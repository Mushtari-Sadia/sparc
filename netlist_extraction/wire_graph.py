"""Deterministic wire tracing + connectivity graph, adapted from the classical-CV
netlist pipeline prototyped in eee-bench/10-04-25/src/cv_functions.py, simplified
for this use case: we already have symbol locations from SAM2, so unlike the
original pipeline (which had to *infer* "not-wire" regions via MSER blobs +
Harris corners + morphological line filtering before it even knew where symbols
were), we can just mask out the known symbol boxes directly. What's left over
is wiring: this module traces it and answers "which symbols are on the same net."

This is the actual differentiator from a plain VLM read: connectivity is not a
guess made by looking at a marked-up image (a task VLMs are known to be weak at
-- exhaustively enumerating every wire in a busy diagram), it's the deterministic
output of line detection + geometric intersection clustering. The VLM's role is
narrowed to what it's comparatively good at: classifying what a cropped/marked
region is.

Pipeline:
  1. Otsu-binarize the image.
  2. Mask out each (padded) symbol bbox -- what remains is wiring + stray text.
  3. Skeletonize, then run LSD (Line Segment Detector) on the skeleton to get
     clean straight-line wire segments.
  4. Merge nearly-collinear/parallel segments that are close together (wires
     broken by anti-aliasing or symbol-mask edges become one segment again).
  5. Union-find cluster all segments that intersect or nearly-touch each other
     into "nets" -- this is what turns a soup of line segments into one group
     per electrically-connected wire, including at T/+ junctions.
  6. For each symbol box, find which net(s) touch its padded boundary (segment
     endpoint inside the box, or segment crossing the box edge).
  7. Two symbols are connected iff they share a net.
"""
from __future__ import annotations
import math
import numpy as np
import cv2


def _binarize(image_arr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thr


def _mask_out_symbols(binary: np.ndarray, boxes: list, pad: int, max_side_frac: float = 0.16) -> np.ndarray:
    """Bbox-based erasure fallback (used only if real per-pixel masks aren't available).
    Boxes are capped to max_side_frac of the image's shorter side (centered on
    the box's own center) before erasing: SAM mark consolidation occasionally
    over-merges into a box that spans a symbol *and* a stretch of real wiring
    (e.g. a resistor plus the wire run leading into it) -- erasing that whole
    oversized box would wipe out real connectivity, not just the symbol.
    Prefer _mask_out_symbols_by_mask when SAM's actual per-pixel masks are available --
    it erases exactly the symbol's ink instead of a rectangle around it, which is both
    more precise and doesn't need this size-capping workaround in the first place."""
    out = binary.copy()
    h, w = out.shape
    max_side = max_side_frac * min(h, w)
    for (x, y, bw, bh) in boxes:
        if bw > max_side or bh > max_side:
            cx, cy = x + bw / 2.0, y + bh / 2.0
            bw, bh = min(bw, max_side), min(bh, max_side)
            x, y = cx - bw / 2.0, cy - bh / 2.0
        x1, y1 = max(0, int(x - pad)), max(0, int(y - pad))
        x2, y2 = min(w, int(x + bw + pad)), min(h, int(y + bh + pad))
        out[y1:y2, x1:x2] = 0
    return out


def _mask_out_symbols_by_mask(binary: np.ndarray, masks_by_id: dict, dilate_px: int = 3) -> np.ndarray:
    """Erase the real SAM footprint of every symbol (unioned, then mildly dilated to
    cover anti-aliased stroke edges just outside the raw mask) instead of a padded
    rectangle. This is the direct fix for the bug found during development: an
    oversized/over-merged mark's *bbox* can span real wiring well outside the symbol's
    actual ink, and erasing that whole rectangle silently deletes real connectivity.
    Erasing the true pixel footprint erases only what's actually symbol ink, regardless
    of how loose the mark's bounding box is."""
    if not masks_by_id:
        return binary.copy()
    h, w = binary.shape
    union = np.zeros((h, w), dtype=bool)
    for m in masks_by_id.values():
        union |= m
    union_u8 = union.astype(np.uint8) * 255
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        union_u8 = cv2.dilate(union_u8, kernel)
    out = binary.copy()
    out[union_u8 > 0] = 0
    return out


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    try:
        from cv2.ximgproc import thinning
        return thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except Exception:
        skel = mask.copy()
        prev = np.zeros_like(skel)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(skel, kernel)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
            temp = cv2.subtract(eroded, opened)
            skel = cv2.bitwise_or(cv2.bitwise_and(skel, cv2.bitwise_not(temp)), temp)
            if np.array_equal(skel, prev):
                break
            prev = skel.copy()
        return skel


def _detect_segments(skel: np.ndarray, min_length: float) -> np.ndarray:
    if skel.dtype != np.uint8:
        skel = np.clip(skel, 0, 255).astype(np.uint8)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(skel)[0]
    segments = []
    if lines is not None:
        # OpenCV versions differ on shape: (N,1,4) vs (N,4) -- normalize either way.
        for (x1, y1, x2, y2) in lines.reshape(-1, 4):
            if math.hypot(x2 - x1, y2 - y1) >= min_length:
                segments.append([float(x1), float(y1), float(x2), float(y2)])
    return np.asarray(segments, dtype=np.float32)


def _merge_parallel_segments(segments: np.ndarray, angle_tol_deg=6.0, offset_tol_px=3.0,
                              gap_tol_px=10.0, max_iters=3, return_counts: bool = False):
    """Greedy agglomeration of collinear/parallel nearby segments into one spanning
    segment -- ported near-verbatim from the eee-bench prototype (cv_functions.py).

    return_counts=True additionally returns, for each output segment, how many raw
    input segments were stitched together to produce it (1 = detected clean in one
    LSD pass, needed no merging; higher = assembled from more anti-aliasing-broken
    fragments across possibly multiple merge iterations). Used as a confidence
    signal: each merge is a heuristic angle/offset/gap judgment call, so a segment
    that needed many of them strung together has more chances to have wrongly
    bridged two things that aren't really the same wire -- see
    `low_fragmentation_connections`, a fully gold-free alternative/complement to
    `confident_connections`'s series/parallel-corroboration filter."""
    if len(segments) == 0:
        return (segments, np.array([], dtype=np.int32)) if return_counts else segments

    S = np.asarray(segments, dtype=np.float32).copy()
    counts = np.ones(len(S), dtype=np.int32)

    def _unit(v):
        n = np.hypot(v[..., 0], v[..., 1]) + 1e-8
        return v / n[..., None]

    def _should_merge(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        au = _unit(np.array([ax2 - ax1, ay2 - ay1], dtype=np.float32))
        bu = _unit(np.array([bx2 - bx1, by2 - by1], dtype=np.float32))
        cosang = abs(au[0] * bu[0] + au[1] * bu[1])
        if cosang < np.cos(np.deg2rad(angle_tol_deg)):
            return False, None
        u = au if (au[0] * bu[0] + au[1] * bu[1]) >= 0 else -au
        n = np.array([-u[1], u[0]], dtype=np.float32)
        a0 = np.array([ax1, ay1], dtype=np.float32)
        b0 = np.array([bx1, by1], dtype=np.float32)
        offset = abs(np.dot(n, b0 - a0))
        if offset > offset_tol_px:
            return False, None

        def proj_t(p):
            return np.dot(p - a0, u)

        a_ts = np.array([proj_t(a0), proj_t(np.array([ax2, ay2], dtype=np.float32))])
        b_ts = np.array([proj_t(b0), proj_t(np.array([bx2, by2], dtype=np.float32))])
        a_min, a_max = a_ts.min(), a_ts.max()
        b_min, b_max = b_ts.min(), b_ts.max()
        gap = max(b_min - a_max, a_min - b_max, 0.0)
        if gap > gap_tol_px:
            return False, None
        t_min, t_max = min(a_min, b_min), max(a_max, b_max)
        signed_off = np.dot(n, b0 - a0)
        a_base = a0 + 0.5 * signed_off * n
        p_min = a_base + t_min * u
        p_max = a_base + t_max * u
        return True, np.array([p_min[0], p_min[1], p_max[0], p_max[1]], dtype=np.float32)

    for _ in range(max_iters):
        used = np.zeros(len(S), dtype=bool)
        merged_list = []
        counts_new_list = []
        i = 0
        while i < len(S):
            if used[i]:
                i += 1
                continue
            cur = S[i].copy()
            cur_count = counts[i]
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(len(S)):
                    if used[j] or j == i:
                        continue
                    ok, m = _should_merge(cur, S[j])
                    if ok:
                        cur = m
                        cur_count += counts[j]
                        used[j] = True
                        changed = True
            merged_list.append(cur)
            counts_new_list.append(cur_count)
            i += 1
        S_new = np.vstack(merged_list) if merged_list else np.empty((0, 4), np.float32)
        counts_new = np.array(counts_new_list, dtype=np.int32) if counts_new_list else np.array([], dtype=np.int32)
        if len(S_new) == len(S):
            S, counts = S_new, counts_new
            break
        S, counts = S_new, counts_new
    S = S.astype(np.float32)
    return (S, counts) if return_counts else S


def _seg_near(a, b, tol=4.0, angle_tol_deg=6.0) -> bool:
    """True if segments a, b intersect, nearly-touch, or collinearly overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ax, ay = ax2 - ax1, ay2 - ay1
    bx, by = bx2 - bx1, by2 - by1

    def cross(x1, y1, x2, y2):
        return x1 * y2 - y1 * x2

    def dot(x1, y1, x2, y2):
        return x1 * x2 + y1 * y2

    La, Lb = math.hypot(ax, ay), math.hypot(bx, by)
    if La < 1e-6 or Lb < 1e-6:
        return False

    denom = cross(ax, ay, bx, by)
    if abs(denom) > 1e-6:
        dx, dy = bx1 - ax1, by1 - ay1
        t = cross(dx, dy, bx, by) / denom
        u = cross(dx, dy, ax, ay) / denom
        if -tol / La <= t <= 1 + tol / La and -tol / Lb <= u <= 1 + tol / Lb:
            return True

    def point_to_seg_dist(px, py, x1, y1, x2, y2):
        vx, vy = x2 - x1, y2 - y1
        L2 = vx * vx + vy * vy
        if L2 == 0:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * vx + (py - y1) * vy) / L2))
        qx, qy = x1 + t * vx, y1 + t * vy
        return math.hypot(px - qx, py - qy)

    dists = [
        point_to_seg_dist(ax1, ay1, bx1, by1, bx2, by2),
        point_to_seg_dist(ax2, ay2, bx1, by1, bx2, by2),
        point_to_seg_dist(bx1, by1, ax1, ay1, ax2, ay2),
        point_to_seg_dist(bx2, by2, ax1, ay1, ax2, ay2),
    ]
    if min(dists) <= tol:
        return True

    cosang = max(-1.0, min(1.0, abs(dot(ax, ay, bx, by) / (La * Lb))))
    ang_deg = math.degrees(math.acos(cosang))
    if ang_deg <= angle_tol_deg:
        perp = abs(cross(bx1 - ax1, by1 - ay1, ax, ay)) / La
        if perp <= tol:
            ux, uy = ax / La, ay / La

            def proj(x, y):
                return (x - ax1) * ux + (y - ay1) * uy

            b_lo, b_hi = sorted([proj(bx1, by1), proj(bx2, by2)])
            overlap = min(La, b_hi) - max(0.0, b_lo)
            if overlap >= -tol:
                return True
    return False


def _closest_points(a, b):
    """Closest pair of points between two segments (used to place a junction node
    where two near-touching -- not necessarily exactly crossing -- segments meet)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    candidates = []
    for (px, py) in [(ax1, ay1), (ax2, ay2)]:
        for (qx, qy) in [(bx1, by1), (bx2, by2)]:
            candidates.append((math.hypot(px - qx, py - qy), (px, py), (qx, qy)))
    d, p, q = min(candidates, key=lambda c: c[0])
    return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)


def _cluster_nets(segments: np.ndarray, tol=4.0, angle_tol_deg=6.0):
    """Returns (nets, seg_idx_to_net_idx) -- the mapping lets callers (e.g. junction
    rescue) know which net a given segment index belongs to."""
    n = len(segments)
    if n == 0:
        return [], {}
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if _seg_near(segments[i], segments[j], tol=tol, angle_tol_deg=angle_tol_deg):
                union(i, j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    net_list = []
    seg_idx_to_net_idx = {}
    for net_idx, members in enumerate(groups.values()):
        net_list.append(np.vstack([segments[i] for i in members]))
        for i in members:
            seg_idx_to_net_idx[i] = net_idx
    return net_list, seg_idx_to_net_idx


def _segment_intersects_rect(seg, rect) -> bool:
    x1, y1, x2, y2 = seg
    rx1, ry1, rx2, ry2 = rect

    def inside(x, y):
        return rx1 <= x <= rx2 and ry1 <= y <= ry2

    if inside(x1, y1) or inside(x2, y2):
        return True

    def seg_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
        d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

    edges = [((rx1, ry1), (rx2, ry1)), ((rx2, ry1), (rx2, ry2)),
             ((rx2, ry2), (rx1, ry2)), ((rx1, ry2), (rx1, ry1))]
    for e1, e2 in edges:
        if seg_intersect((x1, y1), (x2, y2), e1, e2):
            return True
    return False


def _find_junctions(segments: np.ndarray, boxes_by_id: dict, touch_pad: float = 20.0,
                     merge_tol: float = 8.0, rescue_radius: float = 80.0) -> dict:
    """Extract explicit junction/point nodes from the segment set, rather than only
    the coarse "these segments form one net" clustering. Each node gets a wire-degree
    (how many distinct segments terminate there) and knows which symbols it touches.

    This is the signal a pure net-clustering approach throws away: a node with wire
    degree 1 and no symbol touch is a dangling wire end -- almost always a tracing
    miss, not a real feature of the schematic (real circuits don't have stub
    antennas), and its position is a precise, localized pointer to *where* the trace
    likely under-shot a nearby symbol -- so it doubles as a repair signal: for each
    dangling node, if a symbol sits within `rescue_radius`, add that as a recovered
    touch instead of just discarding the dangling end as noise.

    Returns {"nodes": [...], "rescued_touches": [(node_idx, mark_id), ...]}
    where each node has: point, degree, touches (set of mark_id), is_dangling, is_branch.
    """
    ids = list(boxes_by_id.keys())
    boxes = [boxes_by_id[i] for i in ids]

    # candidate points: every segment endpoint, plus every near-intersection between
    # a pair of segments (this is what actually locates a T/+/X junction's position,
    # as opposed to just knowing "these two segments are in the same net").
    candidates = []  # (x, y, seg_idx)
    n = len(segments)
    for i in range(n):
        x1, y1, x2, y2 = segments[i]
        candidates.append((x1, y1, i))
        candidates.append((x2, y2, i))
    for i in range(n):
        for j in range(i + 1, n):
            if _seg_near(segments[i], segments[j], tol=merge_tol / 2, angle_tol_deg=6.0):
                px, py = _closest_points(segments[i], segments[j])
                candidates.append((px, py, i))
                candidates.append((px, py, j))

    if not candidates:
        return {"nodes": [], "rescued_touches": []}

    # union-find merge of candidate points within merge_tol
    m = len(candidates)
    parent = list(range(m))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(m):
        for j in range(i + 1, m):
            if math.hypot(candidates[i][0] - candidates[j][0], candidates[i][1] - candidates[j][1]) <= merge_tol:
                union(i, j)

    groups: dict = {}
    for i in range(m):
        groups.setdefault(find(i), []).append(candidates[i])

    nodes = []
    for members in groups.values():
        xs = [c[0] for c in members]
        ys = [c[1] for c in members]
        point = (sum(xs) / len(xs), sum(ys) / len(ys))
        degree = len({c[2] for c in members})

        touches = set()
        for mid, (bx, by, bw, bh) in zip(ids, boxes):
            rect = (bx - touch_pad, by - touch_pad, bx + bw + touch_pad, by + bh + touch_pad)
            if rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]:
                touches.add(mid)

        nodes.append({
            "point": point, "degree": degree, "touches": touches,
            "is_dangling": degree == 1 and not touches,
            "is_branch": degree >= 3,
        })

    rescued = []
    touched_ids_global = {mid for node in nodes for mid in node["touches"]}
    for node_idx, node in enumerate(nodes):
        if not node["is_dangling"]:
            continue
        px, py = node["point"]
        best, best_d = None, rescue_radius
        for mid, (bx, by, bw, bh) in zip(ids, boxes):
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            # distance to box boundary, not center, so large boxes aren't penalized
            dx = max(bx - px, 0, px - (bx + bw))
            dy = max(by - py, 0, py - (by + bh))
            d = math.hypot(dx, dy)
            if d < best_d:
                best, best_d = mid, d
        if best is not None:
            node["touches"].add(best)
            node["is_dangling"] = False
            rescued.append((node_idx, best))

    return {"nodes": nodes, "rescued_touches": rescued}


def _detect_ground_marks(masks_by_id: dict) -> set:
    """Shape-based ground-symbol detection: a ground symbol is conventionally drawn as
    2-4 horizontal bars of decreasing width stacked vertically (or a similar fan/tick
    shape), which is a distinctive, standardized signature classical shape analysis can
    pick up directly -- no VLM call needed, and "ground" isn't even in the VLM's type
    vocabulary here. Heuristic: find horizontal-bar-shaped connected components within
    the mask (wide relative to tall), require >=2 stacked with mostly-decreasing width
    and small vertical gaps between them (the concentric-bars pattern)."""
    ground_ids = set()
    for mid, mask in masks_by_id.items():
        u8 = mask.astype(np.uint8) * 255
        num, _, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
        bars = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if w >= 3 * max(h, 1) and area >= 4:
                bars.append((y, w))
        if len(bars) < 2:
            continue
        bars.sort(key=lambda b: b[0])  # top to bottom
        widths = [w for _, w in bars]
        decreasing = sum(1 for a, b in zip(widths, widths[1:]) if b <= a * 1.15)
        if decreasing >= len(widths) - 1:  # allow at most one non-decreasing step
            ground_ids.add(mid)
    return ground_ids


def _extract_topology(ids: list, id_to_nets: dict, net_members: dict) -> dict:
    """Derive series/parallel structure directly from net membership -- this is a
    circuit-specific structural feature, not just a netlist-formatting nicety: it's the
    primitive most CktBench/EEE-Bench questions actually reason over ("equivalent
    resistance", "which components are in parallel"), so surfacing it explicitly gives
    the downstream QA step something closer to what it needs than a flat edge list.

    series_links: pairs of components whose shared net has exactly those 2 members
      (a pure pass-through connection, no other component or branch on that net).
    parallel_groups: sets of >=2 components that touch exactly the same pair of nets
      (both terminals shared -- the textbook definition of parallel for 2-terminal parts).
    """
    series_links = set()
    for members in net_members.values():
        if len(members) == 2:
            series_links.add(tuple(sorted(members)))

    two_terminal = {i: tuple(sorted(id_to_nets[i])) for i in ids if len(id_to_nets[i]) == 2}
    by_net_pair: dict = {}
    for i, net_pair in two_terminal.items():
        by_net_pair.setdefault(net_pair, []).append(i)
    parallel_groups = [sorted(members) for members in by_net_pair.values() if len(members) >= 2]

    return {"series_links": sorted(series_links), "parallel_groups": parallel_groups}


def confident_connections(series_links: list, parallel_groups: list) -> set:
    """The reject-thresholded view of wire-tracer connectivity: instead of trusting
    every raw wire-touch connection equally (build_connectivity's `connections`,
    the weakest-measured signal in the pipeline), only keep connections that are
    structurally corroborated by series-link or parallel-group detection (the
    strongest-measured signals). This is what gets passed to the VLM as
    high-confidence connectivity evidence -- see extract.py."""
    confident = set()
    for a, b in series_links:
        confident.add(tuple(sorted((str(a), str(b)))))
    for group in parallel_groups:
        ids = [str(x) for x in group]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                confident.add(tuple(sorted((ids[i], ids[j]))))
    return confident


def build_connectivity(image_arr: np.ndarray, boxes_by_id: dict, masks_by_id: dict = None,
                        min_length: float = 12, symbol_mask_pad: int = 4, touch_pad: int = 20,
                        rescue_radius: float = 80.0) -> dict:
    """boxes_by_id: {mark_id (str) -> (x, y, w, h)} in image_arr pixel coords.
    touch_pad/rescue_radius defaults were calibrated against held-out instance-level
    connection accuracy.
    masks_by_id: {mark_id -> bool ndarray}, preferred over boxes_by_id for erasure when
    available (see _mask_out_symbols_by_mask).
    Returns nets, connections, junction nodes (with dangling/branch flags and rescue
    info), detected ground marks, and series/parallel topology."""
    ids = list(boxes_by_id.keys())
    boxes = [boxes_by_id[i] for i in ids]

    binary = _binarize(image_arr)
    if masks_by_id:
        wire_only = _mask_out_symbols_by_mask(binary, masks_by_id)
    else:
        wire_only = _mask_out_symbols(binary, boxes, pad=symbol_mask_pad)
    skel = _skeletonize(wire_only)
    segs = _detect_segments(skel, min_length=min_length)
    segs, seg_fragment_counts = _merge_parallel_segments(segs, return_counts=True)
    nets, seg_idx_to_net_idx = _cluster_nets(segs)

    # Per-net fragmentation: how many raw LSD segments, at worst, were stitched
    # together to build any one segment in this net -- the max, not the mean, since
    # a single risky multi-fragment stitch anywhere in the net is enough to cast
    # doubt on connections routed through it. See low_fragmentation_connections.
    net_fragment_count: dict = {}
    for seg_idx, net_idx in seg_idx_to_net_idx.items():
        c = int(seg_fragment_counts[seg_idx]) if seg_idx < len(seg_fragment_counts) else 1
        net_fragment_count[net_idx] = max(net_fragment_count.get(net_idx, 1), c)

    id_to_nets = {i: set() for i in ids}
    for net_idx, net_segs in enumerate(nets):
        for i, (x, y, w, h) in zip(ids, boxes):
            rect = (x - touch_pad, y - touch_pad, x + w + touch_pad, y + h + touch_pad)
            if any(_segment_intersects_rect(seg, rect) for seg in net_segs):
                id_to_nets[i].add(net_idx)

    # Text labels ("+V_sat", "R", polarity marks, etc.) are never SAM-marked as
    # components, so they're never erased -- each character/glyph becomes its own
    # tiny LSD "segment", floating disconnected from everything. Restrict junction/
    # dangling analysis to segments belonging to a net that touches >=1 real symbol;
    # a net with zero symbol touches at all is essentially always this kind of text/
    # noise fragment rather than a legitimate (if under-recognized) wire.
    nets_with_touch = {ni for touches in id_to_nets.values() for ni in touches}
    real_seg_mask = np.array([seg_idx_to_net_idx.get(i) in nets_with_touch for i in range(len(segs))])
    real_segs = segs[real_seg_mask] if len(segs) else segs

    # Junction extraction on the noise-filtered segment set, and dangling-end
    # rescue: a node with wire-degree 1 and no symbol touch nearby gets attached to
    # the closest symbol within rescue_radius, and that symbol also gets added to
    # whichever net owns the dangling segment -- directly recovering connections
    # that plain net-clustering + fixed touch_pad would otherwise miss.
    junction_info = _find_junctions(real_segs, boxes_by_id, touch_pad=touch_pad, rescue_radius=rescue_radius)
    for node_idx, mark_id in junction_info["rescued_touches"]:
        node = junction_info["nodes"][node_idx]
        px, py = node["point"]
        best_seg_idx, best_d = None, float("inf")
        for seg_idx, seg in enumerate(segs):
            x1, y1, x2, y2 = seg
            d = min(math.hypot(px - x1, py - y1), math.hypot(px - x2, py - y2))
            if d < best_d:
                best_seg_idx, best_d = seg_idx, d
        if best_seg_idx is not None and best_seg_idx in seg_idx_to_net_idx:
            id_to_nets[mark_id].add(seg_idx_to_net_idx[best_seg_idx])

    net_members: dict = {}
    for i, net_idxs in id_to_nets.items():
        for ni in net_idxs:
            net_members.setdefault(ni, []).append(i)

    connections = set()
    connection_fragment_counts: dict = {}
    for net_idx, members in net_members.items():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pair = tuple(sorted((members[a], members[b])))
                connections.add(pair)
                frag = net_fragment_count.get(net_idx, 1)
                # a pair can in principle be co-touched by more than one net (rare);
                # keep the most favorable (lowest-fragmentation) evidence for it.
                connection_fragment_counts[pair] = min(connection_fragment_counts.get(pair, frag), frag)

    ground_ids = _detect_ground_marks(masks_by_id) if masks_by_id else set()
    topology = _extract_topology(ids, id_to_nets, net_members)

    return {
        "nets": list(net_members.values()),
        "connections": sorted(connections),
        "connection_fragment_counts": {f"{a},{b}": c for (a, b), c in connection_fragment_counts.items()},
        "junctions": junction_info["nodes"],
        "n_rescued": len(junction_info["rescued_touches"]),
        "ground_ids": sorted(ground_ids),
        "series_links": topology["series_links"],
        "parallel_groups": topology["parallel_groups"],
    }
