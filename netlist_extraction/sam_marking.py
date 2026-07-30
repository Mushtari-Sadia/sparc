"""SAM2 automatic mask generation -> Set-of-Mark image, training-free.

No bounding-box training, no per-image tuned thresholds: SAM2's public
checkpoint is used zero-shot. We upscale first (input diagrams are often
small, e.g. 384x384, too small for SAM2's native scale and for small
component labels to be legible to the VLM), then filter proposals to a
"component scale" area band and draw numbered circles directly onto the
image -- a single image, marks baked into the pixels.

Requires the SAM2.1 small checkpoint at CHECKPOINT (see this repo's README
for the download link).
"""
from __future__ import annotations
import os
import numpy as np
import cv2
from PIL import Image

CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "sam2.1_hiera_small.pt")
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
UPSCALE_LONG_SIDE = 768
MIN_AREA_FRAC = 0.00015  # ignore tiny noise blobs (schematics are sparse line art, symbols are small)
MAX_AREA_FRAC = 0.04     # ignore near-whole-image / background masks (per raw proposal)
MAX_MARKS = 25
MERGE_PAD_FRAC = 0.02    # pad a kept box by this * (w+h) before testing containment of smaller
                         # candidates -- scales with each box's own size (a big op-amp triangle
                         # gets a bigger absorption radius than a small resistor), which avoids
                         # both under-merging large symbols and chaining across small distinct ones.
MAX_CLUSTER_AREA_FRAC = 0.10  # safety cap on the *merged* cluster bbox, so padding-driven merges
                               # can't chain together unrelated, far-apart symbols into one blob.
TARGET_MAX_MARKS = 15    # soft ceiling, distinct from MAX_MARKS' hard truncation: most CktBench
                         # diagrams have well under 15 real components (gold component counts
                         # observed range ~5-22, median far below 15), so a first pass landing
                         # above this is a signal of over-segmentation (e.g. 25 marks for a
                         # 7-component circuit, directly observed corrupting a final-assembly
                         # prediction into a spurious 17-resistor answer -- see README), not a
                         # genuinely dense diagram. Retried with progressively larger absorption
                         # padding rather than trusted as-is; MAX_MARKS' hard truncation stays as
                         # a last-resort safety net if escalation still doesn't get there.
MERGE_PAD_ESCALATION = [0.02, 0.05, 0.09, 0.14]  # MERGE_PAD_FRAC values tried in order until
                                                  # mark count drops to TARGET_MAX_MARKS or below


_generator = None


def load_sam2(device: str = "cuda"):
    global _generator
    if _generator is not None:
        return _generator
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    # apply_postprocessing=False: the compiled CUDA post-processing extension isn't
    # built in this env; skipping it is explicitly noted as safe by SAM2's own warning.
    sam2_model = build_sam2(MODEL_CFG, CHECKPOINT, device=device, apply_postprocessing=False)
    # Thresholds are looser than SAM2's natural-image defaults: schematics are sparse
    # line art (thin strokes, open shapes), which produce much weaker/less "stable"
    # region proposals than photographed objects, so the default thresholds return
    # almost nothing on this domain.
    _generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=48,
        pred_iou_thresh=0.5,
        stability_score_thresh=0.6,
        min_mask_region_area=10,
        box_nms_thresh=0.9,
    )
    return _generator


def unload_sam2():
    global _generator
    _generator = None
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()


def _upscale(image: Image.Image) -> Image.Image:
    w, h = image.size
    scale = UPSCALE_LONG_SIDE / max(w, h)
    if scale <= 1.0:
        return image.convert("RGB")
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return image.convert("RGB").resize(new_size, Image.LANCZOS)


def _cluster_masks(masks: list, img_area: int, merge_pad_frac: float = MERGE_PAD_FRAC) -> list:
    """Filter proposals to component scale, then greedily consolidate: process largest
    proposals first, and absorb (drop) any remaining smaller proposal that is mostly
    contained within an already-kept box's *padded* footprint. Padding/containment is
    computed relative to each kept box's own size, not a single global pixel radius --
    a global radius either under-merges large symbols (op-amp triangles) or, if set big
    enough for those, over-merges/chains across small nearby-but-distinct symbols. Using
    "mostly inside the (slightly padded) larger box" scales naturally with symbol size.

    merge_pad_frac is exposed (not just the module constant) so generate_marked_image
    can retry with progressively more aggressive absorption when a first pass
    over-segments -- see TARGET_MAX_MARKS."""
    kept = []
    for m in masks:
        frac = m["area"] / img_area
        if frac < MIN_AREA_FRAC or frac > MAX_AREA_FRAC:
            continue
        kept.append(m)

    if not kept:
        return []

    # largest first: bigger proposals are more likely to already span a whole symbol
    kept.sort(key=lambda m: -m["area"])

    def contained_frac(inner_bbox, outer_bbox_padded):
        ix, iy, iw, ih = inner_bbox
        ox1, oy1, ox2, oy2 = outer_bbox_padded
        ix1, iy1, ix2, iy2 = ix, iy, ix + iw, iy + ih
        cx1, cy1 = max(ix1, ox1), max(iy1, oy1)
        cx2, cy2 = min(ix2, ox2), min(iy2, oy2)
        inter = max(0, cx2 - cx1) * max(0, cy2 - cy1)
        inner_area = max(1, iw * ih)
        return inter / inner_area

    final = []  # list of dicts: bbox (union so far), _pad (frozen at first size), _n_members, _mask
    for m in kept:
        x, y, w, h = m["bbox"]
        absorbed_into = None
        for f in final:
            # use the anchor's FROZEN original-size pad, not one recomputed from its
            # current (possibly already-grown) union bbox -- otherwise each absorption
            # makes the anchor bigger, which makes its pad bigger, which absorbs more,
            # cascading into one giant blob (the same chaining failure as before).
            fx, fy, fw, fh = f["bbox"]
            f_pad = f["_pad"]
            outer_padded = (fx - f_pad, fy - f_pad, fx + fw + f_pad, fy + fh + f_pad)
            if contained_frac((x, y, w, h), outer_padded) > 0.7:
                absorbed_into = f
                break
        if absorbed_into is not None:
            fx, fy, fw, fh = absorbed_into["bbox"]
            nx1, ny1 = min(fx, x), min(fy, y)
            nx2, ny2 = max(fx + fw, x + w), max(fy + fh, y + h)
            absorbed_into["bbox"] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
            absorbed_into["_n_members"] += 1
            absorbed_into["_mask"] |= m["segmentation"]
            continue

        final.append({"bbox": (x, y, w, h), "_pad": merge_pad_frac * (w + h),
                       "_centroid": (x + w / 2.0, y + h / 2.0),
                       "area": w * h, "_n_members": 1,
                       "_mask": m["segmentation"].copy()})
        if len(final) >= MAX_MARKS * 3:  # generous cap before final trim, avoids O(n^2) blowup
            break

    for f in final:
        x, y, w, h = f["bbox"]
        f["_centroid"] = (x + w / 2.0, y + h / 2.0)
        f["area"] = w * h

    final = [f for f in final if f["area"] / img_area <= MAX_CLUSTER_AREA_FRAC]
    final.sort(key=lambda c: (-c["_n_members"], -c["area"]))
    final = final[:MAX_MARKS]

    # left-to-right, top-to-bottom reading order for stable, human-legible numbering
    row_height = max(1.0, (img_area ** 0.5) * 0.03)
    final.sort(key=lambda c: (c["_centroid"][1] // row_height, c["_centroid"][0]))
    return final


def generate_marked_image(image: Image.Image, device: str = "cuda"):
    """Returns (upscaled_plain_image, upscaled_marked_image, num_marks, boxes_by_id, masks_by_id).
    boxes_by_id: {"1": (x, y, w, h), ...} in the upscaled image's pixel coords --
    same numbering as drawn on the marked image, consumed by wire_graph.py for
    geometric connectivity.
    masks_by_id: {"1": bool ndarray (H,W), ...} -- the actual per-pixel SAM footprint
    (union of every raw proposal absorbed into that mark), used to erase exactly the
    symbol's ink rather than a padded rectangle, which can bleed into real wiring."""
    generator = load_sam2(device=device)
    up = _upscale(image)
    arr = np.array(up)
    masks = generator.generate(arr)
    img_area = arr.shape[0] * arr.shape[1]

    # Adaptive re-clustering: a first pass landing above TARGET_MAX_MARKS is treated as
    # over-segmentation, not a genuinely dense diagram (see TARGET_MAX_MARKS docstring),
    # and retried with progressively more aggressive absorption padding. Re-clustering is
    # pure geometry over the already-computed raw proposals -- no extra SAM2 inference,
    # cheap to retry a few times.
    marks = _cluster_masks(masks, img_area, merge_pad_frac=MERGE_PAD_ESCALATION[0])
    for pad in MERGE_PAD_ESCALATION[1:]:
        if len(marks) <= TARGET_MAX_MARKS:
            break
        marks = _cluster_masks(masks, img_area, merge_pad_frac=pad)
    boxes_by_id = {str(i): m["bbox"] for i, m in enumerate(marks, start=1)}
    masks_by_id = {str(i): m["_mask"] for i, m in enumerate(marks, start=1)}

    marked = np.ascontiguousarray(arr.copy())
    for i, m in enumerate(marks, start=1):
        x, y, w, h = (int(round(v)) for v in m["bbox"])
        cx, cy = int(round(m["_centroid"][0])), int(round(m["_centroid"][1]))
        cv2.rectangle(marked, (x, y), (x + w, y + h), (255, 0, 0), 1, lineType=cv2.LINE_AA)
        radius = 9
        cv2.circle(marked, (cx, cy), radius, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(marked, (cx, cy), radius, (200, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        label = str(i)
        font_scale = 0.35
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.putText(marked, label, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 0, 0), 1, cv2.LINE_AA)

    return up, Image.fromarray(marked), len(marks), boxes_by_id, masks_by_id
