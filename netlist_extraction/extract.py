"""End-to-end circuit-diagram -> netlist extraction.

This is the Segment-Trace-Reason (STR) pipeline described in the paper:
  1. **Segment (component evidence).** Detect and erase text (EasyOCR,
     location only) so SAM2 doesn't mistake labels for components, then
     segment components on the text-erased image (SAM2, zero-shot) into a
     numbered Set-of-Mark image. A text-boxed (not erased) copy of the
     original is also kept, so the final VLM call can still read labels.
  2. **Trace (connectivity evidence).** Deterministically trace wire
     connectivity on the same text-erased image with classical CV
     (binarize -> mask out symbols -> skeletonize -> line detection -> net
     clustering), then flag the high-confidence subset corroborated by
     series/parallel structure.
  3. **Reason.** One VLM call sees the plain text-erased image, the numbered
     component-evidence image, the text-boxed image, and the connectivity
     evidence, and returns the full netlist as JSON.

The final VLM is swappable (any of submission/sparc/src/model.py's backends)
-- nothing about it is trained; all the circuit-specific work happens in
steps 1-2, which run locally and are free.

See this repo's README for setup instructions. Usage:
    python extract.py --image path/to/circuit.png --model claude
    python extract.py --image_dir path/to/images --model openai --out results.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

import sam_marking
import text_utils
import wire_graph
import prompts
from json_utils import parse_model_json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from model import Model  # noqa: E402


def _wire_summary_text(series_links, parallel_groups, raw_connections) -> str:
    confident = sorted(wire_graph.confident_connections(series_links, parallel_groups))
    return (
        f"Raw wire-traced connections (marker id pairs, may include false positives): "
        f"{sorted(raw_connections)}\n"
        f"High-confidence connections (series/parallel-corroborated only): {confident}\n"
        f"Series (pass-through) links: {series_links}\n"
        f"Parallel groups (share both terminals): {parallel_groups}"
    )


def build_evidence(image: Image.Image, device: str = "cuda"):
    """Steps 1-2: text erasure, SAM2 component segmentation, wire tracing.
    Returns (plain_erased, marked_erased, text_boxed, wire_summary_text)."""
    upscaled_original = sam_marking._upscale(image)
    text_boxes = text_utils.detect_text_boxes(upscaled_original)
    text_erased = text_utils.erase_text(upscaled_original, text_boxes)
    text_boxed = text_utils.draw_text_boxes(upscaled_original, text_boxes)

    plain_erased, marked_erased, n_marks, boxes_by_id, masks_by_id = \
        sam_marking.generate_marked_image(text_erased, device=device)

    conn = wire_graph.build_connectivity(np.array(plain_erased), boxes_by_id, masks_by_id=masks_by_id)
    wire_summary = _wire_summary_text(conn["series_links"], conn["parallel_groups"], conn["connections"])
    return plain_erased, marked_erased, text_boxed, wire_summary


def extract_netlist(image: Image.Image, model_name: str = "claude", variant: str | None = None,
                     device: str = "cuda") -> dict:
    """Runs the full pipeline on one PIL image and returns the parsed netlist:
    {"components": [{"id": ..., "type": ...}, ...], "connections": [[id_a, id_b], ...],
    "marker_mapping": {...}}."""
    plain_erased, marked_erased, text_boxed, wire_summary = build_evidence(image, device=device)

    sys_p, usr_p = prompts.build_extraction_prompt(wire_summary)
    model = Model(model_name)
    raw = model.predict_vl(sys_p, usr_p, images=[plain_erased, marked_erased, text_boxed], variant=variant)
    return parse_model_json(raw)


def main():
    ap = argparse.ArgumentParser(description="Extract a structured netlist from a circuit diagram image.")
    ap.add_argument("--image", type=str, help="path to a single circuit diagram image")
    ap.add_argument("--image_dir", type=str, help="directory of images to process (batch mode)")
    ap.add_argument("--model", type=str, default="claude",
                     choices=["openai", "claude", "gemini", "qwen", "llava", "internvl", "nemotron", "glm"],
                     help="backend from submission/sparc/src/model.py's Model class")
    ap.add_argument("--variant", type=str, default=None, help="override the backend's default model variant")
    ap.add_argument("--out", type=str, default=None, help="write result(s) as JSON to this path")
    ap.add_argument("--device", type=str, default="cuda", help="device for SAM2 (cuda or cpu)")
    args = ap.parse_args()

    if not args.image and not args.image_dir:
        raise SystemExit("must pass --image or --image_dir")

    if args.image:
        paths = [args.image]
    else:
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        paths = sorted(
            os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.lower().endswith(exts)
        )

    results = {}
    for path in paths:
        print(f"[extract] {path}")
        image = Image.open(path)
        pred = extract_netlist(image, model_name=args.model, variant=args.variant, device=args.device)
        results[os.path.basename(path)] = pred
        print(json.dumps(pred, indent=2))

    sam_marking.unload_sam2()
    text_utils.unload_reader()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results if args.image_dir else next(iter(results.values())), f, indent=2)
        print(f"\nWrote result(s) to {args.out}")


if __name__ == "__main__":
    main()
