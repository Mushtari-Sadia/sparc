"""Text detection (EasyOCR) utilities for the text-erased pipeline.

Used for *location* only, never *recognition* content: EasyOCR's recognizer is
unreliable on this domain's schematic fonts and symbols (e.g. read '10 Q' at
0.72 confidence for a diagram that actually reads '10 Ω' -- the ohm symbol
misread as the letter Q; verified on real CktBench images before building
this). Detection (finding *where* text boxes are) is much more trustworthy, so
that's all this module trusts EasyOCR for -- reading the actual label content,
where it matters, is left to the main VLM in a later pass (see
text_erased_pipeline.py step 3), pointed at exactly the right region instead of
having to search the whole image.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=True)
    return _reader


def unload_reader():
    global _reader
    _reader = None
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def detect_text_boxes(image: Image.Image, conf_threshold: float = 0.15) -> list:
    """Returns [(x0, y0, x1, y1), ...] axis-aligned boxes for detected text
    regions. Threshold is deliberately low: only location is trusted here, and a
    real text region's box is usually fine even when its recognized content is
    garbled, so filtering harder on confidence would just drop real text boxes
    for no benefit."""
    reader = _get_reader()
    arr = np.array(image.convert("RGB"))
    results = reader.readtext(arr)
    boxes = []
    for bbox, _text, conf in results:
        if conf < conf_threshold:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
    return boxes


def erase_text(image: Image.Image, boxes: list, pad: int = 2) -> Image.Image:
    """Paints over each detected text box with white -- this dataset's schematics
    are white-background line art, so a flat white fill removes the text
    without introducing a visible artifact SAM or the VLM would need to explain."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for (x0, y0, x1, y1) in boxes:
        draw.rectangle([x0 - pad, y0 - pad, x1 + pad, y1 + pad], fill=(255, 255, 255))
    return out


def draw_text_boxes(image: Image.Image, boxes: list) -> Image.Image:
    """Draws an outline (not filled -- text stays legible) around each detected
    text region on a copy of the given (un-erased) image, so a later VLM pass
    can be pointed at exactly where labels are without this module ever having
    to trust what EasyOCR thinks they say."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for (x0, y0, x1, y1) in boxes:
        draw.rectangle([x0, y0, x1, y1], outline=(220, 30, 30), width=2)
    return out
