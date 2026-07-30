"""Robust extraction of the netlist JSON object from raw VLM output."""
from __future__ import annotations
import json
import re


def _extract_first_json_object(text: str) -> str | None:
    """Scans for the first '{' and returns the substring up to its *matching* '}'
    (proper brace-depth counting, string-literal-aware so braces inside quoted
    strings don't throw off the count) -- unlike a naive first-brace-to-last-brace
    span, this stops at the end of the first complete object and ignores anything
    after it. This specifically guards against a real, observed failure mode: some
    completions duplicate the entire JSON answer, emitting it twice with a stray
    word in between (e.g. "...}\\nolygon\\n{...}"). A first-to-last-brace scan over
    that spans both copies and the garbage between them, producing invalid JSON and
    a hard parse failure even though the model's actual answer (either copy) was
    perfectly valid on its own."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_model_json(raw_text: str) -> dict:
    """Best-effort extraction of the netlist JSON object from model output
    (handles stray markdown fences, <answer> tags from reasoning models,
    leading/trailing prose, or a duplicated answer -- see
    _extract_first_json_object). Always returns a dict with "components" and
    "connections" keys; sets "_parse_error": True if nothing valid was found."""
    text = raw_text.strip()

    answer_tag = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_tag:
        text = answer_tag.group(1).strip()

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        extracted = _extract_first_json_object(text)
        if extracted is not None:
            text = extracted
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"components": [], "connections": [], "_parse_error": True}
    data.setdefault("components", [])
    data.setdefault("connections", [])
    return data
