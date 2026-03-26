#!/usr/bin/env python3
"""
Normalize RetinaNet dataset JSON files without changing semantic content.

What this script does:
1) Ensures split keys exist: training / validation / testing.
2) Normalizes each sample shape:
   - image: string
   - box: list[list[6]]
   - label: list[int]
3) Keeps original sample order to avoid changing data sequencing.
4) Writes pretty UTF-8 JSON with stable key ordering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


SPLITS = ("training", "validation", "testing")


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _normalize_box(box: Any) -> List[float] | None:
    vals = _as_list(box)
    if len(vals) != 6:
        return None
    try:
        return [float(v) for v in vals]
    except (TypeError, ValueError):
        return None


def _normalize_label(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _normalize_item(item: Any) -> Dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    image = item.get("image", "")
    if not isinstance(image, str):
        image = str(image) if image is not None else ""

    boxes_in = _as_list(item.get("box", []))
    labels_in = _as_list(item.get("label", []))

    boxes_out: List[List[float]] = []
    labels_out: List[int] = []

    # Pair by index and drop malformed pairs to keep lengths aligned.
    for idx in range(min(len(boxes_in), len(labels_in))):
        b = _normalize_box(boxes_in[idx])
        l = _normalize_label(labels_in[idx])
        if b is None or l is None:
            continue
        boxes_out.append(b)
        labels_out.append(l)

    # Preserve extra metadata fields.
    out: Dict[str, Any] = dict(item)
    out["image"] = image
    out["box"] = boxes_out
    out["label"] = labels_out
    return out


def normalize_dataset_file(path: Path) -> Tuple[int, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected top-level object with split keys")

    normalized: Dict[str, Any] = {}
    dropped = 0
    total = 0

    for split in SPLITS:
        items_in = _as_list(raw.get(split, []))
        items_out: List[Dict[str, Any]] = []
        for item in items_in:
            total += 1
            ni = _normalize_item(item)
            if ni is None:
                dropped += 1
                continue
            items_out.append(ni)
        normalized[split] = items_out

    # Keep non-split top-level metadata, if any.
    for k, v in raw.items():
        if k not in SPLITS:
            normalized[k] = v

    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return total, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize RetinaNet dataset JSON files.")
    parser.add_argument("files", nargs="+", help="Dataset JSON file paths.")
    args = parser.parse_args()

    for f in args.files:
        p = Path(f)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        total, dropped = normalize_dataset_file(p)
        print(f"{p}: normalized {total} items, dropped {dropped} malformed items")


if __name__ == "__main__":
    main()

