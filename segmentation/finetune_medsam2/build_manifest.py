#!/usr/bin/env python3
"""
Build segmentation manifest JSON for MedSAM2 manifest mode.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple


VALID_IMAGE_EXTS = (".nii.gz", ".nii", ".mhd")


def _strip_double_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def _list_images(image_dir: Path) -> List[Path]:
    out: List[Path] = []
    for path in image_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(path.name.endswith(ext) for ext in VALID_IMAGE_EXTS):
            out.append(path)
    return sorted(out)


def _find_mask(mask_dir: Path, patient_id: str) -> Optional[Path]:
    for ext in VALID_IMAGE_EXTS:
        candidate = mask_dir / f"{patient_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_detection_boxes(det_report_dir: Optional[Path], patient_id: str) -> List[List[float]]:
    if det_report_dir is None:
        return []
    report_path = det_report_dir / f"{patient_id}.json"
    if not report_path.exists():
        return []
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    nodules = report.get("nodules", [])
    boxes = []
    for nodule in nodules:
        box = nodule.get("box_voxel")
        if isinstance(box, list) and len(box) >= 6:
            try:
                vals = [float(v) for v in box[:6]]
            except (TypeError, ValueError):
                continue
            if vals[3] > vals[0] and vals[4] > vals[1]:
                boxes.append(vals)
    return boxes


def _to_path_str(path: Path, base: Path, relative: bool) -> str:
    if relative:
        try:
            return str(Path(os.path.relpath(path.resolve(), start=base.resolve())))
        except ValueError:
            return str(path.resolve())
    return str(path.resolve())


def _split_patient_ids(
    patient_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[set, set, set]:
    ids = list(patient_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    if n == 0:
        return set(), set(), set()

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    if n >= 3:
        train_end = max(1, min(train_end, n - 2))
        val_end = max(train_end + 1, min(val_end, n - 1))
    elif n == 2:
        train_end = 1
        val_end = 1
    else:
        train_end = 1
        val_end = 1

    return set(ids[:train_end]), set(ids[train_end:val_end]), set(ids[val_end:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build segmentation manifest JSON")
    parser.add_argument("--image_dir", required=True, help="Directory containing CT images")
    parser.add_argument("--mask_dir", default=None, help="Directory containing segmentation masks")
    parser.add_argument(
        "--output_json",
        default="segmentation/manifests/dataset_segmentation.json",
        help="Output manifest path",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--relative_paths",
        action="store_true",
        help="Write image/mask paths as relative to output JSON directory",
    )
    parser.add_argument(
        "--det_report_dir",
        default=None,
        help="Optional detection report directory (expects <patient_id>.json with nodules[].box_voxel)",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    mask_dir = Path(args.mask_dir).resolve() if args.mask_dir else None
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    det_report_dir = Path(args.det_report_dir).resolve() if args.det_report_dir else None

    images = _list_images(image_dir)
    if not images:
        raise RuntimeError(f"No images found under: {image_dir}")

    records: List[Dict] = []
    for image_path in images:
        patient_id = _strip_double_suffix(image_path.name)
        record: Dict[str, object] = {
            "patient_id": patient_id,
            "image": _to_path_str(image_path, output_json.parent, args.relative_paths),
        }

        if mask_dir is not None:
            mask_path = _find_mask(mask_dir, patient_id)
            if mask_path is not None:
                record["mask"] = _to_path_str(mask_path, output_json.parent, args.relative_paths)

        boxes = _load_detection_boxes(det_report_dir, patient_id)
        if boxes:
            record["boxes"] = boxes

        records.append(record)

    patient_ids = sorted({r["patient_id"] for r in records})
    train_ids, val_ids, test_ids = _split_patient_ids(
        patient_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    manifest = {
        "training": [r for r in records if r["patient_id"] in train_ids],
        "validation": [r for r in records if r["patient_id"] in val_ids],
        "testing": [r for r in records if r["patient_id"] in test_ids],
        "summary": {
            "total_records": len(records),
            "total_patients": len(patient_ids),
            "train_patients": len(train_ids),
            "val_patients": len(val_ids),
            "test_patients": len(test_ids),
            "has_mask_records": sum(1 for r in records if "mask" in r),
            "has_detection_boxes_records": sum(1 for r in records if "boxes" in r),
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Manifest saved: {output_json}")
    print(
        f"Split counts: train={len(manifest['training'])}, "
        f"val={len(manifest['validation'])}, test={len(manifest['testing'])}"
    )


if __name__ == "__main__":
    main()
