#!/usr/bin/env python3
"""
Extract lightweight mask-based morphology features for detector candidates.

This is an analysis tool for case_analysis exports. It uses a simple local
threshold + connected-component segmentation inside each predicted box, then
summarizes mask features for TP/FP candidates.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import measure, morphology


def _load_summaries(case_analysis_dir: Path) -> Iterable[Dict]:
    for path in sorted(case_analysis_dir.rglob("summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_case_dir"] = str(path.parent)
        yield data


def _clip_box(box: Sequence[float], shape: Sequence[int], pad: int) -> Tuple[int, int, int, int, int, int]:
    y1 = max(int(np.floor(float(box[0]))) - pad, 0)
    x1 = max(int(np.floor(float(box[1]))) - pad, 0)
    z1 = max(int(np.floor(float(box[2]))) - pad, 0)
    y2 = min(int(np.ceil(float(box[3]))) + pad + 1, int(shape[0]))
    x2 = min(int(np.ceil(float(box[4]))) + pad + 1, int(shape[1]))
    z2 = min(int(np.ceil(float(box[5]))) + pad + 1, int(shape[2]))
    return y1, x1, z1, y2, x2, z2


def _box_dims_mm(box: Sequence[float], spacing: Sequence[float]) -> Tuple[float, float, float]:
    dy = max(float(box[3]) - float(box[0]), 0.0) * float(spacing[0])
    dx = max(float(box[4]) - float(box[1]), 0.0) * float(spacing[1])
    dz = max(float(box[5]) - float(box[2]), 0.0) * float(spacing[2])
    return dy, dx, dz


def _largest_component_near_center(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask.astype(bool, copy=False)
    label_img, n_labels = ndimage.label(mask)
    if n_labels <= 1:
        return mask.astype(bool, copy=False)

    center = np.asarray([(s - 1) / 2.0 for s in mask.shape], dtype=np.float32)
    best_label = 0
    best_key = None
    for label in range(1, n_labels + 1):
        coords = np.argwhere(label_img == label)
        if coords.size == 0:
            continue
        area = coords.shape[0]
        centroid = coords.mean(axis=0)
        dist = float(np.linalg.norm(centroid - center))
        # Prefer components near the bbox center; break ties by larger area.
        key = (dist, -area)
        if best_key is None or key < best_key:
            best_key = key
            best_label = label
    return label_img == best_label


def _projection_features(proj: np.ndarray) -> Dict[str, float]:
    if not np.any(proj):
        return {"area": 0.0, "elongation": 999.0, "solidity": 0.0, "fill": 0.0}
    label_img = measure.label(proj)
    props = measure.regionprops(label_img)
    if not props:
        return {"area": 0.0, "elongation": 999.0, "solidity": 0.0, "fill": 0.0}
    largest = max(props, key=lambda p: p.area)
    minor_axis = float(getattr(largest, "minor_axis_length", 0.0) or 0.0)
    major_axis = float(getattr(largest, "major_axis_length", 0.0) or 0.0)
    elongation = 999.0 if minor_axis <= 1e-6 and major_axis > 0 else major_axis / max(minor_axis, 1e-6)
    minr, minc, maxr, maxc = largest.bbox
    bbox_area = max(1.0, float((maxr - minr) * (maxc - minc)))
    return {
        "area": float(largest.area),
        "elongation": float(min(elongation, 999.0)),
        "solidity": float(getattr(largest, "solidity", 1.0) or 1.0),
        "fill": float(largest.area) / bbox_area,
    }


def _safe_regionprops3d(mask: np.ndarray) -> Dict[str, float]:
    if not np.any(mask):
        return {"elongation_3d": 999.0, "bbox_fill_3d": 0.0}
    label_img = measure.label(mask)
    props = measure.regionprops(label_img)
    if not props:
        return {"elongation_3d": 999.0, "bbox_fill_3d": 0.0}
    largest = max(props, key=lambda p: p.area)
    try:
        major_axis = float(getattr(largest, "major_axis_length", 0.0) or 0.0)
        minor_axis = float(getattr(largest, "minor_axis_length", 0.0) or 0.0)
        elongation = 999.0 if minor_axis <= 1e-6 and major_axis > 0 else major_axis / max(minor_axis, 1e-6)
    except ValueError:
        elongation = 999.0
    bbox = largest.bbox
    bbox_vol = max(1.0, float((bbox[3] - bbox[0]) * (bbox[4] - bbox[1]) * (bbox[5] - bbox[2])))
    return {
        "elongation_3d": float(min(elongation, 999.0)),
        "bbox_fill_3d": float(largest.area) / bbox_vol,
    }


def _extract_mask_features(
    image: np.ndarray,
    box: Sequence[float],
    spacing: Sequence[float],
    threshold: float,
    pad: int,
    closing_radius: int,
) -> Dict[str, float]:
    y1, x1, z1, y2, x2, z2 = _clip_box(box, image.shape, pad)
    patch = image[y1:y2, x1:x2, z1:z2]
    if patch.size == 0:
        return {"mask_voxels": 0.0, "mask_volume_mm3": 0.0, "mask_bbox_fill": 0.0, "component_found": 0.0}

    mask = patch > float(threshold)
    if closing_radius > 0 and np.any(mask):
        mask = morphology.binary_closing(mask, morphology.ball(int(closing_radius)))
    mask = _largest_component_near_center(mask)

    voxel_volume = float(spacing[0]) * float(spacing[1]) * float(spacing[2])
    mask_voxels = int(np.sum(mask))
    box_dy, box_dx, box_dz = _box_dims_mm(box, spacing)
    box_volume = max(float(box_dy * box_dx * box_dz), 1e-6)
    mask_volume = float(mask_voxels) * voxel_volume

    proj_axial = _projection_features(np.any(mask, axis=2))
    proj_sagittal = _projection_features(np.any(mask, axis=1))
    proj_coronal = _projection_features(np.any(mask, axis=0))
    features_3d = _safe_regionprops3d(mask)

    side_areas = [proj_sagittal["area"], proj_coronal["area"]]
    axial_area = max(proj_axial["area"], 1.0)
    best_side_area_ratio = max((min(axial_area, a) / max(axial_area, a, 1.0)) for a in side_areas)
    side_elong_delta = min(
        abs(proj_axial["elongation"] - proj_sagittal["elongation"]),
        abs(proj_axial["elongation"] - proj_coronal["elongation"]),
    )
    side_solidity_delta = min(
        abs(proj_axial["solidity"] - proj_sagittal["solidity"]),
        abs(proj_axial["solidity"] - proj_coronal["solidity"]),
    )
    side_fill_delta = min(
        abs(proj_axial["fill"] - proj_sagittal["fill"]),
        abs(proj_axial["fill"] - proj_coronal["fill"]),
    )

    return {
        "component_found": float(mask_voxels > 0),
        "mask_voxels": float(mask_voxels),
        "mask_volume_mm3": float(mask_volume),
        "mask_bbox_fill": float(mask_volume / box_volume),
        "elongation_3d": features_3d["elongation_3d"],
        "bbox_fill_3d": features_3d["bbox_fill_3d"],
        "axial_area": proj_axial["area"],
        "axial_elongation": proj_axial["elongation"],
        "axial_solidity": proj_axial["solidity"],
        "axial_fill": proj_axial["fill"],
        "sagittal_area": proj_sagittal["area"],
        "sagittal_elongation": proj_sagittal["elongation"],
        "sagittal_solidity": proj_sagittal["solidity"],
        "sagittal_fill": proj_sagittal["fill"],
        "coronal_area": proj_coronal["area"],
        "coronal_elongation": proj_coronal["elongation"],
        "coronal_solidity": proj_coronal["solidity"],
        "coronal_fill": proj_coronal["fill"],
        "best_side_area_ratio": float(best_side_area_ratio),
        "side_elongation_delta": float(side_elong_delta),
        "side_solidity_delta": float(side_solidity_delta),
        "side_fill_delta": float(side_fill_delta),
    }


def _summarize(rows: List[Dict], label: str) -> Dict:
    subset = [r for r in rows if r["label"] == label]
    out = {"count": len(subset)}
    for key in (
        "score",
        "bbox_max_diameter_mm",
        "mask_volume_mm3",
        "mask_bbox_fill",
        "elongation_3d",
        "bbox_fill_3d",
        "best_side_area_ratio",
        "side_elongation_delta",
        "side_solidity_delta",
        "side_fill_delta",
    ):
        vals = np.asarray([float(r[key]) for r in subset if r.get(key) is not None], dtype=np.float32)
        out[key] = {
            "mean": float(np.mean(vals)) if len(vals) else 0.0,
            "median": float(np.median(vals)) if len(vals) else 0.0,
            "p10": float(np.percentile(vals, 10)) if len(vals) else 0.0,
            "p90": float(np.percentile(vals, 90)) if len(vals) else 0.0,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract lightweight mask features for TP/FP detector candidates.")
    parser.add_argument("--case_analysis_dir", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--spacing", type=float, nargs=3, default=[0.703125, 0.703125, 1.25], metavar=("SY", "SX", "SZ"))
    parser.add_argument("--threshold", type=float, default=0.333, help="normalized CT threshold, 0.333 is about -600 HU")
    parser.add_argument("--pad", type=int, default=2)
    parser.add_argument("--closing_radius", type=int, default=1)
    args = parser.parse_args()

    case_analysis_dir = Path(args.case_analysis_dir)
    output_csv = Path(args.output_csv) if args.output_csv else case_analysis_dir / "candidate_mask_features.csv"
    output_json = Path(args.output_json) if args.output_json else case_analysis_dir / "candidate_mask_feature_report.json"

    rows: List[Dict] = []
    skipped = 0
    image_cache: Dict[str, np.ndarray] = {}
    for summary in _load_summaries(case_analysis_dir):
        case_dir = Path(summary["_case_dir"])
        image_path = case_dir / "ct_model_space.nii.gz"
        if not image_path.exists():
            skipped += 1
            continue
        if str(image_path) not in image_cache:
            image_cache[str(image_path)] = np.asarray(nib.load(str(image_path)).get_fdata(dtype=np.float32), dtype=np.float32)
        image = image_cache[str(image_path)]

        boxes = summary.get("pred_boxes_yxz") or []
        scores = summary.get("pred_scores") or []
        is_tp = summary.get("pred_is_tp") or []
        best_iou = summary.get("pred_best_iou") or []
        case_name = str(summary.get("case_name", case_dir.name))
        for idx, box in enumerate(boxes):
            box_dy, box_dx, box_dz = _box_dims_mm(box, args.spacing)
            row = {
                "case_name": case_name,
                "pred_index": idx,
                "label": "TP" if idx < len(is_tp) and bool(is_tp[idx]) else "FP",
                "score": float(scores[idx]) if idx < len(scores) else None,
                "best_iou": float(best_iou[idx]) if idx < len(best_iou) else None,
                "bbox_dy_mm": float(box_dy),
                "bbox_dx_mm": float(box_dx),
                "bbox_dz_mm": float(box_dz),
                "bbox_max_diameter_mm": float(max(box_dy, box_dx, box_dz)),
                "bbox_volume_mm3": float(box_dy * box_dx * box_dz),
            }
            row.update(_extract_mask_features(image, box, args.spacing, args.threshold, args.pad, args.closing_radius))
            rows.append(row)

    report = {
        "case_analysis_dir": str(case_analysis_dir),
        "threshold": float(args.threshold),
        "pad": int(args.pad),
        "closing_radius": int(args.closing_radius),
        "n_candidates": len(rows),
        "n_skipped_cases": skipped,
        "tp": _summarize(rows, "TP"),
        "fp": _summarize(rows, "FP"),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved JSON: {output_json}")
    print(f"Saved CSV: {output_csv}")


if __name__ == "__main__":
    main()
