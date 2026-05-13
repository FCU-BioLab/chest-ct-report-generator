#!/usr/bin/env python3
"""
Run MedSAM2 on detector candidates and extract mask-based morphology features.

This is intentionally an offline analysis tool. It reads case_analysis exports
from RetinaNet evaluation, segments each predicted candidate with MedSAM2 using
the detector bbox as prompt, and writes candidate-level morphology features plus
a conservative rule-based keep/remove recommendation.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress is optional
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = PROJECT_ROOT / "llm" / "ct_report_pipeline"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_summaries(case_analysis_dir: Path) -> Iterable[Dict]:
    for path in sorted(case_analysis_dir.rglob("summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_case_dir"] = str(path.parent)
        yield data


def _normalized_to_hu(image: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    return image.astype(np.float32, copy=False) * float(hu_max - hu_min) + float(hu_min)


def _yxz_to_xyz_box(box_yxz: Sequence[float]) -> Dict[str, float]:
    y1, x1, z1, y2, x2, z2 = [float(v) for v in box_yxz]
    return {
        "x_min": min(x1, x2),
        "x_max": max(x1, x2),
        "y_min": min(y1, y2),
        "y_max": max(y1, y2),
        "z_center": (z1 + z2) / 2.0,
    }


def _box_dims_mm(box: Sequence[float], spacing: Sequence[float]) -> Tuple[float, float, float]:
    dy = max(float(box[3]) - float(box[0]), 0.0) * float(spacing[0])
    dx = max(float(box[4]) - float(box[1]), 0.0) * float(spacing[1])
    dz = max(float(box[5]) - float(box[2]), 0.0) * float(spacing[2])
    return dy, dx, dz


def _clip_box(box: Sequence[float], shape: Sequence[int], pad: int = 0) -> Tuple[int, int, int, int, int, int]:
    y1 = max(0, int(np.floor(float(box[0]))) - int(pad))
    x1 = max(0, int(np.floor(float(box[1]))) - int(pad))
    z1 = max(0, int(np.floor(float(box[2]))) - int(pad))
    y2 = min(int(shape[0]), int(np.ceil(float(box[3]))) + int(pad) + 1)
    x2 = min(int(shape[1]), int(np.ceil(float(box[4]))) + int(pad) + 1)
    z2 = min(int(shape[2]), int(np.ceil(float(box[5]))) + int(pad) + 1)
    return y1, x1, z1, y2, x2, z2


def _largest_component_near_box_center(mask: np.ndarray, box: Sequence[float]) -> np.ndarray:
    if not np.any(mask):
        return mask.astype(bool, copy=False)
    labels, n_labels = ndimage.label(mask)
    if n_labels <= 1:
        return mask.astype(bool, copy=False)
    center = np.asarray(
        [
            (float(box[0]) + float(box[3])) / 2.0,
            (float(box[1]) + float(box[4])) / 2.0,
            (float(box[2]) + float(box[5])) / 2.0,
        ],
        dtype=np.float32,
    )
    best_label = 0
    best_key = None
    for label_id in range(1, n_labels + 1):
        coords = np.argwhere(labels == label_id)
        if coords.size == 0:
            continue
        centroid = coords.mean(axis=0)
        dist = float(np.linalg.norm(centroid - center))
        area = int(coords.shape[0])
        key = (dist, -area)
        if best_key is None or key < best_key:
            best_key = key
            best_label = label_id
    return labels == best_label


def _pca_elongation(mask: np.ndarray, spacing: Sequence[float]) -> Tuple[float, float, float, float]:
    coords = np.argwhere(mask)
    if coords.shape[0] < 3:
        return 999.0, 0.0, 0.0, 0.0
    coords_mm = coords.astype(np.float32)
    coords_mm[:, 0] *= float(spacing[0])
    coords_mm[:, 1] *= float(spacing[1])
    coords_mm[:, 2] *= float(spacing[2])
    coords_mm -= coords_mm.mean(axis=0, keepdims=True)
    cov = np.cov(coords_mm, rowvar=False)
    vals = np.linalg.eigvalsh(cov)
    vals = np.sort(np.maximum(vals, 1e-8))[::-1]
    axes = np.sqrt(vals)
    elongation = float(axes[0] / max(axes[-1], 1e-6))
    return elongation, float(axes[0]), float(axes[1]), float(axes[2])


def _projection_features(proj: np.ndarray) -> Dict[str, float]:
    if not np.any(proj):
        return {"area": 0.0, "elongation": 999.0, "fill": 0.0}
    labels, n_labels = ndimage.label(proj)
    if n_labels == 0:
        return {"area": 0.0, "elongation": 999.0, "fill": 0.0}
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    label_id = int(np.argmax(counts))
    coords = np.argwhere(labels == label_id)
    if coords.shape[0] < 2:
        return {"area": float(coords.shape[0]), "elongation": 999.0, "fill": 0.0}
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    bbox_area = max(1.0, float(np.prod(maxs - mins)))
    centered = coords.astype(np.float32) - coords.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    vals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 1e-8))[::-1]
    elongation = float(np.sqrt(vals[0]) / max(np.sqrt(vals[-1]), 1e-6))
    return {
        "area": float(coords.shape[0]),
        "elongation": float(min(elongation, 999.0)),
        "fill": float(coords.shape[0]) / bbox_area,
    }


def _extract_features(
    image_hu_yxz: np.ndarray,
    mask_yxz: np.ndarray,
    box: Sequence[float],
    spacing: Sequence[float],
    pad: int,
) -> Dict[str, float]:
    mask = np.asarray(mask_yxz > 0, dtype=bool)
    mask = _largest_component_near_box_center(mask, box)

    y1, x1, z1, y2, x2, z2 = _clip_box(box, image_hu_yxz.shape, pad=pad)
    box_mask = mask[y1:y2, x1:x2, z1:z2]
    box_image = image_hu_yxz[y1:y2, x1:x2, z1:z2]
    if box_mask.size == 0:
        box_mask = np.zeros((0,), dtype=bool)

    mask_voxels = int(np.sum(mask))
    box_mask_voxels = int(np.sum(box_mask))
    voxel_volume = float(spacing[0]) * float(spacing[1]) * float(spacing[2])
    dy, dx, dz = _box_dims_mm(box, spacing)
    bbox_volume_mm3 = max(float(dy * dx * dz), 1e-6)
    mask_volume_mm3 = float(mask_voxels) * voxel_volume
    bbox_fill = mask_volume_mm3 / bbox_volume_mm3
    pca_elongation, pca_major, pca_mid, pca_minor = _pca_elongation(mask, spacing)

    mask_coords = np.argwhere(mask)
    if mask_coords.size:
        mins = mask_coords.min(axis=0)
        maxs = mask_coords.max(axis=0) + 1
        mask_bbox_voxels = np.maximum(maxs - mins, 1)
        mask_bbox_volume_mm3 = (
            float(mask_bbox_voxels[0]) * float(spacing[0])
            * float(mask_bbox_voxels[1]) * float(spacing[1])
            * float(mask_bbox_voxels[2]) * float(spacing[2])
        )
        mask_bbox_fill = mask_volume_mm3 / max(mask_bbox_volume_mm3, 1e-6)
    else:
        mask_bbox_fill = 0.0

    axial = _projection_features(np.any(mask, axis=2))
    sagittal = _projection_features(np.any(mask, axis=1))
    coronal = _projection_features(np.any(mask, axis=0))
    round_planes = int(axial["elongation"] <= 2.0) + int(sagittal["elongation"] <= 2.0) + int(coronal["elongation"] <= 2.0)
    axial_area = max(axial["area"], 1.0)
    side_area_ratio = max(
        min(axial_area, sagittal["area"]) / max(axial_area, sagittal["area"], 1.0),
        min(axial_area, coronal["area"]) / max(axial_area, coronal["area"], 1.0),
    )

    hu_values = box_image[box_mask > 0] if box_mask.size and box_image.size else np.asarray([], dtype=np.float32)
    hu_mean = float(np.mean(hu_values)) if len(hu_values) else 0.0
    hu_p10 = float(np.percentile(hu_values, 10)) if len(hu_values) else 0.0
    hu_p90 = float(np.percentile(hu_values, 90)) if len(hu_values) else 0.0

    return {
        "mask_valid": float(mask_voxels > 0),
        "mask_voxels": float(mask_voxels),
        "mask_voxels_inside_bbox": float(box_mask_voxels),
        "mask_volume_mm3": float(mask_volume_mm3),
        "mask_bbox_fill": float(bbox_fill),
        "mask_component_bbox_fill": float(mask_bbox_fill),
        "pca_elongation": float(min(pca_elongation, 999.0)),
        "pca_axis_major": float(pca_major),
        "pca_axis_mid": float(pca_mid),
        "pca_axis_minor": float(pca_minor),
        "axial_area": axial["area"],
        "axial_elongation": axial["elongation"],
        "axial_fill": axial["fill"],
        "sagittal_area": sagittal["area"],
        "sagittal_elongation": sagittal["elongation"],
        "sagittal_fill": sagittal["fill"],
        "coronal_area": coronal["area"],
        "coronal_elongation": coronal["elongation"],
        "coronal_fill": coronal["fill"],
        "round_planes_elong_le2": float(round_planes),
        "best_side_area_ratio": float(side_area_ratio),
        "mask_hu_mean": hu_mean,
        "mask_hu_p10": hu_p10,
        "mask_hu_p90": hu_p90,
    }


def _recommend_keep(row: Dict[str, float], args: argparse.Namespace) -> Tuple[bool, str]:
    if float(row["score"]) >= float(args.protect_score):
        return True, "protect_score"
    if float(row["bbox_max_diameter_mm"]) <= float(args.protect_small_mm):
        return True, "protect_small"
    if float(row["mask_valid"]) <= 0:
        return True, "segmentation_failed_keep"

    bad = []
    if float(row["mask_voxels"]) < float(args.min_mask_voxels):
        bad.append("tiny_mask")
    if float(row["pca_elongation"]) > float(args.max_pca_elongation):
        bad.append("elongated_3d")
    if float(row["mask_bbox_fill"]) < float(args.min_mask_bbox_fill):
        bad.append("low_bbox_fill")
    if float(row["best_side_area_ratio"]) < float(args.min_side_area_ratio):
        bad.append("poor_axial_side_consistency")
    if float(row["round_planes_elong_le2"]) < float(args.min_round_planes):
        bad.append("few_round_planes")

    if len(bad) >= int(args.min_bad_conditions):
        return False, ";".join(bad)
    return True, "keep"


def _summarize_rule(rows: List[Dict[str, float]]) -> Dict:
    out = {}
    for label in ("TP", "FP"):
        subset = [r for r in rows if r["label"] == label]
        removed = [r for r in subset if not bool(r["medsam2_morph_keep"])]
        out[label] = {
            "count": len(subset),
            "removed": len(removed),
            "kept": len(subset) - len(removed),
        }
    return out


def _build_report(args: argparse.Namespace, rows: List[Dict], processed_cases: int, case_analysis_dir: Path) -> Dict:
    return {
        "case_analysis_dir": str(case_analysis_dir),
        "medsam2_checkpoint": str(args.medsam2_checkpoint),
        "medsam2_root": str(args.medsam2_root) if args.medsam2_root else None,
        "propagate": not bool(args.no_propagate),
        "score_min": float(args.score_min),
        "n_cases": int(processed_cases),
        "n_candidates": int(len(rows)),
        "rule": {
            "protect_score": float(args.protect_score),
            "protect_small_mm": float(args.protect_small_mm),
            "min_mask_voxels": float(args.min_mask_voxels),
            "max_pca_elongation": float(args.max_pca_elongation),
            "min_mask_bbox_fill": float(args.min_mask_bbox_fill),
            "min_side_area_ratio": float(args.min_side_area_ratio),
            "min_round_planes": float(args.min_round_planes),
            "min_bad_conditions": int(args.min_bad_conditions),
        },
        "rule_summary": _summarize_rule(rows),
    }


def _write_outputs(rows: List[Dict], output_csv: Path, output_json: Path, report: Dict) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MedSAM2 mask morphology features for detector candidates.")
    parser.add_argument("--case_analysis_dir", required=True)
    parser.add_argument("--medsam2_checkpoint", required=True)
    parser.add_argument("--medsam2_root", default=None)
    parser.add_argument("--medsam2_config", default="sam2.1_hiera_t512.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--spacing", type=float, nargs=3, default=[0.703125, 0.703125, 1.25], metavar=("SY", "SX", "SZ"))
    parser.add_argument("--hu_min", type=float, default=-1024.0)
    parser.add_argument("--hu_max", type=float, default=300.0)
    parser.add_argument("--score_min", type=float, default=0.0)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--max_candidates", type=int, default=0)
    parser.add_argument("--no_propagate", action="store_true", help="segment only the prompted axial slice for speed")
    parser.add_argument("--pad", type=int, default=2)
    parser.add_argument("--protect_score", type=float, default=0.95)
    parser.add_argument("--protect_small_mm", type=float, default=6.0)
    parser.add_argument("--min_mask_voxels", type=float, default=3.0)
    parser.add_argument("--max_pca_elongation", type=float, default=8.0)
    parser.add_argument("--min_mask_bbox_fill", type=float, default=0.01)
    parser.add_argument("--min_side_area_ratio", type=float, default=0.10)
    parser.add_argument("--min_round_planes", type=float, default=0.0)
    parser.add_argument("--min_bad_conditions", type=int, default=2)
    parser.add_argument("--flush_every_cases", type=int, default=1, help="write partial CSV/JSON after this many processed cases")
    parser.add_argument("--verbose_partial_saves", action="store_true", help="print a line every time partial CSV/JSON files are saved")
    parser.add_argument("--show_box_progress", action="store_true", help="show nested MedSAM2 bbox progress bars")
    args = parser.parse_args()

    case_analysis_dir = Path(args.case_analysis_dir)
    output_csv = Path(args.output_csv) if args.output_csv else case_analysis_dir / "medsam2_candidate_features.csv"
    output_json = Path(args.output_json) if args.output_json else case_analysis_dir / "medsam2_candidate_feature_report.json"

    from segmentation import MedSAM2Segmenter

    segmenter = MedSAM2Segmenter(
        checkpoint_path=args.medsam2_checkpoint,
        medsam2_root=args.medsam2_root,
        config_file=args.medsam2_config,
        device=args.device,
    )

    rows: List[Dict] = []
    processed_cases = 0
    processed_candidates = 0
    summaries = list(_load_summaries(case_analysis_dir))
    if args.max_cases and args.max_cases > 0:
        summaries = summaries[: int(args.max_cases)]
    case_iter = summaries
    if tqdm is not None:
        case_iter = tqdm(summaries, desc="MedSAM2 cases", ncols=100)
    for summary in case_iter:
        if args.max_cases and processed_cases >= int(args.max_cases):
            break
        if args.max_candidates and processed_candidates >= int(args.max_candidates):
            break

        case_dir = Path(summary["_case_dir"])
        image_path = case_dir / "ct_model_space.nii.gz"
        if not image_path.exists():
            continue
        boxes = summary.get("pred_boxes_yxz") or []
        scores = summary.get("pred_scores") or []
        is_tp = summary.get("pred_is_tp") or []
        best_iou = summary.get("pred_best_iou") or []

        selected = []
        for idx, box in enumerate(boxes):
            score = float(scores[idx]) if idx < len(scores) else 0.0
            if score < float(args.score_min):
                continue
            selected.append((idx, box, score))
            if args.max_candidates and processed_candidates + len(selected) >= int(args.max_candidates):
                break
        if not selected:
            continue

        case_name = str(summary.get("case_name", case_dir.name))
        if tqdm is not None and hasattr(case_iter, "set_postfix"):
            case_iter.set_postfix({"case": case_name[:24], "boxes": len(selected), "done": processed_candidates})
        else:
            print(f"Processing case {processed_cases + 1}: {case_name} ({len(selected)} boxes)", flush=True)

        image_norm_yxz = np.asarray(nib.load(str(image_path)).get_fdata(dtype=np.float32), dtype=np.float32)
        image_hu_yxz = _normalized_to_hu(image_norm_yxz, hu_min=args.hu_min, hu_max=args.hu_max)
        image_hu_xyz = np.transpose(image_hu_yxz, (1, 0, 2))
        medsam_boxes = [_yxz_to_xyz_box(item[1]) for item in selected]
        masks_xyz = segmenter.segment_from_boxes(
            image_hu_xyz,
            medsam_boxes,
            propagate=not bool(args.no_propagate),
            show_progress=bool(args.show_box_progress),
        )

        for (idx, box, score), mask_xyz in zip(selected, masks_xyz):
            mask_yxz = np.transpose(np.asarray(mask_xyz > 0, dtype=np.uint8), (1, 0, 2))
            dy, dx, dz = _box_dims_mm(box, args.spacing)
            row = {
                "case_name": case_name,
                "pred_index": int(idx),
                "label": "TP" if idx < len(is_tp) and bool(is_tp[idx]) else "FP",
                "score": float(score),
                "best_iou": float(best_iou[idx]) if idx < len(best_iou) else 0.0,
                "bbox_dy_mm": float(dy),
                "bbox_dx_mm": float(dx),
                "bbox_dz_mm": float(dz),
                "bbox_max_diameter_mm": float(max(dy, dx, dz)),
                "bbox_volume_mm3": float(dy * dx * dz),
            }
            row.update(_extract_features(image_hu_yxz, mask_yxz, box, args.spacing, pad=args.pad))
            keep, reason = _recommend_keep(row, args)
            row["medsam2_morph_keep"] = bool(keep)
            row["medsam2_morph_remove_reason"] = reason
            rows.append(row)
            processed_candidates += 1

        processed_cases += 1
        if int(args.flush_every_cases) > 0 and processed_cases % int(args.flush_every_cases) == 0:
            partial_report = _build_report(args, rows, processed_cases, case_analysis_dir)
            _write_outputs(rows, output_csv, output_json, partial_report)
            if args.verbose_partial_saves:
                if tqdm is not None:
                    tqdm.write(f"Partial saved: cases={processed_cases} candidates={len(rows)} csv={output_csv}")
                else:
                    print(
                        f"Partial saved: cases={processed_cases} candidates={len(rows)} csv={output_csv}",
                        flush=True,
                    )

    report = _build_report(args, rows, processed_cases, case_analysis_dir)
    _write_outputs(rows, output_csv, output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved CSV: {output_csv}")
    print(f"Saved JSON: {output_json}")


if __name__ == "__main__":
    main()
