"""
Headless case pipeline runner for n8n orchestration.

Stages:
- preprocess
- detect
- segment
- feature
- report
- run (all stages)
"""

import argparse
import base64
import html
import json
import math
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PIPELINE_ROOT = PROJECT_ROOT / "llm" / "ct_report_pipeline"

# Ensure local pipeline modules resolve first.
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config, get_medsam2_checkpoint, get_medsam2_root
from lung_rads import build_structured_report_input
from report_generator import get_report_generator
from segmentation import MedSAM2Segmenter


def _case_dir(work_dir: Path, case_id: str) -> Path:
    return work_dir / case_id


def _state_path(case_dir: Path) -> Path:
    return case_dir / "state.json"


def _load_state(case_dir: Path) -> Dict:
    p = _state_path(case_dir)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(case_dir: Path, state: Dict) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    with open(_state_path(case_dir), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _update_total_process_time(state: Dict) -> None:
    timings = state.get("process_times", {})
    total = 0.0
    for item in timings.values():
        if isinstance(item, dict):
            total += float(item.get("elapsed_seconds", 0.0) or 0.0)
    state["total_process_seconds"] = round(total, 3)


def _run_stage_timed(case_dir: Path, state: Dict, stage: str, fn: Callable[[], Dict]) -> Dict:
    original_state = json.loads(json.dumps(state))
    state.setdefault("process_times", {})
    state["process_times"][stage] = {
        "started_at": _now_iso(),
        "status": "running",
    }
    _save_state(case_dir, state)

    started = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:
        elapsed = time.perf_counter() - started
        restored = original_state
        restored.setdefault("process_times", {})
        restored["process_times"][stage] = {
            "started_at": state["process_times"][stage].get("started_at", _now_iso()),
            "finished_at": _now_iso(),
            "elapsed_seconds": round(elapsed, 3),
            "status": "failed",
            "error": str(exc),
        }
        _update_total_process_time(restored)
        _save_state(case_dir, restored)
        raise

    elapsed = time.perf_counter() - started
    state["process_times"][stage].update(
        {
            "finished_at": _now_iso(),
            "elapsed_seconds": round(elapsed, 3),
            "status": "ok",
        }
    )
    _update_total_process_time(state)
    return result


def _require(state: Dict, key: str) -> str:
    value = state.get(key)
    if not value:
        raise ValueError(f"Missing state field: {key}")
    return str(value)


def _infer_gt_label_path_from_input(input_path: str) -> str:
    """
    Try to infer labelsTr path from nnDetection imagesTr input.
    Example:
      .../imagesTr/<case>_0000.nii.gz -> .../labelsTr/<case>.nii.gz
    """
    p = Path(input_path)
    if not p.exists() or not p.is_file():
        return ""
    if "imagesTr" not in str(p):
        return ""
    name = p.name
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    else:
        stem = p.stem
    if stem.endswith("_0000"):
        stem = stem[:-5]
    cand = p.parent.parent / "labelsTr" / f"{stem}.nii.gz"
    return str(cand) if cand.exists() else ""


def _extract_lung_mask(arr_zyx: np.ndarray) -> np.ndarray:
    """
    Build a coarse lung mask from CT HU values.
    Shape is SimpleITK array order: z, y, x.
    """
    air = arr_zyx.astype(np.float32) < -400.0
    lung = np.zeros_like(air, dtype=bool)
    structure_2d = np.ones((3, 3), dtype=bool)

    for z in range(air.shape[0]):
        slice_air = ndi.binary_opening(air[z], structure=structure_2d, iterations=1)
        labels, _ = ndi.label(slice_air)
        if labels.max() == 0:
            continue

        border_labels = np.unique(
            np.concatenate(
                [
                    labels[0, :],
                    labels[-1, :],
                    labels[:, 0],
                    labels[:, -1],
                ]
            )
        )
        border_labels = border_labels[border_labels > 0]
        internal_air = slice_air & ~np.isin(labels, border_labels)
        internal_air = ndi.binary_fill_holes(internal_air)
        internal_air = ndi.binary_closing(internal_air, structure=structure_2d, iterations=2)
        lung[z] = internal_air

    lung = ndi.binary_closing(lung, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    lung = ndi.binary_fill_holes(lung)

    labels, num = ndi.label(lung)
    if num == 0:
        return lung.astype(np.uint8)

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep_count = min(2, int((counts > 0).sum()))
    keep_labels = np.argsort(counts)[-keep_count:]
    lung = np.isin(labels, keep_labels)
    return lung.astype(np.uint8)


def _write_sitk_mask(mask_zyx: np.ndarray, reference: sitk.Image, path: Path) -> None:
    mask_img = sitk.GetImageFromArray(mask_zyx.astype(np.uint8))
    mask_img.CopyInformation(reference)
    sitk.WriteImage(mask_img, str(path))


def _write_sitk_ct(arr_zyx: np.ndarray, reference: sitk.Image, path: Path) -> None:
    ct_img = sitk.GetImageFromArray(arr_zyx.astype(np.float32))
    ct_img.CopyInformation(reference)
    sitk.WriteImage(ct_img, str(path))


def stage_preprocess(case_dir: Path, input_path: str) -> Dict:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"input_path not found: {p}")

    out_dir = case_dir / "01_preprocess"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ct = out_dir / "ct.nii.gz"

    if p.is_dir():
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(p))
        if not series_ids:
            raise ValueError(f"No DICOM series found in: {p}")
        dicom_names = reader.GetGDCMSeriesFileNames(str(p), series_ids[0])
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(str(p))

    sitk.WriteImage(image, str(out_ct))

    arr = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    lung_mask = _extract_lung_mask(arr)
    lung_mask_path = out_dir / "lung_mask.nii.gz"
    lung_ct_path = out_dir / "ct_lung_only.nii.gz"
    _write_sitk_mask(lung_mask, image, lung_mask_path)
    lung_only = np.where(lung_mask > 0, arr, -1000.0)
    _write_sitk_ct(lung_only, image, lung_ct_path)

    summary = {
        "input_path": str(p),
        "preprocess_dir": str(out_dir),
        "ct_path": str(out_ct),
        "lung_mask_path": str(lung_mask_path),
        "lung_ct_path": str(lung_ct_path),
        "lung_mask_voxels": int(lung_mask.sum()),
        "shape_dhw": [int(x) for x in arr.shape],
        "spacing_xyz": [float(x) for x in spacing],
        "origin_xyz": [float(x) for x in origin],
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "input_path": str(p),
        "preprocess_dir": str(out_dir),
        "ct_path": str(out_ct),
        "lung_mask_path": str(lung_mask_path),
        "lung_ct_path": str(lung_ct_path),
        "preprocess_summary": str(out_dir / "summary.json"),
    }


def _box_lung_mask_metrics(box: List[float], lung_mask_zyx: np.ndarray) -> Dict:
    if len(box) != 6 or lung_mask_zyx.ndim != 3:
        return {
            "center_in_lung": False,
            "overlap_ratio": 0.0,
            "valid_box": False,
        }

    x1, y1, z1, x2, y2, z2 = [float(v) for v in box]
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    z_min, z_max = sorted([z1, z2])

    depth, height, width = lung_mask_zyx.shape
    xi0 = max(0, int(math.floor(x_min)))
    yi0 = max(0, int(math.floor(y_min)))
    zi0 = max(0, int(math.floor(z_min)))
    xi1 = min(width - 1, int(math.ceil(x_max)))
    yi1 = min(height - 1, int(math.ceil(y_max)))
    zi1 = min(depth - 1, int(math.ceil(z_max)))

    if xi0 > xi1 or yi0 > yi1 or zi0 > zi1:
        return {
            "center_in_lung": False,
            "overlap_ratio": 0.0,
            "valid_box": False,
        }

    cx = min(width - 1, max(0, int(round((x_min + x_max) / 2.0))))
    cy = min(height - 1, max(0, int(round((y_min + y_max) / 2.0))))
    cz = min(depth - 1, max(0, int(round((z_min + z_max) / 2.0))))
    center_in_lung = bool(lung_mask_zyx[cz, cy, cx] > 0)

    roi = lung_mask_zyx[zi0 : zi1 + 1, yi0 : yi1 + 1, xi0 : xi1 + 1]
    overlap_ratio = float(np.mean(roi > 0)) if roi.size else 0.0
    return {
        "center_in_lung": center_in_lung,
        "overlap_ratio": overlap_ratio,
        "valid_box": True,
    }


def _filter_detection_report_by_lung_mask(
    report_path: Path,
    lung_mask_path: str,
    min_overlap_ratio: float = 0.01,
) -> Dict:
    if not lung_mask_path:
        return {
            "enabled": False,
            "reason": "missing_lung_mask_path",
        }

    mask_path = Path(lung_mask_path)
    if not mask_path.exists():
        return {
            "enabled": False,
            "reason": f"lung mask not found: {mask_path}",
        }

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    raw_report_path = report_path.with_name("report_unfiltered.json")
    if not raw_report_path.exists():
        shutil.copy2(report_path, raw_report_path)

    lung_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(np.uint8)
    nodules = list(report.get("nodules", []))
    kept: List[Dict] = []
    removed: List[Dict] = []

    for nodule in nodules:
        source_id = nodule.get("id")
        metrics = _box_lung_mask_metrics(nodule.get("box_voxel", []), lung_mask)
        keep = bool(metrics["center_in_lung"] or metrics["overlap_ratio"] >= min_overlap_ratio)
        nodule["source_detection_id"] = source_id
        nodule["lung_mask_center_in_lung"] = bool(metrics["center_in_lung"])
        nodule["lung_mask_overlap_ratio"] = round(float(metrics["overlap_ratio"]), 4)
        if keep:
            kept.append(nodule)
        else:
            removed.append(
                {
                    "id": nodule.get("id"),
                    "score": nodule.get("score"),
                    "box_voxel": nodule.get("box_voxel"),
                    "center_in_lung": bool(metrics["center_in_lung"]),
                    "overlap_ratio": round(float(metrics["overlap_ratio"]), 4),
                }
            )

    for new_id, nodule in enumerate(kept, 1):
        nodule["id"] = new_id

    filter_summary = {
        "enabled": True,
        "mask_path": str(mask_path),
        "min_overlap_ratio": float(min_overlap_ratio),
        "before_count": len(nodules),
        "after_count": len(kept),
        "removed_count": len(removed),
        "removed": removed,
        "raw_report_path": str(raw_report_path),
    }
    report["nodules"] = kept
    report["lung_mask_filter"] = filter_summary

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return filter_summary


def stage_detect(
    case_dir: Path,
    model_path: str,
    threshold: float,
    device: str,
) -> Dict:
    state = _load_state(case_dir)
    ct_path = _require(state, "ct_path")

    out_dir = case_dir / "02_detect"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "detection.retinanet.inference",
        "--input_path",
        ct_path,
        "--model_path",
        str(model_path),
        "--output_dir",
        str(out_dir),
        "--threshold",
        str(threshold),
        "--device",
        str(device),
    ]

    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

    detect_report = out_dir / "report.json"
    if not detect_report.exists():
        raise FileNotFoundError(f"Detection report not found: {detect_report}")

    lung_filter = _filter_detection_report_by_lung_mask(
        report_path=detect_report,
        lung_mask_path=str(state.get("lung_mask_path", "")),
    )

    return {
        "detection_dir": str(out_dir),
        "detection_report": str(detect_report),
        "detection_model_path": str(model_path),
        "detection_gt_label_path": "",
        "detection_lung_filter": lung_filter,
    }


def _load_detection_boxes(detection_report: Path) -> List[Dict]:
    with open(detection_report, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes = []
    for nodule in data.get("nodules", []):
        box = nodule.get("box_voxel")
        if not box or len(box) != 6:
            continue

        x1, y1, z1, x2, y2, z2 = [float(v) for v in box]
        boxes.append(
            {
                "x_min": max(0.0, min(x1, x2)),
                "y_min": max(0.0, min(y1, y2)),
                "x_max": max(0.0, max(x1, x2)),
                "y_max": max(0.0, max(y1, y2)),
                "z_center": int(round((z1 + z2) / 2.0)),
            }
        )

    return boxes


def _save_mask(mask: np.ndarray, affine: np.ndarray, path: Path) -> None:
    nii = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nib.save(nii, str(path))


def stage_segment(
    case_dir: Path,
    medsam2_checkpoint: str = "",
    propagate: bool = True,
) -> Dict:
    state = _load_state(case_dir)
    ct_path = _require(state, "ct_path")
    detection_report = Path(_require(state, "detection_report"))

    boxes = _load_detection_boxes(detection_report)
    if not boxes:
        out_dir = case_dir / "03_segment"
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "segment_dir": str(out_dir),
            "mask_paths": [],
            "combined_mask_path": "",
            "segment_status": "no_detections",
        }

    cfg = load_config()
    ckpt = medsam2_checkpoint or str(get_medsam2_checkpoint(cfg))
    medsam2_root = str(get_medsam2_root(cfg))

    ct_volume, affine = MedSAM2Segmenter.load_ct_volume(ct_path)

    segmenter = MedSAM2Segmenter(
        checkpoint_path=ckpt,
        medsam2_root=medsam2_root,
    )

    masks = segmenter.segment_from_boxes(ct_volume, boxes, propagate=propagate)

    out_dir = case_dir / "03_segment"
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_paths: List[str] = []
    combined = np.zeros_like(ct_volume, dtype=np.uint8)

    for i, mask in enumerate(masks, 1):
        m = (mask > 0).astype(np.uint8)
        combined = np.maximum(combined, m)
        p = out_dir / f"mask_nodule_{i:03d}.nii.gz"
        _save_mask(m, affine, p)
        mask_paths.append(str(p))

    combined_path = out_dir / "mask_combined.nii.gz"
    _save_mask(combined, affine, combined_path)

    return {
        "segment_dir": str(out_dir),
        "mask_paths": mask_paths,
        "combined_mask_path": str(combined_path),
        "medsam2_checkpoint": str(ckpt),
    }


def _spacing_from_affine(affine: np.ndarray) -> List[float]:
    if affine is None or affine.shape[0] < 3 or affine.shape[1] < 3:
        return [1.0, 1.0, 1.0]
    linear = np.asarray(affine[:3, :3], dtype=np.float64)
    spacing = np.linalg.norm(linear, axis=0)
    spacing = np.where(spacing > 0, spacing, 1.0)
    return [float(spacing[0]), float(spacing[1]), float(spacing[2])]


def _physical_points_from_voxels(voxels: np.ndarray, affine: np.ndarray) -> np.ndarray:
    linear = np.asarray(affine[:3, :3], dtype=np.float64)
    origin = np.asarray(affine[:3, 3], dtype=np.float64)
    return voxels.astype(np.float64) @ linear.T + origin


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    surface = mask & ~eroded
    vox = np.argwhere(surface > 0)
    if vox.size == 0:
        vox = np.argwhere(mask > 0)
    return vox


def _longest_feret_diameter_mm(mask: np.ndarray, affine: np.ndarray) -> Dict:
    vox = _surface_voxels(mask)
    spacing = _spacing_from_affine(affine)
    if vox.shape[0] == 0:
        return {"value": 0.0, "method": "empty_mask", "point_count": 0}
    if vox.shape[0] == 1:
        return {"value": float(max(spacing)), "method": "single_voxel_spacing", "point_count": 1}

    points = _physical_points_from_voxels(vox, affine)
    method = "surface_voxel_pairwise"

    if points.shape[0] > 15000:
        try:
            hull = ConvexHull(points)
            points = points[hull.vertices]
            method = "convex_hull_vertices"
        except QhullError:
            step = int(math.ceil(points.shape[0] / 15000))
            points = points[::step]
            method = "surface_voxel_sampled"

    if points.shape[0] > 15000:
        step = int(math.ceil(points.shape[0] / 15000))
        points = points[::step]
        method = f"{method}_sampled"

    distances = pdist(points)
    if distances.size == 0:
        return {"value": float(max(spacing)), "method": method, "point_count": int(points.shape[0])}
    return {
        "value": float(np.max(distances)),
        "method": method,
        "point_count": int(points.shape[0]),
    }


def _attenuation_composition(values: np.ndarray) -> Dict:
    if values.size == 0:
        return {
            "attenuation_type": "indeterminate",
            "attenuation_confidence": 0.0,
            "attenuation_basis": "empty_mask",
        }

    air_fraction = float(np.mean(values < -900))
    ground_glass_fraction = float(np.mean((values >= -900) & (values < -500)))
    soft_tissue_fraction = float(np.mean((values >= -300) & (values <= 200)))
    calcified_fraction = float(np.mean(values > 200))
    intermediate_fraction = max(0.0, 1.0 - air_fraction - ground_glass_fraction - soft_tissue_fraction - calcified_fraction)

    if air_fraction > 0.30:
        attenuation_type = "indeterminate"
        confidence = 0.30
        basis = "mask_contains_high_air_fraction"
    elif calcified_fraction >= 0.10:
        attenuation_type = "calcified"
        confidence = min(1.0, calcified_fraction + 0.50)
        basis = "calcified_voxel_fraction"
    elif soft_tissue_fraction >= 0.35 and ground_glass_fraction >= 0.20:
        attenuation_type = "part-solid"
        confidence = min(1.0, soft_tissue_fraction + ground_glass_fraction)
        basis = "mixed_ground_glass_and_soft_tissue_fraction"
    elif soft_tissue_fraction >= 0.50:
        attenuation_type = "solid"
        confidence = min(1.0, soft_tissue_fraction)
        basis = "soft_tissue_fraction"
    elif ground_glass_fraction >= 0.55 and soft_tissue_fraction < 0.20:
        attenuation_type = "ground-glass"
        confidence = min(1.0, ground_glass_fraction)
        basis = "ground_glass_fraction"
    elif soft_tissue_fraction >= 0.20 and ground_glass_fraction >= 0.20:
        attenuation_type = "part-solid"
        confidence = min(0.75, soft_tissue_fraction + ground_glass_fraction)
        basis = "low_confidence_mixed_attenuation"
    else:
        attenuation_type = "indeterminate"
        confidence = max(ground_glass_fraction, soft_tissue_fraction, calcified_fraction)
        basis = "low_confidence_attenuation_distribution"

    return {
        "attenuation_type": attenuation_type,
        "attenuation_confidence": float(confidence),
        "attenuation_basis": basis,
        "air_fraction": air_fraction,
        "ground_glass_fraction": ground_glass_fraction,
        "intermediate_fraction": intermediate_fraction,
        "soft_tissue_fraction": soft_tissue_fraction,
        "calcified_fraction": calcified_fraction,
    }


def _compute_features_for_mask(
    ct_volume: np.ndarray,
    mask: np.ndarray,
    nodule_id: int,
    affine: np.ndarray,
    spacing_x: float,
    spacing_y: float,
    spacing_z: float,
    hu_warning: Optional[str] = None,
) -> Dict:
    vox = np.argwhere(mask > 0)
    if vox.size == 0:
        return {}

    voxel_count = int(vox.shape[0])
    volume_mm3 = voxel_count * spacing_x * spacing_y * spacing_z
    equivalent_diameter_mm = 2.0 * (3.0 * volume_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0)

    # Mask/CT are in (x, y, z) axis order.
    x_min, y_min, z_min = vox.min(axis=0).tolist()
    x_max, y_max, z_max = vox.max(axis=0).tolist()

    bbox_x_mm = (x_max - x_min + 1) * spacing_x
    bbox_y_mm = (y_max - y_min + 1) * spacing_y
    bbox_z_mm = (z_max - z_min + 1) * spacing_z

    values = ct_volume[mask > 0]
    p10_hu, p25_hu, p50_hu, p75_hu, p90_hu = np.percentile(values, [10, 25, 50, 75, 90])
    longest = _longest_feret_diameter_mm(mask, affine)
    attenuation = _attenuation_composition(values)

    feature = {
        "nodule_id": int(nodule_id),
        "voxel_count": voxel_count,
        "volume_mm3": float(volume_mm3),
        "equivalent_diameter_mm": float(equivalent_diameter_mm),
        "longest_axis_mm": float(longest["value"]),
        "longest_axis_method": str(longest["method"]),
        "longest_axis_point_count": int(longest["point_count"]),
        "bbox_longest_axis_mm": float(max(bbox_x_mm, bbox_y_mm, bbox_z_mm)),
        "short_axis_mm": float(min(bbox_x_mm, bbox_y_mm, bbox_z_mm)),
        "bbox_x_mm": float(bbox_x_mm),
        "bbox_y_mm": float(bbox_y_mm),
        "bbox_z_mm": float(bbox_z_mm),
        "bbox": {
            "x_min": int(x_min),
            "x_max": int(x_max),
            "y_min": int(y_min),
            "y_max": int(y_max),
            "z_min": int(z_min),
            "z_max": int(z_max),
        },
        "mean_hu": float(np.mean(values)),
        "std_hu": float(np.std(values)),
        "min_hu": float(np.min(values)),
        "max_hu": float(np.max(values)),
        "p10_hu": float(p10_hu),
        "p25_hu": float(p25_hu),
        "median_hu": float(p50_hu),
        "p75_hu": float(p75_hu),
        "p90_hu": float(p90_hu),
        "spacing_mm": [float(spacing_x), float(spacing_y), float(spacing_z)],
    }
    feature.update(attenuation)
    if hu_warning:
        feature["hu_warning"] = hu_warning
    return feature


def _detect_hu_scale_issue(ct_volume: np.ndarray) -> str:
    finite = ct_volume[np.isfinite(ct_volume)]
    if finite.size == 0:
        return "CT volume has no finite voxel intensity values."

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    p1_hu, p99_hu = np.percentile(finite, [1, 99]).astype(float).tolist()

    # If data range is near [0,1], the volume is likely normalized rather than true HU.
    if -0.2 <= p1_hu <= 1.2 and -0.2 <= p99_hu <= 1.2:
        return (
            "CT intensity range looks normalized (about 0-1) instead of HU. "
            f"Observed percentiles: p1={p1_hu:.3f}, p99={p99_hu:.3f}, min={vmin:.3f}, max={vmax:.3f}. "
            "HU-based nodule typing may be unreliable."
        )
    return ""


def stage_feature(case_dir: Path) -> Dict:
    state = _load_state(case_dir)
    ct_path = _require(state, "ct_path")
    mask_paths = state.get("mask_paths", [])
    out_dir = case_dir / "04_feature"
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "lesion_features.json"

    if not mask_paths:
        features: List[Dict] = []
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
        return {
            "feature_dir": str(out_dir),
            "features_path": str(features_path),
            "nodule_count": 0,
            "feature_status": "no_masks",
        }

    ct_volume, affine = MedSAM2Segmenter.load_ct_volume(ct_path)
    hu_warning = _detect_hu_scale_issue(ct_volume)

    spacing_x, spacing_y, spacing_z = _spacing_from_affine(affine)

    features: List[Dict] = []

    for i, p in enumerate(mask_paths, 1):
        m = nib.load(str(p)).get_fdata()
        feat = _compute_features_for_mask(
            ct_volume=ct_volume,
            mask=(m > 0),
            nodule_id=i,
            affine=affine,
            spacing_x=spacing_x,
            spacing_y=spacing_y,
            spacing_z=spacing_z,
            hu_warning=hu_warning or None,
        )
        if feat:
            features.append(feat)

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    result = {
        "feature_dir": str(out_dir),
        "features_path": str(features_path),
        "nodule_count": len(features),
    }
    if hu_warning:
        result["feature_warning"] = hu_warning
        result["feature_status"] = "hu_scale_suspect"
    return result


def stage_report(case_dir: Path, use_llm: bool) -> Dict:
    state = _load_state(case_dir)
    features_path = Path(_require(state, "features_path"))
    case_id = str(state.get("case_id", case_dir.name))
    report_id = f"AUTO_{case_id}"
    scan_date = datetime.now().strftime("%Y-%m-%d")

    with open(features_path, "r", encoding="utf-8") as f:
        features = json.load(f)

    out_dir = case_dir / "05_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    structured_input = build_structured_report_input(
        features,
        report_id=report_id,
        scan_date=scan_date,
    )
    structured_input_path = out_dir / "structured_input.json"
    with open(structured_input_path, "w", encoding="utf-8") as f:
        json.dump(structured_input, f, ensure_ascii=False, indent=2)

    if not features:
        # Keep pipeline executable even when segmentation produced empty masks.
        exam_assessment = structured_input["lung_rads"]["exam"]
        report = {
            "report_id": report_id,
            "scan_date": scan_date,
            "impression": "No measurable pulmonary nodule features were extracted.",
            "findings": [],
            "lung_rads": structured_input["lung_rads"],
            "note": "Feature list is empty; check segmentation masks and thresholds.",
        }

        txt_path = out_dir / "report.txt"
        json_path = out_dir / "report.json"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report["impression"] + "\n")
            f.write(f"Lung-RADS Category {exam_assessment['category']}: {exam_assessment['management']}\n")
            f.write(report["note"] + "\n")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        report_meta = {
            "text_path": str(txt_path),
            "json_path": str(json_path),
            "report_id": report["report_id"],
            "scan_date": report["scan_date"],
            "status": "empty_features",
            "generation_method": "template",
            "source_label": "Template-generated",
            "requested_llm": bool(use_llm),
            "fallback_reason": "",
            "structured_input_path": str(structured_input_path),
        }
        with open(out_dir / "report_meta.json", "w", encoding="utf-8") as f:
            json.dump(report_meta, f, ensure_ascii=False, indent=2)

        return {
            "report_dir": str(out_dir),
            "report_text_path": report_meta["text_path"],
            "report_json_path": report_meta["json_path"],
            "report_meta_path": str(out_dir / "report_meta.json"),
            "structured_input_path": str(structured_input_path),
            "report_status": report_meta["status"],
            "report_generation_method": report_meta["generation_method"],
            "report_source_label": report_meta["source_label"],
            "report_fallback_reason": report_meta["fallback_reason"],
        }

    report_generation_method = "template"
    report_status = "template"
    fallback_reason = ""

    if use_llm:
        try:
            generator = get_report_generator(use_llm=True)
            report = generator.generate_report(
                lesion_features=features,
                report_id=report_id,
                scan_date=scan_date,
                structured_input=structured_input,
                lung_rads_assessment=structured_input["lung_rads"],
            )
            if generator.__class__.__name__ == "ReportGenerator":
                report_generation_method = "llm"
                report_status = "llm"
            else:
                report_generation_method = "template"
                report_status = "template_fallback"
                fallback_reason = "LLM generator was unavailable during initialization."
        except Exception as exc:
            fallback_reason = str(exc)
            generator = get_report_generator(use_llm=False)
            report = generator.generate_report(
                lesion_features=features,
                report_id=report_id,
                scan_date=scan_date,
                structured_input=structured_input,
                lung_rads_assessment=structured_input["lung_rads"],
            )
            report_generation_method = "template"
            report_status = "template_fallback"
    else:
        generator = get_report_generator(use_llm=False)
        report = generator.generate_report(
            lesion_features=features,
            report_id=report_id,
            scan_date=scan_date,
            structured_input=structured_input,
            lung_rads_assessment=structured_input["lung_rads"],
        )

    saved = generator.save_report(report, str(out_dir), formats=["txt", "json"])

    report_source_label = "LLM-generated" if report_generation_method == "llm" else "Template-generated"
    if report_status == "template_fallback":
        report_source_label = "Template-generated (LLM fallback)"

    report_meta = {
        "text_path": saved.get("txt", ""),
        "json_path": saved.get("json", ""),
        "report_id": report.get("report_id", ""),
        "scan_date": report.get("scan_date", ""),
        "status": report_status,
        "generation_method": report_generation_method,
        "source_label": report_source_label,
        "requested_llm": bool(use_llm),
        "fallback_reason": fallback_reason,
        "structured_input_path": str(structured_input_path),
    }

    with open(out_dir / "report_meta.json", "w", encoding="utf-8") as f:
        json.dump(report_meta, f, ensure_ascii=False, indent=2)

    return {
        "report_dir": str(out_dir),
        "report_text_path": report_meta["text_path"],
        "report_json_path": report_meta["json_path"],
        "report_meta_path": str(out_dir / "report_meta.json"),
        "structured_input_path": str(structured_input_path),
        "report_status": report_meta["status"],
        "report_generation_method": report_meta["generation_method"],
        "report_source_label": report_meta["source_label"],
        "report_fallback_reason": report_meta["fallback_reason"],
    }


def _read_text(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def _load_json(path: str):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_seconds(value) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return ""
    if seconds >= 60:
        minutes = int(seconds // 60)
        rem = seconds - minutes * 60
        return f"{minutes}m {rem:.1f}s"
    return f"{seconds:.2f}s"


def _image_to_data_uri(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _normalize_ct_slice(slice_2d: np.ndarray) -> np.ndarray:
    window_min = -1000.0
    window_max = 400.0
    clipped = np.clip(slice_2d.astype(np.float32), window_min, window_max)
    scaled = (clipped - window_min) / (window_max - window_min)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _save_mask_overlay(
    ct_slice: np.ndarray,
    mask_slice: np.ndarray,
    out_path: Path,
    color: str,
) -> None:
    gray = _normalize_ct_slice(ct_slice)
    rgb = np.stack([gray, gray, gray], axis=-1)
    mask = mask_slice > 0
    if np.any(mask):
        if color == "blue":
            rgb[mask, 0] = (rgb[mask, 0] * 0.35).astype(np.uint8)
            rgb[mask, 1] = (rgb[mask, 1] * 0.55).astype(np.uint8)
            rgb[mask, 2] = np.maximum(rgb[mask, 2], 230)
        else:
            rgb[mask, 0] = np.maximum(rgb[mask, 0], 230)
            rgb[mask, 1] = (rgb[mask, 1] * 0.35).astype(np.uint8)
            rgb[mask, 2] = (rgb[mask, 2] * 0.35).astype(np.uint8)

    # NIfTI volumes are stored as x, y, z. Rotate for a readable axial preview.
    preview = np.flipud(np.rot90(rgb))
    Image.fromarray(preview).save(out_path)


def _generate_mask_previews(
    case_dir: Path,
    state: Dict,
    mask_state_key: str,
    out_dir_state_key: str,
    default_subdir: str,
    filename_prefix: str,
    color: str,
    limit: int = 8,
) -> List[str]:
    ct_path = Path(str(state.get("ct_path", "")))
    mask_path = Path(str(state.get(mask_state_key, "")))
    if not ct_path.exists() or not mask_path.exists():
        return []

    try:
        ct = np.asarray(nib.load(str(ct_path)).get_fdata(), dtype=np.float32)
        mask = np.asarray(nib.load(str(mask_path)).get_fdata() > 0, dtype=np.uint8)
    except Exception:
        return []

    if ct.ndim != 3 or mask.ndim != 3 or ct.shape != mask.shape:
        return []

    z_area = mask.sum(axis=(0, 1))
    z_indices = np.where(z_area > 0)[0]
    if z_indices.size == 0:
        return []

    strongest = sorted(z_indices.tolist(), key=lambda z: int(z_area[z]), reverse=True)[:limit]
    selected = sorted(strongest)

    out_dir = Path(str(state.get(out_dir_state_key, case_dir / default_subdir))) / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    preview_paths: List[str] = []
    for z in selected:
        out_path = out_dir / f"{filename_prefix}_z{int(z):03d}.png"
        _save_mask_overlay(ct[:, :, z], mask[:, :, z], out_path, color=color)
        preview_paths.append(str(out_path))

    return preview_paths


def _generate_lung_previews(case_dir: Path, state: Dict) -> List[str]:
    return _generate_mask_previews(
        case_dir=case_dir,
        state=state,
        mask_state_key="lung_mask_path",
        out_dir_state_key="preprocess_dir",
        default_subdir="01_preprocess",
        filename_prefix="lung_mask_overlay",
        color="blue",
        limit=8,
    )


def _generate_segmentation_previews(case_dir: Path, state: Dict) -> List[str]:
    return _generate_mask_previews(
        case_dir=case_dir,
        state=state,
        mask_state_key="combined_mask_path",
        out_dir_state_key="segment_dir",
        default_subdir="03_segment",
        filename_prefix="segmentation_overlay",
        color="red",
        limit=8,
    )


def _generate_ct_viewer(case_dir: Path, state: Dict) -> str:
    ct_path = Path(str(state.get("ct_path", "")))
    if not ct_path.exists():
        return ""

    out_dir = case_dir / "05_report"
    viewer_dir = out_dir / "ct_viewer"
    slices_dir = viewer_dir / "slices"
    axial_dir = slices_dir / "axial"
    coronal_dir = slices_dir / "coronal"
    sagittal_dir = slices_dir / "sagittal"
    for plane_dir in [axial_dir, coronal_dir, sagittal_dir]:
        plane_dir.mkdir(parents=True, exist_ok=True)
    viewer_path = out_dir / "ct_viewer.html"

    try:
        ct_img = nib.load(str(ct_path))
        ct = np.asarray(ct_img.get_fdata(), dtype=np.float32)
        spacing_x, spacing_y, spacing_z = _spacing_from_affine(ct_img.affine)
    except Exception:
        return ""

    if ct.ndim != 3 or min(ct.shape) <= 0:
        return ""

    mask = None
    mask_path = Path(str(state.get("combined_mask_path", "")))
    if mask_path.exists():
        try:
            loaded_mask = np.asarray(nib.load(str(mask_path)).get_fdata() > 0, dtype=np.uint8)
            if loaded_mask.shape == ct.shape:
                mask = loaded_mask
        except Exception:
            mask = None

    def make_plane_preview(
        ct_slice: np.ndarray,
        row_spacing: float,
        col_spacing: float,
        mask_slice: Optional[np.ndarray] = None,
        flip_vertical: bool = False,
    ) -> Image.Image:
        gray = _normalize_ct_slice(ct_slice)
        rgb = np.stack([gray, gray, gray], axis=-1)
        if mask_slice is not None and np.any(mask_slice > 0):
            m = mask_slice > 0
            rgb[m, 0] = np.maximum(rgb[m, 0], 235)
            rgb[m, 1] = (rgb[m, 1] * 0.35).astype(np.uint8)
            rgb[m, 2] = (rgb[m, 2] * 0.35).astype(np.uint8)

        preview = np.flipud(np.rot90(rgb))
        if flip_vertical:
            preview = np.flipud(preview)
        image = Image.fromarray(preview)
        out_width = max(1, int(round(image.width * float(row_spacing))))
        out_height = max(1, int(round(image.height * float(col_spacing))))
        min_scale = min(row_spacing, col_spacing)
        if min_scale > 0:
            out_width = max(1, int(round(out_width / min_scale)))
            out_height = max(1, int(round(out_height / min_scale)))
        return image.resize((out_width, out_height), Image.Resampling.BILINEAR)

    def save_plane_slices(
        axis: str,
        count: int,
        get_slice: Callable[[int], np.ndarray],
        get_mask_slice: Callable[[int], Optional[np.ndarray]],
        row_spacing: float,
        col_spacing: float,
        flip_vertical: bool = False,
    ) -> List[str]:
        paths: List[str] = []
        for idx in range(count):
            preview = make_plane_preview(
                get_slice(idx),
                row_spacing=row_spacing,
                col_spacing=col_spacing,
                mask_slice=get_mask_slice(idx),
                flip_vertical=flip_vertical,
            )
            rel_path = f"ct_viewer/slices/{axis}/slice_{idx:03d}.jpg"
            out_path = out_dir / rel_path
            preview.save(out_path, quality=90)
            paths.append(rel_path)
        return paths

    axial_paths = save_plane_slices(
        "axial",
        ct.shape[2],
        lambda z: ct[:, :, z],
        lambda z: mask[:, :, z] if mask is not None else None,
        spacing_x,
        spacing_y,
    )
    coronal_paths = save_plane_slices(
        "coronal",
        ct.shape[1],
        lambda y: ct[:, y, :],
        lambda y: mask[:, y, :] if mask is not None else None,
        spacing_x,
        spacing_z,
        flip_vertical=True,
    )
    sagittal_paths = save_plane_slices(
        "sagittal",
        ct.shape[0],
        lambda x: ct[x, :, :],
        lambda x: mask[x, :, :] if mask is not None else None,
        spacing_y,
        spacing_z,
        flip_vertical=True,
    )
    max_slider = 1000

    case_id = html.escape(str(state.get("case_id", case_dir.name)))
    axial_json = json.dumps(axial_paths)
    coronal_json = json.dumps(coronal_paths)
    sagittal_json = json.dumps(sagittal_paths)
    detection_data = _load_json(str(state.get("detection_report", ""))) or {}
    nodule_targets = []
    for nodule in detection_data.get("nodules", []):
        box = nodule.get("box_voxel") or []
        if len(box) != 6:
            continue
        x1, y1, z1, x2, y2, z2 = [_safe_float(v) for v in box]
        axial_index = int(round((z1 + z2) / 2.0))
        coronal_index = int(round((y1 + y2) / 2.0))
        sagittal_index = int(round((x1 + x2) / 2.0))
        location = nodule.get("anatomical_location") or {}
        nodule_targets.append(
            {
                "id": int(nodule.get("id", len(nodule_targets) + 1)),
                "label": f"Nodule {int(nodule.get('id', len(nodule_targets) + 1))}",
                "location": str(location.get("lobe_full") or location.get("lobe") or ""),
                "diameter": round(_safe_float(nodule.get("approx_diameter_mm")), 1),
                "slider": int(round(max_slider * axial_index / max(len(axial_paths) - 1, 1))),
                "axial": max(0, min(len(axial_paths) - 1, axial_index)),
                "coronal": max(0, min(len(coronal_paths) - 1, coronal_index)),
                "sagittal": max(0, min(len(sagittal_paths) - 1, sagittal_index)),
            }
        )
    targets_json = json.dumps(nodule_targets)
    viewer_doc = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CT Review - {case_id}</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft JhengHei", sans-serif; color: #1f2933; background: #f6f8fb; }}
    header {{ background: #17324d; color: white; padding: 22px 32px; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 24px; }}
    .viewer {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 18px; }}
    .planes {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .plane {{ background: #111827; border-radius: 6px; padding: 12px; }}
    .plane h2 {{ margin: 0 0 8px; color: white; font-size: 16px; }}
    .legend {{ margin: 0 0 12px; color: #52606d; }}
    .legend span {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; background: #eb2f2f; vertical-align: -1px; margin-right: 6px; }}
    img {{ max-width: 100%; height: auto; image-rendering: auto; }}
    .controls {{ display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: center; margin-top: 14px; }}
    input[type="range"] {{ width: 100%; }}
    .counter {{ min-width: 300px; font-weight: 700; text-align: right; }}
    .nodule-list {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 14px; }}
    .nodule-list button {{ border: 1px solid #9fb3c8; background: #f0f4f8; border-radius: 6px; padding: 8px 10px; color: #1f2933; cursor: pointer; font-weight: 700; text-align: left; }}
    .nodule-list button:hover {{ border-color: #0f766e; background: #e6fffa; }}
    a {{ color: #0f766e; font-weight: 700; text-decoration: none; }}
    @media (max-width: 900px) {{ .planes {{ grid-template-columns: 1fr; }} .counter {{ min-width: 0; text-align: left; }} .controls {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>CT Review</h1>
    <div>Case: {case_id}</div>
  </header>
  <main>
    <p><a href="index.html">Back to summary</a></p>
    <section class="viewer">
      <p class="legend"><span></span>Red overlay indicates segmentation mask.</p>
      <div class="nodule-list" id="noduleList"></div>
      <div class="planes">
        <div class="plane">
          <h2>Axial</h2>
          <img id="axialImage" alt="Axial CT slice">
        </div>
        <div class="plane">
          <h2>Coronal</h2>
          <img id="coronalImage" alt="Coronal CT slice">
        </div>
        <div class="plane">
          <h2>Sagittal</h2>
          <img id="sagittalImage" alt="Sagittal CT slice">
        </div>
      </div>
      <div class="controls">
        <input id="sliceSlider" type="range" min="0" max="{max_slider}" value="{max_slider // 2}">
        <div class="counter" id="sliceCounter"></div>
      </div>
    </section>
  </main>
  <script>
    const axial = {axial_json};
    const coronal = {coronal_json};
    const sagittal = {sagittal_json};
    const noduleTargets = {targets_json};
    const axialImage = document.getElementById('axialImage');
    const coronalImage = document.getElementById('coronalImage');
    const sagittalImage = document.getElementById('sagittalImage');
    const slider = document.getElementById('sliceSlider');
    const counter = document.getElementById('sliceCounter');
    const noduleList = document.getElementById('noduleList');
    function indexFor(list) {{
      return Math.round((Number(slider.value) / {max_slider}) * (list.length - 1));
    }}
    function render() {{
      const axialIndex = indexFor(axial);
      const coronalIndex = indexFor(coronal);
      const sagittalIndex = indexFor(sagittal);
      axialImage.src = axial[axialIndex];
      coronalImage.src = coronal[coronalIndex];
      sagittalImage.src = sagittal[sagittalIndex];
      counter.textContent = `Axial ${{axialIndex + 1}}/${{axial.length}} | Coronal ${{coronalIndex + 1}}/${{coronal.length}} | Sagittal ${{sagittalIndex + 1}}/${{sagittal.length}}`;
    }}
    slider.addEventListener('input', render);
    noduleTargets.forEach((target) => {{
      const button = document.createElement('button');
      const detail = target.location ? ` | ${{target.location}}` : '';
      button.textContent = `${{target.label}} | ${{target.diameter}} mm${{detail}}`;
      button.addEventListener('click', () => {{
        slider.value = String(target.slider);
        render();
      }});
      noduleList.appendChild(button);
    }});
    render();
  </script>
</body>
</html>
"""

    viewer_path.write_text(viewer_doc, encoding="utf-8")
    return str(viewer_path)


def _html_table(headers: List[str], rows: List[List[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _detection_bbox_metrics(nodule: Dict, spacing: List[float]) -> Dict:
    box = nodule.get("box_voxel") or []
    if len(box) != 6:
        return {
            "bbox_x_mm": 0.0,
            "bbox_y_mm": 0.0,
            "bbox_z_mm": 0.0,
            "bbox_volume_mm3": 0.0,
            "bbox_longest_axis_mm": 0.0,
        }

    sx = _safe_float(spacing[0] if len(spacing) > 0 else 1.0, 1.0)
    sy = _safe_float(spacing[1] if len(spacing) > 1 else 1.0, 1.0)
    sz = _safe_float(spacing[2] if len(spacing) > 2 else 1.0, 1.0)
    x1, y1, z1, x2, y2, z2 = [_safe_float(v) for v in box]
    bbox_x_mm = abs(x2 - x1) * sx
    bbox_y_mm = abs(y2 - y1) * sy
    bbox_z_mm = abs(z2 - z1) * sz
    return {
        "bbox_x_mm": bbox_x_mm,
        "bbox_y_mm": bbox_y_mm,
        "bbox_z_mm": bbox_z_mm,
        "bbox_volume_mm3": bbox_x_mm * bbox_y_mm * bbox_z_mm,
        "bbox_longest_axis_mm": max(bbox_x_mm, bbox_y_mm, bbox_z_mm),
    }


def _build_detection_segmentation_comparison_rows(detection: Dict, features: List[Dict]) -> List[List[str]]:
    feature_by_id = {
        int(feat.get("nodule_id")): feat
        for feat in features
        if isinstance(feat, dict) and str(feat.get("nodule_id", "")).isdigit()
    }
    spacing = detection.get("spacing") if isinstance(detection, dict) else None
    if not isinstance(spacing, list):
        spacing = []

    rows: List[List[str]] = []
    for nodule in detection.get("nodules", []):
        if not isinstance(nodule, dict):
            continue
        nodule_id = int(nodule.get("id", 0) or 0)
        feat = feature_by_id.get(nodule_id, {})
        det_metrics = _detection_bbox_metrics(nodule, spacing)

        det_diam = _safe_float(nodule.get("approx_diameter_mm"))
        det_longest = _safe_float(det_metrics.get("bbox_longest_axis_mm"))
        seg_diam = _safe_float(feat.get("equivalent_diameter_mm"))
        seg_longest = _safe_float(feat.get("longest_axis_mm"))
        seg_volume = _safe_float(feat.get("volume_mm3"))
        attenuation_type = str(feat.get("attenuation_type", ""))
        location = nodule.get("anatomical_location") or {}

        rows.append(
            [
                html.escape(str(nodule_id)),
                html.escape(str(location.get("lobe_full") or location.get("lobe") or "")),
                html.escape(f"{det_diam:.2f}"),
                html.escape(f"{det_longest:.2f}"),
                html.escape(f"{seg_longest:.2f}"),
                html.escape(f"{seg_diam:.2f}"),
                html.escape(f"{seg_volume:.2f}"),
                html.escape(attenuation_type),
            ]
        )

    return rows


def generate_summary_html(case_dir: Path, state: Dict) -> Dict:
    out_dir = case_dir / "05_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.html"
    legacy_summary_path = out_dir / "summary.html"

    report_text = _read_text(str(state.get("report_text_path", "")))
    features = _load_json(str(state.get("features_path", ""))) or []
    detection = _load_json(str(state.get("detection_report", ""))) or {}
    report_meta = _load_json(str(state.get("report_meta_path", ""))) or {}
    timings = state.get("process_times", {})
    segmentation_previews = _generate_segmentation_previews(case_dir, state)
    if segmentation_previews:
        state["segmentation_preview_paths"] = segmentation_previews
    ct_viewer_path = _generate_ct_viewer(case_dir, state)
    if ct_viewer_path:
        state["ct_viewer_html_path"] = ct_viewer_path

    timing_rows = []
    for stage in ["preprocess", "detect", "segment", "feature", "report"]:
        item = timings.get(stage, {})
        timing_rows.append(
            [
                html.escape(stage.title()),
                html.escape(str(item.get("status", ""))),
                html.escape(_format_seconds(item.get("elapsed_seconds"))),
            ]
        )

    feature_rows = []
    for feat in features:
        feature_rows.append(
            [
                html.escape(str(feat.get("nodule_id", ""))),
                html.escape(f"{float(feat.get('equivalent_diameter_mm', 0.0)):.2f}"),
                html.escape(f"{float(feat.get('longest_axis_mm', 0.0)):.2f}"),
                html.escape(f"{float(feat.get('volume_mm3', 0.0)):.2f}"),
                html.escape(str(feat.get("attenuation_type", ""))),
            ]
        )
    comparison_rows = _build_detection_segmentation_comparison_rows(detection, features)

    detection_cards = []
    viz_dir = Path(str(state.get("detection_dir", ""))) / "viz"
    kept_detection_source_ids = {
        int(n.get("source_detection_id", n.get("id")))
        for n in detection.get("nodules", [])
        if isinstance(n, dict) and str(n.get("source_detection_id", n.get("id", ""))).isdigit()
    }
    apply_detection_viz_filter = bool((state.get("detection_lung_filter") or {}).get("enabled"))
    for image_path in sorted(viz_dir.glob("*.png"))[:12]:
        match = re.search(r"_pred_(\d+)_", image_path.name)
        if apply_detection_viz_filter and match and int(match.group(1)) not in kept_detection_source_ids:
            continue
        data_uri = _image_to_data_uri(image_path)
        if not data_uri:
            continue
        display_index = len(detection_cards) + 1
        detection_cards.append(
            "<figure>"
            f"<img src=\"{data_uri}\" alt=\"Detection image {display_index}\">"
            f"<figcaption>Detection view {display_index}</figcaption>"
            "</figure>"
        )

    segmentation_cards = []
    for preview in state.get("segmentation_preview_paths", []):
        image_path = Path(str(preview))
        data_uri = _image_to_data_uri(image_path)
        if not data_uri:
            continue
        display_index = len(segmentation_cards) + 1
        segmentation_cards.append(
            "<figure>"
            f"<img src=\"{data_uri}\" alt=\"Segmentation image {display_index}\">"
            f"<figcaption>Segmentation view {display_index}</figcaption>"
            "</figure>"
        )

    generated_at = _now_iso()
    total_time = _format_seconds(state.get("total_process_seconds"))
    nodule_count = state.get("nodule_count", len(features))
    largest_longest = max([_safe_float(feat.get("longest_axis_mm")) for feat in features] or [0.0])
    report_source_label = str(
        report_meta.get("source_label")
        or state.get("report_source_label")
        or "Template-generated"
    )
    report_fallback_reason = str(report_meta.get("fallback_reason") or state.get("report_fallback_reason") or "")
    fallback_note = (
        f"<p class=\"note\">LLM unavailable; template fallback used. Reason: {html.escape(report_fallback_reason)}</p>"
        if report_fallback_reason
        else ""
    )

    doc = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chest CT Summary - {html.escape(str(state.get("case_id", case_dir.name)))}</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft JhengHei", sans-serif; color: #1f2933; background: #f6f8fb; }}
    header {{ background: #17324d; color: white; padding: 28px 40px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 24px 48px; }}
    section {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
    .metric {{ background: #eef4f8; border-radius: 6px; padding: 12px; }}
    .metric span {{ display: block; color: #52606d; font-size: 12px; text-transform: uppercase; }}
    .metric strong {{ display: block; margin-top: 6px; font-size: 22px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e6edf3; padding: 9px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; color: #334e68; }}
    pre {{ white-space: pre-wrap; background: #f8fafc; border: 1px solid #d9e2ec; border-radius: 6px; padding: 14px; line-height: 1.45; }}
    .images {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; border: 1px solid #d9e2ec; border-radius: 6px; overflow: hidden; background: #fff; }}
    img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ padding: 8px 10px; color: #52606d; font-size: 13px; }}
    .note {{ color: #52606d; margin: 0 0 12px; }}
    .paths td:first-child {{ width: 160px; font-weight: 700; color: #334e68; }}
    .table-scroll {{ overflow-x: auto; }}
    .review-link {{ display: inline-block; background: #0f766e; color: white; border-radius: 6px; padding: 10px 14px; font-weight: 700; text-decoration: none; margin-right: 8px; }}
    .source-pill {{ display: inline-block; background: #eef4f8; color: #334e68; border: 1px solid #bcccdc; border-radius: 999px; padding: 6px 10px; font-size: 13px; font-weight: 700; margin-bottom: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>Chest CT Summary</h1>
    <div>Case: {html.escape(str(state.get("case_id", case_dir.name)))} | Generated: {html.escape(generated_at)}</div>
  </header>
  <main>
    <section class="meta">
      <div class="metric"><span>Nodules</span><strong>{html.escape(str(nodule_count))}</strong></div>
      <div class="metric"><span>Largest measured diameter</span><strong>{html.escape(f"{largest_longest:.1f} mm")}</strong></div>
      <div class="metric"><span>Total process time</span><strong>{html.escape(total_time)}</strong></div>
    </section>
    <section>
      <h2>Clinical Report</h2>
      <div class="source-pill">Report generation: {html.escape(report_source_label)}</div>
      {fallback_note}
      <pre>{html.escape(report_text) if report_text else "No report text found."}</pre>
    </section>
    <section>
      <h2>Full CT Review</h2>
      <p class="note">Open synchronized axial, coronal, and sagittal CT views with segmentation mask overlay.</p>
      {"<a class=\"review-link\" href=\"ct_viewer.html\">Review Full CT</a>" if ct_viewer_path else "<p>CT viewer was not generated.</p>"}
    </section>
    <section>
      <h2>Nodule Measurements</h2>
      {_html_table(["ID", "Equivalent diameter mm", "Longest diameter mm", "Volume mm3", "Attenuation"], feature_rows) if feature_rows else "<p>No extracted features.</p>"}
    </section>
    <section>
      <h2>Detection vs Segmentation Measurements</h2>
      <div class="table-scroll">
        {_html_table(["ID", "Location", "Detection diameter mm", "Detection longest mm", "Segmentation longest mm", "Segmentation equivalent diameter mm", "Segmentation volume mm3", "Attenuation"], comparison_rows) if comparison_rows else "<p>No comparison data found.</p>"}
      </div>
    </section>
    <section>
      <h2>Detection Images</h2>
      <div class="images">{''.join(detection_cards) if detection_cards else "<p>No detection images found.</p>"}</div>
    </section>
    <section>
      <h2>Segmentation Images</h2>
      <p class="note">Red overlay shows the combined segmentation mask on axial CT slices.</p>
      <div class="images">{''.join(segmentation_cards) if segmentation_cards else "<p>No segmentation images found.</p>"}</div>
    </section>
    <section>
      <h2>Pipeline Process Time</h2>
      {_html_table(["Stage", "Status", "Elapsed"], timing_rows)}
    </section>
  </main>
</body>
</html>
"""

    index_path.write_text(doc, encoding="utf-8")
    legacy_summary_path.write_text(doc, encoding="utf-8")
    return {
        "summary_html_path": str(index_path),
        "legacy_summary_html_path": str(legacy_summary_path),
        "summary_generated_at": generated_at,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless chest CT case pipeline")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["preprocess", "detect", "segment", "feature", "report", "run"],
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--work-dir", default=str(SCRIPT_DIR / "runtime"))

    # Stage options
    parser.add_argument("--input-path", default="")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--medsam2-checkpoint", default="")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--no-propagate", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    work_dir = Path(args.work_dir)
    case_dir = _case_dir(work_dir, args.case_id)
    state = _load_state(case_dir)
    state["case_id"] = args.case_id
    state["work_dir"] = str(work_dir)
    if args.stage in {"detect", "run"} or "threshold" not in state:
        state["threshold"] = float(args.threshold)
    if args.stage in {"detect", "run"} or "device" not in state:
        state["device"] = str(args.device)
    if args.stage in {"report", "run"} or args.use_llm or "use_llm" not in state:
        state["use_llm"] = bool(args.use_llm)
    if args.stage in {"segment", "run"} or args.no_propagate or "propagate" not in state:
        state["propagate"] = not args.no_propagate

    if args.stage in {"preprocess", "run"}:
        if not args.input_path and "input_path" not in state:
            raise ValueError("--input-path is required for preprocess/run when state has no input_path")

    if args.stage in {"detect", "run"}:
        if not args.model_path and "detection_model_path" not in state:
            raise ValueError("--model-path is required for detect/run when state has no detection_model_path")

    if args.stage == "preprocess":
        state.update(
            _run_stage_timed(
                case_dir,
                state,
                "preprocess",
                lambda: stage_preprocess(case_dir, args.input_path or state["input_path"]),
            )
        )

    elif args.stage == "detect":
        model_path = args.model_path or state["detection_model_path"]
        state.update(
            _run_stage_timed(
                case_dir,
                state,
                "detect",
                lambda: stage_detect(case_dir, model_path, args.threshold, args.device),
            )
        )

    elif args.stage == "segment":
        state.update(
            _run_stage_timed(
                case_dir,
                state,
                "segment",
                lambda: stage_segment(case_dir, args.medsam2_checkpoint, propagate=not args.no_propagate),
            )
        )

    elif args.stage == "feature":
        state.update(
            _run_stage_timed(case_dir, state, "feature", lambda: stage_feature(case_dir))
        )

    elif args.stage == "report":
        state.update(
            _run_stage_timed(
                case_dir,
                state,
                "report",
                lambda: stage_report(case_dir, use_llm=args.use_llm),
            )
        )
        state.update(generate_summary_html(case_dir, state))

    elif args.stage == "run":
        state.update(
            _run_stage_timed(
                case_dir,
                state,
                "preprocess",
                lambda: stage_preprocess(case_dir, args.input_path or state["input_path"]),
            )
        )
        _save_state(case_dir, state)
        model_path = args.model_path or state["detection_model_path"]
        state.update(
            _run_stage_timed(
                case_dir,
                state,
                "detect",
                lambda: stage_detect(case_dir, model_path, args.threshold, args.device),
            )
        )
        _save_state(case_dir, state)
        state.update(
            _run_stage_timed(
                case_dir,
                state,
                "segment",
                lambda: stage_segment(case_dir, args.medsam2_checkpoint, propagate=not args.no_propagate),
            )
        )
        _save_state(case_dir, state)
        state.update(
            _run_stage_timed(case_dir, state, "feature", lambda: stage_feature(case_dir))
        )
        _save_state(case_dir, state)
        state.update(
            _run_stage_timed(
                case_dir,
                state,
                "report",
                lambda: stage_report(case_dir, use_llm=args.use_llm),
            )
        )
        state.update(generate_summary_html(case_dir, state))

    _save_state(case_dir, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

