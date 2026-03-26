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
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import SimpleITK as sitk


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PIPELINE_ROOT = PROJECT_ROOT / "llm" / "ct_report_pipeline"

# Ensure local pipeline modules resolve first.
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config, get_medsam2_checkpoint, get_medsam2_root
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

    summary = {
        "input_path": str(p),
        "ct_path": str(out_ct),
        "shape_dhw": [int(x) for x in arr.shape],
        "spacing_xyz": [float(x) for x in spacing],
        "origin_xyz": [float(x) for x in origin],
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "input_path": str(p),
        "ct_path": str(out_ct),
        "preprocess_summary": str(out_dir / "summary.json"),
    }


def stage_detect(
    case_dir: Path,
    model_path: str,
    threshold: float,
    device: str,
) -> Dict:
    state = _load_state(case_dir)
    ct_path = _require(state, "ct_path")
    source_input_path = str(state.get("input_path", ""))

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

    gt_label_path = _infer_gt_label_path_from_input(source_input_path)
    if gt_label_path:
        cmd.extend(["--gt_label_path", gt_label_path])

    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

    detect_report = out_dir / "report.json"
    if not detect_report.exists():
        raise FileNotFoundError(f"Detection report not found: {detect_report}")

    return {
        "detection_dir": str(out_dir),
        "detection_report": str(detect_report),
        "detection_model_path": str(model_path),
        "detection_gt_label_path": gt_label_path,
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


def _compute_features_for_mask(
    ct_volume: np.ndarray,
    mask: np.ndarray,
    nodule_id: int,
    spacing_x: float,
    spacing_y: float,
    spacing_z: float,
) -> Dict:
    vox = np.argwhere(mask > 0)
    if vox.size == 0:
        return {}

    voxel_count = int(vox.shape[0])
    volume_mm3 = voxel_count * spacing_x * spacing_y * spacing_z
    equivalent_diameter_mm = 2.0 * (3.0 * volume_mm3 / (4.0 * math.pi)) ** (1.0 / 3.0)

    z_min, y_min, x_min = vox.min(axis=0).tolist()
    z_max, y_max, x_max = vox.max(axis=0).tolist()

    bbox_x_mm = (x_max - x_min + 1) * spacing_x
    bbox_y_mm = (y_max - y_min + 1) * spacing_y
    bbox_z_mm = (z_max - z_min + 1) * spacing_z

    values = ct_volume[mask > 0]

    return {
        "nodule_id": int(nodule_id),
        "voxel_count": voxel_count,
        "volume_mm3": float(volume_mm3),
        "equivalent_diameter_mm": float(equivalent_diameter_mm),
        "longest_axis_mm": float(max(bbox_x_mm, bbox_y_mm, bbox_z_mm)),
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
        "spacing_mm": [float(spacing_x), float(spacing_y), float(spacing_z)],
    }


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

    spacing_x = float(abs(affine[0, 0])) if affine.shape[0] > 0 else 1.0
    spacing_y = float(abs(affine[1, 1])) if affine.shape[0] > 1 else 1.0
    spacing_z = float(abs(affine[2, 2])) if affine.shape[0] > 2 else 1.0

    features: List[Dict] = []

    for i, p in enumerate(mask_paths, 1):
        m = nib.load(str(p)).get_fdata()
        feat = _compute_features_for_mask(
            ct_volume=ct_volume,
            mask=(m > 0),
            nodule_id=i,
            spacing_x=spacing_x,
            spacing_y=spacing_y,
            spacing_z=spacing_z,
        )
        if feat:
            features.append(feat)

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    return {
        "feature_dir": str(out_dir),
        "features_path": str(features_path),
        "nodule_count": len(features),
    }


def stage_report(case_dir: Path, use_llm: bool) -> Dict:
    state = _load_state(case_dir)
    features_path = Path(_require(state, "features_path"))
    case_id = str(state.get("case_id", case_dir.name))

    with open(features_path, "r", encoding="utf-8") as f:
        features = json.load(f)

    out_dir = case_dir / "05_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not features:
        # Keep pipeline executable even when segmentation produced empty masks.
        report = {
            "report_id": f"AUTO_{case_id}",
            "scan_date": datetime.now().strftime("%Y-%m-%d"),
            "impression": "No measurable pulmonary nodule features were extracted.",
            "findings": [],
            "note": "Feature list is empty; check segmentation masks and thresholds.",
        }

        txt_path = out_dir / "report.txt"
        json_path = out_dir / "report.json"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report["impression"] + "\n")
            f.write(report["note"] + "\n")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        report_meta = {
            "text_path": str(txt_path),
            "json_path": str(json_path),
            "report_id": report["report_id"],
            "scan_date": report["scan_date"],
            "status": "empty_features",
        }
        with open(out_dir / "report_meta.json", "w", encoding="utf-8") as f:
            json.dump(report_meta, f, ensure_ascii=False, indent=2)

        return {
            "report_dir": str(out_dir),
            "report_text_path": report_meta["text_path"],
            "report_json_path": report_meta["json_path"],
            "report_meta_path": str(out_dir / "report_meta.json"),
            "report_status": report_meta["status"],
        }

    generator = get_report_generator(use_llm=use_llm)
    report = generator.generate_report(
        lesion_features=features,
        report_id=f"AUTO_{case_id}",
    )

    saved = generator.save_report(report, str(out_dir), formats=["txt", "json"])

    report_meta = {
        "text_path": saved.get("txt", ""),
        "json_path": saved.get("json", ""),
        "report_id": report.get("report_id", ""),
        "scan_date": report.get("scan_date", ""),
    }

    with open(out_dir / "report_meta.json", "w", encoding="utf-8") as f:
        json.dump(report_meta, f, ensure_ascii=False, indent=2)

    return {
        "report_dir": str(out_dir),
        "report_text_path": report_meta["text_path"],
        "report_json_path": report_meta["json_path"],
        "report_meta_path": str(out_dir / "report_meta.json"),
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

    if args.stage in {"preprocess", "run"}:
        if not args.input_path and "input_path" not in state:
            raise ValueError("--input-path is required for preprocess/run when state has no input_path")

    if args.stage in {"detect", "run"}:
        if not args.model_path and "detection_model_path" not in state:
            raise ValueError("--model-path is required for detect/run when state has no detection_model_path")

    if args.stage == "preprocess":
        state.update(stage_preprocess(case_dir, args.input_path or state["input_path"]))

    elif args.stage == "detect":
        model_path = args.model_path or state["detection_model_path"]
        state.update(stage_detect(case_dir, model_path, args.threshold, args.device))

    elif args.stage == "segment":
        state.update(stage_segment(case_dir, args.medsam2_checkpoint, propagate=not args.no_propagate))

    elif args.stage == "feature":
        state.update(stage_feature(case_dir))

    elif args.stage == "report":
        state.update(stage_report(case_dir, use_llm=args.use_llm))

    elif args.stage == "run":
        state.update(stage_preprocess(case_dir, args.input_path or state["input_path"]))
        model_path = args.model_path or state["detection_model_path"]
        state.update(stage_detect(case_dir, model_path, args.threshold, args.device))
        state.update(stage_segment(case_dir, args.medsam2_checkpoint, propagate=not args.no_propagate))
        state.update(stage_feature(case_dir))
        state.update(stage_report(case_dir, use_llm=args.use_llm))

    _save_state(case_dir, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

