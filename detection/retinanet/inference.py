#!/usr/bin/env python3
"""
RetinaNet inference pipeline for chest CT nodule detection.

Outputs:
- report.json
- visualization images under output_dir/viz
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from detection.common.location_estimator import LungLocationEstimator
from detection.retinanet.config import RetinaNetConfig
from detection.retinanet.dataset import build_val_transform
from detection.retinanet.trainer import RetinaNetTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_ct_volume(path: Union[str, Path]) -> Tuple[sitk.Image, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load DICOM directory or medical image file.
    Returns: (sitk_image, array[D,H,W], spacing[x,y,z], origin[x,y,z])
    """
    path = Path(path)
    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(path))
        if not series_ids:
            raise ValueError(f"No DICOM series found in: {path}")
        dicom_names = reader.GetGDCMSeriesFileNames(str(path), series_ids[0])
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(str(path))

    array = sitk.GetArrayFromImage(image)  # (D,H,W)
    spacing = np.array(image.GetSpacing())  # (x,y,z)
    origin = np.array(image.GetOrigin())
    return image, array, spacing, origin


def load_preprocessed_npz(path: Union[str, Path]) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, Dict]:
    data = np.load(path)
    frames = data["frames"]
    if frames.dtype == np.uint8:
        frames = frames.astype(np.float32) / 255.0

    tensor = torch.from_numpy(frames).unsqueeze(0).float()  # (1,D,H,W)
    spacing = data["spacing"]
    origin = data["origin"]

    scale_info = {
        "scale": 1.0,
        "pad_top": 0,
        "pad_left": 0,
        "original_shape": frames.shape,
        "is_preprocessed": True,
    }
    return tensor, spacing, origin, scale_info


def preprocess_volume(
    array: np.ndarray,
    target_size: int = 256,
    window_center: float = -600,
    window_width: float = 1500,
) -> Tuple[torch.Tensor, Dict]:
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    vol_norm = np.clip(array, img_min, img_max)
    vol_norm = (vol_norm - img_min) / (img_max - img_min)
    vol_norm = vol_norm.astype(np.float32)

    d, h, w = vol_norm.shape
    max_dim = max(h, w)
    pad_h = max_dim - h
    pad_w = max_dim - w
    pad_top = pad_h // 2
    pad_left = pad_w // 2

    if pad_h > 0 or pad_w > 0:
        padded = np.zeros((d, max_dim, max_dim), dtype=np.float32)
        padded[:, pad_top:pad_top + h, pad_left:pad_left + w] = vol_norm
        vol_norm = padded

    scale = 1.0
    if max_dim != target_size:
        scale = target_size / max_dim
        vol_resized = ndimage.zoom(vol_norm, (1.0, scale, scale), order=1, mode="constant", cval=0.0)
    else:
        vol_resized = vol_norm

    tensor = torch.from_numpy(vol_resized).unsqueeze(0).float()  # (1,D,H,W)
    scale_info = {
        "original_shape": (d, h, w),
        "padded_shape": (d, max_dim, max_dim),
        "resized_shape": vol_resized.shape,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "scale": scale,
        "max_dim": max_dim,
    }
    return tensor, scale_info


def rescale_boxes(boxes: np.ndarray, scale_info: Dict) -> np.ndarray:
    if len(boxes) == 0:
        return boxes

    boxes = boxes.copy()
    scale = scale_info["scale"]
    pad_top = scale_info["pad_top"]
    pad_left = scale_info["pad_left"]

    boxes[:, [0, 1, 3, 4]] /= scale
    boxes[:, [0, 3]] -= pad_left
    boxes[:, [1, 4]] -= pad_top

    d, h, w = scale_info["original_shape"]
    boxes[:, [0, 3]] = np.clip(boxes[:, [0, 3]], 0, w)
    boxes[:, [1, 4]] = np.clip(boxes[:, [1, 4]], 0, h)
    boxes[:, [2, 5]] = np.clip(boxes[:, [2, 5]], 0, d)
    return boxes


def preprocess_with_training_transform(
    input_path: Union[str, Path],
    cfg: RetinaNetConfig,
) -> Tuple[torch.Tensor, Dict]:
    """
    Use the same deterministic preprocessing as training/validation:
    Orientation + Spacing + Intensity scaling.
    """
    val_transform = build_val_transform(
        spacing=cfg.spacing,
        hu_min=cfg.hu_min,
        hu_max=cfg.hu_max,
    )
    data = {
        "image": str(input_path),
        "box": np.zeros((0, 6), dtype=np.float32).tolist(),
        "label": np.zeros((0,), dtype=np.int64).tolist(),
    }
    processed = val_transform(data)
    image_tensor = processed["image"].float()
    meta = getattr(image_tensor, "meta", {}) or {}
    return image_tensor, meta


def _box_corners_xyz(box: np.ndarray) -> np.ndarray:
    x1, y1, z1, x2, y2, z2 = [float(v) for v in box]
    return np.array(
        [
            [x1, y1, z1],
            [x1, y1, z2],
            [x1, y2, z1],
            [x1, y2, z2],
            [x2, y1, z1],
            [x2, y1, z2],
            [x2, y2, z1],
            [x2, y2, z2],
        ],
        dtype=np.float64,
    )


def map_boxes_to_original_xyz(
    boxes: np.ndarray,
    transformed_affine: np.ndarray,
    original_affine: np.ndarray,
    original_shape_dhw: Tuple[int, int, int],
) -> np.ndarray:
    """
    Map predicted boxes from transformed voxel space back to original image voxel space.
    Inputs/outputs are [x1, y1, z1, x2, y2, z2].
    """
    if len(boxes) == 0:
        return boxes

    inv_original_affine = np.linalg.inv(original_affine)
    d, h, w = original_shape_dhw
    mapped = []

    for box in boxes:
        corners = _box_corners_xyz(box)
        corners_h = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float64)], axis=1)
        world = (transformed_affine @ corners_h.T).T
        orig_ijk_h = (inv_original_affine @ world.T).T
        orig_ijk = orig_ijk_h[:, :3]

        x1 = float(np.min(orig_ijk[:, 0]))
        y1 = float(np.min(orig_ijk[:, 1]))
        z1 = float(np.min(orig_ijk[:, 2]))
        x2 = float(np.max(orig_ijk[:, 0]))
        y2 = float(np.max(orig_ijk[:, 1]))
        z2 = float(np.max(orig_ijk[:, 2]))

        x1 = float(np.clip(x1, 0, max(w - 1, 0)))
        x2 = float(np.clip(x2, 0, max(w - 1, 0)))
        y1 = float(np.clip(y1, 0, max(h - 1, 0)))
        y2 = float(np.clip(y2, 0, max(h - 1, 0)))
        z1 = float(np.clip(z1, 0, max(d - 1, 0)))
        z2 = float(np.clip(z2, 0, max(d - 1, 0)))

        mapped.append([x1, y1, z1, x2, y2, z2])

    return np.asarray(mapped, dtype=np.float32)


def _strip_nii_gz(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def auto_find_gt_label_path(input_path: Union[str, Path]) -> Optional[Path]:
    p = Path(input_path)
    if not p.is_file():
        return None
    if "imagesTr" not in str(p):
        return None
    base = _strip_nii_gz(p.name)
    if base.endswith("_0000"):
        base = base[:-5]
    cand = p.parent.parent / "labelsTr" / f"{base}.nii.gz"
    return cand if cand.exists() else None


def load_gt_boxes(label_path: Union[str, Path], target_dhw: Tuple[int, int, int]) -> List[List[float]]:
    arr = np.asarray(nib.load(str(label_path)).get_fdata())

    # labelsTr is commonly (X,Y,Z); convert to (D=Z,H=Y,W=X)
    if arr.shape == (target_dhw[2], target_dhw[1], target_dhw[0]):
        arr = np.transpose(arr, (2, 1, 0))
    elif arr.shape != target_dhw:
        raise ValueError(f"GT label shape {arr.shape} incompatible with CT shape {target_dhw}")

    boxes: List[List[float]] = []
    for v in sorted([x for x in np.unique(arr) if x > 0]):
        coords = np.argwhere(arr == v)  # (z,y,x)
        if coords.size == 0:
            continue
        z1, y1, x1 = coords.min(axis=0).tolist()
        z2, y2, x2 = coords.max(axis=0).tolist()
        boxes.append([float(x1), float(y1), float(z1), float(x2), float(y2), float(z2)])
    return boxes


def iou3d(a: List[float], b: List[float]) -> float:
    ax1, ay1, az1, ax2, ay2, az2 = a
    bx1, by1, bz1, bx2, by2, bz2 = b

    ix1, iy1, iz1 = max(ax1, bx1), max(ay1, by1), max(az1, bz1)
    ix2, iy2, iz2 = min(ax2, bx2), min(ay2, by2), min(az2, bz2)

    iw = max(0.0, ix2 - ix1 + 1.0)
    ih = max(0.0, iy2 - iy1 + 1.0)
    id_ = max(0.0, iz2 - iz1 + 1.0)
    inter = iw * ih * id_

    va = max(0.0, ax2 - ax1 + 1.0) * max(0.0, ay2 - ay1 + 1.0) * max(0.0, az2 - az1 + 1.0)
    vb = max(0.0, bx2 - bx1 + 1.0) * max(0.0, by2 - by1 + 1.0) * max(0.0, bz2 - bz1 + 1.0)
    union = va + vb - inter
    return 0.0 if union <= 0 else float(inter / union)


def save_visualization(
    array: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    output_dir: Path,
    prefix: str,
    gt_boxes: Optional[List[List[float]]] = None,
) -> None:
    viz_dir = output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    for i, box in enumerate(pred_boxes):
        score = float(pred_scores[i])
        x1, y1, z1, x2, y2, z2 = [int(round(v)) for v in box]

        z_center = int((z1 + z2) / 2)
        z_center = max(0, min(array.shape[0] - 1, z_center))

        slice_img = array[z_center]
        vmin, vmax = -1000, 400
        disp_img = np.clip(slice_img, vmin, vmax)
        disp_img = (disp_img - vmin) / (vmax - vmin)

        plt.figure(figsize=(8, 8))
        plt.imshow(disp_img, cmap="gray")

        pred_rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
        plt.gca().add_patch(pred_rect)
        plt.text(x1, max(0, y1 - 4), f"P{i+1}", color="red", fontsize=9, weight="bold")

        if gt_boxes:
            for g_idx, g in enumerate(gt_boxes, 1):
                gx1, gy1, gz1, gx2, gy2, gz2 = [int(round(v)) for v in g]
                if gz1 <= z_center <= gz2:
                    gt_rect = plt.Rectangle(
                        (gx1, gy1), gx2 - gx1, gy2 - gy1,
                        linewidth=2, edgecolor="lime", linestyle="--", facecolor="none"
                    )
                    plt.gca().add_patch(gt_rect)
                    plt.text(gx1, max(0, gy1 - 4), f"GT{g_idx}", color="lime", fontsize=9, weight="bold")
            plt.text(5, 15, "Legend: red=PRED, green=GT", color="yellow", fontsize=9, weight="bold")

        plt.title(f"Pred {i+1} score={score:.3f} z={z_center} (red=pred, green=gt)")
        plt.axis("off")

        out_path = viz_dir / f"{prefix}_pred_{i+1:03d}_score_{score:.3f}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()


def save_gt_only_visualization(
    array: np.ndarray,
    output_dir: Path,
    prefix: str,
    gt_boxes: List[List[float]],
) -> None:
    if not gt_boxes:
        return
    viz_dir = output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Use first GT box center slice for quick debugging view.
    gx1, gy1, gz1, gx2, gy2, gz2 = [int(round(v)) for v in gt_boxes[0]]
    z_center = int((gz1 + gz2) / 2)
    z_center = max(0, min(array.shape[0] - 1, z_center))

    slice_img = array[z_center]
    vmin, vmax = -1000, 400
    disp_img = np.clip(slice_img, vmin, vmax)
    disp_img = (disp_img - vmin) / (vmax - vmin)

    plt.figure(figsize=(8, 8))
    plt.imshow(disp_img, cmap="gray")
    for g_idx, g in enumerate(gt_boxes, 1):
        x1, y1, z1, x2, y2, z2 = [int(round(v)) for v in g]
        if z1 <= z_center <= z2:
            gt_rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="lime", linestyle="--", facecolor="none"
            )
            plt.gca().add_patch(gt_rect)
            plt.text(x1, max(0, y1 - 4), f"GT{g_idx}", color="lime", fontsize=9, weight="bold")
    plt.text(5, 15, "Legend: green=GT (no predictions)", color="yellow", fontsize=9, weight="bold")
    plt.title(f"GT only z={z_center}")
    plt.axis("off")
    out_path = viz_dir / f"{prefix}_gt_only.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def build_report(
    input_path: str,
    model_path: str,
    array: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
) -> Dict:
    report: Dict = {
        "input_path": str(input_path),
        "model_path": str(model_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_shape": list(array.shape),
        "spacing": list(spacing),
        "nodules": [],
    }

    location_estimator = LungLocationEstimator(total_slices=max(int(array.shape[0]), 1))

    for i in range(len(boxes)):
        box = boxes[i].tolist()
        cx = (box[0] + box[3]) / 2
        cy = (box[1] + box[4]) / 2
        cz = (box[2] + box[5]) / 2

        relative_x = cx / max(array.shape[2], 1)
        relative_y = cy / max(array.shape[1], 1)
        slice_ratio = cz / max(array.shape[0], 1)
        location = location_estimator.estimate_location(relative_x, relative_y, slice_ratio)

        pt_world = origin + np.array([cx, cy, cz]) * spacing
        diameter = max(box[3] - box[0], box[4] - box[1], box[5] - box[2]) * spacing[0]

        report["nodules"].append(
            {
                "id": i + 1,
                "score": float(scores[i]),
                "box_voxel": [round(x, 1) for x in box],
                "center_world_mm": [round(float(x), 2) for x in pt_world],
                "approx_diameter_mm": round(float(diameter), 1),
                "anatomical_location": {
                    "lobe": location["lobe"],
                    "lobe_full": location["lobe_full"],
                    "side": location["side"],
                    "confidence": round(float(location["confidence"]), 3),
                },
            }
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="RetinaNet inference pipeline")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results_inference")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window_center", type=float, default=-600)
    parser.add_argument("--window_width", type=float, default=1500)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--gt_label_path", type=str, default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing detector: %s", args.model_path)
    cfg = RetinaNetConfig(output_dir=str(output_dir), cache_dataset=False)
    trainer = RetinaNetTrainer(cfg, inference_only=True)

    try:
        trainer.detector.network = torch.jit.load(args.model_path, map_location=args.device)
    except Exception as e:
        logger.warning("TorchScript load failed (%s), fallback to state_dict", e)
        checkpoint = torch.load(args.model_path, map_location=args.device)
        if "model" in checkpoint:
            trainer.detector.network.load_state_dict(checkpoint["model"])
        else:
            trainer.detector.network.load_state_dict(checkpoint)

    trainer.detector.to(args.device)
    trainer.detector.eval()

    input_path = Path(args.input_path)
    is_npz = input_path.suffix.lower() == ".npz"

    transformed_meta: Dict = {}
    if is_npz:
        input_tensor, spacing, origin, scale_info = load_preprocessed_npz(input_path)
        array = input_tensor[0].cpu().numpy()
    else:
        _, array, spacing, origin = load_ct_volume(args.input_path)
        input_tensor, transformed_meta = preprocess_with_training_transform(input_path, cfg)
        scale_info = {}

    logger.info("Running inference")
    input_tensor = input_tensor.to(args.device)

    with torch.no_grad():
        if cfg.amp and str(args.device).startswith("cuda"):
            with torch.amp.autocast("cuda"):
                outputs = trainer.detector([input_tensor], use_inferer=True)
        else:
            outputs = trainer.detector([input_tensor], use_inferer=True)

    results = outputs[0]
    pred_boxes = results[trainer.detector.target_box_key].detach().cpu().numpy()
    pred_scores = results[trainer.detector.pred_score_key].detach().cpu().numpy()

    keep = pred_scores >= args.threshold
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]

    logger.info("Detected %d nodules with threshold >= %.3f", len(pred_boxes), args.threshold)

    if is_npz:
        orig_boxes = rescale_boxes(pred_boxes, scale_info)
    else:
        transformed_affine = transformed_meta.get("affine", None)
        original_affine = transformed_meta.get("original_affine", None)
        if transformed_affine is not None and original_affine is not None:
            transformed_affine_np = np.asarray(transformed_affine, dtype=np.float64)
            original_affine_np = np.asarray(original_affine, dtype=np.float64)
            orig_boxes = map_boxes_to_original_xyz(
                pred_boxes,
                transformed_affine_np,
                original_affine_np,
                tuple(array.shape),
            )
        else:
            logger.warning("Transform metadata missing affine/original_affine; fallback to legacy rescale")
            input_tensor_legacy, scale_info_legacy = preprocess_volume(
                array,
                target_size=args.image_size,
                window_center=args.window_center,
                window_width=args.window_width,
            )
            _ = input_tensor_legacy  # explicit: used only for scale_info fallback
            orig_boxes = rescale_boxes(pred_boxes, scale_info_legacy)
    report = build_report(args.input_path, args.model_path, array, spacing, origin, orig_boxes, pred_scores)

    gt_boxes: List[List[float]] = []
    gt_path: Optional[Path] = None
    if args.gt_label_path:
        gt_path = Path(args.gt_label_path)
    else:
        gt_path = auto_find_gt_label_path(args.input_path)

    if gt_path and gt_path.exists() and not is_npz:
        try:
            gt_boxes = load_gt_boxes(gt_path, tuple(array.shape))
            best_iou_per_pred = []
            for pb in orig_boxes.tolist() if len(orig_boxes) > 0 else []:
                best_iou_per_pred.append(max([iou3d(pb, gb) for gb in gt_boxes], default=0.0))

            best_iou_per_gt = []
            for gb in gt_boxes:
                best_iou_per_gt.append(max([iou3d(pb, gb) for pb in orig_boxes.tolist()], default=0.0))

            report["gt_eval"] = {
                "gt_label_path": str(gt_path),
                "gt_count": len(gt_boxes),
                "pred_count": int(len(orig_boxes)),
                "best_iou_per_pred": [round(float(x), 4) for x in best_iou_per_pred],
                "best_iou_per_gt": [round(float(x), 4) for x in best_iou_per_gt],
                "max_iou": round(float(max(best_iou_per_pred) if best_iou_per_pred else 0.0), 4),
            }
        except Exception as e:
            report["gt_eval"] = {"error": str(e)}

    json_path = output_dir / "report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Saved report: %s", json_path)

    if len(orig_boxes) > 0:
        save_visualization(array, orig_boxes, pred_scores, output_dir, _strip_nii_gz(input_path.name), gt_boxes=gt_boxes)
        logger.info("Saved visualization to %s", output_dir / "viz")
    elif gt_boxes:
        save_gt_only_visualization(array, output_dir, _strip_nii_gz(input_path.name), gt_boxes)
        logger.info("Saved GT-only visualization to %s", output_dir / "viz")


if __name__ == "__main__":
    main()
