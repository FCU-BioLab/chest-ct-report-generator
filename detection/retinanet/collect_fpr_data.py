#!/usr/bin/env python3
"""
Collect FPR patches from RetinaNet predictions.
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logging.getLogger("monai.transforms.io.array").setLevel(logging.WARNING)
logging.getLogger("monai.data.image_reader").setLevel(logging.WARNING)

PATCH_SIZE = 32
IOU_THRESH = 0.1


def compute_iou_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    iz1 = max(box_a[2], box_b[2])
    ix2 = min(box_a[3], box_b[3])
    iy2 = min(box_a[4], box_b[4])
    iz2 = min(box_a[5], box_b[5])

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) * max(0.0, iz2 - iz1)
    if inter <= 0.0:
        return 0.0

    vol_a = max(0.0, box_a[3] - box_a[0]) * max(0.0, box_a[4] - box_a[1]) * max(0.0, box_a[5] - box_a[2])
    vol_b = max(0.0, box_b[3] - box_b[0]) * max(0.0, box_b[4] - box_b[1]) * max(0.0, box_b[5] - box_b[2])
    return float(inter / (vol_a + vol_b - inter + 1e-6))


def crop_patch(image_np: np.ndarray, center: Tuple[float, float, float], patch_size: int = PATCH_SIZE) -> np.ndarray:
    """Crop a cubic patch from a transformed image tensor in (Y, X, Z) order."""
    half = patch_size // 2
    if image_np.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape={image_np.shape}")

    size_y, size_x, size_z = image_np.shape
    cy, cx, cz = (int(round(v)) for v in center)

    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half
    z1, z2 = cz - half, cz + half

    pad_y1, pad_x1, pad_z1 = max(0, -y1), max(0, -x1), max(0, -z1)
    pad_y2, pad_x2, pad_z2 = max(0, y2 - size_y), max(0, x2 - size_x), max(0, z2 - size_z)

    y1c, y2c = max(y1, 0), min(y2, size_y)
    x1c, x2c = max(x1, 0), min(x2, size_x)
    z1c, z2c = max(z1, 0), min(z2, size_z)

    patch = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
    patch[
        pad_y1:patch_size - pad_y2,
        pad_x1:patch_size - pad_x2,
        pad_z1:patch_size - pad_z2,
    ] = image_np[y1c:y2c, x1c:x2c, z1c:z2c]
    return patch


def _normalize_section_name(section: str) -> str:
    mapping = {"train": "training", "val": "validation", "test": "testing"}
    return mapping.get(section, section)


def _get_scan_id(item: Dict) -> str:
    return str(
        item.get("seriesuid")
        or item.get("lndb_id")
        or Path(item.get("image", "unknown")).stem
    )


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)[:80]


def _load_lung_mask_yxz(mask_path: Path) -> np.ndarray:
    mask_img = sitk.ReadImage(str(mask_path))
    mask_arr_zyx = sitk.GetArrayFromImage(mask_img)
    if mask_arr_zyx.ndim != 3:
        raise ValueError(f"Expected 3D lung mask, got shape={mask_arr_zyx.shape}")
    return (mask_arr_zyx.transpose(1, 2, 0) > 0).astype(np.float32)


def _classify_negative_type(
    pred_score: float,
    best_iou: float,
    hard_negative_min_score: float,
    hard_negative_iou_max: float,
    near_miss_iou_min: float,
    near_miss_iou_max: float,
) -> str:
    if pred_score >= hard_negative_min_score and best_iou <= hard_negative_iou_max:
        return "hard_negative"
    if near_miss_iou_min < best_iou <= near_miss_iou_max:
        return "near_miss_negative"
    return "easy_negative"


def _keep_easy_negative(scan_id: str, pred_index: int, keep_ratio: float) -> bool:
    if keep_ratio >= 1.0:
        return True
    if keep_ratio <= 0.0:
        return False
    digest = hashlib.md5(f"{scan_id}:{pred_index}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / float(0xFFFFFFFF)
    return value < keep_ratio


def collect_from_dataset(
    trainer,
    dataset,
    raw_items: List[Dict],
    section_name: str,
    pos_dir: Path,
    neg_dir: Path,
    config,
    score_thresh: float,
    patch_size: int,
    keep_duplicate_as_negative: bool,
    hard_negative_mining: bool,
    hard_negative_min_score: float,
    hard_negative_iou_max: float,
    hard_negative_max_per_scan: int,
    near_miss_iou_min: float,
    near_miss_iou_max: float,
    easy_negative_keep_ratio: float,
    hard_negative_only: bool,
    lung_mask_dir: Path | None,
    require_lung_mask: bool,
    lung_mask_min_overlap: float,
) -> Tuple[int, int, int, List[Dict]]:
    total_tp = 0
    total_fp = 0
    total_ignored = 0
    metadata: List[Dict] = []
    skipped = 0

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Collecting [{section_name}]"):
            record = raw_items[idx]
            scan_id = _get_scan_id(record)
            scan_name = _safe_stem(scan_id)
            image_path = str(record.get("image", ""))
            lung_mask = None
            lung_mask_path = None

            try:
                if lung_mask_dir is not None:
                    candidate_mask_path = lung_mask_dir / f"{scan_id}.mhd"
                    if candidate_mask_path.exists():
                        lung_mask = _load_lung_mask_yxz(candidate_mask_path)
                        lung_mask_path = str(candidate_mask_path)
                    elif require_lung_mask:
                        raise FileNotFoundError(f"Missing lung mask: {candidate_mask_path}")

                data_item = dataset[idx]
                img_tensor = data_item["image"].to(trainer.device)
                gt_boxes = data_item[trainer.detector.target_box_key].cpu().numpy()

                use_inferer = img_tensor[0, ...].numel() >= np.prod(config.val_patch_size)
                if config.amp:
                    with torch.amp.autocast("cuda"):
                        outputs = trainer.detector([img_tensor], use_inferer=use_inferer)
                else:
                    outputs = trainer.detector([img_tensor], use_inferer=use_inferer)

                pred_boxes = outputs[0][trainer.detector.target_box_key].cpu().numpy()
                pred_scores = outputs[0][trainer.detector.pred_score_key].cpu().numpy()
                if len(pred_boxes) == 0:
                    continue

                score_mask = pred_scores >= score_thresh if score_thresh > 0 else np.ones_like(pred_scores, dtype=bool)
                pred_boxes = pred_boxes[score_mask]
                pred_scores = pred_scores[score_mask]
                if len(pred_boxes) == 0:
                    continue

                if lung_mask is not None and lung_mask.shape != tuple(img_tensor[0].shape):
                    raise ValueError(
                        f"Lung mask shape {lung_mask.shape} does not match image shape {tuple(img_tensor[0].shape)}"
                    )

                order = np.argsort(-pred_scores)
                pred_boxes = pred_boxes[order]
                pred_scores = pred_scores[order]
                img_np = img_tensor[0].cpu().numpy()
                gt_matched = set()
                neg_saved_this_scan = 0
                hard_neg_saved_this_scan = 0

                for pred_index, (pred_box, pred_score) in enumerate(zip(pred_boxes, pred_scores)):
                    best_iou = 0.0
                    best_gt_index = -1
                    for gt_index, gt_box in enumerate(gt_boxes):
                        iou = compute_iou_3d(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_index = gt_index

                    is_candidate_match = best_iou >= IOU_THRESH
                    is_tp = is_candidate_match and best_gt_index not in gt_matched
                    if is_tp:
                        gt_matched.add(best_gt_index)

                    center = (
                        (pred_box[0] + pred_box[3]) / 2.0,
                        (pred_box[1] + pred_box[4]) / 2.0,
                        (pred_box[2] + pred_box[5]) / 2.0,
                    )
                    patch = crop_patch(img_np, center, patch_size)
                    is_duplicate_match = is_candidate_match and not is_tp
                    hard_negative_candidate = (not is_tp) and (not is_duplicate_match)
                    hard_negative_selected = False
                    hard_negative_reason = None
                    neg_type = None
                    lung_patch = None
                    lung_overlap = None
                    if lung_mask is not None:
                        lung_patch = crop_patch(lung_mask, center, patch_size)
                        lung_overlap = float(lung_patch.mean())
                        if lung_overlap < lung_mask_min_overlap:
                            label = "ignored"
                            save_dir = None
                            hard_negative_reason = "lung_overlap_below_min"
                            neg_type = "outside_lung"
                            total_ignored += 1
                            filename = (
                                f"{section_name}_{scan_name}_pred{pred_index:03d}"
                                f"_iou{best_iou:.3f}_s{pred_score:.3f}.npy"
                            )
                            metadata.append(
                                {
                                    "filename": filename,
                                    "relative_path": None,
                                    "label": label,
                                    "score": float(pred_score),
                                    "iou": float(best_iou),
                                    "matched_gt_index": int(best_gt_index),
                                    "is_candidate_match": bool(is_candidate_match),
                                    "is_duplicate_match": bool(is_duplicate_match),
                                    "hard_negative_mining": bool(hard_negative_mining),
                                    "hard_negative_candidate": bool(hard_negative_candidate),
                                    "hard_negative_selected": bool(hard_negative_selected),
                                    "hard_negative_reason": hard_negative_reason,
                                    "neg_type": neg_type,
                                    "scan_id": scan_id,
                                    "seriesuid": record.get("seriesuid"),
                                    "lndb_id": record.get("lndb_id"),
                                    "source_image": image_path,
                                    "source_split": section_name,
                                    "scan_index_in_split": idx,
                                    "pred_box_yxz": [float(v) for v in pred_box.tolist()],
                                    "center_yxz": [float(v) for v in center],
                                    "patch_size": patch_size,
                                    "lung_mask_path": lung_mask_path,
                                    "lung_overlap": lung_overlap,
                                }
                            )
                            continue
                        patch = patch * lung_patch

                    if is_tp:
                        label = "positive"
                        save_dir = pos_dir
                        neg_type = "positive"
                    elif is_duplicate_match and not keep_duplicate_as_negative:
                        label = "ignored"
                        save_dir = None
                        neg_type = "duplicate_match"
                    else:
                        label = "negative"
                        save_dir = neg_dir
                        neg_type = _classify_negative_type(
                            float(pred_score),
                            float(best_iou),
                            hard_negative_min_score,
                            hard_negative_iou_max,
                            near_miss_iou_min,
                            near_miss_iou_max,
                        )
                        if hard_negative_only and neg_type != "hard_negative":
                            label = "ignored"
                            save_dir = None
                            hard_negative_reason = "hard_negative_only_filtered"
                        elif neg_type == "hard_negative":
                            if hard_negative_max_per_scan > 0 and hard_neg_saved_this_scan >= hard_negative_max_per_scan:
                                label = "ignored"
                                save_dir = None
                                hard_negative_reason = "per_scan_limit"
                            else:
                                hard_negative_selected = True
                        elif hard_negative_mining and neg_type == "easy_negative":
                            if not _keep_easy_negative(scan_id, pred_index, easy_negative_keep_ratio):
                                label = "ignored"
                                save_dir = None
                                hard_negative_reason = "easy_negative_subsampled"

                    filename = (
                        f"{section_name}_{scan_name}_pred{pred_index:03d}"
                        f"_iou{best_iou:.3f}_s{pred_score:.3f}.npy"
                    )
                    if save_dir is not None:
                        np.save(str(save_dir / filename), patch)

                    if is_tp:
                        total_tp += 1
                    elif label == "negative":
                        total_fp += 1
                        neg_saved_this_scan += 1
                        if neg_type == "hard_negative":
                            hard_neg_saved_this_scan += 1
                    else:
                        total_ignored += 1

                    metadata.append(
                        {
                            "filename": filename,
                            "relative_path": f"{label}/{filename}" if save_dir is not None else None,
                            "label": label,
                            "score": float(pred_score),
                            "iou": float(best_iou),
                            "matched_gt_index": int(best_gt_index),
                            "is_candidate_match": bool(is_candidate_match),
                            "is_duplicate_match": bool(is_duplicate_match),
                            "hard_negative_mining": bool(hard_negative_mining),
                            "hard_negative_candidate": bool(hard_negative_candidate),
                            "hard_negative_selected": bool(hard_negative_selected),
                            "hard_negative_reason": hard_negative_reason,
                            "neg_type": neg_type,
                            "scan_id": scan_id,
                            "seriesuid": record.get("seriesuid"),
                            "lndb_id": record.get("lndb_id"),
                            "source_image": image_path,
                            "source_split": section_name,
                            "scan_index_in_split": idx,
                            "pred_box_yxz": [float(v) for v in pred_box.tolist()],
                            "center_yxz": [float(v) for v in center],
                            "patch_size": patch_size,
                            "lung_mask_path": lung_mask_path,
                            "lung_overlap": lung_overlap,
                        }
                    )
            except Exception as exc:
                skipped += 1
                if skipped <= 10:
                    logger.warning("Skipped %s[%d] (%s): %s", section_name, idx, scan_id, exc)

    if skipped:
        logger.info("Skipped %d samples in split '%s'", skipped, section_name)

    return total_tp, total_fp, total_ignored, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect FPR patches from RetinaNet predictions.")
    parser.add_argument("--checkpoint", required=True, help="RetinaNet checkpoint path")
    parser.add_argument("--data_path", default="detection/manifests/dataset_luna16.json", help="dataset JSON path")
    parser.add_argument("--output_dir", default="detection/results/fpr_dataset", help="output directory")
    parser.add_argument("--source_split", default="train", choices=["train", "val", "test"], help="dataset split used to collect FPR patches")
    parser.add_argument("--score_thresh", type=float, default=0.05, help="collect predictions with score >= this threshold")
    parser.add_argument("--patch_size", type=int, default=32, help="cubic patch size")
    parser.add_argument("--val_patch_size", type=int, nargs=3, default=None, metavar=("H", "W", "D"), help="sliding-window ROI size for full-volume inference")
    parser.add_argument("--no_cache", action="store_true", help="disable MONAI PersistentDataset disk cache during FPR collection")
    parser.add_argument("--keep_duplicate_as_negative", action="store_true", help="keep duplicate matches (IoU>=0.1 but GT already matched) as negative samples")
    parser.add_argument("--hard_negative_mining", action="store_true", help="keep only hard negatives (high-score, low-IoU, per-scan top-K)")
    parser.add_argument("--hard_negative_min_score", type=float, default=0.3, help="minimum score for hard negatives")
    parser.add_argument("--hard_negative_iou_max", type=float, default=0.05, help="maximum IoU for hard negatives")
    parser.add_argument("--hard_negative_max_per_scan", type=int, default=20, help="max negatives kept per scan when hard-negative mining is enabled (0=unlimited)")
    parser.add_argument("--near_miss_iou_min", type=float, default=0.05, help="minimum IoU for near-miss negatives")
    parser.add_argument("--near_miss_iou_max", type=float, default=0.20, help="maximum IoU for near-miss negatives")
    parser.add_argument("--easy_negative_keep_ratio", type=float, default=0.2, help="deterministic keep ratio for easy negatives during hard-negative mining")
    parser.add_argument("--hard_negative_only", action="store_true", help="keep only detector high-score low-IoU false positives as negatives")
    parser.add_argument("--lung_mask_dir", default=None, help="directory containing lung masks named <seriesuid>.mhd")
    parser.add_argument("--require_lung_mask", action="store_true", help="skip/error when lung mask is missing")
    parser.add_argument("--lung_mask_min_overlap", type=float, default=0.05, help="minimum fraction of patch voxels inside lung mask")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pos_dir = output_dir / "positive"
    neg_dir = output_dir / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    from monai.data import Dataset, PersistentDataset
    from monai.utils import set_determinism

    from .config import RetinaNetConfig
    from .dataset import build_val_transform, prepare_datalist
    from .trainer import RetinaNetTrainer

    config = RetinaNetConfig(
        data_path=args.data_path,
        device=args.device,
        num_workers=args.num_workers,
        output_dir=str(output_dir),
    )
    if args.val_patch_size:
        config.val_patch_size = args.val_patch_size
    config.validate()
    set_determinism(seed=config.split_seed)

    trainer = RetinaNetTrainer(config)
    trainer.detector.network = torch.jit.load(args.checkpoint).to(trainer.device)
    trainer.detector.eval()
    logger.info("Loaded RetinaNet checkpoint: %s", args.checkpoint)

    section_name = _normalize_section_name(args.source_split)
    raw_items = prepare_datalist(
        config.data_path,
        args.source_split,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.split_seed,
    )
    raw_items = [item for item in raw_items if os.path.exists(item.get("image", ""))]

    val_transform = build_val_transform(
        spacing=config.spacing,
        hu_min=config.hu_min,
        hu_max=config.hu_max,
    )
    if args.no_cache:
        logger.info("  🧹 停用 PersistentDataset 磁碟快取 (--no_cache)")
        dataset = Dataset(
            data=raw_items,
            transform=val_transform,
        )
    else:
        cache_dir = str(Path("cache/monai_persistent_cache") / f"fpr_{section_name}")
        logger.info("  📦 使用 PersistentDataset 磁碟快取: %s", cache_dir)
        dataset = PersistentDataset(
            data=raw_items,
            transform=val_transform,
            cache_dir=cache_dir,
        )

    total_tp, total_fp, total_ignored, samples = collect_from_dataset(
        trainer=trainer,
        dataset=dataset,
        raw_items=raw_items,
        section_name=section_name,
        pos_dir=pos_dir,
        neg_dir=neg_dir,
        config=config,
        score_thresh=args.score_thresh,
        patch_size=args.patch_size,
        keep_duplicate_as_negative=args.keep_duplicate_as_negative,
        hard_negative_mining=args.hard_negative_mining,
        hard_negative_min_score=args.hard_negative_min_score,
        hard_negative_iou_max=args.hard_negative_iou_max,
        hard_negative_max_per_scan=args.hard_negative_max_per_scan,
        near_miss_iou_min=args.near_miss_iou_min,
        near_miss_iou_max=args.near_miss_iou_max,
        easy_negative_keep_ratio=args.easy_negative_keep_ratio,
        hard_negative_only=args.hard_negative_only,
        lung_mask_dir=Path(args.lung_mask_dir) if args.lung_mask_dir else None,
        require_lung_mask=args.require_lung_mask,
        lung_mask_min_overlap=args.lung_mask_min_overlap,
    )

    n_hard_selected = sum(1 for s in samples if s.get("hard_negative_selected"))
    n_hard_ignored = sum(1 for s in samples if s.get("hard_negative_mining") and s.get("label") == "ignored")
    n_hard = sum(1 for s in samples if s.get("label") == "negative" and s.get("neg_type") == "hard_negative")
    n_near_miss = sum(1 for s in samples if s.get("label") == "negative" and s.get("neg_type") == "near_miss_negative")
    n_easy = sum(1 for s in samples if s.get("label") == "negative" and s.get("neg_type") == "easy_negative")

    metadata = {
        "version": 2,
        "summary": {
            "source_split": section_name,
            "n_scans": len({_get_scan_id(item) for item in raw_items}),
            "n_samples": len(samples),
            "n_positive": total_tp,
            "n_negative": total_fp,
            "n_ignored_duplicate": total_ignored,
            "n_hard_negative_selected": int(n_hard_selected),
            "n_hard_negative_ignored": int(n_hard_ignored),
            "n_hard_negative": int(n_hard),
            "n_near_miss_negative": int(n_near_miss),
            "n_easy_negative": int(n_easy),
            "negative_to_positive_ratio": float(total_fp / max(total_tp, 1)),
        },
        "config": {
            "checkpoint": args.checkpoint,
            "data_path": args.data_path,
            "score_thresh": args.score_thresh,
            "patch_size": args.patch_size,
            "no_cache": args.no_cache,
            "keep_duplicate_as_negative": args.keep_duplicate_as_negative,
            "hard_negative_mining": args.hard_negative_mining,
            "hard_negative_min_score": args.hard_negative_min_score,
            "hard_negative_iou_max": args.hard_negative_iou_max,
            "hard_negative_max_per_scan": args.hard_negative_max_per_scan,
            "near_miss_iou_min": args.near_miss_iou_min,
            "near_miss_iou_max": args.near_miss_iou_max,
            "easy_negative_keep_ratio": args.easy_negative_keep_ratio,
            "hard_negative_only": args.hard_negative_only,
            "lung_mask_dir": args.lung_mask_dir,
            "require_lung_mask": args.require_lung_mask,
            "lung_mask_min_overlap": args.lung_mask_min_overlap,
            "val_patch_size": config.val_patch_size,
            "spacing": config.spacing,
            "split_seed": config.split_seed,
        },
        "samples": samples,
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("FPR collection complete")
    logger.info("  split: %s", section_name)
    logger.info("  positive: %d", total_tp)
    logger.info("  negative: %d", total_fp)
    logger.info("  ignored_duplicate: %d", total_ignored)
    logger.info("  hard_negative: %d | near_miss_negative: %d | easy_negative: %d", n_hard, n_near_miss, n_easy)
    if args.hard_negative_mining:
        logger.info("  hard_negative_selected: %d", n_hard_selected)
        logger.info("  hard_negative_ignored: %d", n_hard_ignored)
    logger.info("  output: %s", output_dir)


if __name__ == "__main__":
    main()
