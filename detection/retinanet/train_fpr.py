#!/usr/bin/env python3
"""
Train a false-positive reduction classifier on RetinaNet proposal patches.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FPRPatchDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Dict],
        model_type: str,
        num_slices_per_view: int = 1,
        augment: bool = False,
        augment_rotate_prob: float = 0.5,
        augment_noise: float = 0.05,
    ):
        self.samples = list(samples)
        self.model_type = model_type
        self.num_slices_per_view = int(num_slices_per_view)
        self.augment = augment
        self.augment_rotate_prob = float(augment_rotate_prob)
        self.augment_noise = float(augment_noise)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from .fpr_model import patch_to_model_input

        sample = self.samples[idx]
        patch = np.load(sample["file_path"]).astype(np.float32)

        if self.augment:
            for axis in range(3):
                if np.random.random() > 0.5:
                    patch = np.flip(patch, axis=axis).copy()
            if np.random.random() < self.augment_rotate_prob:
                axes = [(0, 1), (0, 2), (1, 2)][np.random.randint(0, 3)]
                patch = np.rot90(patch, k=int(np.random.randint(1, 4)), axes=axes).copy()
            if self.augment_noise > 0:
                patch = np.clip(patch + np.random.uniform(-self.augment_noise, self.augment_noise), 0.0, 1.0)

        patch = patch_to_model_input(
            patch,
            self.model_type,
            num_slices_per_view=self.num_slices_per_view,
        )
        label = torch.tensor(sample["label_id"], dtype=torch.long)
        return torch.from_numpy(patch), label


def _load_metadata(metadata_path: Path) -> Dict:
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if isinstance(metadata, list):
        return {"version": 1, "samples": metadata}
    return metadata


def _infer_metadata_path(pos_dir: Path, neg_dir: Path, metadata_path: str = None) -> Path:
    if metadata_path:
        return Path(metadata_path)
    common_root = pos_dir.parent if pos_dir.parent == neg_dir.parent else pos_dir.parent
    return common_root / "metadata.json"


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _box_max_diameter_mm(box, spacing: Sequence[float]) -> float | None:
    if not isinstance(box, (list, tuple)) or len(box) != 6:
        return None
    try:
        dy = max(float(box[3]) - float(box[0]), 0.0) * float(spacing[0])
        dx = max(float(box[4]) - float(box[1]), 0.0) * float(spacing[1])
        dz = max(float(box[5]) - float(box[2]), 0.0) * float(spacing[2])
    except (TypeError, ValueError, IndexError):
        return None
    return float(max(dy, dx, dz))


def _build_samples(pos_dir: Path, neg_dir: Path, metadata: Dict) -> List[Dict]:
    samples = []
    spacing = metadata.get("config", {}).get("spacing") or [1.0, 1.0, 1.0]
    meta_by_filename = {
        item["filename"]: item
        for item in metadata.get("samples", [])
        if isinstance(item, dict) and "filename" in item
    }

    for label_name, label_id, directory in (("positive", 1, pos_dir), ("negative", 0, neg_dir)):
        for file_path in sorted(directory.glob("*.npy")):
            meta = meta_by_filename.get(file_path.name, {})
            pred_box = meta.get("pred_box_yxz")
            if isinstance(pred_box, (list, tuple)) and len(pred_box) == 6:
                try:
                    pred_box = [float(v) for v in pred_box]
                except (TypeError, ValueError):
                    pred_box = None
            else:
                pred_box = None
            pred_max_diameter_mm = _safe_float(meta.get("pred_max_diameter_mm"))
            if pred_max_diameter_mm is None:
                pred_max_diameter_mm = _box_max_diameter_mm(pred_box, spacing)
            gt_max_diameter_mm = _safe_float(meta.get("gt_max_diameter_mm"))
            effective_diameter_mm = gt_max_diameter_mm if label_id == 1 and gt_max_diameter_mm is not None else pred_max_diameter_mm
            samples.append(
                {
                    "file_path": str(file_path),
                    "filename": file_path.name,
                    "label_name": label_name,
                    "label_id": label_id,
                    "scan_id": str(meta.get("scan_id", file_path.stem.split("_pred", 1)[0])),
                    "source_split": meta.get("source_split", "unknown"),
                    "score": meta.get("score"),
                    "iou": meta.get("iou"),
                    "neg_type": meta.get("neg_type"),
                    "pred_box_yxz": pred_box,
                    "pred_max_diameter_mm": pred_max_diameter_mm,
                    "gt_max_diameter_mm": gt_max_diameter_mm,
                    "effective_diameter_mm": effective_diameter_mm,
                }
            )
    return samples


def _filter_by_source_split(samples: Sequence[Dict], allowed_splits: Sequence[str]) -> List[Dict]:
    allowed = {split.strip() for split in allowed_splits if split.strip()}
    if not allowed:
        return list(samples)
    return [sample for sample in samples if sample.get("source_split") in allowed]


def _filter_by_candidate_diameter(
    samples: Sequence[Dict],
    min_diam_mm: float | None,
    max_diam_mm: float | None,
) -> List[Dict]:
    if min_diam_mm is None and max_diam_mm is None:
        return list(samples)

    filtered = []
    for sample in samples:
        diameter = _safe_float(sample.get("pred_max_diameter_mm"))
        if diameter is None:
            diameter = _safe_float(sample.get("effective_diameter_mm"))
        if diameter is None:
            continue
        if min_diam_mm is not None and diameter < float(min_diam_mm):
            continue
        if max_diam_mm is not None and diameter > float(max_diam_mm):
            continue
        filtered.append(sample)
    return filtered


def _split_by_scan(samples: Sequence[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict], List[str], List[str]]:
    scans_to_samples: Dict[str, List[Dict]] = defaultdict(list)
    for sample in samples:
        scans_to_samples[sample["scan_id"]].append(sample)

    scan_ids = sorted(scans_to_samples.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(scan_ids)

    n_val_scans = max(1, int(round(len(scan_ids) * val_ratio))) if len(scan_ids) > 1 and val_ratio > 0 else 0
    if n_val_scans >= len(scan_ids):
        n_val_scans = max(1, len(scan_ids) - 1)

    val_scan_ids = set(scan_ids[:n_val_scans])
    train_scan_ids = set(scan_ids[n_val_scans:])

    train_samples = [sample for sample in samples if sample["scan_id"] in train_scan_ids]
    val_samples = [sample for sample in samples if sample["scan_id"] in val_scan_ids]
    return train_samples, val_samples, sorted(train_scan_ids), sorted(val_scan_ids)


def _compute_loss(logits: torch.Tensor, targets: torch.Tensor, loss_name: str, gamma: float) -> torch.Tensor:
    if loss_name == "ce":
        return F.cross_entropy(logits, targets)

    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    true_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    true_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_weight = (1.0 - true_probs).pow(gamma)
    return (-focal_weight * true_log_probs).mean()


def _summarize_samples(samples: Sequence[Dict]) -> Dict:
    n_pos = int(sum(sample["label_id"] for sample in samples))
    n_total = len(samples)
    neg_counts = defaultdict(int)
    for sample in samples:
        if sample["label_id"] == 0:
            neg_counts[str(sample.get("neg_type") or "unknown")] += 1
    return {
        "n_samples": n_total,
        "n_positive": n_pos,
        "n_negative": n_total - n_pos,
        "n_scans": len({sample["scan_id"] for sample in samples}),
        "negative_breakdown": dict(sorted(neg_counts.items())),
    }


def _compute_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    best = {
        "threshold": 0.5,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }
    for thresh in np.arange(0.05, 0.951, 0.01):
        pred = probs >= thresh
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        if f1 > best["f1"]:
            best = {
                "threshold": float(round(thresh, 2)),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
    return best


def _val_breakdown(
    samples: Sequence[Dict],
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    small_positive_diam: float,
    low_score_positive_thresh: float,
) -> Dict[str, float]:
    pred = probs >= float(threshold)

    def recall_for(mask: np.ndarray) -> float:
        denom = int(np.sum(mask & (y_true == 1)))
        if denom <= 0:
            return float("nan")
        return float(np.sum(mask & (y_true == 1) & pred) / denom)

    def fpr_for(mask: np.ndarray) -> float:
        denom = int(np.sum(mask & (y_true == 0)))
        if denom <= 0:
            return float("nan")
        return float(np.sum(mask & (y_true == 0) & pred) / denom)

    diam = np.asarray([
        _safe_float(sample.get("effective_diameter_mm"), float("nan"))
        for sample in samples
    ], dtype=np.float32)
    score = np.asarray([
        _safe_float(sample.get("score"), float("nan"))
        for sample in samples
    ], dtype=np.float32)
    neg_types = np.asarray([str(sample.get("neg_type") or "") for sample in samples], dtype=object)

    small_pos = (y_true == 1) & np.isfinite(diam) & (diam <= float(small_positive_diam))
    large_pos = (y_true == 1) & (~small_pos)
    low_score_pos = (y_true == 1) & np.isfinite(score) & (score < float(low_score_positive_thresh))

    return {
        "val_recall_small_pos": recall_for(small_pos),
        "val_recall_large_pos": recall_for(large_pos),
        "val_recall_low_score_pos": recall_for(low_score_pos),
        "val_fpr_hard_neg": fpr_for(neg_types == "hard_negative"),
        "val_fpr_near_miss_neg": fpr_for(neg_types == "near_miss_negative"),
        "val_fpr_easy_neg": fpr_for(neg_types == "easy_negative"),
    }


def _infer_negative_type(
    sample: Dict,
    hard_negative_score_thresh: float,
    hard_negative_iou_max: float,
    near_miss_iou_min: float,
    near_miss_iou_max: float,
) -> str:
    neg_type = sample.get("neg_type")
    if isinstance(neg_type, str) and neg_type:
        return neg_type
    score = sample.get("score")
    iou = sample.get("iou")
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    try:
        iou = float(iou) if iou is not None else None
    except (TypeError, ValueError):
        iou = None
    if score is not None and score >= hard_negative_score_thresh and (iou is None or iou <= hard_negative_iou_max):
        return "hard_negative"
    if iou is not None and near_miss_iou_min < iou <= near_miss_iou_max:
        return "near_miss_negative"
    return "easy_negative"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FPR classifier.")
    parser.add_argument("--pos_dir", default="detection/results/fpr_dataset/positive", help="positive patch directory")
    parser.add_argument("--neg_dir", default="detection/results/fpr_dataset/negative", help="negative patch directory")
    parser.add_argument("--metadata_path", default=None, help="metadata.json from collect_fpr_data.py")
    parser.add_argument("--output_dir", default="detection/results/fpr_resnet18", help="output directory")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="optimizer weight decay")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="validation scan ratio")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience, 0 disables")
    parser.add_argument("--loss", choices=["ce", "focal"], default="focal", help="classification loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="gamma for focal loss")
    parser.add_argument("--augment_rotate_prob", type=float, default=0.5, help="probability of random 90-degree patch rotation during training")
    parser.add_argument("--augment_noise", type=float, default=0.05, help="uniform intensity jitter amplitude during training")
    parser.add_argument("--allowed_source_splits", nargs="*", default=["training"], help="metadata source splits allowed for FPR training")
    parser.add_argument("--min_candidate_diam_mm", type=float, default=None, help="train only on candidates with predicted max diameter >= this value in mm")
    parser.add_argument("--max_candidate_diam_mm", type=float, default=None, help="train only on candidates with predicted max diameter <= this value in mm")
    parser.add_argument("--hard_negative_weight", type=float, default=2.0, help="extra sampling weight multiplier for hard negatives")
    parser.add_argument("--near_miss_weight", type=float, default=1.5, help="extra sampling weight multiplier for near-miss negatives")
    parser.add_argument("--easy_negative_weight", type=float, default=1.0, help="extra sampling weight multiplier for easy negatives")
    parser.add_argument("--positive_weight", type=float, default=1.0, help="sampling weight multiplier for all positives")
    parser.add_argument("--small_positive_weight", type=float, default=1.0, help="extra sampling weight multiplier for small positives")
    parser.add_argument("--small_positive_diam", type=float, default=8.0, help="positive max diameter threshold in mm for small-positive weighting")
    parser.add_argument("--low_score_positive_weight", type=float, default=1.0, help="extra sampling weight multiplier for low-score positives")
    parser.add_argument("--low_score_positive_thresh", type=float, default=0.9, help="detector score threshold below which positives get low-score weighting")
    parser.add_argument("--hard_negative_score_thresh", type=float, default=0.5, help="detector score threshold defining hard negatives")
    parser.add_argument("--hard_negative_iou_max", type=float, default=0.05, help="maximum IoU for hard negatives")
    parser.add_argument("--near_miss_iou_min", type=float, default=0.05, help="minimum IoU for near-miss negatives")
    parser.add_argument("--near_miss_iou_max", type=float, default=0.20, help="maximum IoU for near-miss negatives")
    parser.add_argument("--fpr_model_type", choices=["3d", "2.5d"], default="3d", help="FPR backbone input type")
    parser.add_argument("--fpr_backbone", choices=["resnet18", "resnet34"], default="resnet18", help="FPR classifier backbone")
    parser.add_argument("--fpr_num_slices_per_view", type=int, default=1, help="odd number of slices per view for 2.5D input")
    args = parser.parse_args()

    pos_dir = Path(args.pos_dir)
    neg_dir = Path(args.neg_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = _infer_metadata_path(pos_dir, neg_dir, args.metadata_path)
    metadata = _load_metadata(metadata_path) if metadata_path.exists() else {"samples": []}
    samples = _build_samples(pos_dir, neg_dir, metadata)
    samples = _filter_by_source_split(samples, args.allowed_source_splits)
    before_diam_filter = len(samples)
    samples = _filter_by_candidate_diameter(
        samples,
        min_diam_mm=args.min_candidate_diam_mm,
        max_diam_mm=args.max_candidate_diam_mm,
    )
    for sample in samples:
        if sample["label_id"] == 0:
            sample["neg_type"] = _infer_negative_type(
                sample,
                hard_negative_score_thresh=args.hard_negative_score_thresh,
                hard_negative_iou_max=args.hard_negative_iou_max,
                near_miss_iou_min=args.near_miss_iou_min,
                near_miss_iou_max=args.near_miss_iou_max,
            )
        else:
            sample["neg_type"] = "positive"

    if not samples:
        raise RuntimeError("No FPR patches found. Run collect_fpr_data.py first.")

    train_samples, val_samples, train_scans, val_scans = _split_by_scan(samples, args.val_ratio, args.seed)
    if not train_samples or not val_samples:
        raise RuntimeError("Scan-level split produced an empty train or validation split.")

    logger.info("Loaded %d FPR patches", len(samples))
    if args.min_candidate_diam_mm is not None or args.max_candidate_diam_mm is not None:
        logger.info(
            "  candidate diameter filter: min=%s max=%s kept=%d/%d",
            "None" if args.min_candidate_diam_mm is None else f"{float(args.min_candidate_diam_mm):.2f}mm",
            "None" if args.max_candidate_diam_mm is None else f"{float(args.max_candidate_diam_mm):.2f}mm",
            len(samples),
            before_diam_filter,
        )
    logger.info("  train: %s", _summarize_samples(train_samples))
    logger.info("  val:   %s", _summarize_samples(val_samples))

    with open(output_dir / "data_split.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "val_ratio": args.val_ratio,
                "train_summary": _summarize_samples(train_samples),
                "val_summary": _summarize_samples(val_samples),
                "train_scan_ids": train_scans,
                "val_scan_ids": val_scans,
                "metadata_path": str(metadata_path) if metadata_path.exists() else None,
                "min_candidate_diam_mm": args.min_candidate_diam_mm,
                "max_candidate_diam_mm": args.max_candidate_diam_mm,
                "n_samples_before_candidate_diam_filter": before_diam_filter,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    from .fpr_model import (
        build_fpr_model,
        normalize_fpr_backbone,
        normalize_fpr_model_type,
        normalize_fpr_num_slices_per_view,
    )

    model_type = normalize_fpr_model_type(args.fpr_model_type)
    backbone_name = normalize_fpr_backbone(args.fpr_backbone)
    num_slices_per_view = normalize_fpr_num_slices_per_view(args.fpr_num_slices_per_view)
    if model_type == "3d":
        num_slices_per_view = 1
    logger.info(
        "FPR model config: type=%s backbone=%s num_slices_per_view=%d",
        model_type,
        backbone_name,
        num_slices_per_view,
    )
    train_ds = FPRPatchDataset(
        train_samples,
        model_type=model_type,
        num_slices_per_view=num_slices_per_view,
        augment=True,
        augment_rotate_prob=args.augment_rotate_prob,
        augment_noise=args.augment_noise,
    )
    val_ds = FPRPatchDataset(
        val_samples,
        model_type=model_type,
        num_slices_per_view=num_slices_per_view,
        augment=False,
    )

    train_labels = np.array([sample["label_id"] for sample in train_samples], dtype=np.int64)
    n_pos = int(np.sum(train_labels == 1))
    n_neg = int(np.sum(train_labels == 0))
    sample_weights = np.where(train_labels == 1, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
    neg_type_counts = defaultdict(int)
    small_positive_count = 0
    low_score_positive_count = 0
    for idx, sample in enumerate(train_samples):
        if sample["label_id"] == 1:
            sample_weights[idx] *= float(args.positive_weight)
            diameter = _safe_float(sample.get("effective_diameter_mm"))
            score = _safe_float(sample.get("score"))
            if diameter is not None and diameter <= float(args.small_positive_diam):
                sample_weights[idx] *= float(args.small_positive_weight)
                small_positive_count += 1
            if score is not None and score < float(args.low_score_positive_thresh):
                sample_weights[idx] *= float(args.low_score_positive_weight)
                low_score_positive_count += 1
            continue
        neg_type = sample.get("neg_type") or "easy_negative"
        train_samples[idx]["neg_type"] = neg_type
        neg_type_counts[neg_type] += 1
        if neg_type == "hard_negative":
            sample_weights[idx] *= float(args.hard_negative_weight)
        elif neg_type == "near_miss_negative":
            sample_weights[idx] *= float(args.near_miss_weight)
        else:
            sample_weights[idx] *= float(args.easy_negative_weight)
    if n_neg > 0:
        logger.info(
            "Negative sampling weights: hard=%.2fx (%d) | near_miss=%.2fx (%d) | easy=%.2fx (%d)",
            float(args.hard_negative_weight),
            int(neg_type_counts.get("hard_negative", 0)),
            float(args.near_miss_weight),
            int(neg_type_counts.get("near_miss_negative", 0)),
            float(args.easy_negative_weight),
            int(neg_type_counts.get("easy_negative", 0)),
        )
    logger.info(
        "Positive sampling weights: base=%.2fx | small<=%.2fmm %.2fx (%d) | low_score<%.2f %.2fx (%d)",
        float(args.positive_weight),
        float(args.small_positive_diam),
        float(args.small_positive_weight),
        int(small_positive_count),
        float(args.low_score_positive_thresh),
        float(args.low_score_positive_weight),
        int(low_score_positive_count),
    )
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_fpr_model(
        model_type=model_type,
        backbone_name=backbone_name,
        num_slices_per_view=num_slices_per_view,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "loss": args.loss,
                "focal_gamma": args.focal_gamma,
                "augment_rotate_prob": args.augment_rotate_prob,
                "augment_noise": args.augment_noise,
                "seed": args.seed,
                "device": str(device),
                "allowed_source_splits": args.allowed_source_splits,
                "min_candidate_diam_mm": args.min_candidate_diam_mm,
                "max_candidate_diam_mm": args.max_candidate_diam_mm,
                "hard_negative_weight": args.hard_negative_weight,
                "near_miss_weight": args.near_miss_weight,
                "easy_negative_weight": args.easy_negative_weight,
                "positive_weight": args.positive_weight,
                "small_positive_weight": args.small_positive_weight,
                "small_positive_diam": args.small_positive_diam,
                "low_score_positive_weight": args.low_score_positive_weight,
                "low_score_positive_thresh": args.low_score_positive_thresh,
                "hard_negative_score_thresh": args.hard_negative_score_thresh,
                "hard_negative_iou_max": args.hard_negative_iou_max,
                "near_miss_iou_min": args.near_miss_iou_min,
                "near_miss_iou_max": args.near_miss_iou_max,
                "model_type": model_type,
                "backbone_name": backbone_name,
                "num_slices_per_view": num_slices_per_view,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    best_val_f1 = -1.0
    best_epoch = 0
    best_threshold = 0.5
    patience_counter = 0
    history: List[Dict] = []

    logger.info("Start training FPR model")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False, ncols=100):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = _compute_loss(logits, batch_y, args.loss, args.focal_gamma)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += int((preds == batch_y).sum().item())
            train_total += batch_x.size(0)

        scheduler.step()
        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss = 0.0
        val_total = 0
        val_probs_list: List[np.ndarray] = []
        val_labels_list: List[np.ndarray] = []

        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", leave=False, ncols=100):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                loss = _compute_loss(logits, batch_y, args.loss, args.focal_gamma)
                val_loss += loss.item() * batch_x.size(0)
                val_total += batch_x.size(0)
                probs = torch.softmax(logits, dim=1)[:, 1]
                val_probs_list.append(probs.detach().cpu().numpy())
                val_labels_list.append(batch_y.detach().cpu().numpy())

        val_loss /= max(val_total, 1)
        y_val = np.concatenate(val_labels_list).astype(np.int64) if val_labels_list else np.zeros((0,), dtype=np.int64)
        p_val = np.concatenate(val_probs_list).astype(np.float32) if val_probs_list else np.zeros((0,), dtype=np.float32)
        val_best = _compute_best_threshold(y_val, p_val) if len(y_val) > 0 else {
            "threshold": 0.5,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        val_threshold = float(val_best["threshold"])
        val_f1 = float(val_best["f1"])
        val_precision = float(val_best["precision"])
        val_recall = float(val_best["recall"])
        if len(y_val) > 0:
            val_pred = p_val >= val_threshold
            val_acc = float(np.mean(val_pred == y_val))
        else:
            val_acc = 0.0
        val_breakdown = _val_breakdown(
            val_samples,
            y_val,
            p_val,
            val_threshold,
            small_positive_diam=args.small_positive_diam,
            low_score_positive_thresh=args.low_score_positive_thresh,
        ) if len(y_val) > 0 else {}

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "val_best_threshold": val_threshold,
            **val_breakdown,
        }
        history.append(epoch_info)

        logger.info(
            "Epoch %02d/%d | train_loss=%.4f train_acc=%.3f | val_loss=%.4f val_f1=%.3f @ %.2f (P=%.3f R=%.3f) | smallR=%.3f lowR=%.3f hardFPR=%.3f",
            epoch + 1,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_f1,
            val_threshold,
            val_precision,
            val_recall,
            float(val_breakdown.get("val_recall_small_pos", float("nan"))),
            float(val_breakdown.get("val_recall_low_score_pos", float("nan"))),
            float(val_breakdown.get("val_fpr_hard_neg", float("nan"))),
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_threshold = val_threshold
            patience_counter = 0
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "val_f1": best_val_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "best_threshold": best_threshold,
                    "loss": args.loss,
                    "focal_gamma": args.focal_gamma,
                    "model_type": model_type,
                    "backbone_name": backbone_name,
                    "num_slices_per_view": num_slices_per_view,
                },
                output_dir / "model_best.pt",
            )
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    logger.info("Training finished")
    logger.info("  best_epoch: %d", best_epoch)
    logger.info("  best_val_f1: %.4f", best_val_f1)
    logger.info("  best_threshold: %.2f", best_threshold)
    logger.info("  model: %s", output_dir / "model_best.pt")


if __name__ == "__main__":
    main()
