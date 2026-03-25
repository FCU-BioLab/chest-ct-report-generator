#!/usr/bin/env python3
"""
Train a learned Stage-2 fuser using detector score + FPR probability + optional rich features.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .fpr_fuser import (
    BASIC_FEATURES,
    EXTENDED_FEATURES,
    build_features,
    build_fpr_fuser,
)
from .fpr_model import get_model_type_from_model, load_fpr_model, patch_to_model_input
from .train_fpr import (
    _build_samples,
    _filter_by_source_split,
    _infer_metadata_path,
    _load_metadata,
    _split_by_scan,
    _summarize_samples,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FuserPatchDataset(Dataset):
    def __init__(self, samples: Sequence[Dict], model_type: str):
        self.samples = list(samples)
        self.model_type = model_type

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        patch_3d = np.load(sample["file_path"]).astype(np.float32)
        patch = patch_to_model_input(patch_3d, self.model_type)
        det_score = float(sample.get("score", 0.0))
        label = int(sample["label_id"])

        # Additional cues for harder FP suppression.
        pred_box = sample.get("pred_box_yxz")
        if isinstance(pred_box, (list, tuple)) and len(pred_box) == 6:
            dy = max(float(pred_box[3]) - float(pred_box[0]), 0.0)
            dx = max(float(pred_box[4]) - float(pred_box[1]), 0.0)
            dz = max(float(pred_box[5]) - float(pred_box[2]), 0.0)
        else:
            dy = dx = dz = 0.0
        volume = dy * dx * dz
        min_axis = max(min(dy, dx, dz), 1e-3)
        max_axis = max(dy, dx, dz)
        elongation = max_axis / min_axis

        patch_mean = float(np.mean(patch_3d))
        patch_std = float(np.std(patch_3d))
        patch_p90 = float(np.percentile(patch_3d, 90))
        aux = np.asarray(
            [np.log1p(volume), elongation, patch_mean, patch_std, patch_p90],
            dtype=np.float32,
        )

        return (
            torch.from_numpy(patch),
            torch.tensor(det_score, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.from_numpy(aux),
        )


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


def _resolve_feature_names(feature_set: str) -> List[str]:
    value = str(feature_set).strip().lower()
    if value == "basic":
        return list(BASIC_FEATURES)
    if value == "extended":
        return list(EXTENDED_FEATURES)
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def _extract_feature_tensors(
    samples: Sequence[Dict],
    fpr_model,
    fpr_model_type: str,
    feature_names: Sequence[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ds = FuserPatchDataset(samples, model_type=fpr_model_type)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_feats: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for patch, det_score, label, aux in tqdm(loader, desc="Extracting features", leave=False, ncols=100):
            patch = patch.to(device)
            det_score = det_score.to(device)
            aux = aux.to(device)
            logits = fpr_model(patch)
            fpr_prob = torch.softmax(logits, dim=1)[:, 1]
            extra = {
                "log_volume": aux[:, 0],
                "elongation": aux[:, 1],
                "patch_mean": aux[:, 2],
                "patch_std": aux[:, 3],
                "patch_p90": aux[:, 4],
            }
            feats = build_features(det_score, fpr_prob, extra_features=extra, feature_names=feature_names)
            all_feats.append(feats.cpu())
            all_labels.append(label.cpu())

    x = torch.cat(all_feats, dim=0).float()
    y = torch.cat(all_labels, dim=0).float()
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned Stage-2 fuser.")
    parser.add_argument("--pos_dir", default="detection/results/fpr_dataset/positive", help="positive patch directory")
    parser.add_argument("--neg_dir", default="detection/results/fpr_dataset/negative", help="negative patch directory")
    parser.add_argument("--metadata_path", default=None, help="metadata.json from collect_fpr_data.py")
    parser.add_argument("--fpr_model", required=True, help="trained FPR model checkpoint")
    parser.add_argument("--output_dir", default="detection/results/fpr_fuser_linear", help="output directory")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="validation scan ratio")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--feature_batch_size", type=int, default=128, help="batch size for FPR feature extraction")
    parser.add_argument("--epochs", type=int, default=300, help="training epochs")
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
    parser.add_argument("--patience", type=int, default=40, help="early stopping patience")
    parser.add_argument("--allowed_source_splits", nargs="*", default=["training"], help="metadata source splits allowed for fuser training")
    parser.add_argument("--hard_negative_weight", type=float, default=3.0, help="extra loss weight multiplier for hard negatives")
    parser.add_argument("--hard_negative_score_thresh", type=float, default=0.5, help="detector score threshold defining hard negatives")
    parser.add_argument("--hard_negative_iou_max", type=float, default=0.05, help="maximum IoU for hard negatives")
    parser.add_argument("--feature_set", choices=["basic", "extended"], default="extended", help="feature recipe for fuser")
    parser.add_argument("--fuser_arch", choices=["linear", "mlp"], default="mlp", help="fuser architecture")
    parser.add_argument("--hidden_dims", type=int, nargs="*", default=[32, 16], help="hidden dims for mlp fuser")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout for mlp fuser")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="optimizer weight decay")
    args = parser.parse_args()

    pos_dir = Path(args.pos_dir)
    neg_dir = Path(args.neg_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = _infer_metadata_path(pos_dir, neg_dir, args.metadata_path)
    metadata = _load_metadata(metadata_path) if metadata_path.exists() else {"samples": []}
    samples = _build_samples(pos_dir, neg_dir, metadata)
    samples = _filter_by_source_split(samples, args.allowed_source_splits)
    samples = [s for s in samples if s.get("score") is not None]
    if not samples:
        raise RuntimeError("No valid FPR patches with detector scores found.")

    train_samples, val_samples, train_scans, val_scans = _split_by_scan(samples, args.val_ratio, args.seed)
    if not train_samples or not val_samples:
        raise RuntimeError("Scan-level split produced an empty train or validation split.")

    feature_names = _resolve_feature_names(args.feature_set)

    logger.info("Loaded %d fusion samples", len(samples))
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
                "fpr_model": args.fpr_model,
                "feature_set": args.feature_set,
                "feature_names": feature_names,
                "fuser_arch": args.fuser_arch,
                "hidden_dims": args.hidden_dims,
                "dropout": args.dropout,
                "hard_negative_weight": args.hard_negative_weight,
                "hard_negative_score_thresh": args.hard_negative_score_thresh,
                "hard_negative_iou_max": args.hard_negative_iou_max,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fpr_model = load_fpr_model(args.fpr_model, device=str(device))
    fpr_model_type = get_model_type_from_model(fpr_model)
    logger.info("FPR backbone type for fuser features: %s", fpr_model_type)
    logger.info("Fuser feature_set=%s (%s)", args.feature_set, ", ".join(feature_names))

    logger.info("Extracting train features...")
    x_train, y_train = _extract_feature_tensors(
        train_samples,
        fpr_model,
        fpr_model_type,
        feature_names,
        device,
        args.feature_batch_size,
        args.num_workers,
    )
    logger.info("Extracting val features...")
    x_val, y_val = _extract_feature_tensors(
        val_samples,
        fpr_model,
        fpr_model_type,
        feature_names,
        device,
        args.feature_batch_size,
        args.num_workers,
    )

    model = build_fpr_fuser(
        model_arch=args.fuser_arch,
        input_dim=len(feature_names),
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_pos = int(torch.sum(y_train == 1).item())
    n_neg = int(torch.sum(y_train == 0).item())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    train_sample_weights = torch.ones_like(y_train)
    if args.hard_negative_weight > 1.0:
        hard_flags = []
        for sample in train_samples:
            if sample["label_id"] != 0:
                hard_flags.append(False)
                continue
            score = sample.get("score")
            iou = sample.get("iou")
            score_ok = (score is not None) and (float(score) >= args.hard_negative_score_thresh)
            iou_ok = (iou is None) or (float(iou) <= args.hard_negative_iou_max)
            hard_flags.append(score_ok and iou_ok)
        hard_flags_t = torch.as_tensor(hard_flags, dtype=torch.bool, device=device)
        train_sample_weights[hard_flags_t] = float(args.hard_negative_weight)
        logger.info(
            "Hard-negative weighting for fuser: +%.2fx on %d/%d train negatives",
            float(args.hard_negative_weight),
            int(torch.sum(hard_flags_t).item()),
            int(torch.sum(y_train == 0).item()),
        )

    best_val_f1 = -1.0
    best_epoch = 0
    best_threshold = 0.5
    best_metrics = {}
    patience_counter = 0
    history: List[Dict] = []

    logger.info("Start training fuser")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss_per_sample = criterion(logits, y_train)
        loss = (loss_per_sample * train_sample_weights).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_logits = model(x_val)
            val_loss = criterion(val_logits, y_val).mean().item()
            val_probs = torch.sigmoid(val_logits).detach().cpu().numpy()
            y_val_np = y_val.detach().cpu().numpy().astype(np.int64)
            val_best = _compute_best_threshold(y_val_np, val_probs)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss),
                "val_best_f1": float(val_best["f1"]),
                "val_best_threshold": float(val_best["threshold"]),
                "val_precision": float(val_best["precision"]),
                "val_recall": float(val_best["recall"]),
            }
        )
        logger.info(
            "Epoch %03d/%d | train_loss=%.4f val_loss=%.4f | val_best_f1=%.4f @ %.2f (P=%.3f R=%.3f)",
            epoch + 1,
            args.epochs,
            float(loss.item()),
            float(val_loss),
            float(val_best["f1"]),
            float(val_best["threshold"]),
            float(val_best["precision"]),
            float(val_best["recall"]),
        )

        if val_best["f1"] > best_val_f1:
            best_val_f1 = float(val_best["f1"])
            best_epoch = epoch + 1
            best_threshold = float(val_best["threshold"])
            best_metrics = val_best
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "best_threshold": best_threshold,
                    "feature_names": list(feature_names),
                    "model_arch": args.fuser_arch,
                    "hidden_dims": list(args.hidden_dims),
                    "dropout": float(args.dropout),
                    "val_metrics": best_metrics,
                },
                output_dir / "model_best.pt",
            )
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    logger.info("Fuser training finished")
    logger.info("  best_epoch: %d", best_epoch)
    logger.info("  best_val_f1: %.4f", best_val_f1)
    logger.info("  best_threshold: %.2f", best_threshold)
    logger.info("  model: %s", output_dir / "model_best.pt")


if __name__ == "__main__":
    main()
