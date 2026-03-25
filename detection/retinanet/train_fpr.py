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
    def __init__(self, samples: Sequence[Dict], model_type: str, augment: bool = False):
        self.samples = list(samples)
        self.model_type = model_type
        self.augment = augment

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
            patch = np.clip(patch + np.random.uniform(-0.05, 0.05), 0.0, 1.0)

        patch = patch_to_model_input(patch, self.model_type)
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


def _build_samples(pos_dir: Path, neg_dir: Path, metadata: Dict) -> List[Dict]:
    samples = []
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
                    "pred_box_yxz": pred_box,
                }
            )
    return samples


def _filter_by_source_split(samples: Sequence[Dict], allowed_splits: Sequence[str]) -> List[Dict]:
    allowed = {split.strip() for split in allowed_splits if split.strip()}
    if not allowed:
        return list(samples)
    return [sample for sample in samples if sample.get("source_split") in allowed]


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
    return {
        "n_samples": n_total,
        "n_positive": n_pos,
        "n_negative": n_total - n_pos,
        "n_scans": len({sample["scan_id"] for sample in samples}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FPR classifier.")
    parser.add_argument("--pos_dir", default="detection/results/fpr_dataset/positive", help="positive patch directory")
    parser.add_argument("--neg_dir", default="detection/results/fpr_dataset/negative", help="negative patch directory")
    parser.add_argument("--metadata_path", default=None, help="metadata.json from collect_fpr_data.py")
    parser.add_argument("--output_dir", default="detection/results/fpr_resnet18", help="output directory")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="validation scan ratio")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience, 0 disables")
    parser.add_argument("--loss", choices=["ce", "focal"], default="focal", help="classification loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="gamma for focal loss")
    parser.add_argument("--allowed_source_splits", nargs="*", default=["training"], help="metadata source splits allowed for FPR training")
    parser.add_argument("--hard_negative_weight", type=float, default=3.0, help="extra sampling weight multiplier for hard negatives")
    parser.add_argument("--hard_negative_score_thresh", type=float, default=0.5, help="detector score threshold defining hard negatives")
    parser.add_argument("--hard_negative_iou_max", type=float, default=0.05, help="maximum IoU for hard negatives")
    parser.add_argument("--fpr_model_type", choices=["3d", "2.5d"], default="3d", help="FPR backbone input type")
    args = parser.parse_args()

    pos_dir = Path(args.pos_dir)
    neg_dir = Path(args.neg_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = _infer_metadata_path(pos_dir, neg_dir, args.metadata_path)
    metadata = _load_metadata(metadata_path) if metadata_path.exists() else {"samples": []}
    samples = _build_samples(pos_dir, neg_dir, metadata)
    samples = _filter_by_source_split(samples, args.allowed_source_splits)

    if not samples:
        raise RuntimeError("No FPR patches found. Run collect_fpr_data.py first.")

    train_samples, val_samples, train_scans, val_scans = _split_by_scan(samples, args.val_ratio, args.seed)
    if not train_samples or not val_samples:
        raise RuntimeError("Scan-level split produced an empty train or validation split.")

    logger.info("Loaded %d FPR patches", len(samples))
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
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    from .fpr_model import build_fpr_model, normalize_fpr_model_type

    model_type = normalize_fpr_model_type(args.fpr_model_type)
    train_ds = FPRPatchDataset(train_samples, model_type=model_type, augment=True)
    val_ds = FPRPatchDataset(val_samples, model_type=model_type, augment=False)

    train_labels = np.array([sample["label_id"] for sample in train_samples], dtype=np.int64)
    n_pos = int(np.sum(train_labels == 1))
    n_neg = int(np.sum(train_labels == 0))
    sample_weights = np.where(train_labels == 1, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
    if args.hard_negative_weight > 1.0:
        hard_neg_flags = []
        for sample in train_samples:
            if sample["label_id"] != 0:
                hard_neg_flags.append(False)
                continue
            score = sample.get("score")
            iou = sample.get("iou")
            score_ok = (score is not None) and (float(score) >= args.hard_negative_score_thresh)
            iou_ok = (iou is None) or (float(iou) <= args.hard_negative_iou_max)
            hard_neg_flags.append(score_ok and iou_ok)
        hard_neg_flags = np.asarray(hard_neg_flags, dtype=bool)
        sample_weights[hard_neg_flags] *= float(args.hard_negative_weight)
        logger.info(
            "Hard-negative weighting: +%.2fx for %d/%d train negatives (score>=%.2f, iou<=%.2f)",
            float(args.hard_negative_weight),
            int(np.sum(hard_neg_flags)),
            int(n_neg),
            float(args.hard_negative_score_thresh),
            float(args.hard_negative_iou_max),
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
    model = build_fpr_model(model_type=model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "loss": args.loss,
                "focal_gamma": args.focal_gamma,
                "seed": args.seed,
                "device": str(device),
                "allowed_source_splits": args.allowed_source_splits,
                "hard_negative_weight": args.hard_negative_weight,
                "hard_negative_score_thresh": args.hard_negative_score_thresh,
                "hard_negative_iou_max": args.hard_negative_iou_max,
                "model_type": model_type,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    best_val_f1 = -1.0
    best_epoch = 0
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
        val_tp = val_fp = val_fn = val_tn = 0

        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", leave=False, ncols=100):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                loss = _compute_loss(logits, batch_y, args.loss, args.focal_gamma)
                val_loss += loss.item() * batch_x.size(0)
                val_total += batch_x.size(0)

                preds = logits.argmax(dim=1)
                val_tp += int(((preds == 1) & (batch_y == 1)).sum().item())
                val_fp += int(((preds == 1) & (batch_y == 0)).sum().item())
                val_fn += int(((preds == 0) & (batch_y == 1)).sum().item())
                val_tn += int(((preds == 0) & (batch_y == 0)).sum().item())

        val_loss /= max(val_total, 1)
        val_precision = val_tp / max(val_tp + val_fp, 1)
        val_recall = val_tp / max(val_tp + val_fn, 1)
        val_f1 = 2.0 * val_precision * val_recall / max(val_precision + val_recall, 1e-6)
        val_acc = (val_tp + val_tn) / max(val_total, 1)

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
        }
        history.append(epoch_info)

        logger.info(
            "Epoch %02d/%d | train_loss=%.4f train_acc=%.3f | val_loss=%.4f val_f1=%.3f (P=%.3f R=%.3f)",
            epoch + 1,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_f1,
            val_precision,
            val_recall,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "val_f1": best_val_f1,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "loss": args.loss,
                    "focal_gamma": args.focal_gamma,
                    "model_type": model_type,
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

    logger.info("Training finished")
    logger.info("  best_epoch: %d", best_epoch)
    logger.info("  best_val_f1: %.4f", best_val_f1)
    logger.info("  model: %s", output_dir / "model_best.pt")


if __name__ == "__main__":
    main()
