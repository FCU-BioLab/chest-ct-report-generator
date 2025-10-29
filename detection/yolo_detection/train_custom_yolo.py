#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11 Direct Training Script for Preprocessed YOLO Format Dataset

This script trains YOLOv11 on already-formatted YOLO datasets with automatic
patient-based train/validation splitting to prevent data leakage.

Usage:
    python train_custom_yolo.py ^
        --model models/yolo11_custom_ct_s_optimize.yaml ^
        --data_dir ../../datasets/splits_yolo_lesion/train ^
        --epochs 300 ^
        --batch_size 16 ^
        --imgsz 640 ^
        --lr 0.0007 ^
        --max_negative_ratio 0.3 ^
        --oversample_positive 2.0 ^
        --warmup_epochs 15 ^
        --fliplr 0.5 ^
        --scale 0.3 ^
        --translate 0.15 ^
        --degrees 3.0 ^
        --workers 8 ^
        --optimizer AdamW
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


# ========== Focal Loss Implementation ==========
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by down-weighting easy examples.
    Designed to improve detection of small lesions in CT scans.
    
    Args:
        alpha (float): Weighting factor in range (0,1) to balance positive/negative examples
        gamma (float): Focusing parameter for modulating loss. Higher gamma increases focus on hard examples
    
    Reference: https://arxiv.org/abs/1708.02002
    
    ⚠️ NOTE: This class is currently a PLACEHOLDER and NOT integrated with YOLO training.
    Ultralytics YOLO does not support custom loss functions via model.train() API.
    To use Focal Loss, you would need to:
    1. Modify the YOLO source code to replace the default loss
    2. Or use class weights and sample rebalancing (current approach)
    
    The current implementation achieves similar effects through:
    - Positive sample oversampling (oversample_positive parameter)
    - Negative sample ratio control (max_negative_ratio parameter)
    - Adjusted loss weights (box, cls, dfl parameters)
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE
        return focal_loss.mean()


# ========== Configuration ==========
@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults for medical CT detection."""
    
    # Data paths
    data_dir: str                           # Path to train split (e.g., datasets/splited_dataset/train)
    model: Optional[str] = None             # Placeholder for YOLO model type
    save_dir: str = "./yolo_runs"          # Where to save models and logs
    
    # Training parameters
    num_epochs: int = 200
    batch_size: int = 16
    imgsz: int = 640
    model_size: str = "m"                   # n/s/m/l/x
    learning_rate: float = 0.001
    
    # Validation split
    val_ratio: float = 0.2                  # Ratio of patients for validation
    random_seed: int = 42
    
    # Class balancing
    max_negative_ratio: float = 0.3         # ✅ Max ratio of negative samples (empty labels) to positive samples
    oversample_positive: float = 2.0        # ✅ Positive sample oversampling multiplier (e.g., 2.0 = duplicate positive samples)
    
    # Focal Loss settings
    use_focal_loss: bool = False            # ✅ Enable Focal Loss for class imbalance (experimental with YOLO)
    focal_alpha: float = 0.75               # ✅ Focal Loss alpha parameter
    focal_gamma: float = 2.0                # ✅ Focal Loss gamma parameter
    
    # Optimizer settings
    optimizer: str = "AdamW"                # SGD/Adam/AdamW
    weight_decay: float = 0.0005
    momentum: float = 0.937                 # For SGD only
    warmup_epochs: int = 10                 # Increased from 5 for better stability
    cos_lr: bool = True                     # Cosine LR scheduler
    
    # Augmentation
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    fliplr: float = 0.5
    flipud: float = 0.0
    
    # Advanced settings
    patience: int = 50
    save_period: int = 10
    workers: int = 8
    device: str = "auto"
    
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ========== Logging Setup ==========
def setup_logging(log_dir: Path, timestamp: str) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_{timestamp}.log"
    
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    fh = logging.FileHandler(log_file, encoding="utf-8", mode="w")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return str(log_file)


# ========== Device Selection ==========
def select_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ========== Reproducibility ==========
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ========== Dataset Preparation ==========
def is_negative_sample(label_path: Path) -> bool:
    try:
        content = label_path.read_text(encoding="utf-8").strip()
        return len(content) == 0
    except Exception:
        return True


def collect_patient_data(data_dir: Path) -> Dict[str, List[Tuple[Path, Path, bool]]]:
    patient_data = {}
    positive_count = 0
    negative_count = 0
    
    logging.info(f"Scanning dataset directory: {data_dir}")
    
    # ✅ Auto-detect folder structure (supports both "images" and "images_png")
    images_base_dir = None
    for possible_name in ["images", "images_png"]:
        candidate = data_dir / possible_name
        if candidate.exists():
            images_base_dir = candidate
            logging.info(f"  Found images directory: {possible_name}/")
            break
    
    labels_base_dir = data_dir / "labels"
    
    if images_base_dir is None or not labels_base_dir.exists():
        logging.error(f"Missing images (or images_png) or labels directory in {data_dir}")
        logging.error(f"  Expected structure: {data_dir}/[images|images_png]/PATIENT_ID/*.png")
        logging.error(f"                      {data_dir}/labels/PATIENT_ID/*.txt")
        return patient_data
    
    for patient_dir in sorted(images_base_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        images_dir = patient_dir
        labels_dir = labels_base_dir / patient_id
        if not labels_dir.exists():
            logging.warning(f"Labels directory missing for patient {patient_id}")
            continue
        
        pairs = []
        for img_file in sorted(images_dir.glob("*.png")):
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                is_neg = is_negative_sample(label_file)
                pairs.append((img_file, label_file, is_neg))
                if is_neg:
                    negative_count += 1
                else:
                    positive_count += 1
        if pairs:
            patient_data[patient_id] = pairs
    
    logging.info(f"Found {len(patient_data)} patients")
    logging.info(f"  Positive samples: {positive_count}")
    logging.info(f"  Negative samples: {negative_count}")
    logging.info(f"  Negative/Positive ratio: {negative_count/max(positive_count, 1):.2f}")
    return patient_data


def split_patients(patient_ids: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    random.seed(seed)
    shuffled = patient_ids.copy()
    random.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_patients = shuffled[:val_count]
    train_patients = shuffled[val_count:]
    return train_patients, val_patients


# ✅ 新增這個函式：確保 val split 至少有一個正樣本
def ensure_val_has_positive_samples(patient_data: Dict[str, List[Tuple[Path, Path, bool]]],
                                    val_patients: List[str],
                                    train_patients: List[str]) -> Tuple[List[str], List[str]]:
    def has_positive(patient_id: str) -> bool:
        for _, label_path, is_neg in patient_data[patient_id]:
            if not is_neg:
                return True
        return False

    if any(has_positive(pid) for pid in val_patients):
        return train_patients, val_patients

    logging.warning("Validation split contains no positive samples. Adjusting...")
    for pid in train_patients:
        if has_positive(pid):
            train_patients.remove(pid)
            val_patients.append(pid)
            logging.info(f"Moved patient {pid} from train to val to ensure positive sample presence")
            break

    return train_patients, val_patients


def create_yolo_dataset(patient_data, train_patients, val_patients, output_dir, max_negative_ratio=0.3, oversample_positive=1.0, random_seed=42):
    logging.info(f"Creating YOLO dataset at {output_dir}")
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    def link_files(patient_ids, split):
        positive_samples = []
        negative_samples = []
        for patient_id in patient_ids:
            for img_path, label_path, is_neg in patient_data[patient_id]:
                if is_neg:
                    negative_samples.append((patient_id, img_path, label_path))
                else:
                    positive_samples.append((patient_id, img_path, label_path))
        
        # ✅ Apply rebalancing ONLY to training split
        original_positive_count = len(positive_samples)
        original_negative_count = len(negative_samples)
        
        if split == "train":
            # Oversample positive samples for training split only
            if oversample_positive > 1.0:
                target_count = int(original_positive_count * oversample_positive)
                # Repeat full copies
                full_copies = int(oversample_positive)
                positive_samples = positive_samples * full_copies
                # Add random samples to reach target count
                remaining = target_count - len(positive_samples)
                if remaining > 0:
                    random.seed(random_seed)
                    positive_samples.extend(random.choices(positive_samples[:original_positive_count], k=remaining))
                logging.info(f"  [Train] Oversampling positive samples: {original_positive_count} × {oversample_positive} = {len(positive_samples)}")
            
            # Control negative sample ratio for training split only
            max_negatives = int(len(positive_samples) * max_negative_ratio)
            if len(negative_samples) > max_negatives and max_negative_ratio < float('inf'):
                random.seed(random_seed)
                negative_samples = random.sample(negative_samples, max_negatives)
                logging.info(f"  [Train] Limiting negative samples: {original_negative_count} → {len(negative_samples)} (ratio={max_negative_ratio})")
        else:
            # ⚠️ CRITICAL: Validation split keeps ORIGINAL distribution
            logging.info(f"  [Val] Keeping original distribution: {original_positive_count} positive, {original_negative_count} negative")
        
        all_samples = positive_samples + negative_samples
        
        img_count = pos_count = neg_count = 0
        for patient_id, img_path, label_path in tqdm(all_samples, desc=f"Preparing {split} split"):
            new_name = f"{patient_id}_{img_path.name}"
            target_img = output_dir / "images" / split / new_name
            target_label = output_dir / "labels" / split / f"{new_name.replace('.png', '.txt')}"
            try:
                shutil.copy2(img_path, target_img)
                shutil.copy2(label_path, target_label)
                img_count += 1
                if is_negative_sample(label_path):
                    neg_count += 1
                else:
                    pos_count += 1
            except Exception as e:
                logging.error(f"Failed to copy {img_path.name}: {e}")
        logging.info(f"{split.capitalize()}: {img_count} images (pos {pos_count}, neg {neg_count})")
        return img_count, pos_count, neg_count

    train_count, train_pos, train_neg = link_files(train_patients, "train")
    val_count, val_pos, val_neg = link_files(val_patients, "val")

    yaml_content = f"""# YOLOv11 Dataset Configuration
path: {output_dir.absolute().as_posix()}
train: images/train
val: images/val
nc: 1
names: ['lesion']
"""
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


# ========== Training ==========
def train_yolo(config: TrainingConfig, dataset_yaml: Path, timestamp: str) -> Dict[str, Any]:

    # ========== NaN Debug (optional) ==========
    # logging.info("🔍 Running 1-batch NaN detection check...")
    # torch.autograd.set_detect_anomaly(True)

    # debug_model = YOLO(config.model or "models/yolo11_custom_ct_s_optimize.yaml")
    # debug_model.train(
    #     data=str(dataset_yaml),
    #     epochs=1,
    #     batch=4,
    #     imgsz=config.imgsz,
    #     workers=0,
    #     device=select_device(config.device),
    #     amp=False,
    #     val=False,
    #     lr0=2e-4,
    #     warmup_epochs=5,
    #     mosaic=0.0,
    #     mixup=0.0,
    #     translate=0.05,
    #     scale=0.0,
    #     rect=True,
    #     freeze=10,
    #     cos_lr=False,
    #     fraction=0.02,      # ✅ 只取 2% 的資料快速 smoke test
    # )


    # --- 預先下載 AMP 檢查所需的 yolo11n.pt（避免訓練中途下載）---
    try:
        logging.info("Ensuring yolo11n.pt is available for AMP checks...")
        _ = YOLO("yolo11n.pt")  # 觸發自動下載
        logging.info("✅ yolo11n.pt ready")
    except Exception as e:
        logging.warning(f"Failed to pre-download yolo11n.pt: {e}")

    # --- 正式訓練 ---
    logging.info("=" * 80)
    logging.info("🚀 Starting YOLOv11 Custom CT Model Training")
    logging.info("=" * 80)
    logging.info("")
    
    model_name = config.model if config.model else f"yolo11{config.model_size}.pt"
    
    # ✅ Log ONLY custom configurations NOT in args.yaml
    logging.info("📊 Custom Dataset Rebalancing (Pre-processing):")
    logging.info(f"  - Max Negative Ratio: {config.max_negative_ratio} (limits negative samples in training split)")
    logging.info(f"  - Positive Oversampling: {config.oversample_positive}x (duplicates positive samples)")
    logging.info(f"  ⚠️  Note: Validation split keeps original distribution for fair evaluation")
    if config.use_focal_loss:
        logging.warning(f"  - Focal Loss Flag: Enabled (alpha={config.focal_alpha}, gamma={config.focal_gamma})")
        logging.warning(f"    ⚠️  NOT integrated with YOLO API - using sample rebalancing instead")
    logging.info("")
    
    logging.info(f"📁 Dataset Path:")
    logging.info(f"  - YAML: {dataset_yaml}")
    logging.info(f"  ℹ️  All other training hyperparameters will be saved in: runs/.../args.yaml")
    logging.info("")
    
    logging.info(f"🔧 Loading model: {model_name}...")
    model = YOLO(model_name)
    logging.info(f"✅ Model loaded successfully")
    logging.info(f"")

    train_args = {
        "data": str(dataset_yaml),
        "imgsz": config.imgsz,
        "epochs": config.num_epochs,
        "batch": config.batch_size,
        "device": select_device(config.device),
        "optimizer": config.optimizer,
        # Learning rate and optimizer settings - FROM CONFIG
        "lr0": config.learning_rate,
        "lrf": 0.08,
        "momentum": config.momentum,  # Note: Only effective for SGD optimizer
        "weight_decay": config.weight_decay,
        "warmup_epochs": config.warmup_epochs,
        "warmup_momentum": 0.8, 
        "warmup_bias_lr": 0.05, 
        "cos_lr": config.cos_lr,
        # Augmentation settings - FROM CONFIG
        "degrees": config.degrees,
        "translate": config.translate,
        "scale": config.scale,
        "fliplr": config.fliplr,
        "flipud": config.flipud,
        "mosaic": config.mosaic,
        "mixup": config.mixup,
        "copy_paste": config.copy_paste,
        "hsv_h": 0.0,           
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "auto_augment": "none",  
        "erasing": 0.0,
        "project": str(Path(config.save_dir) / "training"),
        "amp": True,            
        "name": f"yolo11{config.model_size}_{timestamp}",
        "exist_ok": True,
        "save": True,
        "val": True,
        "patience": config.patience,
        "workers": config.workers,
        "verbose": True,
        "plots": True,
        "seed": config.random_seed,
        "rect": False,
        "freeze": 0,
        "close_mosaic": 0,      # ✅ 無Mosaic不需close_mosaic
        # 🎯 Optimized Loss Weights for Gradient Stability
        "box": 5.0,             # ✅ Reduced from 7.0 to prevent gradient explosion
        "cls": 1.0,             # ✅ Reduced from 1.0 (cls_loss was abnormally high)
        "dfl": 1.2,             # ✅ Slightly reduced from 1.5
        # Note: Gradient clipping is handled internally by Ultralytics YOLO
        # max_grad_norm is NOT a valid YOLO argument, removed to prevent SyntaxError
        "nbs": 128,             # ✅ Gradient Accumulation
        "dropout": 0.1,         # ✅ 輕度Dropout
    }

    logging.info("=" * 80)
    logging.info("🎯 Initiating Training Process...")
    logging.info("=" * 80)
    logging.info("")

    # Start timing
    import time
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    logging.info(f"⏱️  Training Start Time: {start_time_str}")
    logging.info("")

    try:
        results = model.train(**train_args)
        
        # End timing and calculate metrics
        end_time = time.time()
        end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
        total_time_seconds = end_time - start_time
        
        # Format time
        hours = int(total_time_seconds // 3600)
        minutes = int((total_time_seconds % 3600) // 60)
        seconds = int(total_time_seconds % 60)
        
        # Calculate average time per epoch
        epochs_trained = results.epochs if hasattr(results, 'epochs') else config.num_epochs
        avg_time_per_epoch = total_time_seconds / epochs_trained if epochs_trained > 0 else 0
        avg_minutes = int(avg_time_per_epoch // 60)
        avg_seconds = int(avg_time_per_epoch % 60)
        
        # Calculate throughput (images per second)
        # Assuming train_dataset_size is available from config or results
        train_images = len(model.trainer.train_loader.dataset) if hasattr(model, 'trainer') and hasattr(model.trainer, 'train_loader') else 0
        images_per_epoch = train_images if train_images > 0 else 0
        total_images_processed = images_per_epoch * epochs_trained
        throughput = total_images_processed / total_time_seconds if total_time_seconds > 0 else 0
        
        logging.info("")
        logging.info("=" * 80)
        logging.info("✅ Training Completed Successfully!")
        logging.info("=" * 80)
        logging.info("")
        
        logging.info("⏱️  Training Time Metrics:")
        logging.info(f"  - Start Time: {start_time_str}")
        logging.info(f"  - End Time: {end_time_str}")
        logging.info(f"  - Total Training Time: {hours}h {minutes}m {seconds}s ({total_time_seconds:.2f} seconds)")
        logging.info(f"  - Average Time per Epoch: {avg_minutes}m {avg_seconds}s ({avg_time_per_epoch:.2f} seconds)")
        logging.info("")
        
        logging.info("📊 Training Performance:")
        logging.info(f"  - Total Epochs Trained: {epochs_trained}")
        if images_per_epoch > 0:
            logging.info(f"  - Images per Epoch: {images_per_epoch:,}")
            logging.info(f"  - Total Images Processed: {total_images_processed:,}")
            logging.info(f"  - Throughput: {throughput:.2f} images/second")
            logging.info(f"  - Average Batch Processing Time: {(avg_time_per_epoch / (images_per_epoch / config.batch_size)):.3f}s per batch")
        logging.info("")
        
        logging.info("💾 Model Checkpoints:")
        logging.info(f"  - Best Model: best.pt (saved at best mAP epoch)")
        logging.info(f"  - Last Model: last.pt (final epoch weights)")
        logging.info("")
        
        # Training efficiency metrics
        if hasattr(results, 'results_dict'):
            results_dict = results.results_dict
            if 'metrics/mAP50(B)' in results_dict:
                best_map50 = results_dict['metrics/mAP50(B)']
                logging.info("🎯 Best Performance Metrics:")
                logging.info(f"  - mAP@50: {best_map50:.4f}")
                if 'metrics/mAP50-95(B)' in results_dict:
                    logging.info(f"  - mAP@50-95: {results_dict['metrics/mAP50-95(B)']:.4f}")
                if 'metrics/precision(B)' in results_dict:
                    logging.info(f"  - Precision: {results_dict['metrics/precision(B)']:.4f}")
                if 'metrics/recall(B)' in results_dict:
                    logging.info(f"  - Recall: {results_dict['metrics/recall(B)']:.4f}")
                logging.info("")
        
        return {"success": True, "results": results, "training_time": total_time_seconds}
        
    except Exception as e:
        logging.error("")
        logging.error("=" * 80)
        logging.error("❌ Training Failed!")
        logging.error("=" * 80)
        logging.error(f"Error: {str(e)}")
        logging.error("")
        import traceback
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())
        return {"success": False, "error": str(e)}


# ========== Main Pipeline ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="",help="Path to the custom YOLO model .yaml or .pt file")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=300)  # ✅ 延長至300 epoch補償較低lr
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model_size", type=str, default="s", choices=["n","s","m","l","x"])
    parser.add_argument("--lr", type=float, default=0.0008)  # ✅ 降低默認學習率（從 0.001 → 0.0008）
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_negative_ratio", type=float, default=0.3)  # ✅ Updated default to 0.3
    parser.add_argument("--oversample_positive", type=float, default=2.0, help="Positive sample oversampling multiplier")
    parser.add_argument("--use_focal_loss", action="store_true", help="Enable Focal Loss (experimental)")
    parser.add_argument("--focal_alpha", type=float, default=0.75, help="Focal Loss alpha parameter")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss gamma parameter")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--no_cos_lr", action="store_true")
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default="./yolo_runs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if not ULTRALYTICS_AVAILABLE:
        print("Error: ultralytics not installed.")
        sys.exit(1)

    config = TrainingConfig(
        model=args.model or None,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        model_size=args.model_size,
        learning_rate=args.lr,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
        max_negative_ratio=args.max_negative_ratio,
        oversample_positive=args.oversample_positive,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        cos_lr=not args.no_cos_lr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        workers=args.workers,
        device=args.device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"./yolo_runs/train_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)
    config.save_dir = str(save_dir) 
    log_file = setup_logging(save_dir / "logs", timestamp)
    set_seed(config.random_seed)

    data_dir = Path(config.data_dir)
    patient_data = collect_patient_data(data_dir)
    patient_ids = list(patient_data.keys())
    train_patients, val_patients = split_patients(patient_ids, config.val_ratio, config.random_seed)
    train_patients, val_patients = ensure_val_has_positive_samples(patient_data, val_patients, train_patients)

    dataset_dir = save_dir / f"dataset_{timestamp}"
    dataset_yaml = create_yolo_dataset(patient_data, train_patients, val_patients, dataset_dir, config.max_negative_ratio, config.oversample_positive, config.random_seed)

    train_results = train_yolo(config, dataset_yaml, timestamp)
    if not train_results["success"]:
        logging.error("Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
 