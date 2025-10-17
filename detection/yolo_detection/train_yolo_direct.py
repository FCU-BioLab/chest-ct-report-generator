#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11 Direct Training Script for Preprocessed YOLO Format Dataset

This script trains YOLOv11 on already-formatted YOLO datasets with automatic
patient-based train/validation splitting to prevent data leakage.

Usage:
    python train_yolo_direct.py --data_dir ./datasets/splited_dataset/train \\
                                --epochs 200 --batch_size 16 --model_size m
"""

import argparse
from html import parser
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
from tqdm import tqdm

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

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
    max_negative_ratio: float = 1.0         # Max ratio of negative samples (empty labels) to positive samples
    
    # Optimizer settings
    optimizer: str = "AdamW"                # SGD/Adam/AdamW
    weight_decay: float = 0.0005
    momentum: float = 0.937                 # For SGD
    warmup_epochs: int = 5
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
    patience: int = 100
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
    
    for patient_dir in sorted(data_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        images_dir = patient_dir / "images"
        labels_dir = patient_dir / "labels"
        if not images_dir.exists() or not labels_dir.exists():
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


def create_yolo_dataset(patient_data, train_patients, val_patients, output_dir, max_negative_ratio=1.0, random_seed=42):
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
        max_negatives = int(len(positive_samples) * max_negative_ratio)
        if len(negative_samples) > max_negatives and max_negative_ratio < float('inf'):
            random.seed(random_seed)
            negative_samples = random.sample(negative_samples, max_negatives)
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
    logging.info("=" * 80)
    logging.info("Starting YOLOv11 Training")
    logging.info("=" * 80)
    model_name = config.model if config.model else f"yolo11{config.model_size}.pt"
    logging.info(f"Loading model: {model_name}")
    model = YOLO(model_name)
    train_args = {
        "data": str(dataset_yaml),
        "imgsz": config.imgsz,
        "epochs": config.num_epochs,
        "batch": config.batch_size,
        "device": select_device(config.device),
        "optimizer": config.optimizer,
        "lr0": config.learning_rate,
        "lrf": 0.01,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "warmup_epochs": config.warmup_epochs,
        "cos_lr": config.cos_lr,
        "degrees": config.degrees,
        "translate": config.translate,
        "scale": config.scale,
        "fliplr": config.fliplr,
        "mosaic": config.mosaic,
        "mixup": config.mixup,
        "project": str(Path(config.save_dir) / "training"),
        "name": f"yolo11{config.model_size}_{timestamp}",
        "exist_ok": True,
        "save": True,
        "val": True,
        "patience": config.patience,
        "workers": config.workers,
        "verbose": True,
        "plots": True,
        "seed": config.random_seed,
    }
    results = model.train(**train_args)
    return {"success": True, "results": results}


# ========== Main Pipeline ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="",help="Path to the custom YOLO model .yaml or .pt file")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model_size", type=str, default="m", choices=["n","s","m","l","x"])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_negative_ratio", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--no_cos_lr", action="store_true")
    parser.add_argument("--mosaic", type=float, default=0.4)
    parser.add_argument("--mixup", type=float, default=0.1)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.3)
    parser.add_argument("--fliplr", type=float, default=0.5)
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
    dataset_yaml = create_yolo_dataset(patient_data, train_patients, val_patients, dataset_dir, config.max_negative_ratio, config.random_seed)

    train_results = train_yolo(config, dataset_yaml, timestamp)
    if not train_results["success"]:
        logging.error("Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
 