#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv7 Training Script with Medical Image Detection Modules

Converted from YOLOv11 Ultralytics API to YOLOv7 native training with:
- Medical preprocessing (HU windowing + CLAHE)
- Optional medical modules (CBAM, SimAM, Swin Transformer, BiFPN)
- Dataset caching
- Multi-GPU support
- EMA and advanced training techniques

Usage:
    python train_yolov7_medical.py --data_dir ./datasets/ct_data --epochs 120 \\
        --use_medical_modules --enable_hu_windowing 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from tqdm import tqdm

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

# Import local modules
try:
    from detection_dataset import CTDetectionDataset
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False

from yolov7_model import load_yolov7_model, YOLOv7Model
from yolov7_dataset import create_yolov7_dataloader
from yolov7_utils import (
    ComputeLoss,
    ModelEMA,
    WarmupCosineSchedule,
    select_device,
    compute_metrics,
)

LOGGER = logging.getLogger(__name__)


# ==================== Configuration ====================
@dataclass
class TrainingConfig:
    """Training configuration for YOLOv7"""
    
    # Data parameters
    data_dir: str
    train_split: str = "train"
    val_split: Optional[str] = None
    val_ratio: float = 0.2
    include_negative_samples: bool = True
    max_negative_per_patient: int = 0
    
    # Training parameters
    num_epochs: int = 120
    batch_size: int = 16
    learning_rate: float = 0.001
    imgsz: int = 640
    
    # Model parameters
    model_config: str = "models/yolov7_medical.yaml"
    use_medical_modules: bool = True
    pretrained_weights: Optional[str] = None
    
    # Optimizer parameters
    optimizer_type: str = "Adam"  # SGD, Adam, AdamW
    weight_decay: float = 5e-4
    momentum: float = 0.937
    warmup_epochs: int = 5
    cos_lr: bool = True
    
    # Medical preprocessing
    enable_hu_windowing: bool = True
    window_center: float = -600.0
    window_width: float = 1500.0
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    robust_percentile_stretch: bool = True
    
    # Advanced training
    use_ema: bool = True
    ema_decay: float = 0.9999
    gradient_clip: float = 10.0
    mixed_precision: bool = True
    multi_scale: bool = True
    
    # Output directories
    save_dir: str = "./yolov7_models"
    log_dir: str = "./yolov7_logs"
    
    # Misc
    random_seed: int = 42
    num_workers: int = 4
    device: str = ""  # Empty = auto-select
    
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==================== Utilities ====================
def setup_logging(log_dir: str) -> str:
    """Setup logging configuration"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"yolov7_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Reset handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    fh = logging.FileHandler(log_file, encoding="utf-8", mode="w")
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(fh)
    root_logger.addHandler(ch)
    
    return str(log_file)


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==================== Dataset Preparation ====================
def prepare_datasets(config: TrainingConfig) -> Tuple[Any, Any]:
    """
    Prepare train and validation datasets
    
    Args:
        config: Training configuration
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    if not DATASET_AVAILABLE:
        raise RuntimeError("CTDetectionDataset is not available")
    
    LOGGER.info("Preparing datasets...")
    
    # Load train dataset
    train_dataset = CTDetectionDataset(
        data_root=config.data_dir,
        split=config.train_split,
        target_size=config.imgsz,
        include_negative_samples=config.include_negative_samples,
        max_negative_per_patient=config.max_negative_per_patient,
        format_type="yolo",
    )
    
    LOGGER.info(f"Train dataset: {len(train_dataset)} samples")
    
    # Prepare validation dataset
    if config.val_split:
        val_dataset = CTDetectionDataset(
            data_root=config.data_dir,
            split=config.val_split,
            target_size=config.imgsz,
            include_negative_samples=config.include_negative_samples,
            max_negative_per_patient=config.max_negative_per_patient,
            format_type="yolo",
        )
    else:
        # Split train dataset
        total_size = len(train_dataset)
        val_size = int(total_size * config.val_ratio)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    LOGGER.info(f"Validation dataset: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: Any,
    val_dataset: Any,
    config: TrainingConfig
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders with medical preprocessing
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    LOGGER.info("Creating dataloaders with medical preprocessing...")
    
    # Training dataloader
    train_loader = create_yolov7_dataloader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        img_size=config.imgsz,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True,
        enable_hu_windowing=config.enable_hu_windowing,
        window_center=config.window_center,
        window_width=config.window_width,
        enable_clahe=config.enable_clahe,
        clahe_clip_limit=config.clahe_clip_limit,
        clahe_tile_size=config.clahe_tile_grid,
        robust_percentile_stretch=config.robust_percentile_stretch,
        augment=True,
    )
    
    # Validation dataloader
    val_loader = create_yolov7_dataloader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        img_size=config.imgsz,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
        enable_hu_windowing=config.enable_hu_windowing,
        window_center=config.window_center,
        window_width=config.window_width,
        enable_clahe=config.enable_clahe,
        clahe_clip_limit=config.clahe_clip_limit,
        clahe_tile_size=config.clahe_tile_grid,
        robust_percentile_stretch=config.robust_percentile_stretch,
        augment=False,
    )
    
    return train_loader, val_loader


# ==================== Training ====================
def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: ComputeLoss,
    device: torch.device,
    epoch: int,
    config: TrainingConfig,
    scaler: Optional[amp.GradScaler] = None,
    ema: Optional[ModelEMA] = None,
) -> Dict[str, float]:
    """
    Train for one epoch
    
    Args:
        model: YOLOv7 model
        train_loader: Training dataloader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device
        epoch: Current epoch
        config: Training configuration
        scaler: AMP gradient scaler
        ema: EMA model
    
    Returns:
        Dictionary of metrics
    """
    model.train()
    
    total_loss = 0.0
    box_loss = 0.0
    obj_loss = 0.0
    cls_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
    
    for batch_idx, (images, targets, metadata) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        if config.mixed_precision and scaler is not None:
            with amp.autocast():
                outputs = model(images)
                loss, loss_items = loss_fn(outputs, targets)
        else:
            outputs = model(images)
            loss, loss_items = loss_fn(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        
        if config.mixed_precision and scaler is not None:
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.gradient_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            if config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update(model)
        
        # Update metrics
        total_loss += loss.item()
        box_loss += loss_items[0].item()
        obj_loss += loss_items[1].item()
        cls_loss += loss_items[2].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'box': f'{loss_items[0].item():.4f}',
            'obj': f'{loss_items[1].item():.4f}',
            'cls': f'{loss_items[2].item():.4f}',
        })
    
    n_batches = len(train_loader)
    
    return {
        'train_loss': total_loss / n_batches,
        'train_box_loss': box_loss / n_batches,
        'train_obj_loss': obj_loss / n_batches,
        'train_cls_loss': cls_loss / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: ComputeLoss,
    device: torch.device,
    config: TrainingConfig,
) -> Dict[str, float]:
    """
    Validate model
    
    Args:
        model: YOLOv7 model
        val_loader: Validation dataloader
        loss_fn: Loss function
        device: Device
        config: Training configuration
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    total_loss = 0.0
    box_loss = 0.0
    obj_loss = 0.0
    cls_loss = 0.0
    
    pbar = tqdm(val_loader, desc="Validation")
    
    for images, targets, metadata in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(images)
        loss, loss_items = loss_fn(outputs, targets)
        
        # Update metrics
        total_loss += loss.item()
        box_loss += loss_items[0].item()
        obj_loss += loss_items[1].item()
        cls_loss += loss_items[2].item()
        
        pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    
    n_batches = len(val_loader)
    
    return {
        'val_loss': total_loss / n_batches,
        'val_box_loss': box_loss / n_batches,
        'val_obj_loss': obj_loss / n_batches,
        'val_cls_loss': cls_loss / n_batches,
    }


def train_yolov7(config: TrainingConfig) -> Dict[str, Any]:
    """
    Main training function
    
    Args:
        config: Training configuration
    
    Returns:
        Training results dictionary
    """
    # Setup
    log_file = setup_logging(config.log_dir)
    LOGGER.info("=" * 80)
    LOGGER.info("YOLOv7 Medical Image Detection Training")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Configuration:\n{json.dumps(config.as_dict(), indent=2)}")
    
    set_global_seed(config.random_seed)
    device = select_device(config.device, config.batch_size)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(config.save_dir) / f"run_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    
    LOGGER.info(f"Save directory: {save_dir}")
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(config)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)
    
    # Load model
    model_cfg_path = SCRIPT_DIR / config.model_config
    if not model_cfg_path.exists() and config.use_medical_modules:
        LOGGER.warning(f"Medical model config not found: {model_cfg_path}")
        LOGGER.warning("Falling back to baseline model")
        model_cfg_path = SCRIPT_DIR / "models" / "yolov7_baseline.yaml"
        config.use_medical_modules = False
    
    LOGGER.info(f"Loading model from {model_cfg_path}")
    LOGGER.info(f"Medical modules: {'ENABLED' if config.use_medical_modules else 'DISABLED'}")
    
    model = load_yolov7_model(
        cfg_path=str(model_cfg_path),
        weights_path=config.pretrained_weights,
        nc=1,  # Single class: lesion
        use_medical_modules=config.use_medical_modules,
        device=str(device),
    )
    
    # Log model parameters
    params = model.count_parameters()
    LOGGER.info("=" * 80)
    LOGGER.info("Model Parameters:")
    LOGGER.info(f"  Total: {params['total']:,}")
    LOGGER.info(f"  Trainable: {params['trainable']:,}")
    if config.use_medical_modules:
        LOGGER.info(f"  Medical Modules: {params['medical_modules']:,}")
        medical_percentage = (params['medical_modules'] / params['total']) * 100
        LOGGER.info(f"  Medical Module Percentage: {medical_percentage:.2f}%")
    LOGGER.info("=" * 80)
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        LOGGER.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Setup training components
    loss_fn = ComputeLoss(model)
    
    # Optimizer
    if config.optimizer_type == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    elif config.optimizer_type == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_type == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer_type}")
    
    # Learning rate scheduler
    if config.cos_lr:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.num_epochs,
            min_lr=config.learning_rate * 0.01,
        )
    else:
        scheduler = None
    
    # EMA
    ema = ModelEMA(model, decay=config.ema_decay) if config.use_ema else None
    
    # Mixed precision
    scaler = amp.GradScaler() if config.mixed_precision else None
    
    # Training loop
    LOGGER.info("=" * 80)
    LOGGER.info("Starting Training")
    LOGGER.info("=" * 80)
    
    best_val_loss = float('inf')
    training_history = []
    start_time = time.time()
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            config=config,
            scaler=scaler,
            ema=ema,
        )
        
        # Validate
        eval_model = ema.ema if ema is not None else model
        val_metrics = validate(
            model=eval_model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            config=config,
        )
        
        # Update learning rate
        if scheduler is not None:
            current_lr = scheduler.step(epoch)
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # Combine metrics
        epoch_metrics = {
            'epoch': epoch,
            'lr': current_lr,
            **train_metrics,
            **val_metrics,
        }
        
        training_history.append(epoch_metrics)
        
        # Log epoch summary
        LOGGER.info("=" * 80)
        LOGGER.info(f"Epoch {epoch}/{config.num_epochs} Summary:")
        LOGGER.info(f"  LR: {current_lr:.6f}")
        LOGGER.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        LOGGER.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        LOGGER.info("=" * 80)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'config': config.as_dict(),
            'metrics': epoch_metrics,
        }
        
        if ema is not None:
            checkpoint['ema'] = ema.ema.state_dict()
        
        # Save last checkpoint
        torch.save(checkpoint, weights_dir / "last.pt")
        
        # Save best checkpoint
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save(checkpoint, weights_dir / "best.pt")
            LOGGER.info(f"✓ Best model saved (val_loss: {best_val_loss:.4f})")
        
        # Save periodic checkpoints
        if epoch % 10 == 0:
            torch.save(checkpoint, weights_dir / f"epoch_{epoch}.pt")
    
    # Training complete
    elapsed_time = time.time() - start_time
    
    LOGGER.info("=" * 80)
    LOGGER.info("Training Complete!")
    LOGGER.info(f"Total time: {elapsed_time / 3600:.2f} hours")
    LOGGER.info(f"Best validation loss: {best_val_loss:.4f}")
    LOGGER.info(f"Model saved to: {weights_dir}")
    LOGGER.info("=" * 80)
    
    # Save training history
    history_file = save_dir / "training_history.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    
    # Save final summary
    summary = {
        'config': config.as_dict(),
        'training_time_hours': elapsed_time / 3600,
        'best_val_loss': best_val_loss,
        'final_epoch': config.num_epochs,
        'model_path': str(weights_dir / "best.pt"),
        'log_file': log_file,
        'history_file': str(history_file),
        'model_parameters': params,
    }
    
    summary_file = save_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    return summary


# ==================== CLI ====================
def main():
    parser = argparse.ArgumentParser(
        description="YOLOv7 Training for Medical CT Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Train with medical modules (default)
          python train_yolov7_medical.py --data_dir ./datasets/ct_data --epochs 120
          
          # Train baseline (without medical modules)
          python train_yolov7_medical.py --data_dir ./datasets/ct_data --epochs 120 \\
              --use_medical_modules 0 --model_config models/yolov7_baseline.yaml
          
          # Advanced training with custom preprocessing
          python train_yolov7_medical.py --data_dir ./datasets/ct_data --epochs 150 \\
              --batch_size 32 --imgsz 800 --window_center -600 --window_width 1500 \\
              --use_ema --mixed_precision --multi_scale
        """)
    )
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--train_split", type=str, default="train", help="Train split name")
    parser.add_argument("--val_split", type=str, default="", help="Validation split name")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--include_negative", action="store_true", default=True)
    parser.add_argument("--max_negative", type=int, default=20)
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    
    # Model parameters
    parser.add_argument("--model_config", type=str, default="models/yolov7_medical.yaml")
    parser.add_argument("--use_medical_modules", type=int, default=1, help="Use medical modules (1/0)")
    parser.add_argument("--pretrained", type=str, default="", help="Pretrained weights path")
    
    # Optimizer
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam", "AdamW"])
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--cos_lr", type=int, default=1)
    
    # Medical preprocessing
    parser.add_argument("--enable_hu_windowing", type=int, default=1)
    parser.add_argument("--window_center", type=float, default=-600.0)
    parser.add_argument("--window_width", type=float, default=1500.0)
    parser.add_argument("--enable_clahe", type=int, default=1)
    parser.add_argument("--clahe_clip_limit", type=float, default=2.0)
    parser.add_argument("--clahe_tile_grid", type=int, default=8)
    
    # Advanced
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--gradient_clip", type=float, default=10.0)
    parser.add_argument("--mixed_precision", action="store_true", default=True)
    parser.add_argument("--multi_scale", action="store_true", default=True)
    
    # Output
    parser.add_argument("--save_dir", type=str, default="./yolov7_models")
    parser.add_argument("--log_dir", type=str, default="./yolov7_logs")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="", help="Device (empty=auto)")
    
    args = parser.parse_args()
    
    # Build config
    config = TrainingConfig(
        data_dir=args.data_dir,
        train_split=args.train_split,
        val_split=args.val_split if args.val_split else None,
        val_ratio=args.val_ratio,
        include_negative_samples=args.include_negative,
        max_negative_per_patient=args.max_negative,
        
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        imgsz=args.imgsz,
        
        model_config=args.model_config,
        use_medical_modules=bool(args.use_medical_modules),
        pretrained_weights=args.pretrained if args.pretrained else None,
        
        optimizer_type=args.optimizer,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        warmup_epochs=args.warmup_epochs,
        cos_lr=bool(args.cos_lr),
        
        enable_hu_windowing=bool(args.enable_hu_windowing),
        window_center=args.window_center,
        window_width=args.window_width,
        enable_clahe=bool(args.enable_clahe),
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid=args.clahe_tile_grid,
        
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        gradient_clip=args.gradient_clip,
        mixed_precision=args.mixed_precision,
        multi_scale=args.multi_scale,
        
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        
        random_seed=args.seed,
        num_workers=args.workers,
        device=args.device,
    )
    
    # Print banner
    print("\n" + "=" * 80)
    print("YOLOv7 Medical CT Tumor Detection Training")
    print("=" * 80)
    print(f"Medical Modules: {'ENABLED' if config.use_medical_modules else 'DISABLED'}")
    print(f"Image Size: {config.imgsz}x{config.imgsz}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"HU Windowing: {config.enable_hu_windowing} (WC={config.window_center}, WW={config.window_width})")
    print(f"CLAHE: {config.enable_clahe}")
    print("=" * 80 + "\n")
    
    # Train
    summary = train_yolov7(config)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Training Time: {summary['training_time_hours']:.2f} hours")
    print(f"Best Validation Loss: {summary['best_val_loss']:.4f}")
    print(f"Model: {summary['model_path']}")
    print(f"Summary: {summary.get('history_file', 'N/A')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
