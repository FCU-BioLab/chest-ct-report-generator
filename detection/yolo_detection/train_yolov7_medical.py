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
    # With medical modules (default config includes medical modules)
    python train_yolov7_medical.py --data_dir ./datasets/ct_data --epochs 120 --use_medical_modules
    
    # Without medical modules
    python train_yolov7_medical.py --data_dir ./datasets/ct_data --epochs 120 --no_medical_modules
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

# Try to import evaluation visualizer
try:
    from yolov7_eval_visualizer import YOLOv7EvalVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False
    LOGGER = logging.getLogger(__name__)
    LOGGER.warning("YOLOv7EvalVisualizer not available, evaluation visualization disabled")

LOGGER = logging.getLogger(__name__)


# ==================== Configuration ====================
@dataclass
class TrainingConfig:
    """Training configuration for YOLOv7 with Enhanced Features"""
    
    # Data parameters
    data_dir: str
    train_split: str = "train"
    val_split: Optional[str] = None
    val_ratio: float = 0.2
    include_negative_samples: bool = True
    max_negative_per_patient: int = 0
    
    # Training parameters
    num_epochs: int = 120
    batch_size: int = 8  # Enhanced: increased from 4 to 8
    accumulation_steps: int = 4  # Enhanced: gradient accumulation for effective batch=32
    learning_rate: float = 0.001
    imgsz: int = 640
    
    # Model parameters
    model_config: str = "models/yolov7_medical.yaml"
    use_medical_modules: bool = True
    pretrained_weights: Optional[str] = None
    
    # Optimizer parameters
    optimizer_type: str = "AdamW"  # Enhanced: AdamW for better convergence
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
    
    # Enhanced: Augmentation parameters
    enable_augmentation: bool = True
    mosaic_prob: float = 0.5
    mixup_prob: float = 0.3
    copy_paste_prob: float = 0.3
    cache_images: bool = False
    
    # Enhanced: Positive sample oversampling
    positive_oversample: bool = True
    positive_ratio: float = 0.7  # 70% positive samples
    
    # Enhanced: Loss configuration
    cls_loss_gain: float = 1.8  # Enhanced for small lesions
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Advanced training
    use_ema: bool = True
    ema_decay: float = 0.9999
    gradient_clip: float = 10.0
    mixed_precision: bool = True
    multi_scale: bool = True
    
    # Enhanced: Evaluation & Visualization
    eval_interval: int = 1
    save_interval: int = 5
    visualize_predictions: bool = True  # Per-epoch visualization
    vis_conf_threshold: float = 0.001  # Low threshold for diagnosis
    vis_nms_iou: float = 0.45
    num_vis_samples: int = 20
    
    # Output directories
    save_dir: str = "./yolov7_logs/runs"
    log_dir: str = "./yolov7_logs/logs"
    
    # Misc
    random_seed: int = 42
    num_workers: int = 4  # Enhanced: increased from 0 to 4
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
    
    # Training dataloader with enhanced features
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
        augment=config.enable_augmentation,  # Enhanced
        mosaic_prob=config.mosaic_prob,  # Enhanced
        mixup_prob=config.mixup_prob,  # Enhanced
        copy_paste_prob=config.copy_paste_prob,  # Enhanced
        cache_images=config.cache_images,  # Enhanced
        positive_oversample=config.positive_oversample,  # Enhanced
        positive_ratio=config.positive_ratio,  # Enhanced
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
        positive_oversample=False,  # No oversampling for validation
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
    
    # Enhanced: Gradient accumulation - zero grad at start
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
    
    for batch_idx, (images, targets, metadata) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        if config.mixed_precision and scaler is not None:
            with amp.autocast():
                outputs = model(images)
                loss, loss_items = loss_fn(outputs, targets)
                loss = loss / config.accumulation_steps  # Enhanced: Scale loss for accumulation
        else:
            outputs = model(images)
            loss, loss_items = loss_fn(outputs, targets)
            loss = loss / config.accumulation_steps  # Enhanced: Scale loss for accumulation
        
        # Backward pass
        if config.mixed_precision and scaler is not None:
            scaler.scale(loss).backward()
            
            # Enhanced: Gradient accumulation - only update every N steps
            if (batch_idx + 1) % config.accumulation_steps == 0:
                # Gradient clipping
                if config.gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            
            # Enhanced: Gradient accumulation - only update every N steps
            if (batch_idx + 1) % config.accumulation_steps == 0:
                # Gradient clipping
                if config.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                
                optimizer.step()
                optimizer.zero_grad()
        
        # Update EMA (only after optimizer update)
        if ema is not None and (batch_idx + 1) % config.accumulation_steps == 0:
            ema.update(model)
        
        # Update metrics (scaled back to original scale)
        total_loss += loss.item() * config.accumulation_steps
        box_loss += loss_items[0].item()
        obj_loss += loss_items[1].item()
        cls_loss += loss_items[2].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * config.accumulation_steps:.4f}',
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


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    Non-Maximum Suppression (NMS) for YOLOv7 predictions
    
    Args:
        prediction: Model output (bs, num_boxes, 6) [x, y, w, h, conf, class]
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        max_det: Maximum detections per image
    
    Returns:
        List of detections per image [(n, 6) for x, y, w, h, conf, class]
    """
    # Convert to list if not already
    if not isinstance(prediction, list):
        prediction = [prediction]
    
    bs = prediction[0].shape[0]
    nc = prediction[0].shape[2] - 5  # number of classes
    xc = prediction[0][..., 4] > conf_thres  # candidates
    
    # Settings
    max_wh = 7680  # maximum box width and height
    max_nms = 30000  # maximum number of boxes for NMS
    
    output = [torch.zeros((0, 6), device=prediction[0].device)] * bs
    
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xi]  # Get predictions for image xi
        x = x[xc[xi]]  # confidence filtering
        
        if not x.shape[0]:
            continue
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :4].clone()
        box[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        box[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        box[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        box[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        
        # Detections matrix (n, 6) [x1, y1, x2, y2, conf, class]
        if nc == 1:
            conf = x[:, 4:5]
            j = torch.zeros((x.shape[0], 1), dtype=torch.long, device=x.device)  # 2D tensor
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            conf *= x[:, 4:5]  # multiply by objectness
        
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        # Check shape
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        
        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        
        output[xi] = x[i]
    
    return output


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: ComputeLoss,
    device: torch.device,
    config: TrainingConfig,
) -> Dict[str, float]:
    """
    Validate model with detection metrics
    
    Args:
        model: YOLOv7 model
        val_loader: Validation dataloader
        loss_fn: Loss function
        device: Device
        config: Training configuration
    
    Returns:
        Dictionary of metrics (loss, mAP, precision, recall)
    """
    # Keep model in training mode for loss calculation
    was_training = model.training
    model.train()
    
    total_loss = 0.0
    box_loss = 0.0
    obj_loss = 0.0
    cls_loss = 0.0
    
    # For detection metrics
    stats = []
    seen = 0
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # IoU vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    pbar = tqdm(val_loader, desc="Validation")
    
    for images, targets, metadata in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        nb, _, height, width = images.shape
        
        # Forward pass for loss
        outputs = model(images)
        loss, loss_items = loss_fn(outputs, targets)
        
        # Update loss metrics
        total_loss += loss.item()
        box_loss += loss_items[0].item()
        obj_loss += loss_items[1].item()
        cls_loss += loss_items[2].item()
        
        # Switch to eval mode for inference
        model.eval()
        
        # Get predictions (inference mode)
        with torch.no_grad():
            preds = model(images)
            
        # Process predictions
        if isinstance(preds, tuple):
            preds = preds[0]  # Get inference output
        
        # Apply NMS
        preds = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.6)
        
        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            seen += 1
            
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), 
                                torch.Tensor(), torch.Tensor(), 
                                torch.tensor(tcls)))
                continue
            
            # Predictions
            predn = pred.clone()
            
            # Evaluate
            if nl:
                tbox = labels[:, 1:5].clone()  # target boxes
                tbox[:, 0] *= width  # x center
                tbox[:, 1] *= height  # y center
                tbox[:, 2] *= width  # width
                tbox[:, 3] *= height  # height
                
                # Convert to x1y1x2y2
                tbox[:, 0] -= tbox[:, 2] / 2
                tbox[:, 1] -= tbox[:, 3] / 2
                tbox[:, 2] = tbox[:, 0] + tbox[:, 2]
                tbox[:, 3] = tbox[:, 1] + tbox[:, 3]
                
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            
            # Move tensors to CPU before appending to stats
            # Convert tcls to tensor
            tcls_tensor = torch.tensor(tcls) if isinstance(tcls, list) else tcls
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls_tensor.cpu()))
        
        # Restore training mode
        model.train()
        
        pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    
    # Restore original training state
    if not was_training:
        model.eval()
    
    n_batches = len(val_loader)
    
    # Compute detection metrics
    metrics = {}
    if len(stats):
        # Stats are already on CPU, just concatenate and convert to numpy
        stats = [torch.cat(x, 0).numpy() for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=None, names={0: 'tumor'})
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            
            metrics.update({
                'precision': mp,
                'recall': mr,
                'mAP@0.5': map50,
                'mAP@0.5:0.95': map,
                'f1': f1.mean(),
            })
    
    metrics.update({
        'val_loss': total_loss / n_batches,
        'val_box_loss': box_loss / n_batches,
        'val_obj_loss': obj_loss / n_batches,
        'val_cls_loss': cls_loss / n_batches,
    })
    
    return metrics


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix
    
    Args:
        detections: (n, 6) x1, y1, x2, y2, conf, class
        labels: (m, 5) class, x1, y1, x2, y2
        iouv: IoU vector for mAP@0.5:0.95
    
    Returns:
        correct: (n, 10) for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def box_iou(box1, box2):
    """
    Calculate IoU between two sets of boxes
    
    Args:
        box1: (n, 4) x1, y1, x2, y2
        box2: (m, 4) x1, y1, x2, y2
    
    Returns:
        iou: (n, m)
    """
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - 
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    
    return inter / (area1[:, None] + area2 - inter)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    """
    Compute Average Precision per class
    
    Args:
        tp: True positives (n, 10)
        conf: Confidence scores (n,)
        pred_cls: Predicted classes (n,)
        target_cls: Target classes
        plot: Plot PR curves
        save_dir: Save directory
        names: Class names
        eps: Small value
    
    Returns:
        tp, fp, p, r, f1, ap, unique_classes
    """
    # Sort by confidence
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]
    
    # Create Precision-Recall curve
    px, py = np.linspace(0, 1, 1000), []
    
    # Per class AP, precision, recall
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros(nc), np.zeros(nc)
    
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions
        
        if n_p == 0 or n_l == 0:
            continue
        
        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        
        # Recall
        recall = tpc / (n_l + eps)
        r[ci] = recall[-1, 0]  # recall at last threshold
        
        # Precision
        precision = tpc / (tpc + fpc)
        p[ci] = precision[-1, 0]  # precision at last threshold
        
        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap_single(recall[:, j], precision[:, j])
    
    # Compute F1
    f1 = 2 * p * r / (p + r + eps)
    
    return tp, fpc, p, r, f1, ap, unique_classes.astype('int32')


def compute_ap_single(recall, precision):
    """Compute AP for single IoU threshold"""
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([1.], precision, [0.]))
    
    # Compute precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    
    # Calculate area under PR curve
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    
    return ap, mpre, mrec


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
    
    # Enhanced: Initialize visualizer if enabled and available
    visualizer = None
    if config.visualize_predictions and VISUALIZER_AVAILABLE:
        vis_dir = save_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        visualizer = YOLOv7EvalVisualizer(
            output_dir=str(vis_dir),
            conf_threshold=config.vis_conf_threshold,
            nms_iou=config.vis_nms_iou,
            max_samples=config.num_vis_samples,
        )
        LOGGER.info(f"✓ Visualizer initialized: {vis_dir}")
    elif config.visualize_predictions and not VISUALIZER_AVAILABLE:
        LOGGER.warning("⚠ Visualizer requested but not available (missing dependencies)")
    
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
    
    # Enhanced: Setup training components with focal loss support
    loss_fn = ComputeLoss(
        model,
        cls_loss_gain=config.cls_loss_gain,
        use_focal_loss=config.use_focal_loss,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
    )
    
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
        
        # Enhanced: Visualize predictions after validation
        if visualizer is not None and epoch % max(1, config.num_epochs // 10) == 0:
            LOGGER.info(f"Generating visualizations for epoch {epoch}...")
            try:
                visualizer.visualize_epoch(
                    model=eval_model,
                    dataloader=val_loader,
                    device=device,
                    epoch=epoch,
                )
                LOGGER.info("✓ Visualizations saved")
            except Exception as e:
                LOGGER.warning(f"⚠ Visualization failed: {e}")
        
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
        
        # Log detection metrics if available
        if 'mAP@0.5' in val_metrics:
            LOGGER.info(f"  Precision: {val_metrics['precision']:.4f}")
            LOGGER.info(f"  Recall: {val_metrics['recall']:.4f}")
            LOGGER.info(f"  mAP@0.5: {val_metrics['mAP@0.5']:.4f}")
            LOGGER.info(f"  mAP@0.5:0.95: {val_metrics['mAP@0.5:0.95']:.4f}")
            LOGGER.info(f"  F1: {val_metrics['f1']:.4f}")
        
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
          # Train with medical modules
          python train_yolov7_medical.py --data_dir ./datasets/ct_data --epochs 120 --use_medical_modules
          
          # Train baseline (without medical modules)
          python train_yolov7_medical.py --data_dir ./datasets/ct_data --epochs 120 \\
              --no_medical_modules --model_config models/yolov7_baseline.yaml
          
          # Advanced training with custom preprocessing
          python train_yolov7_medical.py --data_dir ./datasets/ct_data --epochs 150 \\
              --batch_size 32 --imgsz 800 --window_center -600 --window_width 1500 \\
              --use_medical_modules --use_ema --mixed_precision --multi_scale
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
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    
    # Enhanced: Augmentation parameters
    parser.add_argument("--enable_augmentation", action="store_true", default=False,
                       help="Enable advanced augmentations (Mosaic, MixUp, Copy-Paste)")
    parser.add_argument("--mosaic_prob", type=float, default=0.0, help="Mosaic augmentation probability")
    parser.add_argument("--mixup_prob", type=float, default=0.0, help="MixUp augmentation probability")
    parser.add_argument("--copy_paste_prob", type=float, default=0.0, help="Copy-Paste augmentation probability")
    parser.add_argument("--cache_images", action="store_true", default=False, help="Cache images in memory")
    
    # Enhanced: Positive oversampling
    parser.add_argument("--positive_oversample", action="store_true", default=False,
                       help="Enable positive sample oversampling")
    parser.add_argument("--positive_ratio", type=float, default=0.7, 
                       help="Target positive sample ratio (0.0-1.0)")
    
    # Enhanced: Loss function parameters
    parser.add_argument("--cls_loss_gain", type=float, default=1.0, help="Classification loss gain multiplier")
    parser.add_argument("--use_focal_loss", action="store_true", default=False, help="Use Focal Loss for classification")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Focal Loss alpha parameter")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal Loss gamma parameter")
    
    # Enhanced: Visualization parameters
    parser.add_argument("--visualize_predictions", action="store_true", default=False,
                       help="Generate per-epoch prediction visualizations")
    parser.add_argument("--vis_conf_threshold", type=float, default=0.001, 
                       help="Confidence threshold for visualization")
    parser.add_argument("--vis_nms_iou", type=float, default=0.45, help="NMS IoU threshold for visualization")
    parser.add_argument("--num_vis_samples", type=int, default=20, help="Number of samples to visualize per epoch")
    
    # Model parameters
    parser.add_argument("--model_config", type=str, default="models/yolov7_medical.yaml")
    parser.add_argument("--use_medical_modules", action="store_true", default=False, 
                       help="Enable medical modules (use flag to enable, default: False)")
    parser.add_argument("--no_medical_modules", action="store_true", 
                       help="Explicitly disable medical modules")
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
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.lr,
        imgsz=args.imgsz,
        
        # Enhanced: Augmentation parameters
        enable_augmentation=args.enable_augmentation,
        mosaic_prob=args.mosaic_prob,
        mixup_prob=args.mixup_prob,
        copy_paste_prob=args.copy_paste_prob,
        cache_images=args.cache_images,
        
        # Enhanced: Positive oversampling
        positive_oversample=args.positive_oversample,
        positive_ratio=args.positive_ratio,
        
        # Enhanced: Loss parameters
        cls_loss_gain=args.cls_loss_gain,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        
        # Enhanced: Visualization parameters
        visualize_predictions=args.visualize_predictions,
        vis_conf_threshold=args.vis_conf_threshold,
        vis_nms_iou=args.vis_nms_iou,
        num_vis_samples=args.num_vis_samples,
        
        model_config=args.model_config,
        use_medical_modules=args.use_medical_modules if not args.no_medical_modules else False,
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
    print(f"Batch Size: {config.batch_size} (Effective: {config.batch_size * config.accumulation_steps})")
    print(f"Gradient Accumulation: {config.accumulation_steps} steps")
    print(f"HU Windowing: {config.enable_hu_windowing} (WC={config.window_center}, WW={config.window_width})")
    print(f"CLAHE: {config.enable_clahe}")
    print("")
    print("Enhanced Features:")
    print(f"  Augmentation: {config.enable_augmentation} (Mosaic={config.mosaic_prob:.2f}, MixUp={config.mixup_prob:.2f}, Copy-Paste={config.copy_paste_prob:.2f})")
    print(f"  Positive Oversample: {config.positive_oversample} (Target Ratio={config.positive_ratio:.2f})")
    print(f"  Focal Loss: {config.use_focal_loss} (α={config.focal_alpha:.2f}, γ={config.focal_gamma:.2f})")
    print(f"  Classification Gain: {config.cls_loss_gain:.2f}")
    print(f"  Visualization: {config.visualize_predictions} (Conf={config.vis_conf_threshold:.3f})")
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
