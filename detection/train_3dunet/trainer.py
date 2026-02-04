#!/usr/bin/env python3
"""
3D U-Net Video Trainer
======================

Trainer logic for 3D U-Net.
"""

import logging
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from typing import Dict, Tuple, Optional, List

from .dataset import VolumetricDataset, collate_video_batch
from .model import get_model
from .config import Config
from .config import Config
from .detector import NoduleDetector, DetectedNodule, GroundTruthNodule
from dataclasses import asdict

logger = logging.getLogger(__name__)

# ============ Postprocess Functions ============

def generate_lung_mask(volume: np.ndarray, threshold: float = 0.45, dilate_mm: float = 5.0) -> np.ndarray:
    """
    從 CT volume 生成肺部遮罩
    
    Args:
        volume: (D, H, W) normalized to [0, 1]
        threshold: 肺部閾值 (肺部較暗，低於此值視為肺部)
        dilate_mm: 擴張半徑 (mm)，確保覆蓋邊緣結節
        
    Returns:
        lung_mask: (D, H, W) binary mask
    """
    # 肺部通常比較暗（在 CT 中 HU 較低）
    # 提高閾值避免漏分割
    binary = (volume < threshold).astype(np.uint8)
    
    # 形態學操作：閉運算填補小孔
    struct = morphology.ball(3) if binary.ndim == 3 else morphology.disk(3)
    binary = morphology.binary_closing(binary, struct)
    
    # 填充 2D 孔洞（逐切片）
    for z in range(binary.shape[0]):
        binary[z] = ndimage.binary_fill_holes(binary[z])
    
    # 移除小的連通區域，保留最大的兩個（左右肺）
    labels = measure.label(binary)
    if labels.max() > 0:
        regions = measure.regionprops(labels)
        # 按面積排序，保留最大的兩個
        regions = sorted(regions, key=lambda x: x.area, reverse=True)
        lung_mask = np.zeros_like(binary)
        for region in regions[:2]:  # 最多保留兩個（左右肺）
            if region.area > 500:  # 降低閾值，避免遺漏
                lung_mask[labels == region.label] = 1
    else:
        lung_mask = binary
    
    # 擴張肺遮罩，確保邊緣結節不被過濾
    if dilate_mm > 0:
        # 假設 1 voxel ≈ 1mm（可根據實際 spacing 調整）
        dilate_radius = max(2, int(dilate_mm))
        struct_dilate = morphology.ball(dilate_radius) if lung_mask.ndim == 3 else morphology.disk(dilate_radius)
        lung_mask = morphology.binary_dilation(lung_mask, struct_dilate)
    
    return lung_mask.astype(np.uint8)


def postprocess_prediction(
    prob_mask: np.ndarray,
    lung_mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    min_size_voxels: int = 10,
    apply_closing: bool = True
) -> np.ndarray:
    """
    後處理預測結果
    
    Args:
        prob_mask: (D, H, W) 機率圖 [0, 1]
        lung_mask: (D, H, W) 肺部遮罩（可選）
        threshold: 二值化閾值
        min_size_voxels: 最小連通區域體素數
        apply_closing: 是否應用閉運算
        
    Returns:
        binary_mask: 後處理後的二值遮罩
    """
    # Step 1: Threshold
    binary = (prob_mask > threshold).astype(np.uint8)
    
    # Step 2: Apply lung mask
    if lung_mask is not None:
        binary = binary * (lung_mask > 0).astype(np.uint8)
    
    # Step 3: Connected component filtering
    if binary.sum() > 0:
        labels = measure.label(binary)
        filtered = np.zeros_like(binary)
        for region in measure.regionprops(labels):
            if region.area >= min_size_voxels:
                filtered[labels == region.label] = 1
        binary = filtered
    
    # Step 4: Morphological closing
    if apply_closing and binary.sum() > 0:
        struct = morphology.ball(1) if binary.ndim == 3 else morphology.disk(1)
        binary = morphology.binary_closing(binary, struct).astype(np.uint8)
    
    return binary.astype(np.uint8)


def calc_detection_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    iou_threshold: float = 0.1
) -> Dict[str, float]:
    """
    計算結節檢測指標 (基於連通區域)
    
    Args:
        pred: (D, H, W) 預測的二值遮罩
        gt: (D, H, W) Ground truth 二值遮罩
        iou_threshold: 判定為 TP 的 IoU 閾值
        
    Returns:
        metrics: Dict with TP, FP, FN, Precision, Recall, F1
    """
    # 標記連通區域
    pred_labels = measure.label(pred)
    gt_labels = measure.label(gt)
    
    pred_regions = measure.regionprops(pred_labels)
    gt_regions = measure.regionprops(gt_labels)
    
    n_pred = len(pred_regions)
    n_gt = len(gt_regions)
    
    # 特殊情況處理
    if n_gt == 0 and n_pred == 0:
        return {'TP': 0, 'FP': 0, 'FN': 0, 'Precision': 1.0, 'Recall': 1.0, 'F1': 1.0}
    if n_gt == 0:
        return {'TP': 0, 'FP': n_pred, 'FN': 0, 'Precision': 0.0, 'Recall': 1.0, 'F1': 0.0}
    if n_pred == 0:
        return {'TP': 0, 'FP': 0, 'FN': n_gt, 'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}
    
    # 計算每個 GT 區域是否被偵測到
    gt_matched = [False] * n_gt
    pred_matched = [False] * n_pred
    
    for i, gt_region in enumerate(gt_regions):
        gt_mask = (gt_labels == gt_region.label)
        best_iou = 0
        best_pred_idx = -1
        
        for j, pred_region in enumerate(pred_regions):
            if pred_matched[j]:
                continue
            pred_mask = (pred_labels == pred_region.label)
            
            # 計算 IoU
            inter = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            iou = inter / (union + 1e-6)
            
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = j
        
        if best_iou >= iou_threshold and best_pred_idx >= 0:
            gt_matched[i] = True
            pred_matched[best_pred_idx] = True
    
    TP = sum(gt_matched)
    FN = n_gt - TP
    FP = sum([not m for m in pred_matched])
    
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }


def calc_dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """計算 Dice Score"""
    inter = np.logical_and(pred > 0, gt > 0).sum()
    union = (pred > 0).sum() + (gt > 0).sum()
    return (2.0 * inter) / (union + 1e-6)


def calc_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculate IoU (Jaccard Index)"""
    inter = np.logical_and(pred > 0, gt > 0).sum()
    union = np.logical_or(pred > 0, gt > 0).sum()
    return inter / (union + 1e-6)


def calc_segmentation_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive segmentation metrics.
    
    Returns:
        Dict with: dice, iou, precision, recall
    """
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    
    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, np.logical_not(gt_bin)).sum()
    fn = np.logical_and(np.logical_not(pred_bin), gt_bin).sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall)
    }

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):  # 降低 smooth，讓稀少正樣本的梯度更強
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss - 可調整 FP/FN 權重的 Dice 變體
    
    當 alpha=0.5, beta=0.5 時等同於 Dice Loss
    當 alpha < beta 時，更注重減少 FN（提高 Recall）
    當 alpha > beta 時，更注重減少 FP（提高 Precision）
    
    推薦：alpha=0.3, beta=0.7 對於小目標分割（減少漏檢）
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        super().__init__()
        self.alpha = alpha  # FP 權重
        self.beta = beta    # FN 權重
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - 結合 Focal 機制的 Tversky Loss
    對困難樣本給予更高權重
    
    gamma > 1: 更加專注於困難樣本
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.5, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky


class BoundaryLoss(nn.Module):
    """
    Boundary Loss - 專注於分割邊界區域
    
    使用距離變換計算邊界，對邊界區域的誤差給予更高權重
    適合小目標和邊界精確度要求高的場景
    """
    def __init__(self, boundary_weight=1.0, use_dist_map=True):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.use_dist_map = use_dist_map
    
    def compute_boundary_weight_map(self, target):
        """
        Compute boundary weight map using distance transform
        """
        # Move to CPU for scipy operations
        target_np = target.detach().cpu().numpy()
        batch_size = target_np.shape[0]
        weight_maps = []
        
        for b in range(batch_size):
            mask = target_np[b, 0]  # (D, H, W)
            
            if mask.sum() < 1:
                # No foreground, return uniform weight
                weight_map = np.ones_like(mask)
            else:
                # Compute distance to boundary
                from scipy import ndimage as ndi
                
                # Distance from foreground to boundary
                dist_fg = ndi.distance_transform_edt(mask)
                # Distance from background to boundary  
                dist_bg = ndi.distance_transform_edt(1 - mask)
                
                # Combine - closer to boundary = higher weight
                # Use exponential decay from boundary
                dist_to_boundary = np.minimum(dist_fg, dist_bg)
                weight_map = np.exp(-dist_to_boundary / 3.0)  # decay factor
                
                # Normalize
                weight_map = (weight_map - weight_map.min()) / (weight_map.max() - weight_map.min() + 1e-6)
                weight_map = weight_map + 0.5  # baseline weight of 0.5
            
            weight_maps.append(weight_map)
        
        weight_tensor = torch.tensor(np.stack(weight_maps)[:, np.newaxis], 
                                     dtype=target.dtype, device=target.device)
        return weight_tensor
    
    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        
        if self.use_dist_map:
            # Compute boundary-aware weight map
            weight_map = self.compute_boundary_weight_map(target)
            
            # Weighted BCE
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            weighted_bce = (bce * weight_map).mean()
            
            return self.boundary_weight * weighted_bce
        else:
            # Simple Laplacian edge detection
            # Apply 3D Laplacian filter to get edges
            laplacian_kernel = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                             [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                             [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], 
                                            dtype=pred.dtype, device=pred.device)
            laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3, 3)
            
            # Get edges of target
            target_edges = F.conv3d(target, laplacian_kernel, padding=1).abs()
            target_edges = (target_edges > 0.1).float()
            
            # MSE on boundary regions
            boundary_loss = F.mse_loss(pred_prob * target_edges, target * target_edges)
            
            return self.boundary_weight * boundary_loss


class CombinedLoss(nn.Module):
    """
    Combined Loss for nodule segmentation:
    - Tversky Loss: Handle class imbalance, reduce FN
    - Boundary Loss: Improve boundary accuracy
    - BCE: Voxel-level learning signal
    """
    def __init__(self, tversky_weight=1.0, boundary_weight=0.5, bce_weight=0.5,
                 tversky_alpha=0.3, tversky_beta=0.7, pos_weight=100.0):
        super().__init__()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.boundary = BoundaryLoss(boundary_weight=1.0, use_dist_map=True)
        # Store pos_weight as a buffer so it moves with the module
        self.register_buffer('pos_weight', torch.tensor([pos_weight]))
        
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        loss_tversky = self.tversky(pred, target)
        loss_boundary = self.boundary(pred, target)
        
        # BCE with pos_weight on correct device
        loss_bce = F.binary_cross_entropy_with_logits(
            pred, target, 
            pos_weight=self.pos_weight.to(pred.device)
        )
        
        total = (self.tversky_weight * loss_tversky + 
                 self.boundary_weight * loss_boundary +
                 self.bce_weight * loss_bce)
        
        return total

def calc_detection_rate(logits_batch, masks_batch, threshold=0.1):
    """
    Calculate Nodule Detection Rate (Sensitivity)
    Hit if IoU > threshold
    """
    batch_size = logits_batch.shape[0]
    hits = 0
    
    pred_batch = (torch.sigmoid(logits_batch) > 0.5).float()
    
    for i in range(batch_size):
        pred = pred_batch[i]
        mask = masks_batch[i]
        
        inter = (pred * mask).sum()
        union = pred.sum() + mask.sum() - inter
        
        if union > 0:
            iou = inter / union
            if iou > threshold:
                hits += 1
        elif mask.sum() == 0:
            # GT is empty
            if pred.sum() == 0:
                hits += 1 # Correctly predicted empty
        else:
             pass 

    return hits

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2.0, reduction='mean'):  # alpha 提高到 0.95 增強正樣本權重
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        
        # Calculate alpha weighting
        # alpha_t = alpha where target=1, (1-alpha) where target=0
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class UNet3DTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        
        # Model
        self.model = get_model(config).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Losses - based on config.training.loss_type
        loss_type = getattr(config.training, 'loss_type', 'combined')
        logger.info(f"📉 Initializing loss: {loss_type}")
        
        if loss_type == 'combined':
            # Combined Loss: Tversky + Boundary + BCE
            self.combined_loss = CombinedLoss(
                tversky_weight=getattr(config.training, 'tversky_weight', 1.0),
                boundary_weight=getattr(config.training, 'boundary_weight', 0.5),
                bce_weight=getattr(config.training, 'bce_weight', 0.5),
                tversky_alpha=getattr(config.training, 'tversky_alpha', 0.3),
                tversky_beta=getattr(config.training, 'tversky_beta', 0.7),
                pos_weight=getattr(config.training, 'pos_weight', 100.0)
            )
            self.use_combined_loss = True
        elif loss_type == 'tversky':
            # Tversky Loss only
            self.tversky = TverskyLoss(
                alpha=getattr(config.training, 'tversky_alpha', 0.3),
                beta=getattr(config.training, 'tversky_beta', 0.7)
            )
            self.use_combined_loss = False
        else:
            # Legacy losses (dice + bce + focal)
            self.bce_weighted = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([getattr(config.training, 'pos_weight', 150.0)]).to(self.device)
            )
            self.focal = FocalLoss(alpha=0.90, gamma=2.0)
            self.dice = DiceLoss(smooth=1e-5)
            self.use_combined_loss = False
        
        # Detector
        # Detector
        self.detector = NoduleDetector(
            threshold=getattr(config.postprocessing, 'det_threshold', 0.5), 
            min_size_mm3=getattr(config.postprocessing, 'det_min_size', 30.0)
        )
        
        # Datasets
        self.train_loader = self._create_loader("train", shuffle=True)
        self.val_loader = self._create_loader("val", shuffle=False)
        
        # Scheduler: 進一步降低 max_lr，穩定訓練
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.training.learning_rate * 3,  # 從 5x 降到 3x (max_lr = 3e-4)
            epochs=config.training.epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,  # 更快達到峰值，更多時間下降
            div_factor=3,   # 初始 lr = max_lr / 3
            final_div_factor=30
        )
        
        # Add File Handler to Logger
        file_handler = logging.FileHandler(self.output_dir / "training.log", encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(file_handler)
        logger.info(f"📝 Logging to file: {self.output_dir / 'training.log'}")
        
        self.best_val_score = 0.0
        
        # Store datasets for info export
        self.train_dataset = self.train_loader.dataset
        self.val_dataset = self.val_loader.dataset
        
    def _create_loader(self, split: str, shuffle: bool):
        dataset = VolumetricDataset(
            npz_dir=self.config.data.npz_dir,
            split=split,
            image_size=self.config.model.image_size,
            max_depth=self.config.data.max_depth,
            augmentation=(split == 'train'),
            positive_ratio=getattr(self.config.data, 'positive_ratio', 0.7)
        )
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_video_batch,
            pin_memory=True,
            persistent_workers=(self.config.num_workers > 0),
            prefetch_factor=2 if self.config.num_workers > 0 else None
        )

    def export_dataset_info(self):
        """Export dataset information to Dataset.json"""
        logger.info("📊 Exporting dataset information...")
        
        dataset_info = {
            'export_time': datetime.now().isoformat(),
            'npz_dir': str(self.config.data.npz_dir),
            'splits': {}
        }
        
        # Collect info from train and val datasets
        for split_name, dataset in [('train', self.train_dataset), ('val', self.val_dataset)]:
            info = dataset.get_dataset_info()
            dataset_info['splits'][split_name] = info
            logger.info(f"  {split_name}: {info['total_samples']} samples, {info['unique_patients']} patients")
        
        # Try to load test dataset info
        try:
            test_dataset = VolumetricDataset(
                npz_dir=self.config.data.npz_dir,
                split='test',
                image_size=self.config.model.image_size,
                max_depth=self.config.data.max_depth,
                augmentation=False
            )
            info = test_dataset.get_dataset_info()
            dataset_info['splits']['test'] = info
            logger.info(f"  test: {info['total_samples']} samples, {info['unique_patients']} patients")
        except Exception as e:
            logger.warning(f"  Could not load test dataset: {e}")
        
        # Export to JSON
        json_path = self.output_dir / "Dataset.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Dataset info exported to {json_path}")
        return dataset_info

    def train(self):
        logger.info(f"🚀 Starting training for {self.config.training.epochs} epochs")
        
        # Export dataset info at start
        self.export_dataset_info()
        
        # Enhanced history tracking
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'val_precision': [],
            'val_recall': [],
            'val_det_rate': [],
            'val_det_precision': [],
            'val_det_recall': [],
            'val_det_f1': []
        }
        best_dice = 0.0
        
        for epoch in range(1, self.config.training.epochs + 1):
            train_loss = self.train_epoch(epoch)
            # Validation with comprehensive metrics
            val_metrics = self.validate(epoch)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_dice'].append(val_metrics['dice'])
            history['val_iou'].append(val_metrics['iou'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            history['val_det_rate'].append(val_metrics['det_rate'])
            history['val_det_precision'].append(val_metrics['det_precision'])
            history['val_det_recall'].append(val_metrics['det_recall'])
            history['val_det_f1'].append(val_metrics['det_f1'])
            
            # Log epoch summary
            logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_metrics['loss']:.4f}, Val Dice={val_metrics['dice']:.4f}")
            
            # Plot all metrics
            self.plot_metrics(history)
            
            # Save Best
            val_dice = val_metrics['dice']
            if val_dice > best_dice:
                best_dice = val_dice
                logger.info(f"🆕 New best model saved (Dice={best_dice:.4f})")
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pth")
            
            # Save periodic
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, val_dice)
                
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Train Ep {epoch}")
        
        for batch in pbar:
            images = batch['image'].to(self.device).float()
            masks = batch['mask'].to(self.device).float()
            
            self.optimizer.zero_grad()
            
            logits = self.model(images)
            if isinstance(logits, list):
                # Deep supervision
                loss = 0
                weights = [1.0, 0.5, 0.25, 0.125]
                for i, logit in enumerate(logits[:len(weights)]):
                    if logit.shape != masks.shape:
                         target = F.interpolate(masks, size=logit.shape[2:], mode='nearest')
                    else:
                        target = masks
                    
                    if hasattr(self, 'use_combined_loss') and self.use_combined_loss:
                        loss += weights[i] * self.combined_loss(logit, target)
                    elif hasattr(self, 'tversky'):
                        loss += weights[i] * self.tversky(logit, target)
                    else:
                        l_focal = self.focal(logit, target)
                        l_dice = self.dice(logit, target)
                        loss += weights[i] * (l_focal + 2.0 * l_dice)
            else:
                # Single output
                if hasattr(self, 'use_combined_loss') and self.use_combined_loss:
                    # Use Combined Loss (Tversky + Boundary + BCE)
                    loss = self.combined_loss(logits, masks)
                elif hasattr(self, 'tversky'):
                    # Use Tversky Loss only
                    loss = self.tversky(logits, masks)
                else:
                    # Legacy: BCE + Focal + Dice
                    loss_bce = self.bce_weighted(logits, masks)
                    loss_focal = self.focal(logits, masks)
                    loss_dice = self.dice(logits, masks)
                    loss = loss_bce + 0.5 * loss_focal + 2.0 * loss_dice
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'lr': self.scheduler.get_last_lr()[0]})
            
        return total_loss / len(self.train_loader)

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model and compute comprehensive metrics.
        
        Returns:
            Dict with: loss, dice, iou, precision, recall, det_rate, det_precision, det_recall, det_f1
        """
        self.model.eval()
        
        # Accumulators for segmentation metrics
        total_loss = 0.0
        seg_metrics_sum = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Accumulators for detection metrics
        total_tp, total_fp, total_fn = 0, 0, 0
        total_hits = 0
        total_nodules = 0
        
        count = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Val Ep {epoch}"):
                images = batch['image'].to(self.device).float()
                masks = batch['mask'].to(self.device).float()
                
                logits = self.model(images)
                if isinstance(logits, list):
                    logits = logits[0]
                
                # Calculate validation loss based on loss type
                if hasattr(self, 'use_combined_loss') and self.use_combined_loss:
                    loss = self.combined_loss(logits, masks)
                elif hasattr(self, 'tversky'):
                    loss = self.tversky(logits, masks)
                else:
                    loss_bce = self.bce_weighted(logits, masks)
                    loss_focal = self.focal(logits, masks)
                    loss_dice = self.dice(logits, masks)
                    loss = loss_bce + 0.5 * loss_focal + 2.0 * loss_dice
                total_loss += loss.item()
                
                # Get predictions
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()
                
                # Debug logging for first batch
                if count == 0:
                    logger.info(f"🔍 DEBUG [Val Ep {epoch}]")
                    logger.info(f"  Img: min={images.min():.4f}, max={images.max():.4f}, mean={images.mean():.4f}")
                    logger.info(f"  Msk: sum={masks.sum()}, unique={torch.unique(masks)}")
                    logger.info(f"  Logits: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
                    logger.info(f"  Pred: sum={pred.sum()}, unique={torch.unique(pred)}")
                    
                    # Save debug image
                    self._save_debug_image(images, masks, logits, epoch)
                
                # Calculate per-sample metrics
                batch_size = images.shape[0]
                for i in range(batch_size):
                    pred_np = pred[i, 0].cpu().numpy()
                    mask_np = masks[i, 0].cpu().numpy()
                    
                    # Segmentation metrics
                    seg_m = calc_segmentation_metrics(pred_np, mask_np)
                    for k in seg_metrics_sum:
                        seg_metrics_sum[k] += seg_m[k]
                    
                    # Detection metrics (connected component analysis)
                    det_m = calc_detection_metrics(pred_np, mask_np, iou_threshold=0.1)
                    total_tp += det_m['TP']
                    total_fp += det_m['FP']
                    total_fn += det_m['FN']
                
                # Detection rate (simple IoU-based)
                batch_hits = calc_detection_rate(logits, masks)
                total_hits += batch_hits
                total_nodules += (masks.amax(dim=(1,2,3,4)).sum().item())
                
                count += batch_size
        
        # Compute averages
        n_batches = len(self.val_loader)
        n_samples = count
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        avg_seg = {k: v / n_samples for k, v in seg_metrics_sum.items()} if n_samples > 0 else seg_metrics_sum
        
        # Detection aggregates
        det_precision = total_tp / (total_tp + total_fp + 1e-6)
        det_recall = total_tp / (total_tp + total_fn + 1e-6)
        det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall + 1e-6)
        det_rate = total_hits / (total_nodules + 1e-6) if total_nodules > 0 else 0.0
        
        # Log comprehensive metrics
        logger.info(f"📊 Val Metrics [Ep {epoch}]:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Segmentation: Dice={avg_seg['dice']:.4f}, IoU={avg_seg['iou']:.4f}, Precision={avg_seg['precision']:.4f}, Recall={avg_seg['recall']:.4f}")
        logger.info(f"  Detection: Rate={det_rate*100:.2f}%, Precision={det_precision:.4f}, Recall={det_recall:.4f}, F1={det_f1:.4f}")
        logger.info(f"  Detection Counts: TP={total_tp}, FP={total_fp}, FN={total_fn}")
        
        return {
            'loss': avg_loss,
            'dice': avg_seg['dice'],
            'iou': avg_seg['iou'],
            'precision': avg_seg['precision'],
            'recall': avg_seg['recall'],
            'det_rate': det_rate,
            'det_precision': det_precision,
            'det_recall': det_recall,
            'det_f1': det_f1,
            'det_tp': total_tp,
            'det_fp': total_fp,
            'det_fn': total_fn
        }

    def _match_nodules(self, 
                      predictions: List[DetectedNodule], 
                      gt_nodules: List[GroundTruthNodule], 
                      iou_threshold: float = 0.1) -> Dict:
        """
        Match predictions to GT nodules and assign status (TP/FP/FN).
        Returns counts and updated lists.
        """
        # Reset status
        for p in predictions: p.match_status = "FP" # Default to FP
        for g in gt_nodules: g.match_status = "Missed" # Default to Missed
        
        matches = 0
        
        # Greedy matching: Sort predictions by confidence (prob)
        predictions.sort(key=lambda x: x.probability, reverse=True)
        
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            
            p_bbox = pred.geometry.bbox
            p_vol = pred.geometry.volume_mm3
            
            # Find best overlapping GT that isn't matched yet?
            # Or allow multiple preds to match same GT? 
            # Standard: 1-to-1 matching for counting.
            
            for i, gt in enumerate(gt_nodules):
                # Quick bbox check
                g_bbox = gt.geometry.bbox
                if (p_bbox[0] > g_bbox[3] or p_bbox[3] < g_bbox[0] or
                    p_bbox[1] > g_bbox[4] or p_bbox[4] < g_bbox[1] or
                    p_bbox[2] > g_bbox[5] or p_bbox[5] < g_bbox[2]):
                    continue
                    
                # Calculate IoU
                # We need masks to calc IoU accurately. 
                # But here we only have objects.
                # Approx IoU using bbox? No, inaccurate.
                # We should have done matching on masks in `calc_detection_metrics`.
                # But we want to link objects.
                # Let's use bbox IoU as approximation if masks not available?
                # OR, since we have slice indices and ranges, we can try to compute intersection if simple shapes.
                # BUT, `detector.py` doesn't keep the masks of objects.
                
                # REVISIT: We should do matching when we have masks available, OR accept BBox IoU.
                # Given we are in trainer.py loop where we have masks `pred_post` and `mask_np`.
                # We can match `DetectedNodule` to `GroundTruthNodule` by finding which label they correspond to!
                
                # DetectedNodule has `id` which is the label in `pred_post`.
                # GroundTruthNodule has `id` which is the label in `mask_np`.
                # We can compute IoU between label `pred.id` and label `gt.id` in the masks!
                pass
            pass
            
        # WAIT: I can't easily do IoU here without the masks. 
        # I should put this logic INSIDE the loop in comprehensive_test where I have `pred_post` and `mask_np`.
        # So I will NOT add a complex `_match_nodules` that re-calculates IoU.
        # Instead I will do matching in `comprehensive_test` using the masks.
        
        return {}
    
    def _save_debug_image(self, images, masks, logits, epoch):
        """Save debug visualization image"""
        try:
            import matplotlib.pyplot as plt
            
            # Take first sample
            img_t = images[0, 0].cpu().numpy()
            msk_t = masks[0, 0].cpu().numpy()
            prob_t = torch.sigmoid(logits[0, 0]).cpu().numpy()
            pred_t = (prob_t > 0.5).astype(np.uint8)
            
            # Find slice with largest mask, or center
            if msk_t.sum() > 0:
                z_idx = int(np.argmax(msk_t.sum(axis=(1, 2))))
            else:
                z_idx = msk_t.shape[0] // 2
            
            plt.figure(figsize=(20, 5))
            plt.subplot(1, 5, 1); plt.imshow(img_t[z_idx], cmap='gray'); plt.title(f'Image (z={z_idx})')
            plt.subplot(1, 5, 2); plt.imshow(msk_t[z_idx], cmap='gray'); plt.title('GT Mask')
            plt.subplot(1, 5, 3); plt.imshow(prob_t[z_idx], cmap='jet', vmin=0, vmax=1); plt.title('Prob Map')
            plt.subplot(1, 5, 4); plt.imshow(pred_t[z_idx], cmap='gray'); plt.title('Prediction')
            plt.subplot(1, 5, 5); plt.hist(prob_t.flatten(), bins=50); plt.title('Prob Dist')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"debug_ep{epoch:03d}.png")
            plt.close()
            logger.info(f"  🖼️ Saved debug image to {self.output_dir / f'debug_ep{epoch:03d}.png'}")
        except Exception as e:
            logger.warning(f"  ❌ Failed to save debug image: {e}")
            
    def plot_metrics(self, history: Dict):
        """
        Plot comprehensive training metrics and save multiple visualization PNGs.
        
        Args:
            history: Dict with keys for all tracked metrics
        """
        try:
            import matplotlib.pyplot as plt
            epochs = range(1, len(history['train_loss']) + 1)
            
            # ============ Main Metrics Plot (2x3 grid) ============
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            # Row 1: Loss metrics
            # Train Loss
            axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
            axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Epochs'); axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Validation Loss
            axes[0, 1].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
            axes[0, 1].set_title('Validation Loss', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Epochs'); axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # Train vs Val Loss
            axes[0, 2].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train')
            axes[0, 2].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val')
            axes[0, 2].set_title('Train vs Val Loss', fontsize=12, fontweight='bold')
            axes[0, 2].set_xlabel('Epochs'); axes[0, 2].set_ylabel('Loss')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].legend()
            
            # Row 2: Segmentation and Detection metrics
            # Dice & IoU
            axes[1, 0].plot(epochs, history['val_dice'], 'g-', linewidth=2, label='Dice')
            axes[1, 0].plot(epochs, history['val_iou'], 'c-', linewidth=2, label='IoU')
            axes[1, 0].set_title('Segmentation: Dice & IoU', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Epochs'); axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            
            # Precision & Recall (Segmentation)
            axes[1, 1].plot(epochs, history['val_precision'], 'm-', linewidth=2, label='Precision')
            axes[1, 1].plot(epochs, history['val_recall'], 'y-', linewidth=2, label='Recall')
            axes[1, 1].set_title('Segmentation: Precision & Recall', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epochs'); axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
            
            # Detection Rate
            det_rates_pct = [d * 100 for d in history['val_det_rate']]
            axes[1, 2].plot(epochs, det_rates_pct, 'r-', linewidth=2, label='Detection Rate')
            axes[1, 2].set_title('Nodule Detection Rate', fontsize=12, fontweight='bold')
            axes[1, 2].set_xlabel('Epochs'); axes[1, 2].set_ylabel('Rate (%)')
            axes[1, 2].set_ylim(0, 100)
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "metrics.png", dpi=150)
            plt.close()
            
            # ============ Segmentation Metrics Plot ============
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            axes[0].plot(epochs, history['val_dice'], 'g-', linewidth=2)
            axes[0].set_title('Dice Score (DSC)', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Epochs'); axes[0].set_ylabel('Score')
            axes[0].set_ylim(0, 1); axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(epochs, history['val_iou'], 'c-', linewidth=2)
            axes[1].set_title('IoU (Jaccard)', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Epochs'); axes[1].set_ylabel('Score')
            axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(epochs, history['val_precision'], 'm-', linewidth=2)
            axes[2].set_title('Precision', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Epochs'); axes[2].set_ylabel('Score')
            axes[2].set_ylim(0, 1); axes[2].grid(True, alpha=0.3)
            
            axes[3].plot(epochs, history['val_recall'], 'y-', linewidth=2)
            axes[3].set_title('Recall', fontsize=12, fontweight='bold')
            axes[3].set_xlabel('Epochs'); axes[3].set_ylabel('Score')
            axes[3].set_ylim(0, 1); axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "segmentation_metrics.png", dpi=150)
            plt.close()
            
            # ============ Detection Metrics Plot ============
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            axes[0].plot(epochs, det_rates_pct, 'r-', linewidth=2)
            axes[0].set_title('Detection Rate', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Epochs'); axes[0].set_ylabel('Rate (%)')
            axes[0].set_ylim(0, 100); axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(epochs, history['val_det_precision'], 'b-', linewidth=2)
            axes[1].set_title('Detection Precision', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Epochs'); axes[1].set_ylabel('Score')
            axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(epochs, history['val_det_recall'], 'orange', linewidth=2)
            axes[2].set_title('Detection Recall', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Epochs'); axes[2].set_ylabel('Score')
            axes[2].set_ylim(0, 1); axes[2].grid(True, alpha=0.3)
            
            axes[3].plot(epochs, history['val_det_f1'], 'purple', linewidth=2)
            axes[3].set_title('Detection F1 Score', fontsize=12, fontweight='bold')
            axes[3].set_xlabel('Epochs'); axes[3].set_ylabel('Score')
            axes[3].set_ylim(0, 1); axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "detection_metrics.png", dpi=150)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to plot metrics: {e}")

    def save_checkpoint(self, epoch: int, score: float, is_best: bool = False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'score': score,
            'config': self.config.save(str(self.ckpt_dir / "config.json")) # save config is weird here, returns None
        }
        
        filename = f"checkpoint_ep{epoch}.pt"
        torch.save(state, self.ckpt_dir / filename)
        
        if is_best:
            torch.save(state, self.ckpt_dir / "best_model.pt")
            logger.info(f"🆕 New best model saved (Dice={score:.4f})")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint - handles both full checkpoint dict and raw state_dict"""
        logger.info(f"📥 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle both checkpoint formats:
        # 1) Full checkpoint dict with 'model_state_dict' key (from save_checkpoint)
        # 2) Raw state_dict saved directly (from best_model.pth in train())
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"✅ Checkpoint loaded (Epoch {checkpoint.get('epoch', '?')}, Score {checkpoint.get('score', 0.0):.4f})")
        else:
            # Raw state_dict (like best_model.pth)
            self.model.load_state_dict(checkpoint)
            logger.info(f"✅ Model state loaded from {checkpoint_path}")

    def evaluate(self, split: str = 'test', use_postprocess: bool = True):
        """
        Evaluate on specific split with postprocessing and detection metrics
        
        Args:
            split: 'test', 'val', or 'train'
            use_postprocess: Whether to apply lung mask and postprocessing
        """
        logger.info(f"📊 Evaluating on {split} set (postprocess={use_postprocess})...")
        loader = self._create_loader(split, shuffle=False)
        self.model.eval()
        
        # Accumulators
        all_dice_raw = []
        all_dice_post = []
        total_tp, total_fp, total_fn = 0, 0, 0
        sample_results = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Eval {split}"):
                images = batch['image'].to(self.device).float()
                masks = batch['mask'].to(self.device).float()
                
                logits = self.model(images)
                if isinstance(logits, list):
                    logits = logits[0]
                
                probs = torch.sigmoid(logits)
                
                # Process each sample in batch
                batch_size = images.shape[0]
                for i in range(batch_size):
                    # Get numpy arrays
                    img_np = images[i, 0].cpu().numpy()  # (D, H, W)
                    mask_np = masks[i, 0].cpu().numpy()  # (D, H, W)
                    prob_np = probs[i, 0].cpu().numpy()  # (D, H, W)
                    
                    # Raw prediction (no postprocess)
                    pred_raw = (prob_np > 0.5).astype(np.uint8)
                    dice_raw = calc_dice_score(pred_raw, mask_np)
                    all_dice_raw.append(dice_raw)
                    
                    if use_postprocess:
                        # 停用 Lung Mask（容易漏分割），只使用連通區域過濾
                        # lung_mask = generate_lung_mask(img_np, threshold=0.45)
                        
                        # Apply postprocessing (without lung mask)
                        pred_post = postprocess_prediction(
                            prob_np,
                            lung_mask=None,  # 停用 lung mask
                            threshold=0.5,
                            min_size_voxels=5,  # 降低最小體素數
                            apply_closing=True
                        )
                        
                        dice_post = calc_dice_score(pred_post, mask_np)
                        all_dice_post.append(dice_post)
                        
                        # Calculate detection metrics
                        metrics = calc_detection_metrics(pred_post, mask_np, iou_threshold=0.1)
                    else:
                        pred_post = pred_raw
                        all_dice_post.append(dice_raw)
                        metrics = calc_detection_metrics(pred_raw, mask_np, iou_threshold=0.1)
                    
                    total_tp += metrics['TP']
                    total_fp += metrics['FP']
                    total_fn += metrics['FN']
                    
                    sample_results.append({
                        'npz_path': batch['npz_path'][i] if 'npz_path' in batch else f'sample_{len(sample_results)}',
                        'dice_raw': dice_raw,
                        'dice_post': all_dice_post[-1],
                        'TP': metrics['TP'],
                        'FP': metrics['FP'],
                        'FN': metrics['FN']
                    })
        
        # Compute overall metrics
        avg_dice_raw = np.mean(all_dice_raw) if all_dice_raw else 0.0
        avg_dice_post = np.mean(all_dice_post) if all_dice_post else 0.0
        
        overall_precision = total_tp / (total_tp + total_fp + 1e-6)
        overall_recall = total_tp / (total_tp + total_fn + 1e-6)
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-6)
        
        # Log results
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 {split.upper()} SET EVALUATION RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"📈 Segmentation Metrics:")
        logger.info(f"   • Dice (Raw):         {avg_dice_raw:.4f}")
        logger.info(f"   • Dice (Postprocess): {avg_dice_post:.4f}")
        logger.info(f"\n🎯 Detection Metrics (IoU≥0.1):")
        logger.info(f"   • True Positives:  {total_tp}")
        logger.info(f"   • False Positives: {total_fp}")
        logger.info(f"   • False Negatives: {total_fn}")
        logger.info(f"   • Precision:       {overall_precision:.4f}")
        logger.info(f"   • Recall:          {overall_recall:.4f}")
        logger.info(f"   • F1 Score:        {overall_f1:.4f}")
        logger.info(f"{'='*60}\n")
        
        # Return comprehensive results
        return {
            'dice_raw': avg_dice_raw,
            'dice_post': avg_dice_post,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'sample_results': sample_results
        }

    def comprehensive_test(self, split: str = 'test', save_visualizations: bool = True, export_gif: bool = True,
                         det_min_size: Optional[float] = None, det_threshold: Optional[float] = None, 
                         no_postprocess: bool = False):
        """
        Run comprehensive testing with detailed metrics and visualization.
        
        Args:
            split: Dataset split to test
            save_visualizations: Whether to save visualization files
            export_gif: Whether to save GIF animations
            det_min_size: Min nodule size (mm3) override
            det_threshold: Detection probability threshold override
            no_postprocess: Disable lung mask and postprocessing
        
        Returns:
            Dict with all metrics and paths to saved files
        """
        import matplotlib.pyplot as plt
        
        logger.info(f"🔬 Comprehensive Test on {split} set...")
        
        # Setup output directory
        output_dir = self.output_dir / f"test_{split}"
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        loader = self._create_loader(split, shuffle=False)
        self.model.eval()
        
        # Accumulators
        sample_results = []
        seg_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': []}
        det_totals = {'TP': 0, 'FP': 0, 'FN': 0}
        
        sample_idx = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Testing {split}"):
                images = batch['image'].to(self.device).float()
                masks = batch['mask'].to(self.device).float()
                
                if images.shape[2] > 64:
                    logits = self._sliding_window_inference(images, window_size=64, overlap=32)
                else:
                    logits = self.model(images)
                    if isinstance(logits, list):
                        logits = logits[0]
                
                probs = torch.sigmoid(logits)
                
                batch_size = images.shape[0]
                for i in range(batch_size):
                    # Get numpy arrays
                    img_np = images[i, 0].cpu().numpy()
                    mask_np = masks[i, 0].cpu().numpy()
                    prob_np = probs[i, 0].cpu().numpy()
                    
                    # Predictions
                    # Calculate min size in voxels
                    spacing = batch['spacing'][i].cpu().numpy()
                    # Use provided args or config defaults
                    min_vol_mm3 = det_min_size if det_min_size is not None else getattr(self.config.postprocessing, 'det_min_size', 30.0)
                    thresh = det_threshold if det_threshold is not None else getattr(self.config.postprocessing, 'det_threshold', 0.5)
                    do_closing = getattr(self.config.postprocessing, 'apply_closing', True)

                    voxel_vol = np.prod(spacing)
                    min_voxels = int(min_vol_mm3 / voxel_vol) if voxel_vol > 0 else 5
                    
                    # Generate lung mask
                    lung_mask = None
                    if not no_postprocess:
                        lung_mask = generate_lung_mask(img_np, dilate_mm=10.0)

                    pred_raw = (prob_np > thresh).astype(np.uint8)
                    pred_post = postprocess_prediction(
                        prob_np, lung_mask=lung_mask, threshold=thresh,
                        min_size_voxels=min_voxels, apply_closing=do_closing
                    )
                    
                    # Run Nodule Detector
                    # Prepare metadata (need to unsquelch batch dims or handle batch index)
                    # batch has keys like 'spacing', 'origin' if collate worked
                    batch_metadata = {
                        'slice_indices': batch.get('slice_indices'),
                        'spacing': batch.get('spacing'),
                        'origin': batch.get('origin'),
                        'original_shape': batch.get('original_shape')
                    }
                    
                    # process_batch expects batch logits (B, ...)
                    # We are in a loop over B, but passing single items? 
                    # No, process_batch takes full batch logits. 
                    # But here we are iterating i in batch_size.
                    # It's more efficient to run detector on full batch before loop.
                    # However, to minimize code change, I can pass a mini-batch of size 1 or rewrite the loop.
                    # Let's run detector for the specific sample i using slice on logits.
                    
                    # Or better: Run classifier ONCE for the batch outside data loop, then iterate.
                    # Wait, logits is computed for batch at line 1198.
                    # Let's run detector on the whole batch at once.
                    
                    # (See code change BELOW loop: wait, I need to do this inside loop because I need per-sample results)
                    
                    # Actually, let's call detector process_batch ONCE before the loop.
                    # But existing code structure loops i.
                    # I will insert the call before the loop and use index i.
                    
                    # See ReplacementContent logic below...
                    
                    # Segmentation metrics
                    seg_m = calc_segmentation_metrics(pred_post, mask_np)
                    for k in seg_metrics:
                        seg_metrics[k].append(seg_m[k])
                    
                    # Detection metrics
                    det_m = calc_detection_metrics(pred_post, mask_np, iou_threshold=0.1)
                    det_totals['TP'] += det_m['TP']
                    det_totals['FP'] += det_m['FP']
                    det_totals['FN'] += det_m['FN']
                    
                    # Sample name
                    npz_name = Path(batch['npz_path'][i]).stem if 'npz_path' in batch else f'sample_{sample_idx}'
                    
                    # Create per-case folder
                    case_dir = viz_dir / f'{sample_idx:03d}_{npz_name}'
                    case_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Get detections for this sample
                    # We need to run detector on the single sample or full batch.
                    # Let's run on single sample to fit this structure easily, 
                    # constructing a 1-batch metadata dict.
                    sample_meta = {
                        'slice_indices': batch['slice_indices'][i:i+1] if 'slice_indices' in batch else None,
                        'spacing': batch['spacing'][i:i+1] if 'spacing' in batch else None,
                        'origin': batch['origin'][i:i+1] if 'origin' in batch else None,
                        'original_shape': batch['original_shape'][i:i+1] if 'original_shape' in batch else None,
                    }
                    sample_logits = logits[i:i+1] # (1, 1, D, H, W)
                    
                    sample_detections = self.detector.process_batch(sample_logits, sample_meta)[0]
                    detected_nodules_list = [asdict(d) for d in sample_detections]

                    # Per-case statistics
                    case_stats = {
                        'idx': sample_idx,
                        'name': npz_name,
                        'npz_path': batch['npz_path'][i] if 'npz_path' in batch else None,
                        'segmentation': {
                            'dice': seg_m['dice'],
                            'iou': seg_m['iou'],
                            'precision': seg_m['precision'],
                            'recall': seg_m['recall'],
                            'gt_voxels': int(mask_np.sum()),
                            'pred_voxels': int(pred_post.sum())
                        },
                        'detection': {
                            'TP': det_m['TP'],
                            'FP': det_m['FP'],
                            'FN': det_m['FN'],
                            'precision': det_m['Precision'],
                            'recall': det_m['Recall'],
                            'f1': det_m['F1']
                        },
                        'detected_nodules': detected_nodules_list
                    }
                    
                    # Save per-case JSON
                    with open(case_dir / 'stats.json', 'w', encoding='utf-8') as f:
                        json.dump(case_stats, f, indent=2, ensure_ascii=False)
                        
                    # ---------------------------------------------------------
                    # MATCHING LOGIC FOR DETECTED NODULES (TP/FP/FN)
                    # ---------------------------------------------------------
                    # Get GT Nodules
                    gt_nodules_list = self.detector.analyze_gt(masks[i:i+1], sample_meta)[0]
                    
                    # Match using masks
                    # pred_post is the prediction mask (labels correspond to sample_detections IDs?)
                    # Wait, detector.process_batch re-runs measure.label.
                    # So IDs in sample_detections match the labels in the INTERNAL pred_vol of detector.
                    # They might NOT match `pred_post` if I re-ran labeling there.
                    # `detector.process_batch` does its own labeling.
                    
                    # To be safe: We should map `DetectedNodule` back to the mask I have here?
                    # Or trust that I can reconstruct the matching.
                    
                    # Better: Re-calculate matching here using the list of objects and their properties?
                    # Or just rely on BBox IoU for reporting mapping?
                    # Or: Map the detected nodules to the mask using their Centroid?
                    
                    # Let's use Centroid matching to identify which label in `pred_post` corresponds to `DetectedNodule`.
                    # Actually `pred_post` is binary.
                    # `detector` re-runs labeling.
                    # Let's rely on overlap check:
                    # For each predicted nodule object: create a mask for it (approx or if we had it).
                    # Since we don't have the individual object masks, we can check value at centroid? 
                    # No, centroid might be in hole.
                    
                    # Simplest robust way: 
                    # 1. We have `sample_detections` (Preds) and `gt_nodules_list` (GTs).
                    # 2. We have the full `mask_np` (GT mask).
                    # 3. We cannot easily get individual pred masks from `DetectedNodule` without reconstruction.
                    
                    # BUT `detector.process_batch` was run on `sample_logits`.
                    # It produces `DetectedNodule`s. 
                    # We want to know if each is TP or FP.
                    # We can use the `mask_np` (GT) to check overlap?
                    # For a `DetectedNodule`:
                    #   - Slice range is known.
                    #   - BBox is known.
                    #   - We can iterate over voxels in BBox? No, too slow.
                    
                    # Alternative:
                    # Pass `mask_np` (GT) to `process_batch`? No.
                    
                    # Let's compute matching based on BBox IoU (3D IoU).
                    # It's an approximation but likely sufficient for "which nodule is this".
                    # OR check if Centroid of Pred is inside ANY GT Nodule?
                    # NO, standard is IoU > 0.1.
                    
                    # Let's implement 3D BBox IoU Matching here.
                    tp_details = []
                    fp_details = []
                    
                    # Reset status
                    for p in sample_detections: p.match_status = "FP"
                    for g in gt_nodules_list: g.match_status = "Missed"
                    
                    for p in sample_detections:
                        p_bbox = p.geometry.bbox # z1, y1, x1, z2, y2, x2
                        best_iou = 0
                        best_gt = None
                        
                        p_z1, p_y1, p_x1, p_z2, p_y2, p_x2 = p_bbox
                        p_vol = (p_z2-p_z1)*(p_y2-p_y1)*(p_x2-p_x1)
                        if p_vol == 0: continue
                        
                        for g in gt_nodules_list:
                            g_bbox = g.geometry.bbox
                            g_z1, g_y1, g_x1, g_z2, g_y2, g_x2 = g_bbox
                            g_vol = (g_z2-g_z1)*(g_y2-g_y1)*(g_x2-g_x1)
                            
                            # Intersection
                            iz1 = max(p_z1, g_z1); iy1 = max(p_y1, g_y1); ix1 = max(p_x1, g_x1)
                            iz2 = min(p_z2, g_z2); iy2 = min(p_y2, g_y2); ix2 = min(p_x2, g_x2)
                            
                            if iz1 < iz2 and iy1 < iy2 and ix1 < ix2:
                                inter_vol = (iz2-iz1)*(iy2-iy1)*(ix2-ix1)
                                union_vol = p_vol + g_vol - inter_vol
                                iou = inter_vol / union_vol
                                
                                if iou > best_iou:
                                    best_iou = iou
                                    best_gt = g
                        
                        # Threshold BBox IoU (might need adjustment compared to exact mask IoU)
                        # Exact mask IoU 0.1 ~ BBox IoU 0.1 ? Roughly.
                        if best_iou > 0.05: # Loose threshold for BBox
                            p.match_status = "TP"
                            if best_gt:
                                best_gt.match_status = "Matched"
                    
                    # Console Output
                    logger.info(f"  🔍 Sample {npz_name} Analysis:")
                    
                    # TPs
                    tps = [p for p in sample_detections if p.match_status == "TP"]
                    fps = [p for p in sample_detections if p.match_status == "FP"]
                    fns = [g for g in gt_nodules_list if g.match_status == "Missed"]
                    
                    tp_strs = [f"#{p.id}(V={p.geometry.volume_mm3:.1f}, D={p.geometry.diameter_mm:.1f}mm)" for p in tps]
                    fp_strs = [f"#{p.id}(V={p.geometry.volume_mm3:.1f}, D={p.geometry.diameter_mm:.1f}mm)" for p in fps]
                    fn_strs = [f"#{g.id}(V={g.geometry.volume_mm3:.1f}, D={g.geometry.diameter_mm:.1f}mm)" for g in fns]
                    
                    if tps: logger.info(f"    ✅ TP: {', '.join(tp_strs)}")
                    if fps: logger.info(f"    ❌ FP: {', '.join(fp_strs)}")
                    if fns: logger.info(f"    ⚠️ FN: {', '.join(fn_strs)}")
                    
                    # Update case_stats with new info
                    case_stats['detected_nodules'] = [asdict(d) for d in sample_detections]
                    case_stats['missed_gt_nodules'] = [asdict(g) for g in fns]
                    
                    # Save updated JSON
                    with open(case_dir / 'stats.json', 'w', encoding='utf-8') as f:
                        json.dump(case_stats, f, indent=2, ensure_ascii=False)
                    
                    sample_results.append(case_stats)
                    
                    # Visualization
                    if save_visualizations:
                        self._save_case_visualization(
                            img_np, mask_np, prob_np, pred_raw, pred_post,
                            seg_m, det_m, npz_name, case_dir, export_gif=export_gif
                        )
                    
                    sample_idx += 1
        
        # Compute aggregated metrics
        n_samples = len(sample_results)
        avg_seg = {k: np.mean(v) for k, v in seg_metrics.items()}
        
        det_precision = det_totals['TP'] / (det_totals['TP'] + det_totals['FP'] + 1e-6)
        det_recall = det_totals['TP'] / (det_totals['TP'] + det_totals['FN'] + 1e-6)
        det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall + 1e-6)
        
        # Create summary
        summary = {
            'split': split,
            'n_samples': n_samples,
            'segmentation': {
                'dice_mean': avg_seg['dice'],
                'dice_std': np.std(seg_metrics['dice']),
                'iou_mean': avg_seg['iou'],
                'precision_mean': avg_seg['precision'],
                'recall_mean': avg_seg['recall']
            },
            'detection': {
                'TP': det_totals['TP'],
                'FP': det_totals['FP'],
                'FN': det_totals['FN'],
                'precision': det_precision,
                'recall': det_recall,
                'f1': det_f1
            },
            'sample_results': sample_results
        }
        
        # Save JSON report
        json_path = output_dir / "test_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        # Save flat detections.json (for easier parsing)
        all_detections = []
        for sample in sample_results:
            pid = sample['name']
            for nodule in sample.get('detected_nodules', []):
                nodule_copy = nodule.copy()
                nodule_copy['patient_id'] = pid
                all_detections.append(nodule_copy)
        
        with open(output_dir / "detections.json", 'w', encoding='utf-8') as f:
            json.dump(all_detections, f, indent=2, ensure_ascii=False)
            logger.info(f"📋 Saved {len(all_detections)} detections to detections.json")
        
        # Generate summary plots
        self._generate_test_summary_plots(summary, output_dir)

        # Extract FP/FN cases for analysis
        try:
            logger.info("🔍 Coping FP/FN cases for analysis...")
            analysis_dir = output_dir / "analysis_fp_fn"
            fp_dir = analysis_dir / "FP"
            fn_dir = analysis_dir / "FN"
            
            if analysis_dir.exists():
                shutil.rmtree(analysis_dir)
            fp_dir.mkdir(parents=True, exist_ok=True)
            fn_dir.mkdir(parents=True, exist_ok=True)
            
            fp_count = 0
            fn_count = 0
            
            for sample in sample_results:
                idx = sample['idx']
                name = sample['name']
                folder_name = f"{idx:03d}_{name}"
                src_path = viz_dir / folder_name
                
                if not src_path.exists():
                    continue
                    
                detection = sample.get('detection', {})
                # Copy to FP folder if has FP
                if detection.get('FP', 0) > 0:
                    shutil.copytree(src_path, fp_dir / folder_name)
                    fp_count += 1
                
                # Copy to FN folder if has FN
                if detection.get('FN', 0) > 0:
                    shutil.copytree(src_path, fn_dir / folder_name)
                    fn_count += 1
            
            logger.info(f"✅ Copied {fp_count} FP cases and {fn_count} FN cases to {analysis_dir}")
        except Exception as e:
            logger.warning(f"Failed to copy FP/FN cases: {e}")
        
        # Log results
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 COMPREHENSIVE TEST RESULTS - {split.upper()}")
        logger.info(f"{'='*70}")
        logger.info(f"📈 Segmentation Metrics (n={n_samples}):")
        logger.info(f"   • Dice:      {avg_seg['dice']:.4f} ± {np.std(seg_metrics['dice']):.4f}")
        logger.info(f"   • IoU:       {avg_seg['iou']:.4f}")
        logger.info(f"   • Precision: {avg_seg['precision']:.4f}")
        logger.info(f"   • Recall:    {avg_seg['recall']:.4f}")
        logger.info(f"\n🎯 Detection Metrics:")
        logger.info(f"   • TP={det_totals['TP']}, FP={det_totals['FP']}, FN={det_totals['FN']}")
        logger.info(f"   • Precision: {det_precision:.4f}")
        logger.info(f"   • Recall:    {det_recall:.4f}")
        logger.info(f"   • F1 Score:  {det_f1:.4f}")
        logger.info(f"\n📁 Output saved to: {output_dir}")
        logger.info(f"{'='*70}\n")
        
        return summary
    
    def _save_sample_visualization(self, img_np, mask_np, prob_np, pred_raw, pred_post,
                                   seg_m, det_m, npz_name, sample_idx, viz_dir):
        """Save per-sample visualization with segmentation and detection info"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # Find slice with most GT content
        if mask_np.sum() > 0:
            z_idx = int(np.argmax(mask_np.sum(axis=(1, 2))))
        else:
            z_idx = mask_np.shape[0] // 2
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Row 1: Image, GT, Pred, Overlay
        axes[0, 0].imshow(img_np[z_idx], cmap='gray')
        axes[0, 0].set_title(f'Image (z={z_idx})')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask_np[z_idx], cmap='Reds')
        axes[0, 1].set_title(f'GT Mask (sum={mask_np.sum():.0f})')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pred_post[z_idx], cmap='Blues')
        axes[0, 2].set_title(f'Prediction (sum={pred_post.sum():.0f})')
        axes[0, 2].axis('off')
        
        # Overlay: Red=GT, Blue=Pred, Yellow=Both
        overlay = np.stack([img_np[z_idx]] * 3, axis=-1)
        overlay = (overlay * 255).clip(0, 255).astype(np.uint8)
        overlay[mask_np[z_idx] > 0, 0] = 255  # GT = Red
        overlay[mask_np[z_idx] > 0, 1] = 0
        overlay[mask_np[z_idx] > 0, 2] = 0
        pred_only = (pred_post[z_idx] > 0) & (mask_np[z_idx] == 0)
        overlay[pred_only, 0] = 0
        overlay[pred_only, 1] = 0
        overlay[pred_only, 2] = 255  # Pred only = Blue
        both = (pred_post[z_idx] > 0) & (mask_np[z_idx] > 0)
        overlay[both, 0] = 255
        overlay[both, 1] = 255
        overlay[both, 2] = 0  # Both = Yellow
        
        axes[0, 3].imshow(overlay)
        axes[0, 3].set_title('Overlay: R=GT, B=Pred, Y=Both')
        axes[0, 3].axis('off')
        
        # Row 2: Prob map, Metrics text, Multi-slice view, Histogram
        axes[1, 0].imshow(prob_np[z_idx], cmap='jet', vmin=0, vmax=1)
        axes[1, 0].set_title('Probability Map')
        axes[1, 0].axis('off')
        
        # Metrics text
        axes[1, 1].axis('off')
        metrics_text = (
            f"SEGMENTATION METRICS\n"
            f"{'─'*25}\n"
            f"Dice:      {seg_m['dice']:.4f}\n"
            f"IoU:       {seg_m['iou']:.4f}\n"
            f"Precision: {seg_m['precision']:.4f}\n"
            f"Recall:    {seg_m['recall']:.4f}\n\n"
            f"DETECTION METRICS\n"
            f"{'─'*25}\n"
            f"TP: {det_m['TP']}  FP: {det_m['FP']}  FN: {det_m['FN']}\n"
            f"Precision: {det_m['Precision']:.4f}\n"
            f"Recall:    {det_m['Recall']:.4f}\n"
            f"F1:        {det_m['F1']:.4f}"
        )
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].set_title('Metrics')
        
        # Multi-slice (3 slices around z_idx)
        slices = [max(0, z_idx-2), z_idx, min(mask_np.shape[0]-1, z_idx+2)]
        combined = np.hstack([pred_post[s] for s in slices])
        axes[1, 2].imshow(combined, cmap='Blues')
        axes[1, 2].set_title(f'Multi-slice view (z={slices})')
        axes[1, 2].axis('off')
        
        # Probability histogram
        axes[1, 3].hist(prob_np.flatten(), bins=50, color='steelblue', alpha=0.7)
        axes[1, 3].axvline(x=0.5, color='red', linestyle='--', label='Threshold=0.5')
        axes[1, 3].set_xlabel('Probability')
        axes[1, 3].set_ylabel('Voxel Count')
        axes[1, 3].set_title('Probability Distribution')
        axes[1, 3].legend()
        
        # Overall title
        status = "[OK]" if seg_m['dice'] > 0.5 else ("[WARN]" if seg_m['dice'] > 0.2 else "[FAIL]")
        fig.suptitle(f"{status} {npz_name} | Dice={seg_m['dice']:.4f} | Detection: TP={det_m['TP']}, FP={det_m['FP']}, FN={det_m['FN']}",
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(viz_dir / f'{sample_idx:03d}_{npz_name}.png', dpi=100)
        plt.close()
    
    def _save_case_visualization(self, img_np, mask_np, prob_np, pred_raw, pred_post,
                                  seg_m, det_m, npz_name, case_dir, export_gif: bool = True):
        """
        Save per-case visualization files to individual folder
        
        Saves:
        - overview.png: Combined 8-panel visualization
        - slices/: Multi-slice images
        - overlay.png: GT vs Prediction overlay
        - animation.gif: Animated GIF of all slices (if export_gif=True)
        """
        import matplotlib.pyplot as plt
        
        # Find slice with most GT content
        if mask_np.sum() > 0:
            z_idx = int(np.argmax(mask_np.sum(axis=(1, 2))))
        else:
            z_idx = mask_np.shape[0] // 2
        
        # ====== 1. Overview Panel ======
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Row 1: Image, GT, Pred, Overlay
        axes[0, 0].imshow(img_np[z_idx], cmap='gray')
        axes[0, 0].set_title(f'Image (z={z_idx})')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask_np[z_idx], cmap='Reds')
        axes[0, 1].set_title(f'GT Mask (sum={mask_np.sum():.0f})')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pred_post[z_idx], cmap='Blues')
        axes[0, 2].set_title(f'Prediction (sum={pred_post.sum():.0f})')
        axes[0, 2].axis('off')
        
        # Overlay
        overlay = np.stack([img_np[z_idx]] * 3, axis=-1)
        overlay = (overlay * 255).clip(0, 255).astype(np.uint8)
        overlay[mask_np[z_idx] > 0, 0] = 255
        overlay[mask_np[z_idx] > 0, 1] = 0
        overlay[mask_np[z_idx] > 0, 2] = 0
        pred_only = (pred_post[z_idx] > 0) & (mask_np[z_idx] == 0)
        overlay[pred_only, 0] = 0
        overlay[pred_only, 1] = 0
        overlay[pred_only, 2] = 255
        both = (pred_post[z_idx] > 0) & (mask_np[z_idx] > 0)
        overlay[both, 0] = 255
        overlay[both, 1] = 255
        overlay[both, 2] = 0
        
        axes[0, 3].imshow(overlay)
        axes[0, 3].set_title('Overlay: R=GT, B=Pred, Y=Both')
        axes[0, 3].axis('off')
        
        # Row 2: Prob map, Metrics, Multi-slice, Histogram
        axes[1, 0].imshow(prob_np[z_idx], cmap='jet', vmin=0, vmax=1)
        axes[1, 0].set_title('Probability Map')
        axes[1, 0].axis('off')
        
        # Metrics text
        axes[1, 1].axis('off')
        metrics_text = (
            f"SEGMENTATION\n"
            f"Dice:      {seg_m['dice']:.4f}\n"
            f"IoU:       {seg_m['iou']:.4f}\n"
            f"Precision: {seg_m['precision']:.4f}\n"
            f"Recall:    {seg_m['recall']:.4f}\n\n"
            f"DETECTION\n"
            f"TP:{det_m['TP']} FP:{det_m['FP']} FN:{det_m['FN']}\n"
            f"P:{det_m['Precision']:.3f} R:{det_m['Recall']:.3f} F1:{det_m['F1']:.3f}"
        )
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].set_title('Metrics')
        
        # Multi-slice
        slices = [max(0, z_idx-2), z_idx, min(mask_np.shape[0]-1, z_idx+2)]
        combined = np.hstack([pred_post[s] for s in slices])
        axes[1, 2].imshow(combined, cmap='Blues')
        axes[1, 2].set_title(f'Multi-slice (z={slices})')
        axes[1, 2].axis('off')
        
        # Histogram
        axes[1, 3].hist(prob_np.flatten(), bins=50, color='steelblue', alpha=0.7)
        axes[1, 3].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        axes[1, 3].set_xlabel('Probability')
        axes[1, 3].set_ylabel('Count')
        axes[1, 3].set_title('Probability Distribution')
        axes[1, 3].legend()
        
        status = "[OK]" if seg_m['dice'] > 0.5 else ("[WARN]" if seg_m['dice'] > 0.2 else "[FAIL]")
        fig.suptitle(f"{status} {npz_name} | Dice={seg_m['dice']:.4f}", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(case_dir / 'overview.png', dpi=100)
        plt.close()
        
        # ====== 2. Overlay Image (high-res) ======
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.title(f'{npz_name} (z={z_idx}) - R=GT, B=Pred, Y=Both')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(case_dir / 'overlay.png', dpi=150)
        plt.close()
        
        # ====== 3. Slices folder ======
        slices_dir = case_dir / 'slices'
        slices_dir.mkdir(exist_ok=True)
        
        # Save slices around the center
        for z_offset in range(-3, 4):
            z = z_idx + z_offset
            if 0 <= z < mask_np.shape[0]:
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                
                axes[0].imshow(img_np[z], cmap='gray')
                axes[0].set_title(f'Image z={z}')
                axes[0].axis('off')
                
                axes[1].imshow(mask_np[z], cmap='Reds')
                axes[1].set_title(f'GT')
                axes[1].axis('off')
                
                axes[2].imshow(pred_post[z], cmap='Blues')
                axes[2].set_title(f'Pred')
                axes[2].axis('off')
                
                # Overlay for this slice
                slice_overlay = np.stack([img_np[z]] * 3, axis=-1)
                slice_overlay = (slice_overlay * 255).clip(0, 255).astype(np.uint8)
                slice_overlay[mask_np[z] > 0, 0] = 255
                slice_overlay[mask_np[z] > 0, 1] = 0
                slice_overlay[mask_np[z] > 0, 2] = 0
                pred_only_z = (pred_post[z] > 0) & (mask_np[z] == 0)
                slice_overlay[pred_only_z, 2] = 255
                both_z = (pred_post[z] > 0) & (mask_np[z] > 0)
                slice_overlay[both_z, 0] = 255
                slice_overlay[both_z, 1] = 255
                slice_overlay[both_z, 2] = 0
                
                axes[3].imshow(slice_overlay)
                axes[3].set_title(f'Overlay')
                axes[3].axis('off')
                
                plt.tight_layout()
                plt.savefig(slices_dir / f'slice_{z:03d}.png', dpi=80)
                plt.close()
        
        # ====== 4. Animated GIF ======
        if export_gif:
            self._save_case_gif(img_np, mask_np, pred_post, seg_m, npz_name, case_dir)
    
    def _save_case_gif(self, img_np, mask_np, pred_post, seg_m, npz_name, case_dir, fps: int = 5):
        """
        Save animated GIF showing all slices with Overlay view.
        Colors: Red=GT, Blue=Pred, Yellow=Both
        
        Args:
            img_np: (D, H, W) CT volume normalized to [0, 1]
            mask_np: (D, H, W) Ground truth mask
            pred_post: (D, H, W) Postprocessed prediction
            seg_m: Segmentation metrics dict
            npz_name: Sample name for title
            case_dir: Directory to save GIF
            fps: Frames per second
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        
        D = img_np.shape[0]
        
        # Create overlay function
        def create_overlay(z_idx):
            overlay = np.stack([img_np[z_idx]] * 3, axis=-1)
            overlay = (overlay * 255).clip(0, 255).astype(np.uint8)
            # GT = Red
            overlay[mask_np[z_idx] > 0, 0] = 255
            overlay[mask_np[z_idx] > 0, 1] = 0
            overlay[mask_np[z_idx] > 0, 2] = 0
            # Pred only = Blue
            pred_only = (pred_post[z_idx] > 0) & (mask_np[z_idx] == 0)
            overlay[pred_only, 0] = 0
            overlay[pred_only, 1] = 0
            overlay[pred_only, 2] = 255
            # Both = Yellow
            both = (pred_post[z_idx] > 0) & (mask_np[z_idx] > 0)
            overlay[both, 0] = 255
            overlay[both, 1] = 255
            overlay[both, 2] = 0
            return overlay
        
        # Create single-panel figure
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')
        
        im = ax.imshow(create_overlay(0))
        title = ax.set_title(f'{npz_name} | Dice={seg_m["dice"]:.4f}\nSlice 0/{D-1}\nR=GT, B=Pred, Y=Both', fontsize=10)
        
        def update(frame_idx):
            im.set_data(create_overlay(frame_idx))
            title.set_text(f'{npz_name} | Dice={seg_m["dice"]:.4f}\nSlice {frame_idx}/{D-1}\nR=GT, B=Pred, Y=Both')
            return [im, title]
        
        anim = FuncAnimation(fig, update, frames=D, interval=1000//fps, blit=True)
        
        gif_path = case_dir / 'animation.gif'
        anim.save(str(gif_path), writer=PillowWriter(fps=fps))
        plt.close()
    
    def _generate_test_summary_plots(self, summary, output_dir):
        """Generate summary plots for test results"""
        import matplotlib.pyplot as plt
        import json
        
        sample_results = summary['sample_results']
        
        # Extract data (handle nested structure)
        dices = [s['segmentation']['dice'] if 'segmentation' in s else s.get('dice', 0) for s in sample_results]
        ious = [s['segmentation']['iou'] if 'segmentation' in s else s.get('iou', 0) for s in sample_results]
        precisions = [s['segmentation']['precision'] if 'segmentation' in s else s.get('precision', 0) for s in sample_results]
        recalls = [s['segmentation']['recall'] if 'segmentation' in s else s.get('recall', 0) for s in sample_results]
        
        
        # Identify Detection Error Cases (FP > 0 or FN > 0)
        error_cases = []
        clean_cases = []
        
        # For recalculating detection metrics
        clean_tp_sum = 0
        clean_fp_sum = 0
        clean_fn_sum = 0
        
        # For recalculating segmentation stats (lists)
        clean_dices = []
        clean_ious = []
        clean_precisions = []
        clean_recalls = []
        
        for s in sample_results:
            # Check if detection info exists
            det = s.get('detection', {})
            fp = det.get('FP', 0)
            fn = det.get('FN', 0)
            
            if fp > 0 or fn > 0:
                error_cases.append(s)
            else:
                clean_cases.append(s)
                # Sum up TP/FP/FN for clean cases detection rate
                clean_tp_sum += det.get('TP', 0)
                clean_fp_sum += fp
                clean_fn_sum += fn
                
                # Segmentation metrics stats
                seg = s.get('segmentation', {})
                clean_dices.append(seg.get('dice', 0))
                clean_ious.append(seg.get('iou', 0))
                clean_precisions.append(seg.get('precision', 0))
                clean_recalls.append(seg.get('recall', 0))
                
        # Calculate Filtered Detection Metrics
        # Det Rate = TP / (TP + FN)
        if (clean_tp_sum + clean_fn_sum) > 0:
            filt_det_rate = clean_tp_sum / (clean_tp_sum + clean_fn_sum)
        else:
            filt_det_rate = 0.0
            
        # Det Precision = TP / (TP + FP)
        if (clean_tp_sum + clean_fp_sum) > 0:
            filt_det_precision = clean_tp_sum / (clean_tp_sum + clean_fp_sum)
        else:
            filt_det_precision = 0.0
            
        # Det Recall = TP / (TP + FN)  (Same as Det Rate, but keeping terminology)
        if (clean_tp_sum + clean_fn_sum) > 0:
            filt_det_recall = clean_tp_sum / (clean_tp_sum + clean_fn_sum)
        else:
            filt_det_recall = 0.0
            
        # Det F1
        if (filt_det_precision + filt_det_recall) > 0:
            filt_det_f1 = 2 * (filt_det_precision * filt_det_recall) / (filt_det_precision + filt_det_recall)
        else:
            filt_det_f1 = 0.0

        # Calculate Segmentation metrics excluding error cases
        filtered_seg_metrics = {
            'dice': float(np.mean(clean_dices)) if clean_dices else 0,
            'iou': float(np.mean(clean_ious)) if clean_ious else 0,
            'precision': float(np.mean(clean_precisions)) if clean_precisions else 0,
            'recall': float(np.mean(clean_recalls)) if clean_recalls else 0,
            'count': len(clean_cases)
        }
        
        filtered_det_metrics = {
            'detection_rate': float(filt_det_rate),
            'precision': float(filt_det_precision),
            'recall': float(filt_det_recall),  # Adding explicit recall
            'f1': float(filt_det_f1),
            'sum_tp': int(clean_tp_sum),
            'sum_fp': int(clean_fp_sum),
            'sum_fn': int(clean_fn_sum)
        }
        
        # Save error cases and filtered metrics to JSON
        error_cases_path = output_dir / 'detection_error_cases.json'
        with open(error_cases_path, 'w') as f:
            json.dump({
                'criteria': 'FP > 0 or FN > 0',
                'error_case_count': len(error_cases),
                'total_cases': len(sample_results),
                'clean_case_count': len(clean_cases),
                'filtered_segmentation_metrics': filtered_seg_metrics,
                'filtered_detection_metrics': filtered_det_metrics,
                'error_cases': error_cases
            }, f, indent=2)
        logger.info(f"💾 Saved {len(error_cases)} detection error cases to {error_cases_path}")
        logger.info(f"📈 Filtered (Clean) Seg Dice: {filtered_seg_metrics['dice']:.4f}")
        logger.info(f"📈 Filtered (Clean) Det Rate: {filtered_det_metrics['detection_rate']:.4f}, Det F1: {filtered_det_metrics['f1']:.4f}")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Dice distribution
        axes[0, 0].hist(dices, bins=20, color='green', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=np.mean(dices), color='red', linestyle='--', 
                          label=f'Mean={np.mean(dices):.4f}')
        axes[0, 0].set_xlabel('Dice Score')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Dice Score Distribution')
        axes[0, 0].legend()
        
        # IoU distribution
        axes[0, 1].hist(ious, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=np.mean(ious), color='red', linestyle='--',
                          label=f'Mean={np.mean(ious):.4f}')
        axes[0, 1].set_xlabel('IoU')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('IoU Distribution')
        axes[0, 1].legend()
        
        # Precision vs Recall scatter
        axes[0, 2].scatter(recalls, precisions, alpha=0.6, c=dices, cmap='RdYlGn')
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision vs Recall (color=Dice)')
        axes[0, 2].set_xlim(0, 1)
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].plot([0, 1], [1, 0], 'k--', alpha=0.3)  # iso-F1 reference
        
        # Per-sample Dice bar chart (sorted)
        # Extract dice for sorting
        samples_with_dice = [(s['name'], s['segmentation']['dice'] if 'segmentation' in s else s.get('dice', 0)) for s in sample_results]
        sorted_samples = sorted(samples_with_dice, key=lambda x: x[1], reverse=True)
        top_n = min(30, len(sorted_samples))
        names = [s[0][:15] for s in sorted_samples[:top_n]]
        dices_sorted = [s[1] for s in sorted_samples[:top_n]]
        colors = ['green' if d > 0.5 else ('orange' if d > 0.2 else 'red') for d in dices_sorted]
        axes[1, 0].barh(range(top_n), dices_sorted, color=colors)
        axes[1, 0].set_yticks(range(top_n))
        axes[1, 0].set_yticklabels(names, fontsize=8)
        axes[1, 0].set_xlabel('Dice Score')
        axes[1, 0].set_title(f'Top {top_n} Samples by Dice')
        axes[1, 0].invert_yaxis()
        
        # Detection summary pie chart
        det = summary['detection']
        axes[1, 1].pie([det['TP'], det['FP'], det['FN']], 
                       labels=[f"TP={det['TP']}", f"FP={det['FP']}", f"FN={det['FN']}"],
                       colors=['green', 'orange', 'red'], autopct='%1.1f%%')
        axes[1, 1].set_title(f"Detection Summary\nP={det['precision']:.2f}, R={det['recall']:.2f}, F1={det['f1']:.2f}")
        
        # Summary text
        axes[1, 2].axis('off')
        summary_text = (
            f"TEST SUMMARY\n"
            f"{'='*40}\n\n"
            f"Samples: {summary['n_samples']}\n\n"
            f"SEGMENTATION\n"
            f"  Dice:      {summary['segmentation']['dice_mean']:.4f} ± {summary['segmentation']['dice_std']:.4f}\n"
            f"  IoU:       {summary['segmentation']['iou_mean']:.4f}\n"
            f"  Precision: {summary['segmentation']['precision_mean']:.4f}\n"
            f"  Recall:    {summary['segmentation']['recall_mean']:.4f}\n\n"
            f"DETECTION\n"
            f"  TP: {det['TP']}  FP: {det['FP']}  FN: {det['FN']}\n"
            f"  Precision: {det['precision']:.4f}\n"
            f"  Recall:    {det['recall']:.4f}\n"
            f"  F1 Score:  {det['f1']:.4f}"
        )
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=14, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / "test_summary.png", dpi=150)
        plt.close()
        
        logger.info(f"📊 Summary plot saved to {output_dir / 'test_summary.png'}")

    def visualize_predictions(self, split: str = 'test', save_dir: Optional[str] = None):
        """
        Visualize predictions vs GT for all samples in a split.
        Saves overlay images showing: Image, GT, Pred_Raw, Pred_Post, LungMask, Overlay
        
        Args:
            split: Dataset split to visualize
            save_dir: Directory to save images (default: output_dir/visualize_{split})
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        logger.info(f"🖼️ Visualizing predictions for {split} set...")
        
        # Setup output directory
        if save_dir is None:
            save_dir = self.output_dir / f"visualize_{split}"
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        loader = self._create_loader(split, shuffle=False)
        self.model.eval()
        
        sample_idx = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Visualizing {split}"):
                images = batch['image'].to(self.device).float()
                masks = batch['mask'].to(self.device).float()
                
                logits = self.model(images)
                if isinstance(logits, list):
                    logits = logits[0]
                
                probs = torch.sigmoid(logits)
                
                batch_size = images.shape[0]
                for i in range(batch_size):
                    # Get numpy arrays
                    img_np = images[i, 0].cpu().numpy()  # (D, H, W)
                    mask_np = masks[i, 0].cpu().numpy()  # (D, H, W)
                    prob_np = probs[i, 0].cpu().numpy()  # (D, H, W)
                    
                    # Predictions
                    pred_raw = (prob_np > 0.5).astype(np.uint8)
                    
                    # Postprocessed prediction (無 lung mask)
                    pred_post = postprocess_prediction(
                        prob_np,
                        lung_mask=None,  # 停用 lung mask
                        threshold=0.5,
                        min_size_voxels=5,
                        apply_closing=True
                    )
                    
                    # Calculate metrics for this sample
                    dice_raw = calc_dice_score(pred_raw, mask_np)
                    dice_post = calc_dice_score(pred_post, mask_np)
                    
                    # Find the slice with most GT mask content
                    if mask_np.sum() > 0:
                        z_idx = int(np.argmax(mask_np.sum(axis=(1, 2))))
                    else:
                        z_idx = mask_np.shape[0] // 2
                    
                    # Create visualization figure (2 rows x 4 cols)
                    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                    
                    # Get sample name
                    npz_name = Path(batch['npz_path'][i]).stem if 'npz_path' in batch else f'sample_{sample_idx}'
                    
                    # Row 1: Basic views
                    # Image
                    axes[0, 0].imshow(img_np[z_idx], cmap='gray')
                    axes[0, 0].set_title(f'Image (z={z_idx})')
                    axes[0, 0].axis('off')
                    
                    # GT Mask
                    axes[0, 1].imshow(mask_np[z_idx], cmap='Reds')
                    axes[0, 1].set_title(f'GT Mask (sum={mask_np.sum():.0f})')
                    axes[0, 1].axis('off')
                    
                    # Pred Raw
                    axes[0, 2].imshow(pred_raw[z_idx], cmap='Blues')
                    axes[0, 2].set_title(f'Pred Raw (Dice={dice_raw:.4f})')
                    axes[0, 2].axis('off')
                    
                    # Pred Post
                    axes[0, 3].imshow(pred_post[z_idx], cmap='Greens')
                    axes[0, 3].set_title(f'Pred Post (Dice={dice_post:.4f})')
                    axes[0, 3].axis('off')
                    
                    # Row 2: Analysis views
                    # Difference: Raw - Post (顯示被過濾掉的區域)
                    diff_removed = (pred_raw > 0) & (pred_post == 0)  # 被後處理移除的
                    diff_added = (pred_post > 0) & (pred_raw == 0)    # 被後處理新增的
                    axes[1, 0].imshow(diff_removed[z_idx], cmap='Reds')
                    axes[1, 0].set_title(f'Removed by PostProc (sum={diff_removed.sum():.0f})')
                    axes[1, 0].axis('off')
                    
                    # Probability Map
                    axes[1, 1].imshow(prob_np[z_idx], cmap='jet', vmin=0, vmax=1)
                    axes[1, 1].set_title('Probability Map')
                    axes[1, 1].axis('off')
                    
                    # Overlay: Image + GT (red) + Pred_Raw (blue)
                    overlay_raw = np.stack([
                        img_np[z_idx],  # R
                        img_np[z_idx],  # G
                        img_np[z_idx]   # B
                    ], axis=-1)
                    overlay_raw = (overlay_raw * 255).clip(0, 255).astype(np.uint8)
                    # Add GT in red
                    overlay_raw[mask_np[z_idx] > 0, 0] = 255
                    overlay_raw[mask_np[z_idx] > 0, 1] = 0
                    overlay_raw[mask_np[z_idx] > 0, 2] = 0
                    # Add Pred Raw in blue (where no GT)
                    pred_only = (pred_raw[z_idx] > 0) & (mask_np[z_idx] == 0)
                    overlay_raw[pred_only, 0] = 0
                    overlay_raw[pred_only, 1] = 0
                    overlay_raw[pred_only, 2] = 255
                    # Overlap (both GT and Pred) in yellow
                    overlap = (pred_raw[z_idx] > 0) & (mask_np[z_idx] > 0)
                    overlay_raw[overlap, 0] = 255
                    overlay_raw[overlap, 1] = 255
                    overlay_raw[overlap, 2] = 0
                    
                    axes[1, 2].imshow(overlay_raw)
                    axes[1, 2].set_title('Overlay Raw: R=GT, B=Pred, Y=Both')
                    axes[1, 2].axis('off')
                    
                    # Overlay: Image + GT (red) + Pred_Post (green)
                    overlay_post = np.stack([
                        img_np[z_idx],
                        img_np[z_idx],
                        img_np[z_idx]
                    ], axis=-1)
                    overlay_post = (overlay_post * 255).clip(0, 255).astype(np.uint8)
                    overlay_post[mask_np[z_idx] > 0, 0] = 255
                    overlay_post[mask_np[z_idx] > 0, 1] = 0
                    overlay_post[mask_np[z_idx] > 0, 2] = 0
                    pred_only_post = (pred_post[z_idx] > 0) & (mask_np[z_idx] == 0)
                    overlay_post[pred_only_post, 0] = 0
                    overlay_post[pred_only_post, 1] = 255
                    overlay_post[pred_only_post, 2] = 0
                    overlap_post = (pred_post[z_idx] > 0) & (mask_np[z_idx] > 0)
                    overlay_post[overlap_post, 0] = 255
                    overlay_post[overlap_post, 1] = 255
                    overlay_post[overlap_post, 2] = 0
                    
                    axes[1, 3].imshow(overlay_post)
                    axes[1, 3].set_title('Overlay Post: R=GT, G=Pred, Y=Both')
                    axes[1, 3].axis('off')
                    
                    # Add overall title
                    diff = dice_raw - dice_post
                    status = "⚠️ POST WORSE" if diff > 0.1 else ("✅ OK" if diff < 0.1 else "")
                    fig.suptitle(f'{npz_name}\nDice: Raw={dice_raw:.4f}, Post={dice_post:.4f} (diff={diff:+.4f}) {status}', 
                                fontsize=14, fontweight='bold')
                    
                    plt.tight_layout()
                    plt.savefig(save_dir / f'{sample_idx:03d}_{npz_name}.png', dpi=100)
                    plt.close()
                    
                    sample_idx += 1
        
        logger.info(f"✅ Saved {sample_idx} visualization images to {save_dir}")
        return save_dir

    def _sliding_window_inference(self, volume: torch.Tensor, window_size: int = 64, overlap: int = 32) -> torch.Tensor:
        """
        Run inference using sliding window with Gaussian blending.
        Args:
            volume: (B, C, D, H, W) - B should be 1
            window_size: Depth window size
            overlap: Overlap size
        Returns:
            logits: (B, C, D, H, W)
        """
        B, C, D, H, W = volume.shape
        stride = window_size - overlap
        
        # Output buffer
        # Assuming output channels = 1 (logits)
        logits = torch.zeros((B, 1, D, H, W), device=volume.device)
        count_map = torch.zeros((B, 1, D, H, W), device=volume.device)
        
        # Generate gaussian weight map for blending
        # 1D gaussian along depth
        import math
        def get_gaussian(window_size, sigma_scale=1.0/8):
            tmp = torch.arange(window_size, device=volume.device).float() - (window_size - 1) / 2
            sigma = sigma_scale * (window_size - 1)
            gauss = torch.exp(-0.5 * (tmp / (sigma ** 2)))
            return gauss
            
        weight = get_gaussian(window_size).view(1, 1, -1, 1, 1) # (1, 1, W, 1, 1)
        weight = weight.expand(B, 1, window_size, H, W)
        
        for z in range(0, D, stride):
            z_start = z
            z_end = z + window_size
            
            if z_end > D:
                z_start = max(0, D - window_size)
                z_end = D
            
            # Crop
            chunk = volume[:, :, z_start:z_end, :, :]
            
            # Pad if chunk is smaller than window_size (at very start if D < window_size)
            # But logic above handles z_start for end case.
            # Handle D < window_size case
            if chunk.shape[2] < window_size:
                 pad_d = window_size - chunk.shape[2]
                 chunk = F.pad(chunk, (0,0, 0,0, 0,pad_d))
            
            # Inference
            with torch.no_grad():
                chunk_logits = self.model(chunk)
                if isinstance(chunk_logits, list):
                    chunk_logits = chunk_logits[0]
            
            # Unpad
            if z_end - z_start < window_size:
                chunk_logits = chunk_logits[:, :, :z_end-z_start, :, :]
                
            # Accumulate
            # Safe check shapes
            d_len = z_end - z_start
            w_chunk = weight[:, :, :d_len, :, :]
            
            logits[:, :, z_start:z_end, :, :] += chunk_logits * w_chunk
            count_map[:, :, z_start:z_end, :, :] += w_chunk
            
            if z_end == D:
                break
                
        # Average
        logits /= (count_map + 1e-8)
        return logits
