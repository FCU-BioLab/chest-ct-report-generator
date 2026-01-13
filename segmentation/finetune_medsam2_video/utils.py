#!/usr/bin/env python3
"""
工具函數模組
============

提供視頻訓練的輔助工具函數。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """計算 Dice 係數"""
    pred = pred.view(-1).float()
    target = target.view(-1).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """計算 IoU (Jaccard Index)"""
    pred = pred.view(-1).float()
    target = target.view(-1).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """計算所有評估指標"""
    pred_binary = (pred > 0.5).float()
    target_binary = target.float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # True/False Positives/Negatives
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()
    
    # Metrics
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    specificity = (tn + 1e-6) / (tn + fp + 1e-6)
    accuracy = (tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'specificity': specificity.item(),
        'accuracy': accuracy.item(),
    }


def visualize_video_prediction(
    frames: np.ndarray,
    masks_gt: np.ndarray,
    masks_pred: np.ndarray,
    center_idx: int,
    output_path: str,
    max_cols: int = 6,
) -> None:
    """
    視覺化視頻預測結果
    
    Args:
        frames: (D, H, W) CT 切片
        masks_gt: (D, H, W) Ground truth
        masks_pred: (D, H, W) 預測結果
        center_idx: 中心幀索引
        output_path: 輸出路徑
        max_cols: 每行最多幾個子圖
    """
    D = frames.shape[0]
    num_cols = min(D, max_cols)
    num_rows = (D + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows * 2, num_cols, figsize=(num_cols * 3, num_rows * 6))
    
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows * 2 == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(D):
        row = (i // num_cols) * 2
        col = i % num_cols
        
        # GT
        axes[row, col].imshow(frames[i], cmap='gray')
        if masks_gt[i].max() > 0:
            mask_overlay = np.ma.masked_where(masks_gt[i] == 0, masks_gt[i])
            axes[row, col].imshow(mask_overlay, cmap='Greens', alpha=0.5)
        axes[row, col].set_title(f'GT Frame {i}' + (' ⭐' if i == center_idx else ''))
        axes[row, col].axis('off')
        
        # Pred
        axes[row + 1, col].imshow(frames[i], cmap='gray')
        if masks_pred[i].max() > 0:
            mask_overlay = np.ma.masked_where(masks_pred[i] == 0, masks_pred[i])
            axes[row + 1, col].imshow(mask_overlay, cmap='Reds', alpha=0.5)
        
        # 計算 Dice
        dice = compute_dice(
            torch.from_numpy(masks_pred[i].astype(np.float32)),
            torch.from_numpy(masks_gt[i].astype(np.float32))
        )
        axes[row + 1, col].set_title(f'Pred Frame {i} (Dice: {dice:.2f})')
        axes[row + 1, col].axis('off')
    
    # 隱藏空白子圖
    for i in range(D, num_cols * num_rows):
        row = (i // num_cols) * 2
        col = i % num_cols
        axes[row, col].axis('off')
        axes[row + 1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_video_gif(
    frames: np.ndarray,
    masks: np.ndarray,
    output_path: str,
    fps: int = 5,
) -> None:
    """
    將視頻預測結果保存為 GIF
    
    Args:
        frames: (D, H, W) CT 切片
        masks: (D, H, W) 分割遮罩
        output_path: 輸出路徑
        fps: 幀率
    """
    images = []
    
    for i in range(frames.shape[0]):
        # 建立 RGB 影像
        frame_rgb = np.stack([frames[i], frames[i], frames[i]], axis=-1)
        
        # 疊加 mask（紅色）
        if masks[i].max() > 0:
            mask_bool = masks[i] > 0
            frame_rgb[mask_bool, 0] = np.clip(frame_rgb[mask_bool, 0] * 0.5 + 128, 0, 255)
            frame_rgb[mask_bool, 1] = frame_rgb[mask_bool, 1] * 0.5
            frame_rgb[mask_bool, 2] = frame_rgb[mask_bool, 2] * 0.5
        
        images.append(Image.fromarray(frame_rgb.astype(np.uint8)))
    
    # 保存 GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,
        loop=0,
    )


def apply_ct_windowing(
    ct_array: np.ndarray,
    window_center: float = -600,
    window_width: float = 1500,
) -> np.ndarray:
    """
    套用 CT 窗位/窗寬
    
    Args:
        ct_array: CT 影像 (HU 值)
        window_center: 窗位
        window_width: 窗寬
        
    Returns:
        正規化到 0-255 的 uint8 影像
    """
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    
    ct_clipped = np.clip(ct_array, min_val, max_val)
    ct_normalized = ((ct_clipped - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    return ct_normalized


def estimate_diameter_from_mask(mask: np.ndarray, spacing: np.ndarray) -> float:
    """
    從 mask 估算病灶直徑
    
    Args:
        mask: 3D 分割遮罩
        spacing: (z, y, x) voxel spacing in mm
        
    Returns:
        等效球體直徑 (mm)
    """
    volume_voxels = np.sum(mask > 0)
    volume_mm3 = volume_voxels * np.prod(spacing)
    
    # 假設球形
    diameter_mm = 2 * np.power(3 * volume_mm3 / (4 * np.pi), 1/3)
    
    return diameter_mm


class EarlyStopping:
    """早停機制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: 容忍幾個 epoch 沒有改善
            min_delta: 最小改善量
            mode: 'max' 或 'min'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        檢查是否應該早停
        
        Args:
            score: 當前分數
            
        Returns:
            True 表示應該早停
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class AverageMeter:
    """追蹤平均值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
