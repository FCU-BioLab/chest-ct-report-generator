#!/usr/bin/env python3
"""
UNet++ 肺結節分割訓練 - 工具函數模組
"""

import logging
import random
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json

import numpy as np
import torch
import matplotlib.pyplot as plt


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """設定日誌"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        # 使用 UTF-8 編碼避免中文亂碼
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def set_seed(seed: int = 42):
    """設定隨機種子以確保可重複性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "cuda") -> torch.device:
    """獲取可用設備"""
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """繪製訓練歷史曲線"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True)
    
    # Dice
    ax = axes[0, 1]
    ax.plot(history['val_dice'], label='Validation Dice')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice Score')
    ax.set_title('Validation Dice Score')
    ax.legend()
    ax.grid(True)
    
    # IoU
    ax = axes[1, 0]
    ax.plot(history['val_iou'], label='Validation IoU')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('Validation IoU')
    ax.legend()
    ax.grid(True)
    
    # Learning Rate
    ax = axes[1, 1]
    ax.plot(history['lr'], label='Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"訓練曲線已保存: {save_path}")
    
    plt.close()


def visualize_predictions(
    images: np.ndarray,
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
    save_path: Optional[str] = None,
    num_samples: int = 4
):
    """視覺化預測結果"""
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # 影像
        ax = axes[i, 0]
        if images[i].ndim == 3:
            ax.imshow(images[i, 1], cmap='gray')  # 中間切片
        else:
            ax.imshow(images[i], cmap='gray')
        ax.set_title('Input Image')
        ax.axis('off')
        
        # GT 遮罩
        ax = axes[i, 1]
        ax.imshow(gt_masks[i], cmap='gray')
        ax.set_title('Ground Truth')
        ax.axis('off')
        
        # 預測遮罩
        ax = axes[i, 2]
        ax.imshow(pred_masks[i], cmap='gray')
        ax.set_title('Prediction')
        ax.axis('off')
        
        # 疊加
        ax = axes[i, 3]
        if images[i].ndim == 3:
            overlay = images[i, 1].copy()
        else:
            overlay = images[i].copy()
        
        # 創建 RGB 疊加
        overlay_rgb = np.stack([overlay, overlay, overlay], axis=-1)
        overlay_rgb[gt_masks[i] > 0.5, 1] = 1.0  # GT 為綠色
        overlay_rgb[pred_masks[i] > 0.5, 0] = 1.0  # Pred 為紅色
        
        ax.imshow(overlay_rgb)
        ax.set_title('Overlay (GT=Green, Pred=Red)')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"預測視覺化已保存: {save_path}")
    
    plt.close()


def calculate_dataset_statistics(
    data_dir: str,
    patient_ids: List[str]
) -> Dict:
    """計算資料集統計資訊"""
    cache_dir = Path(data_dir).parent / 'cache' / 'lndb_preprocessed'
    
    stats = {
        'num_patients': len(patient_ids),
        'total_slices': 0,
        'positive_slices': 0,
        'total_nodules': 0,
        'nodule_sizes': [],
        'slice_shapes': []
    }
    
    for patient_id in patient_ids:
        cache_path = cache_dir / f"{patient_id}.npz"
        if not cache_path.exists():
            continue
        
        data = np.load(cache_path, allow_pickle=True)
        volume = data['volume']
        masks = data['masks']
        
        stats['total_slices'] += volume.shape[0]
        stats['slice_shapes'].append(volume.shape[1:])
        
        if len(masks) > 0 and masks.ndim > 0:
            # 軟共識
            if len(masks.shape) > 1:
                consensus = np.mean(masks, axis=0)
            else:
                consensus = masks
            
            # 統計正樣本切片
            for z in range(consensus.shape[0]):
                if np.any(consensus[z] > 0.5):
                    stats['positive_slices'] += 1
    
    if stats['total_slices'] > 0:
        stats['positive_ratio'] = stats['positive_slices'] / stats['total_slices']
    else:
        stats['positive_ratio'] = 0
    
    return stats


def custom_collate_fn(batch):
    """自定義 collate 函數，處理可能的 None 值"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    # 支援 patient_id (LNDb) 或 case_id (MSD)
    id_key = 'patient_id' if 'patient_id' in batch[0] else 'case_id'
    
    result = {
        'image': torch.stack([b['image'] for b in batch]),
        'mask': torch.stack([b['mask'] for b in batch]),
        'patient_id': [b[id_key] for b in batch],  # 統一命名
        'slice_idx': [b['slice_idx'] for b in batch],
        'is_positive': [b['is_positive'] for b in batch]
    }
    
    # 新增 patch_idx（如果存在）
    if 'patch_idx' in batch[0]:
        result['patch_idx'] = [b['patch_idx'] for b in batch]
    
    return result


if __name__ == "__main__":
    # 測試
    setup_logging()
    set_seed(42)
    
    device = get_device()
    print(f"Device: {device}")
