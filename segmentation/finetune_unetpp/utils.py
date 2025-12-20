#!/usr/bin/env python3
"""
工具函數模組
提供 logging、評估指標、資料分割等通用功能
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import warnings
import json

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def setup_logging(log_dir: str = "finetune_logs") -> logging.Logger:
    """
    設定日誌系統
    
    Args:
        log_dir: 日誌輸出目錄
        
    Returns:
        配置好的 logger 物件
    """
    if os.environ.get('PYTORCH_WORKER_ID') is not None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(console_handler)
            logger.propagate = False
        return logger
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_{timestamp}.log"
    
    logger = logging.getLogger(__name__)
    
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(
        str(log_path / log_filename), 
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    return logger


def compute_dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    計算 Dice 係數
    
    Args:
        pred: 預測遮罩
        target: 目標遮罩
        smooth: 平滑項
    
    Returns:
        Dice 係數
    """
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = np.sum(pred * target)
    dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)
    
    return float(dice)


def compute_iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    計算 IoU (Intersection over Union)
    
    Args:
        pred: 預測遮罩
        target: 目標遮罩
        smooth: 平滑項
    
    Returns:
        IoU 值
    """
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return float(iou)


def compute_precision_recall(
    pred: np.ndarray, 
    target: np.ndarray, 
    smooth: float = 1e-6
) -> Tuple[float, float]:
    """
    計算 Precision 和 Recall
    
    Args:
        pred: 預測遮罩
        target: 目標遮罩
        smooth: 平滑項
    
    Returns:
        (precision, recall) 元組
    """
    pred = pred.flatten()
    target = target.flatten()
    
    tp = np.sum(pred * target)
    fp = np.sum(pred * (1 - target))
    fn = np.sum((1 - pred) * target)
    
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    return float(precision), float(recall)


def compute_hausdorff_distance(
    pred: np.ndarray, 
    target: np.ndarray
) -> float:
    """
    計算 Hausdorff 距離 (95th percentile)
    
    Args:
        pred: 預測遮罩
        target: 目標遮罩
    
    Returns:
        Hausdorff 距離
    """
    if np.sum(pred) == 0 or np.sum(target) == 0:
        return float('inf')
    
    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)
    
    if len(pred_points) == 0 or len(target_points) == 0:
        return float('inf')
    
    # 計算從 pred 到 target 的距離
    target_dist = distance_transform_edt(1 - target)
    pred_to_target = target_dist[pred > 0]
    
    # 計算從 target 到 pred 的距離
    pred_dist = distance_transform_edt(1 - pred)
    target_to_pred = pred_dist[target > 0]
    
    # 95th percentile Hausdorff 距離
    hd95 = max(
        np.percentile(pred_to_target, 95) if len(pred_to_target) > 0 else 0,
        np.percentile(target_to_pred, 95) if len(target_to_pred) > 0 else 0
    )
    
    return float(hd95)


def compute_all_metrics(
    pred: np.ndarray, 
    target: np.ndarray,
    compute_hd: bool = False
) -> Dict[str, float]:
    """
    計算所有評估指標
    
    Args:
        pred: 預測遮罩 (二值化)
        target: 目標遮罩 (二值化)
        compute_hd: 是否計算 Hausdorff 距離
    
    Returns:
        包含所有指標的字典
    """
    # 確保是二值化的
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    
    dice = compute_dice(pred, target)
    iou = compute_iou(pred, target)
    precision, recall = compute_precision_recall(pred, target)
    
    metrics = {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / (precision + recall + 1e-6)
    }
    
    if compute_hd:
        hd95 = compute_hausdorff_distance(pred, target)
        metrics['hd95'] = hd95
    
    return metrics


class EarlyStopping:
    """
    早停機制
    
    Args:
        patience: 容忍的 epoch 數
        min_delta: 最小改進量
        mode: 監控模式 ('min' or 'max')
    """
    
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        檢查是否應該停止訓練
        
        Args:
            score: 當前分數
        
        Returns:
            是否應該早停
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


class PatientMetricsTracker:
    """
    Patient-level segmentation metrics tracker.
    
    IMPORTANT: This tracker should only receive samples with POSITIVE GT (lesions present).
    Empty GT samples should NOT be added to this tracker to avoid polluting statistics.
    
    Usage:
        tracker = PatientMetricsTracker()
        for sample in positive_gt_samples:
            metrics = compute_all_metrics(pred, target)
            tracker.add_sample(patient_id, slice_idx, metrics)
        
        overall = tracker.get_overall_metrics()
    """
    
    def __init__(self):
        self.patient_metrics: Dict[str, List[Dict]] = {}
    
    def add_sample(
        self, 
        patient_id: str, 
        slice_idx: int, 
        metrics: Dict[str, float]
    ):
        """添加單個切片的指標"""
        if patient_id not in self.patient_metrics:
            self.patient_metrics[patient_id] = []
        
        self.patient_metrics[patient_id].append({
            'slice_idx': slice_idx,
            **metrics
        })
    
    def get_patient_summary(self) -> Dict[str, Dict]:
        """獲取每個患者的匯總指標"""
        summary = {}
        
        for patient_id, slices in self.patient_metrics.items():
            dices = [s['dice'] for s in slices]
            ious = [s['iou'] for s in slices]
            
            summary[patient_id] = {
                'num_slices': len(slices),
                'mean_dice': np.mean(dices),
                'std_dice': np.std(dices),
                'min_dice': np.min(dices),
                'max_dice': np.max(dices),
                'mean_iou': np.mean(ious),
            }
        
        return summary
    
    def get_overall_metrics(self) -> Dict[str, float]:
        """獲取所有患者的整體指標"""
        all_dices = []
        all_ious = []
        
        for slices in self.patient_metrics.values():
            for s in slices:
                all_dices.append(s['dice'])
                all_ious.append(s['iou'])
        
        if not all_dices:
            return {'mean_dice': 0, 'mean_iou': 0}
        
        return {
            'num_patients': len(self.patient_metrics),
            'num_samples': len(all_dices),
            'mean_dice': np.mean(all_dices),
            'std_dice': np.std(all_dices),
            'median_dice': np.median(all_dices),
            'mean_iou': np.mean(all_ious),
            'std_iou': np.std(all_ious),
        }
    
    def clear(self):
        """清空追蹤器"""
        self.patient_metrics.clear()


def split_dataset(
    patient_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: Optional[int] = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    分割資料集
    
    Args:
        patient_ids: 患者 ID 列表
        train_ratio: 訓練集比例
        val_ratio: 驗證集比例
        test_ratio: 測試集比例
        seed: 隨機種子
    
    Returns:
        (train_ids, val_ids, test_ids) 元組
    """
    if seed is not None:
        np.random.seed(seed)
    
    patient_ids = list(patient_ids)
    np.random.shuffle(patient_ids)
    
    n = len(patient_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]
    
    return train_ids, val_ids, test_ids


def save_dataset_split_info(
    output_dir: str,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    config: Dict = None
):
    """保存資料集分割資訊（與 MedSAM2 格式相容）"""
    info = {
        'split_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train': {
            'count': len(train_ids),
            'patient_ids': [str(pid) for pid in sorted(train_ids, key=lambda x: int(x) if str(x).isdigit() else x)]
        },
        'val': {
            'count': len(val_ids),
            'patient_ids': [str(pid) for pid in sorted(val_ids, key=lambda x: int(x) if str(x).isdigit() else x)]
        },
        'test': {
            'count': len(test_ids),
            'patient_ids': [str(pid) for pid in sorted(test_ids, key=lambda x: int(x) if str(x).isdigit() else x)]
        },
        'total': len(train_ids) + len(val_ids) + len(test_ids)
    }
    
    if config:
        info['config'] = config
    
    output_path = Path(output_dir) / 'dataset_split.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    logger = logging.getLogger(__name__)
    logger.info(f"✅ 資料集分割資訊已保存: {output_path}")


def load_dataset_split_info(split_file: str) -> Dict:
    """載入資料集分割資訊"""
    with open(split_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """
    自定義 collate 函數
    """
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    bboxes = torch.stack([item['bbox'] for item in batch])
    
    patient_ids = [item['patient_id'] for item in batch]
    slice_indices = [item['slice_idx'] for item in batch]
    
    return {
        'image': images,
        'mask': masks,
        'bbox': bboxes,
        'patient_id': patient_ids,
        'slice_idx': slice_indices
    }


def set_seed(seed: int):
    """
    設定隨機種子以確保可重現性
    
    [E] Enhanced for complete reproducibility:
    - Sets Python random, NumPy, PyTorch seeds
    - Sets CUDA seeds and deterministic flags
    - Sets CUBLAS workspace config for GPU determinism
    """
    import random
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For CUBLAS determinism (PyTorch >= 1.8)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def worker_init_fn(worker_id: int):
    """
    DataLoader worker init function for reproducibility.
    
    [E] Each worker gets a unique but deterministic seed derived from
    the main process seed, ensuring reproducible data loading order.
    
    Args:
        worker_id: Worker process ID (0-indexed)
    """
    import random
    # Get worker seed from PyTorch's initial seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    """獲取可用的計算設備"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def format_time(seconds: float) -> str:
    """格式化時間"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


if __name__ == "__main__":
    # 測試工具函數
    print("Testing utility functions...")
    
    # 測試指標計算
    pred = np.random.rand(256, 256) > 0.5
    target = np.random.rand(256, 256) > 0.5
    
    metrics = compute_all_metrics(pred, target, compute_hd=True)
    print(f"Metrics: {metrics}")
    
    # 測試早停
    es = EarlyStopping(patience=3, mode='max')
    scores = [0.8, 0.82, 0.81, 0.80, 0.79, 0.78]
    for i, score in enumerate(scores):
        stop = es(score)
        print(f"Epoch {i+1}: score={score}, stop={stop}")
        if stop:
            break
    
    # 測試資料集分割
    patient_ids = [f"patient_{i:03d}" for i in range(100)]
    train_ids, val_ids, test_ids = split_dataset(patient_ids, seed=42)
    print(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    
    print("\nAll utility tests passed!")
