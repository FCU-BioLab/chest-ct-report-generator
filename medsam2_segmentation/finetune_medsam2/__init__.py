#!/usr/bin/env python3
"""
MedSAM2 Fine-tuning Package
===========================

提供胸部 CT 腫瘤分割的 MedSAM2 微調功能

模組:
- dataset: 資料集載入與處理
- trainer: 模型訓練與評估
- losses: 損失函數
- utils: 工具函數
"""

__version__ = "1.0.0"
__author__ = "MedSAM2 Fine-tuning Team"

from .dataset import ChestTumorDataset, DataAugmentation
from .trainer import MedSAM2Trainer
from .losses import DiceLoss, CombinedLoss, FocalLoss
from .utils import (
    setup_logging,
    suppress_noisy_logs,
    split_dataset,
    compute_all_metrics,
    EarlyStopping,
    custom_collate_fn
)

__all__ = [
    # Dataset
    'ChestTumorDataset',
    'DataAugmentation',
    
    # Trainer
    'MedSAM2Trainer',
    
    # Losses
    'DiceLoss',
    'CombinedLoss',
    'FocalLoss',
    
    # Utils
    'setup_logging',
    'suppress_noisy_logs',
    'split_dataset',
    'compute_all_metrics',
    'EarlyStopping',
    'custom_collate_fn',
]
