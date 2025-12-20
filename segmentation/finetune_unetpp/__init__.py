#!/usr/bin/env python3
"""
UNet++ Fine-tuning for Chest Tumor Segmentation
================================================

使用 UNet++ 架構進行胸部 CT 腫瘤分割微調

主要模組:
- model.py: UNet++ 模型架構
- dataset.py: 資料集載入與處理
- trainer.py: 訓練器
- losses.py: 損失函數
- utils.py: 工具函數
- main.py: 主程式入口
"""

from .model import UNetPlusPlus, get_unetpp_model
from .dataset import ChestTumorDataset, LNDbDataset, DataAugmentation
from .trainer import UNetPPTrainer
from .losses import CombinedLoss, EnhancedCombinedLoss, DiceLoss
from .utils import setup_logging, compute_all_metrics

__version__ = "1.0.0"
__all__ = [
    "UNetPlusPlus",
    "get_unetpp_model",
    "ChestTumorDataset",
    "LNDbDataset",
    "DataAugmentation",
    "UNetPPTrainer",
    "CombinedLoss",
    "EnhancedCombinedLoss",
    "DiceLoss",
    "setup_logging",
    "compute_all_metrics",
]
