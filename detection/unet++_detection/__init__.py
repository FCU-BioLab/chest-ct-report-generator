#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNet++ Detection Module
UNet++ 檢測模組初始化文件

該模組實現了基於 UNet++ 的醫學影像病灶檢測功能，包括：
- UNet++ 網路架構
- 分割和檢測的端到端訓練
- 專用的數據載入和處理
- 性能評估和可視化

作者: GitHub Copilot
日期: 2025-09-18
"""

from .unetpp_model import UNetPlusPlus, UNetPPDetector
from .unetpp_dataset import UNetPPDetectionDataset
from .train_unetpp import train_unetpp_detector
from .test_unetpp import evaluate_unetpp_detector

__version__ = "1.0.0"
__author__ = "GitHub Copilot"
__email__ = "copilot@github.com"

__all__ = [
    'UNetPlusPlus',
    'UNetPPDetector', 
    'UNetPPDetectionDataset',
    'train_unetpp_detector',
    'evaluate_unetpp_detector'
]