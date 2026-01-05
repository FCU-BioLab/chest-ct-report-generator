#!/usr/bin/env python3
"""
MedSAM2 Fine-tuning Package
===========================

提供胸部 CT 腫瘤分割的 MedSAM2 微調功能

模組:
- dataset: 資料集載入與處理
- trainer: 模型訓練與評估
- visualizer: 分割結果可視化
- feature_extractor: 病灶特徵提取
- losses: 損失函數
- utils: 工具函數
- checkpoint_manager: Checkpoint 管理
- patient_analyzer: 患者分析
- feature_saver: 特徵保存
- llm_data_generator: LLM 資料生成
"""

__version__ = "1.2.0"
__author__ = "MedSAM2 Fine-tuning Team"

from .dataset import LNDbDataset, DataAugmentation, CachedSliceDataset
from .trainer import MedSAM2Trainer
from .visualizer import SegmentationVisualizer
from .feature_extractor import LesionFeatureExtractor
from .losses import DiceLoss, CombinedLoss, FocalLoss, EnhancedCombinedLoss, TverskyLoss
from .utils import (
    setup_logging,
    suppress_noisy_logs,
    split_dataset,
    compute_all_metrics,
    EarlyStopping,
    custom_collate_fn
)
from .config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    get_default_config
)

# 重構後的獨立模組
from .checkpoint_manager import CheckpointManager
from .patient_analyzer import PatientAnalyzer
from .feature_saver import FeatureSaver
from .llm_data_generator import LLMDataGenerator

__all__ = [
    # Dataset
    'LNDbDataset',
    'CachedSliceDataset',
    'DataAugmentation',
    
    # Trainer
    'MedSAM2Trainer',
    
    # Visualization & Feature Extraction
    'SegmentationVisualizer',
    'LesionFeatureExtractor',
    
    # Losses
    'DiceLoss',
    'CombinedLoss',
    'FocalLoss',
    'EnhancedCombinedLoss',
    'TverskyLoss',
    
    # Utils
    'setup_logging',
    'suppress_noisy_logs',
    'split_dataset',
    'compute_all_metrics',
    'EarlyStopping',
    'custom_collate_fn',
    
    # Config
    'Config',
    'DataConfig',
    'ModelConfig',
    'TrainingConfig',
    'InferenceConfig',
    'get_default_config',
    
    # 重構後的獨立模組
    'CheckpointManager',
    'PatientAnalyzer',
    'FeatureSaver',
    'LLMDataGenerator',
]

