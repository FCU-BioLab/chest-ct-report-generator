#!/usr/bin/env python3
"""
MedSAM2 視頻模式微調模組
========================

將 CT 切片序列視為「影片」，利用 MedSAM2 的時序傳播能力學習病灶分割。

核心概念：
- 每個病例的連續 CT 切片 → 視頻幀序列
- 病灶標註 → 物件追蹤標籤
- MedSAM2 的 Video Predictor → 時序一致的分割

模組結構：
- video_dataset.py: 視頻格式資料集
- npz_converter.py: LNDb/MSD → NPZ 轉換器
- video_trainer.py: 視頻模式訓練器
- config.py: 配置管理
- main.py: 主程式入口
"""

from .config import VideoConfig, DataConfig, ModelConfig, TrainingConfig
from .video_dataset import VideoLesionDataset, CTVideoSample
from .preprocess import VideoPreprocessor
from .video_trainer import MedSAM2VideoTrainer
from .visualizer import VideoVisualizer

__version__ = "1.0.0"
__all__ = [
    "VideoConfig",
    "DataConfig", 
    "ModelConfig",
    "TrainingConfig",
    "VideoLesionDataset",
    "CTVideoSample",
    "VideoPreprocessor",
    "MedSAM2VideoTrainer",
    "VideoVisualizer",
]
