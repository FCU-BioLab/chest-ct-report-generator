#!/usr/bin/env python3
"""
MedSAM2 視頻模式配置
====================

定義視頻訓練所需的所有配置參數
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple
import json


@dataclass
class DataConfig:
    """資料相關配置"""
    # 資料來源
    lndb_dir: str = ""  # LNDb 資料集目錄
    msd_dir: str = ""   # MSD Lung 資料集目錄
    npz_dir: str = "cache\\lndb_video_npz"  # NPZ 輸出目錄
    # npz_dir: str = "cache\\msd_video_npz"  # NPZ 輸出目錄

    
    # 視頻參數
    context_slices: int = 6  # 中心切片前後各取幾個切片 (總長度 = 2*context + 1)
    min_video_length: int = 5  # 最短視頻長度
    max_video_length: int = 32  # 最長視頻長度（記憶體限制）
    
    # 過濾參數
    min_nodule_diameter: float = 0.0  # 最小結節直徑 (mm)
    max_nodule_diameter: float = 100.0  # 最大結節直徑 (mm)
    
    # 資料分割
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 快取
    use_cache: bool = True
    cache_dir: str = "cache"


@dataclass
class ModelConfig:
    """模型相關配置"""
    # SAM2 配置
    config: str = "sam2.1_hiera_t512.yaml"
    checkpoint: str = "MedSAM2/checkpoints/MedSAM2_latest.pt"
    
    # 影像設定
    image_size: int = 512
    
    # 視頻預測器設定
    fill_hole_area: int = 0
    non_overlap_masks: bool = False
    clear_non_cond_mem_around_input: bool = False
    

@dataclass
class TrainingConfig:
    """訓練相關配置"""
    # 基本參數
    epochs: int = 100  # 增加訓練輪數
    batch_size: int = 1  # 視頻模式通常 batch_size=1
    
    # 優化器 - 使用分層學習率
    learning_rate: float = 1e-4  # 基礎學習率
    decoder_lr_multiplier: float = 5.0  # mask decoder 使用較高學習率（降低倍率防止不穩定）
    weight_decay: float = 0.01  # 正則化
    max_lr: float = 3e-4  # 最大學習率上限，防止訓練崩潰
    
    # 調度器
    warmup_epochs: int = 10  # 增加 warmup 時間，讓課程學習生效
    
    # 損失函數
    dice_weight: float = 1.0
    focal_weight: float = 0.5
    bce_weight: float = 0.5  # 新增 BCE loss
    propagation_weight: float = 0.3  # 傳播一致性損失權重
    
    # 訓練策略
    prompt_type: str = "bbox"  # bbox, point, mask
    propagation_steps: int = 3  # 前向/後向傳播步數
    use_curriculum_learning: bool = True  # 課程學習：前期用 GT 引導，後期自主預測
    
    # 凍結選項（減少過擬合）
    freeze_prompt_encoder: bool = False
    freeze_memory_encoder: bool = False
    freeze_mask_decoder: bool = False  # 不要凍結 mask decoder
    
    # 早停
    early_stopping_patience: int = 15  # 適當增加 patience
    
    # 混合精度
    use_amp: bool = True
    
    # 梯度
    grad_clip: float = 1.0
    accumulation_steps: int = 4


@dataclass
class InferenceConfig:
    """推論相關配置"""
    threshold: float = 0.5
    propagate_full_volume: bool = True  # 是否傳播到整個體積


@dataclass 
class VideoConfig:
    """完整視頻訓練配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # 實驗設定
    experiment_name: str = "medsam2_video"
    output_dir: str = "video_output"
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    
    def save(self, path: str):
        """保存配置到 JSON"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'seed': self.seed,
            'device': self.device,
            'num_workers': self.num_workers,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'VideoConfig':
        """從 JSON 載入配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.data = DataConfig(**config_dict.get('data', {}))
        config.model = ModelConfig(**config_dict.get('model', {}))
        config.training = TrainingConfig(**config_dict.get('training', {}))
        config.inference = InferenceConfig(**config_dict.get('inference', {}))
        config.experiment_name = config_dict.get('experiment_name', 'medsam2_video')
        config.output_dir = config_dict.get('output_dir', 'video_output')
        config.seed = config_dict.get('seed', 42)
        config.device = config_dict.get('device', 'cuda')
        config.num_workers = config_dict.get('num_workers', 4)
        
        return config


def get_default_config() -> VideoConfig:
    """取得預設配置"""
    return VideoConfig()
