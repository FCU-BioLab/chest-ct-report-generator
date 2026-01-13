#!/usr/bin/env python3
"""
MedSAM2 Fine-tuning 配置模組
============================

使用 dataclass 定義所有訓練相關配置參數
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import json


@dataclass
class DataConfig:
    """資料相關配置（僅支援快取模式）"""
    # 快取目錄設定
    cache_dir: str = "cache"
    cache_dataset_type: str = "lndb"  # lndb/msd/both
    
    # 2.5D 模式設定
    use_2_5d: bool = False  # 使用 2.5D 輸入 (Z-1, Z, Z+1)，提升上下文資訊和分割效果

    
    # 資料比例
    data_fraction: float = 1.0
    
    # 資料分割比例 (7:1:2)
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2


@dataclass
class ModelConfig:
    """模型相關配置"""
    # SAM2 配置
    config: str = "sam2.1_hiera_t512.yaml"
    checkpoint: str = "MedSAM2/checkpoints/MedSAM2_CTLesion.pt"
    # checkpoint: str = "MedSAM2/checkpoints/MedSAM2_latest.pt"
    
    # 影像設定
    target_size: int = 512  # SAM 設計規格
    in_channels: int = 3  # RGB 格式


@dataclass
class TrainingConfig:
    """訓練相關配置"""
    # 基本參數
    epochs: int = 100
    batch_size: int = 32
    
    # 優化器
    learning_rate: float = 5e-6  # SAM 需要較低的學習率
    weight_decay: float = 1e-6
    
    # 調度器
    warmup_epochs: int = 5
    
    # 早停
    early_stopping_patience: int = 20
    
    # 梯度累積
    accumulation_steps: int = 1
    
    # 損失函數
    # ✅ 新增 'native' 選項：使用 MedSAM2 原生損失函數
    loss_type: str = "combined"  # combined/enhanced/native/tversky/focal
    
    # 資料增強
    use_augmentation: bool = False
    use_strong_augmentation: bool = False


@dataclass
class InferenceConfig:
    """推論相關配置"""
    # 閾值
    prediction_threshold: float = 0.5
    
    # 最小結節過濾
    min_nodule_diameter: float = 0.0
    
    # 測試過濾參數（全部設為 0 表示不過濾）
    min_area: int = 0  # 最小病灶面積（像素），0 表示不過濾
    min_confidence: float = 0.0  # 最小置信度（SAM2 IoU prediction），0 表示不過濾
    min_dice: float = 0.0  # 最小 Dice 分數（與 GT 的匹配度），0 表示不過濾


@dataclass
class Config:
    """完整配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # 實驗設定
    experiment_name: str = "medsam2_finetune"
    seed: int =27
    num_workers: int = 8
    device: str = "cuda"
    output_dir: Optional[str] = None
    
    def save(self, path: str):
        """保存配置到 JSON（包含完整訓練環境資訊）"""
        import platform
        import torch
        from datetime import datetime
        
        # 環境資訊
        env_info = {
            'timestamp': datetime.now().isoformat(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_memory_gb': round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else None,
            'platform': platform.platform(),
        }
        
        config_dict = {
            'environment': env_info,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'num_workers': self.num_workers,
            'device': self.device,
            'output_dir': self.output_dir
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """從 JSON 載入配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.data = DataConfig(**config_dict.get('data', {}))
        config.model = ModelConfig(**config_dict.get('model', {}))
        config.training = TrainingConfig(**config_dict.get('training', {}))
        config.inference = InferenceConfig(**config_dict.get('inference', {}))
        config.experiment_name = config_dict.get('experiment_name', 'medsam2_finetune')
        config.seed = config_dict.get('seed', 42)
        config.num_workers = config_dict.get('num_workers', 4)
        config.device = config_dict.get('device', 'cuda')
        config.output_dir = config_dict.get('output_dir')
        
        return config
    
    @classmethod
    def from_args(cls, args) -> 'Config':
        """從 argparse 參數建立配置"""
        config = cls()
        
        # Data config
        config.data.use_cache = getattr(args, 'use_cache', True)
        config.data.cache_dir = getattr(args, 'cache_dir', 'cache')
        config.data.cache_dataset_type = getattr(args, 'cache_dataset_type', 'both')
        config.data.data_dir = getattr(args, 'data_dir', '../datasets/aLL_patients_data/LNDb')
        config.data.rad_id = getattr(args, 'rad_id', 'consensus')
        config.data.axis = getattr(args, 'axis', 2)
        config.data.data_fraction = getattr(args, 'data_fraction', 1.0)
        
        # Model config
        config.model.config = getattr(args, 'config', 'sam2.1_hiera_t512.yaml')
        # config.model.checkpoint = getattr(args, 'checkpoint', 'MedSAM2/checkpoints/MedSAM2_CTLesion.pt')
        config.model.checkpoint = getattr(args, 'checkpoint', 'MedSAM2/checkpoints/MedSAM2_latest.pt')
        
        # Training config
        config.training.epochs = getattr(args, 'epochs', 100)
        config.training.batch_size = getattr(args, 'batch_size', 32)
        config.training.learning_rate = getattr(args, 'lr', 5e-6)
        config.training.weight_decay = getattr(args, 'weight_decay', 1e-6)
        config.training.warmup_epochs = getattr(args, 'warmup_epochs', 5)
        config.training.early_stopping_patience = getattr(args, 'early_stopping_patience', 20)
        config.training.accumulation_steps = getattr(args, 'accumulation_steps', 1)
        config.training.loss_type = getattr(args, 'loss_type', 'combined')
        config.training.use_augmentation = getattr(args, 'augmentation', False)
        config.training.use_strong_augmentation = getattr(args, 'strong_augmentation', False)
        
        # Inference config
        config.inference.min_nodule_diameter = getattr(args, 'min_nodule_diameter', 0.0)
        
        # General
        config.seed = getattr(args, 'seed', 42)
        config.num_workers = getattr(args, 'num_workers', 8)
        config.output_dir = getattr(args, 'output_dir', None)
        
        return config


def get_default_config() -> Config:
    """獲取預設配置"""
    return Config()


if __name__ == "__main__":
    # 測試配置
    config = get_default_config()
    print(f"Cache dir: {config.data.cache_dir}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Loss type: {config.training.loss_type}")
    print(f"Seed: {config.seed}")
