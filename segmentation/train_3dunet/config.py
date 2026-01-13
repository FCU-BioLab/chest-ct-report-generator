#!/usr/bin/env python3
"""
3D U-Net Video Finetuning Config
================================

Configuration for 3D U-Net based volumetric segmentation on video (sequence) data.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import json
import os

@dataclass
class DataConfig:
    """Data configuration"""
    # Data sources
    lndb_dir: str = ""  # LNDb dataset directory
    msd_dir: str = ""   # MSD Lung dataset directory
    npz_dir: str = "../../cache/video_npz"  # NPZ output directory
    
    # Volume parameters
    context_slices: int = 6  # Slices before/after center
    min_depth: int = 5
    max_depth: int = 32  # Limited by memory, usually 16-32 for 3D U-Net
    
    # Filtering
    min_nodule_diameter: float = 4.0  # mm
    max_nodule_diameter: float = 100.0  # mm
    
    # Split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Cache
    use_cache: bool = True
    cache_dir: str = "../../cache"


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str = "unet3d"
    in_channels: int = 1
    out_channels: int = 1
    base_filters: int = 32
    layer_multipliers: Tuple[int, ...] = (1, 2, 4, 8, 16)
    use_batchnorm: bool = True
    dropout_rate: float = 0.0
    image_size: int = 256  # Input spatial size (H, W)


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basics
    epochs: int = 100
    batch_size: int = 4  # Can be larger than 1 for 3D U-Net depending on size
    
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Scheduler
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Loss weights
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 15
    
    # Hardware
    use_amp: bool = True
    grad_clip: float = 1.0
    accumulation_steps: int = 1


@dataclass
class Config:
    """Main Configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment
    experiment_name: str = "unet3d_volume"
    output_dir: str = "volume_output_unet3d"
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    
    def save(self, path: str):
        """Save config to JSON"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'seed': self.seed,
            'device': self.device,
            'num_workers': self.num_workers,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load from JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.data = DataConfig(**config_dict.get('data', {}))
        config.model = ModelConfig(**config_dict.get('model', {}))
        config.training = TrainingConfig(**config_dict.get('training', {}))
        config.experiment_name = config_dict.get('experiment_name', 'unet3d_video')
        config.output_dir = config_dict.get('output_dir', 'video_output_unet3d')
        config.seed = config_dict.get('seed', 42)
        config.device = config_dict.get('device', 'cuda')
        config.num_workers = config_dict.get('num_workers', 4)
        
        return config

def get_default_config() -> Config:
    return Config()
