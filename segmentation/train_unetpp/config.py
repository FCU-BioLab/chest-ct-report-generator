#!/usr/bin/env python3
"""
UNet++ 肺結節分割訓練 - 配置模組
===================================

定義所有訓練相關的配置參數
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import json


@dataclass
class DataConfig:
    """資料相關配置"""
    # 資料路徑
    data_dir: str = "datasets/aLL_patients_data/LNDb"
    output_dir: str = "result"
    
    # Spacing 設定
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # 各向同性 spacing (mm)
    
    # HU Windowing
    hu_window_center: float = -400  # Lung window center
    hu_window_width: float = 1200   # Lung window width [-1000, 200]
    
    # Patch 設定
    patch_size: int = 224  # CSEA-Net 論文規格
    
    # 2.5D 設定
    use_2_5d: bool = True  # 啟用 2.5D input (z-1, z, z+1)
    slice_distance_mm: float = 2.0  # 固定毫米距離取鄰近切片
    
    # Val/Test restrictive cropping
    val_num_patches: int = 4  # 每個 slice 抽取的 patch 數
    
    # 採樣比例
    positive_ratio: float = 0.7    # 正樣本（結節中心）比例
    hard_negative_ratio: float = 0.2  # 肺野內無結節區域
    random_negative_ratio: float = 0.1  # 完全隨機
    
    # 資料分割（單次訓練模式）7:2:1
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    
    # 遮罩共識方式
    consensus_method: str = "soft"  # "soft", "intersection", "union", "rad1"
    
    # 快取
    cache_preprocessed: bool = True
    cache_dir: str = "cache/lndb_preprocessed"


@dataclass
class ModelConfig:
    """模型相關配置"""
    # UNet++ 架構
    encoder_name: str = "efficientnet-b4"
    encoder_weights: str = "imagenet"  # 預訓練權重
    in_channels: int = 3  # 2.5D: z-1, z, z+1 疊成 3 channel
    num_classes: int = 1  # 二元分割
    use_2d: bool = True  # 使用純 2D 模式（但輸入是 2.5D）
    
    # 深度監督
    deep_supervision: bool = True


@dataclass
class TrainingConfig:
    """訓練相關配置"""
    # 基本參數
    batch_size: int = 16  # 加大 batch size 提高 GPU 利用率
    epochs: int = 100
    max_samples_per_epoch: int = None  # None = 使用所有樣本
    
    # 優化器
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # 學習率調度
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # 損失函數
    loss_type: str = "bce_dice"  # "adaptive" (GT=0: FocalBCE, GT>0: BCE+Dice), "bce_dice", "combined"
    dice_weight: float = 0.5
    focal_weight: float = 0.3
    tversky_weight: float = 0.2
    tversky_alpha: float = 0.7  # 控制 FN 權重（提高 recall）
    tversky_beta: float = 0.3   # 控制 FP 權重
    
    # Early Stopping
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.001
    
    # 資料增強
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # 梯度累積（記憶體不足時使用）
    gradient_accumulation_steps: int = 1
    
    # 混合精度訓練
    use_amp: bool = True
    
    # CV 設定
    use_cv: bool = False  # 預設使用單次訓練
    cv_fold: Optional[int] = None  # 指定訓練的 fold（None = 全部）
    num_folds: int = 5


@dataclass
class InferenceConfig:
    """推論相關配置"""
    # 閾值
    prediction_threshold: float = 0.5
    
    # 後處理
    use_postprocessing: bool = True
    min_volume_mm3: float = 14.0  # 最小結節體積（約 3mm 直徑球體）
    use_lung_mask: bool = True   # 只保留肺野內的預測
    use_morphology: bool = True  # 形態學平滑
    
    # 輸出格式
    save_nifti: bool = True
    save_json: bool = True  # 結節屬性 JSON（供 LLM 使用）


@dataclass
class Config:
    """完整配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # 實驗設定
    experiment_name: str = "unetpp_lndb"
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"
    
    def __post_init__(self):
        """初始化後處理（不自動創建目錄，由 get_default_config 處理）"""
        pass
    
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
            'hostname': platform.node()
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
            'device': self.device
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
        config.experiment_name = config_dict.get('experiment_name', 'unetpp_lndb')
        config.seed = config_dict.get('seed', 42)
        config.num_workers = config_dict.get('num_workers', 4)
        config.device = config_dict.get('device', 'cuda')
        
        return config


def get_default_config() -> Config:
    """獲取預設配置（自動設定絕對路徑）"""
    # 自動計算專案根目錄
    project_root = Path(__file__).parent.parent.parent  # train_unetpp -> segmentation -> project_root
    
    config = Config()
    
    # 設定絕對路徑（資料集在外接硬碟）
    config.data.data_dir = r"E:\lung_ct_lesion_dataset\LNDb"
    config.data.output_dir = str(project_root / "segmentation" / "result")
    config.data.cache_dir = str(project_root / "segmentation" / "cache" / "lndb_preprocessed")
    
    # 確保目錄存在
    Path(config.data.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.data.cache_dir).mkdir(parents=True, exist_ok=True)
    
    return config


if __name__ == "__main__":
    # 測試配置
    config = get_default_config()
    print(f"Data dir: {config.data.data_dir}")
    print(f"Encoder: {config.model.encoder_name}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Loss type: {config.training.loss_type}")
    
    # 保存測試
    config.save("test_config.json")
    print("\nConfig saved to test_config.json")
