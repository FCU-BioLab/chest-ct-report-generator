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
    patch_size: int = 288  # Adjusted to 288 based on user feedback (balance context vs lesion size)

    
    # 2.5D 設定
    use_2_5d: bool = True  # 啟用 2.5D input (z-2, z-1, z, z+1, z+2)
    slice_distance_mm: float = 1.0  # 固定毫米距離取鄰近切片
    
    # Val/Test restrictive cropping
    val_num_patches: int = 4  # 每個 slice 抽取的 patch 數
    
    # 採樣比例
    positive_ratio: float = 0.5    # 正樣本（結節中心）比例
    hard_negative_ratio: float = 0.3  # 肺野內無結節區域 (Increased for hard negative mining)
    random_negative_ratio: float = 0.2  # 完全隨機
    
    # 資料分割（單次訓練模式）7:2:1
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    
    # 遮罩共識方式
    consensus_method: str = "soft"  # "soft", "intersection", "union", "rad1"
    
    # 快取 - 使用預處理好的 4-patch 資料
    # Set to FALSE to enable on-the-fly cropping (needed for 352x352 since cache is 256)
    cache_preprocessed: bool = False  
    cache_dir: str = "segmentation/cache/lndb_patches"  # 4-patch 預處理資料目錄


@dataclass
class ModelConfig:
    """模型相關配置"""
    # UNet++ 架構
    encoder_name: str = "resnet34"  # Balanced choice: Faster GPU training than EfficientNet, better features
    encoder_weights: str = "imagenet"  # 預訓練權重
    in_channels: int = 5  # 2.5D: z-2, z-1, z, z+1, z+2 疊成 5 channel
    num_classes: int = 1  # 二元分割
    use_2d: bool = True  # 使用純 2D 模式（但輸入是 2.5D）
    
    # 深度監督
    deep_supervision: bool = True


@dataclass
class TrainingConfig:
    """訓練相關配置"""
    # 基本參數
    batch_size: int = 24  # ResNet34 is lighter than ResNeXt, can increase batch size
    epochs: int = 200     # 參照 MedSAM2 成功運行配置
    max_samples_per_epoch: int = None  # None = 使用所有樣本
    
    # 優化器
    optimizer: str = "adamw"
    learning_rate: float = 1e-4  # UNet++ 需要比 Finetune MedSAM2 更高的 LR (5e-6 -> 1e-4)
    weight_decay: float = 1e-4   # Increased regularization (was 1e-5)
    
    # 學習率調度
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    warmup_epochs: int = 5
    min_lr: float = 1e-6  # 提高最低 LR，避免下降太快
    
    # 損失函數
    loss_type: str = "tversky"  # Switch to Tversky to boost Recall (penalize FN more)
    dice_weight: float = 0.7  # 增加 Dice 權重，改善 under-segmentation
    focal_weight: float = 0.3
    tversky_weight: float = 0.0
    tversky_alpha: float = 0.7  # 控制 FN 權重（提高 recall）
    tversky_beta: float = 0.3   # 控制 FP 權重
    
    # Early Stopping
    early_stopping_patience: int = 30  # Increased patience
    early_stopping_min_delta: float = 0.001
    
    # 資料增強
    use_augmentation: bool = True  # Enable augmentation to fix overfitting
    use_strong_augmentation: bool = True  # Enable stronger medically safe augmentations
    augmentation_prob: float = 0.8  # Increased probability
    
    # 梯度累積（記憶體不足時使用）
    gradient_accumulation_steps: int = 4
    grad_clip: float = 1.0  # 梯度裁切閾值
    
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
    prediction_threshold: float = 0.75
    
    # === Metrics Calculator ===
    target_threshold: float = 0.5  # GT 二值化閾值
    min_area_px: int = 5  # 2D lesion-wise 過濾碎片（像素）
    
    # === Volume-level 後處理（完整推論時使用） ===
    use_postprocessing: bool = True
    
    # Step 2: Lung mask 限制
    use_lung_mask: bool = True
    lung_mask_dilate_mm: float = 0.0  # 擴張肺遮罩 (mm)
    
    # Step 3: Connected component filtering
    min_volume_mm3: float = 14.0  # 最小結節體積（約 3mm 直徑球體）
    max_volume_mm3: float = 113097.0  # 最大結節體積（約 60mm 直徑球體）
    
    # Step 4: Morphology
    closing_radius_mm: float = 1.0  # 閉運算半徑 (mm)
    fill_holes: bool = True
    fill_holes_3d: bool = False  # 3D 填充較慢
    
    # 額外過濾
    remove_edge_nodules: bool = False
    min_solidity: float = 0.0
    
    # === Patch-level 後處理（訓練驗證時使用） ===
    # patch 內結節可能被截斷，使用更保守的參數
    patch_min_size_mm3: float = 5.0
    patch_max_size_mm3: float = 50000.0
    patch_closing_radius_mm: float = 0.5
    patch_fill_holes: bool = True
    patch_spacing_mm: float = 1.0  # patch 級別假設的 spacing
    
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
    seed: int = 27
    num_workers: int = 2
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
    # config.data.cache_dir = str(project_root / "segmentation" / "cache" / "lndb_preprocessed")
    
    # 確保目錄存在
    Path(config.data.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Explicitly set absolute path for cache
    config.data.cache_dir = str(project_root / "segmentation" / "cache" / "lndb_patches")
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
