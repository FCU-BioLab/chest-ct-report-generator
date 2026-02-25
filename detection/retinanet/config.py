#!/usr/bin/env python3
"""
RetinaNet 設定檔 (Config)
========================

用於 MONAI 3D RetinaNet 肺結節偵測的 Dataclass 設定。
對齊 bundles/lung_nodule_ct_detection/configs/train_luna16.json。
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class RetinaNetConfig:
    """MONAI 3D RetinaNet 偵測模型設定。"""

    # ─── 資料設定 ─────────────────────────────────────────────
    data_path: str = "cache/lndb_volume_npz_agr1"  # 支援 JSON 檔案路徑
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42

    # 影像預處理
    spacing: List[float] = field(default_factory=lambda: [0.703125, 0.703125, 1.25])
    hu_min: float = -1024.0
    hu_max: float = 300.0

    # ─── 模型設定 ─────────────────────────────────────────────
    spatial_dims: int = 3
    n_input_channels: int = 1
    num_classes: int = 1  # 僅偵測結節 (Nodule)

    # 錨點 (Anchor) 形狀設定，單位為體素 (Voxel)
    # 這些尺寸針對小結節 (約 3-30mm)
    base_anchor_shapes: List[List[int]] = field(
        default_factory=lambda: [[6, 8, 4], [8, 6, 5], [10, 10, 6]]
    )

    # 錨點 feature map 縮放因子 — 每個 feature level 的 scalar 倍率
    # 對應 FPN 的 3 個 feature map 層級
    # (base_anchor_shapes 中已包含 per-axis 的非等向性)
    feature_map_scales: List[int] = field(
        default_factory=lambda: [2, 4, 8]
    )

    # 從 ResNet 返回的 FPN 層級 (數值越小代表解析度越高)
    returned_layers: List[int] = field(default_factory=lambda: [1, 2])

    # ResNet conv1 步長 (Z 軸通常間距較大，因此步長較小)
    conv1_t_stride: List[int] = field(default_factory=lambda: [2, 2, 1])

    # size_divisible — 固定值，對齊 Bundle
    size_divisible: List[int] = field(default_factory=lambda: [16, 16, 8])

    # ─── 訓練設定 ─────────────────────────────────────────────
    epochs: int = 300
    batch_size: int = 2
    lr: float = 0.001
    weight_decay: float = 3e-5
    w_cls: float = 1.0  # Loss 權重: 分類損失 vs 邊框回歸損失
    warmup_epochs: int = 10
    lr_step_size: int = 160
    lr_gamma: float = 0.1
    val_interval: int = 5  # 每 N 個 Epoch 進行一次驗證

    # 訓練時的隨機裁切 (Random Crop) 大小 [H, W, D]
    patch_size: List[int] = field(default_factory=lambda: [192, 192, 80])
    # 驗證時使用的滑動視窗 (Sliding Window) 大小
    val_patch_size: List[int] = field(default_factory=lambda: [192, 192, 80])

    # ATSS 匹配器參數
    atss_num_candidates: int = 4

    # 硬負樣本採樣器 (Hard Negative Sampler) 參數
    hn_batch_size_per_image: int = 64
    hn_positive_fraction: float = 0.3
    hn_pool_size: int = 20
    hn_min_neg: int = 16

    # ─── 偵測 (推論) 設定 ─────────────────────────────────────
    score_thresh: float = 0.02
    nms_thresh: float = 0.22
    topk_candidates_per_level: int = 1000
    detections_per_img: int = 300

    # ─── 評估設定 ─────────────────────────────────────────────
    iou_list: List[float] = field(default_factory=lambda: [0.1])

    # ─── 系統設定 ─────────────────────────────────────────────
    amp: bool = True       # 是否啟用混合精度訓練 (Automatic Mixed Precision)
    device: str = "cuda"   # 運算裝置
    num_workers: int = 4   # DataLoader 工作執行緒數
    seed: int = 42         # 全域隨機種子
    output_dir: Optional[str] = None  # 輸出目錄

    # 預訓練權重 (MONAI Bundle model.pt)
    pretrained_weights: Optional[str] = "bundles/lung_nodule_ct_detection/models/model.pt"

    # 使用 CacheDataset 快取 deterministic transforms
    cache_dataset: bool = True

    def __post_init__(self):
        # 若未指定輸出目錄，則自動依據時間戳記生成
        if self.output_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"detection/results/retinanet_{ts}"
