#!/usr/bin/env python3
"""
RetinaNet Detection Config
==========================

Dataclass-based configuration for MONAI 3D RetinaNet nodule detection.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class RetinaNetConfig:
    """Configuration for MONAI 3D RetinaNet detection."""

    # ─── Data ───────────────────────────────────────────────
    npz_dir: str = "cache/lndb_volume_npz_agr1"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42

    # ─── Model ──────────────────────────────────────────────
    spatial_dims: int = 3
    n_input_channels: int = 1
    num_classes: int = 1  # nodule only

    # Anchor shapes in voxels (for the highest-resolution FPN level)
    # Each shape is [W, H, D] — designed for small nodules (3-30mm)
    base_anchor_shapes: List[List[int]] = field(
        default_factory=lambda: [[6, 8, 4], [8, 6, 4], [10, 10, 6]]
    )

    # FPN returned layers from ResNet (lower = higher resolution)
    returned_layers: List[int] = field(default_factory=lambda: [1, 2])

    # ResNet conv1 stride (Z-axis often has larger spacing → smaller stride)
    conv1_t_stride: List[int] = field(default_factory=lambda: [2, 2, 1])

    # ─── Training ───────────────────────────────────────────
    epochs: int = 300
    batch_size: int = 2
    lr: float = 0.01
    weight_decay: float = 3e-5
    w_cls: float = 1.0  # weight: cls_loss vs box_reg_loss
    warmup_epochs: int = 10
    lr_step_size: int = 150
    lr_gamma: float = 0.1
    val_interval: int = 5  # validate every N epochs

    # Patch size for random cropping during training [H, W, D]
    patch_size: List[int] = field(default_factory=lambda: [192, 192, 80])
    # Patch size for sliding window inference during validation
    val_patch_size: List[int] = field(default_factory=lambda: [512, 512, 208])

    # ATSS Matcher
    atss_num_candidates: int = 4

    # Hard Negative Sampler
    hn_batch_size_per_image: int = 64
    hn_positive_fraction: float = 0.3
    hn_pool_size: int = 20
    hn_min_neg: int = 16

    # ─── Detection (Inference) ──────────────────────────────
    score_thresh: float = 0.02
    nms_thresh: float = 0.22
    topk_candidates_per_level: int = 1000
    detections_per_img: int = 100

    # ─── System ─────────────────────────────────────────────
    amp: bool = True
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    output_dir: Optional[str] = None

    def __post_init__(self):
        if self.output_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"detection/video_result/retinanet_{ts}"
