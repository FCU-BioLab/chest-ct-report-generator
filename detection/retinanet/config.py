#!/usr/bin/env python3
"""
RetinaNet configuration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


def _default_spacing() -> List[float]:
    # Keep bundle-compatible resampling so anchor sizes remain meaningful.
    return [0.703125, 0.703125, 1.25]


def _default_anchor_shapes() -> List[List[int]]:
    # MONAI bundle defaults tuned for small lung nodules.
    return [[6, 8, 4], [8, 6, 5], [10, 10, 6]]


def _default_feature_map_scales() -> List[List[int]]:
    # Per-axis FPN strides must stay aligned with the anisotropic Z spacing.
    return [[4, 4, 2], [8, 8, 4], [16, 16, 8]]


def _default_patch_size() -> List[int]:
    return [192, 192, 80]


def _default_val_patch_size() -> List[int]:
    # Larger ROI gives inference more context than the train crop.
    return [512, 512, 192]


@dataclass
class RetinaNetConfig:
    """MONAI 3D RetinaNet training/inference configuration."""

    data_path: str = "detection/manifests/dataset_lndb.json"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42

    spacing: List[float] = field(default_factory=_default_spacing)
    hu_min: float = -1024.0
    hu_max: float = 300.0

    spatial_dims: int = 3
    n_input_channels: int = 1
    num_classes: int = 1

    base_anchor_shapes: List[List[int]] = field(default_factory=_default_anchor_shapes)
    feature_map_scales: List[List[int]] = field(default_factory=_default_feature_map_scales)
    returned_layers: List[int] = field(default_factory=lambda: [1, 2])
    conv1_t_stride: List[int] = field(default_factory=lambda: [2, 2, 1])
    size_divisible: List[int] = field(default_factory=lambda: [16, 16, 8])

    epochs: int = 300
    batch_size: int = 2
    lr: float = 0.001
    weight_decay: float = 3e-5
    w_cls: float = 1.0
    warmup_epochs: int = 10
    lr_step_size: int = 160
    lr_gamma: float = 0.1
    val_interval: int = 5
    # Early stopping (disabled when patience <= 0).
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0

    patch_size: List[int] = field(default_factory=_default_patch_size)
    val_patch_size: List[int] = field(default_factory=_default_val_patch_size)
    max_boxes_for_crop: int = 8

    atss_num_candidates: int = 4
    hn_batch_size_per_image: int = 64
    hn_positive_fraction: float = 0.3
    hn_pool_size: int = 20
    hn_min_neg: int = 16

    proposal_score_thresh: float = 0.02
    test_score_thresh: float = 0.1
    nms_thresh: float = 0.22
    topk_candidates_per_level: int = 1000
    detections_per_img: int = 300

    iou_list: List[float] = field(default_factory=lambda: [0.1])

    amp: bool = True
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    output_dir: Optional[str] = None

    pretrained_weights: Optional[str] = "bundles/lung_nodule_ct_detection/models/model.pt"
    cache_dataset: bool = True

    def __post_init__(self):
        if self.output_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"detection/results/retinanet_{ts}"
        self.validate()

    def validate(self) -> None:
        if len(self.spacing) != 3:
            raise ValueError("spacing must have 3 values: [x, y, z]")
        if len(self.patch_size) != 3 or len(self.val_patch_size) != 3:
            raise ValueError("patch_size and val_patch_size must have 3 values")
        if len(self.size_divisible) != 3:
            raise ValueError("size_divisible must have 3 values")
        if len(self.base_anchor_shapes) == 0:
            raise ValueError("base_anchor_shapes must not be empty")
        if len(self.feature_map_scales) == 0:
            raise ValueError("feature_map_scales must not be empty")
        if len(self.base_anchor_shapes) != len(self.feature_map_scales):
            raise ValueError("base_anchor_shapes and feature_map_scales must have the same length")
        for shape in self.base_anchor_shapes:
            if len(shape) != 3:
                raise ValueError("each base_anchor_shape must contain 3 values")
        for scale in self.feature_map_scales:
            if len(scale) != 3:
                raise ValueError("each feature_map_scale must contain 3 values")
        for patch, divisible in zip(self.patch_size, self.size_divisible):
            if patch % divisible != 0:
                raise ValueError("patch_size must be divisible by size_divisible on each axis")
        for patch, divisible in zip(self.val_patch_size, self.size_divisible):
            if patch % divisible != 0:
                raise ValueError("val_patch_size must be divisible by size_divisible on each axis")
        if self.max_boxes_for_crop <= 0:
            raise ValueError("max_boxes_for_crop must be > 0")
        if self.early_stop_patience < 0:
            raise ValueError("early_stop_patience must be >= 0")
        if self.early_stop_min_delta < 0:
            raise ValueError("early_stop_min_delta must be >= 0")
