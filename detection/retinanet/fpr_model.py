#!/usr/bin/env python3
"""
FPR model utilities.

Supports Stage-2 classifier backbones with configurable patch views:
- 3D: ResNet on cubic patch, input [B, 1, D, H, W]
- 2.5D: ResNet on orthogonal slices, input [B, 3 * num_slices_per_view, H, W]
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


FPR_MODEL_TYPE_3D = "3d"
FPR_MODEL_TYPE_2P5D = "2.5d"
FPR_BACKBONE_RESNET18 = "resnet18"
FPR_BACKBONE_RESNET34 = "resnet34"


def normalize_fpr_model_type(model_type: str) -> str:
    value = str(model_type).strip().lower()
    if value in {"3d", "resnet3d"}:
        return FPR_MODEL_TYPE_3D
    if value in {"2.5d", "2p5d", "2d", "resnet2d"}:
        return FPR_MODEL_TYPE_2P5D
    raise ValueError(f"Unsupported fpr model type: {model_type}")


def normalize_fpr_backbone(backbone_name: str) -> str:
    value = str(backbone_name).strip().lower()
    if value in {"resnet18", "r18"}:
        return FPR_BACKBONE_RESNET18
    if value in {"resnet34", "r34"}:
        return FPR_BACKBONE_RESNET34
    raise ValueError(f"Unsupported fpr backbone: {backbone_name}")


def normalize_fpr_num_slices_per_view(num_slices_per_view: int) -> int:
    value = int(num_slices_per_view)
    if value <= 0:
        raise ValueError(f"num_slices_per_view must be >= 1, got {num_slices_per_view}")
    if value % 2 == 0:
        raise ValueError(f"num_slices_per_view must be odd so there is a center slice, got {num_slices_per_view}")
    return value


def _build_resnet(backbone_name: str, spatial_dims: int, n_input_channels: int, pretrained: bool = False):
    from monai.networks.nets import resnet18, resnet34

    backbone_name = normalize_fpr_backbone(backbone_name)
    builder = {
        FPR_BACKBONE_RESNET18: resnet18,
        FPR_BACKBONE_RESNET34: resnet34,
    }[backbone_name]
    return builder(
        pretrained=pretrained,
        spatial_dims=spatial_dims,
        n_input_channels=n_input_channels,
        num_classes=2,
    )


def build_fpr_model(
    model_type: str = FPR_MODEL_TYPE_3D,
    pretrained: bool = False,
    backbone_name: str = FPR_BACKBONE_RESNET18,
    num_slices_per_view: int = 1,
):
    backbone_name = normalize_fpr_backbone(backbone_name)
    num_slices_per_view = normalize_fpr_num_slices_per_view(num_slices_per_view)

    model_type = normalize_fpr_model_type(model_type)
    if model_type == FPR_MODEL_TYPE_3D:
        model = _build_resnet(
            backbone_name=backbone_name,
            spatial_dims=3,
            n_input_channels=1,
            pretrained=pretrained,
        )
    else:
        model = _build_resnet(
            backbone_name=backbone_name,
            spatial_dims=2,
            n_input_channels=3 * num_slices_per_view,
            pretrained=pretrained,
        )
    setattr(model, "fpr_model_type", model_type)
    setattr(model, "fpr_backbone_name", backbone_name)
    setattr(model, "fpr_num_slices_per_view", num_slices_per_view)
    return model


def _extract_state_dict_from_checkpoint_obj(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        return checkpoint_obj["model_state_dict"]
    return checkpoint_obj


def infer_fpr_model_type_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    conv1_weight = state_dict.get("conv1.weight")
    if conv1_weight is None:
        return FPR_MODEL_TYPE_3D
    if conv1_weight.ndim == 5:
        return FPR_MODEL_TYPE_3D
    if conv1_weight.ndim == 4:
        return FPR_MODEL_TYPE_2P5D
    return FPR_MODEL_TYPE_3D


def infer_fpr_model_type_from_checkpoint(checkpoint_obj, default: str = FPR_MODEL_TYPE_3D) -> str:
    if isinstance(checkpoint_obj, dict):
        model_type = checkpoint_obj.get("model_type")
        if model_type is not None:
            return normalize_fpr_model_type(model_type)
    state_dict = _extract_state_dict_from_checkpoint_obj(checkpoint_obj)
    if isinstance(state_dict, dict):
        return infer_fpr_model_type_from_state_dict(state_dict)
    return normalize_fpr_model_type(default)


def infer_fpr_backbone_from_checkpoint(checkpoint_obj, default: str = FPR_BACKBONE_RESNET18) -> str:
    if isinstance(checkpoint_obj, dict):
        backbone_name = checkpoint_obj.get("backbone_name")
        if backbone_name is not None:
            return normalize_fpr_backbone(backbone_name)
    return normalize_fpr_backbone(default)


def infer_fpr_num_slices_per_view_from_checkpoint(checkpoint_obj, default: int = 1) -> int:
    if isinstance(checkpoint_obj, dict):
        num_slices_per_view = checkpoint_obj.get("num_slices_per_view")
        if num_slices_per_view is not None:
            return normalize_fpr_num_slices_per_view(num_slices_per_view)
    return normalize_fpr_num_slices_per_view(default)


def _slice_indices(center: int, size: int, num_slices_per_view: int) -> list[int]:
    half = normalize_fpr_num_slices_per_view(num_slices_per_view) // 2
    return [min(max(center + offset, 0), size - 1) for offset in range(-half, half + 1)]


def patch_to_multiview_slices(patch_3d: np.ndarray, num_slices_per_view: int = 1) -> np.ndarray:
    """
    Convert a 3D patch [Y, X, Z] to orthogonal 2.5D slices
    [3 * num_slices_per_view, H, W].
    """
    if patch_3d.ndim != 3:
        raise ValueError(f"Expected 3D patch, got shape={patch_3d.shape}")
    num_slices_per_view = normalize_fpr_num_slices_per_view(num_slices_per_view)
    cy = int(patch_3d.shape[0] // 2)
    cx = int(patch_3d.shape[1] // 2)
    cz = int(patch_3d.shape[2] // 2)

    axial = [patch_3d[:, :, idx] for idx in _slice_indices(cz, patch_3d.shape[2], num_slices_per_view)]
    coronal = [patch_3d[:, idx, :] for idx in _slice_indices(cx, patch_3d.shape[1], num_slices_per_view)]
    sagittal = [patch_3d[idx, :, :] for idx in _slice_indices(cy, patch_3d.shape[0], num_slices_per_view)]

    views = np.stack(axial + coronal + sagittal, axis=0).astype(np.float32, copy=False)
    return views


def patch_to_model_input(patch_3d: np.ndarray, model_type: str, num_slices_per_view: int = 1) -> np.ndarray:
    model_type = normalize_fpr_model_type(model_type)
    patch = patch_3d.astype(np.float32, copy=False)
    if model_type == FPR_MODEL_TYPE_3D:
        return patch[np.newaxis, ...]  # [1, D, H, W]
    return patch_to_multiview_slices(patch, num_slices_per_view=num_slices_per_view)  # [3 * num_slices, H, W]


def batch_patches_to_model_input(patches_3d: np.ndarray, model_type: str, num_slices_per_view: int = 1) -> np.ndarray:
    """
    Args:
        patches_3d: [N, Y, X, Z]
    Returns:
        model input array:
          - 3D:   [N, 1, Y, X, Z]
          - 2.5D: [N, 3 * num_slices_per_view, Y, X]
    """
    model_type = normalize_fpr_model_type(model_type)
    if patches_3d.ndim != 4:
        raise ValueError(f"Expected [N, Y, X, Z], got shape={patches_3d.shape}")
    if model_type == FPR_MODEL_TYPE_3D:
        return patches_3d[:, np.newaxis, ...].astype(np.float32, copy=False)
    views = [patch_to_multiview_slices(p, num_slices_per_view=num_slices_per_view) for p in patches_3d]
    return np.stack(views, axis=0).astype(np.float32, copy=False)


def load_fpr_model(checkpoint_path: str, device: str = "cuda"):
    """
    Load a trained FPR classifier checkpoint.
    """
    checkpoint_obj = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_type = infer_fpr_model_type_from_checkpoint(checkpoint_obj)
    backbone_name = infer_fpr_backbone_from_checkpoint(checkpoint_obj)
    num_slices_per_view = infer_fpr_num_slices_per_view_from_checkpoint(checkpoint_obj)
    model = build_fpr_model(
        model_type=model_type,
        backbone_name=backbone_name,
        num_slices_per_view=num_slices_per_view,
    )

    state_dict = _extract_state_dict_from_checkpoint_obj(checkpoint_obj)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    setattr(model, "fpr_model_type", model_type)
    setattr(model, "fpr_backbone_name", backbone_name)
    setattr(model, "fpr_num_slices_per_view", num_slices_per_view)
    return model


def get_model_type_from_model(model, default: str = FPR_MODEL_TYPE_3D) -> str:
    value: Optional[str] = getattr(model, "fpr_model_type", None)
    if value is None:
        return normalize_fpr_model_type(default)
    return normalize_fpr_model_type(value)


def get_model_backbone_from_model(model, default: str = FPR_BACKBONE_RESNET18) -> str:
    value: Optional[str] = getattr(model, "fpr_backbone_name", None)
    if value is None:
        return normalize_fpr_backbone(default)
    return normalize_fpr_backbone(value)


def get_model_num_slices_per_view(model, default: int = 1) -> int:
    value = getattr(model, "fpr_num_slices_per_view", None)
    if value is None:
        return normalize_fpr_num_slices_per_view(default)
    return normalize_fpr_num_slices_per_view(value)
