#!/usr/bin/env python3
"""
FPR model utilities.

Supports two Stage-2 classifier backbones:
- 3D: ResNet-18 on cubic patch, input [B, 1, D, H, W]
- 2.5D: ResNet-18 on orthogonal slices, input [B, 3, H, W]
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


FPR_MODEL_TYPE_3D = "3d"
FPR_MODEL_TYPE_2P5D = "2.5d"


def normalize_fpr_model_type(model_type: str) -> str:
    value = str(model_type).strip().lower()
    if value in {"3d", "resnet3d"}:
        return FPR_MODEL_TYPE_3D
    if value in {"2.5d", "2p5d", "2d", "resnet2d"}:
        return FPR_MODEL_TYPE_2P5D
    raise ValueError(f"Unsupported fpr model type: {model_type}")


def build_fpr_model(model_type: str = FPR_MODEL_TYPE_3D, pretrained: bool = False):
    from monai.networks.nets import resnet18

    model_type = normalize_fpr_model_type(model_type)
    if model_type == FPR_MODEL_TYPE_3D:
        model = resnet18(
            pretrained=pretrained,
            spatial_dims=3,
            n_input_channels=1,
            num_classes=2,
        )
    else:
        model = resnet18(
            pretrained=pretrained,
            spatial_dims=2,
            n_input_channels=3,
            num_classes=2,
        )
    setattr(model, "fpr_model_type", model_type)
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


def patch_to_multiview_slices(patch_3d: np.ndarray) -> np.ndarray:
    """
    Convert a 3D patch [Y, X, Z] to orthogonal 2.5D slices [3, H, W].
    """
    if patch_3d.ndim != 3:
        raise ValueError(f"Expected 3D patch, got shape={patch_3d.shape}")
    cy = int(patch_3d.shape[0] // 2)
    cx = int(patch_3d.shape[1] // 2)
    cz = int(patch_3d.shape[2] // 2)

    axial = patch_3d[:, :, cz]      # Y-X
    coronal = patch_3d[:, cx, :]    # Y-Z
    sagittal = patch_3d[cy, :, :]   # X-Z

    views = np.stack([axial, coronal, sagittal], axis=0).astype(np.float32, copy=False)
    return views


def patch_to_model_input(patch_3d: np.ndarray, model_type: str) -> np.ndarray:
    model_type = normalize_fpr_model_type(model_type)
    patch = patch_3d.astype(np.float32, copy=False)
    if model_type == FPR_MODEL_TYPE_3D:
        return patch[np.newaxis, ...]  # [1, D, H, W]
    return patch_to_multiview_slices(patch)  # [3, H, W]


def batch_patches_to_model_input(patches_3d: np.ndarray, model_type: str) -> np.ndarray:
    """
    Args:
        patches_3d: [N, Y, X, Z]
    Returns:
        model input array:
          - 3D:   [N, 1, Y, X, Z]
          - 2.5D: [N, 3, Y, X]
    """
    model_type = normalize_fpr_model_type(model_type)
    if patches_3d.ndim != 4:
        raise ValueError(f"Expected [N, Y, X, Z], got shape={patches_3d.shape}")
    if model_type == FPR_MODEL_TYPE_3D:
        return patches_3d[:, np.newaxis, ...].astype(np.float32, copy=False)
    views = [patch_to_multiview_slices(p) for p in patches_3d]
    return np.stack(views, axis=0).astype(np.float32, copy=False)


def load_fpr_model(checkpoint_path: str, device: str = "cuda"):
    """
    Load a trained FPR classifier checkpoint.
    """
    checkpoint_obj = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_type = infer_fpr_model_type_from_checkpoint(checkpoint_obj)
    model = build_fpr_model(model_type=model_type)

    state_dict = _extract_state_dict_from_checkpoint_obj(checkpoint_obj)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    setattr(model, "fpr_model_type", model_type)
    return model


def get_model_type_from_model(model, default: str = FPR_MODEL_TYPE_3D) -> str:
    value: Optional[str] = getattr(model, "fpr_model_type", None)
    if value is None:
        return normalize_fpr_model_type(default)
    return normalize_fpr_model_type(value)
