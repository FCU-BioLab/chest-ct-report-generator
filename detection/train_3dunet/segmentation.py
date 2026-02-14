#!/usr/bin/env python3
"""
Segmentation Utilities
======================

Shared segmentation logic for 3D U-Net training and preprocessing.
GPU acceleration via PyTorch MaxPool3d (optimized dilation/erosion).

Classic lung segmentation approach:
  1. Threshold CT to find all air (< -400 HU equivalent)
  2. Remove external air using clear_border (remove components touching image border)
  3. Remaining air = internal air (lungs + airways)
  4. Morphological closing to smooth edges
  5. Dilation for margin
"""

import numpy as np
from scipy import ndimage as cpu_ndimage
from skimage import measure, morphology, segmentation
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

# --- GPU Support via PyTorch ---
try:
    import torch
    import torch.nn.functional as F
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False


def generate_lung_mask(volume: np.ndarray, threshold: float = 0.45, dilate_mm: float = 5.0) -> np.ndarray:
    """
    從 CT volume 生成肺部遮罩

    Args:
        volume: (D, H, W) normalized to [0, 1] using HU range [-1024, 400]
        threshold: voxels below this value are considered air/lung
        dilate_mm: dilation radius for margin

    Returns:
        lung_mask: (D, H, W) binary mask (numpy array)
    """
    if HAS_GPU:
        return _generate_lung_mask_gpu(volume, threshold, dilate_mm)
    else:
        return _generate_lung_mask_cpu(volume, threshold, dilate_mm)


def _gpu_dilate_3d(x: torch.Tensor, radius: int) -> torch.Tensor:
    """GPU 3D dilation using MaxPool3d. Input: (1,1,D,H,W) float tensor."""
    kernel_size = 2 * radius + 1
    return F.max_pool3d(x, kernel_size=kernel_size, stride=1, padding=radius)


def _gpu_erode_3d(x: torch.Tensor, radius: int) -> torch.Tensor:
    """GPU 3D erosion using -MaxPool3d(-x). Input: (1,1,D,H,W) float tensor."""
    kernel_size = 2 * radius + 1
    return -F.max_pool3d(-x, kernel_size=kernel_size, stride=1, padding=radius)


def _segment_lungs_core(volume: np.ndarray, threshold: float) -> np.ndarray:
    """
    Core lung segmentation logic (CPU).
    
    Classic approach:
    1. Threshold → air_mask (lung + external air)
    2. Per-slice: clear_border to remove external air
    3. Remaining = internal air (lungs)
    4. 3D connected components → keep top 2 largest (left + right lung)
    5. Fill holes to include airways within lungs
    """
    # 1. Threshold: everything below threshold is "air"
    air_mask = (volume < threshold).astype(np.uint8)
    
    # 2. Per-slice: remove external air (connected to border)
    internal_air = np.zeros_like(air_mask)
    for z in range(air_mask.shape[0]):
        # clear_border removes regions touching the image border
        # What remains is internal air (lungs, airways)
        cleared = segmentation.clear_border(air_mask[z])
        internal_air[z] = cleared
    
    # 3. Connected components in 3D — keep top 2 (left + right lung)
    labels = measure.label(internal_air)
    
    if labels.max() > 0:
        label_counts = np.bincount(labels.ravel())
        label_counts[0] = 0  # ignore background
        
        lung_mask = np.zeros_like(internal_air)
        # Get top 2 largest components
        top_labels = np.argsort(label_counts)[::-1][:2]
        for lbl in top_labels:
            if label_counts[lbl] > 500:  # minimum size filter
                lung_mask[labels == lbl] = 1
    else:
        lung_mask = internal_air
    
    # 4. Fill holes per-slice (include airways within lung boundary)
    for z in range(lung_mask.shape[0]):
        lung_mask[z] = cpu_ndimage.binary_fill_holes(lung_mask[z]).astype(np.uint8)
    
    # 5. Morphological closing to smooth edges
    struct = morphology.ball(2)
    lung_mask = morphology.binary_closing(lung_mask, struct).astype(np.uint8)
    
    return lung_mask


@torch.no_grad()
def _generate_lung_mask_gpu(volume: np.ndarray, threshold: float, dilate_mm: float) -> np.ndarray:
    """GPU-accelerated lung mask: core segmentation on CPU, dilation on GPU."""
    # Core segmentation (CPU — uses clear_border which has no GPU equivalent)
    lung_mask = _segment_lungs_core(volume, threshold)
    
    # Dilation on GPU (the expensive 3D morphological operation)
    if dilate_mm > 0:
        dilate_radius = max(2, int(dilate_mm))
        device = torch.device('cuda')
        mask_t = torch.from_numpy(lung_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        mask_t = _gpu_dilate_3d(mask_t, radius=dilate_radius)
        lung_mask = (mask_t[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
    
    return lung_mask


def _generate_lung_mask_cpu(volume: np.ndarray, threshold: float, dilate_mm: float) -> np.ndarray:
    """CPU fallback lung mask generation."""
    lung_mask = _segment_lungs_core(volume, threshold)
    
    # Dilation
    if dilate_mm > 0:
        dilate_radius = max(2, int(dilate_mm))
        struct_dilate = morphology.ball(dilate_radius)
        lung_mask = morphology.binary_dilation(lung_mask, struct_dilate).astype(np.uint8)
    
    return lung_mask


def compute_lung_bbox(lung_mask: np.ndarray, margin: int = 10) -> Tuple[int, int, int, int]:
    """
    計算肺部遮罩的 2D 邊界框 (min_x, min_y, max_x, max_y)
    """
    proj = np.max(lung_mask, axis=0)  # (H, W)

    rows = np.any(proj, axis=1)
    cols = np.any(proj, axis=0)

    if not np.any(rows) or not np.any(cols):
        H, W = lung_mask.shape[1], lung_mask.shape[2]
        return 0, 0, W, H

    min_y, max_y = np.where(rows)[0][[0, -1]]
    min_x, max_x = np.where(cols)[0][[0, -1]]

    H, W = lung_mask.shape[1], lung_mask.shape[2]
    min_y = max(0, min_y - margin)
    max_y = min(H, max_y + margin)
    min_x = max(0, min_x - margin)
    max_x = min(W, max_x + margin)

    return min_x, min_y, max_x, max_y
