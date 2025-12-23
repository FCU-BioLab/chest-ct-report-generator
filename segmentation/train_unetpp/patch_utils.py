#!/usr/bin/env python3
"""
4-Patch 計算與提取工具
======================

提供統一的 4-patch 切割邏輯：
1. 只保留 lung mask 區域
2. 基於 lung bbox 分成 2×2 quadrants
3. 每個 patch (224×224) 覆蓋對應的 1/4 lung mask
4. 超出邊界用零填補
"""

import numpy as np
from typing import List, Tuple
import torch


def get_lung_bbox(lung_mask: np.ndarray, margin: int = 0) -> Tuple[int, int, int, int]:
    """
    取得 lung mask 的 bounding box
    
    Args:
        lung_mask: (H, W) binary mask
        margin: 邊緣擴展像素
        
    Returns:
        (y_min, y_max, x_min, x_max)
    """
    lung_y, lung_x = np.where(lung_mask > 0)
    
    if len(lung_y) == 0:
        h, w = lung_mask.shape
        return 0, h, 0, w
    
    y_min = max(0, lung_y.min() - margin)
    y_max = min(lung_mask.shape[0], lung_y.max() + 1 + margin)
    x_min = max(0, lung_x.min() - margin)
    x_max = min(lung_mask.shape[1], lung_x.max() + 1 + margin)
    
    return y_min, y_max, x_min, x_max


def compute_4patch_positions(lung_mask: np.ndarray, patch_size: int = 224) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    計算 4-patch 的位置
    
    策略：
    1. 取得 lung mask 的 bounding box
    2. 將 bbox 分成 2×2 的 quadrants
    3. 每個 patch 覆蓋對應的 1/4 lung mask
    
    Args:
        lung_mask: (H, W) binary mask
        patch_size: patch 大小 (預設 224)
        
    Returns:
        list of ((y1, x1), (y2, x2)) - 每個 patch 的左上角和右下角座標
    """
    h, w = lung_mask.shape
    half_patch = patch_size // 2
    
    # 取得 lung bbox
    y_min, y_max, x_min, x_max = get_lung_bbox(lung_mask, margin=0)
    
    # 計算 lung bbox 的中點
    y_mid = (y_min + y_max) // 2
    x_mid = (x_min + x_max) // 2
    
    # 4 個 quadrants
    quadrants = [
        (y_min, y_mid, x_min, x_mid),  # Top-Left
        (y_min, y_mid, x_mid, x_max),  # Top-Right
        (y_mid, y_max, x_min, x_mid),  # Bottom-Left
        (y_mid, y_max, x_mid, x_max),  # Bottom-Right
    ]
    
    patches = []
    
    for (q_y1, q_y2, q_x1, q_x2) in quadrants:
        # 計算 quadrant 中心
        q_cy = (q_y1 + q_y2) // 2
        q_cx = (q_x1 + q_x2) // 2
        
        # Patch 左上角（以 quadrant 中心為 patch 中心）
        p_y1 = q_cy - half_patch
        p_x1 = q_cx - half_patch
        
        # 調整確保 quadrant 完全在 patch 內
        if q_y1 < p_y1:
            p_y1 = q_y1
        if q_y2 > p_y1 + patch_size:
            p_y1 = q_y2 - patch_size
        if q_x1 < p_x1:
            p_x1 = q_x1
        if q_x2 > p_x1 + patch_size:
            p_x1 = q_x2 - patch_size
        
        p_y2 = p_y1 + patch_size
        p_x2 = p_x1 + patch_size
        
        patches.append(((p_y1, p_x1), (p_y2, p_x2)))
    
    return patches


def extract_patch_with_lung_mask(
    image: np.ndarray,
    mask: np.ndarray,
    lung_mask: np.ndarray,
    patch_pos: Tuple[Tuple[int, int], Tuple[int, int]],
    patch_size: int = 224
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    提取 patch，超出邊界用零填充，lung mask 外設為零
    
    Args:
        image: (C, H, W) 或 (H, W) 原始影像
        mask: (H, W) 分割 mask
        lung_mask: (H, W) lung mask
        patch_pos: ((y1, x1), (y2, x2)) patch 座標
        patch_size: patch 大小
        
    Returns:
        patch_image: 提取的影像 patch
        patch_mask: 分割 mask patch
        patch_lung: lung mask patch
    """
    (y1, x1), (y2, x2) = patch_pos
    h, w = lung_mask.shape
    
    # 處理多通道
    if image.ndim == 3:
        n_channels = image.shape[0]
        patch_image = np.zeros((n_channels, patch_size, patch_size), dtype=image.dtype)
    else:
        patch_image = np.zeros((patch_size, patch_size), dtype=image.dtype)
    
    patch_mask = np.zeros((patch_size, patch_size), dtype=mask.dtype)
    patch_lung = np.zeros((patch_size, patch_size), dtype=lung_mask.dtype)
    
    # 計算有效區域
    src_y1 = max(0, y1)
    src_y2 = min(h, y2)
    src_x1 = max(0, x1)
    src_x2 = min(w, x2)
    
    dst_y1 = max(0, -y1)
    dst_x1 = max(0, -x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    
    # 複製有效區域
    if image.ndim == 3:
        patch_image[:, dst_y1:dst_y2, dst_x1:dst_x2] = image[:, src_y1:src_y2, src_x1:src_x2]
    else:
        patch_image[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    
    patch_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    patch_lung[dst_y1:dst_y2, dst_x1:dst_x2] = lung_mask[src_y1:src_y2, src_x1:src_x2]
    
    # 將 lung mask 外的區域設為零
    if image.ndim == 3:
        for c in range(n_channels):
            patch_image[c][patch_lung == 0] = 0
    else:
        patch_image[patch_lung == 0] = 0
    
    # mask 也只保留 lung 區域內的部分
    patch_mask[patch_lung == 0] = 0
    
    return patch_image, patch_mask, patch_lung


def stitch_4patches(
    patches: np.ndarray,
    positions: List[Tuple[int, int]],
    full_shape: Tuple[int, int],
    patch_size: int = 224
) -> np.ndarray:
    """
    將 4 個 patches 拼接回 full slice
    
    Args:
        patches: (4, 1, ps, ps) 預測結果（機率）
        positions: [(y1, x1), ...] 左上角座標
        full_shape: (H, W) 完整尺寸
        patch_size: patch 大小
        
    Returns:
        full_pred: (1, H, W) 拼接後的完整預測
    """
    h, w = full_shape
    full_pred = np.zeros((1, h, w), dtype=np.float32)
    
    for i, (y1, x1) in enumerate(positions):
        y1, x1 = int(y1), int(x1)
        y2, x2 = y1 + patch_size, x1 + patch_size
        
        # 計算有效區域
        src_y1 = max(0, -y1)
        src_x1 = max(0, -x1)
        dst_y1 = max(0, y1)
        dst_x1 = max(0, x1)
        dst_y2 = min(h, y2)
        dst_x2 = min(w, x2)
        
        ph = dst_y2 - dst_y1
        pw = dst_x2 - dst_x1
        
        if ph > 0 and pw > 0:
            # Max stitch
            full_pred[:, dst_y1:dst_y2, dst_x1:dst_x2] = np.maximum(
                full_pred[:, dst_y1:dst_y2, dst_x1:dst_x2],
                patches[i, :, src_y1:src_y1+ph, src_x1:src_x1+pw]
            )
    
    return full_pred


if __name__ == "__main__":
    # 簡單測試
    print("Testing patch_utils...")
    
    # 建立模擬 lung mask
    lung_mask = np.zeros((400, 450), dtype=np.float32)
    lung_mask[80:320, 70:380] = 1  # 模擬 lung 區域
    
    # 計算 4-patch 位置
    patches = compute_4patch_positions(lung_mask, patch_size=224)
    
    print(f"Lung bbox: {get_lung_bbox(lung_mask)}")
    for i, ((y1, x1), (y2, x2)) in enumerate(patches):
        print(f"Patch {i+1}: [{y1}:{y2}, {x1}:{x2}]")
    
    print("Done!")
