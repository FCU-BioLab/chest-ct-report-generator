#!/usr/bin/env python3
"""
資料集模組
提供胸部CT腫瘤資料集的載入與處理
支援多種分割方法生成 Ground Truth

此模組共享自 finetune_medsam2，確保資料處理一致性
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import pandas as pd
from skimage import measure
from scipy import ndimage
from tqdm import tqdm
import random


def normalize_patient_id(patient_id, to_format: str = "int"):
    """
    [B] Normalize patient ID to consistent format.
    
    Ensures patient IDs are always in a consistent format throughout the codebase.
    Supports conversion between integer IDs (e.g., 1, 42) and LNDb format (e.g., "LNDb-0001").
    
    Args:
        patient_id: Input ID (int, "1", "LNDb-0001", etc.)
        to_format: Target format:
            - "int": Return as integer
            - "lndb": Return as "LNDb-XXXX" string
    
    Returns:
        Normalized patient ID in the requested format
        
    Raises:
        ValueError: If patient_id cannot be parsed
        TypeError: If patient_id is not int or str
        
    Examples:
        >>> normalize_patient_id("LNDb-0001", "int")
        1
        >>> normalize_patient_id(42, "lndb")
        'LNDb-0042'
        >>> normalize_patient_id("123", "lndb")
        'LNDb-0123'
    """
    # Extract numeric part
    if isinstance(patient_id, int):
        numeric_id = patient_id
    elif isinstance(patient_id, str):
        if patient_id.startswith("LNDb-"):
            numeric_id = int(patient_id.replace("LNDb-", ""))
        elif patient_id.isdigit():
            numeric_id = int(patient_id)
        else:
            raise ValueError(f"Cannot parse patient ID: '{patient_id}'")
    else:
        raise TypeError(f"Unsupported patient_id type: {type(patient_id)}")
    
    if to_format == "int":
        return numeric_id
    elif to_format == "lndb":
        return f"LNDb-{numeric_id:04d}"
    else:
        raise ValueError(f"Unknown format: {to_format}")


class NoduleSegmenter:
    """
    結節分割器
    
    提供多種分割方法生成 Ground Truth 遮罩
    
    Args:
        method: 分割方法 ('sphere', 'threshold', 'region_growing', 'watershed', 'adaptive')
    """
    
    def __init__(self, method: str = 'region_growing'):
        self.method = method
        self.logger = logging.getLogger(__name__)
    
    def generate_3d_mask(
        self, 
        volume: np.ndarray, 
        center_voxel: np.ndarray, 
        radius_voxel: np.ndarray,
        spacing: np.ndarray
    ) -> np.ndarray:
        """
        生成結節的 3D 分割遮罩
        
        Parameters:
        -----------
        volume : ndarray
            CT 體積資料 (Z, Y, X)
        center_voxel : ndarray
            結節中心的體素座標 (x, y, z)
        radius_voxel : ndarray
            結節在各軸的體素半徑 (rx, ry, rz)
        spacing : ndarray
            像素間距 (sx, sy, sz)
        
        Returns:
        --------
        mask : ndarray
            3D 分割遮罩，與 volume 相同形狀
        """
        vx, vy, vz = int(center_voxel[0]), int(center_voxel[1]), int(center_voxel[2])
        rx, ry, rz = radius_voxel
        
        # 擴展搜索區域
        margin = 1.5
        search_rx = int(rx * margin) + 3
        search_ry = int(ry * margin) + 3
        search_rz = int(rz * margin) + 3
        
        # 確保邊界在體積範圍內
        z_min = max(0, vz - search_rz)
        z_max = min(volume.shape[0], vz + search_rz + 1)
        y_min = max(0, vy - search_ry)
        y_max = min(volume.shape[1], vy + search_ry + 1)
        x_min = max(0, vx - search_rx)
        x_max = min(volume.shape[2], vx + search_rx + 1)
        
        # 提取局部區域
        local_volume = volume[z_min:z_max, y_min:y_max, x_min:x_max].copy()
        
        # 局部座標中的結節中心
        local_center = np.array([vz - z_min, vy - y_min, vx - x_min])
        local_radius = np.array([rz, ry, rx])  # Z, Y, X
        
        if self.method == 'sphere':
            local_mask = self._create_ellipsoid_mask(local_volume.shape, local_center, local_radius)
        elif self.method == 'threshold':
            local_mask = self._threshold_segmentation(local_volume, local_center, local_radius)
        elif self.method == 'region_growing':
            local_mask = self._region_growing_segmentation(local_volume, local_center, local_radius)
        else:
            local_mask = self._create_ellipsoid_mask(local_volume.shape, local_center, local_radius)
        
        # 將局部遮罩放回全局遮罩
        full_mask = np.zeros(volume.shape, dtype=bool)
        full_mask[z_min:z_max, y_min:y_max, x_min:x_max] = local_mask
        
        return full_mask
    
    def _create_ellipsoid_mask(self, shape, center, radius):
        """創建橢球體遮罩"""
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        
        rz = max(radius[0], 0.5)
        ry = max(radius[1], 0.5)
        rx = max(radius[2], 0.5)
        
        distance = ((z - center[0]) / rz) ** 2 + \
                   ((y - center[1]) / ry) ** 2 + \
                   ((x - center[2]) / rx) ** 2
        
        return distance <= 1.0
    
    def _threshold_segmentation(self, volume, center, radius):
        """基於閾值的分割"""
        # 獲取種子點附近的 HU 值
        z, y, x = int(center[0]), int(center[1]), int(center[2])
        seed_value = volume[z, y, x]
        
        # 設定閾值範圍
        lower = seed_value - 150
        upper = seed_value + 150
        
        # 創建初始遮罩
        mask = (volume >= lower) & (volume <= upper)
        
        # 限制在橢球體範圍內
        ellipsoid = self._create_ellipsoid_mask(volume.shape, center, radius * 1.5)
        mask = mask & ellipsoid
        
        # 只保留包含種子點的連通區域
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            seed_label = labeled[z, y, x]
            if seed_label > 0:
                mask = labeled == seed_label
            else:
                mask = self._create_ellipsoid_mask(volume.shape, center, radius)
        
        return mask
    
    def _region_growing_segmentation(self, volume, center, radius):
        """區域生長分割"""
        z, y, x = int(center[0]), int(center[1]), int(center[2])
        seed_value = volume[z, y, x]
        
        # 閾值範圍
        tolerance = 100
        lower = seed_value - tolerance
        upper = seed_value + tolerance
        
        # 初始化遮罩和待處理佇列
        mask = np.zeros(volume.shape, dtype=bool)
        visited = np.zeros(volume.shape, dtype=bool)
        
        # 限制範圍
        max_distance = max(radius) * 2
        
        queue = [(z, y, x)]
        mask[z, y, x] = True
        visited[z, y, x] = True
        
        while queue:
            cz, cy, cx = queue.pop(0)
            
            # 6-連通鄰域
            for dz, dy, dx in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                
                # 邊界檢查
                if not (0 <= nz < volume.shape[0] and 
                        0 <= ny < volume.shape[1] and 
                        0 <= nx < volume.shape[2]):
                    continue
                
                if visited[nz, ny, nx]:
                    continue
                
                visited[nz, ny, nx] = True
                
                # 距離檢查
                dist = np.sqrt((nz - center[0])**2 + (ny - center[1])**2 + (nx - center[2])**2)
                if dist > max_distance:
                    continue
                
                # 閾值檢查
                if lower <= volume[nz, ny, nx] <= upper:
                    mask[nz, ny, nx] = True
                    queue.append((nz, ny, nx))
        
        # 如果結果太小，回退到橢球體
        if np.sum(mask) < 10:
            mask = self._create_ellipsoid_mask(volume.shape, center, radius)
        
        return mask


class DataAugmentation:
    """
    資料增強類
    
    提供多種醫學影像資料增強方法，專為改善模型泛化能力設計
    
    Args:
        rotation_range: 隨機旋轉角度範圍 (度)
        scale_range: 隨機縮放範圍 (min, max)
        flip_prob: 翻轉機率
        brightness_range: 亮度調整範圍
        noise_std: 高斯噪聲標準差
        elastic_transform: 是否啟用彈性變形
        strong_augmentation: 是否啟用強增強模式
    """
    
    def __init__(
        self,
        rotation_range: float = 15.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        flip_prob: float = 0.5,
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        noise_std: float = 0.02,
        elastic_transform: bool = False,
        hsv_augmentation: bool = False,
        strong_augmentation: bool = False
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.elastic_transform = elastic_transform
        self.hsv_augmentation = hsv_augmentation
        self.strong_augmentation = strong_augmentation
        
        # 強增強模式下調整參數
        if strong_augmentation:
            self.rotation_range = max(rotation_range, 30.0)
            self.scale_range = (0.8, 1.2)
            self.brightness_range = (0.8, 1.2)
            self.contrast_range = (0.7, 1.4)
            self.noise_std = max(noise_std, 0.05)
            self.elastic_transform = True
            self.hsv_augmentation = True
    
    def __call__(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        應用資料增強
        
        Args:
            image: 輸入影像 [C, H, W] 或 [H, W]
            mask: 分割遮罩 [H, W]
        
        Returns:
            增強後的 (image, mask)
        """
        from scipy import ndimage as ndi
        from skimage import transform as skitransform
        
        # 記錄原始形狀
        original_shape = image.shape
        is_multichannel = len(original_shape) == 3
        
        # === 1. 隨機旋轉 ===
        if self.rotation_range > 0 and random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            if is_multichannel:
                # 對每個通道分別旋轉
                rotated_image = np.zeros_like(image)
                for c in range(image.shape[0]):
                    rotated_image[c] = ndi.rotate(image[c], angle, reshape=False, 
                                                  order=1, mode='constant', cval=0)
                image = rotated_image
            else:
                image = ndi.rotate(image, angle, reshape=False, order=1, 
                                  mode='constant', cval=0)
            mask = ndi.rotate(mask, angle, reshape=False, order=0, 
                             mode='constant', cval=0)
        
        # === 2. 隨機縮放 ===
        if self.scale_range != (1.0, 1.0) and random.random() < 0.5:
            scale = random.uniform(*self.scale_range)
            if is_multichannel:
                h, w = image.shape[1], image.shape[2]
            else:
                h, w = image.shape[0], image.shape[1]
            
            new_h, new_w = int(h * scale), int(w * scale)
            
            # 縮放影像
            if is_multichannel:
                scaled_image = np.zeros((image.shape[0], new_h, new_w), dtype=image.dtype)
                for c in range(image.shape[0]):
                    scaled_image[c] = skitransform.resize(
                        image[c], (new_h, new_w), preserve_range=True, 
                        anti_aliasing=True, order=1
                    )
            else:
                scaled_image = skitransform.resize(
                    image, (new_h, new_w), preserve_range=True, 
                    anti_aliasing=True, order=1
                )
            
            # 縮放遮罩 (使用最近鄰插值保持二值)
            scaled_mask = skitransform.resize(
                mask.astype(float), (new_h, new_w), preserve_range=True, 
                order=0, anti_aliasing=False
            )
            
            # 裁剪或填充回原始大小
            if scale > 1:
                # 中心裁剪
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                if is_multichannel:
                    image = scaled_image[:, start_h:start_h+h, start_w:start_w+w]
                else:
                    image = scaled_image[start_h:start_h+h, start_w:start_w+w]
                mask = scaled_mask[start_h:start_h+h, start_w:start_w+w]
            else:
                # 中心填充
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                if is_multichannel:
                    new_image = np.zeros((image.shape[0], h, w), dtype=image.dtype)
                    new_image[:, pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled_image
                    image = new_image
                else:
                    new_image = np.zeros((h, w), dtype=image.dtype)
                    new_image[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled_image
                    image = new_image
                new_mask = np.zeros((h, w), dtype=mask.dtype)
                new_mask[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled_mask
                mask = new_mask
        
        # === 3. 隨機水平翻轉 ===
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=-1).copy()
            mask = np.flip(mask, axis=-1).copy()
        
        # === 4. 隨機垂直翻轉 ===
        if random.random() < self.flip_prob:
            image = np.flip(image, axis=-2).copy()
            mask = np.flip(mask, axis=-2).copy()
        
        # === 5. 彈性變形 (Elastic Deformation) ===
        if self.elastic_transform and random.random() < 0.3:
            image, mask = self._elastic_transform(image, mask)
        
        # === 6. 亮度調整 ===
        if self.brightness_range != (1.0, 1.0) and random.random() < 0.5:
            factor = random.uniform(*self.brightness_range)
            image = image * factor
            image = np.clip(image, 0, 1)
        
        # === 7. Gamma 校正 ===
        if random.random() < 0.3:
            gamma = random.uniform(0.8, 1.2)
            if self.strong_augmentation:
                gamma = random.uniform(0.7, 1.4)
            image = np.power(image + 1e-8, gamma)
            image = np.clip(image, 0, 1)
        
        # === 8. 對比度調整 (Luminance Contrast Adjustment) ===
        if random.random() < 0.5:
            factor = random.uniform(*self.contrast_range)
            mean_val = np.mean(image)
            image = (image - mean_val) * factor + mean_val
            image = np.clip(image, 0, 1)
        
        # === 9. HSV 色彩空間轉換增強 ===
        if self.hsv_augmentation and random.random() < 0.5:
            image = self._hsv_augmentation(image)
        
        # === 10. 加入高斯噪聲 ===
        if self.noise_std > 0 and random.random() < 0.5:
            noise = np.random.normal(0, self.noise_std, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1)
        
        # 確保遮罩是二值的
        mask = (mask > 0.5).astype(np.float32)
        
        return image.astype(np.float32), mask
    
    def _elastic_transform(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        alpha: float = 30.0,
        sigma: float = 4.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        彈性變形 (Elastic Deformation)
        
        對醫學影像分割特別有效，模擬組織的自然變形
        
        Args:
            image: 輸入影像
            mask: 分割遮罩
            alpha: 變形強度
            sigma: 平滑程度
        
        Returns:
            變形後的 (image, mask)
        """
        from scipy.ndimage import gaussian_filter, map_coordinates
        
        is_multichannel = len(image.shape) == 3
        
        if is_multichannel:
            h, w = image.shape[1], image.shape[2]
        else:
            h, w = image.shape[0], image.shape[1]
        
        # 生成隨機位移場
        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
        
        # 創建網格座標
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # 添加位移
        indices_x = np.clip(x + dx, 0, w - 1)
        indices_y = np.clip(y + dy, 0, h - 1)
        
        # 應用變形到影像
        if is_multichannel:
            transformed_image = np.zeros_like(image)
            for c in range(image.shape[0]):
                transformed_image[c] = map_coordinates(
                    image[c], [indices_y, indices_x], order=1, mode='constant', cval=0
                )
        else:
            transformed_image = map_coordinates(
                image, [indices_y, indices_x], order=1, mode='constant', cval=0
            )
        
        # 應用變形到遮罩 (使用最近鄰插值)
        transformed_mask = map_coordinates(
            mask.astype(float), [indices_y, indices_x], order=0, mode='constant', cval=0
        )
        
        return transformed_image, transformed_mask.astype(mask.dtype)
    
    def _hsv_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        HSV 色彩空間轉換增強
        
        將 RGB 影像轉換到 HSV 空間，隨機調整後轉回 RGB
        對於灰階 CT 影像（3通道複製），主要影響 Value 通道
        
        Args:
            image: 輸入影像 [C, H, W] 或 [H, W]，範圍 [0, 1]
        
        Returns:
            增強後的影像
        """
        import cv2
        
        is_multichannel = len(image.shape) == 3
        
        if is_multichannel:
            # [C, H, W] -> [H, W, C]
            img_hwc = np.transpose(image, (1, 2, 0))
        else:
            # [H, W] -> [H, W, 3]
            img_hwc = np.stack([image] * 3, axis=-1)
        
        # 轉換到 [0, 255] uint8
        img_uint8 = (img_hwc * 255).astype(np.uint8)
        
        # RGB -> HSV
        img_hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # H: Hue 色相 (0-180 in OpenCV)
        # S: Saturation 飽和度 (0-255)
        # V: Value 亮度 (0-255)
        
        # 隨機調整 Hue (小幅度，因為 CT 影像沒有真正的顏色)
        h_shift = random.uniform(-5, 5)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + h_shift) % 180
        
        # 隨機調整 Saturation
        s_factor = random.uniform(0.8, 1.2)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * s_factor, 0, 255)
        
        # 隨機調整 Value (亮度)
        v_factor = random.uniform(0.9, 1.1)
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * v_factor, 0, 255)
        
        # HSV -> RGB
        img_hsv = img_hsv.astype(np.uint8)
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        
        # 轉回 [0, 1] float
        img_out = img_rgb.astype(np.float32) / 255.0
        
        if is_multichannel:
            # [H, W, C] -> [C, H, W]
            return np.transpose(img_out, (2, 0, 1))
        else:
            # 取單通道
            return img_out[:, :, 0]


class ChestTumorDataset(Dataset):
    """
    胸部腫瘤 CT 資料集 (通用版本)
    
    支援載入 NIfTI 格式的 CT 影像和分割遮罩
    
    Args:
        data_dir: 資料集目錄
        patient_ids: 患者 ID 列表
        axis: 切片軸向 (0=sagittal, 1=coronal, 2=axial)
        transform: 資料增強
        cache_data: 是否緩存資料
        target_size: 目標影像大小
    """
    
    def __init__(
        self,
        data_dir: str,
        patient_ids: List[str],
        axis: int = 2,
        transform: Optional[Callable] = None,
        cache_data: bool = False,
        target_size: Tuple[int, int] = (256, 256)
    ):
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.axis = axis
        self.transform = transform
        self.cache_data = cache_data
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)
        
        # 建立樣本索引
        self.samples = []
        self._build_sample_index()
        
        # 資料緩存
        self.cache = {} if cache_data else None
    
    def _build_sample_index(self):
        """建立樣本索引（子類實現）"""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """獲取單個樣本"""
        raise NotImplementedError("Subclass must implement __getitem__")
    
    def _normalize_image(
        self, 
        image: np.ndarray,
        hu_min: float = -1000,
        hu_max: float = 800
    ) -> np.ndarray:
        """
        CT HU 值裁剪與正規化
        
        1. CT 值裁剪到 [hu_min, hu_max] 範圍 (預設 [-1000, 800] HU)
        2. 正規化到 [0, 1]
        
        Args:
            image: CT 影像 (HU 值)
            hu_min: 最小 HU 值 (預設 -1000，適用於肺部)
            hu_max: 最大 HU 值 (預設 800，涵蓋軟組織和骨骼邊緣)
        
        Returns:
            正規化後的影像 [0, 1]
        """
        # Step 1: HU 值裁剪
        image = np.clip(image, hu_min, hu_max)
        
        # Step 2: 正規化到 [0, 1]
        image = (image - hu_min) / (hu_max - hu_min)
        
        return image.astype(np.float32)
    
    def _resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """調整影像大小"""
        from skimage.transform import resize
        return resize(image, size, preserve_range=True, anti_aliasing=True)
    
    def _resize_mask(self, mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """調整遮罩大小"""
        from skimage.transform import resize
        mask = resize(mask.astype(float), size, preserve_range=True, order=0)
        return (mask > 0.5).astype(np.float32)
    
    def _extract_grid_patches(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        patch_size: int = 224
    ) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]:
        """
        從影像中提取 2x2 網格的 patches (區域裁剪)
        
        將影像劃分為 2x2 網格，每個 patch 為 patch_size x patch_size
        返回 4 個 patches 及其對應的遮罩
        
        Args:
            image: 輸入影像 [H, W]
            mask: 分割遮罩 [H, W]
            patch_size: patch 大小 (預設 224)
        
        Returns:
            List of (patch_image, patch_mask, (row, col)) tuples
        """
        h, w = image.shape[:2]
        
        # 如果影像太小，先 resize 到可以切 2x2
        min_size = patch_size * 2
        if h < min_size or w < min_size:
            # 直接 resize 到 patch_size 並返回單個 patch
            from skimage.transform import resize
            resized_img = resize(image, (patch_size, patch_size), preserve_range=True)
            resized_mask = resize(mask.astype(float), (patch_size, patch_size), order=0, preserve_range=True)
            return [(resized_img.astype(np.float32), (resized_mask > 0.5).astype(np.float32), (0, 0))]
        
        patches = []
        
        # 計算網格起始位置 (2x2)
        # 使用等間距切分
        row_starts = [0, h // 2 - patch_size // 2]
        col_starts = [0, w // 2 - patch_size // 2]
        
        # 確保不超出邊界
        row_starts = [max(0, min(r, h - patch_size)) for r in row_starts]
        col_starts = [max(0, min(c, w - patch_size)) for c in col_starts]
        
        for r_idx, r_start in enumerate(row_starts):
            for c_idx, c_start in enumerate(col_starts):
                # 提取 patch
                patch_img = image[r_start:r_start + patch_size, c_start:c_start + patch_size]
                patch_mask = mask[r_start:r_start + patch_size, c_start:c_start + patch_size]
                
                # 確保大小正確
                if patch_img.shape[0] != patch_size or patch_img.shape[1] != patch_size:
                    from skimage.transform import resize
                    patch_img = resize(patch_img, (patch_size, patch_size), preserve_range=True)
                    patch_mask = resize(patch_mask.astype(float), (patch_size, patch_size), order=0, preserve_range=True)
                
                patches.append((
                    patch_img.astype(np.float32), 
                    (patch_mask > 0.5).astype(np.float32),
                    (r_idx, c_idx)
                ))
        
        return patches


class LNDbDataset(ChestTumorDataset):
    """
    LNDb 資料集
    
    載入 LNDb 格式的胸部 CT 腫瘤資料
    專家標註的分割遮罩，預期 Dice > 0.85
    
    支援兩種資料格式:
    1. NIfTI 格式: data_dir/LNDb-XXXX/LNDb-XXXX.nii.gz + mask.nii.gz
    2. MHD 格式: data_dir/data0-5/LNDb-XXXX.mhd + masks/masks/LNDb-XXXX_radX.mhd
    
    Args:
        data_dir: LNDb 資料集目錄
        patient_ids: 患者 ID 列表
        rad_id: 放射科醫師 ID ('consensus', '1', '2', '3')
        axis: 切片軸向
        transform: 資料增強
        cache_data: 是否緩存
        target_size: 目標影像大小
        min_nodule_diameter: 最小結節直徑 (mm)，過濾小結節
    """
    
    def __init__(
        self,
        data_dir: str,
        patient_ids: List[str],
        rad_id: str = "consensus",
        axis: int = 2,
        transform: Optional[Callable] = None,
        cache_data: bool = False,
        target_size: Tuple[int, int] = (256, 256),
        min_nodule_diameter: float = 3.0
    ):
        self.rad_id = rad_id
        self.min_nodule_diameter = min_nodule_diameter
        self.data_format = None  # Will be set in _build_sample_index
        super().__init__(data_dir, patient_ids, axis, transform, cache_data, target_size)
    
    def _find_ct_file(self, patient_id: str) -> Optional[Path]:
        """
        尋找 CT 檔案 (支援 NIfTI 和 MHD 格式)
        
        Args:
            patient_id: 患者 ID，支援多種格式:
                - "LNDb-0001" (完整格式)
                - "1" 或 1 (整數格式，會自動轉換為 LNDb-0001)
        
        Returns:
            CT 檔案路徑，若找不到則返回 None
        """
        # 標準化 patient_id 格式
        if isinstance(patient_id, int) or (isinstance(patient_id, str) and patient_id.isdigit()):
            # 整數格式 -> LNDb-XXXX
            lndb_id = int(patient_id)
            patient_id_formatted = f"LNDb-{lndb_id:04d}"
        elif patient_id.startswith("LNDb-"):
            patient_id_formatted = patient_id
        else:
            # 嘗試解析為數字
            try:
                lndb_id = int(patient_id)
                patient_id_formatted = f"LNDb-{lndb_id:04d}"
            except ValueError:
                patient_id_formatted = patient_id
        
        # 方法 1: NIfTI 格式 (data_dir/LNDb-XXXX/LNDb-XXXX.nii.gz)
        patient_dir = self.data_dir / patient_id_formatted
        if patient_dir.exists():
            nii_files = [
                patient_dir / f"{patient_id_formatted}.nii.gz",
                patient_dir / f"{patient_id_formatted}_ct.nii.gz"
            ]
            for f in nii_files:
                if f.exists():
                    self.data_format = "nifti"
                    return f
        
        # 方法 2: MHD 格式 (data_dir/data0-5/LNDb-XXXX.mhd)
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith("data"):
                mhd_file = subdir / f"{patient_id_formatted}.mhd"
                if mhd_file.exists():
                    self.data_format = "mhd"
                    return mhd_file
        
        return None
    
    def _find_mask_file(self, patient_id: str) -> Optional[Path]:
        """
        尋找遮罩檔案 (支援 NIfTI 和 MHD 格式)
        
        Args:
            patient_id: 患者 ID，支援多種格式:
                - "LNDb-0001" (完整格式)
                - "1" 或 1 (整數格式，會自動轉換為 LNDb-0001)
        
        優先順序:
        1. consensus 遮罩 (多放射科醫師共識)
        2. rad1 遮罩 (第一位放射科醫師)
        
        Returns:
            遮罩檔案路徑，若找不到則返回 None
        """
        # 標準化 patient_id 格式
        if isinstance(patient_id, int) or (isinstance(patient_id, str) and patient_id.isdigit()):
            lndb_id = int(patient_id)
            patient_id_formatted = f"LNDb-{lndb_id:04d}"
        elif patient_id.startswith("LNDb-"):
            patient_id_formatted = patient_id
        else:
            try:
                lndb_id = int(patient_id)
                patient_id_formatted = f"LNDb-{lndb_id:04d}"
            except ValueError:
                patient_id_formatted = patient_id
        
        # 方法 1: NIfTI 格式 (data_dir/LNDb-XXXX/mask*.nii.gz)
        patient_dir = self.data_dir / patient_id_formatted
        if patient_dir.exists():
            if self.rad_id == "consensus":
                patterns = [
                    f"{patient_id_formatted}_mask_consensus*.nii.gz",
                    f"{patient_id_formatted}_mask.nii.gz"
                ]
            else:
                patterns = [f"{patient_id_formatted}_rad{self.rad_id}*.nii.gz"]
            
            for pattern in patterns:
                matches = list(patient_dir.glob(pattern))
                if matches:
                    return matches[0]
        
        # 方法 2: MHD 格式 (data_dir/masks/masks/LNDb-XXXX_radX.mhd)
        masks_dir = self.data_dir / "masks" / "masks"
        if masks_dir.exists():
            # 決定使用哪個放射科醫師的標註
            if self.rad_id == "consensus":
                # 先嘗試找 rad1，然後 rad2, rad3
                for rad in ["1", "2", "3"]:
                    mask_file = masks_dir / f"{patient_id_formatted}_rad{rad}.mhd"
                    if mask_file.exists():
                        return mask_file
            else:
                mask_file = masks_dir / f"{patient_id_formatted}_rad{self.rad_id}.mhd"
                if mask_file.exists():
                    return mask_file
        
        return None
    
    def _build_sample_index(self):
        """建立 LNDb 樣本索引 (支援 NIfTI 和 MHD 格式)"""
        self.samples = []
        
        for patient_id in tqdm(self.patient_ids, desc="Building LNDb index"):
            # 尋找 CT 檔案
            ct_file = self._find_ct_file(patient_id)
            if ct_file is None:
                self.logger.warning(f"CT file not found for {patient_id}")
                continue
            
            # 尋找遮罩檔案
            mask_file = self._find_mask_file(patient_id)
            if mask_file is None:
                self.logger.warning(f"Mask file not found for {patient_id}")
                continue
            
            # 載入影像獲取切片數量
            try:
                ct_img = sitk.ReadImage(str(ct_file))
                ct_array = sitk.GetArrayFromImage(ct_img)
                mask_img = sitk.ReadImage(str(mask_file))
                mask_array = sitk.GetArrayFromImage(mask_img)
                
                # 確保 mask 與 CT 大小匹配
                if ct_array.shape != mask_array.shape:
                    self.logger.warning(
                        f"Shape mismatch for {patient_id}: "
                        f"CT={ct_array.shape}, Mask={mask_array.shape}. Skipping."
                    )
                    continue
                
                # 找出有腫瘤的切片
                if self.axis == 2:  # Axial
                    for z in range(ct_array.shape[0]):
                        if np.any(mask_array[z] > 0):
                            self.samples.append({
                                'patient_id': patient_id,
                                'ct_file': str(ct_file),
                                'mask_file': str(mask_file),
                                'slice_idx': z
                            })
                elif self.axis == 1:  # Coronal
                    for y in range(ct_array.shape[1]):
                        if np.any(mask_array[:, y, :] > 0):
                            self.samples.append({
                                'patient_id': patient_id,
                                'ct_file': str(ct_file),
                                'mask_file': str(mask_file),
                                'slice_idx': y
                            })
                else:  # Sagittal
                    for x in range(ct_array.shape[2]):
                        if np.any(mask_array[:, :, x] > 0):
                            self.samples.append({
                                'patient_id': patient_id,
                                'ct_file': str(ct_file),
                                'mask_file': str(mask_file),
                                'slice_idx': x
                            })
                            
            except Exception as e:
                self.logger.warning(f"Error processing {patient_id}: {e}")
                continue
        
        self.logger.info(f"Built LNDb dataset with {len(self.samples)} samples")
        if self.data_format:
            self.logger.info(f"Data format: {self.data_format.upper()}")
    
    def __getitem__(self, idx: int) -> Dict:
        """獲取單個樣本"""
        sample_info = self.samples[idx]
        
        # 檢查緩存
        cache_key = f"{sample_info['patient_id']}_{sample_info['slice_idx']}"
        if self.cache is not None and cache_key in self.cache:
            return self.cache[cache_key]
        
        # 載入資料
        ct_img = sitk.ReadImage(sample_info['ct_file'])
        ct_array = sitk.GetArrayFromImage(ct_img)
        mask_img = sitk.ReadImage(sample_info['mask_file'])
        mask_array = sitk.GetArrayFromImage(mask_img)
        
        slice_idx = sample_info['slice_idx']
        
        # 提取切片
        if self.axis == 2:  # Axial
            image_slice = ct_array[slice_idx]
            mask_slice = mask_array[slice_idx]
        elif self.axis == 1:  # Coronal
            image_slice = ct_array[:, slice_idx, :]
            mask_slice = mask_array[:, slice_idx, :]
        else:  # Sagittal
            image_slice = ct_array[:, :, slice_idx]
            mask_slice = mask_array[:, :, slice_idx]
        
        # 正規化
        image_slice = self._normalize_image(image_slice)
        
        # 調整大小
        image_slice = self._resize_image(image_slice, self.target_size)
        mask_slice = self._resize_mask(mask_slice, self.target_size)
        
        # 擴展為 3 通道
        image = np.stack([image_slice] * 3, axis=0)  # [3, H, W]
        mask = mask_slice  # [H, W]
        
        # 資料增強
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        # 計算 bounding box
        bbox = self._compute_bbox(mask)
        
        # 轉換為 tensor
        result = {
            'image': torch.from_numpy(image).float(),
            'mask': torch.from_numpy(mask).float().unsqueeze(0),  # [1, H, W]
            'bbox': torch.tensor(bbox).float(),
            'patient_id': sample_info['patient_id'],
            'slice_idx': slice_idx
        }
        
        # 緩存
        if self.cache is not None:
            self.cache[cache_key] = result
        
        return result
    
    def _compute_bbox(self, mask: np.ndarray) -> List[float]:
        """計算遮罩的 bounding box"""
        if not np.any(mask > 0):
            return [0, 0, mask.shape[1], mask.shape[0]]
        
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # 擴展 bbox 一點邊距
        margin = 5
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(mask.shape[1], x_max + margin)
        y_max = min(mask.shape[0], y_max + margin)
        
        return [x_min, y_min, x_max, y_max]


class LNDbPatchDataset(LNDbDataset):
    """
    LNDb Patch 資料集
    
    繼承 LNDbDataset，添加 2x2 網格 patch 提取功能
    每個切片會被分成 4 個 224x224 的 patches
    
    Args:
        data_dir: LNDb 資料集目錄
        patient_ids: 患者 ID 列表
        patch_size: patch 大小 (預設 224)
        rad_id: 放射科醫師 ID
        axis: 切片軸向
        transform: 資料增強
        cache_data: 是否緩存
        filter_empty_patches: 是否過濾沒有病灶的 patches
    """
    
    def __init__(
        self,
        data_dir: str,
        patient_ids: List[str],
        patch_size: int = 224,
        rad_id: str = "consensus",
        axis: int = 2,
        transform: Optional[Callable] = None,
        cache_data: bool = False,
        filter_empty_patches: bool = True,
        empty_patch_ratio: float = 0.2,
        include_full_slices: bool = False,
        full_slice_size: int = 448,
        min_nodule_diameter: float = 3.0
    ):
        self.patch_size = patch_size
        self.filter_empty_patches = filter_empty_patches
        self.empty_patch_ratio = empty_patch_ratio  # 保留的空 patches 比例
        self.include_full_slices = include_full_slices
        self.full_slice_size = full_slice_size
        
        # 調用父類 __init__，target_size 設為 None 因為我們會自己處理大小
        super().__init__(
            data_dir=data_dir,
            patient_ids=patient_ids,
            rad_id=rad_id,
            axis=axis,
            transform=transform,
            cache_data=cache_data,
            target_size=(patch_size * 2, patch_size * 2),  # 需要足夠大以提取 2x2 patches
            min_nodule_diameter=min_nodule_diameter
        )
        
        # 重建 sample index 以包含 patch 資訊
        self._build_patch_index()
    
    def _build_patch_index(self):
        """
        [C] 建立 patch 樣本索引，可選擇過濾空 patches
        
        Enhanced with comprehensive statistics logging to clarify:
        - How many patches are generated vs kept
        - Positive (with lesion) vs empty patch counts  
        - What __len__() returns
        """
        from tqdm import tqdm
        
        # 將原始樣本擴展為 patch 樣本
        original_samples = self.samples.copy()
        self.samples = []
        
        # [C] Enhanced statistics tracking
        stats = {
            'patients_processed': set(),
            'patients_skipped': set(),
            'total_patches': 0,
            'positive_patches': 0,      # with lesion
            'empty_patches_total': 0,
            'empty_patches_kept': 0,
            'empty_patches_filtered': 0,
            'full_slices': 0,
        }
        
        # 如果需要過濾，需要讀取 mask 資料
        if self.filter_empty_patches:
            self.logger.info(f"🔍 Mixed training mode (keeping {self.empty_patch_ratio*100:.0f}% empty patches)...")
            
            for sample_info in tqdm(original_samples, desc="Filtering patches"):
                patient_id = sample_info['patient_id']
                
                # 讀取 mask 資料
                try:
                    mask_img = sitk.ReadImage(sample_info['mask_file'])
                    mask_array = sitk.GetArrayFromImage(mask_img)
                    slice_idx = sample_info['slice_idx']
                    
                    stats['patients_processed'].add(patient_id)
                    
                    # 提取切片
                    if self.axis == 2:  # Axial
                        mask_slice = mask_array[slice_idx]
                    elif self.axis == 1:  # Coronal
                        mask_slice = mask_array[:, slice_idx, :]
                    else:  # Sagittal
                        mask_slice = mask_array[:, :, slice_idx]
                    
                    h, w = mask_slice.shape
                    patch_size = self.patch_size
                    
                    # 計算 2x2 網格的起始位置
                    row_starts = [0, max(0, h // 2 - patch_size // 2)]
                    col_starts = [0, max(0, w // 2 - patch_size // 2)]
                    row_starts = [max(0, min(r, h - patch_size)) for r in row_starts]
                    col_starts = [max(0, min(c, w - patch_size)) for c in col_starts]
                    
                    for patch_row, r_start in enumerate(row_starts):
                        for patch_col, c_start in enumerate(col_starts):
                            stats['total_patches'] += 1
                            
                            # 提取 patch mask
                            patch_mask = mask_slice[r_start:r_start + patch_size, 
                                                    c_start:c_start + patch_size]
                            
                            # 檢查是否有病灶 (mask 中有非零值)
                            has_lesion = np.sum(patch_mask > 0) > 0
                            
                            if has_lesion:
                                stats['positive_patches'] += 1
                            else:
                                stats['empty_patches_total'] += 1
                            
                            # 混合訓練：保留所有有病灶的 + 部分空 patches
                            keep_patch = has_lesion or (random.random() < self.empty_patch_ratio)
                            
                            if keep_patch:
                                patch_sample = sample_info.copy()
                                patch_sample['patch_row'] = patch_row
                                patch_sample['patch_col'] = patch_col
                                patch_sample['patch_idx'] = patch_row * 2 + patch_col
                                patch_sample['has_lesion'] = has_lesion
                                patch_sample['sample_type'] = 'patch'
                                self.samples.append(patch_sample)
                                if not has_lesion:
                                    stats['empty_patches_kept'] += 1
                            else:
                                stats['empty_patches_filtered'] += 1
                                
                except Exception as e:
                    stats['patients_skipped'].add(patient_id)
                    self.logger.warning(f"Error processing {patient_id}: {e}")
                    continue
            
            # 加入完整切片樣本
            if self.include_full_slices:
                for sample_info in original_samples:
                    full_sample = sample_info.copy()
                    full_sample['sample_type'] = 'full_slice'
                    full_sample['patch_row'] = -1
                    full_sample['patch_col'] = -1
                    full_sample['patch_idx'] = -1
                    full_sample['has_lesion'] = True  # Full slices always have lesions (selected earlier)
                    self.samples.append(full_sample)
                    stats['full_slices'] += 1
        else:
            # 不過濾，包含所有 patches
            for sample_info in original_samples:
                patient_id = sample_info['patient_id']
                stats['patients_processed'].add(patient_id)
                
                for patch_row in range(2):
                    for patch_col in range(2):
                        stats['total_patches'] += 1
                        patch_sample = sample_info.copy()
                        patch_sample['patch_row'] = patch_row
                        patch_sample['patch_col'] = patch_col
                        patch_sample['patch_idx'] = patch_row * 2 + patch_col
                        patch_sample['sample_type'] = 'patch'
                        self.samples.append(patch_sample)
            
            # 加入完整切片樣本
            if self.include_full_slices:
                for sample_info in original_samples:
                    full_sample = sample_info.copy()
                    full_sample['sample_type'] = 'full_slice'
                    full_sample['patch_row'] = -1
                    full_sample['patch_col'] = -1
                    full_sample['patch_idx'] = -1
                    self.samples.append(full_sample)
                    stats['full_slices'] += 1
        
        # [C] Comprehensive statistics summary - use print() to ensure visibility
        print("=" * 60)
        print("📊 DATASET STATISTICS SUMMARY")
        print("=" * 60)
        print(f"  Patients processed: {len(stats['patients_processed'])}")
        if stats['patients_skipped']:
            print(f"  ⚠️ Patients skipped (errors): {len(stats['patients_skipped'])}")
        print(f"  Original slices: {len(original_samples)}")
        print(f"  Total patches generated: {stats['total_patches']}")
        
        if self.filter_empty_patches:
            print(f"  ├── Positive patches (has lesion): {stats['positive_patches']}")
            print(f"  └── Empty patches total: {stats['empty_patches_total']}")
            print(f"      ├── Kept ({self.empty_patch_ratio:.0%}): {stats['empty_patches_kept']}")
            print(f"      └── Filtered: {stats['empty_patches_filtered']}")
        
        if stats['full_slices'] > 0:
            print(f"  Full slices added: {stats['full_slices']} ({self.full_slice_size}×{self.full_slice_size})")
        
        # [C] Explicit __len__() explanation
        patch_count = len([s for s in self.samples if s.get('sample_type') == 'patch'])
        full_count = len([s for s in self.samples if s.get('sample_type') == 'full_slice'])
        print("-" * 60)
        print(f"📦 __len__() returns: {len(self.samples)} total samples")
        print(f"   = {patch_count} patches + {full_count} full slices")
        if self.filter_empty_patches:
            positive_ratio = stats['positive_patches'] / max(1, patch_count)
            print(f"   Positive (has lesion) ratio: {positive_ratio:.1%}")
        print("=" * 60)

    
    def __getitem__(self, idx: int) -> Dict:
        """獲取單個 patch 或 full slice 樣本"""
        sample_info = self.samples[idx]
        
        # 判斷是 patch 還是 full slice
        is_full_slice = sample_info.get('sample_type') == 'full_slice'
        
        # 檢查緩存
        cache_key = f"{sample_info['patient_id']}_{sample_info['slice_idx']}_" + \
                    (f"full" if is_full_slice else f"p{sample_info['patch_idx']}")
        if self.cache is not None and cache_key in self.cache:
            return self.cache[cache_key]
        
        # 載入資料
        ct_img = sitk.ReadImage(sample_info['ct_file'])
        ct_array = sitk.GetArrayFromImage(ct_img)
        mask_img = sitk.ReadImage(sample_info['mask_file'])
        mask_array = sitk.GetArrayFromImage(mask_img)
        
        slice_idx = sample_info['slice_idx']
        
        # 提取切片
        if self.axis == 2:  # Axial
            image_slice = ct_array[slice_idx]
            mask_slice = mask_array[slice_idx]
        elif self.axis == 1:  # Coronal
            image_slice = ct_array[:, slice_idx, :]
            mask_slice = mask_array[:, slice_idx, :]
        else:  # Sagittal
            image_slice = ct_array[:, :, slice_idx]
            mask_slice = mask_array[:, :, slice_idx]
        
        # 正規化 (HU clipping to [-1000, 800])
        image_slice = self._normalize_image(image_slice)
        
        if is_full_slice:
            # Full slice 處理：resize 到指定大小
            target_size = (self.full_slice_size, self.full_slice_size)
            image = cv2.resize(image_slice, target_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask_slice.astype(np.float32), target_size, interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.float32)
            p_row, p_col = -1, -1
        else:
            # Patch 處理：提取 2x2 網格 patches
            patches = self._extract_grid_patches(image_slice, mask_slice, self.patch_size)
            
            # 獲取對應的 patch
            patch_idx = sample_info['patch_idx']
            if patch_idx < len(patches):
                image, mask, (p_row, p_col) = patches[patch_idx]
            else:
                # 如果 patch 數量不足，使用第一個
                image, mask, (p_row, p_col) = patches[0]
        
        # 擴展為 3 通道
        image = np.stack([image] * 3, axis=0)  # [3, H, W]
        
        # 資料增強
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        # 計算 bounding box
        bbox = self._compute_bbox(mask)
        
        # [F] Shape validation
        expected_img_size = self.full_slice_size if is_full_slice else self.patch_size
        assert image.shape == (3, expected_img_size, expected_img_size), \
            f"[F] Unexpected image shape: {image.shape}, expected (3, {expected_img_size}, {expected_img_size})"
        assert mask.shape == (expected_img_size, expected_img_size), \
            f"[F] Unexpected mask shape: {mask.shape}, expected ({expected_img_size}, {expected_img_size})"
        
        # [Fix B] 確保 mask 是 binary (0/1)
        # 處理 augmentation/resize 後可能產生的插值值 (0.1, 0.2 等)
        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)
        
        # 轉換為 tensor
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # [1, H, W]
        
        result = {
            'image': torch.from_numpy(image).float(),
            'mask': mask_tensor,
            'bbox': torch.tensor(bbox).float(),
            'patient_id': sample_info['patient_id'],
            'slice_idx': slice_idx,
            'patch_idx': sample_info['patch_idx'],
            'patch_row': sample_info['patch_row'],
            'patch_col': sample_info['patch_col'],
            'sample_type': 'full_slice' if is_full_slice else 'patch'
        }
        
        # [F] First sample debug logging - clarify channel semantics
        if idx == 0 and not hasattr(self, '_first_sample_logged'):
            self._first_sample_logged = True
            print("=" * 60)
            print("🔬 [F] FIRST SAMPLE DEBUG INFO")
            print("=" * 60)
            print(f"  Image tensor shape: {result['image'].shape}")
            print(f"  ↳ 3 channels = DUPLICATED grayscale (same slice × 3)")
            print(f"  ↳ This is for compatibility with ImageNet-pretrained encoders")
            print(f"  Image value range: [{result['image'].min():.3f}, {result['image'].max():.3f}]")
            print(f"  ↳ Expected: [0, 1] after HU normalization")
            print(f"  Mask tensor shape: {result['mask'].shape}")
            mask_nonzero = (result['mask'] > 0).float().mean().item()
            print(f"  Mask non-zero ratio: {mask_nonzero:.4f}")
            print(f"  Sample type: {result['sample_type']}")
            print("=" * 60)
        
        # 緩存
        if self.cache is not None:
            self.cache[cache_key] = result
        
        return result


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """
    自定義 collate 函數
    
    處理可變長度的資料
    """
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    bboxes = torch.stack([item['bbox'] for item in batch])
    
    patient_ids = [item['patient_id'] for item in batch]
    slice_indices = [item['slice_idx'] for item in batch]
    
    return {
        'image': images,
        'mask': masks,
        'bbox': bboxes,
        'patient_id': patient_ids,
        'slice_idx': slice_indices
    }


if __name__ == "__main__":
    # 測試資料集
    print("Testing dataset module...")
    
    # 測試資料增強
    aug = DataAugmentation(flip_prob=0.5, noise_std=0.02)
    test_image = np.random.rand(3, 256, 256).astype(np.float32)
    test_mask = (np.random.rand(256, 256) > 0.5).astype(np.float32)
    
    aug_image, aug_mask = aug(test_image, test_mask)
    print(f"Augmentation test: image={aug_image.shape}, mask={aug_mask.shape}")
    
    print("Dataset module tests passed!")
