"""
UNet++ 肺結節分割訓練 - 採樣模組
===================================

處理極端類別不平衡問題：
1. 正樣本 Patch Sampling（以結節為中心）
2. Hard Negative Mining（肺野內無結節區域）
3. 隨機負樣本採樣
"""

import logging
from typing import Tuple, List, Dict, Optional
import random

import numpy as np
from scipy import ndimage
from skimage import measure


logger = logging.getLogger(__name__)


class PatchSampler:
    """Patch 採樣器"""
    
    def __init__(
        self,
        patch_size: int = 160,
        positive_ratio: float = 0.7,
        hard_negative_ratio: float = 0.2,
        random_negative_ratio: float = 0.1,
        min_nodule_pixels: int = 10,
        center_jitter: float = 0.3
    ):
        """
        初始化採樣器
        
        Args:
            patch_size: Patch 大小（正方形）
            positive_ratio: 正樣本比例
            hard_negative_ratio: Hard Negative 比例
            random_negative_ratio: 隨機負樣本比例
            min_nodule_pixels: 最小結節像素數（過濾太小的結節）
            center_jitter: 結節中心抖動比例（相對於 patch_size）
        """
        self.patch_size = patch_size
        self.positive_ratio = positive_ratio
        self.hard_negative_ratio = hard_negative_ratio
        self.random_negative_ratio = random_negative_ratio
        self.min_nodule_pixels = min_nodule_pixels
        self.center_jitter = center_jitter
        
        # 確保比例總和為 1
        total = positive_ratio + hard_negative_ratio + random_negative_ratio
        if not np.isclose(total, 1.0):
            logger.warning(f"採樣比例總和 {total} != 1.0，將進行歸一化")
            self.positive_ratio /= total
            self.hard_negative_ratio /= total
            self.random_negative_ratio /= total
    
    def find_nodule_centers(
        self,
        mask: np.ndarray,
        spacing: np.ndarray = None
    ) -> List[Dict]:
        """
        找到遮罩中所有結節的中心位置
        
        Args:
            mask: 分割遮罩 (Z, Y, X) 或 (Y, X)
            spacing: 體素間距
            
        Returns:
            結節資訊列表
        """
        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]  # 擴展為 3D
            is_2d = True
        else:
            is_2d = False
        
        labels = measure.label(mask > 0.5)
        regions = measure.regionprops(labels)
        
        nodules = []
        for region in regions:
            if region.area < self.min_nodule_pixels:
                continue
            
            centroid = np.array(region.centroid)
            if is_2d:
                centroid = centroid[1:]  # 移除 Z 維度
            
            nodule_info = {
                'centroid': centroid,
                'area': region.area,
                'bbox': region.bbox,
                'label': region.label
            }
            
            if spacing is not None:
                # 計算實際大小
                nodule_info['volume_mm3'] = region.area * np.prod(spacing)
            
            nodules.append(nodule_info)
        
        return nodules
    
    def sample_positive_patch(
        self,
        volume_shape: Tuple[int, int],  # (H, W)
        nodule_center: np.ndarray,
        jitter: bool = True
    ) -> Tuple[int, int, int, int]:
        """
        採樣以結節為中心的 Patch
        
        Args:
            volume_shape: 影像形狀 (H, W)
            nodule_center: 結節中心座標 (y, x)
            jitter: 是否加入隨機抖動
            
        Returns:
            (y_start, y_end, x_start, x_end)
        """
        center_y, center_x = nodule_center
        
        if jitter:
            # 加入隨機抖動，避免結節總是在正中心
            max_jitter = int(self.patch_size * self.center_jitter)
            center_y += random.randint(-max_jitter, max_jitter)
            center_x += random.randint(-max_jitter, max_jitter)
        
        half_size = self.patch_size // 2
        
        y_start = int(center_y - half_size)
        y_end = y_start + self.patch_size
        x_start = int(center_x - half_size)
        x_end = x_start + self.patch_size
        
        # 邊界處理
        if y_start < 0:
            y_start, y_end = 0, self.patch_size
        elif y_end > volume_shape[0]:
            y_start, y_end = volume_shape[0] - self.patch_size, volume_shape[0]
        
        if x_start < 0:
            x_start, x_end = 0, self.patch_size
        elif x_end > volume_shape[1]:
            x_start, x_end = volume_shape[1] - self.patch_size, volume_shape[1]
        
        return (y_start, y_end, x_start, x_end)
    
    def sample_hard_negative_patch(
        self,
        lung_mask: np.ndarray,
        nodule_mask: np.ndarray,
        max_attempts: int = 50
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        採樣 Hard Negative Patch（肺野內但不含結節）
        
        Args:
            lung_mask: 肺野遮罩 (H, W)
            nodule_mask: 結節遮罩 (H, W)
            max_attempts: 最大嘗試次數
            
        Returns:
            (y_start, y_end, x_start, x_end) 或 None
        """
        # 找到有效採樣區域：肺野內但距離結節較遠
        valid_region = lung_mask.copy()
        
        # 擴展結節區域（避免採樣到結節邊緣）
        if np.any(nodule_mask > 0):
            dilated_nodule = ndimage.binary_dilation(
                nodule_mask > 0,
                iterations=self.patch_size // 4
            )
            valid_region = valid_region & (~dilated_nodule)
        
        # 確保有足夠空間放置 patch
        valid_indices = np.where(valid_region)
        if len(valid_indices[0]) == 0:
            return None
        
        for _ in range(max_attempts):
            # 隨機選擇一個有效點
            idx = random.randint(0, len(valid_indices[0]) - 1)
            center_y = valid_indices[0][idx]
            center_x = valid_indices[1][idx]
            
            half_size = self.patch_size // 2
            y_start = center_y - half_size
            y_end = y_start + self.patch_size
            x_start = center_x - half_size
            x_end = x_start + self.patch_size
            
            # 檢查邊界
            if (y_start >= 0 and y_end <= lung_mask.shape[0] and
                x_start >= 0 and x_end <= lung_mask.shape[1]):
                
                # 確認 patch 內沒有結節
                patch_nodule = nodule_mask[y_start:y_end, x_start:x_end]
                if not np.any(patch_nodule > 0):
                    return (y_start, y_end, x_start, x_end)
        
        return None
    
    def sample_random_patch(
        self,
        volume_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        完全隨機採樣 Patch
        
        Args:
            volume_shape: 影像形狀 (H, W)
            
        Returns:
            (y_start, y_end, x_start, x_end)
        """
        max_y = volume_shape[0] - self.patch_size
        max_x = volume_shape[1] - self.patch_size
        
        y_start = random.randint(0, max(0, max_y))
        x_start = random.randint(0, max(0, max_x))
        
        return (y_start, y_start + self.patch_size, x_start, x_start + self.patch_size)
    
    def sample_patches_for_slice(
        self,
        slice_shape: Tuple[int, int],
        nodule_mask: np.ndarray,
        lung_mask: np.ndarray,
        num_patches: int
    ) -> List[Tuple[int, int, int, int, str]]:
        """
        為單個切片採樣多個 Patch
        
        Args:
            slice_shape: 切片形狀 (H, W)
            nodule_mask: 結節遮罩
            lung_mask: 肺野遮罩
            num_patches: 要採樣的 patch 數量
            
        Returns:
            List of (y_start, y_end, x_start, x_end, sample_type)
        """
        patches = []
        
        # 計算各類型數量
        num_positive = int(num_patches * self.positive_ratio)
        num_hard_neg = int(num_patches * self.hard_negative_ratio)
        num_random = num_patches - num_positive - num_hard_neg
        
        # 找到結節中心
        nodule_centers = self.find_nodule_centers(nodule_mask)
        
        # 1. 正樣本採樣
        if len(nodule_centers) > 0:
            for i in range(num_positive):
                # 隨機選擇一個結節
                nodule = random.choice(nodule_centers)
                bbox = self.sample_positive_patch(
                    slice_shape,
                    nodule['centroid'],
                    jitter=True
                )
                patches.append((*bbox, 'positive'))
        else:
            # 沒有結節，轉為 hard negative
            num_hard_neg += num_positive
            num_positive = 0
        
        # 2. Hard Negative 採樣
        for _ in range(num_hard_neg):
            bbox = self.sample_hard_negative_patch(lung_mask, nodule_mask)
            if bbox is not None:
                patches.append((*bbox, 'hard_negative'))
            else:
                # 降級為隨機採樣
                bbox = self.sample_random_patch(slice_shape)
                patches.append((*bbox, 'random'))
        
        # 3. 隨機採樣
        for _ in range(num_random):
            bbox = self.sample_random_patch(slice_shape)
            patches.append((*bbox, 'random'))
        
        return patches


class SliceSampler:
    """切片採樣器 - 控制 2D/2.5D 切片的選擇"""
    
    def __init__(
        self,
        slice_distance_mm: float = 2.0,
        positive_slice_weight: float = 3.0
    ):
        """
        初始化切片採樣器
        
        Args:
            slice_distance_mm: 2.5D 取樣的固定毫米距離
            positive_slice_weight: 含結節切片的採樣權重
        """
        self.slice_distance_mm = slice_distance_mm
        self.positive_slice_weight = positive_slice_weight
    
    def get_25d_slice_indices(
        self,
        center_z: int,
        num_slices: int,
        spacing_z: float
    ) -> List[int]:
        """
        獲取 2.5D 採樣的切片索引
        
        Args:
            center_z: 中心切片索引
            num_slices: 總切片數
            spacing_z: Z 方向 spacing (mm)
            
        Returns:
            切片索引列表 [prev, center, next]
        """
        offset = int(round(self.slice_distance_mm / spacing_z))
        offset = max(1, offset)  # 至少偏移 1
        
        indices = [
            max(0, center_z - offset),
            center_z,
            min(num_slices - 1, center_z + offset)
        ]
        
        return indices
    
    def find_positive_slices(
        self,
        mask: np.ndarray,
        min_nodule_pixels: int = 5
    ) -> List[int]:
        """
        找到包含結節的切片索引
        
        Args:
            mask: 3D 遮罩 (Z, Y, X)
            min_nodule_pixels: 最小結節像素數
            
        Returns:
            正樣本切片索引列表
        """
        positive_slices = []
        for z in range(mask.shape[0]):
            if np.sum(mask[z] > 0) >= min_nodule_pixels:
                positive_slices.append(z)
        
        return positive_slices
    
    def compute_slice_weights(
        self,
        mask: np.ndarray,
        min_nodule_pixels: int = 5
    ) -> np.ndarray:
        """
        計算各切片的採樣權重
        
        含結節的切片有更高的權重
        
        Args:
            mask: 3D 遮罩 (Z, Y, X)
            min_nodule_pixels: 最小結節像素數
            
        Returns:
            權重數組 (Z,)
        """
        weights = np.ones(mask.shape[0])
        
        for z in range(mask.shape[0]):
            if np.sum(mask[z] > 0) >= min_nodule_pixels:
                weights[z] = self.positive_slice_weight
        
        # 歸一化
        weights /= weights.sum()
        
        return weights
    
    def sample_slice_indices(
        self,
        mask: np.ndarray,
        num_samples: int,
        weighted: bool = True
    ) -> List[int]:
        """
        採樣切片索引
        
        Args:
            mask: 3D 遮罩 (Z, Y, X)
            num_samples: 採樣數量
            weighted: 是否使用加權採樣
            
        Returns:
            採樣的切片索引列表
        """
        if weighted:
            weights = self.compute_slice_weights(mask)
            indices = np.random.choice(
                mask.shape[0],
                size=num_samples,
                replace=True,
                p=weights
            )
        else:
            indices = np.random.choice(
                mask.shape[0],
                size=num_samples,
                replace=True
            )
        
        return indices.tolist()


if __name__ == "__main__":
    # 測試採樣器
    np.random.seed(42)
    
    # 創建測試資料
    volume_shape = (256, 256)
    nodule_mask = np.zeros(volume_shape)
    nodule_mask[100:120, 100:120] = 1  # 模擬一個結節
    lung_mask = np.zeros(volume_shape)
    lung_mask[50:200, 30:230] = 1  # 模擬肺野
    
    # 測試 PatchSampler
    sampler = PatchSampler(patch_size=64)
    patches = sampler.sample_patches_for_slice(
        volume_shape,
        nodule_mask,
        lung_mask,
        num_patches=10
    )
    
    print("採樣結果:")
    for i, (y1, y2, x1, x2, sample_type) in enumerate(patches):
        print(f"  Patch {i}: ({y1}, {y2}, {x1}, {x2}) - {sample_type}")
