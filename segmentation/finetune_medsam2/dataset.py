#!/usr/bin/env python3
"""
資料集模組
提供胸部CT腫瘤資料集的載入與處理
支援多種分割方法生成 Ground Truth（區域生長、閾值分割、分水嶺等）
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import pandas as pd
from skimage import measure, draw
from scipy import ndimage
from skimage import morphology, segmentation
from tqdm import tqdm


class NoduleSegmenter:
    """
    結節分割器
    
    提供多種分割方法生成 Ground Truth 遮罩，比簡單圓形更準確
    
    Args:
        method: 分割方法 ('sphere', 'threshold', 'region_growing', 'watershed', 'adaptive')
        
    Note:
        'adaptive' 方法會根據結節特性自動選擇最佳分割策略，並避免 fallback 到簡單圓形
    """
    
    def __init__(self, method: str = 'region_growing'):
        self.method = method
        self.logger = logging.getLogger(__name__)
        self._fallback_count = 0  # 追蹤 fallback 次數
        self._total_count = 0
    
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
        elif self.method == 'watershed':
            local_mask = self._watershed_segmentation(local_volume, local_center, local_radius)
        elif self.method == 'adaptive':
            local_mask = self._adaptive_segmentation(local_volume, local_center, local_radius)
        else:
            local_mask = self._create_ellipsoid_mask(local_volume.shape, local_center, local_radius)
        
        self._total_count += 1
        
        # 將局部遮罩放回全局遮罩
        full_mask = np.zeros(volume.shape, dtype=bool)
        full_mask[z_min:z_max, y_min:y_max, x_min:x_max] = local_mask
        
        return full_mask
    
    def _create_ellipsoid_mask(self, shape, center, radius):
        """創建橢球體遮罩"""
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        
        # 避免除以零
        rz = max(radius[0], 0.5)
        ry = max(radius[1], 0.5)
        rx = max(radius[2], 0.5)
        
        distance = ((z - center[0]) / rz) ** 2 + \
                   ((y - center[1]) / ry) ** 2 + \
                   ((x - center[2]) / rx) ** 2
        
        return distance <= 1.0
    
    def _create_refined_ellipsoid(self, volume, center, radius):
        """
        創建經過形態學精修的橢球體遮罩
        
        相比簡單橢球體，這個方法會：
        1. 根據 HU 值微調邊界
        2. 加入隨機擾動使邊緣不那麼規則
        3. 應用輕度形態學操作使邊界更自然
        """
        # 先創建基礎橢球體
        base_mask = self._create_ellipsoid_mask(volume.shape, center, radius)
        
        # 獲取橢球體內的 HU 統計
        if np.sum(base_mask) == 0:
            return base_mask
            
        core_values = volume[base_mask]
        mean_hu = np.mean(core_values)
        
        # 創建稍大的區域用於精修
        expanded_mask = self._create_ellipsoid_mask(volume.shape, center, radius * 1.15)
        
        # 基於 HU 值的邊界調整：保留與核心區域 HU 相近的體素
        hu_tolerance = max(np.std(core_values) * 1.5, 80)
        hu_refined = (volume >= mean_hu - hu_tolerance) & (volume <= mean_hu + hu_tolerance)
        
        # 結合橢球體和 HU 精修
        refined_mask = base_mask.copy()
        
        # 在擴展區域內，如果 HU 符合條件則加入
        expansion_zone = expanded_mask & ~base_mask
        refined_mask = refined_mask | (expansion_zone & hu_refined)
        
        # 在基礎區域內，如果 HU 差太多則移除邊緣
        # 只移除邊緣像素，保留核心
        core_mask = self._create_ellipsoid_mask(volume.shape, center, radius * 0.7)
        edge_zone = base_mask & ~core_mask
        refined_mask = core_mask | (edge_zone & hu_refined)
        
        # 輕度形態學平滑
        if np.sum(refined_mask) > 0:
            # 閉運算填充小孔洞
            refined_mask = ndimage.binary_closing(refined_mask, iterations=1)
            # 確保結果不會比原始橢球體大太多
            refined_mask = refined_mask & expanded_mask
        
        # 如果精修後太小，返回原始橢球體
        if np.sum(refined_mask) < np.sum(base_mask) * 0.5:
            return base_mask
            
        self._fallback_count += 1  # 記錄使用了精修橢球體
        return refined_mask
    
    def _threshold_segmentation(self, volume, center, radius):
        """基於 HU 閾值的分割"""
        # 首先創建一個搜索區域（橢球體 * 1.2）
        search_mask = self._create_ellipsoid_mask(volume.shape, center, radius * 1.2)
        
        # 獲取結節中心附近的 HU 值來估計閾值
        core_mask = self._create_ellipsoid_mask(volume.shape, center, radius * 0.5)
        if np.sum(core_mask) > 0:
            core_values = volume[core_mask]
            # 結節通常具有較高的 HU 值（相對於肺實質）
            lower_threshold = max(np.percentile(core_values, 25), -800)
            upper_threshold = min(np.percentile(core_values, 95) + 200, 400)
        else:
            # 預設結節 HU 閾值範圍
            lower_threshold = -700
            upper_threshold = 300
        
        # 應用閾值
        threshold_mask = (volume >= lower_threshold) & (volume <= upper_threshold) & search_mask
        
        # 形態學操作清理
        if np.sum(threshold_mask) > 0:
            # 開運算去除小噪點
            threshold_mask = ndimage.binary_opening(threshold_mask, iterations=1)
            # 閉運算填充小孔洞
            threshold_mask = ndimage.binary_closing(threshold_mask, iterations=1)
            
            # 連通組件分析，保留最接近中心的組件
            labeled, num_features = ndimage.label(threshold_mask)
            if num_features > 1:
                center_int = tuple(int(c) for c in center)
                if 0 <= center_int[0] < volume.shape[0] and \
                   0 <= center_int[1] < volume.shape[1] and \
                   0 <= center_int[2] < volume.shape[2]:
                    target_label = labeled[center_int]
                    if target_label > 0:
                        threshold_mask = labeled == target_label
                    else:
                        # 找到最近的連通組件
                        best_label = 1
                        min_dist = float('inf')
                        for i in range(1, num_features + 1):
                            comp_coords = np.array(np.where(labeled == i)).T
                            if len(comp_coords) > 0:
                                comp_center = comp_coords.mean(axis=0)
                                dist = np.linalg.norm(comp_center - center)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_label = i
                        threshold_mask = labeled == best_label
        
        # 如果分割結果太小或為空，使用精修橢球體
        if np.sum(threshold_mask) < np.prod(radius) * 0.5:
            return self._create_refined_ellipsoid(volume, center, radius)
        
        return threshold_mask
    
    def _region_growing_segmentation(self, volume, center, radius):
        """區域生長分割"""
        # 獲取種子點的 HU 值
        center_int = tuple(int(c) for c in center)
        if not (0 <= center_int[0] < volume.shape[0] and 
                0 <= center_int[1] < volume.shape[1] and 
                0 <= center_int[2] < volume.shape[2]):
            return self._create_refined_ellipsoid(volume, center, radius)
        
        seed_value = volume[center_int]
        
        # 估計結節區域的 HU 統計
        core_mask = self._create_ellipsoid_mask(volume.shape, center, radius * 0.6)
        if np.sum(core_mask) > 0:
            core_values = volume[core_mask]
            mean_hu = np.mean(core_values)
            std_hu = np.std(core_values)
            # 動態調整閾值範圍
            tolerance = max(std_hu * 2, 100)
        else:
            mean_hu = seed_value
            tolerance = 150
        
        # 創建搜索區域限制
        search_mask = self._create_ellipsoid_mask(volume.shape, center, radius * 1.5)
        
        # 簡單區域生長實現
        lower_bound = mean_hu - tolerance
        upper_bound = mean_hu + tolerance
        
        # 初始化遮罩
        mask = np.zeros(volume.shape, dtype=bool)
        visited = np.zeros(volume.shape, dtype=bool)
        
        # 使用堆疊進行區域生長
        stack = [center_int]
        visited[center_int] = True
        
        # 6-連通鄰居偏移
        neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        
        max_iterations = int(np.prod(radius) * 50)
        iteration = 0
        
        while stack and iteration < max_iterations:
            iteration += 1
            z, y, x = stack.pop()
            
            # 檢查是否在閾值範圍內且在搜索區域內
            if lower_bound <= volume[z, y, x] <= upper_bound and search_mask[z, y, x]:
                mask[z, y, x] = True
                
                # 添加鄰居
                for dz, dy, dx in neighbors:
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if (0 <= nz < volume.shape[0] and 
                        0 <= ny < volume.shape[1] and 
                        0 <= nx < volume.shape[2] and 
                        not visited[nz, ny, nx]):
                        visited[nz, ny, nx] = True
                        stack.append((nz, ny, nx))
        
        # 形態學處理
        if np.sum(mask) > 0:
            mask = ndimage.binary_closing(mask, iterations=2)
            mask = ndimage.binary_fill_holes(mask)
        
        # 如果分割結果太小，返回閾值分割結果
        if np.sum(mask) < np.prod(radius) * 0.3:
            return self._threshold_segmentation(volume, center, radius)
        
        return mask
    
    def _watershed_segmentation(self, volume, center, radius):
        """分水嶺分割"""
        try:
            # 創建搜索區域
            search_mask = self._create_ellipsoid_mask(volume.shape, center, radius * 1.5)
            
            # 獲取核心區域作為前景種子
            foreground = self._create_ellipsoid_mask(volume.shape, center, radius * 0.4)
            
            # 創建背景種子（遠離中心的區域）
            background = ~self._create_ellipsoid_mask(volume.shape, center, radius * 1.8)
            background = background & search_mask
            
            # 創建標記
            markers = np.zeros(volume.shape, dtype=np.int32)
            markers[foreground] = 2  # 前景
            markers[background] = 1  # 背景
            
            # 計算梯度
            gradient = ndimage.sobel(volume.astype(float))
            gradient = np.abs(gradient)
            
            # 只在搜索區域內進行分水嶺
            gradient_masked = gradient.copy()
            gradient_masked[~search_mask] = gradient.max()
            
            # 執行分水嶺
            labeled = segmentation.watershed(gradient_masked, markers)
            
            # 提取結節區域
            mask = labeled == 2
            
            # 後處理
            if np.sum(mask) > 0:
                mask = ndimage.binary_closing(mask, iterations=1)
                mask = ndimage.binary_fill_holes(mask)
            
            # 如果結果太小，使用精修橢球體
            if np.sum(mask) < np.prod(radius) * 0.3:
                return self._create_refined_ellipsoid(volume, center, radius)
            
            return mask
            
        except Exception as e:
            self.logger.warning(f"分水嶺分割失敗: {e}")
            return self._create_refined_ellipsoid(volume, center, radius)
    
    def _adaptive_segmentation(self, volume, center, radius):
        """
        自適應分割方法
        
        根據結節特性自動選擇最佳分割策略：
        1. 分析結節的 HU 分佈判斷類型（實性、部分實性、GGO）
        2. 根據類型選擇不同的分割參數
        3. 結合多種方法的結果
        4. 使用形態學後處理確保邊界平滑自然
        """
        center_int = tuple(int(c) for c in center)
        
        # 安全檢查
        if not (0 <= center_int[0] < volume.shape[0] and 
                0 <= center_int[1] < volume.shape[1] and 
                0 <= center_int[2] < volume.shape[2]):
            return self._create_refined_ellipsoid(volume, center, radius)
        
        # 獲取核心區域的 HU 統計來判斷結節類型
        core_mask = self._create_ellipsoid_mask(volume.shape, center, radius * 0.6)
        if np.sum(core_mask) == 0:
            return self._create_refined_ellipsoid(volume, center, radius)
            
        core_values = volume[core_mask]
        mean_hu = np.mean(core_values)
        std_hu = np.std(core_values)
        max_hu = np.max(core_values)
        
        # 判斷結節類型
        # 實性結節: 平均 HU > -100
        # 部分實性: -500 < 平均 HU <= -100
        # GGO: 平均 HU <= -500
        
        if mean_hu > -100:
            # 實性結節：使用區域生長（效果通常較好）
            nodule_type = 'solid'
            tolerance = max(std_hu * 2.5, 120)
        elif mean_hu > -500:
            # 部分實性：使用較寬鬆的閾值
            nodule_type = 'part_solid'
            tolerance = max(std_hu * 3, 150)
        else:
            # GGO：使用更寬鬆的閾值，因為邊界模糊
            nodule_type = 'ggo'
            tolerance = max(std_hu * 4, 200)
        
        # 創建搜索區域
        search_mask = self._create_ellipsoid_mask(volume.shape, center, radius * 1.4)
        
        # 動態閾值分割
        lower_bound = mean_hu - tolerance
        upper_bound = mean_hu + tolerance
        
        # 確保閾值在合理範圍
        lower_bound = max(lower_bound, -1000)
        upper_bound = min(upper_bound, 400)
        
        # 閾值分割
        threshold_mask = (volume >= lower_bound) & (volume <= upper_bound) & search_mask
        
        # 形態學處理
        if np.sum(threshold_mask) > 0:
            # 開運算去噪
            threshold_mask = ndimage.binary_opening(threshold_mask, iterations=1)
            # 閉運算填孔
            threshold_mask = ndimage.binary_closing(threshold_mask, iterations=2)
            threshold_mask = ndimage.binary_fill_holes(threshold_mask)
            
            # 連通組件分析
            labeled, num_features = ndimage.label(threshold_mask)
            if num_features > 1:
                # 找包含中心點的組件
                target_label = labeled[center_int]
                if target_label > 0:
                    threshold_mask = labeled == target_label
                else:
                    # 找最大且最接近中心的組件
                    best_label = 1
                    best_score = -float('inf')
                    for i in range(1, num_features + 1):
                        comp_mask = labeled == i
                        comp_size = np.sum(comp_mask)
                        comp_coords = np.array(np.where(comp_mask)).T
                        if len(comp_coords) > 0:
                            comp_center = comp_coords.mean(axis=0)
                            dist = np.linalg.norm(comp_center - center)
                            # 評分：大小重要，距離也重要
                            score = comp_size / (1 + dist)
                            if score > best_score:
                                best_score = score
                                best_label = i
                    threshold_mask = labeled == best_label
        
        # 檢查結果品質
        result_size = np.sum(threshold_mask)
        expected_size = np.prod(radius) * 4 / 3 * np.pi  # 球體體積近似
        
        # 結果太小：結合橢球體
        if result_size < expected_size * 0.25:
            # 使用精修橢球體作為基礎，但結合部分閾值結果
            base = self._create_ellipsoid_mask(volume.shape, center, radius)
            if np.sum(threshold_mask) > 0:
                # 結合：橢球體核心 + 閾值分割在擴展區的結果
                core = self._create_ellipsoid_mask(volume.shape, center, radius * 0.8)
                combined = core | (threshold_mask & search_mask)
                combined = ndimage.binary_closing(combined, iterations=1)
                return combined
            else:
                return self._create_refined_ellipsoid(volume, center, radius)
        
        # 結果太大：限制在搜索區域內
        if result_size > expected_size * 3:
            # 縮小到與橢球體的交集
            threshold_mask = threshold_mask & self._create_ellipsoid_mask(
                volume.shape, center, radius * 1.2
            )
        
        return threshold_mask
    
    def get_fallback_stats(self):
        """返回 fallback 統計"""
        return {
            'total': self._total_count,
            'fallback': self._fallback_count,
            'fallback_rate': self._fallback_count / max(self._total_count, 1) * 100
        }


class ChestTumorDataset(Dataset):
    """
    胸部CT腫瘤資料集 (LUNA16)
    
    載入 LUNA16 格式的 CT (.mhd) 和註釋 (annotations.csv)，自動提取 2D 切片和生成遮罩
    
    Args:
        data_dir: 資料集根目錄 (包含 subsetX 和 annotations.csv)
        patient_ids: 患者ID列表 (seriesuid)
        axis: 切片軸向 (0, 1, 或 2)
        transform: 資料增強函數
        cache_data: 是否緩存資料到記憶體
        segmentation_method: 分割方法 ('sphere', 'threshold', 'region_growing', 'watershed')
        min_nodule_diameter: 最小結節直徑 (mm)，過濾小於此值的結節 (預設 0 表示不過濾)
    
    病灶大小分類 (根據 Fleischner Society 指南):
        - micro: < 4mm (通常忽略)
        - small: 4-6mm (低風險)
        - medium: 6-8mm (中風險，需追蹤)
        - large: > 8mm (高風險，需進一步檢查)
    """
    
    # 病灶大小分類閾值 (mm)
    SIZE_THRESHOLDS = {
        'micro': 4.0,    # < 4mm
        'small': 6.0,    # 4-6mm
        'medium': 8.0,   # 6-8mm
        'large': float('inf')  # > 8mm
    }
    
    @staticmethod
    def classify_nodule_size(diameter_mm: float) -> str:
        """
        根據直徑分類結節大小
        
        Args:
            diameter_mm: 結節直徑 (mm)
            
        Returns:
            大小分類: 'micro', 'small', 'medium', 'large'
        """
        if diameter_mm < 4.0:
            return 'micro'
        elif diameter_mm < 6.0:
            return 'small'
        elif diameter_mm < 8.0:
            return 'medium'
        else:
            return 'large'
    
    def __init__(
        self, 
        data_dir: str,
        patient_ids: List[str],
        axis: int = 2,
        transform: Optional[Callable] = None,
        cache_data: bool = False,
        segmentation_method: str = 'region_growing',
        min_nodule_diameter: float = 0.0
    ):
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.axis = axis
        self.transform = transform
        self.cache_data = cache_data
        self.segmentation_method = segmentation_method
        self.min_nodule_diameter = min_nodule_diameter
        
        self.logger = logging.getLogger(__name__)
        
        # 初始化過濾統計（會在 _build_sample_index 中更新）
        self._filter_stats = {
            'patients_total': 0,
            'patients_kept': 0,
            'patients_filtered': 0,
            'nodules_total': 0,
            'nodules_kept': 0,
            'nodules_filtered': 0,
            'size_micro': 0,
            'size_small': 0,
            'size_medium': 0,
            'size_large': 0
        }
        
        # 初始化分割器
        self.segmenter = NoduleSegmenter(method=segmentation_method)
        self.logger.info(f"🔧 使用分割方法: {segmentation_method}")
        if min_nodule_diameter > 0:
            self.logger.info(f"🔍 過濾小於 {min_nodule_diameter}mm 的結節")
        
        # 載入註釋
        self.annotations_file = self.data_dir / "annotations.csv"
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"找不到註釋檔案: {self.annotations_file}")
        self.annotations = pd.read_csv(self.annotations_file)
        
        # 建立檔案索引 (patient_id -> file_path)
        self.patient_files = self._index_patient_files()
        
        # 建立所有切片的索引
        self.samples = []
        self._build_sample_index()
        
        # 緩存資料
        if cache_data:
            self.logger.info("🔄 緩存資料到記憶體...")
            self.cached_data = {}
            for idx in tqdm(range(len(self.samples)), desc="Caching"):
                self.cached_data[idx] = self._load_sample(idx)

    def _index_patient_files(self) -> Dict[str, Path]:
        """索引所有 subset 資料夾中的 .mhd 檔案"""
        patient_files = {}
        # 搜尋 subset0 到 subset9
        for i in range(10):
            subset_dir = self.data_dir / f"subset{i}"
            if not subset_dir.exists():
                continue
            for f in subset_dir.glob("*.mhd"):
                patient_files[f.stem] = f
        return patient_files
    
    def _build_sample_index(self):
        """建立所有有效切片的索引"""
        self.logger.info(f"📊 建立資料集索引 ({len(self.patient_ids)} 個患者)...")
        
        # 統計過濾的患者和結節
        patient_stats = {
            'total_input': len(self.patient_ids),  # 輸入的患者總數
            'no_ct_file': 0,        # 找不到 CT 檔案
            'no_nodules': 0,        # 無病灶（annotations 中沒有記錄）
            'has_nodules': 0,       # 有病灶的患者數
            'filtered': 0,          # 因小結節被過濾
            'kept': 0               # 最終保留
        }
        nodule_stats = {'total': 0, 'filtered': 0, 'kept': 0}
        size_distribution = {'micro': 0, 'small': 0, 'medium': 0, 'large': 0}
        
        for patient_id in tqdm(self.patient_ids, desc="Indexing"):
            if patient_id not in self.patient_files:
                patient_stats['no_ct_file'] += 1
                continue
                
            ct_file = self.patient_files[patient_id]
            
            # 獲取該患者的結節註釋
            patient_nodules = self.annotations[self.annotations['seriesuid'] == patient_id]
            if patient_nodules.empty:
                patient_stats['no_nodules'] += 1
                continue
            
            patient_stats['has_nodules'] += 1
            nodule_stats['total'] += len(patient_nodules)
            
            # ⭐ 新邏輯：過濾掉含有任何小於閾值結節的患者
            # 如果患者有任何結節小於 min_nodule_diameter，則跳過整個患者
            if self.min_nodule_diameter > 0:
                min_diameter_in_patient = patient_nodules['diameter_mm'].min()
                
                if min_diameter_in_patient < self.min_nodule_diameter:
                    # 該患者含有小結節，整個跳過
                    patient_stats['filtered'] += 1
                    nodule_stats['filtered'] += len(patient_nodules)
                    continue
                
                # 該患者所有結節都 >= 閾值，保留
                patient_stats['kept'] += 1
                nodule_stats['kept'] += len(patient_nodules)
            else:
                patient_stats['kept'] += 1
                nodule_stats['kept'] += len(patient_nodules)
            
            # 統計大小分佈（只統計保留的結節）
            for _, row in patient_nodules.iterrows():
                size_class = self.classify_nodule_size(row['diameter_mm'])
                size_distribution[size_class] += 1
            
            try:
                # 讀取影像資訊 (不讀取像素資料)
                reader = sitk.ImageFileReader()
                reader.SetFileName(str(ct_file))
                reader.ReadImageInformation()
                
                size = reader.GetSize()      # (x, y, z)
                spacing = reader.GetSpacing() # (sx, sy, sz)
                origin = reader.GetOrigin()   # (ox, oy, oz)
                
                # 找出包含結節的切片
                valid_slices = set()
                
                for _, row in patient_nodules.iterrows():
                    center_world = np.array([row['coordX'], row['coordY'], row['coordZ']])
                    diameter = row['diameter_mm']
                    radius = diameter / 2.0
                    
                    # 計算結節在 Z 軸 (或其他軸) 的範圍
                    # 這裡簡化計算，假設沒有旋轉 (LUNA16 通常是這樣)
                    # 轉換世界座標到體素座標: index = (world - origin) / spacing
                    
                    center_idx = (center_world - np.array(origin)) / np.array(spacing)
                    radius_idx = radius / np.array(spacing)
                    
                    if self.axis == 2: # Axial (Z-axis)
                        min_idx = int(np.floor(center_idx[2] - radius_idx[2]))
                        max_idx = int(np.ceil(center_idx[2] + radius_idx[2]))
                        max_limit = size[2]
                    elif self.axis == 1: # Coronal (Y-axis)
                        min_idx = int(np.floor(center_idx[1] - radius_idx[1]))
                        max_idx = int(np.ceil(center_idx[1] + radius_idx[1]))
                        max_limit = size[1]
                    else: # Sagittal (X-axis)
                        min_idx = int(np.floor(center_idx[0] - radius_idx[0]))
                        max_idx = int(np.ceil(center_idx[0] + radius_idx[0]))
                        max_limit = size[0]
                    
                    # 確保在影像範圍內
                    min_idx = max(0, min_idx)
                    max_idx = min(max_limit - 1, max_idx)
                    
                    for s in range(min_idx, max_idx + 1):
                        valid_slices.add(s)
                
                # 添加到樣本列表，包含直徑資訊
                max_diameter = patient_nodules['diameter_mm'].max()
                for slice_idx in valid_slices:
                    self.samples.append({
                        'patient_id': patient_id,
                        'ct_file': ct_file,
                        'slice_index': slice_idx,
                        'nodules': patient_nodules,  # 保存註釋以便生成 mask
                        'max_diameter_mm': max_diameter,  # 該切片最大結節直徑
                        'size_class': self.classify_nodule_size(max_diameter)  # 大小分類
                    })
            
            except Exception as e:
                self.logger.warning(f"⚠️ 跳過患者 {patient_id}: {e}")
                continue
        
        # 保存統計資訊到類別屬性
        self._filter_stats = {
            'patients_input': patient_stats['total_input'],      # 輸入總數
            'patients_no_ct': patient_stats['no_ct_file'],       # 無 CT 檔案
            'patients_no_nodules': patient_stats['no_nodules'],  # 無病灶
            'patients_has_nodules': patient_stats['has_nodules'],# 有病灶
            'patients_filtered': patient_stats['filtered'],      # 因小結節被過濾
            'patients_kept': patient_stats['kept'],              # 最終保留
            'patients_total': patient_stats['has_nodules'],      # 向後相容（有病灶的患者）
            'nodules_total': nodule_stats['total'],
            'nodules_kept': nodule_stats['kept'],
            'nodules_filtered': nodule_stats['filtered'],
            'size_micro': size_distribution['micro'],
            'size_small': size_distribution['small'],
            'size_medium': size_distribution['medium'],
            'size_large': size_distribution['large']
        }
        
        # 輸出統計資訊（簡化版，詳細版在 main.py 輸出）
        self.logger.info(f"✅ 找到 {len(self.samples)} 個有效切片 (保留 {patient_stats['kept']}/{patient_stats['has_nodules']} 有病灶患者)")
    
    def get_filter_stats(self) -> Dict:
        """
        獲取過濾統計資訊
        
        Returns:
            Dict: 包含以下鍵值:
                - patients_input: 輸入的患者總數
                - patients_no_ct: 找不到 CT 檔案的患者數
                - patients_no_nodules: 無病灶的患者數（annotations 中沒有記錄）
                - patients_has_nodules: 有病灶的患者數
                - patients_filtered: 因小結節被過濾的患者數
                - patients_kept: 最終保留的患者數
                - patients_total: 有病灶的患者數（向後相容）
                - nodules_total: 結節總數
                - nodules_kept: 保留的結節數
                - nodules_filtered: 過濾掉的結節數
                - size_micro: micro 類別結節數 (<4mm)
                - size_small: small 類別結節數 (4-6mm)
                - size_medium: medium 類別結節數 (6-8mm)
                - size_large: large 類別結節數 (>8mm)
        """
        return self._filter_stats.copy()
    
    def _load_sample(self, idx: int) -> Dict:
        """載入單個樣本"""
        sample_info = self.samples[idx]
        ct_file = sample_info['ct_file']
        slice_idx = sample_info['slice_index']
        patient_nodules = sample_info['nodules']
        
        # 讀取 CT 影像
        ct_image = sitk.ReadImage(str(ct_file))
        ct_array = sitk.GetArrayFromImage(ct_image) # (z, y, x)
        
        # 提取切片
        if self.axis == 2:
            ct_slice = ct_array[slice_idx, :, :]
        elif self.axis == 1:
            ct_slice = ct_array[:, slice_idx, :]
        else:
            ct_slice = ct_array[:, :, slice_idx]
            
        # 生成 Mask（使用新的分割方法）
        mask_slice = np.zeros_like(ct_slice, dtype=np.uint8)
        
        origin = np.array(ct_image.GetOrigin())    # (x, y, z)
        spacing = np.array(ct_image.GetSpacing())  # (sx, sy, sz)
        
        for _, row in patient_nodules.iterrows():
            center_world = np.array([row['coordX'], row['coordY'], row['coordZ']])
            diameter = row['diameter_mm']
            radius_mm = diameter / 2.0
            
            # 轉換世界座標到體素座標
            center_voxel = (center_world - origin) / spacing  # (x, y, z)
            
            # 計算各軸的體素半徑
            radius_voxel = np.array([
                radius_mm / spacing[0],  # rx
                radius_mm / spacing[1],  # ry
                radius_mm / spacing[2]   # rz
            ])
            
            # 判斷該結節是否在此切片上
            if self.axis == 2:
                dist_to_slice_voxel = abs(center_voxel[2] - slice_idx)
                in_range = dist_to_slice_voxel < radius_voxel[2] + 0.5
            elif self.axis == 1:
                dist_to_slice_voxel = abs(center_voxel[1] - slice_idx)
                in_range = dist_to_slice_voxel < radius_voxel[1] + 0.5
            else:
                dist_to_slice_voxel = abs(center_voxel[0] - slice_idx)
                in_range = dist_to_slice_voxel < radius_voxel[0] + 0.5
            
            if not in_range:
                continue
            
            try:
                # 生成 3D 分割遮罩
                nodule_mask_3d = self.segmenter.generate_3d_mask(
                    ct_array, center_voxel, radius_voxel, spacing
                )
                
                # 提取對應的 2D 切片
                if self.axis == 2:
                    nodule_mask_2d = nodule_mask_3d[slice_idx, :, :]
                elif self.axis == 1:
                    nodule_mask_2d = nodule_mask_3d[:, slice_idx, :]
                else:
                    nodule_mask_2d = nodule_mask_3d[:, :, slice_idx]
                
                # 合併到總遮罩
                mask_slice = np.maximum(mask_slice, nodule_mask_2d.astype(np.uint8))
                
            except Exception as e:
                self.logger.warning(f"⚠️ 分割失敗，使用圓形後備: {e}")
                # 後備方案：使用圓形
                self._draw_circle_mask(
                    mask_slice, ct_image, center_world, radius_mm, 
                    slice_idx, spacing
                )

        # CT 值裁剪與標準化
        # 將 CT 值裁剪到 [-1000, 800] HU 範圍，涵蓋肺實質到軟組織/鈣化
        hu_min, hu_max = -1000, 800
        ct_clipped = np.clip(ct_slice, hu_min, hu_max)
        
        # 標準化到 0-255 範圍
        ct_normalized = ((ct_clipped - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)
        
        # 堆疊成 3 通道 RGB (MedSAM2 需要 3 通道輸入)
        ct_rgb = np.stack([ct_normalized, ct_normalized, ct_normalized], axis=-1)
        
        # 提取 bounding boxes
        bboxes = self._extract_bboxes(mask_slice)
        
        return {
            'image': ct_rgb,
            'mask': mask_slice,
            'bboxes': bboxes,
            'patient_id': sample_info['patient_id'],
            'slice_index': slice_idx
        }
    
    def _draw_circle_mask(
        self, 
        mask_slice: np.ndarray, 
        ct_image: sitk.Image, 
        center_world: np.ndarray, 
        radius_mm: float, 
        slice_idx: int, 
        spacing: np.ndarray
    ):
        """
        後備方案：在遮罩上繪製圓形（原始方法）
        
        當進階分割方法失敗時使用
        """
        # 轉換中心點到體素座標
        center_idx = ct_image.TransformPhysicalPointToIndex(tuple(center_world))  # (x, y, z)
        
        # 判斷該結節是否在此切片上
        if self.axis == 2:
            dist_to_slice = abs(center_idx[2] - slice_idx) * spacing[2]
        elif self.axis == 1:
            dist_to_slice = abs(center_idx[1] - slice_idx) * spacing[1]
        else:
            dist_to_slice = abs(center_idx[0] - slice_idx) * spacing[0]
        
        if dist_to_slice < radius_mm:
            # 計算切片上的圓半徑
            slice_radius_mm = np.sqrt(radius_mm**2 - dist_to_slice**2)
            
            # 在 mask 上畫圓
            if self.axis == 2:  # Axial: (y, x)
                cy, cx = center_idx[1], center_idx[0]
                pixel_radius = slice_radius_mm / np.mean([spacing[0], spacing[1]])
            elif self.axis == 1:  # Coronal: (z, x)
                cy, cx = center_idx[2], center_idx[0]
                pixel_radius = slice_radius_mm / np.mean([spacing[0], spacing[2]])
            else:  # Sagittal: (z, y)
                cy, cx = center_idx[2], center_idx[1]
                pixel_radius = slice_radius_mm / np.mean([spacing[1], spacing[2]])
            
            rr, cc = draw.disk((cy, cx), pixel_radius, shape=mask_slice.shape)
            mask_slice[rr, cc] = 1
    
    def _extract_bboxes(self, mask: np.ndarray) -> np.ndarray:
        """從遮罩提取 bounding boxes"""
        if mask.sum() == 0:
            return np.array([[0, 0, 1, 1]])  # 空遮罩
        
        labeled_mask = measure.label(mask > 0)
        regions = measure.regionprops(labeled_mask)
        
        bboxes = []
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            bboxes.append([minc, minr, maxc, maxr])  # [x1, y1, x2, y2]
        
        return np.array(bboxes)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        # 從緩存或磁碟載入
        if self.cache_data:
            data = self.cached_data[idx].copy()
        else:
            data = self._load_sample(idx)
        
        # 資料增強
        if self.transform:
            data = self.transform(data)
        
        # 轉換為 tensor
        image = torch.from_numpy(data['image']).permute(2, 0, 1).float()  # [3, H, W]
        mask = torch.from_numpy(data['mask']).unsqueeze(0).float()  # [1, H, W]
        bboxes = torch.from_numpy(data['bboxes']).float()  # [N, 4]
        
        return {
            'image': image,
            'mask': mask,
            'bboxes': bboxes,
            'patient_id': data['patient_id'],
            'slice_index': data['slice_index']
        }


class DataAugmentation:
    """
    資料增強類別
    
    提供隨機旋轉、翻轉、gamma 調整等增強方法
    
    Args:
        rotation_prob: 旋轉機率
        flip_prob: 翻轉機率
        gamma_prob: Gamma 調整機率
        bbox_shift_limit: Bounding Box 擾動範圍 (pixels)
        elastic_prob: 彈性變形機率
        noise_prob: 高斯噪音機率
        contrast_prob: 對比度調整機率
    """
    
    def __init__(
        self,
        rotation_prob: float = 0.5,
        flip_prob: float = 0.5,
        gamma_prob: float = 0.3,
        bbox_shift_limit: int = 10,
        elastic_prob: float = 0.3,
        noise_prob: float = 0.2,
        contrast_prob: float = 0.3,
        scale_prob: float = 0.3
    ):
        self.rotation_prob = rotation_prob
        self.flip_prob = flip_prob
        self.gamma_prob = gamma_prob
        self.bbox_shift_limit = bbox_shift_limit
        self.elastic_prob = elastic_prob
        self.noise_prob = noise_prob
        self.contrast_prob = contrast_prob
        self.scale_prob = scale_prob
    
    def __call__(self, data: Dict) -> Dict:
        """
        應用資料增強
        
        Args:
            data: 包含 'image' 和 'mask' 的字典
            
        Returns:
            增強後的資料
        """
        import random
        
        image = data['image']  # [H, W, 3]
        mask = data['mask']    # [H, W]
        bboxes = data['bboxes']  # [N, 4]
        
        # 隨機旋轉 (90, 180, 270 度)
        if random.random() < self.rotation_prob:
            k = random.choice([1, 2, 3])  # 旋轉次數
            image = np.rot90(image, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
            # TODO: 旋轉 bboxes (較複雜，暫時跳過)
        
        # 隨機水平翻轉
        if random.random() < self.flip_prob:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
            # 調整 bboxes
            if len(bboxes) > 0:
                w = image.shape[1]
                bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        
        # 隨機垂直翻轉
        if random.random() < self.flip_prob:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
            # 調整 bboxes
            if len(bboxes) > 0:
                h = image.shape[0]
                bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]
        
        # Gamma 調整
        if random.random() < self.gamma_prob:
            gamma = random.uniform(0.8, 1.2)
            image = np.power(image / 255.0, gamma) * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 對比度調整 (CLAHE-like)
        if random.random() < self.contrast_prob:
            factor = random.uniform(0.8, 1.3)
            mean_val = np.mean(image)
            image = (image - mean_val) * factor + mean_val
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 高斯噪音
        if random.random() < self.noise_prob:
            noise_std = random.uniform(3, 10)
            noise = np.random.normal(0, noise_std, image.shape)
            image = image.astype(np.float32) + noise
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 隨機縮放 (Scale Augmentation)
        if random.random() < self.scale_prob:
            scale = random.uniform(0.9, 1.1)
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            from skimage.transform import resize
            # 縮放影像
            if len(image.shape) == 3:
                scaled_image = resize(image, (new_h, new_w, image.shape[2]), 
                                      preserve_range=True, anti_aliasing=True).astype(np.uint8)
            else:
                scaled_image = resize(image, (new_h, new_w), 
                                      preserve_range=True, anti_aliasing=True).astype(np.uint8)
            
            # 縮放遮罩
            scaled_mask = resize(mask.astype(np.float32), (new_h, new_w), 
                                 preserve_range=True, order=0).astype(mask.dtype)
            
            # 裁剪或填充回原始大小
            if scale > 1:
                # 中心裁剪
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                image = scaled_image[start_h:start_h+h, start_w:start_w+w]
                mask = scaled_mask[start_h:start_h+h, start_w:start_w+w]
            else:
                # 中心填充
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                if len(scaled_image.shape) == 3:
                    image = np.zeros((h, w, scaled_image.shape[2]), dtype=np.uint8)
                else:
                    image = np.zeros((h, w), dtype=np.uint8)
                mask = np.zeros((h, w), dtype=mask.dtype)
                image[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled_image
                mask[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled_mask
            
            # 調整 bboxes
            if len(bboxes) > 0:
                if scale > 1:
                    bboxes = (bboxes * scale) - np.array([start_w, start_h, start_w, start_h])
                else:
                    bboxes = (bboxes * scale) + np.array([pad_w, pad_h, pad_w, pad_h])
                bboxes = np.clip(bboxes, 0, [w, h, w, h])
            
        # Bounding Box 擾動 (Jitter)
        # 模擬使用者或檢測器給出的不完美提示
        if len(bboxes) > 0 and self.bbox_shift_limit > 0:
            h, w = image.shape[:2]
            noise = np.random.randint(-self.bbox_shift_limit, self.bbox_shift_limit, size=bboxes.shape)
            bboxes = bboxes + noise
            
            # 確保不超出邊界
            bboxes[:, 0] = np.clip(bboxes[:, 0], 0, w) # x1
            bboxes[:, 1] = np.clip(bboxes[:, 1], 0, h) # y1
            bboxes[:, 2] = np.clip(bboxes[:, 2], 0, w) # x2
            bboxes[:, 3] = np.clip(bboxes[:, 3], 0, h) # y2
            
            # 確保 x2 > x1, y2 > y1
            # 如果擾動導致 box 無效，恢復原狀或設為最小尺寸
            invalid_mask = (bboxes[:, 2] <= bboxes[:, 0]) | (bboxes[:, 3] <= bboxes[:, 1])
            if invalid_mask.any():
                # 簡單處理：對於無效的 box，恢復為原始 box (這裡需要原始 box，但為了簡化，我們只確保最小 1px)
                bboxes[invalid_mask, 2] = bboxes[invalid_mask, 0] + 1
                bboxes[invalid_mask, 3] = bboxes[invalid_mask, 1] + 1
        
        data['image'] = image
        data['mask'] = mask
        data['bboxes'] = bboxes
        
        return data


class LNDbDataset(Dataset):
    """
    LNDb (Lung Nodule Database) 資料集
    
    載入 LNDb 格式的 CT (.mhd) 和專家分割遮罩，自動提取 2D 切片
    
    **優勢**: LNDb 提供真實的專家分割遮罩（不是圓形估計），可以獲得更好的訓練效果
    
    Args:
        data_dir: LNDb 資料集根目錄 (包含 data0~data5, masks, trainset_csv)
        patient_ids: 患者ID列表 (LNDb ID, 如 1, 2, 3 或 'LNDb-0001')
        axis: 切片軸向 (0=sagittal, 1=coronal, 2=axial)
        transform: 資料增強函數
        cache_data: 是否緩存資料到記憶體
        rad_id: 使用哪位放射科醫師的標註 (1, 2, 3, 或 'consensus' 使用多數投票)
        min_nodule_diameter: 最小結節直徑 (mm)，過濾小於此值的結節 (預設 0 表示不過濾)
    
    LNDb 資料結構:
        LNDb/
        ├── data0~data5/          # CT 掃描 (.mhd + .raw)
        ├── masks/masks/          # 專家分割遮罩 (.mhd + .raw)
        │   ├── LNDb-0001_rad1.mhd
        │   ├── LNDb-0001_rad2.mhd
        │   └── LNDb-0001_rad3.mhd
        └── trainset_csv/
            ├── trainNodules.csv      # 個別醫師標註
            └── trainNodules_gt.csv   # 融合標註 (GT)
    """
    
    # 病灶大小分類閾值 (mm)
    SIZE_THRESHOLDS = {
        'micro': 4.0,
        'small': 6.0,
        'medium': 8.0,
        'large': float('inf')
    }
    
    @staticmethod
    def classify_nodule_size(diameter_mm: float) -> str:
        """根據直徑分類結節大小"""
        if diameter_mm < 4.0:
            return 'micro'
        elif diameter_mm < 6.0:
            return 'small'
        elif diameter_mm < 8.0:
            return 'medium'
        else:
            return 'large'
    
    def __init__(
        self, 
        data_dir: str,
        patient_ids: List[str],
        axis: int = 2,
        transform: Optional[Callable] = None,
        cache_data: bool = False,
        rad_id: str = 'consensus',
        min_nodule_diameter: float = 0.0
    ):
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.axis = axis
        self.transform = transform
        self.cache_data = cache_data
        self.rad_id = rad_id
        self.min_nodule_diameter = min_nodule_diameter
        
        self.logger = logging.getLogger(__name__)
        
        # 初始化過濾統計
        self._filter_stats = {
            'patients_input': 0, 'patients_no_ct': 0, 'patients_no_nodules': 0,
            'patients_has_nodules': 0, 'patients_filtered': 0, 'patients_kept': 0,
            'patients_total': 0, 'nodules_total': 0, 'nodules_kept': 0, 'nodules_filtered': 0,
            'size_micro': 0, 'size_small': 0, 'size_medium': 0, 'size_large': 0
        }
        
        self.logger.info(f"🔧 使用 LNDb 資料集 (專家分割遮罩)")
        self.logger.info(f"🔧 放射科醫師選擇: {rad_id}")
        if min_nodule_diameter > 0:
            self.logger.info(f"🔍 過濾小於 {min_nodule_diameter}mm 的結節")
        
        # 載入標註
        self._load_annotations()
        
        # 建立檔案索引
        self._index_files()
        
        # 建立樣本索引
        self.samples = []
        self._build_sample_index()
        
        # 緩存資料
        if cache_data:
            self.logger.info("🔄 緩存資料到記憶體...")
            self.cached_data = {}
            for idx in tqdm(range(len(self.samples)), desc="Caching"):
                self.cached_data[idx] = self._load_sample(idx)
    
    def _load_annotations(self):
        """載入 LNDb 標註資料"""
        csv_dir = self.data_dir / 'trainset_csv'
        
        # 載入融合標註 (GT)
        nodules_gt_path = csv_dir / 'trainNodules_gt.csv'
        if nodules_gt_path.exists():
            self.nodules_gt_df = pd.read_csv(nodules_gt_path)
            self.logger.info(f"載入 {len(self.nodules_gt_df)} 個融合標註 (trainNodules_gt.csv)")
        else:
            raise FileNotFoundError(f"找不到 GT 標註檔案: {nodules_gt_path}")
        
        # 載入個別醫師標註
        nodules_path = csv_dir / 'trainNodules.csv'
        if nodules_path.exists():
            self.nodules_df = pd.read_csv(nodules_path)
        else:
            self.nodules_df = None
    
    def _index_files(self):
        """索引所有 CT 和遮罩檔案"""
        # 索引 CT 檔案
        self.ct_files = {}  # {lndb_id: path}
        for i in range(6):
            data_dir = self.data_dir / f'data{i}'
            if data_dir.exists():
                for mhd_file in data_dir.glob('LNDb-*.mhd'):
                    lndb_id = int(mhd_file.stem.split('-')[1])
                    self.ct_files[lndb_id] = mhd_file
        
        # 索引遮罩檔案
        self.mask_files = {}  # {lndb_id: {rad_id: path}}
        mask_dir = self.data_dir / 'masks' / 'masks'
        if not mask_dir.exists():
            mask_dir = self.data_dir / 'masks'
        
        if mask_dir.exists():
            for mhd_file in mask_dir.glob('LNDb-*_rad*.mhd'):
                parts = mhd_file.stem.split('_')
                lndb_id = int(parts[0].split('-')[1])
                rad_id = int(parts[1].replace('rad', ''))
                
                if lndb_id not in self.mask_files:
                    self.mask_files[lndb_id] = {}
                self.mask_files[lndb_id][rad_id] = mhd_file
        
        self.logger.info(f"找到 {len(self.ct_files)} 個 CT 掃描, {len(self.mask_files)} 個有遮罩")
    
    def _parse_patient_id(self, patient_id) -> int:
        """解析患者 ID 為整數"""
        if isinstance(patient_id, int):
            return patient_id
        if isinstance(patient_id, str):
            if patient_id.startswith('LNDb-'):
                return int(patient_id.split('-')[1])
            return int(patient_id)
        return int(patient_id)
    
    def _build_sample_index(self):
        """建立所有有效切片的索引"""
        self.logger.info(f"📊 建立資料集索引 ({len(self.patient_ids)} 個患者)...")
        
        patient_stats = {
            'total_input': len(self.patient_ids),
            'no_ct_file': 0, 'no_mask': 0, 'no_nodules': 0,
            'has_nodules': 0, 'filtered': 0, 'kept': 0
        }
        nodule_stats = {'total': 0, 'filtered': 0, 'kept': 0}
        size_distribution = {'micro': 0, 'small': 0, 'medium': 0, 'large': 0}
        
        for patient_id in tqdm(self.patient_ids, desc="Indexing"):
            lndb_id = self._parse_patient_id(patient_id)
            
            # 檢查 CT 檔案
            if lndb_id not in self.ct_files:
                patient_stats['no_ct_file'] += 1
                continue
            
            # 檢查遮罩
            if lndb_id not in self.mask_files:
                patient_stats['no_mask'] += 1
                continue
            
            ct_file = self.ct_files[lndb_id]
            
            # 獲取該患者的結節標註
            patient_nodules = self.nodules_gt_df[self.nodules_gt_df['LNDbID'] == lndb_id]
            if patient_nodules.empty:
                patient_stats['no_nodules'] += 1
                continue
            
            patient_stats['has_nodules'] += 1
            nodule_stats['total'] += len(patient_nodules)
            
            # 計算結節直徑 (從體積估計: V = 4/3 * π * r³ => d = 2 * (3V/4π)^(1/3))
            if 'Volume' in patient_nodules.columns:
                volumes = patient_nodules['Volume'].values
                diameters = 2 * np.power(3 * volumes / (4 * np.pi), 1/3)
                min_diameter = diameters.min()
            else:
                min_diameter = float('inf')  # 無法判斷，不過濾
                diameters = np.array([])
            
            # 過濾小結節
            if self.min_nodule_diameter > 0 and min_diameter < self.min_nodule_diameter:
                patient_stats['filtered'] += 1
                nodule_stats['filtered'] += len(patient_nodules)
                continue
            
            patient_stats['kept'] += 1
            nodule_stats['kept'] += len(patient_nodules)
            
            # 統計大小分佈
            for d in diameters:
                size_class = self.classify_nodule_size(d)
                size_distribution[size_class] += 1
            
            try:
                # 讀取遮罩以確定有結節的切片
                mask_volume = self._load_mask_volume(lndb_id)
                if mask_volume is None:
                    continue
                
                # 找出有結節的切片
                if self.axis == 2:  # Axial
                    slice_sums = mask_volume.sum(axis=(1, 2))
                elif self.axis == 1:  # Coronal
                    slice_sums = mask_volume.sum(axis=(0, 2))
                else:  # Sagittal
                    slice_sums = mask_volume.sum(axis=(0, 1))
                
                valid_slices = np.where(slice_sums > 0)[0]
                
                # 計算最大結節直徑
                max_diameter = diameters.max() if len(diameters) > 0 else 10.0
                
                for slice_idx in valid_slices:
                    self.samples.append({
                        'lndb_id': lndb_id,
                        'ct_file': ct_file,
                        'slice_index': int(slice_idx),
                        'max_diameter_mm': max_diameter,
                        'size_class': self.classify_nodule_size(max_diameter)
                    })
                
            except Exception as e:
                self.logger.warning(f"⚠️ 跳過患者 LNDb-{lndb_id:04d}: {e}")
                continue
        
        # 保存統計資訊
        self._filter_stats = {
            'patients_input': patient_stats['total_input'],
            'patients_no_ct': patient_stats['no_ct_file'] + patient_stats['no_mask'],
            'patients_no_nodules': patient_stats['no_nodules'],
            'patients_has_nodules': patient_stats['has_nodules'],
            'patients_filtered': patient_stats['filtered'],
            'patients_kept': patient_stats['kept'],
            'patients_total': patient_stats['has_nodules'],
            'nodules_total': nodule_stats['total'],
            'nodules_kept': nodule_stats['kept'],
            'nodules_filtered': nodule_stats['filtered'],
            'size_micro': size_distribution['micro'],
            'size_small': size_distribution['small'],
            'size_medium': size_distribution['medium'],
            'size_large': size_distribution['large']
        }
        
        self.logger.info(f"✅ 找到 {len(self.samples)} 個有效切片 "
                        f"(保留 {patient_stats['kept']}/{patient_stats['has_nodules']} 有病灶患者)")
    
    def _load_mask_volume(self, lndb_id: int) -> Optional[np.ndarray]:
        """載入遮罩體積"""
        if lndb_id not in self.mask_files:
            return None
        
        available_rads = self.mask_files[lndb_id]
        
        if self.rad_id == 'consensus':
            # 多數投票：至少 2/3 的醫師標記
            masks = []
            for rad_id, mask_path in available_rads.items():
                try:
                    itk_mask = sitk.ReadImage(str(mask_path))
                    mask_arr = sitk.GetArrayFromImage(itk_mask)
                    masks.append(mask_arr > 0)
                except:
                    continue
            
            if len(masks) == 0:
                return None
            
            # 多數投票
            combined = np.sum(masks, axis=0)
            threshold = max(1, len(masks) // 2)  # 至少一半
            return (combined >= threshold).astype(np.uint8)
        
        else:
            # 使用指定醫師的標註
            rad_id = int(self.rad_id) if isinstance(self.rad_id, str) else self.rad_id
            if rad_id in available_rads:
                itk_mask = sitk.ReadImage(str(available_rads[rad_id]))
                return (sitk.GetArrayFromImage(itk_mask) > 0).astype(np.uint8)
            elif available_rads:
                # 使用第一個可用的
                first_rad = list(available_rads.keys())[0]
                itk_mask = sitk.ReadImage(str(available_rads[first_rad]))
                return (sitk.GetArrayFromImage(itk_mask) > 0).astype(np.uint8)
            return None
    
    def get_filter_stats(self) -> Dict:
        """獲取過濾統計資訊"""
        return self._filter_stats.copy()
    
    def get_kept_patient_ids(self) -> List[int]:
        """
        獲取過濾後保留的患者 ID 列表
        
        Returns:
            List[int]: 保留的 LNDb ID 列表
        """
        kept_ids = set()
        for sample in self.samples:
            kept_ids.add(sample['lndb_id'])
        return sorted(list(kept_ids))
    
    def _load_sample(self, idx: int) -> Dict:
        """載入單個樣本"""
        sample_info = self.samples[idx]
        lndb_id = sample_info['lndb_id']
        ct_file = sample_info['ct_file']
        slice_idx = sample_info['slice_index']
        
        # 讀取 CT 影像
        ct_image = sitk.ReadImage(str(ct_file))
        ct_array = sitk.GetArrayFromImage(ct_image)  # (z, y, x)
        
        # 提取 CT 切片
        if self.axis == 2:
            ct_slice = ct_array[slice_idx, :, :]
        elif self.axis == 1:
            ct_slice = ct_array[:, slice_idx, :]
        else:
            ct_slice = ct_array[:, :, slice_idx]
        
        # 讀取遮罩
        mask_volume = self._load_mask_volume(lndb_id)
        if mask_volume is not None:
            if self.axis == 2:
                mask_slice = mask_volume[slice_idx, :, :]
            elif self.axis == 1:
                mask_slice = mask_volume[:, slice_idx, :]
            else:
                mask_slice = mask_volume[:, :, slice_idx]
        else:
            mask_slice = np.zeros_like(ct_slice, dtype=np.uint8)
        
        # ✅ Resize 到固定尺寸 (512x512) - LNDb 影像大小不一致
        target_size = (512, 512)
        orig_h, orig_w = ct_slice.shape
        
        if (orig_h, orig_w) != target_size:
            # 使用 cv2 進行 resize
            import cv2
            ct_slice = cv2.resize(ct_slice.astype(np.float32), target_size, interpolation=cv2.INTER_LINEAR)
            mask_slice = cv2.resize(mask_slice.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
            
            # 計算縮放比例（用於調整 bboxes）
            scale_x = target_size[0] / orig_w
            scale_y = target_size[1] / orig_h
        else:
            scale_x, scale_y = 1.0, 1.0
        
        # CT 值裁剪與標準化
        hu_min, hu_max = -1000, 800
        ct_clipped = np.clip(ct_slice, hu_min, hu_max)
        ct_normalized = ((ct_clipped - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)
        
        # 堆疊成 3 通道 RGB
        ct_rgb = np.stack([ct_normalized, ct_normalized, ct_normalized], axis=-1)
        
        # 提取 bounding boxes（從 resize 後的 mask 提取，所以座標已正確）
        bboxes = self._extract_bboxes(mask_slice)
        
        return {
            'image': ct_rgb,
            'mask': mask_slice.astype(np.uint8),
            'bboxes': bboxes,
            'patient_id': f"LNDb-{lndb_id:04d}",
            'slice_index': slice_idx
        }
    
    def _extract_bboxes(self, mask: np.ndarray) -> np.ndarray:
        """從遮罩提取 bounding boxes"""
        if mask.sum() == 0:
            return np.array([[0, 0, 1, 1]])
        
        labeled_mask = measure.label(mask > 0)
        regions = measure.regionprops(labeled_mask)
        
        bboxes = []
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            bboxes.append([minc, minr, maxc, maxr])  # [x1, y1, x2, y2]
        
        return np.array(bboxes) if bboxes else np.array([[0, 0, 1, 1]])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        # 從緩存或磁碟載入
        if self.cache_data:
            data = self.cached_data[idx].copy()
        else:
            data = self._load_sample(idx)
        
        # 資料增強
        if self.transform:
            data = self.transform(data)
        
        # 轉換為 tensor
        image = torch.from_numpy(data['image']).permute(2, 0, 1).float()  # [3, H, W]
        mask = torch.from_numpy(data['mask']).unsqueeze(0).float()  # [1, H, W]
        bboxes = torch.from_numpy(data['bboxes']).float()  # [N, 4]
        
        return {
            'image': image,
            'mask': mask,
            'bboxes': bboxes,
            'patient_id': data['patient_id'],
            'slice_index': data['slice_index']
        }
