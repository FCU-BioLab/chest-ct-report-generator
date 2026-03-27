#!/usr/bin/env python3
"""
資料集模組
提供 LNDb 胸部 CT 資料集的載入與處理
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import pandas as pd
from skimage import measure
from tqdm import tqdm


# =============================================================================
# 工具函數
# =============================================================================

def classify_nodule_size(diameter_mm: float) -> str:
    """
    根據直徑分類結節大小 (Fleischner Society 指南)
    
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


# =============================================================================
# 資料增強
# =============================================================================

class DataAugmentation:
    """
    資料增強類別（專為 CT 肺結節分割優化）
    
    提供多種增強方法：
    - 幾何變換：旋轉、翻轉、縮放、彈性形變
    - 強度變換：Gamma、對比度、亮度、高斯噪音
    - CT 專用：模擬 CT 偽影、局部遮擋
    
    Args:
        rotation_prob: 旋轉機率
        flip_prob: 翻轉機率
        gamma_prob: Gamma 調整機率
        bbox_shift_limit: Bounding Box 擾動範圍 (pixels)
        noise_prob: 高斯噪音機率
        contrast_prob: 對比度調整機率
        scale_prob: 縮放增強機率
        elastic_prob: 彈性形變機率
        brightness_prob: 亮度調整機率
        blur_prob: 模糊機率
        cutout_prob: 隨機遮擋機率
        mixup_prob: MixUp 機率（需要額外處理）
    """
    
    def __init__(
        self,
        rotation_prob: float = 0.5,
        flip_prob: float = 0.5,
        gamma_prob: float = 0.3,
        bbox_shift_limit: int = 10,
        noise_prob: float = 0.2,
        contrast_prob: float = 0.3,
        scale_prob: float = 0.3,
        elastic_prob: float = 0.0,
        brightness_prob: float = 0.2,
        blur_prob: float = 0.1,
        cutout_prob: float = 0.1,
        fine_rotation_prob: float = 0.3
    ):
        self.rotation_prob = rotation_prob
        self.flip_prob = flip_prob
        self.gamma_prob = gamma_prob
        self.bbox_shift_limit = bbox_shift_limit
        self.noise_prob = noise_prob
        self.contrast_prob = contrast_prob
        self.scale_prob = scale_prob
        self.elastic_prob = elastic_prob
        self.brightness_prob = brightness_prob
        self.blur_prob = blur_prob
        self.cutout_prob = cutout_prob
        self.fine_rotation_prob = fine_rotation_prob
    
    def _apply_elastic_transform(self, image: np.ndarray, mask: np.ndarray, 
                                  alpha: float = 50, sigma: float = 5) -> tuple:
        """彈性形變（模擬組織變形）"""
        from scipy.ndimage import gaussian_filter, map_coordinates
        
        shape = image.shape[:2]
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).flatten(), (x + dx).flatten()
        
        # 對影像的每個通道進行變形
        if len(image.shape) == 3:
            transformed_image = np.zeros_like(image)
            for c in range(image.shape[2]):
                transformed_image[:, :, c] = map_coordinates(
                    image[:, :, c], indices, order=1, mode='reflect'
                ).reshape(shape)
        else:
            transformed_image = map_coordinates(
                image, indices, order=1, mode='reflect'
            ).reshape(shape)
        
        # 對 mask 使用最近鄰插值
        transformed_mask = map_coordinates(
            mask.astype(np.float32), indices, order=0, mode='reflect'
        ).reshape(shape)
        
        return transformed_image.astype(np.uint8), transformed_mask.astype(mask.dtype)
    
    def _apply_fine_rotation(self, image: np.ndarray, mask: np.ndarray, 
                              angle_range: tuple = (-15, 15)) -> tuple:
        """小角度旋轉（更自然的變換）"""
        from scipy.ndimage import rotate
        import random
        
        angle = random.uniform(*angle_range)
        
        if len(image.shape) == 3:
            rotated_image = rotate(image, angle, axes=(0, 1), reshape=False, 
                                   order=1, mode='constant', cval=0)
        else:
            rotated_image = rotate(image, angle, reshape=False, 
                                   order=1, mode='constant', cval=0)
        
        rotated_mask = rotate(mask.astype(np.float32), angle, reshape=False, 
                              order=0, mode='constant', cval=0)
        
        return rotated_image.astype(np.uint8), rotated_mask.astype(mask.dtype)
    
    def _apply_gaussian_blur(self, image: np.ndarray, sigma_range: tuple = (0.5, 1.5)) -> np.ndarray:
        """高斯模糊（模擬運動或對焦不準）"""
        from scipy.ndimage import gaussian_filter
        import random
        
        sigma = random.uniform(*sigma_range)
        
        if len(image.shape) == 3:
            blurred = np.zeros_like(image)
            for c in range(image.shape[2]):
                blurred[:, :, c] = gaussian_filter(image[:, :, c], sigma)
        else:
            blurred = gaussian_filter(image, sigma)
        
        return blurred.astype(np.uint8)
    
    def _apply_cutout(self, image: np.ndarray, mask: np.ndarray,
                       num_holes: int = 1, hole_size_ratio: float = 0.1) -> np.ndarray:
        """隨機遮擋（增加魯棒性，不遮擋病灶區域）"""
        import random
        
        h, w = image.shape[:2]
        hole_size = int(min(h, w) * hole_size_ratio)
        
        image_out = image.copy()
        
        for _ in range(num_holes):
            # 嘗試找到非病灶區域
            for _ in range(10):  # 最多嘗試 10 次
                y = random.randint(0, h - hole_size)
                x = random.randint(0, w - hole_size)
                
                # 檢查是否與病灶重疊
                mask_region = mask[y:y+hole_size, x:x+hole_size]
                if not np.any(mask_region > 0):
                    # 用黑色填充
                    if len(image.shape) == 3:
                        image_out[y:y+hole_size, x:x+hole_size, :] = 0
                    else:
                        image_out[y:y+hole_size, x:x+hole_size] = 0
                    break
        
        return image_out
    
    def _apply_brightness(self, image: np.ndarray, 
                           brightness_range: tuple = (-30, 30)) -> np.ndarray:
        """亮度調整"""
        import random
        
        delta = random.uniform(*brightness_range)
        image = image.astype(np.float32) + delta
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def __call__(self, data: Dict) -> Dict:
        """應用資料增強"""
        import random
        
        image = data['image']  # [H, W, 3]
        mask = data['mask']    # [H, W]
        bboxes = data['bboxes']  # [N, 4]
        
        # ============================================
        # 幾何變換
        # ============================================
        
        # 隨機旋轉 (90, 180, 270 度)
        if random.random() < self.rotation_prob:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        
        # 小角度旋轉 (-15° ~ 15°)
        if random.random() < self.fine_rotation_prob:
            image, mask = self._apply_fine_rotation(image, mask)
        
        # 隨機水平翻轉
        if random.random() < self.flip_prob:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
            if len(bboxes) > 0:
                w = image.shape[1]
                bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        
        # 隨機垂直翻轉
        if random.random() < self.flip_prob:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
            if len(bboxes) > 0:
                h = image.shape[0]
                bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]
        
        # 彈性形變（模擬組織變形）
        if random.random() < self.elastic_prob:
            image, mask = self._apply_elastic_transform(image, mask)
        
        # ============================================
        # 強度變換
        # ============================================
        
        # Gamma 調整
        if random.random() < self.gamma_prob:
            gamma = random.uniform(0.8, 1.2)
            image = np.power(image / 255.0, gamma) * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 對比度調整
        if random.random() < self.contrast_prob:
            factor = random.uniform(0.8, 1.3)
            mean_val = np.mean(image)
            image = (image - mean_val) * factor + mean_val
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 亮度調整
        if random.random() < self.brightness_prob:
            image = self._apply_brightness(image)
        
        # 高斯噪音
        if random.random() < self.noise_prob:
            noise_std = random.uniform(3, 10)
            noise = np.random.normal(0, noise_std, image.shape)
            image = image.astype(np.float32) + noise
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 高斯模糊
        if random.random() < self.blur_prob:
            image = self._apply_gaussian_blur(image)
        
        # ============================================
        # 正則化增強
        # ============================================
        
        # 隨機遮擋 (Cutout)
        if random.random() < self.cutout_prob:
            image = self._apply_cutout(image, mask)
        
        # 隨機縮放
        if random.random() < self.scale_prob:
            scale = random.uniform(0.9, 1.1)
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            from skimage.transform import resize
            if len(image.shape) == 3:
                scaled_image = resize(image, (new_h, new_w, image.shape[2]), 
                                      preserve_range=True, anti_aliasing=True).astype(np.uint8)
            else:
                scaled_image = resize(image, (new_h, new_w), 
                                      preserve_range=True, anti_aliasing=True).astype(np.uint8)
            
            scaled_mask = resize(mask.astype(np.float32), (new_h, new_w), 
                                 preserve_range=True, order=0).astype(mask.dtype)
            
            if scale > 1:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                image = scaled_image[start_h:start_h+h, start_w:start_w+w]
                mask = scaled_mask[start_h:start_h+h, start_w:start_w+w]
            else:
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                if len(scaled_image.shape) == 3:
                    image = np.zeros((h, w, scaled_image.shape[2]), dtype=np.uint8)
                else:
                    image = np.zeros((h, w), dtype=np.uint8)
                mask = np.zeros((h, w), dtype=mask.dtype)
                image[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled_image
                mask[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled_mask
            
            # ✅ 優化：先更新 bbox，再驗證有效性
            if len(bboxes) > 0:
                if scale > 1:
                    bboxes = (bboxes * scale) - np.array([start_w, start_h, start_w, start_h])
                else:
                    bboxes = (bboxes * scale) + np.array([pad_w, pad_h, pad_w, pad_h])
                
                # ✅ 先 clip 到有效範圍
                bboxes[:, 0] = np.clip(bboxes[:, 0], 0, w - 1)
                bboxes[:, 1] = np.clip(bboxes[:, 1], 0, h - 1)
                bboxes[:, 2] = np.clip(bboxes[:, 2], 1, w)
                bboxes[:, 3] = np.clip(bboxes[:, 3], 1, h)
                
                # ✅ 確保 bbox 有效（寬高 > 0）
                invalid_mask = (bboxes[:, 2] <= bboxes[:, 0]) | (bboxes[:, 3] <= bboxes[:, 1])
                if invalid_mask.any():
                    # 對於無效 bbox，從 mask 重新計算
                    from skimage import measure
                    labeled_mask = measure.label(mask > 0)
                    regions = measure.regionprops(labeled_mask)
                    if regions:
                        new_bboxes = []
                        for region in regions:
                            minr, minc, maxr, maxc = region.bbox
                            new_bboxes.append([minc, minr, maxc, maxr])
                        bboxes = np.array(new_bboxes)
        
        # ============================================
        # 重新計算 bboxes（確保與 mask 一致）
        # ============================================
        # 在幾何變換後從 mask 重新計算 bbox
        if len(bboxes) > 0:
            from skimage import measure
            labeled_mask = measure.label(mask > 0)
            regions = measure.regionprops(labeled_mask)
            if regions:
                new_bboxes = []
                for region in regions:
                    minr, minc, maxr, maxc = region.bbox
                    new_bboxes.append([minc, minr, maxc, maxr])
                bboxes = np.array(new_bboxes)
            
        # Bounding Box 擾動（最後才做，模擬不完美的檢測）
        if len(bboxes) > 0 and self.bbox_shift_limit > 0:
            h, w = image.shape[:2]
            noise = np.random.randint(-self.bbox_shift_limit, self.bbox_shift_limit, size=bboxes.shape)
            bboxes = bboxes + noise
            bboxes[:, 0] = np.clip(bboxes[:, 0], 0, w)
            bboxes[:, 1] = np.clip(bboxes[:, 1], 0, h)
            bboxes[:, 2] = np.clip(bboxes[:, 2], 0, w)
            bboxes[:, 3] = np.clip(bboxes[:, 3], 0, h)
            
            invalid_mask = (bboxes[:, 2] <= bboxes[:, 0]) | (bboxes[:, 3] <= bboxes[:, 1])
            if invalid_mask.any():
                bboxes[invalid_mask, 2] = bboxes[invalid_mask, 0] + 1
                bboxes[invalid_mask, 3] = bboxes[invalid_mask, 1] + 1
        
        data['image'] = image
        data['mask'] = mask
        data['bboxes'] = bboxes
        
        return data


# =============================================================================
# LNDb 資料集
# =============================================================================

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
        min_nodule_diameter: 最小結節直徑 (mm)，過濾小於此值的結節
    
    LNDb 資料結構:
        LNDb/
        ├── data0~data5/          # CT 掃描 (.mhd + .raw)
        ├── masks/masks/          # 專家分割遮罩 (.mhd + .raw)
        └── trainset_csv/
            ├── trainNodules.csv      # 個別醫師標註
            └── trainNodules_gt.csv   # 融合標註 (GT)
    """
    
    def __init__(
        self, 
        data_dir: str,
        patient_ids: List,
        axis: int = 2,
        transform: Optional[Callable] = None,
        cache_data: bool = False,
        rad_id: str = 'consensus',
        min_nodule_diameter: float = 0.0,
        use_2_5d: bool = True,  # ✅ 新增：2.5D 模式
        fixed_bbox_size: int = 64  # ✅ 新增：固定 bbox 大小 (0 = 使用動態大小)
    ):
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.axis = axis
        self.transform = transform
        self.cache_data = cache_data
        self.rad_id = rad_id
        self.min_nodule_diameter = min_nodule_diameter
        self.use_2_5d = use_2_5d  # ✅ 儲存 2.5D 模式設定
        self.fixed_bbox_size = fixed_bbox_size  # ✅ 儲存固定 bbox 大小
        
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
        self.logger.info(f"🔧 輸入模式: {'2.5D (Z-1, Z, Z+1)' if use_2_5d else '2D (單一切片)'}")
        self.logger.info(f"📦 BBox 大小: {'Fixed ' + str(fixed_bbox_size) + 'x' + str(fixed_bbox_size) + 'px' if fixed_bbox_size > 0 else 'Dynamic (from mask)'}")
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
        
        nodules_gt_path = csv_dir / 'trainNodules_gt.csv'
        if nodules_gt_path.exists():
            self.nodules_gt_df = pd.read_csv(nodules_gt_path)
            self.logger.info(f"載入 {len(self.nodules_gt_df)} 個融合標註")
        else:
            raise FileNotFoundError(f"找不到 GT 標註檔案: {nodules_gt_path}")
        
        nodules_path = csv_dir / 'trainNodules.csv'
        self.nodules_df = pd.read_csv(nodules_path) if nodules_path.exists() else None
    
    def _index_files(self):
        """索引所有 CT 和遮罩檔案"""
        self.ct_files = {}
        for i in range(6):
            data_dir = self.data_dir / f'data{i}'
            if data_dir.exists():
                for mhd_file in data_dir.glob('LNDb-*.mhd'):
                    lndb_id = int(mhd_file.stem.split('-')[1])
                    self.ct_files[lndb_id] = mhd_file
        
        self.mask_files = {}
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
            
            if lndb_id not in self.ct_files:
                patient_stats['no_ct_file'] += 1
                continue
            
            if lndb_id not in self.mask_files:
                patient_stats['no_mask'] += 1
                continue
            
            ct_file = self.ct_files[lndb_id]
            patient_nodules = self.nodules_gt_df[self.nodules_gt_df['LNDbID'] == lndb_id]
            
            if patient_nodules.empty:
                patient_stats['no_nodules'] += 1
                continue
            
            patient_stats['has_nodules'] += 1
            nodule_stats['total'] += len(patient_nodules)
            
            # 計算結節直徑
            if 'Volume' in patient_nodules.columns:
                volumes = patient_nodules['Volume'].values
                diameters = 2 * np.power(3 * volumes / (4 * np.pi), 1/3)
                min_diameter = diameters.min()
            else:
                min_diameter = float('inf')
                diameters = np.array([])
            
            # 過濾小結節
            if self.min_nodule_diameter > 0 and min_diameter < self.min_nodule_diameter:
                patient_stats['filtered'] += 1
                nodule_stats['filtered'] += len(patient_nodules)
                continue
            
            patient_stats['kept'] += 1
            nodule_stats['kept'] += len(patient_nodules)
            
            for d in diameters:
                size_distribution[classify_nodule_size(d)] += 1
            
            try:
                mask_volume = self._load_mask_volume(lndb_id)
                if mask_volume is None:
                    continue
                
                if self.axis == 2:
                    slice_sums = mask_volume.sum(axis=(1, 2))
                elif self.axis == 1:
                    slice_sums = mask_volume.sum(axis=(0, 2))
                else:
                    slice_sums = mask_volume.sum(axis=(0, 1))
                
                valid_slices = np.where(slice_sums > 0)[0]
                max_diameter = diameters.max() if len(diameters) > 0 else 10.0
                
                for slice_idx in valid_slices:
                    self.samples.append({
                        'lndb_id': lndb_id,
                        'ct_file': ct_file,
                        'slice_index': int(slice_idx),
                        'max_diameter_mm': max_diameter,
                        'size_class': classify_nodule_size(max_diameter)
                    })
                
            except Exception as e:
                self.logger.warning(f"⚠️ 跳過患者 LNDb-{lndb_id:04d}: {e}")
                continue
        
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
            
            combined = np.sum(masks, axis=0)
            threshold = max(1, len(masks) // 2)
            return (combined >= threshold).astype(np.uint8)
        else:
            rad_id = int(self.rad_id) if isinstance(self.rad_id, str) else self.rad_id
            if rad_id in available_rads:
                itk_mask = sitk.ReadImage(str(available_rads[rad_id]))
                return (sitk.GetArrayFromImage(itk_mask) > 0).astype(np.uint8)
            elif available_rads:
                first_rad = list(available_rads.keys())[0]
                itk_mask = sitk.ReadImage(str(available_rads[first_rad]))
                return (sitk.GetArrayFromImage(itk_mask) > 0).astype(np.uint8)
            return None
    
    def get_filter_stats(self) -> Dict:
        """獲取過濾統計資訊"""
        return self._filter_stats.copy()
    
    def get_kept_patient_ids(self) -> List[int]:
        """獲取過濾後保留的患者 ID 列表"""
        kept_ids = set()
        for sample in self.samples:
            kept_ids.add(sample['lndb_id'])
        return sorted(list(kept_ids))
    
    def _load_sample(self, idx: int) -> Dict:
        """載入單個樣本（支援 2D 和 2.5D 模式）"""
        sample_info = self.samples[idx]
        lndb_id = sample_info['lndb_id']
        ct_file = sample_info['ct_file']
        slice_idx = sample_info['slice_index']
        
        ct_image = sitk.ReadImage(str(ct_file))
        ct_array = sitk.GetArrayFromImage(ct_image)
        
        # ✅ 2.5D 模式：載入 Z-1, Z, Z+1 三個切片
        if self.use_2_5d:
            # 獲取總切片數
            if self.axis == 2:
                num_slices = ct_array.shape[0]
            elif self.axis == 1:
                num_slices = ct_array.shape[1]
            else:
                num_slices = ct_array.shape[2]
            
            # 計算相鄰切片索引（邊界處理）
            z_prev = max(0, slice_idx - 1)
            z_next = min(num_slices - 1, slice_idx + 1)
            
            # 載入 3 個相鄰切片
            slices = []
            for z in [z_prev, slice_idx, z_next]:
                if self.axis == 2:
                    ct_slice = ct_array[z, :, :]
                elif self.axis == 1:
                    ct_slice = ct_array[:, z, :]
                else:
                    ct_slice = ct_array[:, :, z]
                slices.append(ct_slice)
        else:
            # ✅ 2D 模式：只載入單一切片
            if self.axis == 2:
                ct_slice = ct_array[slice_idx, :, :]
            elif self.axis == 1:
                ct_slice = ct_array[:, slice_idx, :]
            else:
                ct_slice = ct_array[:, :, slice_idx]
            slices = [ct_slice]  # 包裝成列表以統一處理
        
        # 載入 mask（只需要中間切片的 mask）
        mask_volume = self._load_mask_volume(lndb_id)
        if mask_volume is not None:
            if self.axis == 2:
                mask_slice = mask_volume[slice_idx, :, :]
            elif self.axis == 1:
                mask_slice = mask_volume[:, slice_idx, :]
            else:
                mask_slice = mask_volume[:, :, slice_idx]
        else:
            mask_slice = np.zeros_like(slices[0] if self.use_2_5d else ct_slice, dtype=np.uint8)
        
        # Resize 到固定尺寸 (512x512)
        target_size = (512, 512)
        import cv2
        
        # Resize 所有切片
        slices_resized = []
        for s in slices:
            if s.shape != target_size:
                s_resized = cv2.resize(s.astype(np.float32), target_size, 
                                      interpolation=cv2.INTER_LINEAR)
            else:
                s_resized = s.astype(np.float32)
            slices_resized.append(s_resized)
        
        # Resize mask
        if mask_slice.shape != target_size:
            mask_slice = cv2.resize(mask_slice.astype(np.uint8), target_size, 
                                   interpolation=cv2.INTER_NEAREST)
        
        # CT 值裁剪與標準化（對每個切片）
        hu_min, hu_max = -1000, 800
        normalized_slices = []
        for s in slices_resized:
            ct_clipped = np.clip(s, hu_min, hu_max)
            ct_normalized = ((ct_clipped - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)
            normalized_slices.append(ct_normalized)
        
        # ✅ 組合為 RGB 格式
        if self.use_2_5d:
            # 2.5D: 三個通道分別是 Z-1, Z, Z+1
            ct_rgb = np.stack(normalized_slices, axis=-1)  # [H, W, 3]
        else:
            # 2D: 三個通道都是同一張切片
            ct_rgb = np.stack([normalized_slices[0]] * 3, axis=-1)  # [H, W, 3]
        
        bboxes = self._extract_bboxes(mask_slice)
        
        return {
            'image': ct_rgb,
            'mask': mask_slice.astype(np.uint8),
            'bboxes': bboxes,
            'patient_id': f"LNDb-{lndb_id:04d}",
            'slice_index': slice_idx
        }
    
    
    def _extract_bboxes(self, mask: np.ndarray) -> np.ndarray:
        """從遮罩提取 bounding boxes
        
        如果 fixed_bbox_size > 0，則使用固定大小的 bbox（以病灶中心為中心）
        否則使用 mask 的實際邊界
        """
        if mask.sum() == 0:
            return np.array([[0, 0, 1, 1]])
        
        h, w = mask.shape
        labeled_mask = measure.label(mask > 0)
        regions = measure.regionprops(labeled_mask)
        
        bboxes = []
        for region in regions:
            if self.fixed_bbox_size > 0:
                # 使用固定大小 bbox，以病灶 centroid 為中心
                cy, cx = region.centroid  # centroid 回傳 (row, col)
                half_size = self.fixed_bbox_size // 2
                
                x1 = int(cx - half_size)
                y1 = int(cy - half_size)
                x2 = int(cx + half_size)
                y2 = int(cy + half_size)
                
                # Clip to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # 確保 bbox 有效
                if x2 > x1 and y2 > y1:
                    bboxes.append([x1, y1, x2, y2])
            else:
                # 使用 mask 的實際邊界
                minr, minc, maxr, maxc = region.bbox
                bboxes.append([minc, minr, maxc, maxr])
        
        return np.array(bboxes) if bboxes else np.array([[0, 0, 1, 1]])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        if self.cache_data:
            data = self.cached_data[idx].copy()
        else:
            data = self._load_sample(idx)
        
        if self.transform:
            data = self.transform(data)
        
        image = torch.from_numpy(data['image']).permute(2, 0, 1).float()
        mask = torch.from_numpy(data['mask']).unsqueeze(0).float()
        bboxes = torch.from_numpy(data['bboxes']).float()
        
        return {
            'image': image,
            'mask': mask,
            'bboxes': bboxes,
            'patient_id': data['patient_id'],
            'slice_index': data['slice_index']
        }


# =============================================================================
# 快取切片資料集 (用於預處理過的 .npz 檔案)
# =============================================================================

class CachedSliceDataset(Dataset):
    """
    快取切片資料集
    
    載入預處理過的 .npz 切片檔案，支援 LNDb 和 MSD 資料集格式
    
    Args:
        cache_dir: 快取目錄路徑 (例如 'segmentation/cache')
        dataset_type: 資料集類型 ('lndb', 'msd', 'both')
        patient_ids: 可選，指定要載入的患者 ID 列表
        transform: 資料增強函數
        target_size: 目標圖像尺寸 (H, W)
    
    快取資料結構:
        cache/
        ├── lndb_slices/
        │   └── LNDb-XXXX/
        │       ├── meta.json
        │       └── slice_XXXX.npz  (image, mask, lung_mask)
        └── msd_lung_slices/
            └── lung_XXX/
                ├── meta.json
                └── slice_XXXX.npz  (image, mask, lung_mask)
    """
    
    def __init__(
        self,
        cache_dir: str,
        dataset_type: str = 'both',
        patient_ids: Optional[List] = None,
        transform: Optional[Callable] = None,
        target_size: tuple = (512, 512),
        use_2_5d: bool = True,  # ✅ 新增：2.5D 模式
        fixed_bbox_size: int = 64  # ✅ 新增：固定 bbox 大小 (0 = 使用動態大小)
    ):
        self.cache_dir = Path(cache_dir)
        self.dataset_type = dataset_type.lower()
        self.patient_ids = patient_ids
        self.transform = transform
        self.target_size = target_size
        self.use_2_5d = use_2_5d  # ✅ 儲存 2.5D 模式設定
        self.fixed_bbox_size = fixed_bbox_size  # ✅ 儲存固定 bbox 大小
        
        self.logger = logging.getLogger(__name__)
        
        # 初始化過濾統計
        self._filter_stats = {
            'patients_input': 0, 'patients_no_ct': 0, 'patients_no_nodules': 0,
            'patients_has_nodules': 0, 'patients_filtered': 0, 'patients_kept': 0,
            'patients_total': 0, 'nodules_total': 0, 'nodules_kept': 0, 'nodules_filtered': 0,
            'size_micro': 0, 'size_small': 0, 'size_medium': 0, 'size_large': 0
        }
        
        # 建立樣本索引
        self.samples = []
        self._build_sample_index()
        
        self.logger.info(f"✅ CachedSliceDataset: {len(self.samples)} 個切片 "
                        f"(類型: {dataset_type}, 模式: {'2.5D' if use_2_5d else '2D'}, "
                        f"BBox: {'Fixed ' + str(fixed_bbox_size) + 'x' + str(fixed_bbox_size) if fixed_bbox_size > 0 else 'Dynamic'})")  # ✅ 顯示模式
    
    def _build_sample_index(self):
        """建立所有有效切片的索引"""
        lndb_dir = self.cache_dir / 'lndb_slices'
        msd_dir = self.cache_dir / 'msd_lung_slices'
        
        patient_dirs = []
        
        # 收集 LNDb 患者
        if self.dataset_type in ('lndb', 'both') and lndb_dir.exists():
            for patient_dir in sorted(lndb_dir.iterdir()):
                if patient_dir.is_dir():
                    patient_id = patient_dir.name
                    if self.patient_ids is None or patient_id in self.patient_ids:
                        patient_dirs.append(('lndb', patient_dir))
        
        # 收集 MSD 患者
        if self.dataset_type in ('msd', 'both') and msd_dir.exists():
            for patient_dir in sorted(msd_dir.iterdir()):
                if patient_dir.is_dir():
                    patient_id = patient_dir.name
                    if self.patient_ids is None or patient_id in self.patient_ids:
                        patient_dirs.append(('msd', patient_dir))
        
        self._filter_stats['patients_input'] = len(patient_dirs)
        
        # 建立切片索引
        for source, patient_dir in patient_dirs:
            npz_files = sorted(patient_dir.glob('slice_*.npz'))
            
            if not npz_files:
                self._filter_stats['patients_no_nodules'] += 1
                continue
            
            self._filter_stats['patients_has_nodules'] += 1
            self._filter_stats['patients_kept'] += 1
            self._filter_stats['nodules_kept'] += len(npz_files)
            
            for npz_path in npz_files:
                slice_idx = int(npz_path.stem.split('_')[1])
                self.samples.append({
                    'source': source,
                    'patient_id': patient_dir.name,
                    'npz_path': npz_path,
                    'slice_index': slice_idx
                })
        
        self._filter_stats['patients_total'] = self._filter_stats['patients_has_nodules']
        self._filter_stats['nodules_total'] = self._filter_stats['nodules_kept']
    
    def get_filter_stats(self) -> Dict:
        """獲取過濾統計資訊"""
        return self._filter_stats.copy()
    
    def get_kept_patient_ids(self) -> List[str]:
        """獲取保留的患者 ID 列表"""
        kept_ids = set()
        for sample in self.samples:
            kept_ids.add(sample['patient_id'])
        return sorted(list(kept_ids))
    
    def _load_sample(self, idx: int) -> Dict:
        """載入單個樣本（支援 2D 和 2.5D 模式）"""
        import cv2
        
        sample_info = self.samples[idx]
        npz_path = sample_info['npz_path']
        patient_id = sample_info['patient_id']
        slice_idx = sample_info['slice_index']
        
        # ✅ 2.5D 模式：載入相鄰切片
        if self.use_2_5d:
            # 獲取患者目錄和 meta 資訊
            patient_dir = npz_path.parent
            meta_path = patient_dir / 'meta.json'
            
            if meta_path.exists():
                import json
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                num_slices = meta.get('num_slices', slice_idx + 1)
            else:
                num_slices = slice_idx + 1
            
            # 計算相鄰切片索引
            z_prev = max(0, slice_idx - 1)
            z_next = min(num_slices - 1, slice_idx + 1)
            
            # 載入 3 個切片
            slices = []
            for z in [z_prev, slice_idx, z_next]:
                slice_path = patient_dir / f"slice_{z:04d}.npz"
                if slice_path.exists():
                    data = np.load(str(slice_path))
                    img = data['image']  # 2D array
                else:
                    # 如果切片不存在，使用零填充
                    img = np.zeros(self.target_size, dtype=np.float32)
                slices.append(img)
            
            # 載入中間切片的 mask
            data = np.load(str(npz_path))
            mask = data['mask']
        else:
            # ✅ 2D 模式：只載入單一切片
            data = np.load(str(npz_path))
            image = data['image']  # 2D array
            mask = data['mask']    # 2D array
            slices = [image]
        
        # Resize 所有切片到目標尺寸
        slices_resized = []
        for img in slices:
            if img.shape != self.target_size:
                img = cv2.resize(img.astype(np.float32), self.target_size, 
                                interpolation=cv2.INTER_LINEAR)
            slices_resized.append(img)
        
        # Resize mask
        if mask.shape != self.target_size:
            mask = cv2.resize(mask.astype(np.uint8), self.target_size, 
                             interpolation=cv2.INTER_NEAREST)
        
        # 標準化到 0-255（對每個切片）
        normalized_slices = []
        for img in slices_resized:
            if img.max() > 255 or img.min() < -1000:
                # 假設是 HU 值，進行 CT 視窗處理
                hu_min, hu_max = -1000, 800
                img = np.clip(img, hu_min, hu_max)
                img = ((img - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)
            elif img.max() <= 1.0:
                # 已經 normalize 到 0-1
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            normalized_slices.append(img)
        
        # ✅ 組合為 RGB 格式
        if self.use_2_5d:
            # 2.5D: 三個通道分別是 Z-1, Z, Z+1
            image_rgb = np.stack(normalized_slices, axis=-1)  # [H, W, 3]
        else:
            # 2D: 三個通道都是同一張切片
            image_rgb = np.stack([normalized_slices[0]] * 3, axis=-1)  # [H, W, 3]
        
        # 確保 mask 是二值的
        mask = (mask > 0).astype(np.uint8)
        
        # 提取 bounding boxes
        bboxes = self._extract_bboxes(mask)
        
        return {
            'image': image_rgb,
            'mask': mask,
            'bboxes': bboxes,
            'patient_id': patient_id,
            'slice_index': slice_idx
        }
    
    def _extract_bboxes(self, mask: np.ndarray) -> np.ndarray:
        """從遮罩提取 bounding boxes
        
        如果 fixed_bbox_size > 0，則使用固定大小的 bbox（以病灶中心為中心）
        否則使用 mask 的實際邊界
        """
        if mask.sum() == 0:
            return np.array([[0, 0, 1, 1]])
        
        h, w = mask.shape
        labeled_mask = measure.label(mask > 0)
        regions = measure.regionprops(labeled_mask)
        
        bboxes = []
        for region in regions:
            if self.fixed_bbox_size > 0:
                # 使用固定大小 bbox，以病灶 centroid 為中心
                cy, cx = region.centroid  # centroid 回傳 (row, col)
                half_size = self.fixed_bbox_size // 2
                
                x1 = int(cx - half_size)
                y1 = int(cy - half_size)
                x2 = int(cx + half_size)
                y2 = int(cy + half_size)
                
                # Clip to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # 確保 bbox 有效
                if x2 > x1 and y2 > y1:
                    bboxes.append([x1, y1, x2, y2])
            else:
                # 使用 mask 的實際邊界
                minr, minc, maxr, maxc = region.bbox
                bboxes.append([minc, minr, maxc, maxr])
        
        return np.array(bboxes) if bboxes else np.array([[0, 0, 1, 1]])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        data = self._load_sample(idx)
        
        if self.transform:
            data = self.transform(data)
        
        image = torch.from_numpy(data['image']).permute(2, 0, 1).float()
        mask = torch.from_numpy(data['mask']).unsqueeze(0).float()
        bboxes = torch.from_numpy(data['bboxes']).float()
        
        return {
            'image': image,
            'mask': mask,
            'bboxes': bboxes,
            'patient_id': data['patient_id'],
            'slice_index': data['slice_index']
        }


_MANIFEST_TRAIN_KEYS = ("training", "train")
_MANIFEST_VAL_KEYS = ("validation", "val")
_MANIFEST_TEST_KEYS = ("testing", "test")


def _pick_first_record_value(record: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[Any]:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def infer_manifest_patient_id(record: Dict[str, Any]) -> str:
    pid = _pick_first_record_value(
        record,
        ("patient_id", "seriesuid", "lndb_id", "id", "case_id", "name"),
    )
    if pid is not None:
        return str(pid)

    image_path = _pick_first_record_value(record, ("image", "image_path", "ct_path"))
    if image_path:
        stem = Path(str(image_path)).name
        if stem.endswith(".nii.gz"):
            return stem[:-7]
        return Path(stem).stem

    return "unknown"


def _resolve_manifest_path(path_value: Any, manifest_path: Optional[str]) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    if manifest_path:
        return (Path(manifest_path).resolve().parent / path).resolve()
    return path.resolve()


def _normalize_manifest_splits(raw_data: Any) -> Dict[str, List[Dict[str, Any]]]:
    if isinstance(raw_data, list):
        return {"training": raw_data, "validation": [], "testing": []}

    if not isinstance(raw_data, dict):
        raise ValueError("Manifest JSON 必須是 list 或 dict")

    train_items = []
    val_items = []
    test_items = []

    for key in _MANIFEST_TRAIN_KEYS:
        if key in raw_data:
            train_items = raw_data[key] or []
            break
    for key in _MANIFEST_VAL_KEYS:
        if key in raw_data:
            val_items = raw_data[key] or []
            break
    for key in _MANIFEST_TEST_KEYS:
        if key in raw_data:
            test_items = raw_data[key] or []
            break

    if not train_items and not val_items and not test_items:
        # fallback: treat entire dict as one split only when data is under "data"
        if isinstance(raw_data.get("data"), list):
            return {"training": raw_data["data"], "validation": [], "testing": []}
        raise ValueError("Manifest JSON 缺少 training/validation/testing 欄位")

    return {
        "training": train_items,
        "validation": val_items,
        "testing": test_items,
    }


def load_manifest_dataset(manifest_path: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return _normalize_manifest_splits(raw)


def split_manifest_records_by_patient(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    del test_ratio  # keep API symmetric with split_dataset

    import random

    patient_to_records: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        pid = infer_manifest_patient_id(record)
        patient_to_records.setdefault(pid, []).append(record)

    patient_ids = list(patient_to_records.keys())
    rng = random.Random(seed) if seed is not None else random.Random()
    rng.shuffle(patient_ids)

    n_patients = len(patient_ids)
    if n_patients == 0:
        return [], [], []

    train_end = int(n_patients * train_ratio)
    val_end = train_end + int(n_patients * val_ratio)

    if n_patients >= 3:
        train_end = max(1, min(train_end, n_patients - 2))
        val_end = max(train_end + 1, min(val_end, n_patients - 1))
    elif n_patients == 2:
        train_end = 1
        val_end = 1
    else:
        train_end = 1
        val_end = 1

    train_ids = patient_ids[:train_end]
    val_ids = patient_ids[train_end:val_end]
    test_ids = patient_ids[val_end:]

    def _collect(ids: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for pid in ids:
            out.extend(patient_to_records.get(pid, []))
        return out

    return _collect(train_ids), _collect(val_ids), _collect(test_ids)


class NiftiManifestDataset(Dataset):
    """
    以 NIfTI/MHD + manifest JSON 作為輸入的切片資料集。

    每個 manifest record 需至少包含:
    - image / image_path / ct_path
    可選:
    - mask / mask_path / label / seg_path
    - patient_id / seriesuid
    - slice_bboxes / det_bboxes / detections / boxes (3D boxes)
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        manifest_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (512, 512),
        use_2_5d: bool = True,
        fixed_bbox_size: int = 0,
        prompt_mode: str = "gt",
        det_prompt_json: Optional[str] = None,
        prompt_jitter_px: int = 0,
        include_empty_slices: bool = False,
        hu_min: float = -1000.0,
        hu_max: float = 800.0,
        max_cached_volumes: int = 6,
    ):
        self.records = records
        self.manifest_path = manifest_path
        self.transform = transform
        self.target_size = target_size
        self.use_2_5d = use_2_5d
        self.fixed_bbox_size = int(fixed_bbox_size)
        self.prompt_mode = prompt_mode.lower()
        self.prompt_jitter_px = max(0, int(prompt_jitter_px))
        self.include_empty_slices = include_empty_slices
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)
        self.max_cached_volumes = max(1, int(max_cached_volumes))

        if self.prompt_mode not in ("gt", "det", "hybrid"):
            raise ValueError("prompt_mode must be one of: gt, det, hybrid")

        self.logger = logging.getLogger(__name__)
        self._volume_cache: Dict[str, np.ndarray] = {}
        self._volume_cache_order: List[str] = []
        self._external_prompt_lookup = self._load_external_prompt_lookup(det_prompt_json)
        self._resolved_records: List[Dict[str, Any]] = []
        self.samples: List[Dict[str, Any]] = []
        self._filter_stats = {
            "patients_input": 0,
            "patients_no_ct": 0,
            "patients_no_nodules": 0,
            "patients_has_nodules": 0,
            "patients_filtered": 0,
            "patients_kept": 0,
            "patients_total": 0,
            "nodules_total": 0,
            "nodules_kept": 0,
            "nodules_filtered": 0,
            "size_micro": 0,
            "size_small": 0,
            "size_medium": 0,
            "size_large": 0,
        }

        self._resolve_records()
        self._build_sample_index()
        self.logger.info(
            "NiftiManifestDataset: %d slices, %d patients (prompt_mode=%s, 2.5D=%s)",
            len(self.samples),
            self._filter_stats["patients_kept"],
            self.prompt_mode,
            self.use_2_5d,
        )

    def _cache_put(self, key: str, value: np.ndarray) -> None:
        if key in self._volume_cache:
            try:
                self._volume_cache_order.remove(key)
            except ValueError:
                pass
        self._volume_cache[key] = value
        self._volume_cache_order.append(key)
        while len(self._volume_cache_order) > self.max_cached_volumes:
            old_key = self._volume_cache_order.pop(0)
            self._volume_cache.pop(old_key, None)

    def _load_volume(self, path: Path) -> Optional[np.ndarray]:
        if not path.exists():
            return None
        key = str(path)
        cached = self._volume_cache.get(key)
        if cached is not None:
            return cached
        itk_image = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(itk_image)
        arr = arr.astype(np.float32, copy=False)
        self._cache_put(key, arr)
        return arr

    def _resolve_records(self) -> None:
        self._filter_stats["patients_input"] = len(self.records)
        for record in self.records:
            image_value = _pick_first_record_value(record, ("image", "image_path", "ct_path"))
            if image_value is None:
                self._filter_stats["patients_no_ct"] += 1
                continue

            image_path = _resolve_manifest_path(image_value, self.manifest_path)
            if not image_path.exists():
                self._filter_stats["patients_no_ct"] += 1
                continue

            mask_value = _pick_first_record_value(record, ("mask", "mask_path", "label", "seg_path"))
            mask_path = _resolve_manifest_path(mask_value, self.manifest_path) if mask_value else None

            resolved = dict(record)
            resolved["_image_path"] = str(image_path)
            resolved["_mask_path"] = str(mask_path) if mask_path else None
            resolved["_patient_id"] = infer_manifest_patient_id(record)

            record_prompt_map = self._extract_record_prompt_map(record)
            external_prompt_map = self._external_prompt_lookup.get(resolved["_patient_id"], {})
            resolved["_prompt_map"] = self._merge_prompt_maps(record_prompt_map, external_prompt_map)
            self._resolved_records.append(resolved)

    @staticmethod
    def _parse_box2d_list(raw_boxes: Any) -> List[List[float]]:
        out: List[List[float]] = []
        if raw_boxes is None:
            return out
        if isinstance(raw_boxes, np.ndarray):
            raw_boxes = raw_boxes.tolist()
        if not isinstance(raw_boxes, list):
            return out
        for box in raw_boxes:
            if isinstance(box, np.ndarray):
                box = box.tolist()
            if isinstance(box, list) and len(box) >= 4:
                try:
                    x1, y1, x2, y2 = [float(v) for v in box[:4]]
                except (TypeError, ValueError):
                    continue
                if x2 > x1 and y2 > y1:
                    out.append([x1, y1, x2, y2])
        return out

    @staticmethod
    def _normalize_prompt_map(raw_map: Any) -> Dict[int, List[List[float]]]:
        normalized: Dict[int, List[List[float]]] = {}
        if raw_map is None:
            return normalized

        if isinstance(raw_map, list):
            # format: [{"slice_index": 10, "bbox": [...]}, ...]
            for item in raw_map:
                if not isinstance(item, dict):
                    continue
                slice_idx = item.get("slice_index")
                if slice_idx is None:
                    continue
                try:
                    z = int(slice_idx)
                except (TypeError, ValueError):
                    continue
                boxes = NiftiManifestDataset._parse_box2d_list(
                    item.get("bboxes", item.get("boxes", [item.get("bbox")]))
                )
                if boxes:
                    normalized.setdefault(z, []).extend(boxes)
            return normalized

        if not isinstance(raw_map, dict):
            return normalized

        for key, value in raw_map.items():
            try:
                z = int(key)
            except (TypeError, ValueError):
                continue
            boxes = NiftiManifestDataset._parse_box2d_list(value)
            if boxes:
                normalized.setdefault(z, []).extend(boxes)
        return normalized

    @staticmethod
    def _merge_prompt_maps(
        a: Dict[int, List[List[float]]], b: Dict[int, List[List[float]]]
    ) -> Dict[int, List[List[float]]]:
        merged: Dict[int, List[List[float]]] = {}
        for source in (a, b):
            for z, boxes in source.items():
                if not boxes:
                    continue
                merged.setdefault(z, []).extend(boxes)
        return merged

    def _load_external_prompt_lookup(self, det_prompt_json: Optional[str]) -> Dict[str, Dict[int, List[List[float]]]]:
        if not det_prompt_json:
            return {}

        path = _resolve_manifest_path(det_prompt_json, self.manifest_path)
        if not path.exists():
            self.logger.warning("det_prompt_json 不存在: %s", path)
            return {}

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        lookup: Dict[str, Dict[int, List[List[float]]]] = {}

        if isinstance(raw, dict) and isinstance(raw.get("patients"), list):
            raw = raw["patients"]

        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                pid = infer_manifest_patient_id(item)
                prompt_map = self._normalize_prompt_map(
                    item.get("slice_bboxes", item.get("det_bboxes", item.get("detections")))
                )
                if prompt_map:
                    lookup[pid] = self._merge_prompt_maps(lookup.get(pid, {}), prompt_map)
            return lookup

        if isinstance(raw, dict):
            for pid, payload in raw.items():
                if isinstance(payload, dict):
                    prompt_map = self._normalize_prompt_map(
                        payload.get("slice_bboxes", payload.get("det_bboxes", payload))
                    )
                    if prompt_map:
                        lookup[str(pid)] = prompt_map
            return lookup

        return lookup

    def _extract_record_prompt_map(self, record: Dict[str, Any]) -> Dict[int, List[List[float]]]:
        prompt_map = {}
        for key in ("slice_bboxes", "det_bboxes", "prompt_bboxes", "bbox_by_slice"):
            value = record.get(key)
            if value is not None:
                prompt_map = self._merge_prompt_maps(prompt_map, self._normalize_prompt_map(value))

        detections = record.get("detections")
        if detections is not None:
            prompt_map = self._merge_prompt_maps(prompt_map, self._normalize_prompt_map(detections))

        # 3D boxes: [x1, y1, z1, x2, y2, z2]
        boxes_3d = record.get("box", record.get("boxes"))
        if isinstance(boxes_3d, np.ndarray):
            boxes_3d = boxes_3d.tolist()
        if isinstance(boxes_3d, list):
            projected: Dict[int, List[List[float]]] = {}
            for box in boxes_3d:
                if not isinstance(box, list) or len(box) < 6:
                    continue
                try:
                    x1, y1, z1, x2, y2, z2 = [float(v) for v in box[:6]]
                except (TypeError, ValueError):
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                start_z = int(np.floor(min(z1, z2)))
                end_z = int(np.ceil(max(z1, z2)))
                if end_z <= start_z:
                    end_z = start_z + 1
                for z in range(start_z, end_z):
                    projected.setdefault(z, []).append([x1, y1, x2, y2])
            prompt_map = self._merge_prompt_maps(prompt_map, projected)

        return prompt_map

    def _build_sample_index(self) -> None:
        kept_patients = set()
        for record_idx, record in enumerate(self._resolved_records):
            image_path = Path(record["_image_path"])
            mask_path = Path(record["_mask_path"]) if record["_mask_path"] else None
            patient_id = record["_patient_id"]

            image_volume = self._load_volume(image_path)
            if image_volume is None or image_volume.ndim != 3:
                self._filter_stats["patients_no_ct"] += 1
                continue

            depth = int(image_volume.shape[0])
            prompt_map = {
                z: boxes for z, boxes in (record.get("_prompt_map") or {}).items()
                if 0 <= int(z) < depth and len(boxes) > 0
            }

            positive_slices: List[int] = []
            mask_volume = None
            if mask_path is not None and mask_path.exists():
                mask_volume = self._load_volume(mask_path)
                if mask_volume is not None and mask_volume.ndim == 3:
                    positive_slices = np.where((mask_volume > 0).reshape(depth, -1).any(axis=1))[0].tolist()

            if positive_slices:
                target_slices = positive_slices
                self._filter_stats["patients_has_nodules"] += 1
            elif prompt_map:
                target_slices = sorted(prompt_map.keys())
                self._filter_stats["patients_has_nodules"] += 1
            elif self.include_empty_slices:
                target_slices = list(range(depth))
                self._filter_stats["patients_no_nodules"] += 1
            else:
                self._filter_stats["patients_no_nodules"] += 1
                self._filter_stats["patients_filtered"] += 1
                continue

            for z in target_slices:
                self.samples.append(
                    {
                        "record_idx": record_idx,
                        "patient_id": patient_id,
                        "slice_index": int(z),
                    }
                )

            self._filter_stats["nodules_total"] += len(target_slices)
            self._filter_stats["nodules_kept"] += len(target_slices)
            self._filter_stats["patients_kept"] += 1
            kept_patients.add(patient_id)

        self._filter_stats["patients_total"] = self._filter_stats["patients_kept"]
        self._filter_stats["patients_input"] = max(
            self._filter_stats["patients_input"],
            len(self._resolved_records),
        )
        self._kept_patient_ids = sorted(list(kept_patients))

    def _resize_image(self, image_2d: np.ndarray) -> np.ndarray:
        import cv2

        target_h, target_w = self.target_size
        if image_2d.shape == (target_h, target_w):
            return image_2d
        return cv2.resize(image_2d.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _resize_mask(self, mask_2d: np.ndarray) -> np.ndarray:
        import cv2

        target_h, target_w = self.target_size
        if mask_2d.shape == (target_h, target_w):
            return (mask_2d > 0).astype(np.uint8)
        resized = cv2.resize(mask_2d.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return (resized > 0).astype(np.uint8)

    def _normalize_to_uint8(self, image_2d: np.ndarray) -> np.ndarray:
        if image_2d.dtype == np.uint8:
            return image_2d

        arr = image_2d.astype(np.float32, copy=False)
        if arr.max() <= 1.0 and arr.min() >= 0.0:
            arr = arr * 255.0
            return np.clip(arr, 0, 255).astype(np.uint8)

        arr = np.clip(arr, self.hu_min, self.hu_max)
        arr = (arr - self.hu_min) / (self.hu_max - self.hu_min + 1e-8)
        arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _extract_mask_bboxes(self, mask: np.ndarray) -> np.ndarray:
        if mask.sum() == 0:
            return np.zeros((0, 4), dtype=np.float32)

        h, w = mask.shape
        labeled_mask = measure.label(mask > 0)
        regions = measure.regionprops(labeled_mask)
        out = []
        for region in regions:
            if self.fixed_bbox_size > 0:
                cy, cx = region.centroid
                half = self.fixed_bbox_size // 2
                x1 = max(0, int(cx - half))
                y1 = max(0, int(cy - half))
                x2 = min(w, int(cx + half))
                y2 = min(h, int(cy + half))
                if x2 > x1 and y2 > y1:
                    out.append([x1, y1, x2, y2])
            else:
                minr, minc, maxr, maxc = region.bbox
                out.append([minc, minr, maxc, maxr])

        if not out:
            return np.zeros((0, 4), dtype=np.float32)
        return np.asarray(out, dtype=np.float32)

    def _rescale_prompt_boxes(
        self,
        boxes: List[List[float]],
        src_hw: Tuple[int, int],
    ) -> np.ndarray:
        if not boxes:
            return np.zeros((0, 4), dtype=np.float32)
        src_h, src_w = src_hw
        target_h, target_w = self.target_size
        sx = float(target_w) / max(1.0, float(src_w))
        sy = float(target_h) / max(1.0, float(src_h))

        out = []
        for box in boxes:
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = box[:4]
            x1 = max(0.0, min(float(target_w - 1), float(x1) * sx))
            y1 = max(0.0, min(float(target_h - 1), float(y1) * sy))
            x2 = max(1.0, min(float(target_w), float(x2) * sx))
            y2 = max(1.0, min(float(target_h), float(y2) * sy))
            if x2 > x1 and y2 > y1:
                out.append([x1, y1, x2, y2])
        if not out:
            return np.zeros((0, 4), dtype=np.float32)
        return np.asarray(out, dtype=np.float32)

    def _jitter_boxes(self, boxes: np.ndarray) -> np.ndarray:
        if self.prompt_jitter_px <= 0 or boxes.size == 0:
            return boxes
        out = boxes.copy()
        noise = np.random.randint(-self.prompt_jitter_px, self.prompt_jitter_px + 1, size=out.shape)
        out += noise.astype(np.float32)
        h, w = self.target_size
        out[:, 0] = np.clip(out[:, 0], 0, w - 1)
        out[:, 1] = np.clip(out[:, 1], 0, h - 1)
        out[:, 2] = np.clip(out[:, 2], 1, w)
        out[:, 3] = np.clip(out[:, 3], 1, h)
        valid = (out[:, 2] > out[:, 0]) & (out[:, 3] > out[:, 1])
        return out[valid]

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        record = self._resolved_records[sample["record_idx"]]
        z = int(sample["slice_index"])

        image_path = Path(record["_image_path"])
        mask_path = Path(record["_mask_path"]) if record["_mask_path"] else None

        image_volume = self._load_volume(image_path)
        if image_volume is None:
            raise RuntimeError(f"image volume not found: {image_path}")
        depth, src_h, src_w = image_volume.shape
        z_prev = max(0, z - 1)
        z_next = min(depth - 1, z + 1)

        if self.use_2_5d:
            slices = [image_volume[z_prev], image_volume[z], image_volume[z_next]]
        else:
            slices = [image_volume[z]]

        resized_slices = [self._normalize_to_uint8(self._resize_image(slc)) for slc in slices]
        if self.use_2_5d:
            image_rgb = np.stack(resized_slices, axis=-1)
        else:
            image_rgb = np.stack([resized_slices[0], resized_slices[0], resized_slices[0]], axis=-1)

        if mask_path is not None and mask_path.exists():
            mask_volume = self._load_volume(mask_path)
            if mask_volume is not None and mask_volume.ndim == 3 and z < mask_volume.shape[0]:
                mask_2d = mask_volume[z]
            else:
                mask_2d = np.zeros((src_h, src_w), dtype=np.uint8)
        else:
            mask_2d = np.zeros((src_h, src_w), dtype=np.uint8)

        mask = self._resize_mask(mask_2d)
        gt_bboxes = self._extract_mask_bboxes(mask)

        prompt_map = record.get("_prompt_map", {})
        prompt_boxes_src = prompt_map.get(z, [])
        det_bboxes = self._rescale_prompt_boxes(prompt_boxes_src, (src_h, src_w))

        if self.prompt_mode == "gt":
            bboxes = gt_bboxes
        elif self.prompt_mode == "det":
            bboxes = det_bboxes if len(det_bboxes) > 0 else gt_bboxes
        else:  # hybrid
            bboxes = det_bboxes if len(det_bboxes) > 0 else gt_bboxes

        bboxes = self._jitter_boxes(bboxes)
        if len(bboxes) == 0:
            bboxes = np.array([[0, 0, 1, 1]], dtype=np.float32)

        return {
            "image": image_rgb,
            "mask": mask.astype(np.uint8),
            "bboxes": bboxes.astype(np.float32),
            "patient_id": sample["patient_id"],
            "slice_index": z,
        }

    def get_filter_stats(self) -> Dict[str, Any]:
        return self._filter_stats.copy()

    def get_kept_patient_ids(self) -> List[str]:
        return list(self._kept_patient_ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self._load_sample(idx)

        if self.transform is not None:
            data = self.transform(data)

        image = torch.from_numpy(data["image"]).permute(2, 0, 1).float()
        mask = torch.from_numpy(data["mask"]).unsqueeze(0).float()
        bboxes = torch.from_numpy(data["bboxes"]).float()

        return {
            "image": image,
            "mask": mask,
            "bboxes": bboxes,
            "patient_id": data["patient_id"],
            "slice_index": data["slice_index"],
        }
