#!/usr/bin/env python3
"""
資料集模組
提供 LNDb 胸部 CT 資料集的載入與處理
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable

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
    資料增強類別
    
    提供隨機旋轉、翻轉、gamma 調整等增強方法
    
    Args:
        rotation_prob: 旋轉機率
        flip_prob: 翻轉機率
        gamma_prob: Gamma 調整機率
        bbox_shift_limit: Bounding Box 擾動範圍 (pixels)
        noise_prob: 高斯噪音機率
        contrast_prob: 對比度調整機率
        scale_prob: 縮放增強機率
    """
    
    def __init__(
        self,
        rotation_prob: float = 0.5,
        flip_prob: float = 0.5,
        gamma_prob: float = 0.3,
        bbox_shift_limit: int = 10,
        noise_prob: float = 0.2,
        contrast_prob: float = 0.3,
        scale_prob: float = 0.3
    ):
        self.rotation_prob = rotation_prob
        self.flip_prob = flip_prob
        self.gamma_prob = gamma_prob
        self.bbox_shift_limit = bbox_shift_limit
        self.noise_prob = noise_prob
        self.contrast_prob = contrast_prob
        self.scale_prob = scale_prob
    
    def __call__(self, data: Dict) -> Dict:
        """應用資料增強"""
        import random
        
        image = data['image']  # [H, W, 3]
        mask = data['mask']    # [H, W]
        bboxes = data['bboxes']  # [N, 4]
        
        # 隨機旋轉 (90, 180, 270 度)
        if random.random() < self.rotation_prob:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        
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
        
        # 高斯噪音
        if random.random() < self.noise_prob:
            noise_std = random.uniform(3, 10)
            noise = np.random.normal(0, noise_std, image.shape)
            image = image.astype(np.float32) + noise
            image = np.clip(image, 0, 255).astype(np.uint8)
        
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
            
            if len(bboxes) > 0:
                if scale > 1:
                    bboxes = (bboxes * scale) - np.array([start_w, start_h, start_w, start_h])
                else:
                    bboxes = (bboxes * scale) + np.array([pad_w, pad_h, pad_w, pad_h])
                bboxes = np.clip(bboxes, 0, [w, h, w, h])
            
        # Bounding Box 擾動
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
        """載入單個樣本"""
        sample_info = self.samples[idx]
        lndb_id = sample_info['lndb_id']
        ct_file = sample_info['ct_file']
        slice_idx = sample_info['slice_index']
        
        ct_image = sitk.ReadImage(str(ct_file))
        ct_array = sitk.GetArrayFromImage(ct_image)
        
        if self.axis == 2:
            ct_slice = ct_array[slice_idx, :, :]
        elif self.axis == 1:
            ct_slice = ct_array[:, slice_idx, :]
        else:
            ct_slice = ct_array[:, :, slice_idx]
        
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
        
        # Resize 到固定尺寸 (512x512)
        target_size = (512, 512)
        orig_h, orig_w = ct_slice.shape
        
        if (orig_h, orig_w) != target_size:
            import cv2
            ct_slice = cv2.resize(ct_slice.astype(np.float32), target_size, interpolation=cv2.INTER_LINEAR)
            mask_slice = cv2.resize(mask_slice.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
        
        # CT 值裁剪與標準化
        hu_min, hu_max = -1000, 800
        ct_clipped = np.clip(ct_slice, hu_min, hu_max)
        ct_normalized = ((ct_clipped - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)
        
        ct_rgb = np.stack([ct_normalized, ct_normalized, ct_normalized], axis=-1)
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
        target_size: tuple = (512, 512)
    ):
        self.cache_dir = Path(cache_dir)
        self.dataset_type = dataset_type.lower()
        self.patient_ids = patient_ids
        self.transform = transform
        self.target_size = target_size
        
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
                        f"(類型: {dataset_type})")
    
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
        """載入單個樣本"""
        import cv2
        
        sample_info = self.samples[idx]
        npz_path = sample_info['npz_path']
        
        # 載入 .npz 檔案
        data = np.load(str(npz_path))
        image = data['image']  # 2D array
        mask = data['mask']    # 2D array
        
        # Resize 到目標尺寸
        if image.shape != self.target_size:
            image = cv2.resize(image.astype(np.float32), self.target_size, 
                              interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask.astype(np.uint8), self.target_size, 
                             interpolation=cv2.INTER_NEAREST)
        
        # 標準化到 0-255 (如果尚未標準化)
        if image.max() > 255 or image.min() < -1000:
            # 假設是 HU 值，進行 CT 視窗處理
            hu_min, hu_max = -1000, 800
            image = np.clip(image, hu_min, hu_max)
            image = ((image - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)
        elif image.max() <= 1.0:
            # 已經 normalize 到 0-1
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # 轉換為 RGB (3 通道)
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        
        # 確保 mask 是二值的
        mask = (mask > 0).astype(np.uint8)
        
        # 提取 bounding boxes
        bboxes = self._extract_bboxes(mask)
        
        return {
            'image': image,
            'mask': mask,
            'bboxes': bboxes,
            'patient_id': sample_info['patient_id'],
            'slice_index': sample_info['slice_index']
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
