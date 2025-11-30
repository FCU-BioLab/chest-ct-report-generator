#!/usr/bin/env python3
"""
資料集模組
提供胸部CT腫瘤資料集的載入與處理
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
from tqdm import tqdm


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
    """
    
    def __init__(
        self, 
        data_dir: str,
        patient_ids: List[str],
        axis: int = 2,
        transform: Optional[Callable] = None,
        cache_data: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.axis = axis
        self.transform = transform
        self.cache_data = cache_data
        
        self.logger = logging.getLogger(__name__)
        
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
        
        for patient_id in tqdm(self.patient_ids, desc="Indexing"):
            if patient_id not in self.patient_files:
                continue
                
            ct_file = self.patient_files[patient_id]
            
            # 獲取該患者的結節註釋
            patient_nodules = self.annotations[self.annotations['seriesuid'] == patient_id]
            if patient_nodules.empty:
                continue
            
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
                
                # 添加到樣本列表
                for slice_idx in valid_slices:
                    self.samples.append({
                        'patient_id': patient_id,
                        'ct_file': ct_file,
                        'slice_index': slice_idx,
                        'nodules': patient_nodules # 保存註釋以便生成 mask
                    })
            
            except Exception as e:
                self.logger.warning(f"⚠️ 跳過患者 {patient_id}: {e}")
                continue
        
        self.logger.info(f"✅ 找到 {len(self.samples)} 個有效切片")
    
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
            
        # 生成 Mask
        mask_slice = np.zeros_like(ct_slice, dtype=np.uint8)
        
        origin = ct_image.GetOrigin()
        spacing = ct_image.GetSpacing()
        
        for _, row in patient_nodules.iterrows():
            center_world = (row['coordX'], row['coordY'], row['coordZ'])
            diameter = row['diameter_mm']
            radius = diameter / 2.0
            
            # 轉換中心點到體素座標
            center_idx = ct_image.TransformPhysicalPointToIndex(center_world) # (x, y, z)
            
            # 判斷該結節是否在此切片上
            dist_to_slice = 0
            if self.axis == 2:
                dist_to_slice = abs(center_idx[2] - slice_idx) * spacing[2]
            elif self.axis == 1:
                dist_to_slice = abs(center_idx[1] - slice_idx) * spacing[1]
            else:
                dist_to_slice = abs(center_idx[0] - slice_idx) * spacing[0]
                
            if dist_to_slice < radius:
                # 計算切片上的圓半徑
                slice_radius_mm = np.sqrt(radius**2 - dist_to_slice**2)
                
                # 在 mask 上畫圓
                # 注意: sitk 座標是 (x, y, z), numpy 是 (z, y, x)
                # 對於 axial 切片 (z), numpy 是 (y, x)
                
                if self.axis == 2: # Axial: (y, x)
                    cy, cx = center_idx[1], center_idx[0]
                    pixel_radius = slice_radius_mm / np.mean([spacing[0], spacing[1]])
                elif self.axis == 1: # Coronal: (z, x) -> numpy (z, x) ? No, ct_array is (z, y, x) -> slice is (z, x)
                    cy, cx = center_idx[2], center_idx[0]
                    pixel_radius = slice_radius_mm / np.mean([spacing[0], spacing[2]])
                else: # Sagittal: (z, y)
                    cy, cx = center_idx[2], center_idx[1]
                    pixel_radius = slice_radius_mm / np.mean([spacing[1], spacing[2]])
                
                rr, cc = draw.disk((cy, cx), pixel_radius, shape=mask_slice.shape)
                mask_slice[rr, cc] = 1

        # 標準化 CT 切片到 0-255 (Windowing: Lung window usually -1000 to 400, but here min/max)
        # 建議使用固定的 Lung Window (-1000, 400) 或 (-1200, 600)
        # 這裡先維持原有的 min/max 方式，或者改進為 Lung Window
        # 改進：使用多視窗 (Multi-Window) 輸入
        # Channel 1: Lung Window (Wide) -1000 to 400
        w1_min, w1_max = -1000, 400
        img_w1 = np.clip(ct_slice, w1_min, w1_max)
        img_w1 = ((img_w1 - w1_min) / (w1_max - w1_min) * 255).astype(np.uint8)

        # Channel 2: Mediastinum/Soft Tissue Window (W:350 L:50 -> -125 to 225)
        # 幫助區分軟組織、血管和鈣化
        w2_min, w2_max = -125, 225
        img_w2 = np.clip(ct_slice, w2_min, w2_max)
        img_w2 = ((img_w2 - w2_min) / (w2_max - w2_min) * 255).astype(np.uint8)

        # Channel 3: Bone/Wide Window (W:1500 L:300 -> -450 to 1050)
        # 捕捉高密度結構
        w3_min, w3_max = -450, 1050
        img_w3 = np.clip(ct_slice, w3_min, w3_max)
        img_w3 = ((img_w3 - w3_min) / (w3_max - w3_min) * 255).astype(np.uint8)
        
        # 堆疊成 3 通道 RGB
        ct_rgb = np.stack([img_w1, img_w2, img_w3], axis=-1)
        
        # 提取 bounding boxes
        bboxes = self._extract_bboxes(mask_slice)
        
        return {
            'image': ct_rgb,
            'mask': mask_slice,
            'bboxes': bboxes,
            'patient_id': sample_info['patient_id'],
            'slice_index': slice_idx
        }
    
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
    """
    
    def __init__(
        self,
        rotation_prob: float = 0.5,
        flip_prob: float = 0.5,
        gamma_prob: float = 0.3,
        bbox_shift_limit: int = 10
    ):
        self.rotation_prob = rotation_prob
        self.flip_prob = flip_prob
        self.gamma_prob = gamma_prob
        self.bbox_shift_limit = bbox_shift_limit
    
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
