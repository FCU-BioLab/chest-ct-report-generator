#!/usr/bin/env python3
"""
UNet++ 肺結節分割訓練 - 預處理模組
===================================

提供 CT 影像預處理功能：
1. Spacing Resample（統一到各向同性）
2. HU Windowing（肺窗設定）
3. Lung ROI Crop（肺野粗分割和裁切）
4. 預處理結果快取
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import json

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage import morphology, measure
from tqdm import tqdm


logger = logging.getLogger(__name__)

# Lungmask 3D lung mask 目錄
LUNGMASK_DIR = Path(r"E:\lung_ct_lesion_dataset\LNDb\lung_masks")


class CTPreprocessor:
    """CT 影像預處理器"""
    
    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        hu_window_center: float = -400,
        hu_window_width: float = 1200,
        lung_margin: int = 10,
        cache_dir: Optional[str] = None
    ):
        """
        初始化預處理器
        
        Args:
            target_spacing: 目標 spacing (x, y, z) mm
            hu_window_center: HU 窗位（肺窗 -400）
            hu_window_width: HU 窗寬（肺窗 1200，即 [-1000, 200]）
            lung_margin: 肺野 bounding box 邊緣擴展（像素）
            cache_dir: 快取目錄（None 表示不快取）
        """
        self.target_spacing = np.array(target_spacing)
        self.hu_window_center = hu_window_center
        self.hu_window_width = hu_window_width
        self.lung_margin = lung_margin
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_mhd(self, mhd_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        載入 MHD/RAW 格式的 CT 影像
        
        Returns:
            volume: CT 影像 (Z, Y, X)
            spacing: 體素間距 (x, y, z)
            origin: 原點座標
        """
        image = sitk.ReadImage(str(mhd_path))
        volume = sitk.GetArrayFromImage(image)  # (Z, Y, X)
        spacing = np.array(image.GetSpacing())  # (x, y, z)
        origin = np.array(image.GetOrigin())
        
        return volume.astype(np.float32), spacing, origin
    
    def load_lungmask_3d(self, patient_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        載入 lungmask 生成的 3D lung mask
        
        Args:
            patient_id: 病人 ID，如 "LNDb-0001"
            
        Returns:
            lung_mask: Binary lung mask (Z, Y, X)
            spacing: Lung mask 的 spacing (x, y, z)
        """
        lung_path = LUNGMASK_DIR / f"{patient_id}_lung.mhd"
        
        if not lung_path.exists():
            logger.warning(f"Lungmask 不存在: {lung_path}，使用 threshold-based fallback")
            return None, None
        
        lung_sitk = sitk.ReadImage(str(lung_path))
        lung_array = sitk.GetArrayFromImage(lung_sitk)  # (Z, Y, X)
        lung_spacing = np.array(lung_sitk.GetSpacing())  # (x, y, z)
        
        # Label 1=右肺, 2=左肺 → binary
        lung_mask = (lung_array > 0).astype(np.float32)
        
        return lung_mask, lung_spacing
    
    def resample_to_isotropic(
        self,
        volume: np.ndarray,
        current_spacing: np.ndarray,
        target_spacing: Optional[np.ndarray] = None,
        order: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        將影像重新採樣到目標 spacing
        
        Args:
            volume: 輸入影像 (Z, Y, X)
            current_spacing: 當前 spacing (x, y, z)
            target_spacing: 目標 spacing，None 使用預設
            order: 插值階數（0=最近鄰, 1=線性, 3=三次）
            
        Returns:
            resampled: 重新採樣後的影像
            new_spacing: 新的 spacing
        """
        if target_spacing is None:
            target_spacing = self.target_spacing
        
        # 計算縮放因子（注意 volume 是 Z,Y,X 但 spacing 是 x,y,z）
        scale_factors = current_spacing / target_spacing
        # 轉換為 (z, y, x) 順序
        scale_factors = np.array([scale_factors[2], scale_factors[1], scale_factors[0]])
        
        # 計算新形狀
        new_shape = np.round(np.array(volume.shape) * scale_factors).astype(int)
        
        # 使用 scipy.ndimage.zoom 進行重新採樣
        zoom_factors = new_shape / np.array(volume.shape)
        resampled = ndimage.zoom(volume, zoom_factors, order=order)
        
        return resampled, target_spacing
    
    def apply_hu_window(
        self,
        volume: np.ndarray,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None
    ) -> np.ndarray:
        """
        應用 HU 窗設定並歸一化到 [0, 1]
        
        Args:
            volume: CT 影像（HU 值）
            window_center: 窗位
            window_width: 窗寬
            
        Returns:
            歸一化後的影像 [0, 1]
        """
        if window_center is None:
            window_center = self.hu_window_center
        if window_width is None:
            window_width = self.hu_window_width
        
        min_hu = window_center - window_width / 2
        max_hu = window_center + window_width / 2
        
        # Clipping
        volume = np.clip(volume, min_hu, max_hu)
        
        # Normalize to [0, 1]
        volume = (volume - min_hu) / (max_hu - min_hu)
        
        return volume.astype(np.float32)
    
    def segment_lung_mask(self, volume: np.ndarray) -> np.ndarray:
        """
        粗分割肺野遮罩
        
        使用閾值 + 形態學操作提取肺野區域
        
        Args:
            volume: CT 影像（HU 值）
            
        Returns:
            lung_mask: 二值肺野遮罩
        """
        # 1. 閾值分割（肺野 HU 值約 -1000 ~ -500）
        binary = volume < -400  # 低於此閾值的區域
        
        # 2. 清除邊界連接的區域（通常是身體外的空氣）
        cleared = np.zeros_like(binary)
        for z in range(binary.shape[0]):
            slice_binary = binary[z]
            # 填充邊界
            slice_cleared = ndimage.binary_fill_holes(slice_binary)
            # 保留內部區域
            cleared[z] = slice_cleared
        
        # 3. 形態學開運算去除小區域
        struct = morphology.ball(2)
        opened = morphology.binary_opening(cleared, struct[:3, :, :])
        
        # 4. 保留最大的兩個連通區域（左右肺）
        labels = measure.label(opened)
        regions = measure.regionprops(labels)
        
        if len(regions) == 0:
            logger.warning("無法分割肺野，返回全 1 遮罩")
            return np.ones_like(volume, dtype=bool)
        
        # 按面積排序，保留最大的兩個
        regions = sorted(regions, key=lambda x: x.area, reverse=True)
        lung_mask = np.zeros_like(labels, dtype=bool)
        for i, region in enumerate(regions[:2]):
            lung_mask[labels == region.label] = True
        
        # 5. 閉運算填充肺內部空洞
        lung_mask = morphology.binary_closing(lung_mask, struct[:3, :, :])
        
        return lung_mask
    
    def get_lung_bounding_box(
        self,
        lung_mask: np.ndarray,
        margin: Optional[int] = None
    ) -> Tuple[slice, slice, slice]:
        """
        獲取肺野的 bounding box
        
        Args:
            lung_mask: 二值肺野遮罩
            margin: 邊緣擴展像素數
            
        Returns:
            (z_slice, y_slice, x_slice) 用於裁切的 slice 物件
        """
        if margin is None:
            margin = self.lung_margin
        
        # 找到非零區域
        z_indices, y_indices, x_indices = np.where(lung_mask)
        
        if len(z_indices) == 0:
            # 如果沒有肺野，返回完整範圍
            return (
                slice(0, lung_mask.shape[0]),
                slice(0, lung_mask.shape[1]),
                slice(0, lung_mask.shape[2])
            )
        
        z_min, z_max = z_indices.min(), z_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        # 加上 margin
        z_min = max(0, z_min - margin)
        z_max = min(lung_mask.shape[0], z_max + margin + 1)
        y_min = max(0, y_min - margin)
        y_max = min(lung_mask.shape[1], y_max + margin + 1)
        x_min = max(0, x_min - margin)
        x_max = min(lung_mask.shape[2], x_max + margin + 1)
        
        return (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))
    
    def preprocess_volume(
        self,
        mhd_path: str,
        mask_paths: Optional[List[str]] = None,
        patient_id: Optional[str] = None
    ) -> Dict:
        """
        完整預處理流程
        
        Args:
            mhd_path: CT 影像路徑
            mask_paths: 遮罩路徑列表（多位醫師）
            patient_id: 病人 ID，用於載入 lungmask（如 "LNDb-0001"）
            
        Returns:
            包含預處理結果的字典
        """
        # 1. 載入影像
        volume, spacing, origin = self.load_mhd(mhd_path)
        original_shape = volume.shape
        
        # 2. 肺野分割：必須使用 lungmask 3D mhd
        if not patient_id:
            raise ValueError("必須提供 patient_id 以載入 lungmask")
        
        lung_mask, lung_spacing = self.load_lungmask_3d(patient_id)
        if lung_mask is None:
            # Lungmask 不存在，返回 None 表示跳過此病人
            return None
        
        # 3. Resample CT 和 lung mask
        volume_resampled, new_spacing = self.resample_to_isotropic(volume, spacing)
        lung_mask_resampled, _ = self.resample_to_isotropic(
            lung_mask.astype(np.float32), spacing, order=0  # 最近鄰插值
        )
        lung_mask_resampled = lung_mask_resampled > 0.5
        
        # 4. Lung ROI crop（使用 lungmask 的 bbox）
        bbox = self.get_lung_bounding_box(lung_mask_resampled)
        volume_cropped = volume_resampled[bbox]
        lung_mask_cropped = lung_mask_resampled[bbox]
        
        # 5. HU windowing（保留原始 HU 值作為副本，用於特徵提取）
        volume_hu = volume_cropped.copy()
        volume_normalized = self.apply_hu_window(volume_cropped)
        
        # 6. 處理遮罩（如果有）
        masks_processed = []
        if mask_paths:
            for mask_path in mask_paths:
                if Path(mask_path).exists():
                    mask, _, _ = self.load_mhd(mask_path)
                    mask_resampled, _ = self.resample_to_isotropic(
                        mask.astype(np.float32), spacing, order=0
                    )
                    mask_cropped = mask_resampled[bbox]
                    masks_processed.append((mask_cropped > 0.5).astype(np.float32))
        
        result = {
            'volume': volume_normalized,  # 歸一化後的影像 [0, 1]
            'volume_hu': volume_hu,        # 原始 HU 值
            'lung_mask': lung_mask_cropped,
            'masks': masks_processed,      # 多醫師遮罩列表
            'spacing': new_spacing,
            'bbox': (
                (bbox[0].start, bbox[0].stop),
                (bbox[1].start, bbox[1].stop),
                (bbox[2].start, bbox[2].stop)
            ),
            'original_shape': original_shape,
            'original_spacing': spacing.tolist()
        }
        
        return result
    
    def create_soft_consensus_mask(self, masks: List[np.ndarray]) -> np.ndarray:
        """
        創建軟共識遮罩
        
        將多位醫師的標註疊加成軟標籤 [0, 1]
        
        Args:
            masks: 遮罩列表
            
        Returns:
            soft_mask: 軟共識遮罩
        """
        if len(masks) == 0:
            raise ValueError("遮罩列表為空")
        
        if len(masks) == 1:
            return masks[0].astype(np.float32)
        
        stacked = np.stack(masks, axis=0)
        consensus = stacked.sum(axis=0) / len(masks)
        
        return consensus.astype(np.float32)
    
    def save_preprocessed(self, result: Dict, save_path: str):
        """保存預處理結果到 NPZ 檔案"""
        np.savez_compressed(
            save_path,
            volume=result['volume'],
            volume_hu=result['volume_hu'],
            lung_mask=result['lung_mask'],
            masks=np.array(result['masks']) if result['masks'] else np.array([]),
            spacing=result['spacing'],
            bbox=np.array(result['bbox']),
            original_shape=np.array(result['original_shape']),
            original_spacing=np.array(result['original_spacing'])
        )
    
    def load_preprocessed(self, load_path: str) -> Dict:
        """從 NPZ 檔案載入預處理結果"""
        data = np.load(load_path, allow_pickle=True)
        
        masks = data['masks']
        if masks.ndim == 0:
            masks = []
        else:
            masks = list(masks)
        
        return {
            'volume': data['volume'],
            'volume_hu': data['volume_hu'],
            'lung_mask': data['lung_mask'],
            'masks': masks,
            'spacing': data['spacing'],
            'bbox': tuple(tuple(x) for x in data['bbox']),
            'original_shape': tuple(data['original_shape']),
            'original_spacing': list(data['original_spacing'])
        }
    
    def save_slices(self, result: Dict, output_dir: str, patient_id: str):
        """保存為切片式儲存（每個切片一個檔案）"""
        import os
        patient_dir = Path(output_dir) / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        volume = result['volume']
        lung_mask = result['lung_mask']
        masks = result['masks']
        
        # 創建 binary union 遮罩（任一醫師標註即為前景，符合 CSEA-Net 論文）
        if len(masks) > 0:
            # Union: 任一醫師標註 > 0 即為結節
            binary_mask = np.zeros_like(masks[0], dtype=np.float32)
            for m in masks:
                binary_mask = np.logical_or(binary_mask > 0, m > 0).astype(np.float32)
        else:
            binary_mask = np.zeros_like(volume, dtype=np.float32)
        
        num_slices = int(volume.shape[0])  # 確保是 Python int
        
        # 保存元資料（確保所有值都是 JSON 可序列化的）
        spacing = result['spacing']
        if hasattr(spacing, 'tolist'):
            spacing = spacing.tolist()
        else:
            spacing = [float(x) for x in spacing]
        
        original_shape = result['original_shape']
        if hasattr(original_shape, 'tolist'):
            original_shape = original_shape.tolist()
        else:
            original_shape = [int(x) for x in original_shape]
        
        original_spacing = result['original_spacing']
        if hasattr(original_spacing, 'tolist'):
            original_spacing = original_spacing.tolist()
        else:
            original_spacing = [float(x) for x in original_spacing]
        
        # bbox 轉換
        bbox = result['bbox']
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        else:
            bbox = [[int(x) for x in b] if hasattr(b, '__iter__') else int(b) for b in bbox]
        
        meta = {
            'num_slices': num_slices,
            'spacing': spacing,
            'bbox': bbox,
            'original_shape': original_shape,
            'original_spacing': original_spacing,
            'positive_slices': []
        }
        
        # 找出正樣本切片
        for z in range(num_slices):
            if np.any(binary_mask[z] > 0):
                meta['positive_slices'].append(int(z))  # 確保是 Python int
        
        # 保存每個切片
        for z in range(num_slices):
            slice_path = patient_dir / f"slice_{z:04d}.npz"
            np.savez_compressed(
                slice_path,
                image=volume[z].astype(np.float16),
                mask=binary_mask[z].astype(np.float16),
                lung_mask=lung_mask[z].astype(np.bool_)
            )
        
        # 保存元資料
        import json
        with open(patient_dir / "meta.json", 'w') as f:
            json.dump(meta, f)
        
        return num_slices
    
    def load_slice(self, slice_path: str) -> Dict:
        """載入單一切片"""
        data = np.load(slice_path, allow_pickle=True)
        return {
            'image': data['image'].astype(np.float32),
            'mask': data['mask'].astype(np.float32),
            'lung_mask': data['lung_mask']
        }


def preprocess_lndb_dataset(
    data_dir: str,
    output_dir: str,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    patient_ids: Optional[List[str]] = None
):
    """
    批量預處理 LNDb 資料集
    
    Args:
        data_dir: LNDb 資料集根目錄
        output_dir: 輸出目錄
        target_spacing: 目標 spacing
        patient_ids: 要處理的病人 ID 列表（None = 全部）
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = CTPreprocessor(target_spacing=target_spacing)
    
    # 找到所有 CT 檔案
    ct_files = []
    for subfolder in ['data0', 'data1', 'data2', 'data3', 'data4', 'data5']:
        folder = data_dir / subfolder
        if folder.exists():
            ct_files.extend(folder.glob('*.mhd'))
    
    # 找到對應的遮罩
    mask_dir = data_dir / 'mask' / 'masks'
    
    processed_count = 0
    for ct_path in tqdm(ct_files, desc="Preprocessing"):
        patient_id = ct_path.stem  # e.g., LNDb-0001
        
        if patient_ids and patient_id not in patient_ids:
            continue
        
        output_path = output_dir / f"{patient_id}.npz"
        if output_path.exists():
            logger.info(f"跳過已存在: {patient_id}")
            continue
        
        # 找到對應的遮罩（可能有多位醫師）
        mask_paths = []
        for rad_id in range(1, 4):
            mask_path = mask_dir / f"{patient_id}_rad{rad_id}.mhd"
            if mask_path.exists():
                mask_paths.append(str(mask_path))
        
        try:
            result = preprocessor.preprocess_volume(str(ct_path), mask_paths, patient_id=patient_id)
            
            # 如果 lungmask 不存在，跳過此病人
            if result is None:
                logger.warning(f"跳過 {patient_id}：無 lungmask 3D 遮罩")
                continue
            
            preprocessor.save_preprocessed(result, str(output_path))
            processed_count += 1
        except Exception as e:
            logger.error(f"處理 {patient_id} 時出錯: {e}")
    
    logger.info(f"完成！共處理 {processed_count} 個病人")


def preprocess_lndb_slices(
    data_dir: str,
    output_dir: str,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    patient_ids: Optional[List[str]] = None
):
    """
    切片式預處理 LNDb 資料集（每個切片獨立保存）
    
    Args:
        data_dir: LNDb 資料集根目錄
        output_dir: 輸出目錄
        target_spacing: 目標 spacing
        patient_ids: 要處理的病人 ID 列表（None = 全部）
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = CTPreprocessor(target_spacing=target_spacing)
    
    # 找到所有 CT 檔案
    ct_files = []
    for subfolder in ['data0', 'data1', 'data2', 'data3', 'data4', 'data5']:
        folder = data_dir / subfolder
        if folder.exists():
            ct_files.extend(folder.glob('*.mhd'))
    
    # 找到對應的遮罩目錄
    mask_dir = data_dir / 'mask' / 'masks'
    
    # 過濾只保留有 ≥3mm 結節的病人
    nodule_csv = data_dir / 'trainset_csv' / 'trainNodules_gt.csv'
    nodule_patients = None
    if nodule_csv.exists():
        import pandas as pd
        df = pd.read_csv(nodule_csv)
        # Nodule=1 表示是結節（≥3mm 有分割遮罩）
        nodule_patients = set(f"LNDb-{int(pid):04d}" for pid in df[df['Nodule'] == 1]['LNDbID'].unique())
        logger.info(f"找到 {len(nodule_patients)} 個有 ≥3mm 結節的病人")
    
    total_slices = 0
    processed_count = 0
    skipped_no_nodule = 0
    skipped_no_lungmask = 0
    
    for ct_path in tqdm(ct_files, desc="Preprocessing slices"):
        patient_id = ct_path.stem
        
        if patient_ids and patient_id not in patient_ids:
            continue
        
        # 跳過沒有 ≥3mm 結節的病人
        if nodule_patients is not None and patient_id not in nodule_patients:
            skipped_no_nodule += 1
            continue
        
        patient_dir = output_dir / patient_id
        meta_path = patient_dir / "meta.json"
        
        if meta_path.exists():
            logger.debug(f"跳過已存在: {patient_id}")
            continue
        
        # 找遮罩
        mask_paths = []
        for rad_id in range(1, 4):
            mask_path = mask_dir / f"{patient_id}_rad{rad_id}.mhd"
            if mask_path.exists():
                mask_paths.append(str(mask_path))
        
        try:
            result = preprocessor.preprocess_volume(str(ct_path), mask_paths, patient_id=patient_id)
            
            # 如果 lungmask 不存在，跳過此病人
            if result is None:
                skipped_no_lungmask += 1
                logger.warning(f"跳過 {patient_id}：無 lungmask 3D 遮罩")
                continue
            
            num_slices = preprocessor.save_slices(result, str(output_dir), patient_id)
            total_slices += num_slices
            processed_count += 1
        except Exception as e:
            logger.error(f"處理 {patient_id} 時出錯: {e}")
    
    logger.info(f"完成！共處理 {processed_count} 個病人，{total_slices} 個切片")
    if skipped_no_nodule > 0:
        logger.info(f"跳過 {skipped_no_nodule} 個無 ≥3mm 結節的病人")
    if skipped_no_lungmask > 0:
        logger.info(f"跳過 {skipped_no_lungmask} 個無 lungmask 的病人")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="預處理 LNDb 資料集")
    parser.add_argument("--data_dir", type=str, default="datasets/aLL_patients_data/LNDb")
    parser.add_argument("--output_dir", type=str, default="cache/lndb_preprocessed")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    preprocess_lndb_dataset(
        args.data_dir,
        args.output_dir,
        tuple(args.spacing)
    )
