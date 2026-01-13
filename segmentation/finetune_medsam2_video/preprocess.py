#!/usr/bin/env python3
"""
MedSAM2 視頻預處理模組
======================

將 LNDb/MSD 資料集轉換為 MedSAM2 視頻訓練所需的 NPZ 格式。

每個病灶生成一個 NPZ 檔案，包含：
- 以病灶為中心的連續 CT 切片序列（視頻幀）
- 對應的分割遮罩序列
- 病灶資訊（位置、大小等）
- BBox Prompt

用法:
    python preprocess.py --dataset lndb --input_dir /path/to/lndb --output_dir cache/video_npz
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import json
import argparse

import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from skimage import measure
from PIL import Image

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class VideoPreprocessor:
    """
    CT 資料集 → NPZ 視頻格式轉換器
    
    支援的資料集：
    - LNDb: 肺結節資料庫
    - MSD Lung: 醫學分割十項全能 - 肺腫瘤
    """
    
    def __init__(
        self,
        output_dir: str,
        context_slices: int = 6,
        min_video_length: int = 5,
        max_video_length: int = 32,
        min_nodule_diameter: float = 4.0,
        image_size: int = 512,
        window_center: float = -600,
        window_width: float = 1500,
    ):
        """
        初始化預處理器
        
        Args:
            output_dir: NPZ 輸出目錄
            context_slices: 中心切片前後各取幾個切片
            min_video_length: 最短視頻長度
            max_video_length: 最長視頻長度
            min_nodule_diameter: 最小結節直徑過濾 (mm)
            image_size: 輸出影像大小
            window_center: CT 窗位
            window_width: CT 窗寬
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.context_slices = context_slices
        self.min_video_length = min_video_length
        self.max_video_length = max_video_length
        self.min_nodule_diameter = min_nodule_diameter
        self.image_size = image_size
        self.window_center = window_center
        self.window_width = window_width
        
        # 統計資訊
        self.stats = {
            'total_patients': 0,
            'total_lesions': 0,
            'converted_lesions': 0,
            'filtered_small': 0,
            'filtered_edge': 0,
            'errors': 0,
        }
        
        logger.info(f"📁 視頻預處理器初始化完成")
        logger.info(f"  - 輸出目錄: {self.output_dir}")
        logger.info(f"  - 上下文切片: ±{context_slices} (視頻長度 {2*context_slices+1})")
        logger.info(f"  - 最小結節: {min_nodule_diameter}mm")
    
    def convert_lndb(
        self,
        lndb_dir: str,
        patient_ids: Optional[List] = None,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> Dict:
        """
        轉換 LNDb 資料集
        
        Args:
            lndb_dir: LNDb 根目錄
            patient_ids: 指定患者ID（可選），None 表示全部
            split_ratios: (train, val, test) 分割比例
        """
        lndb_path = Path(lndb_dir)
        logger.info(f"🔄 開始轉換 LNDb 資料集: {lndb_path}")
        
        # 載入標註
        nodules_df = self._load_lndb_annotations(lndb_path)
        if nodules_df is None:
            return self.stats
        
        # 索引 CT 和 Mask 檔案
        ct_files = self._index_lndb_ct_files(lndb_path)
        mask_files = self._index_lndb_mask_files(lndb_path)
        
        # 取得所有患者 ID
        if patient_ids is None:
            patient_ids = sorted(nodules_df['LNDbID'].unique())
        
        self.stats['total_patients'] = len(patient_ids)
        
        # 分割資料集
        np.random.seed(42)
        np.random.shuffle(patient_ids)
        n = len(patient_ids)
        train_end = int(n * split_ratios[0])
        val_end = train_end + int(n * split_ratios[1])
        
        splits = {
            'train': patient_ids[:train_end],
            'val': patient_ids[train_end:val_end],
            'test': patient_ids[val_end:],
        }
        
        # 建立輸出目錄
        for split in splits:
            (self.output_dir / split).mkdir(exist_ok=True)
        
        # 處理每個患者
        for split_name, split_ids in splits.items():
            logger.info(f"📦 處理 {split_name} split ({len(split_ids)} 患者)")
            
            for patient_id in tqdm(split_ids, desc=f"Converting {split_name}"):
                try:
                    self._convert_lndb_patient(
                        patient_id=patient_id,
                        lndb_path=lndb_path,
                        ct_files=ct_files,
                        mask_files=mask_files,
                        nodules_df=nodules_df,
                        output_split=split_name,
                    )
                except Exception as e:
                    logger.error(f"❌ 患者 {patient_id} 轉換失敗: {e}")
                    self.stats['errors'] += 1
        
        self._log_stats()
        return self.stats
    
    def _load_lndb_annotations(self, lndb_path: Path) -> Optional[pd.DataFrame]:
        """載入 LNDb 標註"""
        csv_path = lndb_path / 'trainset_csv' / 'trainNodules_gt.csv'
        if not csv_path.exists():
            csv_path = lndb_path / 'trainNodules_gt.csv'
        
        if not csv_path.exists():
            logger.error(f"❌ 找不到標註檔案: {csv_path}")
            return None
        
        df = pd.read_csv(csv_path)
        logger.info(f"📋 載入 {len(df)} 個結節標註")
        return df
    
    def _index_lndb_ct_files(self, lndb_path: Path) -> Dict[int, Path]:
        """索引 LNDb CT 檔案"""
        ct_files = {}
        for i in range(6):
            data_dir = lndb_path / f'data{i}'
            if data_dir.exists():
                for mhd in data_dir.glob('LNDb-*.mhd'):
                    lndb_id = int(mhd.stem.split('-')[1])
                    ct_files[lndb_id] = mhd
        
        logger.info(f"📂 找到 {len(ct_files)} 個 CT 掃描")
        return ct_files
    
    def _index_lndb_mask_files(self, lndb_path: Path) -> Dict[int, Dict[int, Path]]:
        """索引 LNDb Mask 檔案"""
        mask_files = {}
        
        # 嘗試多種可能的 mask 目錄結構
        possible_mask_dirs = [
            lndb_path / 'mask' / 'masks',   # 常見結構: mask/masks/
            lndb_path / 'masks' / 'masks',  # 備用結構: masks/masks/
            lndb_path / 'mask',             # 備用結構: mask/
            lndb_path / 'masks',            # 備用結構: masks/
        ]
        
        mask_dir = None
        for possible_dir in possible_mask_dirs:
            if possible_dir.exists():
                # 檢查是否有 .mhd 檔案
                if list(possible_dir.glob('LNDb-*_rad*.mhd')):
                    mask_dir = possible_dir
                    logger.info(f"✅ 找到 Mask 目錄: {mask_dir}")
                    break
        
        if mask_dir is None:
            logger.warning(f"⚠️ 找不到 Mask 目錄，嘗試過: {[str(d) for d in possible_mask_dirs]}")
            return mask_files
        
        if mask_dir.exists():
            for mhd in mask_dir.glob('LNDb-*_rad*.mhd'):
                parts = mhd.stem.split('_')
                lndb_id = int(parts[0].split('-')[1])
                rad_id = int(parts[1].replace('rad', ''))
                
                if lndb_id not in mask_files:
                    mask_files[lndb_id] = {}
                mask_files[lndb_id][rad_id] = mhd
        
        logger.info(f"📂 找到 {len(mask_files)} 個患者有 Mask")
        return mask_files
    
    def _convert_lndb_patient(
        self,
        patient_id: int,
        lndb_path: Path,
        ct_files: Dict[int, Path],
        mask_files: Dict[int, Dict[int, Path]],
        nodules_df: pd.DataFrame,
        output_split: str,
    ):
        """轉換單個 LNDb 患者的所有病灶"""
        if patient_id not in ct_files or patient_id not in mask_files:
            return
        
        # 載入 CT
        ct_path = ct_files[patient_id]
        ct_image = sitk.ReadImage(str(ct_path))
        ct_array = sitk.GetArrayFromImage(ct_image)  # (Z, Y, X)
        spacing = np.array(ct_image.GetSpacing()[::-1])  # (z, y, x)
        origin = np.array(ct_image.GetOrigin())
        
        # CT 窗位/窗寬正規化
        ct_normalized = self._apply_windowing(ct_array)
        
        # 載入 Mask（選擇第一個可用的放射科醫師）
        rad_masks = mask_files[patient_id]
        mask_path = list(rad_masks.values())[0]
        mask_image = sitk.ReadImage(str(mask_path))
        mask_array = sitk.GetArrayFromImage(mask_image)
        
        # 從 mask 中識別連通區域（每個代表一個病灶）
        labeled_mask = measure.label(mask_array > 0)
        regions = measure.regionprops(labeled_mask)
        
        self.stats['total_lesions'] += len(regions)
        
        # 處理每個病灶
        for region_idx, region in enumerate(regions):
            lesion_id = region_idx + 1
            
            # 計算病灶大小
            volume_mm3 = region.area * np.prod(spacing)
            diameter_mm = 2 * np.power(3 * volume_mm3 / (4 * np.pi), 1/3)
            
            # 過濾小結節
            if diameter_mm < self.min_nodule_diameter:
                self.stats['filtered_small'] += 1
                continue
            
            # 取得病灶中心切片
            z_min, y_min, x_min, z_max, y_max, x_max = region.bbox
            z_center = (z_min + z_max) // 2
            
            # 計算視頻範圍
            z_start = max(0, z_center - self.context_slices)
            z_end = min(ct_array.shape[0], z_center + self.context_slices + 1)
            
            # 檢查視頻長度
            video_length = z_end - z_start
            if video_length < self.min_video_length:
                self.stats['filtered_edge'] += 1
                continue
            
            # 截取視頻區間
            video_frames = ct_normalized[z_start:z_end]  # (D, Y, X)
            video_masks = (labeled_mask[z_start:z_end] == region.label).astype(np.uint8)
            
            # 調整中心索引
            center_idx = z_center - z_start
            
            # 調整大小到 image_size
            video_frames = self._resize_volume(video_frames)
            video_masks = self._resize_volume(video_masks, is_mask=True)
            
            # 計算 bbox（在 resize 後的座標系中）
            if center_idx < video_masks.shape[0] and video_masks[center_idx].max() > 0:
                bbox = self._compute_bbox_from_mask(video_masks[center_idx])
            else:
                # 非常罕見的情況，中心切片沒有 mask
                # 嘗試在整個視頻中找最大的 mask
                max_area = 0
                best_idx = 0
                for i in range(video_masks.shape[0]):
                    area = np.sum(video_masks[i])
                    if area > max_area:
                        max_area = area
                        best_idx = i
                
                if max_area > 0:
                    bbox = self._compute_bbox_from_mask(video_masks[best_idx])
                else:
                    # 完全沒有 mask，使用中心點
                    h, w = self.image_size, self.image_size
                    cx, cy = w // 2, h // 2
                    bbox = np.array([cx-10, cy-10, cx+10, cy+10], dtype=np.float32)

            
            # 病灶中心世界座標
            lesion_center = region.centroid  # (z, y, x) in voxel
            lesion_center_world = np.array([
                origin[0] + lesion_center[2] * spacing[2],  # x
                origin[1] + lesion_center[1] * spacing[1],  # y
                origin[2] + lesion_center[0] * spacing[0],  # z
            ])
            
            # 儲存 NPZ
            output_path = self.output_dir / output_split / f"LNDb-{patient_id:04d}_lesion{lesion_id:02d}.npz"
            
            np.savez_compressed(
                output_path,
                # 視頻資料
                frames=video_frames,  # (D, H, W) uint8
                masks=video_masks,    # (D, H, W) uint8
                
                # 索引資訊
                center_idx=center_idx,
                slice_indices=list(range(z_start, z_end)),
                
                # 病灶資訊
                patient_id=f"LNDb-{patient_id:04d}",
                lesion_id=lesion_id,
                diameter_mm=diameter_mm,
                volume_mm3=volume_mm3,
                
                # 空間資訊
                spacing=spacing,
                origin=origin,
                lesion_center_world=lesion_center_world,
                original_shape=ct_array.shape,
                
                # Prompt
                bbox=bbox,
            )
            
            self.stats['converted_lesions'] += 1
    
    def convert_msd_lung(
        self,
        msd_dir: str,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> Dict:
        """
        轉換 MSD Lung Tumors 資料集
        """
        msd_path = Path(msd_dir)
        logger.info(f"🔄 開始轉換 MSD Lung 資料集: {msd_path}")
        
        images_dir = msd_path / 'imagesTr'
        labels_dir = msd_path / 'labelsTr'
        
        if not images_dir.exists() or not labels_dir.exists():
            logger.error(f"❌ 找不到 MSD 資料目錄")
            return self.stats
        
        # 取得所有案例
        case_files = sorted(images_dir.glob('lung_*.nii.gz'))
        case_ids = [f.stem.replace('.nii', '') for f in case_files]
        
        self.stats['total_patients'] = len(case_ids)
        
        # 分割資料集
        np.random.seed(42)
        np.random.shuffle(case_ids)
        n = len(case_ids)
        train_end = int(n * split_ratios[0])
        val_end = train_end + int(n * split_ratios[1])
        
        splits = {
            'train': case_ids[:train_end],
            'val': case_ids[train_end:val_end],
            'test': case_ids[val_end:],
        }
        
        # 建立輸出目錄
        for split in splits:
            (self.output_dir / split).mkdir(exist_ok=True)
        
        # 處理每個案例
        for split_name, split_ids in splits.items():
            logger.info(f"📦 處理 {split_name} split ({len(split_ids)} 案例)")
            
            for case_id in tqdm(split_ids, desc=f"Converting {split_name}"):
                try:
                    self._convert_msd_case(
                        case_id=case_id,
                        images_dir=images_dir,
                        labels_dir=labels_dir,
                        output_split=split_name,
                    )
                except Exception as e:
                    logger.error(f"❌ 案例 {case_id} 轉換失敗: {e}")
                    self.stats['errors'] += 1
        
        self._log_stats()
        return self.stats
    
    def _convert_msd_case(
        self,
        case_id: str,
        images_dir: Path,
        labels_dir: Path,
        output_split: str,
    ):
        """轉換單個 MSD 案例"""
        # 載入影像和標籤
        image_path = images_dir / f"{case_id}.nii.gz"
        label_path = labels_dir / f"{case_id}.nii.gz"
        
        if not image_path.exists() or not label_path.exists():
            return
        
        image = sitk.ReadImage(str(image_path))
        label = sitk.ReadImage(str(label_path))
        
        image_array = sitk.GetArrayFromImage(image)  # (Z, Y, X)
        label_array = sitk.GetArrayFromImage(label)
        spacing = np.array(image.GetSpacing()[::-1])
        origin = np.array(image.GetOrigin())
        
        # CT 窗位正規化
        ct_normalized = self._apply_windowing(image_array)
        
        # 識別腫瘤區域
        labeled_mask = measure.label(label_array > 0)
        regions = measure.regionprops(labeled_mask)
        
        self.stats['total_lesions'] += len(regions)
        
        for region_idx, region in enumerate(regions):
            lesion_id = region_idx + 1
            
            # 計算大小
            volume_mm3 = region.area * np.prod(spacing)
            diameter_mm = 2 * np.power(3 * volume_mm3 / (4 * np.pi), 1/3)
            
            if diameter_mm < self.min_nodule_diameter:
                self.stats['filtered_small'] += 1
                continue
            
            # 視頻範圍
            z_min, y_min, x_min, z_max, y_max, x_max = region.bbox
            z_center = (z_min + z_max) // 2
            
            z_start = max(0, z_center - self.context_slices)
            z_end = min(image_array.shape[0], z_center + self.context_slices + 1)
            
            video_length = z_end - z_start
            if video_length < self.min_video_length:
                self.stats['filtered_edge'] += 1
                continue
            
            video_frames = ct_normalized[z_start:z_end]
            video_masks = (labeled_mask[z_start:z_end] == region.label).astype(np.uint8)
            center_idx = z_center - z_start
            
            # 調整大小
            video_frames = self._resize_volume(video_frames)
            video_masks = self._resize_volume(video_masks, is_mask=True)
            
            # 計算 bbox
            if center_idx < video_masks.shape[0] and video_masks[center_idx].max() > 0:
                bbox = self._compute_bbox_from_mask(video_masks[center_idx])
            else:
                 # 同上fallback
                max_area = 0
                best_idx = 0
                for i in range(video_masks.shape[0]):
                    area = np.sum(video_masks[i])
                    if area > max_area:
                        max_area = area
                        best_idx = i
                
                if max_area > 0:
                    bbox = self._compute_bbox_from_mask(video_masks[best_idx])
                else:
                    h, w = self.image_size, self.image_size
                    cx, cy = w // 2, h // 2
                    bbox = np.array([cx-10, cy-10, cx+10, cy+10], dtype=np.float32)

            
            # 儲存
            output_path = self.output_dir / output_split / f"{case_id}_lesion{lesion_id:02d}.npz"
            
            np.savez_compressed(
                output_path,
                frames=video_frames,
                masks=video_masks,
                center_idx=center_idx,
                slice_indices=list(range(z_start, z_end)),
                patient_id=case_id,
                lesion_id=lesion_id,
                diameter_mm=diameter_mm,
                volume_mm3=volume_mm3,
                spacing=spacing,
                origin=origin,
                original_shape=image_array.shape,
                bbox=bbox,
            )
            
            self.stats['converted_lesions'] += 1
    
    def _apply_windowing(self, ct_array: np.ndarray) -> np.ndarray:
        """套用 CT 窗位/窗寬正規化"""
        min_val = self.window_center - self.window_width / 2
        max_val = self.window_center + self.window_width / 2
        
        ct_clipped = np.clip(ct_array, min_val, max_val)
        ct_normalized = ((ct_clipped - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        return ct_normalized
    
    def _resize_volume(
        self, 
        volume: np.ndarray, 
        is_mask: bool = False
    ) -> np.ndarray:
        """調整體積大小"""
        D, H, W = volume.shape
        
        if H == self.image_size and W == self.image_size:
            return volume
        
        resized = np.zeros((D, self.image_size, self.image_size), dtype=volume.dtype)
        
        for i in range(D):
            img = Image.fromarray(volume[i])
            if is_mask:
                img_resized = img.resize((self.image_size, self.image_size), Image.NEAREST)
            else:
                img_resized = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            resized[i] = np.array(img_resized)
        
        return resized
    
    def _compute_bbox_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """從 mask 計算 bounding box"""
        if mask.max() == 0:
            h, w = mask.shape
            cx, cy = w // 2, h // 2
            return np.array([cx - 10, cy - 10, cx + 10, cy + 10], dtype=np.float32)
        
        ys, xs = np.where(mask > 0)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        
        # 稍微擴大
        pad = 5
        h, w = mask.shape
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    
    def _log_stats(self):
        """輸出統計資訊"""
        logger.info("=" * 50)
        logger.info("📊 轉換統計")
        logger.info(f"  - 總患者數: {self.stats['total_patients']}")
        logger.info(f"  - 總病灶數: {self.stats['total_lesions']}")
        logger.info(f"  - 成功轉換: {self.stats['converted_lesions']}")
        logger.info(f"  - 過濾（太小）: {self.stats['filtered_small']}")
        logger.info(f"  - 過濾（邊緣）: {self.stats['filtered_edge']}")
        logger.info(f"  - 錯誤: {self.stats['errors']}")
        logger.info("=" * 50)


def main():
    """命令列轉換工具"""
    parser = argparse.ArgumentParser(description='將 CT 資料集轉換為 MedSAM2 視頻格式')
    parser.add_argument('--dataset', type=str, required=True, choices=['lndb', 'msd'],
                       help='資料集類型')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='輸入資料集目錄')
    parser.add_argument('--output_dir', type=str, default='video_npz',
                       help='NPZ 輸出目錄')
    parser.add_argument('--context_slices', type=int, default=6,
                       help='中心切片前後各取幾個切片')
    parser.add_argument('--min_diameter', type=float, default=4.0,
                       help='最小結節直徑 (mm)')
    parser.add_argument('--image_size', type=int, default=512,
                       help='輸出影像大小')
    
    args = parser.parse_args()
    
    preprocessor = VideoPreprocessor(
        output_dir=args.output_dir,
        context_slices=args.context_slices,
        min_nodule_diameter=args.min_diameter,
        image_size=args.image_size,
    )
    
    if args.dataset == 'lndb':
        preprocessor.convert_lndb(args.input_dir)
    else:
        preprocessor.convert_msd_lung(args.input_dir)


if __name__ == '__main__':
    main()
