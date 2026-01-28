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


class VolumePreprocessor:
    """
    CT Settings -> NPZ Volume Converter
    
    Supported datasets:
    - LNDb: Lung Nodule Database
    - MSD Lung: Medical Segmentation Decathlon - Lung Tumors
    """
    
    def __init__(
        self,
        output_dir: str,
        context_slices: int = 32,
        min_depth: int = 1,
        max_depth: int = 70,
        min_nodule_diameter: float = 0.0,
        image_size: int = 256,
        full_volume: bool = False,
        window_center: float = -600,
        window_width: float = 1500,
        min_agreement: int = 1,
    ):
        """
        初始化預處理器
        
        Args:
            output_dir: NPZ output directory
            context_slices: slices before/after center
            min_depth: min volume depth
            max_depth: max volume depth
            min_nodule_diameter: min nodule diameter (mm)
            image_size: output image size
            window_center: CT window center
            window_width: CT window width
            min_agreement: min radiologist agreement level
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.context_slices = context_slices
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_nodule_diameter = min_nodule_diameter
        self.image_size = image_size
        self.full_volume = full_volume
        self.window_center = window_center
        self.window_width = window_width
        self.min_agreement = min_agreement
        
        # Statistics
        self.stats = {
            'total_patients': 0,
            'total_lesions': 0,
            'converted_lesions': 0,
            'filtered_small': 0,
            'filtered_edge': 0,
            'filtered_edge': 0,
            'errors': 0,
        }
        self.generated_files = []


        logger.info(f"📁 Volume Preprocessor initialized")
        logger.info(f"  - Output dir: {self.output_dir}")
        logger.info(f"  - Context: ±{context_slices} (Depth {2*context_slices+1})")
        logger.info(f"  - 最小結節: {min_nodule_diameter}mm")
        logger.info(f"  - 最小共識: >= {min_agreement} 位醫師")
    
    def convert_lndb(
        self,
        lndb_dir: str,
        patient_ids: Optional[List] = None,
    ) -> Dict:
        """
        轉換 LNDb 資料集
        
        Args:
            lndb_dir: LNDb 根目錄
            patient_ids: 指定患者ID（可選），None 表示全部
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
        
        # 建立輸出目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📦 處理 {len(patient_ids)} 位患者 (不進行預先分割)")
        
        for patient_id in tqdm(patient_ids, desc="Converting LNDb"):
            try:
                self._convert_lndb_patient(
                    patient_id=patient_id,
                    lndb_path=lndb_path,
                    ct_files=ct_files,
                    mask_files=mask_files,
                    nodules_df=nodules_df,
                    output_split="", # No split subfolder
                )
            except Exception as e:
                logger.error(f"❌ 患者 {patient_id} 轉換失敗: {e}")
                self.stats['errors'] += 1
                
                self.stats['errors'] += 1
                
        self.save_manifest()
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
        """轉換單個 LNDb 患者的病灶 (基於 CSV 篩選與座標)"""
        if patient_id not in ct_files:
            return
        
        # 篩選該患者的病灶 (Agreement >= min_agreement)
        patient_nodules = nodules_df[
            (nodules_df['LNDbID'] == patient_id) & 
            (nodules_df['AgrLevel'] >= self.min_agreement) &
            (nodules_df['Nodule'] == 1)
        ]
        
        if len(patient_nodules) == 0:
            return

        # 載入 CT
        ct_path = ct_files[patient_id]
        ct_image = sitk.ReadImage(str(ct_path))
        ct_array = sitk.GetArrayFromImage(ct_image)  # (Z, Y, X)
        spacing = np.array(ct_image.GetSpacing()[::-1])  # (z, y, x)
        origin = np.array(ct_image.GetOrigin())          # (x, y, z) ITK order
        
        # CT 窗位/窗寬正規化
        ct_normalized = self._apply_windowing(ct_array)
        
        # 預先載入該患者所有可用的 Mask (RadID -> MaskArray)
        # 這樣不用對每個病灶重複讀取 IO
        loaded_masks = {}
        if patient_id in mask_files:
            for rad_id, m_path in mask_files[patient_id].items():
                try:
                    m_img = sitk.ReadImage(str(m_path))
                    m_arr = sitk.GetArrayFromImage(m_img)
                    loaded_masks[rad_id] = m_arr
                except Exception as e:
                    logger.warning(f"⚠️ 無法讀取 Mask (P{patient_id}, R{rad_id}): {e}")

        # 處理每個符合條件的病灶
        for _, nodule_row in patient_nodules.iterrows():
            finding_id = nodule_row['FindingID']
            
            # 解析有哪些放射科醫師標記了這個病灶 (RadID column: "1,2,3")
            rad_ids_str = str(nodule_row['RadID'])
            target_rad_ids = []
            if ',' in rad_ids_str:
                target_rad_ids = [int(r) for r in rad_ids_str.replace('"', '').split(',') if r.strip().isdigit()]
            elif rad_ids_str.isdigit():
                target_rad_ids = [int(rad_ids_str)]
            
            # 只保留我們有 Mask 檔案的 RadID
            valid_rad_ids = [r for r in target_rad_ids if r in loaded_masks]
            
            if not valid_rad_ids:
                # 雖然 CSV 說有病灶，但找不到對應 Mask (可能檔案遺失)
                # logger.warning(f"⚠️ 跳過病灶 P{patient_id}-F{finding_id}: 無對應 Mask 檔案")
                continue

            # 取得中心座標 (World -> Voxel)
            # CSV x,y,z is in World coordinates
            center_world = np.array([nodule_row['x'], nodule_row['y'], nodule_row['z']])
            
            # World to Voxel conversion: (World - Origin) / Spacing
            # Note: ITK Origin/Spacing are (x, y, z), but array is (z, y, x)
            # center_world is (x, y, z) from CSV
            
            vx = (center_world[0] - origin[0]) / ct_image.GetSpacing()[0]
            vy = (center_world[1] - origin[1]) / ct_image.GetSpacing()[1]
            vz = (center_world[2] - origin[2]) / ct_image.GetSpacing()[2]
            
            center_voxel = np.array([vz, vy, vx])  # (z, y, x)
            z_center_int = int(round(vz))
            y_center_int = int(round(vy))
            x_center_int = int(round(vx))
            
            # --- Mask Fusion ---
            # 找出每個醫師 Mask 中包含該中心點的 Component
            fused_mask_bool = np.zeros(ct_array.shape, dtype=bool)
            has_valid_mask = False
            
            for r_id in valid_rad_ids:
                mask_arr = loaded_masks[r_id]
                # 檢查中心點是否在 Mask 範圍內
                if (0 <= z_center_int < mask_arr.shape[0] and
                    0 <= y_center_int < mask_arr.shape[1] and
                    0 <= x_center_int < mask_arr.shape[2]):
                        
                    # 如果中心點剛好是 0 (醫師可能畫歪了一點)，嘗試搜尋附近
                    # 簡單版：只看中心點是否有值
                    # 進階版：做 label region props，找距離最近的 region
                    
                    val = mask_arr[z_center_int, y_center_int, x_center_int]
                    if val > 0:
                        # 找出該連通區域
                        labeled_temp = measure.label(mask_arr == val) # 假設 mask 值區分病灶
                        # 找出中心點所在的 label
                        target_label = labeled_temp[z_center_int, y_center_int, x_center_int]
                        if target_label > 0:
                            fused_mask_bool |= (labeled_temp == target_label)
                            has_valid_mask = True
                    else:
                        # Fallback: Search small radius? 
                        # 暫時略過複雜搜索，假設座標準確
                        pass

            if not has_valid_mask:
                # 嘗試放寬搜索：在中心點附近 3x3x3 找 mask
                # 這裡為了效能先跳過，統計如果有大量這種情況再加
                # logger.debug(f"P{patient_id}-F{finding_id}: 中心點無 Mask 覆蓋")
                self.stats['filtered_edge'] += 1 # 借用這個計數器或新增一個 'no_mask_match'
                continue

            # 計算最終 fused mask 的屬性
            final_mask_uint8 = fused_mask_bool.astype(np.uint8)
            regions = measure.regionprops(measure.label(final_mask_uint8))
            if not regions:
                continue
            
            # 取最大的 region (理論上只有一個，因為我們是針對單一病灶融合)
            region = regions[0]
            if len(regions) > 1:
                # 非常罕見，如果不同醫師畫的區域完全不重疊
                region = max(regions, key=lambda r: r.area)

            # --- 下面接續原本的切片與儲存邏輯 ---
            
            # 計算病灶大小
            volume_mm3 = region.area * np.prod(spacing)
            diameter_mm = 2 * np.power(3 * volume_mm3 / (4 * np.pi), 1/3)
            
            # 過濾小結節
            if diameter_mm < self.min_nodule_diameter:
                self.stats['filtered_small'] += 1
                continue
            
            # 取得病灶中心切片 (使用 bbox 中心，比 CSV 座標更準確對應切割後的 Mask)
            z_min, y_min, x_min, z_max, y_max, x_max = region.bbox
            z_center = (z_min + z_max) // 2
            
            # 計算視頻範圍
            if self.full_volume:
                 z_start = 0
                 z_end = ct_array.shape[0]
            else:
                z_start = max(0, z_center - self.context_slices)
                z_end = min(ct_array.shape[0], z_center + self.context_slices + 1)
                
                depth = z_end - z_start
                if depth < self.min_depth:
                    self.stats['filtered_edge'] += 1
                    continue
            
            # 截取視頻區間
            video_frames = ct_normalized[z_start:z_end]  # (D, Y, X)
            video_masks = final_mask_uint8[z_start:z_end]
            
            center_idx = z_center - z_start
            
            # 調整大小到 image_size
            video_frames = self._resize_volume(video_frames)
            video_masks = self._resize_volume(video_masks, is_mask=True)
            
            # 計算 bbox
            if center_idx < video_masks.shape[0] and video_masks[center_idx].max() > 0:
                bbox = self._compute_bbox_from_mask(video_masks[center_idx])
            else:
                # 尋找最大面積切片
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

            # 儲存 NPZ
            # 使用 FindingID 作為 lesion 編號，以保持與 CSV 一致
            output_path = self.output_dir / output_split / f"LNDb-{patient_id:04d}_lesion{finding_id:02d}.npz"
            self.generated_files.append(str(output_path))
            
            np.savez_compressed(
                output_path,
                frames=video_frames,
                masks=video_masks,
                center_idx=center_idx,
                slice_indices=list(range(z_start, z_end)),
                patient_id=f"LNDb-{patient_id:04d}",
                lesion_id=finding_id,
                diameter_mm=diameter_mm,
                volume_mm3=volume_mm3,
                spacing=spacing,
                origin=origin,
                lesion_center_csv=center_world, # 紀錄 CSV 原始座標
                original_shape=ct_array.shape,
                bbox=bbox,
                agreement=nodule_row['AgrLevel']
            )
            
            self.stats['converted_lesions'] += 1
    
    def convert_msd_lung(
        self,
        msd_dir: str,
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
        
        self.stats['total_patients'] = len(case_ids)
        
        # 建立輸出目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📦 處理 {len(case_ids)} 案例 (不進行預先分割)")
        
        for case_id in tqdm(case_ids, desc="Converting MSD"):
            try:
                self._convert_msd_case(
                    case_id=case_id,
                    images_dir=images_dir,
                    labels_dir=labels_dir,
                    output_split="", # No split subfolder
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
            
            if self.full_volume:
                z_start = 0
                z_end = image_array.shape[0]
            else:
                z_start = max(0, z_center - self.context_slices)
                z_end = min(image_array.shape[0], z_center + self.context_slices + 1)
                
                depth = z_end - z_start
                if depth < self.min_depth:
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
    
    def save_manifest(self):
        """Save a log of generated files and statistics"""
        manifest_path = self.output_dir / "data.log"
        manifest = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "config": {
                "min_agreement": self.min_agreement,
                "context_slices": self.context_slices,
                "min_nodule_diameter": self.min_nodule_diameter,
                "image_size": self.image_size,
                "full_volume": self.full_volume
            },
            "stats": self.stats,
            "generated_files": self.generated_files
        }
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            
        logger.info(f"📝 Data log saved to: {manifest_path}")
        logger.info(f"  - 成功轉換: {self.stats['converted_lesions']}")
        logger.info(f"  - 過濾（太小）: {self.stats['filtered_small']}")
        logger.info(f"  - 過濾（邊緣）: {self.stats['filtered_edge']}")
        logger.info(f"  - 錯誤: {self.stats['errors']}")
        logger.info("=" * 50)
    
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
    """Command line volume converter"""
    parser = argparse.ArgumentParser(description='Convert CT dataset to NPZ Volume format')
    parser.add_argument('--dataset', type=str, required=True, choices=['lndb', 'msd'],
                       help='Dataset type')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input dataset directory')
    parser.add_argument('--output_dir', type=str, default='volume_npz',
                       help='NPZ Output directory')
    parser.add_argument('--context_slices', type=int, default=32,
                       help='Slices before/after center')
    parser.add_argument('--min_diameter', type=float, default=0.0,
                       help='Min nodule diameter (mm)')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Output image size')
    parser.add_argument('--full_volume', action='store_true',
                       help='Convert full volume instead of cropping')
    
    parser.add_argument('--min_agreement', type=int, default=1,
                       help='Min radiologist agreement level (1-3)')
    
    args = parser.parse_args()
    
    preprocessor = VolumePreprocessor(
        output_dir=args.output_dir,
        context_slices=args.context_slices,
        min_nodule_diameter=args.min_diameter,
        image_size=args.image_size,
        full_volume=args.full_volume,
        min_agreement=args.min_agreement,
    )
    
    if args.dataset == 'lndb':
        preprocessor.convert_lndb(args.input_dir)
    else:
        preprocessor.convert_msd_lung(args.input_dir)


if __name__ == '__main__':
    main()
