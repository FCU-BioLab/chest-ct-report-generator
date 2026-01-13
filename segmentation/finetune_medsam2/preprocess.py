#!/usr/bin/env python3
"""
MedSAM2 肺結節分割訓練 - 預處理模組
=====================================

專為 MedSAM2 設計的 CT 影像預處理，特點：
1. 輸出 512×512 切片（MedSAM2 原生尺寸）
2. 不需要 4Patch 分割（與 UNet++ 不同）
3. Spacing Resample（統一到各向同性 1mm）
4. HU Windowing（肺窗設定 [-1000, 200]）
5. Lungmask 肺野外填黑
6. 只保存有病灶的切片

輸出格式：
    cache/
    └── lndb_slices/
        └── LNDb-XXXX/
            ├── meta.json
            └── slice_XXXX.npz  (image, mask, lung_mask, slice_idx)
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import json

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm


logger = logging.getLogger(__name__)

# =============================================================================
# 預設路徑配置
# =============================================================================

# LNDb 資料集根目錄
LNDB_DATA_DIR = Path(r"E:\lung_ct_lesion_dataset\LNDb")

# Lungmask 3D lung mask 目錄
LUNGMASK_DIR = LNDB_DATA_DIR / "lung_masks"

# 預設輸出目錄
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "cache" / "lndb_slices"


# =============================================================================
# MedSAM2 預處理器
# =============================================================================

class MedSAM2Preprocessor:
    """
    MedSAM2 專用 CT 影像預處理器
    
    與 UNet++ 版本的主要差異：
    1. 不做 4Patch 分割
    2. 直接輸出完整切片（resize 到 512×512 在 Dataset 中處理）
    3. 保留原始解析度資訊供後續使用
    """
    
    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        hu_window_center: float = -400,
        hu_window_width: float = 1200,
        lung_margin: int = 10
    ):
        """
        初始化預處理器
        
        Args:
            target_spacing: 目標 spacing (x, y, z) mm，預設 1mm 各向同性
            hu_window_center: HU 窗位（肺窗 -400）
            hu_window_width: HU 窗寬（肺窗 1200，即 [-1000, 200]）
            lung_margin: 肺野 bounding box 邊緣擴展（像素）
        """
        self.target_spacing = np.array(target_spacing)
        self.hu_window_center = hu_window_center
        self.hu_window_width = hu_window_width
        self.lung_margin = lung_margin
        
        # HU 範圍
        self.hu_min = hu_window_center - hu_window_width / 2  # -1000
        self.hu_max = hu_window_center + hu_window_width / 2  # 200
    
    # -------------------------------------------------------------------------
    # 檔案載入
    # -------------------------------------------------------------------------
    
    def load_mhd(self, mhd_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        載入 MHD/RAW 格式的 CT 影像
        
        Returns:
            volume: CT 影像 (Z, Y, X)，HU 值
            spacing: 體素間距 (x, y, z) mm
            origin: 原點座標
        """
        image = sitk.ReadImage(str(mhd_path))
        volume = sitk.GetArrayFromImage(image)  # (Z, Y, X)
        spacing = np.array(image.GetSpacing())  # (x, y, z)
        origin = np.array(image.GetOrigin())
        
        return volume.astype(np.float32), spacing, origin
    
    def load_lungmask(self, patient_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        載入 lungmask 生成的 3D lung mask
        
        Args:
            patient_id: 病人 ID，如 "LNDb-0001"
            
        Returns:
            lung_mask: Binary lung mask (Z, Y, X)，若不存在則為 None
            spacing: Lung mask 的 spacing (x, y, z)
        """
        lung_path = LUNGMASK_DIR / f"{patient_id}_lung.mhd"
        
        if not lung_path.exists():
            logger.warning(f"⚠️ Lungmask 不存在: {lung_path}")
            return None, None
        
        lung_sitk = sitk.ReadImage(str(lung_path))
        lung_array = sitk.GetArrayFromImage(lung_sitk)  # (Z, Y, X)
        lung_spacing = np.array(lung_sitk.GetSpacing())  # (x, y, z)
        
        # Label: 1=右肺, 2=左肺 → binary
        lung_mask = (lung_array > 0).astype(np.float32)
        
        return lung_mask, lung_spacing
    
    # -------------------------------------------------------------------------
    # 影像處理
    # -------------------------------------------------------------------------
    
    def resample_volume(
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
        
        # 計算縮放因子
        # volume 是 (Z, Y, X) 但 spacing 是 (x, y, z)
        scale_factors = current_spacing / target_spacing
        # 轉換為 (z, y, x) 順序對應 volume 的維度
        scale_factors_zyx = np.array([scale_factors[2], scale_factors[1], scale_factors[0]])
        
        # 使用 scipy.ndimage.zoom 進行重新採樣
        resampled = ndimage.zoom(volume, scale_factors_zyx, order=order)
        
        return resampled, target_spacing
    
    def apply_hu_window(self, volume: np.ndarray) -> np.ndarray:
        """
        應用 HU 窗設定並歸一化到 [0, 1]
        
        Args:
            volume: CT 影像（HU 值）
            
        Returns:
            歸一化後的影像 [0, 1]
        """
        # Clipping to window range
        volume = np.clip(volume, self.hu_min, self.hu_max)
        
        # Normalize to [0, 1]
        volume = (volume - self.hu_min) / (self.hu_max - self.hu_min)
        
        return volume.astype(np.float32)
    
    def get_lung_bbox(self, lung_mask: np.ndarray) -> Tuple[slice, slice, slice]:
        """
        獲取肺野的 bounding box
        
        Args:
            lung_mask: 二值肺野遮罩 (Z, Y, X)
            
        Returns:
            (z_slice, y_slice, x_slice) 用於裁切的 slice 物件
        """
        # 找到非零區域
        z_indices, y_indices, x_indices = np.where(lung_mask > 0)
        
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
        margin = self.lung_margin
        z_min = max(0, z_min - margin)
        z_max = min(lung_mask.shape[0], z_max + margin + 1)
        y_min = max(0, y_min - margin)
        y_max = min(lung_mask.shape[1], y_max + margin + 1)
        x_min = max(0, x_min - margin)
        x_max = min(lung_mask.shape[2], x_max + margin + 1)
        
        return (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))
    
    def apply_lung_mask(self, volume: np.ndarray, lung_mask: np.ndarray) -> np.ndarray:
        """
        肺野外區域填黑
        
        Args:
            volume: CT 影像（HU 值）
            lung_mask: 二值肺野遮罩
            
        Returns:
            遮罩後的影像（肺野外為 hu_min）
        """
        return np.where(lung_mask > 0, volume, self.hu_min)
    
    # -------------------------------------------------------------------------
    # 主要預處理流程
    # -------------------------------------------------------------------------
    
    def preprocess_patient(
        self,
        ct_path: str,
        mask_paths: List[str],
        patient_id: str
    ) -> Optional[Dict]:
        """
        預處理單一病人的 CT 資料
        
        Args:
            ct_path: CT 影像路徑 (.mhd)
            mask_paths: 遮罩路徑列表（多位醫師標註）
            patient_id: 病人 ID，如 "LNDb-0001"
            
        Returns:
            預處理結果字典，若失敗則為 None
        """
        # 1. 載入 CT 影像
        volume, spacing, origin = self.load_mhd(ct_path)
        original_shape = volume.shape
        
        # 2. 載入 lungmask
        lung_mask, lung_spacing = self.load_lungmask(patient_id)
        if lung_mask is None:
            return None  # 無 lungmask，跳過此病人
        
        # 3. Resample CT 和 lung mask 到目標 spacing
        volume_resampled, new_spacing = self.resample_volume(volume, spacing, order=1)
        lung_mask_resampled, _ = self.resample_volume(
            lung_mask, spacing, order=0  # 最近鄰插值保持二值
        )
        lung_mask_resampled = (lung_mask_resampled > 0.5).astype(np.float32)
        
        # 4. 獲取肺野 bounding box 並裁切
        bbox = self.get_lung_bbox(lung_mask_resampled)
        volume_cropped = volume_resampled[bbox]
        lung_mask_cropped = lung_mask_resampled[bbox]
        
        # 5. 肺野外填黑（使用 HU 最小值）
        volume_masked = self.apply_lung_mask(volume_cropped, lung_mask_cropped)
        
        # 6. HU windowing → [0, 1]
        volume_normalized = self.apply_hu_window(volume_masked)
        
        # 7. 處理分割遮罩
        masks_processed = []
        for mask_path in mask_paths:
            if Path(mask_path).exists():
                mask, _, _ = self.load_mhd(mask_path)
                mask_resampled, _ = self.resample_volume(mask.astype(np.float32), spacing, order=0)
                mask_cropped = mask_resampled[bbox]
                masks_processed.append((mask_cropped > 0.5).astype(np.float32))
        
        # 8. 建立 binary union mask（任一醫師標註即為結節）
        if len(masks_processed) > 0:
            binary_mask = np.zeros_like(masks_processed[0], dtype=np.float32)
            for m in masks_processed:
                binary_mask = np.logical_or(binary_mask > 0, m > 0).astype(np.float32)
        else:
            binary_mask = np.zeros_like(volume_normalized, dtype=np.float32)
        
        return {
            'volume': volume_normalized,      # 歸一化影像 [0, 1]
            'mask': binary_mask,              # 二值分割遮罩
            'lung_mask': lung_mask_cropped,   # 肺野遮罩
            'spacing': new_spacing,
            'bbox': (
                (bbox[0].start, bbox[0].stop),
                (bbox[1].start, bbox[1].stop),
                (bbox[2].start, bbox[2].stop)
            ),
            'original_shape': original_shape,
            'original_spacing': spacing.tolist()
        }
    
    # -------------------------------------------------------------------------
    # 切片式保存（MedSAM2 格式）
    # -------------------------------------------------------------------------
    
    def save_slices(
        self,
        result: Dict,
        output_dir: Path,
        patient_id: str,
        save_all: bool = False
    ) -> int:
        """
        將預處理結果保存為切片式格式
        
        Args:
            result: preprocess_patient 的輸出
            output_dir: 輸出目錄
            patient_id: 病人 ID
            save_all: 是否保存所有切片（False = 只保存有病灶的切片）
            
        Returns:
            實際保存的切片數
        """
        patient_dir = output_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        volume = result['volume']
        mask = result['mask']
        lung_mask = result['lung_mask']
        
        num_slices = volume.shape[0]
        
        # 找出有病灶的切片
        positive_slices = []
        for z in range(num_slices):
            if np.any(mask[z] > 0):
                positive_slices.append(z)
        
        # 決定要保存的切片
        slices_to_save = list(range(num_slices)) if save_all else positive_slices
        
        # 保存切片
        saved_count = 0
        for z in slices_to_save:
            slice_path = patient_dir / f"slice_{z:04d}.npz"
            np.savez_compressed(
                slice_path,
                image=volume[z].astype(np.float16),      # 節省空間
                mask=mask[z].astype(np.float16),
                lung_mask=lung_mask[z].astype(np.bool_),
                slice_idx=z  # 保存原始切片索引（供 2.5D 模式使用）
            )
            saved_count += 1
        
        # 保存元資料
        meta = {
            'patient_id': patient_id,
            'num_slices': num_slices,
            'saved_slices': saved_count,
            'positive_slices': positive_slices,
            'spacing': result['spacing'].tolist() if hasattr(result['spacing'], 'tolist') else list(result['spacing']),
            'bbox': result['bbox'],
            'original_shape': list(result['original_shape']),
            'original_spacing': result['original_spacing'],
            'preprocessing': {
                'target_spacing': self.target_spacing.tolist(),
                'hu_window': [self.hu_min, self.hu_max],
                'lung_margin': self.lung_margin
            }
        }
        
        with open(patient_dir / "meta.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        return saved_count
    
    def load_slice(self, slice_path: str) -> Dict:
        """載入單一切片"""
        data = np.load(slice_path, allow_pickle=True)
        return {
            'image': data['image'].astype(np.float32),
            'mask': data['mask'].astype(np.float32),
            'lung_mask': data['lung_mask'].astype(bool),
            'slice_idx': int(data['slice_idx'])
        }


# =============================================================================
# 批量預處理函數
# =============================================================================

def preprocess_lndb_for_medsam2(
    data_dir: str = None,
    output_dir: str = None,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    patient_ids: Optional[List[str]] = None,
    save_all_slices: bool = False
):
    """
    批量預處理 LNDb 資料集（MedSAM2 格式）
    
    Args:
        data_dir: LNDb 資料集根目錄
        output_dir: 輸出目錄
        target_spacing: 目標 spacing (x, y, z) mm
        patient_ids: 要處理的病人 ID 列表（None = 全部）
        save_all_slices: 是否保存所有切片（False = 只保存有病灶的）
    """
    data_dir = Path(data_dir) if data_dir else LNDB_DATA_DIR
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = MedSAM2Preprocessor(target_spacing=target_spacing)
    
    # 找到所有 CT 檔案
    ct_files = []
    for subfolder in ['data0', 'data1', 'data2', 'data3', 'data4', 'data5']:
        folder = data_dir / subfolder
        if folder.exists():
            ct_files.extend(folder.glob('*.mhd'))
    
    logger.info(f"📂 找到 {len(ct_files)} 個 CT 檔案")
    
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
        logger.info(f"📋 找到 {len(nodule_patients)} 個有 ≥3mm 結節的病人")
    
    # 統計
    stats = {
        'total': len(ct_files),
        'processed': 0,
        'skipped_no_nodule': 0,
        'skipped_no_lungmask': 0,
        'skipped_exists': 0,
        'total_slices': 0,
        'errors': 0
    }
    
    for ct_path in tqdm(ct_files, desc="🔄 Preprocessing for MedSAM2"):
        patient_id = ct_path.stem  # e.g., LNDb-0001
        
        # 過濾
        if patient_ids and patient_id not in patient_ids:
            continue
        
        if nodule_patients is not None and patient_id not in nodule_patients:
            stats['skipped_no_nodule'] += 1
            continue
        
        # 檢查是否已處理
        patient_dir = output_dir / patient_id
        meta_path = patient_dir / "meta.json"
        if meta_path.exists():
            stats['skipped_exists'] += 1
            continue
        
        # 找遮罩檔案
        mask_paths = []
        for rad_id in range(1, 4):
            mask_path = mask_dir / f"{patient_id}_rad{rad_id}.mhd"
            if mask_path.exists():
                mask_paths.append(str(mask_path))
        
        try:
            # 預處理
            result = preprocessor.preprocess_patient(str(ct_path), mask_paths, patient_id)
            
            if result is None:
                stats['skipped_no_lungmask'] += 1
                continue
            
            # 保存切片
            num_slices = preprocessor.save_slices(result, output_dir, patient_id, save_all_slices)
            stats['total_slices'] += num_slices
            stats['processed'] += 1
            
        except Exception as e:
            logger.error(f"❌ 處理 {patient_id} 時出錯: {e}")
            stats['errors'] += 1
    
    # 輸出統計
    logger.info("=" * 60)
    logger.info("📊 MedSAM2 預處理完成統計")
    logger.info("=" * 60)
    logger.info(f"  總 CT 檔案: {stats['total']}")
    logger.info(f"  ✅ 成功處理: {stats['processed']} 個病人, {stats['total_slices']} 個切片")
    logger.info(f"  ⏭️  跳過（已存在）: {stats['skipped_exists']}")
    logger.info(f"  ⏭️  跳過（無 ≥3mm 結節）: {stats['skipped_no_nodule']}")
    logger.info(f"  ⏭️  跳過（無 lungmask）: {stats['skipped_no_lungmask']}")
    logger.info(f"  ❌ 錯誤: {stats['errors']}")
    logger.info("=" * 60)
    
    return stats


def verify_preprocessed_data(cache_dir: str = None):
    """
    驗證預處理資料的完整性
    
    Args:
        cache_dir: 快取目錄
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_OUTPUT_DIR
    
    if not cache_dir.exists():
        logger.error(f"❌ 快取目錄不存在: {cache_dir}")
        return
    
    patient_dirs = sorted([d for d in cache_dir.iterdir() if d.is_dir()])
    
    logger.info(f"📂 驗證 {len(patient_dirs)} 個病人資料...")
    
    stats = {
        'valid': 0,
        'invalid': 0,
        'total_slices': 0,
        'issues': []
    }
    
    for patient_dir in tqdm(patient_dirs, desc="Verifying"):
        patient_id = patient_dir.name
        meta_path = patient_dir / "meta.json"
        
        # 檢查 meta.json
        if not meta_path.exists():
            stats['invalid'] += 1
            stats['issues'].append(f"{patient_id}: missing meta.json")
            continue
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # 檢查切片數量
        expected_slices = meta.get('saved_slices', 0)
        actual_slices = len(list(patient_dir.glob('slice_*.npz')))
        
        if expected_slices != actual_slices:
            stats['invalid'] += 1
            stats['issues'].append(f"{patient_id}: expected {expected_slices} slices, found {actual_slices}")
            continue
        
        stats['valid'] += 1
        stats['total_slices'] += actual_slices
    
    logger.info("=" * 60)
    logger.info("📊 驗證結果")
    logger.info("=" * 60)
    logger.info(f"  ✅ 有效: {stats['valid']} 個病人, {stats['total_slices']} 個切片")
    logger.info(f"  ❌ 無效: {stats['invalid']} 個病人")
    
    if stats['issues']:
        logger.info("  問題列表:")
        for issue in stats['issues'][:10]:  # 只顯示前 10 個
            logger.info(f"    - {issue}")
        if len(stats['issues']) > 10:
            logger.info(f"    ... 還有 {len(stats['issues']) - 10} 個問題")
    
    logger.info("=" * 60)
    
    return stats


# =============================================================================
# 主程式
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MedSAM2 專用 LNDb 資料集預處理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 預處理所有資料（只保存有病灶的切片）
  python preprocess.py

  # 保存所有切片（包含無病灶的）
  python preprocess.py --save-all

  # 指定輸出目錄
  python preprocess.py --output-dir ./my_cache

  # 驗證預處理結果
  python preprocess.py --verify
        """
    )
    
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help=f"LNDb 資料集根目錄 (預設: {LNDB_DATA_DIR})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"輸出目錄 (預設: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        help="目標 spacing (x, y, z) mm (預設: 1.0 1.0 1.0)"
    )
    parser.add_argument(
        "--save-all", action="store_true",
        help="保存所有切片（預設只保存有病灶的切片）"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="驗證預處理資料的完整性"
    )
    
    args = parser.parse_args()
    
    # 設定 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.verify:
        verify_preprocessed_data(args.output_dir)
    else:
        logger.info("🚀 MedSAM2 預處理器")
        logger.info(f"   目標 Spacing: {args.spacing}")
        logger.info(f"   保存模式: {'所有切片' if args.save_all else '只保存有病灶的切片'}")
        
        preprocess_lndb_for_medsam2(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            target_spacing=tuple(args.spacing),
            save_all_slices=args.save_all
        )
