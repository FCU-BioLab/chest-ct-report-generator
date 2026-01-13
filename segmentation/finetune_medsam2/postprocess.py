#!/usr/bin/env python3
"""
MedSAM2 結果後處理模組
======================

將 2D 切片預測結果聚合為 3D NIfTI 體積，並進行後處理優化。

功能：
1. 讀取 `features/predictions` 下的 2D 切片預測 (.npy)
2. 從快取目錄讀取原始幾何資訊 (meta.json)
3. 重建 3D 體積 (Resizing & Stacking)
4. 後處理優化:
   - 最大連通域保留 (Largest Connected Component)
   - 移除小物件 (Small Object Removal)
   - 形態學平滑 (Morphological Smoothing)
5. 保存為 .nii.gz

用法:
    python finetune_medsam2/postprocess.py --result_dir result/segmentation_OUTPUT_DIR
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
import cv2

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class MedSAM2Postprocessor:
    """MedSAM2 後處理器"""
    
    def __init__(
        self, 
        result_dir: str, 
        output_dir: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.result_dir = Path(result_dir)
        
        # 如果未指定 output_dir，預設在 result_dir/nifti
        self.output_dir = Path(output_dir) if output_dir else self.result_dir / "nifti"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 嘗試從 training_config.json 自動獲取 cache_dir
        self.cache_dir = self._resolve_cache_dir(cache_dir)
            
        self.predictions_dir = self.result_dir / "features" / "predictions"
        if not self.predictions_dir.exists():
            # 兼容舊路徑可能有變動
            # 檢查是否有直接在 result_dir 下的 predictions
            if (self.result_dir / "predictions").exists():
                self.predictions_dir = self.result_dir / "predictions"
            else:
                raise FileNotFoundError(f"找不到預測目錄: {self.predictions_dir}")

    def _resolve_cache_dir(self, user_cache_dir: Optional[str]) -> Path:
        """解析快取目錄路徑"""
        if user_cache_dir:
            return Path(user_cache_dir)
        
        config_path = self.result_dir / "training_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Check for absolute path or relative path
                cache_path_str = config.get('cache_dir', 'cache')
                
                # 如果是 data.cache_dir 結構 (config.py save format)
                if isinstance(config.get('data'), dict):
                    cache_path_str = config['data'].get('cache_dir', 'cache')
                
                cache_path = Path(cache_path_str)
                if not cache_path.is_absolute():
                     # 假設相對路徑是相對於專案根目錄 (finetune_medsam2 平級)
                    project_root = Path(__file__).parent.parent
                    cache_path = project_root / cache_path_str
                
                if cache_path.exists():
                    logger.info(f"📂 從設定檔自動偵測到快取目錄: {cache_path}")
                    return cache_path
            except Exception as e:
                logger.warning(f"⚠️ 無法讀取設定檔: {e}")
        
        # Fallback default
        default_cache = Path(__file__).parent.parent / "cache"
        logger.warning(f"⚠️ 無法自動偵測快取目錄，使用預設值: {default_cache}")
        return default_cache

    def find_patient_meta(self, patient_id: str) -> Optional[Tuple[Path, Dict]]:
        """尋找患者的 meta.json"""
        # 搜尋 lndb_slices 和 msd_lung_slices
        for subdir in ['lndb_slices', 'msd_lung_slices']:
            meta_path = self.cache_dir / subdir / patient_id / "meta.json"
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    return meta_path, json.load(f)
        return None, None

    def process_all_patients(
        self, 
        keep_largest: bool = True, 
        remove_small: int = 50,
        morphology_mode: str = 'closing',
        threshold: float = 0.5
    ):
        """處理所有患者"""
        patient_dirs = [d for d in self.predictions_dir.iterdir() if d.is_dir()]
        
        if not patient_dirs:
            logger.warning("⚠️ 找不到任何患者的預測結果")
            return

        logger.info(f"🔄 開始處理 {len(patient_dirs)} 位患者 (Threshold={threshold}, Morphology={morphology_mode})...")
        
        success_count = 0
        for patient_dir in tqdm(patient_dirs, desc="Post-processing"):
            patient_id = patient_dir.name
            try:
                self.process_patient(patient_id, keep_largest, remove_small, morphology_mode, threshold)
                success_count += 1
            except Exception as e:
                logger.error(f"❌ 處理患者 {patient_id} 失敗: {e}")
        
        logger.info(f"✅ 完成! 成功轉換: {success_count}/{len(patient_dirs)}")
        logger.info(f"📁 NIfTI files saved to: {self.output_dir}")

    def process_patient(
        self, 
        patient_id: str,
        keep_largest: bool = True,
        remove_small: int = 50,
        morphology_mode: str = 'closing',
        threshold: float = 0.5
    ):
        """處理單一患者"""
        # 1. 獲取 Metadata
        meta_path, meta = self.find_patient_meta(patient_id)
        if meta is None:
            raise FileNotFoundError(f"找不到患者 {patient_id} 的 meta.json (在 {self.cache_dir})")
        
        original_shape = meta.get('original_shape')  # [D, H, W]
        original_spacing = meta.get('spacing')       # [z, y, x]
        original_origin = meta.get('origin')         # [z, y, x]
        
        # 如果 meta 缺少資訊 (舊版預處理)，嘗試推斷或報錯
        if not original_shape:
            raise ValueError(f"meta.json 缺少 'original_shape' 資訊")

        D, H, W = original_shape
        full_volume = np.zeros((D, H, W), dtype=np.uint8)
        
        # 計算 logit 閾值
        # sigmoid(x) > threshold <=> x > logit(threshold)
        # logit(p) = log(p / (1-p))
        if threshold <= 0 or threshold >= 1:
            raise ValueError("Threshold must be between 0 and 1 exclusive")
        
        logit_threshold = np.log(threshold / (1 - threshold))
        
        # 2. 讀取所有預測切片
        # 預期檔名格式: slice_0001_pred.npy
        pred_files = sorted(list((self.predictions_dir / patient_id).glob("slice_*_pred.npy")))
        
        if not pred_files:
            logger.warning(f"患者 {patient_id} 沒有預測檔案")
            return

        for pred_file in pred_files:
            try:
                # 解析 slice index
                # slice_0123_pred.npy -> 123
                slice_idx = int(pred_file.stem.split('_')[1])
                
                if slice_idx >= D:
                    continue

                # 載入 mask (logits)
                pred_logits = np.load(pred_file)
                
                # 應用閾值
                pred_mask = (pred_logits > logit_threshold).astype(np.uint8)
                
                # Resize 回原始大小 (W, H)
                if pred_mask.shape != (H, W):
                    # 使用 cv2 resize (注意 cv2 是 (W, H))
                    pred_mask_resized = cv2.resize(pred_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                else:
                    pred_mask_resized = pred_mask
                
                full_volume[slice_idx] = pred_mask_resized
                
            except Exception as e:
                logger.warning(f"讀取切片 {pred_file.name} 失敗: {e}")

        # 3. 3D 後處理
        processed_volume = self._apply_postprocessing(
            full_volume, 
            keep_largest=keep_largest, 
            remove_small=remove_small, 
            morphology_mode=morphology_mode
        )

        # 4. 保存為 NIfTI
        sitk_image = sitk.GetImageFromArray(processed_volume)
        
        if original_spacing:
            # SimpleITK 順序是 (x, y, z)
            sitk_image.SetSpacing((float(original_spacing[2]), float(original_spacing[1]), float(original_spacing[0])))
        
        if original_origin:
             sitk_image.SetOrigin((float(original_origin[2]), float(original_origin[1]), float(original_origin[0])))

        output_path = self.output_dir / f"{patient_id}.nii.gz"
        sitk.WriteImage(sitk_image, str(output_path))
        
    def _apply_postprocessing(
        self, 
        volume: np.ndarray,
        keep_largest: bool = True,
        remove_small: int = 50,
        morphology_mode: str = 'closing'
    ) -> np.ndarray:
        """應用 3D 後處理"""
        
        # 1. Morphological Smoothing (Closing/Opening)
        struct = ndimage.generate_binary_structure(3, 1) # 6-connectivity
        
        if morphology_mode == 'closing':
             # 填補小洞，平滑邊界
            volume = ndimage.binary_closing(volume, structure=struct).astype(np.uint8)
        elif morphology_mode == 'opening':
             # 斷開細小連接，去除微小噪點
             volume = ndimage.binary_opening(volume, structure=struct).astype(np.uint8)

        # Labels for connected components
        labeled_array, num_features = ndimage.label(volume)
        
        if num_features == 0:
            return volume

        # 計算每個組件的大小
        sizes = ndimage.sum(volume, labeled_array, range(1, num_features + 1))
        
        mask = np.zeros_like(volume, dtype=np.bool_)
        
        # 2. Keep Largest Component (只保留最大的連通域)
        if keep_largest:
            largest_label = np.argmax(sizes) + 1
            mask = (labeled_array == largest_label)
        
        # 3. Remove Small Objects (移除小於閾值的組件)
        elif remove_small > 0:
            # 保留所有大於閾值的組件
            for i, size in enumerate(sizes):
                if size >= remove_small:
                    mask |= (labeled_array == (i + 1))
        
        else:
            return volume # No filtering

        return mask.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='MedSAM2 後處理工具: 聚合 2D 預測為 3D NIfTI')
    
    parser.add_argument('--result_dir', type=str, required=True, 
                        help='訓練/測試結果目錄 (包含 features/predictions)')
    
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='NIfTI 輸出目錄 (預設為 result_dir/nifti)')
    
    parser.add_argument('--cache_dir', type=str, default=None, 
                        help='快取資料目錄 (用於獲取幾何資訊)，若未指定則嘗試自動偵測')
    
    parser.add_argument('--no_largest', action='store_true', 
                        help='禁用"只保留最大連通域" (預設啟用)')
    
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='二值化閾值 (機率 0.0-1.0)，預設 0.5。提高此值可減少過度分割 (Precision↑, Recall↓)')
    
    parser.add_argument('--morphology', type=str, default='closing', choices=['closing', 'opening', 'none'],
                        help='形態學操作類型: closing (填補小洞, 預設), opening (斷開細小連接/去噪), none (無)')
    
    args = parser.parse_args()
    
    processor = MedSAM2Postprocessor(
        result_dir=args.result_dir, 
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )
    
    processor.process_all_patients(
        keep_largest=not args.no_largest,
        remove_small=args.min_size,
        morphology_mode=args.morphology,
        threshold=args.threshold
    )


if __name__ == '__main__':
    main()
