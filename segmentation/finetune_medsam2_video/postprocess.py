#!/usr/bin/env python3
"""
MedSAM2 視頻後處理模組
======================

將模型預測結果（NPZ/NPY）轉換回原始 CT 空間幾何，並保存為 NIfTI 格式。

功能：
1. 讀取預測結果 (Mask)
2. 讀取原始輸入 NPZ (獲取原始幾何資訊: Spacing, Origin, BBox)
3. 將預測 Mask 重採樣回原始解析度
4. 還原到原始 3D 空間位置 (Padding/Paste)
5. 保存為 .nii.gz 供醫學影像軟體查看

用法:
    python postprocess.py --pred_dir results/preds --input_dir cache/video_npz --output_dir results/nifti
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Union
import argparse

import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class VideoPostprocessor:
    """
    模型預測 → NIfTI 轉換器
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_batch(self, pred_dir: str, input_dir: str):
        """
        批量處理目錄中的預測檔
        
        Args:
            pred_dir: 預測結果目錄 (含 .npz 或 .npy)
            input_dir: 原始輸入目錄 (含 .npz)
        """
        pred_path = Path(pred_dir)
        input_path = Path(input_dir)
        
        # 尋找所有預測檔
        pred_files = list(pred_path.glob("**/*.npz")) + list(pred_path.glob("**/*.npy"))
        
        if not pred_files:
            logger.warning(f"⚠️ 在 {pred_dir} 找不到預測檔案")
            return
            
        logger.info(f"🔄 開始後處理 {len(pred_files)} 個檔案...")
        
        success_count = 0
        for pred_file in tqdm(pred_files, desc="Converting to NIfTI"):
            # 嘗試尋找對應的原始輸入檔
            # 假設檔名相同 (LNDb-0001_lesion01.npz)
            relative_name = pred_file.name
            
            # 在 input_dir 中遞迴尋找同名檔案
            input_matches = list(input_path.glob(f"**/{relative_name}"))
            
            if not input_matches:
                # 嘗試去掉 _pred 後綴
                clean_name = relative_name.replace('_pred', '').replace('.npy', '.npz')
                input_matches = list(input_path.glob(f"**/{clean_name}"))
            
            if not input_matches:
                logger.warning(f"⚠️ 找不到對應輸入檔: {pred_file.name}")
                continue
                
            input_file = input_matches[0]
            
            try:
                self.convert_to_nifti(pred_file, input_file)
                success_count += 1
            except Exception as e:
                logger.error(f"❌ 處理 {pred_file.name} 失敗: {e}")
        
        logger.info(f"✅ 完成! 成功轉換: {success_count}/{len(pred_files)}")
        logger.info(f"📁 輸出目錄: {self.output_dir}")

    def convert_to_nifti(
        self, 
        pred_path: Path, 
        input_path: Path, 
        output_name: str = None
    ):
        """
        單個檔案轉換流程
        """
        # 1. 載入預測
        pred_data = np.load(pred_path, allow_pickle=True)
        if isinstance(pred_data, np.lib.npyio.NpzFile):
            # 支援多種 key
            if 'pred' in pred_data:
                pred_mask = pred_data['pred']
            elif 'masks' in pred_data: # GT 作為測試
                pred_mask = pred_data['masks']
            else:
                 # 假設第一個 array 是 mask
                pred_mask = pred_data[list(pred_data.keys())[0]]
        else:
            pred_mask = pred_data  # .npy 直接是 array
            
        # 轉換為 uint8 (0, 1)
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        
        # 2. 載入原始資訊
        input_data = np.load(input_path, allow_pickle=True)
        
        # 必要的 metadata
        spacing = input_data['spacing']  # (z, y, x)
        origin = input_data['origin']
        slice_indices = input_data['slice_indices']
        original_shape = input_data['original_shape'] # (Z, Y, X)
        
        # 3. 重建完整體積 (Original Full Volume)
        full_mask = np.zeros(original_shape, dtype=np.uint8)
        
        D_pred, H_pred, W_pred = pred_mask.shape
        D_orig, H_orig, W_orig = original_shape[0], original_shape[1], original_shape[2]
        
        # 這裡需要注意：
        # preprocess.py 中我們做了 resize 到 512x512
        # 所以這裡需要 resize 回去原始切片大小
        
        for i, z_idx in enumerate(slice_indices):
            if i >= D_pred: break
            
            # 取出單張預測 mask
            mask_slice = pred_mask[i]
            
            # 使用 PIL resize 回原始大小 (W_orig, H_orig)
            # 注意 PIL resize 是 (width, height)
            if mask_slice.shape != (H_orig, W_orig):
                img_pil = Image.fromarray(mask_slice)
                img_resized = img_pil.resize((W_orig, H_orig), Image.NEAREST)
                mask_slice_orig = np.array(img_resized)
            else:
                mask_slice_orig = mask_slice
            
            # 填入完整體積
            if z_idx < D_orig:
                full_mask[z_idx] = mask_slice_orig
                
        # 4. 轉換為 SimpleITK Image
        sitk_image = sitk.GetImageFromArray(full_mask)
        
        # 設定幾何資訊 (注意 SimpleITK spacing 是 (x, y, z))
        sitk_image.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))
        sitk_image.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2])))
        
        # 5. 保存
        if output_name is None:
            output_name = pred_path.stem.replace('_pred', '') + '.nii.gz'
            
        output_path = self.output_dir / output_name
        sitk.WriteImage(sitk_image, str(output_path))


def main():
    parser = argparse.ArgumentParser(description='MedSAM2 預測後處理工具')
    parser.add_argument('--pred_dir', type=str, required=True, help='預測結果目錄')
    parser.add_argument('--input_dir', type=str, required=True, help='原始 NPZ 資料目錄')
    parser.add_argument('--output_dir', type=str, default='results/nifti', help='NIfTI 輸出目錄')
    
    args = parser.parse_args()
    
    postprocessor = VideoPostprocessor(args.output_dir)
    postprocessor.process_batch(args.pred_dir, args.input_dir)


if __name__ == '__main__':
    main()
