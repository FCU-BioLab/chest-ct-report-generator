#!/usr/bin/env python3
"""
MSD Lung Tumours - Lung Mask 生成腳本
=====================================

使用 lungmask 套件為 MSD Lung Tumours 資料集生成肺部遮罩
- 輸入: NIfTI 格式的 CT 影像
- 輸出: NIfTI 格式的 lung mask

使用方式:
    python generate_msd_lungmasks.py
    python generate_msd_lungmasks.py --data_dir E:/path/to/MSD --output_dir E:/path/to/output
"""

import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np
import SimpleITK as sitk
from lungmask import LMInferer

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============= 預設路徑 =============
DEFAULT_MSD_DIR = Path(r"E:\lung_ct_lesion_dataset\MSD Lung Tumours\Task06_Lung")
DEFAULT_OUTPUT_DIR = Path(r"E:\lung_ct_lesion_dataset\MSD Lung Tumours\lung_masks")


def generate_lung_masks(
    data_dir: Path = DEFAULT_MSD_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_name: str = "LTRCLobes",
    force_regenerate: bool = False
):
    """
    為 MSD Lung Tumours 資料集生成肺部遮罩
    
    Args:
        data_dir: MSD 資料集目錄 (Task06_Lung)
        output_dir: 輸出目錄
        model_name: lungmask 模型名稱 ('LTRCLobes' 或 'R231')
        force_regenerate: 是否強制重新生成
    """
    images_dir = data_dir / "imagesTr"
    
    if not images_dir.exists():
        logger.error(f"找不到影像目錄: {images_dir}")
        return
    
    # 建立輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 找出所有 NIfTI 檔案（排除 macOS metadata 檔案）
    nii_files = [f for f in sorted(images_dir.glob("*.nii.gz")) 
                 if not f.name.startswith("._")]
    logger.info(f"找到 {len(nii_files)} 個 CT 影像")
    
    # 初始化 lungmask
    logger.info(f"載入 lungmask 模型: {model_name}")
    inferer = LMInferer(modelname=model_name)
    
    # 處理每個案例
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for nii_path in tqdm(nii_files, desc="生成 Lung Masks"):
        case_id = nii_path.stem.replace(".nii", "")  # e.g., "lung_001"
        output_path = output_dir / f"{case_id}_lung.nii.gz"
        
        # 檢查是否已存在
        if output_path.exists() and not force_regenerate:
            logger.debug(f"跳過已存在: {case_id}")
            skip_count += 1
            continue
        
        try:
            # 讀取 CT 影像
            ct_sitk = sitk.ReadImage(str(nii_path))
            ct_array = sitk.GetArrayFromImage(ct_sitk)  # (Z, Y, X)
            
            # 使用 lungmask 進行分割
            # LMInferer 的 apply 方法接受 SimpleITK Image
            lung_mask = inferer.apply(ct_sitk)  # (Z, Y, X), labels: 0=bg, 1=right, 2=left
            
            # 儲存為 NIfTI
            lung_sitk = sitk.GetImageFromArray(lung_mask.astype(np.uint8))
            lung_sitk.CopyInformation(ct_sitk)  # 複製 spacing, origin, direction
            sitk.WriteImage(lung_sitk, str(output_path))
            
            # 統計
            lung_voxels = np.sum(lung_mask > 0)
            total_voxels = lung_mask.size
            lung_ratio = lung_voxels / total_voxels * 100
            
            logger.debug(f"{case_id}: Lung ratio = {lung_ratio:.1f}%")
            success_count += 1
            
        except Exception as e:
            logger.error(f"處理 {case_id} 時發生錯誤: {e}")
            fail_count += 1
            continue
    
    logger.info("=" * 50)
    logger.info(f"完成! 成功: {success_count}, 跳過: {skip_count}, 失敗: {fail_count}")
    logger.info(f"Lung masks 儲存於: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='生成 MSD Lung Tumours 的 Lung Masks')
    parser.add_argument('--data_dir', type=str, default=str(DEFAULT_MSD_DIR),
                        help='MSD 資料集目錄 (Task06_Lung)')
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help='輸出目錄')
    parser.add_argument('--model', type=str, default='LTRCLobes',
                        choices=['LTRCLobes', 'R231', 'R231CovidWeb'],
                        help='Lungmask 模型')
    parser.add_argument('--force', action='store_true',
                        help='強制重新生成現有的 masks')
    
    args = parser.parse_args()
    
    generate_lung_masks(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        force_regenerate=args.force
    )


if __name__ == "__main__":
    main()
