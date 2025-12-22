#!/usr/bin/env python3
"""
LNDb 肺部分割遮罩生成器
======================

使用 lungmask 套件為 LNDb 資料集生成肺部分割遮罩。

使用方式:
    python lndb_lung_segmentation.py                    # 處理所有 CT
    python lndb_lung_segmentation.py --lndb_id 1        # 處理單一 CT
    python lndb_lung_segmentation.py --start 1 --end 10 # 處理範圍

輸出:
    肺部遮罩儲存至 E:\lung_ct_lesion_dataset\LNDb\lung_masks\LNDb-XXXX_lung.mhd
    - Label 1: 右肺
    - Label 2: 左肺
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import SimpleITK as sitk

try:
    from lungmask import LMInferer
except ImportError:
    print("錯誤: 請先安裝 lungmask 套件")
    print("執行: pip install lungmask")
    sys.exit(1)


# LNDb 資料集路徑
LNDB_BASE_PATH = Path(r'E:\lung_ct_lesion_dataset\LNDb')
LUNG_MASK_OUTPUT_DIR = LNDB_BASE_PATH / 'lung_masks'


def find_all_ct_scans():
    """尋找所有 CT 掃描檔案"""
    scans = {}
    for i in range(6):
        data_dir = LNDB_BASE_PATH / f'data{i}'
        if data_dir.exists():
            for mhd_file in sorted(data_dir.glob('LNDb-*.mhd')):
                lndb_id = int(mhd_file.stem.split('-')[1])
                scans[lndb_id] = mhd_file
    return scans


def generate_lung_mask(ct_path: Path, output_path: Path, inferer: LMInferer) -> bool:
    """
    生成單一 CT 的肺部遮罩
    
    Parameters:
    -----------
    ct_path : Path
        CT 掃描檔案路徑 (.mhd)
    output_path : Path
        輸出遮罩檔案路徑 (.mhd)
    inferer : LMInferer
        lungmask 推論器
    
    Returns:
    --------
    bool : 是否成功
    """
    try:
        # 讀取 CT 影像
        input_image = sitk.ReadImage(str(ct_path))
        
        # 執行肺部分割
        segmentation = inferer.apply(input_image)
        
        # 轉換為 SimpleITK 影像以保留空間資訊
        seg_image = sitk.GetImageFromArray(segmentation)
        seg_image.SetOrigin(input_image.GetOrigin())
        seg_image.SetSpacing(input_image.GetSpacing())
        seg_image.SetDirection(input_image.GetDirection())
        
        # 確保輸出目錄存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 儲存遮罩
        sitk.WriteImage(seg_image, str(output_path))
        
        return True
        
    except Exception as e:
        print(f"錯誤處理 {ct_path}: {e}")
        return False


def process_single(lndb_id: int, scans: dict, inferer: LMInferer, force: bool = False):
    """處理單一 CT 掃描"""
    if lndb_id not in scans:
        print(f"找不到 LNDb ID: {lndb_id}")
        return False
    
    ct_path = scans[lndb_id]
    output_path = LUNG_MASK_OUTPUT_DIR / f'LNDb-{lndb_id:04d}_lung.mhd'
    
    if output_path.exists() and not force:
        print(f"跳過 LNDb-{lndb_id:04d} (已存在)")
        return True
    
    print(f"處理 LNDb-{lndb_id:04d}...")
    success = generate_lung_mask(ct_path, output_path, inferer)
    
    if success:
        print(f"完成: {output_path}")
    return success


def process_batch(scans: dict, inferer: LMInferer, start_id: int = None, 
                  end_id: int = None, force: bool = False):
    """批次處理 CT 掃描"""
    ids_to_process = sorted(scans.keys())
    
    if start_id is not None:
        ids_to_process = [i for i in ids_to_process if i >= start_id]
    if end_id is not None:
        ids_to_process = [i for i in ids_to_process if i <= end_id]
    
    print(f"準備處理 {len(ids_to_process)} 個 CT 掃描")
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for lndb_id in tqdm(ids_to_process, desc="生成肺部遮罩"):
        output_path = LUNG_MASK_OUTPUT_DIR / f'LNDb-{lndb_id:04d}_lung.mhd'
        
        if output_path.exists() and not force:
            skip_count += 1
            continue
        
        ct_path = scans[lndb_id]
        if generate_lung_mask(ct_path, output_path, inferer):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n處理完成:")
    print(f"  成功: {success_count}")
    print(f"  跳過: {skip_count}")
    print(f"  失敗: {fail_count}")


def main():
    parser = argparse.ArgumentParser(
        description='LNDb 肺部分割遮罩生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    python lndb_lung_segmentation.py                    # 處理所有 CT
    python lndb_lung_segmentation.py --lndb_id 1        # 處理單一 CT
    python lndb_lung_segmentation.py --start 1 --end 10 # 處理範圍
    python lndb_lung_segmentation.py --force            # 強制重新生成
        """
    )
    parser.add_argument('--lndb_id', type=int, help='指定處理的 LNDb ID')
    parser.add_argument('--start', type=int, help='起始 LNDb ID')
    parser.add_argument('--end', type=int, help='結束 LNDb ID')
    parser.add_argument('--force', action='store_true', help='強制重新生成已存在的遮罩')
    parser.add_argument('--model', type=str, default='R231', 
                        choices=['R231', 'LTRCLobes', 'R231CovidWeb'],
                        help='使用的模型 (預設: R231)')
    
    args = parser.parse_args()
    
    # 檢查資料集路徑
    if not LNDB_BASE_PATH.exists():
        print(f"錯誤: 找不到 LNDb 資料集: {LNDB_BASE_PATH}")
        sys.exit(1)
    
    # 尋找所有 CT 掃描
    print(f"掃描 LNDb 資料集: {LNDB_BASE_PATH}")
    scans = find_all_ct_scans()
    print(f"找到 {len(scans)} 個 CT 掃描")
    
    if not scans:
        print("錯誤: 未找到任何 CT 掃描")
        sys.exit(1)
    
    # 初始化 lungmask 推論器
    print(f"載入 lungmask 模型: {args.model}")
    inferer = LMInferer(modelname=args.model)
    print("模型載入完成")
    
    # 處理
    if args.lndb_id is not None:
        # 處理單一 CT
        process_single(args.lndb_id, scans, inferer, args.force)
    else:
        # 批次處理
        process_batch(scans, inferer, args.start, args.end, args.force)


if __name__ == '__main__':
    main()
