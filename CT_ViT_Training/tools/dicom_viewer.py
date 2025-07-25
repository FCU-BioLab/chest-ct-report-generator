#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DICOM檔案訪問和顯示工具
用於快速查看和分析DICOM影像

功能:
- 載入DICOM檔案
- 顯示基本資訊
- 視覺化影像內容

作者: GitHub Copilot  
日期: 2025-07-25
"""

import pydicom
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def load_and_display_dicom(dicom_path: str, show_info: bool = True, save_image: bool = False):
    """
    載入並顯示DICOM檔案
    
    Args:
        dicom_path: DICOM檔案路徑
        show_info: 是否顯示詳細資訊
        save_image: 是否保存影像
    """
    try:
        # 載入DICOM檔案
        dicom_data = pydicom.dcmread(dicom_path)
        
        # 獲取基本資訊
        if show_info:
            print("=== DICOM檔案資訊 ===")
            print(f"檔案路徑: {dicom_path}")
            print(f"患者ID: {getattr(dicom_data, 'PatientID', 'Unknown')}")
            print(f"檢查日期: {getattr(dicom_data, 'StudyDate', 'Unknown')}")
            print(f"影像類型: {getattr(dicom_data, 'Modality', 'Unknown')}")
            print(f"SOP Instance UID: {getattr(dicom_data, 'SOPInstanceUID', 'Unknown')}")
            
        # 獲取像素矩陣
        img_array = dicom_data.pixel_array
        print(f"影像尺寸: {img_array.shape}")
        print(f"像素值範圍: [{img_array.min()}, {img_array.max()}]")
        
        # 顯示影像
        plt.figure(figsize=(10, 8))
        plt.imshow(img_array, cmap=plt.cm.bone)
        plt.title(f"DICOM影像\nUID: {getattr(dicom_data, 'SOPInstanceUID', 'Unknown')}")
        plt.colorbar()
        plt.axis('off')
        
        if save_image:
            output_path = Path(dicom_path).parent / f"{Path(dicom_path).stem}_preview.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"影像已保存至: {output_path}")
        
        plt.show()
        
        return dicom_data, img_array
        
    except Exception as e:
        print(f"載入DICOM檔案失敗: {e}")
        return None, None

def batch_dicom_info(dicom_dir: str, limit: int = 10):
    """
    批量顯示DICOM檔案資訊
    
    Args:
        dicom_dir: DICOM檔案目錄
        limit: 處理檔案數量限制
    """
    dicom_path = Path(dicom_dir)
    dicom_files = list(dicom_path.glob("*.dcm"))[:limit]
    
    if not dicom_files:
        print(f"在 {dicom_dir} 中未找到DICOM檔案")
        return
    
    print(f"=== 批量DICOM資訊 (前{len(dicom_files)}個檔案) ===")
    
    for i, file_path in enumerate(dicom_files, 1):
        try:
            dicom_data = pydicom.dcmread(file_path)
            img_array = dicom_data.pixel_array
            
            print(f"\n{i}. 檔案: {file_path.name}")
            print(f"   患者ID: {getattr(dicom_data, 'PatientID', 'Unknown')}")
            print(f"   影像尺寸: {img_array.shape}")
            print(f"   像素範圍: [{img_array.min()}, {img_array.max()}]")
            
        except Exception as e:
            print(f"{i}. 檔案: {file_path.name} - 載入失敗: {e}")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="DICOM檔案訪問工具")
    parser.add_argument("--file", type=str, help="單個DICOM檔案路徑")
    parser.add_argument("--dir", type=str, help="DICOM檔案目錄")
    parser.add_argument("--batch", action="store_true", help="批量模式")
    parser.add_argument("--save", action="store_true", help="保存影像預覽")
    parser.add_argument("--limit", type=int, default=10, help="批量模式處理檔案數量限制")
    
    args = parser.parse_args()
    
    if args.file:
        # 單檔案模式
        load_and_display_dicom(args.file, show_info=True, save_image=args.save)
    
    elif args.dir:
        if args.batch:
            # 批量資訊模式
            batch_dicom_info(args.dir, args.limit)
        else:
            # 目錄中第一個檔案
            dicom_path = Path(args.dir)
            dicom_files = list(dicom_path.glob("*.dcm"))
            if dicom_files:
                load_and_display_dicom(str(dicom_files[0]), show_info=True, save_image=args.save)
            else:
                print(f"在 {args.dir} 中未找到DICOM檔案")
    else:
        # 預設範例
        example_path = "../../matched_data_by_patient/A0001/dicom_files"
        if Path(example_path).exists():
            dicom_files = list(Path(example_path).glob("*.dcm"))
            if dicom_files:
                print("使用範例DICOM檔案:")
                load_and_display_dicom(str(dicom_files[0]), show_info=True)
            else:
                print("範例目錄中未找到DICOM檔案")
        else:
            print("請指定 --file 或 --dir 參數")
            parser.print_help()

if __name__ == "__main__":
    main()
