#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接處理原始資料集：DICOM → PNG 轉換，保留原始資料結構和檔名
輸出到 preprocessed_yolo_lesion 資料夾

使用方法:
    python preprocess_original_dataset.py \\
        --data_root ../../datasets/all_patient_data \\
        --output_dir ../../datasets/preprocessed_yolo_lesion \\
        --enable_preprocessing False
"""

import os
import sys
import json
import numpy as np
import pydicom
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("警告: opencv-python 未安装，將使用 PIL")
    from PIL import Image

def apply_hu_windowing(hu_array: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """HU windowing - 可選的影像前處理"""
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    windowed = np.clip(hu_array, min_val, max_val)
    normalized = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return normalized

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """CLAHE enhancement - 可選的影像前處理"""
    if HAS_CV2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(img)
    else:
        return img

def read_dicom_to_array(dicom_path: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    讀取 DICOM 檔案並轉換為 numpy array
    
    Returns:
        (pixel_array, metadata): 像素陣列和元數據
    """
    try:
        dcm = pydicom.dcmread(dicom_path)
        
        # 獲取像素資料
        pixel_array = dcm.pixel_array.astype(float)
        
        # 應用 DICOM 的 rescale (轉換為 HU 值)
        intercept = getattr(dcm, 'RescaleIntercept', 0)
        slope = getattr(dcm, 'RescaleSlope', 1)
        hu_array = pixel_array * slope + intercept
        
        # 獲取元數據
        metadata = {
            'PatientID': getattr(dcm, 'PatientID', 'Unknown'),
            'SeriesInstanceUID': getattr(dcm, 'SeriesInstanceUID', 'Unknown'),
            'SOPInstanceUID': getattr(dcm, 'SOPInstanceUID', 'Unknown'),
            'Rows': int(dcm.Rows),
            'Columns': int(dcm.Columns),
            'PixelSpacing': getattr(dcm, 'PixelSpacing', [1.0, 1.0]),
            'SliceThickness': getattr(dcm, 'SliceThickness', 1.0),
            'RescaleIntercept': intercept,
            'RescaleSlope': slope,
        }
        
        return hu_array, metadata
        
    except Exception as e:
        print(f"❌ 讀取 DICOM 失敗: {dicom_path}")
        print(f"   錯誤: {e}")
        return None, None

def parse_xml_annotation(xml_path: str) -> List[Dict]:
    """
    解析 XML 標註檔案
    
    Returns:
        標註列表: [{'class': 'A', 'bbox': [xmin, ymin, xmax, ymax]}, ...]
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        annotations = []
        
        # 獲取圖像尺寸
        size_elem = root.find('size')
        if size_elem is not None:
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
        else:
            width = 512
            height = 512
        
        # 解析所有物件
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            annotations.append({
                'class': name,
                'bbox': [xmin, ymin, xmax, ymax],
                'image_width': width,
                'image_height': height
            })
        
        return annotations
        
    except Exception as e:
        print(f"❌ 解析 XML 失敗: {xml_path}")
        print(f"   錯誤: {e}")
        return []

def convert_bbox_to_yolo(bbox: List[int], img_width: int, img_height: int) -> List[float]:
    """
    將 [xmin, ymin, xmax, ymax] 轉換為 YOLO 格式 [x_center, y_center, width, height]
    所有值歸一化到 [0, 1]
    """
    xmin, ymin, xmax, ymax = bbox
    
    # 計算中心點和寬高
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    return [x_center, y_center, width, height]

def convert_dicom_to_png(
    hu_array: np.ndarray,
    output_path: str,
    enable_preprocessing: bool = False,
    window_center: float = -600.0,
    window_width: float = 1500.0,
    enable_clahe: bool = False,
    clahe_clip_limit: float = 2.0
) -> bool:
    """
    將 HU 陣列轉換為 PNG 圖像
    
    Args:
        hu_array: HU 值陣列
        output_path: 輸出路徑
        enable_preprocessing: 是否啟用影像前處理
        window_center: HU 窗位中心
        window_width: HU 窗寬
        enable_clahe: 是否啟用 CLAHE
        clahe_clip_limit: CLAHE clip limit
    
    Returns:
        是否成功
    """
    try:
        if enable_preprocessing:
            # 啟用前處理：HU windowing + CLAHE
            img = apply_hu_windowing(hu_array, window_center, window_width)
            
            if enable_clahe:
                img = apply_clahe(img, clahe_clip_limit)
        else:
            # 不啟用前處理：直接轉換為 0-255 灰階
            # 使用完整的 HU 範圍來顯示胸腔 CT
            hu_min = -1000  # 空氣
            hu_max = 400    # 骨頭
            img = np.clip(hu_array, hu_min, hu_max)
            img = ((img - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)
        
        # 保存圖像
        if HAS_CV2:
            cv2.imwrite(output_path, img)
        else:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img)
            pil_img.save(output_path)
        
        return True
        
    except Exception as e:
        print(f"❌ 轉換圖像失敗: {output_path}")
        print(f"   錯誤: {e}")
        return False

def find_matching_xml(dcm_uid: str, xml_dir: Path) -> Optional[Path]:
    """
    根據 DICOM UID 尋找對應的 XML 標註檔案
    
    Args:
        dcm_uid: DICOM SOPInstanceUID
        xml_dir: XML 標註目錄
    
    Returns:
        XML 檔案路徑或 None
    """
    # XML 檔名格式: {SOPInstanceUID}.xml
    xml_path = xml_dir / f"{dcm_uid}.xml"
    
    if xml_path.exists():
        return xml_path
    
    return None

def process_patient(
    patient_id: str,
    data_root: Path,
    output_root: Path,
    enable_preprocessing: bool = False,
    window_center: float = -600.0,
    window_width: float = 1500.0,
    enable_clahe: bool = False,
    clahe_clip_limit: float = 2.0
) -> Dict:
    """
    處理單個患者的所有資料
    
    Returns:
        處理統計資訊
    """
    patient_dir = data_root / patient_id
    
    if not patient_dir.exists():
        return {'error': f'Patient directory not found: {patient_dir}'}
    
    # 檢查必要的子目錄
    dicom_dir = patient_dir / "dicom_files"
    xml_dir = patient_dir / "xml_annotations"
    
    if not dicom_dir.exists():
        return {'error': f'DICOM directory not found: {dicom_dir}'}
    
    # 創建輸出目錄結構
    output_patient_dir = output_root / patient_id
    output_images_dir = output_patient_dir / "images"
    output_labels_dir = output_patient_dir / "labels"
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 統計資訊
    stats = {
        'patient_id': patient_id,
        'total_dicom': 0,
        'processed_dicom': 0,
        'with_annotation': 0,
        'without_annotation': 0,
        'failed': 0,
        'failed_files': []
    }
    
    # 處理所有 DICOM 檔案
    dicom_files = sorted(dicom_dir.glob("*.dcm"))
    stats['total_dicom'] = len(dicom_files)
    
    for dicom_path in dicom_files:
        try:
            # 讀取 DICOM
            hu_array, metadata = read_dicom_to_array(str(dicom_path))
            
            if hu_array is None:
                stats['failed'] += 1
                stats['failed_files'].append(str(dicom_path))
                continue
            
            # 生成輸出檔名（保留原始檔名）
            dicom_filename = dicom_path.stem  # 不含副檔名
            png_filename = f"{dicom_filename}.png"
            txt_filename = f"{dicom_filename}.txt"
            
            output_png_path = output_images_dir / png_filename
            output_txt_path = output_labels_dir / txt_filename
            
            # 轉換並保存圖像
            success = convert_dicom_to_png(
                hu_array=hu_array,
                output_path=str(output_png_path),
                enable_preprocessing=enable_preprocessing,
                window_center=window_center,
                window_width=window_width,
                enable_clahe=enable_clahe,
                clahe_clip_limit=clahe_clip_limit
            )
            
            if not success:
                stats['failed'] += 1
                stats['failed_files'].append(str(dicom_path))
                continue
            
            # 尋找對應的 XML 標註
            dcm_uid = metadata.get('SOPInstanceUID', '')
            xml_path = find_matching_xml(dcm_uid, xml_dir) if xml_dir.exists() else None
            
            if xml_path and xml_path.exists():
                # 解析 XML 標註
                annotations = parse_xml_annotation(str(xml_path))
                
                if annotations:
                    # 轉換為 YOLO 格式並保存
                    with open(output_txt_path, 'w') as f:
                        for ann in annotations:
                            # 類別映射: A -> 0
                            class_id = 0  # 假設所有病灶都是同一類
                            
                            # 轉換 bbox 格式
                            yolo_bbox = convert_bbox_to_yolo(
                                ann['bbox'],
                                ann['image_width'],
                                ann['image_height']
                            )
                            
                            # 寫入 YOLO 格式: class x_center y_center width height
                            f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
                    
                    stats['with_annotation'] += 1
                else:
                    # XML 存在但無標註，創建空文件
                    output_txt_path.touch()
                    stats['without_annotation'] += 1
            else:
                # 沒有 XML 標註，創建空文件（負樣本）
                output_txt_path.touch()
                stats['without_annotation'] += 1
            
            stats['processed_dicom'] += 1
            
        except Exception as e:
            print(f"❌ 處理 DICOM 失敗: {dicom_path}")
            print(f"   錯誤: {e}")
            stats['failed'] += 1
            stats['failed_files'].append(str(dicom_path))
            continue
    
    return stats

def process_all_patients(
    data_root: str,
    output_dir: str,
    enable_preprocessing: bool = False,
    window_center: float = -600.0,
    window_width: float = 1500.0,
    enable_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    patient_limit: Optional[int] = None
):
    """
    處理所有患者資料
    
    Args:
        data_root: 原始資料目錄 (all_patient_data)
        output_dir: 輸出目錄 (preprocessed_yolo_lesion)
        enable_preprocessing: 是否啟用影像前處理
        window_center: HU 窗位中心
        window_width: HU 窗寬
        enable_clahe: 是否啟用 CLAHE
        clahe_clip_limit: CLAHE clip limit
        patient_limit: 限制處理患者數量（用於測試）
    """
    data_root_path = Path(data_root)
    output_root_path = Path(output_dir)
    output_root_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🚀 醫學影像資料集預處理 (DICOM → PNG + YOLO 標註)")
    print("=" * 80)
    print(f"輸入目錄:   {data_root}")
    print(f"輸出目錄:   {output_dir}")
    print(f"影像前處理: {'✅ 啟用' if enable_preprocessing else '❌ 關閉（保留原始 HU 值顯示）'}")
    
    if enable_preprocessing:
        print(f"  - HU 窗位:  center={window_center}, width={window_width}")
        print(f"  - CLAHE:    {'啟用' if enable_clahe else '停用'} (clip={clahe_clip_limit})")
    else:
        print(f"  - 顯示範圍: HU -1000 (空氣) ~ 400 (骨頭)")
    
    print("=" * 80)
    
    # 尋找所有患者目錄
    patient_dirs = sorted([d for d in data_root_path.iterdir() if d.is_dir() and d.name.startswith('A')])
    
    if patient_limit:
        patient_dirs = patient_dirs[:patient_limit]
        print(f"⚠️  測試模式: 僅處理前 {patient_limit} 位患者\n")
    
    print(f"📋 找到 {len(patient_dirs)} 位患者\n")
    
    # 處理每位患者
    all_stats = []
    
    for patient_dir in tqdm(patient_dirs, desc="處理患者"):
        patient_id = patient_dir.name
        
        stats = process_patient(
            patient_id=patient_id,
            data_root=data_root_path,
            output_root=output_root_path,
            enable_preprocessing=enable_preprocessing,
            window_center=window_center,
            window_width=window_width,
            enable_clahe=enable_clahe,
            clahe_clip_limit=clahe_clip_limit
        )
        
        all_stats.append(stats)
    
    # 彙總統計
    total_stats = {
        'total_patients': len(patient_dirs),
        'total_dicom': sum(s.get('total_dicom', 0) for s in all_stats),
        'processed_dicom': sum(s.get('processed_dicom', 0) for s in all_stats),
        'with_annotation': sum(s.get('with_annotation', 0) for s in all_stats),
        'without_annotation': sum(s.get('without_annotation', 0) for s in all_stats),
        'failed': sum(s.get('failed', 0) for s in all_stats),
    }
    
    # 保存處理報告
    report = {
        'summary': total_stats,
        'preprocessing_settings': {
            'enable_preprocessing': enable_preprocessing,
            'window_center': window_center,
            'window_width': window_width,
            'enable_clahe': enable_clahe,
            'clahe_clip_limit': clahe_clip_limit,
        },
        'patient_details': all_stats
    }
    
    report_path = output_root_path / "processing_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印最終結果
    print("\n" + "=" * 80)
    print("✅ 資料集預處理完成！")
    print("=" * 80)
    print(f"總患者數:     {total_stats['total_patients']}")
    print(f"總 DICOM 數:  {total_stats['total_dicom']}")
    print(f"成功處理:     {total_stats['processed_dicom']}")
    print(f"  - 有標註:   {total_stats['with_annotation']}")
    print(f"  - 無標註:   {total_stats['without_annotation']}")
    print(f"處理失敗:     {total_stats['failed']}")
    print(f"\n處理報告:     {report_path}")
    print("=" * 80)
    
    # 顯示資料結構
    print("\n📁 輸出目錄結構:")
    print(f"{output_dir}/")
    print(f"├── A0001/")
    print(f"│   ├── images/")
    print(f"│   │   ├── A0001_DCM_001_1-01.png")
    print(f"│   │   ├── A0001_DCM_002_1-02.png")
    print(f"│   │   └── ...")
    print(f"│   └── labels/")
    print(f"│       ├── A0001_DCM_001_1-01.txt  (YOLO format)")
    print(f"│       ├── A0001_DCM_002_1-02.txt")
    print(f"│       └── ...")
    print(f"├── A0002/")
    print(f"├── ...")
    print(f"└── processing_report.json")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description="將醫學影像資料集轉換為 YOLO 格式 (保留原始檔名和資料結構)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需參數
    parser.add_argument("--data_root", type=str, default="../../datasets/all_patient_data",
                       help="原始資料目錄 (all_patient_data)")
    parser.add_argument("--output_dir", type=str, default="../../datasets/preprocessed_yolo_lesion",
                       help="輸出目錄")
    
    # 影像前處理參數
    parser.add_argument("--enable_preprocessing", type=lambda x: x.lower() == 'true', default=False,
                       help="是否啟用影像前處理 (True/False)")
    parser.add_argument("--window_center", type=float, default=-600.0,
                       help="HU 窗位中心 (僅當 enable_preprocessing=True 時有效)")
    parser.add_argument("--window_width", type=float, default=1500.0,
                       help="HU 窗寬 (僅當 enable_preprocessing=True 時有效)")
    parser.add_argument("--enable_clahe", type=lambda x: x.lower() == 'true', default=False,
                       help="啟用 CLAHE 對比度增強 (True/False)")
    parser.add_argument("--clahe_clip", type=float, default=2.0,
                       help="CLAHE clip limit")
    
    # 測試參數
    parser.add_argument("--patient_limit", type=int, default=None,
                       help="限制處理患者數量（用於測試）")
    
    args = parser.parse_args()
    
    # 處理資料集
    process_all_patients(
        data_root=args.data_root,
        output_dir=args.output_dir,
        enable_preprocessing=args.enable_preprocessing,
        window_center=args.window_center,
        window_width=args.window_width,
        enable_clahe=args.enable_clahe,
        clahe_clip_limit=args.clahe_clip,
        patient_limit=args.patient_limit
    )

if __name__ == "__main__":
    main()
