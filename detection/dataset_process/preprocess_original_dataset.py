#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
醫學影像前處理：
  DICOM → PNG + YOLO (for 2D)
  DICOM → NIfTI (for 3D)

功能：
 - 僅處理 CT 且原始為灰階的影像，自動跳過 PET/NM/MR 及彩色影像
 - 跳過的模態不生成影像與標註
 - 動態亮度補正（自動窗位 / CLAHE）
 - 自動型態修正（防止 OpenCV CLAHE 錯誤）
 - PNG 輸出到 preprocessed_yolo_lesion/
 - NIfTI 輸出到 preprocessed_nii_lesion/
 - 不再複製 YOLO 標註到 NIfTI
 - 輸出 processing_report.json
"""

import os
import json
import numpy as np
import pydicom
import nibabel as nib
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
    from PIL import Image


# ==============================================================
# JSON 安全轉換
# ==============================================================
def to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    else:
        return obj


# ==============================================================
# 影像處理工具
# ==============================================================
def apply_hu_windowing(hu_array: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    windowed = np.clip(hu_array, min_val, max_val)
    normalized = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return normalized


def auto_window_ct(hu_array: np.ndarray, low_pct=2, high_pct=98) -> np.ndarray:
    """自動估計窗位窗寬"""
    lo, hi = np.percentile(hu_array, (low_pct, high_pct))
    lo, hi = float(lo), float(hi)
    windowed = np.clip(hu_array, lo, hi)
    return ((windowed - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)


def auto_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE 自動對比增強，自動修正型態"""
    # 轉灰階（若是彩色）
    if img.ndim == 3:
        if HAS_CV2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            from PIL import Image
            img = np.array(Image.fromarray(img).convert("L"))

    # 確保型態正確
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if HAS_CV2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    else:
        from PIL import ImageOps
        return np.array(ImageOps.autocontrast(Image.fromarray(img)))


def read_dicom_to_array(dicom_path: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """讀取 DICOM 並轉換為 HU"""
    try:
        dcm = pydicom.dcmread(dicom_path)
        pixel_array = dcm.pixel_array.astype(float)
        
        # 檢查是否為灰階影像 (2D)
        is_grayscale = (pixel_array.ndim == 2)
        
        intercept = getattr(dcm, "RescaleIntercept", 0)
        slope = getattr(dcm, "RescaleSlope", 1)
        hu_array = pixel_array * slope + intercept
        modality = getattr(dcm, "Modality", "Unknown")
        metadata = {
            "SOPInstanceUID": getattr(dcm, "SOPInstanceUID", "Unknown"),
            "SliceLocation": getattr(dcm, "SliceLocation", 0.0),
            "PixelSpacing": getattr(dcm, "PixelSpacing", [1.0, 1.0]),
            "SliceThickness": getattr(dcm, "SliceThickness", 1.0),
            "Modality": modality,
            "IsGrayscale": is_grayscale,
        }
        return hu_array, metadata
    except Exception as e:
        print(f"❌ DICOM 讀取失敗: {dicom_path}")
        print(f"   錯誤: {e}")
        return None, None


def parse_xml_annotation(xml_path: str) -> List[Dict]:
    """解析 XML 標註"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        annotations = []
        size_elem = root.find("size")
        width = int(size_elem.find("width").text)
        height = int(size_elem.find("height").text)

        for obj in root.findall("object"):
            name = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            annotations.append({
                "class": name,
                "bbox": [xmin, ymin, xmax, ymax],
                "image_width": width,
                "image_height": height
            })
        return annotations
    except Exception as e:
        print(f"❌ 解析 XML 失敗: {xml_path}")
        print(f"   錯誤: {e}")
        return []


def convert_bbox_to_yolo(bbox: List[int], img_width: int, img_height: int) -> List[float]:
    xmin, ymin, xmax, ymax = bbox
    x_center = (xmin + xmax) / 2.0 / img_width
    y_center = (ymin + ymax) / 2.0 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return [x_center, y_center, width, height]


def save_single_nifti(hu_array: np.ndarray, output_path: Path, voxel_spacing: List[float]):
    volume = np.expand_dims(hu_array, axis=-1)
    affine = np.diag([*voxel_spacing, 1])
    nib.save(nib.Nifti1Image(volume, affine), str(output_path))


# ==============================================================
# 單一病患處理 (跳過非 CT、非灰階並略過標註)
# ==============================================================
def process_patient(patient_id: str, data_root: Path, yolo_root: Path, nii_root: Path,
                    window_center=-600.0, window_width=1500.0) -> Dict:
    patient_dir = data_root / patient_id
    dicom_dir = patient_dir / "dicom_files"
    xml_dir = patient_dir / "xml_annotations"

    dicom_files = sorted(dicom_dir.glob("*.dcm"))
    if not dicom_files:
        return {"patient_id": patient_id, "status": "no_dicom"}

    yolo_img_dir = yolo_root / "images_png" / patient_id
    yolo_lbl_dir = yolo_root / "labels" / patient_id
    nii_img_dir = nii_root / "images_nii" / patient_id
    for d in [yolo_img_dir, yolo_lbl_dir, nii_img_dir]:
        d.mkdir(parents=True, exist_ok=True)

    processed_ct, skipped_non_ct, skipped_color = 0, 0, 0
    details = []

    for dicom_path in dicom_files:
        hu_array, meta = read_dicom_to_array(str(dicom_path))
        if hu_array is None:
            continue

        modality = meta.get("Modality", "").upper()
        is_grayscale = meta.get("IsGrayscale", False)
        
        # 檢查是否為 CT 且為灰階
        if modality != "CT":
            skipped_non_ct += 1
            details.append({
                "file": dicom_path.name,
                "modality": modality,
                "status": "skipped_non_ct"
            })
            continue  # 完全略過該影像及其標記
        
        if not is_grayscale:
            skipped_color += 1
            details.append({
                "file": dicom_path.name,
                "modality": modality,
                "image_shape": str(hu_array.shape),
                "status": "skipped_color_image"
            })
            continue  # 完全略過彩色影像

        # --------- 動態亮度補正 ----------
        img = apply_hu_windowing(hu_array, window_center, window_width)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        mean_val = np.mean(img)
        if mean_val > 200 or mean_val < 30:
            img = auto_window_ct(hu_array)
            if np.mean(img) > 200 or np.mean(img) < 30:
                img = auto_contrast(img)
        # ---------------------------------

        dicom_name = dicom_path.stem
        png_path = yolo_img_dir / f"{dicom_name}.png"
        label_path = yolo_lbl_dir / f"{dicom_name}.txt"

        if HAS_CV2:
            cv2.imwrite(str(png_path), img)
        else:
            from PIL import Image
            Image.fromarray(img).save(str(png_path))

        nii_slice_path = nii_img_dir / f"{dicom_name}.nii.gz"
        voxel_spacing = [*meta["PixelSpacing"], meta.get("SliceThickness", 1.0)]
        save_single_nifti(hu_array, nii_slice_path, voxel_spacing)

        # 只有 CT 才會讀取對應 XML 標註
        xml_path = xml_dir / f"{meta['SOPInstanceUID']}.xml"
        if xml_path.exists():
            anns = parse_xml_annotation(str(xml_path))
            with open(label_path, "w") as f:
                for ann in anns:
                    bbox = convert_bbox_to_yolo(ann["bbox"], ann["image_width"], ann["image_height"])
                    f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        else:
            label_path.touch()

        processed_ct += 1
        details.append({
            "file": dicom_path.name,
            "mean_brightness": float(np.mean(img)),
            "status": "ok"
        })

    return {
        "patient_id": patient_id,
        "processed_ct": processed_ct,
        "skipped_non_ct": skipped_non_ct,
        "skipped_color": skipped_color,
        "details": details
    }


# ==============================================================
# 主流程
# ==============================================================
def process_all(data_root: str, yolo_dir: str, nii_dir: str, window_center=-600.0, window_width=1500.0):
    data_root, yolo_root, nii_root = Path(data_root), Path(yolo_dir), Path(nii_dir)
    yolo_root.mkdir(parents=True, exist_ok=True)
    nii_root.mkdir(parents=True, exist_ok=True)

    patients = sorted([d for d in data_root.iterdir() if d.is_dir()])
    all_stats = []

    print(f"📋 找到 {len(patients)} 位患者\n")
    for patient in tqdm(patients, desc="處理患者"):
        stats = process_patient(patient.name, data_root, yolo_root, nii_root, window_center, window_width)
        all_stats.append(stats)

    total_ct = sum(s["processed_ct"] for s in all_stats)
    total_non_ct = sum(s["skipped_non_ct"] for s in all_stats)
    total_color = sum(s.get("skipped_color", 0) for s in all_stats)
    total_images = total_ct + total_non_ct + total_color

    report = {
        "summary": {
            "patients": len(patients),
            "total_images": total_images,
            "processed_ct_grayscale": total_ct,
            "skipped_non_ct": total_non_ct,
            "skipped_color": total_color
        },
        "details": all_stats
    }

    with open(yolo_root / "processing_report.json", "w", encoding="utf-8") as f:
        json.dump(to_serializable(report), f, indent=2, ensure_ascii=False)
    with open(nii_root / "processing_report.json", "w", encoding="utf-8") as f:
        json.dump(to_serializable(report), f, indent=2, ensure_ascii=False)

    print("\n✅ 完成！")
    print(f"📊 統計資訊：")
    print(f"   總影像數: {total_images}")
    print(f"   處理的 CT 灰階影像: {total_ct}")
    print(f"   跳過的非 CT 影像: {total_non_ct}")
    print(f"   跳過的彩色影像: {total_color}")
    print(f"📁 YOLO 輸出: {yolo_root}")
    print(f"📁 NIfTI 輸出: {nii_root}")


# ==============================================================
# 入口
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="DICOM → PNG + NIfTI 動態亮度補正版 (CT grayscale only, skip non-CT & color images)")
    parser.add_argument("--data_root", type=str, required=True, help="原始 DICOM 資料夾")
    parser.add_argument("--yolo_dir", type=str, default="../../datasets/preprocessed_yolo_lesion", help="YOLO 輸出資料夾")
    parser.add_argument("--nii_dir", type=str, default="../../datasets/preprocessed_nii_lesion", help="NIfTI 輸出資料夾")
    parser.add_argument("--window_center", type=float, default=-600.0)
    parser.add_argument("--window_width", type=float, default=1500.0)
    args = parser.parse_args()

    process_all(args.data_root, args.yolo_dir, args.nii_dir, args.window_center, args.window_width)


if __name__ == "__main__":
    main()
