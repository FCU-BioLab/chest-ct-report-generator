#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統計報告生成器：
統計每位病患底下的 CT、非CT、RGB 彩色影像數

輸入資料結構：
  data_root/
    patient_001/
      dicom_files/*.dcm
    patient_002/
      dicom_files/*.dcm
    ...

輸出：
  ct_rgb_report.json
  ct_rgb_report.csv
"""

import os
import json
import csv
import pydicom
from pathlib import Path
from tqdm import tqdm
from typing import Dict

def classify_dicom(dicom_path: str) -> Dict[str, str]:
    """判斷 DICOM 類別（CT / 非CT / RGB）"""
    try:
        dcm = pydicom.dcmread(dicom_path)
        modality = getattr(dcm, "Modality", "Unknown").upper()
        pixel_array = dcm.pixel_array

        # RGB 彩色影像
        if pixel_array.ndim == 3 and pixel_array.shape[-1] == 3:
            return {"modality": modality, "type": "RGB"}

        # 非 CT
        if modality != "CT":
            return {"modality": modality, "type": "NonCT"}

        # 一般灰階 CT
        return {"modality": modality, "type": "CT"}

    except Exception as e:
        return {"modality": "Error", "type": "Error"}

def generate_report(data_root: str, output_dir: str = "./"):
    data_root = Path(data_root)
    patients = sorted([d for d in data_root.iterdir() if d.is_dir()])

    report = []
    print(f"📋 找到 {len(patients)} 位患者\n")

    for patient in tqdm(patients, desc="掃描病例"):
        dicom_dir = patient / "dicom_files"
        if not dicom_dir.exists():
            continue

        dicom_files = list(dicom_dir.glob("*.dcm"))
        ct_count, non_ct_count, rgb_count, error_count = 0, 0, 0, 0

        for dcm_path in dicom_files:
            result = classify_dicom(str(dcm_path))
            if result["type"] == "CT":
                ct_count += 1
            elif result["type"] == "NonCT":
                non_ct_count += 1
            elif result["type"] == "RGB":
                rgb_count += 1
            else:
                error_count += 1

        report.append({
            "patient_id": patient.name,
            "total_files": len(dicom_files),
            "CT_count": ct_count,
            "NonCT_count": non_ct_count,
            "RGB_count": rgb_count,
            "Error_count": error_count
        })

    # JSON 輸出
    json_path = Path(output_dir) / "ct_rgb_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # CSV 輸出
    csv_path = Path(output_dir) / "ct_rgb_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patient_id", "total_files", "CT_count", "NonCT_count", "RGB_count", "Error_count"
        ])
        writer.writeheader()
        writer.writerows(report)

    print("\n✅ 統計完成！")
    print(f"📁 JSON 報告: {json_path}")
    print(f"📁 CSV 報告:  {csv_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="生成 CT / 非CT / RGB 統計報告")
    parser.add_argument("--data_root", type=str, required=True, help="原始 DICOM 病患資料夾")
    parser.add_argument("--output_dir", type=str, default="./", help="報告輸出位置")
    args = parser.parse_args()

    generate_report(args.data_root, args.output_dir)


if __name__ == "__main__":
    main()