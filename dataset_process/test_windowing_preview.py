#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
動態 DICOM → PNG 轉換 + 自動亮度補正（完整修正版）

功能：
 - 自動識別灰階 CT vs 彩色/其他模態
 - CT：固定窗 → 若過亮/過暗 → 自動窗 (百分位法) → 若仍異常 → CLAHE
 - 彩色/其他：normalize → 若過亮/過暗 → gamma/CLAHE
 - 影像標註亮度狀態，若補正過會標註 (Adjusted)
 - 產生 conversion_report.json（已修正 float32 報錯問題）
 - 自動檢查統計一致性
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pydicom
import cv2
from tqdm import tqdm
from typing import Tuple, Dict, Any, List

# =======================================================
# 🔧 JSON 安全轉換 (修正 float32 / ndarray 不能序列化)
# =======================================================
def to_serializable(obj):
    """遞迴轉換成可被 JSON 序列化的 Python 基本型別"""
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


# =======================================================
# 🧩 基礎工具
# =======================================================
EPS = 1e-6

def read_dicom(dicom_path: str):
    dcm = pydicom.dcmread(dicom_path)
    pixel_array = dcm.pixel_array
    modality = getattr(dcm, "Modality", "Unknown")
    intercept = getattr(dcm, "RescaleIntercept", 0.0)
    slope = getattr(dcm, "RescaleSlope", 1.0)
    slice_loc = getattr(dcm, "SliceLocation", 0.0)
    return pixel_array, modality, float(intercept), float(slope), slice_loc


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def normalize_any(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    vmin, vmax = np.min(img), np.max(img)
    if vmax - vmin < EPS:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - vmin) / (vmax - vmin + EPS) * 255.0
    return out.astype(np.uint8)


def ct_window_hu(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    lo = center - width / 2.0
    hi = center + width / 2.0
    img = np.clip(hu, lo, hi)
    img = (img - lo) / (hi - lo + EPS) * 255.0
    return img.astype(np.uint8)


def ct_auto_window_hu(hu: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0, min_width: float = 100.0) -> Tuple[np.ndarray, float, float]:
    lo = np.percentile(hu, low_pct)
    hi = np.percentile(hu, high_pct)
    if hi - lo < min_width:
        mid = (hi + lo) / 2.0
        lo, hi = mid - min_width / 2.0, mid + min_width / 2.0
    center = (hi + lo) / 2.0
    width = hi - lo
    return ct_window_hu(hu, center, width), center, width


def eval_brightness(img_u8: np.ndarray, bright_thresh: float, dark_thresh: float) -> Tuple[str, float]:
    mean_val = float(np.mean(img_u8))
    if mean_val > bright_thresh:
        return "Too Bright", mean_val
    if mean_val < dark_thresh:
        return "Too Dark", mean_val
    return "OK", mean_val


def ensure_bgr(img_u8: np.ndarray) -> np.ndarray:
    if img_u8.ndim == 2:
        return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    if img_u8.ndim == 3 and img_u8.shape[2] == 3:
        return img_u8
    if img_u8.ndim == 3 and img_u8.shape[2] == 4:
        return img_u8[..., :3]
    raise ValueError(f"Unsupported image shape: {img_u8.shape}")


def annotate(img_u8: np.ndarray, text: str, color=(0, 255, 0)) -> np.ndarray:
    out = ensure_bgr(img_u8)
    cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return out


# =======================================================
# 🧠 自動亮度補正策略
# =======================================================
def fix_brightness_ct(hu: np.ndarray, bright_thresh: float, dark_thresh: float, preset_center: float, preset_width: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """CT 自動補正"""
    img1 = ct_window_hu(hu, preset_center, preset_width)
    status1, mean1 = eval_brightness(img1, bright_thresh, dark_thresh)
    steps = [{"step": "preset_window", "status": status1, "mean": mean1}]

    if status1 == "OK":
        return img1, {"corrected": False, "steps": steps}

    img2, auto_c, auto_w = ct_auto_window_hu(hu)
    status2, mean2 = eval_brightness(img2, bright_thresh, dark_thresh)
    steps.append({"step": "auto_window", "status": status2, "mean": mean2, "auto_center": auto_c, "auto_width": auto_w})
    if status2 == "OK":
        return img2, {"corrected": True, "steps": steps}

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img3 = clahe.apply(img2)
    status3, mean3 = eval_brightness(img3, bright_thresh, dark_thresh)
    steps.append({"step": "clahe_after_auto", "status": status3, "mean": mean3})
    return img3, {"corrected": True, "steps": steps}


def fix_brightness_color(img_rgb_or_gray: np.ndarray, bright_thresh: float, dark_thresh: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """彩色/其他模態補正"""
    base = normalize_any(img_rgb_or_gray)
    status1, mean1 = eval_brightness(base, bright_thresh, dark_thresh)
    steps = [{"step": "normalize", "status": status1, "mean": mean1}]
    if status1 == "OK":
        return base, {"corrected": False, "steps": steps}

    gamma = 0.7 if status1 == "Too Bright" else 1.4
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    img2 = cv2.LUT(base, lut)
    status2, mean2 = eval_brightness(img2, bright_thresh, dark_thresh)
    steps.append({"step": f"gamma_{gamma:.2f}", "status": status2, "mean": mean2})
    if status2 == "OK":
        return img2, {"corrected": True, "steps": steps}

    if img2.ndim == 3:
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray = img2
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img3_gray = clahe.apply(gray)
    img3 = cv2.cvtColor(img3_gray, cv2.COLOR_GRAY2BGR)
    status3, mean3 = eval_brightness(img3_gray, bright_thresh, dark_thresh)
    steps.append({"step": "clahe", "status": status3, "mean": mean3})
    return img3, {"corrected": True, "steps": steps}


# =======================================================
# 🚀 主流程
# =======================================================
def convert_dynamic(patient_dir: str, output_dir: str, window_center: float, window_width: float,
                    bright_thresh: float, dark_thresh: float, save_original: bool = False):
    pdir = Path(patient_dir)
    odir = Path(output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    dcm_paths = sorted(pdir.glob("*.dcm"))
    if not dcm_paths:
        print("❌ 找不到任何 DICOM 檔案。")
        return

    stats = {
        "total_files": len(dcm_paths),
        "processed": 0,
        "skipped_unknown": 0,
        "by_type": {"CT_gray": 0, "Color_or_Other": 0},
        "brightness_before": {"OK": 0, "Too Bright": 0, "Too Dark": 0},
        "brightness_after": {"OK": 0, "Too Bright": 0, "Too Dark": 0},
        "corrected_count": 0
    }
    details: List[Dict[str, Any]] = []

    print(f"📁 病患資料夾: {pdir}")
    print(f"🧠 共 {len(dcm_paths)} 張影像，開始動態轉換 + 自動補正...\n")

    for dcm in tqdm(dcm_paths, desc="Processing"):
        try:
            px, modality, intercept, slope, slice_loc = read_dicom(str(dcm))
        except Exception as e:
            stats["skipped_unknown"] += 1
            details.append({"file": dcm.name, "status": "read_error", "error": str(e)})
            continue

        try:
            if px.ndim == 2:
                hu = px.astype(np.float32) * slope + intercept
                out_img, fix_info = fix_brightness_ct(hu, bright_thresh, dark_thresh, window_center, window_width)
                img_type = "CT_gray"
            elif px.ndim == 3 and px.shape[2] in (3, 4):
                out_img, fix_info = fix_brightness_color(px[..., :3], bright_thresh, dark_thresh)
                img_type = "Color_or_Other"
            else:
                stats["skipped_unknown"] += 1
                continue
        except Exception as e:
            stats["skipped_unknown"] += 1
            details.append({"file": dcm.name, "status": "process_error", "error": str(e)})
            continue

        first_step, last_step = fix_info["steps"][0], fix_info["steps"][-1]
        stats["by_type"][img_type] += 1
        stats["brightness_before"][first_step["status"]] += 1
        stats["brightness_after"][last_step["status"]] += 1
        if fix_info.get("corrected", False):
            stats["corrected_count"] += 1

        out_path = odir / f"{dcm.stem}.png"
        tag, color = last_step["status"], (0, 255, 0) if last_step["status"] == "OK" else (0, 0, 255)
        suffix = " (Adjusted)" if fix_info.get("corrected", False) else ""
        annotated = annotate(out_img, f"{tag}{suffix}", color=color)
        cv2.imwrite(str(out_path), to_uint8(annotated))

        if save_original:
            base_img = ct_window_hu(hu, window_center, window_width) if img_type == "CT_gray" else normalize_any(px)
            cv2.imwrite(str(odir / f"{dcm.stem}_orig.png"), to_uint8(ensure_bgr(base_img)))

        stats["processed"] += 1
        details.append({
            "file": dcm.name,
            "modality": modality,
            "type": img_type,
            "slice_location": float(slice_loc),
            "before": {"status": first_step["status"], "mean": float(first_step["mean"])},
            "after": {"status": last_step["status"], "mean": float(last_step["mean"])},
            "corrected": fix_info.get("corrected", False),
            "steps": fix_info["steps"]
        })

    assert stats["processed"] + stats["skipped_unknown"] == stats["total_files"], \
        "統計不一致：processed + skipped_unknown 應等於 total_files"

    report = {"summary": stats, "details": details}
    report_path = Path(output_dir) / "conversion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(report), f, indent=2, ensure_ascii=False)

    print("\n✅ 完成動態轉換 + 自動補正！")
    print(f"  ✔ 處理成功: {stats['processed']} / {stats['total_files']}")
    print(f"  ✔ 灰階 CT: {stats['by_type']['CT_gray']} ，彩色/其他: {stats['by_type']['Color_or_Other']}")
    print(f"  💡 亮度（前）OK/亮/暗: {stats['brightness_before']['OK']}/"
          f"{stats['brightness_before']['Too Bright']}/{stats['brightness_before']['Too Dark']}")
    print(f"  💡 亮度（後）OK/亮/暗: {stats['brightness_after']['OK']}/"
          f"{stats['brightness_after']['Too Bright']}/{stats['brightness_after']['Too Dark']}")
    print(f"  🔧 補正張數: {stats['corrected_count']}")
    print(f"  ⚠️ 跳過未知: {stats['skipped_unknown']}")
    print(f"  📂 輸出: {output_dir}")
    print(f"  📝 報告: {report_path}")


# =======================================================
# CLI
# =======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="動態 DICOM→PNG + 自動亮度補正 + 報告修正版")
    parser.add_argument("--patient_dir", type=str, required=True, help="病患 DICOM 資料夾")
    parser.add_argument("--output_dir", type=str, default="converted_dynamic_png_fixed", help="輸出資料夾")
    parser.add_argument("--window_center", type=float, default=-600.0)
    parser.add_argument("--window_width", type=float, default=1500.0)
    parser.add_argument("--bright_thresh", type=float, default=200.0)
    parser.add_argument("--dark_thresh", type=float, default=30.0)
    parser.add_argument("--save_original", action="store_true", default=False, help="同時輸出原始 _orig.png 以便對照")
    args = parser.parse_args()

    convert_dynamic(
        args.patient_dir,
        args.output_dir,
        args.window_center,
        args.window_width,
        args.bright_thresh,
        args.dark_thresh,
        save_original=args.save_original
    )
