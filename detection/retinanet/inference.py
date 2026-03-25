#!/usr/bin/env python3
"""
RetinaNet 推論管線 (Inference Pipeline)
=====================================

臨床情境使用的肺結節偵測推論腳本。
支援 DICOM 目錄或一般影像檔案 (如 NIfTI, MHD) 輸入。

流程:
1. 載入 CT 掃描
2. 預處理 (Windowing, 正規化, Padding, Resize)
3. 執行 RetinaNet 偵測 (滑動視窗)
4. 將結果座標轉換回原始影像空間
5. 輸出報告 (JSON) 與視覺化圖檔
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage

# 加入專案根目錄至 path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from detection.retinanet.config import RetinaNetConfig
from detection.retinanet.trainer import RetinaNetTrainer
from detection.common.location_estimator import LungLocationEstimator

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_ct_volume(path: Union[str, Path]) -> Tuple[sitk.Image, np.ndarray, np.ndarray, np.ndarray]:
    """
    從檔案或 DICOM 目錄載入 CT 體積。
    Returns: (sitk_image, array [D, H, W], spacing [x, y, z], origin [x, y, z])
    """
    path = Path(path)
    if path.is_dir():
        # 假設為 DICOM 目錄
        logger.info(f"📂 從目錄載入 DICOM: {path}")
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(path))
        if not series_ids:
            raise ValueError(f"在 {path} 中未發現 DICOM 系列")
        
        # 使用第一個系列
        dicom_names = reader.GetGDCMSeriesFileNames(str(path), series_ids[0])
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
    else:
        # 單一檔案 (NIfTI, MHD 等)
        logger.info(f"📂 載入檔案: {path}")
        image = sitk.ReadImage(str(path))

    array = sitk.GetArrayFromImage(image)  # (D, H, W)
    spacing = np.array(image.GetSpacing())  # (x, y, z)
    origin = np.array(image.GetOrigin())    # (x, y, z)

    return image, array, spacing, origin


def load_preprocessed_npz(path: Union[str, Path]) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, Dict]:
    """
    載入已預處理的 .npz 檔案 (訓練格式)。
    Returns: (input_tensor, spacing, origin, scale_info)
    """
    data = np.load(path)
    frames = data["frames"]  # (D, 256, 256) uint8 通常
    
    if frames.dtype == np.uint8:
        frames = frames.astype(np.float32) / 255.0
    
    tensor = torch.from_numpy(frames).unsqueeze(0).float() # (1, D, H, W)
    
    spacing = data["spacing"]
    origin = data["origin"]
    
    scale_info = {
        "scale": 1.0,
        "pad_top": 0,
        "pad_left": 0,
        "original_shape": frames.shape, 
        "is_preprocessed": True
    }
    
    return tensor, spacing, origin, scale_info


def preprocess_volume(
    array: np.ndarray, 
    spacing: np.ndarray,
    target_size: int = 256,
    window_center: float = -600,
    window_width: float = 1500,
) -> Tuple[torch.Tensor, Dict]:
    """
    RetinaNet 預處理流程:
    1. Windowing (窗寬窗位調整)
    2. 正規化至 [0, 1]
    3. Padding (填補至正方形)
    4. Resize (縮放至 target_size, 僅 XY 平面)
    
    Returns: 
        tensor: (1, D, H, W) float32 [0, 1]
        scale_info: 用於還原座標的縮放資訊
    """
    # 1. Windowing & Normalization
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    
    vol_norm = np.clip(array, img_min, img_max)
    vol_norm = (vol_norm - img_min) / (img_max - img_min)
    vol_norm = vol_norm.astype(np.float32) # [0.0, 1.0]

    # 2. Padding (填補至正方形)
    D, H, W = vol_norm.shape
    max_dim = max(H, W)
    pad_h = max_dim - H
    pad_w = max_dim - W
    pad_top = pad_h // 2
    pad_left = pad_w // 2

    if pad_h > 0 or pad_w > 0:
        padded = np.zeros((D, max_dim, max_dim), dtype=np.float32)
        padded[:, pad_top:pad_top+H, pad_left:pad_left+W] = vol_norm
        vol_norm = padded
    
    # 3. Resize (縮放 XY 至 target_size)
    scale = 1.0
    if max_dim != target_size:
        scale = target_size / max_dim
        # zoom factors: (1.0, scale, scale) -> (Z, Y, X)
        # 使用線性插值
        vol_resized = ndimage.zoom(vol_norm, (1.0, scale, scale), order=1, mode='constant', cval=0.0)
    else:
        vol_resized = vol_norm
        
    # 轉為 Tensor (C, D, H, W) -> (1, D, H, W)
    tensor = torch.from_numpy(vol_resized).unsqueeze(0).float()
    
    scale_info = {
        "original_shape": (D, H, W),
        "padded_shape": (D, max_dim, max_dim),
        "resized_shape": vol_resized.shape,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "scale": scale,
        "max_dim": max_dim
    }
    
    return tensor, scale_info


def rescale_boxes(
    boxes: np.ndarray, 
    scale_info: Dict
) -> np.ndarray:
    """
    將邊框從縮放後的座標空間還原至原始影像空間。
    """
    if len(boxes) == 0:
        return boxes
    
    boxes = boxes.copy()
    scale = scale_info["scale"]
    pad_top = scale_info["pad_top"]
    pad_left = scale_info["pad_left"]
    
    # 1. 反縮放 (Un-scale)
    # Z 軸未縮放，僅 XY
    boxes[:, [0, 1, 3, 4]] /= scale
    
    # 2. 移除填補 (Un-pad)
    boxes[:, [0, 3]] -= pad_left
    boxes[:, [1, 4]] -= pad_top
    
    # 3. 限制在原始尺寸內
    D, H, W = scale_info["original_shape"]
    boxes[:, [0, 3]] = np.clip(boxes[:, [0, 3]], 0, W)
    boxes[:, [1, 4]] = np.clip(boxes[:, [1, 4]], 0, H)
    boxes[:, [2, 5]] = np.clip(boxes[:, [2, 5]], 0, D)
    
    return boxes


def save_visualization(
    array: np.ndarray, 
    boxes: np.ndarray, 
    scores: np.ndarray, 
    output_dir: Path, 
    prefix: str
):
    """儲存帶有邊框標註的切片影像。"""
    output_dir = output_dir / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, box in enumerate(boxes):
        score = scores[i]
        x1, y1, z1, x2, y2, z2 = box.astype(int)
        
        # 取中心切片
        z_center = int((z1 + z2) / 2)
        z_center = max(0, min(array.shape[0] - 1, z_center))
        
        slice_img = array[z_center]
        
        # 簡易 Windowing 用於顯示 [-1000, 400]
        vmin, vmax = -1000, 400
        disp_img = np.clip(slice_img, vmin, vmax)
        disp_img = (disp_img - vmin) / (vmax - vmin)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(disp_img, cmap='gray')
        
        # 繪製邊框
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, 
            linewidth=2, edgecolor='red', facecolor='none'
        )
        plt.gca().add_patch(rect)
        plt.title(f"結節 {i+1}: 分數 {score:.3f} (z={z_center})")
        plt.axis('off')
        
        out_path = output_dir / f"{prefix}_nodule_{i+1}_score{score:.2f}.png"
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="肺結節偵測臨床推論工具")
    parser.add_argument("--input_path", type=str, required=True, help="DICOM 資料夾或影像檔案 (.nii.gz, .mhd) 路徑")
    parser.add_argument("--model_path", type=str, required=True, help="模型檢查點 (.pt, .pth) 路徑")
    parser.add_argument("--output_dir", type=str, default="results_inference", help="結果輸出目錄")
    parser.add_argument("--threshold", type=float, default=0.5, help="偵測分數門檻值")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window_center", type=float, default=-600, help="Window Center")
    parser.add_argument("--window_width", type=float, default=1500, help="Window Width")
    parser.add_argument("--image_size", type=int, default=256, help="模型輸入影像大小 (XY)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 初始化模型
    logger.info(f"🔧 初始化模型: {args.model_path}...")
    cfg = RetinaNetConfig(output_dir=str(output_dir), cache_dataset=False) # 使用預設設定
    trainer = RetinaNetTrainer(cfg, inference_only=True)
    
    try:
        # 嘗試載入 TorchScript
        trainer.detector.network = torch.jit.load(args.model_path, map_location=args.device)
    except Exception as e:
        logger.warning(f"TorchScript 載入失敗: {e}。嘗試載入 state_dict...")
        checkpoint = torch.load(args.model_path, map_location=args.device)
        if "model" in checkpoint:
            trainer.detector.network.load_state_dict(checkpoint["model"])
        else:
            trainer.detector.network.load_state_dict(checkpoint)
        
    trainer.detector.to(args.device)
    trainer.detector.eval()
    logger.info("✅ 模型載入完成。")
    
    # 2. 載入與預處理資料
    logger.info(f"📦 載入資料: {args.input_path}")
    
    input_path = Path(args.input_path)
    is_npz = input_path.suffix.lower() == ".npz"
    
    if is_npz:
        logger.info("ℹ️ 輸入為 .npz (假設為預處理資料)。跳過 Windowing/Resize。")
        input_tensor, spacing, origin, scale_info = load_preprocessed_npz(input_path)
        array = input_tensor[0].numpy() # 用於視覺化
    else:
        try:
            sitk_img, array, spacing, origin = load_ct_volume(args.input_path)
        except Exception as e:
            logger.error(f"❌ 載入失敗: {e}")
            return

        logger.info(f"   原始形狀: {array.shape} (D, H, W)")
        logger.info(f"   Spacing: {spacing}")
        
        t0 = time.time()
        input_tensor, scale_info = preprocess_volume(
            array, spacing, 
            target_size=args.image_size,
            window_center=args.window_center, 
            window_width=args.window_width
        )
        logger.info(f"   預處理後形狀: {input_tensor.shape}")
        logger.info(f"⏱️ 預處理耗時: {time.time() - t0:.2f}s")
    
    # 3. 推論
    logger.info("🚀 執行推論 (滑動視窗)...")
    t1 = time.time()
    input_tensor = input_tensor.to(args.device)
    
    with torch.no_grad():
        # use_inferer=True 啟用滑動視窗推論
        # 輸入需為 list of tensors 或 batched tensor
        outputs = trainer.detector([input_tensor], use_inferer=True)
        
    logger.info(f"⏱️ 推論耗時: {time.time() - t1:.2f}s")
    
    # 4. 處理結果
    results = outputs[0] # Batch size 1
    
    pred_boxes = results[trainer.detector.target_box_key].cpu().numpy()
    pred_scores = results[trainer.detector.pred_score_key].cpu().numpy()
    
    # 過濾門檻值
    mask = pred_scores >= args.threshold
    pred_boxes = pred_boxes[mask]
    pred_scores = pred_scores[mask]
    
    logger.info(f"🔍 找到 {len(pred_boxes)} 個結節 (分數 >= {args.threshold})")
    
    # 還原至原始座標
    orig_boxes = rescale_boxes(pred_boxes, scale_info)
    
    # 5. 儲存報告
    report = {
        "input_path": str(args.input_path),
        "model_path": str(args.model_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_shape": list(array.shape),
        "spacing": list(spacing),
        "nodules": []
    }
    location_estimator = LungLocationEstimator(total_slices=max(int(array.shape[0]), 1))
    
    for i in range(len(orig_boxes)):
        box = orig_boxes[i].tolist() # [x1, y1, z1, x2, y2, z2]
        
        # 計算世界座標中心
        cx = (box[0] + box[3]) / 2
        cy = (box[1] + box[4]) / 2
        cz = (box[2] + box[5]) / 2
        relative_x = cx / max(array.shape[2], 1)
        relative_y = cy / max(array.shape[1], 1)
        slice_ratio = cz / max(array.shape[0], 1)
        location = location_estimator.estimate_location(relative_x, relative_y, slice_ratio)
        
        # 簡易轉換: origin + index * spacing (ITK 慣例)
        pt_world = origin + np.array([cx, cy, cz]) * spacing
        
        # 估算直徑 (mm) - 取最大邊
        diameter = max(box[3]-box[0], box[4]-box[1], box[5]-box[2]) * spacing[0]
        
        report["nodules"].append({
            "id": i + 1,
            "score": float(pred_scores[i]),
            "box_voxel": [round(x, 1) for x in box],
            "center_world_mm": [round(x, 2) for x in pt_world],
            "approx_diameter_mm": round(diameter, 1),
            "anatomical_location": {
                "lobe": location["lobe"],
                "lobe_full": location["lobe_full"],
                "side": location["side"],
                "confidence": round(float(location["confidence"]), 3),
            }
        })
        
    # 儲存 JSON
    json_path = output_dir / "report.json"
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    logger.info(f"📝 報告已儲存至: {json_path}")
    
    # 儲存視覺化
    if len(orig_boxes) > 0:
        save_visualization(array, orig_boxes, pred_scores, output_dir, Path(args.input_path).stem)
        logger.info(f"🖼️ 視覺化圖檔已儲存至: {output_dir}/viz")


if __name__ == "__main__":
    main()
