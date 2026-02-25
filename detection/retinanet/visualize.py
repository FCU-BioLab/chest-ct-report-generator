#!/usr/bin/env python3
"""
RetinaNet 結果視覺化工具
=======================

讀取 report.json 與原始影像，生成包含偵測框的 GIF 動畫或切片圖。
"""

import argparse
import json
import logging
from pathlib import Path

import imageio
import matplotlib
matplotlib.use('Agg') # 非互動式後端
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """載入影像並轉為 numpy array (D, H, W)。"""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return arr


def main():
    parser = argparse.ArgumentParser(description="RetinaNet 偵測結果視覺化 (GIF)")
    parser.add_argument("--report_path", type=str, required=True, help="report.json 路徑")
    parser.add_argument("--output_dir", type=str, default=None, help="輸出目錄 (預設為 report 所在目錄/viz_gif)")
    parser.add_argument("--z_margin", type=int, default=3, help="結節上下顯示的切片數 (Margin)")
    
    args = parser.parse_args()
    
    report_path = Path(args.report_path)
    if not report_path.exists():
        logger.error(f"找不到報告: {report_path}")
        return
        
    with open(report_path, "r", encoding='utf-8') as f:
        report = json.load(f)
        
    input_path = report.get("input_path")
    # 若 input_path 為相對路徑，嘗試相對於 report 的位置解析
    if not Path(input_path).exists():
         # 嘗試相對於專案根目錄
         base_dir = report_path.parent.parent.parent # 假設 report 在 detection/results/xxx/report.json
         alt_path = base_dir / input_path
         if alt_path.exists():
             input_path = str(alt_path)
         else:
             logger.warning(f"找不到原始影像: {input_path}")
             # 嘗試在 report 同目錄找
             alt_path_2 = report_path.parent / Path(input_path).name
             if alt_path_2.exists():
                 input_path = str(alt_path_2)
    
    logger.info(f"載入影像: {input_path}")
    try:
        vol = load_image(str(input_path)) # (D, H, W)
    except Exception as e:
        logger.error(f"載入影像失敗: {e}")
        return

    nodules = report.get("nodules", [])
    logger.info(f"共有 {len(nodules)} 個結節需視覺化")
    
    # 設定輸出目錄
    if args.output_dir:
        viz_dir = Path(args.output_dir)
    else:
        viz_dir = report_path.parent / "viz_gif"
        
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 準備繪圖
    fig, ax = plt.subplots(figsize=(6, 6))
    
    frames = []
    
    # 為每個結節生成動畫片段，或生成整體的掃描 (若結節多，整體掃描較好；若針對性檢查，個別片段較好)
    # 這裡實作：將所有結節的 Z 範圍聯集起來進行掃描，或是簡單地針對每個結節生成一段，然後串接。
    # 為了清晰，我們針對每個結節生成一段，並在角落標示。
    
    for i, nodule in enumerate(nodules):
        nid = nodule.get("id", i+1)
        score = nodule.get("score", 0.0)
        box = nodule["box_voxel"] # [x1, y1, z1, x2, y2, z2]
        
        z_min, z_max = int(box[2]), int(box[5])
        z_center = (z_min + z_max) // 2
        
        # 定義顯示範圍
        z_start = max(0, z_min - args.z_margin)
        z_end = min(vol.shape[0], z_max + args.z_margin)
        
        logger.info(f"處理結節 {nid} (分數 {score:.2f}) Z範圍: {z_start}-{z_end}")
        
        for z in range(z_start, z_end):
            ax.clear()
            
            # 取得切片並 Windowing
            img_slice = vol[z, :, :] # (H, W) 
            # 注意: 視 SimpleITK 讀取方向而定。通常是 (D, H, W)。
            # 若顯示時發現旋轉，需 transpose。
            # 這裡假設標準方向。
            
            img_slice = np.clip(img_slice, -1000, 400)
            ax.imshow(img_slice, cmap="gray", origin="upper") # CT通常 origin upper
            
            # 繪製邊框 (若此切片在邊框範圍內)
            if z >= box[2] and z <= box[5]:
                x1, y1 = box[0], box[1]
                w = box[3] - box[0]
                h = box[4] - box[1]
                
                rect = plt.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-5, f"ID {nid}", color='red', fontsize=10, fontweight='bold')
            
            ax.set_title(f"Slice {z} - Nodule {nid} (Score {score:.2f})")
            ax.axis('off')
            
            # 截取圖形
            fig.canvas.draw()
            
            try:
                buf = fig.canvas.buffer_rgba()
                image = np.asarray(buf).copy()
            except AttributeError:
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,)).copy()
            
            if image.shape[2] == 4:
                image = image[:, :, :3]
                
            frames.append(image)
            
        # 加入全黑幀作為間隔
        if i < len(nodules) - 1:
            black_frame = np.zeros_like(frames[-1])
            for _ in range(5): # 5幀黑畫面
                frames.append(black_frame)
    
    if frames:
        gif_path = viz_dir / "detections_summary.gif"
        logger.info(f"儲存 GIF 至 {gif_path}...")
        imageio.mimsave(gif_path, frames, fps=5, loop=0)
        logger.info("完成!")
    else:
        logger.warning("沒有畫面可生成。")


if __name__ == "__main__":
    main()
