#!/usr/bin/env python3
"""
RetinaNet 結果評估工具
=====================

計算偵測結果與 Ground Truth 之間的 IoU、Precision、Recall 與 F1 Score。
支援 LNDb 與 LUNA16 格式。
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set

import numpy as np
import pandas as pd

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    計算兩個 3D 邊框 [x1, y1, z1, x2, y2, z2] 的 IoU (Intersection over Union)。
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    z1 = max(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    y2 = min(box1[4], box2[4])
    z2 = min(box1[5], box2[5])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1) * max(0.0, z2 - z1)
    
    vol1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    vol2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])
    
    union = vol1 + vol2 - intersection
    return intersection / union if union > 0 else 0.0


def evaluate(
    predictions: List[Dict], 
    ground_truths: List[Dict], 
    iou_threshold: float = 0.1
):
    """
    評估預測結果。
    
    Args:
        predictions: 預測列表，包含 'box_voxel' 與 'score'
        ground_truths: 真值列表，包含 'box' (voxel) 與 'id'
        iou_threshold: 判定為 True Positive 的 IoU 門檻值
    """
    tp = 0
    fp = 0
    fn = 0
    
    matched_gt: Set[int] = set()
    
    # 依分數排序預測結果 (高分優先)
    predictions.sort(key=lambda x: x['score'], reverse=True)
    
    logger.info("\n--- 評估報告 ---\n")
    
    for pred in predictions:
        p_box = pred['box_voxel']
        p_id = pred.get('id', 'unknown')
        p_score = pred.get('score', 0.0)
        
        best_iou = 0.0
        best_gt_idx = -1
        
        # 尋找最佳匹配的 GT
        for i, gt in enumerate(ground_truths):
            gt_box = gt['box']
            iou = compute_iou(p_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold:
            if best_gt_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_gt_idx)
                logger.info(f"✅ 預測 {p_id} (分數 {p_score:.2f}) 匹配 GT {ground_truths[best_gt_idx]['id']} (IoU: {best_iou:.2f})")
            else:
                fp += 1 # 重複偵測同一 GT
                logger.info(f"⚠️ 預測 {p_id} (分數 {p_score:.2f}) 重複匹配 (IoU: {best_iou:.2f})")
        else:
            fp += 1
            logger.info(f"❌ 預測 {p_id} (分數 {p_score:.2f}) 為偽陽性 (最大 IoU: {best_iou:.2f})")
            
    fn = len(ground_truths) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\n" + "="*30)
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("="*30 + "\n")


def main():
    parser = argparse.ArgumentParser(description="RetinaNet 偵測結果評估")
    parser.add_argument("--report_path", type=str, required=True, help="推論產生的 report.json 路徑")
    parser.add_argument("--gt_path", type=str, help="Ground Truth CSV 路徑 (LNDb trainNodules_gt.csv 或 LUNA16 annotations.csv)")
    parser.add_argument("--dataset", type=str, choices=["lndb", "luna16"], default="lndb", help="資料集類型")
    parser.add_argument("--iou_threshold", type=float, default=0.1, help="IoU 門檻值 (預設 0.1)")
    
    args = parser.parse_args()
    
    report_path = Path(args.report_path)
    if not report_path.exists():
        logger.error(f"找不到報告檔案: {report_path}")
        return

    # 1. 載入預測報告
    with open(report_path, "r", encoding='utf-8') as f:
        report = json.load(f)
    
    preds = report.get("nodules", [])
    if not preds:
        logger.warning("報告中無偵測結果。")
        return

    # 若未提供 GT 路徑，嘗試自動推斷 (基於專案結構假設)
    if not args.gt_path:
        base_dir = Path(__file__).resolve().parent.parent.parent
        if args.dataset == "lndb":
            args.gt_path = str(base_dir / "cache/LNDb/trainset_csv/trainNodules_gt.csv")
        else:
            args.gt_path = str(base_dir / "cache/LUNA16/annotations.csv")
            
    gt_path = Path(args.gt_path)
    if not gt_path.exists():
        logger.error(f"找不到 GT 檔案: {gt_path}")
        # 如果無法載入 GT，只能列出預測結果
        logger.info(f"僅列出預測結果 (共 {len(preds)} 個):")
        for p in preds:
            print(p)
        return

    logger.info(f"載入 GT: {gt_path}")
    df_gt = pd.read_csv(gt_path)
    
    # 2. 篩選對應影像的 GT
    # 需從 report["input_path"] 解析出 ID (LNDbID 或 SeriesUID)
    input_path = Path(report["input_path"])
    fname = input_path.name
    
    gt_boxes = []
    
    if args.dataset == "lndb":
        # LNDb-0001.mhd -> LNDbID = 1
        try:
            lndb_id = int(fname.replace("LNDb-", "").replace(".mhd", "").replace(".nii.gz", "")) # 簡化解析
            # 若為其他格式，需調整
            if "LNDb-" in fname:
                 lndb_id = int(fname.split("LNDb-")[1].split(".")[0])
        except Exception:
            logger.warning(f"無法從檔名 {fname} 解析 LNDbID。")
            return

        gt_nodules = df_gt[df_gt["LNDbID"] == lndb_id]
        
        # LNDb GT 為 LPS 世界座標
        # report.json 中的 box_voxel 是體素座標
        # 我們需要將 GT 轉為體素，或者 Pred 轉為世界座標來比較。
        # report.json 有 "center_world_mm"。可以用世界座標比較 IoU (近似)。
        # 但 IoU 計算通常在 Voxel 空間較準確 (因為 Box 定義於 Voxel Grid)。
        # 若 report 中有 spacing 與 origin，我們可以轉換。
        
        spacing = report.get("spacing", [1.0, 1.0, 1.0])
        # origin = report.get("origin"...)? report.json 預設未儲存 origin，但 inference.py 有讀取。
        # 假設 inference.py 輸出的 center_world_mm 是準確的。
        # 我們可以比較 center_world_mm 與 GT center 的距離，若小於半徑和，則視為匹配?
        # 或我們假設輸入影像已經過 affine 轉換?
        
        # 簡化: 若無 affine 資訊，無法精確轉換 GT -> Voxel。
        # 這裡僅實作基於距離的匹配作為替代 (若 IoU 因座標系問題不可行)。
        # 但腳本目標是計算 IoU。
        
        # 假設: 使用者應提供能對應 voxel 的 GT，或是我們在此重新載入影像取得 Affine?
        # 為了保持 evaluate 獨立性，我們這裡假設 report 包含足夠資訊或 GT 已知。
        
        # 由於 inference.py 輸出 center_world_mm，我們使用半徑/直徑來重建世界座標下的 Box，並計算世界座標 IoU。
        
        for _, row in gt_nodules.iterrows():
            x, y, z = row['x'], row['y'], row['z'] # LPS
            vol = row['Volume']
            r = (3 * vol / (4 * np.pi)) ** (1/3) # mm
            
            # 建立世界座標 Box (以球體外接立方體近似)
            # LNDb x,y,z
            box = [x - r, y - r, z - r, x + r, y + r, z + r]
            gt_boxes.append({"box": box, "id": row['FindingID']})

    elif args.dataset == "luna16":
        # seriesuid.mhd
        uid = fname.split(".")[0]
        gt_nodules = df_gt[df_gt["seriesuid"] == uid]
        
        for _, row in gt_nodules.iterrows():
            x, y, z = row['coordX'], row['coordY'], row['coordZ']
            d = row['diameter_mm']
            r = d / 2.0
            
            box = [x - r, y - r, z - r, x + r, y + r, z + r]
            gt_boxes.append({"box": box, "id": "nodule"})

    logger.info(f"此影像共有 {len(gt_boxes)} 個 Ground Truth 結節")

    # 3. 轉換預測結果至世界座標 Box (如果尚未轉換)
    # report["nodules"] 有 "center_world_mm" 和 "approx_diameter_mm"
    preds_world = []
    
    for p in preds:
        if "center_world_mm" in p and "approx_diameter_mm" in p:
            c = p["center_world_mm"]
            d = p["approx_diameter_mm"]
            r = d / 2.0
            # [x-r, y-r, z-r, x+r, y+r, z+r]
            box_world = [c[0]-r, c[1]-r, c[2]-r, c[0]+r, c[1]+r, c[2]+r]
            
            preds_world.append({
                "box_voxel": box_world, # 借用欄位名，實為 world
                "score": p["score"],
                "id": p["id"]
            })
        else:
            logger.warning("報告中缺少世界座標資訊，無法與 GT 進行比較。")
            return

    # 執行評估 (在世界座標系)
    evaluate(preds_world, gt_boxes, iou_threshold=args.iou_threshold)


if __name__ == "__main__":
    main()
