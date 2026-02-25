#!/usr/bin/env python3
"""
RetinaNet 資料準備腳本
=======================

此腳本用於掃描 LNDb 或 LUNA16 資料夾，並生成 MONAI 相容的 JSON 資料列表。
支援自動劃分訓練集、驗證集與測試集。

使用方式:
    python -m detection.retinanet.prepare_data --dataset lndb --base_dir cache/LNDb
    python -m detection.retinanet.prepare_data --dataset luna16 --base_dir cache/LUNA16
"""

import argparse
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def process_lndb(base_dir: Path, csv_path: Path) -> List[Dict]:
    """
    處理 LNDb 資料集。
    
    Args:
        base_dir: LNDb 資料根目錄 (包含 data*/LNDb-XXXX.mhd)
        csv_path: 標註檔路徑 (trainNodules_gt.csv)
        
    Returns:
        datalist: MONAI 格式的資料列表
    """
    # 1. 尋找所有 .mhd 檔案
    mhd_files = sorted(list(base_dir.glob("data*/*.mhd")))
    logger.info(f"在 {base_dir} 中找到 {len(mhd_files)} 個 MHD 檔案")
    
    if not mhd_files:
        logger.warning(f"在 {base_dir} 未找到任何 .mhd 檔案。請檢查路徑。")
        return []

    # 建立 LNDbID 到檔案路徑的映射
    # 檔名格式: LNDb-XXXX.mhd -> ID: XXXX (int)
    id_to_path = {}
    for p in mhd_files:
        try:
            # LNDb-0001.mhd -> 1
            lndb_id = int(p.stem.replace("LNDb-", ""))
            # 使用絕對路徑或相對路徑視需求而定，這裡存絕對路徑方便讀取，或轉為相對路徑
            id_to_path[lndb_id] = str(p.resolve())
        except ValueError:
            logger.warning(f"跳過格式不符的檔名: {p.name}")

    # 2. 讀取標註檔
    if not csv_path.exists():
        logger.error(f"找不到 CSV 標註檔: {csv_path}")
        return []
        
    df = pd.read_csv(csv_path)
    logger.info(f"已載入標註檔: {len(df)} 筆資料")
    
    dataset = []
    valid_ids = sorted(list(id_to_path.keys()))
    
    for lndb_id in valid_ids:
        image_path = id_to_path[lndb_id]
        
        # 取得此 ID 的結節標註
        nodules = df[df["LNDbID"] == lndb_id]
        
        boxes = []
        labels = []
        
        for _, row in nodules.iterrows():
            # LNDb 原始座標為 LPS 世界座標 (x, y, z)
            # MONAI 使用 RAS 慣例，需否定 X 和 Y
            x_ras = -row['x']    # L → R
            y_ras = -row['y']    # P → A
            z_ras = row['z']     # S → S
            vol = row['Volume']
            # 半徑 r = (3 * V / (4 * pi))^(1/3)
            r = (3 * vol / (4 * np.pi)) ** (1/3)
            
            # 建立 Box [x_min, y_min, z_min, x_max, y_max, z_max] (RAS 座標)
            boxes.append([x_ras - r, y_ras - r, z_ras - r, x_ras + r, y_ras + r, z_ras + r])
            labels.append(1) # 單一類別: 結節 (Nodule)
            
        # 即使沒有結節，也加入資料集作為負樣本 (Negative Sample)
        if len(boxes) == 0:
            boxes = np.zeros((0, 6), dtype=np.float32).tolist()
            labels = np.zeros((0,), dtype=np.int64).tolist()
        
        dataset.append({
            "image": image_path,
            "box": boxes,
            "label": labels,
            "lndb_id": lndb_id,
            "dataset_type": "lndb"
        })
        
    return dataset


def process_luna16(base_dir: Path, csv_path: Path) -> List[Dict]:
    """
    處理 LUNA16 資料集。
    
    Args:
        base_dir: LUNA16 資料根目錄 (包含 subset0/*.mhd)
        csv_path: 標註檔路徑 (annotations.csv)
        
    Returns:
        datalist: MONAI 格式的資料列表
    """
    # 1. 尋找所有 subset 中的 .mhd 檔案
    # LUNA16 結構: subset0/*.mhd
    mhd_files = sorted(list(base_dir.glob("subset*/*.mhd")))
    logger.info(f"在 {base_dir} 中找到 {len(mhd_files)} 個 MHD 檔案")
    
    if not mhd_files:
        logger.warning(f"在 {base_dir} 未找到任何 .mhd 檔案。請檢查路徑。")
        return []

    # 建立 SeriesUID 到檔案路徑的映射
    uid_to_path = {}
    for p in mhd_files:
        uid = p.stem # 檔名即為 SeriesUID
        uid_to_path[uid] = str(p.resolve())
            
    # 2. 讀取標註
    if not csv_path.exists():
        logger.error(f"找不到 CSV 標註檔: {csv_path}")
        return []
        
    df = pd.read_csv(csv_path)
    logger.info(f"已載入標註檔: {len(df)} 筆資料")
    
    # 檢查必要欄位
    required_cols = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"CSV 缺少必要欄位! 預期包含: {required_cols}")
        return []

    dataset = []
    valid_uids = sorted(list(uid_to_path.keys()))
    
    for uid in valid_uids:
        image_path = uid_to_path[uid]
        
        # 取得此 UID 的結節標註
        nodules = df[df["seriesuid"] == uid]
        
        boxes = []
        labels = []
        
        for _, row in nodules.iterrows():
            # LUNA16 座標為世界座標 (LPS: Left=+X, Posterior=+Y, Superior=+Z)
            # MONAI 的 ITK reader 將 affine 轉換為 RAS (NIfTI) 慣例
            # 因此需要將 LPS 轉為 RAS: 否定 X 和 Y 座標
            x_ras = -row['coordX']   # L → R
            y_ras = -row['coordY']   # P → A
            z_ras = row['coordZ']    # S → S (不變)
            d = row['diameter_mm']
            r = d / 2.0
            
            # 建立 Box [x_min, y_min, z_min, x_max, y_max, z_max] (RAS 座標)
            boxes.append([x_ras - r, y_ras - r, z_ras - r, x_ras + r, y_ras + r, z_ras + r])
            labels.append(1) # 單一類別: 結節
            
        if len(boxes) == 0:
            boxes = np.zeros((0, 6), dtype=np.float32).tolist()
            labels = np.zeros((0,), dtype=np.int64).tolist()
        
        dataset.append({
            "image": image_path,
            "box": boxes,
            "label": labels,
            "seriesuid": uid,
            "dataset_type": "luna16"
        })
        
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Lung Nodule Detection 資料準備工具")
    parser.add_argument("--dataset", type=str, required=True, choices=["lndb", "luna16"], help="資料集類型 (lndb 或 luna16)")
    parser.add_argument("--base_dir", type=str, required=True, help="資料集根目錄路徑")
    parser.add_argument("--csv_path", type=str, help="標註 CSV 檔案路徑 (若未指定，將嘗試使用預設路徑)")
    parser.add_argument("--output", type=str, default="dataset.json", help="輸出 JSON 檔案路徑")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="訓練集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="驗證集比例")
    # 測試集比例 = 1.0 - train - val
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).resolve()
    
    # 設定預設 CSV 路徑
    if args.csv_path:
        csv_path = Path(args.csv_path).resolve()
    else:
        if args.dataset == "lndb":
            csv_path = base_dir / "trainset_csv/trainNodules_gt.csv"
        else: # luna16
            csv_path = base_dir / "annotations.csv"
            
    logger.info(f"正在處理 {args.dataset.upper()} 資料集...")
    logger.info(f"資料目錄: {base_dir}")
    logger.info(f"標註檔案: {csv_path}")
    
    if args.dataset == "lndb":
        data = process_lndb(base_dir, csv_path)
    else:
        data = process_luna16(base_dir, csv_path)
        
    logger.info(f"共生成 {len(data)} 筆資料項目")
    
    if not data:
        return

    # 資料分割
    # 固定種子以確保可重現性
    random.seed(args.seed)
    random.shuffle(data)
    
    n_total = len(data)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    
    train_ds = data[:n_train]
    val_ds = data[n_train:n_train + n_val]
    test_ds = data[n_train + n_val:]
    
    json_data = {
        "training": train_ds,
        "validation": val_ds,
        "testing": test_ds
    }
    
    output_path = Path(args.output).resolve()
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
        
    logger.info(f"資料列表已儲存至: {output_path}")
    logger.info(f"分佈統計: 訓練集 {len(train_ds)}, 驗證集 {len(val_ds)}, 測試集 {len(test_ds)}")

if __name__ == "__main__":
    main()
