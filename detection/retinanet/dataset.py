#!/usr/bin/env python3
"""
RetinaNet 偵測資料集 (Dataset)
=============================

使用 MONAI Transform Pipeline 處理影像載入、座標轉換與資料增強。
與 bundles/lung_nodule_ct_detection/configs/train_luna16.json 對齊。
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import monai
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    Lambdad,
)
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    RandCropBoxByPosNegLabeld,
    RandZoomBoxd,
    ClipBoxToImaged,
    RandFlipBoxd,
    RandRotateBox90d,
)
from scipy import ndimage

logger = logging.getLogger(__name__)


def _reshape_empty_box(x):
    """確保空 box tensor 的 shape 為 (0, 6)。"""
    if x.shape[0] == 0:
        return torch.reshape(x, [0, 6])
    return x


def _remap_labels_to_zero_index(x):
    """將 label 從 1-based (dataset JSON) 轉為 0-based (MONAI RetinaNet 期望)。"""
    if x.numel() > 0:
        return x - 1
    return x


# ─── Transform Builders ─────────────────────────────────────────────

def build_det_transform(
    spacing: List[float] = None,
    hu_min: float = -1024.0,
    hu_max: float = 300.0,
) -> Compose:
    """
    Deterministic transforms（可被 CacheDataset 快取）。
    LoadImage → Orient → Spacing → ScaleIntensity → AffineBoxToImageCoordinated
    """
    if spacing is None:
        spacing = [0.703125, 0.703125, 1.25]

    return Compose([
        LoadImaged(keys="image", reader="itkreader"),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "box"]),
        EnsureTyped(keys="label", dtype=torch.long),
        Lambdad(keys="label", func=_remap_labels_to_zero_index),
        Lambdad(keys="box", func=_reshape_empty_box),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=spacing, mode="bilinear"),
        ScaleIntensityRanged(
            keys="image",
            a_min=hu_min, a_max=hu_max,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        AffineBoxToImageCoordinated(
            box_keys="box",
            box_ref_image_keys="image",
        ),
        EnsureTyped(keys=["image", "box"]),
        EnsureTyped(keys="label", dtype=torch.long),
    ])


def build_rand_transform(
    patch_size: List[int],
    batch_size: int = 2,
) -> Compose:
    """
    Random augmentation transforms（每次重新執行，不快取）。
    RandCrop → RandZoom → Clip → RandFlip → RandRotate → Intensity augmentation
    """
    return Compose([
        # 隨機裁切（正負樣本比例 1:1）
        RandCropBoxByPosNegLabeld(
            image_keys="image",
            box_keys="box",
            label_keys="label",
            spatial_size=patch_size,
            whole_box=True,
            num_samples=batch_size,
            pos=1,
            neg=1,
        ),
        # 隨機縮放
        RandZoomBoxd(
            image_keys="image",
            box_keys="box",
            box_ref_image_keys="image",
            prob=0.2,
            min_zoom=0.7,
            max_zoom=1.4,
            padding_mode="constant",
            keep_size=True,
        ),
        # 裁切/縮放後 clip 到影像邊界
        ClipBoxToImaged(
            box_keys="box",
            label_keys="label",
            box_ref_image_keys="image",
            remove_empty=True,
        ),
        # 隨機翻轉 (3 軸)
        RandFlipBoxd(
            image_keys="image", box_keys="box", box_ref_image_keys="image",
            prob=0.5, spatial_axis=0,
        ),
        RandFlipBoxd(
            image_keys="image", box_keys="box", box_ref_image_keys="image",
            prob=0.5, spatial_axis=1,
        ),
        RandFlipBoxd(
            image_keys="image", box_keys="box", box_ref_image_keys="image",
            prob=0.5, spatial_axis=2,
        ),
        # 隨機旋轉 90°
        RandRotateBox90d(
            image_keys="image", box_keys="box", box_ref_image_keys="image",
            prob=0.75, max_k=3, spatial_axes=[0, 1],
        ),
        # 影像強度增強
        RandGaussianNoised(keys="image", prob=0.1, mean=0.0, std=0.1),
        RandGaussianSmoothd(
            keys="image", prob=0.1,
            sigma_x=[0.5, 1.0], sigma_y=[0.5, 1.0], sigma_z=[0.5, 1.0],
        ),
        RandScaleIntensityd(keys="image", factors=0.25, prob=0.15),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
        RandAdjustContrastd(keys="image", prob=0.3, gamma=[0.7, 1.5]),
        EnsureTyped(keys=["image", "box"]),
        EnsureTyped(keys="label", dtype=torch.long),
    ])


def build_train_transform(
    patch_size: List[int],
    batch_size: int = 2,
    spacing: List[float] = None,
    hu_min: float = -1024.0,
    hu_max: float = 300.0,
) -> Compose:
    """建立訓練用完整 Transform Pipeline（向後相容）。"""
    det = build_det_transform(spacing, hu_min, hu_max)
    rand = build_rand_transform(patch_size, batch_size)
    return Compose([det, rand])


def build_val_transform(
    spacing: List[float] = None,
    hu_min: float = -1024.0,
    hu_max: float = 300.0,
) -> Compose:
    """建立驗證/測試用 Transform Pipeline（無增強、無裁切）。"""
    return build_det_transform(spacing, hu_min, hu_max)


# ─── Utilities ───────────────────────────────────────────────────────

def mask_to_boxes_3d(
    mask: np.ndarray,
    spacing: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    從二值化遮罩中使用連通分量 (Connected Components) 提取 3D 邊框。
    """
    if mask.max() == 0:
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    labeled, num_features = ndimage.label(mask > 0)

    boxes = []
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled == i)
        if len(coords) == 0:
            continue

        d_min, h_min, w_min = coords.min(axis=0)
        d_max, h_max, w_max = coords.max(axis=0)

        box = [d_min, h_min, w_min, d_max + 1, h_max + 1, w_max + 1]
        boxes.append(box)

    if len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    boxes = np.array(boxes, dtype=np.float32)
    labels = np.zeros(len(boxes), dtype=np.int64)
    return boxes, labels


def prepare_datalist(
    data_path: str,
    section: str = "training",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> List[Dict]:
    """
    準備資料列表 (僅支援 JSON)。
    
    若 JSON 已有 training/validation/testing 分割，直接使用。
    若為單一列表，依病歷 ID (seriesuid / lndb_id) 做 patient-level 分割，
    確保同一病歷的所有資料不會跨越分割邊界。
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到資料路徑: {data_path}")

    if not (path.is_file() and path.suffix == ".json"):
        raise ValueError(f"不支援的資料路徑格式: {data_path}。請提供 JSON 檔案。")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_key = section
    if section == "train": target_key = "training"
    if section == "val": target_key = "validation"
    if section == "test": target_key = "testing"

    if target_key in data:
        logging.info(f"已從 {path.name} 載入預定義分割資料 ({target_key}: {len(data[target_key])} 筆)")
        return data[target_key]

    # 需要現場分割
    if isinstance(data, list):
        full_list = data
    elif "data" in data:
        full_list = data["data"]
    else:
        logging.warning(f"JSON 中找不到 key '{target_key}'，也不是列表格式")
        return []

    return _patient_level_split(full_list, section, train_ratio, val_ratio, seed)


def _patient_level_split(
    full_list: List[Dict],
    section: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> List[Dict]:
    """
    依病歷 ID 做 patient-level 分割。
    
    支援的 patient ID 欄位（依優先序）：
    - seriesuid (LUNA16)
    - lndb_id (LNDb)
    - image 路徑的檔名 (fallback)
    """
    from collections import defaultdict

    # 群組化：依病歷 ID 分群
    patient_groups = defaultdict(list)
    for item in full_list:
        pid = (
            item.get("seriesuid")
            or item.get("lndb_id")
            or Path(item.get("image", "unknown")).stem
        )
        patient_groups[str(pid)].append(item)

    patient_ids = sorted(patient_groups.keys())
    logging.info(f"Patient-level 分割: 共 {len(patient_ids)} 位病歷, {len(full_list)} 筆資料 (seed={seed})")

    random.seed(seed)
    random.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    if section == "train":
        selected_ids = patient_ids[:n_train]
    elif section == "val":
        selected_ids = patient_ids[n_train:n_train + n_val]
    elif section == "test":
        selected_ids = patient_ids[n_train + n_val:]
    else:
        return []

    result = []
    for pid in selected_ids:
        result.extend(patient_groups[pid])

    logging.info(f"  {section}: {len(selected_ids)} 位病歷, {len(result)} 筆資料")
    return result


def get_full_split_info(
    data_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    取得完整的分割資訊（用於儲存 data.json）。
    
    Returns:
        包含 training/validation/testing 列表及統計的 dict
    """
    path = Path(data_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 已有預定義分割
    if isinstance(data, dict) and "training" in data:
        train_data = data["training"]
        val_data = data.get("validation", [])
        test_data = data.get("testing", [])
    else:
        full_list = data if isinstance(data, list) else data.get("data", [])
        train_data = _patient_level_split(full_list, "train", train_ratio, val_ratio, seed)
        val_data = _patient_level_split(full_list, "val", train_ratio, val_ratio, seed)
        test_data = _patient_level_split(full_list, "test", train_ratio, val_ratio, seed)

    def _count_with_boxes(items):
        return sum(1 for it in items if len(it.get("box", [])) > 0)

    return {
        "source": str(path.resolve()),
        "split_seed": seed,
        "split_ratio": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "statistics": {
            "train_total": len(train_data),
            "train_with_nodule": _count_with_boxes(train_data),
            "val_total": len(val_data),
            "val_with_nodule": _count_with_boxes(val_data),
            "test_total": len(test_data),
            "test_with_nodule": _count_with_boxes(test_data),
        },
        "training": train_data,
        "validation": val_data,
        "testing": test_data,
    }
