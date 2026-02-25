#!/usr/bin/env python3
"""
RetinaNet Data Pipeline 驗證腳本
================================

驗證 MONAI Transform Pipeline 是否正確產出格式。
用法:
    python -m detection.retinanet.test_data_pipeline --data_path dataset_luna16.json
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def test_data_pipeline(data_path: str, n_samples: int = 3):
    """測試 data pipeline。"""
    from detection.retinanet.dataset import prepare_datalist, build_train_transform, build_val_transform
    from detection.retinanet.config import RetinaNetConfig

    cfg = RetinaNetConfig()

    # 1. 準備資料列表
    train_data = prepare_datalist(data_path, "train")
    val_data = prepare_datalist(data_path, "val")
    logger.info(f"📊 訓練集: {len(train_data)} 筆, 驗證集: {len(val_data)} 筆")

    # 只測試前 n_samples 筆
    train_data = train_data[:n_samples]
    val_data = val_data[:min(n_samples, len(val_data))]

    # 2. 測試訓練 Transform
    logger.info("\n" + "=" * 60)
    logger.info("測試訓練 Transform Pipeline")
    logger.info("=" * 60)

    train_transform = build_train_transform(
        patch_size=cfg.patch_size,
        batch_size=cfg.batch_size,
        spacing=cfg.spacing,
        hu_min=cfg.hu_min,
        hu_max=cfg.hu_max,
    )

    all_pass = True

    for i, item in enumerate(train_data):
        logger.info(f"\n--- 訓練樣本 {i} ---")
        logger.info(f"  Image: {item['image'][:80]}...")
        logger.info(f"  原始 Box 數量: {len(item['box'])}")
        if len(item['box']) > 0:
            logger.info(f"  原始 Box[0]: {item['box'][0]}")
        logger.info(f"  原始 Label: {item['label']}")

        try:
            result = train_transform(item)

            # RandCropBoxByPosNegLabeld 回傳 list of dict
            if isinstance(result, list):
                logger.info(f"  RandCrop 產出 {len(result)} 個子樣本")
                for j, sub in enumerate(result):
                    ok = _validate_sample(sub, f"  [Crop {j}]", is_train=True, patch_size=cfg.patch_size)
                    if not ok:
                        all_pass = False
            else:
                ok = _validate_sample(result, "  ", is_train=True, patch_size=cfg.patch_size)
                if not ok:
                    all_pass = False

        except Exception as e:
            logger.error(f"  ❌ Transform 失敗: {e}", exc_info=True)
            all_pass = False

    # 3. 測試驗證 Transform
    logger.info("\n" + "=" * 60)
    logger.info("測試驗證 Transform Pipeline")
    logger.info("=" * 60)

    val_transform = build_val_transform(
        spacing=cfg.spacing,
        hu_min=cfg.hu_min,
        hu_max=cfg.hu_max,
    )

    for i, item in enumerate(val_data):
        logger.info(f"\n--- 驗證樣本 {i} ---")
        logger.info(f"  Image: {item['image'][:80]}...")

        try:
            result = val_transform(item)
            ok = _validate_sample(result, "  ", is_train=False)
            if not ok:
                all_pass = False
        except Exception as e:
            logger.error(f"  ❌ Transform 失敗: {e}", exc_info=True)
            all_pass = False

    # 4. 結論
    logger.info("\n" + "=" * 60)
    if all_pass:
        logger.info("✅ 所有測試通過！Data pipeline 正常運作。")
    else:
        logger.error("❌ 部分測試失敗，請檢查上方錯誤訊息。")
    logger.info("=" * 60)

    return all_pass


def _validate_sample(sample: dict, prefix: str, is_train: bool, patch_size=None) -> bool:
    """驗證單個樣本的格式與數值。"""
    ok = True

    image = sample["image"]
    box = sample["box"]
    label = sample["label"]

    # Shape 檢查
    logger.info(f"{prefix} image shape: {image.shape}, dtype: {image.dtype}")
    logger.info(f"{prefix} box shape: {box.shape}, dtype: {box.dtype}")
    logger.info(f"{prefix} label shape: {label.shape}, dtype: {label.dtype}")

    # image 應為 (1, D, H, W)
    if image.ndim != 4 or image.shape[0] != 1:
        logger.error(f"{prefix} ❌ image shape 錯誤，應為 (1, D, H, W)，實際: {image.shape}")
        ok = False

    # image 數值應在 [0, 1] (允許小幅超出)
    img_min, img_max = image.min().item(), image.max().item()
    logger.info(f"{prefix} image range: [{img_min:.3f}, {img_max:.3f}]")
    if img_min < -0.5 or img_max > 1.5:
        logger.warning(f"{prefix} ⚠️ image 數值超出預期範圍 [0, 1]")

    # box 應為 (N, 6)
    if box.ndim != 2 or box.shape[1] != 6:
        if box.numel() > 0:
            logger.error(f"{prefix} ❌ box shape 錯誤，應為 (N, 6)，實際: {box.shape}")
            ok = False

    # box 座標應 >= 0
    if box.numel() > 0:
        if (box < 0).any():
            logger.error(f"{prefix} ❌ box 包含負值座標！")
            logger.error(f"{prefix}   box:\n{box}")
            ok = False

        # box min < max
        for dim in range(3):
            if (box[:, dim+3] <= box[:, dim]).any():
                logger.error(f"{prefix} ❌ box dim {dim}: 部分 max <= min！")
                ok = False

        # box 座標應在影像範圍內
        _, D, H, W = image.shape
        max_coords = torch.tensor([D, H, W, D, H, W], dtype=box.dtype)
        if (box > max_coords + 1).any():
            logger.warning(f"{prefix} ⚠️ box 部分座標超出影像範圍 (DHW={D},{H},{W})")
            logger.warning(f"{prefix}   box:\n{box}")

        logger.info(f"{prefix} box 數量: {len(box)}")
        if len(box) > 0:
            logger.info(f"{prefix} box[0]: {box[0].tolist()}")

    # label 應為 torch.long
    if label.dtype != torch.long:
        logger.error(f"{prefix} ❌ label dtype 應為 torch.long，實際: {label.dtype}")
        ok = False

    if ok:
        logger.info(f"{prefix} ✅ 通過")

    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="驗證 RetinaNet data pipeline")
    parser.add_argument("--data_path", required=True, help="dataset.json 路徑")
    parser.add_argument("--n_samples", type=int, default=3, help="測試樣本數")
    args = parser.parse_args()

    test_data_pipeline(args.data_path, args.n_samples)
