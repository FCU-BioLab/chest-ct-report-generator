#!/usr/bin/env python3
"""
MSD Lung Tumours UNet++ 訓練
============================

使用 Medical Segmentation Decathlon Task06 (Lung Tumours) 資料集訓練 UNet++

使用方式:
    # 預處理
    python train_msd.py --preprocess
    
    # 訓練
    python train_msd.py --epochs 100
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# 確保可以導入
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_unetpp.config import Config, get_default_config
from train_unetpp.msd_dataset import (
    MSD_LUNG_DIR, MSD_CACHE_DIR,
    preprocess_msd_lung, get_msd_lung_cases, get_msd_train_val_split,
    MSDLungSliceDataset, msd_val_collate_fn
)
from train_unetpp.trainer import UNetPPTrainer
from train_unetpp.utils import setup_logging, set_seed, get_device, custom_collate_fn

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="MSD Lung Tumours UNet++ 訓練")
    
    parser.add_argument('--preprocess', action='store_true', help='執行預處理')
    parser.add_argument('--epochs', type=int, default=100, help='訓練 epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='學習率')
    parser.add_argument('--patch_size', type=int, default=224, help='Patch 大小')
    parser.add_argument('--encoder', type=str, default='efficientnet-b4', help='編碼器')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--output_dir', type=str, default=None, help='輸出目錄')
    
    return parser.parse_args()


def run_training(args):
    """執行訓練"""
    # 設置
    set_seed(args.seed)
    device = get_device()
    
    logger.info(f"使用設備: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 載入配置
    config = get_default_config()
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.data.patch_size = args.patch_size
    config.model.encoder_name = args.encoder
    config.model.in_channels = 3  # 2.5D
    config.seed = args.seed
    config.num_workers = args.num_workers
    config.device = str(device)
    
    # 取得案例並分割 (70/15/15)
    cases = get_msd_lung_cases(MSD_LUNG_DIR)
    logger.info(f"找到 {len(cases)} 個案例")
    
    train_ids, val_ids, test_ids = get_msd_train_val_split(cases, val_ratio=0.15, test_ratio=0.15, seed=args.seed)
    logger.info(f"訓練集: {len(train_ids)} 案例, 驗證集: {len(val_ids)} 案例, 測試集: {len(test_ids)} 案例")
    
    # 檢查快取是否存在
    if not MSD_CACHE_DIR.exists() or len(list(MSD_CACHE_DIR.iterdir())) == 0:
        logger.error(f"快取目錄不存在或為空: {MSD_CACHE_DIR}")
        logger.error("請先執行 --preprocess")
        return
    
    # 創建資料集
    logger.info("建立訓練資料集...")
    train_dataset = MSDLungSliceDataset(
        case_ids=train_ids,
        cache_dir=MSD_CACHE_DIR,
        mode="train",
        patch_size=config.data.patch_size
    )
    
    logger.info("建立驗證資料集...")
    val_dataset = MSDLungSliceDataset(
        case_ids=val_ids,
        cache_dir=MSD_CACHE_DIR,
        mode="val",
        patch_size=config.data.patch_size
    )
    
    # 創建 DataLoader
    actual_workers = min(4, args.num_workers)
    logger.info(f"DataLoader workers: {actual_workers}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=actual_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=actual_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 4-patch stitch
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=False,  # 避免 OOM
        collate_fn=msd_val_collate_fn,
        persistent_workers=actual_workers > 0
    )
    
    # 設置輸出目錄
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("result") / f"msd_lung_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建 Trainer
    data_split = {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids
    }
    trainer = UNetPPTrainer(config, data_split=data_split, output_dir=run_dir)
    
    # 訓練
    logger.info(f"開始訓練，共 {config.training.epochs} 個 epoch")
    logger.info(f"輸出目錄: {run_dir}")
    
    history = trainer.fit(train_loader, val_loader)
    
    logger.info("訓練完成！")
    return trainer, history


def main():
    args = parse_args()
    
    # 先創建輸出目錄（用於日誌）
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("result") / f"msd_lung_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 設置日誌（含檔案輸出）
    log_file = run_dir / "train.log" if not args.preprocess else None
    setup_logging(
        log_file=str(log_file) if log_file else None,
        level=logging.INFO
    )
    
    if args.preprocess:
        logger.info("執行 MSD Lung Tumours 預處理...")
        preprocess_msd_lung(MSD_LUNG_DIR, MSD_CACHE_DIR)
    else:
        # 傳入已創建的 run_dir
        args.output_dir = str(run_dir)
        run_training(args)


if __name__ == "__main__":
    main()

