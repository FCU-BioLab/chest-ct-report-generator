#!/usr/bin/env python3
"""
UNet++ Fine-tuning for Chest Tumor Segmentation
================================================

使用 UNet++ 架構進行胸部 CT 腫瘤分割微調

主要功能:
1. 載入 NIfTI 格式的 CT 和腫瘤遮罩
2. 資料增強 (Data Augmentation)
3. 微調 UNet++ 模型
4. 驗證和評估（包含測試集）
5. 保存最佳模型

支援資料集:
- LNDb (推薦): 專家標註的分割遮罩，預期 Dice > 0.85
- LUNA16: 需生成 GT 遮罩

使用範例:
    # ==========================================
    # LNDb 資料集訓練（推薦，專家標註）
    # ==========================================
    
    # 基本訓練
    python finetune_unetpp/main.py --dataset_type lndb --epochs 100
    
    # 啟用資料增強
    python finetune_unetpp/main.py --dataset_type lndb --augmentation --epochs 100
    
    # 使用輕量版模型
    python finetune_unetpp/main.py --dataset_type lndb --model_type lite --epochs 100
    
    # 從 checkpoint 繼續訓練
    python finetune_unetpp/main.py --resume result/unetpp_XXXXXXXX_XXXXXX/best_model.pth
    
    # 只評估模型
    python finetune_unetpp/main.py --eval_only --resume result/unetpp_XXXXXXXX_XXXXXX/best_model.pth
    
    # 快速測試（使用 10% 資料）
    python finetune_unetpp/main.py --data_fraction 0.1 --epochs 5
"""

import sys
import argparse
from pathlib import Path
import json
from datetime import datetime
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

# 添加父目錄到 sys.path
package_parent = Path(__file__).parent.parent
if str(package_parent) not in sys.path:
    sys.path.insert(0, str(package_parent))

# 匯入自定義模組
from finetune_unetpp.model import UNetPlusPlus, UNetPlusPlusLite, get_unetpp_model, get_smp_unetpp_model, count_parameters
from finetune_unetpp.dataset import LNDbDataset, LNDbPatchDataset, DataAugmentation, custom_collate_fn
from finetune_unetpp.trainer import UNetPPTrainer, create_trainer
from finetune_unetpp.losses import get_loss_function
from finetune_unetpp.utils import (
    setup_logging, 
    split_dataset,
    save_dataset_split_info,
    load_dataset_split_info,
    set_seed,
    get_device,
    PatientMetricsTracker,
    worker_init_fn  # [E] For DataLoader reproducibility
)


def get_patient_ids(data_dir: Path, dataset_type: str = "lndb") -> list:
    """
    獲取資料集中的患者 ID 列表
    
    支援兩種 LNDb 資料集結構:
    1. NIfTI 格式: data_dir/LNDb-XXXX/LNDb-XXXX.nii.gz
    2. MHD 格式: data_dir/data0-5/LNDb-XXXX.mhd (原始 LNDb 下載格式)
    
    Args:
        data_dir: 資料集目錄
        dataset_type: 資料集類型
    
    Returns:
        患者 ID 列表
    """
    patient_ids = set()
    
    if dataset_type == "lndb":
        # 方法 1: 尋找 NIfTI 格式 (data_dir/LNDb-XXXX/)
        for item in data_dir.iterdir():
            if item.is_dir() and item.name.startswith("LNDb"):
                patient_ids.add(item.name)
        
        # 方法 2: 尋找 MHD 格式 (data_dir/data0-5/LNDb-XXXX.mhd)
        # 這是 LNDb 原始下載格式
        for subdir in data_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith("data"):
                for mhd_file in subdir.glob("LNDb-*.mhd"):
                    # 從檔名提取患者 ID (例如 LNDb-0001.mhd -> LNDb-0001)
                    patient_id = mhd_file.stem
                    patient_ids.add(patient_id)
    else:
        # 通用格式: 假設每個子目錄是一個患者
        for patient_dir in data_dir.iterdir():
            if patient_dir.is_dir():
                patient_ids.add(patient_dir.name)
    
    return sorted(list(patient_ids))


def main():
    """主函數"""
    # 命令列參數解析
    parser = argparse.ArgumentParser(
        description="Fine-tune UNet++ for Chest Tumor Segmentation"
    )
    
    # 資料參數
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="lndb",
        choices=["lndb", "luna16"],
        help="資料集類型: lndb=LNDb(專家標註,推薦), luna16=LUNA16"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="../datasets/aLL_patients_data/LNDb",
        help="資料集目錄"
    )
    parser.add_argument(
        "--rad_id",
        type=str,
        default="consensus",
        help="LNDb: 放射科醫師ID (1/2/3) 或 'consensus' 使用多數投票"
    )
    parser.add_argument(
        "--axis", 
        type=int, 
        default=2, 
        choices=[0, 1, 2],
        help="切片軸向 (0=sagittal, 1=coronal, 2=axial)"
    )
    parser.add_argument(
        "--data_fraction", 
        type=float, 
        default=1.0,
        help="使用資料集的比例 (0.0-1.0)，用於快速測試"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="目標影像大小 (height width)"
    )
    parser.add_argument(
        "--use_patches",
        action="store_true",
        help="使用 2x2 網格 patch 模式 (224x224 patches)"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=224,
        help="Patch 大小 (預設 224)"
    )
    parser.add_argument(
        "--hsv_augmentation",
        action="store_true",
        help="啟用 HSV 色彩空間增強"
    )
    parser.add_argument(
        "--filter_empty_patches",
        action="store_true",
        default=True,
        help="過濾沒有病灶的空 patches (預設啟用)"
    )
    parser.add_argument(
        "--no_filter_empty_patches",
        action="store_true",
        help="包含所有 patches (不過濾空 patches)"
    )
    parser.add_argument(
        "--empty_patch_ratio",
        type=float,
        default=0.2,
        help="保留空 patches 的比例 (預設 0.2 = 20%%)"
    )
    parser.add_argument(
        "--include_full_slices",
        action="store_true",
        help="包含完整切片進行混合尺度訓練 (patches + full slices)"
    )
    parser.add_argument(
        "--full_slice_size",
        type=int,
        default=448,
        help="完整切片的大小 (預設 448)"
    )
    
    # 模型參數
    parser.add_argument(
        "--use_smp",
        action="store_true",
        default=True,
        help="使用 segmentation_models.pytorch 的 UNet++ (推薦)"
    )
    parser.add_argument(
        "--no_smp",
        action="store_true",
        help="使用自定義 UNet++ 實現"
    )
    parser.add_argument(
        "--encoder_name",
        type=str,
        default="resnet34",
        help="SMP Encoder 名稱 (resnet34, resnet50, efficientnet-b0, mobilenet_v2, etc.)"
    )
    parser.add_argument(
        "--encoder_weights",
        type=str,
        default="imagenet",
        help="Encoder 預訓練權重 ('imagenet' or 'None')"
    )
    parser.add_argument(
        "--encoder_depth",
        type=int,
        default=5,
        choices=[3, 4, 5],
        help="Encoder 深度 (3-5)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="standard",
        choices=["standard", "lite", "attention", "no_attention"],
        help="自定義模型類型 (僅當 --no_smp 時使用)"
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="輸入通道數"
    )
    parser.add_argument(
        "--features",
        type=int,
        nargs='+',
        default=[64, 128, 256, 512, 1024],
        help="各層特徵通道數 (僅當 --no_smp 時使用)"
    )
    parser.add_argument(
        "--deep_supervision",
        action="store_true",
        default=True,
        help="使用深度監督 (僅當 --no_smp 時使用)"
    )
    parser.add_argument(
        "--no_deep_supervision",
        action="store_true",
        help="禁用深度監督"
    )
    
    # 訓練參數
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="訓練輪數"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="批次大小"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-4,
        help="學習率"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=1e-4,
        help="權重衰減"
    )
    parser.add_argument(
        "--early_stopping_patience", 
        type=int, 
        default=30,
        help="早停容忍 epoch 數"
    )
    parser.add_argument(
        "--accumulation_steps", 
        type=int, 
        default=1,
        help="梯度累積步數（模擬更大的 batch size）"
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Warmup epoch 數"
    )
    
    # 損失函數
    parser.add_argument(
        "--loss_type",
        type=str,
        default="stable",  # [Fix A] 預設使用穩定版 loss
        choices=["dice", "combined", "stable", "enhanced", "tversky", "focal"],
        help="損失函數類型 (stable: BCE+SoftDice 穩定版)"
    )
    
    # 資料增強
    parser.add_argument(
        "--augmentation", 
        action="store_true",
        help="啟用資料增強"
    )
    parser.add_argument(
        "--strong_augmentation",
        action="store_true",
        help="啟用強資料增強"
    )
    
    # 其他參數
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="輸出目錄（預設: result/unetpp_{timestamp}）"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="DataLoader 工作進程數"
    )
    parser.add_argument(
        "--cache_data", 
        action="store_true",
        help="緩存資料到記憶體（需要大量 RAM）"
    )
    parser.add_argument(
        "--eval_only", 
        action="store_true",
        help="只進行評估（不訓練）"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="從 checkpoint 繼續訓練"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="隨機種子"
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="從既有的 dataset_split.json 載入資料集分割"
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        default=True,
        help="保存可視化圖片"
    )
    parser.add_argument(
        "--no_visualizations",
        action="store_true",
        help="禁用可視化輸出"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="使用混合精度訓練"
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="禁用混合精度訓練"
    )
    parser.add_argument(
        "--extract_llm_features",
        action="store_true",
        help="Test 時提取 LLM 特徵 (deep features + morphological)"
    )
    
    args = parser.parse_args()
    
    # 處理互斥參數
    if args.no_smp:
        args.use_smp = False
    if args.no_deep_supervision:
        args.deep_supervision = False
    if args.no_visualizations:
        args.save_visualizations = False
    if args.no_amp:
        args.use_amp = False
    if args.encoder_weights.lower() == "none":
        args.encoder_weights = None
    
    # 設定輸出目錄
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_result_dir = Path(r"C:\GitHub\chest-ct-report-generator\segmentation\result")
        args.output_dir = str(base_result_dir / f"unetpp_{timestamp}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 設定日誌
    logger = setup_logging(str(output_dir))
    
    logger.info(f"📁 輸出目錄: {args.output_dir}")
    
    # 設定隨機種子
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"🎲 設定隨機種子: {args.seed}")
    else:
        logger.info(f"🎲 未設定隨機種子，訓練過程將使用隨機初始化")
    
    # 設定設備
    device = get_device()
    logger.info(f"🖥️ 使用設備: {device}")
    if str(device) == "cuda":
        import torch
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # ==================== 資料準備 ====================
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        # 嘗試相對路徑
        data_dir = Path(__file__).parent.parent.parent / args.data_dir
    
    if not data_dir.exists():
        logger.error(f"❌ 資料目錄不存在: {data_dir}")
        return
    
    # 根據資料集類型獲取患者 ID
    if args.dataset_type == "lndb":
        import pandas as pd
        gt_csv = data_dir / "trainset_csv" / "trainNodules_gt.csv"
        if gt_csv.exists():
            df = pd.read_csv(gt_csv)
            all_patient_ids = sorted(df['LNDbID'].unique().tolist())
            logger.info(f"📊 找到 {len(all_patient_ids)} 個 CT ({len(df)} 個結節標註) [LNDb]")
            
            # 檢查 mask 資料夾
            mask_dir = data_dir / "masks" / "masks"
            if mask_dir.exists():
                mask_count = len(list(mask_dir.glob("*.mhd")))
                logger.info(f"🎭 專家分割遮罩: {mask_count} 個")
        else:
            # 使用舊的方法
            all_patient_ids = get_patient_ids(data_dir, args.dataset_type)
            logger.info(f"📊 找到 {len(all_patient_ids)} 個患者 [LNDb]")
    else:
        all_patient_ids = get_patient_ids(data_dir, args.dataset_type)
        logger.info(f"📊 找到 {len(all_patient_ids)} 個患者 [{args.dataset_type.upper()}]")
    
    if len(all_patient_ids) == 0:
        logger.error("❌ 資料目錄為空或未找到患者!")
        return
    
    # 資料分割
    if args.split_file and Path(args.split_file).exists():
        logger.info(f"📂 從既有分割檔案載入: {args.split_file}")
        split_info = load_dataset_split_info(args.split_file)
        
        # 顯示原始分割日期（如果有）
        if 'split_date' in split_info:
            logger.info(f"   ↳ 原始分割日期: {split_info['split_date']}")
        
        # 支援兩種 JSON 格式:
        # 格式 1 (UNet++): {"train_ids": [...], "val_ids": [...], "test_ids": [...]}
        # 格式 2 (MedSAM2): {"train": {"patient_ids": [...]}, "val": {"patient_ids": [...]}, ...}
        if 'train_ids' in split_info:
            train_ids = split_info['train_ids']
            val_ids = split_info['val_ids']
            test_ids = split_info['test_ids']
        elif 'train' in split_info and 'patient_ids' in split_info['train']:
            train_ids = split_info['train']['patient_ids']
            val_ids = split_info['val']['patient_ids']
            test_ids = split_info['test']['patient_ids']
        else:
            logger.error(f"❌ 不支援的分割檔案格式: {args.split_file}")
            return
        
        # ✅ 統一 ID 類型（LNDb 使用整數，其他使用字串）
        if args.dataset_type == "lndb":
            train_ids = [int(pid) if isinstance(pid, str) and pid.isdigit() else pid for pid in train_ids]
            val_ids = [int(pid) if isinstance(pid, str) and pid.isdigit() else pid for pid in val_ids]
            test_ids = [int(pid) if isinstance(pid, str) and pid.isdigit() else pid for pid in test_ids]
        
        # 驗證載入的 patient IDs 是否存在於資料集中
        loaded_ids = set(train_ids + val_ids + test_ids)
        available_ids = set(all_patient_ids)
        missing_ids = loaded_ids - available_ids
        if missing_ids:
            logger.warning(f"⚠️ 分割檔案中有 {len(missing_ids)} 個患者在資料集中不存在，將被忽略")
            train_ids = [pid for pid in train_ids if pid in available_ids]
            val_ids = [pid for pid in val_ids if pid in available_ids]
            test_ids = [pid for pid in test_ids if pid in available_ids]
        
        logger.info(f"   ↳ Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    else:
        logger.info("🎲 使用隨機切分 (每次執行都不同)" if args.seed is None else f"🎲 使用固定隨機種子 {args.seed} 切分")
        train_ids, val_ids, test_ids = split_dataset(
            all_patient_ids,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=args.seed
        )
    
    logger.info(f"📊 資料集分割: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # 保存分割資訊
    config = vars(args)
    save_dataset_split_info(
        str(output_dir),
        train_ids, val_ids, test_ids,
        config
    )
    
    # 資料比例
    if args.data_fraction < 1.0:
        n_train = max(1, int(len(train_ids) * args.data_fraction))
        n_val = max(1, int(len(val_ids) * args.data_fraction))
        n_test = max(1, int(len(test_ids) * args.data_fraction))
        train_ids = train_ids[:n_train]
        val_ids = val_ids[:n_val]
        test_ids = test_ids[:n_test]
        logger.info(f"✂️ 依比例 {args.data_fraction} 縮減資料集")
    
    # 資料增強
    transform = None
    if args.augmentation or args.strong_augmentation:
        transform = DataAugmentation(
            rotation_range=15.0 if not args.strong_augmentation else 30.0,
            flip_prob=0.5,
            noise_std=0.02 if not args.strong_augmentation else 0.05,
            hsv_augmentation=args.hsv_augmentation,
            strong_augmentation=args.strong_augmentation
        )
        aug_msg = "Data augmentation enabled"
        if args.hsv_augmentation:
            aug_msg += " (with HSV)"
        logger.info(aug_msg)
    
    # 建立資料集
    target_size = tuple(args.target_size)
    
    if args.dataset_type == "lndb":
        # 選擇 Dataset 類別
        DatasetClass = LNDbPatchDataset if args.use_patches else LNDbDataset
        dataset_type_str = "LNDbPatchDataset (224x224 patches)" if args.use_patches else "LNDbDataset"
        
        if args.use_patches:
            # 決定是否過濾空 patches
            filter_empty = args.filter_empty_patches and not args.no_filter_empty_patches
            if filter_empty:
                logger.info("🔍 Filter empty patches enabled")
            
            logger.info(f"📦 Using patch mode: {args.patch_size}x{args.patch_size} patches")
        
        # 建立資料集 - LNDbPatchDataset 不需要 target_size 參數
        if args.use_patches:
            filter_empty = args.filter_empty_patches and not args.no_filter_empty_patches
            train_dataset = DatasetClass(
                data_dir=str(data_dir),
                patient_ids=train_ids,
                patch_size=args.patch_size,
                rad_id=args.rad_id,
                axis=args.axis,
                transform=transform,
                cache_data=args.cache_data,
                filter_empty_patches=filter_empty,
                empty_patch_ratio=args.empty_patch_ratio,
                include_full_slices=args.include_full_slices,
                full_slice_size=args.full_slice_size
            )
            val_dataset = DatasetClass(
                data_dir=str(data_dir),
                patient_ids=val_ids,
                patch_size=args.patch_size,
                rad_id=args.rad_id,
                axis=args.axis,
                transform=None,
                cache_data=args.cache_data,
                filter_empty_patches=filter_empty,
                empty_patch_ratio=args.empty_patch_ratio,
                include_full_slices=args.include_full_slices,
                full_slice_size=args.full_slice_size
            )
            test_dataset = DatasetClass(
                data_dir=str(data_dir),
                patient_ids=test_ids,
                patch_size=args.patch_size,
                rad_id=args.rad_id,
                axis=args.axis,
                transform=None,
                cache_data=args.cache_data,
                filter_empty_patches=filter_empty,
                empty_patch_ratio=args.empty_patch_ratio,
                include_full_slices=args.include_full_slices,
                full_slice_size=args.full_slice_size
            )
        else:
            train_dataset = DatasetClass(
                data_dir=str(data_dir),
                patient_ids=train_ids,
                rad_id=args.rad_id,
                axis=args.axis,
                transform=transform,
                cache_data=args.cache_data,
                target_size=target_size
            )
            val_dataset = DatasetClass(
                data_dir=str(data_dir),
                patient_ids=val_ids,
                rad_id=args.rad_id,
                axis=args.axis,
                transform=None,
                cache_data=args.cache_data,
                target_size=target_size
            )
            test_dataset = DatasetClass(
                data_dir=str(data_dir),
                patient_ids=test_ids,
                rad_id=args.rad_id,
                axis=args.axis,
                transform=None,
                cache_data=args.cache_data,
                target_size=target_size
            )
    else:
        logger.error(f"Unsupported dataset type: {args.dataset_type}")
        return
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # [F] Input channel validation
    if len(train_dataset) > 0:
        first_sample = train_dataset[0]
        logger.info(f"🔬 [F] Input Validation:")
        logger.info(f"   First sample image shape: {first_sample['image'].shape}")
        logger.info(f"   First sample mask shape: {first_sample['mask'].shape}")
        logger.info(f"   Model in_channels: {args.in_channels}")
        if first_sample['image'].shape[0] != args.in_channels:
            logger.error(f"❌ MISMATCH: dataset outputs {first_sample['image'].shape[0]} channels, "
                        f"but model expects {args.in_channels}")
            return
        logger.info(f"   ✅ Channel count verified")
    
    # [E] Create generator for reproducible shuffling
    g = torch.Generator()
    if args.seed is not None:
        g.manual_seed(args.seed)
        logger.info(f"🎲 [E] DataLoader generator seeded with: {args.seed}")
    
    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,  # [E] For reproducibility
        generator=g  # [E] For reproducible shuffling
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn  # [E] For reproducibility
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn  # [E] For reproducibility
    )
    
    # ==================== 模型建立 ====================
    logger.info("\n" + "=" * 40)
    logger.info("Model Setup")
    logger.info("=" * 40)
    
    if args.use_smp:
        # 使用 segmentation_models.pytorch 的 UNet++
        model = get_smp_unetpp_model(
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
            in_channels=args.in_channels,
            classes=1,
            encoder_depth=args.encoder_depth,
        )
        logger.info(f"🔧 使用 SMP UNet++")
        logger.info(f"   Encoder: {args.encoder_name}")
        logger.info(f"   Pretrained: {args.encoder_weights}")
        logger.info(f"   Encoder depth: {args.encoder_depth}")
        # SMP 模型不支援深度監督
        args.deep_supervision = False
    else:
        # 使用自定義 UNet++
        model = get_unetpp_model(
            model_type=args.model_type,
            in_channels=args.in_channels,
            out_channels=1,
            features=args.features,
            deep_supervision=args.deep_supervision
        )
        logger.info(f"🔧 使用自定義 UNet++")
        logger.info(f"   Model type: {args.model_type}")
        logger.info(f"   Deep supervision: {args.deep_supervision}")
    
    num_params = count_parameters(model)
    logger.info(f"   Parameters: {num_params:,}")
    
    # 載入 checkpoint
    start_epoch = 0
    if args.resume:
        if Path(args.resume).exists():
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            logger.info(f"Resumed from {args.resume}, epoch {start_epoch}")
        else:
            logger.warning(f"Checkpoint not found: {args.resume}")
    
    # ==================== 訓練 ====================
    if not args.eval_only:
        logger.info("\n" + "=" * 40)
        logger.info("Training")
        logger.info("=" * 40)
        
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=args.lr,
            weight_decay=args.weight_decay,
            loss_type=args.loss_type,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            output_dir=str(output_dir),
            device=device,
            use_amp=args.use_amp,
            accumulation_steps=args.accumulation_steps,
            deep_supervision=args.deep_supervision,
            save_visualizations=args.save_visualizations
        )
        
        # 如果有 resume，載入優化器狀態
        if args.resume and Path(args.resume).exists():
            trainer.load_checkpoint(args.resume)
        
        # 開始訓練
        history = trainer.train(
            epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
            save_freq=10
        )
        
        # 評估測試集
        logger.info("\n" + "=" * 40)
        logger.info("Test Evaluation")
        logger.info("=" * 40)
        
        test_metrics = trainer.evaluate(test_loader)
        logger.info(f"Test Dice: {test_metrics['dice']:.4f}")
        logger.info(f"Test IoU: {test_metrics['iou']:.4f}")
        
        # 提取 LLM 特徵
        if args.extract_llm_features:
            logger.info("\n" + "=" * 40)
            logger.info("Extracting LLM Features")
            logger.info("=" * 40)
            llm_features = trainer.extract_llm_features(test_loader, str(output_dir))
            logger.info(f"Extracted features for {llm_features['extraction_info']['total_samples']} samples")
        
    else:
        # 只評估
        logger.info("\n" + "=" * 40)
        logger.info("Evaluation Only")
        logger.info("=" * 40)
        
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            output_dir=str(output_dir),
            device=device,
            save_visualizations=args.save_visualizations
        )
        
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        test_metrics = trainer.evaluate(test_loader)
        logger.info(f"Test Dice: {test_metrics['dice']:.4f}")
        logger.info(f"Test IoU: {test_metrics['iou']:.4f}")
        
        # 提取 LLM 特徵
        if args.extract_llm_features:
            logger.info("\n" + "=" * 40)
            logger.info("Extracting LLM Features")
            logger.info("=" * 40)
            llm_features = trainer.extract_llm_features(test_loader, str(output_dir))
            logger.info(f"Extracted features for {llm_features['extraction_info']['total_samples']} samples")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
