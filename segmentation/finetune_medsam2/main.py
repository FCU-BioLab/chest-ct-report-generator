#!/usr/bin/env python3
"""
MedSAM2 Fine-tuning for Chest Tumor Segmentation
================================================

使用 NIfTI 格式的胸部 CT 腫瘤資料集微調 MedSAM2 模型

主要功能:
1. 載入 NIfTI 格式的 CT 和腫瘤遮罩
2. 自動生成 bounding box prompts
3. 資料增強 (Data Augmentation)
4. 微調 MedSAM2 模型
5. 驗證和評估（包含測試集）
6. 保存最佳模型

支援資料集:
- LNDb (推薦): 專家標註的分割遮罩，預期 Dice > 0.85
- LUNA16: 需生成 GT 遮罩，Dice ~0.78

使用範例:
    # ==========================================
    # 快取資料集訓練（推薦）
    # ==========================================
    
    # 基本訓練（使用 LNDb 快取資料）
    python finetune_medsam2/main.py --cache_dataset_type lndb --epochs 100
    
    # 使用 MSD Lung 快取資料
    python finetune_medsam2/main.py --cache_dataset_type msd --epochs 100
    
    # 使用兩個資料集（LNDb + MSD）
    python finetune_medsam2/main.py --cache_dataset_type both --epochs 100
    
    # 啟用資料增強
    python finetune_medsam2/main.py --cache_dataset_type lndb --augmentation --epochs 100
    
    # 過濾小結節 + 強資料增強
    python finetune_medsam2/main.py --cache_dataset_type lndb --min_nodule_diameter 4 --strong_augmentation --epochs 150
    
    # 使用增強損失函數（推薦用於高 DSC）
    python finetune_medsam2/main.py --cache_dataset_type lndb --loss_type enhanced --epochs 100
    
    # 使用 MedSAM2 原生損失函數（與 MedSAM2 訓練一致）
    python finetune_medsam2/main.py --cache_dataset_type lndb --loss_type native --epochs 100
    
    # ==========================================
    # 通用選項
    # ==========================================
    
    # 從 checkpoint 繼續訓練
    python finetune_medsam2/main.py --resume result/segmentation_XXXXXXXX_XXXXXX/best_model.pth
    
    # 只評估模型
    python finetune_medsam2/main.py --eval_only --resume result/segmentation_XXXXXXXX_XXXXXX/best_model.pth
    
    # 測試模式（提取特徵用於 LLM）
    python finetune_medsam2/main.py --test --resume result/segmentation_XXXXXXXX_XXXXXX/best_model.pth
    
    # 快速測試（使用 10% 資料）
    python finetune_medsam2/main.py --data_fraction 0.1 --epochs 5
    
    # 禁用 2.5D 模式（使用傳統 2D）
    python finetune_medsam2/main.py --no_2_5d --epochs 100
"""

import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

# 添加 MedSAM2 路徑到 sys.path
medsam2_path = Path(__file__).parent.parent / "MedSAM2"
if str(medsam2_path) not in sys.path:
    sys.path.insert(0, str(medsam2_path))

# ✅ 修正：添加父目錄到 sys.path 以便能匯入 finetune_medsam2 套件
# 這樣即使直接執行 main.py 也能正確找到套件
package_parent = Path(__file__).parent.parent
if str(package_parent) not in sys.path:
    sys.path.insert(0, str(package_parent))

# ✅ 修正：在主程式初始化 Hydra（只執行一次）
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

config_dir = str((medsam2_path / "sam2" / "configs").absolute())
if Path(config_dir).exists():
    initialize_config_dir(config_dir=config_dir, version_base="1.2")

# 匯入自定義模組
from finetune_medsam2.dataset import LNDbDataset, DataAugmentation, CachedSliceDataset
from finetune_medsam2.trainer import MedSAM2Trainer
from finetune_medsam2.config import Config, get_default_config
from finetune_medsam2.utils import (
    setup_logging, 
    suppress_noisy_logs, 
    split_dataset,
    custom_collate_fn,
    save_dataset_split_info,
    load_dataset_split_info,
    PatientMetricsTracker
)

# 載入預設配置
_default_config = get_default_config()

def main():
    """主函數"""
    # ⚠️ 先解析參數以獲取輸出目錄，然後再初始化 logger
    # 這樣 log 檔案會保存到正確的位置
    
    # 命令列參數解析
    parser = argparse.ArgumentParser(
        description="Fine-tune MedSAM2 for Chest Tumor Segmentation (Cache-Only Mode)"
    )
    
    # 資料參數（僅快取模式）
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=_default_config.data.cache_dir,
        help="快取資料目錄 (預設: cache)"
    )
    parser.add_argument(
        "--cache_dataset_type",
        type=str,
        default=_default_config.data.cache_dataset_type,
        choices=["lndb", "msd", "both"],
        help="快取資料集類型: lndb/msd/both (預設: both)"
    )
    parser.add_argument(
        "--data_fraction", 
        type=float, 
        default=_default_config.data.data_fraction,
        help="使用資料集的比例 (0.0-1.0)，用於快速測試"
    )
    
    # 訓練參數
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=_default_config.training.epochs,
        help="訓練輪數"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=_default_config.training.batch_size,
        help="批次大小"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=_default_config.training.learning_rate,
        help="學習率"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=_default_config.training.weight_decay,
        help="權重衰減"
    )
    parser.add_argument(
        "--early_stopping_patience", 
        type=int, 
        default=_default_config.training.early_stopping_patience,
        help="早停容忍 epoch 數"
    )
    parser.add_argument(
        "--accumulation_steps", 
        type=int, 
        default=_default_config.training.accumulation_steps,
        help="梯度累積步數（模擬更大的 batch size）"
    )
    
    # 模型參數
    parser.add_argument(
        "--config", 
        type=str, 
        default=_default_config.model.config,
        help="MedSAM2 配置檔案"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=_default_config.model.checkpoint,
        help="預訓練模型路徑"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="從 checkpoint 繼續訓練"
    )
    
    # 資料增強
    parser.add_argument(
        "--augmentation", 
        action="store_true",
        help="啟用資料增強"
    )
    
    # 其他參數
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,  # ✅ 改為 None，稍後自動生成時間戳記目錄
        help="輸出目錄（預設: E:/GitHub/chest-ct-report-generator/medsam2_segmentation/result/segmentation_{timestamp}）"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=_default_config.num_workers,
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
        "--test", 
        action="store_true",
        help="測試模式：在測試集上評估並提取病灶特徵用於 LLM Fine-Tuning"
    )
    parser.add_argument(
        "--extract_features", 
        action="store_true",
        help="提取深層特徵向量（需要更多記憶體）"
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        default=True,
        help="保存可視化 PNG 圖片（GT mask、Pred mask、對比圖）"
    )
    parser.add_argument(
        "--no_visualizations",
        action="store_true",
        help="禁用可視化輸出（預設為啟用）"
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="從既有的 dataset_split.json 載入資料集分割（用於復現測試）"
    )
    parser.add_argument(
        "--feature_output_dir", 
        type=str, 
        default=None,
        help="特徵輸出目錄（預設: output_dir/features）"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=_default_config.seed,
        help="隨機種子（預設 42，可重現實驗結果）"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=_default_config.training.loss_type,
        choices=["combined", "enhanced", "native", "tversky", "focal"],
        help="損失函數類型: combined=Dice+BCE(預設), enhanced=多損失組合(推薦高DSC), native=MedSAM2原生(Dice+Focal), tversky=Tversky, focal=Focal"
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=_default_config.training.warmup_epochs,
        help="Warmup epoch 數，學習率從 10%% 線性增加到 100%% (預設 5)"
    )
    parser.add_argument(
        "--strong_augmentation",
        action="store_true",
        help="啟用強資料增強（包含彈性形變、仿射變換等）"
    )
    parser.add_argument(
        "--min_nodule_diameter",
        type=float,
        default=_default_config.inference.min_nodule_diameter,
        help="最小結節直徑 (mm)，過濾小於此值的結節以提高 Dice (建議 4-6mm)"
    )
    parser.add_argument(
        "--use_2_5d",
        action="store_true",
        default=True,
        help="使用 2.5D 輸入 (Z-1, Z, Z+1)，提升上下文資訊和分割效果 (預設啟用)"
    )
    parser.add_argument(
        "--no_2_5d",
        action="store_false",
        dest="use_2_5d",
        help="禁用 2.5D 模式，使用傳統 2D 輸入（三個通道重複同一切片）"
    )
    
    args = parser.parse_args()
    
    # ✅ 如果使用 --resume，自動從 checkpoint 目錄載入設定
    if args.resume:
        resume_path = Path(args.resume)
        resume_dir = resume_path.parent
        
        # 自動設定 output_dir 為 resume 的目錄（除非明確指定）
        if args.output_dir is None:
            args.output_dir = str(resume_dir)
        
        # 自動載入該次訓練的資料集切分（除非明確指定 split_file）
        if args.split_file is None:
            split_file_path = resume_dir / "dataset_split.json"
            if split_file_path.exists():
                args.split_file = str(split_file_path)
        
        # ✅ 新增：自動載入該次訓練的參數配置（除非命令列明確覆寫）
        config_file_path = resume_dir / "training_config.json"
        if config_file_path.exists():
            with open(config_file_path, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
            
            # 需要繼承的參數列表（不包含 resume、output_dir 等控制參數）
            inheritable_params = [
                'dataset_type', 'data_dir', 'rad_id', 'axis', 'data_fraction', 
                'batch_size', 'lr', 'weight_decay', 'early_stopping_patience', 
                'accumulation_steps', 'config', 'checkpoint', 'augmentation', 
                'num_workers', 'cache_data', 'segmentation_method', 'seed', 
                'loss_type', 'warmup_epochs', 'strong_augmentation',
                'min_nodule_diameter'
            ]
            
            # 取得使用者在命令列中明確指定的參數
            # 比較方式：如果目前值等於 parser 預設值，且 saved_config 有不同的值，則採用 saved_config
            parser_defaults = {
                'dataset_type': "lndb",
                'data_dir': "../datasets/aLL_patients_data/LNDb",
                'rad_id': "consensus",
                'axis': 2,
                'data_fraction': 1.0,
                'batch_size': 16,
                'lr': 1e-5,
                'weight_decay': 1e-4,
                'early_stopping_patience': 50,
                'accumulation_steps': 1,
                'config': "sam2.1_hiera_t512.yaml",
                # 'checkpoint': "MedSAM2/checkpoints/MedSAM2_latest.pt",
                'checkpoint': "MedSAM2/checkpoints/MedSAM2_CTLesion.pt",
                'augmentation': False,
                'num_workers': 8,
                'cache_data': False,
                'segmentation_method': "adaptive",
                'seed': None,
                'loss_type': "combined",
                'warmup_epochs': 5,
                'strong_augmentation': False,
                'min_nodule_diameter': 0.0
            }
            
            inherited_params = []
            for param in inheritable_params:
                if param in saved_config:
                    current_val = getattr(args, param, None)
                    default_val = parser_defaults.get(param)
                    saved_val = saved_config[param]
                    
                    # 如果目前值是預設值，且 saved_config 有不同的值，則採用 saved_config
                    if current_val == default_val and saved_val != default_val:
                        setattr(args, param, saved_val)
                        inherited_params.append(f"{param}={saved_val}")
            
            if inherited_params:
                # 稍後在 logger 初始化後輸出訊息
                args._inherited_params = inherited_params
    
    # ✅ 生成時間戳記輸出目錄（僅當 output_dir 仍為 None 時）
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_result_dir = Path(r"C:\GitHub\chest-ct-report-generator\segmentation\result")
        args.output_dir = str(base_result_dir / f"segmentation_{timestamp}")
    
    # 建立輸出目錄
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ✅ 現在初始化 logger（log 檔案會保存到輸出目錄）
    logger = setup_logging(log_dir=args.output_dir)
    suppress_noisy_logs()
    
    logger.info(f"📁 輸出目錄: {args.output_dir}")
    
    # ✅ 輸出繼承的參數資訊
    if hasattr(args, '_inherited_params') and args._inherited_params:
        logger.info(f"🔄 從上次訓練繼承參數:")
        for param_info in args._inherited_params:
            logger.info(f"   ↳ {param_info}")
    
    # 設定隨機種子（如果有提供）
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"🎲 設定隨機種子: {args.seed}")
    else:
        logger.info(f"🎲 未設定隨機種子，訓練過程將使用隨機初始化")
    
    # 檢查 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️ 使用設備: {device}")
    if device.type == "cuda":
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 建立資料增強
    transform = None
    if args.augmentation or args.strong_augmentation:
        if args.strong_augmentation:
            # 強資料增強：更高的增強機率
            transform = DataAugmentation(
                rotation_prob=0.5,
                flip_prob=0.5,
                gamma_prob=0.4,
                noise_prob=0.3,          # 高斯噪聲
                contrast_prob=0.4,       # 對比度調整
                scale_prob=0.3,          # 縮放增強
                elastic_prob=0.3,        # 彈性形變
                bbox_shift_limit=15      # 更大的 bbox 擾動
            )
            logger.info("🔄 已啟用強資料增強（更高機率 + 更多變換）")
        else:
            transform = DataAugmentation(
                rotation_prob=0.5,
                flip_prob=0.5,
                gamma_prob=0.3
            )
            logger.info("🔄 已啟用基本資料增強")
    
    # 建立資料集（僅快取模式）
    logger.info("🔧 建立資料集...")
    
    # ============================================
    # 快取資料集模式（預處理的 .npz 切片）
    # ============================================
    cache_path = Path(args.cache_dir)
    if not cache_path.is_absolute():
        cache_path = Path(__file__).parent.parent / args.cache_dir
    
    if not cache_path.exists():
        logger.error(f"❌ 快取目錄不存在: {cache_path}")
        logger.error(f"請先執行預處理腳本生成快取資料")
        sys.exit(1)
    
    logger.info(f"📂 使用快取資料集: {cache_path}")
    logger.info(f"📊 資料集類型: {args.cache_dataset_type}")
    
    # 取得所有可用的患者 ID
    all_cache_patients = []
    lndb_dir = cache_path / 'lndb_slices'
    msd_dir = cache_path / 'msd_lung_slices'
    
    if args.cache_dataset_type in ('lndb', 'both') and lndb_dir.exists():
        all_cache_patients.extend([d.name for d in lndb_dir.iterdir() if d.is_dir()])
    if args.cache_dataset_type in ('msd', 'both') and msd_dir.exists():
        all_cache_patients.extend([d.name for d in msd_dir.iterdir() if d.is_dir()])
    
    if len(all_cache_patients) == 0:
        logger.error(f"❌ 快取目錄中沒有找到患者資料")
        logger.error(f"   LNDb 目錄: {lndb_dir} (存在: {lndb_dir.exists()})")
        logger.error(f"   MSD 目錄: {msd_dir} (存在: {msd_dir.exists()})")
        logger.error(f"請先執行預處理腳本生成快取資料")
        sys.exit(1)
    
    logger.info(f"📊 找到 {len(all_cache_patients)} 個患者")
    
    # 依比例減少資料集
    if args.data_fraction < 1.0:
        import random
        rng = random.Random(args.seed) if args.seed is not None else random.Random()
        num_samples = max(1, int(len(all_cache_patients) * args.data_fraction))
        all_cache_patients = sorted(rng.sample(all_cache_patients, num_samples))
        logger.info(f"✂️ 依比例 {args.data_fraction} 縮減: 剩餘 {len(all_cache_patients)} 個患者")
    
    # 分割資料集
    train_ids, val_ids, test_ids = split_dataset(all_cache_patients, seed=args.seed)
    logger.info(f"📊 資料集分割: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # 建立資料集
    train_dataset = CachedSliceDataset(
        str(cache_path),
        dataset_type=args.cache_dataset_type,
        patient_ids=train_ids,
        transform=transform,
        use_2_5d=getattr(args, 'use_2_5d', True)  # ✅ 2.5D 模式
    )
    val_dataset = CachedSliceDataset(
        str(cache_path),
        dataset_type=args.cache_dataset_type,
        patient_ids=val_ids,
        transform=None,
        use_2_5d=getattr(args, 'use_2_5d', True)  # ✅ 2.5D 模式
    )
    test_dataset = CachedSliceDataset(
        str(cache_path),
        dataset_type=args.cache_dataset_type,
        patient_ids=test_ids,
        transform=None,
        use_2_5d=getattr(args, 'use_2_5d', True)  # ✅ 2.5D 模式
    )
    
    # 儲存分割資訊
    if not args.eval_only and not args.test:
        save_dataset_split_info(train_ids, val_ids, test_ids, args.output_dir)
    
    # ✅ 輸出資料集過濾統計摘要
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 資料集建立統計摘要")
    logger.info("=" * 70)
    
    # 收集總計統計
    total_stats = {
        'input': 0, 'no_ct': 0, 'no_nodules': 0, 
        'has_nodules': 0, 'filtered': 0, 'kept': 0
    }
    
    for name, dataset, input_count in [
        ("Train", train_dataset, len(train_ids)),
        ("Val  ", val_dataset, len(val_ids)),
        ("Test ", test_dataset, len(test_ids))
    ]:
        stats = dataset.get_filter_stats()
        total_stats['input'] += input_count
        total_stats['no_ct'] += stats.get('patients_no_ct', 0)
        total_stats['no_nodules'] += stats.get('patients_no_nodules', 0)
        total_stats['has_nodules'] += stats.get('patients_has_nodules', 0)
        total_stats['filtered'] += stats.get('patients_filtered', 0)
        total_stats['kept'] += stats.get('patients_kept', 0)
        
        # 計算保留率（基於有病灶的患者）
        has_nodules = stats.get('patients_has_nodules', 0)
        if has_nodules > 0:
            keep_rate = stats['patients_kept'] / has_nodules * 100
        else:
            keep_rate = 0
        
        no_nodules = stats.get('patients_no_nodules', 0)
        logger.info(f"  {name}: 輸入 {input_count} 患者 | 有病灶 {has_nodules} | 無病灶 {no_nodules} "
                   f"→ 過濾 {stats['patients_filtered']} → 保留 {stats['patients_kept']} ({len(dataset)} 切片)")
    
    logger.info("-" * 70)
    logger.info(f"  📋 原始統計: 輸入 {total_stats['input']} 患者 = 有病灶 {total_stats['has_nodules']} + 無病灶 {total_stats['no_nodules']} + 無CT檔 {total_stats['no_ct']}")
    if args.min_nodule_diameter > 0:
        logger.info(f"  🔍 過濾條件: 跳過含有任何 < {args.min_nodule_diameter}mm 結節的患者 (過濾 {total_stats['filtered']} 人)")
    logger.info(f"  ✅ 本次訓練: 共 {total_stats['kept']} 位患者投入訓練")
    total_slices = len(train_dataset) + len(val_dataset) + len(test_dataset)
    logger.info(f"               Train {len(train_dataset)} + Val {len(val_dataset)} + Test {len(test_dataset)} = {total_slices} 切片")
    
    # 輸出結節大小分佈
    total_size_dist = {'micro': 0, 'small': 0, 'medium': 0, 'large': 0}
    for dataset in [train_dataset, val_dataset, test_dataset]:
        for key in total_size_dist:
            total_size_dist[key] += dataset.get_filter_stats().get(f'size_{key}', 0)
    
    logger.info(f"  📏 結節大小分佈: micro(<4mm)={total_size_dist['micro']}, "
               f"small(4-6mm)={total_size_dist['small']}, medium(6-8mm)={total_size_dist['medium']}, "
               f"large(>8mm)={total_size_dist['large']}")
    logger.info("=" * 70)
    logger.info("")
    
    # ✅ 更新 dataset_split.json 為過濾後的患者 ID
    if args.min_nodule_diameter > 0:
        filtered_train_ids = train_dataset.get_kept_patient_ids()
        filtered_val_ids = val_dataset.get_kept_patient_ids()
        filtered_test_ids = test_dataset.get_kept_patient_ids()
        
        # 重新保存過濾後的分割資訊
        save_dataset_split_info(
            filtered_train_ids, filtered_val_ids, filtered_test_ids, 
            args.output_dir,
            original_split_file=args.split_file,
            filter_info={
                'min_nodule_diameter': args.min_nodule_diameter,
                'original_counts': {
                    'train': len(train_ids), 'val': len(val_ids), 'test': len(test_ids)
                },
                'filtered_counts': {
                    'train': len(filtered_train_ids), 
                    'val': len(filtered_val_ids), 
                    'test': len(filtered_test_ids)
                }
            }
        )
        logger.info(f"✅ 已更新 dataset_split.json 為過濾後的 {len(filtered_train_ids)+len(filtered_val_ids)+len(filtered_test_ids)} 位患者")
    
    # 建立 DataLoader
    # ✅ 優化：使用 persistent_workers 和 prefetch_factor 加速資料載入
    use_persistent_workers = args.num_workers > 0  # 只在有 worker 時啟用
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=use_persistent_workers,  # ✅ 避免重複初始化 worker
        prefetch_factor=2 if args.num_workers > 0 else None,  # ✅ 預取 2 個 batch
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=use_persistent_workers,  # ✅ 避免重複初始化 worker
        prefetch_factor=2 if args.num_workers > 0 else None,  # ✅ 預取 2 個 batch
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=use_persistent_workers,  # ✅ 避免重複初始化 worker
        prefetch_factor=2 if args.num_workers > 0 else None,  # ✅ 預取 2 個 batch
        collate_fn=custom_collate_fn
    )
    
    # 建立訓練器
    trainer = MedSAM2Trainer(
        model_config=args.config,
        checkpoint_path=args.checkpoint if not args.resume else None,
        device=device,
        output_dir=args.output_dir,
        loss_type=args.loss_type
    )
    logger.info(f"📊 損失函數類型: {args.loss_type}")
    
    # 載入 checkpoint (如果需要)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # ✅ 測試模式（測試並提取特徵用於 LLM Fine-Tuning）
    if args.test:
        logger.info("🔬 測試模式：評估並提取病灶特徵")
        
        # 確保有 checkpoint
        if not args.resume and not args.checkpoint:
            logger.warning("⚠️ 未指定 checkpoint，將使用初始化權重進行測試")
        
        # 設定特徵輸出目錄
        feature_output_dir = args.feature_output_dir
        if feature_output_dir is None:
            feature_output_dir = Path(args.output_dir) / "features"
        
        # 判斷是否啟用可視化
        save_vis = not args.no_visualizations
        
        # 在測試集上測試並提取特徵
        logger.info("\n📊 測試集評估與特徵提取:")
        test_results = trainer.test_and_extract_features(
            test_loader,
            output_dir=str(feature_output_dir),
            extract_deep_features=args.extract_features,
            save_predictions=True,
            save_visualizations=save_vis,
            spacing=(1.0, 1.0),  # 可以從 DICOM metadata 讀取
            min_area=_default_config.inference.min_area,  # ✅ 從 config 讀取
            min_confidence=_default_config.inference.min_confidence,  # ✅ 從 config 讀取
            min_dice=_default_config.inference.min_dice  # ✅ 從 config 讀取
        )
        
        # 保存測試配置
        test_config = {
            'mode': 'test',
            'checkpoint': args.resume or args.checkpoint,
            'cache_dir': args.cache_dir,  # ✅ 使用 cache_dir 而非 data_dir
            'cache_dataset_type': args.cache_dataset_type,
            'test_samples': test_results['total_samples'],
            'test_lesions': test_results['total_lesions'],
            'test_patients': len(test_results['patient_features']),
            'test_summary': test_results.get('test_summary', {}),
            'extract_features': args.extract_features,
            'feature_output_dir': str(feature_output_dir),
        }
        
        config_path = Path(args.output_dir) / 'test_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(test_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 測試配置已保存: {config_path}")
        logger.info(f"\n🎉 測試完成！特徵已保存至: {feature_output_dir}")
        
        return
    
    # 評估模式
    if args.eval_only:
        logger.info("🔍 評估模式")
        
        # 驗證集評估
        logger.info("\n📊 驗證集評估:")
        val_loss, val_metrics, val_time = trainer.validate(val_loader)
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ 驗證集結果:")
        logger.info(f"  Loss: {val_loss:.4f}")
        logger.info(f"  Dice: {val_metrics['dice']:.4f}")
        logger.info(f"  IoU: {val_metrics['iou']:.4f}")
        logger.info(f"  Recall (Sensitivity): {val_metrics['recall']:.4f}")
        logger.info(f"  Precision (PPV): {val_metrics['precision']:.4f}")
        logger.info(f"  Specificity: {val_metrics['specificity']:.4f}")
        logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  Hausdorff Distance (95%): {val_metrics['hausdorff_95']:.2f} pixels")
        logger.info(f"  Inference Time: {val_time:.1f}s")
        logger.info(f"{'='*80}\n")
        
        # ✅ 測試集評估
        logger.info("\n📊 測試集評估:")
        test_loss, test_metrics, test_time = trainer.validate(test_loader)
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ 測試集結果:")
        logger.info(f"  Loss: {test_loss:.4f}")
        logger.info(f"  Dice: {test_metrics['dice']:.4f}")
        logger.info(f"  IoU: {test_metrics['iou']:.4f}")
        logger.info(f"  Recall (Sensitivity): {test_metrics['recall']:.4f}")
        logger.info(f"  Precision (PPV): {test_metrics['precision']:.4f}")
        logger.info(f"  Specificity: {test_metrics['specificity']:.4f}")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Hausdorff Distance (95%): {test_metrics['hausdorff_95']:.2f} pixels")
        logger.info(f"  Inference Time: {test_time:.1f}s")
        logger.info(f"{'='*80}\n")
        
        return
    
    # ✅ 在訓練開始前保存配置（確保即使訓練中斷也能恢復參數）
    config_dict = {k: v for k, v in vars(args).items() if not k.startswith('_')}
    config_dict['training_started'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    config_path = Path(args.output_dir) / 'training_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 訓練配置已保存: {config_path}")
    
    # 開始訓練
    trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        accumulation_steps=args.accumulation_steps,
        warmup_epochs=args.warmup_epochs
    )
    
    # ✅ 訓練結束後在測試集上評估
    logger.info("\n" + "="*80)
    logger.info("📊 測試集最終評估")
    logger.info("="*80)
    
    # 載入最佳模型
    best_model_path = Path(args.output_dir) / 'best_model.pth'
    if best_model_path.exists():
        trainer.load_checkpoint(str(best_model_path))
    
    # ✅ 新增：建立測試集指標追蹤器
    test_metrics_tracker = PatientMetricsTracker()
    
    # 判斷是否啟用可視化
    save_vis = not args.no_visualizations
    
    # 訓練結束後執行測試並生成可視化
    # 無論是否提取特徵，都會生成可視化結果
    logger.info("🔬 執行測試評估與可視化...")
    feature_output_dir = args.feature_output_dir
    if feature_output_dir is None:
        feature_output_dir = Path(args.output_dir) / "features"
        
    test_results = trainer.test_and_extract_features(
        test_loader,
        output_dir=str(feature_output_dir),
        extract_deep_features=args.extract_features,  # 深層特徵提取是可選的
        save_predictions=True,
        save_visualizations=save_vis
    )
        
    # 同時使用 validate 獲取標準化的測試指標和患者追蹤
    test_loss, test_metrics, test_time = trainer.validate(test_loader, metrics_tracker=test_metrics_tracker)
    
    # ✅ 低分閾值設定
    poor_threshold = 0.75
    
    # ✅ 保存測試集詳細報告（包含 error_cases.json）
    test_metrics_tracker.save_report(args.output_dir, split_name='test', error_threshold=poor_threshold)
    
    # 顯示測試結果
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ 測試集結果:")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Dice: {test_metrics['dice']:.4f}")
    logger.info(f"  IoU: {test_metrics['iou']:.4f}")
    logger.info(f"  Recall (Sensitivity): {test_metrics['recall']:.4f}")
    logger.info(f"  Precision (PPV): {test_metrics['precision']:.4f}")
    logger.info(f"  Specificity: {test_metrics['specificity']:.4f}")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Hausdorff Distance (95%): {test_metrics['hausdorff_95']:.2f} pixels")
    logger.info(f"  Inference Time: {test_time:.1f}s")
    logger.info(f"{'='*80}\n")
    
    # ✅ 顯示低分病例統計（使用相同閾值）
    poor_cases = test_metrics_tracker.get_poor_performers(metric_name='dice', threshold=poor_threshold)
    if poor_cases:
        logger.info(f"⚠️ 發現 {len(poor_cases)} 個低分病例（Dice < {poor_threshold}）:")
        for patient_id, score in poor_cases[:10]:  # 只顯示最差的 10 個
            logger.info(f"   - 患者 {patient_id}: Dice = {score:.4f}")
        if len(poor_cases) > 10:
            logger.info(f"   ... 以及其他 {len(poor_cases) - 10} 個病例")
    else:
        logger.info(f"✅ 沒有低分病例（Dice < {poor_threshold}）")
    logger.info(f"   詳細清單請查看: {args.output_dir}/test_error_cases.json\n")
    
    # 保存訓練配置與測試結果
    config_dict = vars(args)
    config_dict['best_val_dice'] = trainer.best_val_dice
    config_dict['best_val_metrics'] = trainer.best_val_metrics  # 所有 best validation 指標
    config_dict['best_epoch'] = trainer.best_epoch
    config_dict['test_metrics'] = test_metrics
    config_dict['test_loss'] = test_loss
    config_dict['num_poor_cases'] = len(poor_cases)
    
    config_path = Path(args.output_dir) / 'training_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ 訓練配置與測試結果已保存: {config_path}")
    logger.info(f"\n🎉 所有流程完成！")


if __name__ == "__main__":
    main()
