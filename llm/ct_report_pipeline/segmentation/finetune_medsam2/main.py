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
    # LNDb 資料集訓練（推薦，專家標註）
    # ==========================================
    
    # 基本訓練（使用多放射科醫師共識遮罩）
    python finetune_medsam2/main.py --dataset_type lndb --epochs 100
    
    # 啟用資料增強
    python finetune_medsam2/main.py --dataset_type lndb --augmentation --epochs 100
    
    # 使用單一放射科醫師標註
    python finetune_medsam2/main.py --dataset_type lndb --rad_id 1 --epochs 100
    
    # 過濾小結節 + 強資料增強
    python finetune_medsam2/main.py --dataset_type lndb --min_nodule_diameter 4 --strong_augmentation --epochs 150
    
    # ==========================================
    # LUNA16 資料集訓練（需生成 GT）
    # ==========================================
    
    # 基本訓練
    python finetune_medsam2/main.py --dataset_type luna16 --data_dir ../datasets/aLL_patients_data --epochs 100
    
    # 過濾小結節訓練（推薦，提高 Dice）
    python finetune_medsam2/main.py --dataset_type luna16 --min_nodule_diameter 6 --epochs 200
    
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
from finetune_medsam2.dataset import ChestTumorDataset, LNDbDataset, DataAugmentation
from finetune_medsam2.trainer import MedSAM2Trainer
from finetune_medsam2.utils import (
    setup_logging, 
    suppress_noisy_logs, 
    split_dataset,
    custom_collate_fn,
    save_dataset_split_info,
    load_dataset_split_info,
    PatientMetricsTracker
)

# 添加專案根目錄到 sys.path 以匯入 config
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 匯入配置模組
try:
    from config import load_config, get_lndb_root, get_output_paths, get_training_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False


def main():
    """主函數"""
    # ⚠️ 先解析參數以獲取輸出目錄，然後再初始化 logger
    # 這樣 log 檔案會保存到正確的位置
    
    # 命令列參數解析
    parser = argparse.ArgumentParser(
        description="Fine-tune MedSAM2 for Chest Tumor Segmentation"
    )
    
    # 載入配置以取得預設路徑
    if _CONFIG_AVAILABLE:
        try:
            _config = load_config()
            _lndb_default = str(get_lndb_root(_config))
            _output_default = str(get_output_paths(_config)['segmentation_results'])
        except Exception:
            _lndb_default = "../datasets/aLL_patients_data/LNDb"
            _output_default = None
    else:
        _lndb_default = "../datasets/aLL_patients_data/LNDb"
        _output_default = None
    
    # 資料參數
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="lndb",
        choices=["luna16", "lndb"],
        help="資料集類型: luna16=LUNA16(需生成GT), lndb=LNDb(專家標註,推薦)"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=_lndb_default,
        help=f"資料集目錄 (LNDb 或 LUNA16 root) (default: {_lndb_default})"
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
    
    # 訓練參數
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50,
        help="訓練輪數"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="批次大小"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=5e-6,
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
        default=50,
        help="早停容忍 epoch 數"
    )
    parser.add_argument(
        "--accumulation_steps", 
        type=int, 
        default=1,
        help="梯度累積步數（模擬更大的 batch size）"
    )
    
    # 模型參數
    parser.add_argument(
        "--config", 
        type=str, 
        default="sam2.1_hiera_t512.yaml",
        help="MedSAM2 配置檔案"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        # default="MedSAM2/checkpoints/MedSAM2_latest.pt",
        default="MedSAM2/checkpoints/MedSAM2_CTLesion.pt",
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
        default=8,
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
        default=None,
        help="隨機種子（留空則隨機切分，設定數字則可重現）"
    )
    parser.add_argument(
        "--segmentation_method",
        type=str,
        default="adaptive",
        choices=["sphere", "threshold", "region_growing", "watershed", "adaptive"],
        help="GT 分割方法: sphere=球形遮罩, threshold=HU閾值, region_growing=區域生長, watershed=分水嶺, adaptive=自適應(預設,推薦)"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="combined",
        choices=["combined", "enhanced", "tversky", "focal"],
        help="損失函數類型: combined=Dice+BCE(預設), enhanced=多損失組合(推薦高DSC), tversky=Tversky, focal=Focal"
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
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
        default=0.0,
        help="最小結節直徑 (mm)，過濾小於此值的結節以提高 Dice (建議 4-6mm)"
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
        if _output_default:
            base_result_dir = Path(_output_default)
        else:
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
    
    # 獲取所有患者 ID
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"❌ 資料目錄不存在: {data_dir}")
        sys.exit(1)
    
    # 根據資料集類型獲取患者 ID
    all_patients = []
    
    if args.dataset_type == "lndb":
        # ============================================
        # LNDb 資料集：讀取 trainNodules_gt.csv
        # ============================================
        import pandas as pd
        gt_csv = data_dir / "trainset_csv" / "trainNodules_gt.csv"
        if not gt_csv.exists():
            logger.error(f"❌ LNDb GT 檔案不存在: {gt_csv}")
            sys.exit(1)
        
        df = pd.read_csv(gt_csv)
        # LNDbID 是整數
        all_patients = sorted(df['LNDbID'].unique().tolist())
        logger.info(f"📊 找到 {len(all_patients)} 個 CT ({len(df)} 個結節標註) [LNDb]")
        
        # 檢查 mask 資料夾
        mask_dir = data_dir / "masks" / "masks"
        if mask_dir.exists():
            mask_count = len(list(mask_dir.glob("*.mhd")))
            logger.info(f"🎭 專家分割遮罩: {mask_count} 個")
        
    else:
        # ============================================
        # LUNA16 資料集：掃描 subset 資料夾
        # ============================================
        for i in range(10):
            subset_dir = data_dir / f"subset{i}"
            if subset_dir.exists():
                for f in subset_dir.glob("*.mhd"):
                    all_patients.append(f.stem)
        
        # 去除重複並排序
        all_patients = sorted(list(set(all_patients)))
        logger.info(f"📊 找到 {len(all_patients)} 個患者 [LUNA16]")
    
    # 依比例減少資料集
    if args.data_fraction < 1.0:
        import random
        # 使用相同的 seed 確保每次縮減的結果一致 (如果 seed 有設定)
        rng = random.Random(args.seed) if args.seed is not None else random.Random()
        
        num_samples = int(len(all_patients) * args.data_fraction)
        # 確保至少有一個樣本
        num_samples = max(1, num_samples)
        
        all_patients = rng.sample(all_patients, num_samples)
        all_patients = sorted(all_patients)
        logger.info(f"✂️ 依比例 {args.data_fraction} 縮減資料集: 剩餘 {len(all_patients)} 個 CT")
    
    if len(all_patients) == 0:
        logger.error(f"❌ 資料目錄為空或未找到 .mhd 檔案: {data_dir}")
        sys.exit(1)
    
    # 分割資料集
    if args.split_file:
        # ✅ 從既有的 split 檔案載入
        logger.info(f"📂 從既有分割檔案載入: {args.split_file}")
        if args.resume:
            logger.info(f"   ↳ 自動繼承自 resume checkpoint 的訓練設定")
        train_ids, val_ids, test_ids = load_dataset_split_info(args.split_file)
        
        # ✅ 修正：統一 ID 類型（LNDb 使用整數，LUNA16 使用字串）
        if args.dataset_type == "lndb":
            # LNDb: 確保所有 ID 都是整數
            train_ids = [int(pid) if isinstance(pid, str) else pid for pid in train_ids]
            val_ids = [int(pid) if isinstance(pid, str) else pid for pid in val_ids]
            test_ids = [int(pid) if isinstance(pid, str) else pid for pid in test_ids]
        else:
            # LUNA16: 確保所有 ID 都是字串
            train_ids = [str(pid) for pid in train_ids]
            val_ids = [str(pid) for pid in val_ids]
            test_ids = [str(pid) for pid in test_ids]
        
        # 驗證載入的 patient IDs 是否存在於資料集中
        loaded_ids = set(train_ids + val_ids + test_ids)
        available_ids = set(all_patients)
        missing_ids = loaded_ids - available_ids
        if missing_ids:
            logger.warning(f"⚠️ 分割檔案中有 {len(missing_ids)} 個患者在資料集中不存在，將被忽略")
            train_ids = [pid for pid in train_ids if pid in available_ids]
            val_ids = [pid for pid in val_ids if pid in available_ids]
            test_ids = [pid for pid in test_ids if pid in available_ids]
    else:
        # 隨機分割資料集
        train_ids, val_ids, test_ids = split_dataset(all_patients, seed=args.seed)
    
    logger.info(
        f"📊 資料集分割: Train={len(train_ids)}, "
        f"Val={len(val_ids)}, Test={len(test_ids)}"
    )
    
    # ✅ 修正：只在非 eval_only 和非 test 模式時才覆寫 split 檔案
    if not args.eval_only and not args.test:
        save_dataset_split_info(train_ids, val_ids, test_ids, args.output_dir)
    
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
    
    # 建立資料集
    logger.info("🔧 建立資料集...")
    
    if args.dataset_type == "lndb":
        # ============================================
        # LNDb 資料集（專家標註，推薦）
        # ============================================
        if args.rad_id == "consensus":
            logger.info("🎭 使用多放射科醫師共識遮罩 (consensus mode)")
        else:
            logger.info(f"🎭 使用單一放射科醫師遮罩 (RadID={args.rad_id})")
        
        train_dataset = LNDbDataset(
            args.data_dir, 
            train_ids, 
            axis=args.axis, 
            transform=transform,
            cache_data=args.cache_data,
            rad_id=args.rad_id,
            min_nodule_diameter=args.min_nodule_diameter
        )
        val_dataset = LNDbDataset(
            args.data_dir, 
            val_ids, 
            axis=args.axis, 
            cache_data=args.cache_data,
            rad_id=args.rad_id,
            min_nodule_diameter=args.min_nodule_diameter
        )
        test_dataset = LNDbDataset(
            args.data_dir, 
            test_ids, 
            axis=args.axis, 
            cache_data=args.cache_data,
            rad_id=args.rad_id,
            min_nodule_diameter=args.min_nodule_diameter
        )
    else:
        # ============================================
        # LUNA16 資料集（需生成 GT）
        # ============================================
        logger.info(f"🎯 GT 分割方法: {args.segmentation_method}")
        if args.min_nodule_diameter > 0:
            logger.info(f"🔍 過濾小於 {args.min_nodule_diameter}mm 的結節")
        
        train_dataset = ChestTumorDataset(
            args.data_dir, 
            train_ids, 
            axis=args.axis, 
            transform=transform,
            cache_data=args.cache_data,
            segmentation_method=args.segmentation_method,
            min_nodule_diameter=args.min_nodule_diameter
        )
        val_dataset = ChestTumorDataset(
            args.data_dir, 
            val_ids, 
            axis=args.axis, 
            cache_data=args.cache_data,
            segmentation_method=args.segmentation_method,
            min_nodule_diameter=args.min_nodule_diameter
        )
        test_dataset = ChestTumorDataset(
            args.data_dir, 
            test_ids, 
            axis=args.axis, 
            cache_data=args.cache_data,
            segmentation_method=args.segmentation_method,
            min_nodule_diameter=args.min_nodule_diameter
        )
    
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False,
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
            spacing=(1.0, 1.0)  # 可以從 DICOM metadata 讀取
        )
        
        # 保存測試配置
        test_config = {
            'mode': 'test',
            'checkpoint': args.resume or args.checkpoint,
            'data_dir': args.data_dir,
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
        logger.info(f"  DSC (Dice Similarity Coefficient): {val_metrics['DSC']:.4f}")
        logger.info(f"  IoU (Intersection over Union): {val_metrics['IoU']:.4f}")
        logger.info(f"  SEN (Sensitivity): {val_metrics['SEN']:.4f}")
        logger.info(f"  PPV (Positive Predictive Value): {val_metrics['PPV']:.4f}")
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
        logger.info(f"  DSC (Dice Similarity Coefficient): {test_metrics['DSC']:.4f}")
        logger.info(f"  IoU (Intersection over Union): {test_metrics['IoU']:.4f}")
        logger.info(f"  SEN (Sensitivity): {test_metrics['SEN']:.4f}")
        logger.info(f"  PPV (Positive Predictive Value): {test_metrics['PPV']:.4f}")
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
    logger.info(f"  DSC (Dice Similarity Coefficient): {test_metrics['DSC']:.4f}")
    logger.info(f"  IoU (Intersection over Union): {test_metrics['IoU']:.4f}")
    logger.info(f"  SEN (Sensitivity): {test_metrics['SEN']:.4f}")
    logger.info(f"  PPV (Positive Predictive Value): {test_metrics['PPV']:.4f}")
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
