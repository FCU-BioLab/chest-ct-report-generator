#!/usr/bin/env python3
"""
UNet++ 肺病灶分割訓練 - 統一主程式
===================================

支援資料集:
    - LNDb: 肺結節分割 (236 病人)
    - MSD: 肺腫瘤分割 (Task06, 64+32 案例)

使用範例:
    # ===== LNDb 資料集 =====
    # 預處理
    python main.py --dataset lndb --preprocess
    
    # 訓練
    python main.py --dataset lndb --epochs 100
    
    # 5-fold CV
    python main.py --dataset lndb --cv --epochs 100
    
    # 推論
    python main.py --dataset lndb --inference --model_path path/to/model.pth
    
    # ===== MSD Lung 資料集 =====
    # 預處理
    python main.py --dataset msd --preprocess
    
    # 訓練
    python main.py --dataset msd --epochs 100
    
    # 只執行測試
    python main.py --dataset msd --test --model_path path/to/model.pth
    
    # 視覺化資料
    python main.py --dataset msd --visualize
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

# 確保可以導入本地模組
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_unetpp.config import Config, get_default_config
from train_unetpp.model import get_model, count_parameters
from train_unetpp.trainer import UNetPPTrainer
from train_unetpp.utils import setup_logging, set_seed, get_device, plot_training_history, custom_collate_fn

logger = logging.getLogger(__name__)


# =============================================================================
# 命令列參數
# =============================================================================

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="UNet++ 肺病灶分割訓練（支援 LNDb 與 MSD）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 資料集選擇
    parser.add_argument('--dataset', type=str, default='lndb', 
                        choices=['lndb', 'msd'],
                        help='資料集類型: lndb (肺結節) 或 msd (肺腫瘤)')
    
    # 模式選擇
    parser.add_argument('--preprocess', action='store_true', help='執行資料預處理')
    parser.add_argument('--inference', action='store_true', help='執行推論模式')
    parser.add_argument('--test', action='store_true', help='只執行測試評估（需配合 --model_path）')
    parser.add_argument('--visualize', action='store_true', help='視覺化資料集樣本')
    
    # CV 選項 (僅 LNDb)
    parser.add_argument('--cv', action='store_true', help='啟用 5-fold CV (僅 LNDb)')
    parser.add_argument('--fold', type=int, default=None, help='指定訓練的 fold（需配合 --cv）')
    
    # 路徑
    parser.add_argument('--data_dir', type=str, default=None, help='資料集目錄')
    parser.add_argument('--output_dir', type=str, default=None, help='輸出目錄')
    parser.add_argument('--model_path', type=str, default=None, help='模型路徑（用於推論/測試/恢復訓練）')
    
    # 訓練參數
    parser.add_argument('--epochs', type=int, default=None, help='訓練 epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='學習率')
    parser.add_argument('--patch_size', type=int, default=None, help='Patch 大小')

    # 模型參數
    parser.add_argument('--encoder', type=str, default=None, help='編碼器名稱')
    
    # 其他
    parser.add_argument('--data_fraction', type=float, default=1.0, help='使用的資料比例（用於快速測試）')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda', help='設備')
    
    return parser.parse_args()


def create_config(args) -> Config:
    """根據命令列參數建立配置"""
    config = get_default_config()
    
    # 只有 args 有指定時才覆蓋 config
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.patch_size is not None:
        config.data.patch_size = args.patch_size
    if args.encoder is not None:
        config.model.encoder_name = args.encoder
    # in_channels 使用 config 預設值 (5 for 2.5D: z-2, z-1, z, z+1, z+2)
    if args.seed is not None:
        config.seed = args.seed
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.device is not None:
        config.device = args.device

    # CV 設定
    config.training.use_cv = args.cv
    config.training.cv_fold = args.fold

    # 覆蓋路徑（如果有指定）
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.output_dir:
        config.data.output_dir = args.output_dir

    return config


# =============================================================================
# LNDb 資料集相關函數
# =============================================================================

def run_lndb_preprocess(config: Config):
    """執行 LNDb 切片式預處理"""
    from train_unetpp.preprocess import preprocess_lndb_slices
    
    slice_cache_dir = str(Path(config.data.cache_dir).parent / "lndb_slices")
    
    logger.info("開始 LNDb 切片式資料預處理...")
    logger.info(f"輸出目錄: {slice_cache_dir}")
    
    preprocess_lndb_slices(
        config.data.data_dir,
        slice_cache_dir,
        config.data.target_spacing
    )
    
    logger.info("LNDb 預處理完成！")


def run_lndb_training(config: Config, args, run_dir: Path):
    """執行 LNDb 訓練"""
    from train_unetpp.dataset import CachedPatchDataset, get_cached_patch_split, get_patient_split
    
    # 使用 config 中設定的 cache_dir (cache/lndb_patches)
    cache_dir = config.data.cache_dir
    
    if not Path(cache_dir).exists():
        logger.error(f"Patch 快取不存在: {cache_dir}")
        logger.error("請先執行: python train_unetpp/preprocess.py --mode 4patch")
        return None, None
    
    # 資料分割
    # If caching is disabled, we might not have the cache dir structure for get_cached_patch_split
    # So we use the generic get_patient_split if needed
    if config.data.cache_preprocessed:
        train_ids, val_ids, test_ids = get_cached_patch_split(
            cache_dir,
            config.data.train_ratio,
            config.data.val_ratio,
            config.seed
        )
    else:
        # Using slices directly, use the original split logic
        # Data dir is the RAW data dir, but get_patient_split handles looking into cache too
        train_ids, val_ids, test_ids = get_patient_split(
            config.data.data_dir,
            config.data.train_ratio,
            config.data.val_ratio,
            config.data.test_ratio, # Added test ratio
            config.seed
        )
    
    # 應用資料比例
    if args.data_fraction < 1.0:
        n_train = int(len(train_ids) * args.data_fraction)
        n_val = int(len(val_ids) * args.data_fraction)
        train_ids = train_ids[:max(1, n_train)]
        val_ids = val_ids[:max(1, n_val)]
    
    logger.info(f"訓練集: {len(train_ids)} 病人, 驗證集: {len(val_ids)} 病人, 測試集: {len(test_ids)} 病人")
    
    # 建立資料集
    # Phase 2 Upgrade: Dynamic vs Cached
    if config.data.cache_preprocessed:
        logger.info("Using CACHED Patch Dataset (Fixed Size)")
        train_dataset = CachedPatchDataset(cache_dir, train_ids, config, mode="train")
        val_dataset = CachedPatchDataset(cache_dir, val_ids, config, mode="val")
    else:
        logger.info("Using DYNAMIC Slice Dataset (On-the-fly Cropping for 352x352)")
        # Point to the SLICE cache (cache/lndb_slices)
        slice_cache_dir = Path(config.data.cache_dir).parent / "lndb_slices"
        logger.info(f"Loading slices from: {slice_cache_dir}")
        
        # We need to import LNDbSliceDataset inside the function to avoid circular imports if any
        from train_unetpp.dataset import LNDbSliceDataset
        
        train_dataset = LNDbSliceDataset(slice_cache_dir, train_ids, config, mode="train")
        val_dataset = LNDbSliceDataset(slice_cache_dir, val_ids, config, mode="val")
    
    # DataLoader
    actual_workers = min(4, config.num_workers)
    
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
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=actual_workers > 0
    )
    
    # Trainer
    data_split = {'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids}
    trainer = UNetPPTrainer(config, data_split=data_split, output_dir=run_dir)
    
    # Check for resume (Load model weights if provided in args.model_path)
    if args.model_path and Path(args.model_path).exists():
        logger.info(f"Resuming training from checkpoint: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=config.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        # Consider loading optimizer state if needed, but for now just weights is a good start
        # trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 訓練
    history = trainer.fit(train_loader, val_loader)
    
    # 匯出訓練歷史
    plot_training_history(history, str(trainer.output_dir / "training_curves_final.png"))
    
    # === 新增：Test Set 評估 ===
    if len(test_ids) > 0:
        logger.info(f"\n{'='*50}")
        logger.info(f"開始 LNDb Test Set 評估... ({len(test_ids)} 病人)")
        
        # 建立 Test DataLoader
        test_dataset = CachedPatchDataset(cache_dir, test_ids, config, mode="val") # Test 使用 val 模式( deterministic)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=actual_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
            persistent_workers=actual_workers > 0
        )
        
        # 載入最佳模型
        best_model_path = run_dir / "best_model.pth"
        if best_model_path.exists():
            logger.info(f"載入最佳模型: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=config.device, weights_only=False)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.warning("找不到最佳模型，使用目前模型進行測試")
            
        # 執行評估
        test_metrics = trainer.validate(test_loader, epoch=None, save_samples=False)
        
        # 保存 Test 結果
        test_results = {
            'test_ids': test_ids,
            'num_test_patients': len(test_ids),
            'num_test_patches': len(test_dataset),
            'metrics': {
                'dice': float(test_metrics['dice']),
                'iou': float(test_metrics['iou']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall']),
                'pp_dice': float(test_metrics['pp_dice']),
                'pp_iou': float(test_metrics['pp_iou']),
                'pp_precision': float(test_metrics['pp_precision']),
                'pp_recall': float(test_metrics['pp_recall']),
            }
        }
        
        test_metrics_path = run_dir / "test_metrics.json"
        with open(test_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
            
        logger.info("=" * 50)
        logger.info("LNDb Test 評估結果:")
        logger.info(f"  Raw Dice: {test_metrics['dice']:.4f}, IoU: {test_metrics['iou']:.4f}")
        logger.info(f"  PP  Dice: {test_metrics['pp_dice']:.4f}, IoU: {test_metrics['pp_iou']:.4f}")
        logger.info(f"Test 結果已保存: {test_metrics_path}")
        logger.info("=" * 50)
    
    return trainer, history


def run_lndb_test_only(config: Config, args, run_dir: Path):
    """LNDb 只執行測試評估（不訓練）"""
    from train_unetpp.dataset import CachedPatchDataset, get_cached_patch_split
    
    if not args.model_path:
        logger.error("請指定 --model_path")
        return

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"模型不存在: {model_path}")
        return
        
    # 資料準備
    cache_dir = config.data.cache_dir
    if not Path(cache_dir).exists():
        # Fallback
        project_root = Path(__file__).parent.parent
        cache_dir = str(project_root / "segmentation" / "cache" / "lndb_patches")
        
    _, _, test_ids = get_cached_patch_split(
        cache_dir,
        config.data.train_ratio,
        config.data.val_ratio,
        config.seed
    )
    
    if args.data_fraction < 1.0:
        test_ids = test_ids[:max(1, int(len(test_ids) * args.data_fraction))]
        
    logger.info(f"開始 LNDb Test Set 評估... ({len(test_ids)} 病人)")
    
    # 建立 DataLoader
    test_dataset = CachedPatchDataset(cache_dir, test_ids, config, mode="val")
    actual_workers = min(4, config.num_workers)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=actual_workers > 0
    )
    
    # 初始化 Trainer (用於使用其 validate 方法)
    # 只需要 output_dir 和 config
    trainer = UNetPPTrainer(config, output_dir=run_dir)
    
    # 載入模型
    logger.info(f"載入模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    # 執行評估
    test_metrics = trainer.validate(test_loader, epoch=None, save_samples=False)
    
    # 保存結果
    test_results = {
        'model_path': str(model_path),
        'test_ids': test_ids,
        'metrics': {
            'dice': float(test_metrics['dice']),
            'iou': float(test_metrics['iou']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'pp_dice': float(test_metrics['pp_dice']),
            'pp_iou': float(test_metrics['pp_iou']),
            'pp_precision': float(test_metrics['pp_precision']),
            'pp_recall': float(test_metrics['pp_recall']),
        }
    }
    
    test_metrics_path = run_dir / "test_only_metrics.json"
    with open(test_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
        
    logger.info("=" * 50)
    logger.info("LNDb Test 評估結果:")
    logger.info(f"  Raw Dice: {test_metrics['dice']:.4f}, IoU: {test_metrics['iou']:.4f}")
    logger.info(f"  PP  Dice: {test_metrics['pp_dice']:.4f}, IoU: {test_metrics['pp_iou']:.4f}")
    logger.info(f"結果已保存: {test_metrics_path}")
    logger.info("=" * 50)


def run_lndb_cv_training(config: Config, args, run_dir: Path):
    """執行 LNDb 5-fold CV 訓練"""
    logger.info("開始 5-fold CV 訓練...")
    
    fold_range = [args.fold] if args.fold is not None else range(config.training.num_folds)
    all_results = []
    
    for fold_id in fold_range:
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold_id + 1}/{config.training.num_folds}")
        logger.info(f"{'='*50}")
        
        args.fold = fold_id
        fold_dir = run_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        trainer, history = run_lndb_training(config, args, fold_dir)
        
        if history:
            best_dice = max(history['val_dice'])
            best_iou = max(history['val_iou'])
            
            all_results.append({
                'fold': fold_id,
                'best_dice': best_dice,
                'best_iou': best_iou,
                'output_dir': str(trainer.output_dir)
            })
            
            logger.info(f"Fold {fold_id}: Best Dice = {best_dice:.4f}, Best IoU = {best_iou:.4f}")
    
    # 匯總結果
    if len(all_results) > 1:
        logger.info(f"\n{'='*50}")
        logger.info("CV 結果匯總")
        logger.info(f"{'='*50}")
        
        dices = [r['best_dice'] for r in all_results]
        ious = [r['best_iou'] for r in all_results]
        
        logger.info(f"Mean Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
        logger.info(f"Mean IoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}")
        
        summary_path = run_dir / "cv_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'folds': all_results,
                'mean_dice': float(np.mean(dices)),
                'std_dice': float(np.std(dices)),
                'mean_iou': float(np.mean(ious)),
                'std_iou': float(np.std(ious))
            }, f, indent=2)
        
        logger.info(f"CV 結果已保存: {summary_path}")
    
    return all_results


def run_lndb_inference(config: Config, args):
    """執行 LNDb 推論"""
    if not args.model_path:
        raise ValueError("推論模式需要指定 --model_path")
    
    from train_unetpp.inference import Inferencer, load_model_for_inference
    from train_unetpp.preprocess import CTPreprocessor
    from train_unetpp.dataset import get_patient_split
    
    logger.info(f"載入模型: {args.model_path}")
    
    model = load_model_for_inference(args.model_path, config)
    inferencer = Inferencer(model, config)
    
    _, _, test_ids = get_patient_split(config.data.data_dir, seed=config.seed)
    
    if args.data_fraction < 1.0:
        test_ids = test_ids[:max(1, int(len(test_ids) * args.data_fraction))]
    
    logger.info(f"推論 {len(test_ids)} 個病人")
    
    output_dir = Path(config.data.output_dir) / "inference_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor = CTPreprocessor(cache_dir=config.data.cache_dir)
    
    all_results = []
    for patient_id in test_ids:
        cache_path = Path(config.data.cache_dir) / f"{patient_id}.npz"
        if not cache_path.exists():
            logger.warning(f"跳過不存在的病人: {patient_id}")
            continue
        
        data = preprocessor.load_preprocessed(str(cache_path))
        data['patient_id'] = patient_id
        
        result = inferencer.run_inference(data, str(output_dir))
        
        all_results.append({
            'patient_id': patient_id,
            'num_nodules': len(result['nodules']),
            'nodules': result['nodules']
        })
        
        logger.info(f"{patient_id}: 找到 {len(result['nodules'])} 個結節")
    
    summary_path = output_dir / "inference_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"推論完成！結果已保存: {output_dir}")


# =============================================================================
# MSD 資料集相關函數
# =============================================================================

def run_msd_preprocess():
    """執行 MSD 預處理"""
    from train_unetpp.msd_dataset import MSD_LUNG_DIR, MSD_CACHE_DIR, preprocess_msd_lung
    
    logger.info("執行 MSD Lung Tumours 預處理...")
    preprocess_msd_lung(MSD_LUNG_DIR, MSD_CACHE_DIR)
    logger.info("MSD 預處理完成！")


def run_msd_training(config: Config, args, run_dir: Path):
    """執行 MSD 訓練"""
    from train_unetpp.msd_dataset import (
        MSD_LUNG_DIR, MSD_CACHE_DIR,
        get_msd_lung_cases, get_msd_train_val_split,
        MSDLungSliceDataset
    )
    
    device = get_device(args.device)
    
    # 取得案例並分割
    cases = get_msd_lung_cases(MSD_LUNG_DIR)
    logger.info(f"找到 {len(cases)} 個案例")
    
    train_ids, val_ids, test_ids = get_msd_train_val_split(
        cases, val_ratio=0.15, test_ratio=0.15, seed=args.seed
    )
    logger.info(f"訓練集: {len(train_ids)} 案例, 驗證集: {len(val_ids)} 案例, 測試集: {len(test_ids)} 案例")
    
    # 檢查快取
    if not MSD_CACHE_DIR.exists() or len(list(MSD_CACHE_DIR.iterdir())) == 0:
        logger.error(f"快取目錄不存在或為空: {MSD_CACHE_DIR}")
        logger.error("請先執行 --preprocess")
        return None, None
    
    # 建立資料集
    train_dataset = MSDLungSliceDataset(
        case_ids=train_ids, cache_dir=MSD_CACHE_DIR,
        mode="train", patch_size=config.data.patch_size
    )
    val_dataset = MSDLungSliceDataset(
        case_ids=val_ids, cache_dir=MSD_CACHE_DIR,
        mode="val", patch_size=config.data.patch_size
    )
    
    # DataLoader
    actual_workers = min(4, args.num_workers)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size,
        shuffle=True, num_workers=actual_workers,
        pin_memory=True, collate_fn=custom_collate_fn,
        persistent_workers=actual_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size, shuffle=False,
        num_workers=actual_workers, pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=actual_workers > 0
    )
    
    # Trainer
    data_split = {'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids}
    trainer = UNetPPTrainer(config, data_split=data_split, output_dir=run_dir)
    
    # 訓練
    logger.info(f"開始訓練，共 {config.training.epochs} 個 epoch")
    history = trainer.fit(train_loader, val_loader)
    logger.info("訓練完成！")
    
    # Test 評估
    if len(test_ids) > 0:
        _run_msd_test_eval(trainer, test_ids, config, actual_workers, run_dir, device)
    
    return trainer, history


def _run_msd_test_eval(trainer, test_ids, config, actual_workers, run_dir, device):
    """執行 MSD 測試評估"""
    from train_unetpp.msd_dataset import MSD_CACHE_DIR, MSDLungSliceDataset
    from train_unetpp.utils import custom_collate_fn
    
    logger.info("=" * 50)
    logger.info(f"開始 Test 評估... ({len(test_ids)} 案例)")
    
    test_dataset = MSDLungSliceDataset(
        case_ids=test_ids, cache_dir=MSD_CACHE_DIR,
        mode="val", patch_size=config.data.patch_size
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=config.training.batch_size, shuffle=False,
        num_workers=actual_workers, pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=actual_workers > 0
    )
    
    # 載入最佳模型
    best_model_path = run_dir / "best_model.pth"
    if best_model_path.exists():
        logger.info(f"載入最佳模型: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    # 執行測試
    test_metrics = trainer.validate(test_loader, epoch=None, save_samples=False)
    
    # 保存結果
    test_results = {
        'test_ids': test_ids,
        'num_test_cases': len(test_ids),
        'num_test_slices': len(test_dataset),
        'metrics': {
            'dice': float(test_metrics['dice']),
            'iou': float(test_metrics['iou']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
        }
    }
    
    test_metrics_path = run_dir / "test_metrics.json"
    with open(test_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    logger.info("=" * 50)
    logger.info("Test 評估結果:")
    logger.info(f"  Dice: {test_metrics['dice']:.4f}, IoU: {test_metrics['iou']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test 結果已保存: {test_metrics_path}")
    logger.info("=" * 50)


def run_msd_test_only(config: Config, args, run_dir: Path):
    """MSD 只執行測試評估（不訓練）"""
    import matplotlib.pyplot as plt
    from train_unetpp.msd_dataset import (
        MSD_LUNG_DIR, MSD_CACHE_DIR,
        get_msd_lung_cases, get_msd_train_val_split,
        MSDLungSliceDataset
    )
    from train_unetpp.utils import custom_collate_fn
    from tqdm import tqdm
    
    if not args.model_path:
        logger.error("請指定 --model_path")
        return
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"模型不存在: {model_path}")
        return
    
    device = get_device(args.device)
    
    # 取得測試集
    cases = get_msd_lung_cases(MSD_LUNG_DIR)
    _, _, test_ids = get_msd_train_val_split(cases, val_ratio=0.15, test_ratio=0.15, seed=args.seed)
    
    logger.info(f"測試集: {len(test_ids)} 案例")
    
    if len(test_ids) == 0:
        logger.error("沒有測試集案例")
        return
    
    # 建立資料集
    test_dataset = MSDLungSliceDataset(
        case_ids=test_ids, cache_dir=MSD_CACHE_DIR,
        mode="val", patch_size=config.data.patch_size
    )
    
    actual_workers = min(4, args.num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=config.training.batch_size, shuffle=False,
        num_workers=actual_workers, pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=actual_workers > 0
    )
    
    # 載入模型
    model = get_model(config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("=" * 50)
    logger.info("開始 Test 評估...")
    
    # 使用 patch-level 評估（和 validate() 相同邏輯）
    all_preds, all_targets, all_images, all_case_ids = [], [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            masks = batch['mask']
            
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            targets = masks.numpy()
            imgs = images[:, 1, :, :].cpu().numpy()  # 取中間 channel
            
            for i in range(images.shape[0]):
                all_preds.append(preds[i])
                all_targets.append(targets[i])
                all_images.append(imgs[i])
                
                id_key = 'patient_id' if 'patient_id' in batch else 'case_id'
                case_id = batch[id_key][i] if isinstance(batch[id_key], list) else str(batch[id_key][i])
                all_case_ids.append(case_id)
    
    # 計算指標
    total_intersection, total_pred_sum, total_target_sum, total_union = 0, 0, 0, 0
    
    for pred, target in zip(all_preds, all_targets):
        pred_binary = (pred > 0.5).astype(np.float32)
        target_binary = (target > 0.5).astype(np.float32)
        
        intersection = (pred_binary * target_binary).sum()
        total_intersection += intersection
        total_pred_sum += pred_binary.sum()
        total_target_sum += target_binary.sum()
        total_union += pred_binary.sum() + target_binary.sum() - intersection
    
    smooth = 1e-6
    dice = (2 * total_intersection + smooth) / (total_pred_sum + total_target_sum + smooth)
    iou = (total_intersection + smooth) / (total_union + smooth)
    precision = total_intersection / (total_pred_sum + smooth)
    recall = total_intersection / (total_target_sum + smooth)
    
    # 保存結果
    test_results = {
        'model_path': str(model_path),
        'test_ids': test_ids,
        'num_test_cases': len(test_ids),
        'num_test_patches': len(test_dataset),
        'metrics': {
            'dice': float(dice), 'iou': float(iou),
            'precision': float(precision), 'recall': float(recall),
        }
    }
    
    test_metrics_path = run_dir / "test_metrics.json"
    with open(test_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    # 視覺化
    test_vis_dir = run_dir / "test_samples"
    test_vis_dir.mkdir(parents=True, exist_ok=True)
    
    positive_indices = [i for i, t in enumerate(all_targets) if np.sum(t) > 0]
    selected = positive_indices[:8]
    
    if len(selected) > 0:
        n_samples = min(8, len(selected))
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(selected[:n_samples]):
            img = all_images[idx]
            target = all_targets[idx][0] if all_targets[idx].ndim == 3 else all_targets[idx]
            pred = all_preds[idx][0] if all_preds[idx].ndim == 3 else all_preds[idx]
            pred_binary = (pred > 0.5).astype(np.float32)
            case_id = all_case_ids[idx]
            
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f'{case_id}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(target, cmap='Reds', vmin=0, vmax=1)
            axes[i, 1].set_title(f'GT (area={target.sum():.0f})')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='Blues', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Pred (area={pred_binary.sum():.0f})')
            axes[i, 2].axis('off')
            
            overlay = np.zeros((*target.shape, 3))
            overlay[:, :, 0] = target
            overlay[:, :, 2] = pred_binary
            axes[i, 3].imshow(np.clip(overlay, 0, 1))
            axes[i, 3].set_title('Overlay (R=GT, B=Pred)')
            axes[i, 3].axis('off')
        
        plt.suptitle(f'Test Results - Dice: {dice:.4f}, IoU: {iou:.4f}', fontsize=14)
        plt.tight_layout()
        plt.savefig(test_vis_dir / "test_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info("=" * 50)
    logger.info("Test 評估結果:")
    logger.info(f"  Dice: {dice:.4f}, IoU: {iou:.4f}")
    logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
    logger.info(f"結果已保存: {test_metrics_path}")
    logger.info("=" * 50)


def run_msd_visualize(args, run_dir: Path):
    """
    視覺化實際進入訓練的資料格式
    
    顯示：
    - Train 模式：單一 patch (224x224)，lung mask 外為零
    - Val/Test 模式：4-patch 覆蓋圖 + 各 patch 細節
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from tqdm import tqdm
    from train_unetpp.msd_dataset import MSD_CACHE_DIR, get_msd_lung_cases, get_msd_train_val_split, MSDLungSliceDataset
    from train_unetpp.patch_utils import compute_4patch_positions, extract_patch_with_lung_mask, get_lung_bbox
    
    logger.info("=" * 60)
    logger.info("視覺化實際訓練資料格式")
    logger.info("=" * 60)
    
    cases = get_msd_lung_cases()
    train_ids, val_ids, test_ids = get_msd_train_val_split(cases, seed=args.seed)
    
    preview_dir = run_dir / "training_data_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    
    patch_size = args.patch_size
    
    # 選擇一些樣本來視覺化
    sample_cases = {
        'train': train_ids[:3] if len(train_ids) >= 3 else train_ids,
        'val': val_ids[:2] if len(val_ids) >= 2 else val_ids,
    }
    
    for split_name, case_ids in sample_cases.items():
        logger.info(f"\n處理 {split_name} 樣本...")
        split_dir = preview_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for case_id in tqdm(case_ids, desc=f"{split_name}"):
            case_dir = MSD_CACHE_DIR / case_id
            meta_path = case_dir / "meta.json"
            if not meta_path.exists():
                continue
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            # 選擇正樣本切片（最多 3 張）
            positive_slices = meta['positive_slices'][:3] if meta['positive_slices'] else [meta['num_slices'] // 2]
            
            for slice_idx in positive_slices:
                # 載入資料
                slice_path = case_dir / f"slice_{slice_idx:04d}.npz"
                if not slice_path.exists():
                    continue
                
                data = np.load(slice_path)
                image = data['image'].astype(np.float32)
                mask = data['mask'].astype(np.float32)
                lung_mask = data['lung_mask'].astype(np.float32)
                
                h, w = image.shape
                
                # 模擬 2.5D（這裡簡化為複製）
                image_2_5d = np.stack([image, image, image], axis=0)
                
                # ======= Val/Test 模式：4-patch 視覺化 =======
                fig = plt.figure(figsize=(20, 12))
                
                # 上半部：原圖 + 4-patch 位置標示
                ax1 = fig.add_subplot(2, 4, 1)
                ax1.imshow(image, cmap='gray', vmin=0, vmax=1)
                ax1.set_title(f'Original Image\n{case_id} z={slice_idx}')
                ax1.axis('off')
                
                ax2 = fig.add_subplot(2, 4, 2)
                ax2.imshow(lung_mask, cmap='Blues', vmin=0, vmax=1)
                ax2.set_title('Lung Mask')
                ax2.axis('off')
                
                ax3 = fig.add_subplot(2, 4, 3)
                ax3.imshow(mask, cmap='Reds', vmin=0, vmax=1)
                ax3.set_title(f'Tumor Mask\n(area={mask.sum():.0f})')
                ax3.axis('off')
                
                # 4-patch 位置標示
                ax4 = fig.add_subplot(2, 4, 4)
                
                # 創建 overlay
                overlay = np.stack([image, image, image], axis=-1)
                overlay = np.clip(overlay, 0, 1)
                
                # Lung mask 邊界
                y_min, y_max, x_min, x_max = get_lung_bbox(lung_mask)
                
                # 計算 4-patch 位置
                patch_positions = compute_4patch_positions(lung_mask, patch_size)
                colors = ['red', 'green', 'blue', 'orange']
                patch_names = ['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']
                
                ax4.imshow(overlay)
                
                # 繪製 lung bbox
                rect_lung = mpatches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                               linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
                ax4.add_patch(rect_lung)
                
                # 繪製 4 個 patch 位置
                for i, ((py1, px1), (py2, px2)) in enumerate(patch_positions):
                    rect = mpatches.Rectangle((px1, py1), patch_size, patch_size,
                                             linewidth=2, edgecolor=colors[i], facecolor='none')
                    ax4.add_patch(rect)
                
                ax4.set_title('4-Patch Positions\n(cyan=lung bbox)')
                ax4.axis('off')
                
                # 下半部：4 個 extracted patches
                for i, ((py1, px1), (py2, px2)) in enumerate(patch_positions):
                    ax = fig.add_subplot(2, 4, 5 + i)
                    
                    # 提取 patch（含 lung mask zeroing）
                    patch_pos = ((py1, px1), (py2, px2))
                    patch_img, patch_mask, patch_lung = extract_patch_with_lung_mask(
                        image_2_5d, mask, lung_mask, patch_pos, patch_size
                    )
                    
                    # 取中間 channel 顯示
                    patch_2d = patch_img[1] if patch_img.ndim == 3 else patch_img
                    
                    # 創建 overlay
                    patch_overlay = np.stack([patch_2d, patch_2d, patch_2d], axis=-1)
                    patch_overlay = np.clip(patch_overlay, 0, 1)
                    
                    # 標記腫瘤區域
                    patch_overlay[patch_mask > 0.5, 0] = 1.0
                    patch_overlay[patch_mask > 0.5, 1] = 0.0
                    patch_overlay[patch_mask > 0.5, 2] = 0.0
                    
                    ax.imshow(patch_overlay)
                    tumor_area = patch_mask.sum()
                    ax.set_title(f'{patch_names[i]}\npos=[{py1},{px1}]\ntumor={tumor_area:.0f}px', 
                                fontsize=9, color=colors[i])
                    ax.axis('off')
                
                plt.suptitle(f'[{split_name.upper()}] Actual Training Data Format (4-Patch)\n'
                           f'Patch Size: {patch_size}x{patch_size}, Lung mask outside = 0',
                           fontsize=12, fontweight='bold')
                plt.tight_layout()
                
                save_path = split_dir / f"{case_id}_z{slice_idx:04d}_4patch.png"
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close()
    
    logger.info(f"\n視覺化完成！輸出目錄: {preview_dir}")
    logger.info("=" * 60)


# =============================================================================
# 主程式
# =============================================================================

def main():
    """主函數"""
    args = parse_args()
    config = create_config(args)
    
    # 設定輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = args.dataset.upper()
    
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        if args.preprocess:
            run_dir = Path(config.data.output_dir) / f"{dataset_name}_preprocess_{timestamp}"
        else:
            run_dir = Path(config.data.output_dir) / f"unetpp_{args.dataset}_{timestamp}"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 設置日誌
    log_file = run_dir / "train.log" if not args.preprocess else None
    setup_logging(str(log_file) if log_file else None)
    
    # 設定隨機種子
    set_seed(config.seed)
    
    # 設備
    device = get_device(config.device)
    logger.info(f"使用設備: {device}")
    logger.info(f"資料集: {dataset_name}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 根據資料集和模式執行
    if args.dataset == 'lndb':
        if args.preprocess:
            run_lndb_preprocess(config)
        elif args.inference:
            run_lndb_inference(config, args)
        elif args.cv:
            run_lndb_cv_training(config, args, run_dir)
        elif args.test:
             run_lndb_test_only(config, args, run_dir)
        else:
            run_lndb_training(config, args, run_dir)
    
    elif args.dataset == 'msd':
        if args.preprocess:
            run_msd_preprocess()
        elif args.test:
            run_msd_test_only(config, args, run_dir)
        elif args.visualize:
            run_msd_visualize(args, run_dir)
        else:
            run_msd_training(config, args, run_dir)


if __name__ == "__main__":
    main()
