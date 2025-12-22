#!/usr/bin/env python3
"""
UNet++ 肺結節分割訓練 - 主程式
===================================

使用範例:
    # 預處理資料集
    python -m segmentation.train_unetpp.main --preprocess
    
    # 單次訓練（預設）
    python -m segmentation.train_unetpp.main --epochs 100
    
    # 5-fold CV 訓練
    python -m segmentation.train_unetpp.main --cv --epochs 100
    
    # 只訓練特定 fold
    python -m segmentation.train_unetpp.main --cv --fold 0 --epochs 100
    
    # 快速測試
    python -m segmentation.train_unetpp.main --data_fraction 0.1 --epochs 5
    
    # 推論
    python -m segmentation.train_unetpp.main --inference --model_path path/to/model.pth
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

# 支援直接執行和模組執行
try:
    from .config import Config, get_default_config
    from .preprocess import preprocess_lndb_dataset, preprocess_lndb_slices
    from .dataset import LNDbDataset, LNDbSliceDataset, LNDbInferenceDataset, get_patient_split, get_fold_split
    from .model import get_model, count_parameters
    from .trainer import UNetPPTrainer
    from .inference import Inferencer, load_model_for_inference
    from .utils import setup_logging, set_seed, get_device, plot_training_history, custom_collate_fn
except ImportError:
    # 直接執行時，添加父目錄到路徑
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train_unetpp.config import Config, get_default_config
    from train_unetpp.preprocess import preprocess_lndb_dataset, preprocess_lndb_slices
    from train_unetpp.dataset import LNDbDataset, LNDbSliceDataset, LNDbInferenceDataset, get_patient_split, get_fold_split, val_collate_fn
    from train_unetpp.model import get_model, count_parameters
    from train_unetpp.trainer import UNetPPTrainer
    from train_unetpp.inference import Inferencer, load_model_for_inference
    from train_unetpp.utils import setup_logging, set_seed, get_device, plot_training_history, custom_collate_fn


logger = logging.getLogger(__name__)


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="UNet++ 肺結節分割訓練",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 模式選擇
    parser.add_argument('--preprocess', action='store_true', help='執行資料預處理（3D 快取）')
    parser.add_argument('--preprocess-slices', action='store_true', dest='preprocess_slices', 
                        help='執行切片式預處理（2D 快取，加速訓練）')
    parser.add_argument('--inference', action='store_true', help='執行推論模式')
    
    # CV 選項
    parser.add_argument('--cv', action='store_true', help='啟用 5-fold CV')
    parser.add_argument('--fold', type=int, default=None, help='指定訓練的 fold（需配合 --cv）')
    
    # 路徑
    parser.add_argument('--data_dir', type=str, default=None, help='資料集目錄')
    parser.add_argument('--output_dir', type=str, default=None, help='輸出目錄')
    parser.add_argument('--model_path', type=str, default=None, help='模型路徑（用於推論或恢復訓練）')
    
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


def update_config_from_args(config: Config, args) -> Config:
    """根據命令列參數更新配置"""
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.output_dir:
        config.data.output_dir = args.output_dir
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.patch_size:
        config.data.patch_size = args.patch_size
    if args.encoder:
        config.model.encoder_name = args.encoder
    
    config.seed = args.seed
    config.num_workers = args.num_workers
    config.device = args.device
    
    # CV 設定
    config.training.use_cv = args.cv
    config.training.cv_fold = args.fold
    
    return config


def run_preprocessing(config: Config):
    """執行資料預處理"""
    logger.info("開始資料預處理...")
    
    preprocess_lndb_dataset(
        config.data.data_dir,
        config.data.cache_dir,
        config.data.target_spacing
    )
    
    logger.info("預處理完成！")


def run_preprocessing_slices(config: Config):
    """執行切片式資料預處理（每個切片獨立保存，加速 2D 訓練）"""
    logger.info("開始切片式資料預處理...")
    
    # 使用切片式快取目錄
    slice_cache_dir = str(Path(config.data.cache_dir).parent / "lndb_slices")
    
    preprocess_lndb_slices(
        config.data.data_dir,
        slice_cache_dir,
        config.data.target_spacing
    )
    
    logger.info(f"切片式預處理完成！輸出目錄: {slice_cache_dir}")


def run_single_training(config: Config, train_ids: list, val_ids: list, test_ids: list = None, run_dir: Path = None):
    """執行單次訓練"""
    test_count = len(test_ids) if test_ids else 0
    logger.info(f"訓練集: {len(train_ids)} 病人, 驗證集: {len(val_ids)} 病人, 測試集: {test_count} 病人")
    
    # 切片式快取目錄
    slice_cache_dir = str(Path(config.data.cache_dir).parent / "lndb_slices")
    
    # 檢查是否有切片式快取
    if Path(slice_cache_dir).exists():
        logger.info(f"使用切片式資料集 (快取目錄: {slice_cache_dir})")
        
        train_dataset = LNDbSliceDataset(
            slice_cache_dir,
            train_ids,
            config,
            mode="train"
        )
        
        val_dataset = LNDbSliceDataset(
            slice_cache_dir,
            val_ids,
            config,
            mode="val"
        )
    else:
        # 降級使用舊的 3D 快取
        logger.warning("切片式快取不存在，使用 3D 快取（較慢）")
        logger.info("建立訓練資料集...")
        train_dataset = LNDbDataset(
            config.data.data_dir,
            train_ids,
            config,
            mode="train",
            preload=False
        )
        
        logger.info("建立驗證資料集...")
        val_dataset = LNDbDataset(
            config.data.data_dir,
            val_ids,
            config,
            mode="val",
            preload=False
        )
    
    # 創建 DataLoader
    # 啟用多進程載入加速磁碟讀取
    actual_workers = min(4, config.num_workers)  # Windows 建議用 4 個 workers
    logger.info(f"DataLoader workers: {actual_workers}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=actual_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn,
        persistent_workers=False if actual_workers == 0 else True
    )
    
    # Val/Test 使用自定義 collate_fn 處理 4-patch 格式
    # batch_size=1 因為 full_shape 可能不同
    # pin_memory=False 避免 full_mask 消耗過多 GPU 記憶體
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 必須 1 因為 full_shape 可能不同
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=False,  # 避免 OOM
        collate_fn=val_collate_fn,
        persistent_workers=False if actual_workers == 0 else True
    )
    
    # 創建訓練器（傳遞 data split 資訊和輸出目錄）
    data_split = {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids or []
    }
    trainer = UNetPPTrainer(config, data_split=data_split, output_dir=run_dir)
    
    # 訓練
    history = trainer.fit(train_loader, val_loader)
    
    # 繪製最終訓練曲線
    plot_training_history(
        history,
        str(trainer.output_dir / "training_curves_final.png")
    )
    
    return trainer, history


def run_cv_training(config: Config, args):
    """執行 5-fold CV 訓練"""
    logger.info("開始 5-fold CV 訓練...")
    
    fold_range = [args.fold] if args.fold is not None else range(config.training.num_folds)
    all_results = []
    
    for fold_id in fold_range:
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold_id + 1}/{config.training.num_folds}")
        logger.info(f"{'='*50}")
        
        # 獲取分割
        train_ids, val_ids = get_fold_split(config.data.data_dir, fold_id)
        
        # 應用資料比例
        if args.data_fraction < 1.0:
            n_train = int(len(train_ids) * args.data_fraction)
            n_val = int(len(val_ids) * args.data_fraction)
            train_ids = train_ids[:max(1, n_train)]
            val_ids = val_ids[:max(1, n_val)]
        
        # 更新配置的實驗名稱
        config.experiment_name = f"unetpp_lndb_fold{fold_id}"
        
        # 訓練
        trainer, history = run_single_training(config, train_ids, val_ids)
        
        # 記錄結果
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
        
        # 保存匯總結果
        summary_path = Path(config.data.output_dir) / "cv_summary.json"
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


def run_inference(config: Config, args):
    """執行推論"""
    if not args.model_path:
        raise ValueError("推論模式需要指定 --model_path")
    
    logger.info(f"載入模型: {args.model_path}")
    
    # 載入模型
    model = load_model_for_inference(args.model_path, config)
    
    # 創建推論器
    inferencer = Inferencer(model, config)
    
    # 獲取測試集
    _, _, test_ids = get_patient_split(config.data.data_dir, seed=config.seed)
    
    if args.data_fraction < 1.0:
        n_test = int(len(test_ids) * args.data_fraction)
        test_ids = test_ids[:max(1, n_test)]
    
    logger.info(f"推論 {len(test_ids)} 個病人")
    
    # 推論
    output_dir = Path(config.data.output_dir) / "inference_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from .preprocess import CTPreprocessor
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
    
    # 保存匯總
    summary_path = output_dir / "inference_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"推論完成！結果已保存: {output_dir}")


def main():
    """主函數"""
    args = parse_args()
    
    # 設定
    config = get_default_config()
    config = update_config_from_args(config, args)
    
    # 日誌 - 先建立一個暫時的 log，訓練開始後會移動到正確位置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.data.output_dir) / f"unetpp_lndb_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(run_dir / "train.log"))
    
    # 設定隨機種子
    set_seed(config.seed)
    
    # 設備
    device = get_device(config.device)
    logger.info(f"使用設備: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 執行
    if args.preprocess:
        run_preprocessing(config)
    elif hasattr(args, 'preprocess_slices') and args.preprocess_slices:
        run_preprocessing_slices(config)
    elif args.inference:
        run_inference(config, args)
    elif args.cv:
        run_cv_training(config, args)
    else:
        # 單次訓練
        train_ids, val_ids, test_ids = get_patient_split(
            config.data.data_dir,
            config.data.train_ratio,
            config.data.val_ratio,
            config.data.test_ratio,
            config.seed
        )
        
        # 應用資料比例
        if args.data_fraction < 1.0:
            n_train = int(len(train_ids) * args.data_fraction)
            n_val = int(len(val_ids) * args.data_fraction)
            train_ids = train_ids[:max(1, n_train)]
            val_ids = val_ids[:max(1, n_val)]
        
        run_single_training(config, train_ids, val_ids, test_ids, run_dir=run_dir)


if __name__ == "__main__":
    main()
