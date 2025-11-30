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

使用範例:
    # 開始訓練
    python finetune_medsam2/main.py --epochs 50 --batch_size 4 --lr 1e-5
    
    # 從 checkpoint 繼續訓練
    python finetune_medsam2/main.py --resume finetune_output/best_model.pth
    
    # 只評估模型
    python finetune_medsam2/main.py --eval_only --checkpoint finetune_output/best_model.pth
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
from finetune_medsam2.dataset import ChestTumorDataset, DataAugmentation
from finetune_medsam2.trainer import MedSAM2Trainer
from finetune_medsam2.utils import (
    setup_logging, 
    suppress_noisy_logs, 
    split_dataset,
    custom_collate_fn,
    save_dataset_split_info,
    PatientMetricsTracker
)


def main():
    """主函數"""
    # ⚠️ 先解析參數以獲取輸出目錄，然後再初始化 logger
    # 這樣 log 檔案會保存到正確的位置
    
    # 命令列參數解析
    parser = argparse.ArgumentParser(
        description="Fine-tune MedSAM2 for Chest Tumor Segmentation"
    )
    
    # 資料參數
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="../datasets/aLL_patients_data",
        help="患者資料目錄 (LUNA16 root)"
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
        default=4,
        help="批次大小"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-5,
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
        default=7,
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
        default="MedSAM2/checkpoints/MedSAM2_latest.pt",
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
    
    args = parser.parse_args()
    
    # ✅ 生成時間戳記輸出目錄
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_result_dir = Path(r"E:\GitHub\chest-ct-report-generator\medsam2_segmentation\result")
        args.output_dir = str(base_result_dir / f"segmentation_{timestamp}")
    
    # 建立輸出目錄
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ✅ 現在初始化 logger（log 檔案會保存到輸出目錄）
    logger = setup_logging(log_dir=args.output_dir)
    suppress_noisy_logs()
    
    logger.info(f"📁 輸出目錄: {args.output_dir}")
    
    # 設定隨機種子（如果有提供）
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"🎲 設定隨機種子: {args.seed}")
    else:
        logger.info(f"🎲 未設定隨機種子，訓練過程將使用隨機初始化")
    
    # 檢查 CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️ 使用設備: {device}")
    
    # 獲取所有患者 ID
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"❌ 資料目錄不存在: {data_dir}")
        sys.exit(1)
    
    # LUNA16: 掃描 subset 資料夾中的 .mhd 檔案
    all_patients = []
    for i in range(10):
        subset_dir = data_dir / f"subset{i}"
        if subset_dir.exists():
            for f in subset_dir.glob("*.mhd"):
                all_patients.append(f.stem)
    
    # 去除重複並排序
    all_patients = sorted(list(set(all_patients)))
    
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
        logger.info(f"✂️ 依比例 {args.data_fraction} 縮減資料集: 剩餘 {len(all_patients)} 個患者")

    logger.info(f"📊 找到 {len(all_patients)} 個患者 (LUNA16)")
    
    if len(all_patients) == 0:
        logger.error(f"❌ 資料目錄為空或未找到 .mhd 檔案: {data_dir}")
        sys.exit(1)
    
    # 分割資料集
    train_ids, val_ids, test_ids = split_dataset(all_patients, seed=args.seed)
    logger.info(
        f"📊 資料集分割: Train={len(train_ids)}, "
        f"Val={len(val_ids)}, Test={len(test_ids)}"
    )
    
    # ✅ 新增：保存資料集分割資訊
    save_dataset_split_info(train_ids, val_ids, test_ids, args.output_dir)
    
    # 建立資料增強
    transform = None
    if args.augmentation:
        transform = DataAugmentation(
            rotation_prob=0.5,
            flip_prob=0.5,
            gamma_prob=0.3
        )
        logger.info("🔄 已啟用資料增強")
    
    # 建立資料集
    logger.info("🔧 建立資料集...")
    train_dataset = ChestTumorDataset(
        args.data_dir, 
        train_ids, 
        axis=args.axis, 
        transform=transform,
        cache_data=args.cache_data
    )
    val_dataset = ChestTumorDataset(
        args.data_dir, 
        val_ids, 
        axis=args.axis, 
        cache_data=args.cache_data
    )
    test_dataset = ChestTumorDataset(
        args.data_dir, 
        test_ids, 
        axis=args.axis, 
        cache_data=args.cache_data
    )
    
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
        output_dir=args.output_dir
    )
    
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
        
        # 在測試集上測試並提取特徵
        logger.info("\n📊 測試集評估與特徵提取:")
        test_results = trainer.test_and_extract_features(
            test_loader,
            output_dir=str(feature_output_dir),
            extract_deep_features=args.extract_features,
            save_predictions=True,
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
        logger.info(f"  Dice Score: {val_metrics['dice']:.4f}")
        logger.info(f"  IoU/Jaccard: {val_metrics['iou']:.4f}")
        logger.info(f"  Precision: {val_metrics['precision']:.4f}")
        logger.info(f"  Recall/Sensitivity: {val_metrics['recall']:.4f}")
        logger.info(f"  Specificity: {val_metrics['specificity']:.4f}")
        logger.info(f"  Hausdorff Distance (95%): {val_metrics['hausdorff_95']:.2f} pixels")
        logger.info(f"  Inference Time: {val_time:.1f}s")
        logger.info(f"{'='*80}\n")
        
        # ✅ 測試集評估
        logger.info("\n📊 測試集評估:")
        test_loss, test_metrics, test_time = trainer.validate(test_loader)
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ 測試集結果:")
        logger.info(f"  Loss: {test_loss:.4f}")
        logger.info(f"  Dice Score: {test_metrics['dice']:.4f}")
        logger.info(f"  IoU/Jaccard: {test_metrics['iou']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall/Sensitivity: {test_metrics['recall']:.4f}")
        logger.info(f"  Specificity: {test_metrics['specificity']:.4f}")
        logger.info(f"  Hausdorff Distance (95%): {test_metrics['hausdorff_95']:.2f} pixels")
        logger.info(f"  Inference Time: {test_time:.1f}s")
        logger.info(f"{'='*80}\n")
        
        return
    
    # 開始訓練
    trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        accumulation_steps=args.accumulation_steps
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
    test_loss, test_metrics, test_time = trainer.validate(test_loader, metrics_tracker=test_metrics_tracker)
    
    # ✅ 新增：保存測試集詳細報告
    test_metrics_tracker.save_report(args.output_dir, split_name='test')
    
    # 顯示測試結果
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ 測試集結果:")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Dice Score: {test_metrics['dice']:.4f}")
    logger.info(f"  IoU/Jaccard: {test_metrics['iou']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall/Sensitivity: {test_metrics['recall']:.4f}")
    logger.info(f"  Specificity: {test_metrics['specificity']:.4f}")
    logger.info(f"  Hausdorff Distance (95%): {test_metrics['hausdorff_95']:.2f} pixels")
    logger.info(f"  Inference Time: {test_time:.1f}s")
    logger.info(f"{'='*80}\n")
    
    # ✅ 新增：顯示低分病例統計
    poor_cases = test_metrics_tracker.get_poor_performers(metric_name='dice', threshold=0.5)
    if poor_cases:
        logger.info(f"⚠️ 發現 {len(poor_cases)} 個低分病例（Dice < 0.5）:")
        for patient_id, score in poor_cases[:10]:  # 只顯示最差的 10 個
            logger.info(f"   - 患者 {patient_id}: Dice = {score:.4f}")
        if len(poor_cases) > 10:
            logger.info(f"   ... 以及其他 {len(poor_cases) - 10} 個病例")
        logger.info(f"   詳細清單請查看: {args.output_dir}/test_error_cases.json\n")
    
    # 保存訓練配置與測試結果
    config_dict = vars(args)
    config_dict['best_val_dice'] = trainer.best_val_dice
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
