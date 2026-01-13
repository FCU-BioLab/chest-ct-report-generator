#!/usr/bin/env python3
"""
MedSAM2 視頻模式訓練 - 主程式
==============================

將 CT 切片序列視為「影片」，利用 MedSAM2 的時序傳播能力學習病灶分割。

使用方式：

1. 資料轉換（LNDb → NPZ 視頻格式）:
   python main.py convert --dataset lndb --input_dir /path/to/LNDb

2. 訓練:
   python main.py train --npz_dir video_npz --epochs 50

3. 推論:
   python main.py infer --checkpoint video_output/checkpoints/best_model.pt --input /path/to/ct.nii.gz
"""

import argparse
import logging
import sys
from pathlib import Path
import random
import numpy as np
import torch

# 設定路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from finetune_medsam2_video.config import VideoConfig, DataConfig, ModelConfig, TrainingConfig
from finetune_medsam2_video.npz_converter import NPZConverter
from finetune_medsam2_video.video_trainer import MedSAM2VideoTrainer
from finetune_medsam2_video.video_dataset import VideoLesionDataset


def setup_logging(log_level: str = "INFO"):
    """設定日誌"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def set_seed(seed: int):
    """設定隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cmd_convert(args):
    """資料轉換命令"""
    logging.info("🔄 開始資料轉換...")
    
    converter = NPZConverter(
        output_dir=args.output_dir,
        context_slices=args.context_slices,
        min_nodule_diameter=args.min_diameter,
        max_video_length=args.max_video_length,
        image_size=args.image_size,
    )
    
    if args.dataset == 'lndb':
        converter.convert_lndb(
            lndb_dir=args.input_dir,
            split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        )
    elif args.dataset == 'msd':
        converter.convert_msd_lung(
            msd_dir=args.input_dir,
            split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        )
    else:
        logging.error(f"❌ 不支援的資料集: {args.dataset}")
        return
    
    logging.info("✅ 資料轉換完成")


def cmd_train(args):
    """訓練命令"""
    logging.info("🚀 開始視頻模式訓練...")
    
    # 建立配置
    config = VideoConfig()
    
    # 資料配置
    config.data.npz_dir = args.npz_dir
    config.data.max_video_length = args.max_video_length
    
    # 模型配置
    config.model.config = args.model_config
    if args.checkpoint:
        config.model.checkpoint = args.checkpoint
    config.model.image_size = args.image_size
    
    # 訓練配置
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.weight_decay = args.weight_decay
    config.training.warmup_epochs = args.warmup_epochs
    config.training.dice_weight = args.dice_weight
    config.training.focal_weight = args.focal_weight
    config.training.use_amp = not args.no_amp
    config.training.early_stopping_patience = args.patience
    config.training.propagation_steps = args.propagation_steps
    config.training.prompt_type = args.prompt_type  # 新增: bbox 或 point
    
    # 凍結選項
    if args.freeze_prompt_encoder:
        config.training.freeze_prompt_encoder = True
    
    # 其他配置
    config.output_dir = args.output_dir
    config.seed = args.seed
    config.device = args.device
    config.num_workers = args.num_workers
    
    # 設定種子
    set_seed(config.seed)
    
    # 記錄 prompt 類型
    logging.info(f"📍 Prompt 類型: {config.training.prompt_type}")
    
    # 建立訓練器並開始訓練
    trainer = MedSAM2VideoTrainer(config)
    trainer.train()
    
    logging.info("✅ 訓練完成")


def cmd_infer(args):
    """推論命令"""
    logging.info("🔍 開始推論...")
    
    # 載入配置和模型
    config = VideoConfig()
    config.device = args.device
    
    if args.config:
        config = VideoConfig.load(args.config)
    
    trainer = MedSAM2VideoTrainer(config)
    trainer.load_checkpoint(args.checkpoint)
    
    # TODO: 實現推論邏輯
    logging.info("❌ 推論功能尚未實現，請使用 MedSAM2/medsam2_infer_CT_lesion_npz_recist.py")


def cmd_stats(args):
    """資料集統計命令"""
    logging.info("📊 資料集統計...")
    
    for split in ['train', 'val', 'test']:
        dataset = VideoLesionDataset(
            npz_dir=args.npz_dir,
            split=split,
            image_size=512,
        )
        
        if len(dataset) == 0:
            continue
        
        # 收集統計
        num_frames_list = []
        diameters = []
        
        for i in range(len(dataset)):
            info = dataset.get_sample_info(i)
            num_frames_list.append(info['num_frames'])
            if info['diameter_mm'] > 0:
                diameters.append(info['diameter_mm'])
        
        logging.info(f"\n📁 {split.upper()} Split:")
        logging.info(f"  - 樣本數: {len(dataset)}")
        logging.info(f"  - 視頻長度: {np.mean(num_frames_list):.1f} ± {np.std(num_frames_list):.1f} 幀")
        if diameters:
            logging.info(f"  - 病灶直徑: {np.mean(diameters):.1f} ± {np.std(diameters):.1f} mm")
            logging.info(f"    - 最小: {np.min(diameters):.1f} mm")
            logging.info(f"    - 最大: {np.max(diameters):.1f} mm")


def main():
    parser = argparse.ArgumentParser(
        description='MedSAM2 視頻模式訓練',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:

  # 1. 轉換 LNDb 資料集
  python main.py convert --dataset lndb --input_dir /path/to/LNDb

  # 2. 訓練模型
  python main.py train --npz_dir video_npz --epochs 50

  # 3. 查看資料集統計
  python main.py stats --npz_dir video_npz
        """
    )
    
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # ========== convert 子命令 ==========
    convert_parser = subparsers.add_parser('convert', help='將資料集轉換為 NPZ 視頻格式')
    convert_parser.add_argument('--dataset', type=str, required=True,
                               choices=['lndb', 'msd'],
                               help='資料集類型')
    convert_parser.add_argument('--input_dir', type=str, required=True,
                               help='輸入資料集目錄')
    convert_parser.add_argument('--output_dir', type=str, default='video_npz',
                               help='NPZ 輸出目錄')
    convert_parser.add_argument('--context_slices', type=int, default=6,
                               help='中心切片前後各取幾個切片')
    convert_parser.add_argument('--max_video_length', type=int, default=32,
                               help='最大視頻長度')
    convert_parser.add_argument('--min_diameter', type=float, default=4.0,
                               help='最小結節直徑 (mm)')
    convert_parser.add_argument('--image_size', type=int, default=512,
                               help='輸出影像大小')
    convert_parser.add_argument('--train_ratio', type=float, default=0.7)
    convert_parser.add_argument('--val_ratio', type=float, default=0.15)
    convert_parser.add_argument('--test_ratio', type=float, default=0.15)
    
    # ========== train 子命令 ==========
    train_parser = subparsers.add_parser('train', help='訓練視頻分割模型')
    train_parser.add_argument('--npz_dir', type=str, default='video_npz',
                             help='NPZ 資料目錄')
    train_parser.add_argument('--output_dir', type=str, default='video_output',
                             help='輸出目錄')
    train_parser.add_argument('--model_config', type=str, default='sam2.1_hiera_t512.yaml',
                             help='SAM2 模型配置')
    train_parser.add_argument('--checkpoint', type=str, default=None,
                             help='預訓練 checkpoint 路徑')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='訓練 epochs')
    train_parser.add_argument('--batch_size', type=int, default=1,
                             help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, default=1e-5,
                             help='學習率')
    train_parser.add_argument('--propagation_steps', type=int, default=3,
                             help='傳播步數')
    train_parser.add_argument('--max_video_length', type=int, default=32,
                             help='最大視頻長度')
    train_parser.add_argument('--image_size', type=int, default=512,
                             help='影像大小')
    train_parser.add_argument('--patience', type=int, default=5,
                             help='早停 patience (建議 5)')
    train_parser.add_argument('--weight_decay', type=float, default=0.01,
                             help='權重衰減 (防止過擬合，建議 0.01)')
    train_parser.add_argument('--warmup_epochs', type=int, default=5,
                             help='Warmup epochs')
    train_parser.add_argument('--dice_weight', type=float, default=1.0,
                             help='Dice Loss 權重')
    train_parser.add_argument('--focal_weight', type=float, default=0.5,
                             help='Focal Loss 權重')
    train_parser.add_argument('--prompt_type', type=str, default='bbox',
                             choices=['bbox', 'point'],
                             help='Prompt 類型: bbox=邊界框, point=中心點')
    train_parser.add_argument('--no_amp', action='store_true',
                             help='禁用混合精度')
    train_parser.add_argument('--freeze_prompt_encoder', action='store_true',
                             help='凍結 Prompt Encoder (減少過擬合)')
    train_parser.add_argument('--device', type=str, default='cuda',
                             help='計算設備')
    train_parser.add_argument('--num_workers', type=int, default=4,
                             help='資料載入 workers')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='隨機種子')
    
    # ========== infer 子命令 ==========
    infer_parser = subparsers.add_parser('infer', help='推論')
    infer_parser.add_argument('--checkpoint', type=str, required=True,
                             help='模型 checkpoint')
    infer_parser.add_argument('--config', type=str, default=None,
                             help='配置檔案')
    infer_parser.add_argument('--input', type=str, required=True,
                             help='輸入 CT 檔案')
    infer_parser.add_argument('--output', type=str, default='output',
                             help='輸出目錄')
    infer_parser.add_argument('--device', type=str, default='cuda')
    
    # ========== stats 子命令 ==========
    stats_parser = subparsers.add_parser('stats', help='資料集統計')
    stats_parser.add_argument('--npz_dir', type=str, default='video_npz',
                             help='NPZ 資料目錄')
    
    args = parser.parse_args()
    
    # 設定日誌
    setup_logging(args.log_level)
    
    # 執行命令
    if args.command == 'convert':
        cmd_convert(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'infer':
        cmd_infer(args)
    elif args.command == 'stats':
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
