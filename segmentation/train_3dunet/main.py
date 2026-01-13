#!/usr/bin/env python3
"""
3D U-Net Video Finetuning - Main Entry
======================================

CLI for 3D U-Net video/volume segmentation.
"""

import argparse
import logging
import sys
from pathlib import Path
import random
import numpy as np
import torch

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_3dunet.config import Config
from train_3dunet.preprocess import VolumePreprocessor
from train_3dunet.trainer import UNet3DTrainer
from train_3dunet.dataset import VolumetricDataset

def setup_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cmd_convert(args):
    logging.info("🔄 Starting conversion...")
    converter = VolumePreprocessor(
        output_dir=args.output_dir,
        context_slices=args.context_slices,
        min_nodule_diameter=args.min_diameter,
        max_depth=args.max_depth,
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
    
    logging.info("✅ Conversion complete")

def cmd_train(args):
    logging.info("🚀 Starting 3D U-Net Training...")
    
    config = Config()
    config.data.npz_dir = args.npz_dir
    config.data.max_depth = args.max_depth
    
    config.model.base_filters = args.base_filters
    config.model.image_size = args.image_size
    
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    
    config.output_dir = args.output_dir
    config.seed = args.seed
    config.device = args.device
    
    set_seed(config.seed)
    
    trainer = UNet3DTrainer(config)
    trainer.train()
    
    logging.info("✅ Training complete")

def cmd_stats(args):
    logging.info("📊 Dataset Statistics...")
    for split in ['train', 'val', 'test']:
        dataset = VolumetricDataset(
            npz_dir=args.npz_dir,
            split=split,
            image_size=256,
        )
        if len(dataset) == 0: continue
        
        lengths = []
        for i in range(len(dataset)):
            d = dataset[i]['image'].shape[1]
            lengths.append(d)
            
        logging.info(f"\n📁 {split.upper()}:")
        logging.info(f"  - Samples: {len(dataset)}")
        logging.info(f"  - Depth (avg): {np.mean(lengths):.1f}")

def cmd_test(args):
    logging.info("🔍 Starting Testing...")
    
    config = Config()
    config.data.npz_dir = args.npz_dir
    config.model.base_filters = args.base_filters
    config.model.image_size = args.image_size
    config.device = args.device
    
    trainer = UNet3DTrainer(config)
    trainer.load_checkpoint(args.checkpoint)
    trainer.evaluate(args.split)

def main():
    parser = argparse.ArgumentParser(description='3D U-Net Video Training')
    parser.add_argument('--log_level', default='INFO')
    
    subparsers = parser.add_subparsers(dest='command')
    
    # CONVERT
    conv = subparsers.add_parser('convert')
    conv.add_argument('--dataset', required=True, choices=['lndb', 'msd'])
    conv.add_argument('--input_dir', required=True)
    conv.add_argument('--output_dir', default='volume_npz')
    conv.add_argument('--context_slices', type=int, default=16)
    conv.add_argument('--max_depth', type=int, default=32)
    conv.add_argument('--min_diameter', type=float, default=4.0)
    conv.add_argument('--image_size', type=int, default=256)
    conv.add_argument('--train_ratio', type=float, default=0.7)
    conv.add_argument('--val_ratio', type=float, default=0.15)
    conv.add_argument('--test_ratio', type=float, default=0.15)
    
    # TRAIN
    train = subparsers.add_parser('train')
    train.add_argument('--npz_dir', default='volume_npz')
    train.add_argument('--output_dir', default='volume_output_unet3d')
    train.add_argument('--epochs', type=int, default=50)
    train.add_argument('--batch_size', type=int, default=2)
    train.add_argument('--learning_rate', type=float, default=1e-4)
    train.add_argument('--max_depth', type=int, default=32)
    train.add_argument('--base_filters', type=int, default=32)
    train.add_argument('--image_size', type=int, default=256)
    train.add_argument('--device', default='cuda')
    train.add_argument('--seed', type=int, default=42)
    
    # STATS
    stats = subparsers.add_parser('stats')
    stats.add_argument('--npz_dir', default='volume_npz')
    
    # TEST
    test = subparsers.add_parser('test')
    test.add_argument('--npz_dir', default='volume_npz')
    test.add_argument('--checkpoint', required=True)
    test.add_argument('--split', default='test')
    test.add_argument('--base_filters', type=int, default=32)
    test.add_argument('--image_size', type=int, default=256)
    test.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    if args.command == 'convert':
        cmd_convert(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'stats':
        cmd_stats(args)
    elif args.command == 'test':
        cmd_test(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
