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
    config.model.use_attention = getattr(args, 'attention', False)
    
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.loss_type = getattr(args, 'loss_type', 'combined')
    
    # Only override output_dir if explicitly provided
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    # Otherwise use the auto-generated timestamp path from Config.__post_init__
    
    config.seed = args.seed
    config.device = args.device
    
    set_seed(config.seed)
    
    # Log model and loss info
    logging.info(f"📁 Output directory: {config.output_dir}")
    if config.model.use_attention:
        logging.info("🧠 Using AttentionUNet3D (SE + Attention Gates)")
    else:
        logging.info("🔷 Using standard UNet3D")
    logging.info(f"📉 Loss type: {config.training.loss_type}")
    
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
    results = trainer.evaluate(args.split, use_postprocess=not args.no_postprocess)
    
    # Save results to file
    import json
    output_path = Path(args.checkpoint).parent / f"eval_results_{args.split}.json"
    with open(output_path, 'w') as f:
        # Remove sample_results for cleaner output
        summary = {k: v for k, v in results.items() if k != 'sample_results'}
        json.dump(summary, f, indent=2)
    logging.info(f"📁 Results saved to: {output_path}")

def cmd_visualize(args):
    logging.info("🖼️ Starting Visualization...")
    
    config = Config()
    config.data.npz_dir = args.npz_dir
    config.model.base_filters = args.base_filters
    config.model.image_size = args.image_size
    config.device = args.device
    
    trainer = UNet3DTrainer(config)
    trainer.load_checkpoint(args.checkpoint)
    save_dir = trainer.visualize_predictions(args.split, args.output_dir)
    
    logging.info(f"✅ Visualization complete! Images saved to: {save_dir}")

def cmd_fulltest(args):
    """Run comprehensive test with all metrics and visualizations"""
    logging.info("🔬 Starting Comprehensive Test...")
    
    config = Config()
    config.data.npz_dir = args.npz_dir
    config.model.base_filters = args.base_filters
    config.model.image_size = args.image_size
    config.device = args.device
    config.num_workers = 0  # Avoid multiprocessing issues on Windows
    
    trainer = UNet3DTrainer(config)
    trainer.load_checkpoint(args.checkpoint)
    
    summary = trainer.comprehensive_test(
        split=args.split,
        save_visualizations=not args.no_viz,
        export_gif=not args.no_gif
    )
    
    logging.info("✅ Comprehensive test complete!")

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
    train.add_argument('--output_dir', default=None, help='Output directory (default: segmentation/video_result/3dunet_train_TIMESTAMP)')
    train.add_argument('--epochs', type=int, default=50)
    train.add_argument('--batch_size', type=int, default=2)
    train.add_argument('--learning_rate', type=float, default=1e-4)
    train.add_argument('--max_depth', type=int, default=32)
    train.add_argument('--base_filters', type=int, default=32)
    train.add_argument('--image_size', type=int, default=256)
    train.add_argument('--device', default='cuda')
    train.add_argument('--seed', type=int, default=42)
    # Model options
    train.add_argument('--attention', action='store_true', 
                       help='Enable SE + Attention Gate model')
    # Loss options
    train.add_argument('--loss_type', default='combined', 
                       choices=['dice', 'tversky', 'combined'],
                       help='Loss function type')
    
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
    test.add_argument('--no_postprocess', action='store_true', 
                      help='Disable lung mask and postprocessing')
    
    # FULLTEST (comprehensive test with visualization)
    fulltest = subparsers.add_parser('fulltest', help='Full test with segmentation & detection metrics + visualization')
    fulltest.add_argument('--npz_dir', default='volume_npz')
    fulltest.add_argument('--checkpoint', required=True)
    fulltest.add_argument('--split', default='test')
    fulltest.add_argument('--base_filters', type=int, default=32)
    fulltest.add_argument('--image_size', type=int, default=256)
    fulltest.add_argument('--device', default='cuda')
    fulltest.add_argument('--no_viz', action='store_true', help='Skip per-sample visualizations')
    fulltest.add_argument('--no_gif', action='store_true', help='Skip GIF animation export')
    
    # VISUALIZE
    viz = subparsers.add_parser('visualize', help='Visualize predictions vs GT for all samples')
    viz.add_argument('--npz_dir', default='volume_npz')
    viz.add_argument('--checkpoint', required=True)
    viz.add_argument('--split', default='test')
    viz.add_argument('--base_filters', type=int, default=32)
    viz.add_argument('--image_size', type=int, default=256)
    viz.add_argument('--device', default='cuda')
    viz.add_argument('--output_dir', default=None, help='Output directory for images')
    
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
    elif args.command == 'fulltest':
        cmd_fulltest(args)
    elif args.command == 'visualize':
        cmd_visualize(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
