#!/usr/bin/env python3
"""
3D U-Net main entry.

NIfTI task folder layout expected:
- imagesTr/*_0000.nii.gz
- labelsTr/*.nii.gz
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from train_3dunet.config import Config
from train_3dunet.dataset import VolumetricDataset
from train_3dunet.trainer import UNet3DTrainer
from train_3dunet.visualize import run_visualization


DEFAULT_DATA_DIR = "detection/nndet_data/Task100_LUNA16Nodule"


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_common_config(config: Config, args: argparse.Namespace) -> None:
    if not Path(args.data_dir).exists():
        raise FileNotFoundError(f"data_dir not found: {args.data_dir}")
    config.data.data_dir = args.data_dir
    config.data.max_depth = args.max_depth
    n_levels = len(getattr(config.model, "f_maps", (32, 64, 128, 256)))
    config.model.f_maps = tuple(int(args.base_filters) * (2 ** i) for i in range(n_levels))
    config.model.image_size = args.image_size
    config.model.use_attention = getattr(args, "attention", False)
    config.device = args.device


def cmd_train(args: argparse.Namespace) -> None:
    logging.info("Starting 3D U-Net training")
    if args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer.")
    if args.accumulation_steps <= 0:
        raise ValueError("--accumulation_steps must be >= 1.")
    ratios_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(ratios_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"train/val/test ratios must sum to 1.0, got {ratios_sum:.6f} "
            f"({args.train_ratio}, {args.val_ratio}, {args.test_ratio})."
        )

    config = Config()

    apply_common_config(config, args)
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.loss_type = getattr(args, "loss_type", "combined")
    config.training.accumulation_steps = getattr(args, "accumulation_steps", 1)
    config.training.early_stopping_patience = args.early_stopping_patience
    config.training.enable_tensorboard = not getattr(args, "no_tensorboard", False)
    config.seed = args.seed

    if args.output_dir is not None:
        config.output_dir = args.output_dir

    config.data.train_ratio = args.train_ratio
    config.data.val_ratio = args.val_ratio
    config.data.test_ratio = args.test_ratio
    config.data.split_seed = args.split_seed
    config.data.positive_ratio = args.positive_ratio
    config.model.use_checkpointing = args.use_checkpointing

    set_seed(config.seed)

    logging.info("Output directory: %s", config.output_dir)
    logging.info("Model: %s", "AttentionUNet3D" if config.model.use_attention else "UNet3D")
    logging.info("Loss: %s", config.training.loss_type)

    trainer = UNet3DTrainer(config)
    trainer.train()
    logging.info("Training complete")


def cmd_stats(args: argparse.Namespace) -> None:
    logging.info("Dataset statistics")
    for split in ["train", "val", "test"]:
        dataset = VolumetricDataset(
            data_dir=args.data_dir,
            split=split,
            image_size=256,
            max_depth=args.max_depth,
        )
        if len(dataset) == 0:
            continue
        depths = [dataset[i]["image"].shape[1] for i in range(len(dataset))]
        logging.info("%s: samples=%d, avg_depth=%.1f", split, len(dataset), float(np.mean(depths)))


def cmd_test(args: argparse.Namespace) -> None:
    logging.info("Starting test")
    config = Config()
    apply_common_config(config, args)

    if args.full_volume:
        config.data.max_depth = 10000
        config.training.batch_size = 1
        torch.cuda.empty_cache()
        logging.info("Full-volume test enabled")

    config.postprocessing.det_threshold = args.det_prob_threshold
    config.postprocessing.det_min_size = args.det_min_size
    config.postprocessing.apply_closing = not args.no_closing

    trainer = UNet3DTrainer(config)
    trainer.load_checkpoint(args.checkpoint)
    results = trainer.evaluate(args.split, use_postprocess=not args.no_postprocess)

    output_path = Path(args.checkpoint).parent / f"eval_results_{args.split}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        summary = {k: v for k, v in results.items() if k != "sample_results"}
        json.dump(summary, f, indent=2)
    logging.info("Results saved: %s", output_path)


def cmd_visualize(args: argparse.Namespace) -> None:
    logging.info("Starting prediction visualization")
    config = Config()
    apply_common_config(config, args)

    trainer = UNet3DTrainer(config)
    trainer.load_checkpoint(args.checkpoint)
    save_dir = trainer.visualize_predictions(args.split, args.output_dir)
    logging.info("Visualization saved: %s", save_dir)


def cmd_check_data(args: argparse.Namespace) -> None:
    run_visualization(args)


def cmd_fulltest(args: argparse.Namespace) -> None:
    logging.info("Starting comprehensive test")
    config = Config()
    apply_common_config(config, args)
    config.num_workers = 0

    if args.full_volume:
        config.data.max_depth = 10000
        config.training.batch_size = 1
        torch.cuda.empty_cache()
        logging.info("Full-volume test enabled")

    config.postprocessing.det_threshold = args.det_prob_threshold
    config.postprocessing.det_min_size = args.det_min_size
    config.postprocessing.apply_closing = not args.no_closing

    trainer = UNet3DTrainer(config)
    trainer.load_checkpoint(args.checkpoint)
    trainer.comprehensive_test(
        split=args.split,
        save_visualizations=not args.no_viz,
        export_gif=not args.no_gif,
        det_min_size=args.det_min_size,
        det_threshold=args.det_prob_threshold,
        no_postprocess=getattr(args, "no_postprocess", False),
    )
    logging.info("Comprehensive test complete")


def add_data_dir_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="NIfTI task folder (imagesTr/labelsTr)")


def add_common_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max_depth", type=int, default=33)
    p.add_argument("--base_filters", type=int, default=32)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--attention", action="store_true", help="Enable Attention UNet")


def main() -> None:
    parser = argparse.ArgumentParser(description="3D U-Net training/testing")
    parser.add_argument("--log_level", default="INFO")
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser("train")
    add_data_dir_arg(train)
    add_common_runtime_args(train)
    train.add_argument("--output_dir", default=None)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch_size", type=int, default=2)
    train.add_argument("--learning_rate", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--train_ratio", type=float, default=0.7)
    train.add_argument("--val_ratio", type=float, default=0.15)
    train.add_argument("--test_ratio", type=float, default=0.15)
    train.add_argument("--split_seed", type=int, default=42)
    train.add_argument("--loss_type", default="combined", choices=["dice", "tversky", "combined"])
    train.add_argument("--positive_ratio", type=float, default=0.7)
    train.add_argument("--use_checkpointing", action="store_true")
    train.add_argument("--accumulation_steps", type=int, default=1)
    train.add_argument("--early_stopping_patience", type=int, default=150)
    train.add_argument("--no_tensorboard", action="store_true")

    stats = subparsers.add_parser("stats")
    add_data_dir_arg(stats)
    stats.add_argument("--max_depth", type=int, default=33)

    test = subparsers.add_parser("test")
    add_data_dir_arg(test)
    add_common_runtime_args(test)
    test.add_argument("--checkpoint", required=True)
    test.add_argument("--split", default="test")
    test.add_argument("--no_postprocess", action="store_true")
    test.add_argument("--det_prob_threshold", type=float, default=0.5)
    test.add_argument("--det_min_size", type=float, default=30.0)
    test.add_argument("--no_closing", action="store_true")
    test.add_argument("--full_volume", action="store_true")

    fulltest = subparsers.add_parser("fulltest")
    add_data_dir_arg(fulltest)
    add_common_runtime_args(fulltest)
    fulltest.add_argument("--checkpoint", required=True)
    fulltest.add_argument("--split", default="test")
    fulltest.add_argument("--no_viz", action="store_true")
    fulltest.add_argument("--no_gif", action="store_true")
    fulltest.add_argument("--det_prob_threshold", type=float, default=0.5)
    fulltest.add_argument("--det_min_size", type=float, default=10.0)
    fulltest.add_argument("--no_closing", action="store_true")
    fulltest.add_argument("--full_volume", action="store_true")
    fulltest.add_argument("--no_postprocess", action="store_true")

    viz = subparsers.add_parser("visualize")
    add_data_dir_arg(viz)
    add_common_runtime_args(viz)
    viz.add_argument("--checkpoint", required=True)
    viz.add_argument("--split", default="test")
    viz.add_argument("--output_dir", default=None)

    check = subparsers.add_parser("check_data")
    add_data_dir_arg(check)
    check.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    check.add_argument(
        "--mode",
        type=str,
        default="dataset",
        choices=["dataset", "dataset_view", "dataset_batch", "dataset_augment"],
    )
    check.add_argument("--idx", type=int, default=0)
    check.add_argument("--n_samples", type=int, default=9)
    check.add_argument("--n_augments", type=int, default=4)
    check.add_argument("--save", type=str, default=None)
    check.add_argument("--image_size", type=int, default=256)
    check.add_argument("--max_depth", type=int, default=32)

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "fulltest":
        cmd_fulltest(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "check_data":
        cmd_check_data(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
