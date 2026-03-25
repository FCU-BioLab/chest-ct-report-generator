#!/usr/bin/env python3
"""
Training-data visualizer for the 3D U-Net pipeline.

This module is NIfTI-task only and mirrors what the trainer sees through
`VolumetricDataset` (resize, depth crop, normalization, optional augmentation).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .dataset import VolumetricDataset


logger = logging.getLogger(__name__)


class DatasetVisualizer:
    """Visualize preprocessed samples exactly as consumed by training."""

    def __init__(self, data_dir: str, split: str = "train", image_size: int = 256, max_depth: int = 32):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.max_depth = max_depth
        self.dataset = VolumetricDataset(
            data_dir=data_dir,
            split=split,
            image_size=image_size,
            max_depth=max_depth,
            augmentation=False,
        )
        logger.info(
            "Loaded dataset split=%s with %d samples (data_dir=%s)",
            split,
            len(self.dataset),
            data_dir,
        )

    @staticmethod
    def _make_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        overlay = np.stack([frame, frame, frame], axis=-1)
        if mask.sum() > 0:
            overlay[mask > 0, 0] = np.clip(overlay[mask > 0, 0] + 0.4, 0, 1)
            overlay[mask > 0, 1] *= 0.5
            overlay[mask > 0, 2] *= 0.5
        return overlay

    def print_statistics(self) -> None:
        if len(self.dataset) == 0:
            print("No samples found.")
            return

        n_analyze = min(len(self.dataset), 50)
        depths = []
        mask_voxels = []
        mask_ratios = []
        has_mask = 0

        for i in range(n_analyze):
            sample = self.dataset[i]
            image = sample["image"]  # (1, D, H, W)
            mask = sample["mask"]  # (1, D, H, W)
            depths.append(int(image.shape[1]))

            m = mask.numpy()
            vox = float(m.sum())
            mask_voxels.append(vox)
            ratio = vox / m.size * 100.0
            mask_ratios.append(ratio)
            if vox > 0:
                has_mask += 1

        print("=" * 70)
        print("3D U-Net Dataset Statistics")
        print("=" * 70)
        print(f"Split: {self.split}")
        print(f"Source: {self.data_dir}")
        print(f"Samples: {len(self.dataset)}")
        print(f"Image size: {self.image_size}x{self.image_size}")
        print(f"Max depth: {self.max_depth}")
        print(f"Analyzed: {n_analyze}")
        print(f"Depth (D): min={min(depths)}, max={max(depths)}, mean={np.mean(depths):.1f}")
        print(f"Mask-positive samples: {has_mask}/{n_analyze} ({has_mask / n_analyze * 100.0:.1f}%)")
        print(
            "Mask voxels: min={:.0f}, max={:.0f}, mean={:.1f}".format(
                min(mask_voxels),
                max(mask_voxels),
                float(np.mean(mask_voxels)),
            )
        )
        print(
            "Mask ratio (%): min={:.6f}, max={:.6f}, mean={:.6f}".format(
                min(mask_ratios),
                max(mask_ratios),
                float(np.mean(mask_ratios)),
            )
        )
        print("=" * 70)

    def visualize_sample(self, idx: int, save_path: Optional[str] = None) -> None:
        if idx < 0 or idx >= len(self.dataset):
            raise IndexError(f"Sample idx out of range: {idx} / {len(self.dataset)}")

        sample = self.dataset[idx]
        image = sample["image"].numpy()[0]  # (D,H,W), [0,1]
        mask = sample["mask"].numpy()[0]  # (D,H,W)
        source_name = Path(sample["source_path"]).stem
        patient_id = sample["patient_id"]

        d, h, w = image.shape
        n_show = min(d, 16)
        slice_indices = np.linspace(0, d - 1, n_show, dtype=int)

        cols = 4
        rows = int(np.ceil(n_show / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = np.array(axes).reshape(-1)

        mask_ratio = float(mask.sum() / mask.size * 100.0)
        fig.suptitle(
            f"Sample={source_name} | patient={patient_id} | shape={d}x{h}x{w} | mask_ratio={mask_ratio:.4f}%",
            fontsize=11,
            fontweight="bold",
        )

        for ax, z in zip(axes, slice_indices):
            overlay = self._make_overlay(image[z], mask[z])
            ax.imshow(overlay)
            pixels = int(mask[z].sum())
            ax.set_title(f"z={z} | mask={pixels}", fontsize=9)
            ax.axis("off")

        for ax in axes[len(slice_indices):]:
            ax.axis("off")

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved sample visualization to %s", save_path)
            plt.close()
            return
        plt.show()
        plt.close()

    def visualize_batch(self, n_samples: int = 9, save_path: Optional[str] = None) -> None:
        if len(self.dataset) == 0:
            print("No samples found.")
            return
        n_samples = max(1, min(n_samples, len(self.dataset)))
        indices = np.linspace(0, len(self.dataset) - 1, n_samples, dtype=int)

        cols = int(np.ceil(np.sqrt(n_samples)))
        rows = int(np.ceil(n_samples / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)

        for ax, idx in zip(axes, indices):
            sample = self.dataset[int(idx)]
            image = sample["image"].numpy()[0]
            mask = sample["mask"].numpy()[0]
            z = int(np.argmax(mask.sum(axis=(1, 2)))) if mask.sum() > 0 else image.shape[0] // 2
            ax.imshow(self._make_overlay(image[z], mask[z]))
            ax.set_title(Path(sample["source_path"]).stem, fontsize=8)
            ax.axis("off")

        for ax in axes[len(indices):]:
            ax.axis("off")

        fig.suptitle(f"Dataset batch preview ({self.split}, n={n_samples})", fontsize=11, fontweight="bold")
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved batch visualization to %s", save_path)
            plt.close()
            return
        plt.show()
        plt.close()

    def visualize_augmentation(self, idx: int, n_augments: int = 4, save_path: Optional[str] = None) -> None:
        if idx < 0 or idx >= len(self.dataset):
            raise IndexError(f"Sample idx out of range: {idx} / {len(self.dataset)}")

        aug_dataset = VolumetricDataset(
            data_dir=self.data_dir,
            split=self.split,
            image_size=self.image_size,
            max_depth=self.max_depth,
            augmentation=True,
        )

        base_sample = self.dataset[idx]
        base_image = base_sample["image"].numpy()[0]
        base_mask = base_sample["mask"].numpy()[0]
        z = int(np.argmax(base_mask.sum(axis=(1, 2)))) if base_mask.sum() > 0 else base_image.shape[0] // 2
        z = min(z, base_image.shape[0] - 1)

        cols = n_augments + 1
        fig, axes = plt.subplots(2, cols, figsize=(3.8 * cols, 6))

        axes[0, 0].imshow(base_image[z], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")
        axes[1, 0].imshow(self._make_overlay(base_image[z], base_mask[z]))
        axes[1, 0].set_title(f"Original overlay\nmask={int(base_mask[z].sum())}")
        axes[1, 0].axis("off")

        for j in range(n_augments):
            aug_sample = aug_dataset[idx]
            aug_image = aug_sample["image"].numpy()[0]
            aug_mask = aug_sample["mask"].numpy()[0]
            zz = min(z, aug_image.shape[0] - 1)
            axes[0, j + 1].imshow(aug_image[zz], cmap="gray", vmin=0.0, vmax=1.0)
            axes[0, j + 1].set_title(f"Aug #{j + 1}")
            axes[0, j + 1].axis("off")
            axes[1, j + 1].imshow(self._make_overlay(aug_image[zz], aug_mask[zz]))
            axes[1, j + 1].set_title(f"Overlay\nmask={int(aug_mask[zz].sum())}")
            axes[1, j + 1].axis("off")

        fig.suptitle(
            f"Augmentation preview: {Path(base_sample['source_path']).stem}",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved augmentation visualization to %s", save_path)
            plt.close()
            return
        plt.show()
        plt.close()


def run_visualization(args: argparse.Namespace) -> None:
    if not getattr(args, "data_dir", None):
        raise ValueError("`data_dir` is required.")

    viz = DatasetVisualizer(
        data_dir=args.data_dir,
        split=args.split,
        image_size=args.image_size,
        max_depth=args.max_depth,
    )

    if args.mode == "dataset":
        viz.print_statistics()
    elif args.mode == "dataset_view":
        viz.visualize_sample(args.idx, args.save)
    elif args.mode == "dataset_batch":
        viz.visualize_batch(args.n_samples, args.save)
    elif args.mode == "dataset_augment":
        viz.visualize_augmentation(args.idx, args.n_augments, args.save)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize 3D U-Net dataset samples")
    parser.add_argument("--data_dir", type=str, default="detection/nndet_data/Task100_LUNA16Nodule")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--mode",
        type=str,
        default="dataset",
        choices=["dataset", "dataset_view", "dataset_batch", "dataset_augment"],
    )
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--n_samples", type=int, default=9)
    parser.add_argument("--n_augments", type=int, default=4)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_depth", type=int, default=32)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_visualization(args)


if __name__ == "__main__":
    main()
