#!/usr/bin/env python3
"""
Visualize training data format for RetinaNet.

Shows:
1. Raw NPZ volume shape + value range
2. Extracted bounding boxes (from mask)
3. Cropped patch after random crop
4. Overlay: patch slices with bounding box rectangles

Usage:
    python -m detection.retinanet.visualize_data --npz_dir cache/lndb_volume_npz_agr1
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from detection.retinanet.config import RetinaNetConfig
from detection.retinanet.dataset import prepare_datalist, load_npz_as_image, mask_to_boxes_3d
from detection.retinanet.trainer import NoduleDetectionDataset

logger = logging.getLogger(__name__)


def draw_boxes_on_slice(ax, boxes, slice_idx, axis="z", color="lime"):
    """
    Draw 2D rectangles for boxes that intersect the given slice.
    boxes: (N, 6) [x1, y1, z1, x2, y2, z2] in voxel coords.
    axis: which axis this slice is along ('z'=axial, 'y'=coronal, 'x'=sagittal)
    """
    for box in boxes:
        x1, y1, z1, x2, y2, z2 = box

        if axis == "z":
            if z1 <= slice_idx < z2:
                rect = mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor="none"
                )
                ax.add_patch(rect)
        elif axis == "y":
            if y1 <= slice_idx < y2:
                rect = mpatches.Rectangle(
                    (x1, z1), x2 - x1, z2 - z1,
                    linewidth=2, edgecolor=color, facecolor="none"
                )
                ax.add_patch(rect)
        elif axis == "x":
            if x1 <= slice_idx < x2:
                rect = mpatches.Rectangle(
                    (y1, z1), y2 - y1, z2 - z1,
                    linewidth=2, edgecolor=color, facecolor="none"
                )
                ax.add_patch(rect)


def visualize_sample(idx, datalist_item, dataset, output_dir, patch_size):
    """Visualize one sample: raw volume + cropped patch with boxes."""
    npz_path = datalist_item["image"]
    raw_boxes = datalist_item["box"]
    raw_labels = datalist_item["label"]

    # Load raw volume
    raw_image = load_npz_as_image(npz_path)  # (1, D, H, W)
    _, D, H, W = raw_image.shape

    # Get cropped sample from dataset
    sample = dataset[idx]
    crop_image = sample["image"].numpy()  # (1, cD, cH, cW)
    crop_boxes = sample["box"].numpy()    # (M, 6)
    crop_labels = sample["label"].numpy()
    _, cD, cH, cW = crop_image.shape

    # Print summary
    print(f"\n{'='*60}")
    print(f"Sample {idx}: {Path(npz_path).name}")
    print(f"{'='*60}")
    print(f"  Raw volume:    shape=({D},{H},{W}), range=[{raw_image.min():.3f}, {raw_image.max():.3f}]")
    print(f"  Raw boxes:     {len(raw_boxes)} nodules")
    for i, b in enumerate(raw_boxes):
        size = [b[3]-b[0], b[4]-b[1], b[5]-b[2]]
        print(f"    [{i}] box={b.tolist()}, size={[f'{s:.0f}' for s in size]}")
    print(f"  Patch size:    {patch_size}")
    print(f"  Cropped patch: shape=({cD},{cH},{cW}), range=[{crop_image.min():.3f}, {crop_image.max():.3f}]")
    print(f"  Cropped boxes: {len(crop_boxes)} nodules (after clipping)")
    for i, b in enumerate(crop_boxes):
        size = [b[3]-b[0], b[4]-b[1], b[5]-b[2]]
        print(f"    [{i}] box={b.tolist()}, size={[f'{s:.1f}' for s in size]}")

    # ─── Figure 1: Raw volume with GT boxes ─────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        f"Sample {idx}: {Path(npz_path).stem}\n"
        f"Raw: ({D}×{H}×{W}), {len(raw_boxes)} boxes → "
        f"Crop: ({cD}×{cH}×{cW}), {len(crop_boxes)} boxes",
        fontsize=14, fontweight="bold",
    )

    # Top row: Raw volume axial slices (evenly spaced)
    raw_vol = raw_image[0]  # (D, H, W)
    for col in range(4):
        ax = axes[0, col]
        if len(raw_boxes) > 0 and col < len(raw_boxes):
            # Show slice through each box center
            bz = int((raw_boxes[col % len(raw_boxes)][2] + raw_boxes[col % len(raw_boxes)][5]) / 2)
        else:
            bz = int(D * (col + 1) / 5)
        bz = min(bz, D - 1)
        ax.imshow(raw_vol[bz], cmap="gray", vmin=0, vmax=1)
        draw_boxes_on_slice(ax, raw_boxes, bz, axis="z", color="lime")
        ax.set_title(f"Raw axial z={bz}", fontsize=10)
        ax.axis("off")

    # Bottom row: Cropped patch axial slices
    crop_vol = crop_image[0]  # (cD, cH, cW)
    for col in range(4):
        ax = axes[1, col]
        if len(crop_boxes) > 0 and col < len(crop_boxes):
            bz = int((crop_boxes[col % len(crop_boxes)][2] + crop_boxes[col % len(crop_boxes)][5]) / 2)
        else:
            bz = int(cD * (col + 1) / 5)
        bz = min(bz, cD - 1)
        ax.imshow(crop_vol[bz], cmap="gray", vmin=0, vmax=1)
        draw_boxes_on_slice(ax, crop_boxes, bz, axis="z", color="red")
        ax.set_title(f"Crop axial z={bz}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out_path = output_dir / f"sample_{idx}_overview.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📸 Saved: {out_path}")

    # ─── Figure 2: Multi-view of cropped patch ──────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle(f"Cropped Patch Multi-View (Sample {idx})", fontsize=13)

    # Axial (z)
    mid_z = cD // 2
    if len(crop_boxes) > 0:
        mid_z = int((crop_boxes[0][2] + crop_boxes[0][5]) / 2)
    mid_z = min(mid_z, cD - 1)
    axes2[0].imshow(crop_vol[mid_z], cmap="gray", vmin=0, vmax=1)
    draw_boxes_on_slice(axes2[0], crop_boxes, mid_z, axis="z", color="red")
    axes2[0].set_title(f"Axial z={mid_z}")

    # Coronal (y)
    mid_y = cH // 2
    if len(crop_boxes) > 0:
        mid_y = int((crop_boxes[0][1] + crop_boxes[0][4]) / 2)
    mid_y = min(mid_y, cH - 1)
    axes2[1].imshow(crop_vol[:, mid_y, :], cmap="gray", vmin=0, vmax=1, aspect="auto")
    draw_boxes_on_slice(axes2[1], crop_boxes, mid_y, axis="y", color="red")
    axes2[1].set_title(f"Coronal y={mid_y}")

    # Sagittal (x)
    mid_x = cW // 2
    if len(crop_boxes) > 0:
        mid_x = int((crop_boxes[0][0] + crop_boxes[0][3]) / 2)
    mid_x = min(mid_x, cW - 1)
    axes2[2].imshow(crop_vol[:, :, mid_x], cmap="gray", vmin=0, vmax=1, aspect="auto")
    draw_boxes_on_slice(axes2[2], crop_boxes, mid_x, axis="x", color="red")
    axes2[2].set_title(f"Sagittal x={mid_x}")

    for ax in axes2:
        ax.axis("off")

    plt.tight_layout()
    out_path2 = output_dir / f"sample_{idx}_multiview.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  📸 Saved: {out_path2}")


def main():
    parser = argparse.ArgumentParser(description="Visualize RetinaNet training data")
    parser.add_argument("--npz_dir", default="cache/lndb_volume_npz_agr1")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--output_dir", default="detection/retinanet/viz_data")
    parser.add_argument("--patch_h", type=int, default=128)
    parser.add_argument("--patch_w", type=int, default=128)
    parser.add_argument("--patch_d", type=int, default=32)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_size = [args.patch_h, args.patch_w, args.patch_d]

    # Prepare datalist
    datalist = prepare_datalist(
        args.npz_dir, split="train",
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, split_seed=42,
    )

    if not datalist:
        print("❌ No data found!")
        return

    # Create dataset with patch cropping
    dataset = NoduleDetectionDataset(
        datalist, patch_size=patch_size, is_train=True,
    )

    # Filter to samples that actually have nodules for better visualization
    nodule_indices = [i for i, d in enumerate(datalist) if len(d["box"]) > 0]
    print(f"\n📊 Dataset summary:")
    print(f"  Total samples: {len(datalist)}")
    print(f"  With nodules:  {len(nodule_indices)}")
    print(f"  Patch size:    {patch_size}")

    # Visualize
    n = min(args.num_samples, len(nodule_indices))
    for i in range(n):
        idx = nodule_indices[i]
        visualize_sample(idx, datalist[idx], dataset, output_dir, patch_size)

    print(f"\n✅ Done! {n} samples visualized → {output_dir}/")


if __name__ == "__main__":
    main()
