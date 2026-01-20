#!/usr/bin/env python3
"""
Training Data Visualization Tool
=================================

Visualize NPZ training data for 3D U-Net.
Supports:
- Interactive slice-by-slice viewing
- Overlay visualization (CT + mask)
- Grid view of multiple slices
- Statistics and distribution analysis
- Export to image/video
- **Dataset mode**: View data EXACTLY as fed to training (after Dataset transforms)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NPZVisualizer:
    """Visualizer for NPZ training data"""
    
    def __init__(self, npz_dir: str, split: str = "train"):
        self.npz_dir = Path(npz_dir)
        self.split = split
        self.samples = self._load_samples()
        self.current_idx = 0
        logger.info(f"📂 Loaded {len(self.samples)} samples from {split} split")
    
    def _load_samples(self) -> List[Path]:
        """Load sample list from split directory"""
        split_dir = self.npz_dir / self.split
        if not split_dir.exists():
            split_dir = self.npz_dir
        
        npz_files = sorted(split_dir.glob("*.npz"))
        if not npz_files:
            logger.warning(f"⚠️ No NPZ files found in: {split_dir}")
        return npz_files
    
    def load_sample(self, idx: int) -> dict:
        """Load a single sample"""
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Sample index {idx} out of range [0, {len(self.samples)})")
        
        npz_path = self.samples[idx]
        data = np.load(npz_path, allow_pickle=True)
        
        return {
            'frames': data['frames'],
            'masks': data['masks'],
            'center_idx': int(data['center_idx']),
            'patient_id': str(data.get('patient_id', '')),
            'lesion_id': int(data.get('lesion_id', 0)),
            'path': str(npz_path)
        }
    
    def print_statistics(self):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("📊 DATASET STATISTICS")
        print("="*60)
        
        total_samples = len(self.samples)
        print(f"📂 Split: {self.split}")
        print(f"📁 Directory: {self.npz_dir / self.split}")
        print(f"📑 Total samples: {total_samples}")
        
        if total_samples == 0:
            return
        
        # Sample some files for stats
        sample_count = min(total_samples, 50)
        depths = []
        heights = []
        widths = []
        mask_ratios = []
        has_mask = 0
        
        print(f"\n📈 Analyzing {sample_count} samples...")
        
        for i in range(sample_count):
            data = self.load_sample(i)
            frames = data['frames']
            masks = data['masks']
            
            depths.append(frames.shape[0])
            heights.append(frames.shape[1])
            widths.append(frames.shape[2])
            
            mask_sum = masks.sum()
            total_voxels = masks.size
            mask_ratios.append(mask_sum / total_voxels * 100)
            
            if mask_sum > 0:
                has_mask += 1
        
        print(f"\n📐 Volume Dimensions:")
        print(f"   Depth (D):  min={min(depths)}, max={max(depths)}, mean={np.mean(depths):.1f}")
        print(f"   Height (H): min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}")
        print(f"   Width (W):  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}")
        
        print(f"\n🎯 Mask Statistics:")
        print(f"   Samples with masks: {has_mask}/{sample_count} ({has_mask/sample_count*100:.1f}%)")
        print(f"   Mask ratio: min={min(mask_ratios):.4f}%, max={max(mask_ratios):.4f}%, mean={np.mean(mask_ratios):.4f}%")
        print("="*60 + "\n")
    
    def visualize_sample(self, idx: int, save_path: Optional[str] = None):
        """Visualize a single sample with grid view"""
        data = self.load_sample(idx)
        frames = data['frames']
        masks = data['masks']
        center_idx = data['center_idx']
        
        D, H, W = frames.shape
        
        # Calculate grid size (show up to 16 slices)
        n_show = min(D, 16)
        indices = np.linspace(0, D-1, n_show, dtype=int)
        
        # Create figure
        cols = 4
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        axes = axes.flatten() if n_show > 1 else [axes]
        
        fig.suptitle(
            f"Sample: {Path(data['path']).stem}\n"
            f"Patient: {data['patient_id']} | Lesion: {data['lesion_id']} | "
            f"Shape: {D}×{H}×{W} | Center: {center_idx}",
            fontsize=12, fontweight='bold'
        )
        
        for ax_idx, (ax, slice_idx) in enumerate(zip(axes, indices)):
            frame = frames[slice_idx]
            mask = masks[slice_idx]
            
            # Show CT image
            ax.imshow(frame, cmap='gray', vmin=0, vmax=255)
            
            # Overlay mask with transparency
            if mask.sum() > 0:
                mask_overlay = np.ma.masked_where(mask == 0, mask)
                ax.imshow(mask_overlay, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
            
            # Highlight center slice
            title = f"Slice {slice_idx}"
            if slice_idx == center_idx:
                title += " ★"
                ax.set_title(title, fontsize=10, fontweight='bold', color='red')
            else:
                ax.set_title(title, fontsize=10)
            
            ax.axis('off')
        
        # Hide unused axes
        for ax in axes[len(indices):]:
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"💾 Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def interactive_viewer(self, idx: int):
        """Interactive slice-by-slice viewer with slider"""
        data = self.load_sample(idx)
        frames = data['frames']
        masks = data['masks']
        center_idx = data['center_idx']
        
        D, H, W = frames.shape
        
        # Create figure with subplots
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(2, 3, height_ratios=[10, 1], width_ratios=[1, 1, 1])
        
        ax_ct = fig.add_subplot(gs[0, 0])
        ax_mask = fig.add_subplot(gs[0, 1])
        ax_overlay = fig.add_subplot(gs[0, 2])
        ax_slider = fig.add_subplot(gs[1, :])
        
        fig.suptitle(
            f"📁 {Path(data['path']).stem}\n"
            f"Patient: {data['patient_id']} | Lesion: {data['lesion_id']} | Shape: {D}×{H}×{W}",
            fontsize=11
        )
        
        # Initial display
        current_slice = [center_idx]  # Use list to allow modification in closure
        
        im_ct = ax_ct.imshow(frames[current_slice[0]], cmap='gray', vmin=0, vmax=255)
        ax_ct.set_title("CT Image")
        ax_ct.axis('off')
        
        im_mask = ax_mask.imshow(masks[current_slice[0]], cmap='Reds', vmin=0, vmax=1)
        ax_mask.set_title("Mask")
        ax_mask.axis('off')
        
        # Overlay
        overlay = self._create_overlay(frames[current_slice[0]], masks[current_slice[0]])
        im_overlay = ax_overlay.imshow(overlay)
        ax_overlay.set_title("Overlay")
        ax_overlay.axis('off')
        
        # Create slider
        slider = Slider(
            ax=ax_slider,
            label='Slice',
            valmin=0,
            valmax=D-1,
            valinit=center_idx,
            valstep=1
        )
        
        # Info text
        info_text = ax_slider.text(
            1.02, 0.5, f"Slice {current_slice[0]} | Mask pixels: {masks[current_slice[0]].sum()}",
            transform=ax_slider.transAxes, fontsize=10, verticalalignment='center'
        )
        
        def update(val):
            slice_idx = int(slider.val)
            current_slice[0] = slice_idx
            
            im_ct.set_data(frames[slice_idx])
            im_mask.set_data(masks[slice_idx])
            
            overlay = self._create_overlay(frames[slice_idx], masks[slice_idx])
            im_overlay.set_data(overlay)
            
            mask_pixels = masks[slice_idx].sum()
            center_marker = " ★ CENTER" if slice_idx == center_idx else ""
            info_text.set_text(f"Slice {slice_idx}{center_marker} | Mask pixels: {mask_pixels}")
            
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        
        plt.tight_layout()
        plt.show()
    
    def _create_overlay(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create RGB overlay of CT and mask"""
        # Normalize frame to 0-1
        frame_norm = frame.astype(np.float32) / 255.0
        
        # Create RGB image
        overlay = np.stack([frame_norm, frame_norm, frame_norm], axis=-1)
        
        # Add red overlay where mask is positive
        if mask.sum() > 0:
            overlay[mask > 0, 0] = np.clip(overlay[mask > 0, 0] + 0.5, 0, 1)  # Red
            overlay[mask > 0, 1] = overlay[mask > 0, 1] * 0.5  # Reduce green
            overlay[mask > 0, 2] = overlay[mask > 0, 2] * 0.5  # Reduce blue
        
        return overlay
    
    def browse_samples(self):
        """Browse through all samples interactively"""
        if len(self.samples) == 0:
            print("❌ No samples to browse")
            return
        
        print("\n" + "="*60)
        print("🔍 INTERACTIVE BROWSER")
        print("="*60)
        print("Commands:")
        print("  [n]    - Next sample")
        print("  [p]    - Previous sample")
        print("  [g N]  - Go to sample N")
        print("  [v]    - View current sample (grid)")
        print("  [i]    - Interactive viewer (slider)")
        print("  [s]    - Save current sample as image")
        print("  [info] - Sample info")
        print("  [q]    - Quit")
        print("="*60 + "\n")
        
        while True:
            print(f"\n>>> Sample {self.current_idx + 1}/{len(self.samples)}: {self.samples[self.current_idx].stem}")
            cmd = input("Command: ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == 'n':
                self.current_idx = (self.current_idx + 1) % len(self.samples)
            elif cmd == 'p':
                self.current_idx = (self.current_idx - 1) % len(self.samples)
            elif cmd.startswith('g '):
                try:
                    idx = int(cmd.split()[1]) - 1
                    if 0 <= idx < len(self.samples):
                        self.current_idx = idx
                    else:
                        print(f"❌ Index out of range [1, {len(self.samples)}]")
                except ValueError:
                    print("❌ Invalid index")
            elif cmd == 'v':
                self.visualize_sample(self.current_idx)
            elif cmd == 'i':
                self.interactive_viewer(self.current_idx)
            elif cmd == 's':
                save_path = f"sample_{self.current_idx:04d}.png"
                self.visualize_sample(self.current_idx, save_path)
            elif cmd == 'info':
                data = self.load_sample(self.current_idx)
                print(f"  Path: {data['path']}")
                print(f"  Patient ID: {data['patient_id']}")
                print(f"  Lesion ID: {data['lesion_id']}")
                print(f"  Shape: {data['frames'].shape}")
                print(f"  Center idx: {data['center_idx']}")
                print(f"  Mask sum: {data['masks'].sum()}")
                print(f"  Mask ratio: {data['masks'].sum() / data['masks'].size * 100:.4f}%")
    
    def visualize_batch(self, n_samples: int = 9, save_path: Optional[str] = None):
        """Visualize multiple samples in a grid (center slices only)"""
        n_show = min(n_samples, len(self.samples))
        
        cols = 3
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
        axes = axes.flatten() if n_show > 1 else [axes]
        
        fig.suptitle(f"Dataset Overview ({self.split} split) - Center Slices", fontsize=14, fontweight='bold')
        
        for ax_idx, ax in enumerate(axes):
            if ax_idx >= n_show:
                ax.axis('off')
                continue
            
            data = self.load_sample(ax_idx)
            frames = data['frames']
            masks = data['masks']
            center_idx = data['center_idx']
            
            # Show center slice
            frame = frames[center_idx]
            mask = masks[center_idx]
            
            overlay = self._create_overlay(frame, mask)
            ax.imshow(overlay)
            
            mask_ratio = masks.sum() / masks.size * 100
            ax.set_title(
                f"{Path(data['path']).stem}\n"
                f"Shape: {frames.shape[0]}×{frames.shape[1]}×{frames.shape[2]} | Mask: {mask_ratio:.2f}%",
                fontsize=8
            )
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"💾 Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_gif(self, idx: int, save_path: str, fps: int = 5):
        """Export sample as GIF animation"""
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            print("❌ Please install Pillow: pip install Pillow")
            return
        
        data = self.load_sample(idx)
        frames = data['frames']
        masks = data['masks']
        D = frames.shape[0]
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')
        
        overlay = self._create_overlay(frames[0], masks[0])
        im = ax.imshow(overlay)
        title = ax.set_title(f"Slice 0/{D-1}")
        
        def update(frame_idx):
            overlay = self._create_overlay(frames[frame_idx], masks[frame_idx])
            im.set_data(overlay)
            title.set_text(f"Slice {frame_idx}/{D-1}")
            return [im, title]
        
        anim = FuncAnimation(fig, update, frames=D, interval=1000//fps, blit=True)
        anim.save(save_path, writer=PillowWriter(fps=fps))
        logger.info(f"💾 Saved GIF to {save_path}")
        plt.close()


class DatasetVisualizer:
    """
    Visualizer that uses VolumetricDataset to show data EXACTLY as fed to training.
    This includes: resize to image_size, depth cropping, normalization to [0,1].
    """
    
    def __init__(self, npz_dir: str, split: str = "train", image_size: int = 256, max_depth: int = 32):
        self.npz_dir = npz_dir
        self.split = split
        self.image_size = image_size
        self.max_depth = max_depth
        
        # Import Dataset
        try:
            from segmentation.train_3dunet.dataset import VolumetricDataset
            self.dataset = VolumetricDataset(
                npz_dir=npz_dir,
                split=split,
                image_size=image_size,
                max_depth=max_depth,
                augmentation=False  # No augmentation for visualization
            )
            logger.info(f"📊 Loaded Dataset with {len(self.dataset)} samples")
        except ImportError as e:
            logger.error(f"❌ Failed to import VolumetricDataset: {e}")
            self.dataset = None
    
    def print_statistics(self):
        """Print statistics for data as fed to training"""
        if self.dataset is None or len(self.dataset) == 0:
            print("❌ No dataset loaded")
            return
        
        print("\n" + "="*70)
        print("📊 TRAINING DATA STATISTICS (After Dataset Transforms)")
        print("="*70)
        print(f"📂 Split: {self.split}")
        print(f"📁 Source: {self.npz_dir}")
        print(f"📑 Total samples: {len(self.dataset)}")
        print(f"🎯 Image size: {self.image_size}×{self.image_size}")
        print(f"📏 Max depth: {self.max_depth}")
        
        # Analyze samples
        n_analyze = min(len(self.dataset), 50)
        print(f"\n📈 Analyzing {n_analyze} samples...")
        
        depths = []
        mask_ratios = []
        mask_sums = []
        has_mask = 0
        
        for i in range(n_analyze):
            sample = self.dataset[i]
            image = sample['image']  # (1, D, H, W) tensor, normalized [0,1]
            mask = sample['mask']    # (1, D, H, W) tensor
            
            D = image.shape[1]
            depths.append(D)
            
            mask_np = mask.numpy()
            mask_sum = mask_np.sum()
            mask_sums.append(mask_sum)
            mask_ratios.append(mask_sum / mask_np.size * 100)
            
            if mask_sum > 0:
                has_mask += 1
        
        print(f"\n📐 Volume Dimensions (After Processing):")
        print(f"   Depth (D):  min={min(depths)}, max={max(depths)}, mean={np.mean(depths):.1f}")
        print(f"   Height (H): {self.image_size} (fixed)")
        print(f"   Width (W):  {self.image_size} (fixed)")
        
        print(f"\n🎯 Mask Statistics (Training Input):")
        print(f"   Samples with masks: {has_mask}/{n_analyze} ({has_mask/n_analyze*100:.1f}%)")
        print(f"   Mask voxel count: min={min(mask_sums):.0f}, max={max(mask_sums):.0f}, mean={np.mean(mask_sums):.1f}")
        print(f"   Mask ratio: min={min(mask_ratios):.6f}%, max={max(mask_ratios):.4f}%, mean={np.mean(mask_ratios):.6f}%")
        
        # Show class imbalance warning
        mean_ratio = np.mean(mask_ratios)
        if mean_ratio < 0.1:
            print(f"\n⚠️  WARNING: Severe class imbalance! Mean mask ratio is only {mean_ratio:.4f}%")
            print(f"   Consider: 1) Crop to lesion region, 2) Use weighted loss, 3) Oversample positive")
        
        print("="*70 + "\n")
    
    def visualize_sample(self, idx: int, save_path: Optional[str] = None):
        """Visualize a sample as it would be fed to the model"""
        if self.dataset is None:
            print("❌ No dataset loaded")
            return
        
        sample = self.dataset[idx]
        image = sample['image'].numpy()[0]  # (D, H, W), normalized [0,1]
        mask = sample['mask'].numpy()[0]    # (D, H, W)
        patient_id = sample['patient_id']
        npz_path = sample['npz_path']
        
        D, H, W = image.shape
        
        # Show grid of slices
        n_show = min(D, 16)
        indices = np.linspace(0, D-1, n_show, dtype=int)
        
        cols = 4
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        axes = axes.flatten()
        
        mask_ratio = mask.sum() / mask.size * 100
        fig.suptitle(
            f"🔥 TRAINING DATA (After Dataset Transforms)\n"
            f"Sample: {Path(npz_path).stem} | Patient: {patient_id}\n"
            f"Shape: {D}×{H}×{W} | Normalized: [0,1] | Mask ratio: {mask_ratio:.4f}%",
            fontsize=11, fontweight='bold'
        )
        
        for ax_idx, (ax, slice_idx) in enumerate(zip(axes, indices)):
            frame = image[slice_idx]  # Already [0,1]
            msk = mask[slice_idx]
            
            # Create overlay
            overlay = np.stack([frame, frame, frame], axis=-1)
            if msk.sum() > 0:
                overlay[msk > 0, 0] = np.clip(overlay[msk > 0, 0] + 0.4, 0, 1)
                overlay[msk > 0, 1] *= 0.5
                overlay[msk > 0, 2] *= 0.5
            
            ax.imshow(overlay)
            
            msk_pixels = msk.sum()
            title = f"Slice {slice_idx}"
            if msk_pixels > 0:
                title += f" ({int(msk_pixels)} px)"
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        
        # Hide unused
        for ax in axes[len(indices):]:
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"💾 Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def interactive_viewer(self, idx: int):
        """Interactive viewer for training data"""
        if self.dataset is None:
            print("❌ No dataset loaded")
            return
        
        sample = self.dataset[idx]
        image = sample['image'].numpy()[0]  # (D, H, W)
        mask = sample['mask'].numpy()[0]
        
        D, H, W = image.shape
        
        fig = plt.figure(figsize=(15, 6))
        gs = gridspec.GridSpec(2, 4, height_ratios=[10, 1])
        
        ax_ct = fig.add_subplot(gs[0, 0])
        ax_mask = fig.add_subplot(gs[0, 1])
        ax_overlay = fig.add_subplot(gs[0, 2])
        ax_hist = fig.add_subplot(gs[0, 3])
        ax_slider = fig.add_subplot(gs[1, :])
        
        mask_ratio = mask.sum() / mask.size * 100
        fig.suptitle(
            f"🔥 Training Data Viewer | {sample['patient_id']}\n"
            f"Shape: {D}×{H}×{W} | Range: [{image.min():.2f}, {image.max():.2f}] | Mask: {mask_ratio:.4f}%",
            fontsize=11
        )
        
        current_slice = [D // 2]
        
        # CT (normalized 0-1)
        im_ct = ax_ct.imshow(image[current_slice[0]], cmap='gray', vmin=0, vmax=1)
        ax_ct.set_title("CT (normalized 0-1)")
        ax_ct.axis('off')
        
        # Mask
        im_mask = ax_mask.imshow(mask[current_slice[0]], cmap='Reds', vmin=0, vmax=1)
        ax_mask.set_title("Mask")
        ax_mask.axis('off')
        
        # Overlay
        def create_overlay(frame, msk):
            overlay = np.stack([frame, frame, frame], axis=-1)
            if msk.sum() > 0:
                overlay[msk > 0, 0] = np.clip(overlay[msk > 0, 0] + 0.4, 0, 1)
                overlay[msk > 0, 1] *= 0.5
                overlay[msk > 0, 2] *= 0.5
            return overlay
        
        im_overlay = ax_overlay.imshow(create_overlay(image[current_slice[0]], mask[current_slice[0]]))
        ax_overlay.set_title("Overlay")
        ax_overlay.axis('off')
        
        # Histogram
        ax_hist.hist(image.flatten(), bins=50, color='steelblue', alpha=0.7)
        ax_hist.set_title("Intensity Distribution")
        ax_hist.set_xlabel("Value")
        ax_hist.axvline(x=image[current_slice[0]].mean(), color='red', linestyle='--', label='Slice mean')
        
        # Slider
        slider = Slider(ax_slider, 'Slice', 0, D-1, valinit=current_slice[0], valstep=1)
        
        info_text = ax_slider.text(
            1.02, 0.5, f"Slice {current_slice[0]} | Mask: {mask[current_slice[0]].sum():.0f} px",
            transform=ax_slider.transAxes, fontsize=10, verticalalignment='center'
        )
        
        def update(val):
            s = int(slider.val)
            current_slice[0] = s
            
            im_ct.set_data(image[s])
            im_mask.set_data(mask[s])
            im_overlay.set_data(create_overlay(image[s], mask[s]))
            
            info_text.set_text(f"Slice {s} | Mask: {mask[s].sum():.0f} px")
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        plt.tight_layout()
        plt.show()
    
    def visualize_batch(self, n_samples: int = 9, save_path: Optional[str] = None):
        """Visualize multiple training samples in a grid"""
        if self.dataset is None:
            print("❌ No dataset loaded")
            return
        
        n_show = min(n_samples, len(self.dataset))
        
        cols = 3
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
        axes = axes.flatten()
        
        fig.suptitle(
            f"🔥 Training Data Overview ({self.split})\n"
            f"Size: {self.image_size}×{self.image_size} | Max Depth: {self.max_depth}",
            fontsize=12, fontweight='bold'
        )
        
        for ax_idx, ax in enumerate(axes):
            if ax_idx >= n_show:
                ax.axis('off')
                continue
            
            sample = self.dataset[ax_idx]
            image = sample['image'].numpy()[0]  # (D, H, W)
            mask = sample['mask'].numpy()[0]
            
            D = image.shape[0]
            center_idx = D // 2
            
            # Find slice with most mask
            mask_sums = [mask[i].sum() for i in range(D)]
            best_idx = np.argmax(mask_sums) if max(mask_sums) > 0 else center_idx
            
            frame = image[best_idx]
            msk = mask[best_idx]
            
            # Create overlay
            overlay = np.stack([frame, frame, frame], axis=-1)
            if msk.sum() > 0:
                overlay[msk > 0, 0] = np.clip(overlay[msk > 0, 0] + 0.4, 0, 1)
                overlay[msk > 0, 1] *= 0.5
                overlay[msk > 0, 2] *= 0.5
            
            ax.imshow(overlay)
            
            mask_ratio = mask.sum() / mask.size * 100
            ax.set_title(
                f"{Path(sample['npz_path']).stem}\n"
                f"D={D} | Mask: {mask_ratio:.4f}%",
                fontsize=8
            )
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"💾 Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_augmentation(self, idx: int, n_augments: int = 4, save_path: Optional[str] = None):
        """
        Visualize same sample with multiple augmentations to show augmentation effects.
        Shows original + n_augments augmented versions side by side.
        """
        try:
            from segmentation.train_3dunet.dataset import VolumetricDataset
        except ImportError as e:
            print(f"❌ Failed to import: {e}")
            return
        
        # Create dataset WITHOUT augmentation for original
        dataset_orig = VolumetricDataset(
            npz_dir=self.npz_dir,
            split=self.split,
            image_size=self.image_size,
            max_depth=self.max_depth,
            augmentation=False
        )
        
        # Create dataset WITH augmentation
        dataset_aug = VolumetricDataset(
            npz_dir=self.npz_dir,
            split=self.split,
            image_size=self.image_size,
            max_depth=self.max_depth,
            augmentation=True
        )
        
        # Get original sample
        orig_sample = dataset_orig[idx]
        orig_image = orig_sample['image'].numpy()[0]
        orig_mask = orig_sample['mask'].numpy()[0]
        
        D = orig_image.shape[0]
        
        # Find best slice (with most mask)
        mask_sums = [orig_mask[i].sum() for i in range(D)]
        best_idx = np.argmax(mask_sums) if max(mask_sums) > 0 else D // 2
        
        # Create figure
        cols = n_augments + 1  # Original + augmented versions
        fig, axes = plt.subplots(2, cols, figsize=(4*cols, 8))
        
        fig.suptitle(
            f"🔄 Augmentation Visualization\n"
            f"Sample: {Path(orig_sample['npz_path']).stem} | Slice: {best_idx}",
            fontsize=12, fontweight='bold'
        )
        
        def create_overlay(frame, msk):
            overlay = np.stack([frame, frame, frame], axis=-1)
            if msk.sum() > 0:
                overlay[msk > 0, 0] = np.clip(overlay[msk > 0, 0] + 0.4, 0, 1)
                overlay[msk > 0, 1] *= 0.5
                overlay[msk > 0, 2] *= 0.5
            return overlay
        
        # Show original (column 0)
        axes[0, 0].imshow(orig_image[best_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title("Original CT", fontsize=10, fontweight='bold', color='blue')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(create_overlay(orig_image[best_idx], orig_mask[best_idx]))
        axes[1, 0].set_title(f"Original Overlay\nMask: {orig_mask[best_idx].sum():.0f} px", fontsize=9)
        axes[1, 0].axis('off')
        
        # Show augmented versions
        for aug_idx in range(n_augments):
            # Each call to dataset generates a new random augmentation
            aug_sample = dataset_aug[idx]
            aug_image = aug_sample['image'].numpy()[0]
            aug_mask = aug_sample['mask'].numpy()[0]
            
            # Use same slice index (might be different due to augmentation)
            slice_idx = min(best_idx, aug_image.shape[0] - 1)
            
            col = aug_idx + 1
            
            axes[0, col].imshow(aug_image[slice_idx], cmap='gray', vmin=0, vmax=1)
            axes[0, col].set_title(f"Augmented #{aug_idx+1}", fontsize=10, color='green')
            axes[0, col].axis('off')
            
            axes[1, col].imshow(create_overlay(aug_image[slice_idx], aug_mask[slice_idx]))
            axes[1, col].set_title(f"Mask: {aug_mask[slice_idx].sum():.0f} px", fontsize=9)
            axes[1, col].axis('off')
        
        # Add legend
        fig.text(0.5, 0.02, 
                 "Augmentations: Flip (H/V) | Rotate (-15°~+15°) | Scale (0.8x~1.2x) | Intensity Shift (±10%)",
                 ha='center', fontsize=10, style='italic', color='gray')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"💾 Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D U-Net Training Data")
    parser.add_argument("--npz_dir", type=str, default="../../cache/volume_npz",
                        help="Directory containing NPZ files")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                        help="Data split to visualize")
    parser.add_argument("--mode", type=str, default="stats",
                        choices=["stats", "view", "interactive", "browse", "batch", "gif",
                                 "dataset", "dataset_view", "dataset_batch", "dataset_augment"],
                        help="Visualization mode. Use 'dataset*' modes to see data as fed to training")
    parser.add_argument("--idx", type=int, default=0,
                        help="Sample index for view/interactive/gif modes")
    parser.add_argument("--n_samples", type=int, default=9,
                        help="Number of samples for batch mode")
    parser.add_argument("--n_augments", type=int, default=4,
                        help="Number of augmented versions to show in dataset_augment mode")
    parser.add_argument("--save", type=str, default=None,
                        help="Save path for output image/gif")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for dataset modes (should match training config)")
    parser.add_argument("--max_depth", type=int, default=32,
                        help="Max depth for dataset modes (should match training config)")
    
    args = parser.parse_args()
    
    # Dataset modes - show data as fed to training
    if args.mode.startswith("dataset"):
        viz = DatasetVisualizer(args.npz_dir, args.split, args.image_size, args.max_depth)
        
        if args.mode == "dataset":
            viz.print_statistics()
        elif args.mode == "dataset_view":
            viz.visualize_sample(args.idx, args.save)
        elif args.mode == "dataset_batch":
            viz.visualize_batch(args.n_samples, args.save)
        elif args.mode == "dataset_augment":
            viz.visualize_augmentation(args.idx, args.n_augments, args.save)
        return
    
    # Raw NPZ modes
    viz = NPZVisualizer(args.npz_dir, args.split)
    
    if len(viz.samples) == 0:
        print(f"❌ No samples found in {args.npz_dir}/{args.split}")
        return
    
    # Execute based on mode
    if args.mode == "stats":
        viz.print_statistics()
    elif args.mode == "view":
        viz.visualize_sample(args.idx, args.save)
    elif args.mode == "interactive":
        viz.interactive_viewer(args.idx)
    elif args.mode == "browse":
        viz.browse_samples()
    elif args.mode == "batch":
        viz.visualize_batch(args.n_samples, args.save)
    elif args.mode == "gif":
        save_path = args.save or f"sample_{args.idx:04d}.gif"
        viz.export_gif(args.idx, save_path)


if __name__ == "__main__":
    main()
