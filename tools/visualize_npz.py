#!/usr/bin/env python3
"""
視覺化 preprocessing 結果
用法:
    python tools/visualize_npz.py cache/lndb_volume_npz_agr1_cropped/LNDb-0001_lesion01.npz
    python tools/visualize_npz.py cache/lndb_volume_npz_agr1_cropped  # 隨機選一個
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    
    frames = data['frames']      # (D, H, W) uint8
    masks = data['masks']        # (D, H, W) uint8
    center_idx = int(data['center_idx'])
    patient_id = str(data['patient_id'])
    lesion_id = int(data['lesion_id'])
    diameter = float(data['diameter_mm'])
    bbox = data['bbox']          # [x1, y1, x2, y2]
    spacing = data['spacing']
    
    print(f"📋 Patient: {patient_id}, Lesion: {lesion_id}")
    print(f"   Volume: {frames.shape} ({frames.dtype})")
    print(f"   Diameter: {diameter:.1f}mm")
    print(f"   Center slice: {center_idx}")
    print(f"   Spacing: {spacing}")
    print(f"   BBox: {bbox}")
    print(f"   Mask voxels: {np.sum(masks > 0)}")
    
    # --- 1. Center slice with mask overlay + bbox ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{patient_id} | Lesion {lesion_id} | {diameter:.1f}mm", fontsize=14)
    
    # Show center slice ± 2 slices
    offsets = [-2, -1, 0, 1, 2]
    for idx, offset in enumerate(offsets):
        row, col = divmod(idx, 3)
        ax = axes[row][col]
        s = center_idx + offset
        s = max(0, min(s, frames.shape[0] - 1))
        
        ax.imshow(frames[s], cmap='gray')
        
        # Overlay mask
        mask_slice = masks[s]
        if mask_slice.max() > 0:
            masked = np.ma.masked_where(mask_slice == 0, mask_slice)
            ax.imshow(masked, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
        
        # Draw bbox on center slice
        if offset == 0:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.set_title(f"★ Slice {s} (center)", fontweight='bold', color='green')
        else:
            ax.set_title(f"Slice {s} ({offset:+d})")
        ax.axis('off')
    
    # Bottom-right: sagittal view (middle x)
    ax = axes[1][2]
    mid_x = frames.shape[2] // 2
    sagittal = frames[:, :, mid_x]
    ax.imshow(sagittal.T, cmap='gray', aspect='auto', origin='lower')
    ax.axhline(y=center_idx, color='lime', linewidth=1, linestyle='--')
    ax.set_title(f"Sagittal (x={mid_x})")
    ax.set_xlabel("Slice (Z)")
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # --- 2. Scrollable slice viewer ---
    print("\n🔍 Interactive viewer: scroll through slices with slider")
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.15)
    
    im = ax2.imshow(frames[center_idx], cmap='gray')
    mask_overlay = masks[center_idx].astype(float)
    masked = np.ma.masked_where(mask_overlay == 0, mask_overlay)
    overlay = ax2.imshow(masked, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
    title = ax2.set_title(f"Slice {center_idx}/{frames.shape[0]-1}")
    ax2.axis('off')
    
    from matplotlib.widgets import Slider
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, frames.shape[0]-1, 
                    valinit=center_idx, valstep=1)
    
    def update(val):
        s = int(slider.val)
        im.set_data(frames[s])
        mask_s = masks[s].astype(float)
        masked_s = np.ma.masked_where(mask_s == 0, mask_s)
        overlay.set_data(masked_s)
        title.set_text(f"Slice {s}/{frames.shape[0]-1} | Mask pixels: {np.sum(masks[s] > 0)}")
        fig2.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python tools/visualize_npz.py <path_to_npz_or_dir>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_dir():
        # Random sample from directory
        files = list(path.glob("*.npz"))
        if not files:
            print(f"No .npz files found in {path}")
            sys.exit(1)
        chosen = np.random.choice(files)
        print(f"🎲 Randomly selected: {chosen.name}")
        visualize_npz(str(chosen))
    else:
        visualize_npz(str(path))
