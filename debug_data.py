
import os
import numpy as np
import torch
from segmentation.train_3dunet.dataset import VolumetricDataset
from PIL import Image

def debug_dataset():
    # Settings matching your command
    npz_dir = "cache/volume_npz"
    split = "train"
    image_size = 256
    max_depth = 32
    
    print(f"🔍 Checking dataset in: {npz_dir}")
    
    try:
        dataset = VolumetricDataset(
            npz_dir=npz_dir,
            split=split,
            image_size=image_size,
            max_depth=max_depth
        )
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    if len(dataset) == 0:
        print("❌ Dataset is empty!")
        return
        
    print(f"✅ Found {len(dataset)} samples.")
    
    # Check first few samples
    for i in range(min(3, len(dataset))):
        print(f"\n📦 Sample {i}:")
        sample = dataset[i]
        
        img = sample['image'] # (1, D, H, W)
        mask = sample['mask'] # (1, D, H, W)
        
        print(f"   Image Shape: {img.shape}, Range: {img.min():.4f} - {img.max():.4f}")
        print(f"   Mask Shape: {mask.shape}, Unique Values: {torch.unique(mask)}")
        
        if mask.sum() == 0:
            print("   ⚠️ WARNING: Mask is empty (all zeros)!")
        else:
            print(f"   Mask pixels: {mask.sum()} / {mask.numel()}")
            
        # Save center slice
        D = img.shape[1]
        center = D // 2
        
        img_slice = (img[0, center].numpy() * 255).astype(np.uint8)
        mask_slice = (mask[0, center].numpy() * 255).astype(np.uint8)
        
        os.makedirs("debug_vis", exist_ok=True)
        Image.fromarray(img_slice).save(f"debug_vis/sample_{i}_img.png")
        Image.fromarray(mask_slice).save(f"debug_vis/sample_{i}_mask.png")
        print(f"   💾 Saved slice {center} visualization to debug_vis/")

if __name__ == "__main__":
    debug_dataset()
