#!/usr/bin/env python3
"""
3D Video Dataset
================

Dataset for loading 3D video/volume samples for 3D U-Net training.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)

class VolumetricDataset(Dataset):
    """
    Volumetric Lesion Dataset for 3D U-Net.
    Loads NPZ files containing (D, H, W) volume and masks.
    """
    
    def __init__(
        self,
        npz_dir: str,
        split: str = "train",
        image_size: int = 256, # 3D U-Net usually works on smaller patches to fit in memory
        max_depth: int = 32,
        augmentation: bool = False,
    ):
        self.npz_dir = Path(npz_dir)
        self.split = split
        self.image_size = image_size
        self.max_depth = max_depth
        self.augmentation = augmentation
        
        # Load file list
        self.samples = self._load_file_list()
        logger.info(f"📹 Loading {len(self.samples)} samples (split={split})")
    
    def _load_file_list(self) -> List[Path]:
        split_dir = self.npz_dir / self.split
        if not split_dir.exists():
            split_dir = self.npz_dir
        
        npz_files = sorted(split_dir.glob("*.npz"))
        if not npz_files:
            logger.warning(f"⚠️ No NPZ files found in: {split_dir}")
        return npz_files
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        npz_path = self.samples[idx]
        data = np.load(npz_path, allow_pickle=True)
        
        # Load data
        frames = data['frames']  # (D, H, W)
        masks = data['masks']    # (D, H, W)
        center_idx = int(data['center_idx'])
        
        # Truncate/Pad to max_depth
        D = len(frames)
        if D > self.max_depth:
            # Crop around center
            half = self.max_depth // 2
            start = max(0, center_idx - half)
            end = min(D, start + self.max_depth)
            # Adjust if at boundaries
            if end - start < self.max_depth:
                if start == 0:
                    end = min(D, self.max_depth)
                elif end == D:
                    start = max(0, D - self.max_depth)
            
            frames = frames[start:end]
            masks = masks[start:end]
        
        # Resize
        frames, masks = self._resize_video(frames, masks)
        
        # Augmentation (ToDo)
        if self.augmentation and self.split == "train":
            frames, masks = self._augment(frames, masks)
        
        # Normalize to 0-1
        frames = frames.astype(np.float32) / 255.0
        
        # To Tensor (1, D, H, W)
        frames_tensor = torch.from_numpy(frames).unsqueeze(0)
        masks_tensor = torch.from_numpy(masks).long().unsqueeze(0) # (1, D, H, W) for consistency or (D, H, W)? UNet expects (B, 1, D, H, W) input, Mask (B, D, H, W) or (B, 1, D, H, W).
        # Standard Loss usually expects (B, D, H, W) for target if indices, or (B, 1, D, H, W) if BceWithLogits.
        # Let's return masks as (1, D, H, W) to match model output channel 1.
        
        return {
            'image': frames_tensor,
            'mask': masks_tensor,
            'patient_id': str(data.get('patient_id', '')),
            'lesion_id': int(data.get('lesion_id', 0)),
            'npz_path': str(npz_path)
        }
    
    def _resize_video(self, frames, masks):
        D, H, W = frames.shape
        if H == self.image_size and W == self.image_size:
            return frames, masks
        
        r_frames = np.zeros((D, self.image_size, self.image_size), dtype=frames.dtype)
        r_masks = np.zeros((D, self.image_size, self.image_size), dtype=masks.dtype)
        
        for i in range(D):
            f_img = Image.fromarray(frames[i])
            m_img = Image.fromarray(masks[i])
            r_frames[i] = np.array(f_img.resize((self.image_size, self.image_size), Image.BILINEAR))
            r_masks[i] = np.array(m_img.resize((self.image_size, self.image_size), Image.NEAREST))
            
        return r_frames, r_masks

    def _augment(self, frames, masks):
        # Simple flip
        if np.random.rand() > 0.5:
            frames = np.flip(frames, axis=2) # Horizontal
            masks = np.flip(masks, axis=2)
        if np.random.rand() > 0.5:
            frames = np.flip(frames, axis=1) # Vertical
            masks = np.flip(masks, axis=1)
        return frames.copy(), masks.copy()


def collate_video_batch(batch: List[Dict]) -> Dict:
    """
    Collate batch with variable Depth D.
    Pads to max D in batch.
    """
    if not batch:
        return {}
        
    max_d = max(s['image'].shape[1] for s in batch)
    
    images = []
    masks = []
    
    for s in batch:
        img = s['image'] # (1, D, H, W)
        msk = s['mask']  # (1, D, H, W)
        d = img.shape[1]
        
        if d < max_d:
            pad_d = max_d - d
            # Pad at end
            img = torch.nn.functional.pad(img, (0,0, 0,0, 0,pad_d))
            msk = torch.nn.functional.pad(msk, (0,0, 0,0, 0,pad_d))
            
        images.append(img)
        masks.append(msk)
    
    return {
        'image': torch.stack(images), # (B, 1, D, H, W)
        'mask': torch.stack(masks),   # (B, 1, D, H, W)
        'patient_id': [s['patient_id'] for s in batch],
        'npz_path': [s['npz_path'] for s in batch]
    }
