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
import torch.nn.functional as F
from torch.utils.data import Dataset
import warnings

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
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        split_seed: int = 42,
        positive_ratio: float = 0.7, # 70% chance to pick nodule center, 30% random crop
        full_volume: bool = False, # If True, return full volume (no cropping)
    ):
        self.npz_dir = Path(npz_dir)
        self.split = split
        self.image_size = image_size
        self.max_depth = max_depth
        self.augmentation = augmentation
        self.split_ratios = split_ratios
        self.split_seed = split_seed
        self.positive_ratio = positive_ratio
        self.full_volume = full_volume
        
        # Load file list
        self.samples = self._load_file_list()
        logger.info(f"📹 Loading {len(self.samples)} samples (split={split})")
    
    def _load_file_list(self) -> List[Path]:
        from .utils import get_split_files
        return get_split_files(
            npz_dir=self.npz_dir,
            split=self.split,
            ratios=self.split_ratios,
            seed=self.split_seed
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def get_patient_ids(self) -> List[str]:
        """Extract unique patient IDs from loaded samples"""
        patient_ids = []
        for npz_path in self.samples:
            try:
                data = np.load(npz_path, allow_pickle=True)
                pid = str(data.get('patient_id', npz_path.stem))
                patient_ids.append(pid)
            except Exception as e:
                logger.warning(f"Failed to load patient ID from {npz_path}: {e}")
                patient_ids.append(npz_path.stem)
        return patient_ids
    
    def get_dataset_info(self) -> Dict:
        """Get dataset statistics including patient IDs"""
        patient_ids = self.get_patient_ids()
        unique_patients = list(set(patient_ids))
        return {
            'split': self.split,
            'total_samples': len(self.samples),
            'unique_patients': len(unique_patients),
            'patient_ids': unique_patients,
            'sample_paths': [str(p.name) for p in self.samples]
        }
    
    def __getitem__(self, idx: int) -> Dict:
        npz_path = self.samples[idx]
        data = np.load(npz_path, allow_pickle=True)
        
        # Load data
        frames = data['frames']  # (D, H, W)
        masks = data['masks']    # (D, H, W)
        center_idx = int(data['center_idx'])
        
        # Load metadata
        slice_indices = data.get('slice_indices')
        spacing = data.get('spacing')
        origin = data.get('origin')
        original_shape = data.get('original_shape')
        
        
        # Determine crop range
        D = len(frames)
        start, end = 0, D
        
        # Random Crop Strategy (Hard Negative Mining)
        use_random_crop = False
        if self.split == 'train' and D > self.max_depth:
            if np.random.rand() > self.positive_ratio:
                 use_random_crop = True
        
        if not self.full_volume:
            if use_random_crop:
                # Random crop from anywhere
                max_start = max(0, D - self.max_depth)
                start = np.random.randint(0, max_start + 1)
                end = min(D, start + self.max_depth)
            elif D > self.max_depth:
                # Center crop (Positive sample)
                half = self.max_depth // 2
                start = max(0, center_idx - half)
                end = min(D, start + self.max_depth)
                if end - start < self.max_depth:
                    if start == 0: end = min(D, self.max_depth)
                    elif end == D: start = max(0, D - self.max_depth)
        
        # Crop (numpy)
        frames = frames[start:end]
        masks = masks[start:end]
        if slice_indices is not None:
             slice_indices = slice_indices[start:end]

        # Convert to Tensor (Float) immediately
        # (D, H, W) -> (1, D, H, W) normalized [0,1]
        frames_t = torch.from_numpy(frames).float().unsqueeze(0) / 255.0
        masks_t = torch.from_numpy(masks).float().unsqueeze(0)
        
        # Resize using torch (much faster than PIL loop)
        frames_t, masks_t = self._resize_video_torch(frames_t, masks_t)
        
        # Augmentation (Torch)
        if self.augmentation and self.split == "train":
            frames_t, masks_t = self._augment_torch(frames_t, masks_t)
        
        result = {
            'image': frames_t,         # (1, D, H, W)
            'mask': masks_t.long(),    # (1, D, H, W)
            'patient_id': str(data.get('patient_id', '')),
            'lesion_id': int(data.get('lesion_id', 0)),
            'npz_path': str(npz_path)
        }
        
        # Add metadata if available
        if slice_indices is not None:
            result['slice_indices'] = torch.tensor(slice_indices, dtype=torch.long)
        if spacing is not None:
            result['spacing'] = torch.tensor(spacing, dtype=torch.float32)
        if origin is not None:
            result['origin'] = torch.tensor(origin, dtype=torch.float32)
        if original_shape is not None:
            result['original_shape'] = torch.tensor(original_shape, dtype=torch.long)
            
        return result
    
    def _resize_video_torch(self, frames, masks):
        """
        Resize using torch interpolation.
        Input: (C, D, H, W)
        """
        D, H, W = frames.shape[1], frames.shape[2], frames.shape[3]
        if H == self.image_size and W == self.image_size:
            return frames, masks
            
        # F.interpolate takes (N, C, D, H, W) or (N, C, H, W)
        # Here we have (1, D, H, W) -> Treat as (1, D, H, W) or usually (N, C, D, H, W)
        # For simple 2D resizing on D slices, we can merge N and D or use 3D interpolate
        
        # 3D Interpolation causes depth mixing? No, if we use trilinear it mixes.
        # We want to resize H, W only, keeping D intact.
        # Reshape to (D, C, H, W) -> (D, 1, H, W)
        
        f_in = frames.permute(1, 0, 2, 3) # (D, 1, H, W)
        m_in = masks.permute(1, 0, 2, 3)  # (D, 1, H, W)
        
        # Bilinear for image
        f_out = F.interpolate(f_in, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        # Nearest for mask
        m_out = F.interpolate(m_in, size=(self.image_size, self.image_size), mode='nearest')
        
        # Permute back to (1, D, H, W)
        return f_out.permute(1, 0, 2, 3), m_out.permute(1, 0, 2, 3)

    def _augment_torch(self, frames, masks):
        """
        Fast GPU-ready augmentation using Torch tensors
        """
        # 1. Flip
        if np.random.rand() > 0.5:
            frames = torch.flip(frames, dims=[3]) # Horizontal (W)
            masks = torch.flip(masks, dims=[3])
        if np.random.rand() > 0.5:
            frames = torch.flip(frames, dims=[2]) # Vertical (H)
            masks = torch.flip(masks, dims=[2])
            
        # 2. Intensity Shift (offset)
        if np.random.rand() > 0.5:
            shift = (np.random.rand() * 0.2 - 0.1) # -0.1 to 0.1
            frames = torch.clamp(frames + shift, 0.0, 1.0)
            
        # 3. Simple 90 degree rotations (fast, no artifacts)
        # Replacing slow scipy rotation with simple 90deg steps
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4) # 1, 2, or 3
            frames = torch.rot90(frames, k, dims=[2, 3])
            masks = torch.rot90(masks, k, dims=[2, 3])
            
        # Note: Removing Affine (Rotate/Scale) for now as it's complex to implement purely in Torch 
        # without introducing interpolation artifacts or needing GridSample logic, 
        # which is overkill if speed is the priority. 
        # Rot90 + Flips covers 8 symmetries which is good for medical data.
        
        return frames, masks


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
    slice_indices_list = []
    has_slice_indices = 'slice_indices' in batch[0]
    
    # Metadata lists
    spacings = []
    origins = []
    original_shapes = []
    
    for s in batch:
        img = s['image'] # (1, D, H, W)
        msk = s['mask']  # (1, D, H, W)
        d = img.shape[1]
        
        if d < max_d:
            pad_d = max_d - d
            # Pad at end
            img = torch.nn.functional.pad(img, (0,0, 0,0, 0,pad_d))
            msk = torch.nn.functional.pad(msk, (0,0, 0,0, 0,pad_d))
            
            if has_slice_indices and 'slice_indices' in s:
                # Pad with -1
                si = s['slice_indices']
                si = torch.cat([si, torch.full((pad_d,), -1, dtype=si.dtype)])
                slice_indices_list.append(si)
        else:
            if has_slice_indices and 'slice_indices' in s:
                slice_indices_list.append(s['slice_indices'])
            
        images.append(img)
        masks.append(msk)
        
        if 'spacing' in s: spacings.append(s['spacing'])
        if 'origin' in s: origins.append(s['origin'])
        if 'original_shape' in s: original_shapes.append(s['original_shape'])
    
    result = {
        'image': torch.stack(images), # (B, 1, D, H, W)
        'mask': torch.stack(masks),   # (B, 1, D, H, W)
        'patient_id': [s['patient_id'] for s in batch],
        'npz_path': [s['npz_path'] for s in batch]
    }
    
    if slice_indices_list:
        result['slice_indices'] = torch.stack(slice_indices_list) # (B, D)
    if spacings:
        result['spacing'] = torch.stack(spacings)
    if origins:
        result['origin'] = torch.stack(origins)
    if original_shapes:
        result['original_shape'] = torch.stack(original_shapes)
        
    return result
