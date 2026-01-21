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
import scipy.ndimage
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
        # Ensure positive strides for torch
        frames = np.ascontiguousarray(frames)
        masks = np.ascontiguousarray(masks)
        
        frames_tensor = torch.from_numpy(frames).unsqueeze(0)
        masks_tensor = torch.from_numpy(masks).long().unsqueeze(0)
        
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
        """
        3D Data Augmentation
        - Random Flip (H/V)
        - Random Rotate (-15, 15 degrees)
        - Random Scale (0.8, 1.2)
        - Random Intensity Shift
        """
        # 1. Flip
        if np.random.rand() > 0.5:
            frames = np.flip(frames, axis=2) # Horizontal
            masks = np.flip(masks, axis=2)
        if np.random.rand() > 0.5:
            frames = np.flip(frames, axis=1) # Vertical
            masks = np.flip(masks, axis=1)
            
        # 2. Random Rotate (affine on H-W plane)
        # We rotate the whole volume along the Z-axis (depth)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            # scipy.ndimage.rotate rotates in the plane defined by axes. axes=(1,2) is H-W plane
            # reshape=False to keep original size
            frames = scipy.ndimage.rotate(frames, angle, axes=(1, 2), reshape=False, mode='nearest')
            masks = scipy.ndimage.rotate(masks, angle, axes=(1, 2), reshape=False, order=0, mode='constant', cval=0)
            
        # 3. Random Scale (Zoom)
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            # We want to scale H and W, but keep D same usually, or scale all? 
            # Usually for fixed input size models, scaling is tricky if we don't pad/crop back.
            # But here we resize AFTER augmentation? No, wait.
            # The pipeline is: Load -> Crop Depth -> Resize -> Augment -> Normalize.
            # So frames is (D, 256, 256).
            # If we zoom, the shape changes. We must crop or pad back to 256x256.
            
            # Let's use a simpler approach for now: modifying the resize step or just doing random crop/pad is standard.
            # But since we already resized, doing affine zoom means we crop center or pad.
            
            # Using scipy zoom
            # Zoom factors: (1, scale, scale) -> Keep depth, scale spatial
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frames_zoomed = scipy.ndimage.zoom(frames, (1, scale, scale), order=1)
                masks_zoomed = scipy.ndimage.zoom(masks, (1, scale, scale), order=0)
            
            # Crop or Pad back to (D, 256, 256)
            d, h, w = frames.shape
            zd, zh, zw = frames_zoomed.shape
            
            # Create container
            new_frames = np.zeros_like(frames)
            new_masks = np.zeros_like(masks)
            
            # Calculate offsets to center
            dh = (zh - h) // 2
            dw = (zw - w) // 2
            
            # Source slice
            src_y1 = max(0, dh)
            src_y2 = min(zh, dh + h)
            src_x1 = max(0, dw)
            src_x2 = min(zw, dw + w)
            
            # Target slice
            dst_y1 = max(0, -dh)
            dst_y2 = min(h, h - dh) # This logic can be tricky, let's simplify:
            
            # If zoomed in (scale > 1), crop center.
            if scale > 1.0:
                 start_y = (zh - h) // 2
                 start_x = (zw - w) // 2
                 new_frames = frames_zoomed[:, start_y:start_y+h, start_x:start_x+w]
                 new_masks = masks_zoomed[:, start_y:start_y+h, start_x:start_x+w]
            else:
                # If zoomed out (scale < 1), pad center
                start_y = (h - zh) // 2
                start_x = (w - zw) // 2
                new_frames[:, start_y:start_y+zh, start_x:start_x+zw] = frames_zoomed
                new_masks[:, start_y:start_y+zh, start_x:start_x+zw] = masks_zoomed
                
            frames = new_frames
            masks = new_masks

        # 4. Intensity Shift
        if np.random.rand() > 0.5:
            shift = np.random.uniform(-0.1, 0.1) * 255.0 # Since it's still 0-255 range here roughly?
            # Actually input is uint8? No, let's check.
            # In __getitem__, frames is loaded from npz. Preprocess saves as uint8. 
            # So frames is uint8 (0-255).
            frames = frames.astype(np.float32) + shift
            frames = np.clip(frames, 0, 255) # Keep in range
        
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
