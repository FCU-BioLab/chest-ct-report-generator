#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Faster R-CNN Dataset
=======================

Loads preprocessed 3D volumes (.npz) and prepares them for training.
Includes Random Crop to fixed size for batching.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random

class LungNodule3DDataset(Dataset):
    """
    3D Dataset for loading full volumes or crops.
    Input: Preprocessed .npz files with 'image' (D, H, W) and 'boxes' (N, 6).
    """
    def __init__(self, 
                 data_dir: str, 
                 split: str = "train",
                 augment: bool = False,
                 crop_size: Tuple[int, int, int] = (128, 128, 128)): # Increased to 128
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment
        self.crop_size = crop_size
        
        # Load file list
        self.samples = sorted(list(self.data_dir.glob("*.npz")))
        # In a real scenario, filter by split (train/val text files)
        
        print(f"Loaded {len(self.samples)} samples from {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def _crop(self, image, boxes, crop_size):
        D, H, W = image.shape
        d_crop, h_crop, w_crop = crop_size
        
        # Padding if image is smaller than crop
        pad_d = max(0, d_crop - D)
        pad_h = max(0, h_crop - H)
        pad_w = max(0, w_crop - W)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            # Update D, H, W after padding
            D, H, W = image.shape
            
        # Random Crop
        # Improve: Center on a nodule with 50% probability if nodules exist
        if len(boxes) > 0 and random.random() < 0.5:
            # Pick a random box
            idx = random.randint(0, len(boxes) - 1)
            # Boxes are [z, y, x, d, h, w] (center, size) -> wait, dataset loads [z, y, x, d, h, w]
            # My Preprocessor saves: boxes_3d.append([z_new, y_new, x_new, d, h, w])
            # So boxes[idx, :3] is center
            
            z_c, y_c, x_c = boxes[idx, :3]
            
            z1 = int(z_c - d_crop / 2)
            y1 = int(y_c - h_crop / 2)
            x1 = int(x_c - w_crop / 2)
            
            # Add random jitter
            z1 += random.randint(-16, 16)
            y1 += random.randint(-16, 16)
            x1 += random.randint(-16, 16)
        else:
            # Fully random
            z1 = random.randint(0, D - d_crop) if D > d_crop else 0
            y1 = random.randint(0, H - h_crop) if H > h_crop else 0
            x1 = random.randint(0, W - w_crop) if W > w_crop else 0
            
        # Clamp
        z1 = max(0, min(z1, D - d_crop))
        y1 = max(0, min(y1, H - h_crop))
        x1 = max(0, min(x1, W - w_crop))
        
        z2 = z1 + d_crop
        y2 = y1 + h_crop
        x2 = x1 + w_crop
        
        image_crop = image[z1:z2, y1:y2, x1:x2]
        
        # Adjust Boxes
        # Boxes (numpy): [z, y, x, d, h, w] (Center format from npz)
        # Shift Center
        
        if len(boxes) > 0:
            boxes_crop = boxes.copy()
            boxes_crop[:, 0] -= z1
            boxes_crop[:, 1] -= y1
            boxes_crop[:, 2] -= x1
            
            # Filter boxes outside crop
            # Center must be within crop? Or IoU?
            # Simplest: Center must be within [0, crop_size]
            
            valid_mask = (
                (boxes_crop[:, 0] >= 0) & (boxes_crop[:, 0] < d_crop) &
                (boxes_crop[:, 1] >= 0) & (boxes_crop[:, 1] < h_crop) &
                (boxes_crop[:, 2] >= 0) & (boxes_crop[:, 2] < w_crop)
            )
            boxes_crop = boxes_crop[valid_mask]
        else:
            boxes_crop = boxes
            
        return image_crop, boxes_crop

    def _augment(self, image, boxes):
        """
        Apply 3D augmentations to cropped image and boxes.
        Boxes are [z, y, x, d, h, w] (center-size).
        Image is (D, H, W).
        """
        D, H, W = image.shape
        
        # 1. Flip
        # Flip Z
        if random.random() < 0.5:
            image = np.flip(image, axis=0)
            if len(boxes) > 0:
                boxes[:, 0] = D - 1 - boxes[:, 0]
        
        # Flip Y
        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            if len(boxes) > 0:
                boxes[:, 1] = H - 1 - boxes[:, 1]
                
        # Flip X
        if random.random() < 0.5:
            image = np.flip(image, axis=2)
            if len(boxes) > 0:
                boxes[:, 2] = W - 1 - boxes[:, 2]
                
        # 2. Rotate 90 (Axial Plane - YX)
        if random.random() < 0.5:
            k = random.randint(1, 3) # 1=90, 2=180, 3=270
            image = np.rot90(image, k, axes=(1, 2))
            if len(boxes) > 0:
                for _ in range(k):
                    # Rotate 90 deg clockwise in YX plane
                    # (y, x) -> (x, W-1-y) if origin is top-left
                    # But boxes are (center_y, center_x).
                    # W_new = H_old, H_new = W_old. But here H=W usually.
                    
                    old_y = boxes[:, 1].copy()
                    old_x = boxes[:, 2].copy()
                    old_h = boxes[:, 4].copy()
                    old_w = boxes[:, 5].copy()
                    
                    # New X = H - 1 - Old Y (Standard 2D rotation of index 90deg CW or CCW)
                    # np.rot90 is usually CCW.
                    # CCW 90: (y, x) -> (H-1-x, y)
                    
                    boxes[:, 1] = H - 1 - old_x # New Y
                    boxes[:, 2] = old_y # New X
                    boxes[:, 4] = old_w # New H (swapped)
                    boxes[:, 5] = old_h # New W (swapped)
                    
                    # Swap H and W domain if not square?
                    # image shape also flips if not square. But crop is usually cube or square.
                    H, W = W, H
                    
        # 3. Intensity Jitter
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            shift = random.uniform(-0.1, 0.1)
            image = image * scale + shift
            image = np.clip(image, 0, 1)
        
        # 4. Gaussian Noise
        if random.random() < 0.3:
            noise_std = random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1)
        
        # 5. Gaussian Blur (simulates motion/resolution variation)
        if random.random() < 0.2:
            from scipy.ndimage import gaussian_filter
            sigma = random.uniform(0.3, 1.0)
            image = gaussian_filter(image, sigma=sigma)
        
        # 6. Random Gamma Correction
        if random.random() < 0.3:
            gamma = random.uniform(0.8, 1.2)
            image = np.power(image + 1e-8, gamma)
            image = np.clip(image, 0, 1)

        return image, boxes

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        try:
            data = np.load(path)
            image = data['image'] # (D, H, W) float32 0-1
            if np.isnan(image).any():
                 # print(f"Warning: NaN found in {path}. Replacing with 0.")
                 image = np.nan_to_num(image)
            boxes = data['boxes'] # (N, 6) [z, y, x, d, h, w]
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a dummy zero sample to prevent crash
            image = np.zeros(self.crop_size, dtype=np.float32)
            boxes = np.zeros((0, 6), dtype=np.float32)
            
        # Crop to fixed size
        image, boxes = self._crop(image, boxes, self.crop_size)
        
        # Augment (Only for training)
        if self.augment:
             image, boxes = self._augment(image, boxes)
        
        # To Tensor
        # Ensure contiguous array for torch (fixes negative stride from flip)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0) # (1, D, H, W)
        
        # Convert Center-Size (z,y,x, d,h,w) to corners (x1, y1, z1, x2, y2, z2) for R-CNN
        # NOTE: R-CNN expects (x1, y1, z1, x2, y2, z2). 
        # My boxes are (z, y, x, d, h, w)
        
        target = {}
        if len(boxes) > 0:
            # z
            z1 = boxes[:, 0] - boxes[:, 3] / 2
            z2 = boxes[:, 0] + boxes[:, 3] / 2
            
            # y
            y1 = boxes[:, 1] - boxes[:, 4] / 2
            y2 = boxes[:, 1] + boxes[:, 4] / 2
            
            # x
            x1 = boxes[:, 2] - boxes[:, 5] / 2
            x2 = boxes[:, 2] + boxes[:, 5] / 2
            
            # Stack: x1, y1, z1, x2, y2, z2
            boxes_corner = np.stack([x1, y1, z1, x2, y2, z2], axis=1)
            
            # Clip to image boundaries (Crop Size)
            D, H, W = image.shape
            boxes_corner[:, 0] = np.clip(boxes_corner[:, 0], 0, W)
            boxes_corner[:, 1] = np.clip(boxes_corner[:, 1], 0, H)
            boxes_corner[:, 2] = np.clip(boxes_corner[:, 2], 0, D)
            boxes_corner[:, 3] = np.clip(boxes_corner[:, 3], 0, W)
            boxes_corner[:, 4] = np.clip(boxes_corner[:, 4], 0, H)
            boxes_corner[:, 5] = np.clip(boxes_corner[:, 5], 0, D)
            
            # Filter empty boxes (where x2 <= x1 etc)
            keep = (boxes_corner[:, 3] > boxes_corner[:, 0]) & \
                   (boxes_corner[:, 4] > boxes_corner[:, 1]) & \
                   (boxes_corner[:, 5] > boxes_corner[:, 2])
            boxes_corner = boxes_corner[keep]
            
            target['boxes'] = torch.from_numpy(boxes_corner).float()
            target['labels'] = torch.ones((len(boxes_corner),), dtype=torch.int64)
        else:
            target['boxes'] = torch.zeros((0, 6), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            
        target['image_id'] = torch.tensor([idx])
        
        return image_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))
