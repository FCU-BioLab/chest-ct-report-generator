#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv7 Dataset Adapter with Medical Preprocessing

Wraps CTDetectionDataset and provides medical image preprocessing:
- HU windowing
- CLAHE enhancement
- Robust percentile stretching
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
import logging

LOGGER = logging.getLogger(__name__)


def apply_hu_windowing(hu_array: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """
    Apply HU windowing to CT image
    
    Args:
        hu_array: Input HU values array
        window_center: Window center (e.g., -600 for lung)
        window_width: Window width (e.g., 1500 for lung)
    
    Returns:
        Windowed image as uint8 [0, 255]
    """
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    windowed = np.clip(hu_array, img_min, img_max)
    normalized = (windowed - img_min) / max(window_width, 1e-6)
    return (normalized * 255).astype(np.uint8)


def apply_percentile_stretch(img: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    """
    Apply robust percentile stretching
    
    Args:
        img: Input image array
        low_percentile: Lower percentile for clipping
        high_percentile: Upper percentile for clipping
    
    Returns:
        Stretched image as uint8
    """
    p_low, p_high = np.percentile(img, [low_percentile, high_percentile])
    if p_high - p_low < 1e-6:
        return img.astype(np.uint8)
    stretched = np.clip(img, p_low, p_high)
    normalized = (stretched - p_low) / (p_high - p_low)
    return (normalized * 255).astype(np.uint8)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    Args:
        img: Input grayscale image (uint8)
        clip_limit: CLAHE clip limit
        tile_size: Tile grid size
    
    Returns:
        Enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)


class YOLOv7MedicalDataset(Dataset):
    """
    YOLOv7 Dataset with Medical Image Preprocessing
    
    This dataset wraps CTDetectionDataset and applies medical-specific preprocessing:
    - HU windowing for CT scans
    - CLAHE for contrast enhancement
    - Robust percentile stretching as fallback
    
    Returns images in YOLOv7 format: (C, H, W) normalized to [0, 1]
    Labels in YOLO format: [class_id, x_center, y_center, width, height] normalized
    """
    
    def __init__(
        self,
        dataset,  # CTDetectionDataset instance
        img_size: int = 640,
        enable_hu_windowing: bool = True,
        window_center: float = -600.0,
        window_width: float = 1500.0,
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        robust_percentile_stretch: bool = True,
        augment: bool = False,
    ):
        """
        Args:
            dataset: CTDetectionDataset instance
            img_size: Target image size
            enable_hu_windowing: Enable HU windowing
            window_center: HU window center
            window_width: HU window width
            enable_clahe: Enable CLAHE enhancement
            clahe_clip_limit: CLAHE clip limit
            clahe_tile_size: CLAHE tile grid size
            robust_percentile_stretch: Enable percentile stretching fallback
            augment: Enable augmentation (handled by YOLOv7 training)
        """
        self.dataset = dataset
        self.img_size = img_size
        self.augment = augment
        
        # Medical preprocessing config
        self.enable_hu_windowing = enable_hu_windowing
        self.window_center = window_center
        self.window_width = window_width
        self.enable_clahe = enable_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.robust_percentile_stretch = robust_percentile_stretch
        
        LOGGER.info(f"YOLOv7MedicalDataset initialized with {len(self.dataset)} samples")
        LOGGER.info(f"Image size: {img_size}, HU windowing: {enable_hu_windowing}, CLAHE: {enable_clahe}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def preprocess_medical_image(self, item: Dict[str, Any]) -> np.ndarray:
        """
        Apply medical preprocessing pipeline
        
        Args:
            item: Dataset item containing image and metadata
        
        Returns:
            Preprocessed grayscale image (H, W) uint8
        """
        # Try to get HU image
        hu_image = None
        if isinstance(item, dict):
            hu_image = item.get("image_hu") or item.get("hu_image")
        
        # Apply HU windowing if available
        if self.enable_hu_windowing and hu_image is not None:
            hu_array = np.asarray(hu_image)
            img = apply_hu_windowing(hu_array, self.window_center, self.window_width)
        else:
            # Fallback: convert tensor to numpy
            image_tensor = item.get("image")
            if image_tensor is None:
                raise ValueError("No image found in dataset item")
            
            # Convert tensor to numpy array
            if isinstance(image_tensor, torch.Tensor):
                img_array = image_tensor.detach().cpu().numpy()
                
                # Handle different tensor formats
                if img_array.ndim == 3 and img_array.shape[0] in (1, 3):
                    img_array = img_array.squeeze(0) if img_array.shape[0] == 1 else img_array[0]
                elif img_array.ndim == 2:
                    pass
                else:
                    raise ValueError(f"Unexpected image tensor shape: {img_array.shape}")
                
                # Normalize to [0, 255]
                if img_array.max() <= 1.0:
                    img = (img_array * 255).astype(np.uint8)
                else:
                    img = np.clip(img_array, 0, 255).astype(np.uint8)
            else:
                img = np.asarray(image_tensor).astype(np.uint8)
            
            # Apply robust percentile stretch if enabled
            if self.robust_percentile_stretch:
                img = apply_percentile_stretch(img)
        
        # Apply CLAHE enhancement
        if self.enable_clahe:
            img = apply_clahe(img, self.clahe_clip_limit, self.clahe_tile_size)
        
        return img
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Get dataset item with medical preprocessing
        
        Args:
            idx: Index
        
        Returns:
            Tuple of:
                - Image tensor (3, H, W) normalized to [0, 1]
                - Labels tensor (N, 5): [class_id, x_center, y_center, width, height]
                - Metadata dict
        """
        # Get item from underlying dataset
        item = self.dataset[idx]
        
        # Apply medical preprocessing
        img_gray = self.preprocess_medical_image(item)
        
        # Resize to target size
        img_resized = cv2.resize(img_gray, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # Convert grayscale to 3-channel (YOLOv7 expects 3 channels)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        
        # Convert to tensor (C, H, W) and normalize to [0, 1]
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        
        # Process labels
        target = item.get("target", {})
        boxes = target.get("boxes", [])
        labels = target.get("labels", [])
        
        # Convert boxes to YOLO format: [class_id, x_center, y_center, width, height]
        yolo_labels = []
        if boxes is not None and len(boxes) > 0:
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.detach().cpu().numpy()
            else:
                boxes = np.array(boxes)
            
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
            else:
                labels = np.array(labels)
            
            # Ensure boxes are in [0, 1] normalized format
            H, W = img_gray.shape[:2]
            for box, label in zip(boxes, labels):
                # Handle different box formats
                if len(box) == 4:
                    if box[0] <= 1.0:  # Already normalized
                        cx, cy, w, h = box
                    else:  # Pixel coordinates [x1, y1, x2, y2]
                        x1, y1, x2, y2 = box
                        cx = (x1 + x2) / 2.0 / W
                        cy = (y1 + y2) / 2.0 / H
                        w = (x2 - x1) / W
                        h = (y2 - y1) / H
                    
                    # Ensure class_id is 0-based
                    class_id = int(label) if int(label) == 0 else int(label) - 1
                    class_id = max(0, class_id)
                    
                    yolo_labels.append([class_id, cx, cy, w, h])
        
        # Convert to tensor
        if len(yolo_labels) > 0:
            labels_tensor = torch.tensor(yolo_labels, dtype=torch.float32)
        else:
            labels_tensor = torch.zeros((0, 5), dtype=torch.float32)
        
        # Metadata
        metadata = {
            "index": idx,
            "patient_id": item.get("patient_id", "unknown"),
            "sop_instance_uid": item.get("sop_instance_uid", "unknown"),
            "original_size": img_gray.shape[:2],
            "resized_size": (self.img_size, self.img_size),
        }
        
        return img_tensor, labels_tensor, metadata


def create_yolov7_dataloader(
    dataset,
    batch_size: int = 16,
    img_size: int = 640,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    enable_hu_windowing: bool = True,
    window_center: float = -600.0,
    window_width: float = 1500.0,
    enable_clahe: bool = True,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create YOLOv7 DataLoader with medical preprocessing
    
    Args:
        dataset: CTDetectionDataset instance
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of workers
        shuffle: Shuffle data
        pin_memory: Pin memory for faster GPU transfer
        enable_hu_windowing: Enable HU windowing
        window_center: HU window center
        window_width: HU window width
        enable_clahe: Enable CLAHE
        **kwargs: Additional arguments
    
    Returns:
        DataLoader instance
    """
    medical_dataset = YOLOv7MedicalDataset(
        dataset=dataset,
        img_size=img_size,
        enable_hu_windowing=enable_hu_windowing,
        window_center=window_center,
        window_width=window_width,
        enable_clahe=enable_clahe,
        **kwargs
    )
    
    # Custom collate function for variable-length labels
    def collate_fn(batch):
        images, labels, metadata = zip(*batch)
        images = torch.stack(images, 0)
        
        # Add batch index to labels
        for i, label in enumerate(labels):
            if label.shape[0] > 0:
                label = torch.cat([torch.full((label.shape[0], 1), i), label], dim=1)
                labels[i] = label
        
        labels = torch.cat([l for l in labels if l.shape[0] > 0], 0) if any(l.shape[0] > 0 for l in labels) else torch.zeros((0, 6))
        
        return images, labels, metadata
    
    dataloader = torch.utils.data.DataLoader(
        medical_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    
    return dataloader


if __name__ == "__main__":
    print("YOLOv7 Medical Dataset module")
    print("This module provides medical image preprocessing for YOLOv7 training")
