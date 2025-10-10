#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv7 Dataset Adapter with Medical Preprocessing

Wraps CTDetectionDataset and provides medical image preprocessing:
- HU windowing
- CLAHE enhancement
- Robust percentile stretching
- Positive sample oversampling
- Advanced augmentations (Mosaic, MixUp, Copy-Paste)
- Dataset caching for faster training
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, Sampler
from typing import Dict, List, Optional, Tuple, Any
import logging
import random
from collections import defaultdict

LOGGER = logging.getLogger(__name__)

# Try to import augmentation module
try:
    from yolov7_augmentations import YOLOv7Augmenter
    AUGMENTATIONS_AVAILABLE = True
except ImportError:
    AUGMENTATIONS_AVAILABLE = False
    LOGGER.warning("yolov7_augmentations not available, augmentations disabled")


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
    - Optional augmentations (Mosaic, MixUp, Copy-Paste)
    - Dataset caching for faster training
    
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
        mosaic_prob: float = 0.5,
        mixup_prob: float = 0.3,
        copy_paste_prob: float = 0.3,
        cache_images: bool = False,
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
            augment: Enable augmentation
            mosaic_prob: Probability of mosaic augmentation
            mixup_prob: Probability of mixup augmentation
            copy_paste_prob: Probability of copy-paste augmentation
            cache_images: Cache preprocessed images in memory
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
        
        # Augmentation config
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.copy_paste_prob = copy_paste_prob
        
        # Initialize augmenter if available
        if augment and AUGMENTATIONS_AVAILABLE:
            self.augmenter = YOLOv7Augmenter(
                img_size=img_size,
                mosaic_prob=mosaic_prob,
                mixup_prob=mixup_prob,
                copy_paste_prob=copy_paste_prob,
                enable_mosaic=True,
                enable_mixup=True,
                enable_copy_paste=True
            )
        else:
            self.augmenter = None
        
        # Image caching
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None
        
        # Build positive/negative sample indices for oversampling
        self.positive_indices = []
        self.negative_indices = []
        for i in range(len(self.dataset)):
            try:
                item = self.dataset[i]
                target = item.get("target", {})
                boxes = target.get("boxes", [])
                if boxes is not None and len(boxes) > 0:
                    self.positive_indices.append(i)
                else:
                    self.negative_indices.append(i)
            except:
                pass
        
        LOGGER.info(f"YOLOv7MedicalDataset initialized with {len(self.dataset)} samples")
        LOGGER.info(f"  Positive samples: {len(self.positive_indices)}, Negative samples: {len(self.negative_indices)}")
        LOGGER.info(f"  Image size: {img_size}, HU windowing: {enable_hu_windowing}, CLAHE: {enable_clahe}")
        LOGGER.info(f"  Augmentation: {augment}, Caching: {cache_images}")
    
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
        Get dataset item with medical preprocessing and augmentation
        
        Args:
            idx: Index
        
        Returns:
            Tuple of:
                - Image tensor (3, H, W) normalized to [0, 1]
                - Labels tensor (N, 5): [class_id, x_center, y_center, width, height]
                - Metadata dict
        """
        # Check cache first
        if self.cache_images and idx in self.image_cache:
            img_gray, boxes_normalized = self.image_cache[idx]
        else:
            # Get item from underlying dataset
            item = self.dataset[idx]
            
            # Apply medical preprocessing
            img_gray = self.preprocess_medical_image(item)
            
            # Process labels
            target = item.get("target", {})
            boxes = target.get("boxes", [])
            labels = target.get("labels", [])
            
            # Convert boxes to YOLO format: [class_id, x_center, y_center, width, height]
            boxes_normalized = []
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
                        
                        boxes_normalized.append([class_id, cx, cy, w, h])
            
            boxes_normalized = np.array(boxes_normalized, dtype=np.float32) if boxes_normalized else np.zeros((0, 5), dtype=np.float32)
            
            # Cache if enabled
            if self.cache_images:
                self.image_cache[idx] = (img_gray.copy(), boxes_normalized.copy())
        
        # Resize to target size
        img_resized = cv2.resize(img_gray, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # Convert grayscale to 3-channel
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        
        # Apply augmentations if enabled
        if self.augment and self.augmenter is not None and len(boxes_normalized) > 0:
            # Get extra samples for mosaic/mixup
            extra_samples = []
            if random.random() < self.mosaic_prob or random.random() < self.mixup_prob:
                # Sample 3 random positive samples for augmentation
                if len(self.positive_indices) > 3:
                    extra_indices = random.sample(self.positive_indices, min(3, len(self.positive_indices)))
                    for extra_idx in extra_indices:
                        if extra_idx != idx:
                            try:
                                extra_item = self.dataset[extra_idx]
                                extra_img_gray = self.preprocess_medical_image(extra_item)
                                extra_img_resized = cv2.resize(extra_img_gray, (self.img_size, self.img_size))
                                extra_img_rgb = cv2.cvtColor(extra_img_resized, cv2.COLOR_GRAY2RGB)
                                
                                # Get extra labels
                                extra_target = extra_item.get("target", {})
                                extra_boxes = extra_target.get("boxes", [])
                                extra_labels = extra_target.get("labels", [])
                                
                                extra_boxes_normalized = []
                                if extra_boxes is not None and len(extra_boxes) > 0:
                                    if isinstance(extra_boxes, torch.Tensor):
                                        extra_boxes = extra_boxes.detach().cpu().numpy()
                                    if isinstance(extra_labels, torch.Tensor):
                                        extra_labels = extra_labels.detach().cpu().numpy()
                                    
                                    for box, label in zip(extra_boxes, extra_labels):
                                        if len(box) == 4:
                                            if box[0] <= 1.0:
                                                cx, cy, w, h = box
                                            else:
                                                x1, y1, x2, y2 = box
                                                cx = (x1 + x2) / 2.0 / self.img_size
                                                cy = (y1 + y2) / 2.0 / self.img_size
                                                w = (x2 - x1) / self.img_size
                                                h = (y2 - y1) / self.img_size
                                            
                                            class_id = int(label) if int(label) == 0 else int(label) - 1
                                            class_id = max(0, class_id)
                                            extra_boxes_normalized.append([class_id, cx, cy, w, h])
                                
                                extra_boxes_normalized = np.array(extra_boxes_normalized, dtype=np.float32) if extra_boxes_normalized else np.zeros((0, 5), dtype=np.float32)
                                extra_samples.append((extra_img_rgb, extra_boxes_normalized))
                            except:
                                pass
            
            # Apply augmentations
            try:
                img_rgb, boxes_normalized = self.augmenter(img_rgb, boxes_normalized, extra_samples)
            except Exception as e:
                LOGGER.warning(f"Augmentation failed: {e}, using original")
        
        # Convert to tensor (C, H, W) and normalize to [0, 1]
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        
        # Convert labels to tensor
        if len(boxes_normalized) > 0:
            labels_tensor = torch.tensor(boxes_normalized, dtype=torch.float32)
        else:
            labels_tensor = torch.zeros((0, 5), dtype=torch.float32)
        
        # Metadata
        item = self.dataset[idx]
        metadata = {
            "index": idx,
            "patient_id": item.get("patient_id", "unknown"),
            "sop_instance_uid": item.get("sop_instance_uid", "unknown"),
            "original_size": img_gray.shape[:2],
            "resized_size": (self.img_size, self.img_size),
        }
        
        return img_tensor, labels_tensor, metadata


def yolov7_collate_fn(batch):
    """
    Custom collate function for YOLOv7 DataLoader
    Handles variable-length labels and adds batch indices
    
    Args:
        batch: List of (image, label, metadata) tuples
    
    Returns:
        images: Stacked tensor [B, C, H, W]
        labels: Concatenated labels [N, 6] (batch_idx, class, x, y, w, h)
        metadata: List of metadata dicts
    """
    images, labels, metadata = zip(*batch)
    images = torch.stack(images, 0)
    
    # Add batch index to labels
    labels_with_batch = []
    for i, label in enumerate(labels):
        if label.shape[0] > 0:
            # Prepend batch index
            batch_idx = torch.full((label.shape[0], 1), i, dtype=label.dtype, device=label.device)
            label_with_idx = torch.cat([batch_idx, label], dim=1)
            labels_with_batch.append(label_with_idx)
    
    # Concatenate all labels
    if labels_with_batch:
        labels = torch.cat(labels_with_batch, 0)
    else:
        labels = torch.zeros((0, 6))
    
    return images, labels, metadata


class PositiveOversampleSampler(Sampler):
    """
    Sampler that oversamples positive samples (with lesions)
    to balance positive/negative ratio in training
    """
    
    def __init__(
        self,
        dataset: YOLOv7MedicalDataset,
        positive_ratio: float = 0.7,
        shuffle: bool = True
    ):
        """
        Args:
            dataset: YOLOv7MedicalDataset instance
            positive_ratio: Target ratio of positive samples (0.7 = 70% positive)
            shuffle: Shuffle indices
        """
        self.dataset = dataset
        self.positive_ratio = positive_ratio
        self.shuffle = shuffle
        
        self.positive_indices = dataset.positive_indices
        self.negative_indices = dataset.negative_indices
        
        # Calculate sampling
        n_positive = len(self.positive_indices)
        n_negative = len(self.negative_indices)
        
        if n_positive == 0:
            LOGGER.warning("No positive samples found!")
            self.indices = self.negative_indices
        elif n_negative == 0:
            self.indices = self.positive_indices
        else:
            # Calculate how many negative samples to use
            target_n_negative = int(n_positive * (1 - positive_ratio) / positive_ratio)
            target_n_negative = min(target_n_negative, n_negative)
            
            # Oversample positives if needed
            n_positive_repeat = max(1, int(n_positive / max(target_n_negative, 1)))
            
            self.indices = (
                self.positive_indices * n_positive_repeat +
                self.negative_indices[:target_n_negative]
            )
        
        LOGGER.info(f"PositiveOversampleSampler: Total={len(self.indices)}, "
                   f"Pos={len([i for i in self.indices if i in self.positive_indices])}, "
                   f"Neg={len([i for i in self.indices if i in self.negative_indices])}")
    
    def __iter__(self):
        indices = self.indices.copy()
        if self.shuffle:
            random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return len(self.indices)


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
    augment: bool = False,
    mosaic_prob: float = 0.5,
    mixup_prob: float = 0.3,
    copy_paste_prob: float = 0.3,
    cache_images: bool = False,
    positive_oversample: bool = True,
    positive_ratio: float = 0.7,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create YOLOv7 DataLoader with medical preprocessing and augmentation
    
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
        augment: Enable augmentation
        mosaic_prob: Mosaic probability
        mixup_prob: MixUp probability
        copy_paste_prob: Copy-Paste probability
        cache_images: Cache images in memory
        positive_oversample: Use positive oversampling
        positive_ratio: Target positive ratio (0.7 = 70% positive)
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
        augment=augment,
        mosaic_prob=mosaic_prob,
        mixup_prob=mixup_prob,
        copy_paste_prob=copy_paste_prob,
        cache_images=cache_images,
        **kwargs
    )
    
    # Use positive oversampler if training
    sampler = None
    if positive_oversample and shuffle and augment:
        sampler = PositiveOversampleSampler(
            medical_dataset,
            positive_ratio=positive_ratio,
            shuffle=True
        )
        shuffle = False  # Sampler handles shuffling
    
    # Use global collate function (defined at module level for pickle compatibility)
    dataloader = torch.utils.data.DataLoader(
        medical_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=yolov7_collate_fn,
        drop_last=False,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch for faster loading
    )
    
    return dataloader


if __name__ == "__main__":
    print("YOLOv7 Medical Dataset module")
    print("This module provides medical image preprocessing for YOLOv7 training")
