#!/usr/bin/env python3
"""
視頻格式資料集
==============

將 CT 切片序列轉換為 MedSAM2 可處理的視頻格式。

核心概念：
- 每個病灶周圍的連續切片組成一個「視頻」
- 中心切片（有標註）作為 conditioning frame
- 前後切片用於學習時序傳播
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class CTVideoSample:
    """CT 視頻樣本資料結構"""
    # 基本資訊
    patient_id: str
    lesion_id: int
    
    # 視頻資料 (D, H, W) - D 是幀數
    frames: np.ndarray  # uint8, 0-255
    
    # 標註資料 (D, H, W)
    masks: np.ndarray   # uint8, 0 或 label_id
    
    # 中心幀索引（有完整標註的那一幀）
    center_frame_idx: int
    
    # 原始空間資訊
    spacing: np.ndarray  # (z, y, x) spacing in mm
    original_shape: Tuple[int, int, int]  # 原始體積大小
    slice_indices: List[int]  # 對應原始體積的切片索引
    
    # 病灶資訊
    lesion_diameter_mm: float
    lesion_center_world: np.ndarray  # (x, y, z) in mm
    
    # Prompt 資訊（可選）
    bbox: Optional[np.ndarray] = None  # (x1, y1, x2, y2)
    center_point: Optional[np.ndarray] = None  # (x, y)


class VideoLesionDataset(Dataset):
    """
    視頻病灶資料集
    
    從 NPZ 檔案載入預處理好的視頻樣本。
    
    NPZ 格式:
        - frames: (D, H, W) uint8 影像
        - masks: (D, H, W) uint8 遮罩
        - spacing: (3,) float spacing
        - center_idx: int 中心幀索引
        - patient_id: str
        - lesion_id: int
        - diameter_mm: float
        - bbox: (4,) 可選
    """
    
    def __init__(
        self,
        npz_dir: str,
        split: str = "train",
        image_size: int = 512,
        max_video_length: int = 32,
        augmentation: bool = False,
        normalize: bool = True,
    ):
        """
        初始化視頻資料集
        
        Args:
            npz_dir: NPZ 檔案目錄
            split: 資料分割 (train/val/test)
            image_size: 輸出影像大小
            max_video_length: 最大視頻長度
            augmentation: 是否啟用資料增強
            normalize: 是否正規化 (ImageNet 標準)
        """
        self.npz_dir = Path(npz_dir)
        self.split = split
        self.image_size = image_size
        self.max_video_length = max_video_length
        self.augmentation = augmentation
        self.normalize = normalize
        
        # ImageNet 正規化參數
        self.img_mean = np.array([0.485, 0.456, 0.406])
        self.img_std = np.array([0.229, 0.224, 0.225])
        
        # 載入檔案列表
        self.samples = self._load_file_list()
        logger.info(f"📹 載入 {len(self.samples)} 個視頻樣本 (split={split})")
    
    def _load_file_list(self) -> List[Path]:
        """載入指定 split 的所有 NPZ 檔案"""
        split_dir = self.npz_dir / self.split
        if not split_dir.exists():
            # 嘗試從根目錄載入
            split_dir = self.npz_dir
        
        npz_files = sorted(split_dir.glob("*.npz"))
        
        if not npz_files:
            logger.warning(f"⚠️ 找不到 NPZ 檔案: {split_dir}")
        
        return npz_files
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        取得視頻樣本
        
        Returns:
            Dict containing:
                - frames: (D, 3, H, W) torch.Tensor, 正規化後的 RGB 影像
                - masks: (D, H, W) torch.Tensor, 分割遮罩
                - center_idx: int, 中心幀索引
                - bbox: (4,) torch.Tensor, bounding box
                - patient_id: str
                - lesion_id: int
                - num_frames: int
        """
        npz_path = self.samples[idx]
        data = np.load(npz_path, allow_pickle=True)
        
        # 載入基本資料
        frames = data['frames']  # (D, H, W)
        masks = data['masks']    # (D, H, W)
        center_idx = int(data['center_idx'])
        
        # 截斷到最大長度
        if len(frames) > self.max_video_length:
            # 以中心幀為基準截斷
            half = self.max_video_length // 2
            start = max(0, center_idx - half)
            end = min(len(frames), start + self.max_video_length)
            start = max(0, end - self.max_video_length)
            
            frames = frames[start:end]
            masks = masks[start:end]
            center_idx = center_idx - start
        
        num_frames = len(frames)
        
        # 調整大小
        frames, masks = self._resize_video(frames, masks)
        
        # 資料增強
        if self.augmentation and self.split == "train":
            frames, masks = self._augment(frames, masks)
        
        # 轉換為 RGB 並正規化
        frames_rgb = self._to_rgb_normalized(frames)
        
        # 計算 bounding box（從中心幀的 mask）
        bbox = self._compute_bbox(masks[center_idx])
        
        # 計算中心點 (point prompt)
        center_point = self._compute_center_point(masks[center_idx])
        
        # 轉換為 tensor
        frames_tensor = torch.from_numpy(frames_rgb).float()  # (D, 3, H, W)
        masks_tensor = torch.from_numpy(masks).long()          # (D, H, W)
        bbox_tensor = torch.from_numpy(bbox).float()           # (4,)
        center_point_tensor = torch.from_numpy(center_point).float()  # (2,)
        
        # 取得元資料
        patient_id = str(data.get('patient_id', f'patient_{idx}'))
        lesion_id = int(data.get('lesion_id', 0))
        
        return {
            'frames': frames_tensor,
            'masks': masks_tensor,
            'center_idx': center_idx,
            'bbox': bbox_tensor,
            'center_point': center_point_tensor,  # 新增：中心點 (x, y)
            'patient_id': patient_id,
            'lesion_id': lesion_id,
            'num_frames': num_frames,
            'npz_path': str(npz_path),
        }
    
    def _resize_video(
        self, 
        frames: np.ndarray, 
        masks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """調整視頻大小到 image_size x image_size"""
        D, H, W = frames.shape
        
        if H == self.image_size and W == self.image_size:
            return frames, masks
        
        resized_frames = np.zeros((D, self.image_size, self.image_size), dtype=np.uint8)
        resized_masks = np.zeros((D, self.image_size, self.image_size), dtype=np.uint8)
        
        for i in range(D):
            # 使用 PIL 進行高品質縮放
            frame_pil = Image.fromarray(frames[i])
            mask_pil = Image.fromarray(masks[i])
            
            resized_frames[i] = np.array(
                frame_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
            )
            resized_masks[i] = np.array(
                mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)
            )
        
        return resized_frames, resized_masks
    
    def _to_rgb_normalized(self, frames: np.ndarray) -> np.ndarray:
        """
        將灰階影像轉換為正規化的 RGB
        
        Args:
            frames: (D, H, W) uint8
            
        Returns:
            (D, 3, H, W) float32, 正規化後
        """
        D, H, W = frames.shape
        
        # 擴展為 RGB
        frames_rgb = np.stack([frames, frames, frames], axis=1)  # (D, 3, H, W)
        
        # 正規化到 0-1
        frames_rgb = frames_rgb.astype(np.float32) / 255.0
        
        if self.normalize:
            # ImageNet 正規化
            for c in range(3):
                frames_rgb[:, c] = (frames_rgb[:, c] - self.img_mean[c]) / self.img_std[c]
        
        return frames_rgb
    
    def _compute_bbox(self, mask: np.ndarray) -> np.ndarray:
        """從 mask 計算 bounding box"""
        if mask.max() == 0:
            # 無標註，返回中心小框
            h, w = mask.shape
            cx, cy = w // 2, h // 2
            return np.array([cx - 10, cy - 10, cx + 10, cy + 10], dtype=np.float32)
        
        ys, xs = np.where(mask > 0)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        
        # 稍微擴大 bbox
        pad = 5
        h, w = mask.shape
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    
    def _compute_center_point(self, mask: np.ndarray) -> np.ndarray:
        """
        從 mask 計算中心點（質心）
        
        Args:
            mask: (H, W) binary mask
            
        Returns:
            (2,) array of (x, y) coordinates
        """
        if mask.max() == 0:
            # 無標註，返回影像中心
            h, w = mask.shape
            return np.array([w // 2, h // 2], dtype=np.float32)
        
        # 計算質心
        ys, xs = np.where(mask > 0)
        cx = xs.mean()
        cy = ys.mean()
        
        return np.array([cx, cy], dtype=np.float32)
    
    def _augment(
        self, 
        frames: np.ndarray, 
        masks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """資料增強（保持時序一致性）"""
        import random
        
        # 隨機水平翻轉
        if random.random() > 0.5:
            frames = np.flip(frames, axis=2).copy()
            masks = np.flip(masks, axis=2).copy()
        
        # 隨機垂直翻轉
        if random.random() > 0.5:
            frames = np.flip(frames, axis=1).copy()
            masks = np.flip(masks, axis=1).copy()
        
        # 隨機亮度調整
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            frames = np.clip(frames * factor, 0, 255).astype(np.uint8)
        
        # 隨機對比度調整
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            mean = frames.mean()
            frames = np.clip((frames - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        return frames, masks
    
    def get_sample_info(self, idx: int) -> Dict:
        """取得樣本的基本資訊（不載入完整資料）"""
        npz_path = self.samples[idx]
        data = np.load(npz_path, allow_pickle=True)
        
        return {
            'path': str(npz_path),
            'patient_id': str(data.get('patient_id', '')),
            'lesion_id': int(data.get('lesion_id', 0)),
            'num_frames': len(data['frames']),
            'center_idx': int(data['center_idx']),
            'diameter_mm': float(data.get('diameter_mm', 0)),
        }


def collate_video_batch(batch: List[Dict]) -> Dict:
    """
    視頻批次整理函數
    
    由於不同視頻長度可能不同，需要特殊處理。
    對於 batch_size=1，直接返回即可。
    """
    if len(batch) == 1:
        # 單一樣本，直接返回並增加 batch 維度
        sample = batch[0]
        return {
            'frames': sample['frames'].unsqueeze(0),      # (1, D, 3, H, W)
            'masks': sample['masks'].unsqueeze(0),        # (1, D, H, W)
            'center_idx': [sample['center_idx']],
            'bbox': sample['bbox'].unsqueeze(0),          # (1, 4)
            'center_point': sample['center_point'].unsqueeze(0),  # (1, 2)
            'patient_id': [sample['patient_id']],
            'lesion_id': [sample['lesion_id']],
            'num_frames': [sample['num_frames']],
            'npz_path': [sample['npz_path']],
        }
    
    # 多個樣本需要 padding（視頻模式通常 batch=1，這裡預留擴展）
    max_frames = max(s['num_frames'] for s in batch)
    
    frames_list = []
    masks_list = []
    
    for sample in batch:
        D = sample['num_frames']
        frames = sample['frames']  # (D, 3, H, W)
        masks = sample['masks']    # (D, H, W)
        
        # Padding
        if D < max_frames:
            pad_frames = torch.zeros(max_frames - D, *frames.shape[1:])
            pad_masks = torch.zeros(max_frames - D, *masks.shape[1:], dtype=torch.long)
            frames = torch.cat([frames, pad_frames], dim=0)
            masks = torch.cat([masks, pad_masks], dim=0)
        
        frames_list.append(frames)
        masks_list.append(masks)
    
    return {
        'frames': torch.stack(frames_list),  # (B, D, 3, H, W)
        'masks': torch.stack(masks_list),    # (B, D, H, W)
        'center_idx': [s['center_idx'] for s in batch],
        'bbox': torch.stack([s['bbox'] for s in batch]),
        'center_point': torch.stack([s['center_point'] for s in batch]),  # (B, 2)
        'patient_id': [s['patient_id'] for s in batch],
        'lesion_id': [s['lesion_id'] for s in batch],
        'num_frames': [s['num_frames'] for s in batch],
        'npz_path': [s['npz_path'] for s in batch],
    }
