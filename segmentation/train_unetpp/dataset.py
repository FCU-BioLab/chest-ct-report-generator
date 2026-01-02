#!/usr/bin/env python3
"""
UNet++ 肺結節分割訓練 - LNDb 資料集模組
========================================

統一 4-Patch 邏輯：
- Train/Val/Test 都使用相同的 4-patch 提取方式
- 正負樣本判定基於**每個 patch 內是否有 lesion**
- Train 時對 patch 進行 oversampling
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys

try:
    from .patch_utils import compute_4patch_positions, extract_patch_with_lung_mask
    from .config import Config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train_unetpp.patch_utils import compute_4patch_positions, extract_patch_with_lung_mask
    from train_unetpp.config import Config


logger = logging.getLogger(__name__)


# =============================================================================
# Collate Functions
# =============================================================================


# =============================================================================
# LNDb 切片式資料集
# =============================================================================

class LNDbSliceDataset(Dataset):
    """
    切片式 LNDb 資料集
    
    統一 4-Patch 邏輯：
    - Train 和 Val/Test 都使用相同的 4 個 patch 位置
    - Train 時：返回單一 patch（從 4 個中隨機選或 oversample）
    - Val/Test 時：返回全部 4 個 patches
    
    正負樣本判定：基於 **patch 內是否有 lesion**（mask.sum() > 0）
    """
    
    def __init__(
        self,
        slice_cache_dir: str,
        patient_ids: list,
        config,
        mode: str = "train",
        transform=None
    ):
        self.slice_cache_dir = Path(slice_cache_dir)
        self.patient_ids = patient_ids
        self.config = config
        self.mode = mode
        self.patch_size = config.data.patch_size
        
        if transform is not None:
            self.transform = transform
        elif mode == "train" and config.training.use_augmentation:
            self.transform = self._get_train_transform()
        else:
            self.transform = self._get_val_transform()
        
        # 建立 patch-level 樣本索引
        self._build_patch_index()
        
        # 訓練模式下進行 patch-level oversampling
        if mode == "train":
            self._oversample_patches()
    
    def _get_train_transform(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
            ToTensorV2()
        ])
    
    def _get_val_transform(self):
        return A.Compose([ToTensorV2()])
    
    def _build_patch_index(self):
        """
        建立 patch-level 索引
        
        對每個 slice 計算 4 個 patch，標記每個 patch 是正/負樣本
        """
        self.samples = []  # slice-level（保留用於統計）
        self.patch_samples = []  # patch-level（所有模式都用）
        
        for patient_id in self.patient_ids:
            patient_dir = self.slice_cache_dir / patient_id
            meta_path = patient_dir / "meta.json"
            
            if not meta_path.exists():
                logger.warning(f"切片快取不存在: {patient_id}")
                continue
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            positive_slices = set(meta['positive_slices'])
            
            for z in range(meta['num_slices']):
                slice_info = {
                    'patient_id': patient_id,
                    'slice_idx': z,
                    'is_positive': z in positive_slices,
                    'slice_path': str(patient_dir / f"slice_{z:04d}.npz"),
                    'num_slices': meta['num_slices']
                }
                self.samples.append(slice_info)
                
                # 所有模式都建立 patch 索引
                if z in positive_slices:
                    # 載入 mask 來判斷每個 patch 是否為正
                    data = np.load(slice_info['slice_path'], allow_pickle=True)
                    mask = data['mask'].astype(np.float32)
                    lung_mask = data['lung_mask'].astype(np.float32)
                    
                    patch_positions = compute_4patch_positions(lung_mask, self.patch_size)
                    
                    for patch_idx, ((py1, px1), (py2, px2)) in enumerate(patch_positions):
                        # 計算這個 patch 範圍內的 mask
                        py1_c = max(0, py1)
                        py2_c = min(mask.shape[0], py2)
                        px1_c = max(0, px1)
                        px2_c = min(mask.shape[1], px2)
                        
                        patch_mask_area = mask[py1_c:py2_c, px1_c:px2_c].sum()
                        is_patch_positive = patch_mask_area > 0
                        
                        self.patch_samples.append({
                            **slice_info,
                            'patch_idx': patch_idx,
                            'is_patch_positive': is_patch_positive
                        })
                else:
                    # 負樣本切片的 4 個 patches 都是負
                    for patch_idx in range(4):
                        self.patch_samples.append({
                            **slice_info,
                            'patch_idx': patch_idx,
                            'is_patch_positive': False
                        })
        
        pos_patches = sum(1 for s in self.patch_samples if s['is_patch_positive'])
        neg_patches = len(self.patch_samples) - pos_patches
        logger.info(f"Patch-level 索引 ({self.mode}): {pos_patches} 正, {neg_patches} 負")
    
    def _oversample_patches(self):
        """只保留正樣本 patches（移除所有負樣本）"""
        positives = [s for s in self.patch_samples if s['is_patch_positive']]
        
        if not positives:
            logger.warning(f"沒有正樣本 patches！保留所有 {len(self.patch_samples)} 個 patches")
            return
        
        self.patch_samples = positives
        random.shuffle(self.patch_samples)
        
        logger.info(f"只保留正樣本: {len(self.patch_samples)} patches")
    
    def _load_2_5d_slice(self, patient_dir: Path, slice_idx: int, num_slices: int):
        """載入 2.5D 切片"""
        z_prev = max(0, slice_idx - 1)
        z_next = min(num_slices - 1, slice_idx + 1)
        
        slices = []
        for z in [z_prev, slice_idx, z_next]:
            data = np.load(patient_dir / f"slice_{z:04d}.npz", allow_pickle=True)
            slices.append(data['image'].astype(np.float32))
        
        return np.stack(slices, axis=0)
    
    def __len__(self):
        return len(self.patch_samples)
    
    def __getitem__(self, idx):
        """所有模式統一返回單一 patch"""
        return self._get_train_item(idx)
    
    def _get_train_item(self, idx):
        """Train 模式：返回單一 patch"""
        sample = self.patch_samples[idx]
        patient_id = sample['patient_id']
        slice_idx = sample['slice_idx']
        patch_idx = sample['patch_idx']
        num_slices = sample['num_slices']
        patient_dir = self.slice_cache_dir / patient_id
        
        # 載入 2.5D 切片
        image_2_5d = self._load_2_5d_slice(patient_dir, slice_idx, num_slices)
        
        # 載入 mask 和 lung_mask
        center_data = np.load(sample['slice_path'], allow_pickle=True)
        mask = center_data['mask'].astype(np.float32)
        lung_mask = center_data['lung_mask'].astype(np.float32)
        
        # 計算 4 個 patch 位置，取對應的那個
        patch_positions = compute_4patch_positions(lung_mask, self.patch_size)
        (py1, px1), (py2, px2) = patch_positions[patch_idx]
        patch_pos = ((py1, px1), (py2, px2))
        
        # 提取 patch
        image_patch, mask_patch, _ = extract_patch_with_lung_mask(
            image_2_5d, mask, lung_mask, patch_pos, self.patch_size
        )
        
        # 轉為 (H, W, 3) 給 Albumentations
        image_for_aug = np.transpose(image_patch, (1, 2, 0))
        
        if self.transform:
            transformed = self.transform(image=image_for_aug, mask=mask_patch)
            image_tensor = transformed['image'].float()
            mask_tensor = transformed['mask'].float()
        else:
            image_tensor = torch.from_numpy(image_patch).float()
            mask_tensor = torch.from_numpy(mask_patch).float()
        
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'patient_id': patient_id,
            'slice_idx': slice_idx,
            'patch_idx': patch_idx,
            'is_positive': sample['is_patch_positive']
        }



# =============================================================================
# 資料分割函數
# =============================================================================

def get_patient_split(
    data_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """隨機分割病人 ID"""
    slice_cache_dir = Path(data_dir).parent / 'cache' / 'lndb_slices'
    cache_dir = Path(data_dir).parent / 'cache' / 'lndb_preprocessed'
    
    patient_ids = []
    
    if slice_cache_dir.exists():
        patient_ids = [d.name for d in slice_cache_dir.iterdir() 
                       if d.is_dir() and (d / 'meta.json').exists()]
    
    if len(patient_ids) == 0 and cache_dir.exists():
        patient_ids = [f.stem for f in cache_dir.glob('*.npz')]
    
    if len(patient_ids) == 0:
        data_path = Path(data_dir)
        for subfolder in ['data0', 'data1', 'data2', 'data3', 'data4', 'data5']:
            folder = data_path / subfolder
            if folder.exists():
                patient_ids.extend([f.stem for f in folder.glob('*.mhd')])
    
    patient_ids = sorted(set(patient_ids))
    
    # 過濾只保留有結節的病人
    nodule_csv = Path(data_dir) / 'trainset_csv' / 'trainNodules_gt.csv'
    if nodule_csv.exists():
        df = pd.read_csv(nodule_csv)
        nodule_patients = set(f"LNDb-{int(pid):04d}" for pid in df[df['Nodule'] == 1]['LNDbID'].unique())
        patient_ids = [pid for pid in patient_ids if pid in nodule_patients]
    
    np.random.seed(seed)
    np.random.shuffle(patient_ids)
    
    n = len(patient_ids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return patient_ids[:train_end], patient_ids[train_end:val_end], patient_ids[val_end:]


def get_fold_split(data_dir: str, fold_id: int, num_folds: int = 5) -> Tuple[List[str], List[str]]:
    """根據 trainFolds.csv 獲取交叉驗證分割"""
    folds_path = Path(data_dir) / 'trainset_csv' / 'trainFolds.csv'
    
    if folds_path.exists():
        df = pd.read_csv(folds_path)
        id_col = 'LNDbID' if 'LNDbID' in df.columns else df.columns[0]
        fold_col = 'fold' if 'fold' in df.columns else df.columns[1]
        
        val_ids = df[df[fold_col] == fold_id][id_col].tolist()
        train_ids = df[df[fold_col] != fold_id][id_col].tolist()
    else:
        train_ids, val_ids, _ = get_patient_split(data_dir, seed=42 + fold_id)
    
    return train_ids, val_ids
