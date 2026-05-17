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
        # Stronger but medically safe augmentations (No Cutout/Elastic that destroys nodules)
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            # Affine transforms (Safe geometric variations)
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.6),
            
            # Mild distortion (Tissue deformation) - careful with grid size to not break nodules
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
            
            # Intensity transforms (Scanner/Dose variations)
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            
            # Quality transforms (Reconstruction kernels/Noise)
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            
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
                slice_path = patient_dir / f"slice_{z:04d}.npz"
                if not slice_path.exists():
                    continue

                slice_info = {
                    'patient_id': patient_id,
                    'slice_idx': z,
                    'is_positive': z in positive_slices,
                    'slice_path': str(slice_path),
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
        """載入 2.5D 切片 (5 channels: z-2, z-1, z, z+1, z+2)"""
        slices_indices = [
            max(0, slice_idx - 2),
            max(0, slice_idx - 1),
            slice_idx,
            min(num_slices - 1, slice_idx + 1),
            min(num_slices - 1, slice_idx + 2)
        ]
        
        slices = []
        for z in slices_indices:
            path = patient_dir / f"slice_{z:04d}.npz"
            if not path.exists():
                # Fallback to center slice if neighbor is missing
                path = patient_dir / f"slice_{slice_idx:04d}.npz"
            
            data = np.load(path, allow_pickle=True)
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


# =============================================================================
# 預處理 4-Patch 資料集（直接讀取 cache/lndb_patches）
# =============================================================================

class CachedPatchDataset(Dataset):
    """
    直接讀取預處理好的 4-patch npz 檔案，支援 ratio-based sampling
    
    預處理格式:
        cache/lndb_patches/
        └── LNDb-XXXX/
            ├── meta.json
            └── slice_XXXX_patch_X.npz  (image, mask, lung_mask, slice_idx, patch_idx, patch_type)
    
    Patch types:
        - positive: 含病灶的 patch
        - hard_negative: 遠離病灶的肺內 patch（距離 >= dilate_mm）
        - random_negative: 不含病灶但可能接近病灶的 patch
        - coverage: 4-patch 覆蓋中不含病灶的 patch
    
    Sampling strategy:
        - 依 config 中的 positive_ratio / hard_negative_ratio / random_negative_ratio 抽樣
        - train/val/test 使用相同的抽樣比例，確保評估一致性
    """
    
    def __init__(
        self,
        cache_dir: str,
        patient_ids: List[str],
        config,
        mode: str = "train",
        transform=None
    ):
        self.cache_dir = Path(cache_dir)
        self.patient_ids = patient_ids
        self.config = config
        self.mode = mode
        self.patch_size = config.data.patch_size
        
        # 取得抽樣比例
        self.pos_ratio = getattr(config.data, 'positive_ratio', 0.7)
        self.hard_neg_ratio = getattr(config.data, 'hard_negative_ratio', 0.2)
        self.rand_neg_ratio = getattr(config.data, 'random_negative_ratio', 0.1)
        
        if transform is not None:
            self.transform = transform
        elif mode == "train" and config.training.use_augmentation:
            self.transform = self._get_train_transform()
        else:
            self.transform = self._get_val_transform()
        
        # 建立 patch 索引（按 patch_type 分桶）
        self._build_index()
        
        # 計算 epoch 長度
        self._compute_epoch_length()
        
        # Log 統計
        self._log_stats()
    
    def _get_train_transform(self):
        config_aug_prob = getattr(self.config.training, 'augmentation_prob', 0.5)
        use_strong = getattr(self.config.training, 'use_strong_augmentation', False)
        
        if use_strong:
             return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                
                # Affine transforms (Safe geometric variations)
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.6),
                
                # Mild distortion
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
                
                # Intensity transforms
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                A.RandomGamma(gamma_limit=(85, 115), p=0.3),
                
                # Quality transforms
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
                
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
                ToTensorV2()
            ])
    
    def _get_val_transform(self):
        return A.Compose([ToTensorV2()])
    
    def _build_index(self):
        """建立 patch 檔案索引，按 patch_type 分桶"""
        self.samples = []  # 所有 samples
        
        # 按 patch_type 分桶
        self.buckets = {
            'positive': [],
            'hard_negative': [],
            'random_negative': [],
            'coverage': []
        }
        
        for patient_id in self.patient_ids:
            patient_dir = self.cache_dir / patient_id
            meta_path = patient_dir / "meta.json"
            
            if not meta_path.exists():
                continue
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            # 從 meta 中讀取 patch 資訊
            for slice_info in meta.get('patch_info', []):
                slice_idx = slice_info['slice_idx']
                for patch_data in slice_info['patches']:
                    patch_idx = patch_data['patch_idx']
                    has_lesion = patch_data.get('has_lesion', False)
                    
                    # 讀取 patch_type，舊格式用 has_lesion 推斷
                    patch_type = patch_data.get('patch_type', 
                                                'positive' if has_lesion else 'coverage')
                    
                    patch_path = patient_dir / f"slice_{slice_idx:04d}_patch_{patch_idx}.npz"
                    if patch_path.exists():
                        sample = {
                            'patient_id': patient_id,
                            'slice_idx': slice_idx,
                            'patch_idx': patch_idx,
                            'patch_path': str(patch_path),
                            'has_lesion': has_lesion,
                            'patch_type': patch_type,
                            'num_slices': meta['num_slices']
                        }
                        self.samples.append(sample)
                        
                        # 加入對應的 bucket
                        if patch_type in self.buckets:
                            self.buckets[patch_type].append(sample)
                        elif has_lesion:
                            self.buckets['positive'].append(sample)
                        else:
                            self.buckets['coverage'].append(sample)
    
    def _compute_epoch_length(self):
        """計算 epoch 長度，確保 train/val/test 一致"""
        n_pos = len(self.buckets['positive'])
        
        if self.mode == "train":
            # Train: epoch 長度 = 正樣本數 * 2（加快訓練迭代）
            self.epoch_length = n_pos * 2 if n_pos > 0 else len(self.samples)
            self.eval_samples = None  # train 不用固定 samples
        else:
            # Val/Test: 建立固定的 eval_samples，確保每次驗證可重現
            self._build_eval_samples()
            self.epoch_length = len(self.eval_samples)
    
    def _build_eval_samples(self):
        """
        建立用於驗證/測試的固定樣本列表（deterministic）
        """
        # FIX: Validation/Test should use ALL samples to reflect true distribution
        # No subsampling or ratio enforcement
        self.eval_samples = self.samples
        logger.info(f"   Using ALL {len(self.eval_samples)} samples for evaluation")
    
    def _log_stats(self):
        """Log 統計資訊"""
        stats = {k: len(v) for k, v in self.buckets.items()}
        total = sum(stats.values())
        
        logger.info(f"✅ CachedPatchDataset ({self.mode}): {total} patches from {len(self.patient_ids)} patients")
        logger.info(f"   ├─ positive: {stats['positive']}")
        logger.info(f"   ├─ hard_negative: {stats['hard_negative']}")
        logger.info(f"   ├─ random_negative: {stats['random_negative']}")
        logger.info(f"   └─ coverage: {stats['coverage']}")
        
        if self.mode == "train":
            logger.info(f"   Epoch length: {self.epoch_length}, Ratios: pos={self.pos_ratio:.1%}, hard={self.hard_neg_ratio:.1%}, rand={self.rand_neg_ratio:.1%}")
        else:
            logger.info(f"   Epoch length: {self.epoch_length} (deterministic eval samples)")
    
    def _sample_by_ratio(self) -> dict:
        """根據比例從 buckets 中抽樣一個 sample（只用於 train）"""
        r = random.random()
        
        # 優先使用 positive
        if r < self.pos_ratio and len(self.buckets['positive']) > 0:
            return random.choice(self.buckets['positive'])
        
        # 其次 hard_negative
        elif r < self.pos_ratio + self.hard_neg_ratio and len(self.buckets['hard_negative']) > 0:
            return random.choice(self.buckets['hard_negative'])
        
        # 再來 random_negative
        elif len(self.buckets['random_negative']) > 0:
            return random.choice(self.buckets['random_negative'])
        
        # Fallback: coverage 或任何有的
        elif len(self.buckets['coverage']) > 0:
            return random.choice(self.buckets['coverage'])
        
        # 最後 fallback: 任何 sample
        else:
            return random.choice(self.samples) if self.samples else None
    
    def __len__(self):
        return self.epoch_length
    
    def __getitem__(self, idx):
        # Train: ratio-based sampling（忽略 idx）
        # Val/Test: deterministic access（用 idx）
        if self.mode == "train":
            sample = self._sample_by_ratio()
        else:
            # Deterministic: 用 idx 從 eval_samples 取
            sample = self.eval_samples[idx % len(self.eval_samples)]
        
        if sample is None:
            raise RuntimeError("No samples available in dataset")
        
        # 載入 patch 資料
        data = np.load(sample['patch_path'], allow_pickle=True)
        image = data['image'].astype(np.float32)
        mask = data['mask'].astype(np.float32)
        
        # 檢查是否為 2.5D 格式
        is_2_5d = 'is_2_5d' in data.files and bool(data['is_2_5d'])
        
        if is_2_5d and image.ndim == 3:
            # 2.5D 格式: (C, H, W) -> (H, W, C) for albumentations
            # 支援任意 channel 數（3 或 5 都行）
            image = image.transpose(1, 2, 0)  # (H, W, C)
        elif image.ndim == 2:
            # 舊格式 2D: 複製 3 次
            if self.config.data.use_2_5d:
                image = np.stack([image, image, image], axis=-1)  # (H, W, 3)
            else:
                image = image[:, :, np.newaxis]  # (H, W, 1)
        
        # mask 保持 2D for albumentations
        # mask shape: (H, W)
        
        # 應用 transform
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']  # (C, H, W)
            mask = transformed['mask']    # (H, W)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1))
            mask = torch.from_numpy(mask)
        
        # 確保 mask 是 (1, H, W) 格式
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)
        elif mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.permute(2, 0, 1)  # (H, W, 1) -> (1, H, W)
        
        return {
            'image': image.float(),
            'mask': mask.float(),
            'patient_id': sample['patient_id'],
            'slice_idx': sample['slice_idx'],
            'patch_idx': sample['patch_idx'],
            'is_positive': sample['has_lesion'],  # 保持與原有 collate_fn 相容
            'has_lesion': sample['has_lesion'],
            'patch_type': sample['patch_type']
        }


def get_cached_patch_split(cache_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    從 cache 目錄獲取 patient 分割
    
    Returns:
        train_ids, val_ids, test_ids
    """
    cache_path = Path(cache_dir)
    
    # 找到所有有 meta.json 的病人
    patient_ids = []
    for patient_dir in sorted(cache_path.iterdir()):
        if patient_dir.is_dir() and (patient_dir / "meta.json").exists():
            patient_ids.append(patient_dir.name)
    
    logger.info(f"📂 從 cache 找到 {len(patient_ids)} 個病人")
    
    np.random.seed(seed)
    np.random.shuffle(patient_ids)
    
    n = len(patient_ids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_ids = patient_ids[:train_end]
    val_ids = patient_ids[train_end:val_end]
    test_ids = patient_ids[val_end:]
    
    logger.info(f"📊 資料分割: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    return train_ids, val_ids, test_ids
