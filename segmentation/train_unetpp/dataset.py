#!/usr/bin/env python3
"""
UNet++ 肺結節分割訓練 - 資料集模組
===================================

LNDb 資料集載入器，支援：
1. 2.5D 切片提取（固定 mm 距離）
2. 軟共識標註（多醫師）
3. Patch-based 訓練
4. 資料增強
"""

import logging
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys

# 支援直接執行和模組執行
try:
    from .preprocess import CTPreprocessor
    from .sampler import PatchSampler, SliceSampler
    from .config import Config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train_unetpp.preprocess import CTPreprocessor
    from train_unetpp.sampler import PatchSampler, SliceSampler
    from train_unetpp.config import Config


logger = logging.getLogger(__name__)


def val_collate_fn(batch):
    """
    Val/Test 用的自定義 collate function
    
    處理 dataset 返回的特殊格式：
    - images_4patch: (4, 3, ps, ps) -> (B, 4, 3, ps, ps)
    - positions: list of 4 tuples -> list of lists
    - full_mask: (1, H, W) -> list (因為 H,W 可能不同)
    - full_image_mid: (H, W) -> list (用於視覺化)
    - full_shape: tuple -> 分開收集
    """
    images = torch.stack([item['images_4patch'] for item in batch], dim=0)
    masks = [item['full_mask'] for item in batch]  # 保持 list 因為 shape 可能不同
    full_images = [item['full_image_mid'] for item in batch]  # 用於視覺化
    positions = [item['positions'] for item in batch]
    full_shapes = ([item['full_shape'][0] for item in batch], 
                   [item['full_shape'][1] for item in batch])
    patient_ids = [item['patient_id'] for item in batch]
    slice_idxs = [item['slice_idx'] for item in batch]
    is_positives = [item['is_positive'] for item in batch]
    
    return {
        'images_4patch': images,
        'positions': positions,
        'full_mask': masks,  # list of tensors
        'full_image_mid': full_images,  # list of tensors
        'full_shape': (torch.tensor(full_shapes[0]), torch.tensor(full_shapes[1])),
        'patient_id': patient_ids,
        'slice_idx': slice_idxs,
        'is_positive': is_positives
    }


class LNDbDataset(Dataset):
    """LNDb 肺結節分割資料集"""
    
    def __init__(
        self,
        data_dir: str,
        patient_ids: List[str],
        config: Config,
        mode: str = "train",
        transform: Optional[Callable] = None,
        preload: bool = False
    ):
        """
        初始化資料集
        
        Args:
            data_dir: LNDb 資料集根目錄
            patient_ids: 病人 ID 列表
            config: 配置物件
            mode: 'train', 'val', 或 'test'
            transform: 資料增強（若為 None 則使用預設）
            preload: 是否預載入所有資料到記憶體
        """
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.config = config
        self.mode = mode
        self.preload = preload
        
        # 初始化工具
        self.preprocessor = CTPreprocessor(
            target_spacing=config.data.target_spacing,
            hu_window_center=config.data.hu_window_center,
            hu_window_width=config.data.hu_window_width,
            cache_dir=config.data.cache_dir if config.data.cache_preprocessed else None
        )
        
        self.patch_sampler = PatchSampler(
            patch_size=config.data.patch_size,
            positive_ratio=config.data.positive_ratio,
            hard_negative_ratio=config.data.hard_negative_ratio,
            random_negative_ratio=config.data.random_negative_ratio
        )
        
        self.slice_sampler = SliceSampler(
            slice_distance_mm=config.data.slice_distance_mm
        )
        
        # 資料增強
        if transform is not None:
            self.transform = transform
        elif mode == "train" and config.training.use_augmentation:
            self.transform = self._get_train_transform()
        else:
            self.transform = self._get_val_transform()
        
        # 建立樣本索引
        self._build_sample_index()
        
        # 預載入資料
        if preload:
            self._preload_data()
    
    def _get_train_transform(self) -> A.Compose:
        """訓練時的資料增強"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
                border_mode=0
            ),
            A.ElasticTransform(
                alpha=50,
                sigma=5,
                p=0.3
            ),
            A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            ToTensorV2()
        ])
    
    def _get_val_transform(self) -> A.Compose:
        """驗證/測試時的轉換"""
        return A.Compose([
            ToTensorV2()
        ])
    
    def _build_sample_index(self):
        """建立樣本索引（使用快取加速）"""
        self.samples = []
        cache_dir = Path(self.config.data.cache_dir)
        
        # 檢查是否有索引快取
        index_cache_path = cache_dir / "_sample_index.json"
        if index_cache_path.exists():
            import json
            with open(index_cache_path, 'r') as f:
                all_patient_indices = json.load(f)
            
            # 只取需要的病人
            for patient_id in self.patient_ids:
                if patient_id in all_patient_indices:
                    patient_info = all_patient_indices[patient_id]
                    cache_path = str(cache_dir / f"{patient_id}.npz")
                    for z in range(patient_info['num_slices']):
                        self.samples.append({
                            'patient_id': patient_id,
                            'slice_idx': z,
                            'is_positive': z in patient_info['positive_slices'],
                            'cache_path': cache_path
                        })
        else:
            # 首次建立索引（會較慢）
            logger.info("首次建立樣本索引，請稍候...")
            all_patient_indices = {}
            
            from tqdm import tqdm
            for patient_id in tqdm(self.patient_ids, desc="Building index"):
                cache_path = cache_dir / f"{patient_id}.npz"
                
                if cache_path.exists():
                    data = np.load(cache_path, allow_pickle=True)
                    num_slices = data['volume'].shape[0]
                    
                    # 找正樣本切片
                    positive_slices = []
                    if len(data['masks']) > 0:
                        masks = data['masks']
                        if masks.ndim > 0 and len(masks) > 0:
                            soft_mask = self.preprocessor.create_soft_consensus_mask(list(masks))
                            positive_slices = self.slice_sampler.find_positive_slices(soft_mask)
                    
                    all_patient_indices[patient_id] = {
                        'num_slices': num_slices,
                        'positive_slices': positive_slices
                    }
                    
                    for z in range(num_slices):
                        self.samples.append({
                            'patient_id': patient_id,
                            'slice_idx': z,
                            'is_positive': z in positive_slices,
                            'cache_path': str(cache_path)
                        })
            
            # 保存索引快取
            import json
            with open(index_cache_path, 'w') as f:
                json.dump(all_patient_indices, f)
            logger.info(f"索引快取已保存: {index_cache_path}")
        
        logger.info(f"建立 {len(self.samples)} 個樣本索引 ({self.mode})")
        
        # 訓練模式下進行正樣本 oversampling
        if self.mode == "train":
            self._oversample_positives()
    
    def _oversample_positives(self):
        """正樣本 oversampling（限制總樣本數）"""
        positive_samples = [s for s in self.samples if s['is_positive']]
        negative_samples = [s for s in self.samples if not s['is_positive']]
        
        if len(positive_samples) == 0:
            logger.warning("沒有正樣本切片！")
            return
        
        # 獲取每 epoch 最大樣本數
        max_samples = getattr(self.config.training, 'max_samples_per_epoch', None)
        
        if max_samples and max_samples < len(positive_samples) + len(negative_samples):
            # 限制樣本數量
            target_positive = int(max_samples * self.config.data.positive_ratio)
            target_negative = max_samples - target_positive
            
            # 隨機選擇
            random.shuffle(positive_samples)
            random.shuffle(negative_samples)
            
            # 如果正樣本不夠，重複採樣
            if len(positive_samples) < target_positive:
                factor = target_positive // len(positive_samples) + 1
                positive_samples = (positive_samples * factor)[:target_positive]
            else:
                positive_samples = positive_samples[:target_positive]
            
            negative_samples = negative_samples[:target_negative]
        else:
            # 原始 oversampling 邏輯
            target_positive_ratio = self.config.data.positive_ratio
            target_positive_count = int(len(negative_samples) * target_positive_ratio / (1 - target_positive_ratio))
            
            if len(positive_samples) < target_positive_count:
                oversample_factor = target_positive_count // len(positive_samples) + 1
                positive_samples = positive_samples * oversample_factor
            
            positive_samples = positive_samples[:target_positive_count]
        
        # 合併並打亂
        self.samples = positive_samples + negative_samples
        random.shuffle(self.samples)
        
        logger.info(f"Oversampling 後: {len(positive_samples)} 正樣本, {len(negative_samples)} 負樣本, 總計 {len(self.samples)}")
    
    def _preload_data(self):
        """預載入所有資料到記憶體"""
        from tqdm import tqdm
        self.cached_data = {}
        
        unique_paths = list(set(s['cache_path'] for s in self.samples))
        logger.info(f"預載入 {len(unique_paths)} 個病人資料到記憶體...")
        
        for path in tqdm(unique_paths, desc="Preloading"):
            self.cached_data[path] = self.preprocessor.load_preprocessed(path)
        
        logger.info(f"預載入 {len(self.cached_data)} 個病人資料")
    
    def _load_patient_data(self, cache_path: str) -> Dict:
        """載入病人資料（直接從快取檔讀取）"""
        if self.preload and cache_path in self.cached_data:
            return self.cached_data[cache_path]
        # 直接從 .npz 讀取
        return self.preprocessor.load_preprocessed(cache_path)
    
    def _get_25d_slice(
        self,
        volume: np.ndarray,
        center_z: int,
        spacing_z: float
    ) -> np.ndarray:
        """
        提取 2.5D 切片
        
        Args:
            volume: 3D 影像 (Z, Y, X)
            center_z: 中心切片索引
            spacing_z: Z 方向 spacing
            
        Returns:
            2.5D 切片 (3, H, W)
        """
        indices = self.slice_sampler.get_25d_slice_indices(
            center_z,
            volume.shape[0],
            spacing_z
        )
        
        slices = [volume[idx] for idx in indices]
        return np.stack(slices, axis=0)  # (3, H, W)
    
    def _extract_patch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        lung_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取 Patch
        
        Args:
            image: 2.5D 影像 (3, H, W)
            mask: 2D 遮罩 (H, W)
            lung_mask: 2D 肺野遮罩 (H, W)
            
        Returns:
            (image_patch, mask_patch)
        """
        # 在訓練模式下使用採樣策略
        if self.mode == "train":
            patches = self.patch_sampler.sample_patches_for_slice(
                (image.shape[1], image.shape[2]),
                mask,
                lung_mask,
                num_patches=1
            )
            y1, y2, x1, x2, _ = patches[0]
        else:
            # 驗證/測試模式：中心裁切
            h, w = image.shape[1], image.shape[2]
            ps = self.config.data.patch_size
            y1 = (h - ps) // 2
            x1 = (w - ps) // 2
            y2 = y1 + ps
            x2 = x1 + ps
        
        image_patch = image[:, y1:y2, x1:x2]
        mask_patch = mask[y1:y2, x1:x2]
        
        return image_patch, mask_patch
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 載入資料
        data = self._load_patient_data(sample['cache_path'])
        
        volume = data['volume']  # (Z, Y, X), normalized [0, 1]
        lung_mask = data['lung_mask']
        spacing = data['spacing']
        
        # 創建軟共識遮罩
        if len(data['masks']) > 0:
            if self.config.data.consensus_method == "soft":
                mask = self.preprocessor.create_soft_consensus_mask(data['masks'])
            elif self.config.data.consensus_method == "intersection":
                mask = np.minimum.reduce(data['masks'])
            elif self.config.data.consensus_method == "union":
                mask = np.maximum.reduce(data['masks'])
            else:  # rad1
                mask = data['masks'][0] if len(data['masks']) > 0 else np.zeros_like(volume)
        else:
            mask = np.zeros_like(volume)
        
        slice_idx = sample['slice_idx']
        mask_2d = mask[slice_idx]
        lung_mask_2d = lung_mask[slice_idx]
        
        # 根據模式提取切片
        use_2d = getattr(self.config.model, 'use_2d', False)
        if use_2d:
            # 純 2D 模式：單一切片
            image_slice = volume[slice_idx]  # (H, W)
            image_slice = image_slice[np.newaxis, ...]  # (1, H, W)
        else:
            # 2.5D 模式：3 個相鄰切片
            image_slice = self._get_25d_slice(volume, slice_idx, spacing[2])  # (3, H, W)
        
        # 提取 Patch
        image_patch, mask_patch = self._extract_patch(image_slice, mask_2d, lung_mask_2d)
        
        # Albumentations 需要 (H, W, C) 格式
        image_patch = np.transpose(image_patch, (1, 2, 0))  # (H, W, C)
        
        # 應用增強
        if self.transform:
            transformed = self.transform(image=image_patch, mask=mask_patch)
            image_tensor = transformed['image'].float()  # (C, H, W)
            mask_tensor = transformed['mask'].float()    # (H, W)
        else:
            image_tensor = torch.from_numpy(image_patch.transpose(2, 0, 1)).float()
            mask_tensor = torch.from_numpy(mask_patch).float()
        
        # 確保 mask 有 channel 維度
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)  # (1, H, W)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'patient_id': sample['patient_id'],
            'slice_idx': slice_idx,
            'is_positive': sample['is_positive']
        }


class LNDbInferenceDataset(Dataset):
    """LNDb 推論資料集（處理完整 3D volume）"""
    
    def __init__(
        self,
        data_dir: str,
        patient_ids: List[str],
        config: Config
    ):
        """
        初始化推論資料集
        
        Args:
            data_dir: LNDb 資料集根目錄
            patient_ids: 病人 ID 列表
            config: 配置物件
        """
        self.data_dir = Path(data_dir)
        self.patient_ids = patient_ids
        self.config = config
        
        self.preprocessor = CTPreprocessor(
            target_spacing=config.data.target_spacing,
            hu_window_center=config.data.hu_window_center,
            hu_window_width=config.data.hu_window_width,
            cache_dir=config.data.cache_dir
        )
        
        self.slice_sampler = SliceSampler(
            slice_distance_mm=config.data.slice_distance_mm
        )
    
    def __len__(self) -> int:
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        patient_id = self.patient_ids[idx]
        cache_path = Path(self.config.data.cache_dir) / f"{patient_id}.npz"
        
        if not cache_path.exists():
            raise FileNotFoundError(f"快取不存在: {cache_path}")
        
        data = self.preprocessor.load_preprocessed(str(cache_path))
        
        volume = data['volume']  # (Z, Y, X)
        spacing = data['spacing']
        
        # 準備所有切片的 2.5D 輸入
        slices_25d = []
        for z in range(volume.shape[0]):
            indices = self.slice_sampler.get_25d_slice_indices(z, volume.shape[0], spacing[2])
            slice_25d = np.stack([volume[i] for i in indices], axis=0)
            slices_25d.append(slice_25d)
        
        volume_25d = np.stack(slices_25d, axis=0)  # (Z, 3, H, W)
        
        # 創建軟共識遮罩（如果有）
        if len(data['masks']) > 0:
            gt_mask = self.preprocessor.create_soft_consensus_mask(data['masks'])
        else:
            gt_mask = None
        
        return {
            'patient_id': patient_id,
            'volume': torch.from_numpy(volume_25d).float(),
            'spacing': torch.from_numpy(spacing).float(),
            'lung_mask': torch.from_numpy(data['lung_mask'].astype(np.float32)),
            'gt_mask': torch.from_numpy(gt_mask) if gt_mask is not None else None,
            'original_shape': data['original_shape'],
            'bbox': data['bbox']
        }


def get_patient_split(
    data_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    隨機分割病人 ID
    
    Returns:
        (train_ids, val_ids, test_ids)
    """
    # 優先使用切片式快取目錄
    slice_cache_dir = Path(data_dir).parent / 'cache' / 'lndb_slices'
    cache_dir = Path(data_dir).parent / 'cache' / 'lndb_preprocessed'
    
    patient_ids = []
    
    # 1. 優先從切片式快取讀取
    if slice_cache_dir.exists():
        patient_ids = [d.name for d in slice_cache_dir.iterdir() if d.is_dir() and (d / 'meta.json').exists()]
    
    # 2. 降級使用 3D 快取
    if len(patient_ids) == 0 and cache_dir.exists():
        patient_ids = [f.stem for f in cache_dir.glob('*.npz')]
    
    # 3. 從原始資料夾讀取
    if len(patient_ids) == 0:
        data_path = Path(data_dir)
        for subfolder in ['data0', 'data1', 'data2', 'data3', 'data4', 'data5']:
            folder = data_path / subfolder
            if folder.exists():
                patient_ids.extend([f.stem for f in folder.glob('*.mhd')])
    
    patient_ids = sorted(set(patient_ids))
    
    # 過濾只保留有 ≥3mm 結節的病人
    nodule_csv = Path(data_dir) / 'trainset_csv' / 'trainNodules_gt.csv'
    if nodule_csv.exists():
        import pandas as pd
        df = pd.read_csv(nodule_csv)
        # Nodule=1 表示是結節（≥3mm 有分割遮罩）
        nodule_patients = set(f"LNDb-{int(pid):04d}" for pid in df[df['Nodule'] == 1]['LNDbID'].unique())
        original_count = len(patient_ids)
        patient_ids = [pid for pid in patient_ids if pid in nodule_patients]
        logger.info(f"過濾 ≥3mm 結節病人: {original_count} -> {len(patient_ids)} (保留 {len(patient_ids)} 個有結節的病人)")
    
    # 隨機打亂
    np.random.seed(seed)
    np.random.shuffle(patient_ids)
    
    # 分割
    n = len(patient_ids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_ids = patient_ids[:train_end]
    val_ids = patient_ids[train_end:val_end]
    test_ids = patient_ids[val_end:]
    
    return train_ids, val_ids, test_ids


def get_fold_split(
    data_dir: str,
    fold_id: int,
    num_folds: int = 5
) -> Tuple[List[str], List[str]]:
    """
    根據 trainFolds.csv 獲取交叉驗證分割
    
    Args:
        data_dir: LNDb 資料集根目錄
        fold_id: fold 編號 (0 到 num_folds-1)
        num_folds: fold 總數
        
    Returns:
        (train_ids, val_ids)
    """
    folds_path = Path(data_dir) / 'trainset_csv' / 'trainFolds.csv'
    
    if folds_path.exists():
        df = pd.read_csv(folds_path)
        # 假設欄位名稱
        id_col = 'LNDbID' if 'LNDbID' in df.columns else df.columns[0]
        fold_col = 'fold' if 'fold' in df.columns else df.columns[1]
        
        val_ids = df[df[fold_col] == fold_id][id_col].tolist()
        train_ids = df[df[fold_col] != fold_id][id_col].tolist()
    else:
        logger.warning(f"找不到 {folds_path}，使用隨機分割")
        train_ids, val_ids, _ = get_patient_split(data_dir, seed=42 + fold_id)
    
    return train_ids, val_ids


class LNDbSliceDataset(Dataset):
    """切片式 LNDb 資料集（直接讀取單一切片，加速 2D 訓練）"""
    
    def __init__(
        self,
        slice_cache_dir: str,
        patient_ids: list,
        config,
        mode: str = "train",
        transform=None
    ):
        """
        初始化切片式資料集
        
        Args:
            slice_cache_dir: 切片快取目錄（每個病人一個子目錄）
            patient_ids: 病人 ID 列表
            config: 配置物件
            mode: 'train', 'val', 或 'test'
        """
        self.slice_cache_dir = Path(slice_cache_dir)
        self.patient_ids = patient_ids
        self.config = config
        self.mode = mode
        
        self.patch_sampler = PatchSampler(
            patch_size=config.data.patch_size,
            positive_ratio=config.data.positive_ratio,
            hard_negative_ratio=config.data.hard_negative_ratio,
            random_negative_ratio=config.data.random_negative_ratio
        )
        
        # 資料增強
        if transform is not None:
            self.transform = transform
        elif mode == "train" and config.training.use_augmentation:
            self.transform = self._get_train_transform()
        else:
            self.transform = self._get_val_transform()
        
        # 建立樣本索引
        self._build_sample_index()
        
        # 訓練模式下進行正樣本 oversampling
        if mode == "train":
            self._oversample_positives()
    
    def _get_train_transform(self):
        """訓練時的資料增強"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
            ToTensorV2()
        ])
    
    def _get_val_transform(self):
        """驗證時只轉張量"""
        return A.Compose([ToTensorV2()])
    
    def _build_sample_index(self):
        """建立樣本索引（從 meta.json 讀取）"""
        self.samples = []
        
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
                # Train/Val/Test: 每個 slice 一個樣本
                # Val/Test 的 4-patch cropping 在 __getitem__ 中處理
                self.samples.append({
                    'patient_id': patient_id,
                    'slice_idx': z,
                    'is_positive': z in positive_slices,
                    'slice_path': str(patient_dir / f"slice_{z:04d}.npz"),
                    'num_slices': meta['num_slices']
                })
        
        logger.info(f"建立 {len(self.samples)} 個切片樣本索引 ({self.mode})")
    
    def _oversample_positives(self):
        """正樣本 oversampling（保持正負樣本平衡）"""
        positive_samples = [s for s in self.samples if s['is_positive']]
        negative_samples = [s for s in self.samples if not s['is_positive']]
        
        if len(positive_samples) == 0:
            logger.warning("沒有正樣本切片！")
            return
        
        max_samples = getattr(self.config.training, 'max_samples_per_epoch', None)
        positive_ratio = self.config.data.positive_ratio
        
        if max_samples:
            # 有限制時，限制總樣本數並按比例分配
            total_samples = min(max_samples, len(positive_samples) + len(negative_samples))
        else:
            # 無限制時，使用所有樣本但保持比例
            # 以負樣本數為基準，正樣本 oversample 到對應比例
            total_samples = len(negative_samples) + len(positive_samples)
        
        target_positive = int(total_samples * positive_ratio)
        target_negative = total_samples - target_positive
        
        random.shuffle(positive_samples)
        random.shuffle(negative_samples)
        
        # 正樣本 oversample（複製以達到目標數量）
        if len(positive_samples) < target_positive:
            factor = target_positive // len(positive_samples) + 1
            positive_samples = (positive_samples * factor)[:target_positive]
        else:
            positive_samples = positive_samples[:target_positive]
        
        # 負樣本下採樣
        negative_samples = negative_samples[:target_negative]
        
        self.samples = positive_samples + negative_samples
        random.shuffle(self.samples)
        
        logger.info(f"Oversampling 後: {len(positive_samples)} 正樣本, {len(negative_samples)} 負樣本, 總計 {len(self.samples)}")
    
    def _get_4patch_centers(self, lung_mask: np.ndarray, patch_size: int):
        """
        計算 Val/Test 用的 4 個 deterministic patch centers
        
        基於 lung bbox 的 1/3 和 2/3 位置產生 2x2 grid
        
        Returns:
            List of (center_y, center_x) tuples
        """
        h, w = lung_mask.shape
        half = patch_size // 2
        
        # 找到 lung bbox
        lung_y, lung_x = np.where(lung_mask > 0)
        if len(lung_y) == 0:
            # 沒有 lung mask，fallback 到圖像中心
            cy, cx = h // 2, w // 2
            return [(cy, cx)] * 4
        
        y0, y1 = lung_y.min(), lung_y.max()
        x0, x1 = lung_x.min(), lung_x.max()
        
        # 計算 1/3 和 2/3 位置
        cy1 = int(y0 + (y1 - y0) / 3)
        cy2 = int(y0 + 2 * (y1 - y0) / 3)
        cx1 = int(x0 + (x1 - x0) / 3)
        cx2 = int(x0 + 2 * (x1 - x0) / 3)
        
        # 確保 centers 在有效範圍內
        centers = []
        for cy, cx in [(cy1, cx1), (cy1, cx2), (cy2, cx1), (cy2, cx2)]:
            cy = max(half, min(h - half, cy))
            cx = max(half, min(w - half, cx))
            centers.append((cy, cx))
        
        return centers
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        patient_id = sample['patient_id']
        slice_idx = sample['slice_idx']
        num_slices = sample.get('num_slices', 100)
        
        # === 2.5D: 讀取 z-1, z, z+1 三個切片 ===
        patient_dir = self.slice_cache_dir / patient_id
        
        # 計算鄰近切片索引（邊界 replicate）
        z_prev = max(0, slice_idx - 1)
        z_curr = slice_idx
        z_next = min(num_slices - 1, slice_idx + 1)
        
        # 載入三個切片
        slices_2_5d = []
        for z in [z_prev, z_curr, z_next]:
            slice_path = patient_dir / f"slice_{z:04d}.npz"
            data = np.load(slice_path, allow_pickle=True)
            slices_2_5d.append(data['image'].astype(np.float32))
        
        # 讀取中心切片的 mask 和 lung_mask
        center_data = np.load(sample['slice_path'], allow_pickle=True)
        mask = center_data['mask'].astype(np.float32)    # (H, W)
        lung_mask = center_data['lung_mask']             # (H, W)
        
        # 組裝 2.5D image: (3, H, W)
        image_2_5d = np.stack(slices_2_5d, axis=0)  # (3, H, W)
        
        patch_size = self.config.data.patch_size
        h, w = mask.shape
        half = patch_size // 2
        
        if self.mode == "train":
            # === Train 模式：單一 patch ===
            if sample['is_positive']:
                # 以結節為中心採樣
                nodule_y, nodule_x = np.where(mask > 0.5)
                if len(nodule_y) > 0:
                    center_idx = len(nodule_y) // 2
                    center_y, center_x = nodule_y[center_idx], nodule_x[center_idx]
                    # 加入抖動
                    jitter = int(patch_size * 0.3)
                    center_y += random.randint(-jitter, jitter)
                    center_x += random.randint(-jitter, jitter)
                else:
                    center_y, center_x = h // 2, w // 2
            else:
                # 負樣本：在 lung_mask 區域內隨機採樣
                lung_y, lung_x = np.where(lung_mask > 0)
                if len(lung_y) > 0:
                    rand_idx = random.randint(0, len(lung_y) - 1)
                    center_y, center_x = lung_y[rand_idx], lung_x[rand_idx]
                else:
                    center_y, center_x = h // 2, w // 2
            
            # 確保 center 在有效範圍內
            center_y = max(half, min(h - half, center_y))
            center_x = max(half, min(w - half, center_x))
            
            # 裁切 Patch
            y1, y2 = center_y - half, center_y + half
            x1, x2 = center_x - half, center_x + half
            
            image_patch = image_2_5d[:, y1:y2, x1:x2]
            mask_patch = mask[y1:y2, x1:x2]
            
            # Padding 如果不足
            if image_patch.shape[1] < patch_size or image_patch.shape[2] < patch_size:
                padded_img = np.zeros((3, patch_size, patch_size), dtype=np.float32)
                padded_mask = np.zeros((patch_size, patch_size), dtype=np.float32)
                padded_img[:, :image_patch.shape[1], :image_patch.shape[2]] = image_patch
                padded_mask[:mask_patch.shape[0], :mask_patch.shape[1]] = mask_patch
                image_patch, mask_patch = padded_img, padded_mask
            
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
                'is_positive': sample['is_positive']
            }
        
        else:
            # === Val/Test 模式：返回 4 個 patches + full_mask ===
            centers = self._get_4patch_centers(lung_mask, patch_size)
            
            image_patches = []
            positions = []
            
            for cy, cx in centers:
                y1, y2 = cy - half, cy + half
                x1, x2 = cx - half, cx + half
                
                image_patch = image_2_5d[:, y1:y2, x1:x2]
                
                # Padding
                if image_patch.shape[1] < patch_size or image_patch.shape[2] < patch_size:
                    padded = np.zeros((3, patch_size, patch_size), dtype=np.float32)
                    padded[:, :image_patch.shape[1], :image_patch.shape[2]] = image_patch
                    image_patch = padded
                
                image_patches.append(torch.from_numpy(image_patch).float())
                positions.append((y1, x1))  # 左上角座標
            
            # Stack 4 patches: (4, 3, ps, ps)
            images_4patch = torch.stack(image_patches, dim=0)
            
            # Full mask
            full_mask = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)
            
            # Full image 中間 channel（用於視覺化）
            full_image_mid = torch.from_numpy(image_2_5d[1]).float()  # (H, W)
            
            return {
                'images_4patch': images_4patch,  # (4, 3, 224, 224)
                'positions': positions,           # [(y1, x1), ...] 4 個
                'full_mask': full_mask,           # (1, H, W)
                'full_image_mid': full_image_mid, # (H, W) 用於視覺化
                'full_shape': (h, w),
                'patient_id': patient_id,
                'slice_idx': slice_idx,
                'is_positive': sample['is_positive']
            }


if __name__ == "__main__":
    # 測試資料集
    try:
        from .config import get_default_config
    except ImportError:
        from train_unetpp.config import get_default_config
    
    config = get_default_config()
    
    # 獲取分割
    train_ids, val_ids, test_ids = get_patient_split(config.data.data_dir)
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    if len(train_ids) > 0:
        # 創建資料集
        dataset = LNDbDataset(
            config.data.data_dir,
            train_ids[:5],  # 只用 5 個病人測試
            config,
            mode="train"
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Image shape: {sample['image'].shape}")
            print(f"Mask shape: {sample['mask'].shape}")
            print(f"Patient ID: {sample['patient_id']}")
