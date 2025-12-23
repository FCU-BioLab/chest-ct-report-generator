#!/usr/bin/env python3
"""
MSD Lung Tumours 預處理與資料集
================================

統一 4-Patch 邏輯：
- Train/Val/Test 都使用相同的 4-patch 提取方式
- 正負樣本判定基於**每個 patch 內是否有 lesion**
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import random

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

try:
    from .patch_utils import compute_4patch_positions, extract_patch_with_lung_mask
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train_unetpp.patch_utils import compute_4patch_positions, extract_patch_with_lung_mask

logger = logging.getLogger(__name__)

MSD_LUNG_DIR = Path(r"E:\lung_ct_lesion_dataset\MSD Lung Tumours\Task06_Lung")
MSD_CACHE_DIR = Path(r"C:\GitHub\chest-ct-report-generator\segmentation\cache\msd_lung_slices")
MSD_LUNGMASK_DIR = Path(r"E:\lung_ct_lesion_dataset\MSD Lung Tumours\lung_masks")


# =============================================================================
# 預處理器（保持不變）
# =============================================================================

class MSDLungPreprocessor:
    """MSD Lung Tumours 預處理器"""
    
    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        hu_window_center: float = -600,
        hu_window_width: float = 1500,
        cache_dir: Optional[Path] = None,
        lungmask_dir: Optional[Path] = None
    ):
        self.target_spacing = target_spacing
        self.hu_window_center = hu_window_center
        self.hu_window_width = hu_window_width
        self.cache_dir = cache_dir or MSD_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lungmask_dir = lungmask_dir or MSD_LUNGMASK_DIR
    
    def load_nifti(self, nii_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        nii = nib.load(str(nii_path))
        volume = nii.get_fdata()
        if volume.ndim == 4:
            volume = volume[:, :, :, 0]
        volume = np.transpose(volume, (2, 1, 0))
        spacing = np.array(nii.header.get_zooms()[:3])
        return volume.astype(np.float32), spacing
    
    def load_lungmask(self, case_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        lung_path = self.lungmask_dir / f"{case_id}_lung.nii.gz"
        if not lung_path.exists():
            return None, None
        lung_nii = nib.load(str(lung_path))
        lung_array = lung_nii.get_fdata()
        if lung_array.ndim == 4:
            lung_array = lung_array[:, :, :, 0]
        lung_array = np.transpose(lung_array, (2, 1, 0))
        spacing = np.array(lung_nii.header.get_zooms()[:3])
        return (lung_array > 0).astype(np.float32), spacing
    
    def resample_volume(self, volume: np.ndarray, original_spacing: np.ndarray, is_mask: bool = False) -> np.ndarray:
        zoom_factors = [
            original_spacing[2] / self.target_spacing[2],
            original_spacing[1] / self.target_spacing[1],
            original_spacing[0] / self.target_spacing[0],
        ]
        return zoom(volume, zoom_factors, order=0 if is_mask else 3)
    
    def normalize_hu(self, volume: np.ndarray) -> np.ndarray:
        wl, ww = self.hu_window_center, self.hu_window_width
        min_hu, max_hu = wl - ww / 2, wl + ww / 2
        volume = np.clip(volume, min_hu, max_hu)
        return ((volume - min_hu) / (max_hu - min_hu)).astype(np.float32)
    
    def get_lung_bbox_from_mask(self, lung_mask: np.ndarray, margin: int = 10):
        z_idx, y_idx, x_idx = np.where(lung_mask > 0)
        if len(z_idx) == 0:
            return ((0, lung_mask.shape[0]), (0, lung_mask.shape[1]), (0, lung_mask.shape[2]))
        return (
            (max(0, z_idx.min() - margin), min(lung_mask.shape[0], z_idx.max() + margin)),
            (max(0, y_idx.min() - margin), min(lung_mask.shape[1], y_idx.max() + margin)),
            (max(0, x_idx.min() - margin), min(lung_mask.shape[2], x_idx.max() + margin))
        )
    
    def preprocess_case(self, case_id: str, image_path: Path, label_path: Optional[Path] = None) -> Dict:
        volume, spacing = self.load_nifti(image_path)
        if label_path and label_path.exists():
            mask, _ = self.load_nifti(label_path)
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.zeros_like(volume)
        
        lung_mask, _ = self.load_lungmask(case_id)
        if lung_mask is None:
            lung_mask = ((volume > -1000) & (volume < -200)).astype(np.float32)
        
        volume_rs = self.resample_volume(volume, spacing, is_mask=False)
        mask_rs = self.resample_volume(mask, spacing, is_mask=True)
        lung_rs = (self.resample_volume(lung_mask, spacing, is_mask=True) > 0.5).astype(np.float32)
        
        (z0, z1), (y0, y1), (x0, x1) = self.get_lung_bbox_from_mask(lung_rs)
        volume_crop = self.normalize_hu(volume_rs[z0:z1, y0:y1, x0:x1])
        mask_crop = mask_rs[z0:z1, y0:y1, x0:x1]
        lung_crop = lung_rs[z0:z1, y0:y1, x0:x1]
        
        positive_slices = [z for z in range(mask_crop.shape[0]) if mask_crop[z].sum() > 0]
        
        return {
            'volume': volume_crop,
            'mask': mask_crop,
            'lung_mask': lung_crop,
            'positive_slices': positive_slices,
            'spacing': self.target_spacing
        }
    
    def save_slices(self, case_id: str, data: Dict, positive_only: bool = True) -> int:
        """
        保存切片到快取
        
        Args:
            case_id: 案例 ID
            data: 預處理結果
            positive_only: 只保存有病灶的切片（預設 True）
        """
        case_dir = self.cache_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        
        volume, mask, lung_mask = data['volume'], data['mask'], data['lung_mask']
        positive_slices = set(data['positive_slices'])
        
        if positive_only:
            slices_to_save = sorted(positive_slices)
        else:
            slices_to_save = list(range(volume.shape[0]))
        
        saved_count = 0
        for z in slices_to_save:
            np.savez_compressed(case_dir / f"slice_{z:04d}.npz",
                image=volume[z].astype(np.float16),
                mask=mask[z].astype(np.float16),
                lung_mask=lung_mask[z].astype(np.bool_))
            saved_count += 1
        
        with open(case_dir / "meta.json", 'w') as f:
            json.dump({
                'case_id': case_id,
                'num_slices': volume.shape[0],  # 原始總數
                'saved_slices': slices_to_save,  # 實際保存的切片索引
                'positive_slices': data['positive_slices'],
                'positive_only': positive_only,
                'spacing': list(data['spacing']),
                'shape': list(volume.shape)
            }, f, indent=2)
        
        return saved_count


def get_msd_lung_cases(data_dir: Path = MSD_LUNG_DIR) -> List[Tuple[str, Path, Optional[Path]]]:
    cases = []
    images_tr = data_dir / "imagesTr"
    labels_tr = data_dir / "labelsTr"
    
    if images_tr.exists():
        for nii_file in sorted(images_tr.glob("lung_*.nii.gz")):
            if nii_file.name.startswith("._"):
                continue
            case_id = nii_file.stem.replace(".nii", "")
            label_path = labels_tr / f"{case_id}.nii.gz"
            cases.append((case_id, nii_file, label_path if label_path.exists() else None))
    
    return cases


def get_msd_train_val_split(
    cases: List[Tuple[str, Path, Optional[Path]]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    labeled_cases = [c[0] for c in cases if c[2] is not None]
    random.seed(seed)
    random.shuffle(labeled_cases)
    n = len(labeled_cases)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    return (
        labeled_cases[test_size + val_size:],
        labeled_cases[test_size:test_size + val_size],
        labeled_cases[:test_size]
    )


def preprocess_msd_lung(data_dir: Path = MSD_LUNG_DIR, cache_dir: Path = MSD_CACHE_DIR):
    logger.info(f"預處理 MSD Lung: {data_dir} -> {cache_dir}")
    preprocessor = MSDLungPreprocessor(cache_dir=cache_dir)
    cases = get_msd_lung_cases(data_dir)
    
    total_slices, total_positive = 0, 0
    for case_id, image_path, label_path in tqdm(cases, desc="Preprocessing"):
        if label_path is None:
            continue
        try:
            data = preprocessor.preprocess_case(case_id, image_path, label_path)
            num_slices = preprocessor.save_slices(case_id, data)
            total_slices += num_slices
            total_positive += len(data['positive_slices'])
        except Exception as e:
            logger.error(f"處理 {case_id} 失敗: {e}")
    
    logger.info(f"完成: {total_slices} slices, {total_positive} positive")


# =============================================================================
# MSD 資料集（統一 4-Patch 邏輯）
# =============================================================================

class MSDLungSliceDataset(Dataset):
    """
    MSD Lung Tumours 切片資料集
    
    統一 4-Patch 邏輯：
    - Train: 返回單一 patch（patch-level oversampling）
    - Val/Test: 返回 4 patches
    - 正負樣本判定：基於 patch 內是否有 lesion
    """
    
    def __init__(
        self,
        case_ids: List[str],
        cache_dir: Path = MSD_CACHE_DIR,
        mode: str = "train",
        patch_size: int = 224,
        positive_ratio: float = 0.7,
        transform: Optional[A.Compose] = None
    ):
        self.case_ids = case_ids
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.patch_size = patch_size
        self.positive_ratio = positive_ratio
        
        if transform is not None:
            self.transform = transform
        elif mode == "train":
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([ToTensorV2()])
        
        self.samples = []
        self.patch_samples = []
        self._build_patch_index()
        
        if mode == "train":
            self._oversample_patches()
    
    def _build_patch_index(self):
        """建立 patch-level 索引"""
        for case_id in self.case_ids:
            case_dir = self.cache_dir / case_id
            meta_path = case_dir / "meta.json"
            
            if not meta_path.exists():
                continue
            
            with open(meta_path) as f:
                meta = json.load(f)
            
            positive_slices = set(meta['positive_slices'])
            
            # 支援新格式（只保存正樣本切片）和舊格式（保存所有切片）
            if 'saved_slices' in meta:
                slices_to_iterate = meta['saved_slices']
            else:
                slices_to_iterate = list(range(meta['num_slices']))
            
            for z in slices_to_iterate:
                slice_path = case_dir / f"slice_{z:04d}.npz"
                if not slice_path.exists():
                    continue
                
                slice_info = {
                    'case_id': case_id,
                    'slice_idx': z,
                    'is_positive': z in positive_slices,
                    'slice_path': str(slice_path),
                    'num_slices': len(slices_to_iterate)  # 使用實際保存的切片數
                }
                self.samples.append(slice_info)
                
                if self.mode == "train":
                    # 載入 mask 判斷每個 patch 是否為正
                    data = np.load(slice_info['slice_path'])
                    mask = data['mask'].astype(np.float32)
                    lung_mask = data['lung_mask'].astype(np.float32)
                    
                    patch_positions = compute_4patch_positions(lung_mask, self.patch_size)
                    
                    for patch_idx, ((py1, px1), (py2, px2)) in enumerate(patch_positions):
                        py1_c, py2_c = max(0, py1), min(mask.shape[0], py2)
                        px1_c, px2_c = max(0, px1), min(mask.shape[1], px2)
                        
                        patch_mask_area = mask[py1_c:py2_c, px1_c:px2_c].sum()
                        is_patch_positive = patch_mask_area > 0
                        
                        self.patch_samples.append({
                            **slice_info,
                            'patch_idx': patch_idx,
                            'is_patch_positive': is_patch_positive
                        })
        
        if self.mode == "train":
            pos = sum(1 for s in self.patch_samples if s['is_patch_positive'])
            logger.info(f"Patch-level 索引: {pos} 正 / {len(self.patch_samples) - pos} 負")
        else:
            logger.info(f"建立 {len(self.samples)} 個切片索引 ({self.mode})")
    
    def _oversample_patches(self):
        """Patch-level oversampling"""
        positives = [s for s in self.patch_samples if s['is_patch_positive']]
        negatives = [s for s in self.patch_samples if not s['is_patch_positive']]
        
        if not positives:
            self.patch_samples = negatives
            return
        
        total = len(positives) + len(negatives)
        target_pos = int(total * self.positive_ratio)
        target_neg = total - target_pos
        
        random.shuffle(positives)
        random.shuffle(negatives)
        
        if len(positives) < target_pos:
            factor = target_pos // len(positives) + 1
            positives = (positives * factor)[:target_pos]
        else:
            positives = positives[:target_pos]
        
        negatives = negatives[:target_neg]
        
        self.patch_samples = positives + negatives
        random.shuffle(self.patch_samples)
        
        logger.info(f"Oversampling: {len(positives)} 正 + {len(negatives)} 負 = {len(self.patch_samples)}")
    
    def _load_2_5d_slice(self, case_dir: Path, slice_idx: int, num_slices: int):
        """
        載入 2.5D 切片 (z-1, z, z+1)
        
        如果鄰近切片不存在（positive_only 模式），則複製中心切片
        """
        center_path = case_dir / f"slice_{slice_idx:04d}.npz"
        center_data = np.load(center_path)
        center_image = center_data['image'].astype(np.float32)
        
        # 嘗試載入前一個切片
        prev_path = case_dir / f"slice_{slice_idx - 1:04d}.npz"
        if prev_path.exists():
            prev_image = np.load(prev_path)['image'].astype(np.float32)
        else:
            prev_image = center_image  # 複製中心切片
        
        # 嘗試載入下一個切片
        next_path = case_dir / f"slice_{slice_idx + 1:04d}.npz"
        if next_path.exists():
            next_image = np.load(next_path)['image'].astype(np.float32)
        else:
            next_image = center_image  # 複製中心切片
        
        return np.stack([prev_image, center_image, next_image], axis=0)
    
    def __len__(self):
        return len(self.patch_samples) if self.mode == "train" else len(self.samples)
    
    def __getitem__(self, idx):
        if self.mode == "train":
            sample = self.patch_samples[idx]
            case_dir = self.cache_dir / sample['case_id']
            
            image_2_5d = self._load_2_5d_slice(case_dir, sample['slice_idx'], sample['num_slices'])
            center_data = np.load(sample['slice_path'])
            mask = center_data['mask'].astype(np.float32)
            lung_mask = center_data['lung_mask'].astype(np.float32)
            
            patch_positions = compute_4patch_positions(lung_mask, self.patch_size)
            (py1, px1), (py2, px2) = patch_positions[sample['patch_idx']]
            
            image_patch, mask_patch, _ = extract_patch_with_lung_mask(
                image_2_5d, mask, lung_mask, ((py1, px1), (py2, px2)), self.patch_size
            )
            
            image_for_aug = np.transpose(image_patch, (1, 2, 0))
            transformed = self.transform(image=image_for_aug, mask=mask_patch)
            
            mask_tensor = transformed['mask'].float()
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            
            return {
                'image': transformed['image'].float(),
                'mask': mask_tensor,
                'case_id': sample['case_id'],
                'slice_idx': sample['slice_idx'],
                'patch_idx': sample['patch_idx'],
                'is_positive': sample['is_patch_positive']
            }
        
        else:
            sample = self.samples[idx]
            case_dir = self.cache_dir / sample['case_id']
            
            image_2_5d = self._load_2_5d_slice(case_dir, sample['slice_idx'], sample['num_slices'])
            center_data = np.load(sample['slice_path'])
            mask = center_data['mask'].astype(np.float32)
            lung_mask = center_data['lung_mask'].astype(np.float32)
            
            h, w = mask.shape
            patch_positions = compute_4patch_positions(lung_mask, self.patch_size)
            
            image_patches = []
            positions = []
            
            for (py1, px1), (py2, px2) in patch_positions:
                image_patch, _, _ = extract_patch_with_lung_mask(
                    image_2_5d, mask, lung_mask, ((py1, px1), (py2, px2)), self.patch_size
                )
                image_patches.append(torch.from_numpy(image_patch).float())
                positions.append((py1, px1))
            
            full_mask = mask.copy()
            full_mask[lung_mask == 0] = 0
            
            full_image_mid = image_2_5d[1].copy()
            full_image_mid[lung_mask == 0] = 0
            
            return {
                'images_4patch': torch.stack(image_patches, dim=0),
                'positions': positions,
                'full_mask': torch.from_numpy(full_mask).float().unsqueeze(0),
                'full_image_mid': torch.from_numpy(full_image_mid).float(),
                'full_shape': (h, w),
                'case_id': sample['case_id'],
                'slice_idx': sample['slice_idx'],
                'is_positive': sample['is_positive']
            }


def msd_val_collate_fn(batch):
    return {
        'images_4patch': torch.stack([item['images_4patch'] for item in batch]),
        'positions': [item['positions'] for item in batch],
        'full_mask': [item['full_mask'] for item in batch],
        'full_image_mid': [item['full_image_mid'] for item in batch],
        'full_shape': (torch.tensor([item['full_shape'][0] for item in batch]),
                       torch.tensor([item['full_shape'][1] for item in batch])),
        'case_id': [item['case_id'] for item in batch],
        'slice_idx': [item['slice_idx'] for item in batch],
        'is_positive': [item['is_positive'] for item in batch]
    }
