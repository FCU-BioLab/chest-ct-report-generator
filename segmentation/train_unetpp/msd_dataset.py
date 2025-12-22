#!/usr/bin/env python3
"""
MSD Lung Tumours 預處理與資料集
================================

支援 Medical Segmentation Decathlon Task06 (Lung Tumours) 的預處理和訓練
- NIfTI 格式 (.nii.gz)
- 64 個訓練樣本 + 32 個測試樣本
- 肺部腫瘤分割標註

使用方式:
    # 預處理
    python msd_dataset.py --preprocess
    
    # 檢視資料
    python msd_dataset.py --check
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
from scipy import ndimage
from scipy.ndimage import zoom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ============= 資料集路徑 =============
MSD_LUNG_DIR = Path(r"E:\lung_ct_lesion_dataset\MSD Lung Tumours\Task06_Lung")
MSD_CACHE_DIR = Path(r"C:\GitHub\chest-ct-report-generator\segmentation\cache\msd_lung_slices")
MSD_LUNGMASK_DIR = Path(r"E:\lung_ct_lesion_dataset\MSD Lung Tumours\lung_masks")


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
        """
        載入 NIfTI 檔案
        
        Returns:
            volume: (Z, Y, X) 體積資料
            spacing: (X, Y, Z) 像素間距
        """
        nii = nib.load(str(nii_path))
        volume = nii.get_fdata()
        
        # 移除第四維度（如果有）
        if volume.ndim == 4:
            volume = volume[:, :, :, 0]
        
        # 轉置為 (Z, Y, X)
        volume = np.transpose(volume, (2, 1, 0))
        
        # 取得 spacing
        spacing = np.array(nii.header.get_zooms()[:3])  # (X, Y, Z)
        
        return volume.astype(np.float32), spacing
    
    def load_lungmask(self, case_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        載入預生成的 lungmask
        
        Args:
            case_id: 案例 ID，如 "lung_001"
        
        Returns:
            lung_mask: Binary lung mask (Z, Y, X)
            spacing: Lung mask 的 spacing (X, Y, Z)
        """
        lung_path = self.lungmask_dir / f"{case_id}_lung.nii.gz"
        
        if not lung_path.exists():
            logger.warning(f"Lungmask 不存在: {lung_path}，使用 threshold-based fallback")
            return None, None
        
        lung_nii = nib.load(str(lung_path))
        lung_array = lung_nii.get_fdata()
        
        # 處理可能的第四維度
        if lung_array.ndim == 4:
            lung_array = lung_array[:, :, :, 0]
        
        # 轉置為 (Z, Y, X)
        lung_array = np.transpose(lung_array, (2, 1, 0))
        
        # 取得 spacing
        spacing = np.array(lung_nii.header.get_zooms()[:3])  # (X, Y, Z)
        
        # Label 1=右肺, 2=左肺 → binary
        lung_mask = (lung_array > 0).astype(np.float32)
        
        return lung_mask, spacing
    
    def resample_volume(
        self,
        volume: np.ndarray,
        original_spacing: np.ndarray,
        is_mask: bool = False
    ) -> np.ndarray:
        """重新採樣到目標解析度"""
        # 計算縮放因子
        # original_spacing 是 (X, Y, Z)，volume 是 (Z, Y, X)
        zoom_factors = [
            original_spacing[2] / self.target_spacing[2],  # Z
            original_spacing[1] / self.target_spacing[1],  # Y
            original_spacing[0] / self.target_spacing[0],  # X
        ]
        
        order = 0 if is_mask else 3  # mask 用最近鄰，影像用三次插值
        resampled = zoom(volume, zoom_factors, order=order)
        
        return resampled
    
    def normalize_hu(self, volume: np.ndarray) -> np.ndarray:
        """HU 值正規化到 [0, 1]"""
        wl = self.hu_window_center
        ww = self.hu_window_width
        
        min_hu = wl - ww / 2
        max_hu = wl + ww / 2
        
        volume = np.clip(volume, min_hu, max_hu)
        volume = (volume - min_hu) / (max_hu - min_hu)
        
        return volume.astype(np.float32)
    
    def get_lung_bbox(self, volume: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        使用閾值方法估計肺部區域的邊界框
        
        簡化版本：使用 HU 閾值 (-1000 到 -200) 找肺部區域
        """
        # 假設 volume 已經是 HU 值（未正規化）
        # 肺部大約在 -1000 到 -200 HU
        lung_mask = (volume > -1000) & (volume < -200)
        
        # 形態學操作清理
        from scipy.ndimage import binary_closing, binary_opening
        lung_mask = binary_closing(lung_mask, iterations=5)
        lung_mask = binary_opening(lung_mask, iterations=5)
        
        # 找邊界框
        z_indices, y_indices, x_indices = np.where(lung_mask)
        
        if len(z_indices) == 0:
            # 如果找不到肺部，使用整個體積
            return ((0, volume.shape[0]), (0, volume.shape[1]), (0, volume.shape[2]))
        
        margin = 10  # 邊界
        z_min = max(0, z_indices.min() - margin)
        z_max = min(volume.shape[0], z_indices.max() + margin)
        y_min = max(0, y_indices.min() - margin)
        y_max = min(volume.shape[1], y_indices.max() + margin)
        x_min = max(0, x_indices.min() - margin)
        x_max = min(volume.shape[2], x_indices.max() + margin)
        
        return ((z_min, z_max), (y_min, y_max), (x_min, x_max))
    
    def preprocess_case(
        self,
        case_id: str,
        image_path: Path,
        label_path: Optional[Path] = None
    ) -> Dict:
        """
        預處理單一案例
        
        Returns:
            dict with: volume, mask, lung_mask, spacing, bbox, positive_slices
        """
        # 載入影像
        volume, spacing = self.load_nifti(image_path)
        
        # 載入標註（如果有）
        if label_path and label_path.exists():
            mask, _ = self.load_nifti(label_path)
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.zeros_like(volume)
        
        # 載入預生成的 lungmask
        lung_mask, lung_spacing = self.load_lungmask(case_id)
        
        if lung_mask is None:
            # Lungmask 不存在，使用 threshold-based fallback
            logger.warning(f"{case_id}: 使用 threshold-based lung mask fallback")
            lung_mask = (volume > -1000) & (volume < -200)
            lung_mask = lung_mask.astype(np.float32)
        
        # 重新採樣
        volume_resampled = self.resample_volume(volume, spacing, is_mask=False)
        mask_resampled = self.resample_volume(mask, spacing, is_mask=True)
        lung_mask_resampled = self.resample_volume(lung_mask, spacing, is_mask=True)
        lung_mask_resampled = (lung_mask_resampled > 0.5).astype(np.float32)
        
        # 取得肺部 bbox（使用 lungmask 的 bbox）
        bbox = self.get_lung_bbox_from_mask(lung_mask_resampled)
        
        # 裁切
        (z0, z1), (y0, y1), (x0, x1) = bbox
        volume_cropped = volume_resampled[z0:z1, y0:y1, x0:x1]
        mask_cropped = mask_resampled[z0:z1, y0:y1, x0:x1]
        lung_mask_cropped = lung_mask_resampled[z0:z1, y0:y1, x0:x1]
        
        # HU 正規化
        volume_normalized = self.normalize_hu(volume_cropped)
        
        # 找出有腫瘤的切片
        positive_slices = []
        for z in range(mask_cropped.shape[0]):
            if mask_cropped[z].sum() > 0:
                positive_slices.append(z)
        
        return {
            'volume': volume_normalized,
            'mask': mask_cropped,
            'lung_mask': lung_mask_cropped,
            'spacing': self.target_spacing,
            'bbox': bbox,
            'positive_slices': positive_slices,
            'original_spacing': spacing
        }
    
    def get_lung_bbox_from_mask(
        self,
        lung_mask: np.ndarray,
        margin: int = 10
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        從 lung mask 取得邊界框
        
        Args:
            lung_mask: Binary lung mask (Z, Y, X)
            margin: 邊界擴展像素數
        
        Returns:
            ((z0, z1), (y0, y1), (x0, x1))
        """
        z_indices, y_indices, x_indices = np.where(lung_mask > 0)
        
        if len(z_indices) == 0:
            # 找不到肺部，使用整個體積
            return ((0, lung_mask.shape[0]), (0, lung_mask.shape[1]), (0, lung_mask.shape[2]))
        
        z_min = max(0, z_indices.min() - margin)
        z_max = min(lung_mask.shape[0], z_indices.max() + margin)
        y_min = max(0, y_indices.min() - margin)
        y_max = min(lung_mask.shape[1], y_indices.max() + margin)
        x_min = max(0, x_indices.min() - margin)
        x_max = min(lung_mask.shape[2], x_indices.max() + margin)
        
        return ((z_min, z_max), (y_min, y_max), (x_min, x_max))
    
    def save_slices(
        self,
        case_id: str,
        data: Dict
    ) -> int:
        """將預處理後的資料儲存為切片快取"""
        case_dir = self.cache_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        
        volume = data['volume']
        mask = data['mask']
        lung_mask = data['lung_mask']  # 使用預生成的 lungmask
        
        num_slices = volume.shape[0]
        
        for z in range(num_slices):
            slice_data = {
                'image': volume[z].astype(np.float16),
                'mask': mask[z].astype(np.float16),
                'lung_mask': lung_mask[z].astype(np.bool_)  # 同 LNDb 格式
            }
            np.savez_compressed(case_dir / f"slice_{z:04d}.npz", **slice_data)
        
        # 儲存 meta
        meta = {
            'case_id': case_id,
            'num_slices': num_slices,
            'positive_slices': data['positive_slices'],
            'spacing': list(data['spacing']),
            'shape': list(volume.shape)
        }
        with open(case_dir / "meta.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        return num_slices


def get_msd_lung_cases(data_dir: Path = MSD_LUNG_DIR) -> List[Tuple[str, Path, Optional[Path]]]:
    """
    取得所有 MSD Lung 案例
    
    Returns:
        list of (case_id, image_path, label_path or None)
    """
    cases = []
    
    # 訓練集
    images_tr = data_dir / "imagesTr"
    labels_tr = data_dir / "labelsTr"
    
    if images_tr.exists():
        for nii_file in sorted(images_tr.glob("lung_*.nii.gz")):
            if nii_file.name.startswith("._"):
                continue
            case_id = nii_file.stem.replace(".nii", "")
            label_path = labels_tr / f"{case_id}.nii.gz"
            if not label_path.exists():
                label_path = None
            cases.append((case_id, nii_file, label_path))
    
    # 測試集（通常無標註）
    images_ts = data_dir / "imagesTs"
    if images_ts.exists():
        for nii_file in sorted(images_ts.glob("lung_*.nii.gz")):
            if nii_file.name.startswith("._"):
                continue
            case_id = nii_file.stem.replace(".nii", "") + "_test"
            cases.append((case_id, nii_file, None))
    
    return cases


def get_msd_train_val_split(
    cases: List[Tuple[str, Path, Optional[Path]]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    分割訓練/驗證/測試集（只使用有標註的案例）
    
    Returns:
        train_ids, val_ids, test_ids
    """
    # 只使用有標註的案例
    labeled_cases = [c[0] for c in cases if c[2] is not None]
    
    random.seed(seed)
    random.shuffle(labeled_cases)
    
    n = len(labeled_cases)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    
    test_ids = labeled_cases[:test_size]
    val_ids = labeled_cases[test_size:test_size + val_size]
    train_ids = labeled_cases[test_size + val_size:]
    
    return train_ids, val_ids, test_ids


def preprocess_msd_lung(data_dir: Path = MSD_LUNG_DIR, cache_dir: Path = MSD_CACHE_DIR):
    """預處理整個 MSD Lung 資料集"""
    logger.info(f"開始預處理 MSD Lung Tumours 資料集")
    logger.info(f"資料目錄: {data_dir}")
    logger.info(f"快取目錄: {cache_dir}")
    
    preprocessor = MSDLungPreprocessor(cache_dir=cache_dir)
    cases = get_msd_lung_cases(data_dir)
    
    logger.info(f"找到 {len(cases)} 個案例")
    
    total_slices = 0
    total_positive = 0
    
    for case_id, image_path, label_path in tqdm(cases, desc="Preprocessing"):
        if label_path is None:
            logger.info(f"跳過無標註案例: {case_id}")
            continue
        
        try:
            data = preprocessor.preprocess_case(case_id, image_path, label_path)
            num_slices = preprocessor.save_slices(case_id, data)
            
            total_slices += num_slices
            total_positive += len(data['positive_slices'])
            
            logger.debug(f"{case_id}: {num_slices} slices, {len(data['positive_slices'])} positive")
        
        except Exception as e:
            logger.error(f"處理 {case_id} 失敗: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"預處理完成: 共 {total_slices} 切片, {total_positive} 正樣本切片")


class MSDLungSliceDataset(Dataset):
    """MSD Lung Tumours 切片資料集"""
    
    def __init__(
        self,
        case_ids: List[str],
        cache_dir: Path = MSD_CACHE_DIR,
        mode: str = "train",
        patch_size: int = 224,
        transform: Optional[A.Compose] = None
    ):
        self.case_ids = case_ids
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        self.patch_size = patch_size
        
        # 資料增強
        if transform is not None:
            self.transform = transform
        elif mode == "train":
            self.transform = self._get_train_transform()
        else:
            self.transform = self._get_val_transform()
        
        # 建立樣本索引
        self.samples = []
        self._build_sample_index()
        
        # 訓練模式下 oversampling
        if mode == "train":
            self._oversample_positives()
    
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
    
    def _build_sample_index(self):
        """建立樣本索引"""
        self.samples = []
        
        for case_id in self.case_ids:
            case_dir = self.cache_dir / case_id
            meta_path = case_dir / "meta.json"
            
            if not meta_path.exists():
                logger.warning(f"快取不存在: {case_id}")
                continue
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            positive_slices = set(meta['positive_slices'])
            
            for z in range(meta['num_slices']):
                self.samples.append({
                    'case_id': case_id,
                    'slice_idx': z,
                    'is_positive': z in positive_slices,
                    'slice_path': str(case_dir / f"slice_{z:04d}.npz"),
                    'num_slices': meta['num_slices']
                })
        
        logger.info(f"建立 {len(self.samples)} 個樣本索引 ({self.mode})")
    
    def _oversample_positives(self, target_ratio: float = 0.5):
        """正樣本 oversampling"""
        positives = [s for s in self.samples if s['is_positive']]
        negatives = [s for s in self.samples if not s['is_positive']]
        
        if len(positives) == 0:
            logger.warning("沒有正樣本！")
            return
        
        # 計算需要的正樣本數量
        target_positive = int(len(negatives) * target_ratio / (1 - target_ratio))
        
        if len(positives) < target_positive:
            factor = target_positive // len(positives) + 1
            positives = (positives * factor)[:target_positive]
        
        self.samples = positives + negatives
        random.shuffle(self.samples)
        
        logger.info(f"Oversampling 後: {len(positives)} 正, {len(negatives)} 負, 總計 {len(self.samples)}")
    
    def _get_4patch_centers(self, lung_mask: np.ndarray) -> List[Tuple[int, int]]:
        """計算 4-patch centers"""
        h, w = lung_mask.shape
        half = self.patch_size // 2
        
        lung_y, lung_x = np.where(lung_mask > 0)
        if len(lung_y) == 0:
            cy, cx = h // 2, w // 2
            return [(cy, cx)] * 4
        
        y0, y1 = lung_y.min(), lung_y.max()
        x0, x1 = lung_x.min(), lung_x.max()
        
        cy1 = int(y0 + (y1 - y0) / 3)
        cy2 = int(y0 + 2 * (y1 - y0) / 3)
        cx1 = int(x0 + (x1 - x0) / 3)
        cx2 = int(x0 + 2 * (x1 - x0) / 3)
        
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
        case_id = sample['case_id']
        slice_idx = sample['slice_idx']
        num_slices = sample['num_slices']
        
        # 2.5D: 讀取 z-1, z, z+1
        case_dir = self.cache_dir / case_id
        
        z_prev = max(0, slice_idx - 1)
        z_curr = slice_idx
        z_next = min(num_slices - 1, slice_idx + 1)
        
        slices_2_5d = []
        for z in [z_prev, z_curr, z_next]:
            data = np.load(case_dir / f"slice_{z:04d}.npz")
            slices_2_5d.append(data['image'].astype(np.float32))
        
        # 讀取中心切片的 mask 和 lung_mask
        center_data = np.load(sample['slice_path'])
        mask = center_data['mask'].astype(np.float32)
        lung_mask = center_data['lung_mask']
        
        # 組裝 2.5D: (3, H, W)
        image_2_5d = np.stack(slices_2_5d, axis=0)
        
        patch_size = self.patch_size
        h, w = mask.shape
        half = patch_size // 2
        
        if self.mode == "train":
            # Train: 腫瘤為中心或隨機
            if sample['is_positive']:
                tumor_y, tumor_x = np.where(mask > 0.5)
                if len(tumor_y) > 0:
                    center_idx = len(tumor_y) // 2
                    center_y, center_x = tumor_y[center_idx], tumor_x[center_idx]
                    jitter = int(patch_size * 0.3)
                    center_y += random.randint(-jitter, jitter)
                    center_x += random.randint(-jitter, jitter)
                else:
                    center_y, center_x = h // 2, w // 2
            else:
                lung_y, lung_x = np.where(lung_mask > 0)
                if len(lung_y) > 0:
                    rand_idx = random.randint(0, len(lung_y) - 1)
                    center_y, center_x = lung_y[rand_idx], lung_x[rand_idx]
                else:
                    center_y, center_x = h // 2, w // 2
            
            center_y = max(half, min(h - half, center_y))
            center_x = max(half, min(w - half, center_x))
            
            y1, y2 = center_y - half, center_y + half
            x1, x2 = center_x - half, center_x + half
            
            image_patch = image_2_5d[:, y1:y2, x1:x2]
            mask_patch = mask[y1:y2, x1:x2]
            
            # Padding
            if image_patch.shape[1] < patch_size or image_patch.shape[2] < patch_size:
                padded_img = np.zeros((3, patch_size, patch_size), dtype=np.float32)
                padded_mask = np.zeros((patch_size, patch_size), dtype=np.float32)
                padded_img[:, :image_patch.shape[1], :image_patch.shape[2]] = image_patch
                padded_mask[:mask_patch.shape[0], :mask_patch.shape[1]] = mask_patch
                image_patch, mask_patch = padded_img, padded_mask
            
            # Albumentations
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
                'case_id': case_id,
                'slice_idx': slice_idx,
                'is_positive': sample['is_positive']
            }
        
        else:
            # Val/Test: 4-patch
            centers = self._get_4patch_centers(lung_mask)
            
            image_patches = []
            positions = []
            
            for cy, cx in centers:
                y1, y2 = cy - half, cy + half
                x1, x2 = cx - half, cx + half
                
                image_patch = image_2_5d[:, max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                
                # Padding
                if image_patch.shape[1] < patch_size or image_patch.shape[2] < patch_size:
                    padded = np.zeros((3, patch_size, patch_size), dtype=np.float32)
                    padded[:, :image_patch.shape[1], :image_patch.shape[2]] = image_patch
                    image_patch = padded
                
                image_patches.append(torch.from_numpy(image_patch).float())
                positions.append((y1, x1))
            
            images_4patch = torch.stack(image_patches, dim=0)
            full_mask = torch.from_numpy(mask).float().unsqueeze(0)
            full_image_mid = torch.from_numpy(image_2_5d[1]).float()  # 用於視覺化
            
            return {
                'images_4patch': images_4patch,
                'positions': positions,
                'full_mask': full_mask,
                'full_image_mid': full_image_mid,
                'full_shape': (h, w),
                'case_id': case_id,
                'slice_idx': slice_idx,
                'is_positive': sample['is_positive']
            }


def msd_val_collate_fn(batch):
    """Val/Test 用的 collate function"""
    images = torch.stack([item['images_4patch'] for item in batch], dim=0)
    masks = [item['full_mask'] for item in batch]
    full_images = [item['full_image_mid'] for item in batch]
    positions = [item['positions'] for item in batch]
    full_shapes = ([item['full_shape'][0] for item in batch],
                   [item['full_shape'][1] for item in batch])
    case_ids = [item['case_id'] for item in batch]
    slice_idxs = [item['slice_idx'] for item in batch]
    is_positives = [item['is_positive'] for item in batch]
    
    return {
        'images_4patch': images,
        'positions': positions,
        'full_mask': masks,
        'full_image_mid': full_images,
        'full_shape': (torch.tensor(full_shapes[0]), torch.tensor(full_shapes[1])),
        'case_id': case_ids,
        'slice_idx': slice_idxs,
        'is_positive': is_positives
    }


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='MSD Lung Tumours 資料集處理')
    parser.add_argument('--preprocess', action='store_true', help='執行預處理')
    parser.add_argument('--check', action='store_true', help='檢查快取資料')
    parser.add_argument('--data_dir', type=str, default=str(MSD_LUNG_DIR), help='資料目錄')
    parser.add_argument('--cache_dir', type=str, default=str(MSD_CACHE_DIR), help='快取目錄')
    
    args = parser.parse_args()
    
    if args.preprocess:
        preprocess_msd_lung(Path(args.data_dir), Path(args.cache_dir))
    
    elif args.check:
        cache_dir = Path(args.cache_dir)
        cases = list(cache_dir.iterdir())
        print(f"找到 {len(cases)} 個快取案例")
        
        for case_dir in cases[:3]:
            if case_dir.is_dir():
                meta_path = case_dir / "meta.json"
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    print(f"{meta['case_id']}: {meta['num_slices']} slices, {len(meta['positive_slices'])} positive")
    
    else:
        print("請指定 --preprocess 或 --check")
