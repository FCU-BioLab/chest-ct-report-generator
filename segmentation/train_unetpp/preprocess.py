#!/usr/bin/env python3
"""
UNet++ 肺結節分割訓練 - 預處理模組
===================================

專為 UNet++ 設計的 CT 影像預處理，特點：
1. Spacing Resample（統一到 1mm 各向同性）
2. HU Windowing（肺窗設定 [-1000, 200]）
3. Lung ROI Crop（肺野粗分割和裁切）
4. Lungmask 肺野外填黑（非肺區域設為 -1000 HU → 歸一化後為 0）
5. **4-Patch 分割**（將切片分成 2×2 的 224×224 patches）
6. 只保存有病灶的切片（減少儲存空間）
7. 預處理結果快取

與 MedSAM2 的差異：
- UNet++ 使用 4-Patch 策略，輸出 224×224 patches
- MedSAM2 直接使用完整 512×512 切片

輸出格式（切片模式）：
    cache/lndb_slices/
    └── LNDb-XXXX/
        ├── meta.json
        └── slice_XXXX.npz  (image, mask, lung_mask, slice_idx)

輸出格式（4-Patch 模式）：
    cache/lndb_patches/
    └── LNDb-XXXX/
        ├── meta.json
        └── slice_XXXX_patch_X.npz  (image, mask, lung_mask, patch_pos)
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import json
import random

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm


logger = logging.getLogger(__name__)

# =============================================================================
# 預設路徑配置
# =============================================================================

# LNDb 資料集根目錄
LNDB_DATA_DIR = Path(r"E:\lung_ct_lesion_dataset\LNDb")

# Lungmask 3D lung mask 目錄
LUNGMASK_DIR = LNDB_DATA_DIR / "lung_masks"

# 預設輸出目錄（4-Patch 模式）
DEFAULT_PATCH_OUTPUT_DIR = Path(r"C:\GitHub\chest-ct-report-generator\segmentation\cache\lndb_patches")


# =============================================================================
# 4-Patch 工具函數
# =============================================================================

def get_lung_bbox_2d(lung_mask: np.ndarray, margin: int = 0) -> Tuple[int, int, int, int]:
    """
    取得 2D lung mask 的 bounding box
    
    Args:
        lung_mask: (H, W) binary mask
        margin: 邊緣擴展像素
        
    Returns:
        (y_min, y_max, x_min, x_max)
    """
    lung_y, lung_x = np.where(lung_mask > 0)
    
    if len(lung_y) == 0:
        h, w = lung_mask.shape
        return 0, h, 0, w
    
    y_min = max(0, lung_y.min() - margin)
    y_max = min(lung_mask.shape[0], lung_y.max() + 1 + margin)
    x_min = max(0, lung_x.min() - margin)
    x_max = min(lung_mask.shape[1], lung_x.max() + 1 + margin)
    
    return y_min, y_max, x_min, x_max


def compute_4patch_positions(lung_mask: np.ndarray, patch_size: int = 224) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    計算 4-patch 的位置（基於 lung bbox 分成 2×2 quadrants）
    
    策略：
    1. 取得 lung mask 的 bounding box
    2. 將 bbox 分成 2×2 的 quadrants
    3. 每個 patch 覆蓋對應的 1/4 lung mask
    
    Args:
        lung_mask: (H, W) binary mask
        patch_size: patch 大小 (預設 224 for UNet++)
        
    Returns:
        list of ((y1, x1), (y2, x2)) - 每個 patch 的左上角和右下角座標
    """
    h, w = lung_mask.shape
    half_patch = patch_size // 2
    
    # 取得 lung bbox
    y_min, y_max, x_min, x_max = get_lung_bbox_2d(lung_mask, margin=0)
    
    # 計算 lung bbox 的中點
    y_mid = (y_min + y_max) // 2
    x_mid = (x_min + x_max) // 2
    
    # 4 個 quadrants: Top-Left, Top-Right, Bottom-Left, Bottom-Right
    quadrants = [
        (y_min, y_mid, x_min, x_mid),
        (y_min, y_mid, x_mid, x_max),
        (y_mid, y_max, x_min, x_mid),
        (y_mid, y_max, x_mid, x_max),
    ]
    
    patches = []
    
    for (q_y1, q_y2, q_x1, q_x2) in quadrants:
        # 計算 quadrant 中心
        q_cy = (q_y1 + q_y2) // 2
        q_cx = (q_x1 + q_x2) // 2
        
        # Patch 左上角（以 quadrant 中心為 patch 中心）
        p_y1 = q_cy - half_patch
        p_x1 = q_cx - half_patch
        
        # 調整確保 quadrant 完全在 patch 內
        if q_y1 < p_y1:
            p_y1 = q_y1
        if q_y2 > p_y1 + patch_size:
            p_y1 = q_y2 - patch_size
        if q_x1 < p_x1:
            p_x1 = q_x1
        if q_x2 > p_x1 + patch_size:
            p_x1 = q_x2 - patch_size
        
        p_y2 = p_y1 + patch_size
        p_x2 = p_x1 + patch_size
        
        patches.append(((p_y1, p_x1), (p_y2, p_x2)))
    
    return patches


def extract_patch(
    image: np.ndarray,
    mask: np.ndarray,
    lung_mask: np.ndarray,
    patch_pos: Tuple[Tuple[int, int], Tuple[int, int]],
    patch_size: int = 224
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    提取 patch，超出邊界用零填充，lung mask 外設為零
    
    Args:
        image: (H, W) 原始影像
        mask: (H, W) 分割 mask
        lung_mask: (H, W) lung mask
        patch_pos: ((y1, x1), (y2, x2)) patch 座標
        patch_size: patch 大小
        
    Returns:
        patch_image: 提取的影像 patch (patch_size, patch_size)
        patch_mask: 分割 mask patch
        patch_lung: lung mask patch
    """
    (y1, x1), (y2, x2) = patch_pos
    h, w = lung_mask.shape
    
    patch_image = np.zeros((patch_size, patch_size), dtype=image.dtype)
    patch_mask = np.zeros((patch_size, patch_size), dtype=mask.dtype)
    patch_lung = np.zeros((patch_size, patch_size), dtype=lung_mask.dtype)
    
    # 計算有效區域
    src_y1 = max(0, y1)
    src_y2 = min(h, y2)
    src_x1 = max(0, x1)
    src_x2 = min(w, x2)
    
    dst_y1 = max(0, -y1)
    dst_x1 = max(0, -x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    
    # 複製有效區域
    patch_image[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    patch_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    patch_lung[dst_y1:dst_y2, dst_x1:dst_x2] = lung_mask[src_y1:src_y2, src_x1:src_x2]
    
    # 將 lung mask 外的區域設為零
    patch_image[patch_lung == 0] = 0
    patch_mask[patch_lung == 0] = 0
    
    return patch_image, patch_mask, patch_lung


def sample_negative_patch_centers(
    lung_mask_2d: np.ndarray,
    lesion_mask_2d: np.ndarray,
    dilate_mm: float = 15.0,
    spacing_xy: float = 1.0,
    num_hard: int = 1,
    num_random: int = 1,
    patch_size: int = 224
) -> List[Tuple[int, int]]:
    """
    從 lung mask 範圍內採樣負樣本 patch 的中心點
    
    策略：
    1. Hard negative: 在肺內但遠離病灶的區域（離病灶至少 dilate_mm）
    2. Random negative: 在肺內隨機位置（但不含病灶）
    
    Args:
        lung_mask_2d: 2D lung mask (H, W)
        lesion_mask_2d: 2D lesion mask (H, W)
        dilate_mm: 病灶 dilation 距離 (mm)，預設 15mm
        spacing_xy: XY 方向的 spacing (mm)，用於計算實際距離
        num_hard: 要取的 hard negative 數量
        num_random: 要取的 random negative 數量
        patch_size: patch 大小，用於確保 patch 不會超出邊界太多
        
    Returns:
        List of (cy, cx) center coordinates for negative patches
    """
    h, w = lung_mask_2d.shape
    half_patch = patch_size // 2
    
    centers = []
    
    # 計算 dilation 結構元素（圓形）
    dilate_pixels = int(dilate_mm / spacing_xy)
    if dilate_pixels < 1:
        dilate_pixels = 1
    
    # 建立 dilation 結構元素（圓形）
    y_struct, x_struct = np.ogrid[-dilate_pixels:dilate_pixels+1, -dilate_pixels:dilate_pixels+1]
    struct_elem = (y_struct**2 + x_struct**2) <= dilate_pixels**2
    
    # 擴張病灶區域
    if np.any(lesion_mask_2d > 0):
        lesion_dilated = ndimage.binary_dilation(lesion_mask_2d > 0, structure=struct_elem)
    else:
        lesion_dilated = np.zeros_like(lesion_mask_2d, dtype=bool)
    
    # Hard negative 區域：肺內 && 不在擴張後的病灶區
    hard_neg_region = (lung_mask_2d > 0) & ~lesion_dilated
    
    # Random negative 區域：肺內 && 不直接含病灶（但可以接近）
    random_neg_region = (lung_mask_2d > 0) & (lesion_mask_2d == 0)
    
    # 取得候選點
    hard_candidates = np.argwhere(hard_neg_region)
    random_candidates = np.argwhere(random_neg_region)
    
    # 採樣 hard negative
    if len(hard_candidates) > 0 and num_hard > 0:
        # 隨機取 num_hard 個點
        indices = random.sample(range(len(hard_candidates)), min(num_hard, len(hard_candidates)))
        for idx in indices:
            cy, cx = hard_candidates[idx]
            centers.append((int(cy), int(cx)))
    
    # 採樣 random negative
    if len(random_candidates) > 0 and num_random > 0:
        # 隨機取 num_random 個點，排除已選的位置
        used_points = set(centers)
        available = [(int(p[0]), int(p[1])) for p in random_candidates 
                     if (int(p[0]), int(p[1])) not in used_points]
        if available:
            indices = random.sample(range(len(available)), min(num_random, len(available)))
            for idx in indices:
                centers.append(available[idx])
    
    return centers


def center_to_patch_pos(
    cy: int, cx: int, patch_size: int, img_shape: Tuple[int, int]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    將中心點轉換為 patch position 格式
    
    Args:
        cy, cx: 中心點座標
        patch_size: patch 大小
        img_shape: (H, W) 影像尺寸
        
    Returns:
        ((y1, x1), (y2, x2)) patch 位置
    """
    half = patch_size // 2
    h, w = img_shape
    
    y1 = cy - half
    x1 = cx - half
    y2 = y1 + patch_size
    x2 = x1 + patch_size
    
    return ((y1, x1), (y2, x2))


# =============================================================================
# UNet++ 預處理器
# =============================================================================

class UNetPPPreprocessor:
    """
    UNet++ 專用 CT 影像預處理器
    
    特點：
    1. 支援完整切片輸出和 4-Patch 分割
    2. 針對肺結節優化的 HU 窗設定
    3. 多醫師標註 → Binary Union
    """
    
    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        hu_window_center: float = -400,
        hu_window_width: float = 1200,
        lung_margin: int = 10,
        patch_size: int = 224  # UNet++ patch 大小
    ):
        """
        初始化預處理器
        
        Args:
            target_spacing: 目標 spacing (x, y, z) mm，預設 1mm 各向同性
            hu_window_center: HU 窗位（肺窗 -400）
            hu_window_width: HU 窗寬（肺窗 1200，即 [-1000, 200]）
            lung_margin: 肺野 bounding box 邊緣擴展（像素）
            patch_size: Patch 大小（預設 224 for UNet++）
        """
        self.target_spacing = np.array(target_spacing)
        self.hu_window_center = hu_window_center
        self.hu_window_width = hu_window_width
        self.lung_margin = lung_margin
        self.patch_size = patch_size
        
        # HU 範圍
        self.hu_min = hu_window_center - hu_window_width / 2  # -1000
        self.hu_max = hu_window_center + hu_window_width / 2  # 200
    
    # -------------------------------------------------------------------------
    # 檔案載入
    # -------------------------------------------------------------------------
    
    def load_mhd(self, mhd_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        載入 MHD/RAW 格式的 CT 影像
        
        Returns:
            volume: CT 影像 (Z, Y, X)，HU 值
            spacing: 體素間距 (x, y, z) mm
            origin: 原點座標
        """
        image = sitk.ReadImage(str(mhd_path))
        volume = sitk.GetArrayFromImage(image)  # (Z, Y, X)
        spacing = np.array(image.GetSpacing())  # (x, y, z)
        origin = np.array(image.GetOrigin())
        
        return volume.astype(np.float32), spacing, origin
    
    def load_lungmask(self, patient_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        載入 lungmask 生成的 3D lung mask
        
        Args:
            patient_id: 病人 ID，如 "LNDb-0001"
            
        Returns:
            lung_mask: Binary lung mask (Z, Y, X)，若不存在則為 None
            spacing: Lung mask 的 spacing (x, y, z)
        """
        lung_path = LUNGMASK_DIR / f"{patient_id}_lung.mhd"
        
        if not lung_path.exists():
            logger.warning(f"⚠️ Lungmask 不存在: {lung_path}")
            return None, None
        
        lung_sitk = sitk.ReadImage(str(lung_path))
        lung_array = sitk.GetArrayFromImage(lung_sitk)  # (Z, Y, X)
        lung_spacing = np.array(lung_sitk.GetSpacing())  # (x, y, z)
        
        # Label: 1=右肺, 2=左肺 → binary
        lung_mask = (lung_array > 0).astype(np.float32)
        
        return lung_mask, lung_spacing
    
    # -------------------------------------------------------------------------
    # 影像處理
    # -------------------------------------------------------------------------
    
    def resample_volume(
        self,
        volume: np.ndarray,
        current_spacing: np.ndarray,
        target_spacing: Optional[np.ndarray] = None,
        order: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        將影像重新採樣到目標 spacing
        
        Args:
            volume: 輸入影像 (Z, Y, X)
            current_spacing: 當前 spacing (x, y, z)
            target_spacing: 目標 spacing，None 使用預設
            order: 插值階數（0=最近鄰, 1=線性, 3=三次）
            
        Returns:
            resampled: 重新採樣後的影像
            new_spacing: 新的 spacing
        """
        if target_spacing is None:
            target_spacing = self.target_spacing
        
        # 計算縮放因子
        # volume 是 (Z, Y, X) 但 spacing 是 (x, y, z)
        scale_factors = current_spacing / target_spacing
        # 轉換為 (z, y, x) 順序對應 volume 的維度
        scale_factors_zyx = np.array([scale_factors[2], scale_factors[1], scale_factors[0]])
        
        # 計算新形狀
        new_shape = np.round(np.array(volume.shape) * scale_factors_zyx).astype(int)
        
        # 使用 scipy.ndimage.zoom 進行重新採樣
        zoom_factors = new_shape / np.array(volume.shape)
        resampled = ndimage.zoom(volume, zoom_factors, order=order)
        
        return resampled, target_spacing
    
    def apply_hu_window(self, volume: np.ndarray) -> np.ndarray:
        """
        應用 HU 窗設定並歸一化到 [0, 1]
        
        Args:
            volume: CT 影像（HU 值）
            
        Returns:
            歸一化後的影像 [0, 1]
        """
        # Clipping to window range
        volume = np.clip(volume, self.hu_min, self.hu_max)
        
        # Normalize to [0, 1]
        volume = (volume - self.hu_min) / (self.hu_max - self.hu_min)
        
        return volume.astype(np.float32)
    
    def get_lung_bbox_3d(self, lung_mask: np.ndarray) -> Tuple[slice, slice, slice]:
        """
        獲取 3D 肺野的 bounding box
        
        Args:
            lung_mask: 二值肺野遮罩 (Z, Y, X)
            
        Returns:
            (z_slice, y_slice, x_slice) 用於裁切的 slice 物件
        """
        # 找到非零區域
        z_indices, y_indices, x_indices = np.where(lung_mask > 0)
        
        if len(z_indices) == 0:
            # 如果沒有肺野，返回完整範圍
            return (
                slice(0, lung_mask.shape[0]),
                slice(0, lung_mask.shape[1]),
                slice(0, lung_mask.shape[2])
            )
        
        margin = self.lung_margin
        z_min = max(0, z_indices.min() - margin)
        z_max = min(lung_mask.shape[0], z_indices.max() + margin + 1)
        y_min = max(0, y_indices.min() - margin)
        y_max = min(lung_mask.shape[1], y_indices.max() + margin + 1)
        x_min = max(0, x_indices.min() - margin)
        x_max = min(lung_mask.shape[2], x_indices.max() + margin + 1)
        
        return (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))
    
    def apply_lung_mask(self, volume: np.ndarray, lung_mask: np.ndarray) -> np.ndarray:
        """
        肺野外區域填黑
        
        Args:
            volume: CT 影像（HU 值）
            lung_mask: 二值肺野遮罩
            
        Returns:
            遮罩後的影像（肺野外為 hu_min）
        """
        return np.where(lung_mask > 0, volume, self.hu_min)
    
    def create_binary_union_mask(self, masks: List[np.ndarray]) -> np.ndarray:
        """
        創建 Binary Union 遮罩（任一醫師標註即為結節）
        
        符合 CSEA-Net 論文做法，提高 Recall
        
        Args:
            masks: 多位醫師的遮罩列表
            
        Returns:
            binary_mask: Union 後的二值遮罩
        """
        if len(masks) == 0:
            raise ValueError("遮罩列表為空")
        
        binary_mask = np.zeros_like(masks[0], dtype=np.float32)
        for m in masks:
            binary_mask = np.logical_or(binary_mask > 0, m > 0).astype(np.float32)
        
        return binary_mask
    
    # -------------------------------------------------------------------------
    # 主要預處理流程
    # -------------------------------------------------------------------------
    
    def preprocess_patient(
        self,
        ct_path: str,
        mask_paths: List[str],
        patient_id: str
    ) -> Optional[Dict]:
        """
        預處理單一病人的 CT 資料
        
        Args:
            ct_path: CT 影像路徑 (.mhd)
            mask_paths: 遮罩路徑列表（多位醫師標註）
            patient_id: 病人 ID，如 "LNDb-0001"
            
        Returns:
            預處理結果字典，若失敗則為 None
        """
        # 1. 載入 CT 影像
        volume, spacing, origin = self.load_mhd(ct_path)
        original_shape = volume.shape
        
        # 2. 載入 lungmask 
        # TEMPORARILY DISABLED LUNG MASK
        # lung_mask, lung_spacing = self.load_lungmask(patient_id)
        # if lung_mask is None:
        #     return None  # 無 lungmask，跳過此病人
        
        # 替代方案：不使用 lung mask，或者是全 1 mask
        lung_mask = np.ones(volume.shape, dtype=np.float32)
        lung_spacing = spacing
        
        # 3. Resample CT 和 lung mask 到目標 spacing
        volume_resampled, new_spacing = self.resample_volume(volume, spacing, order=1)
        # 注意：lung mask 使用自己的 spacing (lung_spacing) 而非 CT spacing
        lung_mask_resampled, _ = self.resample_volume(
            lung_mask, lung_spacing, order=0  # 最近鄰插值保持二值
        )
        lung_mask_resampled = (lung_mask_resampled > 0.5).astype(np.float32)
        
        # 4. 獲取肺野 bounding box 並裁切
        bbox = self.get_lung_bbox_3d(lung_mask_resampled)
        volume_cropped = volume_resampled[bbox]
        lung_mask_cropped = lung_mask_resampled[bbox]
        
        # 5. 肺野外填黑（使用 HU 最小值）
        # volume_masked = self.apply_lung_mask(volume_cropped, lung_mask_cropped)
        # TEMPORARILY DISABLED: 不做 masking
        volume_masked = volume_cropped
        
        # 6. HU windowing → [0, 1]
        volume_normalized = self.apply_hu_window(volume_masked)
        
        # 7. 處理分割遮罩（多醫師標註）
        masks_processed = []
        for mask_path in mask_paths:
            if Path(mask_path).exists():
                mask, _, _ = self.load_mhd(mask_path)
                mask_resampled, _ = self.resample_volume(mask.astype(np.float32), spacing, order=0)
                mask_cropped = mask_resampled[bbox]
                masks_processed.append((mask_cropped > 0.5).astype(np.float32))
        
        # 8. 建立 binary union mask
        if len(masks_processed) > 0:
            binary_mask = self.create_binary_union_mask(masks_processed)
        else:
            binary_mask = np.zeros_like(volume_normalized, dtype=np.float32)
        
        return {
            'volume': volume_normalized,      # 歸一化影像 [0, 1]
            'mask': binary_mask,              # 二值分割遮罩
            'lung_mask': lung_mask_cropped,   # 肺野遮罩
            'spacing': new_spacing,
            'bbox': (
                (bbox[0].start, bbox[0].stop),
                (bbox[1].start, bbox[1].stop),
                (bbox[2].start, bbox[2].stop)
            ),
            'original_shape': original_shape,
            'original_spacing': spacing.tolist()
        }
    
    
    def save_patches(
        self,
        result: Dict,
        output_dir: Path,
        patient_id: str,
        save_all: bool = False,
        in_channels: int = 5,  # 幾個 channel (z-2, z-1, z, z+1, z+2)
        num_hard_neg: int = 1,  # 每個 slice 額外存幾個 hard negative patch
        num_random_neg: int = 1,  # 每個 slice 額外存幾個 random negative patch
        dilate_mm: float = 15.0  # hard negative 與病灶的最小距離 (mm)
    ) -> Tuple[int, int, int]:
        """
        保存為 4-Patch 格式（UNet++ 專用，支援多 channel 2.5D）
        
        同時保存正樣本 patch（含病灶）和負樣本 patch（hard negative + random negative）
        以確保訓練時有足夠的負樣本避免 false positive。
        
        Args:
            result: preprocess_patient 的輸出
            output_dir: 輸出目錄
            patient_id: 病人 ID
            save_all: 是否保存所有 patch（False = 只保存有病灶的 + negative）
            in_channels: 幾個 channel (預設 5, 對應 z-2, z-1, z, z+1, z+2)
            num_hard_neg: 每個 positive slice 額外存幾個 hard negative patch
            num_random_neg: 每個 positive slice 額外存幾個 random negative patch  
            dilate_mm: hard negative 與病灶的最小距離 (mm)
            
        Returns:
            (保存的切片數, 保存的正樣本 patch 數, 保存的負樣本 patch 數)
        """
        patient_dir = output_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        volume = result['volume']
        mask = result['mask']
        lung_mask = result['lung_mask']
        
        num_slices = int(volume.shape[0])
        
        # 找出有病灶的切片
        positive_slices = [int(z) for z in range(num_slices) if np.any(mask[z] > 0)]
        
        # 決定要處理的切片
        slices_to_save = list(range(num_slices)) if save_all else positive_slices
        
        saved_slices = 0
        saved_positive_patches = 0
        saved_negative_patches = 0
        all_patch_info = []
        
        # 取得 spacing（用於計算 dilation 距離）
        spacing = result.get('spacing', np.array([1.0, 1.0, 1.0]))
        spacing_xy = spacing[0]  # 假設 xy spacing 相同
        
        for z in slices_to_save:
            # === 2.5D: 取得 z-2, z-1, z, z+1, z+2 五個切片 ===
            half = in_channels // 2
            z_indices = [min(max(0, z + offset), num_slices - 1) for offset in range(-half, half + 1)]
            slice_imgs = [volume[zi] for zi in z_indices]
            slice_mask = mask[z]
            slice_lung = lung_mask[z]

            # 計算 4-Patch 位置（基於當前切片的 lung mask）
            patch_positions = compute_4patch_positions(slice_lung, self.patch_size)

            slice_patches = []
            for patch_idx, patch_pos in enumerate(patch_positions):
                # 提取每個 channel 的 patch
                patch_channels = []
                for img in slice_imgs:
                    patch_img, _, _ = extract_patch(
                        img, slice_mask, slice_lung, patch_pos, self.patch_size
                    )
                    patch_channels.append(patch_img)
                # mask/肺遮罩用中心切片
                _, patch_mask, patch_lung_extracted = extract_patch(
                    slice_imgs[half], slice_mask, slice_lung, patch_pos, self.patch_size
                )
                # 組合成 (C, H, W)
                patch_2_5d = np.stack(patch_channels, axis=0)

                # 判斷是否有病灶
                has_lesion = bool(np.any(patch_mask > 0))

                # 保存所有 4 個 patch (有病灶=Positive, 無病灶=Negative)
                patch_path = patient_dir / f"slice_{z:04d}_patch_{patch_idx}.npz"
                patch_type = 'positive' if has_lesion else 'negative'
                
                np.savez_compressed(
                    patch_path,
                    image=patch_2_5d.astype(np.float16),  # (C, H, W) 2.5D
                    mask=patch_mask.astype(np.float16),   # (H, W)
                    lung_mask=patch_lung_extracted.astype(np.bool_),
                    slice_idx=z,
                    patch_idx=patch_idx,
                    patch_pos=np.array(patch_pos),
                    is_2_5d=True,
                    z_range=z_indices,
                    patch_type=patch_type
                )
                
                if has_lesion:
                    saved_positive_patches += 1
                else:
                    saved_negative_patches += 1
                    
                slice_patches.append({
                    'patch_idx': patch_idx,
                    'has_lesion': has_lesion,
                    'patch_type': patch_type,
                    'patch_pos': [[int(patch_pos[0][0]), int(patch_pos[0][1])],
                                  [int(patch_pos[1][0]), int(patch_pos[1][1])]]
                })

            # === 額外採樣負樣本 patch ===
            # 只在有病灶的 slice 上採樣負樣本（確保是有意義的 negative）
            if z in positive_slices and (num_hard_neg > 0 or num_random_neg > 0):
                neg_centers = sample_negative_patch_centers(
                    lung_mask_2d=slice_lung,
                    lesion_mask_2d=slice_mask,
                    dilate_mm=dilate_mm,
                    spacing_xy=spacing_xy,
                    num_hard=num_hard_neg,
                    num_random=num_random_neg,
                    patch_size=self.patch_size
                )
                
                for neg_idx, (cy, cx) in enumerate(neg_centers):
                    neg_patch_pos = center_to_patch_pos(cy, cx, self.patch_size, slice_lung.shape)
                    
                    # 提取每個 channel 的 patch
                    neg_patch_channels = []
                    for img in slice_imgs:
                        neg_patch_img, _, _ = extract_patch(
                            img, slice_mask, slice_lung, neg_patch_pos, self.patch_size
                        )
                        neg_patch_channels.append(neg_patch_img)
                    _, neg_patch_mask, neg_patch_lung = extract_patch(
                        slice_imgs[half], slice_mask, slice_lung, neg_patch_pos, self.patch_size
                    )
                    neg_patch_2_5d = np.stack(neg_patch_channels, axis=0)
                    
                    # 確認這個 patch 沒有病灶（防止採樣到邊緣）
                    if np.any(neg_patch_mask > 0):
                        continue
                    
                    # 決定 patch type
                    patch_type = 'hard_negative' if neg_idx < num_hard_neg else 'random_negative'
                    neg_patch_idx = len(patch_positions) + neg_idx  # 索引從 4 開始
                    
                    neg_patch_path = patient_dir / f"slice_{z:04d}_patch_{neg_patch_idx}.npz"
                    np.savez_compressed(
                        neg_patch_path,
                        image=neg_patch_2_5d.astype(np.float16),
                        mask=neg_patch_mask.astype(np.float16),
                        lung_mask=neg_patch_lung.astype(np.bool_),
                        slice_idx=z,
                        patch_idx=neg_patch_idx,
                        patch_pos=np.array(neg_patch_pos),
                        is_2_5d=True,
                        z_range=z_indices,
                        patch_type=patch_type
                    )
                    saved_negative_patches += 1
                    slice_patches.append({
                        'patch_idx': neg_patch_idx,
                        'has_lesion': False,
                        'patch_type': patch_type,
                        'center': [int(cy), int(cx)],
                        'patch_pos': [[int(neg_patch_pos[0][0]), int(neg_patch_pos[0][1])],
                                      [int(neg_patch_pos[1][0]), int(neg_patch_pos[1][1])]]
                    })

            if slice_patches:
                all_patch_info.append({
                    'slice_idx': int(z),
                    'patches': slice_patches
                })
                saved_slices += 1
        
        # 保存元資料
        # 確保 bbox 是 Python int
        bbox = result['bbox']
        bbox_serializable = [
            [int(bbox[0][0]), int(bbox[0][1])],
            [int(bbox[1][0]), int(bbox[1][1])],
            [int(bbox[2][0]), int(bbox[2][1])]
        ]
        
        meta = {
            'patient_id': patient_id,
            'num_slices': num_slices,
            'saved_slices': saved_slices,
            'saved_positive_patches': saved_positive_patches,
            'saved_negative_patches': saved_negative_patches,
            'saved_patches': saved_positive_patches + saved_negative_patches,
            'positive_slices': positive_slices,
            'patch_info': all_patch_info,
            'spacing': result['spacing'].tolist() if hasattr(result['spacing'], 'tolist') else list(result['spacing']),
            'bbox': bbox_serializable,
            'original_shape': [int(x) for x in result['original_shape']],
            'original_spacing': result['original_spacing'],
            'preprocessing': {
                'target_spacing': self.target_spacing.tolist(),
                'hu_window': [float(self.hu_min), float(self.hu_max)],
                'lung_margin': int(self.lung_margin),
                'patch_size': int(self.patch_size),
                'mode': '4-patch',
                'is_2_5d': True,
                'num_hard_neg_per_slice': num_hard_neg,
                'num_random_neg_per_slice': num_random_neg,
                'dilate_mm': dilate_mm
            }
        }
        
        with open(patient_dir / "meta.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        return saved_slices, saved_positive_patches, saved_negative_patches
    
    def load_slice(self, slice_path: str) -> Dict:
        """載入單一切片"""
        data = np.load(slice_path, allow_pickle=True)
        return {
            'image': data['image'].astype(np.float32),
            'mask': data['mask'].astype(np.float32),
            'lung_mask': data['lung_mask'].astype(bool),
            'slice_idx': int(data['slice_idx'])
        }
    
    def load_patch(self, patch_path: str) -> Dict:
        """載入單一 patch"""
        data = np.load(patch_path, allow_pickle=True)
        return {
            'image': data['image'].astype(np.float32),
            'mask': data['mask'].astype(np.float32),
            'lung_mask': data['lung_mask'].astype(bool),
            'slice_idx': int(data['slice_idx']),
            'patch_idx': int(data['patch_idx']),
            'patch_pos': data['patch_pos']
        }


# =============================================================================
# 批量預處理函數
# =============================================================================

def preprocess_lndb_for_unetpp(
    data_dir: str = None,
    output_dir: str = None,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    patient_ids: Optional[List[str]] = None,
    save_all: bool = False,
    num_hard_neg: int = 1,
    num_random_neg: int = 1,
    dilate_mm: float = 15.0
):
    """
    批量預處理 LNDb 資料集（4-Patch 格式，UNet++ 專用）
    
    Args:
        data_dir: LNDb 資料集根目錄
        output_dir: 輸出目錄
        target_spacing: 目標 spacing (x, y, z) mm
        patient_ids: 要處理的病人 ID 列表（None = 全部）
        save_all: 是否保存所有 patch（False = 只保存有病灶的 + negative）
        num_hard_neg: 每個 positive slice 額外存幾個 hard negative patch
        num_random_neg: 每個 positive slice 額外存幾個 random negative patch
        dilate_mm: hard negative 與病灶的最小距離 (mm)
    """
    data_dir = Path(data_dir) if data_dir else LNDB_DATA_DIR
    output_dir = Path(output_dir) if output_dir else DEFAULT_PATCH_OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === 印出預處理設定 ===
    logger.info("=" * 60)
    logger.info("🔧 UNet++ 預處理設定 (4-Patch 模式)")
    logger.info("=" * 60)
    logger.info(f"  資料目錄: {data_dir}")
    logger.info(f"  輸出目錄: {output_dir}")
    logger.info(f"  目標 Spacing: {target_spacing}")
    logger.info(f"  保存模式: {'所有 patch' if save_all else '只保存有病灶的 + negative'}")
    logger.info(f"  [Negative Sampling]")
    logger.info(f"    num_hard_neg: {num_hard_neg} (每個 positive slice)")
    logger.info(f"    num_random_neg: {num_random_neg} (每個 positive slice)")
    logger.info(f"    dilate_mm: {dilate_mm} mm (hard neg 與病灶的最小距離)")
    logger.info("=" * 60)
    
    preprocessor = UNetPPPreprocessor(target_spacing=target_spacing)
    
    # 找到所有 CT 檔案
    ct_files = []
    for subfolder in ['data0', 'data1', 'data2', 'data3', 'data4', 'data5']:
        folder = data_dir / subfolder
        if folder.exists():
            ct_files.extend(folder.glob('*.mhd'))
    
    logger.info(f"📂 找到 {len(ct_files)} 個 CT 檔案")
    
    # 找到對應的遮罩目錄
    mask_dir = data_dir / 'mask' / 'masks'
    
    # 過濾只保留有 ≥3mm 結節的病人
    nodule_csv = data_dir / 'trainset_csv' / 'trainNodules_gt.csv'
    nodule_patients = None
    if nodule_csv.exists():
        import pandas as pd
        df = pd.read_csv(nodule_csv)
        # Nodule=1 表示是結節（≥3mm 有分割遮罩）
        nodule_patients = set(f"LNDb-{int(pid):04d}" for pid in df[df['Nodule'] == 1]['LNDbID'].unique())
        logger.info(f"📋 找到 {len(nodule_patients)} 個有 ≥3mm 結節的病人")
    
    # 統計
    stats = {
        'total': len(ct_files),
        'processed': 0,
        'skipped_no_nodule': 0,
        'skipped_no_lungmask': 0,
        'skipped_exists': 0,
        'total_slices': 0,
        'total_patches': 0,
        'total_positive_patches': 0,
        'total_negative_patches': 0,
        'errors': 0
    }
    
    for ct_path in tqdm(ct_files, desc="🔄 Preprocessing for UNet++ (4-Patch)"):
        patient_id = ct_path.stem  # e.g., LNDb-0001
        
        # 過濾
        if patient_ids and patient_id not in patient_ids:
            continue
        
        if nodule_patients is not None and patient_id not in nodule_patients:
            stats['skipped_no_nodule'] += 1
            continue
        
        # 檢查是否已處理
        patient_dir = output_dir / patient_id
        meta_path = patient_dir / "meta.json"
        if meta_path.exists():
            stats['skipped_exists'] += 1
            continue
        
        # 找遮罩檔案
        mask_paths = []
        for rad_id in range(1, 4):
            mask_path = mask_dir / f"{patient_id}_rad{rad_id}.mhd"
            if mask_path.exists():
                mask_paths.append(str(mask_path))
        
        try:
            # 預處理
            result = preprocessor.preprocess_patient(str(ct_path), mask_paths, patient_id)
            
            if result is None:
                stats['skipped_no_lungmask'] += 1
                continue
            
            # 保存為 4-Patch 格式
            num_slices, num_pos, num_neg = preprocessor.save_patches(
                result, output_dir, patient_id, save_all,
                num_hard_neg=num_hard_neg, num_random_neg=num_random_neg, dilate_mm=dilate_mm
            )
            stats['total_slices'] += num_slices
            stats['total_patches'] += num_pos + num_neg
            stats['total_positive_patches'] += num_pos
            stats['total_negative_patches'] += num_neg
            
            stats['processed'] += 1
            
        except Exception as e:
            logger.error(f"❌ 處理 {patient_id} 時出錯: {e}")
            stats['errors'] += 1
    
    # 輸出統計
    logger.info("=" * 60)
    logger.info("📊 UNet++ 預處理完成統計 (4-Patch 模式)")
    logger.info("=" * 60)
    logger.info(f"  總 CT 檔案: {stats['total']}")
    logger.info(f"  ✅ 成功處理: {stats['processed']} 個病人")
    logger.info(f"  📦 切片數: {stats['total_slices']}")
    logger.info(f"  🧩 Patch 總數: {stats['total_patches']}")
    logger.info(f"     ├─ 正樣本: {stats['total_positive_patches']}")
    logger.info(f"     └─ 負樣本: {stats['total_negative_patches']}")
    logger.info(f"  ⏭️  跳過（已存在）: {stats['skipped_exists']}")
    logger.info(f"  ⏭️  跳過（無 ≥3mm 結節）: {stats['skipped_no_nodule']}")
    logger.info(f"  ⏭️  跳過（無 lungmask）: {stats['skipped_no_lungmask']}")
    logger.info(f"  ❌ 錯誤: {stats['errors']}")
    logger.info("=" * 60)
    
    return stats


def verify_preprocessed_data(cache_dir: str = None):
    """
    驗證預處理資料的完整性（4-Patch 模式）
    
    Args:
        cache_dir: 快取目錄
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_PATCH_OUTPUT_DIR
    
    if not cache_dir.exists():
        logger.error(f"❌ 快取目錄不存在: {cache_dir}")
        return
    
    patient_dirs = sorted([d for d in cache_dir.iterdir() if d.is_dir()])
    
    logger.info(f"📂 驗證 {len(patient_dirs)} 個病人資料 (4-Patch 模式)...")
    
    stats = {
        'valid': 0,
        'invalid': 0,
        'total_files': 0,
        'issues': []
    }
    
    for patient_dir in tqdm(patient_dirs, desc="Verifying"):
        patient_id = patient_dir.name
        meta_path = patient_dir / "meta.json"
        
        # 檢查 meta.json
        if not meta_path.exists():
            stats['invalid'] += 1
            stats['issues'].append(f"{patient_id}: missing meta.json")
            continue
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # 檢查檔案數量
        expected = meta.get('saved_patches', 0)
        actual = len(list(patient_dir.glob('slice_*_patch_*.npz')))
        
        if expected != actual:
            stats['invalid'] += 1
            stats['issues'].append(f"{patient_id}: expected {expected} files, found {actual}")
            continue
        
        stats['valid'] += 1
        stats['total_files'] += actual
    
    logger.info("=" * 60)
    logger.info("📊 驗證結果")
    logger.info("=" * 60)
    logger.info(f"  ✅ 有效: {stats['valid']} 個病人, {stats['total_files']} 個檔案")
    logger.info(f"  ❌ 無效: {stats['invalid']} 個病人")
    
    if stats['issues']:
        logger.info("  問題列表:")
        for issue in stats['issues'][:10]:
            logger.info(f"    - {issue}")
        if len(stats['issues']) > 10:
            logger.info(f"    ... 還有 {len(stats['issues']) - 10} 個問題")
    
    logger.info("=" * 60)
    
    return stats


# =============================================================================
# 相容性：保留舊版本函數名稱
# =============================================================================

# 別名：保持向後相容
CTPreprocessor = UNetPPPreprocessor


# =============================================================================
# 主程式
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="UNet++ 專用 LNDb 資料集預處理（4-Patch 模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 預處理（預設：只保存有病灶的 + negative patches）
  python preprocess.py

  # 保存所有 patch
  python preprocess.py --save-all

  # 驗證預處理結果
  python preprocess.py --verify
        """
    )
    
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help=f"LNDb 資料集根目錄 (預設: {LNDB_DATA_DIR})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"輸出目錄 (預設: {DEFAULT_PATCH_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        help="目標 spacing (x, y, z) mm (預設: 1.0 1.0 1.0)"
    )
    parser.add_argument(
        "--save-all", action="store_true",
        help="保存所有 patch（預設只保存有病灶的 + negative）"
    )
    parser.add_argument(
        "--num-hard-neg", type=int, default=1,
        help="每個 positive slice 額外存幾個 hard negative patch (預設: 1)"
    )
    parser.add_argument(
        "--num-random-neg", type=int, default=1,
        help="每個 positive slice 額外存幾個 random negative patch (預設: 1)"
    )
    parser.add_argument(
        "--dilate-mm", type=float, default=15.0,
        help="hard negative 與病灶的最小距離 mm (預設: 15.0)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="驗證預處理資料的完整性"
    )
    
    args = parser.parse_args()
    
    # 設定 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.verify:
        verify_preprocessed_data(args.output_dir)
    else:
        preprocess_lndb_for_unetpp(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            target_spacing=tuple(args.spacing),
            save_all=args.save_all,
            num_hard_neg=args.num_hard_neg,
            num_random_neg=args.num_random_neg,
            dilate_mm=args.dilate_mm
        )
