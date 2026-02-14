import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage import measure, morphology
from tqdm import tqdm
from scipy import ndimage

# GPU acceleration
import torch
import torch.nn.functional as F
HAS_GPU = torch.cuda.is_available()

from .segmentation import generate_lung_mask, compute_lung_bbox

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def _isolate_finding_mask(mask_array: np.ndarray, center_z: int, center_y: int, center_x: int, spacing: np.ndarray) -> tuple:
    """
    從合併的 mask 中分離出最靠近指定中心的結節 mask。
    
    Args:
        mask_array: (D, H, W) 合併 mask（可能包含多個結節）
        center_z, center_y, center_x: 結節中心的 voxel 座標
        spacing: (x, y, z) spacing
    
    Returns:
        (isolated_mask, volume_mm3, diameter_mm)
        - isolated_mask: 只包含目標結節的 mask
        - volume_mm3: 結節體積 (mm³)
        - diameter_mm: 等效球直徑 (mm)
    """
    binary = (mask_array > 0).astype(np.uint8)
    labels = measure.label(binary)
    
    if labels.max() == 0:
        # No mask at all
        return binary, 0.0, 0.0
    
    # Check if center voxel is inside a labeled region
    if (0 <= center_z < labels.shape[0] and 
        0 <= center_y < labels.shape[1] and 
        0 <= center_x < labels.shape[2]):
        center_label = labels[center_z, center_y, center_x]
    else:
        center_label = 0
    
    if center_label > 0:
        # Direct hit — use this component
        isolated = (labels == center_label).astype(np.uint8)
    else:
        # Find the closest component by centroid distance
        regions = measure.regionprops(labels)
        if not regions:
            return binary, 0.0, 0.0
        
        center_arr = np.array([center_z, center_y, center_x], dtype=float)
        best_region = min(regions, key=lambda r: np.linalg.norm(np.array(r.centroid) - center_arr))
        isolated = (labels == best_region.label).astype(np.uint8)
    
    voxel_count = int(isolated.sum())
    volume_mm3 = voxel_count * float(np.prod(spacing))
    diameter_mm = 2 * np.power(3 * volume_mm3 / (4 * np.pi), 1/3) if volume_mm3 > 0 else 0.0
    
    return isolated, volume_mm3, diameter_mm


def _process_single_finding(
    ct_normalized: np.ndarray,
    mask_array: np.ndarray,
    ct_shape: tuple,
    spacing: np.ndarray,
    origin: np.ndarray,
    center_x: int, center_y: int, center_z: int,
    cx_world: float, cy_world: float, cz_world: float,
    lung_crop_bbox,
    image_size: int,
    full_volume: bool,
    context_slices: int,
    min_depth: int,
    min_nodule_diameter: float,
    resize_fn=None,
) -> Optional[dict]:
    """
    處理單一 finding：從 CT 與 mask 中擷取、裁切、resize、計算 bbox。
    
    Args:
        ct_normalized: (D, H, W) 經過 windowing 的 CT
        mask_array: (D, H, W) 合併 mask
        ct_shape: 原始 CT 陣列形狀
        spacing, origin: 影像空間參數 (x, y, z)
        center_x, center_y, center_z: 結節中心 voxel 座標
        cx_world, cy_world, cz_world: 結節中心世界座標 (mm)
        lung_crop_bbox: (min_x, min_y, max_x, max_y) 或 None
        image_size: 目標 resize 尺寸
        full_volume: 是否使用完整 volume
        context_slices: 中心前後各取的切片數
        min_depth: 最小深度
        min_nodule_diameter: 最小結節直徑 (mm)
        resize_fn: 可選的 resize 函式 (volume, is_mask) -> (resized, padding_info)。
                   若為 None 則使用 CPU scipy.ndimage.zoom。

    Returns:
        dict 包含 .npz 所需的所有欄位，或 None（被過濾）。
        額外回傳 'status' 欄位：'converted', 'filtered_small', 'filtered_edge'。
    """
    # --- Z 範圍 ---
    if full_volume:
        z_start, z_end = 0, ct_shape[0]
    else:
        z_start = max(0, center_z - context_slices)
        z_end = min(ct_shape[0], center_z + context_slices + 1)
        if (z_end - z_start) < min_depth:
            return {'status': 'filtered_edge'}

    # --- 擷取 ---
    video_frames = ct_normalized[z_start:z_end]
    raw_masks = mask_array[z_start:z_end]

    # --- Mask 分離 ---
    local_cz = center_z - z_start
    video_masks, volume_mm3, diameter_mm = _isolate_finding_mask(
        raw_masks, local_cz, center_y, center_x, spacing
    )

    if diameter_mm < min_nodule_diameter:
        return {'status': 'filtered_small'}

    center_idx = local_cz

    # --- 肺部裁切 ---
    crop_origin = origin.copy()
    if lung_crop_bbox is not None:
        min_x, min_y, max_x, max_y = lung_crop_bbox
        video_frames = video_frames[:, min_y:max_y, min_x:max_x]
        video_masks = video_masks[:, min_y:max_y, min_x:max_x]
        crop_origin[0] += min_x * spacing[0]
        crop_origin[1] += min_y * spacing[1]
        crop_origin[2] += z_start * spacing[2]

    # --- Pad-to-square + Resize ---
    if resize_fn is not None:
        video_frames_resized, padding_info = resize_fn(video_frames, False)
        video_masks_resized, _ = resize_fn(video_masks, True)
    else:
        # CPU 路徑 (worker 用)
        video_frames_resized, padding_info = _resize_volume_cpu(
            video_frames, image_size, is_mask=False
        )
        video_masks_resized, _ = _resize_volume_cpu(
            video_masks, image_size, is_mask=True
        )

    # --- Bbox 計算 ---
    bbox = _compute_bbox(video_masks_resized, center_idx, image_size)

    return {
        'status': 'converted',
        'frames': video_frames_resized,
        'masks': video_masks_resized,
        'center_idx': center_idx,
        'slice_indices': list(range(z_start, z_end)),
        'diameter_mm': diameter_mm,
        'volume_mm3': volume_mm3,
        'spacing': spacing,
        'origin': crop_origin,
        'lesion_center_csv': [cx_world, cy_world, cz_world],
        'original_shape': ct_shape,
        'bbox': bbox,
        'padding_info': padding_info,
        'lung_crop_bbox': lung_crop_bbox if lung_crop_bbox is not None else np.array([0, 0, ct_shape[2], ct_shape[1]]),
        # 以下由呼叫端自行補充：patient_id, lesion_id, agreement
        # 以及未 resize 的 video_frames / video_masks（供 NIfTI 匯出）
        'video_frames_pre_resize': video_frames,
        'video_masks_pre_resize': video_masks,
    }


def _resize_volume_cpu(volume: np.ndarray, image_size: int, is_mask: bool = False):
    """CPU 版 resize：先 pad-to-square 再 zoom（供 worker 使用）。"""
    current_h, current_w = volume.shape[1], volume.shape[2]
    max_dim = max(current_h, current_w)
    pad_h = max_dim - current_h
    pad_w = max_dim - current_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2

    padding_info = {
        'pre_pad_h': current_h,
        'pre_pad_w': current_w,
        'pad_h': pad_h,
        'pad_w': pad_w,
        'pad_top': pad_top,
        'pad_left': pad_left,
    }

    if pad_h > 0 or pad_w > 0:
        padded = np.zeros((volume.shape[0], max_dim, max_dim), dtype=volume.dtype)
        padded[:, pad_top:pad_top+current_h, pad_left:pad_left+current_w] = volume
        volume = padded

    if max_dim != image_size:
        scale = image_size / max_dim
        zoom_factors = (1.0, scale, scale)
        if is_mask:
            resized = ndimage.zoom(volume, zoom_factors, order=0, mode='nearest')
            resized = (resized > 0.5).astype(np.uint8)
        else:
            resized = ndimage.zoom(volume, zoom_factors, order=1, mode='constant', cval=0)
    else:
        resized = volume

    return resized, padding_info


def _compute_bbox(masks_3d: np.ndarray, center_idx: int, image_size: int) -> np.ndarray:
    """從 3D mask 中計算 2D bbox [x1, y1, x2, y2]。"""
    if center_idx < masks_3d.shape[0] and masks_3d[center_idx].max() > 0:
        mask_2d = masks_3d[center_idx]
    else:
        max_area, best_idx = 0, 0
        for i in range(masks_3d.shape[0]):
            area = np.sum(masks_3d[i])
            if area > max_area:
                max_area, best_idx = area, i
        mask_2d = masks_3d[best_idx] if max_area > 0 else None

    if mask_2d is not None and np.any(mask_2d):
        rows = np.any(mask_2d, axis=1)
        cols = np.any(mask_2d, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return np.array([cmin, rmin, cmax, rmax], dtype=np.float32)
    else:
        cx_b, cy_b = image_size // 2, image_size // 2
        return np.array([cx_b-10, cy_b-10, cx_b+10, cy_b+10], dtype=np.float32)

def _lndb_patient_worker(args: dict) -> dict:
    """
    Module-level worker function for multiprocessing.
    Processes one LNDb patient entirely on CPU.
    Returns a result dict with stats.
    """
    pid = args['patient_id']
    try:
        ct_image = sitk.ReadImage(args['ct_path'])
        ct_array = sitk.GetArrayFromImage(ct_image)
        mask_image = sitk.ReadImage(args['mask_path'])
        mask_array = sitk.GetArrayFromImage(mask_image)
        
        origin = np.array(ct_image.GetOrigin())
        spacing = np.array(ct_image.GetSpacing())
        
        # Windowing
        wc, ww = args['window_center'], args['window_width']
        img_min = wc - ww // 2
        img_max = wc + ww // 2
        ct_normalized = np.clip(ct_array, img_min, img_max)
        ct_normalized = ((ct_normalized - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        
        # Pre-compute lung crop ONCE per patient (use RAW HU, not windowed)
        lung_crop_bbox = None
        if args['crop_lungs']:
            # Normalize raw HU to [0,1]: air=-1024→0, soft tissue=400→1
            vol_norm = (ct_array.astype(np.float32) - (-1024)) / (400 - (-1024))
            vol_norm = np.clip(vol_norm, 0, 1)
            # Force CPU segmentation in worker processes (no CUDA)
            from .segmentation import _generate_lung_mask_cpu, compute_lung_bbox
            lung_mask = _generate_lung_mask_cpu(vol_norm, threshold=0.45, dilate_mm=5.0)
            lung_crop_bbox = compute_lung_bbox(lung_mask, margin=10)
            del lung_mask, vol_norm
        
        nodules_df = args['nodules_df']
        findings = nodules_df.groupby('FindingID')
        
        result = {
            'patient_id': pid,
            'success': True,
            'converted': 0,
            'filtered_small': 0,
            'filtered_edge': 0,
            'files': [],
            'error': None
        }
        
        image_size = args['image_size']
        output_dir = Path(args['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for finding_id, group in findings:
            nodule_row = group.iloc[0]
            if 'AgrLevel' in nodule_row:
                agreement_level = nodule_row['AgrLevel']
            else:
                agreement_level = len(group)
            
            if agreement_level < args['min_agreement']:
                continue
            
            cx, cy, cz = nodule_row['x'], nodule_row['y'], nodule_row['z']
            center_idx_float = (np.array([cx, cy, cz]) - origin) / spacing
            center_x, center_y, center_z = np.round(center_idx_float).astype(int)
            
            if not (0 <= center_z < ct_array.shape[0] and
                    0 <= center_y < ct_array.shape[1] and
                    0 <= center_x < ct_array.shape[2]):
                continue
            
            finding_result = _process_single_finding(
                ct_normalized=ct_normalized,
                mask_array=mask_array,
                ct_shape=ct_array.shape,
                spacing=spacing,
                origin=origin,
                center_x=center_x, center_y=center_y, center_z=center_z,
                cx_world=cx, cy_world=cy, cz_world=cz,
                lung_crop_bbox=lung_crop_bbox,
                image_size=image_size,
                full_volume=args['full_volume'],
                context_slices=args['context_slices'],
                min_depth=args['min_depth'],
                min_nodule_diameter=args['min_nodule_diameter'],
                resize_fn=None,  # CPU only
            )
            
            if finding_result is None:
                continue
            status = finding_result['status']
            if status == 'filtered_small':
                result['filtered_small'] += 1
                continue
            elif status == 'filtered_edge':
                result['filtered_edge'] += 1
                continue
            
            # Save
            output_path = output_dir / args['output_split'] / f"LNDb-{pid:04d}_lesion{finding_id:02d}.npz"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.savez_compressed(
                output_path,
                frames=finding_result['frames'],
                masks=finding_result['masks'],
                center_idx=finding_result['center_idx'],
                slice_indices=finding_result['slice_indices'],
                patient_id=f"LNDb-{pid:04d}",
                lesion_id=finding_id,
                diameter_mm=finding_result['diameter_mm'],
                volume_mm3=finding_result['volume_mm3'],
                spacing=finding_result['spacing'],
                origin=finding_result['origin'],
                lesion_center_csv=finding_result['lesion_center_csv'],
                original_shape=finding_result['original_shape'],
                bbox=finding_result['bbox'],
                agreement=agreement_level,
                padding_info=finding_result['padding_info'],
                lung_crop_bbox=finding_result['lung_crop_bbox'],
            )
            
            result['files'].append(str(output_path))
            result['converted'] += 1
        
        return result
    except Exception as e:
        return {
            'patient_id': pid,
            'success': False,
            'converted': 0,
            'filtered_small': 0,
            'filtered_edge': 0,
            'files': [],
            'error': str(e)
        }


class VolumePreprocessor:
    """
    Handles preprocessing of CT volumes into model-ready .npz format.
    Supports LNDb and MSD Lung Tumor datasets.
    """
    
    def __init__(
        self,
        output_dir: str,
        context_slices: int = 32,
        min_depth: int = 1,
        max_depth: int = 70,
        min_nodule_diameter: float = 2.0,
        image_size: int = 256,
        full_volume: bool = False,
        window_center: float = -600,
        window_width: float = 1500,
        min_agreement: int = 1,
        crop_lungs: bool = False,
        export_nifti: int = 0
    ):
        self.output_dir = Path(output_dir)
        self.context_slices = context_slices
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_nodule_diameter = min_nodule_diameter
        self.image_size = image_size
        self.full_volume = full_volume
        self.window_center = window_center
        self.window_width = window_width
        self.min_agreement = min_agreement
        self.crop_lungs = crop_lungs
        self.export_nifti = export_nifti
        
        self.nifti_dir = self.output_dir / "debug_nifti"
        if self.export_nifti > 0:
            self.nifti_dir.mkdir(parents=True, exist_ok=True)

        self.exported_count = 0
        self.generated_files = []
        self.stats = {
            'total_patients': 0,
            'total_lesions': 0,
            'converted_lesions': 0,
            'filtered_small': 0,
            'filtered_edge': 0,
            'errors': 0
        }
        
        logger.info(f"📁 Volume Preprocessor initialized")
        logger.info(f"  - Output dir: {self.output_dir}")
        logger.info(f"  - Context: ±{context_slices} (Depth {2*context_slices + 1})")
        logger.info(f"  - Min Nodule: {min_nodule_diameter}mm")
        logger.info(f"  - Min Agreement: >= {min_agreement} Radiologists")
        logger.info(f"  - Lung Cropping: {crop_lungs}")
        logger.info(f"  - Export NIfTI Debug: {export_nifti > 0} (count: {export_nifti})")
        if HAS_GPU:
            logger.info(f"  - GPU Acceleration: 🚀 PyTorch {torch.__version__} (CUDA {torch.version.cuda})")
        else:
            logger.info(f"  - GPU Acceleration: ❌ CPU only")

    def _apply_windowing(self, image: np.ndarray) -> np.ndarray:
        """Apply windowing and normalize to [0, 255]"""
        img_min = self.window_center - self.window_width // 2
        img_max = self.window_center + self.window_width // 2
        image = np.clip(image, img_min, img_max)
        image = (image - img_min) / (img_max - img_min)
        image = (image * 255).astype(np.uint8)
        return image

    def _resize_volume(self, volume: np.ndarray, is_mask: bool = False) -> Tuple[np.ndarray, dict]:
        """Resize volume (D, H, W) to (D, image_size, image_size).
        Pads to square first to preserve aspect ratio, then resizes.
        Returns (resized_volume, padding_info).
        """
        current_h, current_w = volume.shape[1], volume.shape[2]
        
        # Pad to square (use longer dimension)
        max_dim = max(current_h, current_w)
        pad_h = max_dim - current_h  # total padding on H axis
        pad_w = max_dim - current_w  # total padding on W axis
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        
        padding_info = {
            'pre_pad_h': current_h,
            'pre_pad_w': current_w,
            'pad_h': pad_h,
            'pad_w': pad_w,
            'pad_top': pad_top,
            'pad_left': pad_left,
        }
        
        if pad_h > 0 or pad_w > 0:
            padded = np.zeros((volume.shape[0], max_dim, max_dim), dtype=volume.dtype)
            padded[:, pad_top:pad_top+current_h, pad_left:pad_left+current_w] = volume
            volume = padded
        
        if max_dim == self.image_size:
            return volume, padding_info
        
        if HAS_GPU:
            vol_t = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
            mode = 'nearest' if is_mask else 'trilinear'
            vol_t = F.interpolate(vol_t, size=(volume.shape[0], self.image_size, self.image_size), mode=mode)
            resized = vol_t[0, 0].cpu().numpy()
            if is_mask:
                resized = (resized > 0.5).astype(np.uint8)
        else:
            scale = self.image_size / max_dim
            zoom_factors = (1.0, scale, scale)
            if is_mask:
                resized = ndimage.zoom(volume, zoom_factors, order=0, mode='nearest')
                resized = (resized > 0.5).astype(np.uint8)
            else:
                resized = ndimage.zoom(volume, zoom_factors, order=1, mode='constant', cval=0)
            
        return resized, padding_info

    def _compute_bbox_from_mask(self, mask_2d: np.ndarray) -> np.ndarray:
        """Compute bounding box [x1, y1, x2, y2] from a 2D mask"""
        rows = np.any(mask_2d, axis=1)
        cols = np.any(mask_2d, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # Fallback for empty mask
            h, w = mask_2d.shape
            cx, cy = w // 2, h // 2
            return np.array([cx-10, cy-10, cx+10, cy+10], dtype=np.float32)

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return np.array([cmin, rmin, cmax, rmax], dtype=np.float32)

    def convert_lndb(self, lndb_dir: str) -> Dict:
        """
        Convert LNDb dataset structure to preprocessed .npz files.
        Expects:
          lndb_dir/
            data/ (CT .mhd/.raw)
            mask/ (Mask .mhd/.raw)
            trainset_csv/trainNodules.csv
        """
        lndb_path = Path(lndb_dir)
        logger.info(f"🔄 Starting LNDb conversion: {lndb_path}")
        
        # Paths setup
        data_dir = lndb_path / 'data'
        mask_dir = lndb_path / 'mask' / 'masks' # LNDb typically puts masks here
        csv_path = lndb_path / 'trainset_csv' / 'trainNodules.csv'

        if not csv_path.exists():
             # Try alternate location
             csv_path = lndb_path / 'trainNodules.csv'
             
        if not csv_path.exists():
            logger.error(f"❌ Could not find trainNodules.csv in {lndb_path}")
            return self.stats

        # Load Nodule Intepretations
        df = pd.read_csv(csv_path)
        logger.info(f"📋 Loaded {len(df)} nodule annotations")
        
        if not data_dir.exists():
            # Check for data0, data1, ... 
            # Or just search recursively in root?
            # Recursive might be slow if many files, but LNDb structure is usually dataX.
            # Let's try globbing all data* folders first.
            data_subdirs = sorted(list(lndb_path.glob("data*")))
            if data_subdirs:
                logger.info(f"⚠️ 'data' dir not found, but found subdirs: {[d.name for d in data_subdirs]}")
                # We will search in all of them 
                
                ct_files = []
                for d in data_subdirs:
                    if d.is_dir():
                        ct_files.extend(list(d.glob("*.mhd")))
                        if not ct_files: # Try nii
                             ct_files.extend(list(d.glob("*.nii.gz")))
                
                ct_files = sorted(ct_files)
            else:
                logger.warning(f"⚠️ 'data' subdirectories not found, checking root {lndb_path}")
                ct_files = sorted(list(lndb_path.glob("*.mhd")))
                if not ct_files:
                    ct_files = sorted(list(lndb_path.glob("*.nii.gz")))
        else:
            # Standard 'data' dir exists
            ct_files = sorted(list(data_dir.glob("*.mhd")))
            if not ct_files:
                 ct_files = sorted(list(data_dir.glob("*.nii.gz")))
        
        if not ct_files:
             # Debug: list first few files in data_dir
             logger.warning(f"⚠️ No CT files found in {data_dir}")
             all_files = list(data_dir.glob("*"))[:10]
             logger.warning(f"   Contents sample: {[f.name for f in all_files]}")
             
        logger.info(f"📂 Found {len(ct_files)} CT scans in {data_dir}")
        
        # Map PatientID -> CT File
        # Flexible parsing: look for first integer in filename
        import re
        ct_map = {}
        for f in ct_files:
            try:
                match = re.search(r'(\d+)', f.stem)
                if match:
                    pid = int(match.group(1))
                    ct_map[pid] = f
            except Exception as e:
                logger.debug(f"Skipping CT file {f}: {e}")

        # Prepare Mask Mapping
        mask_files = sorted(list(mask_dir.glob("LNDb-*.mhd")))
        if not mask_files:
             mask_files = sorted(list(mask_dir.glob("LNDb-*.nii.gz")))
                
        # Mask map: patient_id -> finding_id -> path (if individual masks)
        # Or patient_id -> path (if merged)
        # LNDb masks are often per-radiologist.
        # Let's assume we use the 'agreed' masks or we just use available ones.
        # For this implementation, let's look for masks that match the patient.
        mask_map = {} 
        for f in mask_files:
             try:
                # LNDb-0001_finding1_rad1.mhd ? Or just LNDb-0001.mhd?
                # Usually LNDb provides a script to generate consensus masks.
                # Let's assume the user has generated masks named LNDb-XXXX.mhd 
                # or similar covering holes.
                # If the user is using raw LNDb, they might only have per-rad masks.
                # Let's map patient -> list of masks
                pid_str = f.stem.split('-')[1].split('_')[0]
                pid = int(pid_str)
                if pid not in mask_map:
                    mask_map[pid] = {}
                # Store by filename for now
                mask_map[pid][0] = f # Default key
             except Exception as e:
                 logger.debug(f"Skipping mask file {f}: {e}")

        logger.info(f"📂 Found {len(mask_map)} patients with masks")
        
        patients_with_nodules = df['LNDbID'].unique()
        
        processed_count = 0
        
        # Create output dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Only process patients we have data for
        valid_patients = [p for p in patients_with_nodules if p in ct_map and p in mask_map]
        
        logger.info(f"📦 Processing {len(valid_patients)} patients (No pre-split)")
        
        num_workers = getattr(self, 'num_workers', 1)
        
        if num_workers > 1:
            # --- Multiprocessing Mode ---
            logger.info(f"🔧 Using {num_workers} worker processes")
            
            # Prepare per-patient args (must be picklable)
            worker_args = []
            for pid in valid_patients:
                worker_args.append({
                    'patient_id': pid,
                    'ct_path': str(ct_map[pid]),
                    'mask_path': str(mask_map[pid][0]),
                    'nodules_df': df[df['LNDbID'] == pid],
                    'output_dir': str(self.output_dir),
                    'output_split': "",
                    'image_size': self.image_size,
                    'context_slices': self.context_slices,
                    'full_volume': self.full_volume,
                    'min_depth': self.min_depth,
                    'min_agreement': self.min_agreement,
                    'min_nodule_diameter': self.min_nodule_diameter,
                    'crop_lungs': self.crop_lungs,
                    'window_center': self.window_center,
                    'window_width': self.window_width,
                })
            
            with mp.Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(_lndb_patient_worker, worker_args),
                    total=len(worker_args),
                    desc="Converting LNDb"
                ))
            
            # Aggregate results
            for result in results:
                if result['success']:
                    processed_count += 1
                    self.stats['converted_lesions'] += result['converted']
                    self.stats['filtered_small'] += result['filtered_small']
                    self.stats['filtered_edge'] += result['filtered_edge']
                    self.generated_files.extend(result['files'])
                else:
                    logger.error(f"❌ Patient {result['patient_id']} failed: {result['error']}")
                    self.stats['errors'] += 1
        else:
            # --- Single Process Mode ---
            for pid in tqdm(valid_patients, desc="Converting LNDb"):
                try:
                    self._convert_lndb_patient(
                        patient_id=pid,
                        lndb_path=lndb_path,
                        ct_files=ct_map,
                        mask_files=mask_map,
                        nodules_df=df[df['LNDbID'] == pid],
                        output_split="",
                    )
                    processed_count += 1
                except Exception as e:
                    logger.error(f"❌ Patient {pid} failed: {e}")
                    self.stats['errors'] += 1
                
        self._log_stats()
        return self.stats

    def _convert_lndb_patient(
        self,
        patient_id: int,
        lndb_path: Path,
        ct_files: Dict[int, Path],
        mask_files: Dict[int, Dict[int, Path]],
        nodules_df: pd.DataFrame,
        output_split: str,
    ):
        """
        Process a single LNDb patient.
        Extracts nodule cubes based on CSV coordinates and saves as .npz
        """
        ct_path = ct_files[patient_id]
        # For LNDb, we might have multiple masks. 
        # If we have a merged mask, use it. If not, we might need to generate it.
        # Here we assume the mask file corresponds to the CT.
        mask_path = mask_files[patient_id][0] 
        
        # Load Volume
        ct_image = sitk.ReadImage(str(ct_path))
        ct_array = sitk.GetArrayFromImage(ct_image) # (Z, Y, X)
        
        # Load Mask
        mask_image = sitk.ReadImage(str(mask_path))
        mask_array = sitk.GetArrayFromImage(mask_image)
        
        origin = np.array(ct_image.GetOrigin()) # (x, y, z)
        spacing = np.array(ct_image.GetSpacing()) # (x, y, z)
        
        # Normalize Orientation (LNDb is usually okay, but good to check)
        # For simplicity, assuming consistent orientation or relying on data loader augmentation
        
        # Apply Windowing
        ct_normalized = self._apply_windowing(ct_array)
        
        # --- Pre-compute Lung Crop (ONCE per patient, use RAW HU) ---
        lung_crop_bbox = None
        if self.crop_lungs:
            # Normalize raw HU to [0,1]: air=-1024→0, soft tissue=400→1
            vol_norm = (ct_array.astype(np.float32) - (-1024)) / (400 - (-1024))
            vol_norm = np.clip(vol_norm, 0, 1)
            lung_mask = generate_lung_mask(vol_norm)
            lung_crop_bbox = compute_lung_bbox(lung_mask, margin=10)
            del lung_mask, vol_norm  # Free memory
        
        # Group by FindingID to handle multiple radiologists agreement
        # LNDb CSV: FindingID identifies the distinct nodule
        findings = nodules_df.groupby('FindingID')
        
        for finding_id, group in findings:
            nodule_row = group.iloc[0]
            if 'AgrLevel' in nodule_row:
                agreement_level = nodule_row['AgrLevel']
            else:
                agreement_level = len(group)

            if agreement_level < self.min_agreement:
                continue
            
            cx, cy, cz = nodule_row['x'], nodule_row['y'], nodule_row['z']
            center_idx_float = (np.array([cx, cy, cz]) - origin) / spacing
            center_x, center_y, center_z = np.round(center_idx_float).astype(int)
            
            if not (0 <= center_z < ct_array.shape[0] and 
                    0 <= center_y < ct_array.shape[1] and 
                    0 <= center_x < ct_array.shape[2]):
                logger.warning(f"  ⚠️ Nodule {finding_id} center out of bounds for Pat {patient_id}")
                continue

            # 使用 self._resize_volume 作為 resize 函式（支援 GPU）
            resize_fn = lambda vol, is_mask: self._resize_volume(vol, is_mask=is_mask)
            
            finding_result = _process_single_finding(
                ct_normalized=ct_normalized,
                mask_array=mask_array,
                ct_shape=ct_array.shape,
                spacing=spacing,
                origin=origin,
                center_x=center_x, center_y=center_y, center_z=center_z,
                cx_world=cx, cy_world=cy, cz_world=cz,
                lung_crop_bbox=lung_crop_bbox,
                image_size=self.image_size,
                full_volume=self.full_volume,
                context_slices=self.context_slices,
                min_depth=self.min_depth,
                min_nodule_diameter=self.min_nodule_diameter,
                resize_fn=resize_fn,
            )
            
            if finding_result is None:
                continue
            status = finding_result['status']
            if status == 'filtered_small':
                self.stats['filtered_small'] += 1
                continue
            elif status == 'filtered_edge':
                self.stats['filtered_edge'] += 1
                continue
            
            # --- Export NIfTI for Verification ---
            if self.export_nifti > 0 and self.exported_count < self.export_nifti:
                if np.random.rand() < 0.1: 
                    try:
                        crop_origin = finding_result['origin']
                        video_frames = finding_result['video_frames_pre_resize']
                        video_masks = finding_result['video_masks_pre_resize']
                        
                        out_img = sitk.GetImageFromArray(video_frames)
                        out_img.SetSpacing(spacing)
                        out_img.SetOrigin(crop_origin)
                        sitk.WriteImage(out_img, str(self.nifti_dir / f"LNDb-{patient_id}_{finding_id}_img.nii.gz"))
                        
                        out_msk = sitk.GetImageFromArray(video_masks)
                        out_msk.SetSpacing(spacing)
                        out_msk.SetOrigin(crop_origin)
                        sitk.WriteImage(out_msk, str(self.nifti_dir / f"LNDb-{patient_id}_{finding_id}_msk.nii.gz"))
                        
                        out_res = sitk.GetImageFromArray(finding_result['frames'])
                        max_dim = max(video_frames.shape[1], video_frames.shape[2])
                        scale = max_dim / self.image_size
                        new_spacing = [spacing[0] * scale, spacing[1] * scale, spacing[2]]
                        out_res.SetSpacing(new_spacing)
                        sitk.WriteImage(out_res, str(self.nifti_dir / f"LNDb-{patient_id}_{finding_id}_resized.nii.gz"))

                        self.exported_count += 1
                        logger.info(f"💾 Exported debug NIfTI: {self.nifti_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to export NIfTI: {e}")

            # 儲存 NPZ
            output_path = self.output_dir / output_split / f"LNDb-{patient_id:04d}_lesion{finding_id:02d}.npz"
            self.generated_files.append(str(output_path))
            
            np.savez_compressed(
                output_path,
                frames=finding_result['frames'],
                masks=finding_result['masks'],
                center_idx=finding_result['center_idx'],
                slice_indices=finding_result['slice_indices'],
                patient_id=f"LNDb-{patient_id:04d}",
                lesion_id=finding_id,
                diameter_mm=finding_result['diameter_mm'],
                volume_mm3=finding_result['volume_mm3'],
                spacing=finding_result['spacing'],
                origin=finding_result['origin'],
                lesion_center_csv=finding_result['lesion_center_csv'],
                original_shape=finding_result['original_shape'],
                bbox=finding_result['bbox'],
                agreement=agreement_level,
                padding_info=finding_result['padding_info'],
                lung_crop_bbox=finding_result['lung_crop_bbox'],
            )
            
            self.stats['converted_lesions'] += 1
    
    def convert_msd_lung(
        self,
        msd_dir: str,
    ) -> Dict:
        """
        轉換 MSD Lung Tumors 資料集
        """
        msd_path = Path(msd_dir)
        logger.info(f"🔄 開始轉換 MSD Lung 資料集: {msd_path}")
        
        images_dir = msd_path / 'imagesTr'
        labels_dir = msd_path / 'labelsTr'
        
        if not images_dir.exists() or not labels_dir.exists():
            logger.error(f"❌ 找不到 MSD 資料目錄")
            return self.stats
        
        # 取得所有案例
        case_files = sorted(images_dir.glob('lung_*.nii.gz'))
        case_ids = [f.stem.replace('.nii', '') for f in case_files]
        
        self.stats['total_patients'] = len(case_ids)
        
        # 建立輸出目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📦 處理 {len(case_ids)} 案例 (不進行預先分割)")
        
        for case_id in tqdm(case_ids, desc="Converting MSD"):
            try:
                self._convert_msd_case(
                    case_id=case_id,
                    images_dir=images_dir,
                    labels_dir=labels_dir,
                    output_split="", # No split subfolder
                )
            except Exception as e:
                logger.error(f"❌ 案例 {case_id} 轉換失敗: {e}")
                self.stats['errors'] += 1
                
        self._log_stats()
        return self.stats
    
    def _convert_msd_case(
        self,
        case_id: str,
        images_dir: Path,
        labels_dir: Path,
        output_split: str,
    ):
        """轉換單個 MSD 案例"""
        # 載入影像和標籤
        image_path = images_dir / f"{case_id}.nii.gz"
        label_path = labels_dir / f"{case_id}.nii.gz"
        
        if not image_path.exists() or not label_path.exists():
            return
        
        image = sitk.ReadImage(str(image_path))
        label = sitk.ReadImage(str(label_path))
        
        image_array = sitk.GetArrayFromImage(image)  # (Z, Y, X)
        label_array = sitk.GetArrayFromImage(label)
        spacing = np.array(image.GetSpacing()) # It returns (x, y, z) so we use it directly
        origin = np.array(image.GetOrigin())
        
        # CT 窗位正規化
        ct_normalized = self._apply_windowing(image_array)
        
        # 識別腫瘤區域
        labeled_mask = measure.label(label_array > 0)
        regions = measure.regionprops(labeled_mask)
        
        self.stats['total_lesions'] += len(regions)
        
        for region_idx, region in enumerate(regions):
            lesion_id = region_idx + 1
            
            # 計算大小
            volume_mm3 = region.area * np.prod(spacing)
            diameter_mm = 2 * np.power(3 * volume_mm3 / (4 * np.pi), 1/3)
            
            if diameter_mm < self.min_nodule_diameter:
                self.stats['filtered_small'] += 1
                continue
            
            # 視頻範圍
            z_min, y_min, x_min, z_max, y_max, x_max = region.bbox
            z_center = (z_min + z_max) // 2
            
            if self.full_volume:
                z_start = 0
                z_end = image_array.shape[0]
            else:
                z_start = max(0, z_center - self.context_slices)
                z_end = min(image_array.shape[0], z_center + self.context_slices + 1)
                
                depth = z_end - z_start
                if depth < self.min_depth:
                    self.stats['filtered_edge'] += 1
                    continue
            
            video_frames = ct_normalized[z_start:z_end]
            video_masks = (labeled_mask[z_start:z_end] == region.label).astype(np.uint8)
            center_idx = z_center - z_start
            
            # --- Lung Cropping ---
            if self.crop_lungs:
                # 1. Generate lung mask for this chunk
                vol_norm = video_frames.astype(np.float32) / 255.0
                lung_mask = generate_lung_mask(vol_norm)
                
                # 2. Compute bbox
                min_x, min_y, max_x, max_y = compute_lung_bbox(lung_mask, margin=10)
                
                # 3. Crop
                video_frames = video_frames[:, min_y:max_y, min_x:max_x]
                video_masks = video_masks[:, min_y:max_y, min_x:max_x]
                
                # Update origin
                origin[0] += min_x * spacing[0]
                origin[1] += min_y * spacing[1]
                origin[2] += z_start * spacing[2]

            # 調整大小
            video_frames_resized = self._resize_volume(video_frames)
            video_masks_resized = self._resize_volume(video_masks, is_mask=True)
            
            # --- Export NIfTI for Verification ---
            if self.export_nifti > 0 and self.exported_count < self.export_nifti:
                if np.random.rand() < 0.1:
                    try:
                        # sitk is already imported at module level
                        
                        # Save Resized (what the model sees)
                        out_res = sitk.GetImageFromArray(video_frames_resized)
                        new_spacing = [
                            spacing[0] * (video_frames.shape[2] / self.image_size),
                            spacing[1] * (video_frames.shape[1] / self.image_size),
                            spacing[2]
                        ]
                        out_res.SetSpacing(new_spacing)
                        sitk.WriteImage(out_res, str(self.nifti_dir / f"MSD-{case_id}_lesion{lesion_id}_resized.nii.gz"))

                        self.exported_count += 1
                        logger.info(f"💾 Exported debug NIfTI: {self.nifti_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to export NIfTI: {e}")
            
            # 計算 bbox
            if center_idx < video_masks_resized.shape[0] and video_masks_resized[center_idx].max() > 0:
                bbox = self._compute_bbox_from_mask(video_masks_resized[center_idx])
            else:
                 # 同上fallback
                max_area = 0
                best_idx = 0
                for i in range(video_masks_resized.shape[0]):
                    area = np.sum(video_masks_resized[i])
                    if area > max_area:
                        max_area = area
                        best_idx = i
                
                if max_area > 0:
                    bbox = self._compute_bbox_from_mask(video_masks_resized[best_idx])
                else:
                    h, w = self.image_size, self.image_size
                    cx, cy = w // 2, h // 2
                    bbox = np.array([cx-10, cy-10, cx+10, cy+10], dtype=np.float32)

            
            # 儲存
            output_path = self.output_dir / output_split / f"{case_id}_lesion{lesion_id:02d}.npz"
            
            np.savez_compressed(
                output_path,
                frames=video_frames_resized,
                masks=video_masks_resized,
                center_idx=center_idx,
                slice_indices=list(range(z_start, z_end)),
                patient_id=case_id,
                lesion_id=lesion_id,
                diameter_mm=diameter_mm,
                volume_mm3=volume_mm3,
                spacing=spacing,
                origin=origin,
                original_shape=image_array.shape,
                bbox=bbox,
                agreement=3 # Assuming MSD has high confidence
            )
            
            self.stats['converted_lesions'] += 1

    def _log_stats(self):
        logger.info(f"📊 轉換統計")
        logger.info(f"  - 總患者數: {self.stats['total_patients']}")
        logger.info(f"  - 總病灶數: {self.stats['total_lesions']}")
        logger.info(f"  - 成功轉換: {self.stats['converted_lesions']}")
        logger.info(f"  - 過濾（太小）: {self.stats['filtered_small']}")
        logger.info(f"  - 過濾（邊緣）: {self.stats['filtered_edge']}")
        logger.info(f"  - 錯誤: {self.stats['errors']}")
        
        # Save logs
        log_path = self.output_dir / "data.log"
        with open(log_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"📝 Data log saved to: {log_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert CT dataset to NPZ Volume format')
    parser.add_argument('--dataset', type=str, choices=['lndb', 'msd'], required=True, help='Dataset type')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
    # Optional parameters
    parser.add_argument('--context_slices', type=int, default=32, help='Slices before/after center')
    parser.add_argument('--min_nodule', type=float, default=0.0, help='Min nodule diameter (mm)')
    parser.add_argument('--full_volume', action='store_true', help='Export full volume instead of crops')
    parser.add_argument('--image_size', type=int, default=256, help='Target XY image size')
    parser.add_argument('--crop_lungs', action='store_true', help='Crop to lung mask')
    parser.add_argument('--export_nifti', type=int, default=0, help='Export N debug NIfTI files')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel worker processes')
    
    args = parser.parse_args()
    
    preprocessor = VolumePreprocessor(
        output_dir=args.output_dir,
        context_slices=args.context_slices,
        min_nodule_diameter=args.min_nodule,
        full_volume=args.full_volume,
        image_size=args.image_size,
        crop_lungs=args.crop_lungs,
        export_nifti=args.export_nifti
    )
    preprocessor.num_workers = args.workers
    
    if args.dataset == 'lndb':
        preprocessor.convert_lndb(args.input_dir)
    elif args.dataset == 'msd':
        preprocessor.convert_msd_lung(args.input_dir)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
