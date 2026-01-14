#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Preprocessing for DeepLung (Faster R-CNN)
============================================

Handles the conversion of raw DICOM series into preprocessed 3D volumes (NPZ).
Refactored to use SimpleITK for robust DICOM loading, matching segmentation/train_3dunet's methodology.

Key steps:
1. Load DICOM series using SimpleITK (ImageSeriesReader).
2. Convert to numpy array (automatically handles attributes).
3. Resample to fixed spacing (1.0, 1.0, 1.0) mm.
4. Normalize to [0, 1] using windowing.
5. Save as NPZ.

Author: Antigravity
"""

import os
import glob
import logging
import numpy as np
import scipy.ndimage
import SimpleITK as sitk
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepLungPreprocessor:
    def __init__(self, 
                 output_dir: str = "../../cache/deep_lung_cache",
                 target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 min_hu: int = -1000,
                 max_hu: int = 400,
                 num_workers: int = 4):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_spacing = np.array(target_spacing, dtype=np.float32)
        self.min_hu = min_hu
        self.max_hu = max_hu
        
    def process_patient(self, patient_dir: str, xml_dir: Optional[str] = None) -> Optional[str]:
        """
        Process a single patient: DICOM -> 3D Volume (NPZ)
        
        Args:
            patient_dir: Path to patient's directory containing DICOM files
            xml_dir: Path to XML annotations directory (optional)
            
        Returns:
            Path to saved .npz file or None if failed
        """
        patient_id = Path(patient_dir).name
        logger.info(f"Processing patient: {patient_id}")
        
        try:
            # 1. Load DICOM using SimpleITK
            image_sitk = self._load_dicom_sitk(patient_dir)
            if image_sitk is None:
                logger.error(f"No DICOM series found for {patient_id}")
                return None
            
            # Get properties
            image_array = sitk.GetArrayFromImage(image_sitk) # (D, H, W)
            current_spacing = np.array(image_sitk.GetSpacing()[::-1]) # (z, y, x) from (x, y, z)
            origin = np.array(image_sitk.GetOrigin())
            
            # 2. Convert to HU & Windowing
            # SimpleITK usually loads as int16/int32. If it's DICOM, it often applies rescale automatically or we assume raw values are close to HU if RescaleSlope=1.
            # But safer to manually clip regardless.
            # Unlike manually applying slope/intercept, sitk.ReadImage usually handles it for DICOM series.
            
            # 3. Resample
            image_resampled, new_spacing = self._resample(image_array, current_spacing, self.target_spacing)
            
            # 4. Normalize
            image_norm = self._normalize(image_resampled)
            
            # 5. Process Annotations
            boxes = []
            if xml_dir:
                # We need slice information. SITK image doesn't give per-slice SOPInstanceUID easily after loading as volume.
                # However, ImageSeriesReader.GetGDCMSeriesFileNames returns files in order.
                # We can read UIDs from filenames or assume file order matches Z-stack order (which SITK guarantees).
                
                # Logic: Get file list, map index to UID, parse XML.
                boxes = self._process_annotations_sitk(xml_dir, patient_dir, image_array.shape, image_resampled.shape)
            
            # 6. Save
            save_path = self.output_dir / f"{patient_id}_clean.npz"
            np.savez_compressed(save_path, 
                                image=image_norm, 
                                spacing=new_spacing,
                                origin=origin,
                                boxes=np.array(boxes)) 
            
            logger.info(f"Saved {patient_id} to {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Failed to process {patient_id}: {e}", exc_info=True)
            return None

    def _load_dicom_sitk(self, path: str) -> Optional[sitk.Image]:
        """Load DICOM series using SimpleITK"""
        reader = sitk.ImageSeriesReader()
        
        # Look for standard DICOM directory
        dicom_dir = Path(path)
        if (dicom_dir / "dicom_files").exists():
            dicom_dir = dicom_dir / "dicom_files"
            
        series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        
        if not series_ids:
            return None
        
        # Load the first series
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
        reader.SetFileNames(dicom_names)
        
        try:
            image = reader.Execute()
            return image
        except RuntimeError as e:
            logger.error(f"SimpleITK failed to load series: {e}")
            return None

    def _resample(self, image: np.ndarray, scan_spacing: np.ndarray, target_spacing: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resample to target spacing"""
        if np.array_equal(scan_spacing, target_spacing):
            return image, target_spacing
            
        resize_factor = scan_spacing / target_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = scan_spacing / real_resize_factor
        
        image_resampled = scipy.ndimage.zoom(image, real_resize_factor, mode='nearest')
        return image_resampled, new_spacing

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Min-Max Normalization with Clipping"""
        image = np.clip(image, self.min_hu, self.max_hu)
        image = (image - self.min_hu) / (self.max_hu - self.min_hu)
        return image.astype(np.float32)
        
    def _process_annotations_sitk(self, xml_dir: str, dicom_dir: str, 
                                  orig_shape: Tuple, new_shape: Tuple) -> List[List[float]]:
        """
        Process annotations matching SITK Loaded Volume.
        SITK loads files in specific order. We need to match that order to Z-index.
        """
        # Get Ordered File Names
        reader = sitk.ImageSeriesReader()
        base_dicom_dir = Path(dicom_dir)
        if (base_dicom_dir / "dicom_files").exists():
            base_dicom_dir = base_dicom_dir / "dicom_files"
            
        series_ids = reader.GetGDCMSeriesIDs(str(base_dicom_dir))
        if not series_ids:
            return []
        
        dicom_files = reader.GetGDCMSeriesFileNames(str(base_dicom_dir), series_ids[0])
        
        # Helper to extract UID or filename stem to match XML
        # Assuming XML filename matches DICOM filename stem (SOPInstanceUID)
        # We need to map: index_in_volume -> dicom_filename -> xml_filename
        
        # It's safer if we can trust the DICOM files
        # Let's create a map: z_index -> filename_stem
        z_to_stem = {i: Path(f).stem for i, f in enumerate(dicom_files)}
        
        if not os.path.exists(xml_dir):
            return []
            
        # Scaling Factors
        z_factor = new_shape[0] / orig_shape[0]
        y_factor = new_shape[1] / orig_shape[1]
        x_factor = new_shape[2] / orig_shape[2]
        
        final_boxes = []
        
        for z in range(len(dicom_files)):
            stem = z_to_stem[z]
            xml_path = Path(xml_dir) / f"{stem}.xml"
            
            if xml_path.exists():
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    for obj in root.findall('object'):
                        bndbox = obj.find('bndbox')
                        xmin = int(bndbox.find('xmin').text)
                        ymin = int(bndbox.find('ymin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymax = int(bndbox.find('ymax').text)
                        
                        # Convert to Resampled 3D Coords
                        
                        # New Center
                        cz = z * z_factor
                        cy = ((ymin + ymax) / 2) * y_factor
                        cx = ((xmin + xmax) / 2) * x_factor
                        
                        # Size
                        d = 1.0 * z_factor
                        h = (ymax - ymin) * y_factor
                        w = (xmax - xmin) * x_factor
                        
                        final_boxes.append([cz, cy, cx, d, h, w])
                        
                except Exception:
                    pass
                    
        return final_boxes

    def process_dataset(self, data_root: str, dataset_type: str = 'generic', split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """
        Process entire dataset.
        Args:
            data_root: Root directory of dataset
            dataset_type: 'generic' (DICOM+XML) or 'lndb' (MHD+CSV)
        """
        if dataset_type == 'lndb':
            self.convert_lndb(data_root, split_ratios)
        else:
            self.process_generic(data_root, split_ratios)

    def process_generic(self, data_root: str, split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """
        Process properties: Generic DICOM + XML
        """
        data_path = Path(data_root)
        if not data_path.exists():
            logger.error(f"Data root not found: {data_root}")
            return

        # 1. Index all valid patient directories
        patient_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        patient_ids = [d.name for d in patient_dirs]
        
        total = len(patient_ids)
        logger.info(f"Found {total} patients in {data_root}")
        
        # 2. Shuffle and Split
        np.random.seed(42)
        indices = np.random.permutation(total)
        
        n_train = int(total * split_ratios[0])
        n_val = int(total * split_ratios[1])
        
        splits = {
            'train': [patient_dirs[i] for i in indices[:n_train]],
            'val': [patient_dirs[i] for i in indices[n_train:n_train+n_val]],
            'test': [patient_dirs[i] for i in indices[n_train+n_val:]]
        }
        
        # 3. Process
        for split_name, dirs in splits.items():
            split_dir = self.output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing {split_name} split: {len(dirs)} patients")
            
            for p_dir in dirs:
                xml_dir = p_dir / 'xml_annotations'
                if not xml_dir.exists():
                     xml_dir = None
                     
                self._process_patient_to_split(p_dir, xml_dir, split_dir)

    def convert_lndb(self, lndb_dir: str, split_ratios: Tuple[float, float, float]):
        """
        Convert LNDb Dataset (MHD + CSV) to DeepLung NPZ
        Adapted from segmentation/train_3dunet/preprocess.py
        """
        lndb_path = Path(lndb_dir)
        logger.info(f"Processing LNDb dataset from: {lndb_path}")
        
        # Load Annotations
        # Try finding the csv
        csv_path = lndb_path / 'trainset_csv' / 'trainNodules_gt.csv'
        if not csv_path.exists():
            csv_path = lndb_path / 'trainNodules_gt.csv'
            
        if not csv_path.exists():
            logger.error("LNDb CSV not found (trainNodules_gt.csv)")
            return
            
        df = pd.read_csv(csv_path)
        
        # Index CT Files (LNDb-XXXX.mhd)
        # Often in data0, data1... or just root
        ct_files = {}
        # Check root and subdirs data0-5
        search_paths = [lndb_path] + [lndb_path / f'data{i}' for i in range(10)]
        for p in search_paths:
            if p.exists():
                for f in p.glob("LNDb-*.mhd"):
                    # Exclude masks if they are mhd (usually they contain 'rad')
                    if 'rad' not in f.name:
                        pid = int(f.stem.split('-')[1])
                        ct_files[pid] = f
                        
        patient_ids = list(ct_files.keys())
        logger.info(f"Found {len(patient_ids)} LNDb patients.")

        # Split
        np.random.seed(42)
        np.random.shuffle(patient_ids)
        n = len(patient_ids)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])
        
        splits = {
            'train': patient_ids[:n_train],
            'val': patient_ids[n_train:n_train+n_val],
            'test': patient_ids[n_train+n_val:]
        }
        
        for split_name, pids in splits.items():
            split_dir = self.output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for pid in tqdm(pids, desc=f"LNDb {split_name}"):
                self._process_lndb_patient(pid, ct_files[pid], df, split_dir)

    def _process_lndb_patient(self, pid: int, mhd_path: Path, df: pd.DataFrame, output_dir: Path):
        try:
            # 1. Load Image
            image_sitk = sitk.ReadImage(str(mhd_path))
            image_array = sitk.GetArrayFromImage(image_sitk) # (Z, Y, X)
            
            # SimpleITK: (x, y, z) spacing/origin
            # Array: (z, y, x)
            current_spacing_sitk = image_sitk.GetSpacing()
            current_spacing = np.array(current_spacing_sitk[::-1]) # (z, y, x)
            origin_sitk = image_sitk.GetOrigin()
            origin = np.array(origin_sitk) # (x, y, z)
            
            # 2. Get Annotations for this patient
            # Columns: LNDbID, RadID, RadFindingID, FindingID, x, y, z, AgrLevel, Nodule, Volume, Text
            # We filter for this patient
            
            patient_nodules = df[df['LNDbID'] == pid]
            
            boxes_3d = [] # List of [z_new, y_new, x_new, d, h, w]
            
            # Resize factors for (z, y, x)
            resize_factor = current_spacing / self.target_spacing
            
            # Group by FindingID to handle potential duplicates (though _gt csv might be unique)
            # We filter Nodule == 1 (Nodule > 0 usually implies nodule, 0 is non-nodule)
            
            if 'Nodule' in df.columns:
                patient_nodules = patient_nodules[patient_nodules['Nodule'] >= 1]
            
            grouped = patient_nodules.groupby('FindingID')
            for _, group in grouped:
                # Average properties in case of duplicates
                # Coordinates in LNDb GT are in World Coordinates (mm)
                mean_x = group['x'].mean()
                mean_y = group['y'].mean()
                mean_z = group['z'].mean()
                
                # Volume to Radius
                # V = 4/3 * pi * r^3 => r = (3V / 4pi)^(1/3)
                mean_vol = group['Volume'].mean()
                if mean_vol <= 0:
                    continue
                    
                r_mm = (3 * mean_vol / (4 * 3.1415926535))**(1/3)
                
                if r_mm * 2 < 3: # Skip small nodules (<3mm diameter)
                    continue
                
                # Convert World (mm) -> Voxel Index (Original)
                # SimpleITK uses (x, y, z) for points
                point_mm = (mean_x, mean_y, mean_z)
                
                try:
                    # TransformPhysicalPointToContinuousIndex handles origin and direction matrix correctly
                    idx_x, idx_y, idx_z = image_sitk.TransformPhysicalPointToContinuousIndex(point_mm)
                except Exception:
                    # Fallback if point is outside? (Should verify)
                    # Manual: (p - origin) / spacing (assuming alignment)
                    idx_x = (mean_x - origin_sitk[0]) / current_spacing_sitk[0]
                    idx_y = (mean_y - origin_sitk[1]) / current_spacing_sitk[1]
                    idx_z = (mean_z - origin_sitk[2]) / current_spacing_sitk[2]

                # Convert Voxel Index (Original) -> Voxel Index (Resampled)
                # Zoom applies to image indices: new = old * factor
                # Array is (z, y, x)
                
                z_new = idx_z * resize_factor[0]
                y_new = idx_y * resize_factor[1]
                x_new = idx_x * resize_factor[2]
                
                # Box Size (Diameter) in Resampled Pixels
                # Diameter (mm) = 2 * r_mm
                # Diameter (pixels) = Diameter (mm) / Spacing_New (which is 1.0)
                # So d, h, w ~= 2 * r_mm
                
                d = r_mm * 2
                h = r_mm * 2
                w = r_mm * 2
                
                boxes_3d.append([z_new, y_new, x_new, d, h, w])
            
            # 3. Resample Image
            image_resampled, new_spacing = self._resample(image_array, current_spacing, self.target_spacing)
            
            # 4. Normalize
            image_norm = self._normalize(image_resampled)
            
            # 5. Save
            fname = f"LNDb-{pid:04d}.npz"
            save_path = output_dir / fname
            np.savez_compressed(save_path, 
                                image=image_norm, 
                                spacing=new_spacing,
                                origin=origin,
                                boxes=np.array(boxes_3d))
        except Exception as e:
            logger.error(f"Error converting LNDb {pid}: {e}")

    # (Previous process_dataset becomes process_generic or we verify args)


    def _process_patient_to_split(self, patient_dir: Path, xml_dir: Optional[Path], split_output_dir: Path):
        """Helper to save to specific split directory"""
        try:
            # Re-use logic from process_patient but redirect output
            # (Refactoring process_patient would be cleaner, but for now we adapt)
            
            patient_id = patient_dir.name
            
            # 1. Load
            image_sitk = self._load_dicom_sitk(str(patient_dir))
            if image_sitk is None:
                return

            image_array = sitk.GetArrayFromImage(image_sitk)
            current_spacing = np.array(image_sitk.GetSpacing()[::-1])
            origin = np.array(image_sitk.GetOrigin())

            # 2. Resample
            image_resampled, new_spacing = self._resample(image_array, current_spacing, self.target_spacing)

            # 3. Normalize
            image_norm = self._normalize(image_resampled)

            # 4. Annotations
            boxes = []
            if xml_dir:
                boxes = self._process_annotations_sitk(str(xml_dir), str(patient_dir), image_array.shape, image_resampled.shape)

            # 5. Save
            save_path = split_output_dir / f"{patient_id}.npz"
            np.savez_compressed(save_path, 
                                image=image_norm, 
                                spacing=new_spacing,
                                origin=origin,
                                boxes=np.array(boxes))
                                
            logger.info(f"Saved {patient_id} to {save_path}")
            
        except Exception as e:
            logger.error(f"Error processing {patient_dir.name}: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--dataset_type', type=str, default='generic', choices=['generic', 'lndb'], help='Dataset type: generic (DICOM folder) or lndb (MHD+CSV)')
    parser.add_argument('--output_dir', type=str, default='../../cache/deep_lung_cache', help='Output cache directory')
    args = parser.parse_args()
    
    preprocessor = DeepLungPreprocessor(output_dir=args.output_dir)
    preprocessor.process_dataset(args.data_root, dataset_type=args.dataset_type)

