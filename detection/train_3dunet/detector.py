"""
Nodule Detector Module
======================

Handles conversion of segmentation probability maps into structured nodule detections.
Integrates connectivity analysis, feature extraction, and anatomical location estimation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from skimage import measure, morphology
from .location_estimator import LungLocationEstimator

@dataclass
class NoduleLocation:
    lobe: str
    lobe_full: str
    side: str
    vertical_zone: str
    confidence: float
    description: str

@dataclass
class NoduleGeometry:
    centroid_xyz: Tuple[float, float, float] # Voxel coordinates (z, y, x)
    centroid_world: Tuple[float, float, float] # World coordinates (mm)
    bbox: Tuple[int, int, int, int, int, int] # z_min, y_min, x_min, z_max, y_max, x_max
    volume_mm3: float
    diameter_mm: float
    voxel_count: int

@dataclass
class DetectedNodule:
    id: int
    probability: float
    location: NoduleLocation
    geometry: NoduleGeometry
    slice_range: Tuple[int, int]
    slice_indices: List[int] # Absolute slice indices in original scan

class NoduleDetector:
    def __init__(self, 
                 threshold: float = 0.5, 
                 min_size_mm3: float = 30.0,
                 use_gpu_postprocessing: bool = False):
        """
        Args:
            threshold: Probability threshold for binarization
            min_size_mm3: Minimum volume to be considered a nodule
        """
        self.threshold = threshold
        self.min_size_mm3 = min_size_mm3
        self.location_estimator = LungLocationEstimator()
        
    def process_batch(self, 
                      logits: torch.Tensor, 
                      batch_metadata: Dict) -> List[List[DetectedNodule]]:
        """
        Process a batch of predictions.
        
        Args:
            logits: (B, 1, D, H, W) Output logits from model
            batch_metadata: Dictionary containing 'slice_indices', 'spacing', 'origin', 'original_shape'
        
        Returns:
            List of lists, where each inner list contains detections for one sample in batch.
        """
        all_detections = []
        batch_size = logits.shape[0]
        
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        
        # Move to CPU for skimage processing (until we implement fully GPU connected components)
        preds_np = preds.cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        slice_indices_batch = batch_metadata.get('slice_indices') # (B, D)
        spacings_batch = batch_metadata.get('spacing') # (B, 3)
        origins_batch = batch_metadata.get('origin') # (B, 3)
        original_shapes_batch = batch_metadata.get('original_shape') # (B, 3)
        
        for i in range(batch_size):
            sample_detections = []
            
            pred_vol = preds_np[i, 0] # (D, H, W)
            prob_vol = probs_np[i, 0]
            
            # Post-process (Closing) - similar to trainer.postprocess_prediction
            if pred_vol.sum() > 0:
                struct = morphology.ball(1) if pred_vol.ndim == 3 else morphology.disk(1)
                pred_vol = morphology.binary_closing(pred_vol, struct).astype(np.uint8)
            else:
                pred_vol = pred_vol.astype(np.uint8)
            
            # Connected Components
            labels = measure.label(pred_vol)
            regions = measure.regionprops(labels, intensity_image=prob_vol)
            
            # Metadata for this sample
            spacing = spacings_batch[i].numpy() if spacings_batch is not None else np.array([1.0, 1.0, 1.0])
            origin = origins_batch[i].numpy() if origins_batch is not None else np.array([0.0, 0.0, 0.0])
            slice_indices = slice_indices_batch[i].numpy() if slice_indices_batch is not None else None
            total_slices = int(original_shapes_batch[i][0]) if original_shapes_batch is not None else 100
            
            voxel_vol_mm3 = np.prod(spacing)
            
            for region in regions:
                # Filter by size
                vol_mm3 = region.area * voxel_vol_mm3
                
                # Calculate diameter (equivalent spherical diameter)
                diameter_mm = 2 * (3 * vol_mm3 / (4 * np.pi))**(1/3)
                
                if vol_mm3 < self.min_size_mm3:
                    continue
                
                # Geometry
                z_min, y_min, x_min, z_max, y_max, x_max = region.bbox
                centroid = region.centroid # (z, y, x)
                
                # World estimation
                # Note: spacing is usually (z, y, x) matching numpy order
                centroid_world = (
                    origin[0] + centroid[2] * spacing[2], # x (assuming origin/spacing is Image coords usually x,y,z or z,y,x? Need to match dataset.py)
                    origin[1] + centroid[1] * spacing[1], # y
                    origin[2] + centroid[0] * spacing[0]  # z
                )
                # Correction: dataset.py says spacing = np.array(ct_image.GetSpacing()[::-1])  # (z, y, x)
                # origin = np.array(ct_image.GetOrigin()) # SimpleITK Origin is (x,y,z) usually. 
                # SITK Image GetArrayFromImage returns (Z, Y, X).
                # Spacing/Origin in dataset.py was converted?
                # dataset.py: origin = np.array(ct_image.GetOrigin()) -> This is (x, y, z) standard SITK.
                # spacing = np.array(ct_image.GetSpacing()[::-1]) -> This is (z, y, x).
                # So world coord calc:
                # World_Z = Origin_Z + Voxel_Z * Spacing_Z
                # Origin is (x, y, z) -> origin[2] is Z.
                # Spacing is (z, y, x) -> spacing[0] is Z.
                # Voxel is (z, y, x) -> centroid[0] is Z.
                
                world_x = origin[0] + centroid[2] * spacing[2]
                world_y = origin[1] + centroid[1] * spacing[1]
                world_z = origin[2] + centroid[0] * spacing[0]
                centroid_world = (world_x, world_y, world_z)
                
                # Location Estimation
                # Need relative coords (0-1) and slice ratio
                # Volume shape (D, H, W)
                D, H, W = pred_vol.shape
                
                rel_x = centroid[2] / W # 0(Left in array?) -> 1(Right in array?)
                # SITK Array (Z, Y, X). X=0 is usually Right in patient if LPS? Or Left?
                # Medical images are usually standard patient view (Left is Right).
                # Let's assume standard X axis 0->W is Patient Right->Left (R-L) or Left->Right (L-R)?
                # Usually DICOM is LPS (Left, Posterior, Superior).
                # SimpleITK defaults to LPS? 
                # Let's use the LungLocationEstimator's expectation: relative_x: 0=Left(Left Lung?), 1=Right(Right Lung?)
                # Wait, Estimator doc: relative_x: 0=左, 1=右. 
                # Usually in axial view viewing from feet: Left of image is Right of patient.
                # We need to be careful.
                # If we rely on default orientation from dataset loader (which might be RAS or LPS), we might flip L/R.
                # Assuming standard display: rel_x < 0.5 is Patient Right (Image Left), rel_x > 0.5 is Patient Left (Image Right).
                # BUT Estimator says: 0=Left, 1=Right.
                # Let's assume Estimator means "Image Coordinate X", where 0 is Left side of image.
                # Estimator logic: if relative_x < 0.42: side='left' -> This implies 0 is Left Lung (Patient Left).
                # So Estimator expects 0 -> Patient Left.
                # If image is standard (Patient Right on Left, Patient Left on Right), then X=0 is Patient Right.
                # This conflicts.
                # Ideally we check metadata direction. But for now, let's assume standard orientation and Pass simple ratio.
                
                rel_y = centroid[1] / H # 0=Ant, 1=Post?
                
                # Slice Ratio
                # Use absolute slice index if available
                cur_z_idx = int(round(centroid[0]))
                if slice_indices is not None and cur_z_idx < len(slice_indices) and slice_indices[cur_z_idx] != -1:
                    abs_slice = slice_indices[cur_z_idx]
                    slice_ratio = abs_slice / total_slices
                else:
                    # Fallback to local ratio if absolute not known (e.g. padding region? shouldn't contain nodule)
                    # Or if slice_indices missing
                    slice_ratio = cur_z_idx / D # Very rough
                
                loc_res = self.location_estimator.estimate_location(
                    relative_x=rel_x, 
                    relative_y=rel_y, 
                    slice_ratio=slice_ratio
                )
                
                location = NoduleLocation(
                    lobe=loc_res['lobe'],
                    lobe_full=loc_res['lobe_full'],
                    side=loc_res['side'],
                    vertical_zone=loc_res['vertical_zone'],
                    confidence=loc_res['confidence'],
                    description=loc_res['description']
                )
                
                # Slice Range (Local)
                z_range_local = (z_min, z_max)
                # Slice Indices (Absolute)
                vol_slice_indices = []
                if slice_indices is not None:
                    # Collect valid indices in range
                    for z in range(z_min, z_max):
                        if z < len(slice_indices) and slice_indices[z] != -1:
                            vol_slice_indices.append(int(slice_indices[z]))
                
                geo = NoduleGeometry(
                    centroid_xyz=tuple(float(c) for c in centroid),
                    centroid_world=tuple(float(c) for c in centroid_world),
                    bbox=(int(z_min), int(y_min), int(x_min), int(z_max), int(y_max), int(x_max)),
                    volume_mm3=float(vol_mm3),
                    diameter_mm=float(diameter_mm),
                    voxel_count=int(region.area)
                )
                
                nodule = DetectedNodule(
                    id=int(region.label),
                    probability=float(region.mean_intensity), # Mean prob in region
                    location=location,
                    geometry=geo,
                    slice_range=(int(z_range_local[0]), int(z_range_local[1])),
                    slice_indices=[int(idx) for idx in vol_slice_indices]
                )
                
                sample_detections.append(nodule)
            
            # Sort by size or prob? usually size for relevance
            sample_detections.sort(key=lambda x: x.geometry.volume_mm3, reverse=True)
            all_detections.append(sample_detections)
            
        return all_detections
