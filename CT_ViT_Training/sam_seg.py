#!/usr/bin/env python3
"""
MedSAM/MedSAM2 Segmentation with Spatial Alignment
=================================================

Integrated and optimized MedSAM implementation for chest tumor segmentation
with built-in spatial alignment verification and testing capabilities.

Supports both:
- MedSAM2: Advanced medical image segmentation model (recommended)
- Original SAM: Hugging Face SAM models as fallback

Usage:
    # Use MedSAM2 (recommended)
    python sam_seg.py --model medsam2 --config sam2_hiera_s
    
    # Use original SAM
    python sam_seg.py --model facebook/sam-vit-huge
"""

import sys
import json
import logging
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime

import numpy as np
import cv2
import nibabel as nib
import pydicom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# PyTorch availability check
try:
    import torch
    # MedSAM2 imports
    from sam2_train.build_sam import build_sam2_video_predictor
    from sam2_train.sam2_image_predictor import SAM2ImagePredictor
    TORCH_AVAILABLE = True
    MEDSAM2_AVAILABLE = True
except ImportError:
    try:
        import torch
        from transformers import SamModel, SamProcessor
        TORCH_AVAILABLE = True
        MEDSAM2_AVAILABLE = False
    except ImportError:
        TORCH_AVAILABLE = False
        MEDSAM2_AVAILABLE = False


def setup_logging(log_dir: str = "segmentation_result") -> logging.Logger:
    """Setup logging configuration"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_path / "medsam_seg.log"), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


class MedSAMSegmentator:
    """Integrated MedSAM segmentation with spatial alignment verification - supports both SAM and MedSAM2"""
    
    def __init__(self, data_dir: str = "all_patient_data", model_name: str = "medsam2", config_file: str = "sam2.1_hiera_t512.yaml"):
        """Initialize MedSAM Segmentator
        
        Args:
            data_dir: Directory containing patient data
            model_name: Model type - "medsam2", "facebook/sam-vit-huge", or other SAM models
            config_file: Configuration file for MedSAM2 (sam2_hiera_s, sam2_hiera_l, etc.)
        """
        self.data_dir = Path(data_dir)
        self.segmentation_result_base = Path("segmentation_result")
        self.model_name = model_name
        self.config_file = config_file
        self.model = None
        self.processor = None
        self.predictor = None
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu" if TORCH_AVAILABLE else None
        
        if TORCH_AVAILABLE:
            self._load_model()
        else:
            logger.warning("PyTorch unavailable. Running in mock mode.")
        
        logger.info(f"MedSAM initialized - Data: {self.data_dir}, Model: {self.model_name}, Device: {self.device}")
    
    def _load_model(self):
        """Load SAM model and processor"""
        try:
            if self.model_name.startswith("medsam2") and MEDSAM2_AVAILABLE:
                # Load MedSAM2 model
                checkpoint_path = "MedSAM2/checkpoints/MedSAM2_latest.pt"  # Use the downloaded model
                
                # Reinitialize Hydra with the correct config path for MedSAM2
                from hydra import initialize_config_dir
                from hydra.core.global_hydra import GlobalHydra
                import os
                
                # Clear existing Hydra instance
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()
                
                # Initialize with the MedSAM2 config directory
                config_dir = os.path.abspath("MedSAM2/sam2/configs")
                initialize_config_dir(config_dir=config_dir, version_base="1.2")
                
                # Use the config filename without extension
                config_name = self.config_file.replace('.yaml', '')
                
                self.model = build_sam2_video_predictor(
                    config_file=config_name,
                    ckpt_path=checkpoint_path,
                    device=self.device
                )
                self.predictor = SAM2ImagePredictor(sam_model=self.model)
                logger.info(f"MedSAM2 model loaded: {config_name}")
            elif not self.model_name.startswith("medsam2") and not MEDSAM2_AVAILABLE:
                # Fallback to original SAM
                from transformers import SamModel, SamProcessor
                self.model = SamModel.from_pretrained(self.model_name)
                self.processor = SamProcessor.from_pretrained(self.model_name)
                if self.device:
                    self.model.to(self.device)
                logger.info(f"Original SAM model loaded: {self.model_name}")
            else:
                logger.warning("Model configuration mismatch. Using mock mode.")
                self.model = None
                self.processor = None
                self.predictor = None
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to mock mode")
            self.model = None
            self.processor = None
            self.predictor = None
    
    def get_patient_list(self) -> List[str]:
        """Get list of available patients"""
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            return []
        
        patients = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        return sorted(patients)
    
    def load_patient_data(self, patient_id: str) -> Dict:
        """Load patient DICOM and XML files"""
        patient_dir = self.data_dir / patient_id
        if not patient_dir.exists():
            logger.error(f"Patient directory not found: {patient_dir}")
            return {}
        
        # Check for different possible directory structures
        dicom_files = []
        xml_files = []
        
        # Try common directory structures
        for dicom_dir_name in ["dicom", "dicom_files"]:
            dicom_dir = patient_dir / dicom_dir_name
            if dicom_dir.exists():
                dicom_files = list(dicom_dir.glob("*.dcm"))
                break
        
        for xml_dir_name in ["xml", "xml_annotations"]:
            xml_dir = patient_dir / xml_dir_name
            if xml_dir.exists():
                xml_files = list(xml_dir.glob("*.xml"))
                break
        
        if not dicom_files:
            logger.warning(f"No DICOM files found for patient {patient_id}")
        if not xml_files:
            logger.warning(f"No XML annotations found for patient {patient_id}")
        
        return {
            'patient_id': patient_id,
            'dicom_files': sorted(dicom_files),
            'xml_files': xml_files,
            'dicom_count': len(dicom_files),
            'annotation_count': len(xml_files)
        }
    
    def load_dicom_image(self, dicom_path: Path) -> Tuple[np.ndarray, Dict]:
        """Load DICOM image and extract metadata"""
        try:
            dicom_data = pydicom.dcmread(str(dicom_path))
            
            # Extract image
            image = dicom_data.pixel_array.astype(np.float32)
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            
            # Normalize to 0-255
            image = ((image - image.min()) / (np.ptp(image) + 1e-8) * 255).astype(np.uint8)
            
            # Extract metadata
            metadata = {
                'image_position': getattr(dicom_data, 'ImagePositionPatient', [0.0, 0.0, 0.0]),
                'image_orientation': getattr(dicom_data, 'ImageOrientationPatient', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                'pixel_spacing': getattr(dicom_data, 'PixelSpacing', [1.0, 1.0]),
                'slice_thickness': getattr(dicom_data, 'SliceThickness', 1.0),
                'slice_location': getattr(dicom_data, 'SliceLocation', None),
                'instance_number': getattr(dicom_data, 'InstanceNumber', 0),
                'series_uid': getattr(dicom_data, 'SeriesInstanceUID', 'unknown'),
                'study_uid': getattr(dicom_data, 'StudyInstanceUID', 'unknown')
            }
            
            return image, metadata
            
        except Exception as e:
            logger.error(f"Failed to load DICOM {dicom_path}: {e}")
            return None, {}
    
    def parse_xml_annotation(self, xml_path: Path) -> List[Dict]:
        """Parse XML annotation file"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            annotations = []
            
            # Get image size
            size_elem = root.find('size')
            width = int(size_elem.find('width').text) if size_elem is not None else 512
            height = int(size_elem.find('height').text) if size_elem is not None else 512
            
            # Parse objects
            for obj in root.findall('object'):
                name = obj.find('name').text if obj.find('name') is not None else 'unknown'
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    bbox = [int(bndbox.find(coord).text) for coord in ['xmin', 'ymin', 'xmax', 'ymax']]
                    annotations.append({'name': name, 'bbox': bbox, 'width': width, 'height': height})
            
            return annotations
        except Exception as e:
            logger.error(f"Failed to parse XML {xml_path}: {e}")
            return []
    
    def find_matching_annotation(self, dicom_path: Path, xml_files: List[Path]) -> Optional[Path]:
        """Find matching XML annotation for a DICOM file"""
        try:
            dicom_data = pydicom.dcmread(str(dicom_path))
            instance_uid = str(dicom_data.SOPInstanceUID)
            return next((xml for xml in xml_files if instance_uid in xml.name), None)
        except Exception as e:
            logger.error(f"Failed to find annotation for {dicom_path}: {e}")
            return None
    
    def segment_with_medsam(self, image: np.ndarray, bounding_boxes: List[List[int]]) -> List[np.ndarray]:
        """Perform segmentation using MedSAM (supports both SAM and MedSAM2)"""
        if not TORCH_AVAILABLE or (self.model is None and self.predictor is None):
            return self._generate_mock_masks(image, bounding_boxes)
        
        try:
            masks = []
            
            if self.predictor is not None:  # MedSAM2 path
                # Convert image to RGB if grayscale
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Ensure image is in RGB format
                    rgb_image = image
                else:
                    # Convert grayscale to RGB
                    rgb_image = np.stack([image] * 3, axis=-1) if len(image.shape) == 2 else image
                
                # Set image for MedSAM2 predictor
                self.predictor.set_image(rgb_image)
                
                for bbox in bounding_boxes:
                    # Convert bbox format: [xmin, ymin, xmax, ymax] -> [xmin, ymin, xmax, ymax]
                    input_box = np.array(bbox)
                    
                    # Predict mask using MedSAM2
                    masks_pred, scores, logits = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )
                    
                    # Take the first (and only) mask
                    mask = masks_pred[0].astype(np.uint8)
                    masks.append(mask)
                    
            else:  # Original SAM path
                for bbox in bounding_boxes:
                    inputs = self.processor(image, input_boxes=[[bbox]], return_tensors="pt")
                    if self.device:
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    mask = self.processor.image_processor.post_process_masks(
                        outputs.pred_masks.cpu(),
                        inputs["original_sizes"].cpu(),
                        inputs["reshaped_input_sizes"].cpu()
                    )[0]
                    
                    masks.append((mask[0, 0].numpy() > 0.5).astype(np.uint8))
            
            return masks
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return self._generate_mock_masks(image, bounding_boxes)
    
    def _generate_mock_masks(self, image: np.ndarray, bounding_boxes: List[List[int]]) -> List[np.ndarray]:
        """Generate mock segmentation masks"""
        h, w = image.shape[:2]
        masks = []
        
        for x1, y1, x2, y2 in bounding_boxes:
            mask = np.zeros((h, w), dtype=np.uint8)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            radius = ((x2 - x1) // 2, (y2 - y1) // 2)
            cv2.ellipse(mask, center, radius, 0, 0, 360, 1, -1)
            masks.append(mask)
        
        return masks
    
    def create_dicom_to_nifti_affine(self, metadata: Dict) -> np.ndarray:
        """Create proper affine transformation matrix for 3D Slicer alignment"""
        try:
            image_position = np.array(metadata.get('image_position', [0.0, 0.0, 0.0]), dtype=float)
            image_orientation = np.array(metadata.get('image_orientation', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), dtype=float)
            spacing = metadata.get('spacing', (1.0, 1.0, 1.0))
            
            # DICOM orientation vectors
            row_cosines = image_orientation[:3]
            col_cosines = image_orientation[3:6]
            
            # Calculate normal vector
            normal_vector = np.cross(row_cosines, col_cosines)
            normal_vector = normal_vector / np.linalg.norm(normal_vector) if np.linalg.norm(normal_vector) > 0 else normal_vector
            
            # Build direction matrix
            direction_matrix = np.column_stack([
                col_cosines * spacing[1],
                row_cosines * spacing[0],
                normal_vector * spacing[2]
            ])
            
            # Create 4x4 affine matrix
            affine = np.eye(4)
            affine[:3, :3] = direction_matrix
            affine[:3, 3] = image_position
            
            # Apply coordinate transformation for 3D Slicer compatibility
            coord_transform = np.diag([-1, -1, 1, 1])
            affine = coord_transform @ affine
            
            return affine
            
        except Exception as e:
            logger.error(f"Failed to create affine matrix: {e}")
            return np.eye(4)
    
    def save_masks_as_nifti(self, masks_3d: np.ndarray, output_path: str, metadata: Dict) -> None:
        """Save 3D masks as NIfTI with proper spatial alignment"""
        try:
            # Transpose from (z,y,x) to (x,y,z) for NIfTI convention
            if len(masks_3d.shape) == 3:
                masks_3d = masks_3d.transpose(2, 1, 0)
            
            # Create affine matrix
            affine = self.create_dicom_to_nifti_affine(metadata)
            
            # Create NIfTI image
            nifti_img = nib.Nifti1Image(masks_3d.astype(np.uint8), affine)
            nifti_img.header.set_xyzt_units('mm', 'sec')
            
            # Save to file
            nib.save(nifti_img, output_path)
            logger.info(f"Saved NIfTI: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save NIfTI {output_path}: {e}")
    
    def create_3d_mask_volume(self, slice_results: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """Create 3D volume from slice masks with proper spatial alignment"""
        if not slice_results:
            return np.array([]), {}
        
        # Sort slices by spatial position
        def sort_key(result):
            metadata = result['metadata']
            if metadata.get('slice_location') is not None:
                return float(metadata['slice_location'])
            elif metadata.get('image_position') and len(metadata['image_position']) >= 3:
                return float(metadata['image_position'][2])
            else:
                return float(metadata.get('instance_number', 0))
        
        sorted_results = sorted(slice_results, key=sort_key)
        
        # Create 3D volume
        first_mask = sorted_results[0]['masks'][0] if sorted_results[0]['masks'] else np.zeros((512, 512), dtype=np.uint8)
        depth = len(sorted_results)
        height, width = first_mask.shape
        
        volume_3d = np.zeros((depth, height, width), dtype=np.uint8)
        
        for i, result in enumerate(sorted_results):
            if result['masks']:
                # Combine all masks for this slice
                combined_mask = np.logical_or.reduce(result['masks']) if len(result['masks']) > 1 else result['masks'][0]
                volume_3d[i] = combined_mask.astype(np.uint8)
        
        # Create metadata for the volume
        first_metadata = sorted_results[0]['metadata']
        pixel_spacing = first_metadata.get('pixel_spacing', [1.0, 1.0])
        
        # Calculate z-spacing
        z_spacing = 1.0
        if len(sorted_results) > 1:
            first_pos = first_metadata.get('slice_location')
            last_pos = sorted_results[-1]['metadata'].get('slice_location')
            if first_pos is not None and last_pos is not None:
                z_spacing = abs(float(last_pos) - float(first_pos)) / (depth - 1)
        
        volume_metadata = {
            'shape': volume_3d.shape,
            'spacing': (float(z_spacing), float(pixel_spacing[0]), float(pixel_spacing[1])),
            'image_position': first_metadata.get('image_position', [0.0, 0.0, 0.0]),
            'image_orientation': first_metadata.get('image_orientation', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            'series_uid': first_metadata.get('series_uid', 'unknown')
        }
        
        return volume_3d, volume_metadata
    
    def process_patient(self, patient_id: str, save_results: bool = True, create_reference: bool = True) -> Dict:
        """Process a patient with MedSAM segmentation"""
        logger.info(f"Processing patient: {patient_id}")
        
        # Load patient data
        patient_data = self.load_patient_data(patient_id)
        if not patient_data or not patient_data['dicom_files']:
            return {'status': 'error', 'message': 'No patient data found'}
        
        # Create reference NIfTI first (if requested)
        reference_path = None
        if create_reference and save_results:
            logger.info("Creating DICOM reference NIfTI...")
            reference_path = self.create_reference_nifti(patient_id, match_segmentation_range=True)
            if reference_path:
                logger.info(f"Reference NIfTI created: {Path(reference_path).name}")
        
        # Process each annotated slice
        slice_results = []
        
        for dicom_path in patient_data['dicom_files']:
            # Find matching annotation
            xml_path = self.find_matching_annotation(dicom_path, patient_data['xml_files'])
            if not xml_path:
                continue
            
            # Load image and parse annotations
            image, metadata = self.load_dicom_image(dicom_path)
            annotations = self.parse_xml_annotation(xml_path)
            
            if image is None or not annotations:
                continue
            
            # Perform segmentation
            bounding_boxes = [ann['bbox'] for ann in annotations]
            masks = self.segment_with_medsam(image, bounding_boxes)
            
            slice_results.append({
                'dicom_file': dicom_path.name,
                'xml_file': xml_path.name,
                'metadata': metadata,
                'annotations': annotations,
                'masks': masks
            })
        
        if not slice_results:
            return {'status': 'error', 'message': 'No annotated slices found'}
        
        # Create 3D volume
        volume_3d, volume_metadata = self.create_3d_mask_volume(slice_results)
        
        # Save segmentation results
        nifti_path = None
        if save_results and volume_3d.size > 0:
            output_dir = self.segmentation_result_base / patient_id
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nifti_path = output_dir / f"segmentation_{timestamp}.nii.gz"
            self.save_masks_as_nifti(volume_3d, str(nifti_path), volume_metadata)
        
        # Verify alignment if both files exist
        alignment_verified = False
        if reference_path and nifti_path:
            logger.info("Verifying spatial alignment...")
            verification = AlignmentVerifier.verify_alignment(reference_path, str(nifti_path))
            alignment_verified = verification.get('aligned', False)
            if alignment_verified:
                logger.info("[SUCCESS] Spatial alignment verified - files should align perfectly in 3D Slicer!")
            else:
                logger.warning("[WARNING] Spatial alignment verification failed")
        
        return {
            'status': 'success',
            'patient_id': patient_id,
            'processed_slices': len(slice_results),
            'slice_results': slice_results,
            'volume_shape': volume_3d.shape if volume_3d.size > 0 else None,
            'nifti_path': str(nifti_path) if nifti_path else None,
            'reference_path': reference_path,
            'alignment_verified': alignment_verified
        }
    
    def create_reference_nifti(self, patient_id: str, match_segmentation_range: bool = True) -> Optional[str]:
        """Create reference NIfTI file from DICOM images"""
        logger.info(f"Creating reference NIfTI for patient {patient_id}")
        
        patient_data = self.load_patient_data(patient_id)
        if not patient_data or not patient_data['dicom_files']:
            return None
        
        # Find annotated slices if matching segmentation range
        target_files = []
        if match_segmentation_range:
            for dicom_path in patient_data['dicom_files']:
                xml_path = self.find_matching_annotation(dicom_path, patient_data['xml_files'])
                if xml_path and self.parse_xml_annotation(xml_path):
                    target_files.append(dicom_path)
        else:
            target_files = patient_data['dicom_files']
        
        if not target_files:
            return None
        
        # Load images and metadata
        slice_data = []
        for dicom_path in target_files:
            image, metadata = self.load_dicom_image(dicom_path)
            if image is not None:
                slice_data.append({'image': image, 'metadata': metadata})
        
        # Sort by spatial position
        def sort_key(slice_info):
            metadata = slice_info['metadata']
            if metadata.get('slice_location') is not None:
                return float(metadata['slice_location'])
            elif metadata.get('image_position') and len(metadata['image_position']) >= 3:
                return float(metadata['image_position'][2])
            else:
                return float(metadata.get('instance_number', 0))
        
        sorted_slices = sorted(slice_data, key=sort_key)
        
        # Create 3D volume
        depth = len(sorted_slices)
        first_image = sorted_slices[0]['image']
        height, width = first_image.shape[:2]
        
        volume_3d = np.zeros((depth, height, width), dtype=np.uint16)
        
        for i, slice_info in enumerate(sorted_slices):
            image = slice_info['image']
            if len(image.shape) == 3:
                gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray_image = image
            volume_3d[i] = (gray_image * 256).astype(np.uint16)
        
        # Create metadata
        first_metadata = sorted_slices[0]['metadata']
        last_metadata = sorted_slices[-1]['metadata']
        
        pixel_spacing = first_metadata.get('pixel_spacing', [1.0, 1.0])
        z_spacing = 1.0
        if len(sorted_slices) > 1:
            first_loc = first_metadata.get('slice_location')
            last_loc = last_metadata.get('slice_location')
            if first_loc is not None and last_loc is not None:
                z_spacing = abs(float(last_loc) - float(first_loc)) / (depth - 1)
        
        reference_metadata = {
            'shape': volume_3d.shape,
            'spacing': (float(z_spacing), float(pixel_spacing[0]), float(pixel_spacing[1])),
            'image_position': first_metadata.get('image_position', [0.0, 0.0, 0.0]),
            'image_orientation': first_metadata.get('image_orientation', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        }
        
        # Save reference NIfTI
        output_dir = self.segmentation_result_base / patient_id
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reference_path = output_dir / f"reference_{timestamp}.nii.gz"
        
        try:
            # Transpose and save
            volume_3d_transposed = volume_3d.transpose(2, 1, 0)
            affine = self.create_dicom_to_nifti_affine(reference_metadata)
            nifti_img = nib.Nifti1Image(volume_3d_transposed, affine)
            nib.save(nifti_img, str(reference_path))
            logger.info(f"Reference NIfTI saved: {reference_path}")
            return str(reference_path)
        except Exception as e:
            logger.error(f"Failed to save reference NIfTI: {e}")
            return None


class AlignmentVerifier:
    """Verify spatial alignment between NIfTI files"""
    
    @staticmethod
    def verify_alignment(reference_path: str, segmentation_path: str) -> Dict:
        """Verify that two NIfTI files have matching spatial properties"""
        try:
            ref_img = nib.load(reference_path)
            seg_img = nib.load(segmentation_path)
            
            # Compare shapes
            shapes_match = ref_img.shape == seg_img.shape
            
            # Compare affine matrices
            affine_diff = np.abs(ref_img.affine - seg_img.affine)
            max_affine_diff = np.max(affine_diff)
            affines_match = max_affine_diff < 1e-3
            
            # Compare voxel spacing
            ref_pixdim = ref_img.header['pixdim'][1:4]
            seg_pixdim = seg_img.header['pixdim'][1:4]
            spacing_diff = np.abs(ref_pixdim - seg_pixdim)
            max_spacing_diff = np.max(spacing_diff)
            spacing_match = max_spacing_diff < 1e-3
            
            # Calculate origin distance
            ref_origin = ref_img.affine[:3, 3]
            seg_origin = seg_img.affine[:3, 3]
            origin_distance = np.linalg.norm(ref_origin - seg_origin)
            
            overall_aligned = shapes_match and affines_match and spacing_match
            
            return {
                'aligned': overall_aligned,
                'shapes_match': shapes_match,
                'affines_match': affines_match,
                'spacing_match': spacing_match,
                'max_affine_diff': float(max_affine_diff),
                'max_spacing_diff': float(max_spacing_diff),
                'origin_distance': float(origin_distance),
                'reference_shape': ref_img.shape,
                'segmentation_shape': seg_img.shape
            }
            
        except Exception as e:
            logger.error(f"Alignment verification failed: {e}")
            return {'aligned': False, 'error': str(e)}
    
    @staticmethod
    def print_verification_results(results: Dict):
        """Print verification results in a readable format"""
        print("Spatial Alignment Verification")
        print("=" * 50)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        print(f"Overall aligned: {'[YES]' if results['aligned'] else '[NO]'}")
        print(f"Shapes match: {'[YES]' if results['shapes_match'] else '[NO]'}")
        print(f"Affines match: {'[YES]' if results['affines_match'] else '[NO]'}")
        print(f"Spacing match: {'[YES]' if results['spacing_match'] else '[NO]'}")
        print(f"Origin distance: {results['origin_distance']:.3f} mm")
        print(f"Reference shape: {results['reference_shape']}")
        print(f"Segmentation shape: {results['segmentation_shape']}")
        
        if results['aligned']:
            print("\n[SUCCESS] Files should align perfectly in 3D Slicer!")
        else:
            print("\n[WARNING] Files may not align properly in 3D Slicer")


def test_spatial_alignment(patient_id: str = "A0001", model_name: str = "medsam2", config_file: str = "sam2.1_hiera_t512.yaml"):
    """Test spatial alignment for a patient"""
    print(f"Testing spatial alignment for patient: {patient_id}")
    print(f"Using model: {model_name}")
    print("=" * 60)
    
    segmentator = MedSAMSegmentator(model_name=model_name, config_file=config_file)
    
    # 1. Create reference NIfTI
    print("\n1. Creating DICOM reference NIfTI...")
    reference_path = segmentator.create_reference_nifti(patient_id, match_segmentation_range=True)
    
    if reference_path:
        print(f"   Reference created: {Path(reference_path).name}")
    else:
        print("   Failed to create reference")
        return
    
    # 2. Process segmentation
    print("\n2. Processing segmentation...")
    results = segmentator.process_patient(patient_id, save_results=True)
    
    if results['status'] == 'success' and results.get('nifti_path'):
        print(f"   Segmentation created: {Path(results['nifti_path']).name}")
        print(f"   Processed {results['processed_slices']} slices")
        
        # 3. Verify alignment
        print("\n3. Verifying spatial alignment...")
        verification = AlignmentVerifier.verify_alignment(reference_path, results['nifti_path'])
        AlignmentVerifier.print_verification_results(verification)
        
        if verification['aligned']:
            print("\n3D Slicer Instructions:")
            print("1. Load both NIfTI files")
            print("2. Use 'Volumes' module to overlay them")
            print("3. Adjust opacity to see alignment")
        
    else:
        print("   Segmentation failed")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MedSAM Segmentation with Spatial Alignment (SAM/MedSAM2)")
    parser.add_argument("--patient_id", type=str, default="A0001", help="Patient ID to process")
    parser.add_argument("--data_dir", type=str, default="all_patient_data", help="Patient data directory")
    parser.add_argument("--model", type=str, default="medsam2", 
                       help="Model type: 'medsam2' for MedSAM2 or 'facebook/sam-vit-huge' for original SAM")
    parser.add_argument("--config", type=str, default="sam2.1_hiera_t512.yaml", 
                       help="MedSAM2 config: sam2.1_hiera_t512.yaml, efficientmedsam_s_512_FLARE_RECIST.yaml, etc.")
    parser.add_argument("--list_patients", action="store_true", help="List available patients")
    parser.add_argument("--test_alignment", action="store_true", help="Test spatial alignment")
    parser.add_argument("--create_reference", action="store_true", help="Create reference NIfTI only")
    parser.add_argument("--no_reference", action="store_true", help="Skip creating reference DICOM NIfTI")
    parser.add_argument("--verify_alignment", nargs=2, metavar=('REF', 'SEG'), 
                       help="Verify alignment between two NIfTI files")
    
    args = parser.parse_args()
    
    segmentator = MedSAMSegmentator(data_dir=args.data_dir, model_name=args.model, config_file=args.config)
    
    if args.list_patients:
        patients = segmentator.get_patient_list()
        print(f"Available patients ({len(patients)}): {', '.join(patients)}")
        return
    
    if args.test_alignment:
        test_spatial_alignment(args.patient_id, args.model, args.config)
        return
    
    if args.create_reference:
        reference_path = segmentator.create_reference_nifti(args.patient_id)
        if reference_path:
            print(f"Reference NIfTI created: {reference_path}")
        else:
            print("Failed to create reference NIfTI")
        return
    
    if args.verify_alignment:
        ref_path, seg_path = args.verify_alignment
        if not Path(ref_path).exists():
            print(f"Reference file not found: {ref_path}")
            return
        if not Path(seg_path).exists():
            print(f"Segmentation file not found: {seg_path}")
            return
        
        verification = AlignmentVerifier.verify_alignment(ref_path, seg_path)
        AlignmentVerifier.print_verification_results(verification)
        return
    
    # Default: process patient
    create_ref = not args.no_reference  # Default is True unless --no_reference is specified
    results = segmentator.process_patient(args.patient_id, create_reference=create_ref)
    
    if results['status'] == 'success':
        print(f"Patient {results['patient_id']} processed successfully")
        print(f"Processed {results['processed_slices']} slices")
        if results.get('reference_path'):
            print(f"Reference DICOM saved: {results['reference_path']}")
        if results.get('nifti_path'):
            print(f"Segmentation saved: {results['nifti_path']}")
        if results.get('alignment_verified'):
            print("[SUCCESS] Spatial alignment verified - ready for 3D Slicer!")
    else:
        print(f"Processing failed: {results.get('message', 'Unknown error')}")


if __name__ == "__main__":
    main()
