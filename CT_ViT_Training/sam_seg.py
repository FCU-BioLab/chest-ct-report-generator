#!/usr/bin/env python3
"""
MedSAM2 Segmentation for Chest Tumor
===================================

Simplified MedSAM2 implementation for chest tumor segmentation.

Usage:
    python sam_seg.py --patient_id A0001  # Process specific patient
    python sam_seg.py                      # Process all patients
"""

import sys
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

# Model availability check
try:
    import torch
    from sam2_train.build_sam import build_sam2_video_predictor
    from sam2_train.sam2_image_predictor import SAM2ImagePredictor
    MEDSAM2_AVAILABLE = True
except ImportError:
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
    """Simplified MedSAM2 segmentation"""
    
    def __init__(self, data_dir: str = "all_patient_data", config_file: str = "sam2.1_hiera_t512.yaml"):
        self.data_dir = Path(data_dir)
        self.segmentation_result_base = Path("segmentation_result")
        self.config_file = config_file
        self.model = None
        self.predictor = None
        self.device = "cuda" if MEDSAM2_AVAILABLE and torch.cuda.is_available() else "cpu" if MEDSAM2_AVAILABLE else None
        
        if MEDSAM2_AVAILABLE:
            self._load_medsam2()
        else:
            logger.warning("MedSAM2 unavailable. Running in mock mode.")
        
        logger.info(f"MedSAM2 initialized - Device: {self.device}")
    
    def _load_medsam2(self):
        """Load MedSAM2 model"""
        try:
            checkpoint_path = "MedSAM2/checkpoints/MedSAM2_latest.pt"
            
            from hydra import initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            import os
            
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            
            config_dir = os.path.abspath("MedSAM2/sam2/configs")
            initialize_config_dir(config_dir=config_dir, version_base="1.2")
            config_name = self.config_file.replace('.yaml', '')
            
            self.model = build_sam2_video_predictor(
                config_file=config_name, ckpt_path=checkpoint_path, device=self.device
            )
            self.predictor = SAM2ImagePredictor(sam_model=self.model)
            logger.info(f"MedSAM2 loaded: {config_name}")
            
        except Exception as e:
            logger.error(f"MedSAM2 loading failed: {e}. Using mock mode.")
            self.model = None
            self.predictor = None
    
    def get_patient_list(self) -> List[str]:
        """Get list of available patients"""
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            return []
        return sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
    
    def load_patient_data(self, patient_id: str) -> Dict:
        """Load patient DICOM and XML files"""
        patient_dir = self.data_dir / patient_id
        if not patient_dir.exists():
            logger.error(f"Patient directory not found: {patient_dir}")
            return {}
        
        dicom_files = []
        xml_files = []
        
        # Find DICOM and XML files
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
            
            # Extract and normalize image
            image = dicom_data.pixel_array.astype(np.float32)
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            image = ((image - image.min()) / (np.ptp(image) + 1e-8) * 255).astype(np.uint8)
            
            # Extract key metadata
            metadata = {
                'image_position': getattr(dicom_data, 'ImagePositionPatient', [0.0, 0.0, 0.0]),
                'image_orientation': getattr(dicom_data, 'ImageOrientationPatient', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                'pixel_spacing': getattr(dicom_data, 'PixelSpacing', [1.0, 1.0]),
                'slice_thickness': getattr(dicom_data, 'SliceThickness', 1.0),
                'slice_location': getattr(dicom_data, 'SliceLocation', None),
                'instance_number': getattr(dicom_data, 'InstanceNumber', 0),
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
    
    def segment_with_medsam2(self, image: np.ndarray, bounding_boxes: List[List[int]]) -> List[np.ndarray]:
        """Perform segmentation using MedSAM2"""
        if not MEDSAM2_AVAILABLE or self.predictor is None:
            return self._generate_mock_masks(image, bounding_boxes)
        
        try:
            masks = []
            rgb_image = image if len(image.shape) == 3 else np.stack([image] * 3, axis=-1)
            self.predictor.set_image(rgb_image)
            
            for bbox in bounding_boxes:
                input_box = np.array(bbox)
                masks_pred, scores, logits = self.predictor.predict(
                    point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False
                )
                masks.append(masks_pred[0].astype(np.uint8))
            
            return masks
            
        except Exception as e:
            logger.error(f"MedSAM2 segmentation failed: {e}")
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
        """Create affine transformation matrix for NIfTI"""
        try:
            image_position = np.array(metadata.get('image_position', [0.0, 0.0, 0.0]), dtype=float)
            image_orientation = np.array(metadata.get('image_orientation', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), dtype=float)
            spacing = metadata.get('spacing', (1.0, 1.0, 1.0))
            
            # DICOM orientation vectors
            row_cosines = image_orientation[:3]
            col_cosines = image_orientation[3:6]
            normal_vector = np.cross(row_cosines, col_cosines)
            normal_vector = normal_vector / np.linalg.norm(normal_vector) if np.linalg.norm(normal_vector) > 0 else normal_vector
            
            # Spacing for NIfTI (x, y, z)
            x_spacing = spacing[2] if len(spacing) > 2 else spacing[1]
            y_spacing = spacing[1]
            z_spacing = spacing[0]
            
            # Build direction matrix
            direction_matrix = np.column_stack([
                col_cosines * x_spacing,
                row_cosines * y_spacing,
                normal_vector * z_spacing
            ])
            
            # Create 4x4 affine matrix
            affine = np.eye(4)
            affine[:3, :3] = direction_matrix
            affine[:3, 3] = image_position
            
            # Coordinate transformation for 3D Slicer compatibility
            coord_transform = np.diag([-1, -1, 1, 1])
            affine = coord_transform @ affine
            
            return affine
            
        except Exception as e:
            logger.error(f"Failed to create affine matrix: {e}")
            return np.eye(4)
    
    def save_masks_as_nifti(self, masks_3d: np.ndarray, output_path: str, metadata: Dict) -> None:
        """Save 3D masks as NIfTI"""
        try:
            # Transpose from (z,y,x) to (x,y,z) for NIfTI
            if len(masks_3d.shape) == 3:
                masks_3d = masks_3d.transpose(2, 1, 0)
            
            affine = self.create_dicom_to_nifti_affine(metadata)
            nifti_img = nib.Nifti1Image(masks_3d.astype(np.uint8), affine)
            nifti_img.header.set_xyzt_units('mm', 'sec')
            nib.save(nifti_img, output_path)
            logger.info(f"Saved NIfTI: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save NIfTI {output_path}: {e}")
    
    def create_3d_mask_volume(self, slice_results: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """Create 3D volume from slice masks"""
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
        
        # Check if all masks have the same dimensions
        first_mask = None
        for result in sorted_results:
            if result['masks']:
                first_mask = result['masks'][0]
                break
        
        if first_mask is None:
            logger.warning("No valid masks found in slice results")
            return np.array([]), {}
        
        height, width = first_mask.shape
        depth = len(sorted_results)
        volume_3d = np.zeros((depth, height, width), dtype=np.uint8)
        
        for i, result in enumerate(sorted_results):
            if result['masks']:
                combined_mask = np.logical_or.reduce(result['masks']) if len(result['masks']) > 1 else result['masks'][0]
                
                # Check if mask dimensions match
                if combined_mask.shape != (height, width):
                    logger.warning(f"Mask dimension mismatch at slice {i}: expected {height}x{width}, got {combined_mask.shape}")
                    combined_mask = cv2.resize(combined_mask.astype(np.uint8), (width, height))
                
                volume_3d[i] = combined_mask.astype(np.uint8)
        
        # Create metadata
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
        }
        
        return volume_3d, volume_metadata
    
    def create_reference_nifti(self, patient_id: str) -> Optional[str]:
        """Create reference NIfTI file from DICOM images"""
        logger.info(f"Creating reference NIfTI for patient {patient_id}")
        
        patient_data = self.load_patient_data(patient_id)
        if not patient_data or not patient_data['dicom_files']:
            return None
        
        # Load all DICOM files
        slice_data = []
        for dicom_path in patient_data['dicom_files']:
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
        
        # Check if all images have the same dimensions
        first_image = sorted_slices[0]['image']
        height, width = first_image.shape[:2]
        
        # Verify all images have consistent dimensions
        for i, slice_info in enumerate(sorted_slices):
            img_h, img_w = slice_info['image'].shape[:2]
            if img_h != height or img_w != width:
                logger.warning(f"Inconsistent image dimensions detected:")
                logger.warning(f"  Expected: {height}x{width}, Got: {img_h}x{img_w} at slice {i}")
                logger.warning(f"  Resizing slice {i} to match first slice dimensions")
                
                # Resize the image to match the first slice
                if len(slice_info['image'].shape) == 3:
                    resized_image = cv2.resize(slice_info['image'], (width, height))
                else:
                    resized_image = cv2.resize(slice_info['image'], (width, height))
                slice_info['image'] = resized_image
        
        # Create 3D volume
        depth = len(sorted_slices)
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
        pixel_spacing = first_metadata.get('pixel_spacing', [1.0, 1.0])
        z_spacing = 1.0
        
        if len(sorted_slices) > 1:
            first_loc = first_metadata.get('slice_location')
            last_loc = sorted_slices[-1]['metadata'].get('slice_location')
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
            volume_3d_transposed = volume_3d.transpose(2, 1, 0)
            affine = self.create_dicom_to_nifti_affine(reference_metadata)
            nifti_img = nib.Nifti1Image(volume_3d_transposed, affine)
            nib.save(nifti_img, str(reference_path))
            logger.info(f"Reference NIfTI saved: {reference_path}")
            return str(reference_path)
        except Exception as e:
            logger.error(f"Failed to save reference NIfTI: {e}")
            return None
    
    def process_patient(self, patient_id: str, save_results: bool = True, create_reference: bool = True) -> Dict:
        """Process a patient with MedSAM2 segmentation"""
        logger.info(f"Processing patient: {patient_id}")
        
        # Load patient data
        patient_data = self.load_patient_data(patient_id)
        if not patient_data or not patient_data['dicom_files']:
            return {'status': 'error', 'message': 'No patient data found'}
        
        # Create reference NIfTI if requested
        reference_path = None
        if create_reference and save_results:
            logger.info("Creating DICOM reference NIfTI...")
            reference_path = self.create_reference_nifti(patient_id)
            if reference_path:
                logger.info(f"Reference NIfTI created: {Path(reference_path).name}")
        
        # Process annotated slices
        slice_results = []
        for dicom_path in patient_data['dicom_files']:
            xml_path = self.find_matching_annotation(dicom_path, patient_data['xml_files'])
            if not xml_path:
                continue
            
            image, metadata = self.load_dicom_image(dicom_path)
            annotations = self.parse_xml_annotation(xml_path)
            
            if image is None or not annotations:
                continue
            
            # Perform segmentation
            bounding_boxes = [ann['bbox'] for ann in annotations]
            masks = self.segment_with_medsam2(image, bounding_boxes)
            
            slice_results.append({
                'dicom_file': dicom_path.name,
                'xml_file': xml_path.name,
                'metadata': metadata,
                'annotations': annotations,
                'masks': masks
            })
        
        if not slice_results:
            return {'status': 'error', 'message': 'No annotated slices found'}
        
        # Create and save 3D volume
        volume_3d, volume_metadata = self.create_3d_mask_volume(slice_results)
        nifti_path = None
        
        if save_results and volume_3d.size > 0:
            output_dir = self.segmentation_result_base / patient_id
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nifti_path = output_dir / f"segmentation_{timestamp}.nii.gz"
            self.save_masks_as_nifti(volume_3d, str(nifti_path), volume_metadata)
        
        return {
            'status': 'success',
            'patient_id': patient_id,
            'processed_slices': len(slice_results),
            'volume_shape': volume_3d.shape if volume_3d.size > 0 else None,
            'nifti_path': str(nifti_path) if nifti_path else None,
            'reference_path': reference_path
        }
    
    def process_all_patients(self, save_results: bool = True, create_reference: bool = True) -> Dict:
        """Process all available patients"""
        patients = self.get_patient_list()
        if not patients:
            return {'status': 'error', 'message': 'No patients found'}
        
        logger.info(f"Processing {len(patients)} patients: {', '.join(patients)}")
        
        results = {}
        for patient_id in patients:
            try:
                result = self.process_patient(patient_id, save_results=save_results, create_reference=create_reference)
                results[patient_id] = result
                if result['status'] == 'success':
                    logger.info(f"✓ {patient_id}: {result['processed_slices']} slices processed")
                    if result.get('reference_path'):
                        logger.info(f"  Reference created: {Path(result['reference_path']).name}")
                else:
                    logger.warning(f"✗ {patient_id}: {result.get('message', 'Failed')}")
            except Exception as e:
                logger.error(f"✗ {patient_id}: Error - {e}")
                results[patient_id] = {'status': 'error', 'message': str(e)}
        
        successful = sum(1 for r in results.values() if r['status'] == 'success')
        logger.info(f"Processing completed: {successful}/{len(patients)} patients successful")
        
        return {
            'status': 'success',
            'total_patients': len(patients),
            'successful_patients': successful,
            'results': results
        }
    

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MedSAM2 Segmentation for Chest Tumor")
    parser.add_argument("--patient_id", type=str, help="Patient ID to process (if not specified, process all patients)")
    parser.add_argument("--data_dir", type=str, default="all_patient_data", help="Patient data directory")
    parser.add_argument("--config", type=str, default="sam2.1_hiera_t512.yaml", help="MedSAM2 config file")
    parser.add_argument("--list_patients", action="store_true", help="List available patients")
    parser.add_argument("--create_reference_only", action="store_true", help="Only create reference NIfTI from DICOM (no segmentation)")
    parser.add_argument("--no_reference", action="store_true", help="Skip creating reference NIfTI")
    
    args = parser.parse_args()
    
    segmentator = MedSAMSegmentator(data_dir=args.data_dir, config_file=args.config)
    
    if args.list_patients:
        patients = segmentator.get_patient_list()
        print(f"Available patients ({len(patients)}): {', '.join(patients)}")
        return
    
    if args.create_reference_only:
        # Only create reference NIfTI
        if args.patient_id:
            reference_path = segmentator.create_reference_nifti(args.patient_id)
            if reference_path:
                print(f"Reference NIfTI created for {args.patient_id}: {reference_path}")
            else:
                print(f"Failed to create reference NIfTI for {args.patient_id}")
        else:
            # Create reference for all patients
            patients = segmentator.get_patient_list()
            for patient_id in patients:
                reference_path = segmentator.create_reference_nifti(patient_id)
                if reference_path:
                    print(f"✓ {patient_id}: Reference created - {Path(reference_path).name}")
                else:
                    print(f"✗ {patient_id}: Failed to create reference")
        return
    
    create_ref = not args.no_reference
    
    if args.patient_id:
        # Process specific patient
        results = segmentator.process_patient(args.patient_id, create_reference=create_ref)
        if results['status'] == 'success':
            print(f"Patient {results['patient_id']} processed successfully")
            print(f"Processed {results['processed_slices']} slices")
            if results.get('reference_path'):
                print(f"Reference DICOM saved: {results['reference_path']}")
            if results.get('nifti_path'):
                print(f"Segmentation saved: {results['nifti_path']}")
        else:
            print(f"Processing failed: {results.get('message', 'Unknown error')}")
    else:
        # Process all patients
        results = segmentator.process_all_patients(create_reference=create_ref)
        if results['status'] == 'success':
            print(f"Processing completed: {results['successful_patients']}/{results['total_patients']} patients successful")
            
            # Show summary of failed patients
            failed_patients = [pid for pid, result in results['results'].items() if result['status'] != 'success']
            if failed_patients:
                print(f"Failed patients: {', '.join(failed_patients)}")
        else:
            print(f"Processing failed: {results.get('message', 'Unknown error')}")


if __name__ == "__main__":
    main()
