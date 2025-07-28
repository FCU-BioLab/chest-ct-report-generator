#!/usr/bin/env python3
"""
MedSAM Demo for Chest Tumor Segmentation
========================================

This script demonstrates the use of MedSAM (Medical Segment Anything Model) 
for segmenting chest tumors from CT images in the all_patient_data directory.

Author: GitHub Copilot
Date: July 28, 2025
"""

import sys
import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import pydicom
from typing import List, Tuple, Dict, Optional
import argparse
import logging
from datetime import datetime
import nibabel as nib

# Load configuration
def load_config(config_path: str = None) -> Dict:
    """Load configuration from JSON file"""
    if config_path is None:
        # Try to find config.json in parent directory
        current_dir = Path(__file__).parent
        config_path = current_dir.parent / "config.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return {}

# Configure logging
def setup_logging(config_path: str = None):
    """Setup logging with log file in segmentation_result directory"""
    # Load config to get the correct directory
    config = load_config(config_path)
    
    # Get log directory from config or use default
    if config.get('data', {}).get('segmentation_result_dir'):
        log_dir = Path(config['data']['segmentation_result_dir'])
    else:
        log_dir = Path("segmentation_result")
    
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "medsam_demo.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(sys.stdout)
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from transformers import SamModel, SamProcessor
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch and transformers not available. Running in demo mode without actual inference.")
    TORCH_AVAILABLE = False

class MedSAMDemo:
    """
    MedSAM Demo class for chest tumor segmentation
    """
    
    def __init__(self, data_dir: str = "all_patient_data", model_name: str = "facebook/sam-vit-huge", 
                 config_path: str = None):
        """Initialize MedSAM Demo"""
        self.config = load_config(config_path)
        
        # Set directories from config or defaults
        self.data_dir = Path(self.config.get('data', {}).get('all_patient_data_dir', data_dir))
        self.segmentation_result_base = Path(self.config.get('data', {}).get('segmentation_result_dir', "segmentation_result"))
        
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if TORCH_AVAILABLE else None
        
        if TORCH_AVAILABLE:
            self._load_model()
        
        logger.info(f"MedSAM Demo initialized - Data: {self.data_dir}, Device: {self.device}")
    
    def _create_output_directory(self, patient_id: str) -> Path:
        """Create output directory for patient results"""
        output_dir = self.segmentation_result_base / patient_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _load_model(self):
        """Load SAM model and processor"""
        try:
            logger.info(f"Loading SAM model: {self.model_name}")
            self.model = SamModel.from_pretrained(self.model_name)
            self.processor = SamProcessor.from_pretrained(self.model_name)
            
            if self.device:
                self.model.to(self.device)
            
            logger.info("SAM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            self.model = None
            self.processor = None
    
    def get_patient_list(self) -> List[str]:
        """Get list of available patients"""
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            return []
        
        patients = [d.name for d in self.data_dir.iterdir() 
                   if d.is_dir()]
        patients.sort()
        logger.info(f"Found {len(patients)} patients")
        return patients
    
    def load_patient_data(self, patient_id: str) -> Dict:
        """
        Load patient data including DICOM files and XML annotations
        
        Args:
            patient_id: Patient ID (e.g., 'A0001')
            
        Returns:
            Dictionary containing patient data
        """
        patient_dir = self.data_dir / patient_id
        
        if not patient_dir.exists():
            logger.error(f"Patient directory not found: {patient_dir}")
            return {}
        
        # Load file list
        file_list_path = patient_dir / f"{patient_id}_file_list.json"
        file_list = {}
        if file_list_path.exists():
            with open(file_list_path, 'r') as f:
                file_list = json.load(f)
        
        # Get DICOM files
        dicom_dir = patient_dir / "dicom_files"
        dicom_files = []
        if dicom_dir.exists():
            dicom_files = list(dicom_dir.glob("*.dcm"))
        
        # Get XML annotations
        xml_dir = patient_dir / "xml_annotations"
        xml_files = []
        if xml_dir.exists():
            xml_files = list(xml_dir.glob("*.xml"))
        
        patient_data = {
            'patient_id': patient_id,
            'patient_dir': patient_dir,
            'file_list': file_list,
            'dicom_files': dicom_files,
            'xml_files': xml_files,
            'dicom_count': len(dicom_files),
            'annotation_count': len(xml_files)
        }
        
        logger.info(f"Loaded patient {patient_id}: {len(dicom_files)} DICOM files, {len(xml_files)} annotations")
        return patient_data
    
    def load_dicom_image(self, dicom_path: Path) -> Tuple[np.ndarray, Dict]:
        """Load DICOM image and metadata"""
        try:
            dicom_data = pydicom.dcmread(str(dicom_path))
            image = dicom_data.pixel_array.astype(np.float32)
            
            # Apply windowing or default normalization
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                center = float(dicom_data.WindowCenter[0] if hasattr(dicom_data.WindowCenter, '__iter__') else dicom_data.WindowCenter)
                width = float(dicom_data.WindowWidth[0] if hasattr(dicom_data.WindowWidth, '__iter__') else dicom_data.WindowWidth)
                
                img_min, img_max = center - width // 2, center + width // 2
                image = np.clip(image, img_min, img_max)
                image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Extract essential metadata
            def safe_get(attr, default=None):
                value = getattr(dicom_data, attr, default)
                return value[0] if hasattr(value, '__iter__') and not isinstance(value, str) else value
            
            metadata = {
                'instance_uid': str(dicom_data.SOPInstanceUID),
                'slice_location': safe_get('SliceLocation'),
                'pixel_spacing': safe_get('PixelSpacing'),
                'slice_thickness': safe_get('SliceThickness', 1.0)
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
            
            # Get image size or use default
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
        """Perform segmentation using MedSAM with bounding box prompts"""
        if not TORCH_AVAILABLE or self.model is None:
            return self._generate_mock_masks(image, bounding_boxes)
        
        try:
            masks = []
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
        """Generate mock segmentation masks for demo purposes"""
        h, w = image.shape[:2]
        masks = []
        
        for x1, y1, x2, y2 in bounding_boxes:
            mask = np.zeros((h, w), dtype=np.uint8)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            radius = ((x2 - x1) // 2, (y2 - y1) // 2)
            cv2.ellipse(mask, center, radius, 0, 0, 360, 1, -1)
            masks.append(mask)
        
        return masks
    
    def calculate_tumor_features(self, mask: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                                slice_location: float = 0.0) -> Dict:
        """Calculate comprehensive tumor features from segmentation mask"""
        if mask.sum() == 0:
            return {'valid': False, 'error': 'Empty mask'}
        
        try:
            # Get the largest contour
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return {'valid': False, 'error': 'No contours found'}
            
            contour = max(contours, key=cv2.contourArea)
            area_pixels = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Basic measurements
            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            cx = M['m10'] / M['m00'] if M['m00'] != 0 else x + w // 2
            cy = M['m01'] / M['m00'] if M['m00'] != 0 else y + h // 2
            
            # Calculate key metrics
            area_mm2 = area_pixels * spacing[1] * spacing[2]
            volume_mm3 = area_mm2 * spacing[0]
            circularity = 4 * np.pi * area_pixels / (perimeter ** 2) if perimeter > 0 else 0
            
            # Convexity analysis
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area_pixels / hull_area if hull_area > 0 else 0
            
            # Ellipse fitting for axis measurements
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (_, _), (minor, major), _ = ellipse
                major_mm = major * np.sqrt(spacing[1] * spacing[2])
                minor_mm = minor * np.sqrt(spacing[1] * spacing[2])
                aspect_ratio = major_mm / minor_mm if minor_mm > 0 else 1.0
            else:
                equiv_diameter = np.sqrt(4 * area_pixels / np.pi)
                major_mm = minor_mm = equiv_diameter * np.sqrt(spacing[1] * spacing[2])
                aspect_ratio = 1.0
            
            return {
                'valid': True,
                'area_mm2': float(area_mm2),
                'volume_mm3': float(volume_mm3),
                'centroid_mm': (slice_location, cy * spacing[1], cx * spacing[2]),
                'major_axis_mm': float(major_mm),
                'minor_axis_mm': float(minor_mm),
                'aspect_ratio': float(aspect_ratio),
                'circularity': float(circularity),
                'convexity': float(convexity),
                'irregularity': float(1.0 - circularity)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def save_masks_as_nifti(self, masks_3d: np.ndarray, output_path: str, 
                           spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
        """Save 3D segmentation masks as NIfTI format"""
        try:
            affine = np.diag([spacing[2], spacing[1], spacing[0], 1])
            nifti_img = nib.Nifti1Image(masks_3d.astype(np.uint8), affine)
            nib.save(nifti_img, output_path)
            logger.info(f"3D masks saved as NIfTI: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save NIfTI: {e}")
    
    def create_3d_mask_volume(self, slice_results: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """
        Create 3D volume from individual slice masks
        
        Args:
            slice_results: List of slice processing results
            
        Returns:
            Tuple of (3D mask array, metadata)
        """
        if not slice_results:
            return np.array([]), {}
        
        # Sort by slice location if available
        sorted_results = sorted(slice_results, 
                               key=lambda x: x.get('metadata', {}).get('slice_location', x['slice_index']))
        
        # Get dimensions from first slice
        first_mask = sorted_results[0]['masks'][0] if sorted_results[0]['masks'] else None
        if first_mask is None:
            return np.array([]), {}
        
        height, width = first_mask.shape
        depth = len(sorted_results)
        
        # Initialize 3D volume
        volume_3d = np.zeros((depth, height, width), dtype=np.uint8)
        
        # Fill volume with masks
        for i, result in enumerate(sorted_results):
            if result['masks']:
                # Combine multiple masks per slice if present
                combined_mask = np.zeros((height, width), dtype=np.uint8)
                for mask in result['masks']:
                    combined_mask = np.maximum(combined_mask, mask)
                volume_3d[i] = combined_mask
        
        # Extract spacing information
        spacing = (1.0, 1.0, 1.0)  # Default
        if sorted_results[0]['metadata'].get('pixel_spacing'):
            pixel_spacing = sorted_results[0]['metadata']['pixel_spacing']
            if isinstance(pixel_spacing, list) and len(pixel_spacing) >= 2:
                spacing = (
                    sorted_results[0]['metadata'].get('slice_thickness', 1.0),
                    float(pixel_spacing[0]),
                    float(pixel_spacing[1])
                )
        
        metadata = {
            'shape': volume_3d.shape,
            'spacing': spacing,
            'slice_count': depth,
            'total_voxels': int(volume_3d.sum()),
            'volume_mm3': float(volume_3d.sum() * spacing[0] * spacing[1] * spacing[2])
        }
        
        return volume_3d, metadata
    
    def export_features_to_json(self, results: Dict, output_path: str) -> None:
        """Export all calculated features to JSON file"""
        try:
            export_data = {
                'patient_id': results.get('patient_id'),
                'timestamp': datetime.now().isoformat(),
                'summary': results.get('feature_summary', {}),
                'slice_features': [
                    {
                        'slice_index': sr['slice_index'],
                        'dicom_file': sr['dicom_file'],
                        'features': sr.get('features', [])
                    }
                    for sr in results.get('slice_results', [])
                ]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Features exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export features: {e}")
    
    def visualize_results(self, image: np.ndarray, annotations: List[Dict], 
                         masks: List[np.ndarray], save_path: Optional[str] = None) -> None:
        """
        Visualize segmentation results and save to file
        
        Args:
            image: Original image
            annotations: List of annotations with bounding boxes
            masks: List of segmentation masks
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Image with bounding boxes
        img_with_boxes = image.copy()
        for ann in annotations:
            x1, y1, x2, y2 = ann['bbox']
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_with_boxes, ann['name'], (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        axes[1].imshow(img_with_boxes)
        axes[1].set_title('Bounding Box Annotations')
        axes[1].axis('off')
        
        # Segmentation results
        result_img = image.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            # Apply colored mask
            colored_mask = np.zeros_like(result_img)
            colored_mask[mask > 0] = color
            result_img = cv2.addWeighted(result_img, 0.7, colored_mask, 0.3, 0)
            
            # Draw contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result_img, contours, -1, color, 2)
        
        axes[2].imshow(result_img)
        axes[2].set_title('MedSAM Segmentation Results')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            # Save to file without showing
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to: {save_path}")
        
        # Close the figure to free memory and prevent display
        plt.close(fig)
    
    def demo_single_case(self, patient_id: str, slice_index: int = None, 
                        save_results: bool = True, process_all_slices: bool = True) -> Dict:
        """Demo MedSAM on a patient case"""
        logger.info(f"Starting MedSAM demo for patient: {patient_id}")
        
        patient_data = self.load_patient_data(patient_id)
        if not patient_data or not patient_data['dicom_files'] or not patient_data['xml_files']:
            logger.error(f"Invalid patient data for {patient_id}")
            return {}
        
        # Process specific slice or all slices
        if slice_index is not None:
            dicom_path = patient_data['dicom_files'][min(slice_index, len(patient_data['dicom_files'])-1)]
            annotation_path = self.find_matching_annotation(dicom_path, patient_data['xml_files'])
            if annotation_path:
                return self._process_slice(dicom_path, annotation_path, patient_id, slice_index, save_results)
            else:
                return {'patient_id': patient_id, 'status': 'no_annotations'}
        
        # Process all slices or find first annotated slice
        if process_all_slices:
            return self._process_all_annotated_slices(patient_data, save_results)
        else:
            return self._process_first_annotated_slice(patient_data, save_results)
    
    def _process_slice(self, dicom_path: Path, annotation_path: Path, patient_id: str, 
                      slice_index: int, save_results: bool) -> Dict:
        """Process a single slice with DICOM and annotation"""
        # Load DICOM image
        image, metadata = self.load_dicom_image(dicom_path)
        if image is None:
            return {'status': 'failed', 'error': 'Failed to load DICOM'}
        
        # Parse annotations
        annotations = self.parse_xml_annotation(annotation_path)
        if not annotations:
            return {'status': 'no_annotations'}
        
        # Perform segmentation
        bounding_boxes = [ann['bbox'] for ann in annotations]
        masks = self.segment_with_medsam(image, bounding_boxes)
        
        # Calculate features
        features = []
        for j, mask in enumerate(masks):
            if mask.sum() > 0:
                # Get spacing
                pixel_spacing = metadata.get('pixel_spacing', [1.0, 1.0])
                if isinstance(pixel_spacing, list) and len(pixel_spacing) >= 2:
                    spacing = (metadata.get('slice_thickness', 1.0), float(pixel_spacing[0]), float(pixel_spacing[1]))
                else:
                    spacing = (1.0, 1.0, 1.0)
                
                feature = self.calculate_tumor_features(mask, spacing, float(metadata.get('slice_location', 0.0)))
                feature['annotation_info'] = annotations[j] if j < len(annotations) else {}
                features.append(feature)
        
        # Save visualization
        save_path = None
        if save_results:
            output_dir = self._create_output_directory(patient_id)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = output_dir / f"slice_{slice_index:03d}_{timestamp}.png"
            self.visualize_results(image, annotations, masks, str(save_path))
        
        return {
            'patient_id': patient_id,
            'slice_index': slice_index,
            'dicom_file': dicom_path.name,
            'annotation_file': annotation_path.name,
            'image_shape': image.shape,
            'annotations': annotations,
            'masks': masks,
            'features': features,
            'metadata': metadata,
            'status': 'success',
            'save_path': str(save_path) if save_path else None
        }
    
    def _process_all_annotated_slices(self, patient_data: Dict, save_results: bool) -> Dict:
        """Process all slices that have annotations"""
        dicom_files = patient_data['dicom_files']
        xml_files = patient_data['xml_files']
        patient_id = patient_data['patient_id']
        
        logger.info(f"Processing all annotated slices for patient {patient_id}")
        
        all_results = []
        
        # Process each slice with annotations
        for i, dicom_path in enumerate(dicom_files):
            annotation_path = self.find_matching_annotation(dicom_path, xml_files)
            if annotation_path is None:
                continue
                
            logger.info(f"Processing slice {i}: {dicom_path.name}")
            result = self._process_slice(dicom_path, annotation_path, patient_id, i, save_results)
            
            if result['status'] == 'success':
                all_results.append(result)
        
        if not all_results:
            return {'patient_id': patient_id, 'status': 'no_annotations'}
        
        # Create 3D volume and calculate summary
        volume_3d, volume_metadata = self.create_3d_mask_volume(all_results)
        
        # Save 3D volume as NIfTI
        nifti_path = None
        if save_results and volume_3d.size > 0:
            output_dir = self._create_output_directory(patient_id)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nifti_path = output_dir / f"3d_volume_{timestamp}.nii.gz"
            self.save_masks_as_nifti(volume_3d, str(nifti_path), volume_metadata.get('spacing', (1.0, 1.0, 1.0)))
        
        # Calculate feature summary
        valid_features = [f for result in all_results for f in result.get('features', []) if f.get('valid', False)]
        feature_summary = {
            'total_tumors': len(valid_features),
            'total_volume_mm3': sum(f['volume_mm3'] for f in valid_features),
            'average_area_mm2': np.mean([f['area_mm2'] for f in valid_features]) if valid_features else 0,
            'mean_irregularity': np.mean([f['irregularity'] for f in valid_features]) if valid_features else 0
        }
        
        return {
            'patient_id': patient_id,
            'processed_slices': len(all_results),
            'slice_results': all_results,
            'volume_3d_shape': volume_3d.shape if volume_3d.size > 0 else None,
            'volume_metadata': volume_metadata,
            'feature_summary': feature_summary,
            'nifti_path': str(nifti_path) if nifti_path else None,
            'status': 'success'
        }
    
    def _process_first_annotated_slice(self, patient_data: Dict, save_results: bool) -> Dict:
        """Find and process the first slice with annotations"""
        dicom_files = patient_data['dicom_files']
        xml_files = patient_data['xml_files']
        patient_id = patient_data['patient_id']
        
        for i, dicom_path in enumerate(dicom_files):
            annotation_path = self.find_matching_annotation(dicom_path, xml_files)
            if annotation_path and self.parse_xml_annotation(annotation_path):
                logger.info(f"Found first annotated slice at index {i}")
                return self._process_slice(dicom_path, annotation_path, patient_id, i, save_results)
        
        return {'patient_id': patient_id, 'status': 'no_annotations'}

def main():
    """Main function for running the demo"""
    parser = argparse.ArgumentParser(description="MedSAM Demo for Chest Tumor Segmentation")
    parser.add_argument("--patient_id", type=str, default="A0001")
    parser.add_argument("--slice_index", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default="all_patient_data")
    parser.add_argument("--model_name", type=str, default="facebook/sam-vit-huge")
    parser.add_argument("--list_patients", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--single_slice_only", action="store_true")
    parser.add_argument("--export_features", action="store_true")
    
    args = parser.parse_args()
    demo = MedSAMDemo(data_dir=args.data_dir, model_name=args.model_name)
    
    if args.list_patients:
        patients = demo.get_patient_list()
        print(f"Available patients ({len(patients)}): {', '.join(patients)}")
        return
    
    results = demo.demo_single_case(
        patient_id=args.patient_id,
        slice_index=args.slice_index,
        save_results=not args.no_save,
        process_all_slices=not args.single_slice_only
    )
    
    if not results:
        print("Demo failed to complete")
        return
    
    # Export features if requested
    if args.export_features:
        output_dir = demo._create_output_directory(args.patient_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"tumor_features_{timestamp}.json"
        demo.export_features_to_json(results, str(json_path))
        print(f"Features exported to: {json_path}")
    
    # Print summary
    print(f"Patient {results['patient_id']} - Status: {results['status']}")
    
    if 'slice_results' in results:
        print(f"Processed {results['processed_slices']} slices")
        if results.get('feature_summary'):
            fs = results['feature_summary']
            print(f"Found {fs['total_tumors']} tumors, total volume: {fs['total_volume_mm3']:.1f} mm³")
        if results.get('nifti_path'):
            print(f"3D volume saved: {results['nifti_path']}")
    else:
        if results.get('features'):
            valid_features = [f for f in results['features'] if f.get('valid')]
            print(f"Found {len(valid_features)} valid tumors")
            for i, feature in enumerate(valid_features):
                print(f"  Tumor {i+1}: {feature['area_mm2']:.1f} mm², irregularity: {feature['irregularity']:.3f}")


if __name__ == "__main__":
    main()
