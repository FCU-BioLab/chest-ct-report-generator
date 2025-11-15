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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from skimage import measure

# Model availability check
try:
    import torch
    import sys
    from pathlib import Path
    
    # Add MedSAM2 path to sys.path
    medsam2_path = Path(__file__).parent / "MedSAM2"
    if medsam2_path not in sys.path:
        sys.path.insert(0, str(medsam2_path))
    
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    MEDSAM2_AVAILABLE = True
except ImportError as e:
    print(f"MedSAM2 import error: {e}")
    MEDSAM2_AVAILABLE = False


def setup_logging(log_dir: str = "segmentation_result", append_mode: bool = True) -> logging.Logger:
    """Setup logging configuration"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Use timestamp in log filename to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d")
    log_filename = f"medsam_seg_{timestamp}.log"
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_path / log_filename), mode='a' if append_mode else 'w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


class MedSAMSegmentator:
    """Simplified MedSAM2 segmentation"""
    
    def __init__(self, data_dir: str = "../datasets/all_patient_data", config_file: str = "sam2.1_hiera_t512.yaml", 
                 use_timestamp: bool = True, list_only: bool = False):
        self.data_dir = Path(data_dir)
        
        # Create timestamped result directory (skip if just listing patients)
        if list_only:
            self.segmentation_result_base = Path("segmentation_result")
        elif use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.segmentation_result_base = Path("segmentation_result") / timestamp
        else:
            self.segmentation_result_base = Path("segmentation_result")
        
        self.config_file = config_file
        self.model = None
        self.predictor = None
        self.device = "cuda" if MEDSAM2_AVAILABLE and torch.cuda.is_available() else "cpu" if MEDSAM2_AVAILABLE else None
        
        # Skip model loading if just listing patients
        if not list_only:
            if MEDSAM2_AVAILABLE:
                self._load_medsam2()
            else:
                logger.warning("MedSAM2 unavailable. Running in mock mode.")
            
            # Log the result directory being used
            logger.info(f"Results will be saved to: {self.segmentation_result_base}")
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
        """Generate mock segmentation masks from bounding boxes"""
        h, w = image.shape[:2]
        masks = []
        
        for x1, y1, x2, y2 in bounding_boxes:
            mask = np.zeros((h, w), dtype=np.uint8)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            radius = ((x2 - x1) // 2, (y2 - y1) // 2)
            cv2.ellipse(mask, center, radius, 0, 0, 360, 1, -1)
            masks.append(mask)
        
        return masks
    
    def _prepare_display_image(self, image: np.ndarray) -> np.ndarray:
        """Convert and normalize image for display"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            display_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            display_image = image.copy()
        
        # Normalize for display
        return ((display_image - display_image.min()) / 
               (display_image.max() - display_image.min()) * 255).astype(np.uint8)
    
    def _draw_bounding_boxes(self, ax, annotations: List[Dict], color: str = 'red', 
                            linewidth: int = 2, show_labels: bool = True) -> None:
        """Draw bounding boxes on matplotlib axis"""
        for i, ann in enumerate(annotations):
            bbox = ann['bbox']  # [xmin, ymin, xmax, ymax]
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            rect = patches.Rectangle((x, y), w, h, linewidth=linewidth, 
                                   edgecolor=color, facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            if show_labels:
                label = ann.get('name', f'lesion_{i+1}')
                ax.text(x, y-5, label, color=color, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def _create_mask_overlay(self, display_image: np.ndarray, masks: List[np.ndarray], 
                            color: List[float] = [1, 0, 0, 0.5]) -> np.ndarray:
        """Create mask overlay for visualization"""
        combined_mask = np.zeros_like(display_image, dtype=np.uint8)
        
        for mask in masks:
            if mask.shape[:2] == display_image.shape[:2]:
                combined_mask = np.logical_or(combined_mask, mask > 0)
            else:
                # Resize mask if dimensions don't match
                resized_mask = cv2.resize(mask.astype(np.uint8), 
                                        (display_image.shape[1], display_image.shape[0]))
                combined_mask = np.logical_or(combined_mask, resized_mask > 0)
        
        # Create overlay
        overlay = np.zeros((*display_image.shape, 4))
        overlay[combined_mask, :] = color
        return overlay
    
    def save_visualization_images(self, patient_id: str, image: np.ndarray, annotations: List[Dict], 
                                 masks: List[np.ndarray], dicom_filename: str, slice_index: int) -> None:
        """Save visualization images with bounding boxes and segmentation overlay"""
        try:
            # Create visualization directory
            vis_dir = self.segmentation_result_base / patient_id / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare display image
            display_image = self._prepare_display_image(image)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Original image with bounding boxes
            ax1.imshow(display_image, cmap='gray')
            ax1.set_title(f'Original with Bounding Boxes\n{dicom_filename}', fontsize=12)
            ax1.axis('off')
            self._draw_bounding_boxes(ax1, annotations, color='red', linewidth=2)
            
            # Image with segmentation overlay
            ax2.imshow(display_image, cmap='gray')
            ax2.set_title(f'Segmentation Overlay\n{dicom_filename}', fontsize=12)
            ax2.axis('off')
            
            if masks:
                overlay = self._create_mask_overlay(display_image, masks)
                ax2.imshow(overlay)
                self._draw_bounding_boxes(ax2, annotations, color='yellow', 
                                         linewidth=1, show_labels=False)
            
            plt.tight_layout()
            
            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_filename = f"slice_{slice_index:03d}_{timestamp}.png"
            vis_path = vis_dir / vis_filename
            
            plt.savefig(str(vis_path), dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"Visualization saved: {vis_filename}")
            
        except Exception as e:
            logger.error(f"Failed to save visualization for {dicom_filename}: {e}")
    
    def save_individual_images(self, patient_id: str, image: np.ndarray, annotations: List[Dict], 
                              masks: List[np.ndarray], dicom_filename: str, slice_index: int) -> None:
        """Save individual PNG images for original and segmentation"""
        try:
            # Create individual images directory
            img_dir = self.segmentation_result_base / patient_id / "individual_images"
            img_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare display image
            display_image = self._prepare_display_image(image)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"slice_{slice_index:03d}_{timestamp}"
            
            # Save original image with bounding boxes
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(display_image, cmap='gray')
            ax.set_title(f'Original Image with Annotations\n{dicom_filename}', fontsize=14)
            ax.axis('off')
            self._draw_bounding_boxes(ax, annotations, color='red', linewidth=3)
            
            original_path = img_dir / f"{base_filename}_original.png"
            plt.savefig(str(original_path), dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            # Save segmentation overlay image
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(display_image, cmap='gray')
            ax.set_title(f'Segmentation Result\n{dicom_filename}', fontsize=14)
            ax.axis('off')
            
            if masks:
                # Create overlay
                overlay = self._create_mask_overlay(display_image, masks, [1, 0.2, 0.2, 0.6])
                ax.imshow(overlay)
                
                # Add mask contours
                combined_mask = np.zeros_like(display_image, dtype=np.uint8)
                for mask in masks:
                    if mask.shape[:2] == display_image.shape[:2]:
                        combined_mask = np.logical_or(combined_mask, mask > 0)
                    else:
                        resized_mask = cv2.resize(mask.astype(np.uint8), 
                                                (display_image.shape[1], display_image.shape[0]))
                        combined_mask = np.logical_or(combined_mask, resized_mask > 0)
                
                contours, _ = cv2.findContours(combined_mask.astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour = contour.squeeze()
                    if len(contour.shape) == 2 and contour.shape[0] > 2:
                        ax.plot(contour[:, 0], contour[:, 1], 'yellow', linewidth=2, alpha=0.8)
                
                # Draw bounding boxes
                for ann in annotations:
                    bbox = ann['bbox']
                    x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                           edgecolor='cyan', facecolor='none', alpha=0.7,
                                           linestyle='--')
                    ax.add_patch(rect)
            
            segmentation_path = img_dir / f"{base_filename}_segmentation.png"
            plt.savefig(str(segmentation_path), dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"Individual images saved: {base_filename}_original.png, {base_filename}_segmentation.png")
            
        except Exception as e:
            logger.error(f"Failed to save individual images for {dicom_filename}: {e}")
    
    def create_summary_visualization(self, patient_id: str, slice_results: List[Dict]) -> None:
        """Create a summary visualization showing all processed slices"""
        try:
            if not slice_results:
                return
            
            summary_dir = self.segmentation_result_base / patient_id / "summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            
            # Limit to maximum 16 slices for display
            display_slices = slice_results[:16] if len(slice_results) > 16 else slice_results
            n_slices = len(display_slices)
            
            # Calculate grid dimensions
            cols = min(4, n_slices)
            rows = (n_slices + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            if rows == 1:
                axes = [axes] if cols == 1 else axes
            elif cols == 1:
                axes = [[ax] for ax in axes]
            
            patient_data = self.load_patient_data(patient_id)
            
            for idx, result in enumerate(display_slices):
                row, col = idx // cols, idx % cols
                ax = axes[row][col] if rows > 1 else axes[col]
                
                # Load and display the DICOM image
                dicom_path = next((f for f in patient_data['dicom_files'] 
                                 if f.name == result['dicom_file']), None)
                
                if dicom_path:
                    image, _ = self.load_dicom_image(dicom_path)
                    if image is not None:
                        display_image = self._prepare_display_image(image)
                        ax.imshow(display_image, cmap='gray')
                        
                        # Add segmentation overlay
                        if result['masks']:
                            overlay = self._create_mask_overlay(display_image, result['masks'])
                            ax.imshow(overlay)
                        
                        # Add bounding boxes
                        for ann in result['annotations']:
                            bbox = ann['bbox']
                            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                            rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                                   edgecolor='yellow', facecolor='none')
                            ax.add_patch(rect)
                
                ax.set_title(f"Slice {idx+1}\n{result['dicom_file'][:15]}...", fontsize=8)
                ax.axis('off')
            
            # Hide unused subplots
            for idx in range(n_slices, rows * cols):
                row, col = idx // cols, idx % cols
                ax = axes[row][col] if rows > 1 else axes[col]
                ax.axis('off')
            
            plt.suptitle(f'Patient {patient_id} - Segmentation Summary\n'
                        f'{len(slice_results)} slices processed', fontsize=16)
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = summary_dir / f"segmentation_summary_{timestamp}.png"
            plt.savefig(str(summary_path), dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"Summary visualization saved: segmentation_summary_{timestamp}.png")
            
        except Exception as e:
            logger.error(f"Failed to create summary visualization: {e}")
    
    def extract_lesion_features(self, image: np.ndarray, mask: np.ndarray, 
                               metadata: Dict, annotation: Dict) -> Dict:
        """Extract comprehensive features from a lesion for LLM analysis"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image.copy()
            
            # Get masked region
            masked_region = gray_image * mask
            lesion_pixels = gray_image[mask > 0]
            
            if len(lesion_pixels) == 0:
                return {}
            
            # 1. Morphological features
            labeled_mask = measure.label(mask)
            props = measure.regionprops(labeled_mask, intensity_image=gray_image)[0]
            
            # Area and volume features
            pixel_spacing = metadata.get('pixel_spacing', [1.0, 1.0])
            pixel_area_mm2 = float(pixel_spacing[0]) * float(pixel_spacing[1])
            area_pixels = props.area
            area_mm2 = area_pixels * pixel_area_mm2
            
            # Shape features
            perimeter = props.perimeter
            circularity = (4 * np.pi * area_pixels) / (perimeter ** 2) if perimeter > 0 else 0
            solidity = props.solidity
            eccentricity = props.eccentricity
            
            # Bounding box features
            bbox = annotation.get('bbox', [0, 0, 0, 0])
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            compactness = area_pixels / bbox_area if bbox_area > 0 else 0
            
            # 2. Intensity features
            mean_intensity = float(np.mean(lesion_pixels))
            std_intensity = float(np.std(lesion_pixels))
            min_intensity = float(np.min(lesion_pixels))
            max_intensity = float(np.max(lesion_pixels))
            median_intensity = float(np.median(lesion_pixels))
            
            # Background intensity (region outside lesion but inside bbox)
            bbox_mask = np.zeros_like(mask)
            bbox_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            background_mask = (bbox_mask - mask) > 0
            if np.any(background_mask):
                background_pixels = gray_image[background_mask]
                background_mean = float(np.mean(background_pixels))
                contrast_to_background = mean_intensity - background_mean
            else:
                background_mean = 0
                contrast_to_background = 0
            
            # 3. Texture features (using GLCM approximation)
            # Calculate histogram features
            hist, _ = np.histogram(lesion_pixels, bins=32, range=(0, 256))
            hist = hist / np.sum(hist)  # Normalize
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # Gradient features (edge strength)
            gradient_x = cv2.Sobel(masked_region, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(masked_region, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            edge_strength = float(np.mean(gradient_magnitude[mask > 0]))
            
            # 4. Position features
            centroid = props.centroid
            image_center = (gray_image.shape[0] / 2, gray_image.shape[1] / 2)
            distance_from_center = np.sqrt(
                (centroid[0] - image_center[0])**2 + 
                (centroid[1] - image_center[1])**2
            )
            
            # Relative position
            relative_x = centroid[1] / gray_image.shape[1]  # 0=left, 1=right
            relative_y = centroid[0] / gray_image.shape[0]  # 0=top, 1=bottom
            
            # 5. Spatial metadata
            slice_location = metadata.get('slice_location', 0)
            slice_thickness = metadata.get('slice_thickness', 1.0)
            
            # Compile all features
            features = {
                # Identification
                'lesion_name': annotation.get('name', 'unknown'),
                
                # Morphological features
                'area_pixels': int(area_pixels),
                'area_mm2': float(area_mm2),
                'perimeter_pixels': float(perimeter),
                'circularity': float(circularity),
                'solidity': float(solidity),
                'eccentricity': float(eccentricity),
                'compactness': float(compactness),
                'equivalent_diameter_mm': float(props.equivalent_diameter * pixel_spacing[0]),
                'major_axis_length_mm': float(props.major_axis_length * pixel_spacing[0]),
                'minor_axis_length_mm': float(props.minor_axis_length * pixel_spacing[0]),
                
                # Intensity features
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'min_intensity': float(min_intensity),
                'max_intensity': float(max_intensity),
                'median_intensity': float(median_intensity),
                'intensity_range': float(max_intensity - min_intensity),
                'background_mean_intensity': float(background_mean),
                'contrast_to_background': float(contrast_to_background),
                
                # Texture features
                'entropy': float(entropy),
                'edge_strength': float(edge_strength),
                
                # Position features
                'centroid_x': float(centroid[1]),
                'centroid_y': float(centroid[0]),
                'distance_from_center_pixels': float(distance_from_center),
                'relative_position_x': float(relative_x),
                'relative_position_y': float(relative_y),
                
                # Spatial metadata
                'slice_location': float(slice_location) if slice_location is not None else 0.0,
                'slice_thickness': float(slice_thickness),
                'pixel_spacing_x': float(pixel_spacing[0]),
                'pixel_spacing_y': float(pixel_spacing[1]),
                
                # Bounding box
                'bbox': bbox,
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            return {}
    
    def generate_llm_description(self, features: Dict) -> str:
        """Generate human-readable description for LLM"""
        if not features:
            return "No features available."
        
        description_parts = []
        
        # Lesion identification
        description_parts.append(f"Lesion: {features.get('lesion_name', 'Unknown')}")
        
        # Size description
        area_mm2 = features.get('area_mm2', 0)
        diameter_mm = features.get('equivalent_diameter_mm', 0)
        description_parts.append(
            f"Size: {area_mm2:.1f} mm² (equivalent diameter: {diameter_mm:.1f} mm)"
        )
        
        # Shape description
        circularity = features.get('circularity', 0)
        eccentricity = features.get('eccentricity', 0)
        solidity = features.get('solidity', 0)
        
        if circularity > 0.8:
            shape_desc = "nearly circular"
        elif circularity > 0.6:
            shape_desc = "moderately circular"
        else:
            shape_desc = "irregular"
        
        if eccentricity > 0.9:
            elongation_desc = ", highly elongated"
        elif eccentricity > 0.7:
            elongation_desc = ", somewhat elongated"
        else:
            elongation_desc = ""
        
        description_parts.append(f"Shape: {shape_desc}{elongation_desc} (circularity: {circularity:.2f})")
        
        # Intensity description
        mean_int = features.get('mean_intensity', 0)
        contrast = features.get('contrast_to_background', 0)
        
        if contrast > 50:
            intensity_desc = "high contrast"
        elif contrast > 20:
            intensity_desc = "moderate contrast"
        elif contrast > -20:
            intensity_desc = "similar intensity"
        else:
            intensity_desc = "lower intensity"
        
        description_parts.append(
            f"Intensity: mean {mean_int:.1f}, {intensity_desc} compared to background"
        )
        
        # Texture description
        entropy = features.get('entropy', 0)
        edge_strength = features.get('edge_strength', 0)
        
        if entropy > 4.0:
            texture_desc = "heterogeneous"
        elif entropy > 3.0:
            texture_desc = "moderately heterogeneous"
        else:
            texture_desc = "relatively homogeneous"
        
        if edge_strength > 30:
            edge_desc = "well-defined margins"
        elif edge_strength > 15:
            edge_desc = "moderately defined margins"
        else:
            edge_desc = "poorly defined margins"
        
        description_parts.append(f"Texture: {texture_desc}, {edge_desc}")
        
        # Position description
        rel_x = features.get('relative_position_x', 0.5)
        rel_y = features.get('relative_position_y', 0.5)
        
        if rel_x < 0.33:
            horizontal_pos = "left"
        elif rel_x > 0.67:
            horizontal_pos = "right"
        else:
            horizontal_pos = "central"
        
        if rel_y < 0.33:
            vertical_pos = "upper"
        elif rel_y > 0.67:
            vertical_pos = "lower"
        else:
            vertical_pos = "middle"
        
        slice_loc = features.get('slice_location', 0)
        description_parts.append(
            f"Location: {vertical_pos}-{horizontal_pos} region (slice location: {slice_loc:.1f} mm)"
        )
        
        return "\n".join(description_parts)
    
    def save_features_for_llm(self, patient_id: str, slice_results: List[Dict]) -> None:
        """Save extracted features in JSON format for LLM consumption"""
        try:
            output_dir = self.segmentation_result_base / patient_id / "llm_features"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            all_features = {
                'patient_id': patient_id,
                'total_slices': len(slice_results),
                'slices': []
            }
            
            for slice_result in slice_results:
                slice_features = {
                    'dicom_file': slice_result['dicom_file'],
                    'xml_file': slice_result['xml_file'],
                    'lesions': []
                }
                
                # Extract features for each lesion in this slice
                for i, (annotation, mask) in enumerate(zip(slice_result['annotations'], slice_result['masks'])):
                    # Need to reload image for feature extraction
                    patient_data = self.load_patient_data(patient_id)
                    dicom_path = next((f for f in patient_data['dicom_files'] 
                                     if f.name == slice_result['dicom_file']), None)
                    
                    if dicom_path:
                        image, _ = self.load_dicom_image(dicom_path)
                        if image is not None:
                            features = self.extract_lesion_features(
                                image, mask, slice_result['metadata'], annotation
                            )
                            if features:
                                features['description'] = self.generate_llm_description(features)
                                slice_features['lesions'].append(features)
                
                if slice_features['lesions']:
                    all_features['slices'].append(slice_features)
            
            # Save as JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = output_dir / f"features_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_features, f, indent=2, ensure_ascii=False)
            
            logger.info(f"LLM features saved: {json_path.name}")
            
            # Also save a summary text file
            summary_path = output_dir / f"features_summary_{timestamp}.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"Patient ID: {patient_id}\n")
                f.write(f"Total Slices Analyzed: {len(all_features['slices'])}\n")
                f.write(f"{'='*80}\n\n")
                
                for slice_data in all_features['slices']:
                    f.write(f"Slice: {slice_data['dicom_file']}\n")
                    f.write(f"{'-'*80}\n")
                    
                    for lesion in slice_data['lesions']:
                        f.write(f"\n{lesion['description']}\n\n")
                    
                    f.write(f"{'='*80}\n\n")
            
            logger.info(f"LLM features summary saved: {summary_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to save LLM features: {e}")
    
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
    
    def _sort_slices_by_position(self, slice_data: List[Dict], key_name: str = 'metadata') -> List[Dict]:
        """Sort slices by spatial position"""
        def sort_key(item):
            metadata = item[key_name]
            if metadata.get('slice_location') is not None:
                return float(metadata['slice_location'])
            elif metadata.get('image_position') and len(metadata['image_position']) >= 3:
                return float(metadata['image_position'][2])
            else:
                return float(metadata.get('instance_number', 0))
        
        return sorted(slice_data, key=sort_key)
    
    def _resize_image_if_needed(self, image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize image to target shape if dimensions don't match"""
        if image.shape[:2] != target_shape:
            return cv2.resize(image, (target_shape[1], target_shape[0]))
        return image
    
    def create_3d_mask_volume(self, slice_results: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """Create 3D volume from slice masks"""
        if not slice_results:
            return np.array([]), {}
        
        # Sort slices by spatial position
        sorted_results = self._sort_slices_by_position(slice_results, 'metadata')
        
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
                
                # Resize if needed
                if combined_mask.shape != (height, width):
                    logger.warning(f"Mask dimension mismatch at slice {i}: expected {height}x{width}, got {combined_mask.shape}")
                    combined_mask = self._resize_image_if_needed(combined_mask.astype(np.uint8), (height, width))
                
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
        sorted_slices = self._sort_slices_by_position(slice_data, 'metadata')
        
        # Check and resize images if needed
        first_image = sorted_slices[0]['image']
        height, width = first_image.shape[:2]
        
        for i, slice_info in enumerate(sorted_slices):
            img_h, img_w = slice_info['image'].shape[:2]
            if img_h != height or img_w != width:
                logger.warning(f"Inconsistent image dimensions at slice {i}: expected {height}x{width}, got {img_h}x{img_w}")
                logger.warning(f"  Resizing slice {i} to match first slice dimensions")
                slice_info['image'] = self._resize_image_if_needed(slice_info['image'], (height, width))
        
        # Create 3D volume
        depth = len(sorted_slices)
        volume_3d = np.zeros((depth, height, width), dtype=np.uint16)
        
        for i, slice_info in enumerate(sorted_slices):
            image = slice_info['image']
            # Convert to grayscale
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
        for i, dicom_path in enumerate(patient_data['dicom_files']):
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

            # Save visualization images
            if save_results:
                try:
                    # Save combined visualization (side-by-side comparison)
                    self.save_visualization_images(
                        patient_id=patient_id,
                        image=image,
                        annotations=annotations,
                        masks=masks,
                        dicom_filename=dicom_path.name,
                        slice_index=i
                    )
                    
                    # Save individual images (original and segmentation separately)
                    self.save_individual_images(
                        patient_id=patient_id,
                        image=image,
                        annotations=annotations,
                        masks=masks,
                        dicom_filename=dicom_path.name,
                        slice_index=i
                    )
                except Exception as e:
                    logger.warning(f"Failed to save visualization for {dicom_path.name}: {e}")
        
        if not slice_results:
            return {'status': 'error', 'message': 'No annotated slices found'}

        # Create summary visualization
        if save_results:
            try:
                self.create_summary_visualization(patient_id, slice_results)
            except Exception as e:
                logger.warning(f"Failed to create summary visualization: {e}")
        
        # Extract and save features for LLM
        if save_results:
            try:
                self.save_features_for_llm(patient_id, slice_results)
            except Exception as e:
                logger.warning(f"Failed to save LLM features: {e}")

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
    parser.add_argument("--data_dir", type=str, default="../datasets/all_patient_data", help="Patient data directory")
    parser.add_argument("--config", type=str, default="sam2.1_hiera_t512.yaml", help="MedSAM2 config file")
    parser.add_argument("--list_patients", action="store_true", help="List available patients")
    parser.add_argument("--create_reference_only", action="store_true", help="Only create reference NIfTI from DICOM (no segmentation)")
    parser.add_argument("--no_reference", action="store_true", help="Skip creating reference NIfTI")
    parser.add_argument("--no_timestamp", action="store_true", help="Don't use timestamp in result directory")
    
    args = parser.parse_args()
    
    # Determine if we're just listing patients
    list_only = args.list_patients
    
    segmentator = MedSAMSegmentator(data_dir=args.data_dir, config_file=args.config, 
                                   use_timestamp=not args.no_timestamp, list_only=list_only)
    
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
