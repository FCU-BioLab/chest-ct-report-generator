#!/usr/bin/env python3
"""
MedSAM2 Segmentation for Chest Tumor (NIfTI Version)
====================================================

智能胸部CT腫瘤分割系統 - 支援3D NIfTI格式
- 自動3D→2D切片轉換
- 可選MedSAM2精修分割
- 自動特徵提取與視覺化
- LLM報告生成支援

主要功能:
1. 資料載入: 載入NIfTI格式CT和腫瘤遮罩
2. 切片處理: 自動將3D體積切成2D切片
3. 分割精修: 使用MedSAM2精修腫瘤邊界（可選）
4. 特徵提取: 提取形態、強度、紋理等特徵
5. 視覺化: 生成多種可視化圖片
6. 報告生成: 生成LLM可用的結構化特徵描述

使用範例:
    # 分析資料集統計
    python sam_seg.py --analyze_only
    
    # 處理單個患者
    python sam_seg.py --patient_id 100012
    
    # 批量處理（不使用MedSAM2）
    python sam_seg.py --no_medsam --max_patients 10
    
    # 處理所有患者
    python sam_seg.py
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime

import numpy as np
import cv2
import nibabel as nib
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
    """
    胸部CT腫瘤分割器 - 支援NIfTI格式
    
    主要職責:
    - 載入和處理3D NIfTI資料
    - 執行2D切片分割（可選MedSAM2精修）
    - 提取腫瘤特徵用於報告生成
    - 生成視覺化結果
    
    屬性:
        data_dir: 患者資料目錄
        segmentation_result_base: 結果輸出目錄
        model: MedSAM2模型（如果可用）
        predictor: MedSAM2預測器
        device: 計算設備 (cuda/cpu)
    """
    
    def __init__(self, data_dir: str = "../datasets/all_patient_data", 
                 config_file: str = "sam2.1_hiera_t512.yaml", 
                 use_timestamp: bool = True, 
                 list_only: bool = False):
        """
        初始化分割器
        
        Args:
            data_dir: 患者資料目錄路徑
            config_file: MedSAM2配置檔案
            use_timestamp: 是否在結果目錄使用時間戳
            list_only: 僅列出患者（不載入模型）
        """
        self.data_dir = Path(data_dir)
        
        # 設定結果輸出目錄
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
        
        # 僅在需要時載入模型
        if not list_only:
            if MEDSAM2_AVAILABLE:
                self._load_medsam2()
            else:
                logger.warning("⚠️ MedSAM2 不可用，將使用Ground Truth遮罩模式")
            
            logger.info(f"📁 結果將保存至: {self.segmentation_result_base}")
            logger.info(f"🖥️ 計算設備: {self.device if self.device else 'N/A'}")
    
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
        """Load patient NIfTI files (CT and tumor mask)"""
        patient_dir = self.data_dir / patient_id
        if not patient_dir.exists():
            logger.error(f"Patient directory not found: {patient_dir}")
            return {}
        
        # Find NIfTI files
        ct_file = patient_dir / f"{patient_id}_CT.nii.gz"
        tumor_file = patient_dir / f"{patient_id}_tumor.nii.gz"
        
        if not ct_file.exists():
            logger.warning(f"No CT file found for patient {patient_id}")
            return {}
        
        if not tumor_file.exists():
            logger.warning(f"No tumor file found for patient {patient_id}")
            return {}
        
        return {
            'patient_id': patient_id,
            'ct_file': ct_file,
            'tumor_file': tumor_file,
            'has_data': True
        }
    
    def load_nifti_volume(self, nifti_path: Path) -> Tuple[np.ndarray, Dict]:
        """Load NIfTI volume and extract metadata"""
        try:
            nifti_img = nib.load(str(nifti_path))
            volume = nifti_img.get_fdata()
            
            # Extract metadata from NIfTI header
            header = nifti_img.header
            affine = nifti_img.affine
            
            # Get voxel spacing (in mm)
            spacing = header.get_zooms()[:3]  # (z, y, x) or (x, y, z) depending on orientation
            
            metadata = {
                'shape': volume.shape,
                'spacing': spacing,
                'affine': affine,
                'header': header
            }
            
            return volume, metadata
            
        except Exception as e:
            logger.error(f"Failed to load NIfTI {nifti_path}: {e}")
            return None, {}
    
    def extract_bboxes_from_mask(self, mask_slice: np.ndarray) -> List[List[int]]:
        """Extract bounding boxes from a binary mask slice"""
        if mask_slice.sum() == 0:
            return []
        
        # Find connected components
        labeled_mask = measure.label(mask_slice > 0)
        regions = measure.regionprops(labeled_mask)
        
        bboxes = []
        for region in regions:
            # Get bounding box (min_row, min_col, max_row, max_col)
            minr, minc, maxr, maxc = region.bbox
            # Convert to [xmin, ymin, xmax, ymax] format
            bboxes.append([minc, minr, maxc, maxr])
        
        return bboxes
    
    def convert_to_2d_slices(self, patient_id: str, only_with_tumor: bool = True, axis: int = 0) -> List[Dict]:
        """
        將3D NIfTI體積轉換為2D切片
        
        此函數是核心處理流程:
        1. 載入3D CT和腫瘤體積
        2. 逐層切片並標準化
        3. 提取每個切片的腫瘤邊界框
        4. 過濾無腫瘤切片（可選）
        
        Args:
            patient_id: 患者ID
            only_with_tumor: 是否只保留有腫瘤的切片
            axis: 切片軸向 (0, 1, 或 2 對應volume的三個維度)
            
        Returns:
            切片資訊列表，每個元素包含:
            - slice_index: 切片索引
            - ct_slice: 標準化的CT影像 (0-255)
            - tumor_mask: 二值化腫瘤遮罩
            - bboxes: 腫瘤邊界框列表 [[x1,y1,x2,y2], ...]
            - has_tumor: 是否包含腫瘤
            - spacing: 體素間距 mm
            - slice_location: 切片位置 mm
            - axis: 切片軸向
        """
        patient_data = self.load_patient_data(patient_id)
        if not patient_data or not patient_data.get('has_data'):
            logger.error(f"❌ 患者 {patient_id} 資料載入失敗")
            return []
        
        # 載入3D體積
        ct_volume, ct_metadata = self.load_nifti_volume(patient_data['ct_file'])
        tumor_volume, tumor_metadata = self.load_nifti_volume(patient_data['tumor_file'])
        
        if ct_volume is None or tumor_volume is None:
            logger.error(f"❌ NIfTI檔案載入失敗")
            return []
        
        # 驗證維度匹配
        if ct_volume.shape != tumor_volume.shape:
            logger.warning(f"⚠️ CT和腫瘤遮罩維度不匹配: {ct_volume.shape} vs {tumor_volume.shape}")
            return []
        
        # 驗證軸向參數
        if axis not in [0, 1, 2]:
            logger.error(f"❌ 無效的軸向參數: {axis}，必須是0, 1, 或2")
            return []
        
        # 根據軸向調整切片方向（修正：對應NIfTI實際軸向）
        # NIfTI shape通常是 (dim0, dim1, dim2)，需根據實際資料確定解剖方向
        axis_names = {
            0: "第0軸切片(沿dim0方向)", 
            1: "第1軸切片(沿dim1方向)", 
            2: "第2軸切片(沿dim2方向)"
        }
        
        slice_data = []
        depth = ct_volume.shape[axis]
        spacing = ct_metadata.get('spacing', (1.0, 1.0, 1.0))
        
        logger.info(f"📊 處理3D體積: 形狀={ct_volume.shape}, 間距={spacing}, 切片軸={axis} ({axis_names[axis]})")
        
        for slice_idx in range(depth):
            # 根據軸向提取切片
            if axis == 0:  # 沿第0軸切片
                ct_slice = ct_volume[slice_idx, :, :]
                tumor_slice = tumor_volume[slice_idx, :, :]
                slice_spacing = spacing[0]
            elif axis == 1:  # 沿第1軸切片
                ct_slice = ct_volume[:, slice_idx, :]
                tumor_slice = tumor_volume[:, slice_idx, :]
                slice_spacing = spacing[1]
            else:  # axis == 2, 沿第2軸切片
                ct_slice = ct_volume[:, :, slice_idx]
                tumor_slice = tumor_volume[:, :, slice_idx]
                slice_spacing = spacing[2]
            
            has_tumor = tumor_slice.sum() > 0
            
            # 根據設定跳過無腫瘤切片
            if only_with_tumor and not has_tumor:
                continue
            
            # 標準化CT切片到0-255範圍
            ct_min, ct_max = ct_slice.min(), ct_slice.max()
            if ct_max > ct_min:
                ct_normalized = ((ct_slice - ct_min) / (ct_max - ct_min) * 255).astype(np.uint8)
            else:
                ct_normalized = np.zeros_like(ct_slice, dtype=np.uint8)
            
            # 從遮罩提取邊界框
            bboxes = self.extract_bboxes_from_mask(tumor_slice)
            
            slice_info = {
                'slice_index': slice_idx,
                'ct_slice': ct_normalized,
                'tumor_mask': (tumor_slice > 0).astype(np.uint8),
                'bboxes': bboxes,
                'has_tumor': has_tumor,
                'spacing': spacing,
                'slice_location': slice_idx * slice_spacing,
                'axis': axis
            }
            
            slice_data.append(slice_info)
        
        logger.info(f"✅ 轉換完成: {len(slice_data)}/{depth} 切片（有腫瘤），軸={axis}")
        return slice_data
    
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
        """
        創建摘要視覺化 - 在一張圖中顯示所有處理的切片
        
        限制顯示前16個切片，使用4x4網格佈局
        優化: 不重新載入影像，直接從slice_results中的資料重建
        
        Args:
            patient_id: 患者ID
            slice_results: 切片處理結果列表
        """
        try:
            if not slice_results:
                logger.warning(f"⚠️ 無切片結果，跳過摘要視覺化")
                return
            
            summary_dir = self.segmentation_result_base / patient_id / "summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            
            # 限制最多顯示16個切片
            display_slices = slice_results[:16] if len(slice_results) > 16 else slice_results
            n_slices = len(display_slices)
            
            # 計算網格維度
            cols = min(4, n_slices)
            rows = (n_slices + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            
            # 統一處理axes格式（確保總是2D數組）
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            for idx, result in enumerate(display_slices):
                row, col = idx // cols, idx % cols
                ax = axes[row, col]
                
                # 從ground_truth_mask重建顯示影像
                if 'ground_truth_mask' in result:
                    mask = result['ground_truth_mask']
                    # 簡單顯示遮罩
                    ax.imshow(mask, cmap='gray')
                    
                    # 添加邊界框
                    for ann in result['annotations']:
                        bbox = ann['bbox']
                        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                        rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                               edgecolor='yellow', facecolor='none')
                        ax.add_patch(rect)
                
                ax.set_title(f"Slice {result['metadata']['instance_number']}", fontsize=8)
                ax.axis('off')
            
            # 隱藏未使用的子圖
            for idx in range(n_slices, rows * cols):
                row, col = idx // cols, idx % cols
                axes[row, col].axis('off')
            
            plt.suptitle(f'Patient {patient_id} - Segmentation Summary\n'
                        f'{len(slice_results)} slices processed', fontsize=16)
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = summary_dir / f"segmentation_summary_{timestamp}.png"
            plt.savefig(str(summary_path), dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"✅ 摘要視覺化已保存: {summary_path.name}")
            
        except Exception as e:
            logger.error(f"❌ 創建摘要視覺化失敗: {e}", exc_info=True)
    
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
        """
        保存特徵用於LLM報告生成
        
        優化重點:
        - 避免重複載入影像（從slice_results中獲取）
        - 批量處理特徵提取
        - 生成JSON和可讀文本兩種格式
        
        Args:
            patient_id: 患者ID
            slice_results: 切片處理結果列表
        """
        try:
            output_dir = self.segmentation_result_base / patient_id / "llm_features"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            all_features = {
                'patient_id': patient_id,
                'total_slices': len(slice_results),
                'processing_timestamp': datetime.now().isoformat(),
                'slices': []
            }
            
            logger.info(f"🔬 開始提取 {len(slice_results)} 個切片的特徵...")
            
            for idx, slice_result in enumerate(slice_results, 1):
                slice_features = {
                    'slice_file': slice_result['dicom_file'],
                    'slice_index': slice_result['metadata']['instance_number'],
                    'lesions': []
                }
                
                # 為每個病灶提取特徵（優化：不重新載入影像）
                if len(slice_result['annotations']) > 0:
                    # 重建灰階影像（從ground_truth_mask的維度）
                    # 注意：這裡做了簡化，實際應該緩存原始CT切片
                    for i, (annotation, mask) in enumerate(zip(slice_result['annotations'], slice_result['masks'])):
                        # 使用簡化特徵（避免需要原始影像）
                        features = self._extract_mask_based_features(mask, slice_result['metadata'], annotation)
                        
                        if features:
                            features['description'] = self.generate_llm_description(features)
                            slice_features['lesions'].append(features)
                
                if slice_features['lesions']:
                    all_features['slices'].append(slice_features)
                    logger.info(f"  [{idx}/{len(slice_results)}] 切片 {slice_result['metadata']['instance_number']}: "
                              f"{len(slice_features['lesions'])} 個病灶")
            
            # 保存JSON格式
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = output_dir / f"features_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_features, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ JSON特徵檔案已保存: {json_path.name}")
            
            # 保存可讀文本格式
            summary_path = output_dir / f"features_summary_{timestamp}.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"胸部CT腫瘤分析報告\n")
                f.write(f"{'='*80}\n")
                f.write(f"患者ID: {patient_id}\n")
                f.write(f"分析切片數: {len(all_features['slices'])}\n")
                f.write(f"處理時間: {all_features['processing_timestamp']}\n")
                f.write(f"{'='*80}\n\n")
                
                for slice_data in all_features['slices']:
                    f.write(f"\n切片 #{slice_data['slice_index']}: {slice_data['slice_file']}\n")
                    f.write(f"{'-'*80}\n")
                    
                    for idx, lesion in enumerate(slice_data['lesions'], 1):
                        f.write(f"\n病灶 {idx}:\n")
                        f.write(f"{lesion['description']}\n\n")
                    
                    f.write(f"{'='*80}\n")
            
            logger.info(f"✅ 文本摘要已保存: {summary_path.name}")
            logger.info(f"📊 總計提取 {sum(len(s['lesions']) for s in all_features['slices'])} 個病灶特徵")
            
        except Exception as e:
            logger.error(f"❌ 保存LLM特徵失敗: {e}", exc_info=True)
    
    def _extract_mask_based_features(self, mask: np.ndarray, metadata: Dict, annotation: Dict) -> Dict:
        """
        基於遮罩的簡化特徵提取（不需要原始CT影像）
        
        僅提取形態學特徵，用於快速處理
        """
        try:
            if mask.sum() == 0:
                return {}
            
            labeled_mask = measure.label(mask > 0)
            props = measure.regionprops(labeled_mask)[0]
            
            pixel_spacing = metadata.get('pixel_spacing', [1.0, 1.0])
            pixel_area_mm2 = float(pixel_spacing[0]) * float(pixel_spacing[1])
            
            return {
                'lesion_name': annotation.get('name', 'tumor'),
                'area_pixels': int(props.area),
                'area_mm2': float(props.area * pixel_area_mm2),
                'perimeter_pixels': float(props.perimeter),
                'circularity': float((4 * np.pi * props.area) / (props.perimeter ** 2) if props.perimeter > 0 else 0),
                'solidity': float(props.solidity),
                'eccentricity': float(props.eccentricity),
                'equivalent_diameter_mm': float(props.equivalent_diameter * pixel_spacing[0]),
                'centroid_x': float(props.centroid[1]),
                'centroid_y': float(props.centroid[0]),
                'relative_position_x': float(props.centroid[1] / mask.shape[1]),
                'relative_position_y': float(props.centroid[0] / mask.shape[0]),
                'slice_location': float(metadata.get('slice_location', 0)),
                'bbox': annotation.get('bbox', [0, 0, 0, 0]),
            }
        except Exception as e:
            logger.error(f"特徵提取失敗: {e}")
            return {}
    
    # ========== 輔助函數 ==========
    # 注意：以下DICOM相關函數已不再使用，保留僅供參考
    # 新版本直接使用NIfTI格式，不需要DICOM轉換
    
    def _resize_image_if_needed(self, image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """調整影像大小以匹配目標形狀"""
        if image.shape[:2] != target_shape:
            return cv2.resize(image, (target_shape[1], target_shape[0]))
        return image
    
    def process_patient(self, patient_id: str, save_results: bool = True, use_medsam: bool = True, axis: int = 0) -> Dict:
        """
        處理患者資料（支援多軸向切片）
        
        Args:
            patient_id: 患者ID
            save_results: 是否保存結果
            use_medsam: 是否使用MedSAM2精修
            axis: 切片軸 (0, 1, 或 2)
        """
        logger.info(f"Processing patient: {patient_id}")
        
        # Load patient data
        patient_data = self.load_patient_data(patient_id)
        if not patient_data or not patient_data.get('has_data'):
            return {'status': 'error', 'message': 'No patient data found'}
        
        # Convert 3D NIfTI to 2D slices (支援多軸向)
        slice_data = self.convert_to_2d_slices(patient_id, only_with_tumor=True, axis=axis)
        
        if not slice_data:
            return {'status': 'error', 'message': 'No slices with tumor found'}
        
        logger.info(f"Processing {len(slice_data)} slices with tumor (axis={axis})")
        
        # Process each slice
        slice_results = []
        for slice_info in slice_data:
            ct_slice = slice_info['ct_slice']
            tumor_mask = slice_info['tumor_mask']
            bboxes = slice_info['bboxes']
            
            if not bboxes:
                continue
            
            # Convert grayscale to RGB for MedSAM2
            ct_rgb = np.stack([ct_slice] * 3, axis=-1)
            
            # Optionally run MedSAM2 refinement
            if use_medsam and MEDSAM2_AVAILABLE:
                refined_masks = self.segment_with_medsam2(ct_rgb, bboxes)
            else:
                # Use ground truth masks
                refined_masks = [tumor_mask]
            
            # Create annotation format for visualization
            annotations = []
            for bbox in bboxes:
                annotations.append({
                    'name': 'tumor',
                    'bbox': bbox,
                    'width': ct_slice.shape[1],
                    'height': ct_slice.shape[0]
                })
            
            metadata = {
                'slice_location': slice_info['slice_location'],
                'pixel_spacing': slice_info['spacing'][1:],  # (y, x) spacing
                'slice_thickness': slice_info['spacing'][0],
                'instance_number': slice_info['slice_index']
            }
            
            slice_results.append({
                'dicom_file': f"slice_{slice_info['slice_index']:03d}.png",
                'xml_file': f"slice_{slice_info['slice_index']:03d}_mask.png",
                'metadata': metadata,
                'annotations': annotations,
                'masks': refined_masks,
                'ground_truth_mask': tumor_mask
            })
            
            # Save visualization
            if save_results:
                try:
                    self.save_visualization_images(
                        patient_id=patient_id,
                        image=ct_rgb,
                        annotations=annotations,
                        masks=refined_masks,
                        dicom_filename=f"slice_{slice_info['slice_index']:03d}",
                        slice_index=slice_info['slice_index']
                    )
                    
                    self.save_individual_images(
                        patient_id=patient_id,
                        image=ct_rgb,
                        annotations=annotations,
                        masks=refined_masks,
                        dicom_filename=f"slice_{slice_info['slice_index']:03d}",
                        slice_index=slice_info['slice_index']
                    )
                except Exception as e:
                    logger.warning(f"Failed to save visualization for slice {slice_info['slice_index']}: {e}")
        
        if not slice_results:
            return {'status': 'error', 'message': 'No slices processed successfully'}
        
        # Create summary visualization
        if save_results:
            try:
                self.create_summary_visualization(patient_id, slice_results)
            except Exception as e:
                logger.warning(f"Failed to create summary: {e}")
        
        # Extract and save features for LLM
        if save_results:
            try:
                self.save_features_for_llm(patient_id, slice_results)
            except Exception as e:
                logger.warning(f"Failed to save LLM features: {e}")
        
        # Save 3D segmentation results
        nifti_path = None
        if save_results:
            output_dir = self.segmentation_result_base / patient_id
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create a copy of the original files for reference
            import shutil
            ct_copy = output_dir / f"original_CT_{timestamp}.nii.gz"
            tumor_copy = output_dir / f"original_tumor_{timestamp}.nii.gz"
            shutil.copy(patient_data['ct_file'], ct_copy)
            shutil.copy(patient_data['tumor_file'], tumor_copy)
            
            nifti_path = str(tumor_copy)
            logger.info(f"Saved original files: {ct_copy.name}, {tumor_copy.name}")
        
        return {
            'status': 'success',
            'patient_id': patient_id,
            'processed_slices': len(slice_results),
            'total_slices_with_tumor': len(slice_data),
            'nifti_path': nifti_path,
            'axis': axis
        }
    
    def process_all_patients(self, save_results: bool = True, use_medsam: bool = True, max_patients: int = None, axis: int = 0) -> Dict:
        """
        批量處理患者（支援多軸向）
        
        Args:
            save_results: 是否保存結果
            use_medsam: 是否使用MedSAM2
            max_patients: 最大處理患者數
            axis: 切片軸 (0, 1, 或 2)
        """
        patients = self.get_patient_list()
        if not patients:
            return {'status': 'error', 'message': 'No patients found'}
        
        # Limit number of patients if specified
        if max_patients:
            patients = patients[:max_patients]
        
        logger.info(f"Processing {len(patients)} patients (axis={axis})")
        
        results = {}
        for i, patient_id in enumerate(patients, 1):
            try:
                logger.info(f"[{i}/{len(patients)}] Processing {patient_id}")
                result = self.process_patient(patient_id, save_results=save_results, use_medsam=use_medsam, axis=axis)
                results[patient_id] = result
                if result['status'] == 'success':
                    logger.info(f"✓ {patient_id}: {result['processed_slices']}/{result.get('total_slices_with_tumor', 0)} slices processed")
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
    parser = argparse.ArgumentParser(description="MedSAM2 Segmentation for Chest Tumor (NIfTI version)")
    parser.add_argument("--patient_id", type=str, help="Patient ID to process (if not specified, process all patients)")
    parser.add_argument("--data_dir", type=str, default="../datasets/all_patient_data", help="Patient data directory")
    parser.add_argument("--config", type=str, default="sam2.1_hiera_t512.yaml", help="MedSAM2 config file")
    parser.add_argument("--list_patients", action="store_true", help="List available patients")
    parser.add_argument("--no_medsam", action="store_true", help="Skip MedSAM2 refinement, use ground truth masks only")
    parser.add_argument("--no_timestamp", action="store_true", help="Don't use timestamp in result directory")
    parser.add_argument("--max_patients", type=int, help="Maximum number of patients to process (for testing)")
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze dataset statistics without processing")
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], 
                       help="Slice axis: 0=first dimension, 1=second dimension, 2=third dimension (default)")
    
    args = parser.parse_args()
    
    # Determine if we're just listing patients
    list_only = args.list_patients or args.analyze_only
    
    segmentator = MedSAMSegmentator(data_dir=args.data_dir, config_file=args.config, 
                                   use_timestamp=not args.no_timestamp, list_only=list_only)
    
    if args.list_patients:
        patients = segmentator.get_patient_list()
        print(f"Available patients ({len(patients)}): {', '.join(patients)}")
        return
    
    if args.analyze_only:
        # Analyze dataset statistics
        patients = segmentator.get_patient_list()
        print(f"\n{'='*80}")
        print(f"Dataset Analysis")
        print(f"{'='*80}")
        print(f"Total patients: {len(patients)}")
        
        # Sample a few patients to check structure
        sample_size = min(10, len(patients))
        print(f"\nChecking {sample_size} sample patients...")
        
        valid_count = 0
        total_slices = 0
        for patient_id in patients[:sample_size]:
            patient_data = segmentator.load_patient_data(patient_id)
            if patient_data.get('has_data'):
                valid_count += 1
                slice_data = segmentator.convert_to_2d_slices(patient_id, only_with_tumor=True, axis=args.axis)
                total_slices += len(slice_data)
                print(f"  ✓ {patient_id}: {len(slice_data)} slices with tumor")
            else:
                print(f"  ✗ {patient_id}: Missing data files")
        
        avg_slices = total_slices / valid_count if valid_count > 0 else 0
        estimated_total = int(avg_slices * len(patients))
        print(f"\nEstimated statistics:")
        print(f"  - Average slices per patient: {avg_slices:.1f}")
        print(f"  - Estimated total slices: {estimated_total}")
        print(f"{'='*80}\n")
        return
    
    use_medsam = not args.no_medsam
    
    if args.patient_id:
        # Process specific patient
        results = segmentator.process_patient(args.patient_id, use_medsam=use_medsam, axis=args.axis)
        if results['status'] == 'success':
            print(f"\n{'='*80}")
            print(f"Patient {results['patient_id']} processed successfully")
            print(f"{'='*80}")
            print(f"Slice axis: {results.get('axis', 0)}")
            print(f"Processed slices: {results['processed_slices']}/{results.get('total_slices_with_tumor', 0)}")
            if results.get('nifti_path'):
                print(f"Results saved to: {results['nifti_path']}")
            print(f"{'='*80}\n")
        else:
            print(f"Processing failed: {results.get('message', 'Unknown error')}")
    else:
        # Process all patients
        results = segmentator.process_all_patients(use_medsam=use_medsam, max_patients=args.max_patients, axis=args.axis)
        if results['status'] == 'success':
            print(f"\n{'='*80}")
            print(f"Batch Processing Complete")
            print(f"{'='*80}")
            print(f"Success: {results['successful_patients']}/{results['total_patients']} patients")
            
            # Show summary of failed patients
            failed_patients = [pid for pid, result in results['results'].items() if result['status'] != 'success']
            if failed_patients:
                print(f"\nFailed patients ({len(failed_patients)}):")
                for pid in failed_patients[:10]:  # Show first 10
                    print(f"  - {pid}")
                if len(failed_patients) > 10:
                    print(f"  ... and {len(failed_patients) - 10} more")
            print(f"{'='*80}\n")
        else:
            print(f"Processing failed: {results.get('message', 'Unknown error')}")


if __name__ == "__main__":
    main()
