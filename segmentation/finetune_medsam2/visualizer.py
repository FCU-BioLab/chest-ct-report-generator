#!/usr/bin/env python3
"""
分割結果可視化模組
提供 GT 和 Prediction 對比圖生成功能
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class SegmentationVisualizer:
    """
    分割結果可視化工具
    
    生成 Ground Truth 和 Prediction 的對比圖片
    """
    
    def __init__(self, output_dir: str, dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        
        # 直接使用傳入的目錄作為可視化目錄
        self.vis_dir = self.output_dir
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 儲存每個患者的切片結果（用於生成摘要圖）
        self.patient_slice_results: Dict[str, List[Dict]] = {}
    
    def save_slice_comparison(
        self,
        image: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        patient_id: str,
        slice_idx: int,
        dice_score: float = None,
        iou_score: float = None,
        bboxes: np.ndarray = None
    ) -> Dict:
        """
        保存單個切片的 GT 和 Prediction 對比圖
        
        Args:
            image: CT 影像 [H, W] 或 [3, H, W]
            gt_mask: Ground Truth 遮罩 [H, W]
            pred_mask: 預測遮罩 [H, W]
            patient_id: 患者 ID
            slice_idx: 切片索引
            dice_score: Dice 分數（可選）
            iou_score: IoU 分數（可選）
            bboxes: Bounding boxes（可選）
            
        Returns:
            包含圖片路徑和指標的字典
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 確保 patient_id 是安全的檔名
        safe_patient_id = str(patient_id).replace('.', '_').replace('/', '_')[:50]
        
        # 建立患者目錄
        patient_vis_dir = self.vis_dir / safe_patient_id
        patient_vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 處理影像格式
        if len(image.shape) == 3:
            if image.shape[0] == 3:  # [3, H, W] -> [H, W]
                image = image[0]
            elif image.shape[2] == 3:  # [H, W, 3] -> [H, W]
                image = image[:, :, 0]
        
        # 確保遮罩是二值化的
        gt_binary = (gt_mask > 0.5).astype(np.float32)
        pred_binary = (pred_mask > 0.5).astype(np.float32)
        
        # ===== 圖1: Ground Truth =====
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        ax1.imshow(image, cmap='gray', vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
        
        gt_overlay = np.zeros((*gt_binary.shape, 4))
        gt_overlay[gt_binary > 0] = [0, 1, 0, 0.4]
        ax1.imshow(gt_overlay)
        ax1.contour(gt_binary, levels=[0.5], colors=['lime'], linewidths=2)
        
        if bboxes is not None and len(bboxes) > 0:
            for bbox in bboxes:
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                         fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
                    ax1.add_patch(rect)
        
        ax1.set_title(f'Ground Truth\nPatient: {patient_id[:30]}...\nSlice: {slice_idx}', fontsize=12)
        ax1.axis('off')
        
        plt.tight_layout()
        gt_path = patient_vis_dir / f"slice_{slice_idx:04d}_gt.png"
        plt.savefig(gt_path, dpi=self.dpi, bbox_inches='tight', facecolor='black')
        plt.close(fig1)
        
        # ===== 圖2: Prediction =====
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
        ax2.imshow(image, cmap='gray', vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
        
        pred_overlay = np.zeros((*pred_binary.shape, 4))
        pred_overlay[pred_binary > 0] = [1, 0, 0, 0.4]
        ax2.imshow(pred_overlay)
        ax2.contour(pred_binary, levels=[0.5], colors=['red'], linewidths=2)
        
        if bboxes is not None and len(bboxes) > 0:
            for bbox in bboxes:
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                         fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
                    ax2.add_patch(rect)
        
        title = f'Prediction\nPatient: {patient_id[:30]}...\nSlice: {slice_idx}'
        if dice_score is not None:
            title += f'\nDice: {dice_score:.4f}'
        if iou_score is not None:
            title += f' | IoU: {iou_score:.4f}'
        ax2.set_title(title, fontsize=12)
        ax2.axis('off')
        
        plt.tight_layout()
        pred_path = patient_vis_dir / f"slice_{slice_idx:04d}_pred.png"
        plt.savefig(pred_path, dpi=self.dpi, bbox_inches='tight', facecolor='black')
        plt.close(fig2)
        
        # ===== 圖3: 對比圖（左右並排）=====
        fig3, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 左: Ground Truth
        axes[0].imshow(image, cmap='gray', vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
        gt_overlay = np.zeros((*gt_binary.shape, 4))
        gt_overlay[gt_binary > 0] = [0, 1, 0, 0.4]
        axes[0].imshow(gt_overlay)
        axes[0].contour(gt_binary, levels=[0.5], colors=['lime'], linewidths=2)
        axes[0].set_title('Ground Truth', fontsize=14, color='lime')
        axes[0].axis('off')
        
        # 中: Prediction
        axes[1].imshow(image, cmap='gray', vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
        pred_overlay = np.zeros((*pred_binary.shape, 4))
        pred_overlay[pred_binary > 0] = [1, 0, 0, 0.4]
        axes[1].imshow(pred_overlay)
        axes[1].contour(pred_binary, levels=[0.5], colors=['red'], linewidths=2)
        axes[1].set_title('Prediction', fontsize=14, color='red')
        axes[1].axis('off')
        
        # 右: 重疊對比
        axes[2].imshow(image, cmap='gray', vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
        
        overlap = gt_binary * pred_binary
        gt_only = gt_binary * (1 - pred_binary)
        pred_only = pred_binary * (1 - gt_binary)
        
        overlap_rgb = np.zeros((*gt_binary.shape, 4))
        overlap_rgb[gt_only > 0] = [0, 1, 0, 0.5]
        overlap_rgb[pred_only > 0] = [1, 0, 0, 0.5]
        overlap_rgb[overlap > 0] = [1, 1, 0, 0.5]
        axes[2].imshow(overlap_rgb)
        
        axes[2].contour(gt_binary, levels=[0.5], colors=['lime'], linewidths=1.5, linestyles='--')
        axes[2].contour(pred_binary, levels=[0.5], colors=['red'], linewidths=1.5)
        
        title_overlap = 'Comparison (Green=GT, Red=Pred, Yellow=Overlap)'
        if dice_score is not None:
            title_overlap += f'\nDice: {dice_score:.4f}'
        if iou_score is not None:
            title_overlap += f' | IoU: {iou_score:.4f}'
        axes[2].set_title(title_overlap, fontsize=12)
        axes[2].axis('off')
        
        fig3.suptitle(f'Patient: {patient_id[:40]}... | Slice: {slice_idx}', fontsize=14, y=1.02)
        
        plt.tight_layout()
        comparison_path = patient_vis_dir / f"slice_{slice_idx:04d}_comparison.png"
        plt.savefig(comparison_path, dpi=self.dpi, bbox_inches='tight', facecolor='black')
        plt.close(fig3)
        
        result = {
            'gt_path': str(gt_path),
            'pred_path': str(pred_path),
            'comparison_path': str(comparison_path),
            'slice_idx': slice_idx,
            'dice': dice_score if dice_score is not None else 0.0,
            'iou': iou_score if iou_score is not None else 0.0
        }
        
        # 儲存切片結果到患者字典
        if safe_patient_id not in self.patient_slice_results:
            self.patient_slice_results[safe_patient_id] = []
        self.patient_slice_results[safe_patient_id].append(result)
        
        return result
    
    def create_patient_summary_grid(
        self,
        patient_id: str,
        slice_results: List[Dict] = None,
        max_slices: int = 16
    ):
        """
        為單個患者創建切片摘要網格圖
        
        Args:
            patient_id: 患者 ID
            slice_results: 切片結果列表（可選）
            max_slices: 最多顯示的切片數
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        safe_patient_id = str(patient_id).replace('.', '_').replace('/', '_')[:50]
        if slice_results is None:
            slice_results = self.patient_slice_results.get(safe_patient_id, [])
        
        if not slice_results:
            return
        
        patient_vis_dir = self.vis_dir / safe_patient_id
        
        # 限制切片數量
        if len(slice_results) > max_slices:
            indices = np.linspace(0, len(slice_results)-1, max_slices, dtype=int)
            slice_results = [slice_results[i] for i in indices]
        
        n_slices = len(slice_results)
        cols = min(4, n_slices)
        rows = (n_slices + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, result in enumerate(slice_results):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            comparison_path = result.get('comparison_path')
            if comparison_path and Path(comparison_path).exists():
                img = plt.imread(comparison_path)
                ax.imshow(img)
            
            slice_idx = result.get('slice_idx', idx)
            dice = result.get('dice', 0)
            ax.set_title(f'Slice {slice_idx}\nDice: {dice:.3f}', fontsize=10)
            ax.axis('off')
        
        for idx in range(n_slices, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        fig.suptitle(f'Patient Summary: {patient_id[:50]}...', fontsize=14, y=1.02)
        plt.tight_layout()
        
        summary_path = patient_vis_dir / "patient_summary.png"
        plt.savefig(summary_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        self.logger.info(f"📊 患者摘要圖已保存: {summary_path}")
    
    def get_statistics(self) -> Dict:
        """
        獲取可視化統計資訊
        
        Returns:
            包含統計資訊的字典
        """
        total_images = 0
        total_patients = len(self.patient_slice_results)
        
        for patient_id, slices in self.patient_slice_results.items():
            total_images += len(slices) * 3
        
        total_images += total_patients
        
        return {
            'total_images': total_images,
            'total_patients': total_patients,
            'output_dir': str(self.output_dir)
        }
