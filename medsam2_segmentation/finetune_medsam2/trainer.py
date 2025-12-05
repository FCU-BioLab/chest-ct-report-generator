#!/usr/bin/env python3
"""
訓練器模組
提供 MedSAM2 模型的訓練與評估功能
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from .losses import CombinedLoss
from .utils import compute_all_metrics, EarlyStopping, PatientMetricsTracker


class SegmentationVisualizer:
    """
    分割結果可視化工具
    
    生成 Ground Truth 和 Prediction 的對比圖片
    """
    
    def __init__(self, output_dir: str, dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.logger = logging.getLogger(__name__)
        
        # 建立輸出目錄
        self.vis_dir = self.output_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
    
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
    ):
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
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        # 確保 patient_id 是安全的檔名
        safe_patient_id = str(patient_id).replace('.', '_').replace('/', '_')[:50]
        
        # 建立患者目錄
        patient_vis_dir = self.vis_dir / safe_patient_id
        patient_vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 處理影像格式
        if len(image.shape) == 3:
            if image.shape[0] == 3:  # [3, H, W] -> [H, W]
                image = image[0]  # 取第一個通道
            elif image.shape[2] == 3:  # [H, W, 3] -> [H, W]
                image = image[:, :, 0]
        
        # 確保遮罩是二值化的
        gt_binary = (gt_mask > 0.5).astype(np.float32)
        pred_binary = (pred_mask > 0.5).astype(np.float32)
        
        # 創建顏色遮罩
        # GT: 綠色, Pred: 紅色, 重疊: 黃色
        
        # ===== 圖1: Ground Truth =====
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        
        # 顯示原始影像
        ax1.imshow(image, cmap='gray', vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
        
        # 疊加 GT 遮罩（綠色半透明）
        gt_overlay = np.zeros((*gt_binary.shape, 4))
        gt_overlay[gt_binary > 0] = [0, 1, 0, 0.4]  # 綠色，40% 透明度
        ax1.imshow(gt_overlay)
        
        # 繪製 GT 輪廓
        ax1.contour(gt_binary, levels=[0.5], colors=['lime'], linewidths=2)
        
        # 繪製 bounding boxes
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
        
        # 顯示原始影像
        ax2.imshow(image, cmap='gray', vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
        
        # 疊加 Prediction 遮罩（紅色半透明）
        pred_overlay = np.zeros((*pred_binary.shape, 4))
        pred_overlay[pred_binary > 0] = [1, 0, 0, 0.4]  # 紅色，40% 透明度
        ax2.imshow(pred_overlay)
        
        # 繪製 Prediction 輪廓
        ax2.contour(pred_binary, levels=[0.5], colors=['red'], linewidths=2)
        
        # 繪製 bounding boxes
        if bboxes is not None and len(bboxes) > 0:
            for bbox in bboxes:
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                         fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
                    ax2.add_patch(rect)
        
        # 標題包含評估分數
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
        
        # 右: 重疊對比 (GT=綠, Pred=紅, 重疊=黃)
        axes[2].imshow(image, cmap='gray', vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
        
        # 計算重疊區域
        overlap = gt_binary * pred_binary
        gt_only = gt_binary * (1 - pred_binary)
        pred_only = pred_binary * (1 - gt_binary)
        
        # 創建 RGB 重疊圖
        overlap_rgb = np.zeros((*gt_binary.shape, 4))
        overlap_rgb[gt_only > 0] = [0, 1, 0, 0.5]      # GT only: 綠色
        overlap_rgb[pred_only > 0] = [1, 0, 0, 0.5]    # Pred only: 紅色
        overlap_rgb[overlap > 0] = [1, 1, 0, 0.5]      # 重疊: 黃色
        axes[2].imshow(overlap_rgb)
        
        # 繪製輪廓
        axes[2].contour(gt_binary, levels=[0.5], colors=['lime'], linewidths=1.5, linestyles='--')
        axes[2].contour(pred_binary, levels=[0.5], colors=['red'], linewidths=1.5)
        
        title_overlap = 'Comparison (Green=GT, Red=Pred, Yellow=Overlap)'
        if dice_score is not None:
            title_overlap += f'\nDice: {dice_score:.4f}'
        if iou_score is not None:
            title_overlap += f' | IoU: {iou_score:.4f}'
        axes[2].set_title(title_overlap, fontsize=12)
        axes[2].axis('off')
        
        # 總標題
        fig3.suptitle(f'Patient: {patient_id[:40]}... | Slice: {slice_idx}', fontsize=14, y=1.02)
        
        plt.tight_layout()
        comparison_path = patient_vis_dir / f"slice_{slice_idx:04d}_comparison.png"
        plt.savefig(comparison_path, dpi=self.dpi, bbox_inches='tight', facecolor='black')
        plt.close(fig3)
        
        return {
            'gt_path': str(gt_path),
            'pred_path': str(pred_path),
            'comparison_path': str(comparison_path)
        }
    
    def create_patient_summary_grid(
        self,
        patient_id: str,
        slice_results: List[Dict],
        max_slices: int = 16
    ):
        """
        為單個患者創建切片摘要網格圖
        
        Args:
            patient_id: 患者 ID
            slice_results: 切片結果列表
            max_slices: 最多顯示的切片數
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if not slice_results:
            return
        
        safe_patient_id = str(patient_id).replace('.', '_').replace('/', '_')[:50]
        patient_vis_dir = self.vis_dir / safe_patient_id
        
        # 限制切片數量
        if len(slice_results) > max_slices:
            # 等間隔選取
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
            
            # 載入對比圖
            comparison_path = result.get('comparison_path')
            if comparison_path and Path(comparison_path).exists():
                img = plt.imread(comparison_path)
                ax.imshow(img)
            
            slice_idx = result.get('slice_idx', idx)
            dice = result.get('dice', 0)
            ax.set_title(f'Slice {slice_idx}\nDice: {dice:.3f}', fontsize=10)
            ax.axis('off')
        
        # 隱藏多餘的子圖
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


class LesionFeatureExtractor:
    """
    病灶特徵提取器
    
    從 MedSAM2 模型提取多層次特徵用於 LLM Fine-Tuning：
    1. Image Encoder 特徵（全局影像語義）
    2. Prompt Encoder 特徵（病灶區域語義）
    3. Mask Decoder 特徵（分割預測特徵）
    4. 形態學特徵（面積、周長、圓形度等）
    5. 強度特徵（HU值統計）
    """
    
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def compute_morphological_features(mask: np.ndarray, spacing: Tuple[float, float] = (1.0, 1.0)) -> Dict:
        """
        計算分割遮罩的形態學特徵
        
        Args:
            mask: 二值化遮罩 [H, W]
            spacing: 像素間距 (spacing_x, spacing_y) mm
        
        Returns:
            形態學特徵字典
        """
        from scipy import ndimage
        from skimage import measure as sk_measure
        
        features = {
            'area_pixels': 0,
            'area_mm2': 0.0,
            'perimeter_mm': 0.0,
            'equivalent_diameter_mm': 0.0,
            'major_axis_mm': 0.0,
            'minor_axis_mm': 0.0,
            'eccentricity': 0.0,
            'circularity': 0.0,
            'solidity': 0.0,
            'compactness': 0.0,
            'bbox_area_mm2': 0.0,
            'extent': 0.0,  # 填充率
            'centroid_x': 0.0,
            'centroid_y': 0.0,
        }
        
        # 確保 mask 是二值化的
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        if binary_mask.sum() == 0:
            return features
        
        # 像素間距
        px, py = spacing
        pixel_area = px * py  # mm²/pixel
        
        # 找到連通區域
        labeled_mask, num_labels = ndimage.label(binary_mask)
        
        if num_labels == 0:
            return features
        
        # 取最大連通區域
        region_sizes = ndimage.sum(binary_mask, labeled_mask, range(1, num_labels + 1))
        largest_label = np.argmax(region_sizes) + 1
        largest_region = (labeled_mask == largest_label).astype(np.uint8)
        
        # 使用 skimage 計算屬性
        props = sk_measure.regionprops(largest_region)
        if len(props) == 0:
            return features
        
        prop = props[0]
        
        # 面積
        features['area_pixels'] = prop.area
        features['area_mm2'] = prop.area * pixel_area
        
        # 周長
        contours = sk_measure.find_contours(largest_region, 0.5)
        if len(contours) > 0:
            largest_contour = max(contours, key=len)
            perimeter_pixels = len(largest_contour)
            features['perimeter_mm'] = perimeter_pixels * np.sqrt(px**2 + py**2) / 2
        
        # 等效直徑
        features['equivalent_diameter_mm'] = np.sqrt(4 * features['area_mm2'] / np.pi)
        
        # 主軸和副軸長度
        features['major_axis_mm'] = prop.major_axis_length * px
        features['minor_axis_mm'] = prop.minor_axis_length * py
        
        # 離心率
        features['eccentricity'] = prop.eccentricity
        
        # 圓形度 (4π * Area / Perimeter²)
        if features['perimeter_mm'] > 0:
            features['circularity'] = 4 * np.pi * features['area_mm2'] / (features['perimeter_mm'] ** 2)
        
        # 實心度 (Area / ConvexHullArea)
        features['solidity'] = prop.solidity
        
        # 緊密度 (Perimeter² / Area)
        if features['area_mm2'] > 0:
            features['compactness'] = (features['perimeter_mm'] ** 2) / features['area_mm2']
        
        # 邊界框面積
        bbox = prop.bbox  # (min_row, min_col, max_row, max_col)
        bbox_h = (bbox[2] - bbox[0]) * px
        bbox_w = (bbox[3] - bbox[1]) * py
        features['bbox_area_mm2'] = bbox_h * bbox_w
        
        # 填充率 (Area / BBoxArea)
        if features['bbox_area_mm2'] > 0:
            features['extent'] = features['area_mm2'] / features['bbox_area_mm2']
        
        # 質心
        features['centroid_y'] = prop.centroid[0] * py
        features['centroid_x'] = prop.centroid[1] * px
        
        return features
    
    @staticmethod
    def compute_intensity_features(image: np.ndarray, mask: np.ndarray) -> Dict:
        """
        計算病灶區域的強度特徵
        
        Args:
            image: CT 影像（HU 值或歸一化後）
            mask: 二值化遮罩
        
        Returns:
            強度特徵字典
        """
        features = {
            'mean_intensity': 0.0,
            'std_intensity': 0.0,
            'min_intensity': 0.0,
            'max_intensity': 0.0,
            'median_intensity': 0.0,
            'percentile_25': 0.0,
            'percentile_75': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'entropy': 0.0,
            'contrast': 0.0,  # 與背景對比度
        }
        
        binary_mask = (mask > 0.5).astype(bool)
        
        if binary_mask.sum() == 0:
            return features
        
        # 提取病灶區域像素
        lesion_pixels = image[binary_mask]
        
        # 基本統計
        features['mean_intensity'] = float(np.mean(lesion_pixels))
        features['std_intensity'] = float(np.std(lesion_pixels))
        features['min_intensity'] = float(np.min(lesion_pixels))
        features['max_intensity'] = float(np.max(lesion_pixels))
        features['median_intensity'] = float(np.median(lesion_pixels))
        features['percentile_25'] = float(np.percentile(lesion_pixels, 25))
        features['percentile_75'] = float(np.percentile(lesion_pixels, 75))
        
        # 偏度和峰度
        if len(lesion_pixels) > 2 and features['std_intensity'] > 1e-6:
            from scipy import stats
            features['skewness'] = float(stats.skew(lesion_pixels))
            features['kurtosis'] = float(stats.kurtosis(lesion_pixels))
        
        # 熵
        hist, _ = np.histogram(lesion_pixels, bins=64, density=True)
        hist = hist[hist > 0]
        if len(hist) > 0:
            features['entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
        
        # 與背景對比度
        background_mask = ~binary_mask
        if background_mask.sum() > 0:
            background_mean = np.mean(image[background_mask])
            features['contrast'] = float(features['mean_intensity'] - background_mean)
        
        return features
    
    def extract_deep_features(
        self,
        image_embedding: torch.Tensor,
        sparse_embeddings: torch.Tensor,
        dense_embeddings: torch.Tensor,
        high_res_feats: Optional[List[torch.Tensor]] = None
    ) -> Dict:
        """
        從 MedSAM2 提取深層特徵向量
        
        Args:
            image_embedding: Image encoder 輸出 [1, C, H, W]
            sparse_embeddings: Prompt encoder 稀疏嵌入
            dense_embeddings: Prompt encoder 密集嵌入
            high_res_feats: 高解析度特徵列表
        
        Returns:
            深層特徵字典（包含特徵向量）
        """
        features = {}
        
        # 1. Image Embedding 全局特徵（使用全局平均池化）
        if image_embedding is not None:
            img_global = torch.mean(image_embedding, dim=[2, 3])  # [1, C]
            features['image_embedding_global'] = img_global.cpu().numpy().flatten().tolist()
            features['image_embedding_dim'] = img_global.shape[-1]
        
        # 2. Sparse Embeddings 特徵
        if sparse_embeddings is not None:
            sparse_flat = sparse_embeddings.view(-1).cpu().numpy()
            features['sparse_embedding'] = sparse_flat.tolist()
            features['sparse_embedding_dim'] = len(sparse_flat)
        
        # 3. Dense Embeddings 全局特徵
        if dense_embeddings is not None:
            dense_global = torch.mean(dense_embeddings, dim=[2, 3])  # [1, C]
            features['dense_embedding_global'] = dense_global.cpu().numpy().flatten().tolist()
            features['dense_embedding_dim'] = dense_global.shape[-1]
        
        # 4. High Resolution Features（多尺度特徵）
        if high_res_feats is not None:
            for i, hr_feat in enumerate(high_res_feats):
                if hr_feat is not None and isinstance(hr_feat, torch.Tensor):
                    hr_global = torch.mean(hr_feat, dim=[2, 3])
                    features[f'high_res_feat_{i}_global'] = hr_global.cpu().numpy().flatten().tolist()
                    features[f'high_res_feat_{i}_dim'] = hr_global.shape[-1]
        
        return features
    
    def aggregate_lesion_features(
        self,
        morphological: Dict,
        intensity: Dict,
        deep_features: Dict,
        confidence: float = 1.0
    ) -> Dict:
        """
        聚合所有類型的病灶特徵
        
        Args:
            morphological: 形態學特徵
            intensity: 強度特徵
            deep_features: 深層特徵
            confidence: 分割置信度
        
        Returns:
            聚合後的完整特徵字典
        """
        aggregated = {
            'morphological': morphological,
            'intensity': intensity,
            'deep_features': deep_features,
            'confidence': confidence,
            'feature_version': '1.0',
        }
        
        # 生成文字描述（用於 LLM 輸入）
        description = self._generate_lesion_description(morphological, intensity)
        aggregated['text_description'] = description
        
        return aggregated
    
    @staticmethod
    def _generate_lesion_description(morphological: Dict, intensity: Dict) -> str:
        """
        生成病灶的文字描述
        """
        area = morphological.get('area_mm2', 0)
        diameter = morphological.get('equivalent_diameter_mm', 0)
        circularity = morphological.get('circularity', 0)
        solidity = morphological.get('solidity', 0)
        mean_hu = intensity.get('mean_intensity', 0)
        std_hu = intensity.get('std_intensity', 0)
        
        # 大小分類
        if diameter < 3:
            size_desc = "微小"
        elif diameter < 6:
            size_desc = "小"
        elif diameter < 10:
            size_desc = "中等"
        elif diameter < 30:
            size_desc = "大"
        else:
            size_desc = "巨大"
        
        # 形狀分類
        if circularity > 0.8:
            shape_desc = "圓形"
        elif circularity > 0.6:
            shape_desc = "近圓形"
        elif circularity > 0.4:
            shape_desc = "橢圓形"
        else:
            shape_desc = "不規則形"
        
        # 邊界描述
        if solidity > 0.9:
            border_desc = "邊界清晰光滑"
        elif solidity > 0.7:
            border_desc = "邊界較清晰"
        else:
            border_desc = "邊界不規則"
        
        description = (
            f"病灶為{size_desc}{shape_desc}結構，"
            f"等效直徑約 {diameter:.1f}mm，面積約 {area:.2f}mm²，"
            f"{border_desc}，"
            f"平均CT值 {mean_hu:.1f} HU，標準差 {std_hu:.1f} HU。"
        )
        
        return description


class MedSAM2Trainer:
    """
    MedSAM2 訓練器
    
    負責模型載入、訓練、驗證、評估和模型保存
    
    Args:
        model_config: MedSAM2 配置檔案名稱
        checkpoint_path: 預訓練模型路徑
        device: 計算設備 ('cuda' 或 'cpu')
        output_dir: 輸出目錄
    """
    
    def __init__(
        self,
        model_config: str = "sam2.1_hiera_t512.yaml",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        output_dir: str = "finetune_output"
    ):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # 載入模型
        self.logger.info(f"🔧 載入 MedSAM2 模型: {model_config}")
        self._load_model(model_config, checkpoint_path)
        
        # 損失函數（提高 Dice 權重處理類別不平衡）
        self.criterion = CombinedLoss(dice_weight=0.8, bce_weight=0.2)
        
        # 訓練歷史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'val_precision': [],
            'val_recall': [],
            'val_specificity': [],
            'val_accuracy': [],
            'val_hausdorff_95': [],
            'learning_rate': [],
            'epoch_time': [],
            'inference_time_per_sample': []
        }
        
        self.best_val_dice = 0.0
        self.current_epoch = 0
        
        # ✅ 優化：用於緩存 image embeddings（每個 batch 清空避免 OOM）
        self._current_batch_cache = {}
    
    def _load_model(self, config: str, checkpoint: Optional[str]):
        """
        載入 MedSAM2 模型
        
        ✅ 修正：移除重複的 Hydra 初始化
        """
        import sys
        from pathlib import Path
        
        # 添加 MedSAM2 路徑
        medsam2_path = Path(__file__).parent.parent / "MedSAM2"
        if str(medsam2_path) not in sys.path:
            sys.path.insert(0, str(medsam2_path))
        
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # ✅ 修正：Hydra 初始化應該在外部完成，這裡只載入模型
        # 不再重複呼叫 initialize_config_dir
        
        config_name = config.replace('.yaml', '')
        
        # 建立模型
        if checkpoint and Path(checkpoint).exists():
            self.logger.info(f"📥 從 checkpoint 載入: {checkpoint}")
            self.model = build_sam2(config_name, checkpoint, device=self.device)
        else:
            self.logger.info(f"🆕 建立新模型（使用預設權重）")
            self.model = build_sam2(config_name, device=self.device)
        
        self.predictor = SAM2ImagePredictor(self.model)
        
        # 只訓練 mask decoder 和 prompt encoder
        for name, param in self.model.named_parameters():
            if "image_encoder" in name:
                param.requires_grad = False  # 凍結 image encoder
            else:
                param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"✅ 可訓練參數: {trainable_params:,}")
    
    def _prepare_image_features(
        self, 
        image: torch.Tensor,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        計算 MedSAM2 image embeddings
        
        ✅ 優化：支援緩存避免重複計算（僅限當前 batch 內）
        ⚠️ FIX: 使用 _current_batch_cache 而非 _embedding_cache 避免 OOM
        
        Args:
            image: 輸入影像 [3, H, W]
            use_cache: 是否使用緩存
            
        Returns:
            (image_embedding, high_res_feats)
        """
        # 生成緩存 key
        if use_cache:
            cache_key = hash(image.cpu().numpy().tobytes())
            if cache_key in self._current_batch_cache:
                return self._current_batch_cache[cache_key]
        
        with torch.no_grad():
            np_image = image.permute(1, 2, 0).cpu().numpy()
            self.predictor.set_image(np_image)
        
        features = self.predictor._features or {}
        image_embedding = features.get("image_embed")
        if image_embedding is None:
            raise RuntimeError("Predictor features unavailable after set_image call")
        
        high_res_feats = features.get("high_res_feats")
        
        # ✅ 只緩存在當前 batch 內（避免無限累積）
        if use_cache:
            self._current_batch_cache[cache_key] = (image_embedding, high_res_feats)
        
        return image_embedding, high_res_feats
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        optimizer, 
        scheduler,
        accumulation_steps: int = 1
    ) -> Tuple[float, float]:
        """
        訓練一個 epoch
        
        ✅ 優化：支援梯度累積
        
        Args:
            train_loader: 訓練資料載入器
            optimizer: 優化器
            scheduler: 學習率調度器
            accumulation_steps: 梯度累積步數
            
        Returns:
            (平均訓練損失, 訓練耗時秒數)
        """
        start_time = time.time()
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # ✅ FIX: 清空 batch 緩存（每個 epoch 開始時）
        self._current_batch_cache.clear()
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # ✅ FIX: 每個 batch 開始時清空緩存，避免跨 batch 累積記憶體
            self._current_batch_cache.clear()
            
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            bboxes = batch['bboxes']
            
            batch_loss = 0.0
            batch_samples = 0
            
            # 處理 batch 中的每個樣本
            for i in range(len(images)):
                image = images[i]  # [3, H, W]
                gt_mask = masks[i]  # [1, H, W]
                bbox_tensor = bboxes[i]
                
                if len(bbox_tensor) == 0:
                    continue
                
                # ✅ 優化：同一張影像只計算一次 embedding
                image_embedding, high_res_feats = self._prepare_image_features(image)
                bbox_tensor = bbox_tensor.to(self.device)
                
                sample_loss = 0.0
                valid_boxes = 0
                
                for bbox in bbox_tensor:
                    if bbox.sum() == 0:
                        continue
                    
                    box_torch = bbox.unsqueeze(0)
                    sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                        points=None,
                        boxes=box_torch,
                        masks=None,
                    )
                    
                    low_res_masks, _, _, _ = self.model.sam_mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        repeat_image=False,
                        high_res_features=high_res_feats,
                    )
                    
                    pred_mask = F.interpolate(
                        low_res_masks,
                        size=(gt_mask.shape[-2], gt_mask.shape[-1]),
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    # ✅ 修正：統一 squeeze 維度
                    pred_mask = pred_mask.squeeze()
                    gt_mask_squeezed = gt_mask.squeeze()
                    
                    loss = self.criterion(pred_mask, gt_mask_squeezed)
                    sample_loss += loss
                    valid_boxes += 1
                
                if valid_boxes > 0:
                    sample_loss = sample_loss / valid_boxes
                    batch_loss += sample_loss
                    batch_samples += 1
            
            # 反向傳播（支援梯度累積）
            if batch_samples > 0:
                batch_loss = batch_loss / batch_samples
                
                # 梯度累積：除以累積步數
                loss_scaled = batch_loss / accumulation_steps
                loss_scaled.backward()
                
                # 每 accumulation_steps 步更新一次
                if (batch_idx + 1) % accumulation_steps == 0:
                    # ✅ 梯度裁剪防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()), 
                        max_norm=1.0
                    )
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += batch_loss.item()
                num_batches += 1
                
                # ✅ 修改：顯示累積平均 Loss，而非當前 Batch Loss
                current_avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f'{current_avg_loss:.4f}'})
        
        # 清理最後可能剩餘的梯度
        if num_batches % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 
                max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - start_time
        return avg_loss, epoch_time
    
    @torch.no_grad()
    def validate(
        self, 
        val_loader: DataLoader, 
        metrics_tracker: Optional[PatientMetricsTracker] = None
    ) -> Tuple[float, Dict[str, float], float]:
        """
        驗證模型
        
        ✅ 修正：加入異常處理避免 tqdm 卡死
        ✅ 新增：支援患者指標追蹤
        
        Args:
            val_loader: 驗證資料載入器
            metrics_tracker: 患者指標追蹤器（可選）
        
        Returns:
            (平均損失, 評估指標字典, 驗證耗時秒數)
        """
        start_time = time.time()
        self.model.eval()
        total_loss = 0.0
        
        # 初始化指標累加器
        metrics_sum = {
            'dice': 0.0,
            'iou': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0,
            'accuracy': 0.0,
            'hausdorff_95': 0.0
        }
        num_samples = 0
        
        # ✅ FIX: 清空 batch 緩存
        self._current_batch_cache.clear()
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
        
        try:
            for batch in pbar:
                # ✅ FIX: 每個 batch 開始時清空緩存
                self._current_batch_cache.clear()
                
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                patient_ids = batch['patient_id']
                slice_indices = batch['slice_index']
                bboxes = batch['bboxes']
                
                for i in range(len(images)):
                    image = images[i]
                    gt_mask = masks[i]
                    bbox_tensor = bboxes[i]
                    
                    if len(bbox_tensor) == 0:
                        continue
                    
                    # ✅ 優化：使用相同的 embedding 計算邏輯
                    image_embedding, high_res_feats = self._prepare_image_features(image)
                    bbox_tensor = bbox_tensor.to(self.device)
                    
                    sample_metrics = {k: 0.0 for k in metrics_sum.keys()}
                    sample_loss = 0.0
                    valid_boxes = 0
                    
                    for bbox in bbox_tensor:
                        if bbox.sum() == 0:
                            continue
                        
                        box_torch = bbox.unsqueeze(0)
                        
                        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                            points=None,
                            boxes=box_torch,
                            masks=None,
                        )
                        
                        low_res_masks, _, _, _ = self.model.sam_mask_decoder(
                            image_embeddings=image_embedding,
                            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                            repeat_image=False,
                            high_res_features=high_res_feats,
                        )
                        
                        pred_mask = F.interpolate(
                            low_res_masks,
                            size=(gt_mask.shape[-2], gt_mask.shape[-1]),
                            mode='bilinear',
                            align_corners=False
                        )
                        
                        # ✅ 修正：統一 squeeze 維度
                        pred_mask = pred_mask.squeeze()
                        gt_mask_squeezed = gt_mask.squeeze()
                        
                        # 損失計算
                        loss = self.criterion(pred_mask, gt_mask_squeezed)
                        sample_loss += loss.item()
                        valid_boxes += 1
                        
                        # 指標計算
                        batch_metrics = compute_all_metrics(pred_mask, gt_mask_squeezed)
                        for key, value in batch_metrics.items():
                            sample_metrics[key] += value
                    
                    if valid_boxes > 0:
                        normalized_loss = sample_loss / valid_boxes
                        total_loss += normalized_loss
                        
                        # 計算樣本平均指標
                        sample_avg_metrics = {k: v / valid_boxes for k, v in sample_metrics.items()}
                        
                        for key in metrics_sum.keys():
                            metrics_sum[key] += sample_avg_metrics[key]
                        
                        num_samples += 1
                        
                        # ✅ 新增：記錄患者級別指標
                        if metrics_tracker is not None:
                            patient_id = patient_ids[i]
                            slice_idx = slice_indices[i]
                            metrics_tracker.add_slice_metrics(
                                patient_id=patient_id,
                                slice_idx=slice_idx,
                                metrics=sample_avg_metrics
                            )
                        
                        # ✅ 修改：顯示累積平均指標，而非當前 Batch 指標
                        current_avg_loss = total_loss / num_samples
                        current_avg_metrics = {k: v / num_samples for k, v in metrics_sum.items()}
                        
                        pbar.set_postfix({
                            'loss': f'{current_avg_loss:.4f}',
                            'dice': f'{current_avg_metrics["dice"]:.4f}',
                            'iou': f'{current_avg_metrics["iou"]:.4f}'
                        })
        
        except Exception as e:
            self.logger.error(f"❌ 驗證過程發生錯誤: {e}")
            raise
        
        # 計算平均值
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        avg_metrics = {k: v / num_samples if num_samples > 0 else 0.0 
                      for k, v in metrics_sum.items()}
        
        val_time = time.time() - start_time
        return avg_loss, avg_metrics, val_time
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 7,
        accumulation_steps: int = 1
    ):
        """
        訓練模型
        
        Args:
            train_loader: 訓練資料載入器
            val_loader: 驗證資料載入器
            epochs: 訓練輪數
            learning_rate: 學習率
            weight_decay: 權重衰減
            early_stopping_patience: 早停容忍 epoch 數
            accumulation_steps: 梯度累積步數
        """
        # 優化器和學習率調度器
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 早停機制
        early_stopping = EarlyStopping(
            patience=early_stopping_patience, 
            min_delta=0.001, 
            mode='max'
        )
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🚀 開始訓練")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Epochs: {epochs}")
        self.logger.info(f"Learning Rate: {learning_rate}")
        self.logger.info(f"Gradient Accumulation Steps: {accumulation_steps}")
        self.logger.info(f"Early Stopping Patience: {early_stopping_patience}")
        self.logger.info(f"Loss weights: Dice={self.criterion.dice_weight}, BCE={self.criterion.bce_weight}")
        self.logger.info(f"Train samples: {len(train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(val_loader.dataset)}")
        self.logger.info(f"{'='*80}\n")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 訓練
            train_loss, epoch_time = self.train_epoch(
                train_loader, 
                optimizer, 
                scheduler,
                accumulation_steps
            )
            
            # 驗證
            val_loss, val_metrics, val_time = self.validate(val_loader)
            
            # 計算推理時間 (每樣本)
            inference_time_per_sample = (val_time / len(val_loader.dataset)) * 1000 if len(val_loader.dataset) > 0 else 0
            
            # 記錄歷史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_dice'].append(val_metrics['dice'])
            self.train_history['val_iou'].append(val_metrics['iou'])
            self.train_history['val_precision'].append(val_metrics['precision'])
            self.train_history['val_recall'].append(val_metrics['recall'])
            self.train_history['val_specificity'].append(val_metrics['specificity'])
            self.train_history['val_accuracy'].append(val_metrics['accuracy'])
            self.train_history['val_hausdorff_95'].append(val_metrics['hausdorff_95'])
            self.train_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            self.train_history['epoch_time'].append(epoch_time)
            self.train_history['inference_time_per_sample'].append(inference_time_per_sample)
            
            # 輸出結果
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
                f"  Time: {epoch_time:.1f}s ({epoch_time/len(train_loader):.3f}s/batch), "
                f"Inference: {inference_time_per_sample:.1f}ms/sample\n"
                f"  Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}, "
                f"Acc: {val_metrics['accuracy']:.4f}\n"
                f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
                f"Specificity: {val_metrics['specificity']:.4f}\n"
                f"  Hausdorff95: {val_metrics['hausdorff_95']:.2f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # 保存最佳模型
            if val_metrics['dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice']
                self.save_checkpoint('best_model.pth', is_best=True)
                self.logger.info(f"✅ 保存最佳模型 (Dice: {val_metrics['dice']:.4f})")
            
            # 早停檢查
            if early_stopping(epoch, val_metrics['dice']):
                self.logger.info(f"🛑 早停：訓練在 Epoch {epoch+1} 停止")
                break
            
            # 定期保存 checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ 訓練完成！最佳 Dice: {self.best_val_dice:.4f}")
        self.logger.info(f"{'='*80}\n")
        
        # 繪製訓練曲線
        self.plot_training_curves()
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """保存 checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_dice': self.best_val_dice,
            'train_history': self.train_history
        }
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            self.logger.info(f"💾 最佳模型已保存: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """載入 checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_dice = checkpoint['best_val_dice']
        self.train_history = checkpoint['train_history']
        self.logger.info(f"✅ Checkpoint 已載入: {checkpoint_path}")
    
    def plot_training_curves(self):
        """繪製訓練曲線（包含所有評估指標）"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # 無顯示後端
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib 未安裝，無法繪製訓練曲線")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        # 1. Loss 曲線
        axes[0, 0].plot(epochs, self.train_history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, self.train_history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Dice & IoU
        axes[0, 1].plot(epochs, self.train_history['val_dice'], label='Dice', color='green')
        axes[0, 1].plot(epochs, self.train_history['val_iou'], label='IoU', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Dice & IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Precision & Recall
        axes[0, 2].plot(epochs, self.train_history['val_precision'], label='Precision', color='purple')
        axes[0, 2].plot(epochs, self.train_history['val_recall'], label='Recall', color='brown')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_title('Precision & Recall')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. Specificity & Accuracy
        axes[1, 0].plot(epochs, self.train_history['val_specificity'], label='Specificity', color='cyan')
        axes[1, 0].plot(epochs, self.train_history['val_accuracy'], label='Accuracy', color='magenta')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Specificity & Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 5. Hausdorff Distance
        axes[1, 1].plot(epochs, self.train_history['val_hausdorff_95'], label='HD95', color='black')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Pixels')
        axes[1, 1].set_title('Hausdorff Distance (95%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 6. Learning Rate
        axes[1, 2].plot(epochs, self.train_history['learning_rate'], label='LR', color='gray')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('LR')
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_yscale('log')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        # 7. Training Time (新增)
        axes[2, 0].plot(epochs, self.train_history['epoch_time'], label='Epoch Time (s)', color='blue')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Seconds')
        axes[2, 0].set_title('Training Efficiency')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # 8. Inference Time (新增)
        axes[2, 1].plot(epochs, self.train_history['inference_time_per_sample'], label='Inference (ms)', color='red')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Milliseconds')
        axes[2, 1].set_title('Inference Speed')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        # 9. 摘要統計
        axes[2, 2].axis('off')
        summary_text = f"""
Training Summary
================

Best Val Dice: {self.best_val_dice:.4f}
Final Metrics:
  Dice: {self.train_history['val_dice'][-1]:.4f}
  IoU: {self.train_history['val_iou'][-1]:.4f}
  Acc: {self.train_history['val_accuracy'][-1]:.4f}
  
Efficiency:
  Avg Epoch Time: {np.mean(self.train_history['epoch_time']):.1f}s
  Avg Inference: {np.mean(self.train_history['inference_time_per_sample']):.1f}ms
"""
        axes[2, 2].text(0.1, 0.5, summary_text, fontsize=12, va='center', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png')
        plt.close()
        self.logger.info(f"📈 訓練曲線已保存: {self.output_dir / 'training_curves.png'}")
        
        # 9. 摘要統計
        axes[2, 2].axis('off')
        summary_text = f"""
Training Summary
================

Best Val Dice: {self.best_val_dice:.4f}
Final Metrics:
  Dice: {self.train_history['val_dice'][-1]:.4f}
  IoU: {self.train_history['val_iou'][-1]:.4f}
  Precision: {self.train_history['val_precision'][-1]:.4f}
  Recall: {self.train_history['val_recall'][-1]:.4f}
  Specificity: {self.train_history['val_specificity'][-1]:.4f}
  HD95: {self.train_history['val_hausdorff_95'][-1]:.2f}

Total Epochs: {len(epochs)}
        """
        axes[2, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace', verticalalignment='center')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"📊 訓練曲線已保存: {plot_path}")
        plt.close()
    
    @torch.no_grad()
    def test_and_extract_features(
        self,
        test_loader: DataLoader,
        output_dir: Optional[str] = None,
        extract_deep_features: bool = True,
        save_predictions: bool = True,
        save_visualizations: bool = True,
        spacing: Tuple[float, float] = (1.0, 1.0)
    ) -> Dict:
        """
        測試模型並提取病灶特徵用於 LLM Fine-Tuning
        
        Args:
            test_loader: 測試資料載入器
            output_dir: 特徵輸出目錄（預設為 self.output_dir/features）
            extract_deep_features: 是否提取深層特徵向量
            save_predictions: 是否保存預測遮罩
            save_visualizations: 是否保存可視化 PNG 圖片（GT mask、Pred mask、對比圖）
            spacing: 像素間距 (mm)
        
        Returns:
            包含所有測試結果和特徵的字典
        """
        from datetime import datetime
        
        if output_dir is None:
            output_dir = self.output_dir / "features"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化特徵提取器
        feature_extractor = LesionFeatureExtractor(self.model, self.device)
        
        # 初始化可視化器
        visualizer = None
        if save_visualizations:
            vis_dir = output_dir / "visualizations"
            visualizer = SegmentationVisualizer(vis_dir)
            self.logger.info(f"📸 可視化輸出目錄: {vis_dir}")
        
        self.model.eval()
        self._current_batch_cache.clear()
        
        # 結果容器
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'best_val_dice': self.best_val_dice,
                'current_epoch': self.current_epoch,
            },
            'test_metrics': {
                'dice': [], 'iou': [], 'precision': [], 'recall': [],
                'specificity': [], 'accuracy': [], 'hausdorff_95': []
            },
            'patient_features': {},  # patient_id -> features
            'total_samples': 0,
            'total_lesions': 0,
        }
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔬 開始測試並提取特徵")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"測試樣本數: {len(test_loader.dataset)}")
        self.logger.info(f"輸出目錄: {output_dir}")
        self.logger.info(f"提取深層特徵: {extract_deep_features}")
        self.logger.info(f"{'='*80}\n")
        
        pbar = tqdm(test_loader, desc="Testing & Extracting Features")
        
        for batch in pbar:
            self._current_batch_cache.clear()
            
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            patient_ids = batch['patient_id']
            slice_indices = batch['slice_index']
            bboxes = batch['bboxes']
            
            for i in range(len(images)):
                image = images[i]  # [3, H, W]
                gt_mask = masks[i]  # [1, H, W]
                bbox_tensor = bboxes[i]
                patient_id = str(patient_ids[i])
                slice_idx = int(slice_indices[i])
                
                if len(bbox_tensor) == 0:
                    continue
                
                # 計算 image embedding
                image_embedding, high_res_feats = self._prepare_image_features(image)
                bbox_tensor = bbox_tensor.to(self.device)
                
                # 初始化患者特徵
                if patient_id not in all_results['patient_features']:
                    all_results['patient_features'][patient_id] = {
                        'patient_id': patient_id,
                        'slices': {},
                        'summary': {}
                    }
                
                slice_features = {
                    'slice_index': slice_idx,
                    'lesions': [],
                    'metrics': {}
                }
                
                # 處理每個 bbox（病灶）
                all_pred_masks = []
                lesion_idx = 0
                
                for bbox in bbox_tensor:
                    if bbox.sum() == 0:
                        continue
                    
                    box_torch = bbox.unsqueeze(0)
                    
                    # Prompt Encoder
                    sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                        points=None,
                        boxes=box_torch,
                        masks=None,
                    )
                    
                    # Mask Decoder
                    low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        repeat_image=False,
                        high_res_features=high_res_feats,
                    )
                    
                    # 上採樣到原始大小
                    pred_mask = F.interpolate(
                        low_res_masks,
                        size=(gt_mask.shape[-2], gt_mask.shape[-1]),
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    pred_mask_squeezed = pred_mask.squeeze()
                    gt_mask_squeezed = gt_mask.squeeze()
                    
                    # 計算評估指標
                    metrics = compute_all_metrics(pred_mask_squeezed, gt_mask_squeezed)
                    
                    # 取得二值化預測 mask
                    pred_binary = (torch.sigmoid(pred_mask_squeezed) > 0.5).cpu().numpy()
                    all_pred_masks.append(pred_binary)
                    
                    # 原始影像（用於強度特徵計算）
                    original_image = image[0].cpu().numpy()  # 取第一個通道
                    
                    # 提取形態學特徵
                    morphological_features = feature_extractor.compute_morphological_features(
                        pred_binary, spacing
                    )
                    
                    # 提取強度特徵
                    intensity_features = feature_extractor.compute_intensity_features(
                        original_image, pred_binary
                    )
                    
                    # 提取深層特徵（可選）
                    deep_features = {}
                    if extract_deep_features:
                        deep_features = feature_extractor.extract_deep_features(
                            image_embedding,
                            sparse_embeddings,
                            dense_embeddings,
                            high_res_feats
                        )
                    
                    # IoU 預測分數作為置信度
                    confidence = float(iou_predictions.cpu().numpy().mean()) if iou_predictions is not None else 1.0
                    
                    # 聚合病灶特徵
                    lesion_feature = feature_extractor.aggregate_lesion_features(
                        morphological_features,
                        intensity_features,
                        deep_features,
                        confidence
                    )
                    lesion_feature['lesion_id'] = lesion_idx
                    lesion_feature['bbox'] = bbox.cpu().numpy().tolist()
                    lesion_feature['metrics'] = metrics
                    
                    slice_features['lesions'].append(lesion_feature)
                    lesion_idx += 1
                    all_results['total_lesions'] += 1
                    
                    # 累積測試指標
                    for key in all_results['test_metrics'].keys():
                        if key in metrics:
                            all_results['test_metrics'][key].append(metrics[key])
                
                # 計算切片級別指標（平均）
                if slice_features['lesions']:
                    slice_metrics = {}
                    for key in ['dice', 'iou', 'precision', 'recall']:
                        values = [l['metrics'].get(key, 0) for l in slice_features['lesions']]
                        slice_metrics[key] = float(np.mean(values))
                    slice_features['metrics'] = slice_metrics
                
                # 保存切片特徵
                all_results['patient_features'][patient_id]['slices'][slice_idx] = slice_features
                all_results['total_samples'] += 1
                
                # 保存預測遮罩（可選）
                if save_predictions and all_pred_masks:
                    pred_save_dir = output_dir / "predictions" / patient_id
                    pred_save_dir.mkdir(parents=True, exist_ok=True)
                    
                    combined_mask = np.zeros_like(all_pred_masks[0], dtype=np.uint8)
                    for idx, pm in enumerate(all_pred_masks):
                        combined_mask = np.maximum(combined_mask, pm.astype(np.uint8) * (idx + 1))
                    
                    np.save(pred_save_dir / f"slice_{slice_idx:04d}_pred.npy", combined_mask)
                
                # 保存可視化圖片（可選）
                if visualizer is not None and all_pred_masks:
                    # 準備原始影像（歸一化到 0-1）
                    original_image_np = image[0].cpu().numpy()  # 取第一個通道
                    if original_image_np.max() > 1.0:
                        original_image_np = (original_image_np - original_image_np.min()) / (original_image_np.max() - original_image_np.min() + 1e-8)
                    
                    # 合併所有預測遮罩
                    combined_pred = np.zeros_like(all_pred_masks[0], dtype=np.float32)
                    for pm in all_pred_masks:
                        combined_pred = np.maximum(combined_pred, pm.astype(np.float32))
                    
                    # GT mask
                    gt_mask_np = gt_mask.squeeze().cpu().numpy()
                    
                    # 計算切片級別的平均指標
                    slice_dice = slice_features['metrics'].get('dice', 0.0)
                    slice_iou = slice_features['metrics'].get('iou', 0.0)
                    
                    visualizer.save_slice_comparison(
                        patient_id=patient_id,
                        slice_idx=slice_idx,
                        original_image=original_image_np,
                        gt_mask=gt_mask_np,
                        pred_mask=combined_pred,
                        dice_score=slice_dice,
                        iou_score=slice_iou
                    )
                
                # 更新進度條
                pbar.set_postfix({
                    'patients': len(all_results['patient_features']),
                    'lesions': all_results['total_lesions']
                })
        
        # 計算患者級別摘要
        for patient_id, patient_data in all_results['patient_features'].items():
            patient_summary = self._compute_patient_summary(patient_data)
            all_results['patient_features'][patient_id]['summary'] = patient_summary
        
        # 生成患者摘要可視化圖（可選）
        if visualizer is not None:
            self.logger.info("📸 正在生成患者摘要可視化圖...")
            for patient_id in tqdm(all_results['patient_features'].keys(), desc="生成患者摘要圖"):
                visualizer.create_patient_summary_grid(patient_id)
            
            # 輸出可視化統計
            vis_stats = visualizer.get_statistics()
            self.logger.info(f"📊 可視化統計: 已生成 {vis_stats['total_images']} 張圖片，{vis_stats['total_patients']} 個患者")
        
        # 計算總體測試指標
        test_summary = {}
        for key, values in all_results['test_metrics'].items():
            if values:
                test_summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
        all_results['test_summary'] = test_summary
        
        # 保存結果
        self._save_features(all_results, output_dir)
        
        # 輸出摘要
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ 測試完成")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"總樣本數: {all_results['total_samples']}")
        self.logger.info(f"總病灶數: {all_results['total_lesions']}")
        self.logger.info(f"患者數: {len(all_results['patient_features'])}")
        if 'dice' in test_summary:
            self.logger.info(f"平均 Dice: {test_summary['dice']['mean']:.4f} ± {test_summary['dice']['std']:.4f}")
        if 'iou' in test_summary:
            self.logger.info(f"平均 IoU: {test_summary['iou']['mean']:.4f} ± {test_summary['iou']['std']:.4f}")
        if visualizer is not None:
            self.logger.info(f"可視化圖片已保存至: {visualizer.output_dir}")
        self.logger.info(f"特徵已保存至: {output_dir}")
        self.logger.info(f"{'='*80}\n")
        
        return all_results
    
    def _compute_patient_summary(self, patient_data: Dict) -> Dict:
        """
        計算患者級別的特徵摘要
        """
        slices = patient_data.get('slices', {})
        
        summary = {
            'total_slices': len(slices),
            'total_lesions': 0,
            'avg_lesion_area_mm2': 0.0,
            'max_lesion_area_mm2': 0.0,
            'avg_lesion_diameter_mm': 0.0,
            'max_lesion_diameter_mm': 0.0,
            'avg_circularity': 0.0,
            'avg_solidity': 0.0,
            'avg_confidence': 0.0,
            'metrics': {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
        }
        
        all_areas = []
        all_diameters = []
        all_circularities = []
        all_solidities = []
        all_confidences = []
        all_metrics = {k: [] for k in summary['metrics'].keys()}
        
        for slice_data in slices.values():
            for lesion in slice_data.get('lesions', []):
                summary['total_lesions'] += 1
                
                morph = lesion.get('morphological', {})
                all_areas.append(morph.get('area_mm2', 0))
                all_diameters.append(morph.get('equivalent_diameter_mm', 0))
                all_circularities.append(morph.get('circularity', 0))
                all_solidities.append(morph.get('solidity', 0))
                all_confidences.append(lesion.get('confidence', 0))
                
                for key in all_metrics.keys():
                    all_metrics[key].append(lesion.get('metrics', {}).get(key, 0))
        
        if all_areas:
            summary['avg_lesion_area_mm2'] = float(np.mean(all_areas))
            summary['max_lesion_area_mm2'] = float(np.max(all_areas))
            summary['avg_lesion_diameter_mm'] = float(np.mean(all_diameters))
            summary['max_lesion_diameter_mm'] = float(np.max(all_diameters))
            summary['avg_circularity'] = float(np.mean(all_circularities))
            summary['avg_solidity'] = float(np.mean(all_solidities))
            summary['avg_confidence'] = float(np.mean(all_confidences))
            
            for key, values in all_metrics.items():
                summary['metrics'][key] = float(np.mean(values))
        
        return summary
    
    def _save_features(self, results: Dict, output_dir: Path):
        """
        保存特徵到檔案
        
        生成多種格式：
        1. 完整 JSON（包含所有特徵）
        2. 患者級別獨立檔案（每個患者一個資料夾，包含 JSON 和 NPY 特徵）
        3. LLM 訓練用格式（簡化的文字描述 + 標籤）
        4. 測試摘要
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 將 numpy 類型轉換為 Python 原生類型
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        # 1. 保存完整結果（不含深層特徵向量以減小檔案大小）
        full_results_lite = self._create_lite_results(results)
        full_results_path = output_dir / f"full_features_{timestamp}.json"
        serializable_results = convert_to_serializable(full_results_lite)
        
        with open(full_results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"✅ 完整特徵已保存: {full_results_path}")
        
        # 2. 保存患者級別獨立檔案（每個患者一個資料夾）
        patient_base_dir = output_dir / "patients"
        patient_base_dir.mkdir(parents=True, exist_ok=True)
        
        for patient_id, patient_data in results['patient_features'].items():
            self._save_patient_features(
                patient_id, 
                patient_data, 
                patient_base_dir,
                timestamp
            )
        
        self.logger.info(f"✅ 患者獨立特徵已保存: {patient_base_dir}")
        
        # 3. 生成 LLM Fine-Tuning 用的訓練資料（每個患者一個檔案）
        llm_dir = output_dir / "llm_data"
        llm_dir.mkdir(parents=True, exist_ok=True)
        
        llm_training_data = self._generate_llm_training_data(results)
        
        # 保存整合版本
        llm_data_path = llm_dir / f"llm_training_data_all_{timestamp}.json"
        with open(llm_data_path, 'w', encoding='utf-8') as f:
            json.dump(llm_training_data, f, indent=2, ensure_ascii=False)
        
        # 保存每個患者獨立的 LLM 資料
        for sample in llm_training_data:
            patient_id = sample['patient_id']
            # 使用安全的檔名（替換特殊字符）
            safe_patient_id = patient_id.replace('.', '_').replace('/', '_')[:50]
            patient_llm_path = llm_dir / f"{safe_patient_id}_llm.json"
            with open(patient_llm_path, 'w', encoding='utf-8') as f:
                json.dump(sample, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ LLM 訓練資料已保存: {llm_dir}")
        
        # 4. 保存測試摘要
        summary_path = output_dir / f"test_summary_{timestamp}.json"
        summary = {
            'timestamp': results['timestamp'],
            'model_info': results['model_info'],
            'test_summary': results.get('test_summary', {}),
            'total_samples': results['total_samples'],
            'total_lesions': results['total_lesions'],
            'total_patients': len(results['patient_features']),
            'patient_list': list(results['patient_features'].keys()),
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.logger.info(f"✅ 測試摘要已保存: {summary_path}")
    
    def _create_lite_results(self, results: Dict) -> Dict:
        """
        創建不含深層特徵向量的輕量版結果（減小 JSON 檔案大小）
        """
        lite_results = {
            'timestamp': results['timestamp'],
            'model_info': results['model_info'],
            'test_metrics': results['test_metrics'],
            'test_summary': results.get('test_summary', {}),
            'total_samples': results['total_samples'],
            'total_lesions': results['total_lesions'],
            'patient_features': {}
        }
        
        for patient_id, patient_data in results['patient_features'].items():
            lite_patient = {
                'patient_id': patient_data.get('patient_id'),
                'summary': patient_data.get('summary', {}),
                'slices': {}
            }
            
            for slice_idx, slice_data in patient_data.get('slices', {}).items():
                lite_slice = {
                    'slice_index': slice_data.get('slice_index'),
                    'metrics': slice_data.get('metrics', {}),
                    'lesions': []
                }
                
                for lesion in slice_data.get('lesions', []):
                    # 排除深層特徵向量
                    lite_lesion = {
                        'lesion_id': lesion.get('lesion_id'),
                        'bbox': lesion.get('bbox'),
                        'confidence': lesion.get('confidence'),
                        'morphological': lesion.get('morphological', {}),
                        'intensity': lesion.get('intensity', {}),
                        'metrics': lesion.get('metrics', {}),
                        'text_description': lesion.get('text_description', ''),
                    }
                    lite_slice['lesions'].append(lite_lesion)
                
                lite_patient['slices'][slice_idx] = lite_slice
            
            lite_results['patient_features'][patient_id] = lite_patient
        
        return lite_results
    
    def _save_patient_features(
        self, 
        patient_id: str, 
        patient_data: Dict, 
        base_dir: Path,
        timestamp: str
    ):
        """
        保存單個患者的完整特徵到獨立資料夾
        
        結構:
        patients/
        └── {patient_id}/
            ├── metadata.json          # 患者基本資訊和摘要
            ├── features.json          # 完整特徵（不含向量）
            ├── deep_features.npz      # 深層特徵向量（NumPy 格式）
            ├── slices/
            │   ├── slice_{idx}_features.json
            │   └── slice_{idx}_deep.npy
            └── llm_input.txt          # LLM 輸入文字
        """
        # 使用安全的資料夾名稱
        safe_patient_id = patient_id.replace('.', '_').replace('/', '_')[:50]
        patient_dir = base_dir / safe_patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        slices_dir = patient_dir / "slices"
        slices_dir.mkdir(parents=True, exist_ok=True)
        
        # 將 numpy 類型轉換為 Python 原生類型
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        # 1. 保存患者 metadata
        metadata = {
            'patient_id': patient_id,
            'safe_patient_id': safe_patient_id,
            'timestamp': timestamp,
            'summary': patient_data.get('summary', {}),
            'total_slices': len(patient_data.get('slices', {})),
        }
        
        with open(patient_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(metadata), f, indent=2, ensure_ascii=False)
        
        # 2. 收集所有深層特徵向量
        all_deep_features = {
            'image_embeddings': [],      # 每個切片的 image embedding
            'sparse_embeddings': [],     # 每個病灶的 sparse embedding
            'dense_embeddings': [],      # 每個病灶的 dense embedding
            'slice_indices': [],         # 對應的切片索引
            'lesion_indices': [],        # 對應的病灶索引
        }
        
        # 3. 處理每個切片
        features_without_vectors = {
            'patient_id': patient_id,
            'summary': patient_data.get('summary', {}),
            'slices': {}
        }
        
        for slice_idx, slice_data in patient_data.get('slices', {}).items():
            slice_features = {
                'slice_index': slice_data.get('slice_index'),
                'metrics': slice_data.get('metrics', {}),
                'lesions': []
            }
            
            slice_deep_features = []
            
            for lesion in slice_data.get('lesions', []):
                # 提取深層特徵
                deep_feat = lesion.get('deep_features', {})
                
                if deep_feat:
                    # 保存深層特徵向量
                    lesion_deep = {}
                    
                    if 'image_embedding_global' in deep_feat:
                        img_emb = np.array(deep_feat['image_embedding_global'])
                        all_deep_features['image_embeddings'].append(img_emb)
                        lesion_deep['image_embedding'] = img_emb
                    
                    if 'sparse_embedding' in deep_feat:
                        sparse_emb = np.array(deep_feat['sparse_embedding'])
                        all_deep_features['sparse_embeddings'].append(sparse_emb)
                        lesion_deep['sparse_embedding'] = sparse_emb
                    
                    if 'dense_embedding_global' in deep_feat:
                        dense_emb = np.array(deep_feat['dense_embedding_global'])
                        all_deep_features['dense_embeddings'].append(dense_emb)
                        lesion_deep['dense_embedding'] = dense_emb
                    
                    # 高解析度特徵
                    for key in deep_feat:
                        if key.startswith('high_res_feat_') and key.endswith('_global'):
                            hr_emb = np.array(deep_feat[key])
                            lesion_deep[key] = hr_emb
                    
                    all_deep_features['slice_indices'].append(slice_idx)
                    all_deep_features['lesion_indices'].append(lesion.get('lesion_id', 0))
                    slice_deep_features.append(lesion_deep)
                
                # 不含向量的病灶特徵
                lesion_lite = {
                    'lesion_id': lesion.get('lesion_id'),
                    'bbox': lesion.get('bbox'),
                    'confidence': lesion.get('confidence'),
                    'morphological': lesion.get('morphological', {}),
                    'intensity': lesion.get('intensity', {}),
                    'metrics': lesion.get('metrics', {}),
                    'text_description': lesion.get('text_description', ''),
                    'feature_version': lesion.get('feature_version', '1.0'),
                }
                slice_features['lesions'].append(lesion_lite)
            
            features_without_vectors['slices'][slice_idx] = slice_features
            
            # 保存切片級別的深層特徵
            if slice_deep_features:
                slice_deep_path = slices_dir / f"slice_{slice_idx:04d}_deep.npz"
                np.savez_compressed(
                    slice_deep_path,
                    **{f"lesion_{i}_{k}": v 
                       for i, ld in enumerate(slice_deep_features) 
                       for k, v in ld.items()}
                )
        
        # 4. 保存完整特徵 JSON（不含向量）
        with open(patient_dir / "features.json", 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(features_without_vectors), f, indent=2, ensure_ascii=False)
        
        # 5. 保存聚合的深層特徵向量（NPZ 格式）
        if all_deep_features['image_embeddings']:
            deep_features_path = patient_dir / "deep_features.npz"
            
            # 轉換為 numpy array
            save_dict = {}
            
            if all_deep_features['image_embeddings']:
                save_dict['image_embeddings'] = np.array(all_deep_features['image_embeddings'])
            if all_deep_features['sparse_embeddings']:
                save_dict['sparse_embeddings'] = np.array(all_deep_features['sparse_embeddings'])
            if all_deep_features['dense_embeddings']:
                save_dict['dense_embeddings'] = np.array(all_deep_features['dense_embeddings'])
            
            save_dict['slice_indices'] = np.array(all_deep_features['slice_indices'])
            save_dict['lesion_indices'] = np.array(all_deep_features['lesion_indices'])
            
            # 計算聚合特徵（平均）
            if 'image_embeddings' in save_dict:
                save_dict['aggregated_image_embedding'] = np.mean(save_dict['image_embeddings'], axis=0)
            if 'sparse_embeddings' in save_dict:
                save_dict['aggregated_sparse_embedding'] = np.mean(save_dict['sparse_embeddings'], axis=0)
            if 'dense_embeddings' in save_dict:
                save_dict['aggregated_dense_embedding'] = np.mean(save_dict['dense_embeddings'], axis=0)
            
            np.savez_compressed(deep_features_path, **save_dict)
        
        # 6. 保存 LLM 輸入文字
        llm_input_text = self._generate_patient_description(patient_data)
        with open(patient_dir / "llm_input.txt", 'w', encoding='utf-8') as f:
            f.write(llm_input_text)
        
        # 7. 保存患者級別的 LLM 訓練資料
        summary = patient_data.get('summary', {})
        
        # 收集深層特徵向量
        deep_feature_vectors = []
        for slice_data in patient_data.get('slices', {}).values():
            for lesion in slice_data.get('lesions', []):
                deep_feat = lesion.get('deep_features', {})
                if 'image_embedding_global' in deep_feat:
                    deep_feature_vectors.append(deep_feat['image_embedding_global'])
        
        # 聚合深層特徵
        if deep_feature_vectors:
            avg_deep_features = np.mean(deep_feature_vectors, axis=0).tolist()
        else:
            avg_deep_features = []
        
        llm_data = {
            'patient_id': patient_id,
            'safe_patient_id': safe_patient_id,
            'input': llm_input_text,
            'numerical_features': {
                'total_lesions': summary.get('total_lesions', 0),
                'total_slices': summary.get('total_slices', 0),
                'avg_area_mm2': summary.get('avg_lesion_area_mm2', 0),
                'max_area_mm2': summary.get('max_lesion_area_mm2', 0),
                'avg_diameter_mm': summary.get('avg_lesion_diameter_mm', 0),
                'max_diameter_mm': summary.get('max_lesion_diameter_mm', 0),
                'avg_circularity': summary.get('avg_circularity', 0),
                'avg_solidity': summary.get('avg_solidity', 0),
                'avg_confidence': summary.get('avg_confidence', 0),
                'dice': summary.get('metrics', {}).get('dice', 0),
                'iou': summary.get('metrics', {}).get('iou', 0),
                'precision': summary.get('metrics', {}).get('precision', 0),
                'recall': summary.get('metrics', {}).get('recall', 0),
            },
            'deep_features_dim': len(avg_deep_features),
            'deep_features': avg_deep_features,
            'output': "",  # 預留：需從報告資料中獲取
            'metadata': {
                'timestamp': timestamp,
                'feature_version': '1.0',
            }
        }
        
        with open(patient_dir / "llm_training_sample.json", 'w', encoding='utf-8') as f:
            json.dump(llm_data, f, indent=2, ensure_ascii=False)
    
    def _generate_llm_training_data(self, results: Dict) -> List[Dict]:
        """
        生成 LLM Fine-Tuning 用的訓練資料
        
        格式：
        [
            {
                "patient_id": "xxx",
                "input": "病灶特徵描述...",
                "features": {...},  # 數值特徵
                "output": ""  # 預留給報告文字（需人工標註或從其他來源獲取）
            },
            ...
        ]
        """
        training_data = []
        
        for patient_id, patient_data in results['patient_features'].items():
            summary = patient_data.get('summary', {})
            
            # 生成輸入文字（病灶特徵描述）
            input_text = self._generate_patient_description(patient_data)
            
            # 收集數值特徵（用於 embedding）
            numerical_features = {
                'total_lesions': summary.get('total_lesions', 0),
                'avg_area_mm2': summary.get('avg_lesion_area_mm2', 0),
                'max_area_mm2': summary.get('max_lesion_area_mm2', 0),
                'avg_diameter_mm': summary.get('avg_lesion_diameter_mm', 0),
                'max_diameter_mm': summary.get('max_lesion_diameter_mm', 0),
                'avg_circularity': summary.get('avg_circularity', 0),
                'avg_solidity': summary.get('avg_solidity', 0),
                'avg_confidence': summary.get('avg_confidence', 0),
                'dice': summary.get('metrics', {}).get('dice', 0),
                'iou': summary.get('metrics', {}).get('iou', 0),
            }
            
            # 收集深層特徵向量（如果有的話）
            deep_feature_vectors = []
            for slice_data in patient_data.get('slices', {}).values():
                for lesion in slice_data.get('lesions', []):
                    deep_feat = lesion.get('deep_features', {})
                    if 'image_embedding_global' in deep_feat:
                        deep_feature_vectors.append(deep_feat['image_embedding_global'])
            
            # 聚合深層特徵（取平均）
            if deep_feature_vectors:
                avg_deep_features = np.mean(deep_feature_vectors, axis=0).tolist()
            else:
                avg_deep_features = []
            
            training_sample = {
                'patient_id': patient_id,
                'input': input_text,
                'numerical_features': numerical_features,
                'deep_features': avg_deep_features,
                'output': "",  # 預留：需從報告資料中獲取
                'metadata': {
                    'total_slices': summary.get('total_slices', 0),
                    'total_lesions': summary.get('total_lesions', 0),
                }
            }
            
            training_data.append(training_sample)
        
        return training_data
    
    def _generate_patient_description(self, patient_data: Dict) -> str:
        """
        生成患者的文字描述（用於 LLM 輸入）
        """
        summary = patient_data.get('summary', {})
        slices = patient_data.get('slices', {})
        
        description = f"# 胸部 CT 病灶分析報告\n\n"
        description += f"## 摘要\n"
        description += f"- 分析切片數：{summary.get('total_slices', 0)}\n"
        description += f"- 發現病灶數：{summary.get('total_lesions', 0)}\n"
        description += f"- 平均病灶面積：{summary.get('avg_lesion_area_mm2', 0):.2f} mm²\n"
        description += f"- 最大病灶面積：{summary.get('max_lesion_area_mm2', 0):.2f} mm²\n"
        description += f"- 平均病灶直徑：{summary.get('avg_lesion_diameter_mm', 0):.2f} mm\n"
        description += f"- 最大病灶直徑：{summary.get('max_lesion_diameter_mm', 0):.2f} mm\n"
        description += f"- 平均圓形度：{summary.get('avg_circularity', 0):.3f}\n"
        description += f"- 平均實心度：{summary.get('avg_solidity', 0):.3f}\n\n"
        
        description += f"## 詳細病灶資訊\n\n"
        
        for slice_idx, slice_data in sorted(slices.items()):
            lesions = slice_data.get('lesions', [])
            if not lesions:
                continue
            
            description += f"### 切片 {slice_idx}\n"
            for lesion in lesions:
                text_desc = lesion.get('text_description', '')
                if text_desc:
                    description += f"- {text_desc}\n"
            description += "\n"
        
        return description