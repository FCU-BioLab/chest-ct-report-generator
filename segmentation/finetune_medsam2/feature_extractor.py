#!/usr/bin/env python3
"""
病灶特徵提取模組
提供形態學、強度和深層特徵提取功能
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


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
            'extent': 0.0,
            'centroid_x': 0.0,
            'centroid_y': 0.0,
        }
        
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        if binary_mask.sum() == 0:
            return features
        
        px, py = spacing
        pixel_area = px * py
        
        labeled_mask, num_labels = ndimage.label(binary_mask)
        
        if num_labels == 0:
            return features
        
        region_sizes = ndimage.sum(binary_mask, labeled_mask, range(1, num_labels + 1))
        largest_label = np.argmax(region_sizes) + 1
        largest_region = (labeled_mask == largest_label).astype(np.uint8)
        
        props = sk_measure.regionprops(largest_region)
        if len(props) == 0:
            return features
        
        prop = props[0]
        
        features['area_pixels'] = prop.area
        features['area_mm2'] = prop.area * pixel_area
        
        contours = sk_measure.find_contours(largest_region, 0.5)
        if len(contours) > 0:
            largest_contour = max(contours, key=len)
            perimeter_pixels = len(largest_contour)
            features['perimeter_mm'] = perimeter_pixels * np.sqrt(px**2 + py**2) / 2
        
        features['equivalent_diameter_mm'] = np.sqrt(4 * features['area_mm2'] / np.pi)
        features['major_axis_mm'] = prop.major_axis_length * px
        features['minor_axis_mm'] = prop.minor_axis_length * py
        features['eccentricity'] = prop.eccentricity
        
        if features['perimeter_mm'] > 0:
            features['circularity'] = 4 * np.pi * features['area_mm2'] / (features['perimeter_mm'] ** 2)
        
        features['solidity'] = prop.solidity
        
        if features['area_mm2'] > 0:
            features['compactness'] = (features['perimeter_mm'] ** 2) / features['area_mm2']
        
        bbox = prop.bbox
        bbox_h = (bbox[2] - bbox[0]) * px
        bbox_w = (bbox[3] - bbox[1]) * py
        features['bbox_area_mm2'] = bbox_h * bbox_w
        
        if features['bbox_area_mm2'] > 0:
            features['extent'] = features['area_mm2'] / features['bbox_area_mm2']
        
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
            'contrast': 0.0,
        }
        
        binary_mask = (mask > 0.5).astype(bool)
        
        if binary_mask.sum() == 0:
            return features
        
        lesion_pixels = image[binary_mask]
        
        features['mean_intensity'] = float(np.mean(lesion_pixels))
        features['std_intensity'] = float(np.std(lesion_pixels))
        features['min_intensity'] = float(np.min(lesion_pixels))
        features['max_intensity'] = float(np.max(lesion_pixels))
        features['median_intensity'] = float(np.median(lesion_pixels))
        features['percentile_25'] = float(np.percentile(lesion_pixels, 25))
        features['percentile_75'] = float(np.percentile(lesion_pixels, 75))
        
        if len(lesion_pixels) > 2 and features['std_intensity'] > 1e-6:
            from scipy import stats
            features['skewness'] = float(stats.skew(lesion_pixels))
            features['kurtosis'] = float(stats.kurtosis(lesion_pixels))
        
        hist, _ = np.histogram(lesion_pixels, bins=64, density=True)
        hist = hist[hist > 0]
        if len(hist) > 0:
            features['entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
        
        background_mask = ~binary_mask
        if background_mask.sum() > 0:
            background_mean = np.mean(image[background_mask])
            features['contrast'] = float(features['mean_intensity'] - background_mean)
        
        return features
    
    @staticmethod
    def classify_nodule_type(image: np.ndarray, mask: np.ndarray) -> Dict:
        """
        根據 HU 值分佈分類結節類型
        
        結節類型分類標準（基於 Fleischner Society 指南）：
        - Solid nodule: 結節完全遮蓋肺實質
        - Part-solid nodule: 同時具有實性和磨玻璃成分
        - Ground-glass nodule (GGO): 不遮蓋肺實質，輕微增加密度
        - Calcified nodule: 含有高密度鈣化成分
        
        Args:
            image: CT 影像（HU 值）
            mask: 二值化遮罩
        
        Returns:
            結節類型分類結果字典
        """
        result = {
            'nodule_type': 'unknown',
            'nodule_type_chinese': '未知',
            'solid_percentage': 0.0,
            'ggo_percentage': 0.0,
            'calcified_percentage': 0.0,
            'mean_hu': 0.0,
            'confidence': 0.0,
            'description': ''
        }
        
        binary_mask = (mask > 0.5).astype(bool)
        
        if binary_mask.sum() == 0:
            return result
        
        nodule_pixels = image[binary_mask]
        total_pixels = len(nodule_pixels)
        
        if total_pixels == 0:
            return result
        
        mean_hu = float(np.mean(nodule_pixels))
        result['mean_hu'] = mean_hu
        
        # GGO: -700 ~ -400 HU
        ggo_mask = (nodule_pixels >= -700) & (nodule_pixels < -400)
        ggo_count = np.sum(ggo_mask)
        
        # Solid: -400 ~ 200 HU
        solid_mask = (nodule_pixels >= -400) & (nodule_pixels < 200)
        solid_count = np.sum(solid_mask)
        
        # Calcified: > 200 HU
        calcified_mask = nodule_pixels >= 200
        calcified_count = np.sum(calcified_mask)
        
        result['ggo_percentage'] = float(ggo_count / total_pixels * 100)
        result['solid_percentage'] = float(solid_count / total_pixels * 100)
        result['calcified_percentage'] = float(calcified_count / total_pixels * 100)
        
        # Classification logic
        if result['calcified_percentage'] > 50 or mean_hu > 200:
            result['nodule_type'] = 'calcified'
            result['nodule_type_chinese'] = '鈣化結節'
            result['confidence'] = min(result['calcified_percentage'] / 50, 1.0)
            result['description'] = f"鈣化結節，鈣化成分佔 {result['calcified_percentage']:.1f}%，平均 HU 值 {mean_hu:.1f}"
        
        elif result['solid_percentage'] > 80 and result['ggo_percentage'] < 10:
            result['nodule_type'] = 'solid'
            result['nodule_type_chinese'] = '實性結節'
            result['confidence'] = result['solid_percentage'] / 100
            result['description'] = f"實性結節，實性成分佔 {result['solid_percentage']:.1f}%，平均 HU 值 {mean_hu:.1f}"
        
        elif result['ggo_percentage'] > 80 and result['solid_percentage'] < 10:
            result['nodule_type'] = 'ground_glass'
            result['nodule_type_chinese'] = '磨玻璃結節'
            result['confidence'] = result['ggo_percentage'] / 100
            result['description'] = f"純磨玻璃結節 (pure GGO)，GGO 成分佔 {result['ggo_percentage']:.1f}%，平均 HU 值 {mean_hu:.1f}"
        
        elif result['ggo_percentage'] >= 10 and result['solid_percentage'] >= 10:
            result['nodule_type'] = 'part_solid'
            result['nodule_type_chinese'] = '部分實性結節'
            result['confidence'] = min((result['ggo_percentage'] + result['solid_percentage']) / 100, 1.0)
            result['description'] = (
                f"部分實性結節，實性成分佔 {result['solid_percentage']:.1f}%，"
                f"GGO 成分佔 {result['ggo_percentage']:.1f}%，平均 HU 值 {mean_hu:.1f}"
            )
        
        else:
            if mean_hu > -100:
                result['nodule_type'] = 'solid'
                result['nodule_type_chinese'] = '實性結節'
                result['confidence'] = 0.6
            elif mean_hu > -500:
                result['nodule_type'] = 'part_solid'
                result['nodule_type_chinese'] = '部分實性結節'
                result['confidence'] = 0.5
            else:
                result['nodule_type'] = 'ground_glass'
                result['nodule_type_chinese'] = '磨玻璃結節'
                result['confidence'] = 0.6
            result['description'] = f"根據平均 HU 值 ({mean_hu:.1f}) 判定為{result['nodule_type_chinese']}"
        
        return result
    
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
        
        if image_embedding is not None:
            img_global = torch.mean(image_embedding, dim=[2, 3])
            features['image_embedding_global'] = img_global.cpu().numpy().flatten().tolist()
            features['image_embedding_dim'] = img_global.shape[-1]
        
        if sparse_embeddings is not None:
            sparse_flat = sparse_embeddings.view(-1).cpu().numpy()
            features['sparse_embedding'] = sparse_flat.tolist()
            features['sparse_embedding_dim'] = len(sparse_flat)
        
        if dense_embeddings is not None:
            dense_global = torch.mean(dense_embeddings, dim=[2, 3])
            features['dense_embedding_global'] = dense_global.cpu().numpy().flatten().tolist()
            features['dense_embedding_dim'] = dense_global.shape[-1]
        
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
        confidence: float = 1.0,
        nodule_classification: Optional[Dict] = None
    ) -> Dict:
        """
        聚合所有類型的病灶特徵
        
        Args:
            morphological: 形態學特徵
            intensity: 強度特徵
            deep_features: 深層特徵
            confidence: 分割置信度
            nodule_classification: 結節類型分類結果
        
        Returns:
            聚合後的完整特徵字典
        """
        aggregated = {
            'morphological': morphological,
            'intensity': intensity,
            'deep_features': deep_features,
            'nodule_classification': nodule_classification or {},
            'confidence': confidence,
            'feature_version': '1.1',
        }
        
        description = self._generate_lesion_description(morphological, intensity, nodule_classification)
        aggregated['text_description'] = description
        
        return aggregated
    
    @staticmethod
    def _generate_lesion_description(
        morphological: Dict, 
        intensity: Dict, 
        nodule_classification: Optional[Dict] = None
    ) -> str:
        """生成病灶的文字描述（包含結節類型）"""
        area = morphological.get('area_mm2', 0)
        diameter = morphological.get('equivalent_diameter_mm', 0)
        circularity = morphological.get('circularity', 0)
        solidity = morphological.get('solidity', 0)
        mean_hu = intensity.get('mean_intensity', 0)
        std_hu = intensity.get('std_intensity', 0)
        
        nodule_type_chinese = "結節"
        if nodule_classification:
            nodule_type_chinese = nodule_classification.get('nodule_type_chinese', '結節')
        
        # Size classification
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
        
        # Shape classification
        if circularity > 0.8:
            shape_desc = "圓形"
        elif circularity > 0.6:
            shape_desc = "近圓形"
        elif circularity > 0.4:
            shape_desc = "橢圓形"
        else:
            shape_desc = "不規則形"
        
        # Border description
        if solidity > 0.9:
            border_desc = "邊界清晰光滑"
        elif solidity > 0.7:
            border_desc = "邊界較清晰"
        else:
            border_desc = "邊界不規則"
        
        description = f"發現{size_desc}{shape_desc}{nodule_type_chinese}，"
        description += f"等效直徑約 {diameter:.1f}mm，面積約 {area:.2f}mm²，"
        description += f"{border_desc}，"
        description += f"平均CT值 {mean_hu:.1f} HU，標準差 {std_hu:.1f} HU。"
        
        if nodule_classification:
            solid_pct = nodule_classification.get('solid_percentage', 0)
            ggo_pct = nodule_classification.get('ggo_percentage', 0)
            calc_pct = nodule_classification.get('calcified_percentage', 0)
            
            if calc_pct > 5:
                description += f" 含鈣化成分 {calc_pct:.1f}%。"
            if nodule_classification.get('nodule_type') == 'part_solid':
                description += f" 實性成分 {solid_pct:.1f}%，磨玻璃成分 {ggo_pct:.1f}%。"
        
        return description
