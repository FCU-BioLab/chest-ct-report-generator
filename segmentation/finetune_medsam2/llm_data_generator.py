"""
LLM 訓練資料生成模組

用於從分割特徵生成 LLM Fine-Tuning 用的訓練資料
"""

import numpy as np
from typing import Dict, List
from pathlib import Path


class LLMDataGenerator:
    """
    LLM 訓練資料生成器
    
    負責:
    - 生成患者文字描述
    - 生成 LLM Fine-Tuning 訓練資料
    """
    
    # 結節類型中文名稱對照
    NODULE_TYPE_NAMES = {
        'solid': '實性結節',
        'part_solid': '部分實性結節',
        'ground_glass': '磨玻璃結節 (GGO)',
        'calcified': '鈣化結節',
        'unknown': '未分類'
    }
    
    def __init__(self):
        """初始化 LLM 資料生成器"""
        pass
    
    def generate_patient_description(self, patient_data: Dict) -> str:
        """
        生成患者的文字描述（用於 LLM 輸入，包含結節類型分佈）
        
        Args:
            patient_data: 患者資料字典，包含 summary 和 slices
            
        Returns:
            格式化的文字描述
        """
        summary = patient_data.get('summary', {})
        slices = patient_data.get('slices', {})
        
        description = f"# 胸部 CT 病灶分析報告\n\n"
        description += f"## 摘要\n"
        description += f"- 分析切片數：{summary.get('total_slices', 0)}\n"
        description += f"- 發現病灶數：{summary.get('total_lesions', 0)}\n"
        description += f"- 3D 結節數量：{summary.get('unique_nodules_3d', 0)}\n"
        description += f"- 平均病灶面積：{summary.get('avg_lesion_area_mm2', 0):.2f} mm²\n"
        description += f"- 最大病灶面積：{summary.get('max_lesion_area_mm2', 0):.2f} mm²\n"
        description += f"- 平均病灶直徑：{summary.get('avg_lesion_diameter_mm', 0):.2f} mm\n"
        description += f"- 最大病灶直徑：{summary.get('max_lesion_diameter_mm', 0):.2f} mm\n"
        description += f"- 平均圓形度：{summary.get('avg_circularity', 0):.3f}\n"
        description += f"- 平均實心度：{summary.get('avg_solidity', 0):.3f}\n\n"
        
        # 結節類型分佈
        nodule_type_counts = summary.get('nodule_type_counts', {})
        nodule_type_dist = summary.get('nodule_type_distribution', {})
        if nodule_type_counts and summary.get('total_lesions', 0) > 0:
            description += f"## 結節類型分佈\n"
            for ntype, count in nodule_type_counts.items():
                if count > 0:
                    pct = nodule_type_dist.get(ntype, 0)
                    type_name = self.NODULE_TYPE_NAMES.get(ntype, ntype)
                    description += f"- {type_name}：{count} 個 ({pct:.1f}%)\n"
            description += "\n"
        
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
    
    def generate_training_data(self, results: Dict) -> List[Dict]:
        """
        生成 LLM Fine-Tuning 用的訓練資料
        
        Args:
            results: 完整測試結果，包含 patient_features
            
        Returns:
            訓練資料列表，每個患者一個樣本
            
        格式：
        [
            {
                "patient_id": "xxx",
                "input": "病灶特徵描述...",
                "numerical_features": {...},
                "deep_features": [...],
                "output": "",  # 預留給報告文字
                "metadata": {...}
            },
            ...
        ]
        """
        training_data = []
        
        for patient_id, patient_data in results['patient_features'].items():
            summary = patient_data.get('summary', {})
            
            # 生成輸入文字（病灶特徵描述）
            input_text = self.generate_patient_description(patient_data)
            
            # 收集數值特徵（用於 embedding）
            numerical_features = {
                'total_lesions': summary.get('total_lesions', 0),
                'unique_nodules_3d': summary.get('unique_nodules_3d', 0),
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
            
            # 添加結節類型分佈
            nodule_type_counts = summary.get('nodule_type_counts', {})
            for ntype in ['solid', 'part_solid', 'ground_glass', 'calcified']:
                numerical_features[f'nodule_count_{ntype}'] = nodule_type_counts.get(ntype, 0)
            
            # 收集深層特徵向量
            deep_feature_vectors = self._collect_deep_features(patient_data)
            
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
                    'unique_nodules_3d': summary.get('unique_nodules_3d', 0),
                }
            }
            
            training_data.append(training_sample)
        
        return training_data
    
    def generate_patient_llm_data(
        self,
        patient_id: str,
        patient_data: Dict,
        timestamp: str
    ) -> Dict:
        """
        生成單個患者的 LLM 訓練資料
        
        Args:
            patient_id: 患者 ID
            patient_data: 患者資料
            timestamp: 時間戳記
            
        Returns:
            LLM 訓練資料字典
        """
        summary = patient_data.get('summary', {})
        safe_patient_id = patient_id.replace('.', '_').replace('/', '_')[:50]
        
        # 生成輸入文字
        input_text = self.generate_patient_description(patient_data)
        
        # 收集深層特徵向量
        deep_feature_vectors = self._collect_deep_features(patient_data)
        
        # 聚合深層特徵
        if deep_feature_vectors:
            avg_deep_features = np.mean(deep_feature_vectors, axis=0).tolist()
        else:
            avg_deep_features = []
        
        llm_data = {
            'patient_id': patient_id,
            'safe_patient_id': safe_patient_id,
            'input': input_text,
            'numerical_features': {
                'total_lesions': summary.get('total_lesions', 0),
                'total_slices': summary.get('total_slices', 0),
                'unique_nodules_3d': summary.get('unique_nodules_3d', 0),
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
        
        return llm_data
    
    def _collect_deep_features(self, patient_data: Dict) -> List[np.ndarray]:
        """
        從患者資料中收集深層特徵向量
        
        Args:
            patient_data: 患者資料
            
        Returns:
            深層特徵向量列表
        """
        deep_feature_vectors = []
        
        for slice_data in patient_data.get('slices', {}).values():
            for lesion in slice_data.get('lesions', []):
                deep_feat = lesion.get('deep_features', {})
                if 'image_embedding_global' in deep_feat:
                    deep_feature_vectors.append(deep_feat['image_embedding_global'])
        
        return deep_feature_vectors
