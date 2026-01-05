"""
特徵保存模組

用於保存分割特徵到各種格式
"""

import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from .utils import convert_to_serializable
from .llm_data_generator import LLMDataGenerator


class FeatureSaver:
    """
    特徵保存器
    
    負責:
    - 保存完整測試結果
    - 保存患者級別特徵
    - 生成 LLM 訓練資料
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化特徵保存器
        
        Args:
            logger: 日誌記錄器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.llm_generator = LLMDataGenerator()
    
    def save_features(self, results: Dict, output_dir: Path):
        """
        保存特徵到檔案
        
        生成多種格式：
        1. 完整 JSON（包含所有特徵）
        2. 患者級別獨立檔案（每個患者一個資料夾，包含 JSON 和 NPY 特徵）
        3. LLM 訓練用格式（簡化的文字描述 + 標籤）
        4. 測試摘要
        
        Args:
            results: 完整測試結果
            output_dir: 輸出目錄
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存完整結果（不含深層特徵向量以減小檔案大小）
        self._save_full_results(results, output_dir, timestamp)
        
        # 2. 保存患者級別獨立檔案
        self._save_all_patient_features(results, output_dir, timestamp)
        
        # 3. 生成 LLM Fine-Tuning 用的訓練資料
        self._save_llm_training_data(results, output_dir, timestamp)
        
        # 4. 保存測試摘要
        self._save_test_summary(results, output_dir, timestamp)
    
    def _save_full_results(self, results: Dict, output_dir: Path, timestamp: str):
        """保存完整結果（輕量版）"""
        full_results_lite = self._create_lite_results(results)
        full_results_path = output_dir / f"full_features_{timestamp}.json"
        serializable_results = convert_to_serializable(full_results_lite)
        
        with open(full_results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"✅ 完整特徵已保存: {full_results_path}")
    
    def _save_all_patient_features(self, results: Dict, output_dir: Path, timestamp: str):
        """保存所有患者的獨立特徵"""
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
    
    def _save_llm_training_data(self, results: Dict, output_dir: Path, timestamp: str):
        """保存 LLM 訓練資料"""
        llm_dir = output_dir / "llm_data"
        llm_dir.mkdir(parents=True, exist_ok=True)
        
        llm_training_data = self.llm_generator.generate_training_data(results)
        
        # 保存整合版本
        llm_data_path = llm_dir / f"llm_training_data_all_{timestamp}.json"
        with open(llm_data_path, 'w', encoding='utf-8') as f:
            json.dump(llm_training_data, f, indent=2, ensure_ascii=False)
        
        # 保存每個患者獨立的 LLM 資料
        for sample in llm_training_data:
            patient_id = sample['patient_id']
            safe_patient_id = patient_id.replace('.', '_').replace('/', '_')[:50]
            patient_llm_path = llm_dir / f"{safe_patient_id}_llm.json"
            with open(patient_llm_path, 'w', encoding='utf-8') as f:
                json.dump(sample, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ LLM 訓練資料已保存: {llm_dir}")
    
    def _save_test_summary(self, results: Dict, output_dir: Path, timestamp: str):
        """保存測試摘要"""
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
        
        Args:
            results: 完整結果
            
        Returns:
            輕量版結果
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
        
        Args:
            patient_id: 患者 ID
            patient_data: 患者資料
            base_dir: 基礎目錄
            timestamp: 時間戳記
        """
        # 使用安全的資料夾名稱
        safe_patient_id = patient_id.replace('.', '_').replace('/', '_')[:50]
        patient_dir = base_dir / safe_patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        slices_dir = patient_dir / "slices"
        slices_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存患者 metadata
        self._save_patient_metadata(patient_id, patient_data, patient_dir, timestamp)
        
        # 2. 處理每個切片並收集深層特徵
        features_without_vectors, all_deep_features = self._process_slices(
            patient_id, patient_data, slices_dir
        )
        
        # 3. 保存完整特徵 JSON（不含向量）
        with open(patient_dir / "features.json", 'w', encoding='utf-8') as f:
            json.dump(
                convert_to_serializable(features_without_vectors), 
                f, indent=2, ensure_ascii=False
            )
        
        # 4. 保存聚合的深層特徵向量（NPZ 格式）
        self._save_deep_features(all_deep_features, patient_dir)
        
        # 5. 保存 LLM 輸入文字
        llm_input_text = self.llm_generator.generate_patient_description(patient_data)
        with open(patient_dir / "llm_input.txt", 'w', encoding='utf-8') as f:
            f.write(llm_input_text)
        
        # 6. 保存患者級別的 LLM 訓練資料
        llm_data = self.llm_generator.generate_patient_llm_data(
            patient_id, patient_data, timestamp
        )
        with open(patient_dir / "llm_training_sample.json", 'w', encoding='utf-8') as f:
            json.dump(llm_data, f, indent=2, ensure_ascii=False)
    
    def _save_patient_metadata(
        self, 
        patient_id: str, 
        patient_data: Dict, 
        patient_dir: Path, 
        timestamp: str
    ):
        """保存患者 metadata"""
        safe_patient_id = patient_id.replace('.', '_').replace('/', '_')[:50]
        
        metadata = {
            'patient_id': patient_id,
            'safe_patient_id': safe_patient_id,
            'timestamp': timestamp,
            'summary': patient_data.get('summary', {}),
            'total_slices': len(patient_data.get('slices', {})),
        }
        
        with open(patient_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(metadata), f, indent=2, ensure_ascii=False)
    
    def _process_slices(
        self, 
        patient_id: str, 
        patient_data: Dict, 
        slices_dir: Path
    ) -> tuple:
        """
        處理每個切片並收集深層特徵
        
        Returns:
            (features_without_vectors, all_deep_features)
        """
        all_deep_features = {
            'image_embeddings': [],
            'sparse_embeddings': [],
            'dense_embeddings': [],
            'slice_indices': [],
            'lesion_indices': [],
        }
        
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
                    lesion_deep = self._extract_lesion_deep_features(
                        deep_feat, all_deep_features, slice_idx, lesion
                    )
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
        
        return features_without_vectors, all_deep_features
    
    def _extract_lesion_deep_features(
        self, 
        deep_feat: Dict, 
        all_deep_features: Dict, 
        slice_idx: int, 
        lesion: Dict
    ) -> Dict:
        """提取病灶的深層特徵"""
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
        
        return lesion_deep
    
    def _save_deep_features(self, all_deep_features: Dict, patient_dir: Path):
        """保存聚合的深層特徵向量"""
        if not all_deep_features['image_embeddings']:
            return
        
        deep_features_path = patient_dir / "deep_features.npz"
        
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
            save_dict['aggregated_image_embedding'] = np.mean(
                save_dict['image_embeddings'], axis=0
            )
        if 'sparse_embeddings' in save_dict:
            save_dict['aggregated_sparse_embedding'] = np.mean(
                save_dict['sparse_embeddings'], axis=0
            )
        if 'dense_embeddings' in save_dict:
            save_dict['aggregated_dense_embedding'] = np.mean(
                save_dict['dense_embeddings'], axis=0
            )
        
        np.savez_compressed(deep_features_path, **save_dict)
