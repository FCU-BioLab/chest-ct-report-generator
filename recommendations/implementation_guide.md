# 重構實施指南

## 🚀 立即可行的改進方案

基於您現有的CT-ViT系統，我們提供兩個改進路徑：

### 路徑A：漸進式改進（推薦開始）
在現有架構基礎上進行增量改進，風險較低，可立即開始。

### 路徑B：革命性重構  
完全重新設計架構，實現最大化改進，需要更多資源和時間。

## 📋 路徑A：漸進式改進計劃

### 第1週：增強現有分類系統

#### 1.1 改進特徵提取
在現有的`inference.py`中增加更詳細的特徵分析：

```python
# CT_ViT_Training/src/enhanced_inference.py
import cv2
import numpy as np
from typing import Dict, List, Tuple
from scipy import ndimage
from skimage import measure

class EnhancedCTViTInference:
    """增強版CT-ViT推理器"""
    
    def __init__(self, model_path: str):
        # 繼承原有的初始化
        super().__init__(model_path)
        self.nodule_detector = SimpleNoduleDetector()
        
    def enhanced_predict(self, image_path: str) -> Dict:
        """增強預測，包含結節檢測和特徵提取"""
        
        # 1. 原有分類預測
        base_result = self.predict_single_image(image_path, return_attention=True)
        
        # 2. 載入DICOM影像進行結節檢測
        dicom_processor = DICOMProcessor(self.config)
        pixel_array = dicom_processor.load_dicom_file(image_path)
        
        # 3. 簡單結節檢測
        nodules = self.nodule_detector.detect_nodules(pixel_array)
        
        # 4. 提取結節特徵
        nodule_features = []
        for nodule in nodules:
            features = self.extract_nodule_features(pixel_array, nodule)
            nodule_features.append(features)
        
        # 5. 整合結果
        enhanced_result = {
            **base_result,
            'nodules_detected': len(nodules),
            'nodule_details': nodule_features,
            'clinical_assessment': self.generate_clinical_assessment(base_result, nodule_features)
        }
        
        return enhanced_result
    
    def extract_nodule_features(self, ct_image: np.ndarray, nodule_bbox: Tuple) -> Dict:
        """提取結節特徵"""
        x, y, w, h = nodule_bbox
        nodule_region = ct_image[y:y+h, x:x+w]
        
        # 基本測量
        features = {
            'bbox': nodule_bbox,
            'size_mm': self.estimate_size_mm(w, h),
            'mean_hu': float(np.mean(nodule_region)),
            'std_hu': float(np.std(nodule_region)),
            'density_type': self.classify_density(nodule_region),
            'shape_features': self.extract_shape_features(nodule_region),
            'location': self.estimate_anatomical_location(x, y, ct_image.shape)
        }
        
        return features
    
    def classify_density(self, nodule_region: np.ndarray) -> str:
        """分類結節密度"""
        mean_hu = np.mean(nodule_region)
        
        if mean_hu > 100:
            return "solid_calcified"
        elif mean_hu > -200:
            return "solid"
        elif mean_hu > -500:
            return "part_solid"
        else:
            return "ground_glass"
    
    def generate_clinical_assessment(self, base_result: Dict, nodule_features: List[Dict]) -> Dict:
        """生成臨床評估"""
        assessment = {
            'overall_classification': base_result['predicted_label'],
            'confidence': base_result['confidence'],
            'lung_rads_assessment': self.assess_lung_rads(nodule_features),
            'recommendations': self.generate_recommendations(nodule_features)
        }
        
        return assessment
    
    def assess_lung_rads(self, nodule_features: List[Dict]) -> str:
        """評估Lung-RADS分類"""
        if not nodule_features:
            return "Lung-RADS 1 (No nodules detected)"
        
        # 找出最大的結節
        largest_nodule = max(nodule_features, key=lambda x: x['size_mm'])
        
        size_mm = largest_nodule['size_mm']
        density = largest_nodule['density_type']
        
        if size_mm < 6:
            return "Lung-RADS 2 (Benign appearance or behavior)"
        elif size_mm < 8:
            return "Lung-RADS 3 (Probably benign)"
        elif size_mm < 15:
            return "Lung-RADS 4A (Suspicious)"
        else:
            return "Lung-RADS 4B (Very suspicious)"

class SimpleNoduleDetector:
    """簡單結節檢測器（基於閾值和形態學）"""
    
    def detect_nodules(self, ct_slice: np.ndarray) -> List[Tuple]:
        """檢測結節候選區域"""
        # 肺窗調整
        windowed = self.apply_lung_window(ct_slice)
        
        # 二值化
        binary = self.threshold_lung_nodules(windowed)
        
        # 形態學處理
        processed = self.morphological_processing(binary)
        
        # 連通域分析
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        nodules = []
        for contour in contours:
            bbox = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # 篩選合理大小的區域
            if 50 < area < 5000:  # 調整閾值
                nodules.append(bbox)
        
        return nodules
    
    def apply_lung_window(self, ct_slice: np.ndarray) -> np.ndarray:
        """應用肺窗"""
        center, width = -600, 1200
        min_hu = center - width // 2
        max_hu = center + width // 2
        return np.clip(ct_slice, min_hu, max_hu)
    
    def threshold_lung_nodules(self, windowed_image: np.ndarray) -> np.ndarray:
        """結節區域閾值分割"""
        # 尋找軟組織密度區域
        normalized = ((windowed_image - windowed_image.min()) / 
                     (windowed_image.max() - windowed_image.min()) * 255).astype(np.uint8)
        
        # 自動閾值
        _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
```

#### 1.2 改進報告生成
增強現有的RAG系統：

```python
# RAG/enhanced_report_generator.py
import json
from typing import Dict, List
from datetime import datetime

class EnhancedReportGenerator:
    """增強版報告生成器"""
    
    def __init__(self):
        self.lung_rads_criteria = self.load_lung_rads_criteria()
        self.report_templates = self.load_report_templates()
    
    def generate_structured_report(self, analysis_result: Dict) -> Dict:
        """生成結構化報告"""
        
        report = {
            'report_header': self.generate_header(),
            'clinical_information': self.extract_clinical_info(analysis_result),
            'technique': "Non-contrast axial images of the chest were obtained.",
            'findings': self.generate_findings_section(analysis_result),
            'impression': self.generate_impression_section(analysis_result),
            'recommendations': self.generate_recommendations_section(analysis_result),
            'lung_rads_category': analysis_result.get('clinical_assessment', {}).get('lung_rads_assessment', 'N/A')
        }
        
        return report
    
    def generate_findings_section(self, analysis_result: Dict) -> str:
        """生成發現章節"""
        findings = []
        
        # 基本分類結果
        classification = analysis_result.get('predicted_label', 'Unknown')
        confidence = analysis_result.get('confidence', 0)
        
        findings.append(f"AI-based classification: {classification} series (confidence: {confidence:.2%})")
        
        # 結節發現
        nodules = analysis_result.get('nodule_details', [])
        if nodules:
            findings.append(f"\\n{len(nodules)} pulmonary nodule(s) detected:")
            
            for i, nodule in enumerate(nodules, 1):
                nodule_desc = self.describe_nodule(nodule)
                findings.append(f"{i}. {nodule_desc}")
        else:
            findings.append("\\nNo significant pulmonary nodules detected.")
        
        return "\\n".join(findings)
    
    def describe_nodule(self, nodule: Dict) -> str:
        """描述結節特徵"""
        size = nodule.get('size_mm', 0)
        density = nodule.get('density_type', 'unknown')
        location = nodule.get('location', 'unspecified location')
        
        description = f"A {size:.1f}mm {density.replace('_', ' ')} nodule in the {location}"
        
        # 添加HU值信息
        mean_hu = nodule.get('mean_hu', 0)
        description += f" (mean HU: {mean_hu:.0f})"
        
        return description
    
    def generate_impression_section(self, analysis_result: Dict) -> str:
        """生成印象章節"""
        nodules = analysis_result.get('nodule_details', [])
        lung_rads = analysis_result.get('clinical_assessment', {}).get('lung_rads_assessment', '')
        
        if not nodules:
            return "No significant pulmonary abnormalities detected."
        
        impression_parts = []
        
        # 主要發現
        if len(nodules) == 1:
            nodule = nodules[0]
            impression_parts.append(f"Solitary pulmonary nodule measuring {nodule.get('size_mm', 0):.1f}mm.")
        else:
            impression_parts.append(f"Multiple pulmonary nodules detected ({len(nodules)} total).")
        
        # Lung-RADS評估
        if lung_rads:
            impression_parts.append(lung_rads)
        
        return " ".join(impression_parts)
    
    def generate_recommendations_section(self, analysis_result: Dict) -> str:
        """生成建議章節"""
        nodules = analysis_result.get('nodule_details', [])
        
        if not nodules:
            return "Routine follow-up as clinically indicated."
        
        # 根據最大結節大小決定建議
        max_size = max(nodule.get('size_mm', 0) for nodule in nodules)
        
        if max_size < 6:
            return "No follow-up required. Routine screening as appropriate."
        elif max_size < 8:
            return "Follow-up CT recommended in 12 months."
        elif max_size < 15:
            return "Follow-up CT recommended in 6 months."
        else:
            return "Consider chest CT with contrast and/or PET-CT. Multidisciplinary consultation recommended."
```

### 第2-3週：整合改進功能

#### 2.1 修改主要推理腳本
更新`CT_ViT_Training/inference.py`：

```python
# 在現有inference.py中添加
def enhanced_inference_mode(args):
    """增強推理模式"""
    from src.enhanced_inference import EnhancedCTViTInference
    from RAG.enhanced_report_generator import EnhancedReportGenerator
    
    # 初始化增強推理器
    enhanced_inferencer = EnhancedCTViTInference(args.model_path)
    report_generator = EnhancedReportGenerator()
    
    if args.mode == 'enhanced_single':
        # 單張影像增強分析
        result = enhanced_inferencer.enhanced_predict(args.input)
        
        # 生成結構化報告
        structured_report = report_generator.generate_structured_report(result)
        
        # 輸出結果
        print("=== Enhanced Analysis Results ===")
        print(f"Classification: {result['predicted_label']} ({result['confidence']:.2%})")
        print(f"Nodules detected: {result['nodules_detected']}")
        print(f"Lung-RADS: {result['clinical_assessment']['lung_rads_assessment']}")
        
        print("\\n=== Structured Report ===")
        print(structured_report['findings'])
        print(f"\\nImpression: {structured_report['impression']}")
        print(f"Recommendations: {structured_report['recommendations']}")
        
        # 保存詳細結果
        if args.output:
            output_file = os.path.join(args.output, f"enhanced_analysis_{os.path.basename(args.input)}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'analysis_result': result,
                    'structured_report': structured_report
                }, f, indent=2, ensure_ascii=False)
            print(f"\\nDetailed results saved to: {output_file}")

# 更新主函數
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced CT-ViT Inference")
    parser.add_argument("--mode", choices=["single", "batch", "evaluate", "enhanced_single"], 
                       default="enhanced_single", help="Inference mode")
    # ... 其他參數保持不變
    
    args = parser.parse_args()
    
    if args.mode == "enhanced_single":
        enhanced_inference_mode(args)
    else:
        # 原有的推理邏輯
        main(args)
```

#### 2.2 建立配置管理
建立新的配置檔案：

```yaml
# CT_ViT_Training/configs/enhanced_config.yaml
# 增強配置
enhanced_features:
  enable_nodule_detection: true
  enable_feature_extraction: true
  enable_lung_rads_assessment: true
  
nodule_detection:
  min_size_pixels: 10
  max_size_pixels: 200
  hu_threshold_range: [-200, 100]
  
feature_extraction:
  extract_morphological: true
  extract_density: true
  extract_location: true
  
reporting:
  include_lung_rads: true
  include_recommendations: true
  export_structured_format: true
  
clinical_thresholds:
  lung_rads_2_max_size: 6  # mm
  lung_rads_3_max_size: 8  # mm
  lung_rads_4a_max_size: 15 # mm
```

### 第4週：測試與驗證

#### 測試腳本
```python
# tests/test_enhanced_system.py
import unittest
import os
import json
from CT_ViT_Training.src.enhanced_inference import EnhancedCTViTInference
from RAG.enhanced_report_generator import EnhancedReportGenerator

class TestEnhancedSystem(unittest.TestCase):
    
    def setUp(self):
        self.model_path = "../CT_ViT/models/final_model"
        self.test_dicom = "test_data/sample.dcm"
        self.inferencer = EnhancedCTViTInference(self.model_path)
        self.report_generator = EnhancedReportGenerator()
    
    def test_enhanced_prediction(self):
        """測試增強預測功能"""
        result = self.inferencer.enhanced_predict(self.test_dicom)
        
        # 檢查必要欄位
        self.assertIn('predicted_label', result)
        self.assertIn('nodules_detected', result)
        self.assertIn('clinical_assessment', result)
        
        # 檢查臨床評估
        clinical = result['clinical_assessment']
        self.assertIn('lung_rads_assessment', clinical)
        self.assertIn('recommendations', clinical)
    
    def test_report_generation(self):
        """測試報告生成"""
        # 模擬分析結果
        mock_result = {
            'predicted_label': 'A',
            'confidence': 0.85,
            'nodules_detected': 1,
            'nodule_details': [{
                'size_mm': 8.5,
                'density_type': 'solid',
                'location': 'right upper lobe',
                'mean_hu': 45
            }],
            'clinical_assessment': {
                'lung_rads_assessment': 'Lung-RADS 4A (Suspicious)'
            }
        }
        
        report = self.report_generator.generate_structured_report(mock_result)
        
        # 檢查報告結構
        required_sections = ['findings', 'impression', 'recommendations']
        for section in required_sections:
            self.assertIn(section, report)
            self.assertIsInstance(report[section], str)
            self.assertTrue(len(report[section]) > 0)

if __name__ == '__main__':
    unittest.main()
```

## 🎯 立即可執行的步驟

### 今天就可以開始：

1. **備份現有代碼**
```bash
cd d:\GitHub\chest-ct-report-generator
git checkout -b enhanced-system-v1
git add .
git commit -m "Backup before enhancement"
```

2. **建立增強模組**
```bash
mkdir CT_ViT_Training\src\enhanced
mkdir RAG\enhanced
mkdir tests
```

3. **逐步實施改進**
   - 先實現`SimpleNoduleDetector`
   - 再改進`EnhancedCTViTInference`
   - 最後整合`EnhancedReportGenerator`

4. **測試驗證**
   - 使用現有的測試資料驗證功能
   - 比較改進前後的結果差異

## 📊 預期改進效果

### 短期改進（1個月內）
- **結節檢測**: 基本的結節候選區域檢測
- **特徵提取**: 大小、密度、位置基本特徵
- **Lung-RADS評估**: 簡化版本的風險分類
- **結構化報告**: 標準化的報告格式

### 長期改進（3-6個月）
- **精確檢測**: 實現高精度的結節檢測和分割
- **完整特徵**: 形態學、放射組學特徵提取
- **臨床標準**: 完整的Lung-RADS和Fleischner標準
- **專業報告**: 符合放射科標準的報告生成

這個漸進式改進方案可以讓您在不破壞現有系統的前提下，逐步提升系統的臨床實用性和準確性。每個改進步驟都是獨立的，可以根據您的時間和資源安排靈活實施。
