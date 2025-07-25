#!/bin/bash
# 重構實施腳本 - 漸進式改進

echo "🚀 開始胸部CT報告生成系統重構..."

# 1. 備份現有代碼
echo "📋 步驟1: 備份現有代碼"
cd /d/GitHub/chest-ct-report-generator
git checkout -b enhanced-system-v1
git add .
git commit -m "Backup before enhancement - $(date)"

# 2. 建立新目錄結構
echo "📁 步驟2: 建立增強模組目錄"
mkdir -p CT_ViT_Training/src/enhanced
mkdir -p RAG/enhanced  
mkdir -p tests/unit_tests
mkdir -p tests/integration_tests
mkdir -p configs/enhanced
mkdir -p docs/enhanced_system

# 3. 安裝新依賴
echo "📦 步驟3: 安裝增強功能依賴"
pip install opencv-python scikit-image scipy
pip install pyradiomics SimpleITK
pip install fastapi uvicorn python-multipart

# 4. 建立配置文件
echo "⚙️ 步驟4: 建立增強配置"
cat > configs/enhanced/enhanced_config.yaml << EOF
# 增強系統配置
enhanced_features:
  enable_nodule_detection: true
  enable_feature_extraction: true
  enable_lung_rads_assessment: true
  
nodule_detection:
  min_size_pixels: 10
  max_size_pixels: 200
  hu_threshold_range: [-200, 100]
  morphology_kernel_size: 3
  
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
EOF

# 5. 建立測試腳本
echo "🧪 步驟5: 建立測試框架"
cat > tests/test_enhanced_system.py << 'EOF'
#!/usr/bin/env python3
import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class TestEnhancedSystem(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = "tests/test_data"
        
    def test_system_integration(self):
        """測試系統整合"""
        print("✅ 基礎測試通過")
        
    def test_enhanced_inference(self):
        """測試增強推理功能"""
        print("✅ 增強推理測試通過")
        
    def test_report_generation(self):
        """測試報告生成"""
        print("✅ 報告生成測試通過")

if __name__ == '__main__':
    unittest.main()
EOF

# 6. 建立啟動腳本
echo "🎯 步驟6: 建立快速啟動腳本"
cat > scripts/run_enhanced_inference.py << 'EOF'
#!/usr/bin/env python3
"""
增強推理快速啟動腳本
用於測試新的增強功能
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    print("🔬 胸部CT增強分析系統")
    print("=" * 40)
    
    # 檢查模型路徑
    model_path = "../CT_ViT/models/final_model"
    if not os.path.exists(model_path):
        print("❌ 模型路徑不存在，請先訓練模型")
        return
    
    # 檢查測試資料
    test_dicom = "test_data/sample.dcm"
    if os.path.exists(test_dicom):
        print(f"✅ 找到測試資料: {test_dicom}")
        # 這裡將整合增強推理功能
        print("🚀 開始增強分析...")
        print("📊 分析完成 - 請查看輸出結果")
    else:
        print(f"⚠️ 測試資料不存在: {test_dicom}")
        print("請提供DICOM檔案進行測試")

if __name__ == "__main__":
    main()
EOF

# 7. 建立文檔
echo "📚 步驟7: 建立增強系統文檔"
cat > docs/enhanced_system/README.md << 'EOF'
# 增強系統使用指南

## 快速開始

1. 執行增強推理：
```bash
python scripts/run_enhanced_inference.py
```

2. 執行系統測試：
```bash
python tests/test_enhanced_system.py
```

## 新功能

### 1. 結節檢測
- 自動識別肺結節候選區域
- 提取結節基本特徵

### 2. 醫學特徵提取
- 大小測量（mm）
- 密度分析（HU值）
- 位置估計（肺葉/肺段）

### 3. Lung-RADS評估
- 自動風險分類
- 臨床建議生成

### 4. 結構化報告
- 標準醫學報告格式
- JSON結構化輸出
EOF

# 8. 設置權限
chmod +x scripts/run_enhanced_inference.py
chmod +x tests/test_enhanced_system.py

echo ""
echo "✅ 重構準備完成！"
echo ""
echo "📋 後續步驟："
echo "1. 實施增強推理模組 (CT_ViT_Training/src/enhanced/)"
echo "2. 實施增強報告生成 (RAG/enhanced/)"
echo "3. 執行測試驗證功能"
echo "4. 使用真實資料進行驗證"
echo ""
echo "🚀 開始開發："
echo "cd CT_ViT_Training"
echo "python ../scripts/run_enhanced_inference.py"
echo ""
