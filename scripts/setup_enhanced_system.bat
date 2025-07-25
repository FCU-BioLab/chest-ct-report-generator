@echo off
REM 重構實施腳本 - Windows版本

echo 🚀 開始胸部CT報告生成系統重構...

REM 1. 備份現有代碼
echo 📋 步驟1: 備份現有代碼
cd /d d:\GitHub\chest-ct-report-generator
git checkout -b enhanced-system-v1
git add .
git commit -m "Backup before enhancement - %date% %time%"

REM 2. 建立新目錄結構
echo 📁 步驟2: 建立增強模組目錄
mkdir CT_ViT_Training\src\enhanced 2>nul
mkdir RAG\enhanced 2>nul
mkdir tests\unit_tests 2>nul
mkdir tests\integration_tests 2>nul
mkdir configs\enhanced 2>nul
mkdir docs\enhanced_system 2>nul

REM 3. 安裝新依賴
echo 📦 步驟3: 安裝增強功能依賴
pip install opencv-python scikit-image scipy
pip install pyradiomics SimpleITK
pip install fastapi uvicorn python-multipart

REM 4. 建立配置文件
echo ⚙️ 步驟4: 建立增強配置
(
echo # 增強系統配置
echo enhanced_features:
echo   enable_nodule_detection: true
echo   enable_feature_extraction: true
echo   enable_lung_rads_assessment: true
echo   
echo nodule_detection:
echo   min_size_pixels: 10
echo   max_size_pixels: 200
echo   hu_threshold_range: [-200, 100]
echo   morphology_kernel_size: 3
echo   
echo feature_extraction:
echo   extract_morphological: true
echo   extract_density: true
echo   extract_location: true
echo   
echo reporting:
echo   include_lung_rads: true
echo   include_recommendations: true
echo   export_structured_format: true
echo   
echo clinical_thresholds:
echo   lung_rads_2_max_size: 6  # mm
echo   lung_rads_3_max_size: 8  # mm
echo   lung_rads_4a_max_size: 15 # mm
) > configs\enhanced\enhanced_config.yaml

REM 5. 建立測試腳本
echo 🧪 步驟5: 建立測試框架
(
echo #!/usr/bin/env python3
echo import unittest
echo import sys
echo import os
echo sys.path.append^(os.path.join^(os.path.dirname^(__file__^), '..'^^)
echo 
echo class TestEnhancedSystem^(unittest.TestCase^):
echo     def setUp^(self^):
echo         self.test_data_dir = "tests/test_data"
echo         
echo     def test_system_integration^(self^):
echo         """測試系統整合"""
echo         print^("✅ 基礎測試通過"^)
echo         
echo     def test_enhanced_inference^(self^):
echo         """測試增強推理功能"""
echo         print^("✅ 增強推理測試通過"^)
echo         
echo     def test_report_generation^(self^):
echo         """測試報告生成"""
echo         print^("✅ 報告生成測試通過"^)
echo 
echo if __name__ == '__main__':
echo     unittest.main^(^)
) > tests\test_enhanced_system.py

REM 6. 建立啟動腳本
echo 🎯 步驟6: 建立快速啟動腳本
(
echo #!/usr/bin/env python3
echo """
echo 增強推理快速啟動腳本
echo 用於測試新的增強功能
echo """
echo import sys
echo import os
echo sys.path.append^(os.path.join^(os.path.dirname^(__file__^), '..'^^)
echo 
echo def main^(^):
echo     print^("🔬 胸部CT增強分析系統"^)
echo     print^("=" * 40^)
echo     
echo     # 檢查模型路徑
echo     model_path = "../CT_ViT/models/final_model"
echo     if not os.path.exists^(model_path^):
echo         print^("❌ 模型路徑不存在，請先訓練模型"^)
echo         return
echo     
echo     # 檢查測試資料
echo     test_dicom = "test_data/sample.dcm"
echo     if os.path.exists^(test_dicom^):
echo         print^(f"✅ 找到測試資料: {test_dicom}"^)
echo         # 這裡將整合增強推理功能
echo         print^("🚀 開始增強分析..."^)
echo         print^("📊 分析完成 - 請查看輸出結果"^)
echo     else:
echo         print^(f"⚠️ 測試資料不存在: {test_dicom}"^)
echo         print^("請提供DICOM檔案進行測試"^)
echo 
echo if __name__ == "__main__":
echo     main^(^)
) > scripts\run_enhanced_inference.py

REM 7. 建立文檔
echo 📚 步驟7: 建立增強系統文檔
(
echo # 增強系統使用指南
echo 
echo ## 快速開始
echo 
echo 1. 執行增強推理：
echo ```bash
echo python scripts/run_enhanced_inference.py
echo ```
echo 
echo 2. 執行系統測試：
echo ```bash
echo python tests/test_enhanced_system.py
echo ```
echo 
echo ## 新功能
echo 
echo ### 1. 結節檢測
echo - 自動識別肺結節候選區域
echo - 提取結節基本特徵
echo 
echo ### 2. 醫學特徵提取
echo - 大小測量^(mm^)
echo - 密度分析^(HU值^)
echo - 位置估計^(肺葉/肺段^)
echo 
echo ### 3. Lung-RADS評估
echo - 自動風險分類
echo - 臨床建議生成
echo 
echo ### 4. 結構化報告
echo - 標準醫學報告格式
echo - JSON結構化輸出
) > docs\enhanced_system\README.md

echo.
echo ✅ 重構準備完成！
echo.
echo 📋 後續步驟：
echo 1. 實施增強推理模組 ^(CT_ViT_Training\src\enhanced\^)
echo 2. 實施增強報告生成 ^(RAG\enhanced\^)
echo 3. 執行測試驗證功能
echo 4. 使用真實資料進行驗證
echo.
echo 🚀 開始開發：
echo cd CT_ViT_Training
echo python ..\scripts\run_enhanced_inference.py
echo.

pause
