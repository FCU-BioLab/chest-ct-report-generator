# 測試檔案目錄

這個資料夾包含所有與檢測系統相關的測試和檢查工具。

## 📋 檔案說明

### 🧪 測試檔案

#### 核心功能測試
- **`test_detection.py`** - 主要檢測系統測試，包含完整的模型評估和深度特徵提取
- **`test_faster_rcnn.py`** - Faster R-CNN模型基本功能測試
- **`test_feature_extraction.py`** - 深度特徵提取系統測試

#### 特徵相關測試
- **`test_default_features.py`** - 預設特徵提取測試
- **`test_simple_features.py`** - 簡單特徵提取測試
- **`test_patient_folders.py`** - 病人資料夾結構測試

#### 系統性能測試
- **`test_gpu_usage.py`** - GPU使用狀況測試
- **`test_small_scale.py`** - 小規模資料測試
- **`test_visualization.py`** - 視覺化功能測試

### 🔍 檢查工具

#### 環境檢查
- **`check_gpu.py`** - GPU環境檢查工具
- **`check_data_size.py`** - 資料大小檢查工具
- **`check_limits_quick.py`** - 快速系統限制檢查

## 🚀 使用方法

### 運行主要測試
```bash
# 切換到detection目錄
cd detection

# 運行主要檢測測試
python test\test_detection.py --split test

# 運行特徵提取測試
python test\test_feature_extraction.py

# 檢查GPU狀態
python test\check_gpu.py
```

### 運行特定測試
```bash
# 測試深度特徵提取
python test\test_default_features.py

# 測試病人資料夾結構
python test\test_patient_folders.py

# 檢查資料大小
python test\check_data_size.py
```

## 📝 注意事項

1. **相對路徑問題**: 從 detection 目錄運行測試檔案，因為它們依賴相對路徑導入模組
2. **GPU需求**: 大部分測試需要GPU支持，運行前請確保GPU可用
3. **資料依賴**: 測試需要正確的資料集結構，請確保資料路徑正確

## 🔧 故障排除

如果遇到模組導入錯誤，請確保：
1. 從 detection 目錄運行測試
2. 所有必要的核心檔案（faster_rcnn_*.py, deep_feature_extractor.py等）在detection目錄中
3. Python路徑正確設置
