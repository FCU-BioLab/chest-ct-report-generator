# PreprocessedYOLODataset 更新說明

## 概述
`PreprocessedYOLODataset` 已更新，現在支援兩種資料結構：
1. **扁平結構** - 所有圖像和標註在同一層
2. **患者分組結構** - 按患者 ID 分組（由 `dataset_splitter.py` 生成）

## 支援的資料結構

### 1. 扁平結構
```
data_root/
├── train/
│   ├── images/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   └── labels/
│       ├── 000000.txt
│       ├── 000001.txt
│       └── ...
└── val/
    ├── images/
    └── labels/
```

### 2. 患者分組結構 ⭐ 新增支援
```
data_root/
├── train/
│   ├── A0001/
│   │   ├── images/
│   │   │   ├── A0001_DCM_001.png
│   │   │   └── ...
│   │   └── labels/
│   │       ├── A0001_DCM_001.txt
│   │       └── ...
│   ├── A0002/
│   │   ├── images/
│   │   └── labels/
│   └── ...
└── test/
    ├── A0001/
    │   ├── images/
    │   └── labels/
    └── ...
```

## 自動檢測機制

Dataset 類會自動檢測資料結構類型：

```python
def _detect_data_structure(self):
    # 檢查扁平結構
    if (split_dir / "images").exists() and (split_dir / "labels").exists():
        self.data_structure = "flat"
    
    # 檢查患者分組結構
    elif 存在患者目錄:
        self.data_structure = "grouped"
    
    else:
        raise ValueError("無法識別資料結構")
```

## 訓練腳本更新

### 驗證集自動分割
`train_yolov7_preprocessed.py` 現在會自動從訓練集分割驗證集：

```python
# 固定從訓練集分割驗證集
# - 預設使用 15% 的訓練資料作為驗證集
# - 保證資料分割的可重現性（使用 random_seed）
# - 不需要獨立的驗證集目錄
```

### 使用範例

#### 基本訓練（預設 15% 驗證集）
```bash
python train_yolov7_preprocessed.py \
    --data_dir ../../datasets/splited_dataset \
    --epochs 120 \
    --batch_size 24
```

#### 自訂驗證集比例
```bash
python train_yolov7_preprocessed.py \
    --data_dir ../../datasets/splited_dataset \
    --epochs 120 \
    --batch_size 24 \
    --val_ratio 0.2  # 使用 20% 作為驗證集
```

#### 完整配置
```bash
python train_yolov7_preprocessed.py \
    --data_dir ../../datasets/splited_dataset \
    --epochs 120 \
    --batch_size 24 \
    --val_ratio 0.15 \
    --use_medical_modules \
    --use_ema \
    --mixed_precision
```

## 完整工作流程

### Step 1: 預處理原始資料
```bash
cd detection/dataset_process
python preprocess_original_dataset.py \
    --data_root ../../datasets/all_patient_data \
    --output_dir ../../datasets/preprocessed_yolo_lesion \
    --enable_preprocessing False
```

### Step 2: 劃分資料集
```bash
cd ../../dataset_process
python dataset_splitter.py \
    --source_dir ../datasets/preprocessed_yolo_lesion \
    --output_dir ../datasets/splited_dataset \
    --train_ratio 0.9 \
    --test_ratio 0.1
```

### Step 3: 訓練模型
```bash
cd ../detection/yolo_detection
python train_yolov7_preprocessed.py \
    --data_dir ../../datasets/splited_dataset \
    --epochs 120 \
    --batch_size 24
```

## 日誌輸出範例

```
================================================================================
🚀 使用預處理資料訓練 YOLOv7
================================================================================
資料目錄:   ../../datasets/splited_dataset
訓練集:     train
驗證集比例: 15.0% (從訓練集分割)
Epochs:     120
Batch:      24 x 2 = 48
醫學模組:   ✓ 啟用
================================================================================
⚡ 預處理加速: 預期 8-10 倍訓練速度提升！
================================================================================
2025-10-12 14:42:07 - INFO - 使用設備: cuda:0
2025-10-12 14:42:07 - INFO - 載入預處理資料集...
2025-10-12 14:42:08 - INFO - 檢測到患者分組結構，共 193 位患者
2025-10-12 14:42:10 - INFO - PreprocessedYOLODataset 初始化完成:
2025-10-12 14:42:10 - INFO -   分割: train
2025-10-12 14:42:10 - INFO -   資料結構: grouped
2025-10-12 14:42:10 - INFO -   樣本數: 146236
2025-10-12 14:42:10 - INFO -   正樣本: 15598
2025-10-12 14:42:10 - INFO -   負樣本: 130638
2025-10-12 14:42:10 - INFO -   圖像尺寸: 640x640
2025-10-12 14:42:10 - INFO - 完整訓練集: 146236 樣本
2025-10-12 14:42:10 - INFO - 按比例 15.0% 分割驗證集
2025-10-12 14:42:10 - INFO - 訓練集: 124300 樣本 (85.0%)
2025-10-12 14:42:10 - INFO - 驗證集: 21936 樣本 (15.0%)
```

## 主要改進

### 1. 自動結構檢測 ✅
- 無需手動指定資料結構
- 自動適配扁平或分組結構

### 2. 靈活的驗證集處理 ✅
- 自動尋找 val → test
- 支援從訓練集分割
- 防止訓練中斷

### 3. 完整的錯誤提示 ✅
```python
ValueError: 無法識別資料結構。請確保資料目錄符合以下格式之一：
1. 扁平結構: data_root/train/images/ 和 data_root/train/labels/
2. 患者分組: data_root/train/patient_id/images/ 和 data_root/train/patient_id/labels/
```

### 4. 向後相容 ✅
- 原有的扁平結構完全支援
- 不影響現有程式碼

## 故障排除

### 問題 1: "圖像目錄不存在"
**原因**: 舊版本只支援扁平結構

**解決方案**: 更新到新版本
```bash
git pull origin main
```

### 問題 2: "無法識別資料結構"
**原因**: 資料目錄結構不符合要求

**檢查**:
```bash
# 扁平結構應該有
ls data_root/train/images/
ls data_root/train/labels/

# 或患者分組結構應該有
ls data_root/train/A0001/images/
ls data_root/train/A0001/labels/
```

### 問題 3: "驗證集目錄不存在"
**不再是問題！** 新版本會自動：
1. 嘗試使用 test 目錄
2. 從訓練集分割

## 性能對比

### 扁平結構
- ✅ 載入速度快
- ✅ 適合單一實驗
- ❌ K-Fold 交叉驗證較困難

### 患者分組結構
- ✅ 易於管理患者資料
- ✅ 支援 K-Fold 交叉驗證
- ✅ 清晰的資料組織
- ⚠️ 載入時需遍歷子目錄（影響很小）

## 建議用法

### 日常開發和測試
使用扁平結構（更簡單）：
```bash
preprocess_original_dataset.py → 直接輸出扁平結構
```

### 正式訓練和評估
使用患者分組結構（更嚴謹）：
```bash
preprocess_original_dataset.py → dataset_splitter.py → 患者分組結構
```

## 更新日期
2025-10-12
