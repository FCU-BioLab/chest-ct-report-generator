# Dataset Splitter 使用說明

## 概述
`dataset_splitter.py` 已更新為支援 YOLO 格式的預處理資料結構。

## 資料結構要求

### 輸入格式 (來自 preprocess_original_dataset.py)
```
preprocessed_yolo_lesion/
├── A0001/
│   ├── images/              # PNG 圖像
│   │   ├── A0001_DCM_001_1-01.png
│   │   ├── A0001_DCM_002_1-02.png
│   │   └── ...
│   └── labels/              # YOLO 格式標註
│       ├── A0001_DCM_001_1-01.txt
│       ├── A0001_DCM_002_1-02.txt
│       └── ...
├── A0002/
│   ├── images/
│   └── labels/
└── ...
```

### 輸出格式 (劃分後)
```
splited_dataset/
├── train/
│   ├── A0001/
│   │   ├── images/
│   │   └── labels/
│   ├── A0003/
│   │   ├── images/
│   │   └── labels/
│   └── ...
├── test/
│   ├── A0002/
│   │   ├── images/
│   │   └── labels/
│   └── ...
├── train_patients.txt       # 訓練集患者列表
├── test_patients.txt        # 測試集患者列表
└── dataset_split_report.json # 劃分報告
```

## 使用方法

### 基本使用 (使用 config.json)
```bash
cd dataset_process
python dataset_splitter.py
```

### 自訂參數
```bash
python dataset_splitter.py \
    --source_dir ../datasets/preprocessed_yolo_lesion \
    --output_dir ../datasets/splited_dataset \
    --train_ratio 0.9 \
    --test_ratio 0.1 \
    --random_seed 42
```

### K-Fold 交叉驗證 (不同折)
```bash
# Fold 1
python dataset_splitter.py --random_seed 42 --output_dir ../datasets/fold_1

# Fold 2
python dataset_splitter.py --random_seed 43 --output_dir ../datasets/fold_2

# Fold 3
python dataset_splitter.py --random_seed 44 --output_dir ../datasets/fold_3
```

## 參數說明

| 參數 | 類型 | 預設值 | 說明 |
|-----|------|--------|------|
| `--source_dir` | str | config.json | 預處理後的 YOLO 格式資料目錄 |
| `--output_dir` | str | config.json | 劃分後的輸出目錄 |
| `--train_ratio` | float | 0.9 | 訓練集比例 (0.0-1.0) |
| `--test_ratio` | float | 0.1 | 測試集比例 (0.0-1.0) |
| `--random_seed` | int | 42 | 隨機種子 (用於可重現劃分) |
| `--config` | str | ../config.json | 配置文件路徑 |

## 主要變更

### 1. 掃描邏輯變更
**舊版 (DICOM 格式):**
```python
dicom_dir = patient_dir / "dicom_files"
xml_dir = patient_dir / "xml_annotations"

if dicom_dir.exists() and xml_dir.exists():
    # 處理患者
```

**新版 (YOLO 格式):**
```python
images_dir = patient_dir / "images"
labels_dir = patient_dir / "labels"

if images_dir.exists() and labels_dir.exists():
    # 處理患者
```

### 2. 功能保持不變
- ✅ 分層劃分 (按系列 A/B/E/G 分組)
- ✅ 比例控制 (train_ratio + test_ratio = 1.0)
- ✅ 隨機種子 (可重現劃分)
- ✅ 統計報告生成
- ✅ 患者列表文件

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

### Step 3: 驗證結果
```bash
# 查看劃分報告
cat ../datasets/splited_dataset/dataset_split_report.json

# 查看患者列表
cat ../datasets/splited_dataset/train_patients.txt
cat ../datasets/splited_dataset/test_patients.txt
```

## 輸出文件說明

### dataset_split_report.json
```json
{
  "random_seed": 42,
  "split_ratios": {
    "train": 0.9,
    "test": 0.1
  },
  "total_patients": 215,
  "series_distribution": {
    "A": 200,
    "B": 10,
    "E": 3,
    "G": 2
  },
  "splits": {
    "train": {
      "patients": ["A0001", "A0003", ...],
      "count": 193,
      "series_count": {
        "A": 180,
        "B": 9,
        "E": 3,
        "G": 1
      }
    },
    "test": {
      "patients": ["A0002", "A0004", ...],
      "count": 22,
      "series_count": {
        "A": 20,
        "B": 1,
        "E": 0,
        "G": 1
      }
    }
  }
}
```

### train_patients.txt / test_patients.txt
```
A0001
A0003
A0005
...
```

## 注意事項

1. **確保預處理完成**: 必須先執行 `preprocess_original_dataset.py`
2. **資料結構完整**: 每個患者目錄必須包含 `images/` 和 `labels/` 子目錄
3. **比例總和為 1**: `train_ratio + test_ratio` 必須等於 1.0
4. **隨機種子**: 使用相同的 `random_seed` 可重現相同的劃分結果
5. **分層劃分**: 自動按系列 (A/B/E/G) 進行分層，保持比例一致

## 故障排除

### 問題 1: "配置文件不存在"
```bash
# 解決方案: 指定正確的配置文件路徑
python dataset_splitter.py --config ../config.json
```

### 問題 2: "沒有找到有效的患者資料"
```bash
# 檢查源目錄結構
ls -l ../datasets/preprocessed_yolo_lesion/A0001/
# 應該看到 images/ 和 labels/ 目錄
```

### 問題 3: "比例總和必須為1.0"
```bash
# 確保比例正確
python dataset_splitter.py --train_ratio 0.9 --test_ratio 0.1
```

## 更新日期
2025-10-12
