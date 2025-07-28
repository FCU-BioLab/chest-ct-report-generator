# MedSAM Demo for Chest Tumor Segmentation

這個專案展示如何使用 MedSAM (Medical Segment Anything Model) 對 `all_patient_data` 中的胸腔腫瘤病例進行分割。

## 功能特色

- 🏥 支援單一病例的腫瘤分割演示
- 📊 批次處理多個病例
- 🎯 基於 XML 標註的邊界框提示
- 📈 視覺化分割結果
- 📋 生成詳細的處理報告

## 檔案結構

```
CT_ViT_Training/
├── medsam_demo.py              # 主要的 MedSAM demo 類別
├── run_medsam_example.py       # 使用範例腳本
├── medsam_batch_processor.py   # 批次處理腳本
├── medsam_requirements.txt     # 額外的依賴套件
└── README_MedSAM.md           # 這個說明文件
```

## 安裝依賴

1. 首先安裝基本依賴：
```bash
pip install -r requirements.txt
```

2. 安裝 MedSAM 相關依賴：
```bash
pip install -r medsam_requirements.txt
```

## 使用方法

### 1. 簡單範例

運行基本演示：
```bash
cd CT_ViT_Training
python run_medsam_example.py
```

運行互動式演示：
```bash
python run_medsam_example.py --interactive
```

### 2. 單一病例處理

使用命令行處理特定病例：
```bash
# 處理病例 A0001 的第 0 個切片
python medsam_demo.py --patient_id A0001 --slice_index 0

# 列出所有可用的病例
python medsam_demo.py --list_patients

# 處理病例 A0005 的第 10 個切片，不保存結果
python medsam_demo.py --patient_id A0005 --slice_index 10 --no_save
```

### 3. 批次處理

處理多個病例：
```bash
# 處理前 5 個病例
python medsam_batch_processor.py --first_n 5

# 處理指定的病例
python medsam_batch_processor.py --patients A0001 A0002 A0003

# 處理病例範圍
python medsam_batch_processor.py --patient_range A0001-A0010

# 每個病例處理 3 個切片
python medsam_batch_processor.py --first_n 3 --slices_per_patient 3
```

## 程式架構

### MedSAMDemo 類別

核心功能類別，包含以下主要方法：

- `load_patient_data()`: 載入病例資料
- `load_dicom_image()`: 載入 DICOM 影像
- `parse_xml_annotation()`: 解析 XML 標註
- `segment_with_medsam()`: 使用 MedSAM 進行分割
- `visualize_results()`: 視覺化結果
- `demo_single_case()`: 單一病例演示

### 資料格式

程式支援以下資料結構：
```
all_patient_data/
├── A0001/
│   ├── dicom_files/           # DICOM 檔案
│   ├── xml_annotations/       # XML 標註檔案
│   ├── A0001_file_list.json   # 檔案清單
│   └── ...
├── A0002/
└── ...
```

## 輸出結果

### 視覺化輸出

程式會生成包含三個子圖的視覺化結果：
1. 原始影像
2. 帶邊界框標註的影像
3. MedSAM 分割結果

### 批次處理結果

批次處理會在 `medsam_results/` 目錄下生成：
```
medsam_results/
├── visualizations/            # 視覺化圖片
│   ├── A0001_slice000_segmentation.png
│   └── ...
└── reports/                   # 報告檔案
    ├── medsam_batch_results_YYYYMMDD_HHMMSS.json
    ├── medsam_batch_results_YYYYMMDD_HHMMSS.csv
    └── medsam_batch_results_YYYYMMDD_HHMMSS_summary.json
```

## 配置選項

### 模型配置

默認使用 `facebook/sam-vit-huge` 模型，可以通過以下方式修改：
```python
demo = MedSAMDemo(model_name="facebook/sam-vit-base")
```

### 支援的模型

- `facebook/sam-vit-huge` (默認，最佳效果)
- `facebook/sam-vit-large`
- `facebook/sam-vit-base` (最快速度)

## 注意事項

1. **GPU 支援**: 程式會自動檢測 CUDA 並使用 GPU 加速（如果可用）
2. **記憶體需求**: SAM 模型需要較多記憶體，建議至少 8GB RAM
3. **模型下載**: 首次運行時會自動下載 SAM 模型檔案
4. **標註格式**: 支援 Pascal VOC XML 格式的邊界框標註

## 故障排除

### 常見問題

1. **找不到病例資料**
   - 檢查 `all_patient_data` 目錄是否存在
   - 確認病例 ID 格式正確（如 A0001）

2. **模型載入失敗**
   - 檢查網路連接（需要下載模型）
   - 確認 PyTorch 和 transformers 版本兼容

3. **記憶體不足**
   - 嘗試使用較小的模型（如 sam-vit-base）
   - 減少批次處理的病例數量

### 調試模式

啟用詳細日誌：
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 範例輸出

```
🏥 MedSAM Demo for Chest Tumor Segmentation
==================================================

📊 Found 119 patients in the dataset

👥 Available patients (first 10):
  1. A0001
  2. A0002
  3. A0003
  ...

🎯 Selected patient for demo: A0001

📁 Patient data summary:
  - DICOM files: 20
  - Annotations: 20

🔍 Running MedSAM demo on slice 10...

✅ Demo completed successfully!
  - Patient: A0001
  - Slice: 10
  - Image shape: (512, 512, 3)
  - Tumors detected: 1
  - Masks generated: 1
  - Visualization saved: medsam_demo_A0001_slice10_20250728_143022.png

📋 Tumor annotations found:
  1. Type: A, BBox: (286, 310) -> (355, 402)
```

## 授權

本專案遵循原專案的授權條款。SAM 模型遵循 Apache 2.0 授權。
