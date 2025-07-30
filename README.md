# 🏥 胸部CT報告生成系## 📋 目錄
- [系統特色](#系統特色)
- [系統性能](#系統性能) 
- [資料集概況](#資料集概├── 🧬 medsam├── 🛠️ scripts/                # 系統腳本
├── config.json                # 系統配置
├── requirements.txt           # 依賴包列表
├── dataset_analysis_report.md # 資料集分析報告
├── dataset_analysis_detailed.json # 詳細資料集分析
├── CT_ViT_Detection_Upgrade_Guide.md # 升級指南
├── PROJECT_SUMMARY.md         # 專案總結
└── README.md                  # 主要說明文件entation/     # MedSAM2分割系統)
- [專案結構](#專案結構)
- [技術架構](#技術架構)
- [快速開始](#快速開始)
- [配置說明](#配置說明)
- [故障排除](#故障排除)

## 📈 系統性能t CT Report Generation System

一個完整的端到端胸部CT影像分析和醫療報告生成系統，整合了CT-ViT目標檢測、MedSAM2分割、RAG知識檢索和智能報告生成功能。

## 🌟 系統特色

✅ **CT-ViT目標檢測** - 基於Vision Transformer的精確腫瘤檢測與分類  
✅ **MedSAM2醫學分割** - 專業醫學影像分割，提供精確腫瘤邊界  
✅ **多模態分析** - 整合DICOM影像與XML標註資料  
✅ **RAG智能報告** - 結合知識檢索生成專業醫學報告  
✅ **標準化輸出** - 支援JSON/HTML/PDF多格式報告  
✅ **模組化架構** - 完整的訓練、評估、推理流程

## � 目錄
- [系統特色](#系統特色)
- [系統性能](#系統性能) 
- [資料集概況](#資料集概況)
- [專案結構](#專案結構)
- [技術架構](#技術架構)
- [快速開始](#快速開始)
- [使用指南](#使用指南)
- [配置說明](#配置說明)
- [故障排除](#故障排除)

| 指標 | 當前CT-ViT | 升級後檢測模型 | 提升幅度 |
|------|------------|---------------|----------|
| 分類準確度 | 63% | 85-90% | +22-27% |
| 檢測mAP | N/A | 75-80% | 新功能 |
| 空間定位 | ❌ | ✅ | 新功能 |
| 處理速度 | 0.5s/影像 | 0.8s/影像 | 可接受 |

## 🗂️ 資料集概況

- **患者數量**: 352位患者 (A:248, B:38, E:5, G:61)
- **影像數量**: 30,382組DICOM-XML配對
- **標註格式**: Pascal VOC XML格式，包含邊界框座標
- **影像品質**: 高品質胸部CT掃描，適合深度學習訓練

## 📁 專案結構

```
chest-ct-report-generator/
├── 🎯 CT_ViT_Training/          # 核心訓練模組 (主要工作區域)
│   ├── src/                     # 源代碼模組
│   │   ├── config.py           # 配置管理
│   │   ├── data_processing.py  # 數據處理
│   │   ├── detection_dataset.py # 檢測資料集
│   │   ├── detection_model.py  # 檢測模型
│   │   ├── model.py           # 分類模型
│   │   └── utils.py           # 工具函數
│   ├── configs/               # 配置文件
│   ├── scripts/               # 運行腳本
│   ├── CT_ViT_Detection/      # 檢測模型輸出目錄
│   ├── train.py              # 🏋️ 分類模型訓練
│   ├── train_detection.py    # 🎯 檢測模型訓練
│   ├── evaluate_model.py     # 📊 模型評估
│   ├── unified_evaluator.py  # 🔄 統一評估器
│   ├── inference.py          # 🔮 分類推理
│   ├── inference_detection.py # 🔍 檢測推理
│   ├── test_system.py        # 🧪 系統測試
│   └── README.md             # 訓練模組說明
├── 🧬 medsam2_segmentation/     # MedSAM2分割系統
│   ├── sam_seg.py             # 分割主程式
│   ├── test_medsam2_setup.py  # 環境測試
│   ├── README_MEDSAM2.md      # MedSAM2使用說明
│   └── MedSAM2/               # MedSAM2模型檔案
│       └── checkpoints/       # 模型檢查點
├── 🤖 LLM/                     # 語言模型與RAG系統
│   ├── RAG/                   # RAG報告生成
│   └── Fine_Tune/             # 模型微調
├── 📊 datasets/                # 資料集目錄
│   ├── all_patient_data/      # 處理後的患者資料
│   └── original_datasets/     # 原始資料集
├── 📁 dataset_process/         # 資料處理工具
├── 📄 report_data/            # 報告相關資料
├── 🛠️ scripts/                # 系統腳本
├── config.json                # 系統配置
├── requirements.txt           # 依賴包列表
├── dataset_analysis_report.md # 資料集分析報告
├── dataset_analysis_detailed.json # 詳細資料集分析
├── CT_ViT_Detection_Upgrade_Guide.md # 升級指南
├── PROJECT_SUMMARY.md         # 專案總結
└── README.md                  # 主要說明文件
```
├── � medsam2_segmentation/     # MedSAM2分割系統
│   ├── sam_seg.py             # 分割主程式
│   ├── test_medsam2_setup.py  # 環境測試
│   ├── README_MEDSAM2.md      # MedSAM2使用說明
│   └── MedSAM2/               # MedSAM2模型檔案
├── 🤖 LLM/                     # 語言模型與RAG系統
│   ├── RAG/                   # RAG報告生成
│   └── Fine_Tune/             # 模型微調
├── 📊 datasets/                # 資料集目錄
│   ├── all_patient_data/      # 處理後的患者資料
│   └── original_datasets/     # 原始資料集
├── 📁 dataset_process/         # 資料處理工具
├── 📄 report_data/            # 報告相關資料
├── �️ scripts/                # 系統腳本
├── config.json                # 系統配置
├── requirements.txt           # 依賴包列表
├── dataset_analysis_report.md # 資料集分析報告
├── CT_ViT_Detection_Upgrade_Guide.md # 升級指南
└── README.md                  # 主要說明文件
```

### 🚀 主要功能模組

#### 1. 核心訓練與評估 (CT_ViT_Training/)
- **分類訓練**: `train.py` - 原始CT-ViT分類模型
- **檢測訓練**: `train_detection.py` - 升級後的目標檢測模型
- **模型評估**: `evaluate_model.py`, `unified_evaluator.py`
- **推理功能**: `inference.py`, `inference_detection.py`
- **系統測試**: `test_system.py` - 完整系統功能驗證

#### 2. 醫學影像分割系統 (medsam2_segmentation/)
- **MedSAM2分割**: `sam_seg.py` - 專業醫學影像分割
- **環境測試**: `test_medsam2_setup.py` - MedSAM2環境驗證
- **分割模型**: `MedSAM2/` - 預訓練的醫學分割模型

#### 3. 數據處理工具 (dataset_process/)
- **資料處理**: 包含各種資料預處理和轉換工具
- **DICOM處理**: 醫學影像格式處理功能

#### 4. 智能報告生成系統 (LLM/)
- **RAG系統**: `RAG/` - 基於檢索增強的報告生成
- **模型微調**: `Fine_Tune/` - 語言模型微調功能

#### 5. 資料集管理 (datasets/)
本專案使用 **Lung-PET-CT-Dx** 資料集，以下是資料集的組織結構：

##### 📂 資料集結構
```
datasets/
├── original_datasets/         # 原始資料集存放
│   └── Lung-PET-CT-Dx/       # Lung-PET-CT-Dx資料集
│       ├── manifest-*/        # DICOM影像檔案
│       └── Annotations/       # XML標註檔案
└── all_patient_data/          # 處理後的患者資料
    ├── A0001/                 # 患者資料夾
    ├── A0002/
    └── ...
```

##### 🔧 核心處理程式
- **資料載入處理**: [`CT_ViT_Training/src/data_processing.py`](CT_ViT_Training/src/data_processing.py)
  - `DICOMProcessor` 類：DICOM文件讀取與前處理
  - HU值轉換與肺窗調整（中心：-600，寬度：1200）
  - `CTDataset` 類：PyTorch資料集封裝

- **檢測資料處理**: [`CT_ViT_Training/src/detection_dataset.py`](CT_ViT_Training/src/detection_dataset.py)
  - `XMLAnnotationParser` 類：Pascal VOC格式XML解析
  - 邊界框座標提取與類別映射
  - `CTDetectionDataset` 類：目標檢測任務資料集

##### 📊 數據處理流程
```
原始資料集 (datasets/original_datasets/)
    ↓ 資料處理工具
已處理資料 (datasets/all_patient_data/)  
    ↓ data_processing.py / detection_dataset.py
PyTorch資料集 → 模型訓練/推理
```

#### 6. 系統配置與文檔
- **系統配置**: `config.json` - 主要系統配置文件
- **升級指南**: `CT_ViT_Detection_Upgrade_Guide.md` - 系統升級說明
- **資料分析**: `dataset_analysis_report.md` - 資料集統計分析
- **專案總結**: `PROJECT_SUMMARY.md` - 專案概覽和總結

- **檢測資料處理**: [`CT_ViT_Training/src/detection_dataset.py`](CT_ViT_Training/src/detection_dataset.py)
  - `XMLAnnotationParser` 類：Pascal VOC格式XML解析
  - 邊界框座標提取與類別映射
  - `CTDetectionDataset` 類：目標檢測任務資料集

##### 📊 數據處理流程
```
原始資料集 (datasets/original_datasets/)
    ↓ 資料處理工具
已處理資料 (datasets/all_patient_data/)  
    ↓ dataset_splitter.py
分割資料集 (datasets/splits/)
    ↓ data_processing.py / detection_dataset.py
PyTorch資料集 → 模型訓練/推理
```

#### 6. 系統測試與配置
- **系統測試**: `CT_ViT_Training/test_system.py` - 完整系統測試
- **配置管理**: `config.json` - 系統配置文件
- **升級指南**: `CT_ViT_Detection_Upgrade_Guide.md` - 系統升級說明
- **資料分析**: `dataset_analysis_report.md` - 資料集統計分析

## 🏗️ 技術架構

### 系統架構概覽
```
                  胸部CT報告生成系統架構
    ┌─────────────────────────────────────────────────────────────┐
    │                     輸入層 (Input Layer)                    │
    ├─────────────────────────────────────────────────────────────┤
    │  📁 DICOM檔案  │  📋 XML標註  │  ⚙️ 配置檔案  │  👤 患者資料  │
    └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   預處理層 (Preprocessing)                  │
    ├─────────────────────────────────────────────────────────────┤
    │  🔄 DICOM解析     │  📏 影像正規化  │  🖼️ 尺寸調整  │  📊 品質檢查 │
    │  • pydicom      │  • 像素正規化   │  • 224x224    │  • 對比度   │
    │  • 元資料提取    │  • HU值轉換     │  • 張量轉換    │  • 清晰度   │
    └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   AI檢測層 (AI Detection)                   │
    ├─────────────────────────────────────────────────────────────┤
    │           🧠 CT-ViT檢測模型 (CTViTForDetection)             │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  Vision Transformer Backbone                       │    │
    │  │  ├── Patch Embedding (16x16)                      │    │
    │  │  ├── Positional Encoding                          │    │
    │  │  ├── 12層 Transformer Encoder                     │    │
    │  │  └── [CLS] Token Feature (768維)                  │    │
    │  │                     │                             │    │
    │  │                     ▼                             │    │
    │  │  ┌─────────────────────────────────────────────┐  │    │
    │  │  │         多頭檢測層 (Multi-Head)              │  │    │
    │  │  │  ┌─────────────┬─────────────┬───────────┐  │  │    │
    │  │  │  │  分類頭      │  邊界框回歸   │  目標性    │  │  │    │
    │  │  │  │ Classifier  │  BBox Reg   │ Objectness│  │  │    │
    │  │  │  │   (4類)     │   (4座標)    │   (1值)   │  │  │    │
    │  │  │  │  A/B/E/G    │   x,y,w,h   │  0.0-1.0  │  │  │    │
    │  │  │  └─────────────┴─────────────┴───────────┘  │  │    │
    │  │  └─────────────────────────────────────────────┘  │    │
    │  └─────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                  特徵提取層 (Feature extraction)             │
    ├─────────────────────────────────────────────────────────────┤
    │  📊 統計特徵      │  🎯 檢測結果     │  📍 空間特徵   │  🔍 品質評估  │
    │  • 像素統計      │  • 信心度分數    │  • 位置座標    │  • 清晰度    │
    │  • 對比度分析    │  • 類別機率      │  • 大小面積    │  • 對比度    │
    └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   知識檢索層 (RAG System)                   │
    ├─────────────────────────────────────────────────────────────┤
    │  📚 醫學知識庫    │  🔍 語義搜索     │  📖 文獻檢索   │  💡 診斷建議  │
    │  • 疾病資料庫    │  • 向量嵌入      │  • 相關研究    │  • 治療建議  │
    │  • 診斷標準      │  • 相似性匹配    │  • 案例對比    │  • 風險評估  │
    └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                  報告生成層 (Report Generation)              │
    ├─────────────────────────────────────────────────────────────┤
    │  🤖 LLM生成      │  📋 結構化報告   │  📊 視覺化      │  💾 多格式輸出 │
    │  • LLaMA 3.2     │  • JSON格式      │  • 檢測標註    │  • HTML報告   │
    │  • Gemma         │  • 標準醫療格式  │  • 統計圖表    │  • PDF導出    │
    └─────────────────────────────────────────────────────────────┘
```

### 🔄 工作流程

#### 完整資料處理流程
```
原始資料集 (Lung-PET-CT-Dx)
    ↓ copy_all_patients_matched_files.py (DICOM-XML匹配)
配對數據 (matched_data_by_patient/)
    ↓ dataset_splitter.py (分割訓練/驗證/測試)
分割資料集 (dataset_splits/)
    ↓ data_processing.py (數據載入)
PyTorch資料集 → CT-ViT訓練
    ↓ 模型推理
檢測結果 → RAG系統 → 智能報告
```

#### 完整處理管道
```
DICOM輸入 → 影像前處理 → AI檢測分析 → 特徵提取 → 知識檢索 → 報告生成 → 結果輸出
    ↓           ↓           ↓          ↓         ↓         ↓         ↓
  CT掃描    → 正規化轉換 → CT-ViT檢測 → 量化分析 → RAG系統 → LLM生成 → JSON/HTML
```

#### 核心技術組件

1. **原始資料處理**
   - Lung-PET-CT-Dx資料集解析與組織
   - DICOM-XML文件智能匹配（基於SOP Instance UID）
   - 352位患者，30,382組配對數據的自動化處理
   - 分層資料集分割（訓練70%，驗證15%，測試15%）

2. **DICOM預處理**
   - pydicom讀取和解析
   - HU值正規化（肺窗：-600中心，1200寬度）
   - 影像尺寸調整到224×224

3. **CT-ViT檢測模型**
   - Vision Transformer骨幹網路（google/vit-base-patch16-224）
   - 多頭檢測架構(分類+回歸+目標性)
   - 多任務學習損失函數

4. **特徵提取**
   - 影像統計特徵
   - 病灶形態特徵
   - 檢測信心度分析

5. **RAG知識檢索**
   - 醫學知識庫查詢
   - 語義相似性匹配（all-MiniLM-L6-v2）
6. **智能報告生成**
   - 結構化醫療報告
   - 風險等級評估
   - HTML/JSON雙格式輸出

## 🚀 快速開始

### 環境要求
- **Python**: 3.8-3.11
- **系統需求**: 16GB+ RAM (推薦32GB)
- **GPU**: NVIDIA RTX 3060+ (推薦), CUDA 11.0+
- **存儲空間**: 50GB+ (模型和資料)

### 1. 安裝與設置

```bash
# 克隆專案
git clone https://github.com/FCU-BioLab/chest-ct-report-generator.git
cd chest-ct-report-generator

# 安裝依賴
pip install -r requirements.txt

# 檢查系統狀態
cd CT_ViT_Training
python test_system.py
```

### 2. 資料準備

```bash
# 檢查資料結構
ls datasets/all_patient_data/

# 如需處理原始資料集，請參考資料處理工具
cd dataset_process/
```

### 3. 模型訓練

```bash
cd CT_ViT_Training

# 分類模型訓練
python train.py

# 檢測模型訓練 (推薦)
python train_detection.py

# 統一評估系統
python unified_evaluator.py
```

### 4. 影像分割

```bash
# MedSAM2 分割功能
cd medsam2_segmentation
python test_medsam2_setup.py  # 測試環境
python sam_seg.py --help      # 查看分割選項
```

### 5. 智能報告生成

```bash
# 啟動 RAG 報告生成系統
cd LLM/RAG
python GUI.py
```

### 4. 資料集處理

```powershell
# 使用智能資料集分割工具
cd CT_ViT_Training\tools
python dataset_splitter.py `
    --source_dir "..\..\matched_data_by_patient" `
    --output_dir "..\..\dataset_splits" `
    --train_ratio 0.7 `
    --val_ratio 0.15 `
    --test_ratio 0.15
```

### 5. 模型訓練

```powershell
cd CT_ViT_Training

# 分類模型訓練
python train.py

# 檢測模型訓練 (升級版)
python train_detection.py

# 統一評估系統
python unified_evaluator.py
```

### 6. 推理與評估

```powershell
cd CT_ViT_Training

# 分類推理
python inference.py `
    --model_path "models/classification_model" `
    --input "path/to/image.dcm" `
    --mode single

# 檢測推理 (新功能)
python inference_detection.py `
    --model_path "models/detection_model" `
    --input "path/to/image.dcm" `
    --confidence_threshold 0.5

# 批量處理
python inference.py `
    --model_path "models/classification_model" `
    --input "image_directory" `
    --mode batch

# 模型評估
python inference.py `
    --model_path "models/classification_model" `
    --input "../dataset_splits/test" `
    --mode evaluate
```

### 7. 報告生成

```powershell
# 啟動RAG報告生成系統
cd RAG
python GUI.py
```

### 8. 處理單個患者

```powershell
# 檢查特定患者數據
cd matched_data_by_patient\A0001
dir

# 使用推理腳本處理單個DICOM文件
cd ..\..\CT_ViT_Training
python inference.py `
    --model_path "models/classification_model" `
    --input "../matched_data_by_patient/A0001/dicom_files/A0001000.dcm" `
    --mode single
```

### 9. 批量處理患者

```powershell
# 使用資料集工具進行批量分析
cd CT_ViT_Training\tools
python dataset_splitter.py `
    --source_dir "../../matched_data_by_patient" `
    --output_dir "../../dataset_splits" `
    --analyze_only

# 批量推理
cd ..
python inference.py `
    --model_path "models/classification_model" `
    --input "../dataset_splits/test" `
    --mode batch
```

## ⚙️ 配置說明

### 主要配置文件 (`config.json`)

```json
{
  "models": {
    "detection_model_path": "CT_ViT_Training/models/best_detection_model.pth",
    "classification_model_path": "CT_ViT_Training/models/best_classification_model.pth"
  },
  "processing": {
    "image_size": 224,
    "confidence_threshold": 0.5,
    "batch_size": 8,
    "use_gpu": true,
    "preprocessing": {
      "normalize_method": "percentile",
      "percentile_range": [1, 99],
      "target_pixel_range": [0, 255]
    }
  },
  "data": {
    "patient_data_dir": "matched_data_by_patient",
    "dataset_splits_dir": "dataset_splits",
    "output_base_dir": "workflow_outputs"
  },
  "rag": {
    "knowledge_base_path": "RAG/medical_knowledge_base",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "top_k": 5,
    "similarity_threshold": 0.7,
    "enable_rag": true
  },
  "llm": {
    "model_name": "llama3.2",
    "temperature": 0.3,
    "max_tokens": 2048,
    "enable_llm": true
  },
  "output": {
    "save_visualizations": true,
    "save_intermediate_results": true,
    "generate_html_report": true,
    "generate_json_report": true
  },
  "system": {
    "max_concurrent_processes": 4,
    "enable_caching": true,
    "cache_dir": "cache",
    "log_level": "INFO"
  }
}
```

### 訓練配置 (`CT_ViT_Training/configs/default_config.yaml`)

```yaml
# 完整訓練配置
training:
  batch_size: 8
  learning_rate: 2e-5
  num_epochs: 50
  weight_decay: 0.01
  mixed_precision: true

model:
  name: "google/vit-base-patch16-224"
  num_labels: 4
  image_size: 224
  dropout_rate: 0.1

dicom:
  window_center: -600  # 肺窗中心 (HU)
  window_width: 1200   # 肺窗寬度 (HU)
  
data:
  stratified_split: true
  patient_level_split: true
  augmentation: true
```

### RAG配置 (`RAG/config.json`)

```json
{
  "model_name": "all-MiniLM-L6-v2",
  "ollama_model": "gemma3n:e2b",
  "top_k": 5,
  "similarity_threshold": 0.7,
  "max_history": 50,
  "auto_save": true,
  "log_level": "INFO"
}
```

## 📊 輸出範例與高級功能

### JSON報告結構
```json
{
  "report_header": {
    "patient_id": "A0001",
    "examination_date": "2025-01-XX",
    "ai_system_version": "CT-ViT Detection v1.0"
  },
  "clinical_findings": {
    "primary_findings": [
      {
        "description": "檢測到疑似Adenocarcinoma (惡性腺癌)",
        "location": "右肺上葉",
        "size": {"width_mm": "35.4", "height_mm": "46.8"},
        "characteristics": {
          "confidence": 0.892,
          "risk_level": "High Risk (惡性)"
        }
      }
    ]
  },
  "recommendations": [
    "建議儘快安排PET-CT檢查以確認病灶性質",
    "建議胸腔外科專科會診評估治療方案"
  ]
}
```

### 視覺化輸出
- 🖼️ **檢測結果圖**: 標註邊界框和類別標籤
- 📈 **特徵分析圖**: 統計特徵和品質指標
- 🎯 **信心度分布**: 檢測結果可信度分析
- 📊 **風險評估圖**: 視覺化風險等級

### 批次處理範例
```python
import os
from pathlib import Path

# 使用現有的推理系統進行批次處理
def batch_process_patients(patient_list):
    """批次處理多個患者"""
    for patient_id in patient_list:
        patient_dir = Path(f"matched_data_by_patient/{patient_id}")
        if patient_dir.exists():
            dicom_files = list(patient_dir.glob("dicom_files/*.dcm"))
            print(f"患者 {patient_id}: 找到 {len(dicom_files)} 個DICOM檔案")
        else:
            print(f"患者 {patient_id}: 目錄不存在")

# 範例使用
patients = ["A0001", "A0002", "A0003"]
batch_process_patients(patients)
```

## 🔧 故障排除

### 常見問題與解決方案

#### 1. 模型載入失敗
```powershell
問題: FileNotFoundError: 找不到模型文件
解決方案: 
- 檢查模型路徑，確認模型檔案存在
- 運行系統測試確認功能正常: cd CT_ViT_Training; python test_system.py
```

#### 2. CUDA記憶體不足
```powershell
問題: RuntimeError: CUDA out of memory
解決方案:
- 降低batch_size (從8改為4或2)
- 在config.json中設定 "use_gpu": false 使用CPU模式
- 啟用混合精度訓練: "mixed_precision": true
```

#### 3. DICOM讀取錯誤
```powershell
問題: InvalidDicomError: 無法讀取DICOM文件
解決方案:
- 安裝正確版本: pip install pydicom>=2.3.0
- 檢查文件完整性: python -c "import pydicom; pydicom.dcmread('file_path')"
- 確認文件格式為標準DICOM格式
```

#### 4. 依賴包版本衝突
```powershell
問題: 套件相容性問題
解決方案:
- 使用虛擬環境: python -m venv venv; .\venv\Scripts\activate
- 安裝指定版本: pip install -r requirements.txt
- 使用鏡像源: pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 5. GPU檢測問題
```powershell
問題: CUDA不可用
解決方案:
- 檢查CUDA安裝: nvidia-smi
- 驗證PyTorch CUDA支援: python -c "import torch; print(torch.cuda.is_available())"
- 重新安裝對應CUDA版本的PyTorch
```

### 性能優化建議
- **記憶體優化**: 調整batch_size以匹配GPU記憶體
- **速度優化**: 啟用混合精度訓練和數據快取
- **準確度優化**: 使用數據增強和交叉驗證

### 系統需求檢查
- **最低需求**: Python 3.8+, 16GB RAM, 20GB存儲空間
- **推薦配置**: NVIDIA RTX 3060+, 32GB RAM, 50GB SSD, CUDA 11.0+
- **作業系統**: Windows 10+, Ubuntu 18.04+, macOS 10.15+

## 📈 開發路線圖

### 已完成功能 ✅
- [x] CT-ViT檢測模型升級（63% → 85-90%）
- [x] 端到端工作流程整合
- [x] RAG知識檢索系統
- [x] 智能報告生成（JSON/HTML）
- [x] 視覺化檢測結果
- [x] 批量處理功能
- [x] 系統配置管理

### 近期開發計劃 🚧
- [ ] 3D體積分析功能
- [ ] 多模態影像融合
- [ ] 實時處理優化
- [ ] Web界面開發
- [ ] Docker容器化部署

### 長期發展目標 🎯
- [ ] 支援更多影像格式（MRI, PET等）
- [ ] 整合更多醫學知識庫
- [ ] 多語言報告生成
- [ ] 雲端部署與API服務
- [ ] 移動端應用開發
- [ ] 與醫院系統整合（PACS/HIS）

## 🤝 貢獻指南

### 如何貢獻
1. **Fork專案**: 點擊右上角Fork按鈕
2. **創建分支**: `git checkout -b feature/amazing-feature`
3. **提交更改**: `git commit -m 'feat: add amazing feature'`
4. **推送分支**: `git push origin feature/amazing-feature`
5. **發起PR**: 在GitHub上創建Pull Request

### 代碼規範
- 遵循PEP 8 Python代碼風格
- 添加必要的註釋和文檔字符串
- 確保所有測試通過
- 更新相關文檔

### 報告問題
- 使用[GitHub Issues](https://github.com/FCU-BioLab/chest-ct-report-generator/issues)報告Bug
- 提供詳細的錯誤信息和重現步驟
- 說明您的系統環境（OS、Python版本等）

## 📚 參考資源

### 技術文檔
- [PyTorch官方文檔](https://pytorch.org/docs/) - 深度學習框架
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) - 預訓練模型
- [PyDICOM文檔](https://pydicom.github.io/pydicom/) - DICOM處理

### 相關論文
- Vision Transformer (ViT) - "An Image is Worth 16x16 Words"
- DETR - "End-to-End Object Detection with Transformers"
- RAG - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

### 醫學標準
- [Lung-RADS分類標準](https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Lung-Rads)
- DICOM醫學影像標準
- HL7 FHIR醫療數據交換標準

## 📄 授權與免責聲明

### 授權條款
本專案採用 **MIT License** 授權 - 詳見 [LICENSE](LICENSE) 文件

### 免責聲明
⚠️ **重要提醒**: 
- 本系統僅供**研究和教育用途**
- **不得用於實際臨床診斷**
- 任何醫療決策應由合格醫師做出
- 使用者承擔所有使用風險

### 隱私保護
- 所有患者數據已去識別化處理
- 遵循HIPAA和GDPR數據保護規範
- 不收集或存儲個人敏感信息

## 🙏 致謝

### 技術支持
- **Vision Transformer** by Google Research
- **Hugging Face** Transformers Library  
- **PyTorch** 深度學習框架
- **OpenAI** 與 **Meta** 的開源LLM模型

### 數據支持
- 感謝醫學影像數據提供方
- 感謝醫學專家的標註工作
- 感謝開源社群的貢獻

### 開發團隊
- **FCU-BioLab** 逢甲大學生物資訊實驗室
- **GitHub Copilot** AI程式設計助手
- 所有貢獻者和測試用戶

## 📞 支援與聯繫

### 獲取幫助
1. **查看文檔**: 先查閱本README和相關文檔
2. **搜索Issues**: 檢查[GitHub Issues](https://github.com/FCU-BioLab/chest-ct-report-generator/issues)是否有類似問題
3. **創建Issue**: 描述問題並提供必要信息
4. **聯繫維護者**: 通過GitHub或Email聯繫

### 聯繫方式
- **GitHub**: [FCU-BioLab](https://github.com/FCU-BioLab)
- **Issues**: [項目Issues頁面](https://github.com/FCU-BioLab/chest-ct-report-generator/issues)
- **Discussions**: [項目討論區](https://github.com/FCU-BioLab/chest-ct-report-generator/discussions)

---

## 🏥 **胸部CT報告生成系統** 
### *讓AI協助醫療診斷，提升診斷效率和準確性！*

**版本**: v2.1.0 | **最後更新**: 2025年7月25日 | **維護者**: FCU-BioLab Team

*本系統僅供研究和教育用途，實際臨床應用請遵循相關醫療法規和標準。*
    "patient_id": "A0001",
    "examination_date": "2025-01-XX",
    "ai_system_version": "CT-ViT Detection v1.0"
  },
  "clinical_findings": {
    "primary_findings": [
      {
        "description": "檢測到疑似Adenocarcinoma (惡性腺癌)",
        "location": "右肺上葉",
        "size": {"width_mm": "35.4", "height_mm": "46.8"},
        "characteristics": {
          "confidence": 0.892,
          "risk_level": "High Risk (惡性)"
        }
      }
    ]
  },
  "recommendations": [
    "建議儘快安排PET-CT檢查以確認病灶性質",
    "建議胸腔外科專科會診評估治療方案"
  ]
}
```

### 視覺化輸出
- 🖼️ **檢測結果圖**: 標註邊界框和類別標籤
- 📈 **特徵分析圖**: 統計特徵和品質指標
- 🎯 **信心度分布**: 檢測結果可信度分析
- 📊 **風險評估圖**: 視覺化風險等級

### 批次處理範例
```python
import os
from pathlib import Path

# 使用現有的推理系統進行批次處理
def batch_process_patients(patient_list):
    """批次處理多個患者"""
    for patient_id in patient_list:
        patient_dir = Path(f"matched_data_by_patient/{patient_id}")
        if patient_dir.exists():
            dicom_files = list(patient_dir.glob("dicom_files/*.dcm"))
            print(f"患者 {patient_id}: 找到 {len(dicom_files)} 個DICOM檔案")
        else:
            print(f"患者 {patient_id}: 目錄不存在")

# 範例使用
patients = ["A0001", "A0002", "A0003"]
batch_process_patients(patients)
```

### 自定義配置範例
```json
{
  "models": {
    "detection_model_path": "CT_ViT_Training/models/best_detection_model.pth",
    "confidence_threshold": 0.5
  },
  "processing": {
    "image_size": 224,
    "batch_size": 8,
    "use_gpu": true
  },
  "output": {
    "save_visualizations": true,
  }
}
```

---

## 🏥 **胸部CT報告生成系統** 
### *讓AI協助醫療診斷，提升診斷效率和準確性！*

**版本**: v2.1.0 | **最後更新**: 2025年7月30日 | **維護者**: FCU-BioLab Team

*本系統僅供研究和教育用途，實際臨床應用請遵循相關醫療法規和標準。*
