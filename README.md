# 🏥 胸部CT報告生成系統 (Chest CT Report Generation System)

一個完整的端到端胸部CT影像分析和醫療報告生成系統，整合了深度學習檢測、專業醫學分割和智能報告生成功能。

## 📋 目錄
- [專案概況](#專案概況)
- [系統特色](#系統特色)
- [系統性能](#系統性能) 
- [資料集概況](#資料集概況)
- [資料集詳細分析](#資料集詳細分析)
- [專案結構](#專案結構)
- [技術架構](#技術架構)
- [快速開始](#快速開始)
- [使用指南](#使用指南)
- [配置說明](#配置說明)
- [故障排除](#故障排除)
- [開發路線圖](#開發路線圖)
- [貢獻指南](#貢獻指南)
- [支援與聯繫](#支援與聯繫)

## 🎯 專案概況

**胸部CT報告生成系統** 是一個完整的醫學影像分析管道，整合了深度學習檢測、專業醫學分割和智能報告生成功能。

### ✅ 已完成功能
- **Faster R-CNN檢測模型**: 基於ResNet50-FPN的高精度腫瘤檢測（病灶 vs 背景）
- **多類別分類**: 支援A/B/E/G四類病理類型檢測
- **MedSAM2分割系統**: 專業醫學影像分割功能
- **RAG報告生成**: 基於檢索增強的智能報告系統
- **完整資料流程**: 從原始DICOM到結構化報告的完整管道

### 🔧 核心技術組件
1. **影像處理**: DICOM讀取、HU值轉換、肺窗調整
2. **深度學習**: Faster R-CNN目標檢測架構
3. **醫學分割**: MedSAM2專業分割模型
4. **知識檢索**: RAG系統整合醫學知識庫
5. **報告生成**: 結構化醫療報告輸出

## 📈 系統性能

| 指標 | 開發狀態 | 技術特點 |
|------|----------|----------|
| 目標檢測 | ✅ 已完成 | Faster R-CNN + ResNet50-FPN |
| 多類別分類 | ✅ 已完成 | A/B/E/G四類病理檢測 |
| 醫學分割 | ✅ 已完成 | MedSAM2專業分割 |
| 報告生成 | ✅ 已完成 | RAG增強智能報告 |
| 處理速度 | 🔄 優化中 | 目標 < 1秒/影像 |

### 📊 核心性能指標
- **資料集規模**: 352位患者，30,382組DICOM-XML配對
- **支援格式**: DICOM輸入，JSON/HTML/PDF輸出
- **模型架構**: Faster R-CNN + ResNet50-FPN backbone

## 🌟 系統特色

✅ **Faster R-CNN目標檢測** - 基於ResNet50-FPN的高精度腫瘤檢測（多類別分類）  
✅ **MedSAM2醫學分割** - 專業醫學影像分割，提供精確腫瘤邊界  
✅ **多模態分析** - 整合DICOM影像與XML標註資料  
✅ **RAG智能報告** - 結合知識檢索生成專業醫學報告  
✅ **標準化輸出** - 支援JSON/HTML/PDF多格式報告  
✅ **模組化架構** - 完整的訓練、評估、推理流程

## 🗂️ 資料集概況

- **患者數量**: 352位患者 (A:248, B:38, E:5, G:61)
- **影像數量**: 30,382組DICOM-XML配對
- **標註格式**: Pascal VOC XML格式，包含邊界框座標
- **影像品質**: 高品質胸部CT掃描，適合深度學習訓練

### 📊 患者分類分布
- **A系列 (惡性)**: 248患者 (70.5%)
- **B系列 (良性)**: 38患者 (10.8%)
- **E系列**: 5患者 (1.4%)
- **G系列**: 61患者 (17.3%)

## 📋 資料集詳細分析

### 基本統計
- **總患者數**: 352
- **總標註數**: 1,743

### 標註品質分析
- **包含邊界框的標註**: 1,745
- **包含病理類型的標註**: 1,745
- **完整標註（同時包含邊界框和病理）**: 1,743

### 病理類型分布
- **A**: 982 個標註 (主要惡性)
- **G**: 258 個標註
- **a**: 245 個標註 (次要惡性)
- **B**: 174 個標註 (主要良性)
- **g**: 47 個標註
- **E**: 24 個標註
- **b**: 15 個標註 (次要良性)

### 邊界框分析
- **平均邊界框大小**: 3,689 像素²
- **最小邊界框**: 156 像素²
- **最大邊界框**: 34,587 像素²

## 📁 專案結構

```
chest-ct-report-generator/
├── 🎯 detection/                # 核心檢測模組 (主要工作區域)
│   ├── faster_rcnn_dataset.py   # 檢測資料集
│   ├── faster_rcnn_model.py     # 檢測模型
│   ├── train_detection.py       # 🎯 檢測模型訓練
│   ├── train_detection_simple.py # 簡化訓練腳本
│   ├── inference_detection.py   # 🔍 檢測推理
│   ├── test_faster_rcnn.py      # 模型測試
│   ├── check_gpu.py             # GPU檢查工具
│   ├── Faster_RCNN_Detection/   # 檢測模型輸出目錄
│   ├── Simple_Training/         # 簡化訓練輸出
│   └── README.md                # 檢測模組說明
├── 🧬 medsam2_segmentation/      # MedSAM2分割系統
│   ├── sam_seg.py               # 分割主程式
│   ├── test_medsam2_setup.py    # 環境測試
│   └── MedSAM2/                 # MedSAM2模型檔案
├── 🤖 llm/                      # 語言模型與RAG系統
│   ├── RAG/                     # RAG報告生成
│   └── Fine_Tune/               # 模型微調
├── 📊 datasets/                 # 資料集目錄
│   ├── all_patient_data/        # 處理後的患者資料
│   └── original_datasets/       # 原始資料集
├── 📁 dataset_process/          # 資料處理工具
├── 📄 report_data/              # 報告相關資料
├── config.json                  # 系統配置
├── requirements.txt             # 依賴包列表
├── dataset_analysis_detailed.json # 詳細資料集分析
└── README.md                    # 主要說明文件
```

### 🚀 主要功能模組

#### 1. 核心檢測系統 (detection/)
- **檢測訓練**: `train_detection.py` - Faster R-CNN目標檢測模型訓練
- **簡化訓練**: `train_detection_simple.py` - 簡化版訓練流程
- **模型推理**: `inference_detection.py` - 檢測推理功能
- **系統測試**: `test_faster_rcnn.py` - 模型功能測試
- **GPU支援**: `check_gpu.py` - GPU環境檢查

#### 2. 醫學影像分割系統 (medsam2_segmentation/)
- **MedSAM2分割**: `sam_seg.py` - 專業醫學影像分割
- **環境測試**: `test_medsam2_setup.py` - MedSAM2環境驗證

#### 3. 數據處理工具 (dataset_process/)
- **資料處理**: 包含各種資料預處理和轉換工具
- **DICOM處理**: 醫學影像格式處理功能

#### 4. 智能報告生成系統 (llm/)
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
- **檢測資料處理**: [`detection/faster_rcnn_dataset.py`](detection/faster_rcnn_dataset.py)
  - Faster R-CNN資料集載入器
  - Pascal VOC格式XML解析
  - DICOM影像前處理

##### 📊 數據處理流程
```
原始資料集 (datasets/original_datasets/)
    ↓ 資料處理工具
已處理資料 (datasets/all_patient_data/)  
    ↓ faster_rcnn_dataset.py
PyTorch資料集 → Faster R-CNN訓練
```

## 🚀 使用流程

### 快速開始步驟
1. **環境檢查**: `python detection/check_gpu.py`
2. **模型訓練**: `python detection/train_detection.py`
3. **影像分割**: `python medsam2_segmentation/sam_seg.py`
4. **智能報告**: 使用RAG系統生成醫療報告

### 詳細操作流程

#### 1. 環境設置
```bash
# 安裝依賴
pip install -r requirements.txt

# 檢查GPU狀態
python detection/check_gpu.py
```

#### 2. 資料集準備
```bash
# 處理原始資料集
cd dataset_process
python dataset_splitter.py --source_dir "../datasets/all_patient_data" --output_dir "../datasets/dataset_splits"
```

#### 3. 模型訓練
```bash
cd detection

# Faster R-CNN模型訓練 (推薦)
python train_detection.py

# 簡化版訓練
python train_detection_simple.py

# 模型測試
python test_faster_rcnn.py
```

#### 4. 影像分析與推理
```bash
# 檢測推理
python detection/inference_detection.py --model_path "Faster_RCNN_Detection/models/detection_model.pth" --input "path/to/image.dcm"

# 醫學分割
python medsam2_segmentation/sam_seg.py --input "path/to/image.dcm"
```

#### 5. 報告生成
```bash
# 啟動RAG報告生成系統
cd llm/RAG
python GUI.py
```

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
    │  • pydicom      │  • 像素正規化   │  • resize     │  • 對比度   │
    │  • 元資料提取    │  • HU值轉換     │  • 張量轉換    │  • 清晰度   │
    └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                AI檢測層 (Faster R-CNN Detection)            │
    ├─────────────────────────────────────────────────────────────┤
    │  🧠 Faster R-CNN模型 (ResNet50-FPN Backbone)               │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │  ResNet50特徵提取                                   │    │
    │  │  ├── Conv層特徵提取                               │    │
    │  │  ├── FPN多尺度特徵融合                            │    │
    │  │  └── 特徵金字塔輸出                               │    │
    │  │                     │                             │    │
    │  │                     ▼                             │    │
    │  │  ┌─────────────────────────────────────────────┐  │    │
    │  │  │        RPN + ROI Head 檢測層                │  │    │
    │  │  │  ┌─────────────┬─────────────┬───────────┐  │  │    │
    │  │  │  │  分類頭      │  邊界框回歸   │ 區域提議   │  │  │    │
    │  │  │  │ Classifier  │  BBox Reg   │    RPN    │  │  │    │
    │  │  │  │   (4類)     │   (4座標)    │  Proposals│  │  │    │
    │  │  │  │  A/B/E/G    │   x,y,w,h   │           │  │  │    │
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
    ↓ 資料處理工具 (dataset_process/)
配對數據 (datasets/all_patient_data/)
    ↓ faster_rcnn_dataset.py (資料載入)
PyTorch資料集 → Faster R-CNN訓練
    ↓ 模型推理
檢測結果 → RAG系統 → 智能報告
```

#### 完整處理管道
```
DICOM輸入 → 影像前處理 → Faster R-CNN檢測 → 特徵提取 → 知識檢索 → 報告生成 → 結果輸出
    ↓           ↓               ↓            ↓         ↓         ↓         ↓
  CT掃描    → 正規化轉換 → ResNet50-FPN檢測 → 量化分析 → RAG系統 → LLM生成 → JSON/HTML
```

#### 核心技術組件

1. **原始資料處理**
   - Lung-PET-CT-Dx資料集解析與組織
   - DICOM-XML文件匹配處理
   - 352位患者，30,382組配對數據的自動化處理

2. **DICOM預處理**
   - pydicom讀取和解析
   - 影像尺寸調整和正規化
   - HU值轉換與肺窗調整

3. **Faster R-CNN檢測模型**
   - ResNet50-FPN骨幹網路
   - RPN區域提議網路
   - 多類別分類頭（A/B/E/G）

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

# 檢查GPU狀態
python detection/check_gpu.py
```

### 2. 資料準備

```bash
# 檢查資料結構 (Windows使用dir, Linux/Mac使用ls)
dir datasets\all_patient_data\    # Windows
# ls datasets/all_patient_data/   # Linux/Mac

# 如需處理原始資料集，請參考資料處理工具
cd dataset_process
```

### 3. 模型訓練

```bash
cd detection

# Faster R-CNN模型訓練
python train_detection.py

# 簡化版訓練
python train_detection_simple.py

# 模型測試
python test_faster_rcnn.py
```

## 🔧 使用指南

### 詳細操作流程參考

完整的操作流程已整合在「🚀 使用流程」章節中，包含：
- 環境設置與依賴安裝
- 資料集準備與處理
- 模型訓練與評估
- 影像分析與推理
- 報告生成與輸出

### 進階使用技巧

#### 批量處理
```bash
# 批量推理多個DICOM檔案
python detection/inference_detection.py --input "datasets/test_images/" --mode batch
```

#### GPU監控
```bash
# 檢查GPU使用狀況
python detection/check_gpu.py
python detection/test_gpu_usage.py
```

#### 模型可視化
```bash
# 檢測結果可視化
python detection/test_visualization.py
```

## ⚙️ 配置說明

### 主要配置文件 (`config.json`)

```json
{
  "models": {
    "detection_model_path": "detection/Faster_RCNN_Detection/models/best_detection_model.pth",
    "simple_model_path": "detection/Simple_Training/models/simple_model.pth"
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
    "patient_data_dir": "datasets/all_patient_data",
    "output_base_dir": "workflow_outputs"
  },
  "rag": {
    "knowledge_base_path": "llm/RAG/medical_knowledge_base",
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

### RAG配置 (`llm/RAG/config.json`)

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
    "examination_date": "2025-08-XX",
    "ai_system_version": "Faster R-CNN Detection v1.0"
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
- 運行系統測試確認功能正常: python detection/check_gpu.py
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
- [x] Faster R-CNN目標檢測模型（多類別病理檢測）
- [x] MedSAM2專業醫學分割功能
- [x] RAG知識檢索系統
- [x] 智能報告生成（JSON/HTML/PDF）
- [x] 端到端工作流程整合
- [x] 視覺化檢測結果
- [x] 批量處理功能
- [x] 系統配置管理

### 近期開發計劃 🚧
- [ ] **性能優化**: 提升檢測準確度到90%+
- [ ] **模型優化**: 整合檢測與分割模型
- [ ] **實時處理**: 優化推理速度
- [ ] **Web界面**: 開發使用者友好界面
- [ ] **Docker容器化**: 簡化部署流程

### 長期發展目標 🎯
- [ ] **臨床驗證**: 與醫療機構合作驗證系統
- [ ] **功能擴展**: 支援更多病理類型和影像格式
- [ ] **系統整合**: 與醫院PACS系統整合
- [ ] **多語言支援**: 擴展到英文等其他語言
- [ ] **雲端服務**: 提供API服務和雲端部署
- [ ] **移動端**: 開發移動端應用程式

### 🎯 技術發展重點
1. **準確性提升**: 深度學習模型精度優化
2. **處理效率**: 大規模影像處理能力
3. **臨床應用**: 實際醫療環境適應性
4. **標準化**: 符合醫療行業標準

## 🔧 技術支援

### 開發團隊
- **開發者**: FCU-BioLab
- **系統需求**: Python 3.8+, CUDA 11.0+, 16GB+ RAM
- **主要依賴**: PyTorch, MONAI, pydicom, transformers

### 聯繫方式
- **GitHub Issues**: [問題回報](https://github.com/FCU-BioLab/chest-ct-report-generator/issues)
- **技術討論**: [Discussions](https://github.com/FCU-BioLab/chest-ct-report-generator/discussions)

---

*最後更新: 2025年8月9日*

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

**版本**: v2.1.0 | **最後更新**: 2025年8月9日 | **維護者**: FCU-BioLab Team

*本系統僅供研究和教育用途，實際臨床應用請遵循相關醫療法規和標準。*
