# Chest CT Report Generator / 胸部 CT 報告生成器

[English](#english) | [中文](#chinese)

## English

### Overview
This project is a comprehensive AI-powered chest CT analysis and report generation system that combines multiple advanced technologies to assist medical professionals in chest CT scan analysis and standardized report generation. The system integrates Vision Transformer (ViT) for image classification, RAG (Retrieval-Augmented Generation) for report generation, and fine-tuned language models to provide accurate and consistent medical analysis.

### 🌟 Key Features

#### 🏥 Medical Image Analysis
- **CT-ViT Training System**: Professional Vision Transformer training for chest CT DICOM image classification
- **DICOM Processing**: Advanced Hounsfield unit conversion and lung windowing
- **Multi-class Classification**: Automatic classification into A, B, E, G patient series
- **Attention Visualization**: Visual interpretation of model attention patterns

#### 📋 Report Generation
- **Automated Report Generation**: AI-powered chest CT report generation
- **RAG Integration**: Retrieval-Augmented Generation for context-aware reporting  
- **Lung-RADS Compliance**: Integration with standardized Lung-RADS criteria
- **Multi-language Support**: English report generation with extensible language support

#### 🖥️ User Interface & Tools
- **User-friendly GUI**: Intuitive graphical interface for report generation
- **One-click Training**: Automated training scripts for easy model deployment
- **Batch Processing**: Support for processing multiple images efficiently
- **Comprehensive Evaluation**: Confusion matrices, ROC curves, and detailed metrics

### 📁 Project Structure

```
chest-ct-report-generator/
├── 🎯 CT_ViT_Training/           # Vision Transformer Training System
│   ├──  train.py               # Main training script  
│   ├── 🔍 inference.py           # Inference and evaluation
│   ├── 🔧 test_system.py         # System health check
│   ├── � migrate_files.py       # File migration utility
│   ├── �📂 src/                   # Core modules
│   │   ├── ⚙️ config.py          # Configuration management
│   │   ├── 🖼️ data_processing.py # DICOM processing & datasets
│   │   ├── 🧠 model.py           # ViT model & training logic
│   │   └── 🛠️ utils.py           # Utilities & logging
│   ├── 📂 configs/               # Configuration files
│   │   └── 📝 default_config.yaml # Default configuration template
│   ├── 📂 scripts/               # Convenience scripts
│   │   ├── 🪟 run_ct_vit.bat     # Windows one-click startup
│   │   └── � run_ct_vit.sh      # Linux/Mac one-click startup
│   └── 📂 legacy/                # Legacy file backups
│
├── 🤖 RAG/                       # RAG System & GUI
│   ├── 🖥️ Gemma3_GUI.py         # Main GUI application
│   ├── 📦 install.bat            # Installation script
│   └── 📦 uninstall.bat          # Uninstallation script
│
├── 🔬 Fine_Tune/                 # Model Fine-tuning
│   └── 🦙 llama3.2/              # Fine-tuned LLaMA models
│
├── 📊 data/                      # Data directory
│   ├── 📁 dataset_splits/        # Train/validation/test splits
│   │   ├── 📂 train/             # Training data
│   │   ├── 📂 validation/        # Validation data
│   │   └── 📂 test/              # Test data
│   ├── 📁 raw/                   # Raw DICOM data
│   └── 📁 processed/             # Processed datasets
│
├── 🔄 datasets_process/          # Data processing utilities
├── 📁 matched_data_by_patient/   # Organized patient data
├── 📋 requirements.txt           # Unified dependencies
└── � README.md                  # This comprehensive documentation
```

### 🚀 Quick Start

#### 1. System Requirements
- **OS**: Windows 10+, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8 - 3.11
- **Memory**: At least 16GB RAM
- **GPU**: NVIDIA GPU with CUDA 11.0+ (recommended)
- **Storage**: At least 10GB available space

#### 2. Installation
```bash
# Clone the repository
git clone https://github.com/FCU-BioLab/chest-ct-report-generator.git
cd chest-ct-report-generator

# Install unified dependencies
pip install -r requirements.txt
```

#### 3. CT-ViT Training (One-click)
**Windows:**
```bash
cd CT_ViT_Training
scripts\run_ct_vit.bat
```

**Linux/Mac:**
```bash
cd CT_ViT_Training
chmod +x scripts/run_ct_vit.sh
./scripts/run_ct_vit.sh
```

#### 4. Report Generation GUI
```bash
cd RAG
python Gemma3_GUI.py
```

### 🎮 Usage Guide

#### CT Image Classification
1. **Prepare Dataset**: Organize DICOM files in the required structure
2. **Configure Training**: Modify `CT_ViT_Training/configs/default_config.yaml`
3. **Train Model**: Use one-click scripts or manual training
4. **Evaluate Results**: Generate confusion matrices, ROC curves, and metrics

#### Report Generation
1. **Launch GUI**: Start the RAG interface
2. **Input Information**: Provide patient and scan details
3. **Generate Report**: AI-powered report creation
4. **Review & Export**: Review generated reports and export

### 📊 DICOM Processing Features

#### Professional Medical Image Processing
- **Hounsfield Units**: Automatic HU value conversion and normalization
- **Lung Windowing**: Center -600 HU, Width 1200 HU
- **Image Enhancement**: Adaptive contrast and noise suppression
- **Standardization**: ViT-compatible input preprocessing

#### Data Augmentation
- Random rotation (±15°)
- Horizontal flipping (50% probability)
- Brightness adjustment (±20%)
- Contrast adjustment (±20%)
- Random scale and crop

### 🧠 Model Architecture

#### Vision Transformer
- **Base Model**: google/vit-base-patch16-224
- **Patch Size**: 16×16 pixels
- **Image Resolution**: 224×224
- **Attention Heads**: 12 multi-head attention
- **Hidden Dimension**: 768

#### Custom Adaptations
- 4-class output adaptation (A, B, E, G series)
- Medical image preprocessing pipeline
- Class-balanced loss function
- Early stopping and learning rate scheduling

### 📈 Evaluation & Visualization

#### Metrics
- **Accuracy**
- **Precision/Recall/F1-Score**
- **Confusion Matrix**
- **ROC Curves and AUC**
- **Class Distribution Statistics**

#### Visualizations
- 🎯 Confusion matrix heatmaps
- 📈 ROC curve plots
- 📊 Class distribution pie charts
- 🎪 Training progress curves
- 👁️ Attention weight visualization

---

## Chinese

### 概述
本專案是一個綜合性的AI驅動胸部CT分析和報告生成系統，結合多種先進技術來協助醫療專業人員進行胸部CT掃描分析和標準化報告生成。系統整合了Vision Transformer（ViT）進行影像分類、RAG（檢索增強生成）進行報告生成，以及微調的語言模型，以提供準確且一致的醫療分析。

### 🌟 主要功能

#### 🏥 醫學影像分析
- **CT-ViT訓練系統**：專業的Vision Transformer訓練，用於胸部CT DICOM影像分類
- **DICOM處理**：先進的Hounsfield單位轉換和肺部視窗化
- **多類別分類**：自動分類為A、B、E、G患者系列
- **注意力可視化**：模型注意力模式的視覺化解釋

#### 📋 報告生成
- **自動報告生成**：AI驅動的胸部CT報告生成
- **RAG整合**：檢索增強生成，提供情境感知報告
- **Lung-RADS合規**：整合標準化Lung-RADS標準
- **多語言支援**：英文報告生成，支援擴展語言

#### 🖥️ 使用者介面與工具
- **友善的GUI**：報告生成的直觀圖形介面
- **一鍵訓練**：自動化訓練腳本，輕鬆部署模型
- **批次處理**：支援高效處理多張影像
- **全面評估**：混淆矩陣、ROC曲線和詳細指標

### 🚀 快速開始

#### 1. 系統需求
- **作業系統**：Windows 10+、Ubuntu 18.04+、macOS 10.15+
- **Python**：3.8 - 3.11
- **記憶體**：至少16GB RAM
- **GPU**：NVIDIA GPU with CUDA 11.0+（推薦）
- **儲存空間**：至少10GB可用空間

#### 2. 安裝
```bash
# 複製儲存庫
git clone https://github.com/FCU-BioLab/chest-ct-report-generator.git
cd chest-ct-report-generator

# 安裝統一依賴
pip install -r requirements.txt
```

#### 3. CT-ViT訓練（一鍵式）
**Windows：**
```bash
cd CT_ViT_Training
scripts\run_ct_vit.bat
```

**Linux/Mac：**
```bash
cd CT_ViT_Training
chmod +x scripts/run_ct_vit.sh
./scripts/run_ct_vit.sh
```

#### 4. 報告生成GUI
```bash
cd RAG
python Gemma3_GUI.py
```

### 🎮 使用指南

#### CT影像分類
1. **準備資料集**：按照要求的結構組織DICOM檔案
2. **配置訓練**：修改 `CT_ViT_Training/configs/default_config.yaml`
3. **訓練模型**：使用一鍵腳本或手動訓練
4. **評估結果**：生成混淆矩陣、ROC曲線和指標

#### 報告生成
1. **啟動GUI**：開始RAG介面
2. **輸入資訊**：提供患者和掃描細節
3. **生成報告**：AI驅動的報告創建
4. **檢視與匯出**：檢視生成的報告並匯出

### 📚 項目功能

#### 🎯 CT-ViT 訓練系統
本專案的核心是一個完整的Vision Transformer訓練系統，專門用於胸部CT影像的四類分類（A、B、E、G系列）：

**主要特色：**
- **專業DICOM處理**：支援Hounsfield單位轉換、肺部視窗化
- **模組化設計**：清晰的程式碼結構，便於維護和擴展
- **一鍵式啟動**：Windows和Linux/Mac的自動化腳本
- **完整評估**：混淆矩陣、ROC曲線、注意力可視化

**快速使用：**
```bash
cd CT_ViT_Training
# Windows: scripts\run_ct_vit.bat
# Linux/Mac: ./scripts/run_ct_vit.sh
```

#### 🤖 RAG 報告生成系統
整合了檢索增強生成技術的智慧報告生成系統：

**功能包括：**
- AI驅動的報告自動生成
- Lung-RADS標準合規
- 友善的圖形介面
- 多語言報告支援

#### 🔬 模型微調
支援LLaMA等大語言模型的醫療領域微調，提升報告生成的專業性和準確性。

### ⚡ 性能優化與特色

#### 🏥 醫學影像專業處理
- **Hounsfield Units轉換**：自動HU值歸一化
- **肺部視窗化**：中心值-600 HU，寬度1200 HU  
- **數據增強**：針對醫學影像的專業增強策略
- **批次處理**：高效的多影像處理能力

#### 🧠 先進的AI架構
- **Vision Transformer**：基於Google ViT-Base的醫學影像分類
- **注意力機制**：可視化模型關注區域
- **混合精度訓練**：減少記憶體使用，加速訓練
- **早停機制**：防止過擬合，優化模型性能

### 🚨 故障排除

#### 常見問題解決

**Q: CUDA記憶體不足**
```yaml
# 在 CT_ViT_Training/configs/default_config.yaml 中調整
training:
  batch_size: 4              # 減少批次大小
  gradient_accumulation_steps: 2  # 使用梯度累積
system:
  mixed_precision: true      # 啟用混合精度訓練
```

**Q: DICOM檔案讀取失敗**
```python
# 檢查DICOM檔案完整性
import pydicom
try:
    ds = pydicom.dcmread("file.dcm")
    print("DICOM檔案正常")
except Exception as e:
    print(f"讀取錯誤: {e}")
```

**Q: 依賴套件安裝失敗**
```bash
# 升級pip並使用國內鏡像源
python -m pip install --upgrade pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

**Q: GPU未被正確識別**
```python
# 檢查CUDA環境
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU數量: {torch.cuda.device_count()}")
```

### 📊 系統需求建議

#### 最低配置
- **CPU**: Intel i5 / AMD Ryzen 5
- **記憶體**: 16GB RAM
- **儲存**: 20GB 可用空間
- **Python**: 3.8-3.11

#### 推薦配置
- **CPU**: Intel i7 / AMD Ryzen 7
- **記憶體**: 32GB RAM
- **GPU**: NVIDIA RTX 3060 或更高
- **儲存**: 50GB SSD空間
- **CUDA**: 11.0+

### 🔧 進階配置

#### 分散式訓練
```bash
# 多GPU訓練範例
python -m torch.distributed.launch --nproc_per_node=2 train.py --config configs/default_config.yaml
```

#### 超參數調優
```yaml
# 在 default_config.yaml 中啟用
hyperparameter_search:
  enable: true
  strategy: "bayesian"
  search_params:
    learning_rate: [1e-5, 2e-5, 5e-5]
    batch_size: [4, 8, 16]
```

### 🤝 貢獻指南

我們歡迎社群的貢獻！請遵循以下步驟：

#### 如何貢獻
1. **Fork 專案**：點擊GitHub頁面右上角的Fork按鈕
2. **建立功能分支**：`git checkout -b feature/amazing-feature`
3. **提交變更**：`git commit -m 'Add some amazing feature'`
4. **推送到分支**：`git push origin feature/amazing-feature`
5. **發起Pull Request**：在GitHub上創建Pull Request

#### 程式碼規範
- 遵循PEP 8 Python程式碼風格
- 為新功能添加適當的註釋和文檔
- 確保所有測試通過
- 更新相關的配置文件

#### 報告問題
- 使用GitHub Issues報告錯誤
- 提供詳細的錯誤訊息和重現步驟
- 包含系統環境資訊（OS、Python版本等）

### 📞 技術支援

#### 聯絡方式
- **GitHub Issues**: [報告問題或功能請求](https://github.com/FCU-BioLab/chest-ct-report-generator/issues)
- **Email**: [專案維護者信箱]
- **Wiki**: [專案Wiki文檔](https://github.com/FCU-BioLab/chest-ct-report-generator/wiki)

#### 常用資源
- 🔗 [PyTorch官方文檔](https://pytorch.org/docs/)
- 🔗 [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- 🔗 [DICOM標準文檔](https://www.dicomstandard.org/)
- 🔗 [Vision Transformer論文](https://arxiv.org/abs/2010.11929)

### 📄 授權聲明

此專案採用MIT授權條款。

### ⚠️ 免責聲明

**重要**：此系統僅供研究和教育用途，不應用於實際醫療診斷。使用醫學影像數據時請遵守相關的隱私和倫理規範。

---

### 🎉 版本歷史

- **v2.0.0** (2025-07-24): 
  - ✅ 完整系統整合，統一文檔和依賴管理
  - ✅ 整合所有requirements.txt文件
  - ✅ 統一README.md文檔
  - ✅ 刪除重複的配置文件
  - ✅ 優化專案結構

- **v1.5.0** (2025-07-22): 
  - ✅ CT-ViT系統模組化重構
  - ✅ 創建專業的訓練模組
  - ✅ 實現一鍵式啟動腳本
  - ✅ 添加系統健康檢查

- **v1.0.0** (2025-07-20): 
  - ✅ 初始版本，基礎功能實現
  - ✅ DICOM處理功能
  - ✅ Vision Transformer訓練
  - ✅ RAG報告生成系統

### 🔮 未來規劃

#### v2.1.0 計劃功能
- 🔄 增加更多數據增強技術
- 📊 優化評估指標和可視化
- 🚀 支援更多預訓練模型
- 🌐 Web界面開發

#### v3.0.0 長期目標
- 🤖 整合更多AI模型
- 🏥 支援更多醫學影像類型
- 📱 移動端應用開發
- ☁️ 雲端部署支援

**最後更新**：2025年7月24日