# Chest CT Report Generator / 胸部 CT 報告生成器

[English](#english) | [中文](#chinese)

## English

### Overview
A comprehensive AI-powered chest CT analysis and report generation system combining Vision Transformer (ViT) for image classification, RAG (Retrieval-Augmented Generation) for report generation, and fine-tuned language models for accurate medical analysis.

### 🌟 Key Features
- **CT-ViT Training**: Professional Vision Transformer for chest CT DICOM classification (A, B, E, G series) with 63%+ test accuracy
- **Smart Dataset Processing**: Stratified splitting with patient-level consistency and automated data organization
- **Multi-mode Inference**: Single image, batch processing, and comprehensive evaluation modes
- **Detailed Evaluation**: ROC curves, confusion matrices, classification reports with visualization
- **Report Generation**: AI-powered reporting with RAG integration and medical standard compliance
- **Robust Pipeline**: Full training-evaluation-inference workflow with checkpoint management

### 📁 Project Structure
```
chest-ct-report-generator/
├── 🎯 CT_ViT_Training/          # Vision Transformer Training
│   ├── train.py                 # Training script
│   ├── inference.py             # Inference & evaluation
│   ├── src/                     # Core modules
│   └── scripts/                 # One-click startup scripts
├── 🔄 datasets_process/         # Dataset Processing Tools
│   ├── split_dataset.py         # Smart dataset splitting
│   ├── inference_wrapper.py     # Inference wrapper
│   └── example_usage.py         # Interactive examples
├── 🤖 RAG/                      # Report Generation GUI
├── 🔬 Fine_Tune/                # Model Fine-tuning
├── 📊 dataset_splits/           # Processed Dataset Splits
└── 📁 matched_data_by_patient/  # Organized patient data
```

### 🚀 Quick Start

#### 1. Installation
```bash
git clone https://github.com/FCU-BioLab/chest-ct-report-generator.git
cd chest-ct-report-generator
pip install -r requirements.txt
```

#### 2. Dataset Splitting
```bash
cd datasets_process
python split_dataset.py --source_dir "matched_data_by_patient" --output_dir "../dataset_splits"
```

#### 3. Model Training
```bash
cd CT_ViT_Training
scripts\run_ct_vit.bat    # Windows
./scripts/run_ct_vit.sh   # Linux/Mac
```

#### 4. Inference & Evaluation
```bash
cd CT_ViT_Training
# Single image inference with confidence scores
python inference.py --model_path "../CT_ViT/models/final_model" --mode single --input "path/to/image.dcm"

# Batch processing with results export
python inference.py --model_path "../CT_ViT/models/final_model" --mode batch --input "image_list.txt" --output "./results"

# Comprehensive model evaluation with visualizations
python inference.py --model_path "../CT_ViT/models/final_model" --mode evaluate --input "../dataset_splits/test" --output "./evaluation_results"
```
The evaluation mode generates:
- `evaluation_results.json` - Detailed metrics and classification report
- `confusion_matrix.png` - Visual confusion matrix
- `roc_curves.png` - ROC curves for each class
- `class_distribution.png` - Dataset class distribution analysis

#### 5. Report Generation
```bash
cd RAG
python Gemma3_GUI.py
```

### 📊 Key Features Details

#### Smart Dataset Splitting
- **Stratified Sampling**: Maintains class balance across splits
- **Patient-Level Grouping**: Prevents data leakage
- **Flexible Configuration**: Customizable ratios and random seeds
- **Detailed Reports**: Comprehensive statistics in TXT/JSON formats

#### CT-ViT Model Performance
- **Architecture**: google/vit-base-patch16-224 (224×224, 12 attention heads)
- **Classes**: A, B, E, G patient series classification
- **Test Accuracy**: ~63.16% on test set, ~68.63% on validation set
- **Features**: DICOM processing, lung windowing (-600 HU center, 1200 HU width)
- **Evaluation**: Comprehensive metrics including confusion matrices, ROC curves, and attention visualization
- **Training**: Supports checkpoint resumption and detailed logging with TensorBoard integration

### ⚙️ Configuration

#### Training Configuration (`CT_ViT_Training/configs/default_config.yaml`)
```yaml
# Complete training configuration
training:
  batch_size: 8
  learning_rate: 2e-5
  num_epochs: 50
  weight_decay: 0.01

model:
  name: "google/vit-base-patch16-224"
  num_labels: 4
  image_size: 224

dicom:
  window_center: -600  # Lung window center (HU)
  window_width: 1200   # Lung window width (HU)
```

#### Inference Configuration (`datasets_process/config_ct_vit.yaml`)
```yaml
# Simplified inference configuration
model:
  name: "google/vit-base-patch16-224"
  num_labels: 4
  image_size: 224

data:
  dicom:
    window_center: -600
    window_width: 1200

inference:
  batch_size: 8
  device: "auto"
```

### 🔧 Troubleshooting
- **CUDA Memory**: Reduce batch_size in config, enable mixed precision training
- **DICOM Errors**: Check file integrity with `pydicom.dcmread()`, verify DICOM format
- **Transformers Version**: Use `transformers>=4.53.0` to avoid `evaluation_strategy` parameter issues
- **Package Issues**: Use `pip install -r requirements.txt` with verified compatible versions
- **GPU Issues**: Verify CUDA with `torch.cuda.is_available()`, model supports CPU fallback
- **Wandb Integration**: Set `WANDB_DISABLED=true` environment variable if not using wandb logging
- **JSON Serialization**: Recent updates handle numpy array serialization in inference outputs

---

## Chinese

### 概述
綜合性AI驅動胸部CT分析和報告生成系統，整合Vision Transformer影像分類、RAG報告生成和微調語言模型，提供準確的醫療分析。

### 🌟 主要功能
- **CT-ViT訓練**：專業Vision Transformer進行胸部CT DICOM分類（A、B、E、G系列）
- **智慧資料集處理**：分層採樣確保患者級別一致性
- **模型推理評估**：全面評估與注意力可視化
- **報告生成**：AI驅動報告，整合RAG和Lung-RADS標準
- **友善工具**：一鍵訓練、批次處理、互動式GUI

### 🚀 快速開始

#### 1. 安裝
```bash
git clone https://github.com/FCU-BioLab/chest-ct-report-generator.git
cd chest-ct-report-generator
pip install -r requirements.txt
```

#### 2. 資料集劃分
```bash
cd datasets_process
python split_dataset.py --source_dir "matched_data_by_patient" --output_dir "../dataset_splits"
```

#### 3. 模型訓練
```bash
cd CT_ViT_Training
scripts\run_ct_vit.bat    # Windows
./scripts/run_ct_vit.sh   # Linux/Mac
```

#### 4. 推理評估
```bash
cd CT_ViT_Training
# 單張影像推理
python inference.py --model_path "models/best_model" --mode single --input "image.dcm"
# 批次處理
python inference.py --model_path "models/best_model" --mode batch --input "image_dir"
# 模型評估
python inference.py --model_path "models/best_model" --mode evaluate --input "../dataset_splits/test"
```

#### 5. 報告生成
```bash
cd RAG
python Gemma3_GUI.py
```

### 📊 核心功能

#### 智慧資料集劃分
- **分層採樣**：維持各類別平衡
- **患者級分組**：避免資料洩漏
- **彈性配置**：可自訂比例和隨機種子
- **詳細報告**：TXT/JSON格式統計

#### CT-ViT模型
- **架構**：google/vit-base-patch16-224（224×224，12個注意力頭）
- **分類**：A、B、E、G患者系列
- **特色**：DICOM處理、肺窗調整（-600 HU中心，1200 HU寬度）
- **評估**：混淆矩陣、ROC曲線、注意力可視化

### ⚙️ 配置

#### 訓練配置 (`CT_ViT_Training/configs/default_config.yaml`)
```yaml
# 完整訓練配置
training:
  batch_size: 8
  learning_rate: 2e-5
  num_epochs: 50
  weight_decay: 0.01

model:
  name: "google/vit-base-patch16-224"
  num_labels: 4
  image_size: 224

dicom:
  window_center: -600  # 肺窗中心 (HU)
  window_width: 1200   # 肺窗寬度 (HU)
```

#### 推理配置 (`datasets_process/config_ct_vit.yaml`)
```yaml
# 簡化推理配置
model:
  name: "google/vit-base-patch16-224"
  num_labels: 4
  image_size: 224

data:
  dicom:
    window_center: -600
    window_width: 1200

inference:
  batch_size: 8
  device: "auto"
```
```yaml
# datasets_process/config_ct_vit.yaml
model:
  name: "google/vit-base-patch16-224"
  num_labels: 4
  image_size: 224

data:
  dicom:
    window_center: -600
    window_width: 1200

inference:
  batch_size: 8
  device: "auto"
```

### 🔧 故障排除
- **CUDA記憶體**：減少batch_size，啟用mixed_precision
- **DICOM錯誤**：用`pydicom.dcmread()`檢查文件完整性
- **套件問題**：使用`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/`
- **GPU問題**：用`torch.cuda.is_available()`驗證

### 📋 系統需求
- **最低**：Python 3.8-3.11, 16GB RAM, 20GB空間
- **推薦**：NVIDIA RTX 3060+, 32GB RAM, 50GB SSD, CUDA 11.0+

### 🤝 貢獻 & 支援
- **Issues**: [GitHub Issues](https://github.com/FCU-BioLab/chest-ct-report-generator/issues)
- **文檔**: [PyTorch](https://pytorch.org/docs/), [Transformers](https://huggingface.co/docs/transformers/)
- **授權**: MIT License
- **免責聲明**: 僅供研究教育用途，非醫療診斷用途

### 📈 版本歷史
- **v2.1.0** (2025-07-24): 智慧資料集劃分、推理包裝器、完善文檔
- **v2.0.0** (2025-07-24): 系統整合、統一文檔和依賴管理
- **v1.5.0** (2025-07-22): CT-ViT模組化重構、一鍵啟動腳本
- **v1.0.0** (2025-07-20): 初始版本

**最後更新**: 2025年7月24日 | **維護者**: GitHub Copilot & FCU-BioLab Team
