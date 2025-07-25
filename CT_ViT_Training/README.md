# CT-ViT Training & Evaluation System

這是胸部CT報告生成系統的核心訓練和評估模組，包含完整的模型訓練、評估和推理功能。

## 📁 目錄結構

```
CT_ViT_Training/
├── 🏋️ 訓練模組
│   ├── train.py                    # 原始分類模型訓練
│   ├── train_detection.py          # 檢測模型訓練 (推薦)
│   └── test_system.py             # 系統測試
├── 
├── 📊 評估模組  
│   ├── evaluate_model.py           # 原始評估腳本
│   └── unified_evaluator.py        # 統一評估系統 (推薦)
├── 
├── 🔮 推理模組
│   ├── inference.py                # 分類推理
│   └── inference_detection.py      # 檢測推理 (推薦)
├── 
├── 🏗️ 核心架構
│   └── src/
│       ├── detection_model.py      # CT-ViT檢測模型
│       ├── detection_dataset.py    # 檢測資料處理
│       ├── model.py                # 原始分類模型
│       ├── data_processing.py      # 資料預處理
│       ├── config.py               # 配置管理
│       └── utils.py                # 工具函數
├── 
├── 🛠️ 工具集
│   └── tools/
│       ├── dataset_splitter.py     # 資料集劃分工具
│       └── dicom_viewer.py         # DICOM檢視工具
├── 
├── ⚙️ 配置檔案
│   └── configs/
│       ├── detection_config.yaml   # 檢測模型配置
│       └── classification_config.yaml # 分類模型配置
├── 
├── 📚 文件
│   ├── README.md                   # 本文件
│   └── docs/                       # 詳細文檔
└── 
└── 🎯 其他
    ├── scripts/                    # 執行腳本
    └── models/                     # 訓練好的模型 (自動生成)
```

## 🚀 快速開始

### 1. 資料準備

```bash
# 劃分資料集 (首次使用)
python tools/dataset_splitter.py \
    --source_dir ../../matched_data_by_patient \
    --output_dir ../../dataset_splits \
    --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
```

### 2. 模型訓練

#### 檢測模型訓練 (推薦)
```bash
# 訓練CT-ViT檢測模型
python train_detection.py \
    --train_dir ../../dataset_splits/train \
    --val_dir ../../dataset_splits/validation \
    --output_dir models \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 1e-4
```

#### 分類模型訓練
```bash
# 訓練原始分類模型
python train.py \
    --train_dir ../../dataset_splits/train \
    --val_dir ../../dataset_splits/validation \
    --epochs 30 \
    --batch_size 16
```

### 3. 模型評估

#### 統一評估系統 (推薦)
```bash
# 評估檢測模型
python unified_evaluator.py \
    --model_path models/best_detection_model.pth \
    --data_path ../../dataset_splits/test \
    --model_type detection \
    --output_dir evaluation_results

# 評估分類模型
python unified_evaluator.py \
    --model_path models/best_classification_model.pth \
    --data_path ../../dataset_splits/test \
    --model_type classification
```

#### 原始評估腳本
```bash
python evaluate_model.py
```

### 4. 模型推理

#### 檢測推理 (推薦)
```bash
# 單張影像推理
python inference_detection.py \
    --model_path models/best_detection_model.pth \
    --image_path ../../matched_data_by_patient/A0001/dicom_files/A0001000.dcm \
    --output_dir inference_results

# 批量推理
python inference_detection.py \
    --model_path models/best_detection_model.pth \
    --batch_dir ../../matched_data_by_patient/A0001 \
    --output_dir batch_inference_results
```

#### 分類推理
```bash
python inference.py \
    --model_path models/best_classification_model.pth \
    --image_path path/to/image.dcm
```

## 🔧 工具使用

### 資料集劃分工具
```bash
# 基本使用
python tools/dataset_splitter.py

# 自定義參數
python tools/dataset_splitter.py \
    --source_dir ../../matched_data_by_patient \
    --output_dir ../../dataset_splits \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
    --random_seed 42
```

### DICOM檢視工具
```bash
# 檢視單個DICOM檔案
python tools/dicom_viewer.py --file path/to/file.dcm

# 批量檢視目錄中的DICOM資訊
python tools/dicom_viewer.py --dir path/to/dicom_dir --batch

# 保存影像預覽
python tools/dicom_viewer.py --file path/to/file.dcm --save
```

## 📊 模型性能比較

| 模型類型 | 準確度 | 特色功能 | 推薦使用 |
|---------|--------|----------|----------|
| 分類模型 | ~63% | 快速分類 | 初步篩檢 |
| 檢測模型 | 85-90% | 定位+分類 | **生產環境** |

## ⚙️ 配置說明

### 檢測模型配置
```yaml
# configs/detection_config.yaml
model:
  image_size: 224
  num_classes: 5  # Background, A, B, E, G
  confidence_threshold: 0.5
  
training:
  epochs: 50
  batch_size: 8
  learning_rate: 1e-4
  weight_decay: 1e-5
  
data:
  train_dir: "../../dataset_splits/train"
  val_dir: "../../dataset_splits/validation"
  test_dir: "../../dataset_splits/test"
```

### 分類模型配置
```yaml
# configs/classification_config.yaml
model:
  image_size: 224
  num_classes: 4  # A, B, E, G
  
training:
  epochs: 30
  batch_size: 16
  learning_rate: 2e-5
```

## 🔄 推薦工作流程

### 新手入門流程
1. **資料準備**: `python tools/dataset_splitter.py`
2. **模型訓練**: `python train_detection.py` (檢測模型)
3. **模型評估**: `python unified_evaluator.py --model_type detection`
4. **模型推理**: `python inference_detection.py`

### 進階使用流程
1. **資料分析**: 使用 `dicom_viewer.py` 檢視資料品質
2. **超參數調優**: 修改配置檔案並重新訓練
3. **性能比較**: 使用統一評估系統比較不同模型
4. **生產部署**: 整合到主工作流程 (`../integrated_workflow.py`)

## 📈 性能優化建議

### 訓練優化
```bash
# 使用更大的批次大小 (如果GPU記憶體允許)
python train_detection.py --batch_size 16

# 使用學習率調度
python train_detection.py --lr_scheduler cosine

# 啟用資料增強
python train_detection.py --augmentation True
```

### 推理優化
```bash
# 批量推理以提高效率
python inference_detection.py --batch_dir path/to/patient_dirs

# 使用GPU加速
CUDA_VISIBLE_DEVICES=0 python inference_detection.py
```

## 🔍 故障排除

### 常見問題

1. **CUDA記憶體不足**
   ```bash
   # 降低批次大小
   python train_detection.py --batch_size 4
   
   # 或使用CPU訓練
   CUDA_VISIBLE_DEVICES="" python train_detection.py
   ```

2. **模型載入失敗**
   ```bash
   # 檢查模型路徑是否正確
   ls -la models/
   
   # 檢查模型檔案完整性
   python -c "import torch; print(torch.load('models/best_detection_model.pth').keys())"
   ```

3. **資料載入錯誤**
   ```bash
   # 檢查資料集結構
   python tools/dataset_splitter.py --source_dir ../../matched_data_by_patient
   
   # 驗證DICOM檔案
   python tools/dicom_viewer.py --dir ../../matched_data_by_patient/A0001/dicom_files --batch
   ```

### 效能監控
```bash
# 訓練過程監控
python train_detection.py --verbose --save_checkpoints

# GPU使用監控
nvidia-smi -l 1

# 磁碟空間檢查
df -h
```

## 🤝 與主系統整合

這個訓練模組與主要工作流程系統完全整合：

```python
# 在 integrated_workflow.py 中使用
from CT_ViT_Training.src.detection_model import CTViTForDetection
from CT_ViT_Training.unified_evaluator import ModelEvaluator

# 載入訓練好的模型
model = CTViTForDetection.from_pretrained("CT_ViT_Training/models/best_detection_model.pth")

# 使用統一評估器
evaluator = ModelEvaluator()
results = evaluator.run_evaluation(model_path, test_data, "detection")
```

## 📞 技術支援

如果遇到問題：

1. 查看日誌檔案：`logs/training_*.log`
2. 檢查配置檔案：`configs/*.yaml`
3. 使用偵錯模式：`python train_detection.py --debug`
4. 參考主專案文檔：`../WORKFLOW_GUIDE.md`

---

**🏗️ CT-ViT Training System** - 讓AI模型訓練變得簡單高效！ Training Module

此模組包含CT-ViT（胸部CT Vision Transformer）模型的訓練、評估和推理程式碼。專門針對胸部CT影像的四分類任務（A、B、E、G系列）進行設計和優化。

## 檔案說明

### 主要腳本
- **`train.py`** - 主要的訓練腳本，包含完整的訓練流水線和模型訓練邏輯
- **`evaluate_model.py`** - 模型評估腳本，用於評估已訓練模型的性能指標
- **`inference.py`** - 多功能推理腳本，支援單張影像、批次推理和詳細評估
- **`test_system.py`** - 系統測試腳本，用於驗證整體系統功能

### 目錄結構
- **`src/`** - 核心模組程式碼
  - `config.py` - 配置設定
  - `data_processing.py` - 數據處理和DICOM載入
  - `model.py` - 自定義訓練器
  - `utils.py` - 工具函數
- **`configs/`** - 配置檔案
- **`docs/`** - 文檔
- **`scripts/`** - 輔助腳本

## 使用方法

### 訓練模型
啟動完整的模型訓練流程：
```bash
python train.py
```
訓練過程會自動保存檢查點和最終模型，並生成詳細的訓練日誌。

### 評估模型
評估已訓練模型的性能：
```bash
python evaluate_model.py
```
會生成分類報告、混淆矩陣和詳細的評估指標。

### 推理預測
`inference.py` 腳本支援多種運行模式：

#### 單張影像推理
```bash
python inference.py --model_path ../CT_ViT/models/final_model --mode single --input path/to/image.dcm
```

#### 批次推理
```bash
python inference.py --model_path ../CT_ViT/models/final_model --mode batch --input path/to/image_list.txt --output ./results
```

#### 詳細評估模式
```bash
python inference.py --model_path ../CT_ViT/models/final_model --mode evaluate --input path/to/dataset --output ./evaluation_results
```
此模式會生成ROC曲線、混淆矩陣、類別分布圖和完整的評估報告。

## 系統需求
- Python 3.8+
- PyTorch >= 2.0.0
- Transformers >= 4.30.0 (建議 4.53.0+)
- OpenCV >= 4.8.0
- scikit-learn >= 1.3.0
- 其他完整依賴請參考根目錄的 `requirements.txt`

### 硬體需求
- 建議使用GPU進行訓練（支援CUDA自動檢測）
- 最少16GB RAM
- 充足的儲存空間用於DICOM資料和模型檢查點

## 模型架構
使用預訓練的 Vision Transformer (google/vit-base-patch16-224) 進行胸部CT影像的四分類任務。

### 分類標籤
- **A系列**: 正常胸部CT影像
- **B系列**: 特定病理類型
- **E系列**: 另一病理類型
- **G系列**: 第三種病理類型

### 模型特性
- 基於Transformer架構的視覺模型
- 支援224x224像素輸入
- 動態類別檢測和適配
- 內建注意力機制可視化

## 輸出結構
訓練和推理會產生以下輸出：

### 模型檔案
- **`../CT_ViT/models/final_model/`** - 最終訓練完成的模型
- **`../CT_ViT/models/checkpoint-*/`** - 訓練過程中的檢查點

### 評估結果
- **`../CT_ViT/evaluation_*.json`** - 詳細的評估指標（準確率、精確率、召回率、F1分數）
- **`../CT_ViT/confusion_matrix_*.png`** - 混淆矩陣視覺化
- **`../CT_ViT/roc_curves.png`** - ROC曲線圖
- **`../CT_ViT/class_distribution.png`** - 類別分布圖

### 訓練日誌
- **`../CT_ViT/logs/`** - 詳細的訓練日誌
- **`../CT_ViT/training_log.json`** - 訓練過程記錄
- **`../CT_ViT/tensorboard/`** - TensorBoard視覺化檔案

## 功能特色
1. **自動化訓練流程** - 完整的端到端訓練管道
2. **多模式推理** - 支援單張、批次和評估模式
3. **詳細評估指標** - 包含分類報告、混淆矩陣、ROC曲線等
4. **視覺化輸出** - 自動產生各種圖表和分析結果
5. **檢查點恢復** - 支援訓練中斷後的繼續訓練
6. **配置管理** - 靈活的配置系統支援不同實驗設定

## 性能指標
最新訓練結果顯示模型在測試集上達到約63%的準確率，驗證集準確率約69%，適用於胸部CT影像的初步篩檢和分類任務。

## 常見問題與疑難排解

### 1. 版本相容性問題
如果遇到 `evaluation_strategy` 參數錯誤，請確保使用 Transformers >= 4.53.0：
```bash
pip install transformers>=4.53.0
```

### 2. CUDA記憶體不足
如果GPU記憶體不足，可以調整批次大小：
- 修改 `src/config.py` 中的 `batch_size` 參數
- 或在訓練時使用較小的批次大小

### 3. DICOM檔案讀取問題
確保DICOM檔案格式正確且可讀取：
```python
import pydicom
ds = pydicom.dcmread('your_file.dcm')
```

### 4. Wandb整合問題
如果不想使用Wandb記錄，設置環境變數：
```bash
export WANDB_DISABLED=true  # Linux/Mac
$env:WANDB_DISABLED="true"  # Windows PowerShell
```

## 開發與擴展

### 自定義配置
可以通過修改 `src/config.py` 來調整：
- 學習率和訓練參數
- 模型架構設定
- 資料載入配置

### 添加新的評估指標
在 `src/utils.py` 中可以添加自定義的評估函數和指標計算方法。

### 模型改進
- 可以嘗試不同的預訓練模型
- 調整影像預處理方式
- 實驗不同的優化器和學習率調度策略

## 授權與貢獻
此專案為研究用途開發，歓迎提交問題報告和改進建議。
