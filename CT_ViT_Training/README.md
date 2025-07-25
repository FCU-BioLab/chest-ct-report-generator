# CT-ViT Training Module

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
