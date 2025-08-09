# Faster R-CNN 胸部 CT 腫瘤檢測訓練系統

基於 Faster R-CNN 的胸部 CT 腫瘤目標檢測模型，專門針對二分類檢測任務（背景 vs 病灶），提供兩種訓練模式：**K-Fold 交叉驗證**和**簡單訓練/驗證分割**。

## 🌟 主要特色

- **🎯 二分類檢測**: 精確區分背景與病灶並預測邊界框
- **🏗️ Faster R-CNN架構**: 採用 ResNet50-FPN 作為骨幹網路
- **🔄 雙重訓練模式**: K-Fold 交叉驗證 + 簡單訓練/驗證分割
- **📊 詳細記錄**: 自動生成訓練日誌和 TensorBoard 視覺化
- **📈 進度條顯示**: 使用 tqdm 提供詳細的訓練進度
- **⚙️ 靈活配置**: 支援多種參數自訂選項

## 🚀 快速開始

### 基本要求

確保您的資料位於正確位置：
```
E:\GitHub\chest-ct-report-generator\datasets\splited_dataset\
├── train/
│   ├── train_patients.txt          # 訓練患者列表
│   └── [patient_folders]/          # 患者資料夾
│       ├── dicom_files/            # DICOM 檔案
│       └── xml_annotations/        # XML 標註檔案
└── test/
    ├── test_patients.txt           # 測試患者列表
    └── [patient_folders]/          # 患者資料夾
```

### 🔥 訓練命令

## 🎯 方法一：K-Fold 交叉驗證（推薦用於模型評估）

**1. 基本 K-Fold 訓練**
```bash
python detection\train_detection.py
```

**2. 自訂 K-Fold 參數**
```bash
python detection\train_detection.py --k_folds 5 --num_epochs 50 --batch_size 8
```

**3. 快速測試（少量 epoch）**
```bash
python detection\train_detection.py --k_folds 2 --num_epochs 5 --batch_size 4
```

## 🚀 方法二：簡單訓練/驗證分割（推薦用於快速訓練）

**1. 基本簡單訓練**
```bash
python detection\train_detection_simple.py
```

**2. 自訂驗證集比例**
```bash
python detection\train_detection_simple.py --num_epochs 50 --val_split 0.2 --batch_size 8
```

**3. 快速測試**
```bash
python detection\train_detection_simple.py --num_epochs 5 --val_split 0.1 --batch_size 4
```

## ⚙️ 參數配置

### 📋 K-Fold 訓練參數 (`train_detection.py`)

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--data_dir` | `./datasets/splited_dataset` | 數據集目錄路徑 |
| `--k_folds` | `2` | K-fold交叉驗證的fold數量 |
| `--num_epochs` | `50` | 每個fold的訓練輪數 |
| `--batch_size` | `16` | 批次大小 |
| `--learning_rate` | `0.0001` | 學習率 |
| `--save_dir` | `./Faster_RCNN_Detection/models` | 模型保存目錄 |
| `--log_dir` | `./Faster_RCNN_Detection/logs` | 日誌保存目錄 |
| `--random_seed` | `42` | 隨機種子 |

### 📋 簡單訓練參數 (`train_detection_simple.py`)

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--data_dir` | `./datasets/splited_dataset` | 數據集目錄路徑 |
| `--num_epochs` | `50` | 訓練輪數 |
| `--batch_size` | `16` | 批次大小 |
| `--learning_rate` | `0.0001` | 學習率 |
| `--val_split` | `0.2` | 驗證集比例 (0.0-1.0) |
| `--save_dir` | `./Simple_Training/models` | 模型保存目錄 |
| `--log_dir` | `./Simple_Training/logs` | 日誌保存目錄 |
| `--random_seed` | `42` | 隨機種子 |

## 🔧 詳細訓練配置說明

### 📝 當前預設配置（適合快速測試）

#### 基本參數
- **K-Fold數量**: 2 (減少以加快測試)
- **訓練輪數**: 50 (標準訓練，快速測試可調整為 5)
- **批次大小**: 16 (標準值，記憶體不足時可調整為 4)
- **學習率**: 0.0001 (標準值)

#### 優化參數
- **梯度累積批次數**: 2 (補償較小的批次大小)
- **驗證檢查間隔**: 1 (每個epoch都驗證，便於觀察進度)
- **隨機種子**: 42 (保證結果可重現)

#### 目錄設置
- **K-Fold模型保存目錄**: `detection/Faster_RCNN_Detection/models`
- **K-Fold日誌保存目錄**: `detection/Faster_RCNN_Detection/logs`
- **簡單訓練模型保存目錄**: `detection/Simple_Training/models`
- **簡單訓練日誌保存目錄**: `detection/Simple_Training/logs`

### ⚡ 預期效果
- **訓練時間**: 約 30-60 分鐘（視硬件而定）
- **內存使用**: 相對較低（適合普通GPU）
- **進度顯示**: 每個epoch都會顯示驗證結果

### 🔄 進階配置與範例

#### 完整訓練配置（生產環境推薦）
```bash
# K-Fold: 5-fold 交叉驗證，完整訓練
python detection\train_detection.py --k_folds 5 --num_epochs 50 --batch_size 16

# 簡單訓練: 完整配置
python detection\train_detection_simple.py --num_epochs 50 --batch_size 16 --val_split 0.2
```

#### 記憶體優化配置
```bash
# 減少批次大小和 fold 數量
python detection\train_detection.py --batch_size 2 --k_folds 3
python detection\train_detection_simple.py --batch_size 2
```

#### 自訂配置範例
```bash
# 高學習率與自訂種子
python detection\train_detection.py --learning_rate 0.0005 --random_seed 123

# 自訂保存路徑
python detection\train_detection_simple.py --save_dir "./custom_models" --random_seed 123
```

## 📊 訓練輸出

### K-Fold 訓練結果 (`train_detection.py`)
```
detection/Faster_RCNN_Detection/
├── models/
│   ├── best_model_fold_1.pth      # Fold 1 最佳模型
│   ├── best_model_fold_2.pth      # Fold 2 最佳模型
│   ├── ...                        # 其他 fold 模型
│   └── kfold_results.json         # K-fold 總體結果
├── logs/
│   ├── fold_1/                    # Fold 1 TensorBoard 記錄
│   ├── fold_2/                    # Fold 2 TensorBoard 記錄
│   ├── ...                        # 其他 fold 記錄
│   └── kfold_training_*.log       # 訓練日誌
├── visualizations_fold_1/
│   ├── fold_1_predictions_sample_1.png  # Fold 1 預測結果可視化
│   ├── fold_1_predictions_sample_2.png
│   ├── ...
│   └── fold_1_summary.png         # Fold 1 統計摘要
├── visualizations_fold_2/
│   └── ...                        # 其他 fold 可視化
└── kfold_summary_visualizations/
    └── kfold_summary.png          # K-fold 整體結果摘要
```

### 簡單訓練結果 (`train_detection_simple.py`)
```
detection/Simple_Training/
├── models/
│   ├── best_model.pth             # 最佳模型（包含完整檢查點）
│   ├── best_model_weights.pth     # 純模型權重（用於推理）
│   ├── checkpoint_epoch_*.pth     # 定期檢查點
│   └── training_results.json     # 詳細訓練結果
├── logs/
│   ├── events.out.tfevents.*      # TensorBoard 記錄
│   └── simple_training_*.log      # 訓練日誌
└── visualizations/
    ├── final_predictions_sample_1.png  # 預測結果可視化
    ├── final_predictions_sample_2.png
    ├── ...
    └── final_summary.png          # 訓練統計摘要
```

### 結果分析文件

**K-Fold: kfold_results.json 包含：**
- 平均精確度、召回率、F1分數及標準差
- 每個 fold 的詳細結果
- 總訓練時間統計
- 訓練配置參數

**簡單訓練: training_results.json 包含：**
- 最終模型性能指標
- 完整訓練歷史（每個 epoch 的 loss 和指標）
- 訓練配置和數據集統計

### 🎨 自動可視化功能

**兩種訓練方法都會自動生成：**
- **預測結果對比圖**：真實標註 vs 模型預測的視覺對比
- **統計摘要圖表**：預測框數量分佈、置信度分佈等統計分析
- **K-fold 專用**：各 fold 性能比較圖表和整體結果摘要

**可視化特色：**
- 根據置信度使用不同顏色標示預測框
- 詳細的統計圖表和分佈分析
- 自動保存高清 PNG 格式圖片

## 📈 監控和分析

### 🖥️ 實時監控
- **詳細進度條**: 終端會顯示詳細的訓練進度
- **關鍵指標**: 包含Loss、F1分數等關鍵指標的實時更新
- **驗證結果**: 每個epoch的驗證結果即時顯示

### 📊 TensorBoard 視覺化

**K-Fold 訓練：**
```bash
# 查看所有 fold 的訓練過程
tensorboard --logdir detection/Faster_RCNN_Detection/logs

# 查看特定 fold
tensorboard --logdir detection/Faster_RCNN_Detection/logs/fold_1
```

**簡單訓練：**
```bash
# 查看訓練過程
tensorboard --logdir detection/Simple_Training/logs
```

### 📋 查看結果

**K-Fold 結果：**
```powershell
# 查看 K-fold 總體結果
type detection\Faster_RCNN_Detection\models\kfold_results.json

# 查看訓練日誌 (可使用 PowerShell 或文本編輯器)
Get-Content detection\Faster_RCNN_Detection\logs\kfold_training_*.log -Tail 20
```

**簡單訓練結果：**
```powershell
# 查看訓練結果
type detection\Simple_Training\models\training_results.json

# 查看訓練日誌
Get-Content detection\Simple_Training\logs\simple_training_*.log -Tail 20
```

## 🔍 模型推理

**K-Fold 訓練的模型：**
```bash
# 使用最佳 fold 模型進行推理
python detection\inference_detection.py --model_path Faster_RCNN_Detection\models\best_model_fold_1.pth --input_dicom path\to\dicom.dcm --confidence_threshold 0.5
```

**簡單訓練的模型：**
```bash
# 使用簡單訓練的最佳模型進行推理
python detection\inference_detection.py --model_path Simple_Training\models\best_model_weights.pth --input_dicom path\to\dicom.dcm --confidence_threshold 0.5
```

## 🏗️ 模型架構

### Faster R-CNN 組件
- **骨幹網路**: ResNet50-FPN (預訓練於COCO)
- **區域提議網路(RPN)**: 生成候選邊界框
- **分類頭**: 二分類（背景 vs 病灶）
- **回歸頭**: 邊界框座標預測

### K-Fold 交叉驗證優勢
1. **可靠評估**: 多次驗證減少隨機性影響
2. **充分利用數據**: 每個樣本都用於訓練和驗證
3. **統計穩定性**: 提供指標的均值和標準差
4. **模型選擇**: 可選擇最佳 fold 的模型

### 簡單訓練/驗證分割優勢
1. **快速訓練**: 單次訓練即可完成
2. **簡單管理**: 只需管理一個模型
3. **快速迭代**: 適合參數調優和快速實驗
4. **直觀結果**: 訓練過程清晰易懂

### 🎯 選擇建議
- **研究和論文發表**: 使用 K-Fold 交叉驗證 (`train_detection.py`)
- **快速原型和測試**: 使用簡單訓練 (`train_detection_simple.py`)
- **生產環境部署**: 可使用任一方法，但建議先用簡單訓練快速驗證

## 🛠️ 故障排除

### 常見問題

**Q: 記憶體不足錯誤**
```bash
# K-Fold: 減少批次大小和 fold 數量
python detection\train_detection.py --batch_size 4 --k_folds 3

# 簡單訓練: 減少批次大小
python detection\train_detection_simple.py --batch_size 4
```

**Q: 數據集找不到**
```powershell
# 檢查數據路徑（兩種方法相同）
python detection\train_detection.py --data_dir "./datasets/splited_dataset"
python detection\train_detection_simple.py --data_dir "./datasets/splited_dataset"

# 確認數據集結構正確
dir datasets\splited_dataset\train\
```

**Q: 訓練時間過長**
```bash
# K-Fold: 減少 epoch 和 fold 數量
python detection\train_detection.py --num_epochs 20 --k_folds 3

# 簡單訓練: 減少 epoch 數量
python detection\train_detection_simple.py --num_epochs 20
```

**Q: 如何選擇訓練方法？**
- **追求最高準確性**: 使用 K-Fold 交叉驗證
- **快速測試想法**: 使用簡單訓練
- **第一次訓練**: 建議先用簡單訓練熟悉流程

**Q: PyTorch 版本相關問題**
```bash
# 確保使用支援的 PyTorch 版本
pip install torch>=2.0.0

# 檢查版本
python -c "import torch; print(torch.__version__)"
```

**Q: 可視化生成失敗**
```bash
# 檢查可視化依賴
pip install matplotlib>=3.7.0 opencv-python>=4.8.0
```

**Q: 數據載入錯誤如何解決？**
```powershell
# 檢查數據目錄路徑是否正確
dir datasets\splited_dataset\train\
dir datasets\splited_dataset\test\

# 確認患者列表文件存在
type datasets\splited_dataset\train\train_patients.txt
type datasets\splited_dataset\test\test_patients.txt
```

**Q: 如何中斷和恢復訓練？**
- **K-Fold訓練**: 每個fold結束後會保存模型，可以手動修改代碼從特定fold開始
- **簡單訓練**: 支持檢查點保存，可從最近的檢查點恢復訓練

### 檢查系統狀態
```bash
# 測試數據載入
python detection\faster_rcnn_dataset.py

# 檢查 GPU 可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📚 技術細節

### 評估指標
- **Precision**: 檢測精確度（TP / (TP + FP)）
- **Recall**: 檢測召回率（TP / (TP + FN)）
- **F1-Score**: 綜合評估指標（2 * Precision * Recall / (Precision + Recall)）
- **IoU**: 邊界框重疊度閾值（預設 0.5）

### 訓練配置
- **優化器**: AdamW (weight_decay=0.0001)
- **學習率調度**: CosineAnnealingLR
- **損失函數**: Faster R-CNN 內建損失（分類 + 回歸 + RPN）
- **模型輸入**: 512x512 灰階影像轉 RGB
- **隨機種子**: 支援全局隨機種子設定確保結果可重現
- **PyTorch 兼容**: 支援 PyTorch 2.6+ 版本

### 🔄 主要功能
- **🎯 隨機種子支援**: 兩種訓練方法都支援 `--random_seed` 參數
- **🎨 自動可視化**: 訓練完成後自動生成預測結果和統計圖表
- **📊 詳細記錄**: 完整的訓練日誌和進度顯示
- **📈 實時監控**: 詳細進度條顯示 epoch/fold 進度、loss 和 F1 分數

## 🎯 最佳實踐

### 通用建議
1. **資料檢查**: 確保 DICOM 檔案與 XML 標註的 SOPInstanceUID 匹配
2. **參數調整**: 根據 GPU 記憶體調整 batch_size
3. **日誌監控**: 使用 TensorBoard 監控訓練過程
4. **隨機種子**: 固定隨機種子確保結果可重現

### 訓練方法選擇
- **研究和論文發表**: 使用 K-Fold 交叉驗證（更可靠的評估）
- **快速原型和測試**: 使用簡單訓練（快速迭代）
- **生產環境部署**: 建議先用簡單訓練快速驗證，再用 K-Fold 完整評估

## 🚨 注意事項

### K-Fold 交叉驗證
- **訓練時間**: 訓練時間是簡單訓練的 K 倍
- **數據平衡**: 確保各 fold 中病灶樣本分佈均勻
- **結果解讀**: 關注指標的標準差，過大表示模型不穩定

### 簡單訓練/驗證分割
- **驗證集大小**: 通常設置為 15-25% (0.15-0.25)
- **過擬合風險**: 需要密切監控驗證指標

### 通用注意事項
- **數據質量**: 確保 DICOM 和 XML 標註的一致性
- **GPU 記憶體**: 根據硬體調整批次大小
- **進度監控**: 關注訓練過程中的 loss 變化

## 🔗 相關文件

### 主要訓練腳本
- `train_detection.py`: K-Fold 交叉驗證訓練腳本
- `train_detection_simple.py`: 簡單訓練/驗證分割腳本

### 支援模組
- `faster_rcnn_dataset.py`: 數據載入模組
- `faster_rcnn_model.py`: 模型定義（如果需要）
- `inference_detection.py`: 推理腳本（如果需要）

### 配置和文檔
- `README.md`: 本文檔
- `requirements.txt`: Python 依賴套件
- `config/`: 配置文件目錄（如果存在）

## 📋 更新歷史

### 最新更新 (2025-08)
- ✅ **PyTorch 2.6+ 兼容性**: 修正模型載入警告
- ✅ **隨機種子支援**: 確保結果可重現
- ✅ **自動可視化**: 預測結果和統計圖表自動生成
- ✅ **文檔整合**: 整合所有訓練配置到單一文檔
