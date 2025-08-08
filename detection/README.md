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

### 範例命令

```bash
# K-Fold: 10-fold 交叉驗證，較高學習率
python detection\train_detection.py --k_folds 10 --learning_rate 0.0005

# K-Fold: 減少記憶體使用
python detection\train_detection.py --batch_size 4 --num_epochs 30

# K-Fold: 自訂隨機種子
python detection\train_detection.py --k_folds 5 --random_seed 123

# 簡單訓練: 小驗證集，快速訓練
python detection\train_detection_simple.py --val_split 0.1 --num_epochs 30

# 簡單訓練: 自訂路徑和種子
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

### TensorBoard 視覺化

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

### 查看結果

**K-Fold 結果：**
```bash
# 查看 K-fold 總體結果
cat detection/Faster_RCNN_Detection/models/kfold_results.json

# 查看訓練日誌
tail -f detection/Faster_RCNN_Detection/logs/kfold_training_*.log
```

**簡單訓練結果：**
```bash
# 查看訓練結果
cat detection/Simple_Training/models/training_results.json

# 查看訓練日誌
tail -f detection/Simple_Training/logs/simple_training_*.log
```

## 🔍 模型推理

**K-Fold 訓練的模型：**
```bash
# 使用最佳 fold 模型進行推理
python detection\inference_detection.py \
  --model_path Faster_RCNN_Detection/models/best_model_fold_1.pth \
  --input_dicom path/to/dicom.dcm \
  --confidence_threshold 0.5
```

**簡單訓練的模型：**
```bash
# 使用簡單訓練的最佳模型進行推理
python detection\inference_detection.py \
  --model_path Simple_Training/models/best_model_weights.pth \
  --input_dicom path/to/dicom.dcm \
  --confidence_threshold 0.5
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
```bash
# 檢查數據路徑（兩種方法相同）
python detection\train_detection.py --data_dir "./datasets/splited_dataset"
python detection\train_detection_simple.py --data_dir "./datasets/splited_dataset"

# 確認數據集結構正確
ls datasets/splited_dataset/train/
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

**Q: PyTorch 載入警告或錯誤**
```bash
# 如果遇到 torch.load 相關警告，代碼已自動處理
# 確保使用 PyTorch 2.6+ 版本
pip install torch>=2.0.0

# 檢查版本
python -c "import torch; print(torch.__version__)"
```

**Q: 可視化生成失敗**
```bash
# 檢查可視化依賴
pip install matplotlib>=3.7.0 opencv-python>=4.8.0

# 如果可視化失敗，訓練仍會繼續，檢查日誌：
tail -f detection/Simple_Training/logs/simple_training_*.log
```

**Q: 結果不可重現**
```bash
# 使用固定的隨機種子
python detection\train_detection.py --random_seed 42
python detection\train_detection_simple.py --random_seed 42
```

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

### 🔄 最新更新功能
- **🎯 隨機種子支援**: 兩種訓練方法都支援 `--random_seed` 參數
- **🔧 PyTorch 2.6+ 兼容**: 修正 `torch.load` 和模型載入相關警告
- **🎨 自動可視化**: 訓練完成後自動生成預測結果和統計圖表
- **📊 增強記錄**: 更詳細的訓練日誌和進度顯示

### K-Fold 流程
1. 載入完整數據集
2. 隨機分割為 K 個 fold
3. 依序訓練 K 個模型（每次用 K-1 個 fold 訓練，1 個 fold 驗證）
4. 計算平均指標和標準差
5. 保存所有模型和結果

### 簡單訓練流程
1. 載入完整數據集
2. 隨機分割為訓練集和驗證集
3. 訓練單一模型
4. 每個 epoch 進行驗證
5. 保存最佳模型和訓練歷史

### 🚀 訓練進度顯示
兩種訓練方法都配備了詳細的進度條：
- **整體進度**: 顯示 epoch/fold 進度
- **批次進度**: 顯示當前 loss 和平均 loss
- **驗證進度**: 顯示驗證過程
- **實時指標**: 動態更新 F1 分數等指標

## 🎯 最佳實踐

### 通用建議
1. **資料檢查**: 確保 DICOM 檔案與 XML 標註的 SOPInstanceUID 匹配
2. **參數調整**: 根據 GPU 記憶體調整 batch_size
3. **日誌監控**: 使用 TensorBoard 監控訓練過程

### K-Fold 特定建議
4. **結果分析**: 檢查各 fold 間的性能一致性
5. **模型選擇**: 選擇表現最佳的 fold 模型用於推理
6. **時間規劃**: K-fold 訓練時間是簡單訓練的 K 倍

### 簡單訓練特定建議
4. **驗證集大小**: 通常設置為 15-25% (0.15-0.25)
5. **隨機種子**: 固定隨機種子確保結果可重現
6. **檢查點**: 利用定期保存的檢查點進行訓練恢復

## 🚨 注意事項

### K-Fold 交叉驗證
- **記憶體需求**: 需要更多時間和儲存空間
- **數據平衡**: 確保各 fold 中病灶樣本分佈均勻
- **訓練時間**: 訓練時間是簡單訓練的 K 倍
- **結果解讀**: 關注指標的標準差，過大表示模型不穩定

### 簡單訓練/驗證分割
- **隨機性**: 單次分割可能存在偏差
- **驗證集選擇**: 驗證集大小影響訓練穩定性
- **過擬合風險**: 需要密切監控驗證指標
- **可重現性**: 依賴隨機種子設置

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
- ✅ **PyTorch 2.6+ 兼容性**: 修正 `torch.load` 和模型載入警告
- ✅ **隨機種子支援**: 兩種訓練方法都支援 `--random_seed` 參數
- ✅ **自動可視化**: 添加預測結果和統計圖表自動生成
- ✅ **增強記錄**: 更詳細的訓練過程記錄和進度顯示
- ✅ **代碼同步**: 確保兩個訓練腳本功能一致性
- ✅ **文檔整合**: 將所有更新信息整合到主 README 文檔中
