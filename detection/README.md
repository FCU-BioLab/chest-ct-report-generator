# Faster R-CNN 胸部 CT 腫瘤檢測訓練系統

基於 Faster R-CNN 的胸部 CT 腫瘤目標檢測模型，專門針對二分類檢測任務（背景 vs 病灶），提供兩種訓練模式：**K-Fold 交叉驗證**和**簡單訓練/驗證分割**。

## 🌟 主要特色

- **🎯 二分類檢測**: 精確區分背景與病灶並預測邊界框
- **🏗️ Faster R-CNN架構**: 採用 ResNet50-FPN 作為骨幹網路
- **🔄 雙重訓練模式**: K-Fold 交叉驗證 + 簡單訓練/驗證分割
- **📊 統一評估指標**: 三個檢測腳本都支援22項專業評估指標（mAP、IoU變體、ROC/FROC等）
- **🧪 全面指標同步**: `train_detection.py`、`train_detection_simple.py`、`test_detection.py` 都具備相同的評估能力
- **📊 詳細記錄**: 自動生成訓練日誌和 TensorBoard 視覺化
- **📈 進度條顯示**: 使用 tqdm 提供詳細的訓練進度
- **🎨 豐富可視化**: 自動生成預測結果、統計圖表、ROC曲線等
- **⚙️ 靈活配置**: 支援多種參數自訂選項

## 🚀 快速開始

### 基本要求

## 🧪 模型測試與評估 (`test/test_detection.py`)

### 🎯 快速開始

#### 基本測試（預設已啟用全面評估）
```bash
python detection\test\test_detection.py --model_path "./path/to/best_model.pth"
```

#### 自定義測試配置
```bash
# 指定測試數據集和結果保存路徑
python detection\test\test_detection.py \
  --model_path "./Faster_RCNN_Detection/models/best_model_fold_1.pth" \
  --data_dir "./datasets/splited_dataset" \
  --output_dir "./test_results" \
  --confidence_threshold 0.5 \
  --device cuda
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

**4. 全面評估模式（包含22項專業指標）**
```bash
python detection\train_detection_simple.py --num_epochs 50 --batch_size 16 --comprehensive_eval
```

**5. K-Fold 全面評估模式**
```bash
python detection\train_detection.py --k_folds 5 --num_epochs 50 --batch_size 16
# K-Fold 訓練預設已啟用全面評估指標
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
| `--comprehensive_eval` | `False` | 是否使用全面評估指標 |
| `--confidence_threshold` | `0.5` | 置信度閾值 |

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
└── visualizations/                # 可視化結果（啟用--comprehensive_eval時額外生成）
    ├── final_predictions_sample_*.png      # 預測結果可視化
    ├── final_summary.png                  # 基本統計摘要
    ├── comprehensive_metrics.png          # 全面指標雷達圖（僅全面評估）
    ├── comprehensive_metrics_report.txt   # 詳細文字報告（僅全面評估）
    └── roc_froc_curves.png               # ROC/FROC曲線（僅全面評估）
```

### 結果分析文件

**K-Fold: kfold_results.json 包含：**
- 平均精確度、召回率、F1分數及標準差
- 每個 fold 的詳細結果
- 總訓練時間統計
- 訓練配置參數

**簡單訓練: training_results.json 包含：**
- 最終模型性能指標（基本或全面，取決於是否使用 `--comprehensive_eval`）
- 完整訓練歷史（每個 epoch 的 loss 和指標）
- 訓練配置和數據集統計
- 如果啟用全面評估：ROC/FROC分析結果、IoU變體指標等

### 🎨 自動可視化功能

**基本可視化（兩種訓練方法都有）：**
- **預測結果對比圖**：真實標註 vs 模型預測的視覺對比
- **統計摘要圖表**：預測框數量分佈、置信度分佈等統計分析
- **K-fold 專用**：各 fold 性能比較圖表和整體結果摘要

**全面評估可視化（簡單訓練 + `--comprehensive_eval`）：**
- **綜合指標雷達圖**：包含所有核心指標的雷達圖
- **ROC/FROC 曲線**：臨床相關的敏感度分析
- **詳細統計報告**：完整的文字評估報告
- **IoU 品質分析**：GIoU、DIoU、CIoU 比較圖

**可視化特色：**
- 根據置信度使用不同顏色標示預測框
- 詳細的統計圖表和分佈分析
- 自動保存高清 PNG 格式圖片
- 專業的醫學影像分析圖表

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

## 🎯 全面評估指標系統

`train_detection_simple.py` 支援全面的評估指標，提供比基本 Precision/Recall/F1 更詳細的模型評估。

### 🚀 啟用全面評估

```bash
# 基本使用（向後兼容）
python detection\train_detection_simple.py --num_epochs 50 --batch_size 16

# 啟用全面評估指標
python detection\train_detection_simple.py --num_epochs 50 --batch_size 16 --comprehensive_eval

# 自定義置信度閾值
python detection\train_detection_simple.py --confidence_threshold 0.3 --comprehensive_eval
```

### 📊 評估指標分類

#### 一、核心檢測指標（必備）

| 指標 | 英文名稱 | 說明 | 範圍 |
|------|----------|------|------|
| **IoU** | Intersection over Union | 預測框與真實框的重疊度 | [0, 1] |
| **mAP@0.5** | Mean Average Precision at IoU=0.5 | IoU=0.5閾值下的平均精度 | [0, 1] |
| **mAP@[0.5:0.95]** | Mean Average Precision at IoU=0.5:0.95 | COCO標準，多個IoU閾值的平均精度 | [0, 1] |
| **Sensitivity/Recall** | 敏感度/召回率 | 正確檢測出的病灶比例 | [0, 1] |
| **Precision** | 精確率 | 檢測結果中真正病灶的比例 | [0, 1] |
| **F1-score** | F1分數 | 精確率與召回率的調和平均 | [0, 1] |
| **FP per Image** | False Positives per Image | 每張影像的平均假陽性數 | [0, ∞) |

#### 二、定位與錯誤分析指標（必備）

| 指標 | 英文名稱 | 說明 | 範圍 |
|------|----------|------|------|
| **Lesion-level Sensitivity** | 病灶級敏感度 | 個別病灶被正確檢測的比例 | [0, 1] |
| **Case-level Sensitivity** | 病例級敏感度 | 有病灶的病例被正確檢測的比例 | [0, 1] |
| **Bounding Box Error** | 邊界框誤差 | 預測框與真實框的位置/大小誤差 | [0, ∞) |
| **GIoU** | Generalized IoU | 考慮非重疊區域的改進IoU | [-1, 1] |
| **DIoU** | Distance IoU | 考慮中心點距離的改進IoU | [-1, 1] |
| **CIoU** | Complete IoU | 考慮寬高比一致性的改進IoU | [-1, 1] |

#### 三、臨床相關指標（建議）

| 指標 | 英文名稱 | 說明 | 範圍 |
|------|----------|------|------|
| **ROC Curve** | Receiver Operating Characteristic | 接收者操作特徵曲線 | - |
| **AUC** | Area Under Curve | ROC曲線下面積 | [0, 1] |
| **FROC** | Free-response ROC | 敏感度 vs 每張圖的假陽性數 | - |
| **Mean Localization Error** | 平均定位誤差 | 預測框中心點與真實框的平均距離 | [0, ∞) |

#### 四、效率與可用性指標（可選）

| 指標 | 英文名稱 | 說明 | 單位 |
|------|----------|------|------|
| **Inference Time** | 推論時間 | 每張影像的平均處理時間 | 毫秒 |
| **FPS** | Frames Per Second | 每秒處理的影像數量 | 張/秒 |
| **Memory Usage** | 顯存使用量 | GPU記憶體佔用量 | MB |

### 📈 全面評估輸出

當啟用 `--comprehensive_eval` 時，訓練完成會生成：

**控制台詳細輸出：**
```
=== 最終全面評估結果 ===
精確度 (Precision): 0.8500
召回率/敏感度 (Recall/Sensitivity): 0.7800
F1分數 (F1-Score): 0.8137
mAP@0.5: 0.7650
mAP@[0.5:0.95]: 0.5420
病灶級敏感度: 0.7800
病例級敏感度: 0.8200
每圖平均假陽性數: 0.3200
IoU: 0.6850
GIoU: 0.6425
DIoU: 0.6520
CIoU: 0.6380
每圖推理時間: 125.50 ms
FPS: 7.97
ROC AUC: 0.8750
```

**額外可視化文件：**
```
Simple_Training/models/visualizations/
├── comprehensive_metrics.png           # 全面指標雷達圖和統計圖
├── comprehensive_metrics_report.txt    # 詳細文字報告
├── roc_froc_curves.png                # ROC和FROC曲線
├── final_predictions_sample_*.png     # 預測結果可視化
└── final_summary.png                  # 統計摘要圖
```

### 🎯 指標解讀建議

#### 優秀模型的指標範圍

| 指標 | 優秀 | 良好 | 需改進 |
|------|------|------|--------|
| F1-Score | >0.80 | 0.60-0.80 | <0.60 |
| mAP@0.5 | >0.75 | 0.50-0.75 | <0.50 |
| Case-level Sensitivity | >0.85 | 0.70-0.85 | <0.70 |
| FP per Image | <0.5 | 0.5-2.0 | >2.0 |
| IoU | >0.65 | 0.50-0.65 | <0.50 |

#### 關鍵指標關注點

1. **臨床應用**: 優先關注 Case-level Sensitivity 和 FP per Image
2. **研究比較**: 使用 mAP@0.5 和 mAP@[0.5:0.95] 進行客觀比較
3. **定位品質**: 關注 IoU、GIoU、DIoU、CIoU 的值
4. **系統部署**: 考慮 Inference Time 和 Memory Usage

### 🔧 評估指標依賴

#### 必需依賴
```bash
pip install torch torchvision numpy matplotlib tqdm
```

#### 可選依賴（ROC/AUC計算）
```bash
pip install scikit-learn
```

如果未安裝 scikit-learn，ROC/AUC 相關指標將自動跳過。

### 🧪 測試評估指標

```bash
# 運行測試腳本驗證指標計算
python detection\test_comprehensive_metrics.py
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
- **詳細模型分析**: 使用簡單訓練 + 全面評估 (`train_detection_simple.py --comprehensive_eval`)
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

**Q: 全面評估相關問題**
```bash
# scikit-learn 未安裝（ROC/AUC 指標顯示為0）
pip install scikit-learn

# 記憶體不足時禁用全面評估
python detection\train_detection_simple.py --batch_size 4
# 不使用 --comprehensive_eval 標誌

# 測試評估指標是否正常
python detection\test_comprehensive_metrics.py
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

### 基本評估指標
- **Precision**: 檢測精確度（TP / (TP + FP)）
- **Recall**: 檢測召回率（TP / (TP + FN)）
- **F1-Score**: 綜合評估指標（2 * Precision * Recall / (Precision + Recall)）
- **IoU**: 邊界框重疊度閾值（預設 0.5）

### 全面評估指標（`--comprehensive_eval`）
- **mAP@0.5 & mAP@[0.5:0.95]**: COCO標準平均精度
- **GIoU/DIoU/CIoU**: 改進型IoU變體，更準確衡量定位品質
- **病例級/病灶級敏感度**: 臨床相關的檢測敏感度
- **ROC/FROC分析**: 敏感度vs假陽性率分析
- **效率指標**: 推理時間、FPS、記憶體使用量

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
5. **評估選擇**: 
   - 快速測試: 使用基本評估（不加 `--comprehensive_eval`）
   - 完整分析: 使用全面評估（加 `--comprehensive_eval`）

### 訓練方法選擇
- **研究和論文發表**: 使用 K-Fold 交叉驗證（更可靠的評估）
- **快速原型和測試**: 使用簡單訓練（快速迭代）
- **生產環境部署**: 建議先用簡單訓練快速驗證，再用 K-Fold 完整評估
- **詳細模型分析**: 使用簡單訓練 + 全面評估指標（`--comprehensive_eval`）

```

## 🧪 模型測試與評估 (`test_detection.py`)

### � 快速開始

#### 基本測試（預設已啟用全面評估）
```bash
python detection	est_detection.py --model_path "./path/to/best_model.pth"
```

#### 自定義測試配置
```bash
# 指定測試數據集和結果保存路徑
python detection	est_detection.py \
  --model_path "./Faster_RCNN_Detection/models/best_model_fold_1.pth" \
  --data_dir "./datasets/splited_dataset" \
  --output_dir "./test_results" \
  --confidence_threshold 0.5 \
  --device cuda
```

### 📊 測試結果輸出

測試完成後會自動生成以下內容：

**控制台詳細報告：**
- 22項全面評估指標
- 效率指標（推理時間、FPS、記憶體使用）
- 病例級和病灶級統計分析
- ROC/FROC 曲線分析結果

**文件輸出：**
```
test_results/
├── test_results.json              # 完整測試結果（JSON格式）
├── comprehensive_metrics_report.txt # 詳細文字報告
├── visualizations/
│   ├── roc_froc_curves.png       # ROC和FROC曲線
│   ├── comprehensive_metrics.png  # 指標雷達圖和統計圖
│   ├── test_predictions_sample_*.png # 預測結果可視化
│   └── test_summary.png          # 測試統計摘要
└── logs/
    └── test_detection_*.log       # 測試日誌
```

### 🎯 測試指標解讀

#### 控制台輸出示例
```
=== 模型測試完成 ===
總測試樣本數: 156
正確檢測樣本數: 143
總病灶數: 234
正確檢測病灶數: 187

=== 核心檢測指標 ===
IoU: 0.6847
mAP@0.5: 0.7234
mAP@[0.5:0.95]: 0.5678
Sensitivity/Recall: 0.7991
Precision: 0.8123
F1-score: 0.8056
FP per Image: 1.23

=== 定位與錯誤分析 ===
GIoU: 0.6234
DIoU: 0.6456
CIoU: 0.6378
Mean Localization Error: 12.34 pixels

=== 臨床相關指標 ===
ROC AUC: 0.8456
Case-level Sensitivity: 0.8654
Lesion-level Sensitivity: 0.7991

=== 效率指標 ===
Average Inference Time: 45.67 ms
FPS: 21.9 frames/second
Memory Usage: 1024 MB
```

### 🔧 測試參數配置

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model_path` | 必須指定 | 訓練好的模型檔案路徑 |
| `--data_dir` | `./datasets/splited_dataset` | 測試數據集目錄路徑 |
| `--output_dir` | `./test_results` | 測試結果保存目錄 |
| `--confidence_threshold` | `0.5` | 檢測置信度閾值 |
| `--device` | `cuda` | 計算設備（cuda/cpu） |
| `--batch_size` | `16` | 測試批次大小 |
| `--save_visualizations` | `True` | 是否保存可視化結果 |

### 📈 適用場景

1. **模型最終評估**: 訓練完成後的全面性能評估
2. **模型比較**: 比較不同訓練配置或算法的性能
3. **臨床驗證**: 評估模型在臨床應用中的可靠性
4. **性能分析**: 詳細分析模型的優勢和改進點
5. **部署前測試**: 部署到生產環境前的最終驗證

### 🔗 與訓練腳本的關係

- `test_detection.py` 使用與 `train_detection.py` 和 `train_detection_simple.py` 相同的評估指標
- 確保訓練和測試階段評估方法的一致性
- 支援測試任何由訓練腳本產生的模型檔案

## �🚨 注意事項


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
- `train_detection_simple.py`: 簡單訓練/驗證分割腳本（支援全面評估指標）

### 深度特徵提取
- `deep_feature_extractor.py`: 深度特徵提取器
- `feature_loader.py`: 特徵載入與分析工具
- `extract_features.py`: 互動式特徵提取工具

### 支援模組
- `faster_rcnn_dataset.py`: 數據載入模組
- `faster_rcnn_model.py`: 模型定義
- `gpu_dataloader_optimizer.py`: GPU資料載入優化工具

### 測試和檢查工具 (test/)
- `test_detection.py`: 主要模型測試與評估腳本（包含全面評估指標）
- `test_feature_extraction.py`: 深度特徵提取測試
- `test_faster_rcnn.py`: Faster R-CNN基本功能測試
- `check_gpu.py`: GPU環境檢查
- `check_data_size.py`: 資料大小檢查
- 更多測試工具請參考 [`test/README.md`](test/README.md)

### 配置和文檔
- `README.md`: 本文檔（包含全面評估指標說明）
- `requirements.txt`: Python 依賴套件
- `config/`: 配置文件目錄（如果存在）

## 🧠 深層特徵提取系統

### 概述

深層特徵提取系統為LLM生成報告提供結構化的特徵數據，包含以下組件：

1. **deep_feature_extractor.py** - 核心特徵提取器
2. **extract_features.py** - 便捷的交互式提取工具
3. **feature_loader.py** - 特徵加載和分析工具
4. **test/test_detection.py** - 已增強的檢測測試（包含特徵提取選項）

### 🚀 快速開始特徵提取

#### 方法1：使用便捷腳本（推薦）

```bash
cd detection
python extract_features.py
```

這將啟動交互式界面，引導您完成特徵提取過程。

#### 方法2：使用測試腳本附帶提取

```bash
cd detection
python test\test_detection.py --extract_features --split val
```

這將在模型測試的同時提取深層特徵。

#### 方法3：直接使用提取器

```bash
cd detection
python deep_feature_extractor.py --model_path "path/to/model.pth" --data_dir "path/to/data" --save_dir "./features"
```

### 📊 提取的特徵類型

#### 1. 全局特徵 (Global Features)
- **Backbone特徵**: ResNet50各層的全局平均池化特徵
  - layer1: 256維 - 低級特徵 (邊緣、紋理)
  - layer2: 512維 - 中級特徵 (形狀、模式)
  - layer3: 1024維 - 高級特徵 (對象組件)
  - layer4: 2048維 - 最高級特徵 (語義信息)
- **FPN特徵**: 特徵金字塔網絡的多尺度特徵
- **RPN特徵**: 區域建議網絡特徵
- **ROI特徵**: 感興趣區域特徵

#### 2. 檢測特徵 (Detection Features)
- 病灶檢測統計
- 邊界框信息
- 置信度分布
- 病灶面積和體積估計

#### 3. 病例級特徵 (Patient-level Features)
- 跨切片的特徵聚合 (mean, std, max, min)
- 病灶分布統計
- 嚴重程度評估

### 📁 輸出文件格式

特徵提取會生成以下文件：

```
deep_features/
├── [patient_id]/
│   ├── [patient_id]_features.pkl          # 完整的numpy數組特徵
│   ├── [patient_id]_features.json         # JSON格式的可讀特徵
│   └── [patient_id]_feature_report.md     # 病例特徵報告
├── logs/
│   └── feature_extraction_20250903_143022.log
├── all_features_summary.pkl               # 所有特徵的彙總
├── feature_extraction_report.json         # 提取統計報告
└── feature_summary_report.md              # 數據集特徵摘要
```

### 🤖 供LLM使用的特徵格式

#### 結構化特徵描述

每個病例會生成如下的結構化描述供LLM使用：

```
患者ID: A0001
CT掃描切片數: 45
檢測結果: 發現病灶
病灶總數: 8
含病灶切片數: 6
病灶分布比例: 13.3%
病灶總體積(近似): 2845
平均檢測置信度: 0.732
最高檢測置信度: 0.891
嚴重程度評估: 中度
病灶分布詳情:
  上段(肺尖): 2 個切片有病灶
  中段: 3 個切片有病灶
  下段(肺底): 1 個切片有病灶
  最大病灶位於第 23 切片，面積: 1205

深層特徵分析:
  可用特徵類型: 高級語義特徵, 多尺度特徵, 區域特徵
```

### 💻 程式化使用特徵

#### 加載特徵

```python
from feature_loader import FeatureLoader

# 加載特徵
loader = FeatureLoader("./deep_features")

# 獲取病例特徵
patient_features = loader.get_patient_features("A0001")

# 獲取檢測摘要
detection_summary = loader.get_detection_summary("A0001")

# 生成LLM提示特徵
llm_prompt = loader.generate_llm_prompt_features("A0001")
print(llm_prompt)
```

#### 生成報告

```python
from feature_loader import FeatureVisualizer

visualizer = FeatureVisualizer(loader)

# 生成單個病例報告
report = visualizer.create_patient_report("A0001", "A0001_report.md")

# 生成數據集摘要
summary = visualizer.create_dataset_summary("dataset_summary.md")
```

### 🔗 與LLM集成

#### 1. 特徵向量方式
```python
# 獲取數值特徵向量用於embedding
patient_features = loader.get_patient_features("A0001")
backbone_features = patient_features['global_features']['backbone_layer4']['mean']
# backbone_features: numpy array of shape [2048]
```

#### 2. 文本描述方式
```python
# 獲取結構化文本描述
text_features = loader.generate_llm_prompt_features("A0001")
# 可直接作為LLM的輸入提示
```

#### 3. 混合方式
```python
# 結合數值特徵和文本描述
detection_summary = loader.get_detection_summary("A0001")
text_prompt = loader.generate_llm_prompt_features("A0001")

# 構建完整的LLM輸入
llm_input = f"""
基於以下CT掃描分析結果生成醫學報告：

{text_prompt}

請生成專業的放射科報告，包括：
1. 影像檢查所見
2. 診斷印象
3. 建議
"""
```

### ⚙️ 命令行選項

#### deep_feature_extractor.py
```bash
python deep_feature_extractor.py \
    --model_path "models/best_model.pth" \
    --data_dir "datasets/splited_dataset" \
    --save_dir "./deep_features" \
    --split "val" \
    --confidence_threshold 0.5 \
    --device "cuda"
```

#### test/test_detection.py（包含特徵提取）
```bash
python test\test_detection.py \
    --extract_features \
    --split val \
    --model_path "models/best_model.pth" \
    --data_dir "datasets/splited_dataset"
```

#### feature_loader.py（分析工具）
```bash
# 生成數據集摘要
python feature_loader.py --features_dir "./deep_features"

# 生成特定病例報告
python feature_loader.py --features_dir "./deep_features" --patient_id "A0001"
```

### ⚠️ 注意事項

1. **記憶體使用**: 特徵提取需要較多GPU記憶體，建議batch_size不要太大
2. **儲存空間**: 每個病例的特徵文件約1-5MB，請確保有足夠的磁碟空間
3. **處理時間**: 特徵提取速度取決於病例數量和切片數，通常每個病例需要10-30秒
4. **模型依賴**: 確保使用的模型文件與訓練時的架構一致

### 🔧 特徵提取故障排除

#### 常見問題

1. **模型加載失敗**
   - 檢查模型路徑是否正確
   - 確認模型文件完整性
   - 檢查CUDA/CPU兼容性

2. **記憶體不足**
   - 減少batch_size
   - 使用CPU模式
   - 分批處理病例

3. **特徵文件損壞**
   - 重新提取特徵
   - 檢查磁碟空間
   - 確認權限設置

4. **JSON序列化錯誤**
   - 通常不影響pkl文件
   - 可能是numpy類型問題
   - 優先使用pkl格式

### 🚀 擴展功能

#### 自定義特徵提取
可以修改`DeepFeatureExtractor`類來提取更多類型的特徵：

```python
class CustomFeatureExtractor(DeepFeatureExtractor):
    def extract_custom_features(self, image):
        # 添加自定義特徵提取邏輯
        pass
```

#### 特徵後處理
```python
def normalize_features(features):
    # 添加特徵歸一化
    pass

def reduce_dimensionality(features):
    # 添加降維處理
    pass
```

## 📋 更新歷史

### 最新更新 (2025-01)
- ✅ **統一評估指標系統**: 完成 `train_detection.py`、`train_detection_simple.py`、`test_detection.py` 的評估指標同步
- ✅ **全面評估指標**: 三個腳本都支援22項專業評估指標
  - 核心檢測指標（IoU、mAP@0.5、mAP@[0.5:0.95]等）
  - 定位與錯誤分析指標（GIoU、DIoU、CIoU等）
  - 臨床相關指標（ROC、AUC、FROC等）
  - 效率與可用性指標（推理時間、FPS、記憶體使用）
- ✅ **模型測試腳本增強**: `test_detection.py` 支援完整的評估報告和可視化
- ✅ **PyTorch 2.6+ 兼容性**: 修正模型載入警告
- ✅ **隨機種子支援**: 確保結果可重現
- ✅ **自動可視化**: 預測結果和統計圖表自動生成
- ✅ **向後兼容性**: 保持原有簡化評估模式
- ✅ **豐富可視化**: ROC/FROC曲線、雷達圖、統計報告
- ✅ **測試框架**: 提供評估指標測試腳本
- ✅ **文檔整合**: 整合所有訓練配置和評估指標到單一文檔
