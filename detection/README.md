# Faster R-CNN 目標檢測訓練系統

基於 Faster R-CNN 的胸部 CT 腫瘤目標檢測模型，專門針對二分類檢測任務（背景 vs 病灶）。

## 🌟 主要特色

- **🎯 二分類檢測**: 精確區分背景與病灶並預測邊界框
- **🏗️ Faster R-CNN架構**: 採用 ResNet50-FPN 作為骨幹網路
- **🔄 多種訓練模式**: 支援傳統訓練和 K-Fold 交叉驗證
- **🚀 一鍵執行**: 整合所有功能於單一腳本
- **📊 智能分析**: 自動生成訓練報告和視覺化
- **⚙️ 靈活配置**: 豐富的參數自訂選項

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

### 🔥 一鍵訓練

**1. 傳統訓練模式（推薦用於最終模型）**
```bash
python detection\train_detection.py --mode traditional
```

**2. K-Fold 交叉驗證（推薦用於模型評估）**
```bash
python detection\train_detection.py --mode kfold
```

**3. 快速測試（2個epoch）**
```bash
python detection\train_detection.py --mode traditional --num_epochs 2 --batch_size 2
```

## ⚙️ 進階配置

### 自訂參數訓練

```bash
# 自訂傳統訓練
python detection\train_detection.py --mode custom --val_ratio 0.15 --num_epochs 100 --learning_rate 5e-5

# 自訂K-Fold訓練
python detection\train_detection.py --mode custom --use_kfold --k_folds 10 --num_epochs 30
```

### 📋 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--mode` | `custom` | 執行模式：traditional/kfold/custom |
| `--num_epochs` | `50` | 訓練輪數 |
| `--batch_size` | `8` | 批次大小 |
| `--learning_rate` | `1e-4` | 學習率 |
| `--val_ratio` | `0.2` | 驗證集比例 |
| `--k_folds` | `5` | K-Fold 數量 |
| `--num_classes` | `2` | 類別數量（背景+病灶） |
| `--image_size` | `512` | 影像尺寸 |
| `--data_root` | `../datasets/splited_dataset` | 資料根目錄 |
| `--output_dir` | `./Faster_RCNN_Detection` | 輸出目錄 |

## 📊 訓練輸出

### 傳統訓練模式
```
Faster_RCNN_Detection/
├── best_detection_model.pth       # 最佳模型
├── training_args.json             # 訓練參數
├── logs/                          # 訓練日誌
│   ├── training.log              # 詳細日誌
│   └── events.out.tfevents.*     # TensorBoard
└── predictions_epoch_*.png        # 預測視覺化
```

### K-Fold 訓練模式
```
Faster_RCNN_Detection/
├── kfold_final_results.json       # 總體結果
├── fold_1/                        # Fold 1 結果
│   ├── best_detection_model.pth   # 該fold最佳模型
│   ├── fold_results.json          # 該fold詳細結果
│   └── logs/                      # 該fold日誌
├── fold_2/                        # Fold 2 結果
└── ...
```

## 📈 結果分析

### K-Fold 結果分析
```bash
python detection\analyze_kfold_results.py --results_dir Faster_RCNN_Detection
```

### TensorBoard 監控
```bash
# 傳統模式
tensorboard --logdir Faster_RCNN_Detection/logs

# K-Fold 模式
tensorboard --logdir Faster_RCNN_Detection
```

## 🔍 模型推理

```bash
python detection\inference_detection.py \
  --model_path Faster_RCNN_Detection/best_detection_model.pth \
  --input_dicom path/to/dicom.dcm \
  --confidence_threshold 0.5
```

## 📝 訓練日誌

### 傳統模式
- **詳細日誌**: `Faster_RCNN_Detection/logs/training.log`
- **TensorBoard**: `Faster_RCNN_Detection/logs/`

### K-Fold 模式
- **總體結果**: `Faster_RCNN_Detection/kfold_final_results.json`
- **各Fold日誌**: `Faster_RCNN_Detection/fold_X/logs/training.log`

## 🏗️ 模型架構

### Faster R-CNN 組件
- **骨幹網路**: ResNet50-FPN (預訓練於COCO)
- **區域提議網路(RPN)**: 生成候選邊界框
- **分類頭**: 二分類（背景 vs 病灶）
- **回歸頭**: 邊界框座標預測

### 主要優勢
1. **端到端訓練**: 同時優化檢測和分類
2. **成熟架構**: 經過大量驗證的目標檢測方法
3. **預訓練權重**: 基於COCO資料集的預訓練模型
4. **靈活輸入**: 支援灰階醫學影像

## 🛠️ 故障排除

### 常見問題

**Q: 記憶體不足錯誤**
```bash
# 減少批次大小
python detection\train_detection.py --mode traditional --batch_size 2

# 減少工作程序
# (程式已自動設為num_workers=0)
```

**Q: 沒有檢測到任何病灶**
- 檢查置信度閾值設定
- 確認標註檔案格式正確
- 增加訓練輪數

**Q: 訓練速度慢**
- 確認使用GPU訓練
- 檢查資料載入效率
- 考慮使用較小的影像尺寸

### 檢查系統狀態
```bash
# 測試模型創建
python detection\test_faster_rcnn.py

# 檢查資料載入
python detection\faster_rcnn_dataset.py
```

## 📚 技術細節

### 損失函數
- **分類損失**: CrossEntropyLoss
- **回歸損失**: SmoothL1Loss
- **RPN損失**: 結合分類和回歸損失

### 評估指標
- **Precision**: 檢測精度
- **Recall**: 檢測召回率  
- **F1-Score**: 綜合評估指標
- **IoU**: 邊界框重疊度

### 資料擴增
- **隨機縮放**: 多尺度訓練
- **正規化**: 基於ImageNet統計
- **灰階轉RGB**: 適配預訓練模型

## 🎯 最佳實踐

1. **使用K-Fold評估模型性能**，獲得可靠的評估結果
2. **傳統模式訓練最終模型**，確保最佳性能
3. **監控訓練日誌**，及時調整參數
4. **視覺化檢測結果**，驗證模型有效性
5. **保存訓練參數**，確保結果可重現

## 🚨 注意事項

- 確保 DICOM 檔案與 XML 標註的 SOPInstanceUID 匹配
- 訓練前建議先運行測試腳本驗證環境
- K-Fold 模式需要更長時間但提供更可靠的評估
- 模型權重檔案較大，注意儲存空間
