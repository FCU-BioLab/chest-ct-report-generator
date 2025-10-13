# YOLOv7 驗證指標計算說明

## 📊 新增的評估指標

### 修改日期
2025-10-12

### 修改內容
在 `train_yolov7_preprocessed.py` 中增強了驗證階段的評估指標計算和記錄功能。

---

## ✨ 新增指標

訓練過程現在會計算並記錄以下評估指標：

### 1. **Precision (精確率)**
- **定義**: TP / (TP + FP)
- **意義**: 所有預測為正樣本中，真正是正樣本的比例
- **範圍**: 0.0 - 1.0（越高越好）

### 2. **Recall (召回率)**
- **定義**: TP / (TP + FN)
- **意義**: 所有真實正樣本中，被正確預測出來的比例
- **範圍**: 0.0 - 1.0（越高越好）

### 3. **F1-Score (F1 分數)**
- **定義**: 2 × (Precision × Recall) / (Precision + Recall)
- **意義**: Precision 和 Recall 的調和平均數
- **範圍**: 0.0 - 1.0（越高越好）

### 4. **Accuracy (準確率)**
- **定義**: 目前使用 mAP@0.5 作為近似值
- **意義**: 整體檢測準確度
- **範圍**: 0.0 - 1.0（越高越好）

### 5. **mAP@0.5 (平均精確率 @ IoU=0.5)**
- **定義**: IoU 閾值為 0.5 時的平均精確率
- **意義**: COCO 指標，常用於目標檢測評估
- **範圍**: 0.0 - 1.0（越高越好）

### 6. **mAP@0.5:0.95 (平均精確率 @ IoU=0.5:0.95)**
- **定義**: IoU 閾值從 0.5 到 0.95（步長 0.05）的平均 mAP
- **意義**: 更嚴格的評估標準，COCO 主要指標
- **範圍**: 0.0 - 1.0（越高越好）

---

## 📝 輸出格式

### 1. **終端輸出**
每個 epoch 完成後會顯示：
```
Epoch 1/100 - Train Loss: 0.7221, Val Loss: 0.9828, P: 0.8500, R: 0.7800, F1: 0.8140, mAP@0.5: 0.8300, LR: 0.001000, Time: 63.45s, Total: 1.06h
```

### 2. **日誌文件**
保存在 `yolov7_logs/yolov7_training_YYYYMMDD_HHMMSS.log`
包含完整的訓練過程和所有指標。

### 3. **CSV 結果文件**
保存在 `yolov7_models/run_YYYYMMDD_HHMMSS/results.csv`
包含每個 epoch 的所有指標：
```csv
epoch,train_loss,val_loss,precision,recall,f1,accuracy,mAP@0.5,mAP@0.5:0.95,learning_rate,time_elapsed
1,0.7221,0.9828,0.8500,0.7800,0.8140,0.8300,0.8300,0.6520,0.001000,63.45
```

### 4. **JSON 結果文件**
保存在 `yolov7_models/run_YYYYMMDD_HHMMSS/yolov7_training_results_YYYYMMDD_HHMMSS.json`
包含完整的訓練配置、歷史記錄和最終指標：
```json
{
  "timestamp": "20251012_173454",
  "best_epoch": 42,
  "best_val_loss": 0.4521,
  "best_mAP": 0.8932,
  "final_metrics": {
    "precision": 0.8500,
    "recall": 0.7800,
    "f1": 0.8140,
    "accuracy": 0.8300,
    "mAP@0.5": 0.8300,
    "mAP@0.5:0.95": 0.6520
  },
  "history": {
    "epoch": [1, 2, 3, ...],
    "precision": [0.75, 0.78, 0.82, ...],
    "recall": [0.70, 0.73, 0.76, ...],
    ...
  }
}
```

---

## 📈 可視化圖表

訓練完成後會自動生成以下圖表（保存在 `yolov7_models/run_YYYYMMDD_HHMMSS/`）：

### 1. **loss_curve.png**
- 訓練損失 vs 驗證損失
- 標記最佳 epoch

### 2. **metrics_curve.png** ⭐ 新增
- Precision、Recall、F1-Score 隨 epoch 的變化
- 標記最佳 epoch

### 3. **mAP_curve.png** ⭐ 新增
- mAP@0.5 和 mAP@0.5:0.95 隨 epoch 的變化
- 標記最佳 epoch

### 4. **lr_curve.png**
- 學習率變化曲線

### 5. **results.png** ⭐ 增強
- 3×2 綜合結果圖，包含：
  - Loss 曲線
  - Precision & Recall
  - F1-Score & Accuracy
  - mAP@0.5 & mAP@0.5:0.95
  - Learning Rate
  - 訓練摘要文本（包含所有最終指標）

---

## 🎯 最佳模型保存

最佳模型（`best.pt`）現在包含完整的評估指標：

```python
checkpoint = torch.load('yolov7_models/run_20251012_173454/weights/best.pt')
print(checkpoint['metrics'])
# 輸出:
# {
#   'precision': 0.8500,
#   'recall': 0.7800,
#   'f1': 0.8140,
#   'mAP@0.5': 0.8300,
#   'mAP@0.5:0.95': 0.6520
# }
```

---

## 📊 訓練完成摘要

訓練結束時會顯示完整的評估報告：

```
================================================================================
✅ 訓練完成！
總時間: 4.38 小時 (15768.25 秒)
最佳驗證損失: 0.4521 (Epoch 42)
最佳 mAP@0.5: 0.8932 (Epoch 42)
最終指標 (Epoch 100):
  Precision:    0.8500
  Recall:       0.7800
  F1-Score:     0.8140
  Accuracy:     0.8300
  mAP@0.5:      0.8300
  mAP@0.5:0.95: 0.6520
模型保存至: yolov7_models/run_20251012_173454/weights
最佳模型: yolov7_models/run_20251012_173454/weights/best.pt
最後模型: yolov7_models/run_20251012_173454/weights/last.pt
================================================================================
```

---

## 🔧 技術細節

### 指標計算位置
- 核心計算在 `train_yolov7_medical.py` 的 `validate()` 函數
- 使用 `ap_per_class()` 函數計算 AP、Precision、Recall
- IoU 計算使用 `box_iou()` 函數

### NMS 後處理
- 置信度閾值: 0.001
- IoU 閾值: 0.6
- 使用 `torchvision.ops.nms`

### mAP 計算
- mAP@0.5: IoU 閾值 = 0.5
- mAP@0.5:0.95: IoU 閾值從 0.5 到 0.95，步長 0.05（共 10 個閾值）

---

## 📚 相關文件

- **主訓練腳本**: `train_yolov7_preprocessed.py`
- **驗證函數**: `train_yolov7_medical.py` (line 539-679)
- **指標計算**: `train_yolov7_medical.py` (line 680-760)
- **NMS 實現**: `train_yolov7_medical.py` (line 520-538)

---

## ⚙️ 使用方式

運行訓練時，所有指標會自動計算和記錄：

```bash
python train_yolov7_preprocessed.py \
    --data_dir ../../datasets/splited_dataset \
    --epochs 100 \
    --batch_size 16 \
    --max_negative_per_patient 20
```

訓練過程中可以實時查看指標，訓練完成後查看圖表和 JSON 文件獲取完整結果。

---

## 📌 注意事項

1. **初期指標可能為 0**：前幾個 epoch 模型還未收斂時，可能沒有有效檢測，導致指標為 0
2. **Accuracy 近似值**：目前使用 mAP@0.5 作為 Accuracy 的近似值
3. **指標延遲**：指標計算需要推理時間，會稍微增加驗證時間（約 10-20%）
4. **批次影響**：batch_size 不影響指標計算，但會影響訓練穩定性

---

## 🎓 指標解讀建議

### 醫學影像檢測的優先順序：
1. **Recall > Precision**：寧願誤報，不要漏檢（漏掉病灶比誤報更嚴重）
2. **mAP@0.5 > mAP@0.5:0.95**：醫學影像中，精確定位比完美框選更重要
3. **F1-Score**：綜合評估模型性能

### 典型指標範圍：
- **優秀**: Precision > 0.85, Recall > 0.80, mAP@0.5 > 0.85
- **良好**: Precision > 0.75, Recall > 0.70, mAP@0.5 > 0.75
- **需改進**: Precision < 0.70 或 Recall < 0.65

---

## 更新記錄

- **2025-10-12**: 初版發布，新增 Precision、Recall、F1、Accuracy、mAP 等指標的完整記錄和可視化
