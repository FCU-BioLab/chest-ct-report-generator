# YOLOv7 Chest CT Enhancement - 實作完成總結

## ✅ 已完成的改進項目

### 1. 標註驗證與可視化 ✅
- **文件**: `validate_annotations.py`
- **功能**:
  - 隨機抽取 100 張影像並繪製標註
  - 檢測 bbox 錯誤（超出範圍、過小、負座標）
  - 生成 grid 可視化 (25 張/grid)
  - 輸出詳細報告 (Markdown + JSON)
- **使用**: `python validate_annotations.py --data_root ../../datasets/splited_dataset --num_samples 100`

### 2. 增強資料增強 ✅
- **文件**: `yolov7_augmentations.py`
- **新增 Augmentations**:
  - ✅ Mosaic (4-image mosaic)
  - ✅ MixUp
  - ✅ Copy-Paste (小物體增強)
  - ✅ Random Rotation (±15°)
  - ✅ ShiftScaleRotate
  - ✅ RandomBrightnessContrast
  - ✅ GaussNoise & GaussianBlur
  - ✅ CLAHE
  - ✅ Sharpen

### 3. 正樣本過採樣 ✅
- **文件**: `yolov7_dataset.py` (已修改)
- **新增功能**:
  - ✅ `PositiveOversampleSampler` class
  - ✅ 自動識別正負樣本
  - ✅ 可調整正樣本比例 (default: 0.7)
  - ✅ Dataset caching
  - ✅ 整合 augmentation pipeline
- **配置**: `positive_oversample=True, positive_ratio=0.7`

### 4. Batch Size & Workers 優化 ✅
- **文件**: `yolov7_dataset.py`, `enhanced_config.py`
- **變更**:
  - ✅ `batch_size`: 4 → 8
  - ✅ `accumulation_steps`: 1 → 4 (有效 batch=32)
  - ✅ `num_workers`: 0 → 4 (Windows) / 8 (Linux)
  - ✅ `persistent_workers=True`
  - ✅ `prefetch_factor=2`
- **選項**: GroupNorm / SyncBN (配置中)

### 5. 損失函數增強 ✅
- **文件**: `yolov7_utils.py` (已修改)
- **新增**:
  - ✅ `FocalLoss` class
  - ✅ `ComputeLoss` 增強版
  - ✅ 可調整 `cls_loss_gain` (default: 1.8)
  - ✅ Focal Loss 選項 (default: enabled)
- **配置**:
  ```python
  ComputeLoss(
      model,
      cls_loss_gain=1.8,
      use_focal_loss=True,
      focal_alpha=0.25,
      focal_gamma=2.0
  )
  ```

### 6. Per-Epoch 評估可視化 ✅
- **文件**: `yolov7_eval_visualizer.py`
- **功能**:
  - ✅ 低 conf_threshold 預測 (0.001)
  - ✅ TP/FP/FN 計算與標記
  - ✅ 繪製對比圖（綠色=TP, 紅色=FP, 黃色=FN）
  - ✅ 保存到 `./yolov7_logs/vis_epoch_X/`
  - ✅ 生成評估報告 (Markdown + JSON)
  - ✅ 訓練曲線繪製
- **使用**:
  ```python
  visualizer = YOLOv7EvalVisualizer(save_dir='./yolov7_logs')
  stats = visualizer.visualize_epoch(model, val_loader, epoch=1, conf_threshold=0.001)
  ```

---

## 📦 新增文件列表

### 核心工具
1. ✅ **`validate_annotations.py`** (718 lines)
   - 標註驗證與可視化工具
   - 錯誤檢測與報告生成

2. ✅ **`yolov7_augmentations.py`** (610 lines)
   - Mosaic, MixUp, Copy-Paste 實作
   - 醫學影像專屬增強
   - `YOLOv7Augmenter` 統一介面

3. ✅ **`yolov7_eval_visualizer.py`** (650 lines)
   - Per-epoch 評估可視化
   - TP/FP/FN 計算與繪製
   - 統計報告生成

4. ✅ **`enhanced_config.py`** (220 lines)
   - 配置管理系統
   - 預設配置生成 (baseline, recommended, aggressive)
   - YAML 導入/導出

5. ✅ **`quickstart.py`** (280 lines)
   - 快速啟動腳本
   - 整合驗證、配置、訓練流程
   - 命令行界面

6. ✅ **`ENHANCEMENT_README.md`** (完整文檔)
   - 詳細使用說明
   - 配置建議
   - 調試指南

### 修改文件
1. ✅ **`yolov7_dataset.py`**
   - 新增 `PositiveOversampleSampler`
   - 整合 augmentation pipeline
   - Dataset caching
   - 改進 `__getitem__` 支援 augmentation

2. ✅ **`yolov7_utils.py`**
   - 新增 `FocalLoss` class
   - 增強 `ComputeLoss` 支援 focal loss 和可調整 cls_loss_gain
   - 改進損失權重計算

---

## 🚀 使用指南

### 快速開始（3 步驟）

#### 1. 驗證標註
```bash
python quickstart.py validate --num_samples 100
```

**輸出**:
- `yolov7_logs/annotation_validation/validation_report.md`
- `yolov7_logs/annotation_validation/annotation_grid_*.png`

#### 2. 生成配置
```bash
python enhanced_config.py
```

**輸出**:
- `configs/train_baseline.yaml`
- `configs/train_recommended.yaml` ⭐ (推薦)
- `configs/train_aggressive.yaml`

#### 3. 訓練模型
```bash
# 使用推薦配置
python quickstart.py train --config recommended --epochs 120

# 或使用完整 pipeline
python quickstart.py full --epochs 120
```

---

## 📊 配置對比

### Baseline vs Recommended vs Aggressive

| 參數 | Baseline | Recommended ⭐ | Aggressive |
|------|----------|---------------|-----------|
| **Batch Size** | 4 | 8 | 8 |
| **Accumulation** | 1 | 4 | 4 |
| **Num Workers** | 0 | 4 | 8 |
| **Mosaic** | ❌ | ✅ (0.5) | ✅ (0.7) |
| **MixUp** | ❌ | ✅ (0.3) | ✅ (0.5) |
| **Copy-Paste** | ❌ | ✅ (0.3) | ✅ (0.5) |
| **Positive Oversample** | ❌ | ✅ (0.7) | ✅ (0.8) |
| **Focal Loss** | ❌ | ✅ | ✅ |
| **cls_loss_gain** | 1.0 | 1.8 | 2.0 |
| **GroupNorm** | ❌ | ❌ | ✅ |
| **Visualization** | ❌ | ✅ | ✅ |

---

## 🎯 預期效果

### 訓練收斂改善

| 指標 | Before (Baseline) | After (Enhanced) | 改善幅度 |
|------|------------------|------------------|----------|
| **mAP@0.5** (epoch 10) | < 0.01 | 0.05 - 0.15 | **5-15x** ↑ |
| **Recall** (epoch 10) | < 0.05 | 0.30 - 0.50 | **6-10x** ↑ |
| **Precision** (epoch 10) | < 0.10 | 0.20 - 0.40 | **2-4x** ↑ |
| **Loss 穩定性** | 震盪 | 穩定下降 | ✅ |
| **小病灶檢測** | 幾乎不檢測 | 明顯改善 | ✅ |

### 關鍵改進來源

1. **Focal Loss** → 解決類別不平衡，提升小病灶權重
2. **Positive Oversampling** → 正負樣本比例平衡 (7:3)
3. **cls_loss_gain=1.8** → 強化分類損失，避免負樣本主導
4. **Mosaic + MixUp** → 多尺度學習，提升泛化
5. **低 conf_threshold 評估** → 及早發現模型是否有預測能力
6. **Copy-Paste** → 專門針對小物體增強

---

## 📈 監控與調試

### 1. 檢查標註品質
```bash
python validate_annotations.py --data_root <your_data>
# 查看 validation_report.md
# 修復錯誤標註後重新訓練
```

### 2. 監控訓練過程
```bash
# 查看訓練日誌
cat yolov7_logs/logs/yolov7_training_*.log

# 查看可視化
ls yolov7_logs/vis_epoch_*/
```

### 3. 分析評估結果
```bash
# 每個 epoch 後查看
cat yolov7_logs/vis_epoch_1/evaluation_report.md

# 關鍵指標：
# - TP/FP/FN 數量
# - Precision/Recall/F1
# - 可視化圖像中的預測品質
```

### 4. 調試建議

#### 如果 mAP 仍然很低 (< 0.05 at epoch 10):

**A. 檢查是否有預測**
```bash
# 查看 vis_epoch_1/evaluation_report.md
# 如果 Total Predictions = 0 → 模型沒有輸出
# 解決方案：
#   1. 降低 conf_threshold (已設為 0.001)
#   2. 檢查 anchor 大小是否合適
#   3. 增加 obj_loss 權重
```

**B. 檢查預測是否對準**
```bash
# 查看 vis_epoch_1/sample_*.png
# 如果框位置嚴重偏移 (high FP, low TP)
# 解決方案：
#   1. 增加 box_loss 權重 (0.05 → 0.1)
#   2. 檢查標註是否正確
#   3. 降低學習率
```

**C. 檢查過度預測**
```bash
# 如果 FP >> TP (例如 FP=1000, TP=10)
# 解決方案：
#   1. 增加 obj_loss 權重 (1.0 → 1.5)
#   2. 降低 cls_loss_gain (1.8 → 1.2)
#   3. 使用更嚴格的 NMS (iou=0.45 → 0.65)
```

**D. 小病灶漏檢**
```bash
# 如果 FN >> TP (小病灶未檢測到)
# 解決方案：
#   1. 增加 cls_loss_gain (1.8 → 2.5)
#   2. 啟用 Copy-Paste augmentation
#   3. 調整 anchor 大小
#   4. 增加正樣本比例 (0.7 → 0.8)
```

---

## 🔧 進階配置

### 自定義損失權重
```python
# 在 train_yolov7_medical.py 中
loss_fn = ComputeLoss(
    model,
    cls_loss_gain=2.0,      # 提高以強化分類
    use_focal_loss=True,
    focal_alpha=0.25,
    focal_gamma=2.0
)

# 或修改 model YAML
# box: 0.1  (box regression loss weight)
# obj: 1.5  (objectness loss weight)
# cls: 0.5  (classification loss weight, 會乘以 cls_loss_gain)
```

### 調整 Anchor
```yaml
# models/yolov7_medical.yaml
anchors:
  # 根據病灶實際大小調整
  - [5, 5, 10, 10, 15, 15]    # Small (適合小病灶)
  - [15, 15, 30, 30, 45, 45]  # Medium
  - [45, 45, 90, 90, 135, 135] # Large
```

### 多階段訓練
```bash
# Stage 1: Baseline (warm-up, 20 epochs)
python train_yolov7_medical.py \
    --config configs/train_baseline.yaml \
    --epochs 20

# Stage 2: Enhanced (fine-tune, 100 epochs)
python train_yolov7_medical.py \
    --config configs/train_recommended.yaml \
    --epochs 100 \
    --pretrained weights/baseline_last.pt
```

---

## 📚 文件結構

```
detection/yolo_detection/
├── models/
│   ├── yolov7_medical.yaml          # Model architecture
│   └── custom_layers.py             # Medical modules (CBAM, etc.)
│
├── configs/                          # ✨ NEW
│   ├── train_baseline.yaml          # Minimal augmentation
│   ├── train_recommended.yaml       # Recommended settings
│   └── train_aggressive.yaml        # Maximum augmentation
│
├── yolov7_logs/                      # ✨ NEW
│   ├── annotation_validation/       # Annotation check results
│   ├── vis_epoch_1/                 # Per-epoch visualizations
│   ├── vis_epoch_2/
│   ├── ...
│   └── runs/                        # Training runs
│
├── validate_annotations.py           # ✨ NEW - Annotation validator
├── yolov7_augmentations.py          # ✨ NEW - Augmentations
├── yolov7_eval_visualizer.py        # ✨ NEW - Evaluation visualizer
├── enhanced_config.py               # ✨ NEW - Config manager
├── quickstart.py                    # ✨ NEW - Quick start script
├── ENHANCEMENT_README.md            # ✨ NEW - Full documentation
│
├── yolov7_model.py                  # Model loader
├── yolov7_dataset.py                # ✅ ENHANCED - Dataset with oversampling
├── yolov7_utils.py                  # ✅ ENHANCED - Loss with focal loss
└── train_yolov7_medical.py          # Training script
```

---

## 🎓 重要概念說明

### Focal Loss
解決類別不平衡問題，降低易分類樣本的權重：
```
FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
```
- `α = 0.25`: 平衡正負樣本
- `γ = 2.0`: 降低易分類樣本權重

### Positive Oversampling
胸腔 CT 中，有病灶的影像 << 無病灶的影像。
過採樣正樣本可平衡比例，避免模型傾向預測「無病灶」。

### Mosaic Augmentation
將 4 張圖像拼接成一張，有助於：
- 學習不同尺度的物體
- 學習物體之間的關係
- 增加 batch diversity

### Copy-Paste for Small Objects
複製小病灶到其他位置，專門針對小物體檢測改善。

---

## ✅ 檢查清單

在開始訓練前，確認：

- [ ] 標註已驗證 (`python quickstart.py validate`)
- [ ] 配置文件已生成 (`python enhanced_config.py`)
- [ ] 數據集路徑正確 (`--data_root`)
- [ ] GPU 可用 (`nvidia-smi`)
- [ ] 依賴項已安裝 (`pip install -r requirements.txt`)

---

## 📞 問題排查

### Q1: 訓練時顯示 "No module named 'albumentations'"
```bash
pip install albumentations
```

### Q2: num_workers > 0 時出錯 (Windows)
```bash
# 使用 num_workers=0 或設置
set PYTHONHASHSEED=0
```

### Q3: 顯存不足 (OOM)
```python
# 降低 batch_size 並增加 accumulation_steps
batch_size: 4
accumulation_steps: 8  # 有效 batch = 32
```

### Q4: mAP 一直是 0.000
```bash
# 檢查評估可視化
cat yolov7_logs/vis_epoch_1/evaluation_report.md

# 如果 Total Predictions = 0:
#   → 模型沒有輸出，檢查 anchor 和 conf_threshold
# 如果 FP >> TP:
#   → 預測不準，檢查標註品質
# 如果 FN >> TP:
#   → 漏檢嚴重，增加 cls_loss_gain
```

---

## 🎉 完成！

所有增強功能已實作完成，包含：

✅ 標註驗證與可視化
✅ 增強資料增強 (Mosaic, MixUp, Copy-Paste)
✅ 正樣本過採樣
✅ Batch size & workers 優化
✅ Focal Loss 與增強損失函數
✅ Per-epoch 評估可視化
✅ 完整文檔與快速啟動腳本

**立即開始訓練：**
```bash
python quickstart.py full --epochs 120
```

**祝訓練成功！🚀**
