# YOLOv7 Chest CT Detection Enhancement 🚀

## 概述

本次更新針對 YOLOv7 胸腔 CT 病灶檢測進行全面優化，解決 early training stage mAP@0.5 長期停留在 0.01 以下的問題。

---

## 🎯 主要改進

### 1. 標註驗證與可視化工具 ✅

**文件**: `validate_annotations.py`

#### 功能
- 隨機抽取 100 張 train/val 影像
- 將 YOLO 格式標註繪製在 CT slice 上
- 檢測標註錯誤：
  - 超出 [0, 1] 範圍
  - 過小的 bbox (< 3px)
  - 負座標
  - 格式錯誤
- 輸出 Grid 可視化圖（每圖最多 25 張）
- 生成詳細報告 (Markdown + JSON)

#### 使用方法
```bash
# 驗證標註
python validate_annotations.py \
    --data_root ../../datasets/splited_dataset \
    --num_samples 100 \
    --output_dir ./yolov7_logs/annotation_validation

# 輸出：
# - yolov7_logs/annotation_validation/validation_report.md
# - yolov7_logs/annotation_validation/validation_report.json
# - yolov7_logs/annotation_validation/annotation_grid_*.png
```

#### 報告內容
- 總體統計（錯誤率、box 數量）
- 錯誤類型分布
- Box 尺寸統計
- 詳細錯誤列表
- 修復建議

---

### 2. 增強資料增強 ✅

**文件**: `yolov7_augmentations.py`

#### 新增 Augmentations

##### Mosaic (4-image mosaic)
- 將 4 張圖像拼接成一張
- 隨機中心點分割
- 自動調整標註座標
- 有助於學習不同尺度和位置的物體

##### MixUp
- 混合兩張圖像和標籤
- Beta 分佈控制混合比例
- 提升泛化能力

##### Copy-Paste (小物體增強)
- 複製小病灶 (area < 0.05) 到其他位置
- 專門針對小病灶設計
- 避免過度複製造成偽影

##### 醫學影像專屬增強
- Random Rotation (±15°)
- ShiftScaleRotate
- RandomBrightnessContrast
- GaussNoise & GaussianBlur
- CLAHE (Contrast Limited AHE)
- Sharpen

#### 使用方法
```python
from yolov7_augmentations import YOLOv7Augmenter

augmenter = YOLOv7Augmenter(
    img_size=640,
    mosaic_prob=0.5,
    mixup_prob=0.3,
    copy_paste_prob=0.3
)

# 在 dataset 中自動應用
aug_img, aug_labels = augmenter(img, labels, extra_samples)
```

---

### 3. 正樣本過採樣 (Positive Oversampling) ✅

**文件**: `yolov7_dataset.py`

#### 功能
- 自動識別正樣本（有病灶）和負樣本（無病灶）
- 按指定比例過採樣正樣本
- 避免負樣本主導訓練
- 內建 Dataset Caching 提升讀取速度

#### 配置
```python
# 創建 DataLoader 時
train_loader = create_yolov7_dataloader(
    dataset=train_dataset,
    batch_size=8,
    positive_oversample=True,  # 啟用正樣本過採樣
    positive_ratio=0.7,        # 70% 正樣本
    cache_images=True,         # 快取圖像
    augment=True,
    mosaic_prob=0.5,
    mixup_prob=0.3,
    ...
)
```

#### 效果
- **Before**: 正負樣本比例 1:5 → 負樣本主導
- **After**: 正負樣本比例 7:3 → 平衡學習

---

### 4. 批次大小與 Worker 優化 ✅

**變更**:
- `batch_size`: 4 → **8**
- `accumulation_steps`: 1 → **4** (有效 batch size = 32)
- `num_workers`: 0 → **4** (Windows) / **8** (Linux)
- 新增 `persistent_workers=True` (保持 workers 存活)
- 新增 `prefetch_factor=2` (預先載入資料)

#### Gradient Accumulation
```python
# 在訓練循環中
for i, (images, targets, _) in enumerate(train_loader):
    loss = ...
    loss = loss / accumulation_steps  # 除以累積步數
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### GroupNorm / SyncBN 選項
```python
# 配置文件中
use_group_norm: bool = True   # 小 batch 更穩定
use_sync_bn: bool = True      # 多 GPU 同步
```

---

### 5. 損失函數增強 ✅

**文件**: `yolov7_utils.py`

#### Focal Loss
替代 BCE Loss，解決類別不平衡問題：

```
FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
```

- **α** = 0.25 (平衡因子)
- **γ** = 2.0 (難易樣本權重)

#### 增強的分類損失
```python
ComputeLoss(
    model,
    cls_loss_gain=1.8,      # 提高分類損失權重 (針對小病灶)
    use_focal_loss=True,    # 使用 Focal Loss
    focal_alpha=0.25,
    focal_gamma=2.0
)
```

#### 配置
| 參數 | Baseline | Enhanced |
|------|----------|----------|
| `cls_loss_gain` | 1.0 | 1.8 |
| `use_focal_loss` | False | True |
| `focal_alpha` | - | 0.25 |
| `focal_gamma` | - | 2.0 |

---

### 6. Per-Epoch 評估可視化 ✅

**文件**: `yolov7_eval_visualizer.py`

#### 功能
- 每個 epoch 結束後自動評估
- 使用極低 `conf_threshold=0.001` 進行預測
- 計算 TP / FP / FN
- 繪製預測框與標註框對比圖
- 顏色標記：
  - 🟢 **綠色** = True Positive (TP)
  - 🔴 **紅色** = False Positive (FP)
  - 🟡 **黃色** = False Negative (FN)
- 儲存到 `./yolov7_logs/vis_epoch_X/`
- 生成評估報告 (Markdown + JSON)

#### 使用方法
```python
from yolov7_eval_visualizer import YOLOv7EvalVisualizer

visualizer = YOLOv7EvalVisualizer(
    save_dir='./yolov7_logs',
    num_vis_samples=20,
    iou_threshold=0.5
)

# 在訓練循環中
stats = visualizer.visualize_epoch(
    model=model,
    val_loader=val_loader,
    epoch=epoch,
    conf_threshold=0.001,
    device='cuda'
)

# 查看統計
print(f"Epoch {epoch}: Precision={stats.precision:.4f}, Recall={stats.recall:.4f}")
```

#### 輸出結構
```
yolov7_logs/
├── vis_epoch_1/
│   ├── sample_000_A0001.png
│   ├── sample_001_A0002.png
│   ├── ...
│   ├── evaluation_report.md
│   └── stats.json
├── vis_epoch_2/
│   └── ...
└── training_curves.png
```

---

## 📦 完整文件列表

### 新增文件
1. ✅ `validate_annotations.py` - 標註驗證工具
2. ✅ `yolov7_augmentations.py` - 增強資料增強
3. ✅ `yolov7_eval_visualizer.py` - 評估可視化工具
4. ✅ `enhanced_config.py` - 增強配置管理

### 修改文件
1. ✅ `yolov7_dataset.py` - 正樣本過採樣 + caching
2. ✅ `yolov7_utils.py` - Focal Loss + enhanced loss
3. ✅ `train_yolov7_medical.py` - 整合所有改進

---

## 🚀 快速開始

### 1. 驗證標註
```bash
# 檢查標註是否有問題
python validate_annotations.py \
    --data_root ../../datasets/splited_dataset \
    --num_samples 100
```

### 2. 生成配置文件
```bash
# 生成預設配置
python enhanced_config.py

# 輸出：
# - configs/train_baseline.yaml (最小增強)
# - configs/train_recommended.yaml (推薦配置)
# - configs/train_aggressive.yaml (最大增強)
```

### 3. 訓練模型
```bash
# 使用推薦配置
python train_yolov7_medical_enhanced.py --config configs/train_recommended.yaml

# 或使用命令行參數覆蓋
python train_yolov7_medical_enhanced.py \
    --data_dir ../../datasets/splited_dataset \
    --batch_size 8 \
    --num_workers 4 \
    --epochs 120 \
    --use_focal_loss \
    --cls_loss_gain 1.8 \
    --positive_oversample \
    --enable_augmentation
```

---

## 📊 預期改進效果

### Baseline vs Enhanced

| 指標 | Baseline | Enhanced | 改善 |
|------|----------|----------|------|
| mAP@0.5 (early) | < 0.01 | 0.05-0.15 | 5-15x ↑ |
| Recall | 0.02 | 0.30-0.50 | 15-25x ↑ |
| Training Stability | 不穩定 | 穩定收斂 | ✅ |
| Small Object mAP | < 0.01 | 0.10-0.20 | 10-20x ↑ |

### 關鍵改進點

1. **Focal Loss** → 解決類別不平衡
2. **Positive Oversampling** → 避免負樣本主導
3. **cls_loss_gain=1.8** → 強化小病灶學習
4. **Mosaic + MixUp** → 多尺度學習
5. **Low conf_threshold eval** → 及早發現預測問題
6. **Copy-Paste** → 小物體增強

---

## 🔧 配置建議

### 推薦配置 (Recommended)
```yaml
# configs/train_recommended.yaml
batch_size: 8
accumulation_steps: 4
num_workers: 4
learning_rate: 0.001

# Augmentation
enable_augmentation: true
mosaic_prob: 0.5
mixup_prob: 0.3
copy_paste_prob: 0.3

# Loss
use_focal_loss: true
cls_loss_gain: 1.8
focal_alpha: 0.25
focal_gamma: 2.0

# Sampling
positive_oversample: true
positive_ratio: 0.7

# Visualization
visualize_predictions: true
vis_conf_threshold: 0.001
num_vis_samples: 20
```

### Aggressive 配置（激進）
適合數據量大、需要極強泛化的場景：
```yaml
batch_size: 8
accumulation_steps: 4
mosaic_prob: 0.7
mixup_prob: 0.5
copy_paste_prob: 0.5
positive_ratio: 0.8
cls_loss_gain: 2.0
use_group_norm: true
```

### Baseline 配置（對照組）
用於對比實驗：
```yaml
batch_size: 4
accumulation_steps: 1
num_workers: 0
enable_augmentation: false
positive_oversample: false
use_focal_loss: false
cls_loss_gain: 1.0
```

---

## 📈 訓練監控

### 1. TensorBoard / WandB
```python
# 在訓練腳本中已整合
# 自動記錄：
# - Loss curves
# - mAP/Precision/Recall
# - Learning rate
```

### 2. 可視化輸出
```bash
# 每個 epoch 後查看
yolov7_logs/
├── vis_epoch_1/
│   ├── evaluation_report.md  # 查看這個！
│   └── sample_*.png
```

### 3. 關鍵指標
- **mAP@0.5**: 目標 > 0.15 (at epoch 10)
- **Recall**: 目標 > 0.30 (at epoch 10)
- **Precision**: 目標 > 0.20 (at epoch 10)
- **Loss 收斂**: 應在前 20 epochs 顯著下降

---

## 🐛 調試建議

### 如果 mAP 仍然很低：

#### 1. 檢查標註
```bash
python validate_annotations.py --data_root <your_data>
# 查看報告，修復錯誤標註
```

#### 2. 檢查預測可視化
```bash
# 查看 yolov7_logs/vis_epoch_1/evaluation_report.md
# 觀察：
# - 是否有預測框？(FP > 0)
# - 是否對準病灶？(TP vs FN)
# - 是否過度預測？(FP >> TP)
```

#### 3. 調整 conf_threshold
```python
# 在評估時使用更低的閾值
visualizer.visualize_epoch(..., conf_threshold=0.0001)
```

#### 4. 調整 Loss 權重
```yaml
# 如果框偏移嚴重
box: 0.1  # 增加 (default 0.05)

# 如果小病灶漏檢
cls_loss_gain: 2.5  # 增加 (default 1.8)

# 如果過度預測
obj: 1.5  # 增加 (default 1.0)
```

#### 5. 調整 Anchor
```yaml
# 在 yolov7_medical.yaml 中
# 根據病灶實際大小調整 anchor
anchors:
  - [5, 5, 10, 10, 15, 15]   # Small anchors for small lesions
```

---

## 📝 依賴項

確保已安裝：
```bash
pip install torch torchvision
pip install opencv-python-headless
pip install albumentations
pip install matplotlib
pip install pyyaml
pip install tqdm
pip install numpy
```

---

## 🎓 參考資料

### 論文
- YOLOv7: https://arxiv.org/abs/2207.02696
- Focal Loss: https://arxiv.org/abs/1708.02002
- Mosaic Augmentation: YOLOv4
- MixUp: https://arxiv.org/abs/1710.09412

### 醫學影像檢測
- CT HU Windowing: Radiology guidelines
- CLAHE: Adaptive histogram equalization

---

## 📞 支援

如有問題，請檢查：
1. `yolov7_logs/logs/` - 訓練日誌
2. `yolov7_logs/vis_epoch_X/` - 可視化輸出
3. `validation_report.md` - 標註驗證報告

---

**祝訓練成功！🎉**
