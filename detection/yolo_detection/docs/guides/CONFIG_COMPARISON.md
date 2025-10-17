# YOLOv11 訓練配置對比表

快速選擇適合您需求的訓練配置。

## 📊 配置對比

| 項目 | 快速測試 | 標準訓練 | 高精度訓練 | 生產環境 |
|------|---------|---------|-----------|---------|
| **用途** | 參數測試 | 基準測試 | 小病灶檢測 | 最終部署 |
| **訓練時間** | 2-3 小時 | 8-12 小時 | 20-30 小時 | 30-50 小時 |
| **模型大小** | n (nano) | m (medium) | l (large) | x (xlarge) |
| **圖片尺寸** | 640 | 640 | 800 | 800-1024 |
| **Epochs** | 50 | 200 | 300 | 400+ |
| **Batch Size** | 32 | 16 | 8 | 4-8 |
| **學習率** | 0.001 | 0.001 | 0.0005 | 0.0003 |
| **驗證比例** | 0.3 | 0.2 | 0.2 | 0.15 |
| **數據增強** | 輕度 | 中度 | 強度 | 強度+ |
| **預期 mAP@0.5** | 0.65-0.75 | 0.75-0.85 | 0.78-0.88 | 0.80-0.90 |

---

## 🚀 命令範例

### 快速測試配置

```bash
python train_yolo_direct.py \
    --data_dir ../../datasets/splited_dataset/train \
    --epochs 50 \
    --batch_size 32 \
    --model_size n \
    --imgsz 640 \
    --val_ratio 0.3 \
    --lr 0.001 \
    --mosaic 0.5 \
    --mixup 0.0
```

**適用場景：**
- ✅ 測試新參數組合
- ✅ 快速驗證想法
- ✅ 學習和實驗

**不適合：**
- ❌ 生產環境
- ❌ 論文結果
- ❌ 最終部署

---

### 標準訓練配置 ⭐ 推薦

```bash
python train_yolo_direct.py \
    --data_dir ../../datasets/splited_dataset/train \
    --epochs 200 \
    --batch_size 16 \
    --model_size m \
    --imgsz 640 \
    --val_ratio 0.2 \
    --lr 0.001 \
    --optimizer AdamW \
    --warmup_epochs 5 \
    --mosaic 1.0 \
    --mixup 0.0 \
    --fliplr 0.5 \
    --scale 0.5
```

**適用場景：**
- ✅ 第一次訓練
- ✅ 建立性能基準
- ✅ 大多數檢測任務
- ✅ 平衡精度和速度

**推薦理由：**
- 參數經過優化
- 訓練時間合理
- 性能穩定可靠

---

### 高精度配置

```bash
python train_yolo_direct.py \
    --data_dir ../../datasets/splited_dataset/train \
    --epochs 300 \
    --batch_size 8 \
    --model_size l \
    --imgsz 800 \
    --val_ratio 0.2 \
    --lr 0.0005 \
    --optimizer AdamW \
    --warmup_epochs 5 \
    --weight_decay 0.0005 \
    --mosaic 1.0 \
    --mixup 0.15 \
    --copy_paste 0.1 \
    --fliplr 0.5 \
    --scale 0.7 \
    --translate 0.1
```

**適用場景：**
- ✅ 小病灶檢測
- ✅ 高精度要求
- ✅ 研究和論文
- ✅ 競賽

**特點：**
- 更大的模型
- 更大的圖片尺寸
- 更強的數據增強
- 更長的訓練時間

---

### 生產環境配置

```bash
python train_yolo_direct.py \
    --data_dir ../../datasets/splited_dataset/train \
    --epochs 400 \
    --batch_size 4 \
    --model_size x \
    --imgsz 1024 \
    --val_ratio 0.15 \
    --lr 0.0003 \
    --optimizer AdamW \
    --warmup_epochs 10 \
    --weight_decay 0.0005 \
    --cos_lr \
    --mosaic 1.0 \
    --mixup 0.2 \
    --copy_paste 0.15 \
    --fliplr 0.5 \
    --scale 0.9 \
    --translate 0.15 \
    --patience 50
```

**適用場景：**
- ✅ 最終部署模型
- ✅ 臨床應用
- ✅ 最高精度要求

**注意事項：**
- 需要大量 GPU 顯存（>16GB）
- 訓練時間很長（1-2 天）
- 需要足夠的驗證樣本

---

## 🎯 按需求選擇

### 我想要最快看到結果
→ **快速測試配置** (2-3 小時)

### 我第一次訓練這個數據集
→ **標準訓練配置** ⭐ (8-12 小時)

### 我需要檢測小病灶
→ **高精度配置** (20-30 小時)

### 我要部署到生產環境
→ **生產環境配置** (30-50 小時)

---

## 💻 硬體需求

| 配置 | GPU 顯存 | 系統內存 | 磁盤空間 |
|------|---------|---------|---------|
| 快速測試 | 6 GB | 16 GB | 10 GB |
| 標準訓練 | 8 GB | 32 GB | 15 GB |
| 高精度 | 12 GB | 32 GB | 20 GB |
| 生產環境 | 16+ GB | 64 GB | 30 GB |

**沒有 GPU？**
- 所有配置都可以在 CPU 上運行
- 訓練時間會增加 10-50 倍
- 建議使用雲端 GPU（如 Google Colab, AWS, Azure）

---

## 📊 性能對比（預期）

基於 146K 張胸部 CT 圖片：

| 配置 | mAP@0.5 | mAP@0.5:0.95 | 推理速度 | 模型大小 |
|------|---------|--------------|---------|---------|
| 快速測試 | 0.65-0.75 | 0.45-0.55 | ~2ms | 5 MB |
| 標準訓練 | 0.75-0.85 | 0.55-0.65 | ~5ms | 50 MB |
| 高精度 | 0.78-0.88 | 0.58-0.68 | ~8ms | 100 MB |
| 生產環境 | 0.80-0.90 | 0.60-0.70 | ~12ms | 200 MB |

---

## 🔧 參數調優指南

### 如果 Recall 太低（漏檢多）

```bash
# 增加數據增強
--mosaic 1.0 --mixup 0.2

# 降低置信度閾值（推理時）
# model.predict(conf=0.15)  # 預設 0.25

# 使用更大的模型
--model_size l

# 增加訓練輪數
--epochs 300
```

### 如果 Precision 太低（誤檢多）

```bash
# 減少數據增強
--mosaic 0.7 --mixup 0.0

# 提高置信度閾值（推理時）
# model.predict(conf=0.35)

# 增加 weight_decay
--weight_decay 0.001

# 提高 NMS IoU 閾值（推理時）
# model.predict(iou=0.5)
```

### 如果過擬合（驗證損失上升）

```bash
# 增強數據增強
--mosaic 1.0 --mixup 0.2 --scale 0.9

# 增加正則化
--weight_decay 0.001

# 減小學習率
--lr 0.0005

# 早停
--patience 30
```

### 如果欠擬合（訓練損失高）

```bash
# 使用更大的模型
--model_size l  # 或 x

# 增加訓練輪數
--epochs 400

# 提高學習率
--lr 0.002

# 減少數據增強
--mosaic 0.7
```

---

## 🎓 漸進式訓練策略

### 階段 1：快速驗證（第 1 天）

```bash
# 快速測試，驗證數據和流程
python train_yolo_direct.py --epochs 50 --model_size n
```

**目標：** 確認一切正常運行

### 階段 2：基準測試（第 2-3 天）

```bash
# 標準配置，建立基準
python train_yolo_direct.py --epochs 200 --model_size m
```

**目標：** 了解數據集特性，建立性能基準

### 階段 3：參數調優（第 4-7 天）

```bash
# 嘗試不同配置
python train_yolo_direct.py --epochs 300 --model_size l --mosaic 1.0 --mixup 0.15
```

**目標：** 找到最佳參數組合

### 階段 4：最終訓練（第 8-10 天）

```bash
# 使用最佳配置長時間訓練
python train_yolo_direct.py --epochs 400 --model_size l --imgsz 800
```

**目標：** 獲得最佳性能模型

---

## 📈 監控指標

### 訓練過程中關注

1. **Box Loss** - 邊界框回歸損失
   - 應該平穩下降
   - 訓練集和驗證集趨勢一致

2. **Objectness Loss** - 目標性損失
   - 模型識別目標的能力
   - 應該逐漸降低

3. **mAP@0.5** - 主要精度指標
   - 應該穩定上升
   - 驗證集不應該下降

### 最終評估指標

1. **mAP@0.5** - IoU=0.5 時的平均精度
2. **mAP@0.5:0.95** - IoU 0.5-0.95 的平均精度
3. **Precision** - 精確率（檢測的準確度）
4. **Recall** - 召回率（檢測的完整度）
5. **F1 Score** - Precision 和 Recall 的調和平均

---

## ✅ 推薦流程

```bash
# 1. 驗證數據集
python validate_dataset.py --data_dir ../../datasets/splited_dataset/train

# 2. 測試環境
python test_environment.py

# 3. 開始標準訓練
python train_yolo_direct.py \
    --data_dir ../../datasets/splited_dataset/train \
    --epochs 200 \
    --model_size m

# 4. 根據結果調整參數，重新訓練

# 5. 最終使用最佳配置訓練
```

---

## 📞 需要更多幫助？

- 📖 參考 [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)
- 📗 查看 [TRAIN_DIRECT_README.md](TRAIN_DIRECT_README.md)
- 🌐 訪問 [Ultralytics 官方文檔](https://docs.ultralytics.com/)

---

**選擇適合您的配置，開始訓練吧！🚀**
