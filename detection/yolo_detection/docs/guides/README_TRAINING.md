# YOLOv11 Training Scripts

訓練腳本集合，專門用於胸部 CT 病灶檢測。

## 📁 文件說明

| 文件 | 用途 | 適用場景 |
|------|------|----------|
| **train_yolo_direct.py** | ⭐ 主訓練腳本 | YOLO 格式數據集直接訓練 |
| **validate_dataset.py** | 數據集驗證工具 | 訓練前檢查數據完整性 |
| **train_quick_start.bat** | Windows 快速啟動 | 一鍵開始訓練（Windows） |
| **train_interactive.py** | 交互式配置 | 逐步配置訓練參數（跨平台） |
| **QUICK_START_GUIDE.md** | 🚀 快速開始指南 | 新手必讀 |
| **TRAIN_DIRECT_README.md** | 詳細文檔 | 完整參數說明和範例 |
| train_yolov11.py | 標準訓練腳本 | 需要 CTDetectionDataset |
| train_yolo_optimize.py | 優化訓練腳本 | 醫學影像預處理 + 進階功能 |

## 🚀 快速開始（3 步）

### 1️⃣ 驗證數據集

```bash
python validate_dataset.py --data_dir ../../datasets/splited_dataset/train
```

### 2️⃣ 開始訓練

**選項 A: 使用快速啟動（Windows）**
```batch
train_quick_start.bat
```

**選項 B: 使用交互式配置（跨平台）**
```bash
python train_interactive.py
```

**選項 C: 直接命令行（推薦）**
```bash
python train_yolo_direct.py \
    --data_dir ../../datasets/splited_dataset/train \
    --epochs 200 \
    --batch_size 16 \
    --model_size m \
    --val_ratio 0.2
```

### 3️⃣ 查看結果

```bash
# 訓練完成後，檢查輸出目錄
cd yolo_runs/experiments/yolo11m_*/

# 查看訓練曲線
# Windows: start results.png
# Linux/Mac: xdg-open results.png

# 使用最佳模型
python -c "from ultralytics import YOLO; model = YOLO('weights/best.pt'); model.val()"
```

## 📖 詳細文檔

- 📘 [快速開始指南](QUICK_START_GUIDE.md) - **推薦閱讀**
- 📗 [完整使用文檔](TRAIN_DIRECT_README.md) - 所有參數說明
- 📊 數據集驗證結果: ✅ 200 患者, 146,236 張圖片

## 🎯 腳本選擇指南

### 使用 train_yolo_direct.py（推薦） ✅

**適用場景：**
- ✅ 數據已經是 YOLO 格式（images/ + labels/）
- ✅ 數據已經完成預處理（HU 窗位、CLAHE 等）
- ✅ 想要簡單直接的訓練流程

**優勢：**
- 代碼簡潔（600 行）
- 無複雜依賴
- 自動患者級切分
- 完整日誌記錄

**當前數據集狀態：** ✅ 完全適用

### 使用 train_yolov11.py

**適用場景：**
- 需要從 CTDetectionDataset 載入
- 標準 YOLO 訓練流程
- 無醫學影像預處理需求

### 使用 train_yolo_optimize.py

**適用場景：**
- 需要 HU 窗位調整
- 需要 CLAHE 對比度增強
- 進階數據緩存機制
- 測試時增強（TTA）

## ⚙️ 推薦配置

### 初次訓練（基準測試）

```bash
python train_yolo_direct.py \
    --data_dir ../../datasets/splited_dataset/train \
    --epochs 200 \
    --batch_size 16 \
    --model_size m \
    --val_ratio 0.2 \
    --lr 0.001 \
    --optimizer AdamW
```

**預期時間：** ~8-12 小時（RTX 3090）

### 高精度訓練（生產環境）

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
    --mosaic 1.0 \
    --mixup 0.15 \
    --scale 0.7
```

**預期時間：** ~20-30 小時（RTX 3090）

### 快速實驗（測試參數）

```bash
python train_yolo_direct.py \
    --data_dir ../../datasets/splited_dataset/train \
    --epochs 50 \
    --batch_size 32 \
    --model_size n \
    --val_ratio 0.3
```

**預期時間：** ~2-3 小時

## 📊 預期性能

基於 146K 張胸部 CT 圖片：

| 模型 | mAP@0.5 | mAP@0.5:0.95 | 推理速度 |
|------|---------|---------------|----------|
| YOLOv11-n | 0.65-0.75 | 0.45-0.55 | ~2ms |
| YOLOv11-s | 0.70-0.80 | 0.50-0.60 | ~3ms |
| YOLOv11-m | 0.75-0.85 | 0.55-0.65 | ~5ms |
| YOLOv11-l | 0.78-0.88 | 0.58-0.68 | ~8ms |
| YOLOv11-x | 0.80-0.90 | 0.60-0.70 | ~12ms |

*實際結果取決於病灶大小、標註質量和訓練參數*

## 🐛 常見問題

### CUDA Out of Memory

```bash
# 減小批次大小
--batch_size 8

# 或減小圖片尺寸
--imgsz 512

# 或使用更小的模型
--model_size s
```

### 訓練速度慢

```bash
# 增加工作線程
--workers 16

# 減少數據增強
--mosaic 0.5 --mixup 0.0
```

### 驗證集指標不穩定

```bash
# 增加驗證集比例
--val_ratio 0.3

# 或固定隨機種子
--seed 42
```

## 📦 依賴要求

```bash
pip install ultralytics torch torchvision
```

**驗證安裝：**
```bash
python -c "from ultralytics import YOLO; print('✅ Ultralytics installed')"
```

## 🎓 學習資源

- [Ultralytics YOLOv11 文檔](https://docs.ultralytics.com/)
- [訓練技巧](https://docs.ultralytics.com/guides/model-training-tips/)
- [數據增強設定](https://docs.ultralytics.com/modes/train/#augmentation-settings)
- [超參數調優](https://docs.ultralytics.com/guides/hyperparameter-tuning/)

## 📝 訓練檢查清單

- [ ] 已運行 `validate_dataset.py` 驗證數據
- [ ] 已安裝 ultralytics 和 torch
- [ ] 已確認 GPU 可用（可選但推薦）
- [ ] 已選擇適當的模型大小
- [ ] 已設定合理的 batch_size（根據 GPU 顯存）
- [ ] 已準備足夠的磁盤空間（~10-20 GB）
- [ ] 已閱讀 [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)

## 🎉 開始訓練

一切準備就緒！選擇一種方式開始訓練：

```bash
# 方式 1: 快速啟動（Windows）
train_quick_start.bat

# 方式 2: 交互式配置
python train_interactive.py

# 方式 3: 直接命令行
python train_yolo_direct.py --data_dir ../../datasets/splited_dataset/train
```

**Good luck! 🚀**

---

*更新時間: 2025-10-13*  
*數據集狀態: ✅ 已驗證（200 患者, 146,236 圖片）*
