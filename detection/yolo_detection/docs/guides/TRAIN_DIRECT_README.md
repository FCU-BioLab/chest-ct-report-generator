# YOLOv11 Direct Training Guide

## 📋 Overview

`train_yolo_direct.py` 是專門為**已經預處理並轉換成 YOLO 格式**的數據集設計的訓練腳本。它會自動基於患者 ID 進行數據切分，避免數據洩漏。

## 🎯 適用場景

✅ 數據集已經是 YOLO 格式（images/ 和 labels/ 目錄）  
✅ 需要自動進行 train/validation 切分  
✅ 希望基於患者 ID 切分（而非隨機切分圖片）  
✅ 不需要額外的醫學影像預處理（已在預處理階段完成）

## 📁 數據集結構要求

```
datasets/splited_dataset/train/
├── A0001/
│   ├── images/
│   │   ├── A0001_DCM_001_1-01.png
│   │   ├── A0001_DCM_002_1-02.png
│   │   └── ...
│   └── labels/
│       ├── A0001_DCM_001_1-01.txt
│       ├── A0001_DCM_002_1-02.txt
│       └── ...
├── A0003/
│   ├── images/
│   └── labels/
└── ...
```

**關鍵點：**
- 每個患者目錄下必須有 `images/` 和 `labels/` 子目錄
- 圖片格式：PNG
- 標籤格式：YOLO txt（`class x_center y_center width height`，歸一化座標）

## 🚀 快速開始

### 方法 1：使用快速啟動腳本（推薦 Windows 用戶）

```batch
cd detection\yolo_detection
train_quick_start.bat
```

### 方法 2：基本命令行

```bash
python train_yolo_direct.py \
    --data_dir ../../datasets/splited_dataset/train \
    --epochs 200 \
    --batch_size 16 \
    --model_size m
```

### 方法 3：完整參數配置

```bash
python train_yolo_direct.py \
    --data_dir ../../datasets/splited_dataset/train \
    --epochs 200 \
    --batch_size 16 \
    --imgsz 640 \
    --model_size m \
    --lr 0.001 \
    --val_ratio 0.2 \
    --optimizer AdamW \
    --warmup_epochs 5 \
    --mosaic 1.0 \
    --mixup 0.0 \
    --fliplr 0.5 \
    --save_dir ./yolo_runs \
    --workers 8
```

## ⚙️ 參數說明

### 必需參數

| 參數 | 說明 | 範例 |
|------|------|------|
| `--data_dir` | 訓練數據目錄路徑 | `../../datasets/splited_dataset/train` |

### 訓練參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--epochs` | 200 | 訓練輪數 |
| `--batch_size` | 16 | 批次大小 |
| `--imgsz` | 640 | 輸入圖片尺寸 |
| `--model_size` | m | 模型大小 (n/s/m/l/x) |
| `--lr` | 0.001 | 學習率 |

### 數據切分

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--val_ratio` | 0.2 | 驗證集比例 (0.0-1.0) |
| `--seed` | 42 | 隨機種子（確保可重現性） |

### 優化器設定

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--optimizer` | AdamW | 優化器 (SGD/Adam/AdamW) |
| `--weight_decay` | 0.0005 | 權重衰減 |
| `--warmup_epochs` | 5 | 預熱輪數 |

### 數據增強

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--mosaic` | 1.0 | Mosaic 增強機率 |
| `--mixup` | 0.0 | MixUp 增強機率 |
| `--degrees` | 0.0 | 旋轉角度 |
| `--translate` | 0.1 | 平移比例 |
| `--scale` | 0.5 | 縮放比例 |
| `--fliplr` | 0.5 | 水平翻轉機率 |

### 輸出設定

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--save_dir` | ./yolo_runs | 輸出目錄 |
| `--workers` | 8 | 數據載入線程數 |
| `--device` | auto | 裝置 (cuda/cpu/mps/auto) |

## 📊 輸出結構

訓練完成後，輸出目錄結構如下：

```
yolo_runs/
├── dataset_20241013_143022/          # 準備好的 YOLO 數據集
│   ├── images/
│   │   ├── train/                    # 訓練圖片
│   │   └── val/                      # 驗證圖片
│   ├── labels/
│   │   ├── train/                    # 訓練標籤
│   │   └── val/                      # 驗證標籤
│   ├── dataset.yaml                  # YOLO 配置文件
│   └── patient_split.json            # 患者切分記錄
├── experiments/
│   └── yolo11m_20241013_143022/      # 訓練實驗目錄
│       ├── weights/
│       │   ├── best.pt               # 最佳模型 ⭐
│       │   └── last.pt               # 最後一輪模型
│       ├── results.png               # 訓練曲線圖
│       ├── confusion_matrix.png      # 混淆矩陣
│       └── ...
├── logs/
│   └── training_20241013_143022.log  # 訓練日誌
└── summary_20241013_143022.json      # 訓練總結
```

## 📈 監控訓練進度

### 1. 查看終端輸出

訓練過程中會實時顯示：
- Epoch 進度
- 損失值 (loss)
- 指標 (mAP, precision, recall)

### 2. 查看日誌文件

```bash
# 實時查看日誌（Linux/Mac）
tail -f yolo_runs/logs/training_20241013_143022.log

# Windows PowerShell
Get-Content yolo_runs\logs\training_20241013_143022.log -Wait
```

### 3. 查看訓練曲線

訓練完成後，檢查 `experiments/yolo11m_*/results.png` 查看：
- 損失曲線
- mAP 曲線
- Precision/Recall 曲線

## 🎓 模型大小選擇建議

| 模型 | 參數量 | 速度 | 準確度 | 適用場景 |
|------|--------|------|--------|----------|
| **n** (nano) | 2.6M | 最快 | 較低 | 快速原型、資源受限 |
| **s** (small) | 9.4M | 快 | 中等 | 輕量部署 |
| **m** (medium) | 20.1M | 中等 | 良好 | **推薦用於 CT 檢測** ⭐ |
| **l** (large) | 25.3M | 較慢 | 優秀 | 高精度要求 |
| **x** (xlarge) | 54.2M | 最慢 | 最佳 | 競賽、學術研究 |

**建議：** 對於胸部 CT 病灶檢測，推薦從 **m** 或 **l** 開始。

## 🔧 進階配置

### 1. 高精度配置（適合小病灶檢測）

```bash
python train_yolo_direct.py \
    --data_dir ../../datasets/splited_dataset/train \
    --model_size l \
    --imgsz 800 \
    --epochs 300 \
    --batch_size 8 \
    --lr 0.0005 \
    --mosaic 1.0 \
    --mixup 0.15 \
    --scale 0.7
```

### 2. 快速實驗配置

```bash
python train_yolo_direct.py \
    --data_dir ../../datasets/splited_dataset/train \
    --model_size n \
    --epochs 50 \
    --batch_size 32 \
    --val_ratio 0.3
```

### 3. 多 GPU 訓練

```bash
python train_yolo_direct.py \
    --data_dir ../../datasets/splited_dataset/train \
    --device 0,1,2,3 \
    --batch_size 64 \
    --workers 16
```

## 📝 訓練後分析

### 1. 查看訓練總結

```bash
# 查看 JSON 總結文件
cat yolo_runs/summary_20241013_143022.json
```

總結包含：
- 配置參數
- 數據集統計
- 訓練時間
- 最終指標

### 2. 使用訓練好的模型

```python
from ultralytics import YOLO

# 載入最佳模型
model = YOLO('yolo_runs/experiments/yolo11m_*/weights/best.pt')

# 推理
results = model.predict('test_image.png', conf=0.25)

# 驗證
metrics = model.val()
```

## 🐛 常見問題

### Q1: 找不到數據

**錯誤：** `Data directory not found`

**解決方案：**
- 確認路徑正確
- 使用絕對路�徑或相對於腳本的正確相對路徑
- 檢查目錄下是否有患者子目錄

### Q2: CUDA out of memory

**錯誤：** `RuntimeError: CUDA out of memory`

**解決方案：**
```bash
# 減小批次大小
--batch_size 8

# 或減小圖片尺寸
--imgsz 512

# 或使用更小的模型
--model_size s
```

### Q3: 驗證集太小

**錯誤：** 驗證集只有 1-2 個患者

**解決方案：**
```bash
# 增加驗證比例
--val_ratio 0.3

# 或確保有足夠的患者數量（建議至少 20+ 患者）
```

### Q4: 訓練速度慢

**解決方案：**
```bash
# 增加工作線程
--workers 16

# 減少數據增強
--mosaic 0.5 --mixup 0.0

# 使用更小的模型
--model_size s
```

## 📚 與其他腳本的比較

| 特性 | train_yolo_direct.py | train_yolov11.py | train_yolo_optimize.py |
|------|----------------------|------------------|------------------------|
| **數據格式** | YOLO (已預處理) ✅ | 需要 CTDetectionDataset | 需要 CTDetectionDataset |
| **醫學預處理** | ❌ (假設已完成) | ❌ | ✅ HU 窗位調整 + CLAHE |
| **數據切分** | 基於患者 ID ✅ | 基於患者 ID | 基於患者 ID |
| **代碼複雜度** | 簡單 (600 行) | 中等 (900 行) | 複雜 (1200+ 行) |
| **使用場景** | **預處理完成後訓練** ⭐ | 標準訓練流程 | 進階優化 + 醫學處理 |

**選擇建議：**
- ✅ **使用 train_yolo_direct.py**：數據已經預處理成 YOLO 格式
- 使用 train_yolov11.py：需要從原始數據開始
- 使用 train_yolo_optimize.py：需要醫學影像專用預處理

## 🔗 相關資源

- [Ultralytics YOLOv11 官方文檔](https://docs.ultralytics.com/)
- [YOLO 數據格式說明](https://docs.ultralytics.com/datasets/detect/)
- [訓練技巧和最佳實踐](https://docs.ultralytics.com/guides/model-training-tips/)

## 📧 支持

如有問題或建議，請檢查：
1. 訓練日誌文件
2. 數據集結構是否正確
3. 依賴包是否安裝完整

---

**祝訓練順利！** 🚀
