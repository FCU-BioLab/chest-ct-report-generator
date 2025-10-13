# YOLOv7 胸腔 CT 病灶檢測 - 完整指南

> **最後更新**: 2025-10-12  
> **版本**: 2.0 - 預處理加速版

---

## 📋 目錄

1. [專案概述](#專案概述)
2. [快速開始](#快速開始)
3. [兩種訓練方式](#兩種訓練方式)
4. [檔案說明](#檔案說明)
5. [訓練參數](#訓練參數)
6. [常見問題](#常見問題)
7. [📚 更多文檔](#-更多文檔)

---

> 💡 **尋找更多資訊？** 查看 [📚 文檔索引](docs/INDEX.md) 以獲取完整的技術文檔和指南

---

## 專案概述

### 核心功能
- **YOLOv7 醫學圖像檢測**: 整合醫學模組（CBAM、SimAM、Swin Transformer、BiFPN）
- **預處理加速**: 一次性預處理，訓練速度提升 8-10 倍
- **進階資料增強**: Mosaic、MixUp、Copy-Paste、正樣本過採樣
- **Focal Loss**: 解決類別不平衡問題
- **自動視覺化**: 每輪生成 TP/FP/FN 對比圖

### 效能對比

| 訓練方式 | 速度 (秒/batch) | Epoch 時間 | 120 Epochs | GPU 利用率 |
|---------|----------------|-----------|------------|-----------|
| **原始 DICOM** | 4.2 秒 | 6-7 小時 | 30-35 天 | 10-15% |
| **預處理 PNG** | 0.5 秒 | 40 分鐘 | **2-3 天** ✅ | 80-100% |

---

## 快速開始

### 方案 A: 預處理訓練（推薦，快 8-10 倍）🚀

#### 步驟 1: 預處理資料（一次性，60 分鐘）
```cmd
cd detection\dataset_process
python preprocess_for_yolo.py ^
    --data_root ../../datasets/splited_dataset ^
    --output_dir ../../datasets/preprocessed_yolo ^
    --imgsz 640
```

#### 步驟 2: 訓練（2-3 天）
```cmd
cd ..\yolo_detection
python train_yolov7_preprocessed.py ^
    --data_dir ../../datasets/preprocessed_yolo ^
    --epochs 120 ^
    --batch_size 24 ^
    --accumulation_steps 2 ^
    --use_medical_modules ^
    --use_ema ^
    --mixed_precision
```

### 方案 B: 原始 DICOM 訓練（初期實驗用）

```cmd
cd detection\yolo_detection
python train_yolov7_medical.py ^
    --data_dir ../../datasets/splited_dataset ^
    --epochs 120 ^
    --batch_size 16 ^
    --accumulation_steps 2 ^
    --use_medical_modules ^
    --workers 8 ^
    --mixed_precision ^
    --use_ema
```

---

## 兩種訓練方式

### 🔄 方式對比

| 項目 | 原始 DICOM | 預處理 PNG |
|------|-----------|-----------|
| **訓練腳本** | `train_yolov7_medical.py` | `train_yolov7_preprocessed.py` |
| **資料格式** | DICOM + XML | PNG + YOLO TXT |
| **預處理時機** | 每個 batch 即時 | 一次性預處理 |
| **磁碟 I/O** | 每次讀 DICOM（慢） | 讀 PNG（快 10 倍）|
| **適用場景** | 初期實驗、快速驗證 | 正式訓練、多次實驗 |
| **成本** | 無 | 60 分鐘 + 6 GB |

### 🎯 使用建議

**使用原始 DICOM 訓練（方案 B）**:
- ✅ 初期探索，驗證想法（1-10 epochs）
- ✅ 資料量很小（< 1000 樣本）
- ✅ 磁碟空間嚴重不足
- ✅ 需要動態調整預處理參數

**使用預處理 PNG 訓練（方案 A，強烈推薦）**:
- ✅ 正式訓練（120 epochs）
- ✅ 資料量大（> 1000 樣本）
- ✅ 需要多次實驗或超參數調優
- ✅ 有足夠磁碟空間（6-10 GB）

---

## 檔案說明

### 核心訓練腳本

| 檔案 | 用途 | 資料格式 |
|------|------|---------|
| `train_yolov7_medical.py` | 原始 DICOM 訓練 | DICOM + XML |
| `train_yolov7_preprocessed.py` | 預處理 PNG 訓練（快）| PNG + TXT |
| `train_preprocessed.bat` | 一鍵訓練批次腳本 | - |

### YOLOv7 核心模組

| 檔案 | 功能 |
|------|------|
| `yolov7_model.py` | 模型定義（含醫學模組）|
| `yolov7_dataset.py` | 資料增強包裝器 |
| `yolov7_utils.py` | 訓練工具（Focal Loss、EMA、損失計算）|
| `yolov7_augmentations.py` | 進階增強（Mosaic/MixUp/Copy-Paste）|
| `yolov7_eval_visualizer.py` | 評估視覺化（TP/FP/FN）|

### 預處理相關

| 檔案 | 功能 |
|------|------|
| `preprocessed_dataset.py` | 載入預處理資料的 Dataset 類 |
| `../dataset_process/preprocess_for_yolo.py` | DICOM → PNG 預處理腳本 |
| `../dataset_process/run_preprocess_yolo.bat` | 一鍵預處理批次腳本 |

### 工具腳本

| 檔案 | 功能 |
|------|------|
| `validate_annotations.py` | 標註驗證與視覺化 |
| `check_gpu.py` | GPU 可用性檢查 |
| `monitor_gpu.py` | 即時 GPU 監控 |

### 模型配置

| 檔案 | 說明 |
|------|------|
| `models/yolov7_medical.yaml` | 醫學模組版本（CBAM、SimAM、Swin、BiFPN）|
| `models/yolov7_baseline.yaml` | 基礎版本（對照組）|
| `models/custom_layers.py` | 自定義層實作 |

---

## 訓練參數

### 預處理訓練參數（推薦）

#### 基礎訓練
```cmd
python train_yolov7_preprocessed.py ^
    --data_dir ../../datasets/preprocessed_yolo ^
    --epochs 120 ^
    --batch_size 24 ^
    --use_medical_modules
```

#### 完整訓練（推薦）
```cmd
python train_yolov7_preprocessed.py ^
    --data_dir ../../datasets/preprocessed_yolo ^
    --epochs 120 ^
    --batch_size 24 ^
    --accumulation_steps 2 ^
    --use_medical_modules ^
    --use_ema ^
    --mixed_precision ^
    --cos_lr ^
    --positive_oversample ^
    --positive_ratio 0.6
```

### 原始 DICOM 訓練參數

#### 測試訓練（10 輪）
```cmd
python train_yolov7_medical.py ^
    --data_dir ../../datasets/splited_dataset ^
    --epochs 10 ^
    --batch_size 4 ^
    --workers 2 ^
    --enable_augmentation ^
    --positive_oversample ^
    --use_focal_loss ^
    --visualize_predictions
```

#### 推薦配置（含增強）
```cmd
python train_yolov7_medical.py ^
    --data_dir ../../datasets/splited_dataset ^
    --epochs 120 ^
    --batch_size 8 ^
    --accumulation_steps 4 ^
    --enable_augmentation ^
    --mosaic_prob 0.5 ^
    --mixup_prob 0.3 ^
    --copy_paste_prob 0.3 ^
    --positive_oversample ^
    --positive_ratio 0.7 ^
    --use_focal_loss ^
    --cls_loss_gain 1.8 ^
    --visualize_predictions ^
    --workers 4 ^
    --mixed_precision ^
    --use_ema
```

### 關鍵參數說明

| 參數 | 說明 | 推薦值 | 作用 |
|------|------|--------|------|
| `--batch_size` | 批次大小 | 24 (預處理) / 8 (DICOM) | 影響記憶體和訓練速度 |
| `--accumulation_steps` | 梯度累積步數 | 2-4 | 有效 batch = batch_size × steps |
| `--epochs` | 訓練輪數 | 120 | 完整訓練建議 120+ |
| `--workers` | DataLoader workers | 4-8 | 影響資料載入速度 |
| `--use_medical_modules` | 使用醫學模組 | ✅ | CBAM/SimAM/Swin/BiFPN |
| `--mixed_precision` | 混合精度訓練 | ✅ | 節省記憶體、加速訓練 |
| `--use_ema` | EMA 模型 | ✅ | 提升模型穩定性 |
| `--cos_lr` | Cosine 學習率 | ✅ | 平滑學習率衰減 |
| `--enable_augmentation` | 啟用進階增強 | ✅ (DICOM) | Mosaic/MixUp/Copy-Paste |
| `--positive_oversample` | 正樣本過採樣 | ✅ | 平衡正負樣本 |
| `--positive_ratio` | 目標正樣本比例 | 0.6-0.7 | 正樣本占比 |
| `--use_focal_loss` | Focal Loss | ✅ (DICOM) | 解決類別不平衡 |
| `--cls_loss_gain` | 分類損失權重 | 1.8 | 增加分類損失 |
| `--visualize_predictions` | 啟用視覺化 | ✅ (可選) | 每輪生成 TP/FP/FN 圖 |

---

## 常見問題

### Q1: 應該使用哪種訓練方式？

**A**: 強烈推薦使用預處理訓練（方案 A）：
- 初期用原始 DICOM 驗證（1-10 epochs）
- 確定配置後執行預處理（60 分鐘）
- 用預處理資料完整訓練（2-3 天 vs 30-35 天）

### Q2: 預處理需要多少磁碟空間？

**A**: 
- 640×640: 約 6 GB
- 800×800: 約 10 GB
- 512×512: 約 4 GB

### Q3: 如何監控訓練進度？

**A**: 
```cmd
# 查看日誌
Get-Content yolov7_logs\yolov7_training_*.log -Wait

# 即時 GPU 監控
python monitor_gpu.py

# 查看視覺化結果
explorer yolov7_models\run_*\visualizations
```

### Q4: OOM (記憶體不足) 怎麼辦？

**A**: 
1. 減小 batch_size: `--batch_size 16` → `--batch_size 8`
2. 增加累積步數: `--accumulation_steps 4`
3. 減小圖像尺寸: `--imgsz 640` → `--imgsz 512`
4. 關閉部分增強: 移除 `--enable_augmentation`

### Q5: 訓練速度太慢怎麼辦？

**A**: 
1. **使用預處理**（最重要！8-10 倍提升）
2. 檢查 GPU 是否被正確使用: `python check_gpu.py`
3. 調整 workers: `--workers 4` 或 `--workers 8`
4. 啟用混合精度: `--mixed_precision`
5. 增大 batch_size（如果記憶體足夠）

### Q6: 預處理後的資料可以刪除嗎？

**A**: 可以。訓練完成後如果空間不足，可以刪除預處理資料。需要重新訓練時再次執行預處理即可（60 分鐘）。

### Q7: 如何驗證標註是否正確？

**A**: 
```cmd
python validate_annotations.py ^
    --data_root ../../datasets/splited_dataset ^
    --num_samples 100 ^
    --output_dir ./annotation_validation
```

### Q8: 預期的訓練效果是什麼？

**A**: 

| Epoch | Baseline mAP@0.5 | Enhanced mAP@0.5 | 改善 |
|-------|------------------|------------------|------|
| 10 | < 0.01 | 0.05-0.15 | 5-15× |
| 50 | 0.05-0.15 | 0.25-0.40 | 3-5× |
| 120 | 0.15-0.30 | 0.45-0.65 | 2-3× |

### Q9: 如何使用訓練好的模型？

**A**: 訓練完成後：
- 最佳模型: `yolov7_models/run_*/weights/best.pt`
- 最後模型: `yolov7_models/run_*/weights/last.pt`
- 使用 `test_yolov11.py` 進行推論測試

### Q10: 為什麼有兩個 README？

**A**: 
- `README.md`（本檔案）: 主要使用指南，包含預處理訓練
- `README_PREPROCESSED_TRAINING.md`: 詳細的預處理技術說明
- `TRAINING_COMPARISON.md`: 兩種方法的深入對比

---

## 輸出結構

```
yolov7_models/run_<timestamp>/
├── weights/
│   ├── best.pt              # 最佳模型
│   └── last.pt              # 最後一輪
├── visualizations/          # 視覺化結果（如果啟用）
│   ├── epoch_001/
│   ├── epoch_002/
│   └── ...
├── training_history.json    # 訓練歷史
└── summary.json            # 訓練摘要

yolov7_logs/
└── yolov7_training_*.log   # 訓練日誌

preprocessed_yolo/           # 預處理資料（如果使用）
├── data.yaml
├── train/
│   ├── images/
│   ├── labels/
│   └── metadata.json
├── val/
└── test/
```

---

## 📚 更多文檔

需要更深入的技術說明或詳細指南？查看我們的文檔中心：

### 📖 [文檔索引](docs/INDEX.md)

**使用指南**:
- [完整訓練流程指南](docs/guides/TRAINING_GUIDE.md) - Step-by-step 操作與參數配置
- [訓練方法對比](docs/guides/TRAINING_COMPARISON.md) - 原始 DICOM vs 預處理 PNG
- [預處理訓練技術說明](docs/guides/README_PREPROCESSED_TRAINING.md) - 開發者深入技術細節

**技術參考**:
- [負樣本判斷機制](docs/references/NEGATIVE_SAMPLE_DETECTION.md) - 資料處理細節
- [驗證指標說明](docs/references/VALIDATION_METRICS_GUIDE.md) - 指標計算與查看
- [預處理資料集結構](docs/references/PREPROCESSED_DATASET_UPDATE.md) - 支援的資料格式
- [YOLOv7 vs YOLOv11n 分析](docs/references/YOLOV7_VS_YOLOV11_ANALYSIS.md) - 模型架構對比

---

## 版本歷史

### v2.0 (2025-10-12)
- ✨ 新增預處理訓練支援（8-10 倍加速）
- ✨ 新增 `train_yolov7_preprocessed.py`
- ✨ 新增 `preprocessed_dataset.py`
- 📝 整合並精簡文檔
- 📁 重組文檔結構（創建 docs/ 目錄）
- 🗑️ 移除過時和重複文檔

### v1.0 (2025-10-10)
- ✨ 整合所有增強功能到主訓練腳本
- ✨ 支援 Mosaic/MixUp/Copy-Paste 增強
- ✨ 正樣本過採樣
- ✨ Focal Loss
- ✨ 自動視覺化
- 🗑️ 移除冗餘腳本和文檔

---

## 授權

本專案使用 MIT 授權。

## 貢獻

歡迎提交 Issue 和 Pull Request。

---

**最後更新**: 2025-10-12  
**維護者**: GitHub Copilot
