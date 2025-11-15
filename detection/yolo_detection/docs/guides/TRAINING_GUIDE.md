# YOLOv7 訓練快速指南

## 完整工作流程

### Step 1: 預處理原始資料
```bash
cd detection/dataset_process
python preprocess_original_dataset.py \
    --data_root ../../datasets/all_patient_data \
    --output_dir ../../datasets/preprocessed_yolo_lesion \
    --enable_preprocessing False
```

**輸出結構**:
```
preprocessed_yolo_lesion/
├── A0001/
│   ├── images/    # PNG 圖像
│   └── labels/    # YOLO 標註
├── A0002/
└── ...
```

### Step 2: 劃分資料集
```bash
cd ../../dataset_process
python dataset_splitter.py \
    --source_dir ../datasets/preprocessed_yolo_lesion \
    --output_dir ../datasets/splited_dataset \
    --train_ratio 0.9 \
    --test_ratio 0.1
```

**輸出結構**:
```
splited_dataset/
├── train/         # 90% 患者
│   ├── A0001/
│   ├── A0003/
│   └── ...
├── test/          # 10% 患者
│   ├── A0002/
│   └── ...
├── train_patients.txt
├── test_patients.txt
└── dataset_split_report.json
```

### Step 3: 訓練模型
```bash
cd ../detection/yolo_detection
python train_yolov7_preprocessed.py \
    --data_dir ../../datasets/splited_dataset \
    --epochs 120 \
    --batch_size 24 \
    --val_ratio 0.15
```

**訓練邏輯**:
- 載入 `train/` 目錄的所有資料
- 自動分割 15% 作為驗證集
- 剩餘 85% 用於訓練
- `test/` 目錄保留用於最終測試

## 關鍵參數

### 資料相關
| 參數 | 預設值 | 說明 |
|-----|--------|------|
| `--data_dir` | 必填 | 預處理資料目錄 |
| `--train_split` | train | 訓練集目錄名稱 |
| `--val_ratio` | 0.15 | 從訓練集分割的驗證集比例 |

### 訓練相關
| 參數 | 預設值 | 說明 |
|-----|--------|------|
| `--epochs` | 120 | 訓練輪數 |
| `--batch_size` | 24 | 批次大小 |
| `--accumulation_steps` | 2 | 梯度累積步數 |
| `--lr` | 0.001 | 學習率 |
| `--imgsz` | 640 | 圖像尺寸 |

### 模型相關
| 參數 | 預設值 | 說明 |
|-----|--------|------|
| `--use_medical_modules` | True | 使用醫學模組 |
| `--use_ema` | True | 使用 EMA |
| `--mixed_precision` | True | 混合精度訓練 |
| `--cos_lr` | True | Cosine 學習率衰減 |

### 增強相關
| 參數 | 預設值 | 說明 |
|-----|--------|------|
| `--enable_augmentation` | False | 啟用資料增強 |
| `--positive_oversample` | True | 正樣本過採樣 |
| `--positive_ratio` | 0.6 | 正樣本目標比例 |

## 訓練範例

### 1. 快速測試（小規模）
```bash
python train_yolov7_preprocessed.py \
    --data_dir ../../datasets/splited_dataset \
    --epochs 10 \
    --batch_size 8 \
    --val_ratio 0.2
```

### 2. 標準訓練（推薦）
```bash
python train_yolov7_preprocessed.py \
    --data_dir ../../datasets/splited_dataset \
    --epochs 120 \
    --batch_size 24 \
    --val_ratio 0.15 \
    --use_medical_modules \
    --use_ema \
    --mixed_precision
```

### 3. 高性能訓練（大 GPU）
```bash
python train_yolov7_preprocessed.py \
    --data_dir ../../datasets/splited_dataset \
    --epochs 150 \
    --batch_size 32 \
    --accumulation_steps 4 \
    --val_ratio 0.15 \
    --use_medical_modules \
    --use_ema \
    --mixed_precision \
    --positive_oversample \
    --positive_ratio 0.7
```

### 4. 啟用資料增強
```bash
python train_yolov7_preprocessed.py \
    --data_dir ../../datasets/splited_dataset \
    --epochs 150 \
    --batch_size 24 \
    --val_ratio 0.15 \
    --enable_augmentation \
    --mosaic_prob 0.3 \
    --mixup_prob 0.2 \
    --copy_paste_prob 0.1
```

## 訓練輸出

### 目錄結構
```
yolov7_models/
└── run_20251012_144207/
    ├── weights/
    │   ├── best.pt         # 最佳模型
    │   └── last.pt         # 最後一輪
    ├── args.yaml
    ├── results.csv
    ├── results.png
    └── yolov7_training_results_*.json

yolov7_logs/
└── yolov7_training_20251012_144207.log
```

### 日誌示例
```
================================================================================
🚀 使用預處理資料訓練 YOLOv7
================================================================================
資料目錄:   ../../datasets/splited_dataset
訓練集:     train
驗證集比例: 15.0% (從訓練集分割)
Epochs:     120
Batch:      24 x 2 = 48
醫學模組:   ✓ 啟用
================================================================================
⚡ 預處理加速: 預期 8-10 倍訓練速度提升！
================================================================================

2025-10-12 14:42:10 - INFO - 完整訓練集: 146236 樣本
2025-10-12 14:42:10 - INFO - 按比例 15.0% 分割驗證集
2025-10-12 14:42:10 - INFO - 訓練集: 124300 樣本 (85.0%)
2025-10-12 14:42:10 - INFO - 驗證集: 21936 樣本 (15.0%)

Epoch 1/120 - Train Loss: 0.1234, Val Loss: 0.0987
✓ 保存最佳模型 (val_loss: 0.0987)
...
```

## 資料分割策略

### 當前策略（推薦）
```
原始資料 (215 位患者)
    ↓
dataset_splitter.py (90:10 分割)
    ↓
├── train/ (193 位患者, 146236 張圖)
│   ↓
│   train_yolov7_preprocessed.py (85:15 分割)
│   ↓
│   ├── 實際訓練集 (85%, ~124300 張圖)
│   └── 驗證集 (15%, ~21900 張圖)
│
└── test/ (22 位患者, 38069 張圖) → 最終測試
```

### 優點
1. ✅ **患者級別分割**: train 和 test 之間沒有患者重疊
2. ✅ **靈活驗證**: 可隨時調整 val_ratio
3. ✅ **保留測試集**: test 完全獨立，用於最終評估
4. ✅ **可重現性**: 使用 random_seed 確保分割一致

### 驗證集比例建議
| 訓練集大小 | 建議 val_ratio | 驗證集大小 | 訓練集大小 |
|-----------|---------------|-----------|-----------|
| < 10,000 | 0.20 (20%) | 更多驗證樣本 | 但訓練資料較少 |
| 10k - 50k | 0.15 (15%) | ⭐ 推薦 | 平衡 |
| 50k - 100k | 0.10 (10%) | 足夠驗證 | 更多訓練資料 |
| > 100k | 0.05-0.10 | 足夠驗證 | 充分訓練 |

**當前資料集 (146k)**: 建議使用 `--val_ratio 0.10` 到 `0.15`

## 監控訓練

### 查看訓練日誌（Windows PowerShell）
```powershell
# 實時監控
Get-Content yolov7_logs\yolov7_training_*.log -Wait -Tail 50

# 查看所有訓練運行
Get-ChildItem yolov7_models\run_*

# 查看最佳模型
Get-ChildItem yolov7_models\run_*\weights\best.pt
```

## 常見問題

### Q1: 應該使用多大的驗證集？
**A**: 對於 146k 樣本的資料集，建議使用 10-15%（約 14k-22k 樣本）

### Q2: 為什麼不直接使用 test 目錄作為驗證集？
**A**: test 應該保留用於最終模型評估，不應該在訓練過程中使用

### Q3: 如何進行 K-Fold 交叉驗證？
**A**: 
```bash
# 生成不同的 fold
python dataset_splitter.py --random_seed 42 --output_dir ../datasets/fold_1
python dataset_splitter.py --random_seed 43 --output_dir ../datasets/fold_2
python dataset_splitter.py --random_seed 44 --output_dir ../datasets/fold_3

# 分別訓練
python train_yolov7_preprocessed.py --data_dir ../../datasets/fold_1 ...
python train_yolov7_preprocessed.py --data_dir ../../datasets/fold_2 ...
python train_yolov7_preprocessed.py --data_dir ../../datasets/fold_3 ...
```

### Q4: 訓練速度慢怎麼辦？
**A**: 
- 減小 batch_size
- 減少 num_workers
- 關閉資料增強（已預處理）
- 使用 mixed_precision

### Q5: 記憶體不足怎麼辦？
**A**:
```bash
# 減小 batch_size 和增加 accumulation_steps 來保持有效 batch size
python train_yolov7_preprocessed.py \
    --batch_size 8 \
    --accumulation_steps 6  # 有效 batch = 8 x 6 = 48
```

## 效能基準

| 設定 | GPU | Batch | 速度 | 記憶體 |
|-----|-----|-------|------|--------|
| 基礎 | RTX 3090 | 24 | ~3 img/s | ~18 GB |
| 高性能 | RTX 4090 | 32 | ~5 img/s | ~22 GB |
| 低記憶體 | RTX 3060 | 8 | ~1.5 img/s | ~8 GB |

## 更新日期
2025-10-12
