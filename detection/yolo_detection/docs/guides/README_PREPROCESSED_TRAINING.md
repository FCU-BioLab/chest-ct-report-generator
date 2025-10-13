# 📚 預處理訓練技術說明

> **注意**: 本文檔為技術細節說明。快速開始請參考 [README.md](README.md)

## 資料流程

```
原始 DICOM + XML → preprocess_for_yolo.py → PNG + YOLO TXT → train_yolov7_preprocessed.py → 訓練模型
```

## 📁 檔案說明

### 1. `preprocess_for_yolo.py`
**位置**: `detection/dataset_process/`

**功能**: 
- 讀取原始 DICOM 檔案和 XML 標註
- 應用 HU windowing (-600 center, 1500 width)
- 應用 CLAHE 對比度增強
- 轉換為 PNG 圖像
- 轉換標註為 YOLO 格式 (TXT)
- 保留 train/val/test 結構

**輸出結構**:
```
preprocessed_yolo/
├── data.yaml              # YOLOv7 配置
├── train/
│   ├── images/           # PNG 圖像
│   │   ├── 000000.png
│   │   └── ...
│   ├── labels/           # YOLO 標註
│   │   ├── 000000.txt
│   │   └── ...
│   └── metadata.json
├── val/
│   ├── images/
│   ├── labels/
│   └── metadata.json
└── test/
    ├── images/
    ├── labels/
    └── metadata.json
```

**執行**:
```cmd
cd detection\dataset_process
python preprocess_for_yolo.py ^
    --data_root ../../datasets/splited_dataset ^
    --output_dir ../../datasets/preprocessed_yolo ^
    --imgsz 640
```

---

### 2. `preprocessed_dataset.py`
**位置**: `detection/yolo_detection/`

**功能**: 
- **新創建的 Dataset 類**
- 直接讀取預處理後的 PNG 圖像
- 讀取 YOLO 格式標註 (TXT)
- 跳過 DICOM 讀取和即時預處理
- 提供與 `CTDetectionDataset` 兼容的接口

**關鍵類**:
```python
class PreprocessedYOLODataset(Dataset):
    """載入預處理後的 PNG + YOLO labels"""
    
    def __init__(self, data_root, split="train", img_size=640, augment=False):
        # 從 data_root/split/images/ 和 data_root/split/labels/ 載入
        pass
    
    def __getitem__(self, idx):
        # 讀取 PNG 圖像（已預處理）
        # 讀取 YOLO 標註
        # 返回 (image_tensor, labels_tensor, metadata)
        pass
```

**為什麼需要這個檔案**:
- 原本的 `CTDetectionDataset` 只能讀取 DICOM + XML
- 預處理後的資料是 PNG + TXT，格式不同
- 需要新的 Dataset 類來載入預處理資料

---

### 3. `yolov7_dataset.py`
**位置**: `detection/yolo_detection/`

**功能**: 
- **資料增強和預處理包裝器**
- 包裝任何 Dataset（包括 `CTDetectionDataset` 和 `PreprocessedYOLODataset`）
- 應用醫學預處理（HU windowing, CLAHE）- **可選**
- 應用資料增強（Mosaic, MixUp, Copy-Paste）
- 正樣本過採樣
- 創建 DataLoader

**關鍵類**:
```python
class YOLOv7MedicalDataset(Dataset):
    """包裝器 Dataset，添加預處理和增強"""
    
    def __init__(self, dataset, img_size=640, 
                 enable_hu_windowing=True, enable_clahe=True,
                 augment=True, ...):
        self.dataset = dataset  # 可以是任何 Dataset
        # ...
    
    def __getitem__(self, idx):
        # 從底層 dataset 獲取資料
        item = self.dataset[idx]
        
        # 應用預處理（如果啟用）
        if self.enable_hu_windowing:
            img = apply_hu_windowing(...)
        if self.enable_clahe:
            img = apply_clahe(...)
        
        # 應用增強（如果啟用）
        if self.augment:
            img, labels = self.augmenter(...)
        
        return img_tensor, labels_tensor, metadata
```

**何時使用**:
- ✅ **原始 DICOM 訓練**: 需要即時 HU windowing + CLAHE
- ✅ **預處理資料訓練**: 只需要增強，不需要預處理（設定 `enable_hu_windowing=False, enable_clahe=False`）

**資料流程**:
```
方案 1: 原始 DICOM 訓練
CTDetectionDataset → YOLOv7MedicalDataset (HU+CLAHE+增強) → DataLoader

方案 2: 預處理資料訓練
PreprocessedYOLODataset → YOLOv7MedicalDataset (只增強) → DataLoader
```

---

### 4. `train_yolov7_medical.py`
**位置**: `detection/yolo_detection/`

**功能**: 
- **原始訓練腳本**（用於 DICOM 資料）
- 使用 `CTDetectionDataset`
- 每個 batch 都要讀取 DICOM 並預處理
- **速度慢**（4.2 秒/batch）

**為什麼不能直接用**:
```python
# train_yolov7_medical.py 中的程式碼
def prepare_datasets(config):
    # ❌ 只能載入 DICOM + XML
    train_dataset = CTDetectionDataset(
        data_root=config.data_dir,  # 需要 DICOM 檔案
        split=config.train_split,
        format_type="yolo",
    )
    return train_dataset, val_dataset
```

如果你傳入預處理目錄 `--data_dir ../../datasets/preprocessed_yolo`：
- ❌ `CTDetectionDataset` 會找不到 DICOM 檔案
- ❌ 會找不到 XML 標註檔案
- ❌ 無法載入資料

---

### 5. `train_yolov7_preprocessed.py`
**位置**: `detection/yolo_detection/`

**功能**: 
- **新創建的訓練腳本**（用於預處理資料）
- 使用 `PreprocessedYOLODataset`
- 直接讀取 PNG 圖像（已預處理）
- **速度快**（0.5 秒/batch，8-10 倍提升）

**為什麼需要這個檔案**:
- 替換 `prepare_datasets()` 函數，使用 `PreprocessedYOLODataset`
- 關閉 HU windowing 和 CLAHE（資料已處理）
- 保留其他訓練邏輯（損失函數、優化器、EMA 等）

**執行**:
```cmd
cd detection\yolo_detection
python train_yolov7_preprocessed.py ^
    --data_dir ../../datasets/preprocessed_yolo ^
    --epochs 120 ^
    --batch_size 24
```

---

## 🎯 完整工作流程

### 步驟 1: 預處理資料（一次性，30-60 分鐘）
```cmd
cd detection\dataset_process
python preprocess_for_yolo.py ^
    --data_root ../../datasets/splited_dataset ^
    --output_dir ../../datasets/preprocessed_yolo ^
    --imgsz 640
```

**輸出**: `preprocessed_yolo/` 目錄，包含 PNG 圖像和 YOLO 標註

---

### 步驟 2: 訓練（2-3 天）
```cmd
cd detection\yolo_detection
python train_yolov7_preprocessed.py ^
    --data_dir ../../datasets/preprocessed_yolo ^
    --epochs 120 ^
    --batch_size 24 ^
    --use_medical_modules ^
    --use_ema ^
    --mixed_precision
```

**速度對比**:
| 方法 | 秒/batch | Epoch 時間 | 120 Epochs |
|------|----------|-----------|------------|
| 原始 DICOM (`train_yolov7_medical.py`) | 4.2 秒 | ~6-7 小時 | 30-35 天 |
| 預處理 PNG (`train_yolov7_preprocessed.py`) | 0.5 秒 | ~40 分鐘 | **2-3 天** ✅ |

---

## 📊 模組關係圖

```
原始 DICOM 訓練流程:
┌─────────────────────┐
│ train_yolov7_       │
│   medical.py        │ (訓練腳本)
└──────────┬──────────┘
           │ 使用
           ↓
┌─────────────────────┐
│ CTDetectionDataset  │ (讀取 DICOM + XML)
└──────────┬──────────┘
           │ 包裝
           ↓
┌─────────────────────┐
│ YOLOv7Medical       │ (HU windowing + CLAHE + 增強)
│   Dataset           │
└──────────┬──────────┘
           │
           ↓
    訓練速度慢 ❌


預處理資料訓練流程:
┌─────────────────────┐
│ train_yolov7_       │
│   preprocessed.py   │ (新訓練腳本)
└──────────┬──────────┘
           │ 使用
           ↓
┌─────────────────────┐
│ Preprocessed        │ (讀取 PNG + TXT)
│   YOLODataset       │ (新 Dataset)
└──────────┬──────────┘
           │ 包裝
           ↓
┌─────────────────────┐
│ YOLOv7Medical       │ (只增強，預處理已完成)
│   Dataset           │
└──────────┬──────────┘
           │
           ↓
    訓練速度快 ✅
```

---

## ❓ 常見問題

### Q1: 為什麼不能用 `train_yolov7_medical.py` 訓練預處理資料？
**A**: 因為它內部使用 `CTDetectionDataset`，只能讀取 DICOM + XML。預處理資料是 PNG + TXT，格式不同。

### Q2: `yolov7_dataset.py` 的作用是什麼？
**A**: 它是一個**包裝器**，在任何 Dataset 外面添加預處理和增強功能。可以包裝 `CTDetectionDataset`（原始 DICOM）或 `PreprocessedYOLODataset`（預處理 PNG）。

### Q3: 預處理後還需要增強嗎？
**A**: 看情況：
- **不需要增強**: 資料已經預處理，直接訓練最快
- **需要增強**: 可以在 `train_yolov7_preprocessed.py` 中啟用 `--enable_augmentation`

### Q4: 預處理資料可以用原始訓練腳本嗎？
**A**: ❌ 不行。必須使用 `train_yolov7_preprocessed.py`。

### Q5: 原始 DICOM 資料可以用新訓練腳本嗎？
**A**: ❌ 不行。`train_yolov7_preprocessed.py` 只能用於預處理資料。

---

## 📝 總結

| 檔案 | 用途 | 輸入 | 輸出 |
|------|------|------|------|
| `preprocess_for_yolo.py` | 預處理 | DICOM + XML | PNG + YOLO TXT |
| `preprocessed_dataset.py` | 載入預處理資料 | PNG + TXT | Tensor |
| `yolov7_dataset.py` | 增強包裝器 | 任何 Dataset | 增強後 Tensor |
| `train_yolov7_medical.py` | 原始訓練腳本 | DICOM + XML | 訓練模型 |
| `train_yolov7_preprocessed.py` | 預處理訓練腳本 | PNG + TXT | 訓練模型 |

**推薦流程**:
1. ✅ 執行 `preprocess_for_yolo.py` 預處理資料（一次性）
2. ✅ 使用 `train_yolov7_preprocessed.py` 訓練（快 8-10 倍）
3. ✅ 享受 2-3 天完成 120 epochs！
