# 🚀 YOLOv11 快速開始指南

> **最後更新**: 2025-10-17  
> **適合對象**: 新手用戶，想要快速開始訓練

---

## ✅ 數據集驗證結果

```
Total Patients: 200
Total Images: 146,236
Matched Pairs: 146,236
Errors: 0
Warnings: 0

✅ Dataset validation PASSED!
```

您的數據集已經完美準備好，可以開始訓練了！

---

## 📦 可用的工具和腳本

### 核心腳本

| 文件 | 用途 | 說明 |
|------|------|------|
| `train_yolo_direct.py` | ⭐ 主訓練腳本 | 專為 YOLO 格式數據集設計 |
| `validate_dataset.py` | 數據集驗證 | 檢查數據集完整性 |
| `validate_annotations.py` | 標註驗證 | 驗證標註格式和範圍 |
| `test_environment.py` | 環境檢查 | 檢查 Python 和依賴 |
| `check_gpu.py` | GPU 檢查 | 快速檢查 GPU 狀態 |
| `monitor_gpu.py` | GPU 監控 | 實時監控 GPU 使用 |

### 文檔

| 文件 | 內容 |
|------|------|
| [README.md](README.md) | 主文檔和完整指南 |
| [docs/guides/TRAIN_DIRECT_README.md](docs/guides/TRAIN_DIRECT_README.md) | 詳細訓練參數說明 |
| [docs/guides/CONFIG_COMPARISON.md](docs/guides/CONFIG_COMPARISON.md) | 配置對比和選擇 |

---

## 🚀 快速開始（3 步驟）

### 步驟 1: 驗證數據集

**CMD 命令：**
```cmd
python validate_dataset.py --data_dir ..\..\datasets\splited_dataset\train
```

**預期輸出：**
```
✅ Dataset validation PASSED!
Total Patients: 200
Total Images: 146,236
Matched Pairs: 146,236
Errors: 0
```

---

### 步驟 2: 開始訓練

選擇一個適合您的配置：

#### 方法 1：標準配置（推薦新手）⭐

```cmd
python train_yolo_direct.py --data_dir ..\..\datasets\splited_dataset\train --epochs 200 --batch_size 16 --model_size m --val_ratio 0.2
```

**預期時間：** ~8-12 小時（RTX 3090）

#### 方法 2：快速測試（驗證流程）

```cmd
python train_yolo_direct.py --data_dir ..\..\datasets\splited_dataset\train --epochs 50 --batch_size 32 --model_size n --val_ratio 0.3
```

**預期時間：** ~2-3 小時

#### 方法 3：高精度配置（追求最佳效果）

```cmd
python train_yolo_direct.py --data_dir ..\..\datasets\splited_dataset\train --epochs 300 --batch_size 8 --model_size l --imgsz 800 --val_ratio 0.2 --lr 0.0005 --mosaic 1.0 --mixup 0.15 --scale 0.7
```

**預期時間：** ~20-30 小時

---

### 步驟 3: 查看結果

**CMD 命令：**
```cmd
rem 訓練完成後，進入輸出目錄
cd yolo_runs\train_YYYYMMDD_HHMMSS\training\yolo11m_*

rem 查看訓練曲線
start results.png

rem 查看混淆矩陣
start confusion_matrix.png

rem 查看 PR 曲線
start BoxPR_curve.png
```

**使用最佳模型：**
```cmd
python -c "from ultralytics import YOLO; model = YOLO('weights/best.pt'); print(model.val())"
```

---

## 📊 您的數據集統計

```
📁 datasets/splited_dataset/train/
├── 200 患者目錄
├── 146,236 張圖片
├── 146,236 個標籤文件
└── 100% 配對成功率 ✅
```

**訓練集切分（預設 val_ratio=0.2）：**
- 訓練集：160 患者 (~117,000 張圖片)
- 驗證集：40 患者 (~29,000 張圖片)

---

## ⚙️ 常用參數說明

| 參數 | 說明 | 推薦值 |
|------|------|--------|
| `--data_dir` | 數據集目錄 | `..\..\datasets\splited_dataset\train` |
| `--epochs` | 訓練輪數 | 200-300 |
| `--batch_size` | 批次大小 | 16（根據 GPU 調整） |
| `--model_size` | 模型大小 | `m`（推薦）或 `l` |
| `--imgsz` | 圖片尺寸 | 640 或 800 |
| `--val_ratio` | 驗證集比例 | 0.2 |
| `--lr` | 學習率 | 0.001 |
| `--optimizer` | 優化器 | `AdamW` |
| `--mosaic` | Mosaic 增強 | 1.0 |
| `--mixup` | Mixup 增強 | 0.0-0.15 |
| `--fliplr` | 水平翻轉 | 0.5 |

**查看完整參數列表：**
```cmd
python train_yolo_direct.py --help
```

---

## 🎯 模型選擇指南

| 模型 | 參數量 | 速度 | 精度 | 適用場景 |
|------|--------|------|------|----------|
| **n** (nano) | 2.6M | 最快 | 較低 | 快速測試、資源受限 |
| **s** (small) | 9.4M | 快 | 中等 | 輕量部署 |
| **m** (medium) | 20.1M | 中等 | 良好 | **推薦首選** ⭐ |
| **l** (large) | 25.3M | 較慢 | 優秀 | 高精度要求 |
| **x** (xlarge) | 54.2M | 最慢 | 最佳 | 競賽、研究 |

---

## 🔧 常見問題

### Q1: 訓練時出現 CUDA Out of Memory

**解決方案：**
```cmd
rem 選項 1: 減小批次大小
python train_yolo_direct.py ... --batch_size 8

rem 選項 2: 減小圖片尺寸
python train_yolo_direct.py ... --imgsz 512

rem 選項 3: 使用更小的模型
python train_yolo_direct.py ... --model_size s
```

### Q2: 如何檢查 GPU 狀態？

**實時監控：**
```cmd
python monitor_gpu.py
```

**快速檢查：**
```cmd
python check_gpu.py
```

### Q3: 訓練速度很慢怎麼辦？

**解決方案：**
```cmd
rem 增加工作線程
--workers 16

rem 減少數據增強
--mosaic 0.5 --mixup 0.0

rem 使用更小的模型
--model_size s
```

### Q4: 如何暫停和繼續訓練？

YOLOv11 會自動保存最新的模型（`last.pt`），您可以使用以下命令繼續訓練：

```cmd
python -c "from ultralytics import YOLO; model = YOLO('yolo_runs/train_*/training/yolo11m_*/weights/last.pt'); model.train(resume=True)"
```

### Q5: 訓練完成後如何使用模型？

```python
from ultralytics import YOLO

# 載入最佳模型
model = YOLO('yolo_runs/train_*/training/yolo11m_*/weights/best.pt')

# 推理單張圖片
results = model.predict('test_image.png', conf=0.25)

# 批量推理
results = model.predict('test_folder/', save=True)

# 驗證模型
metrics = model.val()

# 導出為 ONNX
model.export(format='onnx')
```

---

## 🛠️ 環境檢查

### 檢查 Python 和依賴

```cmd
python test_environment.py
```

**預期輸出：**
```
✅ Python version: 3.x.x
✅ PyTorch: x.x.x
✅ Ultralytics: x.x.x
✅ CUDA available: True
✅ GPU: NVIDIA RTX 3090
```

### 檢查數據集

```cmd
python validate_dataset.py --data_dir ..\..\datasets\splited_dataset\train --verbose
```

### 檢查標註

```cmd
python validate_annotations.py --data_dir ..\..\datasets\splited_dataset\train
```

---

## 📈 預期訓練結果

基於 146,236 張 CT 圖片的預期性能：

| 配置 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 訓練時間 |
|------|---------|--------------|-----------|--------|----------|
| YOLOv11-n (快速測試) | 0.65-0.75 | 0.45-0.55 | 0.70-0.85 | 0.65-0.80 | ~3h |
| YOLOv11-m (標準) | 0.75-0.85 | 0.55-0.65 | 0.75-0.90 | 0.70-0.85 | ~10h |
| YOLOv11-l (高精度) | 0.78-0.88 | 0.58-0.68 | 0.78-0.92 | 0.73-0.88 | ~25h |

*基於 RTX 3090 GPU*

---

## 📂 訓練輸出結構

```
yolo_runs/
└── train_YYYYMMDD_HHMMSS/
    ├── dataset_YYYYMMDD_HHMMSS/      # 數據集
    │   ├── images/
    │   │   ├── train/                # 訓練圖片
    │   │   └── val/                  # 驗證圖片
    │   ├── labels/
    │   │   ├── train/                # 訓練標籤
    │   │   └── val/                  # 驗證標籤
    │   └── dataset.yaml              # YOLO 配置
    │
    ├── training/
    │   └── yolo11m_YYYYMMDD_HHMMSS/
    │       ├── weights/
    │       │   ├── best.pt           # ⭐ 最佳模型
    │       │   └── last.pt           # 最新模型
    │       ├── results.png           # 訓練曲線
    │       ├── confusion_matrix.png  # 混淆矩陣
    │       ├── BoxPR_curve.png       # PR 曲線
    │       └── results.csv           # 詳細結果
    │
    └── logs/
        └── training_YYYYMMDD_HHMMSS.log  # 訓練日誌
```

---

## 🎓 訓練流程建議

1. **環境檢查** → `python test_environment.py`
2. **數據驗證** → `python validate_dataset.py`
3. **快速測試** → 使用 `model_size=n, epochs=50` 驗證流程
4. **標準訓練** → 使用推薦配置訓練
5. **監控訓練** → 使用 `python monitor_gpu.py`
6. **查看結果** → 檢查訓練曲線和指標
7. **調整參數** → 根據結果調整配置
8. **最終訓練** → 使用最佳配置長時間訓練

---

## 📚 延伸閱讀

- [完整訓練文檔](docs/guides/TRAIN_DIRECT_README.md) - 所有參數詳細說明
- [配置對比表](docs/guides/CONFIG_COMPARISON.md) - 不同配置的對比
- [訓練指南](docs/guides/README_TRAINING.md) - 完整訓練流程
- [指標解釋](docs/references/VALIDATION_METRICS_GUIDE.md) - 評估指標說明

---

## 📞 獲取幫助

如有問題：

1. 查看 [README.md](README.md) 的常見問題部分
2. 運行 `python test_environment.py` 檢查環境
3. 查看訓練日誌 `yolo_runs/train_*/logs/training_*.log`
4. 參考文檔目錄 `docs/`

---

**準備好了嗎？開始您的第一次訓練吧！🚀**

```cmd
python train_yolo_direct.py --data_dir ..\..\datasets\splited_dataset\train --epochs 200 --batch_size 16 --model_size m --val_ratio 0.2
```

---

*最後更新: 2025-10-17*  
*數據集狀態: ✅ 已驗證（200 患者，146,236 張圖片）*
