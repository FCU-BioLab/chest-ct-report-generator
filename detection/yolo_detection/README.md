# YOLOv11 胸腔 CT 病灶檢測系統

> **最後更新**: 2025-10-17  
> **版本**: 3.0 - YOLOv11 直接訓練版  
> **狀態**: ✅ 數據集已驗證（200 患者，146,236 張圖片）

---

## 🎯 快速導航

| 文檔 | 描述 | 適合對象 |
|------|------|----------|
| [🚀 快速開始指南](QUICK_START_GUIDE.md) | 3 步開始訓練 | **新手必讀** ⭐ |
| [📖 完整訓練文檔](docs/guides/TRAIN_DIRECT_README.md) | 詳細參數說明 | 進階用戶 |
| [📊 配置對比表](docs/guides/CONFIG_COMPARISON.md) | 選擇最佳配置 | 所有用戶 |
| [✅ 配置檢查清單](docs/references/SETUP_COMPLETE.md) | 環境檢查清單 | 訓練前必讀 |
| [📋 訓練指南](docs/guides/README_TRAINING.md) | 完整訓練流程 | 所有用戶 |

---

## 📦 可用的訓練腳本

### train_yolo_direct.py ⭐ **推薦使用**

**適用場景：** 數據已經是 YOLO 格式（當前數據集狀態）

**CMD 命令：**
```cmd
python train_yolo_direct.py --data_dir ..\..\datasets\splited_dataset\train --epochs 200 --batch_size 16 --model_size m --val_ratio 0.2
```

**特點：**
- ✅ 專為已預處理的 YOLO 數據集設計
- ✅ 自動基於患者 ID 切分 train/val（避免數據洩漏）
- ✅ 簡潔高效（600+ 行代碼）
- ✅ 完整日誌和結果記錄
- ✅ 支持 35+ 可調參數

**當前數據集：** ✅ 完全適用

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

### 步驟 2: 開始訓練

**基本訓練（推薦）⭐**
```cmd
python train_yolo_direct.py --data_dir ..\..\datasets\splited_dataset\train --epochs 200 --batch_size 16 --model_size m --val_ratio 0.2
```

**高精度訓練**
```cmd
python train_yolo_direct.py --data_dir ..\..\datasets\splited_dataset\train --epochs 300 --batch_size 8 --model_size l --imgsz 800 --val_ratio 0.2 --lr 0.0005 --mosaic 1.0 --mixup 0.15
```

**快速測試**
```cmd
python train_yolo_direct.py --data_dir ..\..\datasets\splited_dataset\train --epochs 50 --batch_size 32 --model_size n --val_ratio 0.3
```

### 步驟 3: 查看結果

**CMD 命令：**
```cmd
rem 訓練完成後，檢查輸出目錄
cd yolo_runs\train_YYYYMMDD_HHMMSS\training\yolo11m_*

rem 查看訓練曲線
rem results.png, confusion_matrix.png, BoxPR_curve.png

rem 使用最佳模型驗證
python -c "from ultralytics import YOLO; model = YOLO('weights/best.pt'); model.val()"
```

---

## 📊 訓練配置建議

### 標準配置（推薦）⭐

適合第一次訓練，建立性能基準：

```cmd
python train_yolo_direct.py --data_dir ..\..\datasets\splited_dataset\train --epochs 200 --batch_size 16 --model_size m --imgsz 640 --val_ratio 0.2 --lr 0.001 --optimizer AdamW --warmup_epochs 5 --mosaic 1.0 --fliplr 0.5
```

**預期結果：**
- mAP@0.5: 0.75-0.85
- 訓練時間: ~8-12 小時（RTX 3090）

### 高精度配置

適合小病灶檢測，追求最高精度：

```cmd
python train_yolo_direct.py --data_dir ..\..\datasets\splited_dataset\train --epochs 300 --batch_size 8 --model_size l --imgsz 800 --val_ratio 0.2 --lr 0.0005 --optimizer AdamW --warmup_epochs 5 --mosaic 1.0 --mixup 0.15 --scale 0.7
```

**預期結果：**
- mAP@0.5: 0.78-0.88
- 訓練時間: ~20-30 小時

### 快速測試配置

適合參數測試和快速驗證：

```cmd
python train_yolo_direct.py --data_dir ..\..\datasets\splited_dataset\train --epochs 50 --batch_size 32 --model_size n --val_ratio 0.3
```

**預期結果：**
- mAP@0.5: 0.65-0.75
- 訓練時間: ~2-3 小時

---

## 📁 文件結構

### 核心腳本

| 文件 | 用途 | 行數 |
|------|------|------|
| `train_yolo_direct.py` | ⭐ 主訓練腳本（YOLO 格式） | 600+ |
| `validate_dataset.py` | 數據集驗證工具 | 200+ |
| `validate_annotations.py` | 標註驗證工具 | 500+ |
| `test_environment.py` | 環境檢查工具 | 100+ |
| `check_gpu.py` | GPU 狀態檢查 | 50+ |
| `monitor_gpu.py` | GPU 實時監控 | 100+ |

### 輔助腳本

| 文件 | 用途 |
|------|------|
| `dataset_filter.py` | 數據集過濾工具 |
| `preprocessed_dataset.py` | 預處理數據集類 |
| `test_yolov11.py` | YOLOv11 測試腳本 |

### 文檔

| 文件 | 內容 |
|------|------|
| `QUICK_START_GUIDE.md` | 快速開始指南（新手必讀）|
| `docs/guides/README_TRAINING.md` | 訓練腳本索引和選擇指南 |
| `docs/guides/TRAIN_DIRECT_README.md` | 完整訓練文檔 |
| `docs/guides/CONFIG_COMPARISON.md` | 配置對比和選擇 |
| `docs/references/SETUP_COMPLETE.md` | 配置完成檢查清單 |
| `docs/references/VALIDATION_METRICS_GUIDE.md` | 指標解釋 |
| `docs/references/NEGATIVE_SAMPLE_DETECTION.md` | 負樣本處理說明 |

### 模型文件

| 文件 | 說明 |
|------|------|
| `yolo11n.pt` | YOLOv11 nano 預訓練模型 |
| `yolo11s.pt` | YOLOv11 small 預訓練模型 |
| `yolo11m.pt` | YOLOv11 medium 預訓練模型 |
| `models/yolo11_custom.yaml` | 自定義模型配置 |
| `models/yolo11_custom_ct.yaml` | CT 專用模型配置 |

---

## 📈 訓練輸出

訓練完成後的目錄結構：

```
yolo_runs/
├── train_YYYYMMDD_HHMMSS/            # 訓練運行目錄
│   ├── dataset_YYYYMMDD_HHMMSS/      # 準備好的數據集
│   │   ├── images/train/             # 訓練圖片
│   │   ├── images/val/               # 驗證圖片
│   │   ├── labels/train/             # 訓練標籤
│   │   ├── labels/val/               # 驗證標籤
│   │   ├── dataset.yaml              # YOLO 配置
│   │   └── patient_split.json        # 患者切分記錄
│   │
│   ├── training/
│   │   └── yolo11m_YYYYMMDD_HHMMSS/
│   │       ├── weights/
│   │       │   ├── best.pt           # ⭐ 最佳模型
│   │       │   └── last.pt           # 最後一輪
│   │       ├── results.png           # 訓練曲線
│   │       ├── confusion_matrix.png  # 混淆矩陣
│   │       ├── BoxPR_curve.png       # PR 曲線
│   │       └── ...
│   │
│   └── logs/
│       └── training_YYYYMMDD_HHMMSS.log  # 訓練日誌
│
└── cache/                            # 數據集緩存
```

---

## 🎯 模型選擇指南

| 模型 | 參數量 | 速度 | 準確度 | 適用場景 |
|------|--------|------|--------|----------|
| **n** (nano) | 2.6M | 最快 | 較低 | 快速原型、資源受限 |
| **s** (small) | 9.4M | 快 | 中等 | 輕量部署 |
| **m** (medium) | 20.1M | 中等 | 良好 | **推薦用於 CT 檢測** ⭐ |
| **l** (large) | 25.3M | 較慢 | 優秀 | 高精度要求 |
| **x** (xlarge) | 54.2M | 最慢 | 最佳 | 競賽、學術研究 |

---

## 🔧 常見問題

### Q1: CUDA Out of Memory

**解決方案：**
```cmd
rem 減小批次大小
--batch_size 8

rem 或減小圖片尺寸
--imgsz 512

rem 或使用更小的模型
--model_size s
```

### Q2: 訓練速度慢

**解決方案：**
```cmd
rem 增加工作線程
--workers 16

rem 減少數據增強
--mosaic 0.5 --mixup 0.0
```

### Q3: 驗證集指標不穩定

**解決方案：**
```cmd
rem 增加驗證集比例
--val_ratio 0.3

rem 或固定隨機種子
--seed 42
```

### Q4: 如何使用訓練好的模型？

```python
from ultralytics import YOLO

# 載入最佳模型
model = YOLO('yolo_runs/train_*/training/yolo11m_*/weights/best.pt')

# 推理
results = model.predict('test_image.png', conf=0.25)

# 驗證
metrics = model.val()

# 導出
model.export(format='onnx')
```

### Q5: 如何檢查 GPU 狀態？

**實時監控：**
```cmd
python monitor_gpu.py
```

**快速檢查：**
```cmd
python check_gpu.py
```

---

## 📊 性能基準

基於 146,236 張胸部 CT 圖片的預期性能：

| 配置 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 訓練時間 |
|------|---------|--------------|-----------|--------|----------|
| YOLOv11-n (快速) | 0.65-0.75 | 0.45-0.55 | 0.70-0.85 | 0.65-0.80 | ~3h |
| YOLOv11-s (標準) | 0.70-0.80 | 0.50-0.60 | 0.73-0.88 | 0.68-0.83 | ~6h |
| YOLOv11-m (推薦) | 0.75-0.85 | 0.55-0.65 | 0.75-0.90 | 0.70-0.85 | ~10h |
| YOLOv11-l (高精度) | 0.78-0.88 | 0.58-0.68 | 0.78-0.92 | 0.73-0.88 | ~25h |

*基於 RTX 3090 GPU*

---

## 🛠️ 工具腳本使用

### 驗證數據集

```cmd
python validate_dataset.py --data_dir ..\..\datasets\splited_dataset\train --verbose
```

### 驗證標註

```cmd
python validate_annotations.py --data_dir ..\..\datasets\splited_dataset\train
```

### 測試環境

```cmd
python test_environment.py
```

### 監控 GPU

```cmd
rem 實時監控（每 2 秒刷新）
python monitor_gpu.py

rem 單次檢查
python check_gpu.py
```

### 測試 YOLOv11

```cmd
python test_yolov11.py --model yolo11m.pt --source ..\..\datasets\splited_dataset\train\A0001
```

---

## 📚 相關文檔

### 快速入門
- [🚀 快速開始指南](QUICK_START_GUIDE.md) - 3 步開始訓練
- [✅ 配置完成檢查](docs/references/SETUP_COMPLETE.md) - 訓練前準備

### 詳細文檔
- [📖 完整訓練文檔](docs/guides/TRAIN_DIRECT_README.md) - 所有參數說明
- [📊 配置對比表](docs/guides/CONFIG_COMPARISON.md) - 選擇最佳配置
- [📋 訓練腳本索引](docs/guides/README_TRAINING.md) - 腳本選擇指南

### 進階主題
- [訓練技巧指南](docs/guides/TRAINING_GUIDE.md) - 訓練技巧
- [訓練方法對比](docs/guides/TRAINING_COMPARISON.md) - 方法對比
- [指標解釋](docs/references/VALIDATION_METRICS_GUIDE.md) - 指標解釋
- [負樣本處理](docs/references/NEGATIVE_SAMPLE_DETECTION.md) - 負樣本處理說明

---

## 🎓 訓練最佳實踐

1. **第一次訓練：** 使用標準配置建立基準
2. **驗證環境：** 運行 `python test_environment.py`
3. **驗證數據：** 運行 `python validate_dataset.py`
4. **開始訓練：** 使用推薦配置
5. **監控訓練：** 使用 `python monitor_gpu.py`
6. **評估結果：** 查看訓練曲線和指標
7. **調整參數：** 根據結果調整增強和學習率
8. **最終訓練：** 使用最佳配置長時間訓練
9. **模型評估：** 在獨立測試集上評估
10. **模型導出：** 導出為 ONNX/TensorRT 用於部署

---

## 📦 依賴要求

```cmd
pip install ultralytics torch torchvision
```

**驗證安裝：**
```cmd
python test_environment.py
```

**檢查版本：**
```cmd
python -c "import torch; import ultralytics; print(f'PyTorch: {torch.__version__}'); print(f'Ultralytics: {ultralytics.__version__}')"
```

---

## 🔗 外部資源

- [Ultralytics YOLOv11 官方文檔](https://docs.ultralytics.com/)
- [YOLOv11 GitHub](https://github.com/ultralytics/ultralytics)
- [訓練技巧指南](https://docs.ultralytics.com/guides/model-training-tips/)
- [數據集格式](https://docs.ultralytics.com/datasets/detect/)

---

## 📞 支持

如有問題：
1. 查看 [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)
2. 運行 `python test_environment.py`
3. 檢查訓練日誌文件 `yolo_runs/train_*/logs/training_*.log`
4. 參考常見問題部分
5. 查看文檔目錄 `docs/`

---

## 📝 更新日誌

### 2025-10-17
- ✅ 更新 README 格式，所有命令改為 CMD 格式
- ✅ 移除不存在的腳本引用（train_yolov11.py, train_yolo_optimize.py）
- ✅ 更新文檔路徑引用
- ✅ 新增工具腳本使用說明

### 2025-10-13
- ✅ 數據集驗證完成（200 患者，146,236 張圖片）
- ✅ 創建 train_yolo_direct.py 主訓練腳本
- ✅ 完整文檔整理

---

**準備就緒！選擇一個配置，開始訓練吧！🚀**

---

*最後更新: 2025-10-17*  
*數據集狀態: ✅ 已驗證（200 患者，146,236 張圖片）*  
*訓練腳本: train_yolo_direct.py v1.0*
