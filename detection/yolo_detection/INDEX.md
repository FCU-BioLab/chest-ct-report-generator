# YOLOv7 Medical Detection - 檔案導覽

## 📖 快速導覽

根據您的需求選擇閱讀：

### 🚀 我想快速開始
→ 閱讀 **[QUICKSTART.md](QUICKSTART.md)**
- 3 步驟開始訓練
- 常用配置範例
- 故障排除指南

### 📚 我想了解完整功能
→ 閱讀 **[README_YOLOV7.md](README_YOLOV7.md)**
- 專案概述
- 詳細參數說明
- 安裝與使用指南
- 自訂模組教學

### 🏗️ 我想理解架構設計
→ 閱讀 **[ARCHITECTURE.md](ARCHITECTURE.md)**
- 架構流程圖
- 模組插入位置
- 視覺化說明
- 效能分析

### 📊 我想查看專案摘要
→ 閱讀 **[SUMMARY.md](SUMMARY.md)**
- 完成檔案清單
- 功能達成度
- 與原始腳本對比
- 下一步建議

---

## 📁 核心檔案說明

### 程式碼檔案

| 檔案 | 行數 | 用途 | 重要性 |
|------|------|------|--------|
| **train_yolov7_medical.py** | 717 | 主訓練腳本 | ⭐⭐⭐⭐⭐ |
| **yolov7_model.py** | 384 | 模型載入與建構 | ⭐⭐⭐⭐⭐ |
| **yolov7_dataset.py** | 296 | 資料集與預處理 | ⭐⭐⭐⭐⭐ |
| **yolov7_utils.py** | 423 | 訓練工具函式 | ⭐⭐⭐⭐⭐ |
| **models/custom_layers.py** | 418 | 醫學模組實作 | ⭐⭐⭐⭐⭐ |
| **setup_yolov7.py** | 265 | 環境驗證 | ⭐⭐⭐⭐ |

### 配置檔案

| 檔案 | 用途 | 使用時機 |
|------|------|----------|
| **models/yolov7_medical.yaml** | 醫學模組配置 | 預設配置 |
| **models/yolov7_baseline.yaml** | Baseline 配置 | 對照實驗 |
| **requirements_yolov7.txt** | 依賴套件 | 環境設置 |

### 文件檔案

| 檔案 | 內容 | 適合對象 |
|------|------|----------|
| **INDEX.md** | 檔案導覽 (本檔案) | 所有人 |
| **QUICKSTART.md** | 快速開始指南 | 初次使用者 |
| **README_YOLOV7.md** | 完整文件與使用說明 | 所有使用者 |
| **ARCHITECTURE.md** | 架構視覺化說明 | 開發者 |
| **SUMMARY.md** | 專案完成摘要 | 管理者/審查者 |

---

## 🎯 使用場景導引

### 場景 1: 我是新手，第一次使用
```
1. 閱讀 QUICKSTART.md
2. 執行 python setup_yolov7.py
3. 執行基本訓練命令
4. 如有問題，查閱 README_YOLOV7.md
```

### 場景 2: 我要進行實驗對比
```
1. 閱讀 QUICKSTART.md 的「比較實驗」章節
2. 執行 Baseline 訓練
3. 執行 Medical 訓練
4. 比較結果
```

### 場景 3: 我要自訂模組
```
1. 閱讀 README_YOLOV7.md 的「自訂模組」章節
2. 參考 models/custom_layers.py
3. 修改 models/yolov7_medical.yaml
4. 測試訓練
```

### 場景 4: 我要部署到生產環境
```
1. 閱讀 ARCHITECTURE.md 理解架構
2. 執行完整訓練
3. 評估效能指標
4. 優化推論速度
```

### 場景 5: 我遇到問題
```
1. 查閱 QUICKSTART.md 的「故障排除」
2. 執行 python setup_yolov7.py 檢查環境
3. 查閱 README_YOLOV7.md 的「故障排除」
4. 如仍無法解決，提交 Issue
```

---

## 🔍 程式碼架構概覽

```
YOLOv7 Medical Training
│
├─ Core Training (train_yolov7_medical.py)
│   ├─ TrainingConfig (配置管理)
│   ├─ train_one_epoch() (單輪訓練)
│   ├─ validate() (驗證)
│   └─ train_yolov7() (主流程)
│
├─ Model Architecture (yolov7_model.py)
│   ├─ YOLOv7Model (模型類別)
│   ├─ Basic Layers (Conv, MP, Concat, etc.)
│   └─ load_yolov7_model() (載入函式)
│
├─ Medical Modules (models/custom_layers.py)
│   ├─ CBAM (通道+空間注意力)
│   ├─ SimAM (無參數注意力)
│   ├─ SwinTransformerBlock (視窗注意力)
│   └─ BiFPN (雙向特徵金字塔)
│
├─ Dataset Processing (yolov7_dataset.py)
│   ├─ Medical Preprocessing
│   │   ├─ apply_hu_windowing()
│   │   ├─ apply_clahe()
│   │   └─ apply_percentile_stretch()
│   ├─ YOLOv7MedicalDataset (資料集類別)
│   └─ create_yolov7_dataloader() (載入器)
│
└─ Training Utils (yolov7_utils.py)
    ├─ ComputeLoss (損失計算)
    ├─ ModelEMA (指數移動平均)
    ├─ WarmupCosineSchedule (學習率排程)
    └─ compute_metrics() (評估指標)
```

---

## 📊 功能矩陣

| 功能 | 說明 | 檔案位置 | 狀態 |
|------|------|----------|------|
| HU Windowing | CT 影像視窗化 | yolov7_dataset.py | ✅ |
| CLAHE | 對比度增強 | yolov7_dataset.py | ✅ |
| CBAM | 注意力模組 | models/custom_layers.py | ✅ |
| Swin Transformer | 視窗注意力 | models/custom_layers.py | ✅ |
| BiFPN | 特徵金字塔 | models/custom_layers.py | ✅ |
| SimAM | 輕量注意力 | models/custom_layers.py | ✅ |
| EMA | 指數移動平均 | yolov7_utils.py | ✅ |
| Mixed Precision | 混合精度訓練 | train_yolov7_medical.py | ✅ |
| Multi-GPU | 多 GPU 支援 | train_yolov7_medical.py | ✅ |
| Gradient Clipping | 梯度裁剪 | train_yolov7_medical.py | ✅ |
| Dataset Caching | 資料集快取 | yolov7_dataset.py | ✅ |
| CLI Interface | 命令列介面 | train_yolov7_medical.py | ✅ |

---

## 🎓 學習路徑建議

### 初學者 (1-2 天)
```
Day 1:
  - 閱讀 QUICKSTART.md
  - 執行 setup_yolov7.py
  - 完成第一次訓練

Day 2:
  - 閱讀 README_YOLOV7.md
  - 嘗試不同參數
  - 比較 baseline vs medical
```

### 中級使用者 (3-5 天)
```
Day 1-2:
  - 深入閱讀 README_YOLOV7.md
  - 理解所有參數

Day 3:
  - 閱讀 ARCHITECTURE.md
  - 理解架構設計

Day 4-5:
  - 自訂模組
  - 進行實驗對比
  - 優化效能
```

### 高級開發者 (1 週+)
```
Week 1:
  - 閱讀所有文件
  - 理解所有程式碼
  - 修改核心邏輯

Week 2+:
  - 開發新模組
  - 整合其他技術
  - 優化效能
  - 部署到生產
```

---

## 📞 支援資源

### 文件資源
- **檔案導覽**: INDEX.md (本檔案)
- **快速指南**: QUICKSTART.md
- **完整文件**: README_YOLOV7.md
- **架構說明**: ARCHITECTURE.md
- **專案摘要**: SUMMARY.md

### 程式碼資源
- **範例訓練**: train_yolov7_medical.py
- **環境驗證**: setup_yolov7.py
- **模組測試**: models/custom_layers.py (main block)

### 命令列工具
```bash
# 環境驗證
python setup_yolov7.py

# 查看幫助
python train_yolov7_medical.py --help

# 測試模組
cd models && python custom_layers.py
```

---

## ✅ 快速檢查清單

### 開始訓練前
- [ ] 已安裝所有依賴套件
- [ ] 已執行 `python setup_yolov7.py` 驗證
- [ ] 資料集路徑正確
- [ ] GPU/CUDA 可用 (optional)

### 訓練過程中
- [ ] 查看訓練日誌
- [ ] 監控 GPU 使用率
- [ ] 檢查 loss 下降趨勢
- [ ] 定期查看驗證結果

### 訓練完成後
- [ ] 檢查 best.pt 模型
- [ ] 評估 mAP 指標
- [ ] 比較不同配置結果
- [ ] 保存實驗記錄

---

## 🚀 下一步

選擇您的路徑：

1. **開始訓練** → [QUICKSTART.md](QUICKSTART.md)
2. **深入學習** → [README_YOLOV7.md](README_YOLOV7.md)
3. **理解架構** → [ARCHITECTURE.md](ARCHITECTURE.md)
4. **查看摘要** → [SUMMARY.md](SUMMARY.md)

---

**提示**: 建議按照 QUICKSTART → README_YOLOV7 → ARCHITECTURE → SUMMARY 的順序閱讀，可以獲得最佳學習效果。

**最後更新**: 2025-01-08  
**版本**: 1.0.0  
**狀態**: ✅ 完整實作
