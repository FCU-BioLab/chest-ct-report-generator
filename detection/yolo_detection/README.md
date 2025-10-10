# YOLOv7 胸腔 CT 病灶檢測

## 📁 專案結構

### **核心 Python 檔案**
- `train_yolov7_medical.py` - ⭐ 主訓練腳本（已整合所有增強功能）
- `yolov7_model.py` - YOLOv7 模型定義
- `yolov7_dataset.py` - 資料集載入與前處理
- `yolov7_utils.py` - 訓練工具（含 Focal Loss）
- `yolov7_augmentations.py` - 進階資料增強（Mosaic/MixUp/Copy-Paste）
- `yolov7_eval_visualizer.py` - 評估視覺化工具
- `validate_annotations.py` - 註解驗證工具

### **其他訓練/測試腳本**
- `train_yolov11.py` - YOLOv11 訓練（對比用）
- `train_yolo_optimize.py` - 優化訓練腳本
- `test_yolov11.py` - YOLOv11 推論測試（測試訓練結果）

### **說明文件**
- `QUICK_REFERENCE.md` - 快速參考（推薦閱讀）
- `ENHANCED_TRAINING_GUIDE.md` - 完整訓練指南
- `ENHANCEMENT_README.md` - 功能說明
- `IMPLEMENTATION_SUMMARY.md` - 實作細節
- `INTEGRATION_CHECKLIST.md` - 整合檢查清單

### **模型配置**
- `models/yolov7_medical.yaml` - 醫療模組版本
- `models/yolov7_baseline.yaml` - 基礎版本
- `models/custom_layers.py` - 自定義層

---

## 🚀 快速開始

### **1. 快速測試（10 輪）**
```cmd
python train_yolov7_medical.py --data_dir ../../datasets/splited_dataset --epochs 10 --batch_size 4 --workers 2 --enable_augmentation --positive_oversample --use_focal_loss --visualize_predictions
```

### **2. 推薦配置（完整訓練）**
```cmd
python train_yolov7_medical.py --data_dir ../../datasets/splited_dataset --epochs 120 --batch_size 8 --accumulation_steps 4 --enable_augmentation --mosaic_prob 0.5 --mixup_prob 0.3 --copy_paste_prob 0.3 --positive_oversample --positive_ratio 0.7 --use_focal_loss --cls_loss_gain 1.8 --visualize_predictions --workers 4 --mixed_precision --use_ema
```

### **3. 基礎訓練（對照組）**
```cmd
python train_yolov7_medical.py --data_dir ../../datasets/splited_dataset --epochs 120 --batch_size 16 --workers 4 --mixed_precision --use_ema
```

---

## 🎯 增強功能

### ✅ 已整合功能
1. **梯度累積** - 有效批次大小 = batch_size × accumulation_steps
2. **進階增強** - Mosaic, MixUp, Copy-Paste
3. **正樣本過採樣** - 自動平衡正負樣本
4. **Focal Loss** - 解決類別不平衡
5. **每輪視覺化** - TP/FP/FN 自動視覺化
6. **醫療預處理** - HU windowing, CLAHE

### 📊 預期效果

| 階段 | Baseline mAP@0.5 | Enhanced mAP@0.5 | 改善 |
|------|------------------|------------------|------|
| Epoch 10 | < 0.01 | 0.05-0.15 | 5-15× |
| Epoch 50 | 0.05-0.15 | 0.25-0.40 | 3-5× |
| Epoch 120 | 0.15-0.30 | 0.45-0.65 | 2-3× |

---

## 📖 參數說明

### **關鍵參數**
| 參數 | 說明 | 推薦值 |
|------|------|--------|
| `--batch_size` | 批次大小 | `8` |
| `--accumulation_steps` | 梯度累積步數 | `4` |
| `--enable_augmentation` | 啟用進階增強 | ✅ |
| `--mosaic_prob` | Mosaic 機率 | `0.5` |
| `--mixup_prob` | MixUp 機率 | `0.3` |
| `--copy_paste_prob` | Copy-Paste 機率 | `0.3` |
| `--positive_oversample` | 正樣本過採樣 | ✅ |
| `--positive_ratio` | 目標正樣本比例 | `0.7` |
| `--use_focal_loss` | 使用 Focal Loss | ✅ |
| `--cls_loss_gain` | 分類損失權重 | `1.8` |
| `--visualize_predictions` | 啟用視覺化 | ✅ |

### **完整參數列表**
請參考 `ENHANCED_TRAINING_GUIDE.md`

---

## 📁 輸出結構

```
yolov7_models/run_<timestamp>/
├── weights/
│   ├── best.pt              # 最佳模型
│   ├── last.pt              # 最後一輪
│   └── epoch_*.pt           # 檢查點
├── visualizations/          # 視覺化結果
│   ├── epoch_1/
│   ├── epoch_12/
│   └── ...
├── training_history.json    # 訓練歷史
└── summary.json             # 訓練摘要
```

---

## 🔧 安裝依賴

所有依賴已整合到專案根目錄的 `requirements.txt`：

```cmd
cd e:\GitHub\chest-ct-report-generator
pip install -r requirements.txt
```

或手動安裝 YOLOv7 核心依賴：
```cmd
pip install torch torchvision albumentations opencv-python-headless matplotlib pyyaml tqdm pydicom SimpleITK
```

---

## 🐛 常見問題

### Q: OOM (記憶體不足)
```cmd
--batch_size 4 --accumulation_steps 8
```

### Q: 視覺化未生成
```cmd
pip install albumentations opencv-python-headless matplotlib
```

### Q: Focal Loss 未啟用
確認同時設定：
```cmd
--use_focal_loss --focal_alpha 0.25 --focal_gamma 2.0
```

### Q: 增強未生效
必須同時設定：
```cmd
--enable_augmentation --mosaic_prob 0.5 --mixup_prob 0.3
```

---

## 📚 詳細文檔

- **QUICK_REFERENCE.md** - 快速參考卡（推薦閱讀）
- **ENHANCED_TRAINING_GUIDE.md** - 完整參數說明
- **ENHANCEMENT_README.md** - 功能詳細說明
- **INTEGRATION_CHECKLIST.md** - 整合檢查清單

---

## 🎓 使用建議

1. **首次使用** - 先執行 10 輪快速測試，確認環境正常
2. **完整訓練** - 使用推薦配置進行 120 輪訓練
3. **效果對比** - 同時執行基礎訓練，對比增強效果
4. **監控訓練** - 檢查視覺化結果，調整參數

---

**版本**: 1.0 - 完整整合版  
**狀態**: ✅ Production Ready  
**最後更新**: 2024-10-10
