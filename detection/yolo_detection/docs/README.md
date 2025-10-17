# 📁 文檔目錄說明# 📁 文檔目錄說明



> 快速了解 detection/yolo_detection/ 目錄下的文檔組織> 快速了解 yolo_detection/ 目錄下的文檔組織



------



## 📂 目錄結構## 📂 目錄結構



``````

yolo_detection/yolo_detection/

││

├── 📄 README.md                      ← 🏠 主文檔（從這裡開始！）├── 📄 README.md                      ← 🏠 從這裡開始！

├── 🚀 QUICK_START_GUIDE.md           ← 新手快速上手（3步驟）├── 📚 DOCUMENTATION.md                ← 完整文檔索引

│├── 🚀 QUICK_START_GUIDE.md           ← 新手快速上手

├── 📁 docs/                          ← 詳細文檔目錄│

│   │├── 📁 docs/                          ← 詳細文檔目錄

│   ├── 📄 INDEX.md                   ← 文檔索引和導航│   │

│   ││   ├── 📖 guides/                    ← 使用指南（如何使用）

│   ├── 📖 guides/                    ← 使用指南（如何使用）│   │   ├── TRAIN_DIRECT_README.md    # 完整參數說明

│   │   ├── TRAIN_DIRECT_README.md    # 完整參數說明（35+參數）│   │   ├── CONFIG_COMPARISON.md       # 配置選擇對比

│   │   ├── CONFIG_COMPARISON.md      # 配置選擇對比│   │   └── README_TRAINING.md         # 訓練腳本選擇

│   │   ├── README_TRAINING.md        # 訓練腳本選擇│   │

│   │   ├── TRAINING_GUIDE.md         # 訓練技巧和最佳實踐│   ├── 📋 references/                ← 參考資料（深入了解）

│   │   └── TRAINING_COMPARISON.md    # 訓練方法對比│   │   ├── SETUP_COMPLETE.md          # 環境檢查清單

│   ││   │   ├── NEGATIVE_SAMPLE_DETECTION.md  # 負樣本控制

│   ├── 📋 references/                ← 參考資料（深入了解）│   │   ├── VALIDATION_METRICS_GUIDE.md   # 驗證指標解釋

│   │   ├── SETUP_COMPLETE.md         # 環境檢查清單│   │   ├── PREPROCESSED_DATASET_UPDATE.md  # 數據集說明

│   │   ├── NEGATIVE_SAMPLE_DETECTION.md  # 負樣本控制機制│   │   ├── DOCS_UPDATE_REPORT.md      # 文檔更新記錄

│   │   ├── VALIDATION_METRICS_GUIDE.md   # 驗證指標解釋│   │   └── DOCUMENTATION_REORGANIZATION_REPORT.md  # 整理報告

│   │   ├── PREPROCESSED_DATASET_UPDATE.md  # 數據集結構說明│   │

│   │   ├── DOCS_UPDATE_REPORT.md     # 文檔更新記錄│   └── 🗄️ archives/                  ← 歷史文檔（已過時）

│   │   └── DOCUMENTATION_REORGANIZATION_REPORT.md  # 整理報告│       └── DOCS_INDEX.md              # 舊版索引（已棄用）

│   ││

│   └── 🗄️ archives/                  ← 歷史文檔（已過時）├── 🔥 train_yolo_direct.py           ← 主訓練腳本

│       └── DOCS_INDEX.md             # 舊版索引（已棄用）├── ✅ validate_dataset.py             ← 數據集驗證

│├── 🧪 test_environment.py             ← 環境測試

├── 🔥 train_yolo_direct.py           ← ⭐ 主訓練腳本└── ... （其他 Python 文件）

├── ✅ validate_dataset.py            ← 數據集驗證工具```

├── ✅ validate_annotations.py        ← 標註驗證工具

├── 🧪 test_environment.py            ← 環境檢查工具---

├── 🖥️ check_gpu.py                   ← GPU 狀態檢查

├── 📊 monitor_gpu.py                 ← GPU 實時監控## 🎯 快速導航

├── 🧪 test_yolov11.py                ← YOLOv11 測試

│### 我想要...

├── 📁 models/                        ← 模型配置文件

│   ├── yolo11_custom.yaml            # 自定義模型配置| 需求 | 前往文檔 | 位置 |

│   └── yolo11_custom_ct.yaml         # CT 專用配置|------|----------|------|

│| **快速開始訓練** | [QUICK_START_GUIDE.md](../QUICK_START_GUIDE.md) | 根目錄 |

├── 📁 yolo_runs/                     ← 訓練輸出目錄| **了解項目概況** | [README.md](../README.md) | 根目錄 |

│   └── train_YYYYMMDD_HHMMSS/        # 每次訓練的結果| **查找所有文檔** | [DOCUMENTATION.md](../DOCUMENTATION.md) | 根目錄 |

│| **了解所有參數** | [guides/TRAIN_DIRECT_README.md](guides/TRAIN_DIRECT_README.md) | guides/ |

└── 📦 *.pt                           ← 預訓練模型 (yolo11n, yolo11s, yolo11m)| **選擇訓練配置** | [guides/CONFIG_COMPARISON.md](guides/CONFIG_COMPARISON.md) | guides/ |

```| **選擇訓練腳本** | [guides/README_TRAINING.md](guides/README_TRAINING.md) | guides/ |

| **檢查訓練環境** | [references/SETUP_COMPLETE.md](references/SETUP_COMPLETE.md) | references/ |

---| **控制負樣本** | [references/NEGATIVE_SAMPLE_DETECTION.md](references/NEGATIVE_SAMPLE_DETECTION.md) | references/ |

| **理解訓練指標** | [references/VALIDATION_METRICS_GUIDE.md](references/VALIDATION_METRICS_GUIDE.md) | references/ |

## 🎯 快速導航

---

### 我想要...

## 📖 文檔分類說明

| 需求 | 前往文檔 | 位置 |

|------|----------|------|### 🏠 根目錄（3 個文檔）

| **快速開始訓練** | [QUICK_START_GUIDE.md](../QUICK_START_GUIDE.md) | 根目錄 ⭐ |**特點**: 最常用、快速訪問

| **了解項目概況** | [README.md](../README.md) | 根目錄 |- **README.md** - 項目主入口，概覽所有功能

| **查找所有文檔** | [INDEX.md](INDEX.md) | docs/ |- **DOCUMENTATION.md** - 完整文檔索引，按需查找

| **完整參數說明** | [TRAIN_DIRECT_README.md](guides/TRAIN_DIRECT_README.md) | guides/ |- **QUICK_START_GUIDE.md** - 新手必讀，3 步開始

| **選擇最佳配置** | [CONFIG_COMPARISON.md](guides/CONFIG_COMPARISON.md) | guides/ |

| **理解訓練指標** | [VALIDATION_METRICS_GUIDE.md](references/VALIDATION_METRICS_GUIDE.md) | references/ |### 📖 guides/（使用指南，3 個文檔）

| **檢查環境配置** | [SETUP_COMPLETE.md](references/SETUP_COMPLETE.md) | references/ |**特點**: 操作性強、詳細說明

| **了解負樣本處理** | [NEGATIVE_SAMPLE_DETECTION.md](references/NEGATIVE_SAMPLE_DETECTION.md) | references/ |- **TRAIN_DIRECT_README.md** - 35+ 參數完整說明

- **CONFIG_COMPARISON.md** - 4 種配置對比選擇

---- **README_TRAINING.md** - 3 個訓練腳本對比



## 📚 文檔分類### 📋 references/（參考資料，6 個文檔）

**特點**: 深入解釋、技術細節

### 🟢 新手文檔（必讀）- **SETUP_COMPLETE.md** - 訓練前環境檢查

1. [README.md](../README.md) - 項目主文檔- **NEGATIVE_SAMPLE_DETECTION.md** - 類別平衡策略

2. [QUICK_START_GUIDE.md](../QUICK_START_GUIDE.md) - 3 步快速開始- **VALIDATION_METRICS_GUIDE.md** - 性能指標解讀

3. [SETUP_COMPLETE.md](references/SETUP_COMPLETE.md) - 環境檢查- **PREPROCESSED_DATASET_UPDATE.md** - 數據處理說明

- **DOCS_UPDATE_REPORT.md** - 版本變更記錄

### 🟡 進階文檔- **DOCUMENTATION_REORGANIZATION_REPORT.md** - 文檔整理報告

1. [TRAIN_DIRECT_README.md](guides/TRAIN_DIRECT_README.md) - 完整參數說明

2. [CONFIG_COMPARISON.md](guides/CONFIG_COMPARISON.md) - 配置對比### 🗄️ archives/（歷史文檔）

3. [TRAINING_GUIDE.md](guides/TRAINING_GUIDE.md) - 訓練技巧**特點**: 已過時但保留

4. [VALIDATION_METRICS_GUIDE.md](references/VALIDATION_METRICS_GUIDE.md) - 指標解釋- **DOCS_INDEX.md** - 舊版索引（已被 DOCUMENTATION.md 取代）



### 🔴 開發者文檔---

1. [NEGATIVE_SAMPLE_DETECTION.md](references/NEGATIVE_SAMPLE_DETECTION.md) - 負樣本機制

2. [PREPROCESSED_DATASET_UPDATE.md](references/PREPROCESSED_DATASET_UPDATE.md) - 數據集結構## 🎓 閱讀順序建議

3. [TRAINING_COMPARISON.md](guides/TRAINING_COMPARISON.md) - 方法對比

### 新手用戶

---```

1. README.md（5 分鐘）

## 🔧 可用工具腳本   ↓

2. QUICK_START_GUIDE.md（10 分鐘）

### 訓練相關   ↓

```cmd3. 開始訓練

rem 主訓練腳本```

python train_yolo_direct.py --data_dir ..\..\datasets\splited_dataset\train --epochs 200 --batch_size 16 --model_size m

```### 進階用戶

```

### 驗證相關1. DOCUMENTATION.md（瀏覽索引）

```cmd   ↓

rem 驗證數據集2. guides/CONFIG_COMPARISON.md（選配置）

python validate_dataset.py --data_dir ..\..\datasets\splited_dataset\train   ↓

3. guides/TRAIN_DIRECT_README.md（查參數）

rem 驗證標註   ↓

python validate_annotations.py --data_dir ..\..\datasets\splited_dataset\train4. references/（按需查閱）

``````



### 環境檢查### 故障排查

```cmd```

rem 檢查環境1. references/SETUP_COMPLETE.md（檢查環境）

python test_environment.py   ↓

2. 相關文檔的「故障排除」章節

rem 檢查 GPU```

python check_gpu.py

---

rem 監控 GPU

python monitor_gpu.py## 💡 維護指南

```

### 新增文檔時

### 測試相關1. 確定類型：使用指南 or 參考資料

```cmd2. 放入對應目錄：`guides/` or `references/`

rem 測試 YOLOv113. 更新 `DOCUMENTATION.md` 索引

python test_yolov11.py --model yolo11m.pt --source ..\..\datasets\splited_dataset\train\A00014. 更新 `README.md`（如果是重要文檔）

```

### 更新文檔時

---1. 檢查交叉引用鏈接是否正確

2. 同步更新索引文檔

## 📖 文檔閱讀順序建議3. 記錄變更到 `DOCS_UPDATE_REPORT.md`



### 第一次使用（新手）### 歸檔文檔時

1. 📄 [README.md](../README.md) - 快速瀏覽項目概況1. 移動到 `archives/`

2. 🚀 [QUICK_START_GUIDE.md](../QUICK_START_GUIDE.md) - 按步驟操作2. 從索引中移除

3. ✅ [SETUP_COMPLETE.md](references/SETUP_COMPLETE.md) - 確認環境正確3. 在歸檔文件中添加說明

4. 🔥 開始訓練！

---

### 想要調整參數（進階）

1. 📖 [TRAIN_DIRECT_README.md](guides/TRAIN_DIRECT_README.md) - 了解所有參數## 📞 需要幫助？

2. 📊 [CONFIG_COMPARISON.md](guides/CONFIG_COMPARISON.md) - 選擇最佳配置

3. 🎓 [TRAINING_GUIDE.md](guides/TRAINING_GUIDE.md) - 學習訓練技巧### 找不到想要的文檔？

4. 📈 [VALIDATION_METRICS_GUIDE.md](references/VALIDATION_METRICS_GUIDE.md) - 理解評估指標→ 查看 [DOCUMENTATION.md](../DOCUMENTATION.md) 的完整索引



### 深入了解系統（開發者）### 不知道從哪開始？

1. 🔬 [NEGATIVE_SAMPLE_DETECTION.md](references/NEGATIVE_SAMPLE_DETECTION.md) - 負樣本處理邏輯→ 閱讀 [QUICK_START_GUIDE.md](../QUICK_START_GUIDE.md)

2. 📁 [PREPROCESSED_DATASET_UPDATE.md](references/PREPROCESSED_DATASET_UPDATE.md) - 數據集結構

3. ⚖️ [TRAINING_COMPARISON.md](guides/TRAINING_COMPARISON.md) - 訓練方法對比### 想了解所有功能？

4. 📝 [DOCUMENTATION_REORGANIZATION_REPORT.md](references/DOCUMENTATION_REORGANIZATION_REPORT.md) - 文檔變更歷史→ 閱讀 [README.md](../README.md)



------



## 🆘 遇到問題？**文檔結構版本**: 3.0  

**最後更新**: 2025-10-14

1. **訓練相關問題** → 查看 [README.md](../README.md) 的「常見問題」部分
2. **環境問題** → 運行 `python test_environment.py` 檢查
3. **數據集問題** → 運行 `python validate_dataset.py` 驗證
4. **GPU 問題** → 運行 `python check_gpu.py` 檢查
5. **參數不理解** → 查看 [TRAIN_DIRECT_README.md](guides/TRAIN_DIRECT_README.md)
6. **指標不理解** → 查看 [VALIDATION_METRICS_GUIDE.md](references/VALIDATION_METRICS_GUIDE.md)

---

## 📝 文檔更新日誌

### 2025-10-17
- ✅ 重寫主 README.md，所有命令改為 CMD 格式
- ✅ 更新 QUICK_START_GUIDE.md，移除不存在的腳本
- ✅ 刪除過時文檔（DOCUMENTATION.md, DOCUMENTATION_SUMMARY.md）
- ✅ 更新 docs/INDEX.md 和 docs/README.md
- ✅ 確保所有文檔路徑引用正確

### 2025-10-14
- ✅ 重組文檔結構（guides/ 和 references/）
- ✅ 創建 DOCUMENTATION.md 作為完整索引
- ✅ 移動文檔到適當目錄

---

## 🔗 外部資源

- [Ultralytics YOLOv11 官方文檔](https://docs.ultralytics.com/)
- [YOLOv11 GitHub](https://github.com/ultralytics/ultralytics)
- [訓練技巧指南](https://docs.ultralytics.com/guides/model-training-tips/)
- [數據集格式](https://docs.ultralytics.com/datasets/detect/)

---

*最後更新: 2025-10-17*  
*文檔總數: 12+ Markdown 文件*  
*維護者: YOLOv11 CT Detection Team*
