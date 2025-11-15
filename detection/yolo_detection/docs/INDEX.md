# 📚 YOLOv11 檢測文檔索引# 📚 YOLOv7 檢測文檔索引



> **最後更新**: 2025-10-17  > **最後更新**: 2025-10-12  

> 本目錄包含 YOLOv11 胸腔 CT 病灶檢測的所有技術文檔> 本目錄包含 YOLOv7 胸腔 CT 病灶檢測的所有技術文檔



------



## 📖 文檔結構## 📖 文檔結構



### 🏠 主文檔### 🏠 主文檔

- **[../README.md](../README.md)** - 快速開始與完整使用指南（⭐ 從這裡開始）- **[../README.md](../README.md)** - 快速開始與完整使用指南（⭐ 從這裡開始）

- **[../QUICK_START_GUIDE.md](../QUICK_START_GUIDE.md)** - 3 步快速開始訓練（新手必讀）

---

---

## 📘 使用指南 (guides/)

## 📘 使用指南 (guides/)

完整的訓練流程與操作指南

完整的訓練流程與操作指南

### 1. [訓練完整指南](guides/TRAINING_GUIDE.md) 

### 1. [訓練指南](guides/README_TRAINING.md) - 從預處理到訓練的完整工作流程

- 訓練腳本選擇和使用- Step-by-step 操作步驟

- 完整訓練流程- 參數配置詳解

- 配置參數說明- 4種訓練範例（測試、標準、高性能、增強）



### 2. [訓練詳細文檔](guides/TRAIN_DIRECT_README.md)### 2. [訓練方法對比](guides/TRAINING_COMPARISON.md)

- train_yolo_direct.py 完整參數說明- 原始 DICOM vs 預處理 PNG 詳細對比

- 35+ 可調參數詳解- 瓶頸分析與投資回報

- 高級配置選項- 使用場景建議



### 3. [配置對比表](guides/CONFIG_COMPARISON.md)### 3. [預處理訓練技術說明](guides/README_PREPROCESSED_TRAINING.md)

- 不同配置的對比分析- 預處理資料流程詳解

- 選擇最適合的配置- 核心檔案功能說明

- 性能預期- Dataset 類別技術細節

- 適合開發者深入了解

### 4. [訓練技巧](guides/TRAINING_GUIDE.md)

- 訓練最佳實踐---

- 常見問題解決

- 性能優化技巧## 📚 技術參考 (references/)



### 5. [訓練方法對比](guides/TRAINING_COMPARISON.md)深入的技術說明與機制解析

- 不同訓練方法的優缺點

- 適用場景分析### 1. [負樣本判斷機制](references/NEGATIVE_SAMPLE_DETECTION.md)

- 負樣本判斷邏輯詳解

---- 資料結構範例

- 過濾策略實作

## 📚 技術參考 (references/)- 包含驗證腳本



深入的技術說明與機制解析### 2. [驗證指標說明](references/VALIDATION_METRICS_GUIDE.md)

- Precision、Recall、F1、mAP 等指標詳解

### 1. [配置完成檢查清單](references/SETUP_COMPLETE.md)- 輸出格式說明（終端、CSV、JSON）

- 訓練前環境檢查- 可視化圖表說明

- 依賴安裝驗證- 指標查看位置

- 數據集準備確認

### 3. [預處理資料集結構](references/PREPROCESSED_DATASET_UPDATE.md)

### 2. [驗證指標說明](references/VALIDATION_METRICS_GUIDE.md)- 支援的兩種資料結構（扁平 vs 患者分組）

- Precision、Recall、F1、mAP 等指標詳解- 自動檢測機制

- 輸出格式說明（終端、CSV、JSON）- 訓練腳本更新說明

- 可視化圖表說明

- 指標查看位置### 4. [YOLOv7 vs YOLOv11n 分析](references/YOLOV7_VS_YOLOV11_ANALYSIS.md)

- 模型架構對比（參數量、FLOPs）

### 3. [負樣本判斷機制](references/NEGATIVE_SAMPLE_DETECTION.md)- 訓練速度差異分析

- 負樣本判斷邏輯詳解- 效果對比與優化建議

- 資料結構範例

- 過濾策略實作---

- 包含驗證腳本

## 🚀 快速導航

### 4. [預處理資料集結構](references/PREPROCESSED_DATASET_UPDATE.md)

- 支援的資料結構### 我想要...

- 自動檢測機制

- 訓練腳本更新說明**開始訓練模型**

→ 閱讀 [README.md](../README.md) 的「快速開始」章節

### 5. [文檔重組報告](references/DOCUMENTATION_REORGANIZATION_REPORT.md)

- 文檔整理歷史記錄**了解預處理的優勢**

- 結構變更說明→ 閱讀 [訓練方法對比](guides/TRAINING_COMPARISON.md)



---**配置訓練參數**

→ 參考 [訓練完整指南](guides/TRAINING_GUIDE.md) 的「關鍵參數」章節

## 🚀 快速導航

**理解驗證指標**

### 我想要...→ 查看 [驗證指標說明](references/VALIDATION_METRICS_GUIDE.md)



**開始訓練模型****了解負樣本處理**

→ 閱讀 [QUICK_START_GUIDE.md](../QUICK_START_GUIDE.md)→ 閱讀 [負樣本判斷機制](references/NEGATIVE_SAMPLE_DETECTION.md)



**了解所有訓練參數****深入技術細節**

→ 閱讀 [完整訓練文檔](guides/TRAIN_DIRECT_README.md)→ 閱讀 [預處理訓練技術說明](guides/README_PREPROCESSED_TRAINING.md)



**選擇最佳配置****對比 YOLOv7 vs YOLOv11n**

→ 參考 [配置對比表](guides/CONFIG_COMPARISON.md)→ 參考 [模型對比分析](references/YOLOV7_VS_YOLOV11_ANALYSIS.md)



**理解驗證指標**---

→ 查看 [驗證指標說明](references/VALIDATION_METRICS_GUIDE.md)

## 📝 文檔維護

**了解負樣本處理**

→ 閱讀 [負樣本判斷機制](references/NEGATIVE_SAMPLE_DETECTION.md)- 所有文檔使用 Markdown 格式

- 主要內容集中在主 README.md，避免重複

**檢查訓練環境**- guides/ 目錄存放操作指南

→ 查看 [配置完成檢查清單](references/SETUP_COMPLETE.md)- references/ 目錄存放技術參考

- 定期檢查並移除過時內容

**學習訓練技巧**

→ 閱讀 [訓練技巧指南](guides/TRAINING_GUIDE.md)---



---## 🔄 版本歷史



## 🎯 文檔使用建議### 2025-10-12

- 重新組織文檔結構

### 新手用戶- 刪除過時文檔（DOCUMENTATION_CLEANUP_2025-10-12.md、METRICS_QUICK_REFERENCE.md、QUICK_SOLUTION_GUIDE.md）

1. 先閱讀 [QUICK_START_GUIDE.md](../QUICK_START_GUIDE.md)- 創建 docs/ 目錄並分類整理

2. 運行 `python test_environment.py` 檢查環境- 建立文檔索引系統

3. 使用推薦配置開始第一次訓練
4. 遇到問題查看 [README.md](../README.md) 的常見問題部分

### 進階用戶
1. 參考 [TRAIN_DIRECT_README.md](guides/TRAIN_DIRECT_README.md) 了解所有參數
2. 根據需求選擇配置 [CONFIG_COMPARISON.md](guides/CONFIG_COMPARISON.md)
3. 學習優化技巧 [TRAINING_GUIDE.md](guides/TRAINING_GUIDE.md)
4. 深入理解指標 [VALIDATION_METRICS_GUIDE.md](references/VALIDATION_METRICS_GUIDE.md)

### 開發者
1. 了解數據集結構 [PREPROCESSED_DATASET_UPDATE.md](references/PREPROCESSED_DATASET_UPDATE.md)
2. 理解負樣本機制 [NEGATIVE_SAMPLE_DETECTION.md](references/NEGATIVE_SAMPLE_DETECTION.md)
3. 查看文檔變更歷史 [DOCUMENTATION_REORGANIZATION_REPORT.md](references/DOCUMENTATION_REORGANIZATION_REPORT.md)

---

## 📝 文檔維護

- 所有文檔使用 Markdown 格式
- 主要內容集中在主 README.md，避免重複
- guides/ 目錄存放操作指南和使用說明
- references/ 目錄存放技術參考和深入文檔
- archives/ 目錄存放歷史文檔（已過時）
- 定期檢查並更新內容

---

## 🔗 相關資源

### 外部文檔
- [Ultralytics YOLOv11 官方文檔](https://docs.ultralytics.com/)
- [YOLOv11 GitHub](https://github.com/ultralytics/ultralytics)
- [訓練技巧指南](https://docs.ultralytics.com/guides/model-training-tips/)
- [數據集格式](https://docs.ultralytics.com/datasets/detect/)

### 內部腳本
- `train_yolo_direct.py` - 主訓練腳本
- `validate_dataset.py` - 數據集驗證
- `validate_annotations.py` - 標註驗證
- `test_environment.py` - 環境檢查
- `check_gpu.py` - GPU 檢查
- `monitor_gpu.py` - GPU 監控

---

*最後更新: 2025-10-17*  
*維護者: YOLOv11 CT Detection Team*
