# 📚 YOLOv7 檢測文檔索引

> **最後更新**: 2025-10-12  
> 本目錄包含 YOLOv7 胸腔 CT 病灶檢測的所有技術文檔

---

## 📖 文檔結構

### 🏠 主文檔
- **[../README.md](../README.md)** - 快速開始與完整使用指南（⭐ 從這裡開始）

---

## 📘 使用指南 (guides/)

完整的訓練流程與操作指南

### 1. [訓練完整指南](guides/TRAINING_GUIDE.md) 
- 從預處理到訓練的完整工作流程
- Step-by-step 操作步驟
- 參數配置詳解
- 4種訓練範例（測試、標準、高性能、增強）

### 2. [訓練方法對比](guides/TRAINING_COMPARISON.md)
- 原始 DICOM vs 預處理 PNG 詳細對比
- 瓶頸分析與投資回報
- 使用場景建議

### 3. [預處理訓練技術說明](guides/README_PREPROCESSED_TRAINING.md)
- 預處理資料流程詳解
- 核心檔案功能說明
- Dataset 類別技術細節
- 適合開發者深入了解

---

## 📚 技術參考 (references/)

深入的技術說明與機制解析

### 1. [負樣本判斷機制](references/NEGATIVE_SAMPLE_DETECTION.md)
- 負樣本判斷邏輯詳解
- 資料結構範例
- 過濾策略實作
- 包含驗證腳本

### 2. [驗證指標說明](references/VALIDATION_METRICS_GUIDE.md)
- Precision、Recall、F1、mAP 等指標詳解
- 輸出格式說明（終端、CSV、JSON）
- 可視化圖表說明
- 指標查看位置

### 3. [預處理資料集結構](references/PREPROCESSED_DATASET_UPDATE.md)
- 支援的兩種資料結構（扁平 vs 患者分組）
- 自動檢測機制
- 訓練腳本更新說明

### 4. [YOLOv7 vs YOLOv11n 分析](references/YOLOV7_VS_YOLOV11_ANALYSIS.md)
- 模型架構對比（參數量、FLOPs）
- 訓練速度差異分析
- 效果對比與優化建議

---

## 🚀 快速導航

### 我想要...

**開始訓練模型**
→ 閱讀 [README.md](../README.md) 的「快速開始」章節

**了解預處理的優勢**
→ 閱讀 [訓練方法對比](guides/TRAINING_COMPARISON.md)

**配置訓練參數**
→ 參考 [訓練完整指南](guides/TRAINING_GUIDE.md) 的「關鍵參數」章節

**理解驗證指標**
→ 查看 [驗證指標說明](references/VALIDATION_METRICS_GUIDE.md)

**了解負樣本處理**
→ 閱讀 [負樣本判斷機制](references/NEGATIVE_SAMPLE_DETECTION.md)

**深入技術細節**
→ 閱讀 [預處理訓練技術說明](guides/README_PREPROCESSED_TRAINING.md)

**對比 YOLOv7 vs YOLOv11n**
→ 參考 [模型對比分析](references/YOLOV7_VS_YOLOV11_ANALYSIS.md)

---

## 📝 文檔維護

- 所有文檔使用 Markdown 格式
- 主要內容集中在主 README.md，避免重複
- guides/ 目錄存放操作指南
- references/ 目錄存放技術參考
- 定期檢查並移除過時內容

---

## 🔄 版本歷史

### 2025-10-12
- 重新組織文檔結構
- 刪除過時文檔（DOCUMENTATION_CLEANUP_2025-10-12.md、METRICS_QUICK_REFERENCE.md、QUICK_SOLUTION_GUIDE.md）
- 創建 docs/ 目錄並分類整理
- 建立文檔索引系統
