# 資料集處理工具集

此資料夾包含用於處理胸部 CT 影像資料集的各種工具和腳本。

## 📁 工具列表

### 🔧 資料處理工具

#### 1. `preprocess_original_dataset.py`
**功能**：DICOM 影像前處理工具
- 將 DICOM 轉換為 PNG + YOLO 格式
- 將 DICOM 轉換為 NIfTI 格式（3D）
- 支援灰階和 RGB 彩色影像處理
- 自動亮度補正（窗位調整 / CLAHE）
- 輸出處理報告 JSON

**使用方法**：
```bash
# 處理灰階 CT 影像
python preprocess_original_dataset.py

# 處理 RGB 彩色影像
python preprocess_original_dataset.py --rgb
```

**相關文件**：
- `WINDOWING_SETTINGS_GUIDE.md` - CT 窗位窗寬設定指南

---

#### 2. `dataset_splitter.py`
**功能**：資料集劃分工具（K-Fold 交叉驗證）
- 按病患 ID 分割訓練/測試集
- 支援分層抽樣（保持各系列比例）
- 支援兩種資料結構（集中式/分散式）
- 可重現的隨機劃分
- 生成詳細統計報告

**使用方法**：
```bash
python dataset_splitter.py \
    --source_dir ../datasets/preprocessed_yolo_lesion \
    --output_dir ../datasets/splits_yolo_lesion \
    --train_ratio 0.9 \
    --test_ratio 0.1 \
    --random_seed 42
```

**相關文件**：
- `DATASET_SPLITTER_USAGE.md` - 詳細使用說明

---

#### 3. `dataset_analysis.py`
**功能**：資料集統計分析工具
- 統計每位病患的 CT、非 CT、RGB 影像數量
- 生成 JSON 和 CSV 格式報告
- 自動分類 DICOM 影像類型

**使用方法**：
```bash
python dataset_analysis.py
```

---

#### 4. `refactor_all_patients_files.py`
**功能**：病患檔案整理工具
- 按病患 ID 分類整理 DICOM 和 XML 檔案
- 自動複製和組織原始資料
- 生成處理摘要報告

---

### 🔍 視覺化工具

#### 5. `visualize_patient_yolo.py`
**功能**：YOLO 標註視覺化檢查工具
- 按病患分組檢視 YOLO 標註
- 互動式瀏覽模式
- 統計正負樣本分布
- 支援批次輸出視覺化結果

**使用方法**：
```bash
# 視覺化單個患者
python visualize_patient_yolo.py \
    --data_dir ../datasets/preprocessed_yolo_lesion \
    --patient A0001 \
    --num_samples 10

# 互動模式
python visualize_patient_yolo.py \
    --data_dir ../datasets/preprocessed_yolo_lesion \
    --patient A0001 \
    --interactive
```

**相關文件**：
- `README_VISUALIZATION.md` - 詳細使用說明

---

#### 6. `dicom_viewer.py`
**功能**：DICOM 檔案瀏覽器
- 瀏覽 DICOM 影像和對應的 XML 標記
- 支援 2D/3D 檢視
- GUI 互動介面
- 顯示病灶邊界框

**使用方法**：
```bash
python dicom_viewer.py
```

---

#### 7. `test_windowing_preview.py`
**功能**：DICOM 窗位窗寬測試工具
- 動態測試不同窗位窗寬設定
- 自動亮度補正預覽
- 生成轉換報告 JSON
- 比較不同參數效果

**使用方法**：
```bash
python test_windowing_preview.py
```

---

## 📚 文件說明

### 使用指南
- **`README_VISUALIZATION.md`** - YOLO 標註視覺化工具完整使用說明
- **`DATASET_SPLITTER_USAGE.md`** - 資料集劃分工具使用指南
- **`WINDOWING_SETTINGS_GUIDE.md`** - CT 影像窗位窗寬設定指南（不同病變類型的參數建議）

---

## 🔄 典型工作流程

### 流程 1: 從原始 DICOM 到訓練資料集

```bash
# Step 1: 整理原始檔案
python refactor_all_patients_files.py

# Step 2: 分析資料集
python dataset_analysis.py

# Step 3: 前處理 DICOM 影像
python preprocess_original_dataset.py

# Step 4: 劃分訓練/測試集
python dataset_splitter.py \
    --source_dir ../datasets/preprocessed_yolo_lesion \
    --output_dir ../datasets/splits_yolo_lesion \
    --train_ratio 0.9 \
    --test_ratio 0.1

# Step 5: 視覺化檢查
python visualize_patient_yolo.py \
    --data_dir ../datasets/splits_yolo_lesion/train \
    --patient all \
    --num_samples 5
```

### 流程 2: 資料檢查與視覺化

```bash
# 檢視 DICOM 原始資料
python dicom_viewer.py

# 測試窗位窗寬設定
python test_windowing_preview.py

# 視覺化處理後的 YOLO 資料
python visualize_patient_yolo.py --data_dir <path> --interactive
```

---

## ⚙️ 設定檔

大部分工具會讀取專案根目錄的 `config.json` 來獲取資料路徑和參數設定。

---

## 💡 提示

1. **資料路徑**：確保 `config.json` 中的路徑設定正確
2. **依賴套件**：確保已安裝所需的 Python 套件（pydicom, nibabel, opencv-python, matplotlib 等）
3. **視覺化**：建議使用互動模式進行資料檢查
4. **參數調整**：參考 `WINDOWING_SETTINGS_GUIDE.md` 針對不同病變類型調整參數

---

## 📧 問題回報

如有問題或建議，請聯繫開發團隊。
