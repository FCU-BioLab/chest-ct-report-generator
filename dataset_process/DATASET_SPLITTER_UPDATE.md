# Dataset Splitter 更新說明

## 📋 更新日期
2025-10-21

## 🎯 更新目的
使 `dataset_splitter.py` 能夠支援 `preprocessed_yolo_lesion` 資料集的目錄結構

---

## 📂 支援的資料集結構

### 結構類型 1: **集中式結構** (preprocessed_yolo_lesion)
```
preprocessed_yolo_lesion/
├── images_png/          # 集中式圖片目錄
│   ├── A0001/
│   │   ├── A0001_DCM_001_1-01.png
│   │   ├── A0001_DCM_002_1-02.png
│   │   └── ...
│   ├── A0002/
│   └── ...
└── labels/              # 集中式標籤目錄
    ├── A0001/
    │   ├── A0001_DCM_001_1-01.txt
    │   ├── A0001_DCM_002_1-02.txt
    │   └── ...
    ├── A0002/
    └── ...
```

**特點：**
- ✅ 所有患者的圖片集中在 `images_png/` 目錄下
- ✅ 所有患者的標籤集中在 `labels/` 目錄下
- ✅ 每個患者一個子資料夾

### 結構類型 2: **分散式結構** (all_patient_data)
```
all_patient_data/
├── A0001/               # 每個患者獨立目錄
│   ├── images/
│   │   ├── slice_001.png
│   │   └── ...
│   └── labels/
│       ├── slice_001.txt
│       └── ...
├── A0002/
└── ...
```

**特點：**
- ✅ 每個患者有自己的 `images/` 和 `labels/` 目錄
- ✅ 結構更獨立，易於單獨處理

---

## 🔄 主要修改內容

### 1. **`scan_patients()` 函數**
- 新增自動檢測資料集結構類型的邏輯
- 支援兩種不同的目錄結構
- 在患者資訊中記錄結構類型

```python
# 檢測集中式結構
if images_png_dir.exists() and labels_dir.exists():
    # preprocessed_yolo_lesion 格式
    
# 檢測分散式結構  
else:
    # all_patient_data 格式
```

### 2. **`copy_patient_data()` 函數**
- 根據結構類型採用不同的複製策略
- **集中式結構**：創建 `train/images_png/` 和 `train/labels/`，然後複製各患者子資料夾
- **分散式結構**：直接複製患者目錄到 `train/` 或 `test/`

### 3. **患者資訊字典擴充**
新增欄位記錄額外資訊：
```python
{
    'series': 'A',
    'path': '/path/to/images',
    'labels_path': '/path/to/labels',  # 集中式結構專用
    'structure_type': 'centralized'     # 或 'distributed'
}
```

---

## ✅ 使用範例

### 切分 preprocessed_yolo_lesion 資料集

```bash
cd E:\GitHub\chest-ct-report-generator\dataset_process

python dataset_splitter.py \
  --source_dir "E:\GitHub\chest-ct-report-generator\datasets\preprocessed_yolo_lesion" \
  --output_dir "E:\GitHub\chest-ct-report-generator\datasets\splits_yolo_lesion" \
  --train_ratio 0.9 \
  --test_ratio 0.1 \
  --random_seed 42
```

### 切分 all_patient_data 資料集

```bash
python dataset_splitter.py \
  --source_dir "E:\GitHub\chest-ct-report-generator\datasets\all_patient_data" \
  --output_dir "E:\GitHub\chest-ct-report-generator\datasets\dataset_splits" \
  --train_ratio 0.8 \
  --test_ratio 0.2 \
  --random_seed 42
```

---

## 📊 執行結果

### 測試資料集資訊
- **資料集名稱**: preprocessed_yolo_lesion
- **總患者數**: 355
- **系列分布**:
  - A 系列: 251 患者
  - B 系列: 38 患者
  - E 系列: 5 患者
  - G 系列: 61 患者

### 劃分結果 (train_ratio=0.9, test_ratio=0.1)
- **訓練集**: 317 患者 (188,649 張圖片)
  - A: 225, B: 34, E: 4, G: 54
- **測試集**: 38 患者 (16,810 張圖片)
  - A: 26, B: 4, E: 1, G: 7

### 輸出目錄結構
```
splits_yolo_lesion/
├── train/
│   ├── images_png/
│   │   ├── A0001/
│   │   ├── A0002/
│   │   └── ... (317 患者)
│   └── labels/
│       ├── A0001/
│       ├── A0002/
│       └── ... (317 患者)
├── test/
│   ├── images_png/
│   │   └── ... (38 患者)
│   └── labels/
│       └── ... (38 患者)
├── train_patients.txt
├── test_patients.txt
└── dataset_split_report.json
```

---

## 🔑 關鍵特性

✅ **自動檢測**：自動識別資料集結構類型  
✅ **分層抽樣**：保持各系列（A/B/E/G）比例一致  
✅ **可重現性**：固定隨機種子（預設 42）  
✅ **進度顯示**：每 50 個患者顯示一次進度  
✅ **完整報告**：生成 JSON 格式的詳細報告  
✅ **向後相容**：仍支援原有的 all_patient_data 結構  

---

## 📝 注意事項

1. **隨機種子**: 使用相同的 `random_seed` 可確保劃分結果可重現
2. **比例驗證**: 程式會自動驗證 `train_ratio + test_ratio = 1.0`
3. **目錄覆蓋**: 如果輸出目錄已存在，會先刪除後重新創建
4. **系列識別**: 根據患者 ID 的第一個字元（A/B/E/G）識別系列

---

## 🚀 後續建議

1. **K-Fold 劃分**: 可進一步擴展支援 K-Fold 交叉驗證的多組劃分
2. **統計視覺化**: 生成劃分後的統計圖表（各系列分布等）
3. **驗證機制**: 新增資料完整性驗證（檢查圖片和標籤數量是否匹配）
4. **增量更新**: 支援增量更新已存在的劃分（僅複製新增患者）

---

## 📧 問題回報
如有問題或建議，請聯繫開發團隊。
