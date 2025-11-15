# 負樣本判斷機制詳解 🔍

## 核心判斷邏輯

### **簡單規則：看標籤檔案**

```python
# 在 PreprocessedYOLODataset._build_sample_indices() 中
for idx in range(len(self.image_files)):
    label_file = self.label_files[idx]
    
    # 判斷是否有標註
    has_annotation = label_file.exists() and label_file.stat().st_size > 0
    
    if has_annotation:
        self.positive_indices.append(idx)  # ✅ 正樣本（有病灶）
    else:
        self.negative_indices.append(idx)  # ❌ 負樣本（無病灶）
```

### **判斷標準**

| 條件 | 判斷結果 |
|------|---------|
| 標籤檔案**不存在** | ❌ **負樣本** |
| 標籤檔案**存在但為空**（0 字節） | ❌ **負樣本** |
| 標籤檔案**存在且有內容** | ✅ **正樣本** |

---

## 📂 資料結構範例

### **正樣本（有病灶）**

```
train/
├── images/
│   └── A0001_slice_0050.png    ← 影像檔案
└── labels/
    └── A0001_slice_0050.txt    ← 標籤檔案（有內容）
```

**標籤檔案內容**:
```txt
0 0.512 0.423 0.086 0.094
0 0.678 0.556 0.072 0.081
```
- 格式：`class_id x_center y_center width height`
- 每一行代表一個病灶邊界框
- **有內容 = 正樣本**

---

### **負樣本（無病灶）**

#### **情況 1：標籤檔案不存在**
```
train/
├── images/
│   └── A0001_slice_0010.png    ← 影像檔案
└── labels/
    └── (沒有對應的 .txt 檔案)   ← 標籤檔案不存在
```

#### **情況 2：標籤檔案為空**
```
train/
├── images/
│   └── A0001_slice_0020.png    ← 影像檔案
└── labels/
    └── A0001_slice_0020.txt    ← 標籤檔案存在但大小為 0 字節
```

**標籤檔案內容**:
```txt
(空白檔案，無任何內容)
```

---

## 🔬 詳細實作流程

### **步驟 1：載入圖像列表**

```python
# 在 _load_image_files() 中
self.image_files = sorted(list(images_dir.glob("*.png")))
# 結果: ['A0001_slice_0001.png', 'A0001_slice_0002.png', ...]
```

### **步驟 2：生成對應標籤路徑**

```python
self.label_files = [
    labels_dir / (img.stem + ".txt") 
    for img in self.image_files
]
# 結果: ['A0001_slice_0001.txt', 'A0001_slice_0002.txt', ...]
```

### **步驟 3：逐一檢查標籤檔案**

```python
def _build_sample_indices(self):
    self.positive_indices = []  # 正樣本索引列表
    self.negative_indices = []  # 負樣本索引列表
    
    for idx in range(len(self.image_files)):
        label_file = self.label_files[idx]
        
        # 核心判斷邏輯
        has_annotation = (
            label_file.exists() and           # 檔案存在？
            label_file.stat().st_size > 0     # 檔案大小 > 0？
        )
        
        if has_annotation:
            self.positive_indices.append(idx)
        else:
            self.negative_indices.append(idx)
```

---

## 📊 實際範例

### **資料集結構**

```
datasets/splited_dataset/train/
├── images/
│   ├── A0001_slice_0001.png  ← 有病灶
│   ├── A0001_slice_0002.png  ← 無病灶
│   ├── A0001_slice_0003.png  ← 有病灶
│   ├── A0002_slice_0001.png  ← 無病灶
│   └── ...
└── labels/
    ├── A0001_slice_0001.txt  ← 檔案存在，有內容（18 字節）
    ├── A0001_slice_0002.txt  ← 檔案存在，但為空（0 字節）
    ├── A0001_slice_0003.txt  ← 檔案存在，有內容（25 字節）
    └── (A0002_slice_0001.txt 不存在)
```

### **判斷結果**

| 圖像檔案 | 標籤檔案狀態 | 檔案大小 | 判斷結果 |
|---------|------------|---------|---------|
| A0001_slice_0001.png | 存在 | 18 字節 | ✅ **正樣本** |
| A0001_slice_0002.png | 存在 | 0 字節 | ❌ **負樣本** |
| A0001_slice_0003.png | 存在 | 25 字節 | ✅ **正樣本** |
| A0002_slice_0001.png | **不存在** | - | ❌ **負樣本** |

### **生成的索引**

```python
positive_indices = [0, 2]      # A0001_slice_0001, A0001_slice_0003
negative_indices = [1, 3]      # A0001_slice_0002, A0002_slice_0001
```

---

## 🎯 過濾負樣本的邏輯

### **在 `dataset_filter.py` 中**

```python
def filter_negative_samples(dataset, max_negative_per_patient=20):
    # 1. 獲取所有負樣本索引
    negative_indices = dataset.negative_indices  # [1, 3, 5, 7, ...]
    
    # 2. 按患者 ID 分組
    patient_negatives = defaultdict(list)
    for idx in negative_indices:
        img_path = dataset.image_files[idx]
        patient_id = extract_patient_id(img_path)  # 'A0001'
        patient_negatives[patient_id].append(idx)
    
    # 結果示例:
    # {
    #   'A0001': [1, 5, 12, 18, 24, ...],  ← 患者 A0001 的所有負樣本索引
    #   'A0002': [3, 7, 15, 23, ...],      ← 患者 A0002 的所有負樣本索引
    # }
    
    # 3. 限制每個患者的負樣本數量
    selected_negative_indices = []
    for patient_id, neg_indices in patient_negatives.items():
        if len(neg_indices) <= max_negative_per_patient:
            # 全部保留
            selected_negative_indices.extend(neg_indices)
        else:
            # 隨機選擇 max_negative_per_patient 個
            selected = random.sample(neg_indices, max_negative_per_patient)
            selected_negative_indices.extend(selected)
    
    # 4. 合併正樣本（全部保留）和篩選後的負樣本
    selected_indices = sorted(
        list(dataset.positive_indices) +  # 所有正樣本
        selected_negative_indices          # 篩選後的負樣本
    )
    
    return Subset(dataset, selected_indices)
```

---

## 💡 為什麼這樣判斷？

### **優點**

1. **簡單可靠**
   - 不需要讀取圖像內容
   - 不需要分析像素值
   - 只看檔案系統資訊

2. **高效快速**
   ```python
   # 檢查檔案存在和大小非常快
   label_file.exists()         # 只查詢檔案系統
   label_file.stat().st_size   # 只讀取檔案元數據
   ```

3. **符合 YOLO 標準**
   - YOLO 格式的標準做法
   - 無標註 = 負樣本
   - 有標註 = 正樣本

### **限制**

1. **依賴標註品質**
   - 如果標註錯誤（該標的沒標），會誤判為負樣本
   - 如果空檔案沒刪除，會誤判為負樣本

2. **無法檢測標註內容錯誤**
   - 不檢查座標是否合理
   - 不檢查格式是否正確

---

## 🔧 驗證負樣本判斷

### **快速檢查腳本**

```python
#!/usr/bin/env python3
"""檢查資料集中的正負樣本分布"""

from pathlib import Path
from yolo_detection.preprocessed_dataset import PreprocessedYOLODataset

# 載入資料集
dataset = PreprocessedYOLODataset(
    data_root="../../datasets/splited_dataset",
    split="train",
    img_size=640,
)

print(f"總樣本數: {len(dataset)}")
print(f"正樣本數: {len(dataset.positive_indices)} ({len(dataset.positive_indices)/len(dataset)*100:.1f}%)")
print(f"負樣本數: {len(dataset.negative_indices)} ({len(dataset.negative_indices)/len(dataset)*100:.1f}%)")

# 檢查前 10 個樣本
print("\n前 10 個樣本:")
for i in range(min(10, len(dataset))):
    img_path = dataset.image_files[i]
    label_path = dataset.label_files[i]
    
    label_exists = label_path.exists()
    label_size = label_path.stat().st_size if label_exists else 0
    
    sample_type = "正樣本" if i in dataset.positive_indices else "負樣本"
    
    print(f"{i}: {img_path.name} | 標籤: {'存在' if label_exists else '不存在'} | "
          f"大小: {label_size} 字節 | {sample_type}")
```

### **預期輸出**

```
總樣本數: 146236
正樣本數: 22711 (15.5%)
負樣本數: 123525 (84.5%)

前 10 個樣本:
0: A0001_slice_0001.png | 標籤: 存在 | 大小: 18 字節 | 正樣本
1: A0001_slice_0002.png | 標籤: 存在 | 大小: 0 字節 | 負樣本
2: A0001_slice_0003.png | 標籤: 不存在 | 大小: 0 字節 | 負樣本
3: A0001_slice_0004.png | 標籤: 存在 | 大小: 25 字節 | 正樣本
...
```

---

## 📚 相關程式碼位置

1. **負樣本判斷**: `preprocessed_dataset.py` 第 156-169 行
   ```python
   def _build_sample_indices(self):
       ...
       has_annotation = label_file.exists() and label_file.stat().st_size > 0
   ```

2. **負樣本過濾**: `dataset_filter.py` 第 40-110 行
   ```python
   def filter_negative_samples(dataset, max_negative_per_patient=20):
       ...
   ```

3. **訓練整合**: `train_yolov7_preprocessed.py` 第 47-72 行
   ```python
   def prepare_preprocessed_datasets(config):
       ...
       full_train_dataset = filter_negative_samples(...)
   ```

---

## ✅ 總結

### **負樣本 = 無病灶的 CT 切片**

判斷方式：
1. ✅ 標籤檔案不存在
2. ✅ 標籤檔案存在但為空（0 字節）
3. ❌ 標籤檔案存在且有內容（= 正樣本）

### **為什麼要過濾負樣本？**

- 預處理資料集包含大量負樣本（84.5%）
- 每個患者可能有 100+ 個負樣本切片
- 過多負樣本會：
  - 大幅增加訓練時間
  - 導致樣本不平衡
  - 不一定提升模型性能

### **過濾策略**

- 保留所有正樣本（100%）
- 限制每個患者的負樣本數量（預設 20 個）
- 從 146,236 樣本 → ~30,000 樣本
- 訓練時間從 34 天 → 8 天（4.2x 加速）
