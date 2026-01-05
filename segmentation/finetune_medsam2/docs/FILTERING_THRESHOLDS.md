# Lesion Filtering Thresholds 使用指南

## 問題背景

在 `test_and_extract_features()` 中，原本的實作會將每個 bbox 都計為一個 lesion，導致：
- **Lesion 數量過多**：同一個 3D 結節在多個切片中被重複計數
- **低質量預測被保留**：SAM2 的低置信度預測和錯誤預測都被計入

## 解決方案

新增 **三階段過濾機制**，依序過濾低質量預測：

### 過濾器 1: 面積閾值
- **閾值**: 50 pixels（固定）
- **目的**: 過濾噪音和碎片
- **過濾對象**: 預測面積 < 50px 的 mask

### 過濾器 2: SAM2 置信度閾值
- **參數**: `min_confidence` (預設 0.5)
- **來源**: SAM2 的 `iou_predictions` 輸出
- **目的**: 過濾 SAM2 自身不確定的預測
- **過濾對象**: IoU prediction < min_confidence 的預測

### 過濾器 3: Dice 分數閾值
- **參數**: `min_dice` (預設 0.3)
- **來源**: 預測 mask 與 GT mask 的 Dice 分數
- **目的**: 過濾與 Ground Truth 不匹配的預測
- **過濾對象**: Dice score < min_dice 的預測

---

## 使用方法

### 基本用法（使用預設閾值）

```python
from finetune_medsam2 import MedSAM2Trainer

trainer = MedSAM2Trainer(...)
trainer.load_checkpoint("best_model.pth")

# 使用預設閾值: min_confidence=0.5, min_dice=0.3
results = trainer.test_and_extract_features(
    test_loader=test_loader,
    output_dir="test_results"
)
```

### 調整閾值（更嚴格過濾）

```python
# 更嚴格的過濾，減少 false positives
results = trainer.test_and_extract_features(
    test_loader=test_loader,
    output_dir="test_results",
    min_confidence=0.7,  # 只保留高置信度預測
    min_dice=0.5         # 只保留與 GT 高度匹配的預測
)
```

### 寬鬆過濾（保留更多候選）

```python
# 寬鬆過濾，適合 recall 優先的場景
results = trainer.test_and_extract_features(
    test_loader=test_loader,
    output_dir="test_results",
    min_confidence=0.3,  # 保留較多候選
    min_dice=0.1         # 允許較低的 Dice 分數
)
```

---

## 閾值調整建議

### 場景 1: Lesion 數量仍然過多
**症狀**: 過濾後仍有大量 false positives

**建議**:
```python
min_confidence=0.7,  # 提高置信度要求
min_dice=0.5         # 提高 Dice 要求
```

### 場景 2: Lesion 數量過少（漏檢）
**症狀**: 真實的結節被過濾掉

**建議**:
```python
min_confidence=0.3,  # 降低置信度要求
min_dice=0.2         # 降低 Dice 要求
```

### 場景 3: 平衡 Precision 和 Recall
**症狀**: 需要在準確率和召回率之間取得平衡

**建議**:
```python
min_confidence=0.5,  # 預設值
min_dice=0.3         # 預設值
```

---

## 過濾統計輸出

執行測試後，會自動輸出過濾統計：

```
📊 過濾統計:
   總 bbox 數: 1500
   過濾 (面積 < 50px): 120 (8.0%)
   過濾 (置信度 < 0.50): 450 (30.0%)
   過濾 (Dice < 0.30): 230 (15.3%)
   ✅ 保留病灶: 700 (46.7%)
```

**解讀**:
- **總 bbox 數**: Dataset 中所有 bbox 的數量
- **過濾 (面積)**: 被面積閾值過濾的數量
- **過濾 (置信度)**: 被 SAM2 置信度過濾的數量
- **過濾 (Dice)**: 被 Dice 分數過濾的數量
- **保留病灶**: 最終保留的 lesion 數量

---

## 結果檔案

過濾統計會保存在輸出的 JSON 檔案中：

```json
{
  "filtering_stats": {
    "total_bboxes": 1500,
    "filtered_by_area": 120,
    "filtered_by_confidence": 450,
    "filtered_by_dice": 230,
    "kept_lesions": 700
  },
  "total_lesions": 700,
  ...
}
```

---

## 常見問題

### Q1: 為什麼需要三個過濾器？
**A**: 每個過濾器針對不同類型的錯誤：
- 面積過濾 → 噪音和碎片
- 置信度過濾 → SAM2 不確定的預測
- Dice 過濾 → 與 GT 不匹配的預測

### Q2: 閾值設定的經驗法則？
**A**: 
- `min_confidence`: 0.3-0.7（0.5 為平衡點）
- `min_dice`: 0.2-0.5（0.3 為平衡點）

### Q3: 如何知道閾值設定是否合理？
**A**: 觀察過濾統計：
- 如果 `kept_lesions` 比例 < 30%，閾值可能過嚴
- 如果 `kept_lesions` 比例 > 80%，閾值可能過寬
- 理想範圍：40-60%

### Q4: 過濾會影響 metrics 計算嗎？
**A**: 會！只有通過過濾的 lesions 會被用於計算平均 Dice/IoU 等指標，因此過濾後的 metrics 通常會更高（因為移除了低質量預測）。

---

## 進階技巧

### 動態閾值調整

根據資料集特性動態調整：

```python
# 小結節資料集（容易漏檢）
if dataset_type == "small_nodules":
    min_confidence = 0.3
    min_dice = 0.2

# 大結節資料集（容易誤檢）
elif dataset_type == "large_nodules":
    min_confidence = 0.7
    min_dice = 0.5
```

### 批次測試不同閾值

```python
thresholds = [
    (0.3, 0.2),  # 寬鬆
    (0.5, 0.3),  # 預設
    (0.7, 0.5),  # 嚴格
]

for conf, dice in thresholds:
    results = trainer.test_and_extract_features(
        test_loader=test_loader,
        output_dir=f"results_conf{conf}_dice{dice}",
        min_confidence=conf,
        min_dice=dice
    )
    print(f"Conf={conf}, Dice={dice}: {results['total_lesions']} lesions")
```

---

## 更新日誌

- **2026-01-03**: 新增 `min_confidence` 和 `min_dice` 參數，實作三階段過濾機制
