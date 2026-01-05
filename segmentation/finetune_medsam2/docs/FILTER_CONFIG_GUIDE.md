# 測試過濾參數配置指南

## 📍 配置位置

所有測試過濾參數都在 **`config.py`** 的 `InferenceConfig` 中：

```python
@dataclass
class InferenceConfig:
    """推論相關配置"""
    # 閾值
    prediction_threshold: float = 0.7
    
    # 最小結節過濾
    min_nodule_diameter: float = 0.0
    
    # 測試過濾參數
    min_area: int = 0  # 最小病灶面積（像素），0 表示不過濾
    min_confidence: float = 0.5  # 最小置信度（SAM2 IoU prediction）
    min_dice: float = 0.3  # 最小 Dice 分數（與 GT 的匹配度）
```

---

## 🔧 如何修改

### 方法 1: 直接修改 config.py（推薦）

編輯 `segmentation/finetune_medsam2/config.py` 第 85-87 行：

```python
# 測試過濾參數
min_area: int = 0        # 改為 20 可過濾小於 20px 的預測
min_confidence: float = 0.3  # 改為 0.3 更寬鬆（保留更多預測）
min_dice: float = 0.2    # 改為 0.2 更寬鬆
```

### 方法 2: 使用自訂配置檔案

創建 `my_config.json`:

```json
{
  "inference": {
    "min_area": 20,
    "min_confidence": 0.3,
    "min_dice": 0.2
  }
}
```

然後載入（需要在 main.py 中添加支援）。

---

## 📊 參數說明

### 1. `min_area` - 最小病灶面積

- **預設值**: 0（不過濾）
- **單位**: 像素
- **作用**: 過濾面積太小的預測（通常是噪音）
- **建議值**:
  - `0`: 不過濾（保留所有預測）
  - `20-50`: 過濾小噪音
  - `100+`: 只保留較大的病灶

**範例**:
```python
min_area: int = 0   # 不過濾，保留所有（您當前的設定）
min_area: int = 50  # 過濾 < 50px 的預測（之前的設定）
```

### 2. `min_confidence` - 最小置信度

- **預設值**: 0.5
- **範圍**: 0.0 - 1.0
- **作用**: 過濾 SAM2 模型置信度低的預測
- **建議值**:
  - `0.3`: 寬鬆（保留更多預測）
  - `0.5`: 平衡（預設）
  - `0.7`: 嚴格（只保留高置信度）

**範例**:
```python
min_confidence: float = 0.3  # 更寬鬆，保留更多
min_confidence: float = 0.5  # 預設
min_confidence: float = 0.7  # 更嚴格
```

### 3. `min_dice` - 最小 Dice 分數

- **預設值**: 0.3
- **範圍**: 0.0 - 1.0
- **作用**: 過濾與 Ground Truth 匹配度低的預測
- **建議值**:
  - `0.2`: 寬鬆
  - `0.3`: 平衡（預設）
  - `0.5`: 嚴格

**範例**:
```python
min_dice: float = 0.2  # 更寬鬆
min_dice: float = 0.3  # 預設
min_dice: float = 0.5  # 更嚴格
```

---

## 💡 使用場景

### 場景 1: 保留所有預測（不過濾）

```python
min_area: int = 0
min_confidence: float = 0.0
min_dice: float = 0.0
```

**適用**: 想看到所有預測結果，包括低質量的

### 場景 2: 平衡過濾（預設）

```python
min_area: int = 0
min_confidence: float = 0.5
min_dice: float = 0.3
```

**適用**: 一般使用，過濾明顯錯誤的預測

### 場景 3: 嚴格過濾

```python
min_area: int = 50
min_confidence: float = 0.7
min_dice: float = 0.5
```

**適用**: 只要高質量的預測，用於最終報告

### 場景 4: 保留小病灶（您的需求）

```python
min_area: int = 0        # ✅ 不過濾小面積
min_confidence: float = 0.3  # 降低置信度要求
min_dice: float = 0.2    # 降低 Dice 要求
```

**適用**: 檢測小結節，不想漏掉任何可能的病灶

---

## 🔍 如何查看過濾效果

修改 config.py 後，執行測試：

```bash
cd segmentation
python finetune_medsam2/main.py --test --extract_features \
    --resume result/segmentation_*/best_model.pth
```

查看日誌輸出：

```
📊 過濾閾值:
   - 最小面積: 0 px
   - 最小置信度 (IoU prediction): 0.50
   - 最小 Dice 分數: 0.30

📊 過濾統計:
   總 bbox 數: 9273
   過濾 (面積 < 0px): 0 (0.0%)      ← 面積過濾數量
   過濾 (置信度 < 0.50): 0 (0.0%)   ← 置信度過濾數量
   過濾 (Dice < 0.30): 14 (0.2%)    ← Dice 過濾數量
   ✅ 保留病灶: 9259 (99.8%)        ← 最終保留的數量
```

---

## 📈 您之前的測試結果分析

### 之前（min_area=50）

```
總 bbox 數: 9273
過濾 (面積 < 50px): 8885 (95.8%)  ← 大量被過濾
✅ 保留病灶: 374 (4.0%)
```

### 現在（min_area=0）

預期結果：

```
總 bbox 數: 9273
過濾 (面積 < 0px): 0 (0.0%)      ← 不再過濾
過濾 (置信度 < 0.5): ~100 (1%)
過濾 (Dice < 0.3): ~100 (1%)
✅ 保留病灶: ~9000 (97%)          ← 大幅增加！
```

---

## ⚙️ 快速參考

| 需求 | min_area | min_confidence | min_dice |
|------|----------|----------------|----------|
| **保留所有** | 0 | 0.0 | 0.0 |
| **平衡** | 0 | 0.5 | 0.3 |
| **嚴格** | 50 | 0.7 | 0.5 |
| **小病灶** | 0 | 0.3 | 0.2 |

---

## 📝 修改步驟

1. 打開 `segmentation/finetune_medsam2/config.py`
2. 找到第 85-87 行的 `InferenceConfig`
3. 修改參數值
4. 保存檔案
5. 重新執行測試

**就這麼簡單！** ✅
