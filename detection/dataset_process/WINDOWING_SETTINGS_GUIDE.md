# CT 影像窗位窗寬設定指南

> 針對不同肺部病變類型的 HU 窗位窗寬和 CLAHE 參數設定指南

## 📚 目錄

- [基礎概念](#基礎概念)
- [窗位窗寬參數說明](#窗位窗寬參數說明)
- [不同病變類型的設定](#不同病變類型的設定)
- [CLAHE 參數調整](#clahe-參數調整)
- [實際應用建議](#實際應用建議)
- [參數快速查詢表](#參數快速查詢表)

---

## 基礎概念

### 什麼是 HU 值（Hounsfield Unit）？

HU 值是 CT 掃描中用來表示組織密度的標準單位：

| 組織類型 | HU 值範圍 | 說明 |
|---------|----------|------|
| **空氣** | -1000 HU | 最低密度 |
| **肺組織** | -900 ~ -500 HU | 充氣的肺泡 |
| **脂肪** | -120 ~ -90 HU | 皮下脂肪、縱隔脂肪 |
| **水** | 0 HU | 基準值 |
| **軟組織** | +30 ~ +70 HU | 肌肉、實質器官 |
| **血液** | +30 ~ +45 HU | 未凝固的血液 |
| **鈣化/骨骼** | +400 ~ +1000 HU | 高密度組織 |

### 窗位窗寬的作用

```
窗位（Window Center）：顯示灰度的中心點
窗寬（Window Width）：顯示的 HU 值範圍

實際顯示範圍計算：
  最小值 = 窗位 - (窗寬 / 2)
  最大值 = 窗位 + (窗寬 / 2)
```

**範例：肺窗設定**
```
窗位 = -600 HU
窗寬 = 1500 HU

顯示範圍：
  最小值 = -600 - (1500/2) = -1350 HU
  最大值 = -600 + (1500/2) = +150 HU
```

---

## 窗位窗寬參數說明

### 在 `preprocess_for_yolo.py` 中的參數

```bash
--window_center <數值>  # 窗位中心（HU 值）
--window_width <數值>   # 窗寬範圍（HU 值）
--enable_clahe          # 啟用 CLAHE 對比度增強
--clahe_clip <數值>     # CLAHE 對比度限制（1.0-4.0）
```

---

## 不同病變類型的設定

### 1. 早期肺癌 / 磨玻璃結節（GGO）

**病變特徵**：
- HU 值：-800 ~ -300 HU
- 外觀：淡淡的霧狀陰影
- 密度：接近肺組織，對比度低

**推薦設定**：
```bash
python preprocess_for_yolo.py \
    --data_root ../../datasets/splited_dataset \
    --output_dir ../../datasets/preprocessed_yolo_ggo \
    --imgsz 640 \
    --window_center -600 \
    --window_width 1500 \
    --enable_clahe \
    --clahe_clip 1.5 \
    --max_negative 20
```

| 參數 | 數值 | 說明 |
|-----|------|------|
| `window_center` | -600 | 針對肺組織密度 |
| `window_width` | 1500 | 涵蓋 -1350 ~ +150 HU |
| `clahe_clip` | 1.5 | 輕微增強，保留細微差異 |

**適用場景**：
- ✅ 肺癌早期篩查
- ✅ 磨玻璃結節追蹤
- ✅ 亞實性結節檢測
- ❌ 不適合實性腫瘤

---

### 2. 部分實性結節 / 混合密度腫瘤

**病變特徵**：
- HU 值：-300 ~ +30 HU
- 外觀：中心實性，周圍磨玻璃
- 密度：混合密度，惡性風險高

**推薦設定**：
```bash
python preprocess_for_yolo.py \
    --data_root ../../datasets/splited_dataset \
    --output_dir ../../datasets/preprocessed_yolo_mixed \
    --imgsz 640 \
    --window_center -400 \
    --window_width 1800 \
    --enable_clahe \
    --clahe_clip 2.0 \
    --max_negative 20
```

| 參數 | 數值 | 說明 |
|-----|------|------|
| `window_center` | -400 | 介於肺窗和縱隔窗之間 |
| `window_width` | 1800 | 涵蓋 -1300 ~ +500 HU |
| `clahe_clip` | 2.0 | 平衡增強 |

**適用場景**：
- ✅ 部分實性結節
- ✅ 浸潤性腺癌
- ✅ 需要評估實性成分比例

---

### 3. 實性肺腫瘤 / 中晚期肺癌

**病變特徵**：
- HU 值：+30 ~ +100 HU
- 外觀：清晰的軟組織密度腫塊
- 密度：與軟組織相近，邊界明顯

**推薦設定（方案A：肺-縱隔混合窗）**：
```bash
python preprocess_for_yolo.py \
    --data_root ../../datasets/splited_dataset \
    --output_dir ../../datasets/preprocessed_yolo_solid \
    --imgsz 640 \
    --window_center -400 \
    --window_width 1800 \
    --enable_clahe \
    --clahe_clip 2.5 \
    --max_negative 20
```

**推薦設定（方案B：寬窗設定）**：
```bash
python preprocess_for_yolo.py \
    --data_root ../../datasets/splited_dataset \
    --output_dir ../../datasets/preprocessed_yolo_wide \
    --imgsz 640 \
    --window_center -300 \
    --window_width 2000 \
    --enable_clahe \
    --clahe_clip 2.5 \
    --max_negative 20
```

| 方案 | window_center | window_width | HU 範圍 | 優點 |
|-----|--------------|--------------|---------|------|
| **方案A** | -400 | 1800 | -1300 ~ +500 | 兼顧肺部和腫瘤對比度 |
| **方案B** | -300 | 2000 | -1300 ~ +700 | 涵蓋更廣，適合複雜病例 |

**適用場景**：
- ✅ 實性肺癌
- ✅ 鱗狀細胞癌
- ✅ 小細胞肺癌
- ✅ 評估腫瘤與周圍組織關係

---

### 4. 縱隔腫瘤 / 淋巴結轉移

**病變特徵**：
- HU 值：+20 ~ +60 HU
- 位置：縱隔、肺門淋巴結
- 密度：軟組織密度

**推薦設定**：
```bash
python preprocess_for_yolo.py \
    --data_root ../../datasets/splited_dataset \
    --output_dir ../../datasets/preprocessed_yolo_mediastinum \
    --imgsz 640 \
    --window_center 40 \
    --window_width 400 \
    --enable_clahe \
    --clahe_clip 2.0 \
    --max_negative 20
```

| 參數 | 數值 | 說明 |
|-----|------|------|
| `window_center` | 40 | 軟組織密度中心 |
| `window_width` | 400 | 涵蓋 -160 ~ +240 HU |
| `clahe_clip` | 2.0 | 標準增強 |

**適用場景**：
- ✅ 縱隔淋巴結腫大
- ✅ 胸腺瘤
- ✅ 淋巴瘤
- ✅ 評估 N 分期（淋巴結轉移）

---

### 5. 通用肺部腫瘤檢測（推薦）

**適用於各種類型的肺部腫瘤混合數據集**

**推薦設定**：
```bash
python preprocess_for_yolo.py \
    --data_root ../../datasets/splited_dataset \
    --output_dir ../../datasets/preprocessed_yolo_general \
    --imgsz 640 \
    --window_center -400 \
    --window_width 1800 \
    --enable_clahe \
    --clahe_clip 2.5 \
    --max_negative 20
```

**特點**：
- ✅ 涵蓋磨玻璃結節到實性腫瘤
- ✅ 可觀察腫瘤與縱隔的關係
- ✅ 適合多類型病變的混合數據集
- ✅ 平衡的對比度和細節保留

---

## CLAHE 參數調整

### CLAHE（對比度限制自適應直方圖均衡化）

**參數說明**：
```bash
--enable_clahe         # 啟用 CLAHE
--clahe_clip <數值>    # 對比度限制因子（1.0-4.0）
```

### Clip Limit 設定指南

| Clip 值 | 效果 | 適用場景 | 優缺點 |
|---------|------|---------|--------|
| **1.0** | 保守增強 | 高品質影像、磨玻璃結節 | ✅ 噪聲少<br>❌ 對比度提升有限 |
| **1.5-2.0** | 輕微增強 | 早期病變、細微結構 | ✅ 保留細節<br>✅ 噪聲可控 |
| **2.0-2.5** | 平衡增強 | **通用設定（推薦）** | ✅ 病灶清晰<br>✅ 對比度好 |
| **2.5-3.0** | 明顯增強 | 實性腫瘤、邊界定義 | ✅ 對比度高<br>⚠️ 可能引入噪聲 |
| **3.0-4.0** | 激進增強 | 低對比度影像 | ✅ 極高對比度<br>❌ 噪聲明顯 |

### 不同病變類型的 CLAHE 建議

```python
# 磨玻璃結節（保持細膩）
--clahe_clip 1.5

# 部分實性結節（平衡）
--clahe_clip 2.0

# 實性腫瘤（突出邊界）
--clahe_clip 2.5

# 縱隔腫瘤（標準增強）
--clahe_clip 2.0

# 低品質影像（需要更多增強）
--clahe_clip 3.0
```

---

## 實際應用建議

### 策略 1：單一窗位預處理（快速方案）

**適用情況**：
- 數據集病變類型相對單一
- 計算資源有限
- 需要快速開始訓練

**執行方式**：
根據主要病變類型選擇一組參數進行預處理。

---

### 策略 2：多窗位預處理（最佳方案）

**適用情況**：
- 數據集包含多種類型病變
- 需要最佳檢測性能
- 有充足的儲存空間

**實現方式**：

#### 選項 A：分別預處理
```bash
# 1. 肺窗版本
python preprocess_for_yolo.py \
    --data_root ../../datasets/splited_dataset \
    --output_dir ../../datasets/preprocessed_lung \
    --window_center -600 --window_width 1500

# 2. 混合窗版本
python preprocess_for_yolo.py \
    --data_root ../../datasets/splited_dataset \
    --output_dir ../../datasets/preprocessed_mixed \
    --window_center -400 --window_width 1800

# 3. 縱隔窗版本
python preprocess_for_yolo.py \
    --data_root ../../datasets/splited_dataset \
    --output_dir ../../datasets/preprocessed_mediastinum \
    --window_center 40 --window_width 400
```

#### 選項 B：修改程式碼支援多窗位輸出

修改 `preprocess_for_yolo.py` 生成多通道影像：

```python
# 偽代碼示意
windows = [
    {"name": "lung", "center": -600, "width": 1500},
    {"name": "mixed", "center": -400, "width": 1800},
    {"name": "mediastinum", "center": 40, "width": 400}
]

for window in windows:
    img = apply_hu_windowing(hu_array, window["center"], window["width"])
    save_image(img, f"{idx:06d}_{window['name']}.png")
```

---

### 策略 3：動態窗位選擇（進階方案）

**適用情況**：
- 需要根據病變位置自動選擇窗位
- 追求極致性能

**實現思路**：
1. 根據標註框位置判斷病變類型
2. 位於肺野 → 使用肺窗
3. 位於縱隔 → 使用縱隔窗
4. 跨越多區域 → 使用混合窗

---

## 參數快速查詢表

### 按病變類型查詢

| 病變類型 | center | width | HU 範圍 | clahe | 程式碼標籤 |
|---------|--------|-------|---------|-------|-----------|
| 磨玻璃結節 | -600 | 1500 | -1350~+150 | 1.5 | `_ggo` |
| 部分實性結節 | -400 | 1800 | -1300~+500 | 2.0 | `_mixed` |
| 實性腫瘤 | -400 | 1800 | -1300~+500 | 2.5 | `_solid` |
| 縱隔腫瘤 | 40 | 400 | -160~+240 | 2.0 | `_mediastinum` |
| **通用設定** | **-400** | **1800** | **-1300~+500** | **2.5** | `_general` |

### 按 HU 值範圍查詢

| 目標 HU 範圍 | 建議窗位 | 建議窗寬 | 適用病變 |
|-------------|---------|---------|---------|
| -1350 ~ +150 | -600 | 1500 | 磨玻璃結節、肺野 |
| -1300 ~ +500 | -400 | 1800 | 通用肺部腫瘤 |
| -1300 ~ +700 | -300 | 2000 | 複雜病例、多器官 |
| -160 ~ +240 | 40 | 400 | 縱隔、淋巴結 |
| -1000 ~ +1000 | 0 | 2000 | 全範圍（不推薦） |

### 按臨床分期查詢

| 臨床需求 | 主要窗位 | 輔助窗位 | 說明 |
|---------|---------|---------|------|
| **T 分期**<br>(腫瘤大小) | 肺窗<br>(-600/1500) | 混合窗<br>(-400/1800) | 評估腫瘤邊界和大小 |
| **N 分期**<br>(淋巴結) | 縱隔窗<br>(40/400) | - | 評估淋巴結腫大 |
| **M 分期**<br>(遠端轉移) | 骨窗<br>(400/1800) | 腹部窗<br>(50/350) | 評估骨/肝轉移 |
| **早期篩查** | 肺窗<br>(-600/1500) | - | 偵測小結節 |

---

## 常見問題

### Q1: 如何選擇窗位窗寬？

**A**: 根據您的主要檢測目標：
- 早期篩查 → 肺窗 (-600/1500)
- 通用腫瘤檢測 → 混合窗 (-400/1800) ⭐推薦
- 縱隔病變 → 縱隔窗 (40/400)

### Q2: CLAHE 一定要啟用嗎？

**A**: 強烈建議啟用：
- ✅ 醫學影像對比度通常較低
- ✅ CLAHE 能顯著提升病灶可見性
- ✅ 有助於 YOLO 模型學習特徵
- ⚠️ 但要注意 clip 值不要設太高

### Q3: 為什麼不直接使用最寬的窗位？

**A**: 窗位太寬會導致：
- ❌ 對比度降低，病灶不明顯
- ❌ 細微的密度差異被壓縮
- ❌ 模型難以學習有效特徵

### Q4: 可以在訓練時動態調整窗位嗎？

**A**: 可以作為數據增強策略：
```python
# 在訓練時隨機調整窗位（數據增強）
random_center = base_center + random.uniform(-100, 100)
random_width = base_width + random.uniform(-200, 200)
```

### Q5: 不同 CT 掃描儀的影響？

**A**: 不同廠牌/型號的 CT 可能有細微差異：
- 建議先用小批量數據測試
- 觀察預處理後的影像品質
- 必要時微調參數（±50-100 HU）

---

## 參考資料

### 標準窗位窗寬（臨床常用）

| 窗口名稱 | Center | Width | 用途 |
|---------|--------|-------|------|
| 肺窗 | -600 | 1500 | 肺實質、肺結節 |
| 縱隔窗 | 40 | 400 | 心臟、大血管、淋巴結 |
| 骨窗 | 400 | 1800 | 肋骨、脊椎、骨折 |
| 腹部窗 | 50 | 350 | 肝、脾、腎 |
| 腦窗 | 40 | 80 | 腦實質 |
| 骨窗（頭部） | 600 | 2800 | 顱骨、骨折 |

### HU 值參考範圍

```
組織密度從低到高：
-1000 ─── 空氣
 -900 ─┐
 -800  │  肺組織
 -700  │
 -600 ─┘
 -500 ─── 脂肪
    0 ─── 水
  +30 ─┐
  +40  │  軟組織
  +50  │  (肌肉、器官)
  +60 ─┘
 +100 ─── 血液/腫瘤
 +400 ─── 鈣化開始
+1000 ─── 骨骼
```

---

## 修改記錄

| 日期 | 版本 | 修改內容 |
|-----|------|---------|
| 2025-10-12 | 1.0 | 初始版本：完整的窗位窗寬設定指南 |

---

## 授權

本文件為 Chest CT Report Generator 專案的一部分。

---

**💡 提示**：建議將此文件與 `README_YOLO_PREPROCESS.md` 一起閱讀，以獲得完整的預處理流程理解。
