# YOLO 標註視覺化工具 - 修復說明

## 🔧 修復內容

### 問題
OpenCV GUI 不可用的錯誤：
```
cv2.error: The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support.
```

### 解決方案
已將顯示後端從 OpenCV 改為 **matplotlib**，並添加鍵盤事件支持。

---

## ✅ 修復後的功能

### 1. 自動檢測並使用 Matplotlib
程序會自動檢測 matplotlib 是否可用，並優先使用它來顯示圖像。

### 2. 鍵盤操作（互動模式）
在 matplotlib 視窗中，可以直接使用鍵盤操作：

| 按鍵 | 功能 |
|-----|------|
| `n` 或 `→` | 下一張 |
| `p` 或 `←` | 上一張 |
| `r` | 隨機跳轉 |
| `s` | 儲存當前圖像 |
| `q` 或 `ESC` | 退出 |
| **關閉視窗** | 自動下一張 |

### 3. 無需命令行輸入
不再需要在視窗關閉後手動輸入命令，所有操作都通過鍵盤完成。

---

## 🚀 使用方法

### 互動模式（推薦）
```bash
python visualize_yolo_annotations.py \
    --data_dir ../../datasets/preprocessed_yolo_tumor \
    --split train \
    --interactive
```

**操作流程**：
1. 程序會顯示第一張圖像
2. 在 matplotlib 視窗中按相應按鍵
3. 視窗會自動關閉並顯示下一張
4. 重複步驟 2-3

### 批次檢查模式
```bash
# 檢查 20 個樣本並儲存
python visualize_yolo_annotations.py \
    --data_dir ../../datasets/preprocessed_yolo_tumor \
    --split train \
    --num_samples 20 \
    --output_dir ./check_results
```

### 快速檢查（不顯示視窗）
```bash
# 只生成統計，不顯示圖像
python visualize_yolo_annotations.py \
    --data_dir ../../datasets/preprocessed_yolo_tumor \
    --split all \
    --num_samples 50 \
    --no_show \
    --output_dir ./stats_only
```

---

## 💡 使用技巧

### 技巧 1: 快速瀏覽
- 使用 `n` 或 `→` 快速瀏覽所有圖像
- 發現問題時按 `s` 儲存
- 使用 `r` 隨機跳轉檢查分布

### 技巧 2: 詳細檢查
- 使用 `p` 和 `n` 來回對比
- 儲存有問題的圖像以便後續分析

### 技巧 3: 多分割檢查
```bash
# 依次檢查各分割
python visualize_yolo_annotations.py --data_dir ... --split train --interactive
python visualize_yolo_annotations.py --data_dir ... --split val --interactive
python visualize_yolo_annotations.py --data_dir ... --split test --interactive
```

---

## 🔍 檢查清單

使用這個工具檢查以下內容：

- [ ] 標註框位置是否正確對應到病灶
- [ ] 標註框大小是否合理
- [ ] 是否有漏標的病灶
- [ ] 是否有誤標的區域
- [ ] 負樣本（無標註框）是否確實沒有病灶
- [ ] 圖像品質是否良好
- [ ] 窗位窗寬設定是否合適
- [ ] 不同分割的樣本分布是否均衡

---

## 📊 輸出示例

### 終端輸出
```
================================================================================
🔍 YOLO 標註視覺化檢查工具
================================================================================
資料目錄: ../../datasets/preprocessed_yolo_tumor
類別數量: 1
類別名稱: lesion

================================================================================
🎮 互動式檢查模式
================================================================================
資料分割: TRAIN
圖像數量: 23977
顯示方式: Matplotlib (互動模式)

操作說明:
  - 按 'n' 或 '→' 或關閉視窗: 下一張
  - 按 'p' 或 '←': 上一張
  - 按 'r': 隨機跳轉
  - 按 's': 儲存當前圖像
  - 按 'q' 或 ESC: 退出
================================================================================

🎲 隨機跳轉到第 5432 張
💾 已儲存: ./visualization_saves/train/saved_012345.png
👋 退出互動模式
```

### 視覺化圖像
每張圖像包含：
- 頂部資訊面板（深灰色背景）
  - 檔案名稱
  - 資料分割
  - 圖像尺寸
  - 標註框數量
  - 類別列表
- 中間主圖像區域
  - 彩色標註框
  - 中心點標記
  - 類別標籤（#1 lesion, #2 lesion...）
- 底部操作說明

---

## ⚠️ 注意事項

### 1. Matplotlib 後端
如果遇到顯示問題，可能需要設定 matplotlib 後端：
```python
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
```

### 2. 記憶體使用
互動模式會逐張載入圖像，記憶體使用量較低。但如果圖像很大，可能需要注意。

### 3. 儲存的圖像
使用 `s` 鍵儲存的圖像會放在 `./visualization_saves/<split>/` 目錄下。

### 4. 大量圖像檢查
對於 23977 張圖像，建議：
- 先使用批次模式抽樣檢查（`--num_samples 50`）
- 發現問題後使用互動模式詳細檢查
- 使用隨機跳轉功能確保檢查分布均勻

---

## 🐛 故障排除

### 問題 1: matplotlib 視窗無法顯示
**解決**：
```bash
pip install --upgrade matplotlib
# 或嘗試不同的後端
```

### 問題 2: 鍵盤操作無反應
**解決**：
- 確保 matplotlib 視窗是當前活動視窗
- 嘗試點擊視窗後再按鍵

### 問題 3: 視窗顯示後立即關閉
**解決**：
- 這通常不是問題，是正常的下一張行為
- 如果想停留查看，不要按任何鍵

### 問題 4: 圖像顯示不清晰
**解決**：
- 調整 figsize 參數（在程式碼中）
- 使用全螢幕模式查看

---

## 📝 後續改進建議

1. **添加縮放功能**：支持滑鼠滾輪縮放查看細節
2. **添加對比模式**：同時顯示原圖和標註圖
3. **添加過濾功能**：只顯示有標註的圖像或無標註的圖像
4. **批次標記**：標記有問題的圖像，最後統一處理
5. **統計圖表**：顯示標註框大小分布、類別分布等

---

**最後更新**: 2025-10-12
**狀態**: ✅ 已修復 OpenCV GUI 問題
