# YOLO 標註視覺化檢查工具使用說明

## 📋 功能簡介

`visualize_yolo_annotations.py` 是一個用於檢查預處理後 YOLO 標註框的視覺化工具，可以幫助您：

- ✅ 驗證標註框是否正確對應到圖像上
- ✅ 檢查標註框的位置、大小是否合理
- ✅ 統計正負樣本分布
- ✅ 查看類別分布情況
- ✅ 互動式瀏覽標註數據

---

## 🚀 快速開始

### 基本用法

```bash
# 檢查 train 分割的 10 個樣本
python visualize_yolo_annotations.py \
    --data_dir ../../datasets/preprocessed_yolo \
    --split train \
    --num_samples 10
```

### 檢查所有分割

```bash
# 檢查 train/val/test 所有分割
python visualize_yolo_annotations.py \
    --data_dir ../../datasets/preprocessed_yolo \
    --split all \
    --num_samples 20
```

### 儲存視覺化結果

```bash
# 將視覺化結果儲存到資料夾
python visualize_yolo_annotations.py \
    --data_dir ../../datasets/preprocessed_yolo \
    --split train \
    --num_samples 50 \
    --output_dir ./visualization_output
```

### 互動式檢查模式

```bash
# 啟用互動模式，可以用鍵盤切換圖像
python visualize_yolo_annotations.py \
    --data_dir ../../datasets/preprocessed_yolo \
    --split train \
    --interactive
```

---

## 📚 完整參數說明

| 參數 | 說明 | 預設值 | 範例 |
|-----|------|-------|------|
| `--data_dir` | 預處理後的資料目錄 | **必需** | `../../datasets/preprocessed_yolo` |
| `--split` | 要檢查的資料分割 | `train` | `train`, `val`, `test`, `all` |
| `--num_samples` | 要視覺化的樣本數量 | `10` | `20`, `50`, `100` |
| `--output_dir` | 視覺化結果輸出目錄 | `None` | `./vis_output` |
| `--no_random` | 不隨機抽樣，按順序選擇 | `False` | 加上此參數 |
| `--interactive` | 啟用互動式檢查模式 | `False` | 加上此參數 |
| `--no_show` | 不顯示圖像，只儲存 | `False` | 加上此參數 |

---

## 🎮 互動模式操作

啟用互動模式後，可以使用以下按鍵操作：

| 按鍵 | 功能 |
|-----|------|
| `n` 或 `→` | 下一張圖像 |
| `p` 或 `←` | 上一張圖像 |
| `r` | 隨機跳轉到某一張 |
| `s` | 儲存當前視覺化結果 |
| `q` 或 `ESC` | 退出互動模式 |

**範例**：
```bash
python visualize_yolo_annotations.py \
    --data_dir ../../datasets/preprocessed_yolo \
    --split train \
    --interactive
```

---

## 📊 輸出內容

### 1. 視覺化圖像

每張視覺化圖像包含：

```
┌─────────────────────────────────────────┐
│  檔案: 000123.png                        │
│  分割: TRAIN                             │
│  尺寸: 640 x 640                         │
│  標註框: 3 個                            │
│  類別: lesion                            │
├─────────────────────────────────────────┤
│                                         │
│     [原始圖像 + 標註框]                  │
│                                         │
│     - 彩色邊界框                         │
│     - 中心點標記                         │
│     - 類別標籤 (#1 lesion)               │
│                                         │
└─────────────────────────────────────────┘
```

### 2. 統計資訊

程式會輸出詳細的統計資訊：

```
================================================================================
📈 TRAIN 資料集統計
================================================================================
總圖像數:       1000
已檢查圖像:     50
  - 有標註框:   28 (56.0%)
  - 無標註框:   22 (44.0%)
總標註框數:     85
平均框數/圖:    3.04

類別分布:
  - lesion: 85 (100.0%)
```

### 3. JSON 統計文件

如果指定了 `--output_dir`，會生成 JSON 統計文件：

```json
{
  "total_images": 1000,
  "checked_images": 50,
  "images_with_boxes": 28,
  "images_without_boxes": 22,
  "total_boxes": 85,
  "class_distribution": {
    "lesion": 85
  }
}
```

---

## 💡 使用場景

### 場景 1：快速驗證預處理結果

在完成資料預處理後，快速檢查標註是否正確：

```bash
# 隨機檢查 20 個樣本
python visualize_yolo_annotations.py \
    --data_dir ../../datasets/preprocessed_yolo \
    --split train \
    --num_samples 20
```

### 場景 2：詳細檢查所有分割

仔細檢查 train/val/test 的標註品質：

```bash
# 檢查所有分割並儲存結果
python visualize_yolo_annotations.py \
    --data_dir ../../datasets/preprocessed_yolo \
    --split all \
    --num_samples 30 \
    --output_dir ./full_check
```

### 場景 3：手動逐張檢查

需要手動審核每張圖像的標註：

```bash
# 互動模式逐張檢查
python visualize_yolo_annotations.py \
    --data_dir ../../datasets/preprocessed_yolo \
    --split train \
    --interactive
```

### 場景 4：批次生成報告

為整個資料集生成視覺化報告：

```bash
# 批次處理，不顯示圖像
python visualize_yolo_annotations.py \
    --data_dir ../../datasets/preprocessed_yolo \
    --split all \
    --num_samples 100 \
    --output_dir ./report \
    --no_show
```

### 場景 5：檢查特定樣本

檢查前 N 個樣本（不隨機）：

```bash
# 按順序檢查前 50 個樣本
python visualize_yolo_annotations.py \
    --data_dir ../../datasets/preprocessed_yolo \
    --split train \
    --num_samples 50 \
    --no_random
```

---

## 🔍 視覺化效果說明

### 標註框顏色

- 每個類別使用不同的顏色
- 顏色通過 HSV 色彩空間均勻分布
- 預設情況下，單類別（lesion）使用紅色

### 標註框元素

1. **邊界框**：矩形框標示病灶範圍
2. **中心點**：實心圓點標示框的中心位置
3. **類別標籤**：顯示框的編號和類別名稱
4. **背景色塊**：標籤文字的深色背景，提高可讀性

### 資訊面板

頂部資訊面板顯示：
- 檔案名稱
- 資料分割（TRAIN/VAL/TEST）
- 圖像尺寸
- 標註框數量
- 類別列表

---

## 🛠️ 常見問題

### Q1: 沒有安裝 OpenCV 怎麼辦？

**A**: 需要安裝 opencv-python：

```bash
pip install opencv-python
```

或者使用 requirements.txt：

```bash
pip install -r requirements.txt
```

### Q2: 圖像顯示太小/太大？

**A**: 可以在程式碼中調整顯示尺寸，或使用互動模式時調整視窗大小。

### Q3: 標註框位置不對？

**A**: 可能的原因：
1. YOLO 座標格式錯誤（應為歸一化座標）
2. 圖像尺寸與預處理時不一致
3. 標註文件與圖像不對應

檢查方法：
- 查看 `metadata.json` 確認預處理參數
- 確認圖像尺寸是否一致
- 手動檢查 `.txt` 標註文件內容

### Q4: 顯示的圖像是空的？

**A**: 檢查：
1. 圖像路徑是否正確
2. 圖像文件是否損壞
3. 是否有對應的標註文件

### Q5: 想要檢查所有圖像而不只是抽樣？

**A**: 將 `--num_samples` 設為很大的數字：

```bash
python visualize_yolo_annotations.py \
    --data_dir ../../datasets/preprocessed_yolo \
    --split train \
    --num_samples 10000
```

---

## 📝 輸出目錄結構

使用 `--output_dir` 後的目錄結構：

```
visualization_output/
├── train/
│   ├── 000001.png
│   ├── 000005.png
│   └── ...
├── val/
│   ├── 000002.png
│   └── ...
├── test/
│   └── ...
├── train_stats.json
├── val_stats.json
└── test_stats.json
```

---

## 🎯 最佳實踐

### 1. 預處理後立即檢查

```bash
# 完成預處理
python preprocess_for_yolo.py \
    --data_root ../../datasets/splited_dataset \
    --output_dir ../../datasets/preprocessed_yolo

# 立即檢查
python visualize_yolo_annotations.py \
    --data_dir ../../datasets/preprocessed_yolo \
    --split all \
    --num_samples 30
```

### 2. 分階段檢查

```bash
# 階段 1: 快速檢查每個分割
python visualize_yolo_annotations.py --data_dir ... --split train --num_samples 10
python visualize_yolo_annotations.py --data_dir ... --split val --num_samples 10
python visualize_yolo_annotations.py --data_dir ... --split test --num_samples 10

# 階段 2: 如果發現問題，詳細檢查
python visualize_yolo_annotations.py --data_dir ... --split train --interactive
```

### 3. 生成檢查報告

```bash
# 生成完整的視覺化報告
python visualize_yolo_annotations.py \
    --data_dir ../../datasets/preprocessed_yolo \
    --split all \
    --num_samples 100 \
    --output_dir ./inspection_report \
    --no_show

# 檢查報告
ls ./inspection_report
```

---

## 🔗 相關文件

- **預處理工具**：`preprocess_for_yolo.py` - 將 DICOM 轉換為 YOLO 格式
- **窗位設定指南**：`WINDOWING_SETTINGS_GUIDE.md` - HU 窗位窗寬設定說明
- **預處理說明**：`README_YOLO_PREPROCESS.md` - 預處理流程文檔

---

## 📧 技術支援

如果遇到問題：

1. 檢查錯誤訊息
2. 確認資料目錄結構正確
3. 驗證標註文件格式
4. 查看相關文檔

---

**最後更新**: 2025-10-12
