# 3D U-Net Segmentation Pipeline

本文件詳細說明了 3D U-Net 肺結節分割模組的完整工作流程。該模組位於 `segmentation/train_3dunet` 目錄下。

## 總覽

整個 Pipeline 分為四個主要階段：
1. **數據預處理 (Preprocessing)**: 將原始 CT 數據 (LNDb/MSD) 轉換為標準化的 NPZ 視頻/體積格式。
2. **模型訓練 (Training)**: 使用 3D U-Net 模型在處理好的 NPZ 數據上進行訓練。
3. **完整測試 (Full Test)**: 在測試集上評估模型性能，生成視覺化與 GIF 動畫。
4. **推理與評估 (Inference/Evaluation)**: 快速評估模型 Dice Score。

---

## 1. 數據預處理 (Preprocessing)

**目標**: 從原始 CT 掃描中提取以結節為中心的 3D 體積块 (Volumes)，並保存為 `.npz` 格式。

**代碼位置**: `preprocess.py`

**主要步驟**:
1. **讀取數據**: 讀取 LNDb 或 MSD Lung 的原始圖像 (`.mhd`/`.nii.gz`) 和標註。
2. **預處理**:
   - **Windowing**: 應用肺窗 (Window Center: -600, Width: 1500) 並歸一化到 0-255。
   - **Cropping**: 根據標註的 BBox，提取以結節為中心的 Context Slices。
     - 默認 Context: ±16 slices (總深度 33 slices)。
   - **Resizing**: 將每張切片調整為 `image_size` (默認 256x256)。
   - **Filtering**: 過濾掉過小 (<4mm) 或位於邊緣的結節。
3. **保存**: 劃分 (Train/Val/Test) 後保存為 NPZ 文件。
   - **包含內容**: `frames` (圖像), `masks` (標籤), `bbox` (提示框), `spacing`, `origin` 等。

**執行命令 (CMD)**:

```cmd
REM 轉換 LNDb 數據集
python -m segmentation.train_3dunet.main convert ^
    --dataset lndb ^
    --input_dir E:\lung_ct_lesion_dataset\LNDb ^
    --output_dir cache/volume_npz ^
    --context_slices 16 ^
    --image_size 256

REM 轉換 MSD Lung 數據集
python -m segmentation.train_3dunet.main convert ^
    --dataset msd ^
    --input_dir E:\lung_ct_lesion_dataset\Task06_Lung ^
    --output_dir cache/volume_npz
```

---

## 2. 模型訓練 (Training)

**目標**: 訓練 3D U-Net 模型以分割肺結節。

**代碼位置**: `trainer.py`, `model.py`, `dataset.py`

**模型架構 (`model.py`)**:
- 標準 **3D U-Net** 架構。
- **Input**: (B, 1, D, H, W) - 單通道 3D 體積。
- **Output**: (B, 1, D, H, W) - Logits (二分類)。
- **可選功能**:
  - `--attention`: 啟用 SE Block + Attention Gate 提升分割精度。

**數據加載 (`dataset.py`)**:
- 讀取預處理後的 `.npz` 文件。
- **Padding/Cropping**: 處理不同深度的結節，統一到 `max_depth` (默認 32)。
- **Augmentation**: 訓練時進行隨機翻轉 (Horizontal/Vertical/Depth Flip)。

**訓練流程 (`trainer.py`)**:
- **Loss 選項** (`--loss_type`):
  - `combined` (預設): Tversky Loss + Boundary Loss + BCE Loss
  - `tversky`: Tversky Loss (可調 FP/FN 權重)
  - `dice`: 傳統 Dice Loss + BCE + Focal Loss
- **Optimizer**: AdamW
- **Scheduler**: OneCycleLR
- **監控**: 每個 Epoch 計算 Validation Set 的 Dice Score，保存最佳模型。

**執行命令 (CMD)**:

```cmd
REM 基礎訓練
python -m segmentation.train_3dunet.main train ^
    --npz_dir cache/volume_npz ^
    --epochs 50 ^
    --batch_size 4 ^
    --max_depth 32 ^
    --image_size 256

REM 啟用 Attention 模型 + Combined Loss
python -m segmentation.train_3dunet.main train ^
    --npz_dir cache/volume_npz ^
    --epochs 100 ^
    --attention ^
    --loss_type combined
```

---

## 3. 完整測試 (Full Test)

**目標**: 評估模型並生成詳細視覺化報告。

**代碼位置**: `trainer.py` (`comprehensive_test` 方法)

**輸出內容**:
- `test_results.json`: 每個樣本的 Segmentation + Detection 指標
- `test_summary.png`: 彙總統計圖
- 每個 Case 資料夾:
  - `overview.png`: 8 面板總覽
  - `overlay.png`: GT vs Pred 疊加圖
  - `animation.gif`: 所有切片的 Overlay GIF 動畫
  - `slices/`: 逐切片圖像
  - `stats.json`: 該樣本的指標

**執行命令 (CMD)**:

```cmd
REM 完整測試 (含視覺化 + GIF)
python -m segmentation.train_3dunet.main fulltest ^
    --npz_dir cache/volume_npz ^
    --checkpoint segmentation\video_result\3dunet_train_XXXXXX\best_model.pth ^
    --split test ^
    --attention

REM 跳過 GIF 輸出 (加快速度)
python -m segmentation.train_3dunet.main fulltest ^
    --checkpoint path\to\model.pth ^
    --npz_dir cache/volume_npz ^
    --no_gif

REM 只計算指標，不生成視覺化
python -m segmentation.train_3dunet.main fulltest ^
    --checkpoint path\to\model.pth ^
    --npz_dir cache/volume_npz ^
    --no_viz
```

---

## 4. 快速測試 (Quick Test)

**目標**: 快速評估模型 Dice Score。

```cmd
python -m segmentation.train_3dunet.main test ^
    --checkpoint path\to\best_model.pth ^
    --npz_dir cache/volume_npz ^
    --split test ^
    --attention
```

---

## 5. 統計分析 (Statistics)

查看數據集的分佈情況（樣本數、平均深度等）。

```cmd
python -m segmentation.train_3dunet.main stats --npz_dir cache/volume_npz
```

---

## 6. 視覺化預測 (Visualize)

生成預測對比 GT 的視覺化圖像（不計算 Detection 指標）。

```cmd
python -m segmentation.train_3dunet.main visualize ^
    --checkpoint path\to\model.pth ^
    --npz_dir cache/volume_npz ^
    --split test ^
    --attention
```

---

## 目錄結構

```
segmentation/train_3dunet/
├── main.py          # 入口程式 (CLI)
├── config.py        # 配置定義
├── preprocess.py    # 數據預處理 (ETL)
├── dataset.py       # Pytorch Dataset
├── model.py         # 3D U-Net 模型定義
├── buildingblocks.py # 模型組件 (DoubleConv, SE, Attention Gate)
├── trainer.py       # 訓練、驗證、測試邏輯
├── visualize.py     # NPZ 數據視覺化工具
├── pipeline.md      # Pipeline 說明 (本文件)
└── README.md        # 快速入門
```
