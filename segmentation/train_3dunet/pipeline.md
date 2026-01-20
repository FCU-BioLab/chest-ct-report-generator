# 3D U-Net Segmentation Pipeline

本文件詳細說明了 3D U-Net 肺結節分割模組的完整工作流程。該模組位於 `segmentation/train_3dunet` 目錄下。

## 總覽

整個 Pipeline 分為三個主要階段：
1. **數據預處理 (Preprocessing)**: 將原始 CT 數據 (LNDb/MSD) 轉換為標準化的 NPZ 視頻/體積格式。
2. **模型訓練 (Training)**: 使用 3D U-Net 模型在處理好的 NPZ 數據上進行訓練。
3. **推理與評估 (Inference/Evaluation)**: 在測試集上評估模型性能。

---

## 1. 數據預處理 (Preprocessing)

**目標**: 從原始 CT 掃描中提取以結節為中心的 3D 體積块 (Volumes)，並保存為 `.npz` 格式。

**代碼位置**: `preprocess.py`

**主要步驟**:
1. **讀取數據**: 讀取 LNDb 或 MSD Lung 的原始圖像 (`.mhd`/`.nii.gz`) 和標註。
2. **預處理**:
   - **Windowing**: 應用肺窗 (Window Center: -600, Width: 1500) 並歸一化到 0-255。
   - **Resampling**: 保持原始 spacing (代碼中似乎未顯式重採樣 spacing，但記錄了 spacing)。
   - **Cropping**: 根據標註的 BBox，提取以結節為中心的 Context Slices。
     - 默認 Context: ±6 slices (總深度 13 slices)。
   - **Resizing**: 將每張切片調整為 `image_size` (默認 256x256 或 512x512)。
   - **Filtering**: 過濾掉過小 (<4mm) 或位於邊緣的結節。
3. **保存**: 用戶劃分 (Train/Val/Test) 後保存為 NPZ 文件。
   - **包含內容**: `frames` (圖像), `masks` (標籤), `bbox` (提示框), `spacing`, `origin` 等。

**執行命令 (CMD)**:

```cmd
REM 轉換 LNDb 數據集
python segmentation/train_3dunet/main.py convert ^
    --dataset lndb ^
    --input_dir E:\lung_ct_lesion_dataset\LNDb ^
    --output_dir cache/volume_npz ^
    --context_slices 16 ^
    --image_size 256

REM 轉換 MSD Lung 數據集
python segmentation/train_3dunet/main.py convert ^
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
- **特點**: 使用 DoubleConv (Conv3D-BN-ReLU)，Maxpool3D 下採樣，Upsample/ConvTranspose3D 上採樣。

**數據加載 (`dataset.py`)**:
- 讀取預處理後的 `.npz` 文件。
- **Padding/Cropping**: 處理不同深度的結節，統一裁切或 Padding 到 `max_depth` (默認 32)。
- **Augmentation**: 訓練時進行隨機翻轉 (Horizontal/Vertical Flip)。

**訓練流程 (`trainer.py`)**:
- **Loss**: 混合損失 (BCE Loss + Dice Loss)。
- **Optimizer**: AdamW。
- **Scheduler**: 支持 Warmup (在 Config 中定義)。
- **監控**: 每個 Epoch 計算 Validation Set 的 Dice Score，保存最佳模型。

**執行命令 (CMD)**:

```cmd
python segmentation/train_3dunet/main.py train ^
    --npz_dir cache/volume_npz ^
    --epochs 50 ^
    --batch_size 4 ^
    --max_depth 32 ^
    --base_filters 32 ^
    --image_size 256 ^
    --device cuda
```

**配置 (`config.py`)**:
- 所有超參數可以在 `config.py` 中調整，主要參數也可通過 CLI 覆蓋。

---

## 3. 測試與評估 (Evaluation)

**目標**: 加載訓練好的權重，評估測試集性能。

**主要步驟**:
1. 加載 Checkpoint。
2. 遍歷 Test Set。
3. 計算 Dice Score。

**執行命令 (CMD)**:

```cmd
python segmentation/train_3dunet/main.py test ^
    --checkpoint volume_output_unet3d/checkpoints/best_model.pt ^
    --npz_dir cache/volume_npz ^
    --image_size 256
```

---

## 4. 統計分析 (Statistics)

查看數據集的分佈情況（樣本數、平均深度等）。

```cmd
python segmentation/train_3dunet/main.py stats --npz_dir cache/volume_npz
```

## 目錄結構

```
segmentation/train_3dunet/
├── main.py          # 入口程式 (CLI)
├── config.py        # 配置定義
├── preprocess.py    # 數據預處理 (ETL)
├── dataset.py       # Pytorch Dataset
├── model.py         # 3D U-Net 模型定義
├── trainer.py       # 訓練與驗證邏輯
└── README.md        # 說明文件
```
