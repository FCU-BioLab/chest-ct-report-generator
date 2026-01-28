# 3D U-Net Nodule Detection Pipeline (肺結節偵測流程)

本文件詳細說明了 3D U-Net 肺結節偵測模組的完整工作流程。

## 總覽

本模組不僅進行體積分割，更核心的目標是**輸出結構化的病灶清單**。流程分為四個階段：
1. **數據預處理 (Preprocessing)**: 轉換 CT 數據為標準化 NPZ，並保留關鍵的空間 Metadata (Spacing, Origin, Slice Indices)。
2. **模型訓練 (Training)**: 使用 3D U-Net 學習結節的體積特徵。
3. **偵測與後處理 (Detection & Post-processing)**: 將分割機率圖轉換為具備真實物理座標與解剖位置的病灶清單。
4. **評估 (Evaluation)**: 使用 Precision, Recall, F1 Score 評估偵測性能。

---

## 1. 數據預處理 (Preprocessing)

**目標**: 提取以結節為中心的 3D 體積块 (Volumes)，並嵌入絕對空間定位資訊。

**代碼位置**: `preprocess.py`

**關鍵增強 (Key Enhancements)**:
- 舊版僅保存圖像與 Mask。
- **新版** 額外保存：
    - `origin` & `spacing`: 原始 CT 的物理坐標原點與像素間距 (用於計算真實世界座標)。
    - **支援 Full Volume**: 可選擇轉換完整 CT 掃描而不進行裁切 (`--full_volume`)。
    - **醫師一致性篩選 (Agreement Filtering)**: 
        - 根據 `trainNodules_gt.csv` 中的 `AgrLevel` 篩選病灶。
        - **Mask Fusion**: 若多位醫師標記同一病灶，會將所有 Mask 進行聯集 (Union) 產生更完整的 Ground Truth。
    - **Data Log**: 轉換結束後會生成 `data.log` 記錄詳細統計與配置。

**主要內容**:
- **Windowing**: 肺窗 (-600, 1500)。
- **Sampling**: 提取 Context Slices (默認 ±16，總長 33)。
- **Filtering**: 根據直徑 (`min_diameter`) 與 醫師一致性 (`min_agreement`) 篩選。
- **Output**: 保存為 `.npz`，包含 `frames`, `masks`, `spacing`, `origin`, `slice_indices`, `agreement`。

**執行命令 (CMD)**:

```cmd
REM 轉換 LNDb 數據集 (使用 Full Volume, 篩選至少 2 位醫師同意)
python -m detection.train_3dunet.main convert ^
    --dataset lndb ^
    --input_dir E:\lung_ct_lesion_dataset\LNDb ^
    --output_dir cache/volume_npz ^
    --full_volume ^
    --min_agreement 2 ^
    --image_size 256

REM 轉換 MSD Lung 數據集 (標準模式)
python -m detection.train_3dunet.main convert ^
    --dataset msd ^
    --input_dir E:\lung_ct_lesion_dataset\Task06_Lung ^
    --output_dir cache/volume_npz ^
    --context_slices 32
```

---

```

---

## 2. 數據檢查 (Data Inspection)

**目標**: 在訓練前驗證轉換後的數據是否正確，以及模型 Loader 是否能正確讀取。

**執行命令 (CMD)**:

```cmd
REM 1. 檢查數據集統計
python -m detection.train_3dunet.main check_data ^
    --npz_dir cache/volume_npz ^
    --split train ^
    --mode stats

REM 2. 交互式瀏覽 (Interactive Browse)
python -m detection.train_3dunet.main check_data ^
    --npz_dir cache/volume_npz ^
    --split train ^
    --mode browse

REM 3. 檢查 Dataset Loader 轉換效果 (重要)
python -m detection.train_3dunet.main check_data ^
    --npz_dir cache/volume_npz ^
    --split train ^
    --mode dataset_batch ^
    --max_depth 32
```

---

## 3. 模型訓練 (Training)

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
- **Dynamic Splitting**: 在訓練時根據 `seed` 和 `ratios` 動態分割數據集，無需在轉換階段預分割。
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
python -m detection.train_3dunet.main train ^
    --npz_dir cache/volume_npz ^
    --epochs 50 ^
    --batch_size 4 ^
    --max_depth 32 ^
    --image_size 256 ^
    --train_ratio 0.8 ^
    --val_ratio 0.1 ^
    --test_ratio 0.1 ^
    --split_seed 42

REM 啟用 Attention 模型 + Combined Loss
python -m detection.train_3dunet.main train ^
    --npz_dir cache/volume_npz ^
    --epochs 100 ^
    --attention ^
    --loss_type combined
```

---

## 4. 偵測與推理 (Detection & Inference)

**目標**: 將模型的機率輸出轉換為臨床可用的結構化報告。

**核心組件**: `detector.py` (`NoduleDetector`), `location_estimator.py`

**處理流程**:

1. **模型推論**: 3D U-Net 輸出機率熱圖 (Probability Map)。
2. **NoduleDetector 後處理**:
   - **閾值切分**: `prob > det_prob_threshold` (建議 0.9)。
   - **形態學閉運算 (Closing)**: 填補結節內部的孔洞。
   - **連通區域標記 (CCL)**: 識別獨立的結節候選者。
   - **尺寸過濾**: 移除體積小於 `det_min_size` (建議 5.0 mm³) 的雜訊。
   - **幾何計算**: 計算質心 (Centroid)、體積 (Volume)、直徑 (Diameter)。
3. **空間定位**:
   - 使用 `metadata` (Origin, Spacing) 將像素座標轉換為 **世界座標 (World Coordinates)**。
   - 調用 `LungLocationEstimator` 判斷結節所在的 **肺葉 (Lobe)** (如 RUL, LLL)。
4. **輸出生成**:
   - 匯總所有資訊生成 `DetectedNodule` 物件。

**輸出格式 (`detections.json`)**:

```json
[
  {
    "id": 1,
    "probability": 0.985,
    "location": {
      "lobe": "RUL",
      "description": "Right Upper Lobe"
    },
    "geometry": {
      "centroid_xyz_mm": [120.5, -80.2, -150.0],
      "volume_mm3": 45.2,
      "diameter_mm": 5.2
    },
    "slice_range": [45, 52]
  }
]
```

**執行命令 (CMD)**:

```cmd
python -m detection.train_3dunet.main fulltest ^
    --checkpoint path\to\model.pth ^
    --npz_dir cache/volume_npz ^
    --det_prob_threshold 0.9 ^
    --det_min_size 5.0
```
---

## 5. 快速測試 (Quick Test)

**目標**: 快速評估模型 Dice Score。

```cmd
python -m detection.train_3dunet.main test ^
    --checkpoint path\to\best_model.pth ^
    --npz_dir cache/volume_npz ^
    --split test ^
    --attention
```

---

## 6. 統計分析 (Statistics)

查看數據集的分佈情況（樣本數、平均深度等）。

```cmd
python -m detection.train_3dunet.main stats --npz_dir cache/volume_npz
```

---

## 7. 視覺化預測 (Visualize)

生成預測對比 GT 的視覺化圖像（不計算 Detection 指標）。

```cmd
python -m detection.train_3dunet.main visualize ^
    --checkpoint path\to\model.pth ^
    --npz_dir cache/volume_npz ^
    --split test ^
    --attention
```

---

## 目錄結構

```
detection/train_3dunet/
├── main.py            # 入口程式 (CLI)
├── config.py          # 配置定義 (含 Detection 參數)
├── preprocess.py      # 數據預處理 (含 Metadata 提取)
├── dataset.py         # Pytorch Dataset
├── model.py           # 3D U-Net 模型定義
├── buildingblocks.py  # 模型組件
├── trainer.py         # 訓練與測試流程
├── detector.py        # [NEW] 結節偵測邏輯
├── location_estimator.py # [NEW] 肺葉定位
├── visualize.py       # 視覺化工具
├── pipeline.md        # 本文件
└── README.md          # 快速入門
```
