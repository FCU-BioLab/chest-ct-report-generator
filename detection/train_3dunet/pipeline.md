# 3D U-Net 肺結節偵測 Pipeline

本文件說明 3D U-Net 肺結節偵測模組的完整工作流程。

## 總覽

```
CT 原始影像 → 預處理 (NPZ) → 模型訓練 → 推理 + 後處理 → 結構化偵測報告
```

四個主要階段：

1. **數據預處理** — 轉換 CT (`.mhd`/`.nii.gz`) 為標準化 `.npz`，保留空間 Metadata
2. **模型訓練** — 3D U-Net 學習結節的體積特徵
3. **偵測與後處理** — 機率圖 → 連通區域分析 → 結構化病灶清單
4. **評估** — Dice/IoU (分割) + Precision/Recall/F1 (偵測)

---

## 1. 數據預處理

**代碼**: `preprocess.py` — `VolumePreprocessor` 類別

### 處理流程

```
CT + Mask → Windowing → 肺部裁切 → 逐結節擷取 → Resize → .npz
```

1. **Windowing**: 肺窗 (WC=-600, WW=1500) 正規化到 [0, 255]
2. **肺部裁切**: 自動分割肺部區域 (`segmentation.py`)，裁去體外空氣
3. **逐結節處理** (`_process_single_finding`):
   - 擷取以結節為中心的 Z 軸範圍 (±`context_slices`)
   - `_isolate_finding_mask` — 連通區域分析，分離目標結節
   - 計算體積 (mm³) 與等效直徑 (mm)
   - 按 `min_diameter` 與 `min_agreement` 篩選
   - Pad-to-square + Resize 到 `image_size`
   - 計算 2D Bounding Box
4. **輸出**: `.npz` 檔案，包含以下欄位：
   - `frames` / `masks` — (D, H, W) uint8
   - `spacing` / `origin` — 物理空間參數
   - `slice_indices` — 原始切片索引
   - `diameter_mm` / `volume_mm3` / `agreement`
   - `bbox` / `padding_info` / `lung_crop_bbox`

### 支援的數據集

| 數據集 | 指令 | 特點 |
|--------|------|------|
| LNDb | `--dataset lndb` | 多醫師標記，支援 Agreement 篩選 |
| MSD Lung | `--dataset msd` | Task06_Lung，NIfTI 格式 |

### 執行命令

```cmd
REM LNDb 轉換 (Full Volume, 至少 2 位醫師同意)
python -m detection.train_3dunet.main convert ^
    --dataset lndb ^
    --input_dir E:\lung_ct_lesion_dataset\LNDb ^
    --output_dir cache/lndb_volume_npz_agr1 ^
    --full_volume ^
    --min_agreement 2 ^
    --image_size 256

REM MSD Lung 轉換
python -m detection.train_3dunet.main convert ^
    --dataset msd ^
    --input_dir E:\lung_ct_lesion_dataset\Task06_Lung ^
    --output_dir cache/msd_volume_npz ^
    --context_slices 32
```

---

## 2. 數據檢查

在訓練前驗證轉換後的數據。

```cmd
REM 統計概覽
python -m detection.train_3dunet.main check_data ^
    --npz_dir cache/lndb_volume_npz_agr1 --mode stats

REM 交互式瀏覽
python -m detection.train_3dunet.main check_data ^
    --npz_dir cache/lndb_volume_npz_agr1 --mode browse

REM 驗證 DataLoader 輸出
python -m detection.train_3dunet.main check_data ^
    --npz_dir cache/lndb_volume_npz_agr1 --mode dataset_batch
```

---

## 3. 模型訓練

**代碼**: `trainer.py` (`UNet3DTrainer`), `model.py`, `dataset.py`

### 模型架構 (`model.py`)

| 模型 | 說明 | 啟用方式 |
|------|------|---------|
| `UNet3D` | 標準 3D U-Net | 預設 |
| `AttentionUNet3D` | SE Block + Attention Gate | `--attention` |

- **Input**: (B, 1, D, H, W) 單通道 3D 體積
- **Output**: (B, 1, D, H, W) Logits

### 數據加載 (`dataset.py`)

- **動態分割**: 根據 `--split_seed` 與 `--train_ratio` 等比例參數動態分割 train/val/test
- **深度裁切**: 超過 `max_depth` 時，正樣本做 Center Crop，負樣本做 Random Crop
- **Hard Negative Mining**: `positive_ratio` 控制結節中心 vs 隨機背景的採樣比例
- **Augmentation**: 隨機翻轉 (H/V/Depth) + 90° 旋轉 + 亮度偏移

### 損失函式 (`--loss_type`)

| 類型 | 說明 |
|------|------|
| `combined` (預設) | Tversky + Boundary + BCE (加權組合) |
| `tversky` | Tversky Loss (可調 FP/FN 權重) |
| `dice` | Dice + BCE + Focal Loss |

### 訓練最佳化

- **Mixed Precision (AMP)**: `--use_amp` — FP16 訓練，減少 ~50% 顯存
- **Gradient Checkpointing**: `--use_checkpointing` — 以計算換顯存
- **Gradient Accumulation**: `--accumulation_steps N` — 等效增大 Batch Size
- **Scheduler**: OneCycleLR (快速升溫 + 長時間退火)

### 執行命令

```cmd
python -m detection.train_3dunet.main train ^
    --npz_dir cache/lndb_volume_npz_agr1 ^
    --epochs 200 ^
    --batch_size 1 ^
    --accumulation_steps 4 ^
    --attention ^
    --loss_type combined ^
    --positive_ratio 0.9 ^
    --use_checkpointing ^
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 ^
    --split_seed 42
```

---

## 4. 偵測與推理

**代碼**: `detector.py` (`NoduleDetector`), `location_estimator.py`

### 後處理流程

```
Model Output (Logits)
  → Sigmoid → Probability Map
    → 閾值切分 (det_prob_threshold)
    → 形態學閉運算 (Closing)
    → 3D 連通區域標記 (CCL)
    → 尺寸過濾 (det_min_size mm³)
    → 幾何計算 (質心/體積/直徑)
    → 座標轉換 (像素 → 世界座標 mm)
    → 肺葉定位 (LungLocationEstimator)
    → DetectedNodule 物件
```

- **Sliding Window Inference**: 當 Depth > 64 時，自動採用滑動窗口 + Gaussian Blending

### 輸出格式

```json
{
  "id": 1,
  "probability": 0.985,
  "location": {
    "lobe": "RUL",
    "description": "右肺上葉 (RUL)"
  },
  "geometry": {
    "centroid_world": [120.5, -80.2, -150.0],
    "volume_mm3": 45.2,
    "diameter_mm": 5.2
  },
  "slice_range": [45, 52]
}
```

### 執行命令

```cmd
REM 完整測試 (含偵測報告 + 視覺化 + GIF)
python -m detection.train_3dunet.main fulltest ^
    --checkpoint path\to\best_model.pth ^
    --npz_dir cache/lndb_volume_npz_agr1 ^
    --split test ^
    --attention ^
    --det_prob_threshold 0.5 ^
    --det_min_size 10.0 ^
    --full_volume
```

**輸出**:
- `test_results.json` — 每個樣本的分割與偵測指標
- `detections.json` — 所有偵測到的結節總表
- `test_summary.png` — 彙總統計圖
- 每個 Case 資料夾: `overview.png`, `stats.json`, `animation.gif`

---

## 5. 快速測試 / 統計 / 視覺化

```cmd
REM 快速 Dice Score 測試
python -m detection.train_3dunet.main test ^
    --checkpoint path\to\model.pth ^
    --npz_dir cache/lndb_volume_npz_agr1 ^
    --attention

REM 數據集統計
python -m detection.train_3dunet.main stats ^
    --npz_dir cache/lndb_volume_npz_agr1

REM 視覺化預測 vs GT
python -m detection.train_3dunet.main visualize ^
    --checkpoint path\to\model.pth ^
    --npz_dir cache/lndb_volume_npz_agr1 ^
    --attention
```

---

## 目錄結構

```
detection/train_3dunet/
├── main.py                 # CLI 入口
├── config.py               # 配置管理 (dataclass)
├── preprocess.py           # 數據預處理 (LNDb/MSD → NPZ)
├── dataset.py              # PyTorch Dataset + Collate
├── model.py                # UNet3D / AttentionUNet3D
├── buildingblocks.py       # 模型基礎組件 (Conv, SE, AttentionGate)
├── trainer.py              # 訓練 / 驗證 / 測試 / 視覺化
├── detector.py             # NoduleDetector (後處理 + 偵測)
├── location_estimator.py   # 肺葉位置估算
├── segmentation.py         # 肺部分割 (GPU/CPU)
├── visualize.py            # 數據視覺化工具
├── utils.py                # 工具函式 (Split, etc.)
├── tests/                  # 自動化測試
├── pipeline.md             # 本文件
└── README.md               # 快速入門
```
