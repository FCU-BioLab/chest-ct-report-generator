# 肺結節偵測模組 (Nodule Detection Module)

基於 3D U-Net 的肺結節偵測與分割系統。除了精確分割病灶體積外，還包含完整的後處理流程，輸出結構化的病灶報告（含解剖位置、體積、直徑與信心度）。

## 快速開始

### 1. 數據預處理

```cmd
python -m detection.train_3dunet.main convert ^
    --dataset lndb ^
    --input_dir E:\lung_ct_lesion_dataset\LNDb ^
    --output_dir cache/lndb_volume_npz_agr1 ^
    --full_volume ^
    --min_agreement 2 ^
    --image_size 256
```

### 2. 訓練

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

### 3. 完整測試

```cmd
python -m detection.train_3dunet.main fulltest ^
    --checkpoint path\to\best_model.pth ^
    --npz_dir cache/lndb_volume_npz_agr1 ^
    --split test ^
    --attention ^
    --det_prob_threshold 0.5 ^
    --det_min_size 10.0 ^
    --full_volume
```

### 4. 快速測試 / 統計

```cmd
REM Dice Score 測試
python -m detection.train_3dunet.main test ^
    --checkpoint path\to\model.pth ^
    --npz_dir cache/lndb_volume_npz_agr1 --attention

REM 數據集統計
python -m detection.train_3dunet.main stats ^
    --npz_dir cache/lndb_volume_npz_agr1
```

---

## 主要參數

### 預處理 (`convert`)

| 參數 | 預設 | 說明 |
|------|------|------|
| `--dataset` | 必填 | `lndb` 或 `msd` |
| `--full_volume` | off | 保留完整 CT 體積 (不裁切 Z 軸) |
| `--min_agreement` | 1 | 最低醫師一致性 (1-3) |
| `--context_slices` | 32 | 結節中心前後各取的切片數 |
| `--min_diameter` | 3.0 | 最小結節直徑 (mm) |
| `--image_size` | 256 | 輸出影像尺寸 |

### 訓練 (`train`)

| 參數 | 預設 | 說明 |
|------|------|------|
| `--attention` | off | 啟用 SE Block + Attention Gate |
| `--loss_type` | `combined` | `combined` / `tversky` / `dice` |
| `--positive_ratio` | 0.7 | 正樣本比例 (1.0=全正, 0.9=10% Hard Negative) |
| `--use_checkpointing` | off | Gradient Checkpointing (省顯存) |
| `--accumulation_steps` | 1 | 梯度累積步數 |
| `--batch_size` | 2 | Batch Size |
| `--learning_rate` | 1e-4 | 學習率 |

### 測試 (`fulltest` / `test`)

| 參數 | 預設 | 說明 |
|------|------|------|
| `--det_prob_threshold` | 0.5 | 偵測機率閾值 |
| `--det_min_size` | 10.0 | 最小偵測體積 (mm³) |
| `--full_volume` | off | 完整體積推理 (Sliding Window) |
| `--no_viz` | off | 跳過視覺化 |
| `--no_gif` | off | 跳過 GIF 動畫 |
| `--no_postprocess` | off | 跳過後處理 |

---

## 專案結構

| 文件 | 說明 |
|------|------|
| `main.py` | CLI 入口 |
| `config.py` | 配置管理 (dataclass) |
| `preprocess.py` | 數據預處理 (LNDb/MSD → NPZ) |
| `dataset.py` | PyTorch Dataset + Collate |
| `model.py` | UNet3D / AttentionUNet3D |
| `buildingblocks.py` | 模型基礎組件 (Conv, SE, Attention Gate) |
| `trainer.py` | 訓練 / 驗證 / 測試 / 視覺化 |
| `detector.py` | NoduleDetector (後處理 + 偵測) |
| `location_estimator.py` | 肺葉位置估算 |
| `segmentation.py` | 肺部分割 (GPU/CPU 雙路徑) |
| `visualize.py` | 數據視覺化工具 |
| `utils.py` | 工具函式 (Split 等) |

## 詳細說明

請參閱 [pipeline.md](pipeline.md) 獲取完整的 Pipeline 說明。
