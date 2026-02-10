# 肺結節偵測模組 (Nodule Detection Module)

本模組實現了基於 3D U-Net 的肺結節偵測與分割系統。除了精確分割病灶體積外，還包含完整的後處理流程，能輸出結構化的病灶報告（包含解剖位置、體積、直徑與信心度）。

## 快速開始 (Quick Start)

### 1. 數據預處理

將 LNDb 或 MSD 數據集轉換為 NPZ 體積格式：

```cmd
REM 使用 Full Volume 轉換
python -m detection.train_3dunet.main convert ^
    --dataset lndb ^
    --input_dir E:\lung_ct_lesion_dataset\LNDb ^
    --output_dir cache/lndb_volume_npz_agr1 ^
    --full_volume ^
    --image_size 256
```

### 2. 訓練

```cmd
python -m detection.train_3dunet.main train ^
    --epochs 200 ^
    --batch_size 1 ^
    --accumulation_steps 4 ^
    --attention ^
    --train_ratio 0.8 ^
    --val_ratio 0.1 ^
    --test_ratio 0.1 ^
    --split_seed 42 ^
    --loss_type combined ^
    --positive_ratio 0.9 ^
    --use_checkpointing
```

**參數說明**:
- `--attention`: 啟用 SE Block + Attention Gate
- `--loss_type`: `combined`(預設) / `tversky` / `dice`
- `--split_seed`: 分割數據集的隨機種子
- `--train_ratio` (等): 訓練/驗證/測試集比例
- `--positive_ratio`: 正樣本(結節中心)採樣比例 (0.0-1.0)。設為 0.9 表示 10% 時間採樣隨機背景 (Hard Negative Mining)。
- `--use_checkpointing`: 啟用 Gradient Checkpointing (以時間換空間，節省顯存)。
- `--use_amp`:啟用混合精度訓練 (預設開啟)。

### 3. 完整測試 (含偵測報告)

運行完整測試，生成視覺化報告、GIF 動畫及偵測指標：

```cmd
python -m detection.train_3dunet.main fulltest ^
    --npz_dir cache/lndb_volume_npz_agr1 ^
    --checkpoint path\to\best_model.pth ^
    --split test ^
    --attention ^
    --det_prob_threshold 0.9 ^
    --det_prob_threshold 0.5 ^
    --det_min_size 5.0 ^
    --full_volume
```
*註: `det_prob_threshold 0.9` 與 `det_min_size 1.0` 為實驗驗證後的最佳參數，能在保留小結節的同時過濾誤報。*

**輸出內容**:
- `test_results.json`: 每個樣本的指標 (Dice, IoU, Precision, Recall)
- `detections.json`: **所有偵測到的結節總表** (含解剖位置與幾何資訊)
- `test_summary.png`: 彙總統計圖
- 每個 Case 資料夾:
  - `animation.gif`: Overlay GIF 動畫
  - `stats.json`: 該案例的詳細偵測數據

**選項**:
- `--no_gif`: 跳過 GIF 輸出
- `--no_gif`: 跳過 GIF 輸出
- `--no_viz`: 跳過所有視覺化
- `--full_volume`: **[NEW]** 使用完整 CT 體積進行測試 (Sliding Window Inference)，解決大體積 OOM 問題。

### 4. 快速測試

只計算 Dice Score：

```cmd
python -m detection.train_3dunet.main test ^
    --npz_dir cache/lndb_volume_npz_agr1 ^
    --checkpoint path\to\model.pth ^
    --attention
```

### 5. 統計

```cmd
python -m detection.train_3dunet.main stats --npz_dir cache/lndb_volume_npz_agr1
```

---

## 專案結構

| 文件 | 說明 |
|------|------|
| `main.py` | CLI 入口 |
| `config.py` | 配置管理 |
| `model.py` | 3D U-Net 模型 |
| `buildingblocks.py` | 模型組件 (SE, Attention Gate) |
| `dataset.py` | PyTorch Dataset |
| `trainer.py` | 訓練/測試邏輯 |
| `detector.py` | 結節偵測與後處理 (NoduleDetector) |
| `location_estimator.py` | 肺葉位置估算 |
| `preprocess.py` | 數據轉換 (LNDb/MSD → NPZ) |
| `visualize.py` | 數據視覺化工具 |
| `pipeline.md` | 詳細 Pipeline 說明 |

## 詳細說明

請參閱 [pipeline.md](pipeline.md) 獲取完整的 Pipeline 說明。
