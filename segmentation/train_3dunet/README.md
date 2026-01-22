# 3D U-Net 體積分割 (Volume Segmentation)

本模組實現了基於 3D U-Net 的肺結節/病灶體積分割，使用視頻（切片序列）數據進行訓練。

## 快速開始 (Quick Start)

### 1. 數據預處理

將 LNDb 或 MSD 數據集轉換為 NPZ 體積格式：

```cmd
python -m segmentation.train_3dunet.main convert ^
    --dataset lndb ^
    --input_dir E:\lung_ct_lesion_dataset\LNDb ^
    --output_dir cache/volume_npz ^
    --context_slices 16 ^
    --image_size 256
```

### 2. 訓練

```cmd
python -m segmentation.train_3dunet.main train ^
    --npz_dir cache/volume_npz ^
    --epochs 100 ^
    --batch_size 4 ^
    --attention ^
    --loss_type combined
```

**參數說明**:
- `--attention`: 啟用 SE Block + Attention Gate
- `--loss_type`: `combined`(預設) / `tversky` / `dice`

### 3. 完整測試

運行完整測試，生成視覺化報告和 GIF 動畫：

```cmd
python -m segmentation.train_3dunet.main fulltest ^
    --npz_dir cache/volume_npz ^
    --checkpoint path\to\best_model.pth ^
    --split test
```

**輸出**:
- `test_results.json`: 每個樣本的指標
- `test_summary.png`: 彙總統計圖
- 每個 Case 的 `animation.gif`: Overlay GIF 動畫

**選項**:
- `--no_gif`: 跳過 GIF 輸出
- `--no_viz`: 跳過所有視覺化

### 4. 快速測試

只計算 Dice Score：

```cmd
python -m segmentation.train_3dunet.main test ^
    --npz_dir cache/volume_npz ^
    --checkpoint path\to\model.pth
```

### 5. 統計

```cmd
python -m segmentation.train_3dunet.main stats --npz_dir cache/volume_npz
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
| `preprocess.py` | 數據轉換 (LNDb/MSD → NPZ) |
| `visualize.py` | 數據視覺化工具 |
| `pipeline.md` | 詳細 Pipeline 說明 |

## 詳細說明

請參閱 [pipeline.md](pipeline.md) 獲取完整的 Pipeline 說明。
