# 3D U-Net 視頻微調 (Video Finetuning)

本模組實現了基於 3D U-Net 的肺結節/病灶體積分割，使用視頻（切片序列）數據進行訓練。

其設計與 `finetune_medsam2_video` 中使用的數據格式兼容，可以共用預處理後的 NPZ 檔案。

## 使用方法 (Usage)

### 1. 數據預處理 (Preprocess Data)

將 LNDb 或 MSD 數據集轉換為 NPZ 體積格式 (Volume Format)。

```bash
python segmentation/train_3dunet/main.py convert --dataset lndb --input_dir E:\lung_ct_lesion_dataset\LNDb --output_dir cache/volume_npz --max_depth 32
```

### 2. 訓練 (Train)

訓練 3D U-Net 模型。

```bash
python segmentation/train_3dunet/main.py train --npz_dir cache/volume_npz --epochs 50 --batch_size 4 --max_depth 32
```

### 3. 統計 (Statistics)

查看數據集統計資訊。

```bash
python segmentation/train_3dunet/main.py stats --npz_dir cache/volume_npz
```

### 4. 測試 (Test)

評估模型在測試集上的表現。

```bash
python segmentation/train_3dunet/main.py test --checkpoint volume_output_unet3d/checkpoints/best_model.pt --npz_dir cache/volume_npz
```

## 專案結構 (Structure)

- `main.py`: 主程式入口點。
- `config.py`: 配置管理。
- `model.py`: 3D U-Net 模型架構。
- `video_dataset.py`: 用於加載 NPZ 視頻檔案的 Pytorch Dataset。
- `video_trainer.py`: 訓練循環邏輯。
- `preprocess.py`: 數據轉換邏輯 (LNDb/MSD -> NPZ)。
