# 3D U-Net Training Pipeline (Current)

This document describes the **active** pipeline used by `detection.train_3dunet.main`.

Input format is NIfTI task layout:

- `imagesTr/*_0000.nii.gz`
- `labelsTr/*.nii.gz`

No NPZ input is required by train/test/check-data commands.

## End-to-End Flow

1. Dataset loading
- `VolumetricDataset` reads `imagesTr/labelsTr`.
- Split is patient-level (`train/val/test`) using `split_seed` and split ratios.

2. Preprocessing in dataset
- CT intensity windowing: HU `[-1000, 400]` -> normalized `[0, 1]`.
- Depth handling:
  - train: positive-aware crop (`positive_ratio`) with random negative crops.
  - val/test: no GT-centered crop (prevents label leakage); uses center crop unless full-volume mode is enabled.
- XY resize to `image_size`.
- Optional train-only augmentation (flip/rot90/intensity shift).

3. Training
- Model: `UNet3D` or `AttentionUNet3D` (`--attention`).
- Loss:
  - `combined` (Tversky + Boundary + BCE), or
  - `tversky`, or
  - legacy `dice` branch.
- Optimizer: `AdamW`.
- Scheduler: `OneCycleLR`.
- Mixed precision: `torch.amp` (`cuda` only).
- Gradient accumulation supported (`--accumulation_steps`).
- Early stopping on validation detection F1 (`--early_stopping_patience`).

4. Validation metrics per epoch
- Segmentation: Dice/IoU/Precision/Recall.
- Detection-style component metrics: TP/FP/FN, Precision/Recall/F1.

5. Evaluation / full test
- `test`: summary metrics.
- `fulltest`: detailed metrics + visual artifacts + per-case reports.

## Commands

### Train
```cmd
python -m detection.train_3dunet.main train ^
  --data_dir "detection\nndet_data\Task100_LUNA16Nodule" ^
  --epochs 200 ^
  --batch_size 1 ^
  --accumulation_steps 4 ^
  --attention ^
  --loss_type combined ^
  --positive_ratio 0.9 ^
  --use_checkpointing ^
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 ^
  --split_seed 42 ^
  --device cuda
```

### Test
```cmd
python -m detection.train_3dunet.main test ^
  --checkpoint "path\to\best_model.pth" ^
  --data_dir "detection\nndet_data\Task100_LUNA16Nodule" ^
  --split test ^
  --attention
```

### Full test (with visual outputs)
```cmd
python -m detection.train_3dunet.main fulltest ^
  --checkpoint "path\to\best_model.pth" ^
  --data_dir "detection\nndet_data\Task100_LUNA16Nodule" ^
  --split test ^
  --attention ^
  --full_volume
```

### Dataset sanity check
```cmd
python -m detection.train_3dunet.main check_data ^
  --data_dir "detection\nndet_data\Task100_LUNA16Nodule" ^
  --split train ^
  --mode dataset
```

## Operational Notes

- Ensure split ratios sum to `1.0`.
- For realistic final evaluation, prefer full-volume inference (`--full_volume` in test/fulltest path).
- Legacy NPZ conversion code remains in `preprocess.py` only for compatibility and is not part of the active training path.
