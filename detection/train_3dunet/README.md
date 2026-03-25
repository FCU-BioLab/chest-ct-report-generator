# 3D U-Net (NIfTI Task Pipeline)

This module now uses **NIfTI task folders** as input:

- `imagesTr/*_0000.nii.gz`
- `labelsTr/*.nii.gz`

No NPZ input is required for training/testing/check-data commands.

## 1) Train

```cmd
python -m detection.train_3dunet.main train ^
  --data_dir "detection\nndet_data\Task100_LUNA16Nodule" ^
  --epochs 300 ^
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

## 2) Evaluate

```cmd
python -m detection.train_3dunet.main test ^
  --checkpoint "path\to\best_model.pth" ^
  --data_dir "detection\nndet_data\Task100_LUNA16Nodule" ^
  --split test ^
  --attention ^
  --det_prob_threshold 0.5 ^
  --det_min_size 10.0
```

## 3) Full Evaluation + Visual Artifacts

```cmd
python -m detection.train_3dunet.main fulltest ^
  --checkpoint "path\to\best_model.pth" ^
  --data_dir "detection\nndet_data\Task100_LUNA16Nodule" ^
  --split test ^
  --attention ^
  --det_prob_threshold 0.5 ^
  --det_min_size 10.0 ^
  --full_volume
```

## 4) Dataset Check / Visualization

```cmd
python -m detection.train_3dunet.main check_data ^
  --data_dir "detection\nndet_data\Task100_LUNA16Nodule" ^
  --split train ^
  --mode dataset
```

Available `--mode`:

- `dataset`: print statistics
- `dataset_view`: single sample grid
- `dataset_batch`: batch preview
- `dataset_augment`: augmentation preview

## Key Arguments

- `--data_dir`: NIfTI task folder
- `--base_filters`: base channel count (model widths become `[base, 2x, 4x, 8x]`)
- `--accumulation_steps`: gradient accumulation steps
- `--early_stopping_patience`: early-stopping patience in epochs
- `--attention`: use AttentionUNet3D
- `--use_checkpointing`: enable gradient checkpointing
- `--no_tensorboard`: disable live TensorBoard curves

## Live Curves and Metrics

- During training, metrics are updated every epoch:
  - `metrics_live.png` (quick live snapshot)
  - `metrics_history.json` (full history)
  - `tensorboard/` (live scalars)
- To watch real-time line charts:

```cmd
tensorboard --logdir "detection\video_result\3dunet_train_YYYYMMDD_HHMMSS\tensorboard"
```

- Validation now includes:
  - Detection metrics: Precision / Recall / F1 (IoU=0.1)
  - LUNA16 metrics: FROC sensitivities and CPM

## Notes

- `convert` / NPZ preprocessing logic is legacy and not used by this NIfTI-task training path.
- `Config.load(...)` still accepts old config files containing `npz_dir` and auto-maps it to `data_dir`.
