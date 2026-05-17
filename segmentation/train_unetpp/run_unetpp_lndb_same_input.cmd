@echo off
setlocal

cd /d "%~dp0\..\.."

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "NO_ALBUMENTATIONS_UPDATE=1"

venv\Scripts\python.exe -m segmentation.train_unetpp.main ^
  --dataset lndb ^
  --epochs 100 ^
  --batch_size 2 ^
  --num_workers 0 ^
  --lr 3e-5 ^
  --loss_type bce_dice ^
  --grad_clip 0.5 ^
  --early_stopping_patience 20 ^
  --encoder_weights none ^
  --output_dir F:\unetpp_results\unetpp_same_input_stable ^
  --device cuda

endlocal
