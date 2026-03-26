# Detection

## 目前啟用模組

- `detection.retinanet`
  - 3D RetinaNet + FPR
  - 含訓練、測試、推論、資料生成流程
- `detection.common`
  - 共用工具與輔助函式

## 常用命令

```bash
python -m detection.retinanet.prepare_data --dataset lndb --base_dir "cache/LNDb" --output "detection/manifests/dataset_lndb.json"
python -m detection.retinanet.prepare_luna16_new --base_dir "<LUNA16_NEW_ROOT>" --output_json "detection/manifests/dataset_luna16_new.json"
python -m detection.retinanet.main train --data_path "detection/manifests/dataset_lndb.json" --epochs 300 --output_dir "results/experiment_1"
python -m detection.retinanet.main test --data_path "detection/manifests/dataset_lndb.json" --output_dir "results/experiment_1"
python -m detection.retinanet.inference --input_path "data/patient_01" --model_path "results/experiment_1/model_best.pt" --output_dir "results/patient_01"
```

## 資料檔案位置規範

- Dataset JSON 預設輸出目錄：`detection/manifests`
- 產生 k-fold 時若未指定 `--output_dir`，同樣輸出到 `detection/manifests`
