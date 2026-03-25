# Detection

This repository keeps the main detection pipeline:

- `detection.retinanet`: main 3D RetinaNet pipeline (existing production path)

## 1) RetinaNet (existing path)

```bash
python -m detection.retinanet.prepare_data --dataset lndb --base_dir "cache/LNDb" --output "dataset_lndb.json"
python -m detection.retinanet.main train --data_path "dataset_lndb.json" --epochs 300 --output_dir "results/experiment_1"
python -m detection.retinanet.inference --input_path "data/patient_01" --model_path "results/experiment_1/model_best.pt" --output_dir "results/patient_01"
```

## Other modules

- `detection.common`: shared utilities
