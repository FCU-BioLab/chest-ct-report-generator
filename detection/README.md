# Detection

Active modules:

- `detection.retinanet`: 3D RetinaNet + FPR model + training/testing/inference logic.
- `detection.common`: shared utilities.

## Main commands

```bash
python -m detection.retinanet.prepare_data --dataset lndb --base_dir "cache/LNDb" --output "detection/manifests/dataset_lndb.json"
python -m detection.retinanet.prepare_luna16_new --base_dir "<LUNA16_NEW_ROOT>" --output_json "detection/manifests/dataset_luna16_new.json"
python -m detection.retinanet.main train --data_path "detection/manifests/dataset_lndb.json" --epochs 300 --output_dir "results/experiment_1"
python -m detection.retinanet.main test --data_path "detection/manifests/dataset_lndb.json" --output_dir "results/experiment_1"
python -m detection.retinanet.inference --input_path "data/patient_01" --model_path "results/experiment_1/model_best.pt" --output_dir "results/patient_01"
```
