# Chest CT Report Generator

胸部 CT 結節偵測、分割、特徵量化與報告生成專案。

## Current Architecture

- `detection/retinanet/`
  - 主偵測流程（3D RetinaNet）
  - 含 FPR 模型與相關訓練/測試/推論程式
- `llm/ct_report_pipeline/`
  - 分割、特徵萃取、報告生成（模板/LLM）
- `n8n/`
  - headless pipeline orchestration（preprocess -> detect -> segment -> feature -> report）
- `dataset_process/`
  - 精簡後僅保留 manifest 輔助腳本（`create_lidc_minimal_manifests.ps1`）

## Repository Layout

```text
chest-ct-report-generator/
├── detection/
│   ├── retinanet/
│   └── common/
├── llm/
│   └── ct_report_pipeline/
├── n8n/
├── dataset_process/
├── segmentation/
├── README.md
└── PIPELINE_ZH_TW.md
```

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Detection (RetinaNet + FPR)

```bash
python -m detection.retinanet.prepare_data --dataset lndb --base_dir "cache/LNDb" --output "detection/manifests/dataset_lndb.json"
python -m detection.retinanet.main train --data_path "detection/manifests/dataset_lndb.json" --epochs 300 --output_dir "results/experiment_1"
python -m detection.retinanet.main test --data_path "detection/manifests/dataset_lndb.json" --output_dir "results/experiment_1"
python -m detection.retinanet.inference --input_path "data/patient_01" --model_path "results/experiment_1/model_best.pt" --output_dir "results/patient_01"
```

## End-to-End Case Pipeline (n8n headless)

```bash
python n8n/run_case_pipeline.py --stage run --case-id case001 --input-path <CT_PATH> --model-path <RETINANET_MODEL>
```

## Notes

- 本專案已移除 `detection/train_3dunet` 與 `detection/scripts` 舊路徑。
- `detection/nndet_data`、`detection/results`、`detection/video_result` 屬資料/輸出目錄，保留不刪。
