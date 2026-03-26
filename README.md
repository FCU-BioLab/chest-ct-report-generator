# Chest CT Report Generator

胸部 CT 結節偵測、分割、特徵量化與報告生成專案。

## 專案架構

- `detection/retinanet/`
  - 3D RetinaNet 偵測主流程
  - 含 FPR 模型與其訓練、測試、推論程式
- `llm/ct_report_pipeline/`
  - 分割、特徵萃取、報告生成（模板/LLM）
- `n8n/`
  - Headless pipeline 編排（preprocess -> detect -> segment -> feature -> report）
- `dataset_process/`
  - 資料 manifest 與資料集 JSON 整理工具

## 目錄概覽

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

## 快速開始

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 偵測模組（RetinaNet + FPR）

```bash
python -m detection.retinanet.prepare_data --dataset lndb --base_dir "cache/LNDb" --output "detection/manifests/dataset_lndb.json"
python -m detection.retinanet.main train --data_path "detection/manifests/dataset_lndb.json" --epochs 300 --output_dir "results/experiment_1"
python -m detection.retinanet.main test --data_path "detection/manifests/dataset_lndb.json" --output_dir "results/experiment_1"
python -m detection.retinanet.inference --input_path "data/patient_01" --model_path "results/experiment_1/model_best.pt" --output_dir "results/patient_01"
```

## JSON 生成邏輯 Smoke Test（Windows CMD）

用於驗證「清空後重新生成」資料集 JSON 的流程。

```bat
del /Q detection\manifests\*.json
del /Q cache\*.json

venv\Scripts\python.exe -m detection.retinanet.prepare_luna16_new --base_dir "E:\lung_ct_lesion_dataset\LUNA16-New" --output_image_dir "E:\lung_ct_lesion_dataset\LUNA16-New\retina_mhd" --no_image_write --max_series 20

venv\Scripts\python.exe -m detection.retinanet.make_kfold_json --input_json "detection/manifests/dataset_luna16_new.json" --group_keys seriesuid --num_folds 5

venv\Scripts\python.exe -c "from detection.retinanet.dataset import prepare_datalist; p='detection/manifests/dataset_luna16_new_fold0.json'; print('train', len(prepare_datalist(p,'train'))); print('val', len(prepare_datalist(p,'val'))); print('test', len(prepare_datalist(p,'test')))"
```

預期結果：命令正常結束、`detection/manifests` 產生 fold JSON，並印出 `train/val/test` 筆數。

## n8n 端到端流程（Headless）

```bash
python n8n/run_case_pipeline.py --stage run --case-id case001 --input-path <CT_PATH> --model-path <RETINANET_MODEL>
```

## 備註

- 本專案已移除 `detection/train_3dunet` 與 `detection/scripts` 舊路徑。
- `detection/nndet_data`、`detection/results`、`detection/video_result` 為資料/輸出目錄，保留不刪。
