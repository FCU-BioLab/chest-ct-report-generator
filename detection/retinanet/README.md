# MONAI RetinaNet 肺結節偵測模組

本目錄包含基於 MONAI 3D RetinaNet 的肺結節偵測完整解決方案。
支援 LNDb 與 LUNA16 資料集，提供從資料準備、訓練、推論到視覺化的完整流程。

## 目錄結構

- `retinanet/`
  - `config.py`: 模型與訓練參數設定 (Dataclass)
  - `dataset.py`: 資料集定義 (Dataset) 與轉換邏輯
  - `trainer.py`: 訓練核心邏輯 (Trainer)
  - `main.py`: 命令列入口 (CLI)，支援 train/test/predict
  - `prepare_data.py`: 資料準備腳本 (LNDb/LUNA16 -> JSON)
  - `inference.py`: 臨床推論腳本 (DICOM/MHD -> 報告)
  - `evaluate.py`: 結果評估腳本 (計算 IoU/F1)
  - `visualize.py`: 結果視覺化腳本 (GIF)

## 安裝需求

請確保已安裝 MONAI 與相關套件：
```bash
pip install "monai[all]>=1.3" pandas simpleitk imageio matplotlib
```

## 使用指南

### 1. 資料準備
掃描原始資料集並生成 `dataset.json`。

**LNDb 範例:**
```bash
python -m detection.retinanet.prepare_data \
  --dataset lndb \
  --base_dir "cache/LNDb" \
  --output "dataset_lndb.json"
```

**LUNA16 範例:**
```bash
python -m detection.retinanet.prepare_data \
  --dataset luna16 \
  --base_dir "cache/LUNA16" \
  --output "dataset_luna16.json"
```

### 2. 訓練模型 (Training)
使用 `main.py train` 指令。請指定由步驟 1 生成的 JSON 檔案 (支援原始 MHD/NIfTI 資料)。

**使用 JSON 資料列表:**
```bash
python -m detection.retinanet.main train \
  --data_path "dataset_lndb.json" \
  --epochs 300 \
  --output_dir "results/experiment_1"
```



### 3. 推論與預測

**單檔快速預測 (用於開發):**
```bash
python -m detection.retinanet.main predict \
  --checkpoint "results/experiment_1/model_best.pt" \
  --input "cache/LNDb/data0/LNDb-0001.mhd"
```

**臨床推論 (完整管線，含預處理):**
輸入可為 DICOM 資料夾或 `.nii.gz`/`.mhd` 檔案。
```bash
python -m detection.retinanet.inference \
  --input_path "data/patient_01" \
  --model_path "results/experiment_1/model_best.pt" \
  --output_dir "results/patient_01"
```

### 4. 評估與視覺化

**評估指標 (IoU, Recall):**
```bash
python -m detection.retinanet.evaluate \
  --report_path "results/patient_01/report.json" \
  --dataset lndb
```

**生成視覺化 GIF:**
```bash
python -m detection.retinanet.visualize \
  --report_path "results/patient_01/report.json"
```

## 注意事項
- 支援直接使用 JSON 資料列表讀取原始影像 (MHD/NIfTI) 進行訓練。
- 預設設定針對 3-30mm 的小結節最佳化 (Anchor Sizes)。
