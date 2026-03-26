# MONAI 3D RetinaNet（肺結節偵測）

本目錄包含 RetinaNet 偵測完整流程（核心訓練/推論 + 資料前處理 + FPR + 視覺化工具）。  
以下指令全部以 **Windows CMD (`cmd.exe`)** 為準。

## 1) 模組分類

### 核心流程（訓練/測試/推論）
- `config.py`
- `dataset.py`
- `trainer.py`
- `main.py`
- `inference.py`
- `metrics.py`
- `postprocess.py`

### 資料前處理
- `prepare_data.py`（LNDb / LUNA16）
- `prepare_luna16_new.py`（LUNA16-New）
- `make_kfold_json.py`（k-fold split）

### FPR（假陽性抑制）
- `collect_fpr_data.py`
- `train_fpr.py`
- `train_fpr_fuser.py`
- `fpr_model.py`
- `fpr_fuser.py`

### 工具
- `evaluate.py`
- `visualize.py`
- `visualize_data.py`
- `visualize_predictions.py`
- `test_data_pipeline.py`

## 2) 環境安裝（CMD）

```cmd
cd /d C:\GitHub\chest-ct-report-generator
pip install "monai[all]>=1.3" pandas simpleitk imageio matplotlib nibabel scipy scikit-learn tqdm
```

## 3) 建立資料清單（manifest）

### LNDb

```cmd
python -m detection.retinanet.prepare_data ^
  --dataset lndb ^
  --base_dir "cache\LNDb" ^
  --output "detection\manifests\dataset_lndb.json"
```

### LUNA16

```cmd
python -m detection.retinanet.prepare_data ^
  --dataset luna16 ^
  --base_dir "cache\LUNA16" ^
  --output "detection\manifests\dataset_luna16.json"
```

### LUNA16-New

```cmd
python -m detection.retinanet.prepare_luna16_new ^
  --base_dir "E:\lung_ct_lesion_dataset\LUNA16-New" ^
  --output_json "detection\manifests\dataset_luna16_new.json"
```

### 產生 k-fold

```cmd
python -m detection.retinanet.make_kfold_json ^
  --input_json "detection\manifests\dataset_luna16_new.json" ^
  --group_keys seriesuid ^
  --num_folds 5
```

## 4) 訓練 / 測試 / 推論

### 訓練

```cmd
python -m detection.retinanet.main train ^
  --data_path "detection\manifests\dataset_lndb.json" ^
  --epochs 300 ^
  --batch_size 1 ^
  --output_dir "results\retinanet_exp1"
```

### 測試

```cmd
python -m detection.retinanet.main test ^
  --checkpoint "results\retinanet_exp1\model_best.pt" ^
  --data_path "detection\manifests\dataset_lndb.json"
```

### 單例推論

```cmd
python -m detection.retinanet.main predict ^
  --checkpoint "results\retinanet_exp1\model_best.pt" ^
  --input "cache\LNDb\data0\LNDb-0001.mhd"
```

### 完整推論流程（輸出 report）

```cmd
python -m detection.retinanet.inference ^
  --input_path "data\patient_01" ^
  --model_path "results\retinanet_exp1\model_best.pt" ^
  --output_dir "results\patient_01"
```

## 5) 評估與視覺化

### 評估 report

```cmd
python -m detection.retinanet.evaluate ^
  --report_path "results\patient_01\report.json" ^
  --dataset lndb
```

### 報告視覺化

```cmd
python -m detection.retinanet.visualize ^
  --report_path "results\patient_01\report.json"
```

### 資料可視化檢查

```cmd
python -m detection.retinanet.visualize_data ^
  --data_path "detection\manifests\dataset_lndb.json" ^
  --output_dir "detection\retinanet\viz_data"
```

## 6) FPR 流程（可選）

### 收集 FPR 訓練資料

```cmd
python -m detection.retinanet.collect_fpr_data ^
  --data_path "detection\manifests\dataset_lndb.json" ^
  --checkpoint "results\retinanet_exp1\model_best.pt" ^
  --output_dir "results\fpr_data"
```

### 訓練 FPR 模型

```cmd
python -m detection.retinanet.train_fpr ^
  --data_json "results\fpr_data\fpr_dataset.json" ^
  --output_dir "results\fpr_model"
```

### 訓練 FPR Fuser（可選）

```cmd
python -m detection.retinanet.train_fpr_fuser ^
  --data_json "results\fpr_data\fpr_dataset.json" ^
  --fpr_model_path "results\fpr_model\model_best.pt" ^
  --output_dir "results\fpr_fuser"
```

## 7) 資料管線快速測試（可選）

```cmd
python -m detection.retinanet.test_data_pipeline ^
  --data_path "detection\manifests\dataset_lndb.json"
```

## 8) 常見注意事項

- 請在專案根目錄 `C:\GitHub\chest-ct-report-generator` 執行。
- 路徑含空白時請加雙引號 `"..."`。
- CMD 多行指令續行符號是 `^`（不是 `\`）。
