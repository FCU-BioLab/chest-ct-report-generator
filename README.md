# Chest CT Report Generator

## 快速啟動：n8n 後端與醫師版 HTML

以下指令以 Windows CMD 為主，假設專案放在 `C:\GitHub\chest-ct-report-generator`，輸出放在 `F:\chest-ct-report-output`。

### 1. 啟動 n8n

開一個 CMD 視窗執行：

```cmd
cd /d C:\GitHub\chest-ct-report-generator
n8n\start-local-n8n.cmd
```

看到 `Editor is now accessible via: http://localhost:5678` 代表 n8n 已經起來。這個視窗不要關掉。

### 2. 執行一個 Case

再開另一個 CMD 視窗執行。建議每次測試都換新的 `case_id`，避免舊輸出或舊圖片被混用。

```cmd
curl -X POST http://localhost:5678/webhook/ldZr0BeUQRKujShk/webhook/chest-ct-pipeline -H "Content-Type: application/json" -d "{\"case_id\":\"case-007\",\"input_path\":\"C:/GitHub/chest-ct-report-generator/detection/nndet_data/Task100_LUNA16Nodule/imagesTr/c_1_3_6_1_4_1_14519_5_2_1_6279_6001_100225287222365663678666836860_0000.nii.gz\",\"model_path\":\"C:/GitHub/chest-ct-report-generator/detection/results/retinanet_20260222_223955/model_best.pt\",\"repo_root\":\"C:/GitHub/chest-ct-report-generator\",\"python_exe\":\"C:/GitHub/chest-ct-report-generator/venv/Scripts/python.exe\",\"work_dir\":\"F:/chest-ct-report-output\",\"threshold\":0.5,\"device\":\"cuda\",\"use_llm\":true}"
```

如果要分多行貼到 CMD，行尾的 `^` 後面不能有空白：

```cmd
curl -X POST http://localhost:5678/webhook/ldZr0BeUQRKujShk/webhook/chest-ct-pipeline ^
  -H "Content-Type: application/json" ^
  -d "{\"case_id\":\"case-007\",\"input_path\":\"C:/GitHub/chest-ct-report-generator/detection/nndet_data/Task100_LUNA16Nodule/imagesTr/c_1_3_6_1_4_1_14519_5_2_1_6279_6001_100225287222365663678666836860_0000.nii.gz\",\"model_path\":\"C:/GitHub/chest-ct-report-generator/detection/results/retinanet_20260222_223955/model_best.pt\",\"repo_root\":\"C:/GitHub/chest-ct-report-generator\",\"python_exe\":\"C:/GitHub/chest-ct-report-generator/venv/Scripts/python.exe\",\"work_dir\":\"F:/chest-ct-report-output\",\"threshold\":0.5,\"device\":\"cuda\",\"use_llm\":true}"
```

### 3. 查看輸出

主要輸出會在：

```text
F:\chest-ct-report-output\<case_id>\
```

醫師版首頁：

```text
F:\chest-ct-report-output\<case_id>\05_report\index.html
```

完整 CT 複查頁：

```text
F:\chest-ct-report-output\<case_id>\05_report\ct_viewer.html
```

報告文字與 JSON：

```text
F:\chest-ct-report-output\<case_id>\05_report\AUTO_<case_id>.txt
F:\chest-ct-report-output\<case_id>\05_report\AUTO_<case_id>.json
```

Segmentation mask：

```text
F:\chest-ct-report-output\<case_id>\03_segment\mask_combined.nii.gz
F:\chest-ct-report-output\<case_id>\03_segment\mask_nodule_001.nii.gz
```

### 4. 只重生醫師版 HTML

如果 pipeline 已經跑完，只想用現有結果重生 `index.html` 與 `ct_viewer.html`：

```cmd
cd /d C:\GitHub\chest-ct-report-generator
venv\Scripts\python.exe n8n\run_case_pipeline.py --stage report --case-id case-007 --work-dir F:\chest-ct-report-output --use-llm
```

### 5. LLM 與套版報告

`use_llm:true` 會先嘗試使用本機 LLM。若 LLM 無法載入或生成失敗，pipeline 會自動改用套版報告，不會讓整個流程失敗。醫師版 `index.html` 會顯示報告來源：

- `LLM-generated`
- `Template-generated`
- `Template-generated (LLM fallback)`

### 6. 注意事項

- 如果 curl 回傳 `webhook is not registered`，請確認 n8n 已啟動，且 workflow 是 Active。
- 目前本機 webhook URL 是 `http://localhost:5678/webhook/ldZr0BeUQRKujShk/webhook/chest-ct-pipeline`。
- 若 detection image 還看到 GT 標記，通常是舊 case 的舊圖片；請換新的 `case_id` 重新跑。
- 醫師版 `index.html` 會顯示 pipeline process time、Detection/Segmentation 特徵比較、三軸 CT viewer 連結、Segmentation mask 連結。
- CT viewer 以 Axial 為主做 nodule 跳轉，並用同一條 slider 同步 Axial、Coronal、Sagittal 三個切面。

胸部 CT 結節偵測、分割、特徵量化與報告生成專案。

## 文件索引

- [外接硬碟 Repo + 內接硬碟 venv 使用說明](instruction.md)
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

## 分割模組（MedSAM2：Manifest + Prompt）

建議改用 `manifest` 模式（不依賴 `.npz`），可直接串接 detection prompt：

```bash
python -m segmentation.finetune_medsam2.build_manifest --image_dir "E:/dataset/images" --mask_dir "E:/dataset/masks" --output_json "segmentation/manifests/dataset_segmentation.json" --relative_paths

python -m segmentation.finetune_medsam2.main --data_mode manifest --manifest "segmentation/manifests/dataset_segmentation.json" --prompt_mode hybrid --epochs 100
```

可選參數：
- `--det_prompt_json <path>`：載入 detection prompt JSON
- `--prompt_jitter_px <int>`：訓練時對 prompt bbox 加擾動

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
