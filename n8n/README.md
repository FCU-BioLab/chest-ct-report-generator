# n8n 流程執行器

此資料夾提供 n8n 與本專案 Python pipeline 的串接方式。

## 主要檔案

- `run_case_pipeline.py`
- `docker-compose.yml`
- `.env`
- `.env.example`
- `workflows/chest_ct_pipeline_5_stages.json`

## 外接硬碟/搬遷後的必要設定

這個 workflow 不再寫死 `C:\GitHub\...`，改為以下兩種方式擇一提供路徑：

1. 設定環境變數

```powershell
$env:CHEST_CT_REPO_ROOT = 'E:/chest-ct-report-generator'
$env:CHEST_CT_PYTHON = 'E:/chest-ct-report-generator/venv/Scripts/python.exe'
```

2. 呼叫 webhook 時直接帶入 `repo_root` 與 `python_exe`

若沒有提供 `python_exe`，workflow 會預設使用 `<repo_root>/venv/Scripts/python.exe`。

## 方式一：Windows CMD 直接啟動（建議）

```powershell
npm install -g n8n
n8n
```

開啟 UI：`http://localhost:5678`

## 方式二：Docker（選用）

```powershell
cd E:/chest-ct-report-generator/n8n
docker-compose up -d
docker-compose ps
docker-compose logs -f n8n
```

停止/重啟：

```powershell
docker-compose stop
docker-compose start
docker-compose down
```

## 呼叫案例流程（CLI 範例）

### preprocess

```powershell
E:/chest-ct-report-generator/venv/Scripts/python.exe E:/chest-ct-report-generator/n8n/run_case_pipeline.py --stage preprocess --case-id case-001 --input-path <CT_OR_DICOM_PATH>
```

### detect

```powershell
E:/chest-ct-report-generator/venv/Scripts/python.exe E:/chest-ct-report-generator/n8n/run_case_pipeline.py --stage detect --case-id case-001 --model-path <DETECTION_MODEL_PATH> --threshold 0.5 --device cuda
```

### segment

```powershell
E:/chest-ct-report-generator/venv/Scripts/python.exe E:/chest-ct-report-generator/n8n/run_case_pipeline.py --stage segment --case-id case-001
```

### feature

```powershell
E:/chest-ct-report-generator/venv/Scripts/python.exe E:/chest-ct-report-generator/n8n/run_case_pipeline.py --stage feature --case-id case-001
```

### report

```powershell
E:/chest-ct-report-generator/venv/Scripts/python.exe E:/chest-ct-report-generator/n8n/run_case_pipeline.py --stage report --case-id case-001 --use-llm
```

## Webhook payload 範例

```json
{
  "case_id": "case-001",
  "input_path": "E:/dataset/case-001.nii.gz",
  "model_path": "E:/models/detection/model.pt",
  "repo_root": "E:/chest-ct-report-generator",
  "python_exe": "E:/chest-ct-report-generator/venv/Scripts/python.exe",
  "threshold": 0.5,
  "device": "cuda",
  "use_llm": true
}
```

## 備註

- 建議先確保 `run_case_pipeline.py --help` 可正常執行。
- 若你把 repo 整個搬到外接硬碟，只要更新 `CHEST_CT_REPO_ROOT`，workflow 指向會一起更新。
- 若 `venv` 也一起搬，建議重建一次虛擬環境，避免舊的絕對路徑殘留。
