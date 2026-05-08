# n8n 流程執行器

這個資料夾把 Chest CT 報告流程接到 n8n。目前可用的主流程是：

```text
Webhook -> preprocess -> detect -> segment -> feature -> report -> Respond
```

Python 端入口是 `n8n/run_case_pipeline.py`，每個 case 的狀態會寫到 `n8n/runtime/<case_id>/state.json`。

## 建議執行方式

在 Windows 開發機上建議用本機 n8n，而不是 Docker。原因是 workflow 的 Execute Command 會直接呼叫 Windows venv。

CMD 啟動：

```cmd
cd /d C:\GitHub\chest-ct-report-generator
n8n\start-local-n8n.cmd
```

也可以手動啟動：

```cmd
npm.cmd install -g n8n@1
set CHEST_CT_REPO_ROOT=C:\GitHub\chest-ct-report-generator
set CHEST_CT_PYTHON=C:\GitHub\chest-ct-report-generator\venv\Scripts\python.exe
n8n.cmd
```

PowerShell 啟動：

```powershell
npm.cmd install -g n8n@1
$env:CHEST_CT_REPO_ROOT = "C:/GitHub/chest-ct-report-generator"
$env:CHEST_CT_PYTHON = "C:/GitHub/chest-ct-report-generator/venv/Scripts/python.exe"
n8n.cmd
```

如果 npm 全域目錄被 Windows 權限擋住，可以改用專案提供的啟動腳本。它會把 npm prefix/cache 放到 `C:/tmp`：

```powershell
powershell -ExecutionPolicy Bypass -File n8n/start-local-n8n.ps1
```

啟動腳本會使用 `n8n/local-data` 作為 n8n 使用者資料夾，避免改到既有的 `n8n/data/database.sqlite`。

開啟 n8n UI：

```text
http://localhost:5678
```

匯入 workflow：

```text
n8n/workflows/chest_ct_pipeline_5_stages.json
```

啟用後 webhook path 是：

```text
POST http://localhost:5678/webhook/chest-ct-pipeline
```

測試 workflow 時，n8n 也會提供 test webhook URL；正式啟用後才使用 `/webhook/`。

## Webhook payload

最小 payload：

```json
{
  "case_id": "case-001",
  "input_path": "C:/path/to/ct-or-dicom-folder",
  "model_path": "C:/path/to/model_best.pt",
  "repo_root": "C:/GitHub/chest-ct-report-generator",
  "python_exe": "C:/GitHub/chest-ct-report-generator/venv/Scripts/python.exe"
}
```

完整 payload：

```json
{
  "case_id": "case-001",
  "input_path": "C:/path/to/ct-or-dicom-folder",
  "model_path": "C:/path/to/model_best.pt",
  "repo_root": "C:/GitHub/chest-ct-report-generator",
  "python_exe": "C:/GitHub/chest-ct-report-generator/venv/Scripts/python.exe",
  "work_dir": "C:/GitHub/chest-ct-report-generator/n8n/runtime",
  "threshold": 0.5,
  "device": "cuda",
  "medsam2_checkpoint": "C:/GitHub/chest-ct-report-generator/segmentation/MedSAM2/checkpoints/MedSAM2_CTLesion.pt",
  "use_llm": false,
  "no_propagate": false
}
```

PowerShell 呼叫範例：

```powershell
$body = @{
  case_id = "case-001"
  input_path = "C:/path/to/ct-or-dicom-folder"
  model_path = "C:/path/to/model_best.pt"
  repo_root = "C:/GitHub/chest-ct-report-generator"
  python_exe = "C:/GitHub/chest-ct-report-generator/venv/Scripts/python.exe"
  threshold = 0.5
  device = "cuda"
  use_llm = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:5678/webhook/chest-ct-pipeline" `
  -ContentType "application/json" `
  -Body $body
```

成功後回傳會包含 `state_json_path`。最終報告路徑也會寫在該 `state.json`：

```json
{
  "summary_html_path": ".../05_report/summary.html",
  "report_text_path": ".../05_report/AUTO_case-001.txt",
  "report_json_path": ".../05_report/AUTO_case-001.json",
  "total_process_seconds": 123.456
}
```

`summary.html` 會嵌入 detection 視覺化圖片，並列出每個 stage 的 process time。

## CLI 單機測試

先用 CLI 確認 Python pipeline 可以跑，再接 n8n：

```powershell
C:/GitHub/chest-ct-report-generator/venv/Scripts/python.exe `
  C:/GitHub/chest-ct-report-generator/n8n/run_case_pipeline.py `
  --stage run `
  --case-id case-001 `
  --input-path "C:/path/to/ct-or-dicom-folder" `
  --model-path "C:/path/to/model_best.pt" `
  --threshold 0.5 `
  --device cuda
```

也可以分段測：

```powershell
C:/GitHub/chest-ct-report-generator/venv/Scripts/python.exe C:/GitHub/chest-ct-report-generator/n8n/run_case_pipeline.py --stage preprocess --case-id case-001 --input-path "C:/path/to/ct-or-dicom-folder"
C:/GitHub/chest-ct-report-generator/venv/Scripts/python.exe C:/GitHub/chest-ct-report-generator/n8n/run_case_pipeline.py --stage detect --case-id case-001 --model-path "C:/path/to/model_best.pt" --threshold 0.5 --device cuda
C:/GitHub/chest-ct-report-generator/venv/Scripts/python.exe C:/GitHub/chest-ct-report-generator/n8n/run_case_pipeline.py --stage segment --case-id case-001
C:/GitHub/chest-ct-report-generator/venv/Scripts/python.exe C:/GitHub/chest-ct-report-generator/n8n/run_case_pipeline.py --stage feature --case-id case-001
C:/GitHub/chest-ct-report-generator/venv/Scripts/python.exe C:/GitHub/chest-ct-report-generator/n8n/run_case_pipeline.py --stage report --case-id case-001
```

## Docker 注意事項

`docker-compose.yml` 可以啟動 n8n UI，但 Docker 裡的 Execute Command 是在 Linux 容器內執行，不能直接執行 Windows 的 `venv/Scripts/python.exe`。如果要用 Docker，有兩個可行方向：

- 在容器內建立完整 Python/CUDA 環境，並把 `python_exe` 改成容器內 Python。
- 讓 n8n Docker 只負責 webhook，再用 HTTP Request 呼叫 Windows host 上的 Python API。

目前專案內的 `chest_ct_pipeline_5_stages.json` 是為「Windows 本機 n8n + Windows venv」設計。

## 輸出位置

每個 case 會建立：

```text
n8n/runtime/<case_id>/
  01_preprocess/
  02_detect/
  03_segment/
  04_feature/
  05_report/
  state.json
```

`n8n/.env`、`n8n/data`、`n8n/local-data`、`n8n/runtime` 是本機狀態與輸出，不應提交到版本控制。
