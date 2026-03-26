# n8n 流程執行器

此資料夾提供 n8n 與本專案 Python pipeline 的串接方式。

## 主要檔案

- `run_case_pipeline.py`
- `docker-compose.yml`
- `.env`
- `.env.example`

## 方式一：Windows CMD 直接啟動（建議）

若你的 Python 相依都在：`C:\GitHub\chest-ct-report-generator\venv`

建議直接在 Windows CMD 啟動 n8n，不用 Docker（Docker 容器無法直接使用 Windows venv）。

```bash
npm install -g n8n
n8n
```

開啟 UI：`http://localhost:5678`

## 方式二：Docker（選用）

```bash
cd C:\GitHub\chest-ct-report-generator\n8n
docker-compose up -d
docker-compose ps
docker-compose logs -f n8n
```

停止/重啟：

```bash
docker-compose stop
docker-compose start
docker-compose down
```

## 呼叫案例流程（CLI 範例）

### preprocess

```bash
C:\GitHub\chest-ct-report-generator\venv\Scripts\python.exe C:\GitHub\chest-ct-report-generator\n8n\run_case_pipeline.py --stage preprocess --case-id case-001 --input-path <CT_OR_DICOM_PATH>
```

### detect

```bash
C:\GitHub\chest-ct-report-generator\venv\Scripts\python.exe C:\GitHub\chest-ct-report-generator\n8n\run_case_pipeline.py --stage detect --case-id case-001 --model-path <DETECTION_MODEL_PATH> --threshold 0.5 --device cuda
```

### segment

```bash
C:\GitHub\chest-ct-report-generator\venv\Scripts\python.exe C:\GitHub\chest-ct-report-generator\n8n\run_case_pipeline.py --stage segment --case-id case-001
```

### feature

```bash
C:\GitHub\chest-ct-report-generator\venv\Scripts\python.exe C:\GitHub\chest-ct-report-generator\n8n\run_case_pipeline.py --stage feature --case-id case-001
```

### report

```bash
C:\GitHub\chest-ct-report-generator\venv\Scripts\python.exe C:\GitHub\chest-ct-report-generator\n8n\run_case_pipeline.py --stage report --case-id case-001 --use-llm
```

## 備註

- 建議先確保 `run_case_pipeline.py --help` 可正常執行。
- 若要一鍵串全流程，請在 n8n 工作流中按順序串接各 stage。
