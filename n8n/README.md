# n8n Flow Runner

This folder keeps orchestration scripts for n8n.
LLM core modules remain in `llm/ct_report_pipeline`.

## Files
- `run_case_pipeline.py`
- `docker-compose.yml`
- `.env`
- `.env.example`

## Start n8n service (Windows CMD, recommended)

If your pipeline dependencies are installed in:

`C:\GitHub\chest-ct-report-generator\venv`

run n8n directly on Windows CMD, not Docker.  
Docker n8n runs in Linux container and cannot use your Windows venv Python.

1. Install n8n globally (once):
```bash
npm install -g n8n
```

2. Start n8n from CMD:
```bash
n8n
```

3. Open UI:
- `http://localhost:5678`

## Docker mode (optional)

Use this only if you also prepare a Linux Python runtime/dependencies inside container.

1. Go to n8n folder:
```bash
cd C:\GitHub\chest-ct-report-generator\n8n
```

2. (Optional) adjust credentials in `.env`.

3. Start service:
```bash
docker-compose up -d
```

4. Check status/logs:
```bash
docker-compose ps
docker-compose logs -f n8n
```

4. Open UI:
- `http://localhost:5678`

## Stop/Restart n8n

```bash
docker-compose stop
docker-compose start
docker-compose down
```

## Script stages
- `preprocess`
- `detect`
- `segment`
- `feature`
- `report`
- `run` (all stages)

## Required runtime (pipeline script)
Use project venv:

```bash
C:\GitHub\chest-ct-report-generator\venv\Scripts\python.exe
```

## Example (5-stage flow)

```bash
C:\GitHub\chest-ct-report-generator\venv\Scripts\python.exe C:\GitHub\chest-ct-report-generator\n8n\run_case_pipeline.py --stage preprocess --case-id case-001 --input-path <CT_OR_DICOM_PATH>
```

```bash
C:\GitHub\chest-ct-report-generator\venv\Scripts\python.exe C:\GitHub\chest-ct-report-generator\n8n\run_case_pipeline.py --stage detect --case-id case-001 --model-path <DETECTION_MODEL_PATH> --threshold 0.5 --device cuda
```

```bash
C:\GitHub\chest-ct-report-generator\venv\Scripts\python.exe C:\GitHub\chest-ct-report-generator\n8n\run_case_pipeline.py --stage segment --case-id case-001
```

```bash
C:\GitHub\chest-ct-report-generator\venv\Scripts\python.exe C:\GitHub\chest-ct-report-generator\n8n\run_case_pipeline.py --stage feature --case-id case-001
```

```bash
C:\GitHub\chest-ct-report-generator\venv\Scripts\python.exe C:\GitHub\chest-ct-report-generator\n8n\run_case_pipeline.py --stage report --case-id case-001 --use-llm
```

Each run updates:
- `n8n/runtime/<case-id>/state.json`
