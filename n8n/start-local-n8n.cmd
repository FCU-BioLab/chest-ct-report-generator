@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

if not "%~1"=="" set "REPO_ROOT=%~1"

set "PYTHON_EXE=%REPO_ROOT%\venv\Scripts\python.exe"
if not "%~2"=="" set "PYTHON_EXE=%~2"

set "NPM_PREFIX=C:\tmp\npm-global"
set "NPM_CACHE=C:\tmp\npm-cache"
set "N8N_USER_FOLDER=%REPO_ROOT%\n8n\local-data"

set "CHEST_CT_REPO_ROOT=%REPO_ROOT%"
set "CHEST_CT_PYTHON=%PYTHON_EXE%"
set "N8N_HOST=localhost"
set "N8N_PORT=5678"
set "N8N_PROTOCOL=http"
set "N8N_SECURE_COOKIE=false"
set "N8N_ENCRYPTION_KEY=replace-with-your-own-long-random-key"
set "N8N_USER_FOLDER=%N8N_USER_FOLDER%"

if not exist "%NPM_PREFIX%" mkdir "%NPM_PREFIX%"
if not exist "%NPM_CACHE%" mkdir "%NPM_CACHE%"
if not exist "%N8N_USER_FOLDER%" mkdir "%N8N_USER_FOLDER%"

set "N8N_CMD=%NPM_PREFIX%\n8n.cmd"
if not exist "%N8N_CMD%" (
    cmd.exe /c "set npm_config_prefix=%NPM_PREFIX%&& set npm_config_cache=%NPM_CACHE%&& npm.cmd install -g n8n@1"
    if errorlevel 1 exit /b 1
)

set "DB_PATH=%N8N_USER_FOLDER%\.n8n\database.sqlite"
set "WORKFLOW_PATH=%REPO_ROOT%\n8n\workflows\chest_ct_pipeline_5_stages.json"
if not exist "%DB_PATH%" (
    if exist "%WORKFLOW_PATH%" (
        call "%N8N_CMD%" import:workflow --input "%WORKFLOW_PATH%"
        if errorlevel 1 exit /b 1
    )
)

call "%N8N_CMD%" start
