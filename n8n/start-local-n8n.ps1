param(
    [string]$RepoRoot = "C:/GitHub/chest-ct-report-generator",
    [string]$PythonExe = "C:/GitHub/chest-ct-report-generator/venv/Scripts/python.exe",
    [string]$NpmPrefix = "C:/tmp/npm-global",
    [string]$NpmCache = "C:/tmp/npm-cache",
    [string]$N8nUserFolder = "C:/GitHub/chest-ct-report-generator/n8n/local-data"
)

$ErrorActionPreference = "Stop"

$env:CHEST_CT_REPO_ROOT = $RepoRoot
$env:CHEST_CT_PYTHON = $PythonExe
$env:npm_config_prefix = $NpmPrefix
$env:npm_config_cache = $NpmCache
$env:N8N_USER_FOLDER = $N8nUserFolder
$env:N8N_BLOCK_ENV_ACCESS_IN_NODE = "false"
Remove-Item Env:DB_SQLITE_POOL_SIZE -ErrorAction SilentlyContinue
$env:N8N_GIT_NODE_DISABLE_BARE_REPOS = "true"
$env:N8N_DIAGNOSTICS_ENABLED = "false"
$env:N8N_VERSION_NOTIFICATIONS_ENABLED = "false"
$env:N8N_TEMPLATES_ENABLED = "false"

New-Item -ItemType Directory -Force -Path $NpmPrefix | Out-Null
New-Item -ItemType Directory -Force -Path $NpmCache | Out-Null
New-Item -ItemType Directory -Force -Path $N8nUserFolder | Out-Null

$n8nCmd = Join-Path $NpmPrefix "n8n.cmd"
if (-not (Test-Path $n8nCmd)) {
    npm.cmd install -g n8n@1
}

Remove-Item Env:npm_config_prefix -ErrorAction SilentlyContinue
Remove-Item Env:npm_config_cache -ErrorAction SilentlyContinue

$dbPath = Join-Path $N8nUserFolder ".n8n/database.sqlite"
$workflowPath = Join-Path $RepoRoot "n8n/workflows/chest_ct_pipeline_5_stages.json"
if ((-not (Test-Path $dbPath)) -and (Test-Path $workflowPath)) {
    & $n8nCmd import:workflow --input $workflowPath
}

& $n8nCmd start
