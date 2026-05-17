param(
    [int]$Epochs = 100,
    [int]$BatchSize = 2,
    [int]$AccumulationSteps = 8,
    [int]$NumWorkers = 0,
    [double]$LearningRate = 5e-6,
    [double]$WeightDecay = 1e-6,
    [int]$EarlyStoppingPatience = 20,
    [int]$Seed = 27,
    [double]$DataFraction = 1.0,
    [string]$OutputDir = "",
    [switch]$Use25D,
    [switch]$NoVisualizations
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repoRoot

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$python = Join-Path $repoRoot "venv\Scripts\python.exe"
$checkpoint = Join-Path $repoRoot "segmentation\MedSAM2\checkpoints\MedSAM2_CTLesion.pt"
$cacheDir = "cache"

if (-not (Test-Path $python)) {
    throw "Python venv not found: $python"
}
if (-not (Test-Path $checkpoint)) {
    throw "MedSAM2 checkpoint not found: $checkpoint"
}
if (-not (Test-Path (Join-Path $repoRoot "segmentation\cache\lndb_slices"))) {
    throw "LNDb cache link not found: segmentation\cache\lndb_slices"
}

$argsList = @(
    "-m", "segmentation.finetune_medsam2.main",
    "--data_mode", "cache",
    "--cache_dir", $cacheDir,
    "--cache_dataset_type", "lndb",
    "--checkpoint", $checkpoint,
    "--epochs", $Epochs,
    "--batch_size", $BatchSize,
    "--accumulation_steps", $AccumulationSteps,
    "--num_workers", $NumWorkers,
    "--lr", $LearningRate,
    "--weight_decay", $WeightDecay,
    "--loss_type", "precision",
    "--warmup_epochs", 5,
    "--early_stopping_patience", $EarlyStoppingPatience,
    "--data_fraction", $DataFraction,
    "--seed", $Seed
)

if ($OutputDir.Trim().Length -gt 0) {
    $argsList += @("--output_dir", $OutputDir)
}
if ($Use25D) {
    $argsList += "--use_2_5d"
}
if ($NoVisualizations) {
    $argsList += "--no_visualizations"
}

& $python @argsList
