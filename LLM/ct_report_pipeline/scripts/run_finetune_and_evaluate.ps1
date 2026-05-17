param(
    [string]$Python = "python",
    [string]$ModelName = "meta-llama/Llama-3.2-1B-Instruct",
    [string]$TrainData = "assets/data/finetune_reportdata_pipeline_train.jsonl",
    [string]$ValData = "assets/data/finetune_reportdata_pipeline_val.jsonl",
    [int]$Epochs = 5,
    [int]$BatchSize = 1,
    [int]$GradientAccumulation = 8,
    [double]$LearningRate = 0.0002,
    [int]$LoraRank = 16,
    [int]$MaxLength = 2048,
    [int]$GenerationEvalSamples = 0,
    [int]$GenerationEvalMaxNewTokens = 1024,
    [switch]$Use8Bit
)

$ErrorActionPreference = "Stop"

$PipelineRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $PipelineRoot

$argsList = @(
    "scripts/finetune_llama.py",
    "--model_name", $ModelName,
    "--data_path", $TrainData,
    "--val_data_path", $ValData,
    "--epochs", $Epochs,
    "--batch_size", $BatchSize,
    "--gradient_accumulation", $GradientAccumulation,
    "--learning_rate", $LearningRate,
    "--lora_r", $LoraRank,
    "--max_length", $MaxLength,
    "--generation_eval_samples", $GenerationEvalSamples,
    "--generation_eval_max_new_tokens", $GenerationEvalMaxNewTokens
)

if ($Use8Bit) {
    $argsList += "--use_8bit"
}

Write-Host "Running fine-tune and full validation generation evaluation..."
Write-Host "Pipeline root: $PipelineRoot"
Write-Host "Train data: $TrainData"
Write-Host "Validation data: $ValData"

& $Python @argsList
