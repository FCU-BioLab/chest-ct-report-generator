param(
    [string]$LunaRoot = "E:\lung_ct_lesion_dataset\LUNA16",
    [string]$AnnotationsCsv = "annotations.csv",
    [string]$OutputDir = "",
    [int]$BatchSize = 200,
    [bool]$IncludeAnnotation = $true,
    [switch]$DryRun,
    [string]$NbiaUsername = "nbia_guest",
    [string]$NbiaPassword = "",
    [string]$ClientId = "NBIA"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-AccessToken {
    param(
        [string]$Username,
        [string]$Password,
        [string]$NbiaClientId
    )

    $tokenEndpoint = "https://services.cancerimagingarchive.net/nbia-api/oauth/token"
    $form = "username={0}&password={1}&client_id={2}&grant_type=password" -f `
        [System.Uri]::EscapeDataString($Username), `
        [System.Uri]::EscapeDataString($Password), `
        [System.Uri]::EscapeDataString($NbiaClientId)

    $tokenResponse = Invoke-RestMethod `
        -Method Post `
        -Uri $tokenEndpoint `
        -ContentType "application/x-www-form-urlencoded" `
        -Body $form

    if (-not $tokenResponse.access_token) {
        throw "Cannot get NBIA access token."
    }

    return $tokenResponse.access_token
}

function New-ManifestFromSeries {
    param(
        [string[]]$SeriesUids,
        [string]$AccessToken,
        [bool]$NeedAnnotation
    )

    $manifestEndpoint = "https://services.cancerimagingarchive.net/nbia-api/services/getManifestTextV2"
    $pairs = New-Object System.Collections.Generic.List[string]
    foreach ($uid in $SeriesUids) {
        $pairs.Add("list=$([System.Uri]::EscapeDataString($uid))")
    }
    $pairs.Add("includeAnnotation=$($NeedAnnotation.ToString().ToLowerInvariant())")
    $formBody = [string]::Join("&", $pairs)

    $response = Invoke-RestMethod `
        -Method Post `
        -Uri $manifestEndpoint `
        -Headers @{ Authorization = "Bearer $AccessToken" } `
        -ContentType "application/x-www-form-urlencoded" `
        -Body $formBody

    if ($response -is [string]) {
        return $response.TrimEnd("`r", "`n")
    }

    if ($response -is [byte[]]) {
        return [System.Text.Encoding]::UTF8.GetString($response).TrimEnd("`r", "`n")
    }

    if ($response -is [System.Array] -and $response.Length -gt 0) {
        $allNumeric = $true
        foreach ($item in $response) {
            if ($item -isnot [byte] -and $item -isnot [int]) {
                $allNumeric = $false
                break
            }
        }
        if ($allNumeric) {
            $bytes = New-Object byte[] ($response.Length)
            for ($i = 0; $i -lt $response.Length; $i++) {
                $bytes[$i] = [byte]$response[$i]
            }
            return [System.Text.Encoding]::UTF8.GetString($bytes).TrimEnd("`r", "`n")
        }
    }

    throw "Unexpected manifest response type: $($response.GetType().FullName)"
}

if (-not (Test-Path -LiteralPath $LunaRoot)) {
    throw "LUNA16 path not found: $LunaRoot"
}

$csvPath = Join-Path $LunaRoot $AnnotationsCsv
if (-not (Test-Path -LiteralPath $csvPath)) {
    throw "annotations.csv not found: $csvPath"
}

if ($BatchSize -lt 1) {
    throw "BatchSize must be >= 1"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $LunaRoot "lidc_minimal_download"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "Reading: $csvPath"
$rows = Import-Csv -Path $csvPath
if (-not $rows) {
    throw "No row in CSV: $csvPath"
}
if (-not ($rows[0].PSObject.Properties.Name -contains "seriesuid")) {
    throw "CSV must contain 'seriesuid' column."
}

$allUids = $rows |
    ForEach-Object { $_.seriesuid } |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    ForEach-Object { $_.Trim() } |
    Sort-Object -Unique

if (-not $allUids) {
    throw "No valid seriesuid found."
}

$allUidPath = Join-Path $OutputDir "all_seriesuids.txt"
Set-Content -Path $allUidPath -Value $allUids -Encoding ASCII
Write-Host ("Unique seriesuid: {0}" -f $allUids.Count)
Write-Host "Saved: $allUidPath"

$batchCount = [int][Math]::Ceiling($allUids.Count / [double]$BatchSize)
Write-Host ("Batch size: {0}, batches: {1}" -f $BatchSize, $batchCount)

$manifestFiles = New-Object System.Collections.Generic.List[string]
$batchUidFiles = New-Object System.Collections.Generic.List[string]
$downloadCmdFiles = New-Object System.Collections.Generic.List[string]

$accessToken = $null
if (-not $DryRun) {
    Write-Host "Requesting NBIA token..."
    $accessToken = Get-AccessToken -Username $NbiaUsername -Password $NbiaPassword -NbiaClientId $ClientId
    Write-Host "Token acquired."
}

for ($i = 0; $i -lt $batchCount; $i++) {
    $start = $i * $BatchSize
    $end = [Math]::Min($start + $BatchSize - 1, $allUids.Count - 1)
    $uidsBatch = $allUids[$start..$end]
    $batchId = "{0:D4}" -f ($i + 1)

    $uidFile = Join-Path $OutputDir ("batch_{0}.seriesuids.txt" -f $batchId)
    Set-Content -Path $uidFile -Value $uidsBatch -Encoding ASCII
    $batchUidFiles.Add($uidFile) | Out-Null

    if ($DryRun) {
        Write-Host ("[DryRun] Prepared batch {0}: {1} UIDs" -f $batchId, $uidsBatch.Count)
        continue
    }

    Write-Host ("Creating manifest {0} ({1} UIDs)..." -f $batchId, $uidsBatch.Count)
    $manifestText = New-ManifestFromSeries -SeriesUids $uidsBatch -AccessToken $accessToken -NeedAnnotation $IncludeAnnotation
    if (-not ($manifestText -match "^downloadServerUrl=")) {
        throw "Manifest format check failed for batch $batchId. First line: $($manifestText.Split([Environment]::NewLine)[0])"
    }

    $manifestFile = Join-Path $OutputDir ("batch_{0}.tcia" -f $batchId)
    Set-Content -Path $manifestFile -Value $manifestText -Encoding ASCII
    $manifestFiles.Add($manifestFile) | Out-Null

    $cmdFile = Join-Path $OutputDir ("download_batch_{0}.cmd" -f $batchId)
    $cmd = @(
        "@echo off"
        "REM Edit the path to your NBIA Data Retriever executable if needed."
        "set NBIA_EXE=C:\Program Files\NBIA Data Retriever\NBIA Data Retriever.exe"
        "if not exist ""%NBIA_EXE%"" set NBIA_EXE=C:\Program Files\NBIA Data Retriever\NBIADataRetriever.exe"
        """%NBIA_EXE%"" ""$manifestFile"""
    ) -join [Environment]::NewLine
    Set-Content -Path $cmdFile -Value $cmd -Encoding ASCII
    $downloadCmdFiles.Add($cmdFile) | Out-Null
}

$summary = [PSCustomObject]@{
    luna_root = $LunaRoot
    annotations_csv = $csvPath
    output_dir = $OutputDir
    unique_seriesuid_count = $allUids.Count
    batch_size = $BatchSize
    batch_count = $batchCount
    include_annotation = $IncludeAnnotation
    dry_run = $DryRun.IsPresent
    all_seriesuids_file = $allUidPath
    batch_uid_files = $batchUidFiles
    manifest_files = $manifestFiles
    download_cmd_files = $downloadCmdFiles
    generated_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
}

$summaryPath = Join-Path $OutputDir "summary.json"
$summary | ConvertTo-Json -Depth 4 | Set-Content -Path $summaryPath -Encoding UTF8
Write-Host "Saved summary: $summaryPath"

if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run done. Re-run without -DryRun to generate .tcia manifests."
} else {
    Write-Host ""
    Write-Host "Manifest generation done."
    Write-Host "Open each .tcia file with NBIA Data Retriever, or run generated .cmd files."
}
