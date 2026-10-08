<#
.SYNOPSIS
Downloads MedSG-Bench task text, coordinates, and labels without image archives.

.DESCRIPTION
Downloads the eight JSON task files from both MedSG-Bench and MedSG-Train,
plus README.md and croissant.json. It intentionally excludes every .zip image
archive and bench_task1.parquet. The selected files total about 200 MiB at the
repository's current revision.

.EXAMPLE
.\scripts\download_medsg_text_labels.ps1
#>
[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\..\..\external\medsg-bench'),
    [ValidateRange(1, 20)]
    [int]$Retries = 5,
    [switch]$RestartPartial
)

$ErrorActionPreference = 'Stop'
$repository = 'MedSG-Bench/MedSG-Bench'
$revision = 'main'
$files = @(
    'README.md',
    'croissant.json',
    'MedSG-Bench/Task1.json',
    'MedSG-Bench/Task2.json',
    'MedSG-Bench/Task3.json',
    'MedSG-Bench/Task4.json',
    'MedSG-Bench/Task5.json',
    'MedSG-Bench/Task6.json',
    'MedSG-Bench/Task7.json',
    'MedSG-Bench/Task8.json',
    'MedSG-Train/Task1.json',
    'MedSG-Train/Task2.json',
    'MedSG-Train/Task3.json',
    'MedSG-Train/Task4.json',
    'MedSG-Train/Task5.json',
    'MedSG-Train/Task6.json',
    'MedSG-Train/Task7.json',
    'MedSG-Train/Task8.json'
)

$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $curl) {
    throw 'curl.exe is required but was not found on PATH.'
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$destinationPath = (Resolve-Path -LiteralPath $Destination).Path
$headers = @()
if ($env:HF_TOKEN) {
    $headers += '--header'
    $headers += "Authorization: Bearer $($env:HF_TOKEN)"
}

$completed = 0
foreach ($file in $files) {
    $target = Join-Path $destinationPath $file
    $partial = "$target.part"
    $targetDirectory = Split-Path -Parent $target
    New-Item -ItemType Directory -Force -Path $targetDirectory | Out-Null

    if (Test-Path -LiteralPath $target) {
        $completed++
        Write-Host "[$completed/$($files.Count)] Exists: $file"
        continue
    }
    if ($RestartPartial -and (Test-Path -LiteralPath $partial)) {
        Remove-Item -LiteralPath $partial
    }

    # Braces prevent PowerShell from treating `?download` as part of $file.
    $url = "https://huggingface.co/datasets/$repository/resolve/$revision/${file}?download=true"
    Write-Host "[$($completed + 1)/$($files.Count)] Downloading text/labels: $file"
    $arguments = @(
        '--fail', '--location', '--continue-at', '-', '--retry', "$Retries",
        '--retry-all-errors', '--connect-timeout', '30', '--output', $partial
    ) + $headers + @($url)

    & $curl.Source @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed for $file (curl exit code $LASTEXITCODE). The partial file, if any, was kept for resumption."
    }
    if (-not (Test-Path -LiteralPath $partial) -or (Get-Item -LiteralPath $partial).Length -eq 0) {
        throw "Download completed without a non-empty output file: $file"
    }

    Move-Item -LiteralPath $partial -Destination $target
    $completed++
}

$totalBytes = (Get-ChildItem -LiteralPath $destinationPath -File -Recurse | Measure-Object -Property Length -Sum).Sum
Write-Host "Completed $completed text/label file(s) in $destinationPath ($([math]::Round($totalBytes / 1MB, 2)) MiB)."
