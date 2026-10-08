<#
.SYNOPSIS
Downloads the requested public HEAL-MedVQA Parquet shards and repository metadata with restart support.

.EXAMPLE
.\scripts\download_heal_medvqa.ps1

.EXAMPLE
.\scripts\download_heal_medvqa.ps1 -Destination 'D:\datasets\HEAL-MedVQA'

.NOTES
The selected 41 Parquet files occupy about 12.7 GiB at the repository's current
revision. Downloading is sequential and restartable: incomplete files retain a
.part suffix and curl resumes them on the next run.
#>
[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\..\..\external\heal-medvqa'),
    [ValidateRange(1, 20)]
    [int]$Retries = 5,
    [switch]$RestartPartial
)

$ErrorActionPreference = 'Stop'

$repository = 'MM-Hallu/HEAL-MedVQA'
$revision = 'main'
$files = @(
    '.gitattributes',
    'README.md',
    'data/test-00000-of-00001.parquet',
    'data/train-00000-of-00001.parquet',
    'test-00000-of-00006.parquet',
    'test-00001-of-00006.parquet',
    'test-00002-of-00006.parquet',
    'test-00003-of-00006.parquet',
    'test-00004-of-00006.parquet',
    'test-00005-of-00006.parquet',
    'train-00000-of-00033.parquet',
    'train-00001-of-00033.parquet',
    'train-00002-of-00033.parquet',
    'train-00003-of-00033.parquet',
    'train-00004-of-00033.parquet',
    'train-00005-of-00033.parquet',
    'train-00006-of-00033.parquet',
    'train-00007-of-00033.parquet',
    'train-00008-of-00033.parquet',
    'train-00009-of-00033.parquet',
    'train-00010-of-00033.parquet',
    'train-00011-of-00033.parquet',
    'train-00012-of-00033.parquet',
    'train-00013-of-00033.parquet',
    'train-00014-of-00033.parquet',
    'train-00015-of-00033.parquet',
    'train-00016-of-00033.parquet',
    'train-00017-of-00033.parquet',
    'train-00018-of-00033.parquet',
    'train-00019-of-00033.parquet',
    'train-00020-of-00033.parquet',
    'train-00021-of-00033.parquet',
    'train-00022-of-00033.parquet',
    'train-00023-of-00033.parquet',
    'train-00024-of-00033.parquet',
    'train-00025-of-00033.parquet',
    'train-00026-of-00033.parquet',
    'train-00027-of-00033.parquet',
    'train-00028-of-00033.parquet',
    'train-00029-of-00033.parquet',
    'train-00030-of-00033.parquet',
    'train-00031-of-00033.parquet',
    'train-00032-of-00033.parquet'
)

$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $curl) {
    throw 'curl.exe is required but was not found on PATH.'
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$destinationPath = (Resolve-Path -LiteralPath $Destination).Path
$driveName = (Split-Path -Qualifier $destinationPath).Replace(':', '').Replace('\\', '')
$drive = Get-PSDrive -Name $driveName -ErrorAction SilentlyContinue
if ($drive -and $drive.Free -lt 14GB) {
    Write-Warning "Only $([math]::Round($drive.Free / 1GB, 2)) GiB is free. Reserve at least 14 GiB before downloading."
}

$headers = @()
if ($env:HF_TOKEN) {
    $headers += '--header'
    $headers += "Authorization: Bearer $($env:HF_TOKEN)"
}

$completed = 0
foreach ($file in $files) {
    $target = Join-Path $destinationPath $file
    $partial = "$target.part"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null

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
    Write-Host "[$($completed + 1)/$($files.Count)] Downloading: $file"
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

$totalBytes = (Get-ChildItem -LiteralPath $destinationPath -Filter '*.parquet' -File -Recurse | Measure-Object -Property Length -Sum).Sum
Write-Host "Completed $completed shard(s) in $destinationPath ($([math]::Round($totalBytes / 1GB, 2)) GiB)."
