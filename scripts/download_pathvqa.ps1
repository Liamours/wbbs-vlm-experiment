<#
.SYNOPSIS
Downloads the complete public PathVQA Hugging Face mirror with restart support.

.DESCRIPTION
Downloads the 13 requested Parquet shards plus the repository README,
.gitattributes, and preprocessing script. The selected data occupy about
785 MiB at the repository's current revision.

.EXAMPLE
.\scripts\download_pathvqa.ps1
#>
[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\..\..\external\pathvqa'),
    [ValidateRange(1, 20)]
    [int]$Retries = 5,
    [switch]$RestartPartial
)

$ErrorActionPreference = 'Stop'
$repository = 'flaviagiammarino/path-vqa'
$revision = 'main'
$files = @(
    '.gitattributes',
    'README.md',
    'scripts/processing.py',
    'data/test-00000-of-00003-e9adadb4799f44d3.parquet',
    'data/test-00001-of-00003-7ea98873fc919813.parquet',
    'data/test-00002-of-00003-1628308435019820.parquet',
    'data/train-00000-of-00007-f2d0e9ef9f022d38.parquet',
    'data/train-00001-of-00007-47d8e0220bf6c933.parquet',
    'data/train-00002-of-00007-7fb5037c4c5da7be.parquet',
    'data/train-00003-of-00007-74b9b7b81cc55f90.parquet',
    'data/train-00004-of-00007-77eea90af4a55dce.parquet',
    'data/train-00005-of-00007-5332ec423be520bd.parquet',
    'data/train-00006-of-00007-637a58c700b604af.parquet',
    'data/validation-00000-of-00003-90a5518d26493b67.parquet',
    'data/validation-00001-of-00003-cbfe947a3418595c.parquet',
    'data/validation-00002-of-00003-9ec816895bd3bc20.parquet'
)

$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $curl) {
    throw 'curl.exe is required but was not found on PATH.'
}

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$destinationPath = (Resolve-Path -LiteralPath $Destination).Path
$driveName = (Split-Path -Qualifier $destinationPath).Replace(':', '').Replace('\\', '')
$drive = Get-PSDrive -Name $driveName -ErrorAction SilentlyContinue
if ($drive -and $drive.Free -lt 1GB) {
    Write-Warning "Only $([math]::Round($drive.Free / 1GB, 2)) GiB is free. Reserve at least 1 GiB before downloading."
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
Write-Host "Completed $completed file(s) in $destinationPath ($([math]::Round($totalBytes / 1MB, 2)) MiB of Parquet data)."
