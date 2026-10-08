<#
.SYNOPSIS
Checks that the requested HEAL-MedVQA shards are present and non-empty.

.EXAMPLE
.\scripts\verify_heal_medvqa.ps1
#>
[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\..\..\external\heal-medvqa')
)

$ErrorActionPreference = 'Stop'
$files = @(
    '.gitattributes', 'README.md',
    'data/test-00000-of-00001.parquet', 'data/train-00000-of-00001.parquet',
    'test-00000-of-00006.parquet', 'test-00001-of-00006.parquet', 'test-00002-of-00006.parquet',
    'test-00003-of-00006.parquet', 'test-00004-of-00006.parquet', 'test-00005-of-00006.parquet',
    'train-00000-of-00033.parquet', 'train-00001-of-00033.parquet', 'train-00002-of-00033.parquet',
    'train-00003-of-00033.parquet', 'train-00004-of-00033.parquet', 'train-00005-of-00033.parquet',
    'train-00006-of-00033.parquet', 'train-00007-of-00033.parquet', 'train-00008-of-00033.parquet',
    'train-00009-of-00033.parquet', 'train-00010-of-00033.parquet', 'train-00011-of-00033.parquet',
    'train-00012-of-00033.parquet', 'train-00013-of-00033.parquet', 'train-00014-of-00033.parquet',
    'train-00015-of-00033.parquet', 'train-00016-of-00033.parquet', 'train-00017-of-00033.parquet',
    'train-00018-of-00033.parquet', 'train-00019-of-00033.parquet', 'train-00020-of-00033.parquet',
    'train-00021-of-00033.parquet', 'train-00022-of-00033.parquet', 'train-00023-of-00033.parquet',
    'train-00024-of-00033.parquet', 'train-00025-of-00033.parquet', 'train-00026-of-00033.parquet',
    'train-00027-of-00033.parquet', 'train-00028-of-00033.parquet', 'train-00029-of-00033.parquet',
    'train-00030-of-00033.parquet', 'train-00031-of-00033.parquet', 'train-00032-of-00033.parquet'
)

if (-not (Test-Path -LiteralPath $Destination)) {
    throw "Download directory does not exist: $Destination"
}

$results = foreach ($file in $files) {
    $path = Join-Path $Destination $file
    if (Test-Path -LiteralPath $path) {
        $item = Get-Item -LiteralPath $path
        [PSCustomObject]@{ File = $file; Present = $item.Length -gt 0; SizeMiB = [math]::Round($item.Length / 1MB, 2) }
    } else {
        [PSCustomObject]@{ File = $file; Present = $false; SizeMiB = 0 }
    }
}

$results | Format-Table -AutoSize
$missing = @($results | Where-Object { -not $_.Present })
if ($missing.Count -gt 0) {
    Write-Error "$($missing.Count) of $($files.Count) expected shards are missing or empty."
    exit 1
}

$totalMiB = ($results | Measure-Object -Property SizeMiB -Sum).Sum
Write-Host "Verified presence of all $($files.Count) non-empty shards ($([math]::Round($totalMiB / 1024, 2)) GiB). Run standardize_heal_medvqa.py for a full Parquet decode."
