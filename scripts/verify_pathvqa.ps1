<#
.SYNOPSIS
Checks that the complete selected PathVQA mirror download is present and non-empty.
#>
[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\..\..\external\pathvqa')
)

$ErrorActionPreference = 'Stop'
$files = @(
    '.gitattributes', 'README.md', 'scripts/processing.py',
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
    Write-Error "$($missing.Count) of $($files.Count) expected files are missing or empty."
    exit 1
}

$totalMiB = ($results | Measure-Object -Property SizeMiB -Sum).Sum
Write-Host "Verified all $($files.Count) PathVQA files ($([math]::Round($totalMiB, 2)) MiB)."
