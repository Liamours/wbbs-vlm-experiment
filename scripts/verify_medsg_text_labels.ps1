<#
.SYNOPSIS
Checks the text-and-label-only MedSG-Bench download.
#>
[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $PSScriptRoot '..\..\..\external\medsg-bench')
)

$ErrorActionPreference = 'Stop'
$files = @(
    'README.md', 'croissant.json',
    'MedSG-Bench/Task1.json', 'MedSG-Bench/Task2.json', 'MedSG-Bench/Task3.json', 'MedSG-Bench/Task4.json',
    'MedSG-Bench/Task5.json', 'MedSG-Bench/Task6.json', 'MedSG-Bench/Task7.json', 'MedSG-Bench/Task8.json',
    'MedSG-Train/Task1.json', 'MedSG-Train/Task2.json', 'MedSG-Train/Task3.json', 'MedSG-Train/Task4.json',
    'MedSG-Train/Task5.json', 'MedSG-Train/Task6.json', 'MedSG-Train/Task7.json', 'MedSG-Train/Task8.json'
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
    Write-Error "$($missing.Count) of $($files.Count) expected text/label files are missing or empty."
    exit 1
}

$totalMiB = ($results | Measure-Object -Property SizeMiB -Sum).Sum
Write-Host "Verified all $($files.Count) text/label files ($([math]::Round($totalMiB, 2)) MiB)."
