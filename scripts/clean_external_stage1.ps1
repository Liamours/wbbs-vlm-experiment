[CmdletBinding()]
param(
    [switch]$Execute,
    [switch]$Verify
)

$ErrorActionPreference = 'Stop'

if ($Execute -and $Verify) {
    throw 'Use either -Verify or -Execute, not both.'
}

$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$externalRoot = (Resolve-Path (Join-Path $workspace 'external')).Path
$retentionRecord = Join-Path $workspace 'datasets\benchmarks\RAW_SOURCE_RETENTION.md'
$manifests = @{
    'datasets\benchmarks\heal_medvqa.standard.jsonl' = '0F6BEB73F1F26A4AB609E4921F2B3A8132039E936F05164FA36854E12401F9C9'
    'datasets\benchmarks\pathvqa.standard.jsonl' = '00FFDA31DD26AA9F7BE43167FE83F7D59F42508C4F9533082DB18B19F457D380'
    'datasets\benchmarks\medsg_bench_text_labels.standard.jsonl' = '58732E0E593FEB15CF666588AACC89AA67508108C0236F8B7D375EECABD61E47'
    'datasets\benchmarks\slake.standard.jsonl' = '3383799EC162427F42F50E041B4BD39639708F54B9E3557922FC586AAA30586C'
}
$targets = @('heal-medvqa', 'pathvqa', 'medsg-bench', 'slake') |
    ForEach-Object { (Resolve-Path (Join-Path $externalRoot $_)).Path }

if (-not (Test-Path -LiteralPath $retentionRecord)) {
    throw "Retention record is missing: $retentionRecord"
}
foreach ($entry in $manifests.GetEnumerator()) {
    $path = Join-Path $workspace $entry.Key
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required retained manifest is missing: $path"
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    if ($actual -ne $entry.Value) {
        throw "Retained manifest hash mismatch: $path"
    }
}

if ($Verify) {
    $planned = foreach ($target in $targets) {
        if (-not $target.StartsWith($externalRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to inspect outside the workspace external root: $target"
        }
        $bytes = (Get-ChildItem -LiteralPath $target -Recurse -File -Force | Measure-Object Length -Sum).Sum
        [pscustomobject]@{ path = $target; bytes = $bytes; gib = [math]::Round($bytes / 1GB, 3) }
    }
    $planned | Format-Table -AutoSize
    Write-Output "Verified retained manifests and planned workspace-only cleanup; no files were changed."
    return
}

if (-not $Execute) {
    throw 'Safety stop: run with -Verify to validate or -Execute to delete the four listed workspace directories.'
}

$deleted = @()
foreach ($target in $targets) {
    if (-not $target.StartsWith($externalRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to delete outside the workspace external root: $target"
    }
    $bytes = (Get-ChildItem -LiteralPath $target -Recurse -File -Force | Measure-Object Length -Sum).Sum
    Remove-Item -LiteralPath $target -Recurse -Force
    $deleted += [pscustomobject]@{ path = $target; bytes = $bytes; gib = [math]::Round($bytes / 1GB, 3) }
}

$auditDirectory = Join-Path $workspace 'results\analyses\dataset-audits\storage_cleanup'
New-Item -ItemType Directory -Path $auditDirectory -Force | Out-Null
@{
    completed_at = (Get-Date).ToString('o')
    scope = 'workspace external raw benchmark cleanup only'
    retention_record = $retentionRecord
    deleted = $deleted
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $auditDirectory 'stage1_external_cleanup.json') -Encoding utf8

Write-Output "Deleted $($deleted.Count) verified raw benchmark directories from $externalRoot."
