[CmdletBinding()]
param(
    [switch]$Preflight,
    [switch]$Smoke,
    [switch]$Resume,
    [switch]$EvaluateTest,
    [ValidateSet('4050', '5070')]
    [string]$Profile = '5070',
    [string]$ConfigPath
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$project = Split-Path -Parent (Split-Path -Parent $root)
Set-Location $project
$env:PYTHONPATH = "$root\src" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })

$python = Join-Path $root '.venv\Scripts\python.exe'
$train = Join-Path $root '.venv\Scripts\qwen3vl-train.exe'
$predict = Join-Path $root '.venv\Scripts\qwen3vl-predict.exe'
$evaluate = Join-Path $root '.venv\Scripts\wbbs-evaluate-qwen3.exe'
$variant = if ($Smoke) { 'smoke' } else { '2epoch' }
$config = if ($ConfigPath) {
    $ConfigPath
} else {
    "$root\configs\qwen3vl_r3_multitask_${variant}_${Profile}$(if ($Smoke) { '' } else { '_fast' }).json"
}
if (-not (Test-Path -LiteralPath $config)) {
    throw "Missing configuration: $config"
}
$exports = 'datasets\model_exports\qwen3_json\r3'
$run = (Get-Content -LiteralPath $config -Raw | ConvertFrom-Json).runtime.output_dir
$results = $run -replace '^models', 'results\inferences'
if ($results -eq $run) { throw "Run directory must be under models: $run" }
$validation = "$exports\multitask_val.jsonl"
$test = "$exports\multitask_test.jsonl"
$maxNewTokens = 96

function Invoke-Step([string]$name, [scriptblock]$command) {
    Write-Output "[$(Get-Date -Format s)] $name"
    & $command
    if ($LASTEXITCODE -ne 0) {
        throw "$name failed with exit code $LASTEXITCODE"
    }
}

function Test-ExportChecksums {
    $checksumPath = Join-Path $exports 'SHA256SUMS.txt'
    if (-not (Test-Path -LiteralPath $checksumPath)) {
        throw "Missing R3 export checksum manifest: $checksumPath"
    }
    foreach ($line in Get-Content -LiteralPath $checksumPath) {
        if ($line -notmatch '^([0-9a-f]{64})\s+(.+)$') {
            throw "Invalid checksum line: $line"
        }
        $expected = $matches[1].ToUpperInvariant()
        $path = Join-Path $exports $matches[2]
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing checked export: $path"
        }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        if ($actual -ne $expected) {
            throw "Export checksum mismatch: $path"
        }
    }
}

function Invoke-Evaluation([string]$label, [string]$adapter, [string]$split, [string]$manifestPath) {
    $output = "$results\evaluation\$label\$split"
    New-Item -ItemType Directory -Path $output -Force | Out-Null
    $predictions = "$output\predictions.jsonl"
    Invoke-Step "$label $split inference" {
        & $predict --config $config --adapter $adapter --input $manifestPath --output $predictions --max-new-tokens $maxNewTokens
    }
    Invoke-Step "$label $split evaluation" {
        & $evaluate --task multitask --expected $manifestPath --predictions $predictions --output "$output\metrics.json"
    }
    Invoke-Step "$label $split audit" {
        & $python -m analysis.qwen_multitask --expected $manifestPath --predictions $predictions --output "$output\audit.json"
    }
}

Test-ExportChecksums
Invoke-Step 'R3 multitask image and manifest preflight' {
    & $train --config $config --dry-run
}
Invoke-Step 'R3 launch guardrails' {
    $arguments = @("$root\scripts\check_qwen3_r3_launch.py", '--config', $config, '--profile', $Profile)
    if ($Preflight) { $arguments += '--skip-gpu' }
    if ($Resume) { $arguments += '--resume' }
    & $python @arguments
}
if ($Preflight) {
    Write-Output "[$(Get-Date -Format s)] R3 $Profile $variant preflight complete; no model training was started."
    exit 0
}
if ((Test-Path -LiteralPath $run) -and -not $Resume) {
    throw "Refusing to overwrite existing run directory: $run"
}

if ($Smoke) {
    Invoke-Step "R3 $Profile 50-step smoke training" {
        & $train --config $config
    }
    Write-Output "[$(Get-Date -Format s)] R3 $Profile smoke run complete"
    exit 0
}

Invoke-Step 'R3 two-epoch conventional training' {
    if ($Resume) {
        & $train --config $config --resume
    } else {
        & $train --config $config
    }
}

foreach ($epoch in 1, 2) {
    $label = "epoch-$('{0:D2}' -f $epoch)"
    $adapter = "$run\checkpoints\$label"
    $output = "$results\selection"
    New-Item -ItemType Directory -Path $output -Force | Out-Null
    $predictions = "$output\${label}_val_predictions.jsonl"
    Invoke-Step "$label full validation inference" {
        & $predict --config $config --adapter $adapter --input $validation --output $predictions --max-new-tokens $maxNewTokens
    }
    Invoke-Step "$label full validation evaluation" {
        & $evaluate --task multitask --expected $validation --predictions $predictions --output "$output\${label}_val_metrics.json"
    }
    Invoke-Step "$label full validation audit" {
        & $python -m analysis.qwen_multitask --expected $validation --predictions $predictions --output "$output\${label}_val_audit.json"
    }
}

Invoke-Step 'Validation-only best-checkpoint selection' {
    & $python "$root\scripts\select_qwen3_multitask_best.py" --run-dir $run --selection-dir "$results\selection"
}

foreach ($label in 'best', 'latest') {
    Invoke-Evaluation $label "$run\$label" 'validation' $validation
}

if ($EvaluateTest) {
    Invoke-Evaluation 'best' "$run\best" 'test' $test
}

Write-Output "[$(Get-Date -Format s)] R3 $Profile fast multitask run complete"
