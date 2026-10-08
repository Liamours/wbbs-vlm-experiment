$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$project = Split-Path -Parent (Split-Path -Parent $root)
Set-Location $project
$env:PYTHONPATH = "$root\src" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })

$python = Join-Path $root '.venv\Scripts\python.exe'
$train = Join-Path $root '.venv\Scripts\qwen3vl-train.exe'
$predict = Join-Path $root '.venv\Scripts\qwen3vl-predict.exe'
$evaluate = Join-Path $root '.venv\Scripts\wbbs-evaluate-qwen3.exe'
$config = "$root\configs\qwen3vl_r2_multitask_2epoch_4050.json"
$release = 'datasets\releases\r2'
$exports = 'datasets\model_exports\qwen3_json\r2'
$imageRoot = 'datasets\sources\bs80k-imaging-raw'
$run = 'models\r2-multitask-qwen3vl-2b-json-2epoch'
$results = $run -replace '^models', 'results\inferences'
if ($results -eq $run) { throw "Run directory must be under models: $run" }
$validation = "$exports\multitask_val.jsonl"
$test = "$exports\multitask_test.jsonl"

function Invoke-Step([string]$name, [scriptblock]$command) {
    Write-Output "[$(Get-Date -Format s)] $name"
    & $command
    if ($LASTEXITCODE -ne 0) {
        throw "$name failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Evaluation([string]$label, [string]$adapter, [string]$split, [string]$input) {
    $output = "$results\evaluation\$label\$split"
    New-Item -ItemType Directory -Path $output -Force | Out-Null
    $predictions = "$output\predictions.jsonl"
    Invoke-Step "$label $split inference" {
        & $predict --config $config --adapter $adapter --input $input --output $predictions --max-new-tokens 160
    }
    Invoke-Step "$label $split evaluation" {
        & $evaluate --task multitask --expected $input --predictions $predictions --output "$output\metrics.json"
    }
    Invoke-Step "$label $split audit" {
        & $python -m analysis.qwen_multitask --expected $input --predictions $predictions --output "$output\audit.json"
    }
}

Invoke-Step 'R2 multitask export' {
    & $python -m postprocess.qwen --release $release --output $exports --schema multitask_json --image-root $imageRoot
}
Invoke-Step 'Two-epoch conventional training' {
    & $train --config $config
}

foreach ($epoch in 1, 2) {
    $label = "epoch-$('{0:D2}' -f $epoch)"
    $adapter = "$run\checkpoints\$label"
    $output = "$results\selection"
    New-Item -ItemType Directory -Path $output -Force | Out-Null
    $predictions = "$output\${label}_val_predictions.jsonl"
    Invoke-Step "$label full validation inference" {
        & $predict --config $config --adapter $adapter --input $validation --output $predictions --max-new-tokens 160
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
    Invoke-Evaluation $label "$run\$label" 'test' $test
}

Write-Output "[$(Get-Date -Format s)] complete"
