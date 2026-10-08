$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$project = Split-Path -Parent (Split-Path -Parent $root)
Set-Location $project

$predict = Join-Path $root '.venv\Scripts\qwen3vl-predict.exe'
$evaluate = Join-Path $root '.venv\Scripts\wbbs-evaluate-qwen3.exe'
$audit = Join-Path $root '.venv\Scripts\wbbs-audit-qwen3.exe'
$config = "$root\configs\qwen3vl_r2_grounded_vqa_4050.json"
$adapter = 'models\r2-grounded-vqa-qwen3vl-2b-json'
$results = $adapter -replace '^models', 'results\inferences'
New-Item -ItemType Directory -Force $results | Out-Null
$test = 'datasets\model_exports\qwen3_json\r2\grounded_vqa_test.jsonl'
$validation = 'datasets\model_exports\qwen3_json\r2\grounded_vqa_val.jsonl'

function Invoke-Step([string]$name, [scriptblock]$command) {
    Write-Output "[$(Get-Date -Format s)] $name"
    & $command
    if ($LASTEXITCODE -ne 0) {
        throw "$name failed with exit code $LASTEXITCODE"
    }
}

Invoke-Step '1/6 test inference' {
    & $predict --config $config --adapter $adapter --input $test --output "$results\test_full_predictions.jsonl" --max-new-tokens 128
}
Invoke-Step '2/6 test evaluation' {
    & $evaluate --output-schema json --expected $test --predictions "$results\test_full_predictions.jsonl" --output "$results\test_full_metrics.json"
}
Invoke-Step '3/6 validation inference' {
    & $predict --config $config --adapter $adapter --input $validation --output "$results\val_full_predictions.jsonl" --max-new-tokens 128
}
Invoke-Step '4/6 validation evaluation' {
    & $evaluate --output-schema json --expected $validation --predictions "$results\val_full_predictions.jsonl" --output "$results\val_full_metrics.json"
}
Invoke-Step '5/6 test audit' {
    & $audit --schema json --manifest $test --predictions "$results\test_full_predictions.jsonl" --output "$results\test_full_audit.json"
}
Invoke-Step '6/6 validation audit' {
    & $audit --schema json --manifest $validation --predictions "$results\val_full_predictions.jsonl" --output "$results\val_full_audit.json"
}

Write-Output "[$(Get-Date -Format s)] complete"
