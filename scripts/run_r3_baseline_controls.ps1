$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$project = Split-Path -Parent (Split-Path -Parent $root)
Set-Location $project
$env:PYTHONPATH = "$root\src" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })

$python = Join-Path $root '.venv\Scripts\python.exe'
& $python -m analysis.baseline_controls `
    --train datasets\model_exports\qwen3_json\r3\multitask_train.jsonl `
    --validation datasets\model_exports\qwen3_json\r3\multitask_val.jsonl `
    --output results\metrics\r3-multitask-no-image-controls
if ($LASTEXITCODE -ne 0) {
    throw "R3 no-image controls failed with exit code $LASTEXITCODE"
}
