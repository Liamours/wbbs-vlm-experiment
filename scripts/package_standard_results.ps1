$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
$out = "temp\wbbs-r3-results-standardized"
$zip = "results\inferences\wbbs-r3-results-standardized.zip"
Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue

$sources = @{
  "qwen3vl/val"       = "results\inferences\r3-unified-qwen3vl-2b-2epoch-5070\evaluation\best\validation"
  "qwen3vl/test"      = "results\inferences\r3-unified-qwen3vl-2b-2epoch-5070\evaluation\best\test"
  "paligemma/val"     = "results\inferences\r3-multitask-paligemma-3b-2epoch-5070\evaluation\best\validation"
  "paligemma/test"    = "results\inferences\r3-multitask-paligemma-3b-2epoch-5070\evaluation\best\test"
}

foreach ($pair in $sources.GetEnumerator()) {
  $dest = Join-Path $out $pair.Key
  New-Item -ItemType Directory -Path $dest -Force | Out-Null
  Copy-Item "$($pair.Value)\predictions.jsonl" "$dest\predictions.jsonl"
  $metrics = if (Test-Path "$($pair.Value)\metrics_full.json") { "$($pair.Value)\metrics_full.json" } else { "$($pair.Value)\metrics.json" }
  Copy-Item $metrics "$dest\metrics.json"
  if (Test-Path "$($pair.Value)\audit.json") { Copy-Item "$($pair.Value)\audit.json" "$dest\audit.json" }
  if (Test-Path "$($pair.Value)\bootstrap_confidence_intervals.json") { Copy-Item "$($pair.Value)\bootstrap_confidence_intervals.json" "$dest\bootstrap_confidence_intervals.json" }
}

Compress-Archive -Path $out -DestinationPath $zip -CompressionLevel Optimal -Force
Write-Output "done: $zip"
