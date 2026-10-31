[CmdletBinding()]
param(
    [ValidateSet('qwen3vl', 'paligemma', 'both')]
    [string]$Model = 'both',
    [switch]$Preflight,
    [switch]$Smoke,
    [string]$ConfigPath
)

# Blind-image check for the two reported adapters: the validation split with a blank image in place of every scan.
# Same decoding, batch size and 4-bit loading as the reported validation runs. There is no test-split option on purpose.
# Re-running the same command continues an interrupted run from the rows already written.

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$project = Split-Path -Parent (Split-Path -Parent $root)
Set-Location $project
$env:PYTHONPATH = "$root\src" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

if (-not $ConfigPath) { $ConfigPath = "$root\configs\blind_image_261031_1105.json" }
if (-not (Test-Path -LiteralPath $ConfigPath)) { throw "Missing configuration: $ConfigPath" }
$cfg = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json

$outputDir = Join-Path $project $cfg.output_dir
$logDir = Join-Path $project $cfg.log_dir
$manifestDir = Join-Path $outputDir 'manifests'
$blindManifest = Join-Path $manifestDir 'multitask_val_blind.jsonl'
$smokeManifest = Join-Path $manifestDir 'multitask_val_blind_smoke.jsonl'
$provenancePath = Join-Path $manifestDir 'blind_manifest.provenance.json'
$sourceManifest = Join-Path $project $cfg.source_manifest
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$launcherLog = Join-Path $logDir "r3-blind-image-launcher-$stamp.log"
$uv = (Get-Command uv -ErrorAction Stop).Source
$selectedModels = if ($Model -eq 'both') { @('qwen3vl', 'paligemma') } else { @($Model) }

function Write-Log([string]$message) {
    $line = "[$(Get-Date -Format s)] $message"
    Write-Host $line
    Add-Content -LiteralPath $launcherLog -Value $line -Encoding UTF8
}

function Format-Duration([double]$seconds) {
    $span = [TimeSpan]::FromSeconds([math]::Max(0, [math]::Round($seconds)))
    return ('{0:00}:{1:00}:{2:00}' -f [math]::Floor($span.TotalHours), $span.Minutes, $span.Seconds)
}

function Get-LineCount([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { return 0 }
    $stream = [System.IO.File]::Open($path, 'Open', 'Read', 'ReadWrite')
    try {
        $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)
        $count = 0
        while ($null -ne ($line = $reader.ReadLine())) {
            if ($line.Trim().Length -gt 0) { $count++ }
        }
        return $count
    } finally {
        $stream.Dispose()
    }
}

function Assert-ValidationOnly {
    if ($cfg.split -ne 'val') { throw "The blind-image check runs on the validation split only, config says: $($cfg.split)" }
    foreach ($path in @($sourceManifest, $blindManifest, $smokeManifest)) {
        if ((Split-Path -Leaf $path) -match 'test') { throw "Refusing a manifest named like the locked test split: $path" }
    }
}

function Assert-NoBlockedProcess {
    $running = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $cfg.blocked_processes -contains $_.ProcessName } | Select-Object -ExpandProperty ProcessName -Unique)
    if ($running.Count -gt 0) {
        throw "Close these apps first, they compete for RAM and the GPU: $($running -join ', ')"
    }
}

function Get-GpuState {
    $query = & nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0 -or -not $query) { throw 'nvidia-smi found no GPU' }
    $parts = ($query | Select-Object -First 1).Split(',') | ForEach-Object { $_.Trim() }
    return [pscustomobject]@{ Name = $parts[0]; TotalMb = [int]$parts[1]; FreeMb = [int]$parts[2] }
}

function Assert-ResourcesFree {
    $gpu = Get-GpuState
    if ($gpu.FreeMb -lt $cfg.min_free_vram_mb) {
        throw "GPU has $($gpu.FreeMb) MiB free, the run needs $($cfg.min_free_vram_mb) MiB. Close whatever uses the GPU"
    }
    $freeGb = (Get-Item -LiteralPath $project).PSDrive.Free / 1GB
    if ($freeGb -lt $cfg.min_free_disk_gb) { throw "Only $([math]::Round($freeGb, 1)) GB free on the project drive, need $($cfg.min_free_disk_gb) GB" }
    Write-Log "GPU $($gpu.Name), $($gpu.FreeMb) of $($gpu.TotalMb) MiB free; disk $([math]::Round($freeGb, 1)) GB free"
}

function Test-SourceChecksum {
    $sums = Join-Path $project $cfg.source_checksums
    $name = Split-Path -Leaf $sourceManifest
    $line = Get-Content -LiteralPath $sums | Where-Object { $_ -match "^([0-9a-f]{64})\s+\*?$([regex]::Escape($name))$" } | Select-Object -First 1
    if (-not $line) { throw "No checksum for $name in $sums" }
    $expected = ($line -split '\s+')[0].ToUpperInvariant()
    if ((Get-FileHash -LiteralPath $sourceManifest -Algorithm SHA256).Hash -ne $expected) { throw "Checksum mismatch: $sourceManifest" }
}

function Invoke-Uv([string[]]$arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $uv run --no-sync --project $root @arguments 2>&1 | ForEach-Object { "$_" } | Out-Host } finally { $ErrorActionPreference = $previous }
    if ($LASTEXITCODE -ne 0) { throw "uv run $($arguments[0]) failed with exit code $LASTEXITCODE" }
}

function Initialize-Manifests {
    Write-Log 'Building or verifying the blank-image validation manifest (CPU only)'
    Invoke-Uv @('python', '-m', 'analysis.blind_image', '--config', $ConfigPath)
    $provenance = Get-Content -LiteralPath $provenancePath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (@($provenance.splits_in_manifest).Count -ne 1 -or $provenance.splits_in_manifest[0] -ne 'val') { throw 'Blind manifest holds a split other than val' }
    if ($provenance.rows -ne $cfg.expected_rows) { throw "Blind manifest has $($provenance.rows) rows, expected $($cfg.expected_rows)" }
    if ((Get-FileHash -LiteralPath $blindManifest -Algorithm SHA256).Hash -ne $provenance.manifest_sha256.ToUpperInvariant()) { throw 'Blind manifest changed after it was built' }
    if ((Get-FileHash -LiteralPath $smokeManifest -Algorithm SHA256).Hash -ne $provenance.smoke_sha256.ToUpperInvariant()) { throw 'Smoke manifest changed after it was built' }
    return $provenance
}

function Test-ModelFiles([string]$key) {
    $entry = $cfg.models.$key
    foreach ($relative in @($entry.config, "$($entry.adapter)\adapter_config.json", "$($entry.adapter)\adapter_model.safetensors", $entry.real_image_validation_metrics)) {
        if (-not (Test-Path -LiteralPath (Join-Path $project $relative))) { throw "$key preflight: missing $relative" }
    }
    $snapshots = Join-Path $project $entry.base_snapshot_dir
    if (-not (Test-Path -LiteralPath $snapshots) -or -not (Get-ChildItem -LiteralPath $snapshots -Directory)) { throw "$key preflight: no base model snapshot in $($entry.base_snapshot_dir)" }
    $exe = Join-Path $root ".venv\Scripts\$($entry.predict_entry).exe"
    if (-not (Test-Path -LiteralPath $exe)) { throw "$key preflight: $exe is missing, run: uv sync --project repositories\wbbs-vqa_vgrounding --extra training --extra eval" }
    if ($key -eq 'paligemma') {
        $paligemma = Get-Content -LiteralPath (Join-Path $project $entry.config) -Raw | ConvertFrom-Json
        if ($paligemma.quantization -ne '4bit') { throw 'paligemma config is not 4bit' }
    }
}

function Save-ProgressFile([string]$key, [string]$mode, [int]$done, [int]$total, [double]$rate, [double]$etaSeconds) {
    $payload = [ordered]@{
        model = $key; mode = $mode; rows_completed = $done; rows_total = $total
        rows_per_second = [math]::Round($rate, 4); eta_seconds = [math]::Round($etaSeconds)
        updated = (Get-Date -Format s)
    }
    ($payload | ConvertTo-Json) | Set-Content -LiteralPath (Join-Path $logDir "r3-blind-image-$key-$mode.progress.json") -Encoding UTF8
}

function Wait-WithEta($process, [string]$key, [string]$mode, [string]$predictions, [int]$total, [int]$startDone, [int]$pollSeconds) {
    $minimumSample = [math]::Max(16, [math]::Ceiling($total * 0.01))
    $firstTime = $null
    $firstCount = 0
    $peakUsedMb = 0
    $begun = Get-Date
    while (-not $process.HasExited) {
        Start-Sleep -Seconds $pollSeconds
        $done = Get-LineCount $predictions
        if ($null -eq $firstTime -and $done -gt $startDone) {
            $firstTime = Get-Date
            $firstCount = $done
        }
        $gpu = Get-GpuState
        $peakUsedMb = [math]::Max($peakUsedMb, $gpu.TotalMb - $gpu.FreeMb)
        $elapsed = ((Get-Date) - $begun).TotalSeconds
        $rate = 0.0
        $eta = 0.0
        $message = "$key $mode $done/$total ($([math]::Round(100.0 * $done / $total, 1))%) elapsed $(Format-Duration $elapsed) GPU in use $($gpu.TotalMb - $gpu.FreeMb) MiB (peak $peakUsedMb)"
        if ($null -ne $firstTime -and ($done - $firstCount) -ge $minimumSample) {
            $rate = ($done - $firstCount) / [math]::Max(1.0, ((Get-Date) - $firstTime).TotalSeconds)
            $eta = ($total - $done) / $rate
            $message += " $([math]::Round($rate, 3)) rows/s ETA $(Format-Duration $eta) finish $((Get-Date).AddSeconds($eta).ToString('HH:mm'))"
        } else {
            $message += ' ETA pending (loading the model or warming up)'
        }
        Write-Log $message
        Save-ProgressFile $key $mode $done $total $rate $eta
    }
    $process.WaitForExit()
}

function Invoke-Prediction([string]$key, [string]$mode, [string]$manifest, [int]$total) {
    if ($manifest -ne $blindManifest -and $manifest -ne $smokeManifest) { throw "Refusing to predict on $manifest, only the blind validation manifests are allowed" }
    $entry = $cfg.models.$key
    $directory = Join-Path $outputDir "$key\$mode"
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $predictions = Join-Path $directory 'predictions.jsonl'
    if ($mode -eq 'smoke' -and (Test-Path -LiteralPath $predictions)) { Remove-Item -LiteralPath $predictions }
    $done = Get-LineCount $predictions
    if ($done -gt $total) { throw "$predictions has $done rows, more than the $total expected" }
    if ($done -eq $total) {
        Write-Log "$key $mode predictions already complete ($done rows)"
        return $predictions
    }
    $arguments = @(
        $entry.predict_entry,
        '--config', (Join-Path $project $entry.config),
        '--adapter', (Join-Path $project $entry.adapter),
        '--input', $manifest,
        '--output', $predictions,
        '--max-new-tokens', [string]$cfg.max_new_tokens,
        '--batch-size', [string]$entry.batch_size
    )
    if ($done -gt 0) {
        $arguments += '--resume'
        Write-Log "$key $mode resuming after $done of $total rows"
    }
    $quoted = ($arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $commandLine = "run --no-sync --project `"$root`" $quoted"
    $runName = "r3-blind-image-$key-$mode-$stamp"
    $stdout = Join-Path $logDir "$runName.stdout.log"
    $stderr = Join-Path $logDir "$runName.stderr.log"
    Write-Log "$key $mode command: uv $commandLine"
    Write-Log "$key $mode logs: $stdout and $stderr"
    $started = Get-Date
    $process = Start-Process -FilePath $uv -ArgumentList $commandLine -WorkingDirectory $project -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $null = $process.Handle
    try {
        $poll = if ($mode -eq 'smoke') { $cfg.smoke_poll_seconds } else { $cfg.poll_seconds }
        Wait-WithEta $process $key $mode $predictions $total $done $poll
    } finally {
        if (-not $process.HasExited) {
            & taskkill /T /F /PID $process.Id | Out-Null
            Write-Log "$key $mode stopped; rerun the same command to continue from the rows already written"
        }
    }
    if ($process.ExitCode -ne 0) { throw "$key $mode inference failed with exit code $($process.ExitCode), see $stderr" }
    $finished = Get-LineCount $predictions
    if ($finished -ne $total) { throw "$key $mode wrote $finished of $total rows" }
    $seconds = ((Get-Date) - $started).TotalSeconds
    Write-Log "$key $mode inference complete: $finished rows in $(Format-Duration $seconds) (this attempt, model loading included)"
    return $predictions
}

function Invoke-Evaluation([string]$key, [string]$predictions) {
    $directory = Split-Path -Parent $predictions
    Write-Log "$key validation evaluation with the existing evaluator"
    Invoke-Uv @('wbbs-evaluate-qwen3', '--task', 'multitask', '--expected', $sourceManifest, '--predictions', $predictions, '--output', (Join-Path $directory 'metrics.json'))
    Invoke-Uv @('python', '-m', 'analysis.qwen_multitask', '--expected', $sourceManifest, '--predictions', $predictions, '--output', (Join-Path $directory 'audit.json'))
}

function Show-Projection([string]$key, [int]$total) {
    $timing = $cfg.models.$key.reference_timing
    $seconds = $total / $timing.rows_per_second
    $where = if ($timing.measured_on_this_gpu) { 'measured on this GPU' } else { 'measured on another GPU, so a lower bound here' }
    Write-Log "$key reference: $($timing.rows_per_second) rows/s ($($timing.measured_on); $where) -> about $(Format-Duration $seconds) for $total rows"
}

Write-Log "Blind-image check, split val, models: $($selectedModels -join ', '), mode: $(if ($Preflight) { 'preflight' } elseif ($Smoke) { 'smoke' } else { 'full' })"
Write-Log "Config $ConfigPath; launcher log $launcherLog"
Assert-ValidationOnly
Test-SourceChecksum
foreach ($key in $selectedModels) { Test-ModelFiles $key }
$provenance = Initialize-Manifests
Write-Log "Manifest ready: $($provenance.rows) validation rows, tasks $((($provenance.task_counts | ConvertTo-Json -Compress)))"
foreach ($key in $selectedModels) { Show-Projection $key $provenance.rows }

if ($Preflight) {
    $running = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $cfg.blocked_processes -contains $_.ProcessName } | Select-Object -ExpandProperty ProcessName -Unique)
    Write-Log "Preflight complete. Big apps running now: $(if ($running.Count) { $running -join ', ' } else { 'none' }). No model was loaded."
    exit 0
}

Assert-NoBlockedProcess
Assert-ResourcesFree

if ($Smoke) {
    foreach ($key in $selectedModels) {
        Assert-NoBlockedProcess
        $null = Invoke-Prediction $key 'smoke' $smokeManifest $provenance.smoke_rows
    }
    Write-Log "Smoke complete. Open the smoke predictions under $outputDir, check that the JSON parses, then start the full run without -Smoke."
    exit 0
}

foreach ($key in $selectedModels) {
    $smokeFile = Join-Path $outputDir "$key\smoke\predictions.jsonl"
    if ((Get-LineCount $smokeFile) -ne $provenance.smoke_rows) { throw "Run the smoke test for $key first: .\scripts\run_blind_image_check.ps1 -Model $key -Smoke" }
}
foreach ($key in $selectedModels) {
    Assert-NoBlockedProcess
    $predictions = Invoke-Prediction $key 'validation' $blindManifest $provenance.rows
    Invoke-Evaluation $key $predictions
}
Write-Log "Blind-image check complete. Metrics: $outputDir\<model>\validation\metrics.json; compare with the real-image validation metrics named in the config."
