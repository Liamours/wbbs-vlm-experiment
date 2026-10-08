# Dataset download scripts

The standardized comparison files these scripts write to `datasets/benchmarks/` are archived at `archive/datasets/benchmarks/`; re-running a script recreates them in `datasets/benchmarks/`.

## HEAL-MedVQA

The scripts download and verify the 41 requested Parquet files for the public `MM-Hallu/HEAL-MedVQA` release, plus its `README.md` and `.gitattributes`. They do not download files until executed.

```powershell
.\scripts\download_heal_medvqa.ps1
.\scripts\verify_heal_medvqa.ps1
```

The default destination is `external/heal-medvqa`. About 14 GiB of free disk space is recommended. Downloads resume from `.part` files after interruption. Pass `-RestartPartial` only to discard an incomplete shard and restart that shard from zero. The PowerShell verifier checks presence and non-empty files; the standardizer below performs a full Parquet decode.

After a verified complete download, standardize the configured root-level
`train-*.parquet` and `test-*.parquet` shards. It reads embedded image paths
without duplicating image bytes and converts each source segmentation RLE to a
tight grounding box.

```powershell
uv run --project repo\wbbs-vqa_vgrounding --with pyarrow python repo\wbbs-vqa_vgrounding\scripts\standardize_heal_medvqa.py
```

## PathVQA

The scripts download all 13 listed image-embedded Parquet shards, plus the mirror README, attributes, and `scripts/processing.py`.

```powershell
.\scripts\download_pathvqa.ps1
.\scripts\verify_pathvqa.ps1
```

The default destination is `external/pathvqa`; allow at least 1 GiB of free disk space.

To standardize the image-embedded Parquet release without extracting duplicate
image files:

```powershell
uv run --project repo\wbbs-vqa_vgrounding --with pyarrow python repo\wbbs-vqa_vgrounding\scripts\standardize_pathvqa.py
```

## MedSG-Bench text and labels only

This downloader selects `README.md`, `croissant.json`, and the 16 `Task*.json` files from `MedSG-Bench/` and `MedSG-Train/`. It explicitly excludes all `.zip` archives and `bench_task1.parquet`, so it does not download image payloads.

```powershell
.\scripts\download_medsg_text_labels.ps1
.\scripts\verify_medsg_text_labels.ps1
```

The default destination is `external/medsg-bench`; allow about 250 MiB of free space.

The standardizer preserves the image references and box labels but deliberately
does not validate image files, because this snapshot excludes the ZIP archives.

```powershell
uv run --project repo\wbbs-vqa_vgrounding python repo\wbbs-vqa_vgrounding\scripts\standardize_medsg_bench.py
```

## SLAKE standardization

After extracting the official archive to `external/slake/raw`, export the three
SLAKE splits to the repository's comparison-v1 schema:

```powershell
uv run --project repo\wbbs-vqa_vgrounding python repo\wbbs-vqa_vgrounding\scripts\standardize_slake.py
uv run --project repo\wbbs-vqa_vgrounding python -m analysis.benchmark `
  --standard-file datasets/benchmarks/slake.standard.jsonl `
  --release-name slake `
  --output-dir results/analyses/dataset-audits/slake
```

The exporter intentionally emits VQA records only: SLAKE co-releases masks and
boxes, but the original QA rows do not establish a per-question spatial target.

## Current standardized comparison exports

| Dataset | Standardized export | Audit |
|---|---|---|
| SLAKE | `datasets/benchmarks/slake.standard.jsonl` | `results/analyses/dataset-audits/slake/` |
| PathVQA | `datasets/benchmarks/pathvqa.standard.jsonl` | `results/analyses/dataset-audits/pathvqa/` |
| MedSG-Bench text/labels | `datasets/benchmarks/medsg_bench_text_labels.standard.jsonl` | `results/analyses/dataset-audits/medsg-bench/` |
| HEAL-MedVQA | `datasets/benchmarks/heal_medvqa.standard.jsonl` | `results/analyses/dataset-audits/heal-medvqa/` |

## WBBS R2 merged language release

R2 is generated from the frozen R1 manifests. It writes only the two merged
training groups `vqa.jsonl` and `vgrounding.jsonl`; each row records its
`region_level` (`whole_body`, `bone_region`, or `metastasis`) and interaction
type. Image references use the portable `bs80k/...` root, and grounded target
objects use the `r2-target-v1` JSON schema and the lesion -> bone-region ->
whole-body classification hierarchy. Rows retain raw/effective label fields
and a conflict flag. Generate a smoke release first, then
the complete release:

```powershell
uv run --project repo\wbbs-vqa_vgrounding python repo\wbbs-vqa_vgrounding\scripts\build_r2_diversified.py --input-release datasets/releases/r1 --output-dir datasets/releases/r2_sample --limit 200
uv run --project repo\wbbs-vqa_vgrounding python repo\wbbs-vqa_vgrounding\scripts\build_r2_diversified.py --input-release datasets/releases/r1 --output-dir datasets/releases/r2
```

Benchmark and compare the complete release:

```powershell
uv run --project repo\wbbs-vqa_vgrounding python -m analysis.benchmark --release-dir datasets/releases/r2 --release-name wbbs-r2 --output-dir results/analyses/dataset-audits/wbbs-r2
```

The combined comparison tables are written to
`results/analyses/dataset-audits/comparison/`.

## WBBS R3 compact release

R3 retains R2's full canonical evidence and frozen patient splits while
deterministically retaining 10--20% of consumer task rows. Its coverage seed
ensures every canonical image is present in at least one selected task row.
`multitask.jsonl` is the primary consumer manifest; each row declares its task
type and whether answer, classification, and/or box locations are required.

```powershell
uv run --project repo\wbbs-vqa_vgrounding python repo\wbbs-vqa_vgrounding\scripts\build_r3_compact.py --input-release datasets/releases/r2 --output-dir datasets/releases/r3 --fraction 0.15
uv run --project repo\wbbs-vqa_vgrounding python -m analysis.benchmark --release-dir datasets/releases/r3 --release-name wbbs-r3 --output-dir results/analyses/dataset-audits/wbbs-r3
```

## WBBS R3 fast Qwen3-VL baseline

`configs/qwen3vl_r3_multitask_2epoch_5070_fast.json` and
`configs/qwen3vl_r3_multitask_2epoch_4050_fast.json` train one
`Qwen/Qwen3-VL-2B-Instruct` LoRA adapter on all R3 train rows for two
conventional epochs. It learns VQA, localization, and grounded-VQA through one
structured JSON output. The profile fixes each image at 50,176 pixels and caps
generation at 96 tokens; the R3 target audit found a 90-token maximum. Its
12-GB RTX 5070-safe microbatch is one with gradient accumulation of eight;
it uses AdamW (5e-5 learning rate, 0.01 weight decay, betas 0.9/0.999), 3% linear warm-up,
cosine decay, and gradient clipping at 1.0. Console progress reports the
mean epoch loss, learning rate, throughput, ETA, and peak VRAM.

The runner preserves R1/R2 artifacts, verifies the existing R3 export checksums,
trains, selects the best epoch from the complete R3 validation split, and
evaluates/audits both the latest and best adapters on validation only. The locked
test split is evaluated once, for the validation-selected adapter, only when
`-EvaluateTest` is explicitly supplied.

```powershell
.\scripts\run_qwen3_r3_multitask_2epoch_fast.ps1 -Preflight
.\scripts\run_qwen3_r3_multitask_2epoch_fast.ps1
.\scripts\run_qwen3_r3_multitask_2epoch_fast.ps1 -Profile 4050 -Preflight
.\scripts\run_qwen3_r3_multitask_2epoch_fast.ps1 -Profile 4050
```

The 4050 profile keeps the same training microbatch and optimizer settings but
uses inference batch size one to fit paired images on lower VRAM.

Before a real run, use the non-training launch check. It validates model-cache
availability, disk capacity, immutable manifest hashes, and (on the target PC)
the selected CUDA GPU. It does not load weights onto the GPU or train.

```powershell
.\scripts\run_qwen3_r3_multitask_2epoch_fast.ps1 -Profile 5070 -Preflight
.\scripts\run_qwen3_r3_multitask_2epoch_fast.ps1 -Profile 4050 -Preflight
```

The `-Smoke` option is prepared but intentionally separate from conventional
training. It runs exactly 50 optimizer steps into its own output directory,
reporting peak VRAM and throughput for an evidence-based ETA.

```powershell
.\scripts\run_qwen3_r3_multitask_2epoch_fast.ps1 -Profile 5070 -Smoke
.\scripts\run_qwen3_r3_multitask_2epoch_fast.ps1 -Profile 4050 -Smoke
```

Conventional runs write a `run_manifest.json` with configuration, export
checksums, GPU/CUDA/PyTorch data, and optimizer settings. They also save two
rotating durable checkpoints every 250 optimizer steps. Resume an interrupted
run without repeating completed optimizer steps:

```powershell
.\scripts\run_qwen3_r3_multitask_2epoch_fast.ps1 -Profile 5070 -Resume
```

Generate the no-image comparison controls before VLM training. `majority` uses
only task-level training priors; `prompt_only` uses prompt tokens and
anatomical-region priors, never images. Both are evaluated and schema-audited
with the same R3 multitask evaluator as the VLM.

```powershell
.\scripts\run_r3_baseline_controls.ps1
```
