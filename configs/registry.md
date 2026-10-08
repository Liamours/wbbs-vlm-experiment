# Training config registry

New configs are named `<model>_<YYMMDD>_<HHMM>.json` (e.g. `paligemma_260726_0042.json`
= PaliGemma config created 2026-07-26 00:42) — no more descriptive suffixes like
`_fast`, `_2epoch`, `_smoke` in the filename. Whatever a config is for, why it
exists, and what run it produced belongs in this table, not encoded into the
filename. Add a row here for every new config before or immediately after
creating it.

Existing configs created before this convention (2026-07-26 and earlier) keep
their original descriptive filenames — several are referenced by fixed paths
in active automation (`scripts/run_qwen3_r3_multitask_2epoch_fast.ps1` builds
its config path from a formula that assumes the old names) and in already-
written provenance/documentation (`run_manifest.json` files, the exported
results package). They are catalogued below rather than renamed, to avoid
breaking that traceability. Only new configs going forward use the new
naming.

On 2026-10-07 every config's paths were rebased from repo-relative `data/` and `runs/` to project-relative `datasets/` and `models/`, and configs now resolve against the project folder (`src/project_paths.py`). Settings are unchanged. Config sha256 values in `run_manifest.json` files written before that date describe the old path strings.

## Registry

| Config file | Model | Task | Created | Run directory | Steps/epochs | Notes |
|---|---|---|---|---|---|---|
| `paligemma_4050.example.json` | PaliGemma-3B | vqa | 260714 | `runs/paligemma-vqa-4050` | 100 | Example/template config, not a real dataset |
| `paligemma_r1_anatomy_vqa.example.json` | PaliGemma-3B | vqa | 260715 | `runs/r1-anatomy-vqa-paligemma` | 1000 | R1 example config |
| `paligemma_r1_anatomy_grounding.example.json` | PaliGemma-3B | grounding | 260715 | `runs/r1-anatomy-grounding-paligemma` | 1000 | R1 example config |
| `paligemma_r1_anatomy_grounded_vqa.example.json` | PaliGemma-3B | grounded_vqa | 260715 | `runs/r1-anatomy-grounded-vqa-paligemma` | 1000 | R1 example config |
| `qwen3vl_r1_grounded_vqa_4050.preflight.json` | Qwen3-VL-2B | grounded_vqa | 260716 | `runs/preflight/qwen3vl-r1-grounded-vqa` | 1 | R1 preflight smoke check |
| `qwen3vl_r1_grounded_vqa_4050.accumulation-preflight.json` | Qwen3-VL-2B | grounded_vqa | 260716 | `runs/preflight/qwen3vl-r1-grounded-vqa-accumulation` | 1 | R1 grad-accumulation preflight |
| `qwen3vl_r1_grounded_vqa_4050.calibration.json` | Qwen3-VL-2B | grounded_vqa | 260716 | `runs/calibration/qwen3vl-r1-grounded-vqa` | 25 | R1 tag-grammar calibration |
| `qwen3vl_r1_grounded_vqa_4050.tags-balanced.calibration.json` | Qwen3-VL-2B | grounded_vqa | 260716 | `runs/calibration/qwen3vl-r1-grounded-vqa-tags-balanced` | 25 | R1 balanced-sampling tag calibration |
| `qwen3vl_r1_grounded_vqa_4050.json-balanced.calibration.json` | Qwen3-VL-2B | grounded_vqa | 260716 | `runs/calibration/qwen3vl-r1-grounded-vqa-json-balanced` | 25 | R1 balanced-sampling JSON-schema calibration |
| `qwen3vl_r1_grounded_vqa_4050.json` | Qwen3-VL-2B | grounded_vqa | 260716 | `runs/r1-grounded-vqa-qwen3vl-2b-json` | 1000 | R1 JSON-schema baseline |
| `qwen3vl_r2_grounded_vqa_4050.json` | Qwen3-VL-2B | grounded_vqa | 260716 | `runs/r2-grounded-vqa-qwen3vl-2b-json` | 1000 | R2 baseline |
| `qwen3vl_r2_multitask_2epoch_4050.json` | Qwen3-VL-2B | multitask | 260717 | `runs/r2-multitask-qwen3vl-2b-json-2epoch` | 2 epochs | R2 multitask baseline |
| `qwen3vl_r3_multitask_smoke_4050.json` | Qwen3-VL-2B | multitask | 260722 | `runs/r3-multitask-qwen3vl-2b-smoke-4050` | 50 | R3 smoke test, 4050 profile |
| `qwen3vl_r3_multitask_smoke_5070.json` | Qwen3-VL-2B | multitask | 260722 | `runs/r3-multitask-qwen3vl-2b-smoke-5070` | 50 | R3 smoke test, 5070 profile |
| `qwen3vl_r3_multitask_2epoch_4050.json` | Qwen3-VL-2B | multitask | 260722 | `runs/r3-multitask-qwen3vl-2b-json-2epoch` | 2 epochs | R3 multitask, 4050 profile, superseded by `_fast` variant |
| `qwen3vl_r3_multitask_2epoch_4050_fast.json` | Qwen3-VL-2B | multitask | 260723 | `runs/r3-unified-qwen3vl-2b-2epoch-4050-v2` | 2 epochs | RTX 4050 launch, stopped 2026-07-23 at 2.6% before any checkpoint (see `context/qwen3vl-unified-baseline-execution-log.md`) |
| `qwen3vl_r3_multitask_2epoch_5070_fast.json` | Qwen3-VL-2B | multitask | 260723 | `runs/r3-unified-qwen3vl-2b-2epoch-5070` | 2 epochs (7,716 steps) | **Completed R3 unified baseline.** epoch-01 selected as `best` on full validation (`multitask_macro` score 0.731), locked test evaluated. Full results in `wbbs-r3-baseline-results.zip` |
| `medgemma_r3_multitask_2epoch_5070_fast.json` | MedGemma-4B | multitask | 260724 | `runs/r3-unified-medgemma-4b-2epoch-5070` | 2 | Not yet run in this session's tracked work |
| `medgemma_r3_multitask_smoke_5070.json` | MedGemma-4B | multitask | 260724 | `runs/r3-multitask-medgemma-4b-smoke-5070` | 50 | Smoke config |
| `paligemma_r3_grounded_vqa_smoke_5070.json` | PaliGemma-3B | grounded_vqa | 260725 | `runs/r3-grounded-vqa-paligemma-3b-smoke-5070` | 20 | Smoke config |
| `paligemma_r3_grounded_vqa_2epoch_5070.json` | PaliGemma-3B | grounded_vqa | 260725 | `runs/r3-grounded-vqa-paligemma-3b-2epoch-5070` | 1,258 steps (~2 epochs) | **Completed R3 grounded_vqa-only baseline.** No epoch-boundary/validation-based checkpoint selection (single run to `max_steps`). Locked test evaluated. Task-scope mismatch vs the Qwen3-VL multitask baseline — see `context/model-comparison-fairness-protocol.md`. Full results in `wbbs-r3-baseline-results.zip` |
| `paligemma_260726_0042.json` | PaliGemma-3B | multitask | 260726 0042 | `runs/r3-multitask-paligemma-3b-2epoch-5070` | 7,716 steps (~2 epochs) | **Ready, not yet launched.** Built to close the task-scope and checkpoint-selection confounds vs the Qwen3-VL baseline: same task (`multitask`), same LoRA rank/alpha/dropout, same 7-module target coverage, same effective batch size, LR, step count, seed, and now the same epoch-boundary-checkpoint + validation-based best-of-2 selection protocol (`scripts/run_paligemma_multitask.ps1`). Verified end to end with a real small-scale GPU run before relying on it — see `context/model-comparison-fairness-protocol.md` Verification log. Awaiting go-ahead to launch the real multi-hour run. |
