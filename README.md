# WBBS VQA and visual-grounding dataset pipeline

This repository builds evidence-first manifests for bone-scintigraphy VQA,
referring-expression grounding, and grounded VQA. Dataset construction is pure
Python and runs with `uv`; model training is optional and separate.

## Modules

| Directory | Responsibility |
| --- | --- |
| `src/preprocess` | Source inventory, BS80K adapter, templates, verification, canonical manifests |
| `src/analysis` | Validation, audit, and dataset summaries |
| `src/metric` | Small VQA and grounding metrics |
| `src/postprocess` | Model-specific exports such as PaliGemma JSONL |
| `src/training` | Existing fine-tuning code, kept separate from dataset construction |

`external/` contains study-only shallow clones. Nothing under `src/` imports it.

## Project layout

This repo sits at `repo\wbbs-vqa_vgrounding` inside the project folder. Data and models live in the project folder, not in the repo: `datasets\` (sources, releases, exports), `models\` (base weights and adapters), and `results\`. `src\project_paths.py` defines these folders; set `WBBS_PROJECT_ROOT` to point a standalone clone at another project folder. Run every command below from the project folder; training configs resolve their paths against it.

Publication documentation: [dataset card](DATASET_CARD.md), [license review](LICENSE_REVIEW.md), [changelog](CHANGELOG.md), and [publication plan](context/publish-readiness-plan.md).
The fixed non-image baseline results are in [BASELINES.md](BASELINES.md).

## Build a dataset

The builder accepts a source evidence JSONL. It writes four model-agnostic
artifacts: `canonical.jsonl`, `vqa.jsonl`, `grounding.jsonl`, and
`grounded_vqa.jsonl`.

```powershell
uv run --project repo\wbbs-vqa_vgrounding python -m preprocess.pipeline `
  --source datasets/source/evidence.jsonl `
  --output datasets/derived/v1
```

Every source record must contain `patient_id`, `image`, `image_size`, `view`,
and either `target` or the legacy `region` plus `bbox` fields. `qa` entries are
copied into VQA records without paraphrasing. Grounding records come from
explicit `grounding` entries and exact `caption_*` fields.

```json
{
  "patient_id": "1001",
  "image": "images/1001_ant.png",
  "image_size": [512, 1024],
  "view": "ANT",
  "region": "chest_left",
  "bbox": [120, 210, 330, 480],
  "caption_description": "The left chest region is shown.",
  "qa": [{
    "question": "Is abnormal tracer uptake present in the left chest?",
    "answer": "no",
    "template_id": "region_abnormality",
    "answer_rule": "diagnosis"
  }]
}
```

The builder preserves a complete supplied split (`source_split`) after checking
duplicate-linked patients. Otherwise it assigns deterministic 80/10/10
patient-level splits and keeps declared duplicate groups together.

## Validate and export

```powershell
uv run --project repo\wbbs-vqa_vgrounding python -m analysis.validate --input datasets/derived/v1/grounding.jsonl --kind grounding
uv run --project repo\wbbs-vqa_vgrounding python -m postprocess.paligemma --input datasets/derived/v1/vqa.jsonl --output datasets/derived/v1/paligemma_vqa.jsonl --task vqa
```

PaliGemma coordinate tokens are export-only. Canonical and task manifests store
original-image numeric boxes.

## Optional modules

```powershell
uv sync --extra dev
uv sync --extra analysis
uv sync --extra training
```

The historical image audit is available through `uv run --project repo\wbbs-vqa_vgrounding python -m analysis.audit`.
Training remains a separate future concern; it does not define the dataset schema.

## Compare an external VQA or grounding dataset

External JSON/JSONL files can be normalized to the loss-aware `comparison-v1`
interchange format and audited with the same metrics as WBBS:

```powershell
uv run --project repo\wbbs-vqa_vgrounding python -m analysis.benchmark `
  --standard-file path\to\source.json `
  --standardize `
  --standard-output datasets\benchmarks\source.standard.jsonl `
  --release-name source

uv run --project repo\wbbs-vqa_vgrounding python -m analysis.benchmark `
  --standard-file datasets\benchmarks\source.standard.jsonl `
  --release-name source `
  --output-dir results\analyses\dataset-audits\source
```

After generating reports for several datasets, combine them into comparison
tables:

```powershell
uv run --project repo\wbbs-vqa_vgrounding python -m analysis.benchmark `
  --compare-files results\analyses\dataset-audits\wbbs-r3\dataset_benchmark.json results\analyses\dataset-audits\source\dataset_benchmark.json `
  --output-dir results\analyses\dataset-audits\comparison
```

Use `--bbox-format xywh` or `--bbox-format coco` for COCO-style boxes. For
unusual field names, pass a JSON mapping with `--field-map`; unavailable fields
remain `null`/NR rather than being treated as zero. The interchange schema and
dataset download catalog are documented in
[`context/dataset-comparison-format.md`](context/dataset-comparison-format.md)
and [`reference/dataset-comparison-catalog.md`](reference/dataset-comparison-catalog.md).

## Build the WBBS R0 release

R0 is the first conservative vertical slice. Every task row receives both the
anterior and posterior images for one patient-region pair. View-specific rows
ask about one named image; neutral rows omit the view and use both boxes. BS80K
supplies images, region boxes, diagnosis, and XML metastasis boxes. LIBS-160K
supplies exact English captions only; its crop geometry is never used to infer
a box. The release excludes low-precision shoulder boxes, known posterior-elbow
artifacts, missing/corrupt images, and outlier whole-body images.

```powershell
uv run --project repo\wbbs-vqa_vgrounding python -m preprocess.release `
  --dataset-root datasets\sources `
  --output datasets\releases\r0 `
  --policy repo\wbbs-vqa_vgrounding\configs\dataset\r0.json
```

The release directory is self-auditing:

- `source_inventory.json`: source paths, counts, and annotation checksums.
- `metastases.jsonl`: every XML metastasis box and its unique, ambiguous, or
  unassigned region relationship.
- `paired_evidence.jsonl`: complete ANT+POST evidence pairs used by every task row.
- `evidence_candidates.jsonl`, `evidence_accepted.jsonl`, and
  `evidence_rejected.jsonl`: source-to-release traceability.
- `canonical.jsonl`, `vqa.jsonl`, `grounding.jsonl`, `grounded_vqa.jsonl`:
  model-agnostic task manifests. Grounding uses a short region expression; the
  original LIBS captions remain only in canonical evidence.
- `grounding_bbx.jsonl`: grounding export with targets such as
  `<ANT><BBX>134,129,187,281</BBX></ANT>`. Both-view targets also include a
  `<PST>...</PST>` block. Numeric boxes remain alongside them for auditability.
- `quality_report.json` and `release_summary.json`: paper and review inputs.

### Merged R2 language release

After the R1 release is frozen, generate `datasets/releases/r2/` with
`scripts/build_r2_diversified.py`. R2 presents one `vqa.jsonl` and one
`vgrounding.jsonl` across whole-body, bone-region, and metastasis levels. Rows
carry deterministic surface-form IDs plus `question_form` metadata; geometry,
labels, evidence IDs, and patient splits are unchanged. Classification follows
the lesion -> bone-region -> whole-body hierarchy described in the R2 contract.
Raw region labels and propagated labels are exposed separately, with a conflict
flag for annotation-layer disagreements.
R2 paths use a portable `bs80k/...` root, and grounded targets are structured
JSON objects rather than model-specific tag strings.
See [the R2 contract](context/r2-controlled-language.md) and the resulting
[cross-dataset comparison](results/analyses/dataset-audits/comparison/dataset_comparison.md).
The frozen release manifest and SHA-256 checksums are in
`datasets/releases/r2/R2.0_RELEASE.md` and `datasets/releases/r2/SHA256SUMS.txt`.

## Prepare paired-image baselines

R1 baseline exports compose ANT on the left and POST on the right at training time. Their coordinate targets are transformed to that composite image; neutral two-box grounding rows remain in the dataset but are excluded from the single-box PaliGemma baseline with explicit rejection logs.

```powershell
uv run --project repo\wbbs-vqa_vgrounding python -m training.prepare `
  --release datasets\releases\r1 `
  --output datasets\model_exports\r1

uv run --project repo\wbbs-vqa_vgrounding --extra training vlm-train `
  --config repo\wbbs-vqa_vgrounding\configs\paligemma_r1_anatomy_vqa.example.json `
  --dry-run
```

The provided PaliGemma checkpoint is gated by its publisher. Before an actual
training run, accept the model terms on Hugging Face and authenticate the local
account. The dry-run needs no model access; a one-step run is the recommended
environment check.

After training, write deterministic test predictions and score them with:

```powershell
uv run --project repo\wbbs-vqa_vgrounding --extra training wbbs-predict `
  --config repo\wbbs-vqa_vgrounding\configs\paligemma_r1_anatomy_vqa.example.json `
  --adapter models\r1-anatomy-vqa-paligemma `
  --input datasets\model_exports\r1\anatomy_vqa_test.jsonl `
  --output results\inferences\r1-anatomy-vqa-paligemma\test_predictions.jsonl

uv run --project repo\wbbs-vqa_vgrounding python -m metric.evaluate `
  --expected datasets\model_exports\r1\anatomy_vqa_test.jsonl `
  --predictions results\inferences\r1-anatomy-vqa-paligemma\test_predictions.jsonl `
  --task vqa
```

## Qwen3-VL paired grounded-VQA baseline

Qwen3-VL receives ANT and POST as two native image inputs. Its primary export
joins anatomy and lesion grounded-VQA rows without removing neutral two-view
targets. The target is a concise clinical answer followed by the frozen
`<REG>`, `<CLS>`, `<ANT>/<PST>`, and `<BBX>` structure.

```powershell
uv sync --extra training --extra dev
uv run --project repo\wbbs-vqa_vgrounding wbbs-export-qwen3 --release datasets\releases\r1 --output datasets\model_exports\qwen3\r1
uv run --project repo\wbbs-vqa_vgrounding --extra training qwen3vl-train --config repo\wbbs-vqa_vgrounding\configs\qwen3vl_r1_grounded_vqa_4050.preflight.json --limit-examples 1
```

The preflight configuration uses one pair, one update, and a conservative
per-image pixel budget. It must pass before using the 1,000-step RTX 4050
experiment configuration. Generate and evaluate test predictions only after
selecting the final adapter:

```powershell
uv run --project repo\wbbs-vqa_vgrounding --extra training qwen3vl-predict `
  --config repo\wbbs-vqa_vgrounding\configs\qwen3vl_r1_grounded_vqa_4050.json `
  --adapter models\r1-grounded-vqa-qwen3vl-2b `
  --input datasets\model_exports\qwen3\r1\grounded_vqa_test.jsonl `
  --output results\inferences\r1-grounded-vqa-qwen3vl-2b\test_predictions.jsonl

uv run --project repo\wbbs-vqa_vgrounding wbbs-evaluate-qwen3 `
  --expected datasets\model_exports\qwen3\r1\grounded_vqa_test.jsonl `
  --predictions results\inferences\r1-grounded-vqa-qwen3vl-2b\test_predictions.jsonl
```

## Publication readiness

Publication readiness is tracked in
[publish-readiness-plan.md](context/publish-readiness-plan.md). R0 is a frozen,
reproducible silver release; clinician-reviewed data will be versioned separately
as R0.1 rather than modifying R0.
