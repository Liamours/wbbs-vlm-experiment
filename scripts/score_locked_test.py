from __future__ import annotations

import argparse
import json
from pathlib import Path

from metric.full_report import evaluate_full
from metric.qwen import evaluate_qwen_multitask_predictions, patient_cluster_bootstrap_qwen_multitask
from preprocess.canonical import read_jsonl

# Stratifications not already broken out by evaluate_full: task and per-class
# are already top-level/per_class in its multitask output, and region is
# already per_region. view_scope (Anterior/Posterior/paired) and region_level
# (whole_body/bone_region/metastasis, i.e. "target level") are added here.
_VIEW_SCOPE_VALUES = ("ANT", "POST", "BOTH")


def score(expected_path: Path, predictions_path: Path) -> dict:
    expected = read_jsonl(expected_path)
    predictions = read_jsonl(predictions_path)
    report = evaluate_full(expected, predictions, task="multitask", output_schema="json")
    prediction_by_id = {row["task_id"]: row for row in predictions}

    def subset_metrics(field: str, value: str) -> dict:
        subset_expected = [row for row in expected if row.get(field) == value]
        if not subset_expected:
            return {"rows": 0, "note": f"no rows with {field}={value!r} in this manifest"}
        subset_predictions = [prediction_by_id[row["task_id"]] for row in subset_expected]
        return evaluate_qwen_multitask_predictions(subset_expected, subset_predictions)

    region_levels = sorted({row.get("region_level") for row in expected if row.get("region_level")})
    report["stratifications"] = {
        "view_scope": {value: subset_metrics("view_scope", value) for value in _VIEW_SCOPE_VALUES},
        "region_level": {value: subset_metrics("region_level", value) for value in region_levels},
        "shoulder": subset_metrics("label", "shoulder"),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a locked-test prediction file: full multi-metric report, view_scope/region_level stratifications, and patient-cluster bootstrap CIs.")
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--bootstrap-output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument("--bootstrap-replicates", action="store_true", help="Keep every resample's metric values, needed to pair two models.")
    args = parser.parse_args()

    report = score(args.expected, args.predictions)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    expected = read_jsonl(args.expected)
    predictions = read_jsonl(args.predictions)
    bootstrap = patient_cluster_bootstrap_qwen_multitask(expected, predictions, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed, include_replicates=args.bootstrap_replicates)
    args.bootstrap_output.parent.mkdir(parents=True, exist_ok=True)
    args.bootstrap_output.write_text(json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics_output": str(args.metrics_output), "bootstrap_output": str(args.bootstrap_output), "rows": report["rows"], "bootstrap_patients": bootstrap["patients"]}, sort_keys=True))


if __name__ == "__main__":
    main()
