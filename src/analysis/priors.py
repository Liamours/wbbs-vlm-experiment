from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from metric.paired import classification_metrics, multi_view_grounding_metrics
from preprocess.canonical import read_jsonl, write_jsonl


JOBS = {
    "anatomy_vqa": ("vqa.jsonl", "vqa"),
    "anatomy_grounding": ("grounding.jsonl", "grounding"),
    "anatomy_grounded_vqa": ("grounded_vqa.jsonl", "grounded_vqa"),
    "lesion_vqa": ("lesion_vqa.jsonl", "vqa"),
    "lesion_grounding": ("lesion_grounding.jsonl", "grounding"),
    "lesion_grounded_vqa": ("lesion_grounded_vqa.jsonl", "grounded_vqa"),
}


def run_prior_baseline(train: Iterable[dict[str, Any]], evaluation: Iterable[dict[str, Any]], task: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_rows, eval_rows = list(train), list(evaluation)
    classifiers = _fit_classifiers(train_rows) if task in {"vqa", "grounded_vqa"} else None
    boxes = _fit_boxes(train_rows, _target_field(task)) if task in {"grounding", "grounded_vqa"} else None
    predictions = []
    for row in eval_rows:
        prediction = {"task_id": row["task_id"]}
        if classifiers is not None:
            prediction["prediction_label"] = _predict_class(row, classifiers)
        if boxes is not None:
            prediction["prediction_targets"] = _predict_boxes(row, boxes, _target_field(task))
        predictions.append(prediction)
    report: dict[str, Any] = {"model": "question_anatomy_prior/v1", "task": task, "train_rows": len(train_rows), "evaluation_rows": len(eval_rows)}
    if task == "vqa":
        report["classification"] = classification_metrics((row["answer_label"] for row in eval_rows), (row["prediction_label"] for row in predictions), "yes", "no")
    elif task == "grounding":
        report["grounding"] = multi_view_grounding_metrics(eval_rows, predictions, "targets")
    else:
        report["classification"] = classification_metrics((row["answer_label"] for row in eval_rows), (row["prediction_label"] for row in predictions), "abnormal", "normal")
        report["grounding"] = multi_view_grounding_metrics(eval_rows, predictions, "evidence_targets")
    return predictions, report


def run_release_priors(release: str | Path, output: str | Path) -> dict[str, Any]:
    source, target = Path(release), Path(output)
    target.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"release": str(source), "jobs": {}}
    for name, (filename, task) in JOBS.items():
        rows = read_jsonl(source / filename)
        train = [row for row in rows if row["split"] == "train"]
        results = {}
        for split in ("val", "test"):
            predictions, report = run_prior_baseline(train, [row for row in rows if row["split"] == split], task)
            write_jsonl(target / f"{name}_{split}_predictions.jsonl", predictions)
            (target / f"{name}_{split}_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            results[split] = report
        summary["jobs"][name] = results
    (target / "prior_baseline_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_r3_priors(release: str | Path, output: str | Path) -> dict[str, Any]:
    """Image-free prior baseline for the merged R2/R3 schema.

    R3 keeps one vqa.jsonl and one vgrounding.jsonl instead of R1's per-task
    files, so jobs are carved out by `region_level` (bone_region, metastasis,
    whole_body) and, for vgrounding, by `interaction_type` (localization vs
    localization_and_classification) instead of by filename.
    """
    source, target = Path(release), Path(output)
    target.mkdir(parents=True, exist_ok=True)
    vqa_rows = read_jsonl(source / "vqa.jsonl")
    vgrounding_rows = read_jsonl(source / "vgrounding.jsonl")
    jobs: dict[str, tuple[list[dict[str, Any]], str]] = {}
    for level in ("bone_region", "metastasis", "whole_body"):
        rows = [row for row in vqa_rows if row["region_level"] == level]
        if rows:
            jobs[f"vqa_{level}"] = (rows, "vqa")
    for level in ("bone_region", "metastasis"):
        localization = [row for row in vgrounding_rows if row["region_level"] == level and row["interaction_type"] == "localization"]
        grounded_vqa = [row for row in vgrounding_rows if row["region_level"] == level and row["interaction_type"] == "localization_and_classification"]
        if localization:
            jobs[f"grounding_{level}"] = (localization, "grounding")
        if grounded_vqa:
            jobs[f"grounded_vqa_{level}"] = (grounded_vqa, "grounded_vqa")
    summary: dict[str, Any] = {"release": str(source), "jobs": {}}
    pooled: dict[str, dict[str, list[Any]]] = {
        task: {split: [] for split in ("val", "test")} for task in ("vqa", "grounding", "grounded_vqa")
    }
    for name, (rows, task) in jobs.items():
        train = [row for row in rows if row["split"] == "train"]
        results = {}
        for split in ("val", "test"):
            eval_rows = [row for row in rows if row["split"] == split]
            predictions, report = run_prior_baseline(train, eval_rows, task)
            write_jsonl(target / f"{name}_{split}_predictions.jsonl", predictions)
            (target / f"{name}_{split}_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            results[split] = report
            pooled[task][split].append((eval_rows, predictions))
        summary["jobs"][name] = results
    # Pooled aggregates: bone_region + metastasis (+ whole_body for vqa) rows
    # combined, matching exactly the row composition of the shared VLM test sets
    # (datasets/model_exports/qwen3_json/r3/*.jsonl) so heuristic and image-based
    # baselines are compared on identical evaluation rows.
    for task in ("vqa", "grounding", "grounded_vqa"):
        results = {}
        for split in ("val", "test"):
            parts = pooled[task][split]
            if not parts:
                continue
            eval_rows = [row for rows, _ in parts for row in rows]
            predictions = [prediction for _, preds in parts for prediction in preds]
            if task == "vqa":
                report = {"model": "question_anatomy_prior/v1", "task": task, "evaluation_rows": len(eval_rows)}
                report["classification"] = classification_metrics((row["answer_label"] for row in eval_rows), (p["prediction_label"] for p in predictions), "yes", "no")
            elif task == "grounding":
                report = {"model": "question_anatomy_prior/v1", "task": task, "evaluation_rows": len(eval_rows)}
                report["grounding"] = multi_view_grounding_metrics(eval_rows, predictions, "targets")
            else:
                report = {"model": "question_anatomy_prior/v1", "task": task, "evaluation_rows": len(eval_rows)}
                report["classification"] = classification_metrics((row["answer_label"] for row in eval_rows), (p["prediction_label"] for p in predictions), "abnormal", "normal")
                report["grounding"] = multi_view_grounding_metrics(eval_rows, predictions, "evidence_targets")
            results[split] = report
        summary["jobs"][f"{task}_pooled"] = results
    (target / "prior_baseline_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _fit_classifiers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    total = Counter()
    for row in rows:
        label = row["answer_label"]
        by_key[_class_key(row)][label] += 1
        total[label] += 1
    return {"by_key": {key: _majority(counts) for key, counts in by_key.items()}, "default": _majority(total)}


def _predict_class(row: dict[str, Any], model: dict[str, Any]) -> str:
    return model["by_key"].get(_class_key(row), model["default"])


def _fit_boxes(rows: list[dict[str, Any]], target_field: str) -> dict[tuple[str, str], list[float]]:
    sums: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    for row in rows:
        sizes = {image["view"]: image["image_size"] for image in row["images"]}
        for target in row[target_field]:
            width, height = sizes[target["view"]]
            values = [target["bbox"][0] / width, target["bbox"][1] / height, target["bbox"][2] / width, target["bbox"][3] / height]
            accumulator = sums[(row["label"], target["view"])]
            for index, value in enumerate(values):
                accumulator[index] += value
            accumulator[4] += 1
    return {key: [value / values[4] for value in values[:4]] for key, values in sums.items()}


def _predict_boxes(row: dict[str, Any], boxes: dict[tuple[str, str], list[float]], target_field: str) -> list[dict[str, Any]]:
    sizes = {image["view"]: image["image_size"] for image in row["images"]}
    predicted = []
    for target in row[target_field]:
        width, height = sizes[target["view"]]
        normalized = boxes[(row["label"], target["view"])]
        bbox = [round(normalized[0] * width), round(normalized[1] * height), round(normalized[2] * width), round(normalized[3] * height)]
        predicted.append({"view": target["view"], "bbox": bbox})
    return predicted


def _class_key(row: dict[str, Any]) -> tuple[str, str]:
    return row["label"], row.get("view_scope", "BOTH")


def _target_field(task: str) -> str:
    return "targets" if task == "grounding" else "evidence_targets"


def _majority(counts: Counter[str]) -> str:
    return sorted(counts, key=lambda label: (-counts[label], label))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run image-free question/anatomy-prior baselines for a WBBS release.")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", choices=("r1", "r3"), default="r1", help="r1: separate per-task files. r3: merged vqa.jsonl/vgrounding.jsonl split by region_level.")
    args = parser.parse_args()
    runner = run_r3_priors if args.schema == "r3" else run_release_priors
    print(json.dumps(runner(args.release, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
