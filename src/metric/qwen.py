from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

from postprocess.qwen import parse_qwen_json_prediction, parse_qwen_multitask_json_prediction, parse_qwen_prediction
from preprocess.canonical import read_jsonl

from .grounding import iou


_MULTITASK_FIELDS = {
    "vqa": {"task", "answer", "class"},
    "grounding": {"task", "region", "boxes"},
    "grounded_vqa": {"task", "answer", "region", "class", "boxes"},
}
_HEADLINE_METRIC_PATHS = (
    "vqa.balanced_accuracy",
    "vqa.macro_f1",
    "grounding.mean_iou",
    "grounding.iou_at_0_5",
    "grounding.strict_grounding_accuracy",
    "grounded_vqa.balanced_accuracy",
    "grounded_vqa.macro_f1",
    "grounded_vqa.mean_iou",
    "grounded_vqa.iou_at_0_5",
    "grounded_vqa.strict_grounded_accuracy",
)


def evaluate_qwen_grounded_predictions(
    expected: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
    output_schema: str = "tags",
) -> dict[str, Any]:
    if output_schema not in {"tags", "json"}:
        raise ValueError("output_schema must be tags or json")
    prediction_parser = parse_qwen_json_prediction if output_schema == "json" else parse_qwen_prediction
    expected_rows = {row["task_id"]: row for row in expected}
    predicted_rows = {row["task_id"]: row for row in predictions}
    if set(expected_rows) != set(predicted_rows):
        raise ValueError("Predictions must cover expected task_id values exactly")
    class_correct = 0
    strict_correct = 0
    ious: list[float] = []
    valid_boxes = 0
    target_boxes = 0
    labels: list[tuple[str, str | None]] = []
    for task_id, row in expected_rows.items():
        parsed = prediction_parser(predicted_rows[task_id]["prediction"])
        class_ok = _canonical_binary_prediction(parsed["answer_label"], "abnormal", "normal") == row["answer_label"]
        class_correct += class_ok
        labels.append((row["answer_label"], parsed["answer_label"]))
        predicted_boxes = {box["view"]: box["bbox"] for box in parsed["boxes"] if box["view"] is not None}
        row_ious = []
        for target in row["targets"]:
            target_boxes += 1
            predicted = predicted_boxes.get(_tag_view(target.get("view")))
            score = iou(predicted, target["bbox"]) if predicted is not None else 0.0
            ious.append(score)
            row_ious.append(score)
            valid_boxes += predicted is not None
        if class_ok and len(row_ious) == len(row["targets"]) and all(score >= 0.5 for score in row_ious):
            strict_correct += 1
    rows = len(expected_rows)
    return {
        "rows": rows,
        "classification_accuracy": round(class_correct / rows, 6) if rows else None,
        **_classification_metrics(labels),
        "valid_boxes": valid_boxes,
        "mean_iou": round(sum(ious) / len(ious), 6) if ious else None,
        "iou_at_0_5": round(sum(score >= 0.5 for score in ious) / target_boxes, 6) if target_boxes else None,
        "strict_grounded_accuracy": round(strict_correct / rows, 6) if rows else None,
    }


def _classification_metrics(labels: list[tuple[str, str | None]]) -> dict[str, Any]:
    return _binary_classification_metrics(labels, "abnormal", "normal")


def _binary_classification_metrics(labels: list[tuple[str, str | None]], positive: str, negative: str) -> dict[str, Any]:
    canonical = [(target, _canonical_binary_prediction(predicted, positive, negative)) for target, predicted in labels]
    tp = sum(target == positive and predicted == positive for target, predicted in canonical)
    tn = sum(target == negative and predicted == negative for target, predicted in canonical)
    fp = sum(target == negative and predicted == positive for target, predicted in canonical)
    positive_total = sum(target == positive for target, _ in canonical)
    negative_total = sum(target == negative for target, _ in canonical)
    fn = positive_total - tp
    sensitivity = _rate(tp, positive_total)
    specificity = _rate(tn, negative_total)
    positive_precision = _rate(tp, tp + fp)
    positive_f1 = _rate(2 * tp, 2 * tp + fp + fn)
    negative_fp = fn
    negative_fn = fp
    negative_precision = _rate(tn, tn + negative_fp)
    negative_recall = specificity
    negative_f1 = _rate(2 * tn, 2 * tn + negative_fp + negative_fn)
    macro_f1 = None if positive_f1 is None or negative_f1 is None else round((positive_f1 + negative_f1) / 2, 6)
    exact_accuracy = _rate(tp + tn, len(canonical))
    return {
        "invalid_classifications": sum(predicted not in {positive, negative} for _, predicted in canonical),
        "accuracy": exact_accuracy,
        "exact_canonical_answer_accuracy": exact_accuracy,
        "balanced_accuracy": round((sensitivity + specificity) / 2, 6) if sensitivity is not None and specificity is not None else None,
        "macro_f1": macro_f1,
        "per_class": {
            positive: {"precision": positive_precision, "recall": sensitivity, "f1": positive_f1, "support": positive_total},
            negative: {"precision": negative_precision, "recall": negative_recall, "f1": negative_f1, "support": negative_total},
        },
        "precision": positive_precision,
        "recall": sensitivity,
        "specificity": specificity,
        "f1": positive_f1,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _canonical_binary_prediction(value: str | None, positive: str, negative: str) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    aliases = {positive: positive, negative: negative}
    if {positive, negative} == {"abnormal", "normal"}:
        aliases.update({"malignant": "abnormal", "benign": "normal"})
    elif {positive, negative} == {"malignant", "benign"}:
        aliases.update({"abnormal": "malignant", "normal": "benign"})
    return aliases.get(normalized)


def evaluate_qwen_multitask_predictions(expected: Iterable[dict[str, Any]], predictions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    expected_rows = {row["task_id"]: row for row in expected}
    predicted_rows = {row["task_id"]: row for row in predictions}
    if set(expected_rows) != set(predicted_rows):
        raise ValueError("Predictions must cover expected task_id values exactly")
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any], str]]] = {task: [] for task in ("vqa", "grounding", "grounded_vqa")}
    for task_id, row in expected_rows.items():
        task = row.get("task")
        if task not in groups:
            raise ValueError(f"Unsupported multitask row task: {task}")
        prediction = predicted_rows[task_id]["prediction"]
        groups[task].append((row, parse_qwen_multitask_json_prediction(prediction), prediction))
    vqa = _multitask_vqa_metrics(groups["vqa"])
    grounding = _multitask_grounding_metrics(groups["grounding"])
    grounded = _multitask_grounded_metrics(groups["grounded_vqa"])
    score_terms = [vqa["balanced_accuracy"], grounding["iou_at_0_5"], grounded["strict_grounded_accuracy"]]
    if any(term is None for term in score_terms):
        selection_score = None
    else:
        selection_score = round(sum(score_terms) / len(score_terms), 6)
    per_region = {}
    for region in sorted({str(row["label"]) for row in expected_rows.values() if isinstance(row.get("label"), str) and row["label"]}):
        regional_groups: dict[str, list[tuple[dict[str, Any], dict[str, Any], str]]] = {task: [] for task in groups}
        for row in expected_rows.values():
            if row.get("label") != region:
                continue
            task = row["task"]
            prediction = predicted_rows[row["task_id"]]["prediction"]
            regional_groups[task].append((row, parse_qwen_multitask_json_prediction(prediction), prediction))
        per_region[region] = {
            "vqa": _multitask_vqa_metrics(regional_groups["vqa"]),
            "grounding": _multitask_grounding_metrics(regional_groups["grounding"]),
            "grounded_vqa": _multitask_grounded_metrics(regional_groups["grounded_vqa"]),
        }
    return {
        "rows": len(expected_rows),
        "vqa": vqa,
        "grounding": grounding,
        "grounded_vqa": grounded,
        "model_selection_metric": "mean(vqa.balanced_accuracy, grounding.iou_at_0_5, grounded_vqa.strict_grounded_accuracy)",
        "model_selection_score": selection_score,
        "per_region": per_region,
    }


def patient_cluster_bootstrap_qwen_multitask(
    expected: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
    resamples: int = 2000,
    seed: int = 20260722,
    include_replicates: bool = False,
) -> dict[str, Any]:
    """Patient-cluster bootstrap confidence intervals for locked-test reporting.

    With the same seed and the same patients, two models see identical resamples, so their per-replicate values can be paired.
    """

    if resamples < 1:
        raise ValueError("resamples must be positive")
    expected_rows = list(expected)
    prediction_map = {row["task_id"]: row for row in predictions}
    if {row["task_id"] for row in expected_rows} != set(prediction_map):
        raise ValueError("Predictions must cover expected task_id values exactly")
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in expected_rows:
        patient_id = row.get("patient_id")
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError("Patient-cluster bootstrap requires non-empty patient_id on every expected row")
        clusters.setdefault(patient_id, []).append(row)
    patients = sorted(clusters)
    if len(patients) < 2:
        raise ValueError("Patient-cluster bootstrap requires at least two patients")
    samples: dict[str, list[float]] = {path: [] for path in _HEADLINE_METRIC_PATHS}
    replicates: dict[str, list[float | None]] = {path: [] for path in _HEADLINE_METRIC_PATHS}
    generator = random.Random(seed)
    for replicate in range(resamples):
        sampled_expected: list[dict[str, Any]] = []
        sampled_predictions: list[dict[str, Any]] = []
        for draw, patient_id in enumerate(generator.choices(patients, k=len(patients))):
            for row in clusters[patient_id]:
                task_id = f"{row['task_id']}#bootstrap:{replicate}:{draw}"
                sampled_expected.append({**row, "task_id": task_id})
                sampled_predictions.append({**prediction_map[row["task_id"]], "task_id": task_id})
        report = evaluate_qwen_multitask_predictions(sampled_expected, sampled_predictions)
        for path in _HEADLINE_METRIC_PATHS:
            value = _metric_value(report, path)
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            if valid:
                samples[path].append(float(value))
            replicates[path].append(float(value) if valid else None)
    result = {
        "method": "patient_cluster_bootstrap_percentile",
        "confidence_level": 0.95,
        "resamples": resamples,
        "seed": seed,
        "patients": len(patients),
        "metrics": {
            path: _bootstrap_interval(values)
            for path, values in samples.items()
        },
    }
    if include_replicates:
        result["replicates"] = replicates
    return result


def _metric_value(report: dict[str, Any], path: str) -> Any:
    value: Any = report
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _bootstrap_interval(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"valid_resamples": 0, "lower_95": None, "upper_95": None}
    ordered = sorted(values)
    return {
        "valid_resamples": len(ordered),
        "lower_95": round(ordered[round((len(ordered) - 1) * 0.025)], 6),
        "upper_95": round(ordered[round((len(ordered) - 1) * 0.975)], 6),
    }


def _multitask_vqa_metrics(values: list[tuple[dict[str, Any], dict[str, Any], str]]) -> dict[str, Any]:
    labels = [(row["answer_label"], parsed["answer_label"] if parsed["task"] == "vqa" else None) for row, parsed, _ in values]
    valid = sum(_valid_multitask_output(row, prediction) for row, _, prediction in values)
    return {"rows": len(values), "valid_output_rows": valid, "valid_output_rate": _rate(valid, len(values)), **_binary_classification_metrics(labels, "yes", "no")}


def _multitask_grounding_metrics(values: list[tuple[dict[str, Any], dict[str, Any], str]]) -> dict[str, Any]:
    target_boxes = 0
    valid_boxes = 0
    ious: list[float] = []
    strict = 0
    region_correct = 0
    valid = 0
    for row, parsed, prediction in values:
        output_valid = _valid_multitask_output(row, prediction)
        valid += output_valid
        is_task = output_valid and parsed["task"] == "grounding"
        region_ok = is_task and parsed["region"] == str(row["label"]).replace(" ", "_")
        region_correct += region_ok
        predicted_boxes = {box["view"]: box["bbox"] for box in parsed["boxes"] if box["view"] is not None} if is_task else {}
        row_ious = []
        for target in row["targets"]:
            target_boxes += 1
            predicted = predicted_boxes.get(_tag_view(target.get("view")))
            score = iou(predicted, target["bbox"]) if predicted is not None else 0.0
            ious.append(score)
            row_ious.append(score)
            valid_boxes += predicted is not None
        if region_ok and len(row_ious) == len(row["targets"]) and all(score >= 0.5 for score in row_ious):
            strict += 1
    rows = len(values)
    return {
        "rows": rows,
        "valid_output_rows": valid,
        "valid_output_rate": _rate(valid, rows),
        "region_accuracy": _rate(region_correct, rows),
        "valid_boxes": valid_boxes,
        "mean_iou": round(sum(ious) / len(ious), 6) if ious else None,
        "iou_at_0_5": _rate(sum(score >= 0.5 for score in ious), target_boxes),
        "strict_grounding_accuracy": _rate(strict, rows),
    }


def _multitask_grounded_metrics(values: list[tuple[dict[str, Any], dict[str, Any], str]]) -> dict[str, Any]:
    class_correct = 0
    strict_correct = 0
    valid_boxes = 0
    target_boxes = 0
    ious: list[float] = []
    labels: list[tuple[str, str | None]] = []
    valid = 0
    for row, parsed, prediction in values:
        output_valid = _valid_multitask_output(row, prediction)
        valid += output_valid
        is_task = output_valid and parsed["task"] == "grounded_vqa"
        predicted_class = parsed["answer_label"] if is_task else None
        class_ok = _canonical_binary_prediction(predicted_class, "abnormal", "normal") == row["answer_label"]
        class_correct += class_ok
        labels.append((row["answer_label"], predicted_class))
        region_ok = is_task and parsed["region"] == str(row["label"]).replace(" ", "_")
        predicted_boxes = {box["view"]: box["bbox"] for box in parsed["boxes"] if box["view"] is not None} if is_task else {}
        row_ious = []
        for target in row["targets"]:
            target_boxes += 1
            predicted = predicted_boxes.get(_tag_view(target.get("view")))
            score = iou(predicted, target["bbox"]) if predicted is not None else 0.0
            ious.append(score)
            row_ious.append(score)
            valid_boxes += predicted is not None
        if class_ok and region_ok and len(row_ious) == len(row["targets"]) and all(score >= 0.5 for score in row_ious):
            strict_correct += 1
    rows = len(values)
    return {
        "rows": rows,
        "valid_output_rows": valid,
        "valid_output_rate": _rate(valid, rows),
        "classification_accuracy": _rate(class_correct, rows),
        **_classification_metrics(labels),
        "valid_boxes": valid_boxes,
        "mean_iou": round(sum(ious) / len(ious), 6) if ious else None,
        "iou_at_0_5": _rate(sum(score >= 0.5 for score in ious), target_boxes),
        "strict_grounded_accuracy": _rate(strict_correct, rows),
    }


def _valid_multitask_output(row: dict[str, Any], prediction: Any) -> bool:
    """Check the locked JSON response contract before assigning model credit."""

    if not isinstance(prediction, str):
        return False
    try:
        payload = json.loads(prediction)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    task = row.get("task")
    if task not in _MULTITASK_FIELDS or payload.get("task") != task or set(payload) != _MULTITASK_FIELDS[task]:
        return False
    if task in {"vqa", "grounded_vqa"}:
        valid_classes = {"yes", "no"} if task == "vqa" else {"normal", "abnormal"}
        if not isinstance(payload.get("answer"), str) or not payload["answer"].strip() or payload.get("class") not in valid_classes:
            return False
    if task in {"grounding", "grounded_vqa"}:
        if not isinstance(payload.get("region"), str) or not payload["region"].strip():
            return False
        boxes = payload.get("boxes")
        if not isinstance(boxes, list) or not boxes:
            return False
        sizes = {
            _tag_view(image.get("view")): image.get("image_size")
            for image in row.get("images", [])
            if isinstance(image, dict)
        }
        views: set[str] = set()
        for item in boxes:
            if not isinstance(item, dict) or set(item) != {"view", "bbox"} or item.get("view") not in {"ANT", "PST"}:
                return False
            view = item["view"]
            bbox = item.get("bbox")
            if view in views or not isinstance(bbox, list) or len(bbox) != 4 or any(not isinstance(value, int) or isinstance(value, bool) for value in bbox):
                return False
            x1, y1, x2, y2 = bbox
            if not all(math.isfinite(value) for value in bbox) or x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
                return False
            image_size = sizes.get(view)
            if isinstance(image_size, list) and len(image_size) == 2 and (x2 > image_size[0] or y2 > image_size[1]):
                return False
            views.add(view)
    return True


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _tag_view(view: str | None) -> str | None:
    if view == "POST":
        return "PST"
    return view


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL grounded-VQA or multitask predictions.")
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--limit-examples", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--output-schema", choices=("tags", "json"), default="tags")
    parser.add_argument("--task", choices=("grounded_vqa", "multitask"), default="grounded_vqa")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit_examples is not None and args.limit_examples < 1:
        raise ValueError("limit_examples must be positive")
    expected = read_jsonl(args.expected)
    predictions = read_jsonl(args.predictions)
    if args.sample_seed is not None:
        random.Random(args.sample_seed).shuffle(expected)
    if args.limit_examples is not None:
        expected = expected[: args.limit_examples]
        predictions = predictions[: args.limit_examples]
    report = evaluate_qwen_multitask_predictions(expected, predictions) if args.task == "multitask" else evaluate_qwen_grounded_predictions(expected, predictions, args.output_schema)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
