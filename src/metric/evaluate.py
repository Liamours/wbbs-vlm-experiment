from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from preprocess.canonical import read_jsonl

from .grounding import iou


LOCATION = re.compile(r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>")


def evaluate_predictions(expected: Iterable[dict[str, Any]], predictions: Iterable[dict[str, Any]], task: str) -> dict[str, Any]:
    expected_rows = {row["task_id"]: row for row in expected}
    predicted_rows = {row["task_id"]: row for row in predictions}
    if set(expected_rows) != set(predicted_rows):
        raise ValueError("Predictions must cover expected task_id values exactly")
    values = [(expected_rows[task_id], predicted_rows[task_id]["prediction"]) for task_id in sorted(expected_rows)]
    if task == "vqa":
        return _classification_report(values, "yes", "no")
    if task == "grounding":
        return _grounding_report(values)
    if task == "grounded_vqa":
        return {"classification": _classification_report(values, "abnormal", "normal"), "grounding": _grounding_report(values)}
    raise ValueError("task must be vqa, grounding, or grounded_vqa")


def _classification_report(values: list[tuple[dict[str, Any], str]], positive: str, negative: str) -> dict[str, Any]:
    labels = []
    for expected, prediction in values:
        target = expected.get("answer_label")
        if target not in {positive, negative}:
            raise ValueError("Expected rows require a supported answer_label")
        labels.append((target, _parse_class(prediction, positive, negative)))
    tp = sum(target == positive and predicted == positive for target, predicted in labels)
    tn = sum(target == negative and predicted == negative for target, predicted in labels)
    fp = sum(target == negative and predicted == positive for target, predicted in labels)
    fn = sum(target == positive and predicted == negative for target, predicted in labels)
    invalid = sum(predicted is None for _, predicted in labels)
    sensitivity = _rate(tp, tp + fn)
    specificity = _rate(tn, tn + fp)
    return {
        "rows": len(labels),
        "invalid_predictions": invalid,
        "accuracy": _rate(tp + tn, len(labels)),
        "balanced_accuracy": round((sensitivity + specificity) / 2, 6) if sensitivity is not None and specificity is not None else None,
        "precision": _rate(tp, tp + fp),
        "recall": sensitivity,
        "specificity": specificity,
        "f1": _f1(tp, fp, fn),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _grounding_report(values: list[tuple[dict[str, Any], str]]) -> dict[str, Any]:
    scores = []
    for expected, prediction in values:
        box = _parse_box(prediction)
        if box is not None:
            scores.append(iou(box, expected["box_1024"]))
    return {
        "rows": len(values),
        "valid_boxes": len(scores),
        "mean_iou": round(sum(scores) / len(scores), 6) if scores else None,
        "iou_at_0_5": _rate(sum(score >= 0.5 for score in scores), len(values)),
    }


def _parse_class(prediction: str, positive: str, negative: str) -> str | None:
    value = prediction.lower()
    for label in (positive, negative):
        if re.search(rf"(?:class\s*=\s*|^){re.escape(label)}\b", value):
            return label
    return None


def _parse_box(prediction: str) -> list[int] | None:
    match = LOCATION.search(prediction)
    if not match:
        return None
    y1, x1, y2, x2 = (int(item) for item in match.groups())
    return [x1, y1, x2, y2] if x1 < x2 and y1 < y2 else None


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _f1(tp: int, fp: int, fn: int) -> float | None:
    return _rate(2 * tp, 2 * tp + fp + fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PaliGemma-format WBBS predictions.")
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--task", choices=("vqa", "grounding", "grounded_vqa"), required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_predictions(read_jsonl(args.expected), read_jsonl(args.predictions), args.task), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
