from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .grounding import iou


def classification_metrics(expected: Iterable[str], predicted: Iterable[str | None], positive: str, negative: str) -> dict[str, Any]:
    pairs = list(zip(expected, predicted, strict=True))
    if not all(value in {positive, negative} for value, _ in pairs):
        raise ValueError("Expected labels must be binary")
    tp = sum(target == positive and value == positive for target, value in pairs)
    tn = sum(target == negative and value == negative for target, value in pairs)
    fp = sum(target == negative and value == positive for target, value in pairs)
    fn = sum(target == positive and value == negative for target, value in pairs)
    sensitivity, specificity = _rate(tp, tp + fn), _rate(tn, tn + fp)
    positive_f1 = _rate(2 * tp, 2 * tp + fp + fn)
    negative_f1 = _rate(2 * tn, 2 * tn + fn + fp)
    return {
        "rows": len(pairs),
        "invalid_predictions": sum(value not in {positive, negative} for _, value in pairs),
        "accuracy": _rate(tp + tn, len(pairs)),
        "balanced_accuracy": round((sensitivity + specificity) / 2, 6) if sensitivity is not None and specificity is not None else None,
        "precision": _rate(tp, tp + fp),
        "recall": sensitivity,
        "specificity": specificity,
        "f1": positive_f1,
        "macro_f1": round((positive_f1 + negative_f1) / 2, 6) if positive_f1 is not None and negative_f1 is not None else None,
        "per_class": {
            positive: {"precision": _rate(tp, tp + fp), "recall": sensitivity, "f1": positive_f1, "support": tp + fn},
            negative: {"precision": _rate(tn, tn + fn), "recall": specificity, "f1": negative_f1, "support": tn + fp},
        },
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def multi_view_grounding_metrics(expected: Iterable[dict[str, Any]], predicted: Iterable[dict[str, Any]], target_field: str) -> dict[str, Any]:
    predictions = {row["task_id"]: row for row in predicted}
    scores, by_region = [], defaultdict(list)
    for row in expected:
        predicted_row = predictions.get(row["task_id"], {})
        predicted_targets = {target["view"]: target["bbox"] for target in predicted_row.get("prediction_targets", []) if isinstance(target, dict)}
        for target in row[target_field]:
            score = iou(target["bbox"], predicted_targets[target["view"]]) if target["view"] in predicted_targets else 0.0
            scores.append(score)
            by_region[row["label"]].append(score)
    macro_region_iou = sum(sum(values) / len(values) for values in by_region.values()) / len(by_region) if by_region else None
    return {
        "targets": len(scores),
        "mean_iou": round(sum(scores) / len(scores), 6) if scores else None,
        "iou_at_0_5": _rate(sum(score >= 0.5 for score in scores), len(scores)),
        "macro_region_iou": round(macro_region_iou, 6) if macro_region_iou is not None else None,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None
