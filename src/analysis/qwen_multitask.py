from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from preprocess.canonical import read_jsonl


_TASK_FIELDS = {
    "vqa": {"task", "answer", "class"},
    "grounding": {"task", "region", "boxes"},
    "grounded_vqa": {"task", "answer", "region", "class", "boxes"},
}


def audit_multitask_manifest(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return _audit_rows(rows, predictions=None)


def audit_multitask_predictions(expected: Iterable[dict[str, Any]], predictions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    expected_rows = list(expected)
    prediction_map = {str(row.get("task_id")): row for row in predictions}
    report = _audit_rows(expected_rows, predictions=prediction_map)
    expected_ids = {str(row["task_id"]) for row in expected_rows}
    report["unexpected_prediction_rows"] = len(set(prediction_map) - expected_ids)
    return report


def _audit_rows(rows: Iterable[dict[str, Any]], predictions: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    issues: Counter[str] = Counter()
    examples = []
    checked = 0
    valid = 0
    missing_predictions = 0
    for row in rows:
        checked += 1
        expected_task = row.get("task")
        row_issues = _validate_payload(row.get("target"), expected_task, row)
        if predictions is not None:
            prediction = predictions.get(str(row["task_id"]))
            if prediction is None:
                row_issues.append("missing_prediction")
                missing_predictions += 1
            else:
                row_issues.extend(f"prediction_{issue}" for issue in _validate_payload(prediction.get("prediction"), expected_task, None))
        if row_issues:
            issues.update(row_issues)
            if len(examples) < 20:
                examples.append({"task_id": row["task_id"], "issues": sorted(set(row_issues))})
        else:
            valid += 1
    return {
        "rows": checked,
        "valid_rows": valid,
        "invalid_rows": checked - valid,
        "missing_prediction_rows": missing_predictions,
        "issue_counts": dict(sorted(issues.items())),
        "examples": examples,
    }


def _validate_payload(value: Any, expected_task: Any, row: dict[str, Any] | None) -> list[str]:
    if expected_task not in _TASK_FIELDS:
        return ["unsupported_task"]
    if not isinstance(value, str):
        return ["not_text"]
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return ["invalid_json"]
    if not isinstance(payload, dict):
        return ["json_not_object"]
    issues = []
    if payload.get("task") != expected_task:
        issues.append("task_mismatch")
    if set(payload) != _TASK_FIELDS[expected_task]:
        issues.append("field_set_mismatch")
    if expected_task in {"vqa", "grounded_vqa"} and not isinstance(payload.get("answer"), str):
        issues.append("invalid_answer")
    valid_classes = {"yes", "no"} if expected_task == "vqa" else {"normal", "abnormal"}
    if expected_task in {"vqa", "grounded_vqa"} and payload.get("class") not in valid_classes:
        issues.append("invalid_class")
    if expected_task in {"grounding", "grounded_vqa"}:
        if not isinstance(payload.get("region"), str) or not payload["region"].strip():
            issues.append("invalid_region")
        issues.extend(_box_issues(payload.get("boxes")))
    if row is not None:
        if expected_task in {"vqa", "grounded_vqa"} and payload.get("class") != row.get("answer_label"):
            issues.append("class_target_mismatch")
        if expected_task in {"grounding", "grounded_vqa"}:
            if payload.get("region") != str(row.get("label", "")).replace(" ", "_"):
                issues.append("region_target_mismatch")
            expected_boxes = {_view(target["view"]): [int(value) for value in target["bbox"]] for target in row.get("targets", [])}
            actual_boxes = {item["view"]: item["bbox"] for item in payload.get("boxes", []) if isinstance(item, dict) and item.get("view") in {"ANT", "PST"}}
            if actual_boxes != expected_boxes:
                issues.append("box_target_mismatch")
    return issues


def _box_issues(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["invalid_boxes"]
    views = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"view", "bbox"}:
            return ["invalid_box_object"]
        if item["view"] not in {"ANT", "PST"}:
            return ["invalid_box_view"]
        bbox = item["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4 or any(not isinstance(number, int) or isinstance(number, bool) for number in bbox):
            return ["invalid_integer_bbox"]
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            return ["non_increasing_bbox"]
        views.append(item["view"])
    return ["duplicate_box_view"] if len(views) != len(set(views)) else []


def _view(value: str) -> str:
    return "PST" if value == "POST" else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit strict multitask Qwen JSON targets or predictions.")
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = read_jsonl(args.expected)
    report = audit_multitask_predictions(expected, read_jsonl(args.predictions)) if args.predictions else audit_multitask_manifest(expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": report["rows"], "invalid_rows": report["invalid_rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
