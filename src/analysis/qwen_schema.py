from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable

from preprocess.canonical import read_jsonl


HEAD = re.compile(r"^(?P<answer>.*?)<REG>(?P<region>[^<>]+)</REG>\s*<CLS>(?P<label>normal|abnormal)</CLS>\s*(?P<tail>.*)$", re.DOTALL)
VIEW_BOX = re.compile(r"<(ANT|PST)><BBX>([^<>]*)</BBX></(ANT|PST)>")
INTEGER = re.compile(r"\d+")
TAG_NAMES = ("<REG>", "</REG>", "<CLS>", "</CLS>", "<ANT>", "</ANT>", "<PST>", "</PST>", "<BBX>", "</BBX>")
JSON_FIELDS = frozenset(("answer", "region", "class", "boxes"))


def parse_strict_structure(text: str, require_answer: bool = True) -> dict[str, Any]:
    match = HEAD.fullmatch(text)
    if match is None:
        return {"valid": False, "format_valid": False, "issues": ["invalid_tag_order_or_closure"], "boxes": []}
    issues: list[str] = []
    answer = match.group("answer").strip()
    if require_answer and not answer:
        issues.append("missing_answer")
    tail = match.group("tail")
    cursor = 0
    boxes = []
    while cursor < len(tail):
        while cursor < len(tail) and tail[cursor].isspace():
            cursor += 1
        if cursor == len(tail):
            break
        box_match = VIEW_BOX.match(tail, cursor)
        if box_match is None:
            issues.append("invalid_box_block")
            break
        view, raw, closing_view = box_match.groups()
        if view != closing_view:
            issues.append("mismatched_view_closure")
        box, box_issues = _parse_integer_box(raw)
        issues.extend(box_issues)
        if box is not None:
            boxes.append({"view": view, "bbox": box})
        cursor = box_match.end()
    if not boxes:
        issues.append("missing_valid_box")
    views = [box["view"] for box in boxes]
    if len(views) != len(set(views)):
        issues.append("duplicate_view_box")
    return _structure_result(issues, answer, match.group("region"), match.group("label"), boxes, require_answer)


def parse_strict_json_structure(text: str, require_answer: bool = True) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"valid": False, "format_valid": False, "issues": ["invalid_json"], "boxes": []}
    if not isinstance(payload, dict):
        return {"valid": False, "format_valid": False, "issues": ["json_not_object"], "boxes": []}
    issues: list[str] = []
    missing = JSON_FIELDS - set(payload)
    unexpected = set(payload) - JSON_FIELDS
    if missing:
        issues.append("missing_json_fields")
    if unexpected:
        issues.append("unexpected_json_fields")
    answer = payload.get("answer")
    if not isinstance(answer, str):
        issues.append("invalid_answer")
        answer = ""
    else:
        answer = answer.strip()
    if require_answer and not answer:
        issues.append("missing_answer")
    region = payload.get("region")
    if not isinstance(region, str) or not region.strip():
        issues.append("invalid_region")
        region = None
    label = payload.get("class")
    if label not in {"normal", "abnormal"}:
        issues.append("invalid_class")
        label = None
    boxes = _json_boxes(payload.get("boxes"), issues)
    if not boxes:
        issues.append("missing_valid_box")
    views = [box["view"] for box in boxes]
    if len(views) != len(set(views)):
        issues.append("duplicate_view_box")
    return _structure_result(issues, answer, region, label, boxes, require_answer)


def validate_grounded_row(row: dict[str, Any], schema: str = "tags") -> dict[str, Any]:
    parsed = _structure_parser(schema)(row["target"])
    issues = list(parsed["issues"])
    if parsed.get("region") != str(row["label"]).replace(" ", "_"):
        issues.append("region_mismatch")
    if parsed.get("answer_label") != row.get("answer_label"):
        issues.append("class_mismatch")
    expected = {_tag_view(item["view"]): [int(value) for value in item["bbox"]] for item in row["targets"]}
    actual = {item["view"]: item["bbox"] for item in parsed["boxes"]}
    if set(actual) != set(expected):
        issues.append("view_set_mismatch")
    for view, box in actual.items():
        if view in expected and box != expected[view]:
            issues.append("bbox_mismatch")
        size = _view_size(row, view)
        if size is not None and not _in_bounds(box, size):
            issues.append("bbox_out_of_bounds")
    parsed["issues"] = sorted(set(issues))
    parsed["valid"] = not parsed["issues"]
    return parsed


def audit_grounded_manifest(rows: Iterable[dict[str, Any]], schema: str = "tags", example_limit: int = 20) -> dict[str, Any]:
    checked = 0
    invalid_rows = 0
    issue_counts: Counter[str] = Counter()
    examples = []
    view_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for row in rows:
        if row.get("task") != "grounded_vqa":
            continue
        checked += 1
        label_counts[str(row.get("answer_label"))] += 1
        result = validate_grounded_row(row, schema)
        for box in result["boxes"]:
            view_counts[box["view"]] += 1
        issue_counts.update(result["issues"])
        invalid_rows += bool(result["issues"])
        if result["issues"] and len(examples) < example_limit:
            examples.append({"task_id": row["task_id"], "issues": result["issues"], "target": row["target"]})
    return {
        "rows": checked,
        "valid_rows": checked - invalid_rows,
        "invalid_rows": invalid_rows,
        "issue_counts": dict(sorted(issue_counts.items())),
        "class_counts": dict(sorted(label_counts.items())),
        "view_box_counts": dict(sorted(view_counts.items())),
        "examples": examples,
    }


def audit_predictions(
    expected_rows: Iterable[dict[str, Any]],
    prediction_rows: Iterable[dict[str, Any]],
    schema: str = "tags",
    example_limit: int = 20,
) -> dict[str, Any]:
    expected = {row["task_id"]: row for row in expected_rows if row.get("task") == "grounded_vqa"}
    predictions = list(prediction_rows)
    issue_counts: Counter[str] = Counter()
    examples = []
    strict_rows = 0
    format_valid_rows = 0
    correct_class = 0
    correct_boxes = 0
    total_boxes = 0
    missing_expected = 0
    parser = _structure_parser(schema)
    for prediction in predictions:
        task_id = prediction.get("task_id")
        row = expected.get(task_id)
        if row is None:
            missing_expected += 1
            continue
        result = parser(str(prediction.get("prediction", "")))
        if result["valid"]:
            strict_rows += 1
        if result["format_valid"]:
            format_valid_rows += 1
        issue_counts.update(result["issues"])
        expected_boxes = {_tag_view(item["view"]): [int(value) for value in item["bbox"]] for item in row["targets"]}
        actual_boxes = {item["view"]: item["bbox"] for item in result["boxes"]}
        correct_class += result.get("answer_label") == row.get("answer_label")
        for view, target in expected_boxes.items():
            total_boxes += 1
            correct_boxes += actual_boxes.get(view) == target
        if result["issues"] and len(examples) < example_limit:
            examples.append({"task_id": task_id, "issues": result["issues"], "prediction": prediction.get("prediction", "")})
    rows = len(predictions) - missing_expected
    return {
        "rows": rows,
        "missing_expected_rows": missing_expected,
        "strict_valid_rows": strict_rows,
        "strict_valid_rate": _rate(strict_rows, rows),
        "format_valid_rows": format_valid_rows,
        "format_valid_rate": _rate(format_valid_rows, rows),
        "exact_class_rate": _rate(correct_class, rows),
        "exact_box_rate": _rate(correct_boxes, total_boxes),
        "issue_counts": dict(sorted(issue_counts.items())),
        "examples": examples,
    }


def audit_tokenizer(rows: Iterable[dict[str, Any]], model_id: str, schema: str = "tags") -> dict[str, Any]:
    _check_schema(schema)
    from transformers import AutoProcessor

    tokenizer = AutoProcessor.from_pretrained(model_id).tokenizer
    targets = [row["target"] for row in rows if row.get("task") == "grounded_vqa"]
    target_lengths = _token_lengths(tokenizer, targets)
    if schema == "tags":
        tag_tokens = {}
        for tag in TAG_NAMES:
            ids = tokenizer(tag, add_special_tokens=False)["input_ids"]
            tag_tokens[tag] = {
                "token_count": len(ids),
                "tokens": tokenizer.convert_ids_to_tokens(ids),
                "is_native_special_token": tag in tokenizer.all_special_tokens,
            }
        box_values = [match.group(2) for target in targets for match in VIEW_BOX.finditer(target)]
        prefixes = [target.split("<REG>", 1)[0] for target in targets]
        return {
            "model_id": model_id,
            "schema": schema,
            "tags": tag_tokens,
            "target_token_lengths": _summary(target_lengths),
            "coordinate_token_lengths": _summary(_token_lengths(tokenizer, box_values)),
            "prefix_tag_collisions": {tag: sum(tag in prefix for prefix in prefixes) for tag in TAG_NAMES},
        }
    parsed = [parse_strict_json_structure(target) for target in targets]
    box_values = [",".join(str(value) for value in box["bbox"]) for result in parsed for box in result["boxes"]]
    return {
        "model_id": model_id,
        "schema": schema,
        "json_fields": sorted(JSON_FIELDS),
        "target_token_lengths": _summary(target_lengths),
        "coordinate_token_lengths": _summary(_token_lengths(tokenizer, box_values)),
    }


def run_audit(
    manifests: list[Path],
    output: Path,
    model_id: str,
    predictions: Path | None = None,
    schema: str = "tags",
) -> dict[str, Any]:
    _check_schema(schema)
    loaded = {str(path): read_jsonl(path) for path in manifests}
    all_rows = [row for rows in loaded.values() for row in rows]
    report = {
        "schema": schema,
        "manifests": {path: audit_grounded_manifest(rows, schema) for path, rows in loaded.items()},
        "tokenizer": audit_tokenizer(all_rows, model_id, schema),
    }
    if predictions is not None:
        report["predictions"] = audit_predictions(all_rows, read_jsonl(predictions), schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_render_report(report), encoding="utf-8")
    return report


def _structure_result(
    issues: list[str], answer: str, region: str | None, label: str | None, boxes: list[dict[str, Any]], require_answer: bool
) -> dict[str, Any]:
    unique = sorted(set(issues))
    format_issues = [issue for issue in unique if not (require_answer and issue == "missing_answer")]
    return {
        "valid": not unique,
        "format_valid": not format_issues,
        "issues": unique,
        "answer": answer,
        "region": region,
        "answer_label": label,
        "boxes": boxes,
    }


def _json_boxes(value: Any, issues: list[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        issues.append("invalid_boxes")
        return []
    boxes = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"view", "bbox"}:
            issues.append("invalid_box_object")
            continue
        view = item["view"]
        if view not in {"ANT", "PST"}:
            issues.append("invalid_view")
            continue
        bbox = item["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4 or any(not isinstance(number, int) or isinstance(number, bool) for number in bbox):
            issues.append("invalid_integer_bbox")
            continue
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            issues.append("non_increasing_bbox")
            continue
        boxes.append({"view": view, "bbox": bbox})
    return boxes


def _structure_parser(schema: str):
    _check_schema(schema)
    return parse_strict_json_structure if schema == "json" else parse_strict_structure


def _check_schema(schema: str) -> None:
    if schema not in {"tags", "json"}:
        raise ValueError("schema must be tags or json")


def _parse_integer_box(raw: str) -> tuple[list[int] | None, list[str]]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4 or not all(INTEGER.fullmatch(part) for part in parts):
        return None, ["invalid_integer_bbox"]
    box = [int(part) for part in parts]
    if box[0] >= box[2] or box[1] >= box[3]:
        return None, ["non_increasing_bbox"]
    return box, []


def _view_size(row: dict[str, Any], view: str) -> list[int] | None:
    source_view = "POST" if view == "PST" else view
    for image in row.get("images", []):
        if image.get("view") == source_view:
            size = image.get("image_size")
            if isinstance(size, list) and len(size) == 2:
                return [int(size[0]), int(size[1])]
    return None


def _in_bounds(box: list[int], size: list[int]) -> bool:
    width, height = size
    return 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height


def _tag_view(view: str) -> str:
    return "PST" if view == "POST" else view


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {"count": len(values), "min": min(values), "median": median(values), "mean": round(fmean(values), 4), "max": max(values)}


def _token_lengths(tokenizer, values: list[str]) -> list[int]:
    lengths = []
    for start in range(0, len(values), 2048):
        batch = tokenizer(values[start : start + 2048], add_special_tokens=False)["input_ids"]
        lengths.extend(len(ids) for ids in batch)
    return lengths


def _render_report(report: dict[str, Any]) -> str:
    lines = ["# Qwen structured-output audit", "", f"- Schema: `{report['schema']}`.", "", "## Manifests", ""]
    for path, result in report["manifests"].items():
        lines.append(f"- `{path}`: {result['rows']:,} grounded rows; {result['invalid_rows']:,} invalid fields.")
    token = report["tokenizer"]
    lines.extend(["", "## Tokenizer", "", f"- Model: `{token['model_id']}`."])
    if report["schema"] == "tags":
        for tag, detail in token["tags"].items():
            lines.append(f"- `{tag}`: {detail['token_count']} ordinary tokens; native special token: {detail['is_native_special_token']}.")
    else:
        lines.append(f"- JSON fields: {', '.join(token['json_fields'])}.")
    if "predictions" in report:
        prediction = report["predictions"]
        lines.extend(["", "## Predictions", "", f"- Strict-valid rate: {prediction['strict_valid_rate']}.", f"- Format-valid rate: {prediction['format_valid_rate']}.", f"- Exact-class rate: {prediction['exact_class_rate']}.", f"- Exact-box rate: {prediction['exact_box_rate']}."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Qwen structured targets, tokens, and generated predictions.")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--schema", choices=("tags", "json"), default="tags")
    args = parser.parse_args()
    report = run_audit(args.manifest, args.output, args.model_id, args.predictions, args.schema)
    print(json.dumps({"output": str(args.output), "schema": args.schema, "manifests": {path: item["rows"] for path, item in report["manifests"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
