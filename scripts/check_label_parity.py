"""Check that the two Qwen JSON exports carry the same labels and that the controls and models were scored on the same ones.

The models were trained and scored on one export (reference), the question-only controls on a regeneration of it
(other) whose image paths differ. This script checks, from the files only, that:

1. both exports hold identical task ids, labels, targets, prompts and splits, and differ only in the image path prefix;
2. each control's test predictions cover exactly the test task ids of both exports, and so do the model predictions;
3. scoring the control and model predictions against either export gives identical metrics, and those metrics equal the
   archived ones;
4. the archived control predictions are what the control code produces from either export's training split.

It reads only JSON files and writes one report. Nothing here touches a model, an image or the GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from analysis.baseline_controls import build_control_predictions  # noqa: E402
from metric.qwen import evaluate_qwen_multitask_predictions  # noqa: E402
from project_paths import PROJECT_ROOT  # noqa: E402

IMAGE_MARKER = "bs80k-imaging-raw/"
SCORED_SECTIONS = ("vqa", "grounding", "grounded_vqa")
MAX_LISTED = 10


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_image_path(path: str) -> tuple[str, str]:
    """Return (prefix, remainder) around the shared scan folder name, using forward slashes."""
    normalized = path.replace("\\", "/")
    if IMAGE_MARKER not in normalized:
        return "", normalized
    prefix, remainder = normalized.split(IMAGE_MARKER, 1)
    return prefix + IMAGE_MARKER, remainder


def compare_rows(left: dict[str, Any], right: dict[str, Any], mismatches: Counter, prefixes: Counter) -> None:
    for key in sorted(set(left) | set(right)):
        if key == "images":
            compare_images(left.get(key), right.get(key), mismatches, prefixes)
        elif left.get(key) != right.get(key) or (key in left) != (key in right):
            mismatches[key] += 1


def compare_images(left: Any, right: Any, mismatches: Counter, prefixes: Counter) -> None:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        mismatches["images.count"] += 1
        return
    for left_image, right_image in zip(left, right):
        for key in sorted((set(left_image) | set(right_image)) - {"image"}):
            if left_image.get(key) != right_image.get(key):
                mismatches[f"images.{key}"] += 1
        left_prefix, left_rest = split_image_path(left_image.get("image", ""))
        right_prefix, right_rest = split_image_path(right_image.get("image", ""))
        prefixes[(left_prefix, right_prefix)] += 1
        if left_rest != right_rest:
            mismatches["images.image_after_prefix"] += 1


def compare_export_file(reference: Path, other: Path) -> dict[str, Any]:
    reference_rows = read_jsonl(reference)
    other_rows = read_jsonl(other)
    reference_ids = [row["task_id"] for row in reference_rows]
    other_ids = [row["task_id"] for row in other_rows]
    result: dict[str, Any] = {
        "rows_reference": len(reference_rows),
        "rows_other": len(other_rows),
        "duplicate_task_ids": len(reference_ids) - len(set(reference_ids)) + len(other_ids) - len(set(other_ids)),
        "same_task_id_set": set(reference_ids) == set(other_ids),
        "same_task_id_order": reference_ids == other_ids,
        "sha256_reference": sha256(reference),
        "sha256_other": sha256(other),
    }
    mismatches: Counter = Counter()
    prefixes: Counter = Counter()
    other_by_id = {row["task_id"]: row for row in other_rows}
    paired = 0
    for row in tqdm(reference_rows, desc=reference.name, unit="row", leave=False):
        partner = other_by_id.get(row["task_id"])
        if partner is None:
            continue
        paired += 1
        compare_rows(row, partner, mismatches, prefixes)
    result["rows_paired"] = paired
    result["field_mismatch_counts"] = dict(sorted(mismatches.items()))
    result["image_prefix_pairs"] = [
        {"reference": ref, "other": oth, "images": count} for (ref, oth), count in sorted(prefixes.items())
    ]
    result["split_values"] = sorted({str(row.get("split")) for row in reference_rows} | {str(row.get("split")) for row in other_rows})
    result["identical_apart_from_image_prefix"] = (
        result["same_task_id_set"]
        and result["same_task_id_order"]
        and result["duplicate_task_ids"] == 0
        and paired == len(reference_rows) == len(other_rows)
        and not mismatches
    )
    return result


def coverage(predictions: list[dict[str, Any]], expected_ids: dict[str, list[str]]) -> dict[str, Any]:
    ids = [row["task_id"] for row in predictions]
    report: dict[str, Any] = {"rows": len(ids), "duplicate_task_ids": len(ids) - len(set(ids))}
    for export, expected in expected_ids.items():
        report[f"covers_exactly_{export}"] = set(ids) == set(expected) and len(ids) == len(expected)
        report[f"same_order_as_{export}"] = ids == expected
        report[f"missing_vs_{export}"] = len(set(expected) - set(ids))
        report[f"unexpected_vs_{export}"] = len(set(ids) - set(expected))
    return report


def diff_paths(fresh: Any, archived: Any, prefix: str = "") -> tuple[list[str], list[str]]:
    """Compare archived against fresh over the keys of fresh.

    Returns (paths whose value differs, paths that fresh has and archived lacks).
    """
    if isinstance(fresh, dict):
        if not isinstance(archived, dict):
            return [prefix or "<root>"], []
        changed: list[str] = []
        absent: list[str] = []
        for key, value in fresh.items():
            path = f"{prefix}.{key}".lstrip(".")
            if key not in archived:
                absent.append(path)
            else:
                sub_changed, sub_absent = diff_paths(value, archived[key], path)
                changed.extend(sub_changed)
                absent.extend(sub_absent)
        return changed, absent
    return ([] if fresh == archived else [prefix or "<root>"]), []


def score_both(name: str, predictions: list[dict[str, Any]], exports: dict[str, list[dict[str, Any]]], archived_metrics_path: Path | None) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for export, rows in exports.items():
        try:
            reports[export] = evaluate_qwen_multitask_predictions(rows, predictions)
        except ValueError as error:
            errors[export] = str(error)
    result: dict[str, Any] = {"scoring_errors": errors}
    if len(reports) == len(exports):
        names = list(reports)
        result["metrics_identical_across_exports"] = all(reports[names[0]] == reports[other] for other in names[1:])
        reference = reports[names[0]]
        result["headline"] = {
            "vqa_balanced_accuracy": reference["vqa"]["balanced_accuracy"],
            "vqa_f1": reference["vqa"]["f1"],
            "vqa_confusion": reference["vqa"]["confusion"],
            "grounding_iou_at_0_5": reference["grounding"]["iou_at_0_5"],
            "grounded_vqa_balanced_accuracy": reference["grounded_vqa"]["balanced_accuracy"],
            "grounded_vqa_strict_grounded_accuracy": reference["grounded_vqa"]["strict_grounded_accuracy"],
        }
        if archived_metrics_path is not None and archived_metrics_path.exists():
            archived = json.loads(archived_metrics_path.read_text(encoding="utf-8"))
            changed: list[str] = []
            absent: list[str] = []
            for section in SCORED_SECTIONS:
                section_changed, section_absent = diff_paths(reference[section], archived.get(section, {}), section)
                changed.extend(section_changed)
                absent.extend(section_absent)
            result["matches_archived_metrics"] = not changed
            result["archived_metrics_path"] = project_relative(archived_metrics_path)
            result["archived_values_that_differ"] = changed[:MAX_LISTED]
            result["metrics_added_since_archived"] = absent
        else:
            result["matches_archived_metrics"] = None
    return result


def regenerated_controls_match(train_rows: Iterable[dict[str, Any]], test_rows: list[dict[str, Any]], control: str, archived: list[dict[str, Any]]) -> dict[str, Any]:
    fresh = build_control_predictions(list(train_rows), test_rows, control)  # type: ignore[arg-type]
    archived_by_id = {row["task_id"]: row["prediction"] for row in archived}
    differing = [row["task_id"] for row in fresh if archived_by_id.get(row["task_id"]) != row["prediction"]]
    return {"rows": len(fresh), "differing_rows": len(differing), "examples": differing[:MAX_LISTED]}


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    exports_root = PROJECT_ROOT / "datasets/training-exports/qwen3_json"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=exports_root / "r3-20260723", help="Export the models were trained and scored on.")
    parser.add_argument("--other", type=Path, default=exports_root / "r3", help="Export the controls were built and scored on.")
    parser.add_argument("--files", nargs="+", default=["multitask_test.jsonl", "multitask_val.jsonl", "multitask_train.jsonl"])
    parser.add_argument("--controls", type=Path, default=PROJECT_ROOT / "results/analyses/metrics/r3-multitask-no-image-controls")
    parser.add_argument("--model-predictions", type=Path, default=PROJECT_ROOT / "results/inferences/r3-reported/qwen3vl/test/predictions.jsonl")
    parser.add_argument("--model-metrics", type=Path, default=PROJECT_ROOT / "results/inferences/r3-reported/qwen3vl/test/metrics.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/analyses/label-parity/report.json")
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool | None) -> None:
        checks.append({"name": name, "passed": passed})

    files: dict[str, Any] = {}
    for name in tqdm(args.files, desc="export files", unit="file"):
        files[name] = compare_export_file(args.reference / name, args.other / name)
        record(f"{name}: identical apart from the image path prefix", files[name]["identical_apart_from_image_prefix"])

    reference_test = read_jsonl(args.reference / "multitask_test.jsonl")
    other_test = read_jsonl(args.other / "multitask_test.jsonl")
    expected_ids = {"reference": [row["task_id"] for row in reference_test], "other": [row["task_id"] for row in other_test]}
    exports = {"reference": reference_test, "other": other_test}

    prediction_sets: dict[str, tuple[list[dict[str, Any]], Path | None]] = {}
    for control in sorted(path.name for path in args.controls.iterdir() if (path / "test/predictions.jsonl").exists()):
        prediction_sets[f"control:{control}"] = (read_jsonl(args.controls / control / "test/predictions.jsonl"), args.controls / control / "test/metrics.json")
    prediction_sets["model:qwen3vl"] = (read_jsonl(args.model_predictions), args.model_metrics)

    cover: dict[str, Any] = {}
    scoring: dict[str, Any] = {}
    for label, (predictions, metrics_path) in tqdm(prediction_sets.items(), desc="prediction sets", unit="set"):
        cover[label] = coverage(predictions, expected_ids)
        record(f"{label}: covers exactly the test task ids of both exports", cover[label]["covers_exactly_reference"] and cover[label]["covers_exactly_other"])
        scoring[label] = score_both(label, predictions, exports, metrics_path)
        record(f"{label}: metrics identical when scored on either export", scoring[label].get("metrics_identical_across_exports"))
        record(f"{label}: metrics equal the archived metrics", scoring[label].get("matches_archived_metrics"))

    model_ids = {row["task_id"] for row in prediction_sets["model:qwen3vl"][0]}
    controls_match_model = all(
        {row["task_id"] for row in rows} == model_ids for label, (rows, _) in prediction_sets.items() if label.startswith("control:")
    )
    record("controls and model predictions hold the same test task ids", controls_match_model)

    regenerated: dict[str, Any] = {}
    for export_name, root in (("reference", args.reference), ("other", args.other)):
        train_rows = read_jsonl(root / "multitask_train.jsonl")
        for label, (predictions, _) in prediction_sets.items():
            if label.startswith("control:"):
                control = label.split(":", 1)[1]
                key = f"{control} from {export_name} train"
                regenerated[key] = regenerated_controls_match(train_rows, exports[export_name], control, predictions)
                record(f"{label}: regenerated from the {export_name} export, predictions identical", regenerated[key]["differing_rows"] == 0)

    provenance_path = args.controls / "control_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    test_provenance = provenance.get("inputs", {}).get("test", {})
    current = {"reference": sha256(args.reference / "multitask_test.jsonl"), "other": sha256(args.other / "multitask_test.jsonl")}
    scored_sha = test_provenance.get("sha256")
    notes = {
        "control_scored_file_recorded": test_provenance.get("path"),
        "control_scored_sha256_recorded": scored_sha,
        "test_sha256_now": current,
        "control_scored_file_still_in_project": scored_sha in current.values(),
    }

    failed = [check["name"] for check in checks if check["passed"] is False]
    report = {
        "reference_export": project_relative(args.reference),
        "other_export": project_relative(args.other),
        "all_checks_passed": not failed and all(check["passed"] is not None for check in checks),
        "failed_checks": failed,
        "checks": checks,
        "export_files": files,
        "prediction_coverage": cover,
        "scoring": scoring,
        "regenerated_controls": regenerated,
        "control_provenance": notes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    passed = sum(check["passed"] is True for check in checks)
    print(f"checks passed {passed}/{len(checks)}; failed: {failed or 'none'}; report: {args.output}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
