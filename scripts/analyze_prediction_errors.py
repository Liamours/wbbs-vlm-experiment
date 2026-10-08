"""Text-only failure analysis for R3 locked-test predictions.

Cross-references each model's results/*/test/predictions.jsonl against the
ground-truth test manifest (no images touched). Reports, per model: parse
failure rate, false-negative examples (predicted normal/no on an actually
abnormal/yes case, the clinically dangerous error), worst-IoU grounding
examples, and which failures the two models share versus each has alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from postprocess.qwen import parse_qwen_multitask_json_prediction  # noqa: E402

from project_paths import PROJECT_ROOT as ROOT
TEST_MANIFEST = ROOT / "datasets/model_exports/qwen3_json/r3/multitask_test.jsonl"
MODELS = {
    "qwen3vl": ROOT / "results/inferences/qwen3vl/test/predictions.jsonl",
    "paligemma": ROOT / "results/inferences/paligemma-latest/test/predictions.jsonl",
}
OUTPUT_DIR = ROOT / "results/analyses/error_analysis"
ABNORMAL_LABELS = {"yes", "abnormal"}


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def load_manifest() -> dict[str, dict]:
    rows = {}
    with TEST_MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["task_id"]] = row
    return rows


def load_predictions(path: Path) -> dict[str, str]:
    preds = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            preds[row["task_id"]] = row["prediction"]
    return preds


def analyze_model(name: str, pred_path: Path, manifest: dict[str, dict]) -> dict:
    predictions = load_predictions(pred_path)
    parse_failures = []
    false_negatives = []
    false_positives = []
    worst_iou = []
    region_errors: dict[str, dict[str, int]] = {}
    correct = 0
    scored = 0

    for task_id, prediction_text in predictions.items():
        row = manifest.get(task_id)
        if row is None:
            continue
        reference = parse_qwen_multitask_json_prediction(row["target"])
        predicted = parse_qwen_multitask_json_prediction(prediction_text)

        if predicted["task"] is None:
            parse_failures.append({"task_id": task_id, "raw_prediction": prediction_text[:300]})
            continue

        ref_label = reference["answer_label"]
        pred_label = predicted["answer_label"]
        if ref_label is not None and pred_label is not None:
            scored += 1
            region = region_errors.setdefault(row.get("label", "unknown"), {"correct": 0, "wrong": 0})
            if ref_label == pred_label:
                correct += 1
                region["correct"] += 1
            else:
                region["wrong"] += 1
                example = {
                    "task_id": task_id,
                    "region": row.get("label"),
                    "prompt": row.get("prompt"),
                    "reference_answer": reference["answer"],
                    "predicted_answer": predicted["answer"],
                    "reference_label": ref_label,
                    "predicted_label": pred_label,
                }
                if ref_label in ABNORMAL_LABELS and pred_label not in ABNORMAL_LABELS:
                    false_negatives.append(example)
                elif ref_label not in ABNORMAL_LABELS and pred_label in ABNORMAL_LABELS:
                    false_positives.append(example)

        predicted_boxes = {box["view"]: box["bbox"] for box in predicted["boxes"]}
        for target in row.get("targets", []):
            target_view = "PST" if target.get("view") == "POST" else target.get("view")
            predicted_box = predicted_boxes.get(target_view)
            score = iou(predicted_box, target["bbox"]) if predicted_box is not None else 0.0
            worst_iou.append({"task_id": task_id, "region": row.get("label"), "view": target_view, "iou": round(score, 4)})

    worst_iou.sort(key=lambda item: item["iou"])
    false_negatives.sort(key=lambda item: item["region"] or "")

    return {
        "model": name,
        "rows_predicted": len(predictions),
        "rows_scored_for_classification": scored,
        "classification_accuracy_on_scored_rows": round(correct / scored, 4) if scored else None,
        "parse_failure_count": len(parse_failures),
        "parse_failure_rate": round(len(parse_failures) / len(predictions), 4) if predictions else None,
        "false_negative_count": len(false_negatives),
        "false_positive_count": len(false_positives),
        "region_breakdown": {
            region: {**counts, "error_rate": round(counts["wrong"] / (counts["correct"] + counts["wrong"]), 4)}
            for region, counts in sorted(region_errors.items())
        },
        "parse_failure_examples": parse_failures[:10],
        "false_negative_examples": false_negatives[:15],
        "false_positive_examples": false_positives[:10],
        "worst_grounding_examples": worst_iou[:15],
    }


def cross_model_comparison(manifest: dict[str, dict], per_model_preds: dict[str, dict[str, str]]) -> dict:
    wrong_by_model: dict[str, set[str]] = {}
    for name, predictions in per_model_preds.items():
        wrong = set()
        for task_id, prediction_text in predictions.items():
            row = manifest.get(task_id)
            if row is None:
                continue
            reference = parse_qwen_multitask_json_prediction(row["target"])
            predicted = parse_qwen_multitask_json_prediction(prediction_text)
            if reference["answer_label"] is not None and predicted["answer_label"] != reference["answer_label"]:
                wrong.add(task_id)
        wrong_by_model[name] = wrong
    names = list(wrong_by_model)
    shared = set.intersection(*wrong_by_model.values()) if wrong_by_model else set()
    only = {name: sorted(wrong_by_model[name] - set.union(*(w for other, w in wrong_by_model.items() if other != name))) for name in names}
    return {
        "wrong_count_by_model": {name: len(wrong) for name, wrong in wrong_by_model.items()},
        "shared_wrong_count": len(shared),
        "shared_wrong_task_ids_sample": sorted(shared)[:20],
        "only_wrong_count_by_model": {name: len(ids) for name, ids in only.items()},
        "only_wrong_task_ids_sample": {name: ids[:10] for name, ids in only.items()},
    }


def main() -> None:
    manifest = load_manifest()
    per_model_preds = {name: load_predictions(path) for name, path in MODELS.items()}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, path in MODELS.items():
        report = analyze_model(name, path, manifest)
        reports[name] = report
        (OUTPUT_DIR / f"{name}_error_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{name}: scored={report['rows_scored_for_classification']} accuracy={report['classification_accuracy_on_scored_rows']} "
              f"parse_failures={report['parse_failure_count']} false_negatives={report['false_negative_count']} false_positives={report['false_positive_count']}")

    comparison = cross_model_comparison(manifest, per_model_preds)
    (OUTPUT_DIR / "cross_model_comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("cross-model:", json.dumps(comparison["wrong_count_by_model"]), "shared_wrong=", comparison["shared_wrong_count"])


if __name__ == "__main__":
    main()
