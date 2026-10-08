"""Training-dynamics comparison across every recorded run.

Pure file inspection of models/*/training_state.json, training_progress.json,
and selection/epoch-*_val_metrics.json. No model loading, no dataset access.
"""

from __future__ import annotations

import json
from pathlib import Path

from project_paths import PROJECT_ROOT as ROOT
OUTPUT_DIR = ROOT / "results/analyses/training_dynamics"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def r1_pilot_curve() -> dict:
    state = read_json(ROOT / "models/r1-grounded-vqa-qwen3vl-2b-json/training_state.json")
    return {
        "run": "r1-grounded-vqa-qwen3vl-2b-json",
        "description": "R1 grounded_vqa, 1000-step JSON-schema pilot, balanced_family_class sampling",
        "validation_history": state["validation_history"],
        "final_evaluation_loss": state["evaluation_loss"],
    }


def calibration_comparison() -> dict:
    runs = {
        "tags_unbalanced_25step": "models/calibration/qwen3vl-r1-grounded-vqa",
        "json_balanced_25step": "models/calibration/qwen3vl-r1-grounded-vqa-json-balanced",
        "tags_balanced_25step": "models/calibration/qwen3vl-r1-grounded-vqa-tags-balanced",
    }
    result = {}
    for label, rel in runs.items():
        state = read_json(ROOT / rel / "training_state.json")
        result[label] = {"evaluation_loss": state["evaluation_loss"], "max_steps": state["max_steps"]}
    ranked = sorted(result.items(), key=lambda item: item[1]["evaluation_loss"])
    return {"runs": result, "ranked_lowest_loss_first": [name for name, _ in ranked]}


def epoch_selection(run_dir: str, model_label: str) -> dict:
    selection_dir = ROOT / run_dir / "selection"
    epochs = {}
    for path in sorted(selection_dir.glob("epoch-*_val_metrics.json")):
        epoch_label = path.stem.removesuffix("_val_metrics")
        metrics = read_json(path)
        epochs[epoch_label] = {
            "model_selection_score": metrics.get("model_selection_score"),
            "vqa_balanced_accuracy": metrics.get("vqa", {}).get("balanced_accuracy"),
            "grounding_iou_at_0_5": metrics.get("grounding", {}).get("iou_at_0_5"),
            "grounded_vqa_strict_grounded_accuracy": metrics.get("grounded_vqa", {}).get("strict_grounded_accuracy"),
        }
    scores = [(name, values["model_selection_score"]) for name, values in epochs.items() if values["model_selection_score"] is not None]
    delta = None
    direction = None
    if len(scores) >= 2:
        scores.sort()
        delta = round(scores[-1][1] - scores[0][1], 6)
        direction = "later epoch improved" if delta > 0 else ("later epoch regressed" if delta < 0 else "no change")
    selection = read_json(ROOT / run_dir / "model_selection.json")
    return {
        "model": model_label,
        "epochs": epochs,
        "first_to_last_epoch_selection_score_delta": delta,
        "direction": direction,
        "chosen_epoch": selection.get("best_epoch") if selection else None,
        "chosen_validation_score": selection.get("best_validation_score") if selection else None,
    }


def incomplete_or_anomalous_runs() -> list[dict]:
    findings = []

    v2_progress = read_json(ROOT / "models/r3-unified-qwen3vl-2b-2epoch-4050-v2/training_progress.json")
    if v2_progress:
        findings.append({
            "run": "r3-unified-qwen3vl-2b-2epoch-4050-v2",
            "finding": "abandoned mid-run on the RTX 4050 laptop before switching to the RTX 5070 desktop",
            "last_step": v2_progress.get("last_step"),
            "completed_rows": v2_progress.get("completed_rows"),
            "total_rows": v2_progress.get("total_rows"),
            "fraction_complete": round(v2_progress.get("completed_rows", 0) / v2_progress.get("total_rows", 1), 4),
            "rows_per_second_on_4050": v2_progress.get("rows_per_second"),
        })

    smoke_progress = read_json(ROOT / "models/r3-multitask-qwen3vl-2b-smoke-4050/training_progress.json")
    if smoke_progress:
        nan_steps = [entry["step"] for entry in smoke_progress.get("validation_history", []) if entry.get("loss") != entry.get("loss")]
        if nan_steps:
            findings.append({
                "run": "r3-multitask-qwen3vl-2b-smoke-4050",
                "finding": "validation loss was NaN in this smoke run",
                "nan_at_steps": nan_steps,
                "note": "this was a short smoke/sanity run, not a reported baseline; flagged for awareness, not treated as a real result",
            })

    return findings


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "r1_pilot_validation_curve": r1_pilot_curve(),
        "r1_calibration_comparison_25step": calibration_comparison(),
        "r3_qwen3vl_epoch_selection": epoch_selection("models/r3-unified-qwen3vl-2b-2epoch-5070", "Qwen3-VL-2B-Instruct"),
        "r3_paligemma_epoch_selection": epoch_selection("models/r3-multitask-paligemma-3b-2epoch-5070", "PaliGemma-3B"),
        "incomplete_or_anomalous_runs": incomplete_or_anomalous_runs(),
    }

    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("R1 pilot: loss", [entry["loss"] for entry in report["r1_pilot_validation_curve"]["validation_history"]])
    print("R1 calibration ranking (lowest loss first):", report["r1_calibration_comparison_25step"]["ranked_lowest_loss_first"])
    for key in ("r3_qwen3vl_epoch_selection", "r3_paligemma_epoch_selection"):
        section = report[key]
        print(f"{section['model']}: epochs={list(section['epochs'].keys())} delta={section['first_to_last_epoch_selection_score_delta']} "
              f"({section['direction']}), chosen={section['chosen_epoch']}")
    for finding in report["incomplete_or_anomalous_runs"]:
        print("flagged:", finding["run"], "-", finding["finding"])


if __name__ == "__main__":
    main()
