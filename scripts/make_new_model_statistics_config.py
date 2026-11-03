"""Write configs/statistics/r3-new-models.json: the statistics config of r3.json with the models trained without images and
with class-balanced epochs added as sources, compared with the reported Qwen3-VL adapter and the question-only control.

Run after the test predictions of the new models exist (scripts/run_queue.py):
  uv run --no-sync python scripts/make_new_model_statistics_config.py
  uv run --no-sync python scripts/compute_r3_statistics.py --config configs/statistics/r3-new-models.json
  uv run --no-sync python scripts/compute_r3_supplementary.py --config configs/statistics/r3-new-models.json
A model whose predictions or metrics file is missing is left out.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "statistics"
NEW = [
    ("blind_trained", "Qwen3-VL-2B-Instruct trained with black images", "results/inferences/r3-qwen3vl/r3-blind-image-qwen3vl-2b-epoch1-4050/test"),
    ("balanced", "Qwen3-VL-2B-Instruct trained on class-balanced epochs", "results/inferences/r3-qwen3vl/r3-class-balanced-qwen3vl-2b-epoch1-4050/test"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads((CONFIGS / "r3.json").read_text(encoding="utf-8"))
    config["output_directory"] = "results/analyses/metrics/new-models/r3-statistics"
    config["log_directory"] = "results/logs/new-models-statistics"
    config["strict_archived_intervals"] = False
    keep = {"qwen3vl", "question_only"}
    config["sources"] = [source for source in config["sources"] if source["name"] in keep]
    comparisons = []
    for name, label, folder in NEW:
        predictions, metrics = ROOT / folder / "predictions.jsonl", ROOT / folder / "metrics.json"
        if not (predictions.exists() and metrics.exists()):
            print(f"left out {name}: {predictions.name} or {metrics.name} missing")
            continue
        config["sources"].append({"name": name, "label": label, "kind": "model", "predictions": {"path": f"{folder}/predictions.jsonl", "sha256": sha256(predictions)}, "reference_metrics": {"path": f"{folder}/metrics.json"}})
        comparisons += [{"name": f"{name}_vs_qwen3vl", "first": name, "second": "qwen3vl", "tables": ["table8"]}, {"name": f"{name}_vs_question_only", "first": name, "second": "question_only", "tables": ["table8"]}]
    config["comparisons"] = comparisons
    config["tables"] = {"table8": config["tables"]["table8"]}
    (CONFIGS / "r3-new-models.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"sources {[s['name'] for s in config['sources']]}, comparisons {[c['name'] for c in comparisons]}")


if __name__ == "__main__":
    main()
