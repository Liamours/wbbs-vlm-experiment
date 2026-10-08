from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and materialize the best Qwen multitask epoch from validation metrics.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True, help="Folder with epoch-*_val_metrics.json; lives under results, never under models.")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    selection_dir = args.selection_dir.resolve()
    candidates = []
    for metrics_path in sorted(selection_dir.glob("epoch-*_val_metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        score = metrics.get("model_selection_score")
        if not isinstance(score, (int, float)):
            raise ValueError(f"Missing model_selection_score: {metrics_path}")
        epoch = int(metrics_path.name.split("_", 1)[0].split("-")[1])
        candidates.append((float(score), -epoch, epoch, metrics_path))
    if not candidates:
        raise FileNotFoundError(f"No epoch validation metrics found in {selection_dir}")
    score, _, epoch, metrics_path = max(candidates)
    source = run_dir / "checkpoints" / f"epoch-{epoch:02d}"
    target = run_dir / "best"
    if not source.is_dir():
        raise FileNotFoundError(f"Best checkpoint is missing: {source}")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing best adapter: {target}")
    shutil.copytree(source, target)
    payload = {
        "selection_metric": "mean(vqa.balanced_accuracy, grounding.iou_at_0_5, grounded_vqa.strict_grounded_accuracy)",
        "best_epoch": epoch,
        "best_validation_score": score,
        "best_checkpoint": str(source),
        "best_adapter": str(target),
        "metrics": str(metrics_path),
    }
    (run_dir / "model_selection.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
