"""Box accuracy of the blank-image test split by target level (bone region and hotspot), real image against black image, validation set.

Run: uv run --project repositories/wbbs-vqa_vgrounding --extra analysis python repositories/wbbs-vqa_vgrounding/scripts/compute_blind_image_levels.py
Writes results/analyses/blind-image/box_accuracy_by_level.csv. The locked test split is never read.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis.r3_supplementary import TARGET_COLUMNS, target_matrix  # noqa: E402
from project_paths import PROJECT_ROOT  # noqa: E402

MANIFEST = PROJECT_ROOT / "datasets/training-exports/qwen3_json/r3-20260723/multitask_val.jsonl"
REAL = {
    "qwen3vl": PROJECT_ROOT / "results/inferences/r3-qwen3vl/r3-unified-qwen3vl-2b-2epoch-5070/evaluation/best/validation/predictions.jsonl",
    "paligemma": PROJECT_ROOT / "results/inferences/r3-paligemma/r3-multitask-paligemma-3b-2epoch-5070/evaluation/best/validation/predictions.jsonl",
}
BLIND = {name: PROJECT_ROOT / "results/analyses/blind-image" / name / "validation/predictions.jsonl" for name in REAL}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    manifest = read(MANIFEST)
    position = {patient: index for index, patient in enumerate(sorted({row["patient_id"] for row in manifest}))}
    rows = []
    for model in REAL:
        for condition, path in (("real image", REAL[model]), ("black image", BLIND[model])):
            prediction = {row["task_id"]: row["prediction"] for row in read(path)}
            for task in ("grounding", "grounded_vqa"):
                for level in ("bone_region", "metastasis"):
                    total = target_matrix(manifest, prediction, task, level, position).sum(axis=0)
                    rows.append({"model": model, "condition": condition, "task": task, "level": level, "targets": int(total[0]), "box_accuracy_iou_0_5": round(total[TARGET_COLUMNS.index("iou_at_0_5")] / total[0], 4), "mean_iou": round(total[1] / total[0], 4)})
    out = PROJECT_ROOT / "results/analyses/blind-image/box_accuracy_by_level.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
