"""Image-ablation shortcut probe.

Runs the R3 Qwen3-VL adapter on real test prompts but with blank/noise
images substituted for the real ANT/POST scans, then compares each
prediction against what the same adapter predicted on the real image
(already recorded in results/qwen3vl/test/predictions.jsonl). Does not
touch the dataset image directory at all: the substitute images are
generated here.

Small and GPU-light by design: SAMPLE_SIZE real rows, batch size 2.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from postprocess.qwen import parse_qwen_multitask_json_prediction  # noqa: E402
from training.qwen import predict  # noqa: E402
from training.qwen_config import load_qwen_experiment  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
from project_paths import PROJECT_ROOT as ROOT
TEST_MANIFEST = ROOT / "datasets/model_exports/qwen3_json/r3/multitask_test.jsonl"
REAL_PREDICTIONS = ROOT / "results/inferences/qwen3vl/test/predictions.jsonl"
CONFIG = REPO / "configs/qwen3vl_r3_multitask_2epoch_5070_fast.json"
ADAPTER = ROOT / "models/r3-unified-qwen3vl-2b-2epoch-5070/best"
OUTPUT_DIR = ROOT / "results/analyses/image_ablation_probe"
SAMPLE_SIZE = 8
IMAGE_SIZE = (256, 1024)  # (width, height), matches the manifest's image_size


def make_synthetic_images() -> dict[str, Path]:
    images_dir = OUTPUT_DIR / "synthetic_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    blank = Image.new("RGB", IMAGE_SIZE, color=(0, 0, 0))
    blank_path = images_dir / "blank.png"
    blank.save(blank_path)
    paths["blank"] = blank_path

    rng = np.random.default_rng(4050)
    noise = Image.fromarray(rng.integers(0, 256, size=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype="uint8"))
    noise_path = images_dir / "noise.png"
    noise.save(noise_path)
    paths["noise"] = noise_path

    return paths


def load_manifest_rows() -> list[dict]:
    rows = []
    with TEST_MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def load_real_predictions() -> dict[str, str]:
    preds = {}
    with REAL_PREDICTIONS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            preds[row["task_id"]] = row["prediction"]
    return preds


def build_probe_manifest(sample_rows: list[dict], condition: str, synthetic_images: dict[str, Path]) -> Path:
    image_path = synthetic_images[condition]
    probe_rows = []
    for row in sample_rows:
        probe_row = dict(row)
        probe_row["task_id"] = f"{row['task_id']}:{condition}"
        probe_row["images"] = [{"view": image["view"], "image": str(image_path), "image_size": list(IMAGE_SIZE)} for image in row["images"]]
        probe_rows.append(probe_row)
    manifest_path = OUTPUT_DIR / f"probe_manifest_{condition}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for probe_row in probe_rows:
            handle.write(json.dumps(probe_row) + "\n")
    return manifest_path


def run_condition(condition: str, manifest_path: Path) -> Path:
    output_path = OUTPUT_DIR / f"predictions_{condition}.jsonl"
    config = load_qwen_experiment(CONFIG)
    predict(config, ADAPTER, manifest_path, output_path, max_new_tokens=96, batch_size=2)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(4050)

    manifest_rows = load_manifest_rows()
    real_predictions = load_real_predictions()
    candidates = [row for row in manifest_rows if row["task_id"] in real_predictions]
    sample_rows = random.sample(candidates, SAMPLE_SIZE)

    synthetic_images = make_synthetic_images()

    report_rows = []
    for condition in ("blank", "noise"):
        manifest_path = build_probe_manifest(sample_rows, condition, synthetic_images)
        print(f"running inference: condition={condition} rows={len(sample_rows)}")
        pred_path = run_condition(condition, manifest_path)
        ablated_predictions = {}
        with pred_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                original_task_id = row["task_id"].rsplit(":", 1)[0]
                ablated_predictions[original_task_id] = row["prediction"]

        for row in sample_rows:
            task_id = row["task_id"]
            real_parsed = parse_qwen_multitask_json_prediction(real_predictions[task_id])
            ablated_parsed = parse_qwen_multitask_json_prediction(ablated_predictions.get(task_id, ""))
            report_rows.append({
                "task_id": task_id,
                "condition": condition,
                "region": row.get("label"),
                "true_label": row.get("answer_label"),
                "real_image_predicted_label": real_parsed["answer_label"],
                "ablated_predicted_label": ablated_parsed["answer_label"],
                "label_unchanged_under_ablation": real_parsed["answer_label"] == ablated_parsed["answer_label"],
                "real_image_answer": real_parsed["answer"],
                "ablated_answer": ablated_parsed["answer"],
            })

    unchanged = sum(1 for row in report_rows if row["label_unchanged_under_ablation"])
    summary = {
        "sample_size": SAMPLE_SIZE,
        "conditions": ["blank", "noise"],
        "total_comparisons": len(report_rows),
        "label_unchanged_under_ablation_count": unchanged,
        "label_unchanged_under_ablation_rate": round(unchanged / len(report_rows), 4) if report_rows else None,
        "interpretation": (
            "High unchanged-rate means the model's classification answer for these rows does not depend on "
            "real image content, consistent with the class-imbalance shortcut already documented in BASELINES.md. "
            "Low unchanged-rate means the model is sensitive to image content even though its aggregate accuracy "
            "is shortcut-prone."
        ),
        "rows": report_rows,
    }
    (OUTPUT_DIR / "report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"unchanged_under_ablation: {unchanged}/{len(report_rows)} ({summary['label_unchanged_under_ablation_rate']})")


if __name__ == "__main__":
    main()
