"""Synthetic multitask test rows and model outputs for the statistics tests. No real data."""

from __future__ import annotations

import json
import random
from typing import Any

IMAGE_SIZE = [256, 1024]
REGIONS = ("head", "left chest", "pelvis", "right shoulder")


def make_expected(n_patients: int = 14, seed: int = 42, positive_rate: float = 0.3) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for patient in range(n_patients):
        patient_id = f"p{patient:02d}"
        for task in ("vqa", "grounding", "grounded_vqa"):
            for index in range(rng.randint(1, 3)):
                region = rng.choice(REGIONS)
                positive = rng.random() < positive_rate
                if task == "vqa":
                    answer_label = "yes" if positive else "no"
                elif task == "grounded_vqa":
                    answer_label = "abnormal" if positive else "normal"
                else:
                    answer_label = None
                views = ["ANT"] if rng.random() < 0.5 else ["ANT", "POST"]
                targets = [{"view": view, "bbox": _box(rng)} for view in views]
                rows.append(
                    {
                        "task_id": f"{patient_id}:{task}:{index}",
                        "task": task,
                        "patient_id": patient_id,
                        "split": "test",
                        "label": region,
                        "answer_label": answer_label,
                        "prompt": f"Question about the {region}.",
                        "targets": targets,
                        "images": [
                            {"image": f"wholeBodyANT/{patient_id}.jpg", "image_size": IMAGE_SIZE, "view": "ANT"},
                            {"image": f"wholeBodyPOST/{patient_id}.jpg", "image_size": IMAGE_SIZE, "view": "POST"},
                        ],
                    }
                )
    return rows


def make_predictions(expected: list[dict[str, Any]], seed: int = 7, skill: float = 0.6) -> list[dict[str, Any]]:
    """One model output per row: mostly right, with invalid JSON, wrong task, wrong region and shifted boxes mixed in."""

    rng = random.Random(seed)
    predictions = []
    for row in expected:
        roll = rng.random()
        if roll < 0.06:
            text = "this is not json"
        elif roll < 0.10:
            text = json.dumps({"task": "other", "answer": "?", "class": "maybe"})
        else:
            text = json.dumps(_payload(row, rng, skill), separators=(",", ":"))
        predictions.append({"task_id": row["task_id"], "prediction": text})
    return predictions


def metric_report_for_files(expected: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """The archived-metrics file for these predictions, produced by the existing evaluator."""

    from metric.qwen import evaluate_qwen_multitask_predictions

    return evaluate_qwen_multitask_predictions(expected, predictions)


def _payload(row: dict[str, Any], rng: random.Random, skill: float) -> dict[str, Any]:
    task = row["task"]
    region = row["label"].replace(" ", "_") if rng.random() < 0.93 else "head"
    boxes = [{"view": target["view"].replace("POST", "PST"), "bbox": _shift(target["bbox"], rng, skill)} for target in row["targets"]]
    if len(boxes) == 2 and rng.random() < 0.15:
        boxes = boxes[:1]
    if rng.random() < 0.03:
        boxes = [{"view": "ANT", "bbox": [10, 10, 5000, 40]}]
    if task == "vqa":
        return {"task": "vqa", "answer": "synthetic", "class": _label(row, rng, skill, ("yes", "no"))}
    if task == "grounding":
        return {"task": "grounding", "region": region, "boxes": boxes}
    return {"task": "grounded_vqa", "answer": "synthetic", "region": region, "class": _label(row, rng, skill, ("abnormal", "normal")), "boxes": boxes}


def _label(row: dict[str, Any], rng: random.Random, skill: float, vocabulary: tuple[str, str]) -> str:
    if rng.random() < 0.02:
        return "maybe"
    if rng.random() < skill:
        return row["answer_label"]
    return vocabulary[rng.random() < 0.5]


def _box(rng: random.Random) -> list[int]:
    x = rng.randint(10, 200)
    y = rng.randint(10, 900)
    return [x, y, x + rng.randint(8, 40), y + rng.randint(8, 60)]


def _shift(box: list[int], rng: random.Random, skill: float) -> list[int]:
    spread = 2 if rng.random() < skill else 30
    x1, y1, x2, y2 = (value + rng.randint(-spread, spread) for value in box)
    return [max(0, x1), max(0, y1), max(x1, 0) + max(2, x2 - x1), max(y1, 0) + max(2, y2 - y1)]
