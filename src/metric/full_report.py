from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from postprocess.qwen import parse_qwen_multitask_json_prediction
from preprocess.canonical import read_jsonl

from .grounding import iou, iou_threshold_sweep
from .qwen import evaluate_qwen_grounded_predictions, evaluate_qwen_multitask_predictions
from .text import evaluate_text_answers


def evaluate_full(expected: Iterable[dict[str, Any]], predictions: Iterable[dict[str, Any]], task: str = "multitask", output_schema: str = "json") -> dict[str, Any]:
    """Classification + bbox metrics (existing) plus text-generation quality and an IoU-threshold sweep (new).

    Every R3 export target (vqa/grounding/grounded_vqa/multitask) is rendered as
    the same {"task","answer","region","class","boxes"} JSON shape, so one
    parser reads both the reference (row["target"]) and the prediction here.
    """

    expected_rows = list(expected)
    base = (
        evaluate_qwen_multitask_predictions(expected_rows, predictions)
        if task == "multitask"
        else evaluate_qwen_grounded_predictions(expected_rows, predictions, output_schema)
    )
    predicted_by_id = {row["task_id"]: row["prediction"] for row in predictions}
    text_pairs: list[tuple[str, str]] = []
    all_ious: list[float] = []
    for row in expected_rows:
        reference = parse_qwen_multitask_json_prediction(row["target"])
        predicted = parse_qwen_multitask_json_prediction(predicted_by_id[row["task_id"]])
        if reference["answer"] and predicted["answer"]:
            text_pairs.append((predicted["answer"], reference["answer"]))
        predicted_boxes = {box["view"]: box["bbox"] for box in predicted["boxes"]}
        for target in row.get("targets", []):
            target_view = "PST" if target.get("view") == "POST" else target.get("view")
            predicted_box = predicted_boxes.get(target_view)
            all_ious.append(iou(predicted_box, target["bbox"]) if predicted_box is not None else 0.0)
    base["text_quality"] = evaluate_text_answers(text_pairs)
    base["bbox_threshold_sweep"] = iou_threshold_sweep(all_ious)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Full evaluation: classification + bbox + text-generation quality.")
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--task", choices=("grounded_vqa", "multitask"), default="multitask")
    parser.add_argument("--output-schema", choices=("tags", "json"), default="json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_full(read_jsonl(args.expected), read_jsonl(args.predictions), args.task, args.output_schema)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
