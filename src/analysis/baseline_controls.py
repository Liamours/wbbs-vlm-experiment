"""Reproducible no-image controls for the R3 multitask benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Literal

from metric.qwen import evaluate_qwen_multitask_predictions
from preprocess.canonical import read_jsonl

from .qwen_multitask import audit_multitask_predictions


ControlName = Literal["majority", "prompt_only"]
_CLASS_TASKS = {"vqa", "grounded_vqa"}


def build_control_predictions(train_rows: Iterable[dict[str, Any]], rows: Iterable[dict[str, Any]], control: ControlName) -> list[dict[str, Any]]:
    if control not in {"majority", "prompt_only"}:
        raise ValueError(f"Unsupported control: {control}")
    priors = _build_priors(list(train_rows))
    predictions = []
    for row in rows:
        task = str(row["task"])
        region = priors["global_region"] if control == "majority" else _prompt_region(str(row["prompt"]), priors["regions"])
        region = region or priors["global_region"]
        payload: dict[str, Any]
        if task == "vqa":
            label = _majority_class(task, priors) if control == "majority" else _prompt_class(task, str(row["prompt"]), priors)
            payload = {"task": "vqa", "answer": f"No-image prior: {label}.", "class": label}
        elif task == "grounding":
            payload = {"task": "grounding", "region": region, "boxes": _prior_boxes(region, priors)}
        elif task == "grounded_vqa":
            label = _majority_class(task, priors) if control == "majority" else _prompt_class(task, str(row["prompt"]), priors)
            payload = {
                "task": "grounded_vqa",
                "answer": f"No-image prior: {label}.",
                "region": region,
                "class": label,
                "boxes": _prior_boxes(region, priors),
            }
        else:
            raise ValueError(f"Unsupported multitask row: {task}")
        predictions.append({"task_id": row["task_id"], "prediction": json.dumps(payload, separators=(",", ":")), "parsed": payload})
    return predictions


def create_controls(train: Path, validation: Path, output: Path, test: Path | None = None) -> dict[str, Any]:
    """Create no-image controls without accessing test unless it is supplied explicitly."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing controls directory: {output}")
    train_rows = read_jsonl(train)
    result: dict[str, Any] = {"train": str(train), "inputs": {}, "controls": {}}
    inputs: dict[str, Path] = {"validation": validation}
    if test is not None:
        inputs["test"] = test
    for split, path in inputs.items():
        rows = read_jsonl(path)
        result["inputs"][split] = {"path": str(path), "sha256": _sha256(path), "rows": len(rows)}
        for control in ("majority", "prompt_only"):
            predictions = build_control_predictions(train_rows, rows, control)
            directory = output / control / split
            directory.mkdir(parents=True, exist_ok=False)
            _write_jsonl(directory / "predictions.jsonl", predictions)
            metrics = evaluate_qwen_multitask_predictions(rows, predictions)
            audit = audit_multitask_predictions(rows, predictions)
            (directory / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (directory / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result["controls"].setdefault(control, {})[split] = {"rows": len(predictions), "metrics": metrics, "audit": audit}
    result["train_sha256"] = _sha256(train)
    (output / "control_provenance.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _build_priors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    token_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    token_totals: dict[str, Counter[str]] = defaultdict(Counter)
    region_counts: Counter[str] = Counter()
    boxes: dict[str, dict[str, list[list[int]]]] = defaultdict(lambda: defaultdict(list))
    all_boxes: dict[str, list[list[int]]] = defaultdict(list)
    for row in rows:
        task = str(row["task"])
        if task in _CLASS_TASKS:
            label = str(row["answer_label"])
            class_counts[task][label] += 1
            for token in _tokens(str(row["prompt"])):
                token_counts[task][label][token] += 1
                token_totals[task][label] += 1
        if task in {"grounding", "grounded_vqa"}:
            region = str(row["label"]).replace(" ", "_")
            region_counts[region] += 1
            for target in row["targets"]:
                view = "PST" if target["view"] == "POST" else target["view"]
                bbox = [int(value) for value in target["bbox"]]
                boxes[region][view].append(bbox)
                all_boxes[view].append(bbox)
    if not region_counts:
        raise ValueError("Training rows contain no grounding regions")
    return {
        "class_counts": class_counts,
        "token_counts": token_counts,
        "token_totals": token_totals,
        "regions": tuple(sorted(region_counts, key=lambda value: (-len(value), value))),
        "global_region": _mode(region_counts),
        "boxes": boxes,
        "all_boxes": all_boxes,
    }


def _majority_class(task: str, priors: dict[str, Any]) -> str:
    counts: Counter[str] = priors["class_counts"][task]
    if not counts:
        raise ValueError(f"No class prior for task: {task}")
    return _mode(counts)


def _prompt_class(task: str, prompt: str, priors: dict[str, Any]) -> str:
    counts: Counter[str] = priors["class_counts"][task]
    if not counts:
        raise ValueError(f"No class prior for task: {task}")
    vocabulary = {token for labels in priors["token_counts"][task].values() for token in labels}
    total_rows = sum(counts.values())
    scores = {}
    for label, count in counts.items():
        score = math.log(count / total_rows)
        denominator = priors["token_totals"][task][label] + len(vocabulary)
        for token in _tokens(prompt):
            score += math.log((priors["token_counts"][task][label][token] + 1) / denominator)
        scores[label] = score
    return min(scores, key=lambda label: (-scores[label], label))


def _prompt_region(prompt: str, regions: tuple[str, ...]) -> str | None:
    normalized = prompt.lower().replace("-", "_")
    for region in regions:
        pattern = rf"(?<![a-z0-9_]){re.escape(region)}(?![a-z0-9_])"
        if re.search(pattern, normalized):
            return region
    return None


def _prior_boxes(region: str, priors: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for view in ("ANT", "PST"):
        samples = priors["boxes"].get(region, {}).get(view) or priors["all_boxes"].get(view)
        if not samples:
            continue
        coordinates = [round(median([bbox[index] for bbox in samples])) for index in range(4)]
        coordinates[2] = max(coordinates[2], coordinates[0] + 1)
        coordinates[3] = max(coordinates[3], coordinates[1] + 1)
        output.append({"view": view, "bbox": coordinates})
    if not output:
        raise ValueError("No bounding-box prior is available")
    return output


def _mode(values: Counter[str]) -> str:
    return min(values, key=lambda value: (-values[value], value))


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", value.lower().replace("-", "_"))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, help="Locked test input; supply only for the authorized final evaluation.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = create_controls(args.train, args.validation, args.output, test=args.test)
    print(json.dumps({"output": str(args.output), "controls": sorted(report["controls"])}, sort_keys=True))


if __name__ == "__main__":
    main()
