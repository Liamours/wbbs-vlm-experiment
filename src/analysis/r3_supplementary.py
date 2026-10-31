"""Breakdowns of the R3 locked-test predictions and a decomposition of the question-only control.

CPU only, reads the archived predictions through the same config as r3_statistics.
Outputs: recall and localization per region level and per region, and the question-only
control split into region words, view words, and wording (template) words.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from metric.cluster_bootstrap import draw_patient_weights, metrics_from_totals, percentile_interval, resample_totals
from metric.instance_stats import CLASS_LABELS, COLUMNS, judge_row
from preprocess.canonical import read_jsonl

from .baseline_controls import _build_priors, _prompt_class, _tokens
from .r3_statistics import InputReader, StatisticsConfig, load_inputs

LOGGER = logging.getLogger("r3_supplementary")
VIEW_WORDS = {"anterior", "posterior", "ant", "post", "pst", "front", "back", "view", "image"}
VARIANTS = ("prior_only", "region_words", "view_words", "wording_words", "all_words")
CLASS_TASKS = ("vqa", "grounded_vqa")


def breakdown(config: StatisticsConfig, out: Path) -> dict[str, int]:
    with InputReader(config) as reader:
        inputs = load_inputs(config, reader)
    rows_by_group: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for source in config.sources:
        prediction = {row["task_id"]: row["prediction"] for row in inputs.predictions[source.name]}
        for row in inputs.manifest:
            judged = judge_row(row, prediction[row["task_id"]])
            for level, group in (("region_level", str(row.get("region_level"))), ("region", str(row.get("label")))):
                totals = rows_by_group[(source.name, row["task"], level, group)]
                for name, value in judged.items():
                    totals[name] += value
    fields = ["source", "task", "grouping", "group", "n", "positives", "recall", "negatives", "specificity", "targets", "iou_at_0_5", "strict"]
    table = []
    for (source, task, grouping, group), t in sorted(rows_by_group.items()):
        record = {"source": source, "task": task, "grouping": grouping, "group": group, "n": int(t["n"])}
        if task in CLASS_TASKS:
            record.update(positives=int(t["pos"]), recall=_ratio(t["tp"], t["pos"]), negatives=int(t["neg"]), specificity=_ratio(t["tn"], t["neg"]))
        if task in ("grounding", "grounded_vqa"):
            record.update(targets=int(t["targets"]), iou_at_0_5=_ratio(t["iou_hit"], t["targets"]), strict=_ratio(t["strict"], t["n"]))
        table.append(record)
    _write_csv(out / "breakdown.csv", fields, table)
    return {"breakdown_rows": len(table)}


def decomposition(config: StatisticsConfig, out: Path) -> dict[str, int]:
    with InputReader(config) as reader:
        inputs = load_inputs(config, reader)
    train = read_jsonl(config.class_count_manifests["train"])
    manifest = inputs.manifest
    patients = inputs.patients
    position = {patient: index for index, patient in enumerate(patients)}
    region_words = {token for row in train if row.get("label") for token in _tokens(str(row["label"]))}

    def filtered(prompt: str, variant: str) -> str:
        tokens = _tokens(prompt)
        if variant == "prior_only":
            return ""
        if variant == "region_words":
            return " ".join(t for t in tokens if t in region_words and t not in VIEW_WORDS)
        if variant == "view_words":
            return " ".join(t for t in tokens if t in VIEW_WORDS)
        if variant == "wording_words":
            return " ".join(t for t in tokens if t not in region_words and t not in VIEW_WORDS)
        return " ".join(tokens)

    weights = draw_patient_weights(len(patients), config.resamples, config.seed)
    rows = []
    started = time.perf_counter()
    for variant in VARIANTS:
        train_variant = [dict(row, prompt=filtered(str(row["prompt"]), variant)) for row in train]
        priors = _build_priors(train_variant)
        for task in CLASS_TASKS:
            positive, negative = CLASS_LABELS[task]
            names = COLUMNS["vqa"]
            matrix = np.zeros((len(patients), len(names)), dtype=np.float64)
            for row in manifest:
                if row["task"] != task:
                    continue
                predicted = _prompt_class(task, filtered(str(row["prompt"]), variant), priors)
                target = row["answer_label"]
                values = {"n": 1.0, "valid": 1.0, "pos": float(target == positive), "neg": float(target == negative),
                          "tp": float(target == positive and predicted == positive), "tn": float(target == negative and predicted == negative),
                          "fp": float(target == negative and predicted == positive), "invalid": 0.0}
                matrix[position[row["patient_id"]]] += [values[name] for name in names]
            point = metrics_from_totals("vqa", {name: matrix[:, i].sum(keepdims=True) for i, name in enumerate(names)}, config.rounding_decimals)
            replicates = metrics_from_totals("vqa", resample_totals(matrix, weights, "vqa"), config.rounding_decimals)
            for metric in ("accuracy", "balanced_accuracy", "sensitivity", "specificity"):
                interval = percentile_interval(replicates[metric], config.confidence_level, config.rounding_decimals)
                rows.append({"variant": variant, "task": task, "metric": metric, "estimate": round(float(point[metric][0]), config.rounding_decimals),
                             "lower_95": interval["lower"], "upper_95": interval["upper"]})
        LOGGER.info("variant %s done | elapsed %.1fs", variant, time.perf_counter() - started)
    _write_csv(out / "control_decomposition.csv", ["variant", "task", "metric", "estimate", "lower_95", "upper_95"], rows)
    return {"decomposition_rows": len(rows)}


def run(config: StatisticsConfig) -> dict[str, Any]:
    out = config.output_directory.parent / "r3-supplementary"
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"output_directory": str(out)}
    summary.update(breakdown(config, out))
    summary.update(decomposition(config, out))
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _write_csv(path: Path, fields: list[str], rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
