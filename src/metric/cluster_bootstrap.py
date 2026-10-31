"""Vectorized patient-cluster bootstrap on per-patient sufficient statistics.

A resample is a vector of patient multiplicities. With the (patients, columns)
matrix of instance_stats.patient_totals, every resample's counts are one matrix
product, so 2,000 resamples cost milliseconds. The same weight matrix is applied
to every model and control, which pairs their resamples by construction.

Compatibility with metric.qwen.patient_cluster_bootstrap_qwen_multitask:
the resamples come from the same stream (random.Random(seed).choices over the
sorted patients), rates are rounded to the same number of decimals at the same
steps, and the percentile interval uses the same order-statistic rule
(_bootstrap_interval), so replicates and intervals agree with the archived ones.
"""

from __future__ import annotations

import random
from typing import Mapping, Sequence

import numpy as np

from .instance_stats import CLASS_LABELS, COLUMNS, LOCALIZATION_TASKS

DEFAULT_LEVEL = 0.95
DEFAULT_DECIMALS = 6
HARD_LABEL_AUC_ALIAS = {"hard_label_auc": "balanced_accuracy"}


def draw_patient_weights(n_patients: int, resamples: int, seed: int) -> np.ndarray:
    """Multiplicity of each patient in each resample: an int32 matrix (resamples, patients), rows summing to n_patients."""

    if resamples < 1:
        raise ValueError("resamples must be positive")
    if n_patients < 2:
        raise ValueError("Patient-cluster bootstrap requires at least two patients")
    generator = random.Random(seed)
    population = range(n_patients)
    weights = np.zeros((resamples, n_patients), dtype=np.int32)
    for replicate in range(resamples):
        weights[replicate] = np.bincount(generator.choices(population, k=n_patients), minlength=n_patients)
    return weights


def point_totals(matrix: np.ndarray, task: str) -> dict[str, np.ndarray]:
    """Column totals over all patients, each as a length-1 array (the observed sample)."""

    return {name: matrix[:, index].sum(keepdims=True) for index, name in enumerate(COLUMNS[task])}


def resample_totals(matrix: np.ndarray, weights: np.ndarray, task: str) -> dict[str, np.ndarray]:
    """Column totals of every resample: weights (R, P) times matrix (P, K)."""

    if weights.shape[1] != matrix.shape[0]:
        raise ValueError(f"weights cover {weights.shape[1]} patients but the matrix has {matrix.shape[0]}")
    totals = weights.astype(np.float64) @ matrix
    return {name: totals[:, index] for index, name in enumerate(COLUMNS[task])}


def metrics_from_totals(task: str, totals: Mapping[str, np.ndarray], decimals: int = DEFAULT_DECIMALS) -> dict[str, np.ndarray]:
    """Rate metrics from count totals; NaN where the denominator is zero (undefined, never zero)."""

    metrics = {"valid_output_rate": _rate(totals["valid"], totals["n"], decimals)}
    if task in CLASS_LABELS:
        metrics.update(_classification_metrics(totals, decimals))
    if task in LOCALIZATION_TASKS:
        metrics.update(_localization_metrics(task, totals, decimals))
    return metrics


def metric_names(task: str) -> tuple[str, ...]:
    zeros = {name: np.zeros(1) for name in COLUMNS[task]}
    return tuple(metrics_from_totals(task, zeros))


def count_report(task: str, totals: Mapping[str, np.ndarray]) -> dict[str, int]:
    """Integer counts of the observed sample, for the confusion and reproduction checks."""

    counts = {"rows": _count(totals["n"]), "valid_output_rows": _count(totals["valid"])}
    if task in CLASS_LABELS:
        counts.update(
            {
                "positives": _count(totals["pos"]),
                "negatives": _count(totals["neg"]),
                "tp": _count(totals["tp"]),
                "tn": _count(totals["tn"]),
                "fp": _count(totals["fp"]),
                "fn": _count(totals["pos"]) - _count(totals["tp"]),
                "invalid_predictions": _count(totals["invalid"]),
            }
        )
    if task in LOCALIZATION_TASKS:
        counts.update({"target_boxes": _count(totals["targets"]), "valid_boxes": _count(totals["boxes"])})
    return counts


def percentile_interval(values: Sequence[float] | np.ndarray, level: float = DEFAULT_LEVEL, decimals: int = DEFAULT_DECIMALS) -> dict[str, float | int | None]:
    """Percentile interval over the finite values, picked as order statistics.

    Same rule as metric.qwen._bootstrap_interval: sorted[round((n - 1) * tail)]
    and sorted[round((n - 1) * (1 - tail))]. Undefined values (NaN) are skipped
    and counted out of valid_resamples.
    """

    array = np.asarray(values, dtype=np.float64)
    finite = np.sort(array[np.isfinite(array)])
    count = int(finite.size)
    if count == 0:
        return {"valid_resamples": 0, "lower": None, "upper": None}
    tail = round((1.0 - level) / 2.0, 10)
    upper_tail = round(1.0 - tail, 10)
    return {
        "valid_resamples": count,
        "lower": round(float(finite[round((count - 1) * tail)]), decimals),
        "upper": round(float(finite[round((count - 1) * upper_tail)]), decimals),
    }


def paired_bootstrap(first: Sequence[float] | np.ndarray, second: Sequence[float] | np.ndarray, level: float = DEFAULT_LEVEL, decimals: int = DEFAULT_DECIMALS) -> dict[str, float | int | None]:
    """Difference first - second over resamples where both metrics are defined.

    p_value is the two-sided bootstrap p: twice the smaller tail of the
    difference distribution around zero, with one added to numerator and
    denominator so it never reaches zero (same rule as scripts/compare_models.py).
    """

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != second_array.shape:
        raise ValueError("Paired replicates must have the same length")
    both = np.isfinite(first_array) & np.isfinite(second_array)
    differences = first_array[both] - second_array[both]
    pairs = int(differences.size)
    if pairs == 0:
        return {"valid_pairs": 0, "mean_difference": None, "lower": None, "upper": None, "p_value": None}
    interval = percentile_interval(differences, level, decimals)
    at_most_zero = int((differences <= 0).sum())
    at_least_zero = int((differences >= 0).sum())
    p_value = min(1.0, 2.0 * (min(at_most_zero, at_least_zero) + 1) / (pairs + 1))
    return {
        "valid_pairs": pairs,
        "mean_difference": round(float(differences.mean()), decimals),
        "lower": interval["lower"],
        "upper": interval["upper"],
        "p_value": round(p_value, 6),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""

    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _classification_metrics(totals: Mapping[str, np.ndarray], decimals: int) -> dict[str, np.ndarray]:
    tp, tn, fp = totals["tp"], totals["tn"], totals["fp"]
    fn = totals["pos"] - tp
    sensitivity = _rate(tp, totals["pos"], decimals)
    specificity = _rate(tn, totals["neg"], decimals)
    balanced = _round((sensitivity + specificity) / 2, decimals)
    abnormal_f1 = _rate(2 * tp, 2 * tp + fp + fn, decimals)
    normal_f1 = _rate(2 * tn, 2 * tn + fn + fp, decimals)
    return {
        "accuracy": _rate(tp + tn, totals["n"], decimals),
        "balanced_accuracy": balanced,
        # A generated label has one operating point; its ROC area is (sensitivity + specificity) / 2.
        "hard_label_auc": balanced.copy(),
        "macro_f1": _round((abnormal_f1 + normal_f1) / 2, decimals),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "abnormal_precision": _rate(tp, tp + fp, decimals),
        "abnormal_f1": abnormal_f1,
        # Negative-class precision uses the positives missed (fn) as its false positives, as metric.qwen does.
        "normal_precision": _rate(tn, tn + fn, decimals),
        "normal_f1": normal_f1,
    }


def _localization_metrics(task: str, totals: Mapping[str, np.ndarray], decimals: int) -> dict[str, np.ndarray]:
    strict_name = "strict_grounding_accuracy" if task == "grounding" else "strict_grounded_accuracy"
    return {
        "mean_iou": _rate(totals["iou_sum"], totals["targets"], decimals),
        "iou_at_0_5": _rate(totals["iou_hit"], totals["targets"], decimals),
        "region_accuracy": _rate(totals["region_ok"], totals["n"], decimals),
        strict_name: _rate(totals["strict"], totals["n"], decimals),
    }


def _rate(numerator: np.ndarray, denominator: np.ndarray, decimals: int) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return _round(result, decimals)


def _round(values: np.ndarray, decimals: int) -> np.ndarray:
    # Python's round, not np.round, so ties resolve exactly as in metric.qwen (np.round scales by 10**decimals first).
    return np.array([value if value != value else round(value, decimals) for value in np.asarray(values, dtype=np.float64).tolist()], dtype=np.float64).reshape(np.shape(values))


def _count(total: np.ndarray) -> int:
    return int(round(float(total[0])))
