from __future__ import annotations

import math

import numpy as np
import pytest

from metric.cluster_bootstrap import (
    HARD_LABEL_AUC_ALIAS,
    count_report,
    draw_patient_weights,
    holm_adjust,
    metrics_from_totals,
    paired_bootstrap,
    percentile_interval,
    point_totals,
    resample_totals,
)
from metric.instance_stats import COLUMNS, build_instances, patient_ids, patient_totals
from metric.qwen import _bootstrap_interval, patient_cluster_bootstrap_qwen_multitask
from synthetic_multitask import make_expected, make_predictions


def _vqa_matrix(rows_per_patient: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    """Patient matrix for the vqa task from (pos, neg, tp, tn, fp) per patient; rows are pos + neg, all outputs valid."""

    names = COLUMNS["vqa"]
    matrix = np.zeros((len(rows_per_patient), len(names)))
    for index, (pos, neg, tp, tn, fp) in enumerate(rows_per_patient):
        values = {"n": pos + neg, "valid": pos + neg, "pos": pos, "neg": neg, "tp": tp, "tn": tn, "fp": fp, "invalid": 0}
        matrix[index] = [values[name] for name in names]
    return matrix


def test_holm_matches_hand_computation() -> None:
    assert holm_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm_adjust([0.5]) == [0.5]
    assert holm_adjust([0.4, 0.9]) == pytest.approx([0.8, 0.9])
    assert holm_adjust([0.0004, 0.0004, 0.2])[:2] == pytest.approx([0.0012, 0.0012])


def test_holm_never_lowers_a_p_value_and_keeps_input_order() -> None:
    raw = [0.2, 0.001, 0.03, 0.03, 0.5]
    adjusted = holm_adjust(raw)
    assert all(after >= before for before, after in zip(raw, adjusted))
    assert adjusted[1] == pytest.approx(0.005)
    assert adjusted[2] == adjusted[3]
    assert all(value <= 1.0 for value in adjusted)


def test_percentile_interval_uses_order_statistics_and_skips_undefined_values() -> None:
    values = np.arange(2000, dtype=float)
    interval = percentile_interval(values)
    assert (interval["valid_resamples"], interval["lower"], interval["upper"]) == (2000, 50.0, 1949.0)
    with_gaps = np.concatenate([values, np.full(100, np.nan)])
    gapped = percentile_interval(with_gaps)
    assert gapped == interval
    assert percentile_interval(np.full(5, np.nan)) == {"valid_resamples": 0, "lower": None, "upper": None}


def test_percentile_interval_agrees_with_the_existing_interval_function() -> None:
    generator = np.random.default_rng(42)
    for size in (7, 100, 1999, 2000):
        values = np.round(generator.normal(size=size), 6)
        expected = _bootstrap_interval([float(value) for value in values])
        got = percentile_interval(values)
        assert (got["valid_resamples"], got["lower"], got["upper"]) == (expected["valid_resamples"], expected["lower_95"], expected["upper_95"])


def test_weights_are_reproducible_and_each_resample_draws_n_patients() -> None:
    first = draw_patient_weights(281, 50, 20260722)
    again = draw_patient_weights(281, 50, 20260722)
    other = draw_patient_weights(281, 50, 42)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    assert first.shape == (50, 281)
    assert (first.sum(axis=1) == 281).all()
    with pytest.raises(ValueError):
        draw_patient_weights(1, 10, 42)
    with pytest.raises(ValueError):
        draw_patient_weights(10, 0, 42)


def test_clusters_enter_and_leave_resamples_whole() -> None:
    # Patient 0 has 3 rows, patient 1 has 1 row, patient 2 has 2 rows.
    matrix = _vqa_matrix([(2, 1, 1, 1, 0), (0, 1, 0, 1, 0), (1, 1, 1, 0, 1)])
    weights = np.array([[2, 0, 1], [0, 3, 0], [1, 1, 1]], dtype=np.int32)
    totals = resample_totals(matrix, weights, "vqa")
    assert totals["n"].tolist() == [8, 3, 6]
    assert totals["pos"].tolist() == [5, 0, 3]
    assert totals["tp"].tolist() == [3, 0, 2]


def test_two_models_on_one_manifest_share_every_resampled_row_count() -> None:
    expected = make_expected()
    patients = patient_ids(expected)
    weights = draw_patient_weights(len(patients), 30, 20260722)
    first = build_instances(expected, make_predictions(expected, seed=1), patients)
    second = build_instances(expected, make_predictions(expected, seed=2, skill=0.2), patients)
    for task in first:
        a = resample_totals(patient_totals(first[task], len(patients)), weights, task)
        b = resample_totals(patient_totals(second[task], len(patients)), weights, task)
        assert np.array_equal(a["n"], b["n"])
        if "pos" in a:
            assert np.array_equal(a["pos"], b["pos"])
        assert np.array_equal(first[task].patient_index, second[task].patient_index)


def test_known_toy_bootstrap_accuracy() -> None:
    # Three patients with one row each: right, wrong, right.
    matrix = _vqa_matrix([(1, 0, 1, 0, 0), (0, 1, 0, 0, 1), (1, 0, 1, 0, 0)])
    weights = np.array([[1, 1, 1], [3, 0, 0], [0, 3, 0], [0, 1, 2]], dtype=np.int32)
    metrics = metrics_from_totals("vqa", resample_totals(matrix, weights, "vqa"))
    assert metrics["accuracy"].tolist() == [0.666667, 1.0, 0.0, 0.666667]
    interval = percentile_interval(metrics["accuracy"])
    assert (interval["lower"], interval["upper"]) == (0.0, 1.0)


def test_hard_label_auc_equals_balanced_accuracy_sensitivity_and_specificity() -> None:
    # 4 positives (3 found) and 8 negatives (5 kept): sensitivity 0.75, specificity 0.625.
    matrix = _vqa_matrix([(4, 8, 3, 5, 3)])
    metrics = metrics_from_totals("vqa", point_totals(matrix, "vqa"))
    sensitivity, specificity = metrics["sensitivity"][0], metrics["specificity"][0]
    assert (sensitivity, specificity) == (0.75, 0.625)
    assert metrics["balanced_accuracy"][0] == pytest.approx((sensitivity + specificity) / 2)
    assert metrics["hard_label_auc"][0] == metrics["balanced_accuracy"][0] == 0.6875
    # Area under the one-point ROC curve (0,0) - (1 - specificity, sensitivity) - (1,1), by the trapezoid rule.
    fpr = 1 - specificity
    area = 0.5 * fpr * sensitivity + 0.5 * (1 - fpr) * (sensitivity + 1)
    assert metrics["hard_label_auc"][0] == pytest.approx(area)
    assert HARD_LABEL_AUC_ALIAS["hard_label_auc"] == "balanced_accuracy"


def test_class_metrics_follow_the_reference_definitions() -> None:
    # tp=3 fp=3 fn=1 tn=5: abnormal precision 0.5, abnormal F1 6/10, normal precision 5/6, normal F1 10/14.
    matrix = _vqa_matrix([(4, 8, 3, 5, 3)])
    metrics = metrics_from_totals("vqa", point_totals(matrix, "vqa"))
    assert metrics["abnormal_precision"][0] == 0.5
    assert metrics["abnormal_f1"][0] == 0.6
    assert metrics["normal_precision"][0] == round(5 / 6, 6)
    assert metrics["normal_f1"][0] == round(10 / 14, 6)
    assert metrics["macro_f1"][0] == round((0.6 + round(10 / 14, 6)) / 2, 6)
    assert metrics["accuracy"][0] == round(8 / 12, 6)
    counts = count_report("vqa", point_totals(matrix, "vqa"))
    assert (counts["tp"], counts["fn"], counts["fp"], counts["tn"]) == (3, 1, 3, 5)


def test_resample_without_the_positive_class_is_undefined_not_zero() -> None:
    # Only patient 0 holds positives. Resamples that skip patient 0 have no sensitivity.
    matrix = _vqa_matrix([(2, 2, 1, 2, 0), (0, 3, 0, 3, 0), (0, 2, 0, 1, 1)])
    weights = np.array([[1, 1, 1], [0, 2, 1], [2, 0, 1], [0, 0, 3]], dtype=np.int32)
    metrics = metrics_from_totals("vqa", resample_totals(matrix, weights, "vqa"))
    sensitivity = metrics["sensitivity"]
    assert np.isnan(sensitivity[1]) and np.isnan(sensitivity[3])
    assert not np.isnan(sensitivity[0]) and not np.isnan(sensitivity[2])
    assert np.isnan(metrics["balanced_accuracy"][1]) and np.isnan(metrics["hard_label_auc"][3])
    assert not np.isnan(metrics["specificity"]).any()
    interval = percentile_interval(sensitivity)
    assert interval["valid_resamples"] == 2
    other = np.array([0.1, 0.2, 0.3, 0.4])
    paired = paired_bootstrap(sensitivity, other)
    assert paired["valid_pairs"] == 2
    assert paired["mean_difference"] == pytest.approx(np.mean([sensitivity[0] - 0.1, sensitivity[2] - 0.3]), abs=1e-6)


def test_paired_bootstrap_difference_interval_and_p_value() -> None:
    first = np.full(2000, 0.60)
    second = np.full(2000, 0.50)
    always_higher = paired_bootstrap(first, second)
    assert always_higher["mean_difference"] == pytest.approx(0.1)
    assert (always_higher["lower"], always_higher["upper"]) == (pytest.approx(0.1), pytest.approx(0.1))
    assert always_higher["p_value"] == pytest.approx(2 * 1 / 2001, abs=1e-6)
    tie = paired_bootstrap(first, first)
    assert tie["p_value"] == 1.0
    centred = np.concatenate([np.full(1000, 0.1), np.full(1000, -0.1)])
    assert paired_bootstrap(centred, np.zeros(2000))["p_value"] == pytest.approx(min(1.0, 2 * 1001 / 2001), abs=1e-6)
    none = paired_bootstrap([math.nan, 0.2], [0.1, math.nan])
    assert none["valid_pairs"] == 0 and none["p_value"] is None
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1])


def test_vectorized_bootstrap_reproduces_the_existing_resampling_function() -> None:
    expected = make_expected(n_patients=16, seed=3)
    predictions = make_predictions(expected, seed=11)
    resamples, seed = 40, 20260722
    reference = patient_cluster_bootstrap_qwen_multitask(expected, predictions, resamples=resamples, seed=seed, include_replicates=True)

    patients = patient_ids(expected)
    weights = draw_patient_weights(len(patients), resamples, seed)
    instances = build_instances(expected, predictions, patients)
    mine: dict[str, np.ndarray] = {}
    for task, instance in instances.items():
        totals = resample_totals(patient_totals(instance, len(patients)), weights, task)
        for name, values in metrics_from_totals(task, totals).items():
            mine[f"{task}.{name}"] = values

    assert reference["patients"] == len(patients)
    for path, replicates in reference["replicates"].items():
        expected_values = np.array([math.nan if value is None else value for value in replicates])
        assert np.allclose(mine[path], expected_values, atol=1e-9, equal_nan=True), path
        interval = percentile_interval(mine[path])
        archived = reference["metrics"][path]
        assert interval["valid_resamples"] == archived["valid_resamples"], path
        assert (interval["lower"], interval["upper"]) == (archived["lower_95"], archived["upper_95"]), path
