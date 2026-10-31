from __future__ import annotations

import json
import math

import numpy as np
import pytest

from analysis.r3_statistics import check_against_reference, observed_sample, reference_values
from metric.instance_stats import TASKS, build_instances, judge_row, patient_ids, patient_totals
from metric.qwen import evaluate_qwen_multitask_predictions
from synthetic_multitask import make_expected, make_predictions


def test_vectorized_point_estimates_match_the_existing_evaluator() -> None:
    expected = make_expected(n_patients=20, seed=5)
    predictions = make_predictions(expected, seed=9)
    report = evaluate_qwen_multitask_predictions(expected, predictions)

    patients = patient_ids(expected)
    sample = observed_sample(build_instances(expected, predictions, patients), len(patients), 6)

    total_checked = 0
    for task in TASKS:
        summary, mismatches = check_against_reference("synthetic", task, sample, report[task], tolerance=1e-9)
        assert mismatches == []
        total_checked += summary["checked"]
        rates, counts = reference_values(task, report[task])
        for name, value in rates.items():
            if name in sample.point[task]:
                observed = sample.point[task][name]
                assert (math.isnan(observed) and value is None) or observed == pytest.approx(value, abs=1e-9)
        for name, value in counts.items():
            if name in sample.counts[task]:
                assert sample.counts[task][name] == value
    assert total_checked > 40


def test_invalid_outputs_are_counted_like_the_existing_evaluator() -> None:
    expected = make_expected(n_patients=6, seed=1)
    predictions = [{"task_id": row["task_id"], "prediction": "not json"} for row in expected]
    report = evaluate_qwen_multitask_predictions(expected, predictions)
    patients = patient_ids(expected)
    sample = observed_sample(build_instances(expected, predictions, patients), len(patients), 6)
    for task in TASKS:
        summary, mismatches = check_against_reference("broken", task, sample, report[task], tolerance=1e-9)
        assert mismatches == []
    assert sample.point["vqa"]["valid_output_rate"] == 0.0
    assert sample.point["grounding"]["mean_iou"] == 0.0


def test_non_string_prediction_is_invalid_not_a_crash() -> None:
    row = make_expected(n_patients=2, seed=2)[0]
    judged = judge_row(row, None)
    assert judged["valid"] == 0.0


def test_patient_matrix_adds_up_to_the_task_totals() -> None:
    expected = make_expected(n_patients=10, seed=4)
    predictions = make_predictions(expected, seed=3)
    patients = patient_ids(expected)
    instances = build_instances(expected, predictions, patients)
    for task in TASKS:
        matrix = patient_totals(instances[task], len(patients))
        assert matrix.sum(axis=0)[0] == instances[task].rows
        assert matrix.shape[0] == len(patients)
        assert np.array_equal(matrix[:, 0], np.bincount(instances[task].patient_index, minlength=len(patients)))


def test_predictions_must_cover_the_manifest_exactly() -> None:
    expected = make_expected(n_patients=4, seed=6)
    predictions = make_predictions(expected)
    with pytest.raises(ValueError):
        build_instances(expected, predictions[:-1])
    extra = predictions + [{"task_id": "unknown", "prediction": json.dumps({})}]
    with pytest.raises(ValueError):
        build_instances(expected, extra)
    with pytest.raises(ValueError):
        build_instances(expected + expected[:1], predictions)


def test_rows_without_a_patient_id_are_rejected() -> None:
    expected = make_expected(n_patients=3, seed=8)
    expected[0] = {**expected[0], "patient_id": ""}
    with pytest.raises(ValueError):
        patient_ids(expected)
