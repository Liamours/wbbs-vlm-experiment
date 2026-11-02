from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.candidate_bank import build_bank, candidates, check_bank, coverage, prefix_text
from analysis.score_auc import agreement, class_score, roc_points, summarise, weighted_auc
from project_paths import PROJECT_ROOT
from training.score import left_pad, sequence_logprobs


def _row(task_id, task, level, label, answer, cls, patient="1"):
    skeleton = {"task": task, "answer": answer, "class": cls}
    if task == "grounded_vqa":
        skeleton = {"task": task, "answer": answer, "region": label, "class": cls, "boxes": [{"view": "ANT", "bbox": [1, 2, 3, 4]}]}
    return {"task_id": task_id, "task": task, "region_level": level, "label": label, "patient_id": patient, "target": json.dumps(skeleton, separators=(",", ":"))}


TRAIN = [
    _row("a", "vqa", "metastasis", "right knee", "No—the hotspot in the right knee joint is non-metastatic.", "no"),
    _row("b", "vqa", "metastasis", "pelvis", "No—the hotspot in the pelvis is non-metastatic.", "no"),
    _row("c", "vqa", "metastasis", "pelvis", "Yes, the hotspot in the pelvis is metastatic.", "yes"),
    _row("d", "grounded_vqa", "bone_region", "head", "The head shows abnormal tracer uptake.", "abnormal"),
    _row("e", "grounded_vqa", "bone_region", "head", "The head does not show abnormal tracer uptake.", "normal"),
]


def test_bank_slots_the_region_and_rebuilds_every_training_prefix():
    bank = build_bank(TRAIN)
    assert dict(bank[("vqa", "metastasis", "no")]) == {"No—the hotspot in the {region} is non-metastatic.": 2}
    assert check_bank(TRAIN) == 5
    assert coverage(bank, 1)["vqa/metastasis/no"] == 1.0


def test_candidates_cover_both_classes_and_end_at_the_class_value():
    bank = build_bank(TRAIN)
    found = candidates(_row("t", "vqa", "metastasis", "right knee", "No, whatever.", "no"), bank)
    assert sorted(c["class"] for c in found) == ["no", "yes"]
    yes = next(c for c in found if c["class"] == "yes")
    assert yes["prefix"] == '{"task":"vqa","answer":"Yes, the hotspot in the right knee joint is metastatic.","class":"yes'
    grounded = candidates(_row("g", "grounded_vqa", "bone_region", "head", "x", "normal"), bank)[0]["prefix"]
    assert grounded.endswith('"region":"head","class":"abnormal') or grounded.endswith('"region":"head","class":"normal')
    assert "boxes" not in grounded


def test_prefix_keeps_the_ascii_escaping_of_the_training_targets():
    assert chr(92) + "u2014" in prefix_text({"task": "vqa", "answer": "", "class": ""}, chr(0x2014), "no")


@pytest.mark.skipif(not (PROJECT_ROOT / "datasets/training-exports/qwen3_json/r3/multitask_train.jsonl").exists(), reason="training export not present")
def test_real_training_split_rebuilds_from_the_bank():
    rows = [json.loads(line) for line in (PROJECT_ROOT / "datasets/training-exports/qwen3_json/r3/multitask_train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert check_bank(rows) == sum(r["task"] in ("vqa", "grounded_vqa") for r in rows)


def test_weighted_auc_matches_sklearn_with_ties_and_duplicated_patients():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    scores = np.round(rng.normal(size=60), 1)
    positives = rng.random(60) < 0.3
    patients = rng.integers(0, 12, size=60)
    ones = np.ones((1, 12))
    assert weighted_auc(scores, positives, patients, ones)[0] == pytest.approx(sklearn_metrics.roc_auc_score(positives, scores))
    multiplicity = rng.integers(0, 3, size=12)
    keep = np.repeat(np.arange(60), multiplicity[patients])
    assert weighted_auc(scores, positives, patients, multiplicity[None, :])[0] == pytest.approx(sklearn_metrics.roc_auc_score(positives[keep], scores[keep]))


def test_resample_without_positives_is_undefined_not_zero():
    value = weighted_auc(np.array([0.1, 0.2]), np.array([True, False]), np.array([0, 1]), np.array([[0, 2]]))
    assert np.isnan(value[0])


def test_roc_points_integrate_to_the_auc():
    rng = np.random.default_rng(1)
    scores, positives = np.round(rng.normal(size=80), 1), rng.random(80) < 0.4
    points = np.array(roc_points(scores, positives))
    area = np.trapezoid(points[:, 1], points[:, 0])
    assert area == pytest.approx(weighted_auc(scores, positives, np.arange(80), np.ones((1, 80)))[0])


def test_class_score_is_the_log_odds_of_the_summed_probabilities():
    logps = [np.log(0.1), np.log(0.2), np.log(0.3)]
    assert class_score(logps, ["no", "no", "yes"], "yes") == pytest.approx(np.log(0.3 / 0.3))


def test_sequence_logprobs_reads_the_right_positions_of_a_left_padded_window():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 6, 11, generator=generator)
    tokens, trailing = [[3, 5, 7], [2, 4]], [0, 1]
    got = sequence_logprobs(logits, tokens, trailing)
    steps = torch.log_softmax(logits, dim=-1)
    assert got[0] == pytest.approx(float(steps[0, 2, 3] + steps[0, 3, 5] + steps[0, 4, 7]))
    assert got[1] == pytest.approx(float(steps[1, 2, 2] + steps[1, 3, 4]))
    with pytest.raises(ValueError):
        sequence_logprobs(logits, [[1] * 6], [0])


def test_left_pad_aligns_the_ends():
    padded = left_pad([torch.tensor([1, 2, 3]), torch.tensor([4])], 0)
    assert padded.tolist() == [[1, 2, 3], [0, 0, 4]]


def test_agreement_compares_the_side_and_the_top_answer():
    records = [
        {"task_id": "1", "task": "vqa", "score": 1.0, "top_answer": "A"},
        {"task_id": "2", "task": "vqa", "score": -1.0, "top_answer": "B"},
        {"task_id": "3", "task": "vqa", "score": 1.0, "top_answer": "C"},
    ]
    reference = {"1": {"class": "yes", "answer": "A"}, "2": {"class": "yes", "answer": "B"}}
    assert agreement(records, reference) == {"compared": 2, "label_agreement": 0.5, "answer_agreement": 1.0}


def test_summary_reports_pooled_and_level_auc(tmp_path: Path):
    records = []
    for index in range(40):
        positive = index % 4 in (0, 1)
        records.append({"task_id": str(index), "task": "vqa", "region_level": "metastasis" if index % 2 else "bone_region", "patient_id": str(index % 8), "answer_label": "yes" if positive else "no", "score": 1.0 + index / 100 if positive else -1.0})
    patients = [str(i) for i in range(8)]
    weights = np.random.default_rng(2).multinomial(8, [1 / 8] * 8, size=50)
    records_grounded = [{**r, "task": "grounded_vqa", "answer_label": "abnormal" if r["answer_label"] == "yes" else "normal"} for r in records]
    rows = summarise(records + records_grounded, patients, weights, 0.95)
    assert {(r["task"], r["scope"]) for r in rows} == {(t, s) for t in ("vqa", "grounded_vqa") for s in ("all", "bone_region", "metastasis")}
    assert all(r["auc"] == 1.0 for r in rows)
