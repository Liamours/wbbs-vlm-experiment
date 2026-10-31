"""Per-instance sufficient statistics for the R3 multitask metrics.

Every prediction is parsed and judged once. The judgement is a short row of
counts and sums (correct, positive, IoU sum, ...) that add up over rows, so a
patient-cluster resample becomes a matrix product (see cluster_bootstrap)
instead of a second pass over the predictions.

The judging rules are those of metric.qwen.evaluate_qwen_multitask_predictions
(_multitask_vqa_metrics, _multitask_grounding_metrics, _multitask_grounded_metrics);
the helpers that decide validity, class aliases and view names are imported from
there so the two code paths cannot drift apart. Nothing here imports torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from postprocess.qwen import parse_qwen_multitask_json_prediction

from .grounding import iou
from .qwen import _canonical_binary_prediction, _tag_view, _valid_multitask_output

TASKS = ("vqa", "grounding", "grounded_vqa")
CLASS_LABELS = {"vqa": ("yes", "no"), "grounded_vqa": ("abnormal", "normal")}
LOCALIZATION_TASKS = ("grounding", "grounded_vqa")

_CLASSIFICATION_COLUMNS = ("pos", "neg", "tp", "tn", "fp", "invalid")
_LOCALIZATION_COLUMNS = ("targets", "iou_sum", "iou_hit", "boxes", "region_ok", "strict")
COLUMNS = {
    "vqa": ("n", "valid") + _CLASSIFICATION_COLUMNS,
    "grounding": ("n", "valid") + _LOCALIZATION_COLUMNS,
    "grounded_vqa": ("n", "valid") + _CLASSIFICATION_COLUMNS + _LOCALIZATION_COLUMNS,
}
_EMPTY_PARSE = {"task": None, "answer": None, "region": None, "answer_label": None, "boxes": []}


@dataclass(frozen=True)
class TaskInstances:
    """One task of one prediction file: a row of statistics per test instance, in manifest order."""

    task: str
    task_ids: tuple[str, ...]
    patient_index: np.ndarray
    columns: dict[str, np.ndarray]

    @property
    def rows(self) -> int:
        return len(self.task_ids)


def patient_ids(expected: Iterable[Mapping[str, Any]]) -> list[str]:
    """Sorted patient ids, the cluster order used by metric.qwen.patient_cluster_bootstrap_qwen_multitask."""

    found: set[str] = set()
    for row in expected:
        patient_id = row.get("patient_id")
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError("Patient-cluster bootstrap requires non-empty patient_id on every expected row")
        found.add(patient_id)
    if len(found) < 2:
        raise ValueError("Patient-cluster bootstrap requires at least two patients")
    return sorted(found)


def build_instances(
    expected: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
    patients: Sequence[str] | None = None,
    progress_label: str | None = None,
) -> dict[str, TaskInstances]:
    """Judge every prediction once and group the rows by task.

    Pass the same `patients` list for every model and control so that patient
    positions, and therefore resamples, are shared.
    """

    expected_rows = list(expected)
    prediction_map = {row["task_id"]: row for row in predictions}
    expected_ids = [row["task_id"] for row in expected_rows]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected rows contain duplicate task_id values")
    if set(expected_ids) != set(prediction_map):
        raise ValueError("Predictions must cover expected task_id values exactly")
    patient_list = list(patients) if patients is not None else patient_ids(expected_rows)
    position = {patient: index for index, patient in enumerate(patient_list)}

    ids: dict[str, list[str]] = {task: [] for task in TASKS}
    owners: dict[str, list[int]] = {task: [] for task in TASKS}
    values: dict[str, list[tuple[float, ...]]] = {task: [] for task in TASKS}
    iterator = tqdm(expected_rows, desc=progress_label, leave=False) if progress_label else expected_rows
    for row in iterator:
        task = row.get("task")
        if task not in COLUMNS:
            raise ValueError(f"Unsupported multitask row task: {task}")
        patient_id = row.get("patient_id")
        if patient_id not in position:
            raise ValueError(f"Row {row['task_id']} has a patient_id outside the patient list: {patient_id!r}")
        judged = judge_row(row, prediction_map[row["task_id"]]["prediction"])
        ids[task].append(row["task_id"])
        owners[task].append(position[patient_id])
        values[task].append(tuple(judged[name] for name in COLUMNS[task]))

    instances = {}
    for task in TASKS:
        matrix = np.asarray(values[task], dtype=np.float64).reshape(len(values[task]), len(COLUMNS[task]))
        instances[task] = TaskInstances(
            task=task,
            task_ids=tuple(ids[task]),
            patient_index=np.asarray(owners[task], dtype=np.int64),
            columns={name: matrix[:, index] for index, name in enumerate(COLUMNS[task])},
        )
    return instances


def patient_totals(instances: TaskInstances, n_patients: int) -> np.ndarray:
    """Sum the statistics of each patient: a (patients, columns) matrix."""

    names = COLUMNS[instances.task]
    matrix = np.zeros((n_patients, len(names)), dtype=np.float64)
    for index, name in enumerate(names):
        matrix[:, index] = np.bincount(instances.patient_index, weights=instances.columns[name], minlength=n_patients)
    return matrix


def judge_row(row: Mapping[str, Any], prediction: Any) -> dict[str, float]:
    """Statistics of one instance, following metric.qwen's per-task rules."""

    task = row["task"]
    parsed = parse_qwen_multitask_json_prediction(prediction) if isinstance(prediction, str) else dict(_EMPTY_PARSE)
    output_valid = _valid_multitask_output(row, prediction)
    judged: dict[str, float] = {"n": 1.0, "valid": float(output_valid)}
    if task == "vqa":
        # metric.qwen reads the VQA class from any parsed output of the right task, valid or not.
        predicted_class = parsed["answer_label"] if parsed["task"] == "vqa" else None
        judged.update(_classification(row["answer_label"], predicted_class, *CLASS_LABELS["vqa"]))
        return judged
    is_task = output_valid and parsed["task"] == task
    class_ok = True
    if task == "grounded_vqa":
        positive, negative = CLASS_LABELS["grounded_vqa"]
        predicted_class = parsed["answer_label"] if is_task else None
        judged.update(_classification(row["answer_label"], predicted_class, positive, negative))
        class_ok = _canonical_binary_prediction(predicted_class, positive, negative) == row["answer_label"]
    region_ok = bool(is_task and parsed["region"] == str(row["label"]).replace(" ", "_"))
    judged.update(_localization(row, parsed, is_task, region_ok, class_ok))
    return judged


def _classification(target: Any, predicted: Any, positive: str, negative: str) -> dict[str, float]:
    canonical = _canonical_binary_prediction(predicted, positive, negative)
    return {
        "pos": float(target == positive),
        "neg": float(target == negative),
        "tp": float(target == positive and canonical == positive),
        "tn": float(target == negative and canonical == negative),
        "fp": float(target == negative and canonical == positive),
        "invalid": float(canonical not in {positive, negative}),
    }


def _localization(row: Mapping[str, Any], parsed: Mapping[str, Any], is_task: bool, region_ok: bool, class_ok: bool) -> dict[str, float]:
    boxes = {box["view"]: box["bbox"] for box in parsed["boxes"] if box["view"] is not None} if is_task else {}
    scores: list[float] = []
    found = 0
    for target in row["targets"]:
        predicted = boxes.get(_tag_view(target.get("view")))
        scores.append(iou(predicted, target["bbox"]) if predicted is not None else 0.0)
        found += predicted is not None
    return {
        "targets": float(len(scores)),
        "iou_sum": float(sum(scores)),
        "iou_hit": float(sum(score >= 0.5 for score in scores)),
        "boxes": float(found),
        "region_ok": float(region_ok),
        "strict": float(class_ok and region_ok and all(score >= 0.5 for score in scores)),
    }
