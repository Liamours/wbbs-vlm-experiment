"""Score-based AUC of the two adapters from the scored candidates (output of training.score).

An instance's score is log P(positive class) - log P(negative class), each the log-sum-exp over the candidate answer
sentences of that class (analysis.candidate_bank). The AUC is the rank statistic of the score against the source label;
intervals resample the test patients with the same weights as the tables (metric.cluster_bootstrap), pooled and within
each region level, because the region level carries most of the class prior.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.baseline_controls import question_only_scores
from metric.cluster_bootstrap import draw_patient_weights, holm_adjust, paired_bootstrap, percentile_interval
from preprocess.canonical import read_jsonl
from project_paths import PROJECT_ROOT

POSITIVE = {"vqa": "yes", "grounded_vqa": "abnormal"}


def class_score(logps: Sequence[float], classes: Sequence[str], positive: str) -> float:
    """log P(positive) - log P(negative) from candidate log-probabilities."""

    values = np.asarray(logps, dtype=np.float64)
    is_positive = np.asarray([name == positive for name in classes])
    return float(_logsumexp(values[is_positive]) - _logsumexp(values[~is_positive]))


def _logsumexp(values: np.ndarray) -> float:
    top = values.max()
    return float(top + np.log(np.exp(values - top).sum()))


def weighted_auc(scores: np.ndarray, positives: np.ndarray, patient_index: np.ndarray, patient_weights: np.ndarray) -> np.ndarray:
    """AUC of each patient resample: ties count one half; NaN where a resample has no positive or no negative.

    patient_weights is (resamples, patients); patient_index maps each instance to its patient column. One sort serves all
    resamples.
    """

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    is_positive = positives[order]
    weights = patient_weights[:, patient_index[order]].astype(np.float64)
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    group_positive = np.add.reduceat(weights * is_positive, starts, axis=1)
    group_negative = np.add.reduceat(weights * ~is_positive, starts, axis=1)
    below = np.cumsum(group_negative, axis=1) - group_negative
    total_positive, total_negative = group_positive.sum(axis=1), group_negative.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (group_positive * (below + 0.5 * group_negative)).sum(axis=1) / (total_positive * total_negative)


def roc_points(scores: np.ndarray, positives: np.ndarray) -> list[tuple[float, float, float]]:
    """(false positive rate, true positive rate, threshold) at each distinct score, highest threshold first."""

    order = np.argsort(-scores, kind="stable")
    sorted_scores, is_positive = scores[order], positives[order]
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    true_positives = np.cumsum(is_positive)[ends]
    false_positives = np.cumsum(~is_positive)[ends]
    return [(0.0, 0.0, float("inf"))] + [(float(f / false_positives[-1]), float(t / true_positives[-1]), float(sorted_scores[end])) for f, t, end in zip(false_positives, true_positives, ends)]


def load_scores(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def agreement(records: Sequence[Mapping[str, Any]], reference: Mapping[str, str]) -> dict[str, float | int]:
    """How often the score's side and top answer agree with the stored greedy output of the same adapter."""

    label = text = compared = 0
    for record in records:
        stored = reference.get(record["task_id"])
        if stored is None:
            continue
        compared += 1
        label += (record["score"] > 0) == (stored["class"] == POSITIVE[record["task"]])
        text += record["top_answer"] == stored["answer"]
    if compared == 0:
        return {"compared": 0, "label_agreement": float("nan"), "answer_agreement": float("nan")}
    return {"compared": compared, "label_agreement": label / compared, "answer_agreement": text / compared}


def read_reference(path: Path) -> dict[str, dict[str, str]]:
    """Class and answer of each stored greedy prediction that parses as JSON."""

    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            try:
                parsed = json.loads(record["prediction"])
                result[record["task_id"]] = {"class": parsed["class"], "answer": parsed["answer"]}
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return result


def summarise(records: Sequence[Mapping[str, Any]], patients: Sequence[str], weights: np.ndarray, level: float) -> list[dict[str, Any]]:
    """AUC with its interval for each task, pooled and within each region level."""

    column = {patient: index for index, patient in enumerate(patients)}
    rows = []
    for task, positive in POSITIVE.items():
        task_records = [r for r in records if r["task"] == task]
        for scope in ["all", *sorted({r["region_level"] for r in task_records})]:
            chosen = [r for r in task_records if scope in ("all", r["region_level"])]
            scores = np.asarray([r["score"] for r in chosen])
            positives = np.asarray([r["answer_label"] == positive for r in chosen])
            index = np.asarray([column[r["patient_id"]] for r in chosen])
            observed = weighted_auc(scores, positives, index, np.ones((1, len(patients))))[0]
            interval = percentile_interval(weighted_auc(scores, positives, index, weights), level)
            rows.append({"task": task, "scope": scope, "n": len(chosen), "positives": int(positives.sum()), "auc": round(float(observed), 6), "lower": interval["lower"], "upper": interval["upper"], "valid_resamples": interval["valid_resamples"]})
    return rows


def paired_rows(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]], patients: Sequence[str], weights: np.ndarray, level: float) -> list[dict[str, Any]]:
    """Pooled AUC difference first - second per task on the same resamples, Holm-adjusted over the tasks."""

    column = {patient: index for index, patient in enumerate(patients)}
    rows = []
    for task, positive in POSITIVE.items():
        replicates = []
        ids = []
        for records in (first, second):
            chosen = sorted((r for r in records if r["task"] == task), key=lambda r: r["task_id"])
            ids.append([r["task_id"] for r in chosen])
            replicates.append(weighted_auc(np.asarray([r["score"] for r in chosen]), np.asarray([r["answer_label"] == positive for r in chosen]), np.asarray([column[r["patient_id"]] for r in chosen]), weights))
        if ids[0] != ids[1]:
            raise ValueError(f"The two models scored different {task} instances")
        rows.append({"task": task, **paired_bootstrap(replicates[0], replicates[1], level)})
    for row, adjusted in zip(rows, holm_adjust([row["p_value"] for row in rows])):
        row["holm_p"] = adjusted
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path, models: Sequence[str] | None = None) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = _project_path(config["output_dir"])
    bootstrap = config["bootstrap"]
    test_manifest = _project_path(config["splits"]["test"]["manifest"])
    with test_manifest.open(encoding="utf-8") as handle:
        test_rows = [json.loads(line) for line in handle if line.strip()]
    patients = sorted({row["patient_id"] for row in test_rows})
    if len(patients) != bootstrap["expected_patients"]:
        raise ValueError(f"Test manifest has {len(patients)} patients, expected {bootstrap['expected_patients']}")
    weights = draw_patient_weights(len(patients), bootstrap["resamples"], bootstrap["seed"])
    scored = {}
    for system in config["systems"]:
        file = out / system["model"] / system["split"] / "scores.jsonl"
        if file.exists() and (models is None or system["name"] in models):
            records = load_scores(file)
            if len(records) == config["splits"][system["split"]]["scored_rows"]:
                scored[system["name"]] = records
            else:
                print(f"skipped {system['name']}: {len(records)} of {config['splits'][system['split']]['scored_rows']} rows scored")
    control = question_only_scores(read_jsonl(_project_path(config["train_manifest"])), test_rows)
    control_path = out / "question_only" / "test"
    control_path.mkdir(parents=True, exist_ok=True)
    (control_path / "scores.jsonl").write_text("".join(json.dumps(record) + chr(10) for record in control), encoding="utf-8")
    scored["question_only"] = control
    summary, roc = [], []
    for name, records in scored.items():
        summary += [{"model": name, **row} for row in summarise(records, patients, weights, bootstrap["level"])]
        for task, positive in POSITIVE.items():
            chosen = [r for r in records if r["task"] == task]
            roc += [{"model": name, "task": task, "fpr": f, "tpr": t, "threshold": h} for f, t, h in roc_points(np.asarray([r["score"] for r in chosen]), np.asarray([r["answer_label"] == positive for r in chosen]))]
    write_csv(out / "auc.csv", summary)
    write_csv(out / "roc.csv", roc)
    write_csv(out / "auc_paired.csv", [{"first": a, "second": b, **row} for a, b in config["pairs"] if a in scored and b in scored for row in paired_rows(scored[a], scored[b], patients, weights, bootstrap["level"])])
    print(f"wrote {out / 'auc.csv'}")


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", nargs="+", help="Score files to use; default is every model of the config.")
    args = parser.parse_args()
    run(args.config, args.models)


if __name__ == "__main__":
    main()
