from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

from analysis.r3_statistics import (
    InputMismatchError,
    ReproductionError,
    StateMismatchError,
    load_config,
    run_statistics,
)
from metric.qwen import evaluate_qwen_multitask_predictions, patient_cluster_bootstrap_qwen_multitask
from synthetic_multitask import make_expected, make_predictions

SEED = 20260722
RESAMPLES = 30


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _workspace(root: Path, seed: int = SEED, strict_intervals: bool = True) -> Path:
    expected = make_expected(n_patients=18, seed=21)
    first = make_predictions(expected, seed=1, skill=0.8)
    second = make_predictions(expected, seed=2, skill=0.4)
    control = make_predictions(expected, seed=3, skill=0.1)
    patients = sorted({row["patient_id"] for row in expected})

    _write_jsonl(root / "data/manifest.jsonl", expected)
    moved = [{**row, "images": [{**image, "image": "elsewhere/" + image["image"]} for image in row["images"]]} for row in expected]
    _write_jsonl(root / "data/manifest_frozen.jsonl", moved)
    _write_jsonl(root / "data/patient_splits.jsonl", [{"patient_id": p, "split": "test"} for p in patients] + [{"patient_id": "x1", "split": "train"}])
    _write_jsonl(root / "data/control.jsonl", control)
    _write_jsonl(root / "data/second.jsonl", second)
    with zipfile.ZipFile(root / "data/archive.zip", "w") as archive:
        archive.writestr("bundle/first.jsonl", "\n".join(json.dumps(row) for row in first) + "\n")
    for name, predictions in (("first", first), ("second", second), ("control", control)):
        (root / f"data/{name}_metrics.json").write_text(json.dumps(evaluate_qwen_multitask_predictions(expected, predictions)), encoding="utf-8")
    intervals = patient_cluster_bootstrap_qwen_multitask(expected, second, resamples=RESAMPLES, seed=SEED)
    (root / "data/second_intervals.json").write_text(json.dumps(intervals), encoding="utf-8")

    config = {
        "seed": seed,
        "resamples": RESAMPLES,
        "strict_archived_intervals": strict_intervals,
        "output_directory": "out",
        "log_directory": "logs",
        "manifest": {"path": "data/manifest.jsonl", "parity_path": "data/manifest_frozen.jsonl", "expected_rows": len(expected), "expected_patients": len(patients)},
        "patient_splits": {"path": "data/patient_splits.jsonl", "split": "test"},
        "class_count_manifests": {"test": "data/manifest.jsonl"},
        "archive": {"path": "data/archive.zip", "root": "bundle"},
        "sources": [
            {"name": "first", "predictions": {"archive_member": "first.jsonl"}, "reference_metrics": {"path": "data/first_metrics.json"}},
            {
                "name": "second",
                "predictions": {"path": "data/second.jsonl"},
                "reference_metrics": {"path": "data/second_metrics.json"},
                "reference_intervals": {"path": "data/second_intervals.json"},
            },
            {"name": "control", "kind": "control", "predictions": {"path": "data/control.jsonl"}, "reference_metrics": {"path": "data/control_metrics.json"}},
        ],
        "comparisons": [
            {"name": "first_vs_second", "first": "first", "second": "second", "tables": ["table5"]},
            {"name": "first_vs_control", "first": "first", "second": "control", "tables": ["table8"]},
        ],
        "holm_excluded_metrics": ["hard_label_auc"],
        "tables": {
            "table5": {"metrics": {"vqa": ["accuracy", "balanced_accuracy", "hard_label_auc"], "grounded_vqa": ["accuracy", "balanced_accuracy", "hard_label_auc"]}},
            "table8": {"metrics": {"vqa": ["sensitivity", "balanced_accuracy"], "grounding": ["mean_iou"]}},
        },
    }
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _run(root: Path, **options):
    return run_statistics(load_config(root / "config.json", project_root=root), progress=False, **options)


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_full_run_writes_every_result_file_with_holm_within_each_table(tmp_path: Path) -> None:
    _workspace(tmp_path)
    summary = _run(tmp_path)
    assert (summary["units_computed"], summary["units_resumed"]) == (9, 0)
    assert (summary["pairs_computed"], summary["pairs_resumed"]) == (4, 0)
    out = tmp_path / "out"
    for name in ("intervals.csv", "paired.csv", "confusion_counts.csv", "class_counts.csv", "verification.json", "run_manifest.json", "resamples.npz", "state.jsonl"):
        assert (out / name).exists(), name
    assert len((out / "state.jsonl").read_text(encoding="utf-8").splitlines()) == 13

    paired = _rows(out / "paired.csv")
    assert len([row for row in paired if row["table"] == "table5"]) == 6
    assert len([row for row in paired if row["table"] == "table8"]) == 3
    for row in paired:
        if row["metric"] != "hard_label_auc" and row["p_value"]:
            assert float(row["p_holm"]) >= float(row["p_value"])
    table5 = [row for row in paired if row["table"] == "table5"]
    assert {row["holm_family_size"] for row in table5} == {"4"}
    for task in ("vqa", "grounded_vqa"):
        by_metric = {row["metric"]: row for row in table5 if row["task"] == task}
        assert by_metric["hard_label_auc"]["p_holm"] == by_metric["balanced_accuracy"]["p_holm"]
        assert by_metric["hard_label_auc"]["point_first"] == by_metric["balanced_accuracy"]["point_first"]

    intervals = _rows(out / "intervals.csv")
    assert {row["source"] for row in intervals} == {"first", "second", "control"}
    assert all(row["resamples"] == str(RESAMPLES) for row in intervals)


def test_rerun_resumes_every_unit_and_leaves_results_unchanged(tmp_path: Path) -> None:
    _workspace(tmp_path)
    _run(tmp_path)
    out = tmp_path / "out"
    before = {name: (out / name).read_bytes() for name in ("intervals.csv", "paired.csv", "confusion_counts.csv")}
    summary = _run(tmp_path)
    assert (summary["units_computed"], summary["units_resumed"], summary["pairs_computed"], summary["pairs_resumed"]) == (0, 9, 0, 4)
    assert before == {name: (out / name).read_bytes() for name in before}


def test_interrupted_run_recomputes_only_the_unfinished_unit(tmp_path: Path) -> None:
    _workspace(tmp_path)
    _run(tmp_path)
    state = tmp_path / "out" / "state.jsonl"
    lines = state.read_text(encoding="utf-8").splitlines()
    state.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:20], encoding="utf-8")
    summary = _run(tmp_path)
    assert summary["units_computed"] == 0
    assert summary["pairs_computed"] == 1
    assert len(state.read_text(encoding="utf-8").splitlines()) == 13


def test_changed_reference_metrics_stop_the_run_before_any_result(tmp_path: Path) -> None:
    _workspace(tmp_path)
    metrics_path = tmp_path / "data" / "first_metrics.json"
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    report["vqa"]["balanced_accuracy"] += 0.01
    metrics_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ReproductionError, match="balanced_accuracy"):
        _run(tmp_path)
    assert not (tmp_path / "out" / "state.jsonl").exists()


def test_changed_archived_interval_stops_the_run_unless_relaxed(tmp_path: Path) -> None:
    _workspace(tmp_path)
    path = tmp_path / "data" / "second_intervals.json"
    intervals = json.loads(path.read_text(encoding="utf-8"))
    intervals["metrics"]["vqa.macro_f1"]["upper_95"] += 0.05
    path.write_text(json.dumps(intervals), encoding="utf-8")
    with pytest.raises(ReproductionError, match="vqa.macro_f1"):
        _run(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    config["strict_archived_intervals"] = False
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert _run(tmp_path)["units_resumed"] == 9


def test_different_seed_cannot_extend_an_existing_state_file(tmp_path: Path) -> None:
    _workspace(tmp_path)
    _run(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    config["seed"] = 1
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(StateMismatchError):
        _run(tmp_path)


def test_restart_moves_old_results_aside(tmp_path: Path) -> None:
    _workspace(tmp_path)
    _run(tmp_path)
    summary = _run(tmp_path, restart=True)
    assert summary["units_computed"] == 9
    assert list((tmp_path / "out").glob("superseded-*"))


def test_manifest_that_differs_from_the_frozen_one_beyond_image_paths_is_rejected(tmp_path: Path) -> None:
    _workspace(tmp_path)
    frozen = tmp_path / "data" / "manifest_frozen.jsonl"
    rows = [json.loads(line) for line in frozen.read_text(encoding="utf-8").splitlines()]
    rows[0]["targets"][0]["bbox"][0] += 1
    _write_jsonl(frozen, rows)
    with pytest.raises(InputMismatchError, match="frozen manifest"):
        _run(tmp_path)


def test_patients_must_match_the_split_file(tmp_path: Path) -> None:
    _workspace(tmp_path)
    splits = tmp_path / "data" / "patient_splits.jsonl"
    rows = [json.loads(line) for line in splits.read_text(encoding="utf-8").splitlines()]
    _write_jsonl(splits, rows[1:])
    with pytest.raises(InputMismatchError, match="patient_splits"):
        _run(tmp_path)


def test_pinned_hash_mismatch_is_reported(tmp_path: Path) -> None:
    _workspace(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    config["sources"][2]["predictions"]["sha256"] = "0" * 64
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(InputMismatchError, match="sha256"):
        _run(tmp_path)


def test_check_only_verifies_without_bootstrapping(tmp_path: Path) -> None:
    _workspace(tmp_path)
    summary = _run(tmp_path, check_only=True)
    assert summary["check_only"] is True
    assert (tmp_path / "out" / "verification.json").exists()
    assert not (tmp_path / "out" / "state.jsonl").exists()
