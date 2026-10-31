"""Phase 1 statistics for the R3 locked test: intervals, paired tests and controls.

Flow of one run (configs/statistics/r3.json):

1. read and hash every input; check the test manifest against the patient splits
   and the frozen manifest of the archived runs;
2. judge every prediction file once (metric.instance_stats) and check that the
   observed metrics reproduce the archived metric files; any mismatch stops the run;
3. draw the patient resamples once (metric.cluster_bootstrap) and apply them to
   every model and control, one unit per source and task, appending each finished
   unit to state.jsonl so that a re-run resumes;
4. pair the sources named in the config, apply Holm correction within each table and comparison,
   and write the result files.

Result files are written from state.jsonl and the replicate files only, so a
finished run can be re-assembled without recomputing anything.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np
from tqdm import tqdm

from metric.cluster_bootstrap import (
    HARD_LABEL_AUC_ALIAS,
    count_report,
    draw_patient_weights,
    metric_names,
    metrics_from_totals,
    paired_bootstrap,
    percentile_interval,
    point_totals,
    resample_totals,
    holm_adjust,
)
from metric.instance_stats import CLASS_LABELS, TASKS, build_instances, patient_ids, patient_totals
from project_paths import PROJECT_ROOT

LOGGER = logging.getLogger("r3_statistics")

_RATE_KEYS = {
    "accuracy": "accuracy",
    "classification_accuracy": "accuracy",
    "exact_canonical_answer_accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "macro_f1",
    "f1": "abnormal_f1",
    "precision": "abnormal_precision",
    "recall": "sensitivity",
    "specificity": "specificity",
    "valid_output_rate": "valid_output_rate",
    "mean_iou": "mean_iou",
    "iou_at_0_5": "iou_at_0_5",
    "region_accuracy": "region_accuracy",
    "strict_grounding_accuracy": "strict_grounding_accuracy",
    "strict_grounded_accuracy": "strict_grounded_accuracy",
}
_COUNT_KEYS = {
    "rows": "rows",
    "valid_output_rows": "valid_output_rows",
    "valid_boxes": "valid_boxes",
    "invalid_classifications": "invalid_predictions",
}
_PER_CLASS = ((0, "abnormal", "sensitivity", "positives"), (1, "normal", "specificity", "negatives"))


class InputMismatchError(RuntimeError):
    """An input file differs from what the config pins (hash, rows, patients)."""


class ReproductionError(RuntimeError):
    """The vectorized estimates differ from the archived metrics or intervals."""


class StateMismatchError(RuntimeError):
    """The state file or the stored resamples belong to a different run."""


@dataclass(frozen=True)
class SourceSpec:
    name: str
    label: str
    kind: str
    predictions: Mapping[str, Any]
    reference_metrics: Mapping[str, Any] | None
    reference_intervals: Mapping[str, Any] | None


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    first: str
    second: str
    tables: tuple[str, ...]


@dataclass(frozen=True)
class StatisticsConfig:
    raw: Mapping[str, Any]
    root: Path
    seed: int
    resamples: int
    confidence_level: float
    rounding_decimals: int
    reproduction_tolerance: float
    strict_archived_intervals: bool
    output_directory: Path
    log_directory: Path
    manifest: Mapping[str, Any]
    patient_splits: Mapping[str, Any]
    class_count_manifests: Mapping[str, Path]
    archive: Mapping[str, Any] | None
    sources: tuple[SourceSpec, ...]
    comparisons: tuple[ComparisonSpec, ...]
    tables: Mapping[str, Mapping[str, Any]]
    holm_excluded_metrics: tuple[str, ...]


@dataclass
class SourceSample:
    """Observed-sample view of one source: per-task patient matrices, point metrics and counts."""

    matrices: dict[str, np.ndarray]
    patient_index: dict[str, np.ndarray]
    point: dict[str, dict[str, float]]
    counts: dict[str, dict[str, int]]


@dataclass
class Inputs:
    manifest: list[dict[str, Any]]
    patients: list[str]
    predictions: dict[str, list[dict[str, Any]]]
    reference_metrics: dict[str, dict[str, Any]]
    reference_intervals: dict[str, dict[str, Any]]
    hashes: dict[str, str]
    manifest_report: dict[str, Any]


# ---------------------------------------------------------------- configuration


def load_config(path: Path | str, project_root: Path | str | None = None) -> StatisticsConfig:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("seed", "resamples", "output_directory", "log_directory", "manifest", "patient_splits", "sources", "comparisons", "tables"):
        if key not in raw:
            raise ValueError(f"Statistics config is missing '{key}'")
    if not isinstance(raw["seed"], int) or raw["resamples"] < 1:
        raise ValueError("seed must be an integer and resamples positive")
    sources = tuple(
        SourceSpec(
            name=item["name"],
            label=item.get("label", item["name"]),
            kind=item.get("kind", "model"),
            predictions=item["predictions"],
            reference_metrics=item.get("reference_metrics"),
            reference_intervals=item.get("reference_intervals"),
        )
        for item in raw["sources"]
    )
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("Source names must be unique")
    comparisons = tuple(ComparisonSpec(item["name"], item["first"], item["second"], tuple(item["tables"])) for item in raw["comparisons"])
    if len({comparison.name for comparison in comparisons}) != len(comparisons):
        raise ValueError("Comparison names must be unique")
    tables = raw["tables"]
    for comparison in comparisons:
        if comparison.first not in names or comparison.second not in names:
            raise ValueError(f"Comparison {comparison.name} names an unknown source")
        for table in comparison.tables:
            if table not in tables:
                raise ValueError(f"Comparison {comparison.name} names an unknown table: {table}")
    for table, definition in tables.items():
        for task, metrics in definition["metrics"].items():
            if task not in TASKS:
                raise ValueError(f"Table {table} names an unknown task: {task}")
            unknown = sorted(set(metrics) - set(metric_names(task)))
            if unknown:
                raise ValueError(f"Table {table}, task {task}: unknown metrics {unknown}")
    return StatisticsConfig(
        raw=raw,
        root=root,
        seed=raw["seed"],
        resamples=raw["resamples"],
        confidence_level=float(raw.get("confidence_level", 0.95)),
        rounding_decimals=int(raw.get("rounding_decimals", 6)),
        reproduction_tolerance=float(raw.get("reproduction_tolerance", 1e-6)),
        strict_archived_intervals=bool(raw.get("strict_archived_intervals", True)),
        output_directory=root / raw["output_directory"],
        log_directory=root / raw["log_directory"],
        manifest=raw["manifest"],
        patient_splits=raw["patient_splits"],
        class_count_manifests={split: root / location for split, location in raw.get("class_count_manifests", {}).items()},
        archive=raw.get("archive"),
        sources=sources,
        comparisons=comparisons,
        tables=tables,
        holm_excluded_metrics=tuple(raw.get("holm_excluded_metrics", HARD_LABEL_AUC_ALIAS)),
    )


# ---------------------------------------------------------------- reading inputs


class InputReader:
    """Reads files and archive members, checks pinned hashes, remembers every hash."""

    def __init__(self, config: StatisticsConfig) -> None:
        self._config = config
        self._zip: zipfile.ZipFile | None = None
        self.hashes: dict[str, str] = {}

    def __enter__(self) -> "InputReader":
        archive = self._config.archive
        if archive is not None:
            path = self._config.root / archive["path"]
            self._expect("archive", _sha256_file(path), archive.get("sha256"))
            self._zip = zipfile.ZipFile(path)
        return self

    def __exit__(self, *_: object) -> None:
        if self._zip is not None:
            self._zip.close()

    def read(self, location: Mapping[str, Any], label: str) -> bytes:
        if "archive_member" in location:
            if self._zip is None:
                raise InputMismatchError(f"{label}: an archive member is named but no archive is configured")
            prefix = self._config.archive.get("root", "")
            member = f"{prefix}/{location['archive_member']}" if prefix else location["archive_member"]
            data = self._zip.read(member)
        else:
            data = (self._config.root / location["path"]).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        self._expect(label, digest, location.get("sha256"))
        self.hashes[label] = digest
        return data

    def _expect(self, label: str, digest: str, expected: str | None) -> None:
        self.hashes.setdefault(label, digest)
        if expected and digest != expected:
            raise InputMismatchError(f"{label}: sha256 {digest} differs from the pinned {expected}")


def load_inputs(config: StatisticsConfig, reader: InputReader) -> Inputs:
    manifest, patients, manifest_report = _check_manifest(config, reader)
    predictions: dict[str, list[dict[str, Any]]] = {}
    reference_metrics: dict[str, dict[str, Any]] = {}
    reference_intervals: dict[str, dict[str, Any]] = {}
    for source in config.sources:
        predictions[source.name] = _jsonl(reader.read(source.predictions, f"predictions:{source.name}"))
        if source.reference_metrics:
            reference_metrics[source.name] = json.loads(reader.read(source.reference_metrics, f"reference_metrics:{source.name}"))
        if source.reference_intervals:
            reference_intervals[source.name] = json.loads(reader.read(source.reference_intervals, f"reference_intervals:{source.name}"))
    return Inputs(manifest, patients, predictions, reference_metrics, reference_intervals, dict(reader.hashes), manifest_report)


def _check_manifest(config: StatisticsConfig, reader: InputReader) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    spec = config.manifest
    rows = _jsonl(reader.read(spec, "manifest"))
    patients = patient_ids(rows)
    report: dict[str, Any] = {"path": spec["path"], "rows": len(rows), "patients": len(patients)}
    for key, observed in (("expected_rows", len(rows)), ("expected_patients", len(patients))):
        if key in spec and spec[key] != observed:
            raise InputMismatchError(f"manifest: {key} is {spec[key]} but the file has {observed}")
    wrong_stem = [
        row["task_id"]
        for row in rows
        for image in row.get("images", [])
        if PurePosixPath(str(image["image"]).replace("\\", "/")).stem != row["patient_id"]
    ]
    if wrong_stem:
        raise InputMismatchError(f"manifest: patient_id differs from the image file stem in {len(wrong_stem)} rows, first {wrong_stem[0]}")
    splits = config.patient_splits
    split_rows = _jsonl(reader.read(splits, "patient_splits"))
    split_patients = sorted({str(row["patient_id"]) for row in split_rows if row.get("split") == splits["split"]})
    if split_patients != patients:
        raise InputMismatchError(
            f"manifest patients ({len(patients)}) differ from the {splits['split']} patients of patient_splits ({len(split_patients)})"
        )
    report["patient_splits_match"] = splits["split"]
    if config.manifest.get("parity_path"):
        other = _jsonl(reader.read({"path": spec["parity_path"], "sha256": spec.get("parity_sha256")}, "manifest_parity"))
        differing = [
            first["task_id"]
            for first, second in zip(rows, other, strict=False)
            if first["task_id"] != second["task_id"] or _without_image_paths(first) != _without_image_paths(second)
        ]
        if len(rows) != len(other) or differing:
            raise InputMismatchError(
                f"manifest differs from the frozen manifest {spec['parity_path']} beyond image paths: "
                f"{len(rows)} vs {len(other)} rows, {len(differing)} differing rows"
            )
        report["parity"] = {"path": spec["parity_path"], "rows": len(other), "differing_rows": 0, "ignored": "images[].image"}
    return rows, patients, report


def _without_image_paths(row: Mapping[str, Any]) -> dict[str, Any]:
    images = [{key: value for key, value in image.items() if key != "image"} for image in row.get("images", [])]
    return {**row, "images": images}


# ---------------------------------------------------------------- observed sample and reproduction


def observed_sample(instances: Mapping[str, Any], n_patients: int, decimals: int) -> SourceSample:
    matrices, indexes, point, counts = {}, {}, {}, {}
    for task in TASKS:
        matrix = patient_totals(instances[task], n_patients)
        totals = point_totals(matrix, task)
        matrices[task] = matrix
        indexes[task] = instances[task].patient_index
        point[task] = {name: float(values[0]) for name, values in metrics_from_totals(task, totals, decimals).items()}
        counts[task] = count_report(task, totals)
    return SourceSample(matrices, indexes, point, counts)


def reference_values(task: str, report: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Archived metrics of one task as (rates, counts) under this module's metric names."""

    rates = {name: report[key] for key, name in _RATE_KEYS.items() if key in report}
    counts = {name: report[key] for key, name in _COUNT_KEYS.items() if key in report}
    if task in CLASS_LABELS:
        counts.update(report.get("confusion") or {})
        per_class = report.get("per_class") or {}
        for position, prefix, recall_name, support_name in _PER_CLASS:
            entry = per_class.get(CLASS_LABELS[task][position])
            if entry:
                rates[f"{prefix}_precision"] = entry.get("precision")
                rates[f"{prefix}_f1"] = entry.get("f1")
                rates[recall_name] = entry.get("recall")
                counts[support_name] = entry.get("support")
    return rates, counts


def check_against_reference(
    source: str, task: str, sample: SourceSample, report: Mapping[str, Any], tolerance: float
) -> tuple[dict[str, Any], list[str]]:
    rates, counts = reference_values(task, report)
    point, observed_counts = sample.point[task], sample.counts[task]
    mismatches: list[str] = []
    checked = 0
    worst = 0.0
    for name, value in rates.items():
        if name not in point:
            continue
        checked += 1
        expected = math.nan if value is None else float(value)
        observed = point[name]
        if math.isnan(observed) and math.isnan(expected):
            continue
        difference = abs(observed - expected)
        if difference <= tolerance + 1e-12:
            worst = max(worst, difference)
        else:
            mismatches.append(f"{source} {task} {name}: computed {observed} archived {value}")
    for name, value in counts.items():
        if name not in observed_counts:
            continue
        checked += 1
        if observed_counts[name] != value:
            mismatches.append(f"{source} {task} {name}: computed {observed_counts[name]} archived {value}")
    return {"checked": checked, "max_abs_difference": worst}, mismatches


def check_archived_intervals(source: SourceSpec, reference: Mapping[str, Any], records: Mapping[tuple[str, str], Mapping[str, Any]], config: StatisticsConfig) -> tuple[dict[str, Any], list[str]]:
    mismatches: list[str] = []
    for key, observed in (("seed", config.seed), ("resamples", config.resamples)):
        if reference.get(key) != observed:
            mismatches.append(f"{source.name}: archived {key} {reference.get(key)} differs from {observed}")
    checked = 0
    worst = 0.0
    for path, archived in reference["metrics"].items():
        task, metric = path.split(".", 1)
        interval = records[(source.name, task)]["interval"][metric]
        checked += 1
        pairs = ((interval["lower"], archived.get("lower_95")), (interval["upper"], archived.get("upper_95")))
        if interval["valid_resamples"] != archived.get("valid_resamples"):
            mismatches.append(f"{source.name} {path}: valid resamples {interval['valid_resamples']} vs archived {archived.get('valid_resamples')}")
        for observed, expected in pairs:
            if observed is None or expected is None or abs(observed - expected) > config.reproduction_tolerance + 1e-12:
                mismatches.append(f"{source.name} {path}: interval end computed {observed} archived {expected}")
            else:
                worst = max(worst, abs(observed - expected))
    return {"checked": checked, "max_abs_difference": worst}, mismatches


# ---------------------------------------------------------------- state file


class StateFile:
    """Append-only record of finished units; the last record of a key wins."""

    def __init__(self, path: Path, fingerprint: str) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.units: dict[tuple[str, str], dict[str, Any]] = {}
        self.pairs: dict[tuple[str, str], dict[str, Any]] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        good: list[str] = []
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                if number == len(lines):
                    LOGGER.warning("Dropping an incomplete last line of %s (interrupted append)", self.path)
                    continue
                raise StateMismatchError(f"{self.path} line {number} is not valid JSON") from error
            if record.get("fingerprint") != self.fingerprint:
                raise StateMismatchError(
                    f"{self.path} holds results of a different configuration or input. Re-run with --restart to move them aside."
                )
            self._store(record)
            good.append(line)
        if len(good) != len([line for line in lines if line.strip()]):
            self.path.write_text("\n".join(good) + ("\n" if good else ""), encoding="utf-8", newline="\n")

    def _store(self, record: dict[str, Any]) -> None:
        if record["stage"] == "unit":
            self.units[(record["source"], record["task"])] = record
        else:
            self.pairs[(record["comparison"], record["task"])] = record

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        record = {**record, "fingerprint": self.fingerprint}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._store(record)
        return record


# ---------------------------------------------------------------- units and pairs


def compute_unit(config: StatisticsConfig, source: str, task: str, matrix: np.ndarray, weights: np.ndarray, replicate_path: Path, point: Mapping[str, float], counts: Mapping[str, int]) -> dict[str, Any]:
    started = time.perf_counter()
    replicates = metrics_from_totals(task, resample_totals(matrix, weights, task), config.rounding_decimals)
    _write_replicates(replicate_path, replicates)
    return {
        "stage": "unit",
        "source": source,
        "task": task,
        "replicates": replicate_path.relative_to(config.output_directory).as_posix(),
        "point": {name: _clean(value, config.rounding_decimals) for name, value in point.items()},
        "counts": dict(counts),
        "interval": {name: percentile_interval(values, config.confidence_level, config.rounding_decimals) for name, values in replicates.items()},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def pair_metrics(config: StatisticsConfig, comparison: ComparisonSpec, task: str) -> list[str]:
    ordered: list[str] = []
    for table in comparison.tables:
        for name in config.tables[table]["metrics"].get(task, []):
            if name not in ordered:
                ordered.append(name)
    return ordered


def pair_signature(config: StatisticsConfig, comparison: ComparisonSpec, task: str) -> str:
    payload = {
        "first": comparison.first,
        "second": comparison.second,
        "tables": {table: config.tables[table]["metrics"].get(task, []) for table in comparison.tables},
    }
    return _digest(payload)


def compute_pair(
    config: StatisticsConfig,
    comparison: ComparisonSpec,
    task: str,
    first_unit: Mapping[str, Any],
    second_unit: Mapping[str, Any],
    first_replicates: Mapping[str, np.ndarray],
    second_replicates: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    by_metric = {}
    for name in pair_metrics(config, comparison, task):
        paired = paired_bootstrap(first_replicates[name], second_replicates[name], config.confidence_level, config.rounding_decimals)
        first_point, second_point = first_unit["point"][name], second_unit["point"][name]
        paired["point_first"] = first_point
        paired["point_second"] = second_point
        paired["point_difference"] = None if first_point is None or second_point is None else round(first_point - second_point, config.rounding_decimals)
        by_metric[name] = paired
    return {
        "stage": "pair",
        "comparison": comparison.name,
        "task": task,
        "signature": pair_signature(config, comparison, task),
        "metrics": by_metric,
    }


# ---------------------------------------------------------------- run


def run_statistics(config: StatisticsConfig, restart: bool = False, check_only: bool = False, progress: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    out = config.output_directory
    out.mkdir(parents=True, exist_ok=True)
    with InputReader(config) as reader:
        inputs = load_inputs(config, reader)
    patients = inputs.patients
    fingerprint = _digest(
        {
            "seed": config.seed,
            "resamples": config.resamples,
            "level": config.confidence_level,
            "decimals": config.rounding_decimals,
            "hashes": inputs.hashes,
        }
    )
    LOGGER.info("Inputs read: %d manifest rows, %d patients, %d sources", len(inputs.manifest), len(patients), len(config.sources))
    if restart and not check_only:
        _move_previous_results(out)

    samples, verification = _verify_sources(config, inputs, progress)
    _write_json(out / "verification.json", {"manifest": inputs.manifest_report, "inputs": inputs.hashes, **verification})
    LOGGER.info("Point estimates reproduce the archived metrics (%s)", _verification_line(verification))
    if check_only:
        return {"check_only": True, "output_directory": str(out), "elapsed_seconds": round(time.perf_counter() - started, 2)}

    state = StateFile(out / "state.jsonl", fingerprint)
    weights = _stored_weights(out / "resamples.npz", patients, config)

    unit_total = len(config.sources) * len(TASKS)
    pair_jobs = [(comparison, task) for comparison in config.comparisons for task in TASKS if pair_metrics(config, comparison, task)]
    total = unit_total + len(pair_jobs)
    bar = tqdm(total=total, desc="statistics", disable=not progress)
    computed = skipped = 0
    finished = 0
    run_started = time.perf_counter()

    def advance(label: str) -> None:
        nonlocal finished
        finished += 1
        bar.update(1)
        done = finished
        elapsed = time.perf_counter() - run_started
        eta = elapsed / done * (total - done) if done else 0.0
        LOGGER.info("%s | unit %d/%d | elapsed %s | eta %s", label, done, total, _format_seconds(elapsed), _format_seconds(eta))

    for source in config.sources:
        for task in TASKS:
            record = state.units.get((source.name, task))
            replicate_path = out / "replicates" / f"{source.name}__{task}.npz"
            if record is not None and replicate_path.exists():
                skipped += 1
                advance(f"skip {source.name} {task}")
                continue
            sample = samples[source.name]
            record = compute_unit(config, source.name, task, sample.matrices[task], weights, replicate_path, sample.point[task], sample.counts[task])
            state.append(record)
            computed += 1
            advance(f"done {source.name} {task}")

    interval_checks, interval_mismatches = {}, []
    for source in config.sources:
        if source.name in inputs.reference_intervals:
            summary, mismatches = check_archived_intervals(source, inputs.reference_intervals[source.name], state.units, config)
            interval_checks[source.name] = summary
            interval_mismatches.extend(mismatches)
    if interval_mismatches:
        message = "Intervals differ from the archived ones:\n" + "\n".join(interval_mismatches[:20])
        if config.strict_archived_intervals:
            bar.close()
            raise ReproductionError(message + "\nSet strict_archived_intervals to false to continue after reviewing the difference.")
        LOGGER.warning(message)
    _write_json(out / "verification.json", {"manifest": inputs.manifest_report, "inputs": inputs.hashes, "archived_intervals": interval_checks, **verification})

    computed_pairs = skipped_pairs = 0
    for comparison, task in pair_jobs:
        record = state.pairs.get((comparison.name, task))
        if record is not None and record["signature"] == pair_signature(config, comparison, task):
            skipped_pairs += 1
            advance(f"skip {comparison.name} {task}")
            continue
        first_unit, second_unit = state.units[(comparison.first, task)], state.units[(comparison.second, task)]
        first_replicates = _read_replicates(out / first_unit["replicates"])
        second_replicates = _read_replicates(out / second_unit["replicates"])
        state.append(compute_pair(config, comparison, task, first_unit, second_unit, first_replicates, second_replicates))
        computed_pairs += 1
        advance(f"done {comparison.name} {task}")
    bar.close()

    write_results(config, state, inputs, fingerprint)
    elapsed = round(time.perf_counter() - started, 2)
    LOGGER.info("Finished: %d units computed, %d resumed, %d pairs computed, %d resumed, %s", computed, skipped, computed_pairs, skipped_pairs, _format_seconds(elapsed))
    return {
        "output_directory": str(out),
        "units_computed": computed,
        "units_resumed": skipped,
        "pairs_computed": computed_pairs,
        "pairs_resumed": skipped_pairs,
        "elapsed_seconds": elapsed,
    }


def _verify_sources(config: StatisticsConfig, inputs: Inputs, progress: bool) -> tuple[dict[str, SourceSample], dict[str, Any]]:
    samples: dict[str, SourceSample] = {}
    report: dict[str, Any] = {"point_estimates": {}}
    mismatches: list[str] = []
    for source in tqdm(config.sources, desc="verify", disable=not progress):
        instances = build_instances(inputs.manifest, inputs.predictions[source.name], inputs.patients, progress_label=f"judge {source.name}" if progress else None)
        sample = observed_sample(instances, len(inputs.patients), config.rounding_decimals)
        samples[source.name] = sample
        reference = inputs.reference_metrics.get(source.name)
        report["point_estimates"][source.name] = {}
        for task in TASKS:
            if reference is None:
                report["point_estimates"][source.name][task] = {"checked": 0, "note": "no archived metrics configured"}
                continue
            if task not in reference:
                mismatches.append(f"{source.name}: archived metrics lack task {task}")
                continue
            summary, problems = check_against_reference(source.name, task, sample, reference[task], config.reproduction_tolerance)
            report["point_estimates"][source.name][task] = summary
            mismatches.extend(problems)
    first = next(iter(samples.values()))
    for name, sample in samples.items():
        for task in TASKS:
            if not np.array_equal(sample.patient_index[task], first.patient_index[task]):
                mismatches.append(f"{name} {task}: instance order differs from the first source, resamples would not be paired")
    if mismatches:
        raise ReproductionError(f"{len(mismatches)} mismatches with the archived metrics:\n" + "\n".join(mismatches[:30]))
    return samples, report


def _verification_line(report: Mapping[str, Any]) -> str:
    checked = sum(task.get("checked", 0) for source in report["point_estimates"].values() for task in source.values())
    return f"{checked} values compared"


def _stored_weights(path: Path, patients: list[str], config: StatisticsConfig) -> np.ndarray:
    weights = draw_patient_weights(len(patients), config.resamples, config.seed)
    if path.exists():
        with np.load(path) as stored:
            same = list(stored["patients"]) == patients and np.array_equal(stored["weights"], weights)
        if not same:
            raise StateMismatchError(f"{path} differs from the resamples drawn for this seed and patient list. Re-run with --restart.")
    else:
        _write_npz(path, weights=weights, patients=np.asarray(patients))
    return weights


def _move_previous_results(out: Path) -> None:
    previous = [item for item in out.iterdir() if not item.name.startswith("superseded-")]
    if not previous:
        return
    target = out / f"superseded-{datetime.now():%Y%m%d-%H%M}"
    target.mkdir()
    for item in previous:
        item.rename(target / item.name)
    LOGGER.info("Moved %d previous result entries to %s", len(previous), target)


# ---------------------------------------------------------------- result files


def write_results(config: StatisticsConfig, state: StateFile, inputs: Inputs, fingerprint: str) -> None:
    out = config.output_directory
    labels = {source.name: source.label for source in config.sources}
    interval_rows, confusion_rows = [], []
    for source in config.sources:
        for task in TASKS:
            record = state.units[(source.name, task)]
            order = {name: index for index, name in enumerate(metric_names(task))}
            for name, interval in sorted(record["interval"].items(), key=lambda item: (order.get(item[0], len(order)), item[0])):
                interval_rows.append(
                    {
                        "source": source.name,
                        "label": labels[source.name],
                        "task": task,
                        "metric": name,
                        "estimate": record["point"][name],
                        "lower_95": interval["lower"],
                        "upper_95": interval["upper"],
                        "valid_resamples": interval["valid_resamples"],
                        "resamples": config.resamples,
                    }
                )
            if task in CLASS_LABELS:
                counts = record["counts"]
                confusion_rows.append(
                    {
                        "source": source.name,
                        "label": labels[source.name],
                        "task": task,
                        **{key: counts[key] for key in ("positives", "negatives", "tp", "fn", "fp", "tn", "invalid_predictions")},
                        "sensitivity": record["point"]["sensitivity"],
                        "specificity": record["point"]["specificity"],
                    }
                )
    interval_fields = ["source", "label", "task", "metric", "estimate", "lower_95", "upper_95", "valid_resamples", "resamples"]
    _write_csv(out / "intervals.csv", interval_fields, interval_rows)
    _write_json(out / "intervals.json", interval_rows)
    confusion_fields = ["source", "label", "task", "positives", "negatives", "tp", "fn", "fp", "tn", "invalid_predictions", "sensitivity", "specificity"]
    _write_csv(out / "confusion_counts.csv", confusion_fields, confusion_rows)

    paired_rows = assemble_paired_rows(config, state)
    paired_fields = [
        "table", "comparison", "first", "second", "task", "metric", "point_first", "point_second", "point_difference",
        "mean_difference", "lower_95", "upper_95", "valid_pairs", "p_value", "p_holm", "holm_family_size",
    ]
    _write_csv(out / "paired.csv", paired_fields, paired_rows)
    _write_json(out / "paired.json", paired_rows)

    counts = class_counts(config.class_count_manifests, inputs.manifest)
    _write_json(out / "class_counts.json", counts)
    _write_csv(
        out / "class_counts.csv",
        ["split", "task", "rows", "positive_label", "positives", "negative_label", "negatives", "positive_share", "majority_class_accuracy"],
        [row for split in counts.values() for row in split["classification_rows"]],
    )
    _write_json(
        out / "run_manifest.json",
        {
            "created": datetime.now().isoformat(timespec="seconds"),
            "fingerprint": fingerprint,
            "seed": config.seed,
            "resamples": config.resamples,
            "confidence_level": config.confidence_level,
            "patients": len(inputs.patients),
            "numpy": np.__version__,
            "config": config.raw,
            "inputs": inputs.hashes,
        },
    )


def assemble_paired_rows(config: StatisticsConfig, state: StateFile) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison in config.comparisons:
        for task in TASKS:
            record = state.pairs.get((comparison.name, task))
            if record is None:
                continue
            for table in comparison.tables:
                for name in config.tables[table]["metrics"].get(task, []):
                    values = record["metrics"][name]
                    rows.append(
                        {
                            "table": table,
                            "comparison": comparison.name,
                            "first": comparison.first,
                            "second": comparison.second,
                            "task": task,
                            "metric": name,
                            "point_first": values["point_first"],
                            "point_second": values["point_second"],
                            "point_difference": values["point_difference"],
                            "mean_difference": values["mean_difference"],
                            "lower_95": values["lower"],
                            "upper_95": values["upper"],
                            "valid_pairs": values["valid_pairs"],
                            "p_value": values["p_value"],
                            "p_holm": None,
                            "holm_family_size": None,
                        }
                    )
    excluded = set(config.holm_excluded_metrics)
    for table, comparison in sorted({(row["table"], row["comparison"]) for row in rows}):
        table_rows = [row for row in rows if row["table"] == table and row["comparison"] == comparison]
        family = [row for row in table_rows if row["p_value"] is not None and row["metric"] not in excluded]
        for row, adjusted in zip(family, holm_adjust([row["p_value"] for row in family])):
            row["p_holm"] = round(adjusted, 6)
            row["holm_family_size"] = len(family)
        for row in table_rows:
            base = HARD_LABEL_AUC_ALIAS.get(row["metric"])
            if base is None or row["metric"] not in excluded:
                continue
            for other in family:
                if (other["comparison"], other["task"], other["metric"]) == (row["comparison"], row["task"], base):
                    row["p_holm"], row["holm_family_size"] = other["p_holm"], other["holm_family_size"]
    return rows


def class_counts(manifests: Mapping[str, Path], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rows and class balance per split; the test split comes from the verified manifest."""

    result: dict[str, Any] = {}
    sources = {split: (_jsonl(path.read_bytes()) if split != "test" else test_rows) for split, path in manifests.items()}
    for split, rows in sources.items():
        per_task: dict[str, Counter] = {task: Counter() for task in TASKS}
        for row in tqdm(rows, desc=f"count {split}", leave=False):
            per_task[row["task"]][str(row.get("answer_label"))] += 1
        classification_rows = []
        for task, (positive, negative) in CLASS_LABELS.items():
            counter = per_task[task]
            total = sum(counter.values())
            if total == 0:
                continue
            classification_rows.append(
                {
                    "split": split,
                    "task": task,
                    "rows": total,
                    "positive_label": positive,
                    "positives": counter[positive],
                    "negative_label": negative,
                    "negatives": counter[negative],
                    "positive_share": round(counter[positive] / total, 6),
                    "majority_class_accuracy": round(max(counter[positive], counter[negative]) / total, 6),
                }
            )
        result[split] = {
            "rows": len(rows),
            "patients": len({row.get("patient_id") for row in rows}),
            "rows_per_task": {task: sum(counter.values()) for task, counter in per_task.items()},
            "classification_rows": classification_rows,
        }
    return result


# ---------------------------------------------------------------- small helpers


def _jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _clean(value: float, decimals: int) -> float | None:
    return round(float(value), decimals) if math.isfinite(value) else None


def _format_seconds(seconds: float) -> str:
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes}m{rest:02d}s" if minutes else f"{rest}s"


def _write_replicates(path: Path, replicates: Mapping[str, np.ndarray]) -> None:
    _write_npz(path, **replicates)


def _read_replicates(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f"{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_csv(path: Path, fields: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f"{path.name}.partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: ("" if row.get(field) is None else row[field]) for field in fields})
    temporary.replace(path)
