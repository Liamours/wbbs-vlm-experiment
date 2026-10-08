"""Build a compact R3 release from frozen R2 manifests.

R3 retains all R2 source-evidence manifests, frozen patient splits, and image
references. It samples only consumer task rows. The deterministic coverage seed
ensures every canonical image remains present in at least one selected task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CORE_FILES = ("canonical.jsonl", "paired_evidence.jsonl", "metastases.jsonl", "patient_splits.jsonl")
TASK_FILES = ("vqa.jsonl", "vgrounding.jsonl")
MULTITASK_FILE = "multitask.jsonl"
METADATA_FILE = "metadata_vqa.jsonl"
DEFAULT_FRACTION = 0.15
DEFAULT_SEED = "wbbs-r3-compact-v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def _digest(seed: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).hexdigest()


def _images(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(image["image"]) for image in row.get("images", []) if isinstance(image, dict) and image.get("image")))


def _stratum(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("split", "missing")),
        str(row.get("region_level", "missing")),
        str(row.get("interaction_type", "missing")),
        str(row.get("answer_label", "missing")),
    )


def _quotas(rows: list[dict[str, Any]], target: int) -> dict[tuple[str, str, str, str], int]:
    groups = Counter(_stratum(row) for row in rows)
    total = len(rows)
    raw = {group: target * count / total for group, count in groups.items()}
    quotas = {group: math.floor(value) for group, value in raw.items()}
    for group in sorted(groups, key=lambda item: (-(raw[item] - quotas[item]), item))[: target - sum(quotas.values())]:
        quotas[group] += 1
    return quotas


def _sample_rows(rows: list[dict[str, Any]], target: int, forced_ids: set[str], seed: str) -> list[dict[str, Any]]:
    if target < len(forced_ids):
        raise ValueError(f"Coverage requires {len(forced_ids)} rows, exceeding target {target}.")
    by_id = {str(row["task_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Task IDs must be unique before compact sampling.")
    forced = [by_id[task_id] for task_id in sorted(forced_ids)]
    quotas = _quotas(rows, target)
    forced_counts = Counter(_stratum(row) for row in forced)
    candidates: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["task_id"]) not in forced_ids:
            candidates[_stratum(row)].append(row)
    proposed: list[dict[str, Any]] = []
    for group, group_rows in candidates.items():
        needed = max(0, quotas.get(group, 0) - forced_counts[group])
        proposed.extend(sorted(group_rows, key=lambda item: _digest(seed, str(item["task_id"])))[:needed])
    capacity = target - len(forced)
    proposed = sorted(proposed, key=lambda item: _digest(seed, str(item["task_id"])))[:capacity]
    selected = [*forced, *proposed]
    selected_ids = {str(row["task_id"]) for row in selected}
    remaining = target - len(selected)
    if remaining > 0:
        extras = [row for row in rows if str(row["task_id"]) not in selected_ids]
        for row in sorted(extras, key=lambda item: _digest(seed, str(item["task_id"])))[:remaining]:
            selected.append(row)
    if len(selected) != target:
        raise AssertionError(f"Selected {len(selected)} rows; expected {target}.")
    return sorted(selected, key=lambda row: (str(row.get("split")), str(row["task_id"])))


def _coverage_seed(vqa: list[dict[str, Any]], vgrounding: list[dict[str, Any]], canonical_images: set[str], seed: str) -> dict[str, set[str]]:
    """Select one paired grounding row per image pair, then VQA rows for leftovers."""

    selected = {"vqa": set(), "vgrounding": set()}
    covered: set[str] = set()
    by_pair: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in vgrounding:
        pair = _images(row)
        if len(pair) != 2:
            continue
        current = by_pair.get(pair)
        if current is None or _digest(seed, str(row["task_id"])) < _digest(seed, str(current["task_id"])):
            by_pair[pair] = row
    for pair, row in sorted(by_pair.items()):
        selected["vgrounding"].add(str(row["task_id"]))
        covered.update(pair)
    for row in sorted(vqa, key=lambda item: _digest(seed, str(item["task_id"]))):
        if any(image in canonical_images - covered for image in _images(row)):
            selected["vqa"].add(str(row["task_id"]))
            covered.update(_images(row))
    missing = canonical_images - covered
    if missing:
        raise ValueError(f"Unable to preserve task coverage for {len(missing)} canonical images.")
    return selected


def _distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    fields = ("split", "region_level", "interaction_type", "answer_label")
    return {field: dict(sorted(Counter(str(row.get(field, "missing")) for row in rows).items())) for field in fields}


def _multitask_rows(selected: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Create R3's primary consumer manifest without changing source fields."""

    combined: list[dict[str, Any]] = []
    requirements = {
        "vqa": {"answer": True, "classification": True, "locations": False},
        "grounding": {"answer": False, "classification": False, "locations": True},
        "grounded_vqa": {"answer": True, "classification": True, "locations": True},
    }
    def prompt(row: dict[str, Any]) -> str:
        value = row.get("question") or row.get("query")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Task row has no question/query prompt: {row.get('task_id')}")
        return value
    for row in selected["vqa"]:
        item = dict(row)
        item.update({"schema_version": "r3-multitask-v1", "task_type": "vqa", "prompt": prompt(row), "required_outputs": requirements["vqa"]})
        combined.append(item)
    for row in selected["vgrounding"]:
        task_type = "grounded_vqa" if row.get("interaction_type") == "localization_and_classification" else "grounding"
        item = dict(row)
        item.update({"schema_version": "r3-multitask-v1", "task_type": task_type, "prompt": prompt(row), "required_outputs": requirements[task_type]})
        combined.append(item)
    return sorted(combined, key=lambda row: (str(row.get("split")), str(row["task_type"]), str(row["task_id"])))


def _metadata_rows(canonical: list[dict[str, Any]], paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create non-training view questions solely from recorded manifest fields."""

    rows: list[dict[str, Any]] = []
    by_image: dict[str, dict[str, Any]] = {}
    for record in canonical:
        image = record.get("image")
        if not image:
            continue
        current = by_image.get(str(image))
        if current is None or str(record["record_id"]) < str(current["record_id"]):
            by_image[str(image)] = record
    for record in by_image.values():
        view = record.get("view")
        if view not in {"ANT", "POST"}:
            continue
        word = "anterior" if view == "ANT" else "posterior"
        rows.append(
            {
                "schema_version": "r3-metadata-v1",
                "task_id": f"{record['record_id']}:metadata:view",
                "record_id": record["record_id"],
                "patient_id": record.get("patient_id"),
                "split": record["split"],
                "task_type": "metadata_vqa",
                "metadata_kind": "view",
                "answer_source": "canonical.view",
                "training_eligible": False,
                "benchmark_eligible": False,
                "prompt": "Which projection is shown in this whole-body bone scan?",
                "answer": f"This is the {word} ({view}) view.",
                "answer_label": view,
                "images": [{"image": record["image"], "image_size": record["image_size"], "view": view}],
                "targets": [],
            }
        )
    by_images: dict[tuple[str, ...], dict[str, Any]] = {}
    for pair in paired:
        images = pair.get("images", [])
        key = tuple(sorted(str(image.get("image")) for image in images if isinstance(image, dict) and image.get("image")))
        if len(key) != 2:
            continue
        current = by_images.get(key)
        if current is None or str(pair["evidence_id"]) < str(current["evidence_id"]):
            by_images[key] = pair
    for pair in by_images.values():
        images = [{key: image[key] for key in ("image", "image_size", "view")} for image in pair["images"]]
        rows.append(
            {
                "schema_version": "r3-metadata-v1",
                "task_id": f"{pair['evidence_id']}:metadata:paired-views",
                "evidence_id": pair["evidence_id"],
                "patient_id": pair.get("patient_id"),
                "split": pair["split"],
                "task_type": "metadata_vqa",
                "metadata_kind": "paired_views",
                "answer_source": "paired_evidence.images.view",
                "training_eligible": False,
                "benchmark_eligible": False,
                "prompt": "Which projections are included in this paired bone scan?",
                "answer": "The inputs are the anterior (ANT) and posterior (POST) views.",
                "answer_label": "ANT_POST",
                "images": images,
                "targets": [],
            }
        )
    return sorted(rows, key=lambda row: (str(row["split"]), str(row["task_id"])))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_checksums(destination: Path, files: Iterable[Path]) -> None:
    lines = [f"{_sha256(path)}  {path.name}" for path in sorted(files, key=lambda item: item.name)]
    (destination / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_release(source: Path, destination: Path, fraction: float = DEFAULT_FRACTION, seed: str = DEFAULT_SEED, replace: bool = False) -> dict[str, Any]:
    if not 0.10 <= fraction <= 0.20:
        raise ValueError("fraction must be between 0.10 and 0.20 inclusive")
    if destination.exists() and any(destination.iterdir()) and not replace:
        raise FileExistsError(f"Destination must be empty: {destination}")
    source = source.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for filename in (*CORE_FILES, *TASK_FILES):
        if not (source / filename).is_file():
            raise FileNotFoundError(f"Missing R2 manifest: {source / filename}")
    vqa = _read_jsonl(source / "vqa.jsonl")
    vgrounding = _read_jsonl(source / "vgrounding.jsonl")
    canonical = _read_jsonl(source / "canonical.jsonl")
    paired = _read_jsonl(source / "paired_evidence.jsonl")
    canonical_images = {str(row["image"]) for row in canonical if row.get("image")}
    forced = _coverage_seed(vqa, vgrounding, canonical_images, seed)
    selected = {
        "vqa": _sample_rows(vqa, math.ceil(len(vqa) * fraction), forced["vqa"], seed),
        "vgrounding": _sample_rows(vgrounding, math.ceil(len(vgrounding) * fraction), forced["vgrounding"], seed),
    }
    selected_images = {image for rows in selected.values() for row in rows for image in _images(row)}
    if selected_images != canonical_images:
        raise AssertionError("R3 task rows do not cover exactly the canonical image set.")
    for filename in CORE_FILES:
        shutil.copy2(source / filename, destination / filename)
    counts = {name: _write_jsonl(destination / f"{name}.jsonl", rows) for name, rows in selected.items()}
    multitask = _multitask_rows(selected)
    counts["multitask"] = _write_jsonl(destination / MULTITASK_FILE, multitask)
    metadata = _metadata_rows(canonical, paired)
    counts["metadata_vqa"] = _write_jsonl(destination / METADATA_FILE, metadata)
    source_hashes = {filename: _sha256(source / filename) for filename in (*CORE_FILES, *TASK_FILES)}
    summary = {
        "release": "wbbs-r3",
        "derived_from": str(source),
        "source_release": "r2",
        "deterministic": True,
        "sampling": {"unit": "task_row", "fraction": fraction, "seed": seed, "coverage_policy": "every_canonical_image_in_at_least_one_task_row"},
        "source_task_rows": {"vqa": len(vqa), "vgrounding": len(vgrounding)},
        "exported_task_rows": counts,
        "forced_coverage_rows": {name: len(task_ids) for name, task_ids in forced.items()},
        "canonical_image_coverage": {"required": len(canonical_images), "selected": len(selected_images), "missing": 0},
        "source_distributions": {name: _distribution(rows) for name, rows in {"vqa": vqa, "vgrounding": vgrounding}.items()},
        "exported_distributions": {name: _distribution(rows) for name, rows in selected.items()},
        "schema": {
            "primary_task_manifest": MULTITASK_FILE,
            "compatibility_task_manifests": list(TASK_FILES),
            "supplementary_metadata_manifest": METADATA_FILE,
            "multitask_schema_version": "r3-multitask-v1",
            "target_schema": "r2-target-v1",
            "image_reference_root": "bs80k",
        },
        "source_manifest_sha256": source_hashes,
    }
    for filename in ("release_summary.json", "r3_sampling_report.json"):
        (destination / filename).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = "\n".join(
        [
            "# WBBS R3 compact release",
            "",
            "R3 is a deterministic task-level compact release derived from frozen R2.",
            "It retains all canonical evidence, paired evidence, metastasis provenance, and patient splits.",
            "Every canonical image appears in at least one selected task row.",
            "`multitask.jsonl` is the primary consumer manifest; `vqa.jsonl` and `vgrounding.jsonl` are compatibility projections.",
            "",
            f"- Sampling fraction: {fraction:.1%}",
            f"- VQA rows: {counts['vqa']:,} / {len(vqa):,}",
            f"- Visual-grounding rows: {counts['vgrounding']:,} / {len(vgrounding):,}",
            f"- Unified multitask rows: {counts['multitask']:,}",
            f"- Supplementary view-metadata rows: {counts['metadata_vqa']:,} (not training/benchmark eligible)",
            f"- Canonical image coverage: {len(selected_images):,} / {len(canonical_images):,}",
            f"- Seed: `{seed}`",
            "",
            "R3 does not change source labels, boxes, evidence IDs, image references, or patient splits.",
        ]
    )
    (destination / "R3.0_RELEASE.md").write_text(manifest + "\n", encoding="utf-8")
    payload = [destination / filename for filename in (*CORE_FILES, *TASK_FILES, MULTITASK_FILE, METADATA_FILE, "release_summary.json", "r3_sampling_report.json")]
    _write_checksums(destination, payload)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-release", type=Path, default=Path("datasets/releases/r2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--replace", action="store_true", help="Replace R3 manifests and release metadata in an existing output directory.")
    args = parser.parse_args()
    print(json.dumps(build_release(args.input_release, args.output_dir, args.fraction, args.seed, args.replace), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
