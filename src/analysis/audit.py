"""Non-destructive audits and baselines for the local bone-scan datasets."""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import io
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from project_paths import SOURCES
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError


LIBS_SPLITS = (("train", "train"), ("test", "train"), ("valid", "valid"))
TOKEN_RE = re.compile(r"[a-z]+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)
LONG_NUMBER_RE = re.compile(r"\b\d{6,}\b")


@dataclass(frozen=True)
class WholeRecord:
    identifier: str
    source_path: str
    view: str
    label: str
    pair_id: str
    file_bytes: int
    width: int
    height: int
    aspect_ratio: float
    mean: float
    std: float
    entropy: float
    histogram: tuple[float, ...]
    a_hash: str
    sha256: str
    quality_flags: tuple[str, ...]
    thumbnail: tuple[int, ...]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = list(values)
    if not materialized:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None, "std": None}
    return {
        "count": len(materialized),
        "min": round(min(materialized), 4),
        "median": round(statistics.median(materialized), 4),
        "mean": round(statistics.fmean(materialized), 4),
        "max": round(max(materialized), 4),
        "std": round(statistics.pstdev(materialized), 4),
    }


def _top_counts(counter: Counter[Any], limit: int = 20) -> list[dict[str, Any]]:
    return [{"value": str(value), "count": count} for value, count in counter.most_common(limit)]


def _normalise_text(text: str) -> str:
    return " ".join(text.lower().split())


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_metrics(image: Image.Image) -> dict[str, Any]:
    source_mode = image.mode
    grayscale = image.convert("L")
    width, height = grayscale.size
    histogram = grayscale.histogram()
    total = sum(histogram)
    mean = sum(value * count for value, count in enumerate(histogram)) / total
    variance = sum(((value - mean) ** 2) * count for value, count in enumerate(histogram)) / total
    entropy = -sum((count / total) * math.log2(count / total) for count in histogram if count)
    histogram_16 = tuple(round(sum(histogram[start : start + 16]) / total, 6) for start in range(0, 256, 16))
    thumbnail_image = grayscale.resize((16, 16))
    pixels = (
        thumbnail_image.get_flattened_data()
        if hasattr(thumbnail_image, "get_flattened_data")
        else thumbnail_image.getdata()
    )
    thumbnail = tuple(int(pixel) for pixel in pixels)
    threshold = statistics.fmean(thumbnail)
    bits = "".join("1" if pixel >= threshold else "0" for pixel in thumbnail)
    flags: list[str] = []
    if min(width, height) < 64:
        flags.append("small_dimension")
    if math.sqrt(variance) < 8:
        flags.append("low_contrast")
    if mean < 8:
        flags.append("very_dark")
    return {
        "mode": source_mode,
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 6),
        "mean": round(mean, 6),
        "std": round(math.sqrt(variance), 6),
        "entropy": round(entropy, 6),
        "histogram": histogram_16,
        "a_hash": f"{int(bits, 2):064x}",
        "quality_flags": tuple(flags),
        "thumbnail": thumbnail,
    }


def _decode_sample(payloads: Iterable[str]) -> dict[str, Any]:
    widths: list[float] = []
    heights: list[float] = []
    modes: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    errors = 0
    for payload in payloads:
        try:
            raw = base64.b64decode(payload, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                metrics = _image_metrics(image)
                widths.append(metrics["width"])
                heights.append(metrics["height"])
                modes[metrics["mode"]] += 1
                formats[str(image.format)] += 1
        except (binascii.Error, UnidentifiedImageError, OSError, ValueError):
            errors += 1
    return {
        "sample_size": len(widths) + errors,
        "decoded": len(widths),
        "decode_errors": errors,
        "width": _summary(widths),
        "height": _summary(heights),
        "modes": dict(modes),
        "formats": dict(formats),
    }


def _read_libs_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != {"text_id", "text", "image_ids"}:
            raise ValueError(f"Unexpected LIBS schema at {path}:{line_number}")
        if not isinstance(row["text_id"], int) or not isinstance(row["text"], str):
            raise ValueError(f"Invalid LIBS text row at {path}:{line_number}")
        if not isinstance(row["image_ids"], list) or not all(isinstance(item, int) for item in row["image_ids"]):
            raise ValueError(f"Invalid LIBS image_ids at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _scan_libs_tsv(path: Path, sample_size: int, seed: int) -> tuple[dict[str, Any], set[int], list[str]]:
    rng = random.Random(seed)
    ids: set[int] = set()
    samples: list[str] = []
    payload_lengths: list[float] = []
    malformed_rows = 0
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            row_count += 1
            if len(row) != 2 or not row[0].isdigit():
                malformed_rows += 1
                continue
            image_id, payload = int(row[0]), row[1]
            ids.add(image_id)
            payload_lengths.append(len(payload))
            if len(samples) < sample_size:
                samples.append(payload)
            else:
                replacement = rng.randrange(row_count)
                if replacement < sample_size:
                    samples[replacement] = payload
    return (
        {
            "tsv_rows": row_count,
            "unique_tsv_ids": len(ids),
            "malformed_rows": malformed_rows,
            "payload_characters": _summary(payload_lengths),
        },
        ids,
        samples,
    )


def _payload_signatures(path: Path, selected_ids: set[int]) -> dict[int, tuple[tuple[int, int], str]]:
    signatures: dict[int, tuple[tuple[int, int], str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 2 or not row[0].isdigit() or int(row[0]) not in selected_ids:
                continue
            image_id = int(row[0])
            raw = base64.b64decode(row[1], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                metrics = _image_metrics(image)
                signatures[image_id] = ((metrics["width"], metrics["height"]), hashlib.sha256(bytes(metrics["thumbnail"])).hexdigest())
            if len(signatures) == len(selected_ids):
                break
    return signatures


def _audit_libs(raw_root: Path, analysis_root: Path, sample_size: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_root = raw_root / "LIBS-160K-EN"
    split_data: dict[str, dict[str, Any]] = {}
    label_rows: list[dict[str, Any]] = []
    split_ids: dict[str, set[int]] = {}
    split_tsvs: dict[str, Path] = {}
    for split_name, folder in LIBS_SPLITS:
        text_path = dataset_root / folder / f"{split_name}_texts.jsonl"
        tsv_path = dataset_root / folder / f"{split_name}_imgs.tsv"
        rows = _read_libs_rows(text_path)
        reference_ids = [image_id for row in rows for image_id in row["image_ids"]]
        tsv_summary, tsv_ids, samples = _scan_libs_tsv(tsv_path, sample_size, seed + len(split_data))
        image_counts = [len(row["image_ids"]) for row in rows]
        text_lengths = [len(row["text"]) for row in rows]
        split_data[split_name] = {
            "text_rows": len(rows),
            "text_characters": _summary(text_lengths),
            "images_per_text": _summary(image_counts),
            "referenced_assignments": len(reference_ids),
            "unique_referenced_ids": len(set(reference_ids)),
            "duplicate_assignments": len(reference_ids) - len(set(reference_ids)),
            "referenced_missing_from_tsv": len(set(reference_ids) - tsv_ids),
            "tsv_ids_unreferenced": len(tsv_ids - set(reference_ids)),
            "sampled_image_audit": _decode_sample(samples),
            **tsv_summary,
        }
        split_ids[split_name] = tsv_ids
        split_tsvs[split_name] = tsv_path
        for row in rows:
            label_rows.append(
                {
                    "split": split_name,
                    "text_id": row["text_id"],
                    "text": row["text"],
                    "normalized_text": _normalise_text(row["text"]),
                    "token_count": len(_tokens(row["text"])),
                    "image_count": len(row["image_ids"]),
                }
            )

    train_rows = _read_libs_rows(dataset_root / "train" / "train_texts.jsonl")
    image_to_labels: dict[int, list[int]] = defaultdict(list)
    for row in train_rows:
        for image_id in row["image_ids"]:
            image_to_labels[image_id].append(row["text_id"])
    cardinality = Counter(len(labels) for labels in image_to_labels.values())
    label_pairs = Counter(tuple(sorted(labels)) for labels in image_to_labels.values())
    train_labels = [row for row in label_rows if row["split"] == "train"]
    normalised = Counter(row["normalized_text"] for row in train_labels)
    tokens = Counter(token for row in train_labels for token in _tokens(row["text"]))
    pii_flags = {
        "email_like_strings": sum(bool(EMAIL_RE.search(row["text"])) for row in train_labels),
        "long_numeric_strings": sum(bool(LONG_NUMBER_RE.search(row["text"])) for row in train_labels),
    }
    cross_split: list[dict[str, Any]] = []
    pairs = (("train", "test"), ("train", "valid"), ("test", "valid"))
    rng = random.Random(seed)
    for left, right in pairs:
        shared = split_ids[left] & split_ids[right]
        selected = set(rng.sample(sorted(shared), min(64, len(shared))))
        if selected:
            left_signatures = _payload_signatures(split_tsvs[left], selected)
            right_signatures = _payload_signatures(split_tsvs[right], selected)
            same_dimensions = sum(left_signatures[item][0] == right_signatures[item][0] for item in selected)
            same_thumbnail = sum(left_signatures[item][1] == right_signatures[item][1] for item in selected)
        else:
            same_dimensions = same_thumbnail = 0
        cross_split.append(
            {
                "left": left,
                "right": right,
                "shared_numeric_ids": len(shared),
                "pixel_audit_sample": len(selected),
                "same_dimensions": same_dimensions,
                "same_16x16_pixel_signatures": same_thumbnail,
            }
        )
    _write_csv(
        analysis_root / "libs_text_labels.csv",
        label_rows,
        ["split", "text_id", "text", "normalized_text", "token_count", "image_count"],
    )
    return (
        {
            "splits": split_data,
            "text": {
                "unique_train_labels": len(train_labels),
                "unique_normalized_train_texts": len(normalised),
                "duplicate_normalized_train_text_groups": sum(count > 1 for count in normalised.values()),
                "top_tokens": _top_counts(tokens),
                "potential_pii_heuristics": pii_flags,
            },
            "multi_label_structure": {
                "image_label_cardinality": {str(key): value for key, value in sorted(cardinality.items())},
                "distinct_label_pairs": len(label_pairs),
                "label_pair_frequency": _summary(label_pairs.values()),
            },
            "cross_split_numeric_id_audit": cross_split,
        },
        train_labels,
    )


def _whole_record(path: Path, view: str, label: str) -> WholeRecord:
    with Image.open(path) as image:
        image.load()
        metrics = _image_metrics(image)
    return WholeRecord(
        identifier=f"{view}/{label}/{path.name}",
        source_path=str(path.resolve()),
        view=view,
        label=label,
        pair_id=path.stem,
        file_bytes=path.stat().st_size,
        width=metrics["width"],
        height=metrics["height"],
        aspect_ratio=metrics["aspect_ratio"],
        mean=metrics["mean"],
        std=metrics["std"],
        entropy=metrics["entropy"],
        histogram=metrics["histogram"],
        a_hash=metrics["a_hash"],
        sha256=_file_sha256(path),
        quality_flags=metrics["quality_flags"],
        thumbnail=metrics["thumbnail"],
    )


def _audit_whole(raw_root: Path, analysis_root: Path, manifests_root: Path) -> tuple[dict[str, Any], list[WholeRecord]]:
    all_records: list[WholeRecord] = []
    unreadable: list[str] = []
    by_view_label: dict[str, dict[str, list[WholeRecord]]] = defaultdict(lambda: defaultdict(list))
    for view in ("wholeANT", "wholePOST"):
        for label in ("Normal", "Abnormal"):
            for path in sorted((raw_root / view / label).glob("*.jpg"), key=lambda item: item.name):
                try:
                    record = _whole_record(path, view.removeprefix("whole"), label)
                    all_records.append(record)
                    by_view_label[view][label].append(record)
                except (UnidentifiedImageError, OSError, ValueError):
                    unreadable.append(str(path))
    manifest_rows = {
        "wholeANT": [],
        "wholePOST": [],
    }
    feature_rows: list[dict[str, Any]] = []
    exact_hashes: dict[str, list[str]] = defaultdict(list)
    perceptual_hashes: dict[str, list[str]] = defaultdict(list)
    for record in all_records:
        source = f"whole{record.view}"
        manifest_rows[source].append(
            {
                "id": record.identifier,
                "image": record.source_path,
                "label": record.label,
                "view": record.view,
                "group_id": record.pair_id,
            }
        )
        feature_rows.append(
            {
                "id": record.identifier,
                "view": record.view,
                "label": record.label,
                "group_id": record.pair_id,
                "file_bytes": record.file_bytes,
                "width": record.width,
                "height": record.height,
                "aspect_ratio": record.aspect_ratio,
                "mean": record.mean,
                "std": record.std,
                "entropy": record.entropy,
                "a_hash": record.a_hash,
                "sha256": record.sha256,
                "quality_flags": ";".join(record.quality_flags),
            }
        )
        exact_hashes[record.sha256].append(record.identifier)
        perceptual_hashes[record.a_hash].append(record.identifier)
    for source, rows in manifest_rows.items():
        _write_jsonl(manifests_root / f"{source}.jsonl", rows)
    _write_csv(
        analysis_root / "whole_image_features.csv",
        feature_rows,
        [
            "id", "view", "label", "group_id", "file_bytes", "width", "height", "aspect_ratio",
            "mean", "std", "entropy", "a_hash", "sha256", "quality_flags",
        ],
    )
    view_summary: dict[str, Any] = {}
    for view, labels in by_view_label.items():
        view_summary[view] = {}
        for label, records in labels.items():
            view_summary[view][label] = {
                "images": len(records),
                "width": _summary(record.width for record in records),
                "height": _summary(record.height for record in records),
                "aspect_ratio": _summary(record.aspect_ratio for record in records),
                "mean_intensity": _summary(record.mean for record in records),
                "contrast": _summary(record.std for record in records),
                "entropy": _summary(record.entropy for record in records),
                "quality_flag_counts": dict(Counter(flag for record in records for flag in record.quality_flags)),
            }
    ant = {record.pair_id: record for records in by_view_label["wholeANT"].values() for record in records}
    post = {record.pair_id: record for records in by_view_label["wholePOST"].values() for record in records}
    overlap = set(ant) & set(post)
    matrix = Counter((ant[pair].label, post[pair].label) for pair in overlap)
    exact_groups = {digest: ids for digest, ids in exact_hashes.items() if len(ids) > 1}
    perceptual_groups = {digest: ids for digest, ids in perceptual_hashes.items() if len(ids) > 1}
    _write_json(analysis_root / "whole_duplicate_groups.json", {"exact": exact_groups, "a_hash": perceptual_groups})
    return (
        {
            "views": view_summary,
            "unreadable_files": unreadable,
            "pair_audit": {
                "matched_filenames": len(overlap),
                "ant_only": len(set(ant) - set(post)),
                "post_only": len(set(post) - set(ant)),
                "label_matrix": {f"ANT_{left}__POST_{right}": count for (left, right), count in sorted(matrix.items())},
            },
            "duplicates": {
                "exact_duplicate_groups": len(exact_groups),
                "a_hash_duplicate_groups": len(perceptual_groups),
            },
        },
        all_records,
    )


def _cluster_rows(records: list[WholeRecord], coordinates: Any, labels: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": record.identifier,
            "view": record.view,
            "label": record.label,
            "group_id": record.pair_id,
            "x": round(float(coordinates[index, 0]), 6),
            "y": round(float(coordinates[index, 1]), 6),
            "cluster": int(labels[index]),
        }
        for index, record in enumerate(records)
    ]


def _cluster_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    clusters: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        clusters[row["cluster"]][f"{row['view']}_{row['label']}"] += 1
    return {str(cluster): {"count": sum(counts.values()), "composition": dict(counts)} for cluster, counts in sorted(clusters.items())}


def _run_unsupervised(
    records: list[WholeRecord],
    text_rows: list[dict[str, Any]],
    output_root: Path,
    sample_size: int,
    cluster_count: int,
    seed: int,
) -> dict[str, Any]:
    try:
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA, TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError:
        return {
            "status": "skipped",
            "reason": "Install the analysis extra: uv sync --extra analysis",
            "variants": {},
        }
    rng = random.Random(seed)
    sample = records if len(records) <= sample_size else rng.sample(records, sample_size)
    sample.sort(key=lambda record: record.identifier)
    n_clusters = max(2, min(cluster_count, len(sample)))
    output_dir = output_root / "analysis" / "unsupervised"
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"status": "complete", "sample_size": len(sample), "variants": {}}
    matrices = {
        "engineered": np.asarray(
            [
                [record.width, record.height, record.aspect_ratio, record.mean, record.std, record.entropy, *record.histogram]
                for record in sample
            ],
            dtype=np.float32,
        ),
        "pixels_16x16": np.asarray([record.thumbnail for record in sample], dtype=np.float32),
    }
    for name, matrix in matrices.items():
        scaled = StandardScaler().fit_transform(matrix)
        coordinates = PCA(n_components=2, random_state=seed).fit_transform(scaled)
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(scaled)
        rows = _cluster_rows(sample, coordinates, labels)
        _write_csv(
            output_dir / f"whole_{name}_clusters.csv",
            rows,
            ["id", "view", "label", "group_id", "x", "y", "cluster"],
        )
        result["variants"][name] = {"method": "StandardScaler + PCA(2) + KMeans", "clusters": _cluster_summary(rows)}

    texts = [row["normalized_text"] for row in text_rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
    tfidf = vectorizer.fit_transform(texts)
    text_coordinates = TruncatedSVD(n_components=2, random_state=seed).fit_transform(tfidf)
    text_labels = KMeans(n_clusters=max(2, min(cluster_count, len(text_rows))), n_init=10, random_state=seed).fit_predict(tfidf)
    text_cluster_rows = [
        {
            "text_id": text_rows[index]["text_id"],
            "text": text_rows[index]["text"],
            "x": round(float(text_coordinates[index, 0]), 6),
            "y": round(float(text_coordinates[index, 1]), 6),
            "cluster": int(text_labels[index]),
        }
        for index in range(len(text_rows))
    ]
    _write_csv(
        output_dir / "libs_text_tfidf_clusters.csv",
        text_cluster_rows,
        ["text_id", "text", "x", "y", "cluster"],
    )
    result["variants"]["text_tfidf"] = {
        "method": "TF-IDF(1,2) + TruncatedSVD(2) + KMeans",
        "clusters": _cluster_summary({**row, "view": "text", "label": "label"} for row in text_cluster_rows),
    }
    optional: dict[str, Any] = {}
    try:
        import umap

        embedding = umap.UMAP(n_components=2, random_state=seed).fit_transform(matrices["engineered"])
        umap_rows = _cluster_rows(sample, embedding, [0] * len(sample))
        _write_csv(output_dir / "whole_engineered_umap.csv", umap_rows, ["id", "view", "label", "group_id", "x", "y", "cluster"])
        optional["umap"] = "complete"
    except ModuleNotFoundError:
        optional["umap"] = "not installed"
    try:
        import hdbscan

        hdbscan_labels = hdbscan.HDBSCAN(min_cluster_size=25).fit_predict(matrices["engineered"])
        optional["hdbscan_clusters"] = {str(key): value for key, value in sorted(Counter(int(label) for label in hdbscan_labels).items())}
    except ModuleNotFoundError:
        optional["hdbscan"] = "not installed"
    result["optional_variants"] = optional
    return result


def _render_report(summary: dict[str, Any]) -> str:
    libs = summary["libs"]
    whole = summary["whole"]
    ant = whole["views"]["wholeANT"]
    post = whole["views"]["wholePOST"]
    lines = [
        "# Bone-scan dataset preparation audit",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Scope",
        "",
        "Raw inputs were read only. No decoded or resized images were written by this run.",
        "",
        "## LIBS-160K-EN",
        "",
        f"- Train / test / valid TSV rows: {libs['splits']['train']['tsv_rows']:,} / {libs['splits']['test']['tsv_rows']:,} / {libs['splits']['valid']['tsv_rows']:,}.",
        f"- Train contains {libs['text']['unique_train_labels']} text labels and {libs['multi_label_structure']['distinct_label_pairs']} observed label pairs.",
        f"- Per-image label cardinality: {libs['multi_label_structure']['image_label_cardinality']}.",
        "- Repeated numeric IDs across files were audited using sampled decoded pixels; treat IDs as split-local keys unless provenance says otherwise.",
        "",
        "## Whole-body scans",
        "",
        f"- ANT: Normal {ant['Normal']['images']:,}, Abnormal {ant['Abnormal']['images']:,}.",
        f"- POST: Normal {post['Normal']['images']:,}, Abnormal {post['Abnormal']['images']:,}.",
        f"- Matched ANT/POST filenames: {whole['pair_audit']['matched_filenames']:,}; keep each filename as one group during any train/validation/test split.",
        f"- Unreadable files: {len(whole['unreadable_files'])}.",
        "",
        "## Unsupervised baselines",
        "",
        f"- Status: {summary['unsupervised']['status']}.",
        "- Outputs use engineered grayscale features, 16x16 pixel thumbnails, and TF-IDF text features. They are exploratory diagnostics, not clinical labels.",
        "",
        "## Next decision",
        "",
        "Review the report and manifests before producing training JSONL. Use whole-body scans first for binary VQA; do not represent LIBS's two labels as a single clinical report without an explicit target policy.",
    ]
    return "\n".join(lines) + "\n"


def run_preparation(
    raw_root: Path,
    output_root: Path,
    libs_image_sample_size: int = 512,
    cluster_sample_size: int = 3000,
    cluster_count: int = 6,
    seed: int = 4050,
    skip_unsupervised: bool = False,
) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    output_root = output_root.resolve()
    if not (raw_root / "LIBS-160K-EN").is_dir() or not (raw_root / "wholeANT").is_dir() or not (raw_root / "wholePOST").is_dir():
        raise FileNotFoundError("Raw root must contain LIBS-160K-EN, wholeANT, and wholePOST")
    analysis_root = output_root / "analysis"
    manifests_root = output_root / "manifests"
    reports_root = output_root / "reports"
    for directory in (analysis_root, manifests_root, reports_root):
        directory.mkdir(parents=True, exist_ok=True)
    libs_summary, train_text_rows = _audit_libs(raw_root, analysis_root, libs_image_sample_size, seed)
    whole_summary, whole_records = _audit_whole(raw_root, analysis_root, manifests_root)
    unsupervised = {"status": "skipped", "reason": "Disabled with --skip-unsupervised", "variants": {}}
    if not skip_unsupervised:
        unsupervised = _run_unsupervised(whole_records, train_text_rows, output_root, cluster_sample_size, cluster_count, seed)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "libs": libs_summary,
        "whole": whole_summary,
        "unsupervised": unsupervised,
    }
    _write_json(analysis_root / "summary.json", summary)
    (reports_root / "dataset_audit.md").write_text(_render_report(summary), encoding="utf-8")
    (output_root / "README.md").write_text(
        "# Derived dataset artifacts\n\nGenerated by `wbbs-audit`. Raw data remains outside this directory.\n",
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create non-destructive LIBS/whole-body dataset audits.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=SOURCES / "libs160k-imaging-raw",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SOURCES / "libs160k-imaging-preprocessed",
    )
    parser.add_argument("--libs-image-sample-size", type=int, default=512)
    parser.add_argument("--cluster-sample-size", type=int, default=3000)
    parser.add_argument("--cluster-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=4050)
    parser.add_argument("--skip-unsupervised", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.libs_image_sample_size < 1 or args.cluster_sample_size < 2 or args.cluster_count < 2:
        raise ValueError("Sample sizes and cluster count must be positive; cluster values must be at least 2")
    summary = run_preparation(
        raw_root=args.raw_root,
        output_root=args.output_root,
        libs_image_sample_size=args.libs_image_sample_size,
        cluster_sample_size=args.cluster_sample_size,
        cluster_count=args.cluster_count,
        seed=args.seed,
        skip_unsupervised=args.skip_unsupervised,
    )
    print(f"Prepared analysis at {summary['output_root']}")


if __name__ == "__main__":
    main()
