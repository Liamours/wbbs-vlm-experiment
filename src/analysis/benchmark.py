"""Read-only benchmark audit for VQA and visual-grounding releases.

The auditor intentionally uses deterministic, dependency-free metrics so that a
release can be profiled before training. It measures both linguistic structure
and annotation coverage; it does not score a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Iterable

from preprocess.canonical import read_jsonl
from project_paths import AUDITS


TASK_FILES: tuple[tuple[str, str], ...] = (
    ("vgrounding", "vgrounding.jsonl"),
    ("vqa", "vqa.jsonl"),
    ("grounding", "grounding.jsonl"),
    ("grounded_vqa", "grounded_vqa.jsonl"),
    ("lesion_vqa", "lesion_vqa.jsonl"),
    ("lesion_grounding", "lesion_grounding.jsonl"),
    ("lesion_grounded_vqa", "lesion_grounded_vqa.jsonl"),
)
STANDARD_SCHEMA_VERSION = "1.0"
STANDARD_TASKS = {"vqa", "grounding", "grounded_vqa", "other"}
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*|\d+(?:\.\d+)?|[\u3400-\u4dbf\u4e00-\u9fff]+")
SPACE_RE = re.compile(r"\s+")
QUESTION_KEYS = ("question", "query")
VIEW_WORDS = {
    "ant": "<VIEW>",
    "post": "<VIEW>",
    "anterior": "<VIEW>",
    "posterior": "<VIEW>",
    "neutral": "<VIEW>",
}
CLASS_WORDS = {
    "abnormal": "<CLASS>",
    "normal": "<CLASS>",
    "metastasis": "<CLASS>",
    "metastatic": "<CLASS>",
    "non-metastatic": "<CLASS>",
    "nonmetastatic": "<CLASS>",
}


def _normalise(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = "".join(character if character.isalnum() or character.isspace() or character in "<>-'" else " " for character in text)
    return SPACE_RE.sub(" ", text).strip()


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _language_hint(text: str) -> str:
    has_cjk = any("\u3400" <= character <= "\u9fff" for character in text)
    has_latin = any(character.isascii() and character.isalpha() for character in text)
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "zh"
    if has_latin:
        return "en"
    return "other"


def _text(row: dict[str, Any]) -> str:
    for key in QUESTION_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# ---------------------------------------------------------------------------
# Dataset comparison interchange format
# ---------------------------------------------------------------------------
#
# External medical VQA/grounding datasets use several incompatible JSON
# layouts.  The helpers below intentionally keep the interchange format small
# and loss-aware: values that cannot be inferred are represented by ``None``
# ("not reported"), while the source row index and dataset name are retained
# for traceability.  A standardised JSONL file can then be audited with
# ``benchmark_standard`` using exactly the same metrics as a WBBS release.

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "record_id": ("record_id", "id", "qid", "question_id", "uid", "task_id"),
    "image_id": ("image_id", "image", "image_name", "image_path", "filename", "file_name"),
    "patient_id": ("patient_id", "subject_id", "patient", "subject"),
    "image_size": ("image_size", "size", "dimensions", "resolution"),
    "split": ("split", "partition", "subset", "set", "phase"),
    "question": ("question", "query", "prompt", "caption", "text"),
    "answer": ("answer", "answers", "gt_answer", "ground_truth", "answer_label", "label", "response"),
    "task": ("task", "task_type", "task_name", "type"),
    "region": ("region", "anatomy", "anatomical_region", "organ", "body_part", "label"),
    "view": ("view", "projection", "orientation"),
    "target_granularity": ("target_granularity", "granularity", "target_kind", "kind"),
}


def _first_field(row: dict[str, Any], field: str, field_map: dict[str, str | list[str]] | None = None) -> Any:
    """Return the first non-empty value for a standard field.

    ``field_map`` is deliberately simple JSON-compatible configuration: each
    standard field maps to one source key or a list of source keys.  Alias
    lookup remains available when a mapping is omitted.
    """

    keys: tuple[str, ...]
    if field_map and field in field_map:
        mapped = field_map[field]
        keys = (mapped,) if isinstance(mapped, str) else tuple(mapped)
    else:
        keys = _FIELD_ALIASES.get(field, (field,))
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _as_text(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def _as_answers(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [text for item in value if (text := _as_text(item))]
    if (text := _as_text(value)):
        return [text]
    return []


def _image_value(row: dict[str, Any], field_map: dict[str, str | list[str]] | None = None) -> str | None:
    value = _first_field(row, "image_id", field_map)
    if isinstance(value, dict):
        value = value.get("id") or value.get("image") or value.get("path") or value.get("file_name")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None and isinstance(row.get("images"), list) and row["images"]:
        image = row["images"][0]
        if isinstance(image, dict):
            value = image.get("image_id") or image.get("id") or image.get("image") or image.get("path")
        else:
            value = image
    return _as_text(value)


def _image_size_value(row: dict[str, Any], field_map: dict[str, str | list[str]] | None = None) -> list[float] | None:
    value = _first_field(row, "image_size", field_map)
    if isinstance(value, dict):
        value = [value.get("width", value.get("w")), value.get("height", value.get("h"))]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        width, height = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return [width, height]


def _coerce_bbox(value: Any, bbox_format: str = "xyxy") -> list[float] | None:
    """Convert common bbox representations to ``[x1, y1, x2, y2]``."""

    if isinstance(value, dict):
        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            value = [value["x1"], value["y1"], value["x2"], value["y2"]]
        elif all(key in value for key in ("left", "top", "right", "bottom")):
            value = [value["left"], value["top"], value["right"], value["bottom"]]
        elif all(key in value for key in ("x", "y", "width", "height")):
            value = [value["x"], value["y"], value["width"], value["height"]]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        numbers = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if bbox_format.lower() in {"xywh", "coco", "xywh_abs"}:
        numbers[2] += numbers[0]
        numbers[3] += numbers[1]
    return numbers


def _target_boxes(row: dict[str, Any], bbox_format: str = "xyxy") -> list[dict[str, Any]]:
    """Extract boxes from WBBS, COCO-like, and generic annotation layouts."""

    targets: list[Any] = []
    for key in ("target_boxes", "targets", "evidence_targets", "boxes", "annotations"):
        value = row.get(key)
        if isinstance(value, list):
            targets.extend(value)
    if isinstance(row.get("bbox"), (list, tuple, dict)):
        targets.append({"bbox": row["bbox"], "label": row.get("region") or row.get("label")})
    if isinstance(row.get("evidence_bbox"), (list, tuple, dict)):
        targets.append({"bbox": row["evidence_bbox"], "label": row.get("region") or row.get("label")})

    result: list[dict[str, Any]] = []
    for target in targets:
        if isinstance(target, dict):
            bbox = target.get("bbox") or target.get("box") or target.get("bounding_box")
            target_image = target.get("image_id") or target.get("image") or target.get("file_name")
            label = target.get("label") or target.get("name") or target.get("category") or target.get("region")
        else:
            bbox, target_image, label = target, None, None
        converted = _coerce_bbox(bbox, bbox_format)
        if converted is None:
            continue
        item: dict[str, Any] = {"bbox": converted}
        if (text := _as_text(target_image)):
            item["image_id"] = text
        if (text := _as_text(label)):
            item["label"] = text
        if isinstance(target, dict) and (view := _as_text(target.get("view"))):
            item["view"] = view
        result.append(item)
    return result


def _infer_task(row: dict[str, Any], task_hint: str | None, boxes: list[dict[str, Any]]) -> str:
    raw = _as_text(_first_field(row, "task"))
    value = (task_hint or raw or "").lower().replace("-", "_").replace(" ", "_")
    if value:
        if "ground" in value and ("vqa" in value or "qa" in value):
            return "grounded_vqa"
        if "ground" in value or "local" in value:
            return "grounding"
        if "vqa" in value or value in {"qa", "question_answering"}:
            return "vqa"
    question = _text(row)
    answers = _as_answers(_first_field(row, "answer"))
    if boxes and question and answers:
        return "grounded_vqa"
    if boxes:
        return "grounding"
    if question or answers:
        return "vqa"
    return "other"


def _infer_granularity(row: dict[str, Any], boxes: list[dict[str, Any]], region: str | None) -> str | None:
    value = _as_text(_first_field(row, "target_granularity"))
    # A region label on a VQA row describes the question context, not a
    # grounding target.  Only report inferred granularity when boxes exist;
    # explicit source metadata is retained even for box-less rows.
    if not value and not boxes:
        return None
    box_labels = " ".join(str(box.get("label", "")) for box in boxes if box.get("label"))
    text = (value or box_labels or region or "").lower().replace("_", " ").replace("-", " ")
    if any(token in text for token in ("whole body", "wholebody", "entire body")):
        return "whole_body"
    if any(token in text for token in ("lesion", "metast", "hotspot", "focus")):
        return "lesion_or_metastasis"
    if text or boxes:
        return "anatomical_region"
    return None


def standardize_rows(
    rows: Iterable[dict[str, Any]],
    dataset_name: str = "unknown",
    task_hint: str | None = None,
    field_map: dict[str, str | list[str]] | None = None,
    bbox_format: str = "xyxy",
) -> list[dict[str, Any]]:
    """Normalize arbitrary VQA/grounding rows to the comparison schema.

    The function never guesses missing values as zeros.  Unavailable metadata
    is emitted as ``None`` so comparison reports can display it as NR.
    """

    standard: list[dict[str, Any]] = []
    for index, source_row in enumerate(rows):
        if not isinstance(source_row, dict):
            continue
        boxes = _target_boxes(source_row, bbox_format)
        question = _as_text(_first_field(source_row, "question", field_map))
        answer_values = _as_answers(_first_field(source_row, "answer", field_map))
        region = _as_text(_first_field(source_row, "region", field_map))
        task = _infer_task(source_row, task_hint, boxes)
        record_id = _as_text(_first_field(source_row, "record_id", field_map)) or f"{dataset_name}:{index}"
        image_id = _image_value(source_row, field_map)
        image_size = _image_size_value(source_row, field_map)
        split_value = _as_text(_first_field(source_row, "split", field_map))
        patient_id = _as_text(_first_field(source_row, "patient_id", field_map))
        view = _as_text(_first_field(source_row, "view", field_map))
        for box in boxes:
            box.setdefault("image_id", image_id)
        standard.append(
            {
                "schema_version": STANDARD_SCHEMA_VERSION,
                "dataset": dataset_name,
                "record_id": record_id,
                "split": split_value,
                "task": task,
                "image_id": image_id,
                "image_size": image_size,
                "patient_id": patient_id,
                "question": question,
                "answer": answer_values[0] if answer_values else None,
                "answers": answer_values,
                "answer_type": _question_type(question or "", task),
                "target_boxes": boxes,
                "target_granularity": _infer_granularity(source_row, boxes, region),
                "region": region,
                "view": view,
                "source": {"row_index": index, "dataset": dataset_name},
            }
        )
    return standard


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return read_jsonl(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("data", "questions", "annotations", "items", "records"):
            if isinstance(value.get(key), list):
                return [row for row in value[key] if isinstance(row, dict)]
        split_rows: list[dict[str, Any]] = []
        for key, items in value.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    row = dict(item)
                    row.setdefault("split", key)
                    split_rows.append(row)
        if split_rows:
            return split_rows
        return [value]
    raise ValueError(f"Unsupported dataset JSON root in {path}")


def standardize_dataset(
    input_path: str | Path,
    output_path: str | Path,
    dataset_name: str | None = None,
    task_hint: str | None = None,
    field_map: dict[str, str | list[str]] | None = None,
    bbox_format: str = "xyxy",
) -> dict[str, Any]:
    """Read JSON/JSONL, write standard JSONL, and return a conversion summary."""

    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dataset file not found: {source}")
    name = dataset_name or source.stem
    rows = standardize_rows(_read_rows(source), name, task_hint, field_map, bbox_format)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    tasks = Counter(str(row.get("task") or "other") for row in rows)
    non_null = {
        field: sum(row.get(field) not in (None, "", []) for row in rows)
        for field in ("split", "image_id", "patient_id", "question", "answer", "target_boxes", "target_granularity")
    }
    return {
        "schema_version": STANDARD_SCHEMA_VERSION,
        "dataset": name,
        "rows": len(rows),
        "tasks": dict(sorted(tasks.items())),
        "non_null_fields": non_null,
        "output": str(destination),
    }


def _top(counter: Counter[str], limit: int = 15) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def _entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 1 or len(counter) <= 1:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def _normalised_entropy(counter: Counter[str]) -> float:
    if len(counter) <= 1:
        return 0.0
    return round(_entropy(counter) / math.log2(len(counter)), 6)


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    materialised = list(values)
    if not materialised:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None, "std": None}
    return {
        "count": len(materialised),
        "min": round(min(materialised), 6),
        "median": round(median(materialised), 6),
        "mean": round(fmean(materialised), 6),
        "max": round(max(materialised), 6),
        "std": round(pstdev(materialised), 6),
    }


def _ngrams(tokens: list[str], n: int) -> Iterable[str]:
    return (" ".join(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1)))


def _region_values(rows: Iterable[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for row in rows:
        for key in ("label", "region"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                values.add(value.lower().replace("_", " "))
        target = row.get("target")
        if isinstance(target, dict) and isinstance(target.get("name"), str):
            values.add(target["name"].lower().replace("_", " "))
        for target_box in row.get("target_boxes") or []:
            if isinstance(target_box, dict) and isinstance(target_box.get("label"), str):
                values.add(target_box["label"].lower().replace("_", " "))
    return values


def _mask_skeleton(text: str, regions: set[str]) -> str:
    masked = _normalise(text)
    replacements: list[tuple[str, str]] = []
    replacements.extend((value, "<REGION>") for value in regions if value)
    replacements.extend((value, replacement) for value, replacement in VIEW_WORDS.items())
    replacements.extend((value, replacement) for value, replacement in CLASS_WORDS.items())
    for value, replacement in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        masked = re.sub(rf"\b{re.escape(value)}\b", replacement, masked)
    return SPACE_RE.sub(" ", masked).strip()


def _question_type(text: str, task_name: str) -> str:
    value = _normalise(text)
    if not value:
        return "missing"
    if re.match(r"^(locate|where)\b", value) and re.search(r"\b(determine whether|identify whether|labelled|labeled)\b", value):
        return "localization_plus_classification"
    if task_name.endswith("grounding") or value.startswith(("locate ", "where ")):
        return "localization"
    if re.match(r"^(is|are|does|do|did|can|has|have|was|were)\b", value):
        return "yes_no"
    if re.match(r"^(how many|what number|count)\b", value):
        return "counting"
    if re.match(r"^(which|what type|what kind)\b", value):
        return "classification"
    if re.match(r"^(what|describe|identify)\b", value):
        return "identification_or_description"
    if re.search(r"\b(compare|difference|versus|versus)\b", value):
        return "comparison"
    if re.match(r"^(why|how)\b", value):
        return "reasoning"
    return "other"


def _answer_label(row: dict[str, Any]) -> str:
    for key in ("answer_label", "lesion_class", "source_hotspot_label", "pair_diagnosis"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    answer = row.get("answer")
    if isinstance(answer, str) and answer.strip():
        return _normalise(answer)
    return "missing"


def _bbox_records(row: dict[str, Any]) -> list[tuple[list[Any], list[Any] | None]]:
    records: list[tuple[list[Any], list[Any] | None]] = []
    standard_targets = row.get("target_boxes")
    if isinstance(standard_targets, list):
        image_size = row.get("image_size")
        for target in standard_targets:
            if isinstance(target, dict) and isinstance(target.get("bbox"), list):
                records.append((target["bbox"], image_size if isinstance(image_size, list) else None))
    if isinstance(row.get("images"), list):
        image_sizes = {
            image.get("view"): image.get("image_size")
            for image in row["images"]
            if isinstance(image, dict)
        }
    else:
        image_sizes = {row.get("view"): row.get("image_size")}
    targets = row.get("targets") or row.get("evidence_targets")
    if isinstance(targets, list):
        for target in targets:
            if isinstance(target, dict) and isinstance(target.get("bbox"), list):
                records.append((target["bbox"], image_sizes.get(target.get("view"))))
    elif isinstance(row.get("target"), dict) and isinstance(row["target"].get("locations"), list):
        for target in row["target"]["locations"]:
            if isinstance(target, dict) and isinstance(target.get("bbox"), list):
                records.append((target["bbox"], image_sizes.get(target.get("view"))))
    for key in ("bbox", "evidence_bbox"):
        if isinstance(row.get(key), list):
            records.append((row[key], image_sizes.get(row.get("view"))))
    return records


def _bbox_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    area_fractions: list[float] = []
    invalid = 0
    out_of_bounds = 0
    count = 0
    for row in rows:
        for bbox, image_size in _bbox_records(row):
            count += 1
            if len(bbox) != 4 or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox):
                invalid += 1
                continue
            x1, y1, x2, y2 = [float(value) for value in bbox]
            width, height = x2 - x1, y2 - y1
            if x1 < 0 or y1 < 0 or width <= 0 or height <= 0:
                invalid += 1
                continue
            if isinstance(image_size, list) and len(image_size) == 2:
                image_width, image_height = image_size
                if x2 > image_width or y2 > image_height:
                    out_of_bounds += 1
                if image_width and image_height:
                    area_fractions.append((width * height) / (float(image_width) * float(image_height)))
            widths.append(width)
            heights.append(height)
            areas.append(width * height)
    return {
        "boxes": count,
        "invalid": invalid,
        "out_of_bounds": out_of_bounds,
        "width": _summary(widths),
        "height": _summary(heights),
        "area": _summary(areas),
        "area_fraction": _summary(area_fractions),
    }


def _linguistic_metrics(rows: list[dict[str, Any]], task_name: str, regions: set[str]) -> dict[str, Any]:
    texts = [_text(row) for row in rows]
    normalised = [_normalise(text) for text in texts if text]
    token_lists = [_tokens(text) for text in texts if text]
    all_tokens = [token for tokens in token_lists for token in tokens]
    token_counter = Counter(all_tokens)
    bigrams = Counter(ngram for tokens in token_lists for ngram in _ngrams(tokens, 2))
    trigrams = Counter(ngram for tokens in token_lists for ngram in _ngrams(tokens, 3))
    duplicate_groups = Counter(normalised)
    skeletons = [_mask_skeleton(text, regions) for text in texts if text]
    skeleton_counter = Counter(skeletons)
    templates = Counter(str(row.get("template_id")) for row in rows if row.get("template_id"))
    types = Counter(_question_type(text, task_name) for text in texts)
    languages = Counter(_language_hint(text) for text in texts if text)
    structured_forms = Counter(str(row.get("question_form")) for row in rows if row.get("question_form"))
    region_levels = Counter(str(row.get("region_level")) for row in rows if row.get("region_level"))
    interaction_types = Counter(str(row.get("interaction_type")) for row in rows if row.get("interaction_type"))
    row_unique_words = [len(set(tokens)) for tokens in token_lists]
    row_ttr = [len(set(tokens)) / len(tokens) for tokens in token_lists if tokens]
    return {
        "rows": len(rows),
        "nonempty_text_rows": len(normalised),
        "missing_text_rows": sum(not text for text in texts),
        "token_count": _summary(len(tokens) for tokens in token_lists),
        "unique_words": len(token_counter),
        "type_token_ratio": round(len(token_counter) / len(all_tokens), 6) if all_tokens else 0.0,
        "row_unique_words": _summary(row_unique_words),
        "row_type_token_ratio": _summary(row_ttr),
        "hapax_word_ratio": round(sum(count == 1 for count in token_counter.values()) / len(token_counter), 6) if token_counter else 0.0,
        "unique_bigrams": len(bigrams),
        "unique_trigrams": len(trigrams),
        "top_words": _top(token_counter),
        "top_bigrams": _top(bigrams),
        "top_trigrams": _top(trigrams),
        "exact_text": {
            "unique_normalized": len(duplicate_groups),
            "duplicate_rows": sum(count - 1 for count in duplicate_groups.values()),
            "duplicate_rate": round(1 - len(duplicate_groups) / len(normalised), 6) if normalised else 0.0,
            "duplicate_groups": sum(count > 1 for count in duplicate_groups.values()),
        },
        "masked_skeleton": {
            "unique": len(skeleton_counter),
            "duplicate_rate": round(1 - len(skeleton_counter) / len(skeletons), 6) if skeletons else 0.0,
            "top_share": round(skeleton_counter.most_common(1)[0][1] / len(skeletons), 6) if skeletons else 0.0,
            "entropy_bits": round(_entropy(skeleton_counter), 6),
            "normalized_entropy": _normalised_entropy(skeleton_counter),
            "top_skeletons": _top(skeleton_counter, 10),
        },
        "template_ids": {
            "rows_with_template_id": sum(templates.values()),
            "missing_template_id_rate": round(1 - sum(templates.values()) / len(rows), 6) if rows else 0.0,
            "unique": len(templates),
            "top_share": round(templates.most_common(1)[0][1] / len(rows), 6) if templates and rows else 0.0,
            "normalized_entropy": _normalised_entropy(templates),
            "counts": dict(sorted(templates.items())),
        },
        "question_types": dict(sorted(types.items())),
        "structured_question_forms": dict(sorted(structured_forms.items())),
        "region_levels": dict(sorted(region_levels.items())),
        "interaction_types": dict(sorted(interaction_types.items())),
        "language_hints": dict(sorted(languages.items())),
    }


def _coverage_metrics(canonical: list[dict[str, Any]], paired: list[dict[str, Any]], tasks: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    task_rows: list[dict[str, Any]] = []
    for task_name, rows in tasks.items():
        for row in rows:
            task_rows.append({"task": task_name, **row})
    region_counts = Counter(
        (row.get("target", {}).get("name") if isinstance(row.get("target"), dict) else row.get("region"))
        for row in canonical
    )
    view_counts = Counter(row.get("view") for row in canonical)
    split_counts = Counter(row.get("split") for row in canonical)
    diagnosis_counts = Counter(row.get("diagnosis") for row in canonical)
    task_regions: dict[str, Counter[str]] = defaultdict(Counter)
    task_views: dict[str, Counter[str]] = defaultdict(Counter)
    task_splits: dict[str, Counter[str]] = defaultdict(Counter)
    task_labels: dict[str, Counter[str]] = defaultdict(Counter)
    task_granularities: dict[str, Counter[str]] = defaultdict(Counter)
    region_level_counts = Counter()
    grounded_region_level_counts = Counter()
    grounded_standard_granularities: set[str] = set()
    task_box_counts: dict[str, list[int]] = defaultdict(list)
    for row in task_rows:
        task_name = row["task"]
        region = row.get("label") or row.get("region") or "missing"
        task_regions[task_name][str(region)] += 1
        task_views[task_name][str(row.get("view_scope") or row.get("view") or "missing")] += 1
        task_splits[task_name][str(row.get("split") or "missing")] += 1
        task_labels[task_name][_answer_label(row)] += 1
        if row.get("region_level") or row.get("grounding_level"):
            region_level_counts[str(row.get("region_level") or row.get("grounding_level"))] += 1
        if row.get("target_granularity"):
            task_granularities[task_name][str(row["target_granularity"])] += 1
        interaction_type = str(row.get("interaction_type") or "").lower()
        is_grounding_row = task_name in {"grounding", "grounded_vqa", "vgrounding"} or "localization" in interaction_type
        if is_grounding_row:
            if row.get("region_level") or row.get("grounding_level"):
                grounded_region_level_counts[str(row.get("region_level") or row.get("grounding_level"))] += 1
            if row.get("target_granularity"):
                grounded_standard_granularities.add(str(row["target_granularity"]).lower())
        task_box_counts[task_name].append(len(_bbox_records(row)))
    canonical_target_kinds = Counter(
        str(row.get("target", {}).get("kind", "missing")) if isinstance(row.get("target"), dict) else "missing"
        for row in canonical
    )
    whole_body_values = {"whole body", "whole_body", "whole-body", "wholebody"}
    standard_granularities = {
        str(value).lower()
        for values in task_granularities.values()
        for value in values
    }
    is_standard = bool(task_rows) and not canonical and not paired and all("schema_version" in row for row in task_rows)
    anatomical_present = "anatomical_region" in grounded_standard_granularities or "bone_region" in grounded_region_level_counts or (
        not is_standard and any(name in tasks for name in ("grounding", "grounded_vqa"))
    )
    lesion_present = "lesion_or_metastasis" in grounded_standard_granularities or "metastasis" in grounded_region_level_counts or (
        not is_standard and any(name in tasks for name in ("lesion_grounding", "lesion_grounded_vqa"))
    )
    whole_body_task_present = (
        "whole_body" in region_level_counts
        or any(str(value).lower() in whole_body_values for value in list(region_counts) + list(canonical_target_kinds))
        or "whole_body" in standard_granularities
    )
    whole_body_present = "whole_body" in grounded_region_level_counts or "whole_body" in grounded_standard_granularities
    return {
        "canonical": {
            "records": len(canonical),
            "patients": len({row.get("patient_id") for row in canonical if row.get("patient_id") is not None}),
            "regions": dict(sorted((str(key), value) for key, value in region_counts.items())),
            "views": dict(sorted((str(key), value) for key, value in view_counts.items())),
            "splits": dict(sorted((str(key), value) for key, value in split_counts.items())),
            "diagnoses": dict(sorted((str(key), value) for key, value in diagnosis_counts.items())),
            "target_kinds": dict(sorted(canonical_target_kinds.items())),
        },
        "paired": {
            "records": len(paired),
            "patients": len({row.get("patient_id") for row in paired if row.get("patient_id") is not None}),
            "regions": dict(sorted(Counter(str(row.get("region", "missing")) for row in paired).items())),
            "splits": dict(sorted(Counter(str(row.get("split", "missing")) for row in paired).items())),
        },
        "tasks": {
            task_name: {
                "regions": dict(sorted(task_regions[task_name].items())),
                "view_scopes": dict(sorted(task_views[task_name].items())),
                "splits": dict(sorted(task_splits[task_name].items())),
                "answer_labels": dict(sorted(task_labels[task_name].items())),
                "target_granularities": dict(sorted(task_granularities[task_name].items())),
                "boxes_per_row": _summary(task_box_counts[task_name]),
            }
            for task_name in sorted(tasks)
        },
        "grounding_granularity": {
            "anatomical_region": anatomical_present,
            "lesion_or_metastasis": lesion_present,
            "whole_body": whole_body_present,
        },
        "whole_body_task_present": whole_body_task_present,
        "region_levels": dict(sorted(region_level_counts.items())),
    }


def _integrity_metrics(canonical: list[dict[str, Any]], paired: list[dict[str, Any]], tasks: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {"canonical": {}, "tasks": {}, "split_leakage": {}}
    canonical_ids = [row.get("record_id") for row in canonical]
    result["canonical"] = {
        "duplicate_record_ids": len(canonical_ids) - len(set(canonical_ids)),
        "missing_patient_ids": sum(not isinstance(row.get("patient_id"), str) or not row["patient_id"] for row in canonical),
        "missing_target": sum(not isinstance(row.get("target"), dict) for row in canonical),
        "bbox": _bbox_metrics(canonical),
        "flag_counts": dict(sorted(Counter(flag for row in canonical for flag, value in (row.get("flags") or {}).items() if value).items())),
    }
    for task_name, rows in tasks.items():
        ids = [row.get("task_id") for row in rows]
        result["tasks"][task_name] = {
            "duplicate_task_ids": len(ids) - len(set(ids)),
            "missing_task_ids": sum(not isinstance(value, str) or not value for value in ids),
            "missing_split": sum(not row.get("split") for row in rows),
            "bbox": _bbox_metrics(rows),
        }
    evidence_to_patient = {str(row.get("evidence_id")): str(row.get("patient_id")) for row in paired if row.get("evidence_id") is not None}
    evidence_to_split = {str(row.get("evidence_id")): str(row.get("split")) for row in paired if row.get("evidence_id") is not None}
    patient_splits: dict[str, set[str]] = defaultdict(set)
    evidence_splits: dict[str, set[str]] = defaultdict(set)
    for task_rows in tasks.values():
        for row in task_rows:
            evidence_id = str(row.get("evidence_id")) if row.get("evidence_id") is not None else ""
            if evidence_id in evidence_to_patient and row.get("split"):
                patient_splits[evidence_to_patient[evidence_id]].add(str(row["split"]))
            if evidence_id and row.get("split"):
                evidence_splits[evidence_id].add(str(row["split"]))
    result["split_leakage"] = {
        "paired_patients_crossing_splits": sum(len(splits) > 1 for splits in {**{key: set(value) for key, value in patient_splits.items()}}.values()),
        "evidence_ids_crossing_splits": sum(len(splits) > 1 for splits in evidence_splits.values()),
        "paired_source_split_disagreements": sum(
            str(row.get("split")) != evidence_to_split[str(row.get("evidence_id"))]
            for task_rows in tasks.values()
            for row in task_rows
            if str(row.get("evidence_id")) in evidence_to_split and row.get("split")
        ),
    }
    return result


def _warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for task_name, metrics in report["tasks"].items():
        if metrics["linguistic"]["exact_text"]["duplicate_rate"] >= 0.1:
            warnings.append(f"{task_name}: exact duplicate text rate is {metrics['linguistic']['exact_text']['duplicate_rate']:.1%}.")
        if metrics["linguistic"]["masked_skeleton"]["top_share"] >= 0.5:
            warnings.append(f"{task_name}: one masked question skeleton accounts for {metrics['linguistic']['masked_skeleton']['top_share']:.1%} of rows.")
        if metrics["linguistic"]["template_ids"]["top_share"] >= 0.5:
            warnings.append(f"{task_name}: one template ID accounts for {metrics['linguistic']['template_ids']['top_share']:.1%} of rows.")
        if metrics["linguistic"]["template_ids"]["missing_template_id_rate"] > 0:
            warnings.append(f"{task_name}: {metrics['linguistic']['template_ids']['missing_template_id_rate']:.1%} of rows lack a template ID.")
    labels = report["coverage"]["canonical"]["diagnoses"]
    if labels:
        largest = max(labels.values()) / sum(labels.values())
        if largest >= 0.9:
            warnings.append(f"Canonical diagnosis is imbalanced: the largest class contains {largest:.1%} of records.")
    if not report["coverage"].get("whole_body_task_present", report["coverage"]["grounding_granularity"]["whole_body"]):
        warnings.append("No whole-body target/task was detected in the release manifests.")
    integrity = report["integrity"]
    if report.get("format") == "comparison-v1":
        invalid_boxes = integrity["canonical"]["bbox"]["invalid"]
        out_of_bounds_boxes = integrity["canonical"]["bbox"]["out_of_bounds"]
    else:
        invalid_boxes = integrity["canonical"]["bbox"]["invalid"] + sum(
            item["bbox"]["invalid"] for item in integrity["tasks"].values()
        )
        out_of_bounds_boxes = integrity["canonical"]["bbox"]["out_of_bounds"] + sum(
            item["bbox"]["out_of_bounds"] for item in integrity["tasks"].values()
        )
    if invalid_boxes:
        warnings.append(f"{invalid_boxes} target boxes have invalid or zero area coordinates.")
    if out_of_bounds_boxes:
        warnings.append(f"{out_of_bounds_boxes} target boxes exceed their reported image dimensions.")
    if integrity["split_leakage"]["paired_patients_crossing_splits"] or integrity["split_leakage"]["evidence_ids_crossing_splits"]:
        warnings.append("Cross-split patient or evidence leakage was detected.")
    if integrity["split_leakage"].get("images_crossing_splits", 0):
        warnings.append(
            f"{integrity['split_leakage']['images_crossing_splits']} image IDs occur in more than one split; use image-level grouping for leakage-safe evaluation."
        )
    return warnings


def benchmark_release(release_dir: str | Path, output_dir: str | Path | None = None, release_name: str | None = None) -> dict[str, Any]:
    release_path = Path(release_dir).resolve()
    if not release_path.is_dir():
        raise FileNotFoundError(f"Release directory not found: {release_path}")
    canonical = read_jsonl(release_path / "canonical.jsonl")
    paired = read_jsonl(release_path / "paired_evidence.jsonl")
    tasks: dict[str, list[dict[str, Any]]] = {}
    multitask_path = release_path / "multitask.jsonl"
    if multitask_path.exists():
        for row in read_jsonl(multitask_path):
            task_type = row.get("task_type")
            if task_type not in {"vqa", "grounding", "grounded_vqa"}:
                raise ValueError(f"Invalid multitask task_type: {task_type}")
            tasks.setdefault(task_type, []).append(row)
        scoped_files = {"multitask": "multitask.jsonl"}
    else:
        for task_name, filename in TASK_FILES:
            path = release_path / filename
            if path.exists():
                tasks[task_name] = read_jsonl(path)
        scoped_files = {name: filename for name, filename in TASK_FILES if (release_path / filename).exists()}
    all_rows = canonical + paired + [row for rows in tasks.values() for row in rows]
    regions = _region_values(all_rows)
    report: dict[str, Any] = {
        "benchmark_version": "0.1",
        "release": release_name or release_path.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": {
            "release_dir": str(release_path),
            "files": scoped_files,
        },
        "tasks": {
            task_name: {
                "rows": len(rows),
                "linguistic": _linguistic_metrics(rows, task_name, regions),
                "bbox": _bbox_metrics(rows),
            }
            for task_name, rows in sorted(tasks.items())
        },
        "coverage": _coverage_metrics(canonical, paired, tasks),
        "integrity": _integrity_metrics(canonical, paired, tasks),
    }
    report["warnings"] = _warnings(report)
    quality_path = release_path / "quality_report.json"
    if quality_path.exists():
        report["existing_quality_report"] = json.loads(quality_path.read_text(encoding="utf-8"))
    if output_dir is None:
        output_path = AUDITS / report["release"]
    else:
        output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "dataset_benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_path / "dataset_benchmark.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _standard_integrity_metrics(rows: list[dict[str, Any]], tasks: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ids = [row.get("record_id") for row in rows]
    patient_splits: dict[str, set[str]] = defaultdict(set)
    image_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row.get("split")
        if row.get("patient_id") and split:
            patient_splits[str(row["patient_id"])].add(str(split))
        if row.get("image_id") and split:
            image_splits[str(row["image_id"])].add(str(split))
    task_metrics: dict[str, Any] = {}
    for name, task_rows in tasks.items():
        task_ids = [row.get("record_id") for row in task_rows]
        task_metrics[name] = {
            "duplicate_task_ids": len(task_ids) - len(set(task_ids)),
            "missing_task_ids": sum(not isinstance(value, str) or not value for value in task_ids),
            "missing_split": sum(not row.get("split") for row in task_rows),
            "bbox": _bbox_metrics(task_rows),
        }
    return {
        "canonical": {
            "duplicate_record_ids": len(ids) - len(set(ids)),
            "missing_patient_ids": sum(not row.get("patient_id") for row in rows),
            "missing_target": sum(not row.get("target_boxes") for row in rows),
            "bbox": _bbox_metrics(rows),
            "flag_counts": {},
        },
        "tasks": task_metrics,
        "split_leakage": {
            "paired_patients_crossing_splits": sum(len(splits) > 1 for splits in patient_splits.values()),
            "evidence_ids_crossing_splits": 0,
            "paired_source_split_disagreements": 0,
            "images_crossing_splits": sum(len(splits) > 1 for splits in image_splits.values()),
        },
    }


def benchmark_standard(
    standard_file: str | Path,
    output_dir: str | Path | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Benchmark a standard comparison JSONL produced by ``standardize_dataset``."""

    source = Path(standard_file).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Standard dataset file not found: {source}")
    rows = _read_rows(source)
    tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row.get("task") or "other")
        tasks[task if task in STANDARD_TASKS else "other"].append(row)
    release = dataset_name or source.stem
    report: dict[str, Any] = {
        "benchmark_version": "0.2",
        "format": "comparison-v1",
        "release": release,
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": {
            "standard_file": str(source),
            "rows": len(rows),
            "schema_versions": dict(Counter(str(row.get("schema_version") or "NR") for row in rows)),
        },
        "tasks": {
            task_name: {
                "rows": len(task_rows),
                "linguistic": _linguistic_metrics(task_rows, task_name, _region_values(rows)),
                "bbox": _bbox_metrics(task_rows),
            }
            for task_name, task_rows in sorted(tasks.items())
        },
        "coverage": _coverage_metrics([], [], dict(tasks)),
        "integrity": _standard_integrity_metrics(rows, dict(tasks)),
        "warnings": [],
    }
    report["warnings"] = _warnings(report)
    if output_dir is None:
        output_path = AUDITS / release
    else:
        output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "dataset_benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_path / "dataset_benchmark.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def compare_benchmark_reports(
    report_files: Iterable[str | Path],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Combine compatible benchmark JSON reports into comparison tables."""

    reports: list[dict[str, Any]] = []
    paths: list[Path] = []
    for file in report_files:
        path = Path(file).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark report not found: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("tasks"), dict):
            raise ValueError(f"Not a benchmark report: {path}")
        paths.append(path)
        reports.append(value)
    if not reports:
        raise ValueError("At least one benchmark report is required")

    metric_rows: list[dict[str, Any]] = []
    question_type_rows: list[dict[str, Any]] = []
    structured_question_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for report in reports:
        coverage = report.get("coverage", {})
        granularity = coverage.get("grounding_granularity", {})
        dataset_rows.append(
            {
                "dataset": report.get("release", "unknown"),
                "format": report.get("format", "wbbs-release"),
                "task_families": len(report["tasks"]),
                "anatomical_region_grounding": bool(granularity.get("anatomical_region")),
                "lesion_or_metastasis_grounding": bool(granularity.get("lesion_or_metastasis")),
                "whole_body_grounding": bool(granularity.get("whole_body")),
                "warnings": len(report.get("warnings", [])),
            }
        )
        for task_name, task in sorted(report["tasks"].items()):
            linguistic = task.get("linguistic", {})
            exact = linguistic.get("exact_text", {})
            skeleton = linguistic.get("masked_skeleton", {})
            metric_rows.append(
                {
                    "dataset": report.get("release", "unknown"),
                    "task": task_name,
                    "rows": task.get("rows", 0),
                    "unique_words": linguistic.get("unique_words", 0),
                    "unique_normalized_text": exact.get("unique_normalized", 0),
                    "exact_duplicate_rate": exact.get("duplicate_rate", 0.0),
                    "masked_skeletons": skeleton.get("unique", 0),
                    "top_skeleton_share": skeleton.get("top_share", 0.0),
                    "mean_row_ttr": (linguistic.get("row_type_token_ratio") or {}).get("mean"),
                    "language_hints": "; ".join(
                        f"{key}: {value:,}" for key, value in sorted((linguistic.get("language_hints") or {}).items())
                    ),
                    "question_forms": "; ".join(
                        f"{key}: {value:,}" for key, value in sorted((linguistic.get("structured_question_forms") or {}).items())
                    ),
                    "region_levels": "; ".join(
                        f"{key}: {value:,}" for key, value in sorted((linguistic.get("region_levels") or {}).items())
                    ),
                    "unique_bigrams": linguistic.get("unique_bigrams", 0),
                    "unique_trigrams": linguistic.get("unique_trigrams", 0),
                }
            )
            question_types = linguistic.get("question_types", {})
            total_questions = sum(int(value) for value in question_types.values())
            for question_type, count in sorted(question_types.items()):
                question_type_rows.append(
                    {
                        "dataset": report.get("release", "unknown"),
                        "task": task_name,
                        "question_type": question_type,
                        "count": int(count),
                        "share": round(int(count) / total_questions, 6) if total_questions else 0.0,
                    }
                )
            structured_forms = linguistic.get("structured_question_forms", {})
            structured_total = sum(int(value) for value in structured_forms.values())
            for question_form, count in sorted(structured_forms.items()):
                structured_question_rows.append(
                    {
                        "dataset": report.get("release", "unknown"),
                        "task": task_name,
                        "question_form": question_form,
                        "count": int(count),
                        "share": round(int(count) / structured_total, 6) if structured_total else 0.0,
                    }
                )

    destination = Path(output_dir).resolve() if output_dir else paths[0].parent / "comparison"
    destination.mkdir(parents=True, exist_ok=True)
    metric_fields = list(metric_rows[0]) if metric_rows else []
    dataset_fields = list(dataset_rows[0]) if dataset_rows else []
    with (destination / "dataset_comparison_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    with (destination / "dataset_comparison_capabilities.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=dataset_fields)
        writer.writeheader()
        writer.writerows(dataset_rows)
    question_type_fields = list(question_type_rows[0]) if question_type_rows else []
    with (destination / "dataset_comparison_question_types.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=question_type_fields)
        writer.writeheader()
        writer.writerows(question_type_rows)
    structured_fields = list(structured_question_rows[0]) if structured_question_rows else ["dataset", "task", "question_form", "count", "share"]
    with (destination / "dataset_comparison_question_forms.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=structured_fields)
        writer.writeheader()
        writer.writerows(structured_question_rows)

    lines = [
        "# Dataset comparison",
        "",
        "Generated from benchmark JSON reports using the shared comparison-v1 metrics.",
        "",
        "## Capabilities",
        "",
        "| Dataset | Format | Task families | Region grounding | Lesion/metastasis grounding | Whole-body grounding | Warnings |",
        "|---|---|---:|---|---|---|---:|",
    ]
    for row in dataset_rows:
        lines.append(
            f"| {row['dataset']} | {row['format']} | {row['task_families']} | "
            f"{'yes' if row['anatomical_region_grounding'] else 'no'} | "
            f"{'yes' if row['lesion_or_metastasis_grounding'] else 'no'} | "
            f"{'yes' if row['whole_body_grounding'] else 'no'} | {row['warnings']} |"
        )
    lines.extend(
        [
            "",
            "## Linguistic and template metrics",
            "",
            "| Dataset | Task | Rows | Unique words | Unique normalized text | Exact duplicate rate | Masked skeletons | Top skeleton share | Mean row TTR | Language mix |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in metric_rows:
        mean_ttr = "NR" if row["mean_row_ttr"] is None else f"{row['mean_row_ttr']:.3f}"
        lines.append(
            f"| {row['dataset']} | {row['task']} | {row['rows']:,} | {row['unique_words']:,} | "
            f"{row['unique_normalized_text']:,} | {row['exact_duplicate_rate']:.1%} | "
            f"{row['masked_skeletons']:,} | {row['top_skeleton_share']:.1%} | {mean_ttr} | {row['language_hints']} |"
        )
    lines.extend(
        [
            "",
            "## Question types",
            "",
            "| Dataset | Task | Question type | Count | Share |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in question_type_rows:
        lines.append(
            f"| {row['dataset']} | {row['task']} | {row['question_type']} | {row['count']:,} | {row['share']:.1%} |"
        )
    if structured_question_rows:
        lines.extend(
            [
                "",
                "## Structured question forms",
                "",
                "| Dataset | Task | Form | Count | Share |",
                "|---|---|---|---:|---:|",
            ]
        )
        for row in structured_question_rows:
            lines.append(
                f"| {row['dataset']} | {row['task']} | {row['question_form']} | {row['count']:,} | {row['share']:.1%} |"
            )
    (destination / "dataset_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "format": "comparison-v1",
        "reports": [str(path) for path in paths],
        "datasets": dataset_rows,
        "metric_rows": metric_rows,
        "question_type_rows": question_type_rows,
        "structured_question_rows": structured_question_rows,
        "output_dir": str(destination),
    }


def render_markdown(report: dict[str, Any]) -> str:
    coverage = report["coverage"]
    lines = [
        f"# Dataset benchmark: {report['release']}",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "This is a deterministic, read-only audit of dataset composition, language structure, template concentration, grounding coverage, and manifest integrity. It is not a model evaluation.",
        "",
        "## Executive summary",
        "",
        (f"- Standard comparison rows: {report['scope']['rows']:,}." if report.get("format") == "comparison-v1" else f"- Canonical records: {coverage['canonical']['records']:,} from {coverage['canonical']['patients']:,} patients."),
        ("- Source format: comparison-v1 (missing fields are reported as NR/None)." if report.get("format") == "comparison-v1" else f"- Paired evidence units: {coverage['paired']['records']:,} from {coverage['paired']['patients']:,} patients."),
        f"- Task families audited: {len(report['tasks'])}.",
        f"- Warnings: {len(report['warnings'])}.",
        "",
        "## Task coverage",
        "",
        "| Task | Rows | Unique words | Unique normalized text | Masked skeletons | Top skeleton share |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task_name, metrics in report["tasks"].items():
        linguistic = metrics["linguistic"]
        lines.append(
            f"| {task_name} | {metrics['rows']:,} | {linguistic['unique_words']:,} | "
            f"{linguistic['exact_text']['unique_normalized']:,} | {linguistic['masked_skeleton']['unique']:,} | "
            f"{linguistic['masked_skeleton']['top_share']:.1%} |"
        )
    if report.get("format") == "comparison-v1":
        box_integrity_label = "Target boxes"
        box_integrity = report["integrity"]["canonical"]["bbox"]
    else:
        box_integrity_label = "Canonical boxes"
        box_integrity = report["integrity"]["canonical"]["bbox"]
    lines.extend(
        [
            "",
            "## Linguistic diversity",
            "",
            "| Task | Tokens (mean) | Mean row TTR | Corpus TTR | Unique bigrams | Unique trigrams | Exact duplicate rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task_name, metrics in report["tasks"].items():
        linguistic = metrics["linguistic"]
        lines.append(
            f"| {task_name} | {linguistic['token_count']['mean']:.2f} | {linguistic['row_type_token_ratio']['mean']:.3f} | "
            f"{linguistic['type_token_ratio']:.6f} | {linguistic['unique_bigrams']:,} | {linguistic['unique_trigrams']:,} | "
            f"{linguistic['exact_text']['duplicate_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Question types",
            "",
            "| Task | Type counts |",
            "|---|---|",
        ]
    )
    for task_name, metrics in report["tasks"].items():
        types = ", ".join(f"{key}: {value:,}" for key, value in metrics["linguistic"]["question_types"].items())
        lines.append(f"| {task_name} | {types} |")
    lines.extend(
        [
            "",
            "### Structured question metadata",
            "",
            "| Task | Forms | Region levels |",
            "|---|---|---|",
        ]
    )
    for task_name, metrics in report["tasks"].items():
        linguistic = metrics["linguistic"]
        forms = ", ".join(f"{key}: {value:,}" for key, value in linguistic.get("structured_question_forms", {}).items()) or "NR"
        levels = ", ".join(f"{key}: {value:,}" for key, value in linguistic.get("region_levels", {}).items()) or "NR"
        lines.append(f"| {task_name} | {forms} | {levels} |")
    lines.extend(
        [
            "",
            "## Grounding and structural coverage",
            "",
            f"- Anatomical-region grounding: {'yes' if coverage['grounding_granularity']['anatomical_region'] else 'no'}.",
            f"- Lesion/metastasis grounding: {'yes' if coverage['grounding_granularity']['lesion_or_metastasis'] else 'no'}.",
            f"- Whole-body grounding: {'yes' if coverage['grounding_granularity']['whole_body'] else 'no'}.",
            f"- Whole-body task coverage: {'yes' if coverage.get('whole_body_task_present', coverage['grounding_granularity']['whole_body']) else 'no'}.",
            f"- Region-level task rows: {coverage.get('region_levels', {})}.",
            f"- Canonical target kinds: {coverage['canonical']['target_kinds']}.",
            f"- Canonical regions: {coverage['canonical']['regions']}.",
            f"- Canonical views: {coverage['canonical']['views']}.",
            f"- Canonical diagnoses: {coverage['canonical']['diagnoses']}.",
            "",
            "## Integrity checks",
            "",
            f"- Duplicate canonical IDs: {report['integrity']['canonical']['duplicate_record_ids']}.",
            f"- Duplicate task IDs: {sum(item['duplicate_task_ids'] for item in report['integrity']['tasks'].values())}.",
            f"- Invalid/out-of-bounds {box_integrity_label.lower()}: {box_integrity['invalid']} / {box_integrity['out_of_bounds']}.",
            f"- Patients crossing splits: {report['integrity']['split_leakage']['paired_patients_crossing_splits']}.",
            f"- Evidence IDs crossing splits: {report['integrity']['split_leakage']['evidence_ids_crossing_splits']}.",
            f"- Image IDs crossing splits: {report['integrity']['split_leakage'].get('images_crossing_splits', 0)}.",
            "",
            "## Warnings",
            "",
        ]
    )
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Reproducibility", "", "- JSON report: `dataset_benchmark.json`.", "- This report is generated from the release JSONL files without changing them.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a VQA/grounding dataset release.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--release-dir", type=Path)
    source.add_argument("--standard-file", type=Path, help="Standard comparison JSONL produced by --standardize.")
    source.add_argument("--compare-files", nargs="+", type=Path, help="Benchmark JSON reports to combine into comparison tables.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--release-name")
    parser.add_argument("--standardize", action="store_true", help="Convert --standard-file input to standard JSONL before benchmarking.")
    parser.add_argument("--standard-output", type=Path, help="Output JSONL path for --standardize (defaults beside input).")
    parser.add_argument("--task-hint")
    parser.add_argument("--bbox-format", default="xyxy", choices=("xyxy", "xywh", "coco"))
    parser.add_argument("--field-map", type=Path, help="Optional JSON object mapping standard fields to source keys.")
    args = parser.parse_args()
    if args.release_dir is not None:
        report = benchmark_release(args.release_dir, args.output_dir, args.release_name)
    elif args.standard_file is not None:
        standard_path = args.standard_file
        if args.standardize:
            output_path = args.standard_output or standard_path.with_suffix(".standard.jsonl")
            field_map = None
            if args.field_map:
                value = json.loads(args.field_map.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("--field-map must contain a JSON object")
                field_map = value
            summary = standardize_dataset(standard_path, output_path, args.release_name, args.task_hint, field_map=field_map, bbox_format=args.bbox_format)
            print(json.dumps(summary, indent=2, sort_keys=True))
            standard_path = output_path
        report = benchmark_standard(standard_path, args.output_dir, args.release_name)
        print(json.dumps({"release": report["release"], "tasks": {key: value["rows"] for key, value in report["tasks"].items()}, "warnings": len(report["warnings"])}, indent=2, sort_keys=True))
    else:
        summary = compare_benchmark_reports(args.compare_files, args.output_dir)
        print(json.dumps({"datasets": [row["dataset"] for row in summary["datasets"]], "output_dir": summary["output_dir"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
