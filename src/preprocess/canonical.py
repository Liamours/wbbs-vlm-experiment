from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "val", "test")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"JSONL file not found: {source}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {source}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {source}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {source}")
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def canonicalize(rows: Iterable[dict[str, Any]], seed: int = 4050) -> list[dict[str, Any]]:
    materialized = [_canonical_record(row) for row in rows]
    record_ids = [record["record_id"] for record in materialized]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Canonical record_id values must be unique")
    assignments = _assign_splits(materialized, seed)
    return [{**record, "split": assignments[record["patient_id"]]} for record in materialized]


def validate_box(value: Any, field: str = "bbox") -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{field} must be [x1, y1, x2, y2]")
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        raise ValueError(f"{field} must contain numbers")
    x1, y1, x2, y2 = [float(item) for item in value]
    if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
        raise ValueError(f"{field} must satisfy 0 <= x1 < x2 and 0 <= y1 < y2")
    return [_number(item) for item in (x1, y1, x2, y2)]


def _canonical_record(row: dict[str, Any]) -> dict[str, Any]:
    patient_id = _identifier(row, "patient_id")
    image = _text(row, "image")
    view = _text(row, "view")
    image_size = _image_size(row)
    target = _target(row)
    _check_box_bounds(target["bbox"], image_size, "target.bbox")
    captions = {
        key.removeprefix("caption_"): value
        for key, value in row.items()
        if key.startswith("caption_") and isinstance(value, str) and value.strip()
    }
    record = {
        "record_id": _record_id(row),
        "image": image,
        "image_size": image_size,
        "patient_id": patient_id,
        "view": view,
        "target": target,
        "diagnosis": row.get("diagnosis"),
        "captions": captions,
        "hotspots": _hotspots(row.get("hotspots", [])),
        "qa": _questions(row.get("qa", [])),
        "grounded_qa": _grounded_questions(row.get("grounded_qa", [])),
        "grounding": _grounding(row.get("grounding", []), target["bbox"]),
        "flags": _flags(row),
        "source_label": _optional_text(row.get("source_label")),
        "duplicate_group_id": _optional_identifier(row.get("duplicate_group_id")),
        "duplicate_of_patient_id": _optional_identifier(row.get("duplicate_of_patient_id")),
        "duplicate_of_sibling_component": _optional_identifier(row.get("duplicate_of_sibling_component")),
        "source_split": _source_split(row),
        "provenance": row.get("provenance", {}),
    }
    for index, hotspot in enumerate(record["hotspots"]):
        _check_box_bounds(hotspot["bbox"], image_size, f"hotspots[{index}].bbox")
    for index, question in enumerate(record["qa"]):
        if "evidence_bbox" in question:
            _check_box_bounds(question["evidence_bbox"], image_size, f"qa[{index}].evidence_bbox")
    for index, question in enumerate(record["grounded_qa"]):
        _check_box_bounds(question["evidence_bbox"], image_size, f"grounded_qa[{index}].evidence_bbox")
    for index, grounding in enumerate(record["grounding"]):
        _check_box_bounds(grounding["bbox"], image_size, f"grounding[{index}].bbox")
    if not isinstance(record["provenance"], dict):
        raise ValueError("provenance must be an object")
    return record


def _target(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("target")
    if isinstance(raw, dict):
        kind = _text(raw, "kind")
        name = _text(raw, "name")
        bbox = validate_box(raw.get("bbox"), "target.bbox")
    else:
        name = _text(row, "region", default="whole_body")
        kind = "whole_body" if name == "whole_body" else "region"
        bbox = validate_box(row.get("bbox"), "bbox")
    return {"kind": kind, "name": name, "bbox": bbox}


def _image_size(row: dict[str, Any]) -> list[int]:
    raw = row.get("image_size")
    if raw is None and "width" in row and "height" in row:
        raw = [row["width"], row["height"]]
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError("image_size must be [width, height]")
    if not all(isinstance(item, int) and item > 0 for item in raw):
        raise ValueError("image_size must contain positive integers")
    return raw


def _hotspots(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("hotspots must be a list")
    hotspots: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"hotspots[{index}] must be an object")
        hotspot = {"bbox": validate_box(item.get("bbox"), f"hotspots[{index}].bbox")}
        for key in ("label", "class", "certainty", "assignment", "source_id"):
            if isinstance(item.get(key), str) and item[key].strip():
                hotspot[key] = item[key]
        hotspots.append(hotspot)
    return hotspots


def _questions(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("qa must be a list")
    questions: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"qa[{index}] must be an object")
        question = _text(item, "question", default=item.get("prompt"))
        answer = _answer(item.get("answer", item.get("target")), f"qa[{index}].answer")
        normalized = {
            "question": question,
            "answer": answer,
            "template_id": _text(item, "template_id", default=item.get("type", "source_qa")),
            "template_arguments": item.get("template_arguments", {}),
            "answer_rule": item.get("answer_rule", "source_annotation"),
        }
        if not isinstance(normalized["template_arguments"], dict):
            raise ValueError(f"qa[{index}].template_arguments must be an object")
        if not isinstance(normalized["answer_rule"], str):
            raise ValueError(f"qa[{index}].answer_rule must be text")
        if "evidence_bbox" in item:
            normalized["evidence_bbox"] = validate_box(item["evidence_bbox"], f"qa[{index}].evidence_bbox")
        answer_label = item.get("answer_label")
        if answer_label is not None:
            if not isinstance(answer_label, str) or not answer_label.strip():
                raise ValueError(f"qa[{index}].answer_label must be non-empty text")
            normalized["answer_label"] = answer_label.strip()
        questions.append(normalized)
    return questions


def _grounded_questions(raw: Any) -> list[dict[str, Any]]:
    questions = _questions(raw)
    for index, (question, source) in enumerate(zip(questions, raw, strict=True)):
        if not isinstance(source, dict):
            raise ValueError(f"grounded_qa[{index}] must be an object")
        question["target"] = _text(source, "target")
        if "evidence_bbox" not in question:
            raise ValueError(f"grounded_qa[{index}].evidence_bbox is required")
    return questions


def _grounding(raw: Any, default_bbox: list[float]) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("grounding must be a list")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"grounding[{index}] must be an object")
        rows.append(
            {
                "query": _text(item, "query", default=item.get("expression")),
                "label": _text(item, "label"),
                "bbox": validate_box(item.get("bbox", default_bbox), f"grounding[{index}].bbox"),
                "source": _text(item, "source", default="source_annotation"),
            }
        )
    return rows


def _flags(row: dict[str, Any]) -> dict[str, Any]:
    flags = dict(row.get("flags", {}))
    if not isinstance(flags, dict):
        raise ValueError("flags must be an object")
    for key in ("low_precision_region", "region_overlap", "outlier", "anomaly_score", "likely_corrupt_image"):
        if key in row:
            flags[key] = row[key]
    return flags


def _assign_splits(records: list[dict[str, Any]], seed: int) -> dict[str, str]:
    patients = {record["patient_id"] for record in records}
    parent = {patient: patient for patient in patients}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    components: dict[str, str] = {}
    for record in records:
        patient = record["patient_id"]
        linked = record["duplicate_of_patient_id"]
        if linked in patients:
            union(patient, linked)
        for key in ("duplicate_group_id", "duplicate_of_sibling_component"):
            value = record[key]
            if value:
                if value in components:
                    union(patient, components[value])
                else:
                    components[value] = patient

    provided = {_source_split(record) for record in records if _source_split(record)}
    if provided:
        if any(_source_split(record) is None for record in records):
            raise ValueError("Either every source record must define split or none may define it")
        patient_splits: dict[str, set[str]] = defaultdict(set)
        for record in records:
            patient_splits[record["patient_id"]].add(_source_split(record))
        if any(len(values) != 1 for values in patient_splits.values()):
            raise ValueError("A patient has records in multiple source splits")
        assignments = {patient: next(iter(values)) for patient, values in patient_splits.items()}
        for patient in patients:
            component_splits = {assignments[other] for other in patients if find(other) == find(patient)}
            if len(component_splits) != 1:
                raise ValueError("Duplicate-linked patients cross source split boundaries")
        return {patient: assignments[patient] for patient in patients}

    grouped: dict[str, list[str]] = defaultdict(list)
    for patient in sorted(patients):
        grouped[find(patient)].append(patient)
    ordered = sorted(grouped.values(), key=lambda group: _stable_key(seed, "|".join(group)))
    total = len(patients)
    train_cutoff = total * 0.8
    val_cutoff = total * 0.9
    assignments: dict[str, str] = {}
    assigned = 0
    for group in ordered:
        split = "train" if assigned < train_cutoff else "val" if assigned < val_cutoff else "test"
        assignments.update({patient: split for patient in group})
        assigned += len(group)
    return assignments


def _source_split(record: dict[str, Any]) -> str | None:
    value = record.get("source_split")
    if value is None:
        return None
    if value == "valid":
        value = "val"
    if value not in SPLITS:
        raise ValueError("source_split must be train, val, valid, or test")
    return value


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _record_id(row: dict[str, Any]) -> str:
    value = row.get("record_id", row.get("id"))
    if value is not None:
        return _identifier({"record_id": value}, "record_id")
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _identifier(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if isinstance(value, (str, int)) and str(value).strip():
        return str(value)
    raise ValueError(f"{key} must be a non-empty string or integer")


def _optional_identifier(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (str, int)):
        return str(value)
    raise ValueError("Duplicate identifiers must be strings or integers")


def _optional_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (str, int)) and str(value).strip():
        return str(value).strip()
    raise ValueError("Optional text must be a non-empty string or integer")


def _text(row: dict[str, Any], key: str, default: Any = None) -> str:
    value = row.get(key, default)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"{key} must be non-empty text")


def _answer(value: Any, field: str) -> str | list[str]:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list) and value and all(isinstance(item, str) and item.strip() for item in value):
        return [item.strip() for item in value]
    raise ValueError(f"{field} must be non-empty text or a non-empty text list")


def _number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _check_box_bounds(box: list[float], image_size: list[int], field: str) -> None:
    _, _, x2, y2 = box
    width, height = image_size
    if x2 > width or y2 > height:
        raise ValueError(f"{field} exceeds image_size")
