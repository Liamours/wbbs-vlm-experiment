from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from preprocess.canonical import SPLITS, read_jsonl, validate_box


def validate_manifest(path: str | Path, kind: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    validators = {
        "canonical": _validate_canonical,
        "vqa": _validate_vqa,
        "grounding": _validate_grounding,
        "grounded_vqa": _validate_grounded_vqa,
        "vgrounding": _validate_vgrounding,
        "multitask": _validate_multitask,
        "metadata_vqa": _validate_metadata_vqa,
        "lesion_vqa": _validate_vqa,
        "lesion_grounding": _validate_grounding,
        "lesion_grounded_vqa": _validate_grounded_vqa,
    }
    if kind not in validators:
        raise ValueError(f"Unsupported manifest kind: {kind}")
    task_ids: set[str] = set()
    for index, row in enumerate(rows):
        validators[kind](row)
        identifier = row.get("record_id", row.get("task_id"))
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"Row {index} has no identifier")
        if identifier in task_ids:
            raise ValueError(f"Duplicate identifier: {identifier}")
        task_ids.add(identifier)
    return {"kind": kind, "rows": len(rows), "splits": dict(Counter(row.get("split") for row in rows))}


def _validate_canonical(row: dict[str, Any]) -> None:
    _common(row, "record_id")
    if not isinstance(row.get("target"), dict):
        raise ValueError("Canonical rows require target")
    _validate_box_in_image(row["target"].get("bbox"), row["image_size"], "target.bbox")
    if not isinstance(row.get("patient_id"), str) or not row["patient_id"]:
        raise ValueError("Canonical rows require patient_id")


def _validate_vqa(row: dict[str, Any]) -> None:
    _common(row, "task_id")
    _text(row, "question")
    if not isinstance(row.get("answer"), (str, list)):
        raise ValueError("VQA rows require answer")
    _text(row, "template_id")
    _text(row, "answer_rule")
    _validate_label_provenance(row)
    if isinstance(row.get("labels"), dict):
        _validate_label_levels(row["labels"])


def _validate_grounding(row: dict[str, Any]) -> None:
    _common(row, "task_id")
    _text(row, "query")
    _text(row, "label")
    _validate_label_provenance(row)
    if "targets" in row:
        _validate_targets(row["targets"], row)
    else:
        _validate_box_in_image(row.get("bbox"), row["image_size"], "bbox")
    if isinstance(row.get("target"), dict) and row["target"].get("schema_version") == "r2-target-v1":
        _validate_structured_target(row["target"], row)


def _validate_grounded_vqa(row: dict[str, Any]) -> None:
    _validate_vqa(row)
    if "evidence_targets" in row:
        _validate_targets(row["evidence_targets"], row)
    else:
        _validate_box_in_image(row.get("evidence_bbox"), row["image_size"], "evidence_bbox")
    if "target" in row:
        if isinstance(row["target"], dict):
            _validate_structured_target(row["target"], row)
        else:
            _text(row, "target")


def _validate_structured_target(target: dict[str, Any], row: dict[str, Any]) -> None:
    _validate_label_levels(target)
    locations = target.get("locations")
    if not isinstance(locations, list):
        raise ValueError("R2 targets require locations list")
    image_sizes = {item["view"]: item["image_size"] for item in row.get("images", []) if isinstance(item, dict) and item.get("view")}
    for index, location in enumerate(locations):
        if not isinstance(location, dict) or location.get("view") not in image_sizes:
            raise ValueError(f"Invalid R2 target view at index {index}")
        _validate_box_in_image(location.get("bbox"), image_sizes[location["view"]], f"target.locations[{index}].bbox")


def _validate_label_levels(value: dict[str, Any]) -> None:
    region = value.get("region")
    if not isinstance(region, dict) or not isinstance(region.get("label"), str) or region.get("level") not in {"whole_body", "bone_region", "metastasis"}:
        raise ValueError("R2 labels require region.label and region.level")
    classification = value.get("classification")
    if not isinstance(classification, dict) or classification.get("label") not in {"benign", "malignant"}:
        raise ValueError("R2 labels require classification.label=benign or malignant")


def _validate_label_provenance(row: dict[str, Any]) -> None:
    if "effective_region_label" in row and row["effective_region_label"] not in {"benign", "malignant"}:
        raise ValueError("effective_region_label must be benign or malignant")
    if "label_conflict" in row and not isinstance(row["label_conflict"], bool):
        raise ValueError("label_conflict must be boolean")
    raw = row.get("source_region_label")
    if raw is not None and not isinstance(raw, (str, dict)):
        raise ValueError("source_region_label must be a string or view dictionary")


def _validate_vgrounding(row: dict[str, Any]) -> None:
    """Validate the merged R2 grounding manifest."""

    interaction = str(row.get("interaction_type", "localization"))
    if interaction == "localization_and_classification":
        _validate_grounded_vqa(row)
    else:
        _validate_grounding(row)


def _validate_multitask(row: dict[str, Any]) -> None:
    if row.get("schema_version") != "r3-multitask-v1":
        raise ValueError("R3 multitask rows require schema_version=r3-multitask-v1")
    task_type = row.get("task_type")
    requirements = row.get("required_outputs")
    expected = {
        "vqa": {"answer": True, "classification": True, "locations": False},
        "grounding": {"answer": False, "classification": False, "locations": True},
        "grounded_vqa": {"answer": True, "classification": True, "locations": True},
    }
    if task_type not in expected or requirements != expected[task_type]:
        raise ValueError("R3 multitask rows require a valid task_type and required_outputs contract")
    _text(row, "prompt")
    if task_type == "vqa":
        _validate_vqa(row)
    elif task_type == "grounding":
        _validate_grounding(row)
    else:
        _validate_grounded_vqa(row)


def _validate_metadata_vqa(row: dict[str, Any]) -> None:
    if row.get("schema_version") != "r3-metadata-v1" or row.get("task_type") != "metadata_vqa":
        raise ValueError("R3 metadata rows require r3-metadata-v1 and task_type=metadata_vqa")
    _text(row, "task_id")
    _text(row, "prompt")
    _text(row, "answer")
    metadata_kind = row.get("metadata_kind")
    if metadata_kind not in {"view", "paired_views"} or not isinstance(row.get("answer_source"), str):
        raise ValueError("R3 metadata rows require a supported metadata_kind and answer_source")
    if row.get("training_eligible") is not False or row.get("benchmark_eligible") is not False or row.get("targets") != []:
        raise ValueError("R3 metadata rows must be explicitly excluded from training/benchmarking and have no targets")
    images = row.get("images")
    expected_views = {"ANT"} if row.get("answer_label") == "ANT" else {"POST"}
    if not isinstance(images, list) or len(images) != (1 if metadata_kind == "view" else 2):
        raise ValueError("R3 metadata rows require one view image or one ANT+POST image pair")
    views = {image.get("view") for image in images if isinstance(image, dict)}
    if (metadata_kind == "view" and views != expected_views) or (metadata_kind == "paired_views" and views != {"ANT", "POST"}):
        raise ValueError("R3 metadata rows require images consistent with their recorded view answer")
    for image in images:
        if not isinstance(image, dict):
            raise ValueError("R3 metadata images must be objects")
        _text(image, "image")
        _image_size(image.get("image_size"))
    if row.get("split") not in SPLITS:
        raise ValueError("Rows require a valid split")


def _common(row: dict[str, Any], identifier: str) -> None:
    _text(row, identifier)
    if "images" in row:
        images = row["images"]
        views = {item.get("view") for item in images if isinstance(item, dict)}
        whole_body = str(row.get("region_level", "")).lower() == "whole_body"
        if not isinstance(images, list) or (not whole_body and (views != {"ANT", "POST"} or len(images) != 2)) or (whole_body and (len(images) != 1 or views not in ({"ANT"}, {"POST"}))):
            raise ValueError("Rows require ANT+POST images, except whole_body rows which require one ANT or POST image")
        for image in images:
            if not isinstance(image, dict):
                raise ValueError("Paired images must be objects")
            _text(image, "image")
            _image_size(image.get("image_size"))
    else:
        _text(row, "image")
        _image_size(row.get("image_size"))
    if row.get("split") not in SPLITS:
        raise ValueError("Rows require a valid split")


def _text(row: dict[str, Any], key: str) -> None:
    if not isinstance(row.get(key), str) or not row[key].strip():
        raise ValueError(f"Rows require non-empty {key}")


def _validate_box_in_image(box: Any, image_size: list[int], field: str) -> None:
    _, _, x2, y2 = validate_box(box, field)
    width, height = image_size
    if x2 > width or y2 > height:
        raise ValueError(f"{field} exceeds image_size")


def _validate_targets(targets: Any, row: dict[str, Any]) -> None:
    if not isinstance(targets, list) or not targets:
        raise ValueError("Paired rows require non-empty targets")
    image_sizes = {item["view"]: item["image_size"] for item in row["images"]}
    views = []
    for index, target in enumerate(targets):
        if not isinstance(target, dict) or target.get("view") not in image_sizes:
            raise ValueError(f"Invalid target view at index {index}")
        view = target["view"]
        _validate_box_in_image(target.get("bbox"), image_sizes[view], f"targets[{index}].bbox")
        views.append(view)
    if len(views) != len(set(views)):
        raise ValueError("Paired targets may not repeat a view")


def _image_size(image_size: Any) -> None:
    if not isinstance(image_size, list) or len(image_size) != 2 or not all(isinstance(item, int) and item > 0 for item in image_size):
        raise ValueError("Rows require image_size=[width, height]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a WBBS dataset manifest.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--kind", choices=("canonical", "vqa", "grounding", "grounded_vqa", "vgrounding", "multitask", "metadata_vqa", "lesion_vqa", "lesion_grounding", "lesion_grounded_vqa"), required=True)
    args = parser.parse_args()
    print(json.dumps(validate_manifest(args.input, args.kind), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
