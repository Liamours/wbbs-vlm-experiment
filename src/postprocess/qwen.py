from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from preprocess.canonical import read_jsonl, write_jsonl
from preprocess.tokens import format_view_bbx


EXPORTS = {
    "anatomy_vqa": ("vqa.jsonl", "vqa"),
    "anatomy_grounding": ("grounding.jsonl", "grounding"),
    "anatomy_grounded_vqa": ("grounded_vqa.jsonl", "grounded_vqa"),
    "lesion_vqa": ("lesion_vqa.jsonl", "vqa"),
    "lesion_grounding": ("lesion_grounding.jsonl", "grounding"),
    "lesion_grounded_vqa": ("lesion_grounded_vqa.jsonl", "grounded_vqa"),
}
VIEW_BOX = re.compile(r"<(ANT|PST)><BBX>([^<]+)</BBX></\1>")
BARE_BOX = re.compile(r"<BBX>([^<]+)</BBX>")
REGION = re.compile(r"<REG>([^<]+)</REG>")
CLASS = re.compile(r"<CLS>(normal|abnormal)</CLS>")


def export_qwen(rows: Iterable[dict[str, Any]], task: str, split: str | None = None, schema: str = "tags") -> list[dict[str, Any]]:
    if task not in {"vqa", "grounding", "grounded_vqa"}:
        raise ValueError("task must be vqa, grounding, or grounded_vqa")
    if schema not in {"tags", "json", "multitask_json"}:
        raise ValueError("schema must be tags, json, or multitask_json")
    exported = []
    for row in rows:
        if split and row.get("split") != split:
            continue
        exported.append(_export_row(row, task, schema))
    return exported


def build_qwen_exports(
    release: str | Path, output: str | Path, schema: str = "tags", image_root: str | Path | None = None
) -> dict[str, Any]:
    source, target = Path(release).resolve(), Path(output).resolve()
    target.mkdir(parents=True, exist_ok=True)
    if (source / "multitask.jsonl").exists():
        return _build_r3_qwen_exports(source, target, schema, image_root)
    if (source / "vgrounding.jsonl").exists():
        return _build_r2_qwen_exports(source, target, schema, image_root)
    if image_root is not None:
        raise ValueError("image_root is only used for merged R2 Qwen exports")
    summary: dict[str, Any] = {"release": str(source), "schema": schema, "exports": {}}
    grounded: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "val", "test")}
    for name, (filename, task) in EXPORTS.items():
        rows = read_jsonl(source / filename)
        counts = {}
        for split in grounded:
            exported = export_qwen(rows, task, split, schema)
            _rebase_images(exported, source, target)
            write_jsonl(target / f"{name}_{split}.jsonl", exported)
            counts[split] = len(exported)
            if task == "grounded_vqa":
                grounded[split].extend(exported)
        summary["exports"][name] = {"source": filename, "task": task, "splits": counts}
    for split, rows in grounded.items():
        write_jsonl(target / f"grounded_vqa_{split}.jsonl", rows)
    summary["primary_grounded_vqa"] = {split: len(rows) for split, rows in grounded.items()}
    (target / "qwen_export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _build_r2_qwen_exports(source: Path, target: Path, schema: str, image_root: str | Path | None) -> dict[str, Any]:
    """Export paired, merged R2/R3 grounding tasks for Qwen3-VL.

    Merged manifest image references are deliberately portable (``bs80k/...``),
    so their local source root must be supplied explicitly. Only paired tasks
    are exported: the existing Qwen training path consumes ANT+POST inputs.
    """

    if image_root is None:
        raise ValueError("Merged R2/R3 Qwen exports require image_root pointing to bs80k-imaging-raw")
    root = Path(image_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"R2 image_root does not exist: {root}")
    grounding_rows = read_jsonl(source / "vgrounding.jsonl")
    vqa_rows = read_jsonl(source / "vqa.jsonl")
    groups = {
        "vqa": (vqa_rows, "vqa"),
        "vgrounding_localization": ([row for row in grounding_rows if row.get("interaction_type") == "localization"], "grounding"),
        "vgrounding_grounded_vqa": (
            [row for row in grounding_rows if row.get("interaction_type") == "localization_and_classification"],
            "grounded_vqa",
        ),
    }
    summary: dict[str, Any] = {
        "release": str(source),
        "format": "wbbs-merged-v1",
        "schema": schema,
        "image_root": str(root),
        "exports": {},
    }
    primary: dict[str, int] = {}
    multitask: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "val", "test")}
    for name, (group_rows, task) in groups.items():
        counts: dict[str, int] = {}
        for split in ("train", "val", "test"):
            exported = export_qwen(group_rows, task, split, schema)
            _rebase_r2_images(exported, target, root)
            manifest = target / f"{name}_{split}.jsonl"
            write_jsonl(manifest, exported)
            counts[split] = len(exported)
            multitask[split].extend(exported)
            if task == "grounded_vqa":
                write_jsonl(target / f"grounded_vqa_{split}.jsonl", exported)
                primary[split] = len(exported)
        summary["exports"][name] = {"task": task, "rows": len(group_rows), "splits": counts}
    summary["primary_grounded_vqa"] = primary
    summary["multitask"] = {}
    for split, exported in multitask.items():
        write_jsonl(target / f"multitask_{split}.jsonl", exported)
        summary["multitask"][split] = len(exported)
    summary_path = target / "qwen_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_sha256sums(target, [*target.glob("*.jsonl"), summary_path])
    return summary


def _build_r3_qwen_exports(source: Path, target: Path, schema: str, image_root: str | Path | None) -> dict[str, Any]:
    """Export R3's primary unified multitask manifest for Qwen3-VL."""

    if image_root is None:
        raise ValueError("Merged R2/R3 Qwen exports require image_root pointing to bs80k-imaging-raw")
    root = Path(image_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Merged image_root does not exist: {root}")
    paired_patients = {
        str(row["evidence_id"]): row.get("patient_id")
        for row in read_jsonl(source / "paired_evidence.jsonl")
        if row.get("evidence_id") is not None
    }
    rows = _attach_paired_patient_ids(read_jsonl(source / "multitask.jsonl"), paired_patients)
    groups = {
        "vqa": ([row for row in rows if row.get("task_type") == "vqa"], "vqa"),
        "vgrounding_localization": ([row for row in rows if row.get("task_type") == "grounding"], "grounding"),
        "vgrounding_grounded_vqa": ([row for row in rows if row.get("task_type") == "grounded_vqa"], "grounded_vqa"),
    }
    summary: dict[str, Any] = {"release": str(source), "format": "wbbs-r3-multitask-v1", "schema": schema, "image_root": str(root), "exports": {}}
    primary: dict[str, int] = {}
    multitask: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "val", "test")}
    for name, (group_rows, task) in groups.items():
        counts: dict[str, int] = {}
        for split in ("train", "val", "test"):
            exported = export_qwen(group_rows, task, split, schema)
            _rebase_r2_images(exported, target, root)
            write_jsonl(target / f"{name}_{split}.jsonl", exported)
            counts[split] = len(exported)
            multitask[split].extend(exported)
            if task == "grounded_vqa":
                write_jsonl(target / f"grounded_vqa_{split}.jsonl", exported)
                primary[split] = len(exported)
        summary["exports"][name] = {"task": task, "rows": len(group_rows), "splits": counts}
    summary["primary_grounded_vqa"] = primary
    summary["multitask"] = {}
    for split, exported in multitask.items():
        write_jsonl(target / f"multitask_{split}.jsonl", exported)
        summary["multitask"][split] = len(exported)
    summary_path = target / "qwen_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_sha256sums(target, [*target.glob("*.jsonl"), summary_path])
    return summary


def _attach_paired_patient_ids(rows: Iterable[dict[str, Any]], paired_patients: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach recorded paired-evidence patient IDs without inferring from paths."""

    enriched = []
    for row in rows:
        patient_id = row.get("patient_id") or paired_patients.get(str(row.get("evidence_id")))
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError(f"R3 task row lacks a patient ID linked through paired evidence: {row.get('task_id')}")
        enriched.append({**row, "patient_id": patient_id})
    return enriched


def _rebase_images(rows: list[dict[str, Any]], source: Path, target: Path) -> None:
    for row in rows:
        for image in row["images"]:
            resolved = (source / image["image"]).resolve()
            image["image"] = Path(os.path.relpath(resolved, target)).as_posix()


def _rebase_r2_images(rows: list[dict[str, Any]], target: Path, image_root: Path) -> None:
    for row in rows:
        for image in row["images"]:
            portable = str(image["image"]).replace("\\", "/").split("/")
            if len(portable) < 3 or portable[0] != "bs80k":
                raise ValueError(f"Merged release image path must use the portable bs80k/ prefix: {image['image']}")
            resolved = (image_root.joinpath(*portable[1:])).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"Merged release image not found under image_root: {image['image']}")
            image["image"] = Path(os.path.relpath(resolved, target)).as_posix()


def _write_sha256sums(output: Path, paths: Iterable[Path]) -> None:
    entries = []
    for path in sorted({path.resolve() for path in paths}):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {path.name}")
    (output / "SHA256SUMS.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")


def parse_qwen_prediction(prediction: str) -> dict[str, Any]:
    boxes = []
    for match in VIEW_BOX.finditer(prediction):
        box = _parse_box(match.group(2))
        if box is not None:
            boxes.append({"view": match.group(1), "bbox": box})
    if not boxes:
        for match in BARE_BOX.finditer(prediction):
            box = _parse_box(match.group(1))
            if box is not None:
                boxes.append({"view": None, "bbox": box})
    region = REGION.search(prediction)
    diagnosis = CLASS.search(prediction)
    return {
        "region": region.group(1) if region else None,
        "answer_label": diagnosis.group(1) if diagnosis else None,
        "boxes": boxes,
    }


def parse_qwen_json_prediction(prediction: str) -> dict[str, Any]:
    try:
        payload = json.loads(prediction)
    except json.JSONDecodeError:
        return {"region": None, "answer_label": None, "boxes": []}
    if not isinstance(payload, dict):
        return {"region": None, "answer_label": None, "boxes": []}
    boxes = []
    for item in payload.get("boxes", []):
        if not isinstance(item, dict) or item.get("view") not in {"ANT", "PST"}:
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            parsed = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if parsed[0] < parsed[2] and parsed[1] < parsed[3]:
            boxes.append({"view": item["view"], "bbox": parsed})
    label = payload.get("class")
    return {
        "region": payload.get("region") if isinstance(payload.get("region"), str) else None,
        "answer_label": label if label in {"normal", "abnormal"} else None,
        "boxes": boxes,
    }


def parse_qwen_multitask_json_prediction(prediction: str) -> dict[str, Any]:
    try:
        payload = json.loads(prediction)
    except json.JSONDecodeError:
        return {"task": None, "answer": None, "region": None, "answer_label": None, "boxes": []}
    if not isinstance(payload, dict):
        return {"task": None, "answer": None, "region": None, "answer_label": None, "boxes": []}
    boxes = []
    for item in payload.get("boxes", []):
        if not isinstance(item, dict) or item.get("view") not in {"ANT", "PST"}:
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            parsed = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if parsed[0] < parsed[2] and parsed[1] < parsed[3]:
            boxes.append({"view": item["view"], "bbox": parsed})
    task = payload.get("task")
    return {
        "task": task if task in {"vqa", "grounding", "grounded_vqa"} else None,
        "answer": payload.get("answer") if isinstance(payload.get("answer"), str) else None,
        "region": payload.get("region") if isinstance(payload.get("region"), str) else None,
        "answer_label": payload.get("class") if isinstance(payload.get("class"), str) else None,
        "boxes": boxes,
    }


def _export_row(row: dict[str, Any], task: str, schema: str) -> dict[str, Any]:
    images = _images(row)
    prompt = row["question"] if task != "grounding" else row["query"]
    targets = row.get("evidence_targets", row.get("targets", []))
    if task != "vqa" and (not isinstance(targets, list) or not targets):
        raise ValueError("Qwen export rows require one or more targets")
    target = _multitask_json_target(row, task, targets) if schema == "multitask_json" else _json_target(row, task, targets) if schema == "json" else _tag_target(row, task, targets)
    return {
        "task_id": row["task_id"],
        "task": task,
        "split": row["split"],
        "patient_id": row.get("patient_id"),
        "images": images,
        "prompt": prompt,
        "target": target,
        "answer_label": row.get("answer_label"),
        "label": row["label"],
        "targets": targets,
        "region_level": row.get("region_level"),
        "view_scope": row.get("view_scope"),
    }


def _tag_target(row: dict[str, Any], task: str, targets: list[dict[str, Any]]) -> str:
    if task == "vqa":
        return row["answer"]
    if task == "grounding":
        return f"The location of the {row['label']} is <REG>{_region(row['label'])}</REG>{format_view_bbx(targets)}"
    if isinstance(row.get("target"), dict):
        region = row["target"].get("region", {})
        region_name = region.get("anatomical_region") or row.get("label") or region.get("label")
        classification = row.get("answer_label")
        if classification not in {"normal", "abnormal"}:
            source = row["target"].get("classification", {}).get("label")
            classification = "abnormal" if source == "malignant" else "normal"
        return f"{row['answer']} <REG>{_region(region_name)}</REG><CLS>{classification}</CLS>{format_view_bbx(targets)}"
    return f"{row['answer']} {row['target']}"


def _json_target(row: dict[str, Any], task: str, targets: list[dict[str, Any]]) -> str:
    if task == "vqa":
        payload = {"answer": row["answer"], "class": row.get("answer_label")}
    elif task == "grounding":
        payload = {"region": _region(row["label"]), "boxes": _json_boxes(targets)}
    else:
        payload = {
            "answer": row["answer"],
            "region": _region(row["label"]),
            "class": row.get("answer_label"),
            "boxes": _json_boxes(targets),
        }
    return json.dumps(payload, separators=(",", ":"))


def _multitask_json_target(row: dict[str, Any], task: str, targets: list[dict[str, Any]]) -> str:
    if task == "vqa":
        label = row.get("answer_label")
        if label not in {"yes", "no"}:
            raise ValueError("R2 VQA rows require yes/no answer_label")
        payload = {"task": "vqa", "answer": row["answer"], "class": label}
    elif task == "grounding":
        payload = {"task": "grounding", "region": _region(row["label"]), "boxes": _json_boxes(targets)}
    else:
        label = row.get("answer_label")
        if label not in {"normal", "abnormal"}:
            raise ValueError("R2 grounded-VQA rows require normal/abnormal answer_label")
        payload = {
            "task": "grounded_vqa",
            "answer": row["answer"],
            "region": _region(row["label"]),
            "class": label,
            "boxes": _json_boxes(targets),
        }
    return json.dumps(payload, separators=(",", ":"))


def _json_boxes(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "view": "PST" if target["view"] == "POST" else target["view"],
            "bbox": [int(value) if float(value).is_integer() else value for value in target["bbox"]],
        }
        for target in targets
    ]


def _images(row: dict[str, Any]) -> list[dict[str, Any]]:
    images = row.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Qwen rows require one or two images")
    by_view = {image.get("view"): image for image in images if isinstance(image, dict)}
    if len(images) == 1 and set(by_view) in ({"ANT"}, {"POST"}):
        return [next(iter(by_view.values()))]
    if len(images) != 2:
        raise ValueError("Qwen rows require either one image or paired ANT and POST images")
    if set(by_view) != {"ANT", "POST"}:
        raise ValueError("Qwen rows require ANT and POST images")
    return [by_view["ANT"], by_view["POST"]]


def _region(label: str) -> str:
    return label.replace(" ", "_")


def _parse_box(raw: str) -> list[float] | None:
    try:
        values = [float(value.strip()) for value in raw.split(",")]
    except ValueError:
        return None
    if len(values) != 4 or values[0] >= values[2] or values[1] >= values[3]:
        return None
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paired WBBS manifests for Qwen3-VL.")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema", choices=("tags", "json", "multitask_json"), default="tags")
    parser.add_argument("--image-root", type=Path, help="Downloaded BS80K image root; required for merged R2 releases")
    args = parser.parse_args()
    print(json.dumps(build_qwen_exports(args.release, args.output, args.schema, args.image_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
