from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from preprocess.canonical import read_jsonl, write_jsonl


def export_paligemma(rows: Iterable[dict[str, Any]], task: str, split: str | None = None) -> list[dict[str, Any]]:
    exported, _ = export_paligemma_with_rejections(rows, task, split=split)
    return exported


def export_paligemma_with_rejections(
    rows: Iterable[dict[str, Any]], task: str, split: str | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if task not in {"vqa", "grounding", "grounded_vqa"}:
        raise ValueError("task must be vqa, grounding, or grounded_vqa")
    exported, rejected = [], []
    for row in rows:
        if split and row.get("split") != split:
            continue
        try:
            exported.append(_export_row(row, task))
        except _UnsupportedRow as error:
            rejected.append({"task_id": row.get("task_id"), "reason": str(error), "split": row.get("split")})
    return exported, rejected


def _export_row(row: dict[str, Any], task: str) -> dict[str, Any]:
    if task == "vqa":
        result = {**_image_fields(row), "task_id": _task_id(row), "prompt": row["question"], "target": _answer_text(row["answer"])}
        if isinstance(row.get("answer_label"), str):
            result["answer_label"] = row["answer_label"]
        return result
    if task == "grounding":
        box_1024, target_view = _target_box_1024(row, "targets")
        return {
            **_image_fields(row),
            "task_id": _task_id(row),
            "query": row["query"],
            "label": row["label"],
            "target_view": target_view,
            "box_1024": box_1024,
        }
    box_1024, target_view = _target_box_1024(row, "evidence_targets")
    label = row["label"].replace(" ", "_")
    diagnosis = row["answer_label"]
    if diagnosis not in {"normal", "abnormal"}:
        raise ValueError("grounded VQA answer_label must be normal or abnormal")
    return {
        **_image_fields(row),
        "task_id": _task_id(row),
        "prompt": row["question"],
        "target": f"class={diagnosis} {_location_text(box_1024)} region={label}",
        "answer_label": diagnosis,
        "label": row["label"],
        "target_view": target_view,
        "box_1024": box_1024,
    }


def _image_fields(row: dict[str, Any]) -> dict[str, Any]:
    if "images" not in row:
        return {"image": row["image"], "image_layout": "single"}
    images = row["images"]
    if isinstance(images, list) and len(images) == 1 and str(row.get("region_level", "")).lower() == "whole_body":
        image = images[0]
        if not isinstance(image, dict):
            raise ValueError("Whole-body image must be an object")
        return {"image": image["image"], "image_layout": "single", "image_size": image["image_size"]}
    if not isinstance(images, list) or len(images) != 2:
        raise ValueError("Paired export requires exactly two images")
    by_view = {item.get("view"): item for item in images if isinstance(item, dict)}
    if set(by_view) != {"ANT", "POST"}:
        raise ValueError("Paired export requires ANT and POST images")
    return {
        "images": [{"view": view, "image": by_view[view]["image"], "image_size": by_view[view]["image_size"]} for view in ("ANT", "POST")],
        "image_layout": "ant_post_horizontal",
        "composite_image_size": _composite_size(by_view),
    }


def _target_box_1024(row: dict[str, Any], field: str) -> tuple[list[int], str | None]:
    if field not in row:
        if "bbox" not in row or "image_size" not in row:
            raise ValueError("Grounding rows require target boxes")
        return normalize_box_1024(row["bbox"], row["image_size"]), row.get("view")
    targets = row[field]
    if not isinstance(targets, list) or len(targets) != 1:
        raise _UnsupportedRow("multi_target_not_supported_by_paligemma_baseline")
    target = targets[0]
    if "images" not in row:
        return normalize_box_1024(target["bbox"], row["image_size"]), target.get("view")
    by_view = {image["view"]: image for image in row["images"]}
    view = target.get("view")
    if view not in by_view:
        raise ValueError("Target view is not part of paired images")
    ant_width = by_view["ANT"]["image_size"][0]
    canvas = _composite_size(by_view)
    x1, y1, x2, y2 = target["bbox"]
    offset = ant_width if view == "POST" else 0
    return normalize_box_1024([x1 + offset, y1, x2 + offset, y2], canvas), view


def _composite_size(images_by_view: dict[str, dict[str, Any]]) -> list[int]:
    ant_size, post_size = images_by_view["ANT"]["image_size"], images_by_view["POST"]["image_size"]
    return [ant_size[0] + post_size[0], max(ant_size[1], post_size[1])]


def normalize_box_1024(box: list[float], image_size: list[int]) -> list[int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    scaled = [
        round(x1 / width * 1023),
        round(y1 / height * 1023),
        round(x2 / width * 1023),
        round(y2 / height * 1023),
    ]
    return [min(1023, max(0, value)) for value in scaled]


def _location_text(box: list[int]) -> str:
    x1, y1, x2, y2 = box
    return f"<loc{y1:04d}><loc{x1:04d}><loc{y2:04d}><loc{x2:04d}>"


def _task_id(row: dict[str, Any]) -> str:
    value = row.get("task_id", row.get("record_id"))
    if not isinstance(value, str) or not value:
        raise ValueError("Rows require task_id or record_id")
    return value


class _UnsupportedRow(ValueError):
    pass


def _answer_text(answer: Any) -> str:
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list) and all(isinstance(item, str) for item in answer):
        return ", ".join(answer)
    raise ValueError("VQA answer must be text or a text list")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export task manifests to the PaliGemma training format.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("vqa", "grounding", "grounded_vqa"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--rejected-output", type=Path)
    args = parser.parse_args()
    exported, rejected = export_paligemma_with_rejections(read_jsonl(args.input), args.task, split=args.split)
    write_jsonl(args.output, exported)
    rejected_output = args.rejected_output or args.output.with_name(f"{args.output.stem}_rejected.jsonl")
    if rejected:
        write_jsonl(rejected_output, rejected)
    else:
        rejected_output.unlink(missing_ok=True)
    print(json.dumps({"rows": len(exported), "rejected": len(rejected), "rejected_output": str(rejected_output) if rejected else None}, sort_keys=True))


if __name__ == "__main__":
    main()
