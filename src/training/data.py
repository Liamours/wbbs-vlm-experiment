from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


@dataclass(frozen=True)
class Example:
    images: tuple[Path, ...]
    prompt: str
    target: str


def load_examples(path: Path, task: str) -> list[Example]:
    rows = _read_jsonl(path)
    return [_to_example(row, task, path.parent) for row in rows]


def open_images(examples: Iterable[Example]) -> list[Image.Image]:
    result: list[Image.Image] = []
    for example in examples:
        opened = []
        for path in example.images:
            with Image.open(path) as source:
                opened.append(source.convert("RGB"))
        result.append(opened[0] if len(opened) == 1 else _compose_ant_post(opened))
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if raw.strip():
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    if not rows:
        raise ValueError(f"Training data is empty: {path}")
    return rows


def _to_example(row: dict[str, Any], task: str, root: Path) -> Example:
    images = _images(row, root)
    # open_images() always composes ANT+POST into one canvas, so exactly one
    # <image> token belongs at the start regardless of source image count.
    # PaliGemmaProcessor otherwise falls back to undocumented auto-inference
    # and warns on every call.
    image_token = "<image> "
    paired_prefix = "ANT is the left panel and POST is the right panel. " if len(images) == 2 else ""
    if task == "vqa":
        return Example(images=images, prompt=f"{image_token}answer {paired_prefix}{_required(row, 'prompt')}", target=_required(row, "target"))
    if task == "caption":
        return Example(images=images, prompt=f"{image_token}caption en {paired_prefix}", target=_required(row, "target"))
    if task == "grounding":
        return Example(images=images, prompt=f"{image_token}detect {paired_prefix}{_required(row, 'query')}", target=_grounding_target(row))
    if task == "grounded_vqa":
        return Example(images=images, prompt=f"{image_token}answer and detect {paired_prefix}{_required(row, 'prompt')}", target=_required(row, "target"))
    if task == "multitask":
        # Rows carry their own per-row task (vqa/grounding/grounded_vqa); the row's
        # own 'prompt' is already a complete natural-language instruction for that
        # task, so we only need to name the task, mirroring the Qwen3-VL multitask
        # trainer's "Task type: X." convention for a fair cross-model comparison.
        row_task = row.get("task")
        if row_task not in {"vqa", "grounding", "grounded_vqa"}:
            raise ValueError(f"Unsupported multitask row task: {row_task!r}")
        return Example(images=images, prompt=f"{image_token}{paired_prefix}Task type: {row_task}. {_required(row, 'prompt')}", target=_required(row, "target"))
    raise ValueError(f"Unsupported task: {task}")


def _images(row: dict[str, Any], root: Path) -> tuple[Path, ...]:
    if "images" in row:
        values = row["images"]
        views = [item.get("view") for item in values] if isinstance(values, list) else []
        if views not in (["ANT"], ["POST"], ["ANT", "POST"]):
            raise ValueError("Rows must contain ANT and/or POST images in view order")
        if views == ["ANT", "POST"]:
            paths = tuple(root / next(item["image"] for item in values if item["view"] == view) for view in ("ANT", "POST"))
        else:
            paths = (root / values[0]["image"],)
    else:
        paths = (root / _required(row, "image"),)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
    return paths


def _compose_ant_post(images: list[Image.Image]) -> Image.Image:
    if len(images) != 2:
        raise ValueError("Paired image composition requires ANT and POST images")
    width, height = sum(image.width for image in images), max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    offset = 0
    for image in images:
        canvas.paste(image, (offset, 0))
        offset += image.width
    return canvas


def _grounding_target(row: dict[str, Any]) -> str:
    box = row.get("box_1024")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("Grounding rows require box_1024=[x1,y1,x2,y2]")
    x1, y1, x2, y2 = box
    if not all(isinstance(value, int) and 0 <= value < 1024 for value in box):
        raise ValueError("box_1024 values must be integer coordinates in [0, 1023]")
    if x1 >= x2 or y1 >= y2:
        raise ValueError("box_1024 must have x1 < x2 and y1 < y2")
    label = _required(row, "label")
    return f"<loc{y1:04d}><loc{x1:04d}><loc{y2:04d}><loc{x2:04d}> {label}"


def _required(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Dataset row requires non-empty '{key}'")
    return value
