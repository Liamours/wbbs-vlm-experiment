"""Render deterministic, non-destructive visual-perturbation inputs for WBBS."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from preprocess.canonical import read_jsonl, write_jsonl


def _portable_source(image_root: Path, value: str) -> Path:
    parts = value.replace("\\", "/").split("/")
    if len(parts) < 3 or parts[0] != "bs80k":
        raise ValueError(f"Expected portable bs80k image reference, got: {value}")
    source = image_root.joinpath(*parts[1:])
    if not source.is_file():
        raise FileNotFoundError(f"Source image not found: {source}")
    return source


def _overlap(left: list[int], right: list[int]) -> bool:
    return max(left[0], right[0]) < min(left[2], right[2]) and max(left[1], right[1]) < min(left[3], right[3])


def _control_box(target: list[int], width: int, height: int) -> list[int]:
    box_width, box_height = target[2] - target[0], target[3] - target[1]
    candidates = [
        [0, 0, box_width, box_height],
        [width - box_width, 0, width, box_height],
        [0, height - box_height, box_width, height],
        [width - box_width, height - box_height, width, height],
        [(width - box_width) // 2, 0, (width - box_width) // 2 + box_width, box_height],
        [(width - box_width) // 2, height - box_height, (width - box_width) // 2 + box_width, height],
    ]
    for candidate in candidates:
        if candidate[0] >= 0 and candidate[1] >= 0 and candidate[2] <= width and candidate[3] <= height and not _overlap(candidate, target):
            return candidate
    raise ValueError(f"Cannot place a same-area non-target control for target {target} in {width}x{height}")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _render(source: Path, destination: Path, bbox: list[int]) -> None:
    with Image.open(source) as image:
        rendered = image.convert("RGB")
        ImageDraw.Draw(rendered).rectangle(bbox, fill="black")
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(destination, format="PNG")


def render(spec_file: Path, image_root: Path, output: Path) -> dict[str, int]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing perturbation output: {output}")
    specs = read_jsonl(spec_file)
    output.mkdir(parents=True)
    manifests = []
    images_rendered = 0
    for spec in specs:
        targets = {target["view"]: [int(value) for value in target["bbox"]] for target in spec["mask_boxes"]}
        source_images = {image["view"]: image for image in spec["images"]}
        if set(targets) != {"ANT", "POST"} or set(source_images) != {"ANT", "POST"}:
            raise ValueError(f"Visual perturbations require ANT+POST masks and images: {spec['perturbation_id']}")
        rendered_images: dict[str, list[dict[str, Any]]] = {"target_mask": [], "non_target_control": []}
        control_boxes = {}
        for view in ("ANT", "POST"):
            image = source_images[view]
            source = _portable_source(image_root, image["image"])
            width, height = image["image_size"]
            target = targets[view]
            if target[0] < 0 or target[1] < 0 or target[2] > width or target[3] > height:
                raise ValueError(f"Target mask exceeds image bounds: {spec['perturbation_id']} {view}")
            control = _control_box(target, width, height)
            control_boxes[view] = control
            stem = _slug(str(spec["perturbation_id"]))
            target_path = output / "images" / stem / "target_mask" / f"{view}.png"
            control_path = output / "images" / stem / "non_target_control" / f"{view}.png"
            _render(source, target_path, target)
            _render(source, control_path, control)
            rendered_images["target_mask"].append({"view": view, "image": target_path.relative_to(output).as_posix(), "image_size": [width, height]})
            rendered_images["non_target_control"].append({"view": view, "image": control_path.relative_to(output).as_posix(), "image_size": [width, height]})
            images_rendered += 2
        for condition, expected_class in (("target_mask", "normal"), ("non_target_control", "abnormal")):
            manifests.append(
                {
                    "perturbation_id": spec["perturbation_id"],
                    "task_id": f"{spec['perturbation_id']}:{condition}",
                    "condition": condition,
                    "patient_id": spec["patient_id"],
                    "source_task_id": spec["source_task_id"],
                    "prompt": spec["prompt"],
                    "images": rendered_images[condition],
                    "expected_class": expected_class,
                    "target_mask_boxes": targets,
                    "control_mask_boxes": control_boxes,
                }
            )
    write_jsonl(output / "visual_perturbations.jsonl", manifests)
    summary = {"specifications": len(specs), "conditions": len(manifests), "rendered_images": images_rendered}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Render target-mask and non-target-control WBBS visual perturbations.")
    parser.add_argument("--spec-file", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.spec_file, args.image_root, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
