"""Export the primary WBBS-R3 manifest as a local workspace for manual figures.

The workspace is deliberately local: it copies inverted source images and must
never be included in a derived-artifact release because BS80K images remain
upstream-controlled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps


from project_paths import DATASETS, SOURCES

MANIFEST = DATASETS / "releases" / "r3" / "multitask.jsonl"
RAW_ROOT = SOURCES / "bs80k-imaging-raw"
DEFAULT_OUTPUT = DATASETS / "manual-figure-assets"
COLORS = {
    "abnormal": (220, 38, 38, 128),
    "positive": (220, 38, 38, 128),
    "malignant": (220, 38, 38, 128),
    "normal": (22, 163, 74, 128),
    "negative": (22, 163, 74, 128),
    "benign": (22, 163, 74, 128),
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", value).strip("._")


def normal_view(value: str | None) -> str:
    return {"ANT": "ANT", "ANTERIOR": "ANT", "POST": "POST", "POSTERIOR": "POST"}.get(
        (value or "").upper(), (value or "").upper()
    )


def source_path(image: str) -> Path:
    prefix, relative = image.split("/", 1)
    if prefix != "bs80k":
        raise ValueError(f"Unsupported source prefix: {prefix}")
    return RAW_ROOT / relative


def row_label(row: dict) -> str:
    classification = row.get("target", {}).get("classification", {}).get("label")
    return str(row.get("answer_label") or classification or row.get("effective_region_label") or "").lower()


def source_boxes(row: dict, view: str) -> list[dict]:
    candidates = [*row.get("targets", []), *row.get("evidence_targets", []), *row.get("target", {}).get("locations", [])]
    boxes, seen = [], set()
    for item in candidates:
        bbox = item.get("bbox") if isinstance(item, dict) else None
        if not bbox or normal_view(item.get("view")) != normal_view(view):
            continue
        key = tuple(bbox)
        if key not in seen:
            seen.add(key)
            boxes.append({"bbox_xyxy": list(bbox), "view": normal_view(view)})
    return boxes


def write_svg_overlay(path: Path, size: tuple[int, int], boxes: list[dict], label: str) -> None:
    red, green, blue, alpha = COLORS.get(label, (51, 65, 85, 128))
    rectangles = "".join(
        f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" />'
        for x1, y1, x2, y2 in (item["bbox_xyxy"] for item in boxes)
    )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size[0]}" height="{size[1]}" '
        f'viewBox="0 0 {size[0]} {size[1]}"><g fill="none" stroke="rgb({red},{green},{blue})" '
        f'stroke-opacity="{alpha / 255:.3f}" stroke-width="2">{rectangles}</g></svg>\n',
        encoding="utf-8",
    )


def read_rows(limit: int | None) -> list[dict]:
    rows = []
    with MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export(output: Path, limit: int | None) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to mix exports in existing folder: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(limit)
    grouped: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for row in rows:
        for image in row["images"]:
            grouped[image["image"]].append((row, image))

    index = []
    for ordinal, (image_key, items) in enumerate(sorted(grouped.items()), 1):
        source = source_path(image_key)
        if not source.exists():
            raise FileNotFoundError(source)
        image = ImageOps.invert(Image.open(source).convert("L"))
        folder_name = safe_name(Path(image_key).with_suffix("").as_posix())
        image_dir = output / "images" / folder_name
        instances_dir, overlays_dir = image_dir / "instances", image_dir / "overlays"
        instances_dir.mkdir(parents=True)
        overlays_dir.mkdir()
        image.save(image_dir / "image-inverted.png", format="PNG", compress_level=1)
        view = normal_view(items[0][1].get("view"))
        image_record = {
            "source_image": image_key,
            "raw_source_path": str(source),
            "view": view,
            "image_size": list(image.size),
            "instance_count": len(items),
            "note": "Local figure workspace only. Do not redistribute this raw-derived image.",
        }
        write_json(image_dir / "image.json", image_record)
        instance_records = []
        for row, image_meta in items:
            task_name = safe_name(row["task_id"])
            boxes = source_boxes(row, view)
            record = {
                "task_id": row["task_id"],
                "task_type": row["task_type"],
                "task_group": row.get("task_group"),
                "split": row["split"],
                "prompt": row.get("prompt"),
                "answer": row.get("answer"),
                "answer_label": row.get("answer_label"),
                "label": row.get("label"),
                "region_level": row.get("region_level"),
                "current_image": image_meta,
                "current_image_boxes": boxes,
                "all_linked_images": row["images"],
            }
            write_json(instances_dir / f"{task_name}.json", record)
            if boxes:
                write_svg_overlay(overlays_dir / f"{task_name}.svg", image.size, boxes, row_label(row))
            instance_records.append({"task_id": row["task_id"], "task_type": row["task_type"], "boxes": len(boxes)})
        write_json(image_dir / "instances.json", instance_records)
        index.append({"folder": f"images/{folder_name}", "source_image": image_key, "view": view, "instances": len(items)})
        if ordinal % 100 == 0 or ordinal == len(grouped):
            print(f"{ordinal}/{len(grouped)} source images exported")

    manifest_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    write_json(output / "index.json", {
        "workspace": "WBBS-R3 local manual-figure workspace",
        "primary_manifest": str(MANIFEST),
        "primary_manifest_sha256": manifest_hash,
        "source_image_count": len(index),
        "task_row_count": len(rows),
        "image_task_associations": sum(entry["instances"] for entry in index),
        "images": index,
    })
    (output / "README.md").write_text(
        "# WBBS-R3 local manual-figure workspace\n\n"
        "Each `images/<source-image>/` folder contains `image-inverted.png`, `image.json`, one JSON per primary-manifest task instance, and a transparent SVG with solid 2-pixel outlines for each instance with targets in that image. Red denotes abnormal/positive and green normal/negative labels. Paired task rows are deliberately present in both linked source-image folders. This workspace contains raw-derived images and is local only: do not include it in a public or Zenodo release.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="Export only the first N primary-manifest rows for a smoke test.")
    args = parser.parse_args()
    export(args.output, args.limit)


if __name__ == "__main__":
    main()
