from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from preprocess.canonical import read_jsonl


def create_region_contact_sheet(paired_evidence_path: str | Path, output: str | Path, split: str = "test") -> Path:
    source = Path(paired_evidence_path)
    rows = [row for row in read_jsonl(source) if row["split"] == split]
    selected = _one_per_region(rows)
    cell_width, cell_height = 256, 520
    columns = 3
    canvas = Image.new("RGB", (columns * cell_width, ((len(selected) + columns - 1) // columns) * cell_height), color="white")
    for index, row in enumerate(selected):
        cell = _render_pair(row, source.parent, cell_width, cell_height)
        canvas.paste(cell, ((index % columns) * cell_width, (index // columns) * cell_height))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    return target


def _one_per_region(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_region: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_region.setdefault(row["region"], []).append(row)
    return [min(group, key=lambda row: _stable_key(row["evidence_id"])) for _, group in sorted(by_region.items())]


def _render_pair(row: dict[str, Any], root: Path, width: int, height: int) -> Image.Image:
    panel_height = height - 18
    result = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(result)
    draw.text((2, 2), row["region"], fill="black")
    images = {item["view"]: item for item in row["images"]}
    for index, view in enumerate(("ANT", "POST")):
        item = images[view]
        path = (root / item["image"]).resolve()
        with Image.open(path) as source:
            image = source.convert("RGB")
        panel_width = width // 2
        scaled = image.resize((panel_width, panel_height))
        result.paste(scaled, (index * panel_width, 18))
        x_scale, y_scale = panel_width / image.width, panel_height / image.height
        x1, y1, x2, y2 = item["bbox"]
        draw.rectangle((index * panel_width + x1 * x_scale, 18 + y1 * y_scale, index * panel_width + x2 * x_scale, 18 + y2 * y_scale), outline="red", width=2)
        draw.text((index * panel_width + 2, 20), view, fill="yellow")
    return result


def _stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one paired test example per anatomical region for visual QA.")
    parser.add_argument("--paired-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    args = parser.parse_args()
    print(create_region_contact_sheet(args.paired_evidence, args.output, split=args.split))


if __name__ == "__main__":
    main()
