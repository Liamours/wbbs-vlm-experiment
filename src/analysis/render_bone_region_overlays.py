"""Render every BS80K scan with reliable template-matched region boxes."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


from project_paths import SOURCES as DATASET
RAW = DATASET / "bs80k-imaging-raw"
BOXES = DATASET / "bs80k-bone_region-bb" / "bounding_boxes.csv"
OUTPUTS = {"ANT": DATASET / "bs80k-bone_region-bb" / "anterior-png", "POST": DATASET / "bs80k-bone_region-bb" / "posterior-png"}
RED, GREEN = (220, 38, 38, 128), (22, 163, 74, 128)


def load_boxes(include_shoulders: bool) -> dict[tuple[str, str], list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]]:
    grouped = defaultdict(list)
    with BOXES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            component = row["component"]
            if component.startswith("sho") and not include_shoulders:
                continue
            view = "ANT" if component.endswith("ANT") else "POST"
            x, y, width, height = (int(row[key]) for key in ("x", "y", "width", "height"))
            color = RED if row["diagnosis"].lower() == "abnormal" else GREEN
            grouped[(row["id"], view)].append(((x, y, x + width, y + height), color))
    return grouped


def render(view: str, image_path: Path, annotations: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]], output_path: Path) -> None:
    image = ImageOps.invert(Image.open(image_path).convert("L")).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for bbox, color in annotations:
        draw.rectangle(bbox, outline=color, width=2)
    Image.alpha_composite(image, layer).convert("RGB").save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-shoulders", action="store_true", help="Include known low-precision shoulder matches.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--view", choices=("ANT", "POST"), help="Render one view only.")
    args = parser.parse_args()
    annotations = load_boxes(args.include_shoulders)
    rendered = 0
    sources = (("ANT", RAW / "wholeBodyANT"), ("POST", RAW / "wholeBodyPOST"))
    for view, source_folder in sources:
        if args.view and view != args.view:
            continue
        output_folder = OUTPUTS[view]
        output_folder.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(source_folder.glob("*.jpg"), key=lambda path: int(path.stem)):
            output_path = output_folder / f"{image_path.stem}.png"
            if output_path.exists() and not args.overwrite:
                continue
            render(view, image_path, annotations[(image_path.stem, view)], output_path)
            rendered += 1
    print(f"Rendered {rendered} PNG files. Shoulders included: {args.include_shoulders}.")


if __name__ == "__main__": main()
