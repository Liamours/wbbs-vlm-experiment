"""Replace vector figure-workspace boxes with full-size raster overlays."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw


from project_paths import DATASETS

WORKSPACE = DATASETS / "manual-figure-assets"
MANIFEST = DATASETS / "releases" / "r3" / "multitask.jsonl"
COLORS = {
    "abnormal": (220, 38, 38, 128), "positive": (220, 38, 38, 128), "malignant": (220, 38, 38, 128),
    "normal": (22, 163, 74, 128), "negative": (22, 163, 74, 128), "benign": (22, 163, 74, 128),
}
BOX_LABELS = {
    row["task_id"]: str(
        row.get("answer_label")
        or row.get("target", {}).get("classification", {}).get("label")
        or row.get("effective_region_label")
        or ""
    ).lower()
    for row in (json.loads(line) for line in MANIFEST.open(encoding="utf-8"))
}


def render_folder(folder: Path) -> int:
    base = Image.open(folder / "image-inverted.png").convert("RGBA")
    made = 0
    for instance_path in (folder / "instances").glob("*.json"):
        record = json.loads(instance_path.read_text(encoding="utf-8"))
        boxes = record["current_image_boxes"]
        if not boxes:
            continue
        output = folder / "overlays" / f"{instance_path.stem}.png"
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        color = COLORS.get(BOX_LABELS.get(record["task_id"], ""), (51, 65, 85, 128))
        for item in boxes:
            draw.rectangle(item["bbox_xyxy"], outline=color, width=2)
        Image.alpha_composite(base, overlay).convert("RGB").save(output, format="PNG", compress_level=1)
        made += 1
        svg = folder / "overlays" / f"{instance_path.stem}.svg"
        if svg.exists():
            svg.unlink()
    return made


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    folders = list((args.workspace / "images").glob("*"))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, made in enumerate(pool.map(render_folder, folders), 1):
            if index % 100 == 0 or index == len(folders):
                print(f"{index}/{len(folders)} folders; {made} new PNG overlays in latest folder")


if __name__ == "__main__":
    main()
