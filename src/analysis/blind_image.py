"""Build the blank-image validation manifest for the blind-image check.

The manifest keeps every task id, prompt, label and target of the validation export and points each image reference at
a blank PNG of the same size. It is built from the validation split only and refuses any other split. Outputs go
under the configured output folder (results/analyses/blind-image), never under datasets/.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from project_paths import PROJECT_ROOT

MANIFEST_NAME = "multitask_val_blind.jsonl"
SMOKE_MANIFEST_NAME = "multitask_val_blind_smoke.jsonl"
PROVENANCE_NAME = "blind_manifest.provenance.json"


@dataclass(frozen=True)
class BlindImageConfig:
    split: str
    source_manifest: Path
    expected_rows: int
    output_dir: Path
    blank_fill: tuple[int, int, int]
    smoke_rows: int
    seed: int


def load_config(path: Path) -> BlindImageConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    fill = tuple(int(value) for value in raw["blank_fill"])
    if len(fill) != 3 or any(not 0 <= value <= 255 for value in fill):
        raise ValueError("blank_fill must be three integers between 0 and 255")
    return BlindImageConfig(
        split=str(raw["split"]),
        source_manifest=_project_path(raw["source_manifest"]),
        expected_rows=int(raw["expected_rows"]),
        output_dir=_project_path(raw["output_dir"]),
        blank_fill=fill,  # type: ignore[arg-type]
        smoke_rows=int(raw["smoke_rows"]),
        seed=int(raw["seed"]),
    )


def build_blind_manifest(config: BlindImageConfig, check_sizes: bool = True) -> dict[str, Any]:
    _refuse_locked_split(config)
    rows = _read_jsonl(config.source_manifest)
    _check_rows(rows, config)
    sizes = _declared_sizes(rows)
    if check_sizes:
        _check_real_sizes(rows, config.source_manifest.parent)

    images_dir = config.output_dir / "images"
    manifests_dir = config.output_dir / "manifests"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    blank_files = {size: _write_blank_png(images_dir, size, config.blank_fill) for size in sorted(sizes)}

    blind_rows = [_blind_row(row, blank_files) for row in rows]
    manifest_path = manifests_dir / MANIFEST_NAME
    _write_jsonl_once(manifest_path, blind_rows)

    smoke_indices = sorted(random.Random(config.seed).sample(range(len(rows)), min(config.smoke_rows, len(rows))))
    smoke_path = manifests_dir / SMOKE_MANIFEST_NAME
    _write_jsonl_once(smoke_path, [blind_rows[index] for index in smoke_indices])

    provenance = {
        "split": config.split,
        "source_manifest": _relative(config.source_manifest),
        "source_sha256": _sha256(config.source_manifest),
        "rows": len(blind_rows),
        "task_counts": dict(sorted(Counter(row["task"] for row in rows).items())),
        "splits_in_manifest": sorted({str(row["split"]) for row in blind_rows}),
        "image_references": sum(len(row["images"]) for row in rows),
        "blank_fill": list(config.blank_fill),
        "blank_images": {blank.name: {"size": list(size), "sha256": _sha256(blank)} for size, blank in blank_files.items()},
        "real_sizes_checked": check_sizes,
        "manifest": _relative(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "smoke_manifest": _relative(smoke_path),
        "smoke_rows": len(smoke_indices),
        "smoke_sha256": _sha256(smoke_path),
        "seed": config.seed,
        "built": date.today().isoformat(),
    }
    (manifests_dir / PROVENANCE_NAME).write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


def _refuse_locked_split(config: BlindImageConfig) -> None:
    if config.split != "val":
        raise ValueError(f"The blind-image check runs on the validation split only, not {config.split!r}")
    if "test" in config.source_manifest.name.lower():
        raise ValueError(f"Refusing a manifest named like the locked test split: {config.source_manifest}")
    if _is_under(config.output_dir, PROJECT_ROOT / "datasets"):
        raise ValueError("Blank images and manifests must not be written under datasets/")


def _check_rows(rows: list[dict[str, Any]], config: BlindImageConfig) -> None:
    splits = {row.get("split") for row in rows}
    if splits != {config.split}:
        raise ValueError(f"Source manifest must hold only the {config.split!r} split, found {sorted(map(str, splits))}")
    if len(rows) != config.expected_rows:
        raise ValueError(f"Source manifest has {len(rows)} rows, expected {config.expected_rows}")
    ids = [row["task_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Source manifest has duplicate task ids")


def _declared_sizes(rows: list[dict[str, Any]]) -> set[tuple[int, int]]:
    sizes: set[tuple[int, int]] = set()
    for row in rows:
        for image in row["images"]:
            size = image.get("image_size")
            if not isinstance(size, list) or len(size) != 2 or any(not isinstance(value, int) or value < 1 for value in size):
                raise ValueError(f"Missing or invalid image_size in {row['task_id']}")
            sizes.add((size[0], size[1]))
    return sizes


def _check_real_sizes(rows: list[dict[str, Any]], root: Path) -> None:
    problems: list[str] = []
    for row in tqdm(rows, desc="Checking image sizes", unit="row"):
        for image in row["images"]:
            path = root / image["image"]
            if not path.exists():
                problems.append(f"missing {path}")
                continue
            with Image.open(path) as handle:
                actual = handle.size
            if list(actual) != image["image_size"]:
                problems.append(f"{path}: file is {list(actual)}, manifest says {image['image_size']}")
    if problems:
        raise ValueError(f"{len(problems)} image problems, first: {problems[:3]}")


def _write_blank_png(images_dir: Path, size: tuple[int, int], fill: tuple[int, int, int]) -> Path:
    path = images_dir / f"blank_{size[0]}x{size[1]}.png"
    if path.exists():
        with Image.open(path) as existing:
            colors = existing.convert("RGB").getcolors(maxcolors=1)
            if existing.size != size or colors is None or colors[0][1] != fill:
                raise FileExistsError(f"Existing blank image does not match the config: {path}")
        return path
    Image.new("RGB", size, color=fill).save(path)
    return path


def _blind_row(row: dict[str, Any], blank_files: dict[tuple[int, int], Path]) -> dict[str, Any]:
    result = copy.deepcopy(row)
    for image in result["images"]:
        width, height = image["image_size"]
        # The manifest folder and the images folder are siblings, and both loaders resolve paths against the manifest folder.
        image["image"] = f"../images/{blank_files[(width, height)].name}"
    return result


def _write_jsonl_once(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite a different manifest: {path}")
        return
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-size-check", action="store_true", help="Trust the image_size fields instead of reading every real image header.")
    args = parser.parse_args()
    provenance = build_blind_manifest(load_config(args.config), check_sizes=not args.skip_size_check)
    print(json.dumps({key: provenance[key] for key in ("manifest", "rows", "smoke_rows", "manifest_sha256", "splits_in_manifest")}, sort_keys=True))


if __name__ == "__main__":
    main()
