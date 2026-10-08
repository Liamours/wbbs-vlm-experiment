from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def build_source_inventory(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    entries = [
        _file_entry(root, "region_boxes", "bs80k-bone_region-bb/bounding_boxes.csv", csv_rows=True),
        _file_entry(root, "whole_body_boxes", "bs80k-wholebody-bb/bounding_boxes.csv", csv_rows=True),
        _tree_entry(root, "nidus_annotations_ant", "bs80k-lesion-bb/ant", ".xml"),
        _tree_entry(root, "nidus_annotations_post", "bs80k-lesion-bb/post", ".xml"),
        _tree_entry(root, "whole_body_images_ant", "bs80k-imaging-raw/wholeBodyANT", ".jpg", digest=False),
        _tree_entry(root, "whole_body_images_post", "bs80k-imaging-raw/wholeBodyPOST", ".jpg", digest=False),
        _file_entry(root, "libs_text_train", "libs160k-imaging-raw/LIBS-160K-EN/train/train_texts.jsonl", json_rows=True),
        _file_entry(root, "libs_text_test", "libs160k-imaging-raw/LIBS-160K-EN/train/test_texts.jsonl", json_rows=True),
        _file_entry(root, "libs_text_valid", "libs160k-imaging-raw/LIBS-160K-EN/valid/valid_texts.jsonl", json_rows=True),
    ]
    return {
        "schema_version": "wbbs-source-inventory/v1",
        "dataset_root": str(root),
        "sources": entries,
    }


def write_source_inventory(dataset_root: str | Path, output: str | Path) -> dict[str, Any]:
    inventory = build_source_inventory(dataset_root)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return inventory


def _file_entry(root: Path, name: str, relative: str, csv_rows: bool = False, json_rows: bool = False) -> dict[str, Any]:
    path = root / relative
    entry: dict[str, Any] = {"name": name, "path": relative, "exists": path.is_file()}
    if not path.is_file():
        return entry
    entry["bytes"] = path.stat().st_size
    entry["sha256"] = _sha256_file(path)
    if csv_rows:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            entry["rows"] = sum(1 for _ in csv.DictReader(handle))
    if json_rows:
        entry["rows"] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return entry


def _tree_entry(root: Path, name: str, relative: str, suffix: str, digest: bool = True) -> dict[str, Any]:
    path = root / relative
    entry: dict[str, Any] = {"name": name, "path": relative, "exists": path.is_dir()}
    if not path.is_dir():
        return entry
    files = sorted(item for item in path.rglob(f"*{suffix}") if item.is_file())
    entry["files"] = len(files)
    entry["bytes"] = sum(item.stat().st_size for item in files)
    if digest:
        hasher = hashlib.sha256()
        for item in files:
            hasher.update(item.relative_to(path).as_posix().encode("utf-8"))
            hasher.update(b"\0")
            with item.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
        entry["sha256"] = hasher.hexdigest()
    return entry


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
