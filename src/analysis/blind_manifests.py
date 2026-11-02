"""Black-image copies of the training and test exports, for the model trained without images.

Every row keeps its task id, prompt, label and target; every image reference points at a black PNG of the same size.
The copies go to results/analyses/blind-image/manifests, never under datasets/. The validation copy is built by
analysis.blind_image; this module builds the training copy and the test copy, and the test copy is used to score the
blind-trained model once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_paths import PROJECT_ROOT

from .blind_image import _blind_row, _declared_sizes, _read_jsonl, _sha256, _write_blank_png, _write_jsonl_once

OUTPUT = PROJECT_ROOT / "results" / "analyses" / "blind-image"


def build(source: Path, name: str, expected_rows: int, fill: tuple[int, int, int] = (0, 0, 0), output: Path = OUTPUT) -> dict:
    if str(output.resolve()).startswith(str((PROJECT_ROOT / "datasets").resolve())):
        raise ValueError("Black images and manifests must not be written under datasets/")
    rows = _read_jsonl(source)
    if len(rows) != expected_rows:
        raise ValueError(f"{source.name} has {len(rows)} rows, expected {expected_rows}")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate task ids")
    images = output / "images"
    manifests = output / "manifests"
    images.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    blanks = {size: _write_blank_png(images, size, fill) for size in sorted(_declared_sizes(rows))}
    path = manifests / f"{name}.jsonl"
    _write_jsonl_once(path, [_blind_row(row, blanks) for row in rows])
    record = {"source": source.name, "source_sha256": _sha256(source), "rows": len(rows), "manifest": path.name, "manifest_sha256": _sha256(path), "blank_sizes": sorted(f"{w}x{h}" for w, h in blanks)}
    (manifests / f"{name}.provenance.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.name, args.expected_rows), sort_keys=True))


if __name__ == "__main__":
    main()
