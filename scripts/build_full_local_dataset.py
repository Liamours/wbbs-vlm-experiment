"""Assemble the full local ScintiGround-38K copy: frozen release manifests plus every referenced BS-80K image.

The output keeps BS-80K images, so it is a local working copy and is never published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from tqdm import tqdm


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_training_export(rows: list[dict], export_dir: Path) -> None:
    by_id = {row["task_id"]: row for row in rows}
    exported = [row for split in ("train", "val", "test") for row in read_jsonl(export_dir / f"multitask_{split}.jsonl")]
    if len(exported) != len(rows) or {row["task_id"] for row in exported} != set(by_id):
        raise ValueError(f"Training export task IDs differ from the release: {len(exported)} vs {len(rows)} rows")
    for row in exported:
        source = by_id[row["task_id"]]
        same_images = [Path(i["image"]).name for i in row["images"]] == [Path(i["image"]).name for i in source["images"]]
        if row["prompt"] != source["prompt"] or row["split"] != source["split"] or not same_images:
            raise ValueError(f"Training export row differs from the release: {row['task_id']}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True, help="Frozen R3 release directory")
    parser.add_argument("--image-root", type=Path, required=True, help="BS-80K bs80k-imaging-raw directory")
    parser.add_argument("--training-export", type=Path, required=True, help="Directory holding multitask_{train,val,test}.jsonl used for training")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")

    rows = read_jsonl(args.release / "multitask.jsonl")
    check_training_export(rows, args.training_export)
    images = sorted({image["image"] for row in rows for image in row["images"]})

    args.output.mkdir(parents=True)
    for source in sorted(p for p in args.release.iterdir() if p.is_file()):
        shutil.copy2(source, args.output / source.name)
    for portable in tqdm(images, desc="images", unit="img"):
        relative = Path(*Path(portable).parts[1:])
        target = args.output / portable
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.image_root / relative, target)

    files = sorted(p for p in args.output.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    lines = [f"{sha256(p)}  {p.relative_to(args.output).as_posix()}" for p in tqdm(files, desc="checksums", unit="file")]
    (args.output / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"rows {len(rows)} | images {len(images)} | files {len(files)} | output {args.output}")


if __name__ == "__main__":
    main()
