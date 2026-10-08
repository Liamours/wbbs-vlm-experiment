"""Export the local PathVQA Hugging Face mirror to comparison-v1 JSONL.

PathVQA images are embedded in the source Parquet files.  The export keeps a
stable virtual image ID (Parquet shard plus original embedded image path) and
does not duplicate the image bytes on disk.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from analysis.benchmark import standardize_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("external/pathvqa"),
        help="PathVQA mirror directory containing data/*.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/benchmarks/pathvqa.standard.jsonl"),
        help="Comparison-v1 JSONL output path.",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Parquet rows to normalize at once.")
    return parser.parse_args()


def split_for_file(path: Path) -> str:
    if path.name.startswith("train-"):
        return "train"
    if path.name.startswith("test-"):
        return "test"
    if path.name.startswith("validation-"):
        return "validation"
    raise ValueError(f"Cannot infer source split from {path.name}")


def require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise SystemExit(
            "PyArrow is required for Parquet input. Run: uv run --with pyarrow python scripts/standardize_pathvqa.py"
        ) from error
    return pq


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    source_dir = args.source_dir.resolve()
    files = sorted((source_dir / "data").glob("*.parquet"))
    if len(files) != 13:
        raise FileNotFoundError(f"Expected 13 PathVQA Parquet shards in {source_dir / 'data'}, found {len(files)}")

    pq = require_pyarrow()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    splits: Counter[str] = Counter()
    rows_written = 0

    with partial.open("w", encoding="utf-8") as handle:
        for source_file in files:
            split = split_for_file(source_file)
            relative_file = source_file.relative_to(source_dir).as_posix()
            parquet = pq.ParquetFile(source_file)
            source_index = 0
            # Read only image.path: embedded JPEG bytes can remain in Parquet.
            for batch in parquet.iter_batches(batch_size=args.batch_size, columns=["image.path", "question", "answer"]):
                prepared: list[dict[str, Any]] = []
                for row in batch.to_pylist():
                    image = row.get("image")
                    image_path = image.get("path") if isinstance(image, dict) else None
                    if not isinstance(image_path, str) or not image_path.strip():
                        raise ValueError(f"Missing embedded image path in {relative_file} row {source_index}")
                    image_id = f"{relative_file}::{image_path.replace('\\', '/')}"
                    prepared.append(
                        {
                            "record_id": f"pathvqa:{split}:{relative_file}:{source_index}",
                            "split": split,
                            "task": "vqa",
                            "image_id": image_id,
                            "question": row.get("question"),
                            "answer": row.get("answer"),
                            "source_parquet": relative_file,
                            "source_row_index": source_index,
                            "source_embedded_image_path": image_path.replace("\\", "/"),
                        }
                    )
                    source_index += 1

                standard = standardize_rows(prepared, dataset_name="pathvqa", task_hint="vqa")
                for standard_row, source_row in zip(standard, prepared, strict=True):
                    standard_row["source"] = {
                        "dataset": "pathvqa",
                        "parquet": source_row["source_parquet"],
                        "row_index": source_row["source_row_index"],
                        "embedded_image_path": source_row["source_embedded_image_path"],
                        "image_storage": "embedded_in_parquet",
                    }
                    handle.write(json.dumps(standard_row, ensure_ascii=False, sort_keys=True) + "\n")
                rows_written += len(standard)
                splits[split] += len(standard)

    partial.replace(output)
    print(
        json.dumps(
            {
                "dataset": "pathvqa",
                "source_dir": str(source_dir),
                "output": str(output),
                "rows": rows_written,
                "splits": dict(splits),
                "image_policy": "Image bytes remain embedded in the source Parquet shards; no duplicate extraction was performed.",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
