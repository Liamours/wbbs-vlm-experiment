"""Export HEAL-MedVQA's configured Parquet splits to comparison-v1 JSONL.

The configured Hugging Face release consists of root-level ``train-*.parquet``
and ``test-*.parquet`` shards with images embedded in Parquet.  Each source
mask is start/length RLE in column-major image order.  This exporter converts
that pixel supervision to its tight ``[x1, y1, x2, y2]`` box while retaining
the source mask metadata for traceability.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from analysis.benchmark import standardize_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("external/heal-medvqa"),
        help="HEAL-MedVQA release directory containing root train/test Parquet shards.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/benchmarks/heal_medvqa.standard.jsonl"),
        help="Comparison-v1 JSONL output path.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Parquet rows to normalize at once.")
    return parser.parse_args()


def require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise SystemExit(
            "PyArrow is required for Parquet input. Run: uv run --with pyarrow python scripts/standardize_heal_medvqa.py"
        ) from error
    return pq


def source_shards(source_dir: Path) -> list[Path]:
    files = sorted((*source_dir.glob("train-*.parquet"), *source_dir.glob("test-*.parquet")))
    train = [path for path in files if path.name.startswith("train-")]
    test = [path for path in files if path.name.startswith("test-")]
    if len(train) != 33 or len(test) != 6:
        raise FileNotFoundError(
            f"Expected 33 train and 6 test root Parquet shards in {source_dir}; found {len(train)} train and {len(test)} test."
        )
    return files


def rle_to_box(rle: Iterable[Any], height: Any, width: Any) -> tuple[list[float] | None, int]:
    """Convert HEAL's column-major absolute-start RLE to an xyxy bounding box."""

    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        raise ValueError(f"Invalid mask dimensions: height={height!r}, width={width!r}")
    values = list(rle) if isinstance(rle, (list, tuple)) else []
    if len(values) % 2:
        raise ValueError("HEAL mask_rle must contain alternating absolute starts and run lengths")

    min_x, min_y = width, height
    max_x = max_y = -1
    pixel_count = 0
    for raw_start, raw_length in zip(values[::2], values[1::2], strict=True):
        if not isinstance(raw_start, int) or not isinstance(raw_length, int) or raw_start < 0 or raw_length < 0:
            raise ValueError(f"Invalid RLE run: start={raw_start!r}, length={raw_length!r}")
        if raw_length == 0:
            continue
        end = raw_start + raw_length
        if end > height * width:
            raise ValueError(f"RLE run exceeds mask bounds: start={raw_start}, length={raw_length}, mask={width}x{height}")
        first_x, first_y = divmod(raw_start, height)
        last_x, last_y = divmod(end - 1, height)
        min_x = min(min_x, first_x)
        max_x = max(max_x, last_x)
        if first_x == last_x:
            min_y = min(min_y, first_y, last_y)
            max_y = max(max_y, first_y, last_y)
        else:
            # A column-major contiguous run crossing columns covers the lower
            # end of its first column, whole intermediate columns, and the
            # upper end of its final column.
            min_y = 0
            max_y = height - 1
        pixel_count += raw_length

    if max_x < 0:
        return None, 0
    return [float(min_x), float(min_y), float(max_x + 1), float(max_y + 1)], pixel_count


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    source_dir = args.source_dir.resolve()
    files = source_shards(source_dir)
    pq = require_pyarrow()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    split_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    missing_masks = 0
    rows_written = 0

    columns = [
        "image.path",
        "image_id",
        "question_id",
        "question",
        "answer",
        "anatomy",
        "question_type",
        "mask_rle",
        "mask_h",
        "mask_w",
    ]
    with partial.open("w", encoding="utf-8") as handle:
        for source_file in files:
            split = "train" if source_file.name.startswith("train-") else "test"
            source_index = 0
            parquet = pq.ParquetFile(source_file)
            for batch in parquet.iter_batches(batch_size=args.batch_size, columns=columns):
                prepared: list[dict[str, Any]] = []
                for row in batch.to_pylist():
                    image = row.get("image")
                    image_path = image.get("path") if isinstance(image, dict) else None
                    image_id = row.get("image_id")
                    question_id = row.get("question_id")
                    if not isinstance(image_id, str) or not image_id.strip() or not isinstance(question_id, str) or not question_id.strip():
                        raise ValueError(f"Missing image_id or question_id in {source_file.name} row {source_index}")
                    if not isinstance(image_path, str) or not image_path.strip():
                        raise ValueError(f"Missing embedded image path in {source_file.name} row {source_index}")
                    box, mask_pixels = rle_to_box(row.get("mask_rle"), row.get("mask_h"), row.get("mask_w"))
                    if box is None:
                        missing_masks += 1
                    prepared.append(
                        {
                            "record_id": f"heal-medvqa:{question_id}",
                            "split": split,
                            "task": "grounded_vqa",
                            "image_id": image_id,
                            "image_size": [row["mask_w"], row["mask_h"]],
                            "question": row.get("question"),
                            "answer": row.get("answer"),
                            "region": row.get("anatomy"),
                            "target_boxes": ([{"bbox": box, "image_id": image_id, "label": row.get("anatomy")} ] if box else []),
                            "source_shard": source_file.name,
                            "source_row_index": source_index,
                            "source_image_path": image_path.replace("\\", "/"),
                            "source_question_id": question_id,
                            "source_question_type": row.get("question_type"),
                            "source_mask_height": row.get("mask_h"),
                            "source_mask_width": row.get("mask_w"),
                            "source_mask_rle_runs": len(row.get("mask_rle") or []) // 2,
                            "source_mask_pixels": mask_pixels,
                        }
                    )
                    source_index += 1

                standard = standardize_rows(prepared, dataset_name="heal-medvqa", task_hint="grounded_vqa")
                for standard_row, source_row in zip(standard, prepared, strict=True):
                    standard_row["source"] = {
                        "dataset": "heal-medvqa",
                        "shard": source_row["source_shard"],
                        "row_index": source_row["source_row_index"],
                        "question_id": source_row["source_question_id"],
                        "question_type": source_row["source_question_type"],
                        "embedded_image_path": source_row["source_image_path"],
                        "image_storage": "embedded_in_parquet",
                        "mask_encoding": "column_major_absolute_start_length_rle",
                        "mask_height": source_row["source_mask_height"],
                        "mask_width": source_row["source_mask_width"],
                        "mask_runs": source_row["source_mask_rle_runs"],
                        "mask_pixels": source_row["source_mask_pixels"],
                        "box_derivation": "tight_box_from_source_mask_rle",
                    }
                    handle.write(json.dumps(standard_row, ensure_ascii=False, sort_keys=True) + "\n")
                    type_counts[str(source_row["source_question_type"])] += 1
                rows_written += len(standard)
                split_counts[split] += len(standard)

    partial.replace(output)
    print(
        json.dumps(
            {
                "dataset": "heal-medvqa",
                "source_dir": str(source_dir),
                "output": str(output),
                "rows": rows_written,
                "splits": dict(split_counts),
                "question_types": dict(type_counts),
                "empty_source_masks": missing_masks,
                "grounding_policy": "Each source column-major RLE mask was converted to its tight xyxy bounding box; image bytes remain embedded in Parquet.",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
