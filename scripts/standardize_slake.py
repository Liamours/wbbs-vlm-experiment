"""Export the local SLAKE archive to the repository's comparison-v1 JSONL format.

The source release has VQA records plus co-released masks/boxes, but it does not
publish a per-question box/mask link.  Consequently this exporter labels each
row as ``vqa`` and retains source location and modality as metadata rather than
inventing visual-grounding targets.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from analysis.benchmark import standardize_rows


SPLITS = {
    "train": "train.json",
    "validation": "validate.json",
    "test": "test.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("external/slake/raw"),
        help="Extracted SLAKE root containing imgs/, train.json, validate.json, and test.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/benchmarks/slake.standard.jsonl"),
        help="Comparison-v1 JSONL output path.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Expected a JSON list of objects: {path}")
    return value


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"SLAKE source directory not found: {source_dir}")

    normalized_source_rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    source_counts: Counter[str] = Counter()

    for split, filename in SPLITS.items():
        rows = load_rows(source_dir / filename)
        source_counts[split] = len(rows)
        for source_index, row in enumerate(rows):
            image_name = str(row.get("img_name") or "").strip().replace("\\", "/")
            if not image_name:
                raise ValueError(f"Missing img_name in {filename} row {source_index}")
            image_id = f"imgs/{image_name}"
            if not (source_dir / image_id).is_file():
                missing_images.append(image_id)

            qid = row.get("qid")
            record_id = f"slake:{split}:{qid}" if qid is not None else f"slake:{split}:{source_index}"
            normalized_source_rows.append(
                {
                    "record_id": record_id,
                    "split": split,
                    "task": "vqa",
                    "image_id": image_id,
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "region": row.get("location"),
                    "modality": row.get("modality"),
                    "source_qid": qid,
                    "source_img_id": row.get("img_id"),
                    "source_question_language": row.get("q_lang"),
                    "source_answer_type": row.get("answer_type"),
                    "source_base_type": row.get("base_type"),
                    "source_content_type": row.get("content_type"),
                    "source_triple": row.get("triple"),
                    "source_split": split,
                    "source_row_index": source_index,
                }
            )

    if missing_images:
        examples = ", ".join(sorted(set(missing_images))[:5])
        raise FileNotFoundError(f"{len(missing_images)} source image references are missing (examples: {examples})")

    standard_rows = standardize_rows(normalized_source_rows, dataset_name="slake", task_hint="vqa")
    for standard_row, source_row in zip(standard_rows, normalized_source_rows, strict=True):
        standard_row["source"] = {
            "dataset": "slake",
            "source_split": source_row["source_split"],
            "source_row_index": source_row["source_row_index"],
            "source_qid": source_row["source_qid"],
            "source_img_id": source_row["source_img_id"],
            "source_question_language": source_row["source_question_language"],
            "source_answer_type": source_row["source_answer_type"],
            "source_base_type": source_row["source_base_type"],
            "source_content_type": source_row["source_content_type"],
            "source_triple": source_row["source_triple"],
        }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in standard_rows),
        encoding="utf-8",
    )

    summary = {
        "dataset": "slake",
        "source_dir": str(source_dir),
        "output": str(output),
        "rows": len(standard_rows),
        "source_splits": dict(source_counts),
        "export_splits": dict(Counter(str(row["split"]) for row in standard_rows)),
        "tasks": dict(Counter(str(row["task"]) for row in standard_rows)),
        "referenced_images": len({str(row["image_id"]) for row in standard_rows}),
        "grounding_policy": "VQA only: source boxes/masks have no verified per-question target alignment.",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
