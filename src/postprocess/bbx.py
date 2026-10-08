from __future__ import annotations

import argparse
from pathlib import Path

from preprocess.canonical import read_jsonl, write_jsonl
from preprocess.tokens import format_bbx, format_view_bbx


def export_bbx_grounding(source: str | Path, output: str | Path) -> int:
    rows = []
    for row in read_jsonl(source):
        targets = row.get("targets")
        if isinstance(targets, list):
            target = format_view_bbx(targets)
            bboxes = targets
        else:
            bbox = row["bbox"]
            target = format_bbx(bbox)
            bboxes = [{"bbox": bbox}]
        rows.append(
            {
                "task_id": row["task_id"],
                **({"images": row["images"]} if "images" in row else {"image": row["image"]}),
                "query": row["query"],
                "label": row["label"],
                "target": target,
                "targets": bboxes,
                "evidence_id": row["evidence_id"],
                "split": row["split"],
                **({"source_label": row["source_label"]} if "source_label" in row else {}),
                **({"source_labels": row["source_labels"]} if "source_labels" in row else {}),
                **({"view_scope": row["view_scope"]} if "view_scope" in row else {}),
            }
        )
    write_jsonl(output, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export grounding rows with <BBX> coordinate targets.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_bbx_grounding(args.input, args.output))


if __name__ == "__main__":
    main()
