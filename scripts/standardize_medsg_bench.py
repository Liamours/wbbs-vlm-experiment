"""Export MedSG-Bench text/coordinate labels to comparison-v1 JSONL.

The local MedSG snapshot deliberately contains JSON annotations only; image ZIP
archives are excluded.  This exporter therefore preserves every source image
reference but does not attempt to verify that an image is present locally.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from analysis.benchmark import standardize_rows


PARTITIONS = (("MedSG-Bench", "test"), ("MedSG-Train", "train"))
TASK_FILES = tuple(f"Task{number}.json" for number in range(1, 9))
ORDINALS = {
    "first": 0,
    "1st": 0,
    "1": 0,
    "second": 1,
    "2nd": 1,
    "2": 1,
    "third": 2,
    "3rd": 2,
    "3": 2,
    "fourth": 3,
    "4th": 3,
    "4": 3,
    "fifth": 4,
    "5th": 4,
    "5": 4,
    "sixth": 5,
    "6th": 5,
    "6": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("external/medsg-bench"),
        help="Directory containing MedSG-Bench/ and MedSG-Train/ JSON labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/benchmarks/medsg_bench_text_labels.standard.jsonl"),
        help="Comparison-v1 JSONL output path.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"Expected a JSON list of objects: {path}")
    return value


def target_index(task_filename: str, text: str) -> int:
    """Return the target-image position encoded by each MedSG task design."""

    task_number = int(re.search(r"\d+", task_filename).group())
    if task_number in {1, 6}:
        return 0  # registered difference / patch localization is on the source image
    if task_number in {2, 5}:
        return 1

    text = text.lower()
    mentions = re.findall(
        r"(?:in\s+(?:the\s+)?|image[-\s]?)(first|second|third|fourth|fifth|sixth|[1-6](?:st|nd|rd|th)?)"
        r"(?:\s+image)?",
        text,
    )
    if not mentions:
        raise ValueError(f"No target-image reference in {task_filename}: {text!r}")
    return ORDINALS[mentions[-1]]


def valid_box(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 4 or not all(isinstance(item, (int, float)) for item in value):
        raise ValueError(f"Expected [x1, y1, x2, y2] coordinate answer, received {value!r}")
    x1, y1, x2, y2 = (float(item) for item in value)
    if x2 < x1 or y2 < y1:
        raise ValueError(f"Invalid coordinate order in answer: {value!r}")
    return [x1, y1, x2, y2]


BOX_PATTERN = re.compile(
    r"<\|box_start\|>\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
    r"\s*,\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*<\|box_end\|>"
)


def conversation_pairs(row: dict[str, Any]) -> list[tuple[str, str, int]]:
    """Return every human/assistant pair from a MedSG-188K instruction row."""

    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        return []
    pairs: list[tuple[str, str, int]] = []
    for index, turn in enumerate(conversations):
        if not isinstance(turn, dict) or turn.get("from") != "human":
            continue
        question = str(turn.get("value") or "").replace("<image>\n", "").strip()
        if not question:
            raise ValueError(f"Empty human prompt in instruction record {row.get('id')!r}")
        response_turn = conversations[index + 1] if index + 1 < len(conversations) else None
        if not isinstance(response_turn, dict) or response_turn.get("from") != "gpt":
            raise ValueError(f"Human prompt has no following assistant response in {row.get('id')!r}")
        response = str(response_turn.get("value") or "").strip()
        if not response:
            raise ValueError(f"Empty assistant response in instruction record {row.get('id')!r}")
        pairs.append((question, response, len(pairs)))
    if not pairs:
        raise ValueError(f"No human/assistant pairs in instruction record {row.get('id')!r}")
    return pairs


def box_from_response(response: str) -> list[float]:
    match = BOX_PATTERN.search(response)
    if not match:
        raise ValueError(f"No <|box_start|> response coordinates found: {response!r}")
    return valid_box([float(value) for value in match.groups()])


def normalise_file(
    path: Path, partition: str, split: str
) -> list[dict[str, Any]]:
    source_rows = read_rows(path)
    prepared: list[dict[str, Any]] = []
    for source_index, row in enumerate(source_rows):
        images = row.get("images")
        if not isinstance(images, list) or not images or not all(isinstance(item, str) and item.strip() for item in images):
            raise ValueError(f"Missing image references in {path.name} row {source_index}")
        image_size = row.get("size")
        if not (isinstance(image_size, list) and len(image_size) == 2 and all(isinstance(value, (int, float)) for value in image_size)):
            image_size = None

        if isinstance(row.get("conversations"), list):
            annotations = [(question, response, box_from_response(response), pair_index) for question, response, pair_index in conversation_pairs(row)]
        else:
            question = str(row.get("question") or "").strip()
            if not question:
                raise ValueError(f"Missing question in {path.name} row {source_index}")
            answer_box = valid_box(row.get("answer"))
            annotations = [(question, None, answer_box, 0)]

        for question, response, answer_box, annotation_index in annotations:
            target_text = " ".join(part for part in (question, response or "", str(row.get("additional_info") or "")) if part)
            image_position = target_index(path.name, target_text)
            if image_position >= len(images):
                raise ValueError(
                    f"Target image {image_position + 1} is outside the {len(images)} images in {path.name} row {source_index}"
                )
            target_image = images[image_position].replace("\\", "/")
            prepared.append(
                {
                    # MedSG source IDs are not globally unique within every
                    # task file, so the source-row position is part of the
                    # deterministic comparison record ID.
                    "record_id": f"medsg-bench:{partition}:{path.stem.lower()}:{source_index}:{annotation_index}",
                    "split": split,
                    "task": "grounding",
                    "image_id": target_image,
                    "image_size": image_size,
                    "question": question,
                    "target_boxes": [{"bbox": answer_box, "image_id": target_image, "label": row.get("task")}],
                    "source_partition": partition,
                    "source_task_file": path.name,
                    "source_row_index": source_index,
                    "source_annotation_index": annotation_index,
                    "source_record_id": row.get("id"),
                    "source_task": row.get("task"),
                    "source_images": [image.replace("\\", "/") for image in images],
                    "source_target_image_position": image_position + 1,
                    "source_response": response,
                    "source_additional_info": row.get("additional_info"),
                    "source_need_format": row.get("need_format"),
                }
            )

    standard = standardize_rows(prepared, dataset_name="medsg-bench", task_hint="grounding")
    for standard_row, source_row in zip(standard, prepared, strict=True):
        # MedSG coordinates express cross-image correspondence, not a named
        # anatomical region.  Do not relabel that supervision as anatomy.
        standard_row["target_granularity"] = None
        standard_row["source"] = {
            "dataset": "medsg-bench",
            "partition": source_row["source_partition"],
            "task_file": source_row["source_task_file"],
            "row_index": source_row["source_row_index"],
            "annotation_index": source_row["source_annotation_index"],
            "record_id": source_row["source_record_id"],
            "task": source_row["source_task"],
            "images": source_row["source_images"],
            "target_image_position": source_row["source_target_image_position"],
            "additional_info": source_row["source_additional_info"],
            "need_format": source_row["source_need_format"],
            "response": source_row["source_response"],
            "image_archive_present": False,
        }
    return standard


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    expected = [source_dir / partition / filename for partition, _ in PARTITIONS for filename in TASK_FILES]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing MedSG JSON labels: " + ", ".join(str(path) for path in missing))

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    source_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    row_count = 0
    with partial.open("w", encoding="utf-8") as handle:
        for partition, split in PARTITIONS:
            for filename in TASK_FILES:
                standard = normalise_file(source_dir / partition / filename, partition, split)
                source_counts[split] += len(standard)
                task_counts.update(str(row["source"]["task"]) for row in standard)
                for row in standard:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                row_count += len(standard)
    partial.replace(output)

    print(
        json.dumps(
            {
                "dataset": "medsg-bench",
                "source_dir": str(source_dir),
                "output": str(output),
                "rows": row_count,
                "splits": dict(source_counts),
                "source_tasks": dict(sorted(task_counts.items())),
                "image_policy": "Text/label-only snapshot; source image references preserved without local image validation.",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
