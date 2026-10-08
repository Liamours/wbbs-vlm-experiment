"""Convert the archive's reformatted BS-80K box CSVs back to the original layout `preprocess.release` reads.

Reads `datasets/sources/downloads/bs80k/labels/{bone_region-bb,whole_body-bb}/bounding_boxes.csv.bz2` and writes
`bs80k-bone_region-bb/bounding_boxes.csv` and `bs80k-wholebody-bb/bounding_boxes.csv` under the dataset root.
`patient_NNNNN` is the zero-padded BS-80K ID, `anterior|posterior` become `ANT|POST`, and the region plus view form the component name.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from project_paths import SOURCES  # noqa: E402

VIEWS = {"anterior": "ANT", "posterior": "POST"}
REGION_COLUMNS = ["id", "component", "x", "y", "width", "height", "match_score", "label", "diagnosis", "duplicate_of_patient_id", "duplicate_of_sibling_component", "libs160k_duplicate_matches"]
WHOLE_BODY_COLUMNS = ["id", "view", "x", "y", "width", "height", "outlier", "likely_corrupt_image"]


def patient_id(value: str) -> str:
    return str(int(value.removeprefix("patient_")))


def convert(source: Path, target: Path, columns: list[str], row_map) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(source, "rt", encoding="utf-8", newline="") as reader, target.open("w", encoding="utf-8", newline="") as writer:
        out = csv.DictWriter(writer, fieldnames=columns, lineterminator="\n")
        out.writeheader()
        count = 0
        for row in tqdm(csv.DictReader(reader), desc=target.parent.name, unit="row"):
            out.writerow(row_map(row))
            count += 1
    return count


def region_row(row: dict[str, str]) -> dict[str, str]:
    duplicate = row["duplicate_of_patient_id"]
    return {
        **{key: row[key] for key in ("x", "y", "width", "height", "match_score", "label", "diagnosis", "duplicate_of_sibling_component", "libs160k_duplicate_matches")},
        "id": patient_id(row["patient_id"]),
        "component": row["region"] + VIEWS[row["view"]],
        "duplicate_of_patient_id": patient_id(duplicate) if duplicate.startswith("patient_") else duplicate,
    }


def whole_body_row(row: dict[str, str]) -> dict[str, str]:
    return {
        **{key: row[key] for key in ("x", "y", "width", "height", "outlier", "likely_corrupt_image")},
        "id": patient_id(row["patient_id"]),
        "view": VIEWS[row["view"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=SOURCES)
    args = parser.parse_args()
    labels = args.dataset_root / "downloads/bs80k/labels"
    regions = convert(labels / "bone_region-bb/bounding_boxes.csv.bz2", args.dataset_root / "bs80k-bone_region-bb/bounding_boxes.csv", REGION_COLUMNS, region_row)
    bodies = convert(labels / "whole_body-bb/bounding_boxes.csv.bz2", args.dataset_root / "bs80k-wholebody-bb/bounding_boxes.csv", WHOLE_BODY_COLUMNS, whole_body_row)
    print(f"region rows {regions} | whole-body rows {bodies} | root {args.dataset_root}")


if __name__ == "__main__":
    main()
