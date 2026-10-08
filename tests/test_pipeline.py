import json
from pathlib import Path

from analysis.validate import validate_manifest
from preprocess.canonical import read_jsonl
from preprocess.pipeline import build_dataset


def test_pipeline_keeps_duplicate_linked_patients_in_one_split(tmp_path: Path) -> None:
    source = tmp_path / "evidence.jsonl"
    rows = [
        {
            "record_id": "one",
            "patient_id": "1",
            "image": "images/1_ant.png",
            "image_size": [100, 200],
            "view": "ANT",
            "region": "chest_left",
            "bbox": [10, 20, 40, 80],
            "caption_description": "The left chest region.",
            "qa": [{"question": "Is the left chest abnormal?", "answer": "no", "template_id": "region_abnormality", "answer_rule": "diagnosis"}],
        },
        {
            "record_id": "two",
            "patient_id": "2",
            "duplicate_of_patient_id": "1",
            "image": "images/2_ant.png",
            "image_size": [100, 200],
            "view": "ANT",
            "region": "chest_left",
            "bbox": [10, 20, 40, 80],
            "grounding": [{"query": "left chest", "label": "chest_left"}],
            "qa": [{"question": "Is the left chest abnormal?", "answer": "yes", "template_id": "region_abnormality", "answer_rule": "diagnosis"}],
        },
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    output = tmp_path / "derived"
    summary = build_dataset(source, output)
    canonical = read_jsonl(output / "canonical.jsonl")

    assert summary["tasks"] == {"vqa": 2, "grounding": 2, "grounded_vqa": 2}
    assert {row["split"] for row in canonical}.__len__() == 1
    assert validate_manifest(output / "grounding.jsonl", "grounding")["rows"] == 2


def test_pipeline_rejects_duplicate_leakage_in_source_splits(tmp_path: Path) -> None:
    source = tmp_path / "evidence.jsonl"
    row = {
        "patient_id": "1",
        "image": "images/1_ant.png",
        "image_size": [100, 200],
        "view": "ANT",
        "region": "chest_left",
        "bbox": [10, 20, 40, 80],
        "source_split": "train",
    }
    duplicate = {**row, "patient_id": "2", "duplicate_of_patient_id": "1", "source_split": "test"}
    source.write_text("\n".join(json.dumps(item) for item in (row, duplicate)) + "\n", encoding="utf-8")

    try:
        build_dataset(source, tmp_path / "derived")
    except ValueError as error:
        assert "cross source split" in str(error)
    else:
        raise AssertionError("Expected duplicate leakage validation to fail")
