import importlib.util
import json
from pathlib import Path

from analysis.validate import validate_manifest

def _load_builder():
    path = Path(__file__).parents[1] / "scripts" / "build_r3_compact.py"
    spec = importlib.util.spec_from_file_location("build_r3_compact", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_r3_compact_preserves_all_canonical_image_coverage_and_splits(tmp_path: Path) -> None:
    builder = _load_builder()
    source = tmp_path / "r2"
    output = tmp_path / "r3"
    source.mkdir()
    canonical = [
        {"record_id": "a", "patient_id": "p1", "split": "train", "image": "bs80k/wholeBodyANT/1.jpg", "image_size": [100, 200], "view": "ANT"},
        {"record_id": "b", "patient_id": "p1", "split": "train", "image": "bs80k/wholeBodyPOST/1.jpg", "image_size": [100, 200], "view": "POST"},
        {"record_id": "c", "patient_id": "p2", "split": "test", "image": "bs80k/wholeBodyANT/2.jpg", "image_size": [100, 200], "view": "ANT"},
    ]
    _write(source / "canonical.jsonl", canonical)
    _write(source / "paired_evidence.jsonl", [{"evidence_id": "pair-1", "patient_id": "p1", "split": "train"}])
    _write(source / "metastases.jsonl", [])
    _write(source / "patient_splits.jsonl", [{"patient_id": "p1", "split": "train"}, {"patient_id": "p2", "split": "test"}])
    _write(
        source / "vqa.jsonl",
        [
                {"task_id": "vqa-pair", "split": "train", "region_level": "bone_region", "interaction_type": "classification", "question": "Is it abnormal?", "answer": "No", "answer_label": "no", "images": [{"image": canonical[0]["image"]}, {"image": canonical[1]["image"]}]},
                {"task_id": "vqa-single", "split": "test", "region_level": "whole_body", "interaction_type": "classification", "question": "Is it abnormal?", "answer": "No", "answer_label": "no", "images": [{"image": canonical[2]["image"]}]},
                {"task_id": "vqa-extra", "split": "train", "region_level": "bone_region", "interaction_type": "classification", "question": "Is it abnormal?", "answer": "Yes", "answer_label": "yes", "images": [{"image": canonical[0]["image"]}]},
        ],
    )
    _write(
        source / "vgrounding.jsonl",
        [
                {"task_id": "ground-pair", "split": "train", "region_level": "bone_region", "interaction_type": "localization", "query": "Locate it.", "answer_label": None, "images": [{"image": canonical[0]["image"]}, {"image": canonical[1]["image"]}]},
                {"task_id": "ground-extra", "split": "train", "region_level": "bone_region", "interaction_type": "localization_and_classification", "query": "Locate it.", "answer_label": "normal", "images": [{"image": canonical[0]["image"]}, {"image": canonical[1]["image"]}]},
        ],
    )

    summary = builder.build_release(source, output, fraction=0.20)

    assert summary["canonical_image_coverage"] == {"required": 3, "selected": 3, "missing": 0}
    assert summary["exported_task_rows"] == {"vqa": 1, "vgrounding": 1, "multitask": 2, "metadata_vqa": 3}
    assert (output / "canonical.jsonl").read_text(encoding="utf-8") == (source / "canonical.jsonl").read_text(encoding="utf-8")
    selected = [json.loads(line) for path in (output / "vqa.jsonl", output / "vgrounding.jsonl") for line in path.read_text(encoding="utf-8").splitlines()]
    assert {image["image"] for row in selected for image in row["images"]} == {row["image"] for row in canonical}
    assert {row["split"] for row in selected} == {"train", "test"}
    multitask = [json.loads(line) for line in (output / "multitask.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "vqa" in {row["task_type"] for row in multitask}
    assert {row["task_type"] for row in multitask} <= {"vqa", "grounding", "grounded_vqa"}
    assert all(row["prompt"] and row["schema_version"] == "r3-multitask-v1" for row in multitask)
    metadata = [json.loads(line) for line in (output / "metadata_vqa.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(metadata) == 3 and all(row["training_eligible"] is False for row in metadata)
    assert validate_manifest(output / "metadata_vqa.jsonl", "metadata_vqa")["rows"] == 3
