import importlib.util
import json
from pathlib import Path


def _load_builder():
    path = Path(__file__).parents[1] / "scripts" / "build_r2_diversified.py"
    spec = importlib.util.spec_from_file_location("build_r2_diversified", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_r2_merges_levels_and_preserves_semantic_fields(tmp_path: Path) -> None:
    builder = _load_builder()
    source = tmp_path / "r1"
    output = tmp_path / "r2"
    source.mkdir()
    canonical = []
    for view in ("ANT", "POST"):
        canonical.append(
            {
                "record_id": f"record-{view}",
                "patient_id": "p1",
                "image": f"{view}.png",
                "image_size": [100, 200],
                "view": view,
                "split": "test",
                "diagnosis": "normal",
                "target": {"kind": "region", "name": "head", "bbox": [1, 2, 20, 40]},
                "flags": {},
            }
        )
    _write(source / "canonical.jsonl", canonical)
    _write(
        source / "paired_evidence.jsonl",
        [
            {
                "evidence_id": "pair-1",
                "patient_id": "p1",
                "region": "head",
                "split": "test",
                "images": [
                    {"record_id": "record-ANT", "view": "ANT", "image": "ANT.png", "image_size": [100, 200], "bbox": [1, 2, 20, 40]},
                    {"record_id": "record-POST", "view": "POST", "image": "POST.png", "image_size": [100, 200], "bbox": [1, 2, 20, 40]},
                ],
            }
        ],
    )
    _write(source / "metastases.jsonl", [])
    _write(source / "patient_splits.jsonl", [{"patient_id": "p1", "split": "test"}])
    common = {"evidence_id": "pair-1", "split": "test", "label": "head", "view_scope": "ANT", "source_labels": {"ANT": "0", "POST": "0"}, "images": [{"view": "ANT", "image": "ANT.png", "image_size": [100, 200]}, {"view": "POST", "image": "POST.png", "image_size": [100, 200]}]}
    _write(source / "vqa.jsonl", [{**common, "task_id": "vqa-1", "question": "Is it abnormal?", "answer": "No", "answer_label": "no"}])
    _write(source / "grounding.jsonl", [{**common, "task_id": "ground-1", "query": "Locate it.", "targets": [{"view": "ANT", "bbox": [1, 2, 20, 40]}]}])
    _write(source / "grounded_vqa.jsonl", [{**common, "task_id": "grounded-1", "question": "Locate it.", "answer": "Normal", "answer_label": "normal", "target": "<REG>head</REG>", "evidence_targets": [{"view": "ANT", "bbox": [1, 2, 20, 40]}]}])
    lesion = {**common, "metastasis_id": "m1", "source_hotspot_label": "Normal", "lesion_class": "non_metastatic_hotspot", "targets": [{"view": "ANT", "bbox": [3, 4, 8, 9]}]}
    _write(source / "lesion_vqa.jsonl", [{**lesion, "task_id": "lesion-vqa-1", "question": "Is it metastatic?", "answer": "No", "answer_label": "no"}])
    _write(source / "lesion_grounding.jsonl", [{**lesion, "task_id": "lesion-ground-1", "query": "Locate it."}])
    _write(source / "lesion_grounded_vqa.jsonl", [{**lesion, "task_id": "lesion-grounded-1", "question": "Locate it.", "answer": "Normal", "answer_label": "normal", "target": "<REG>head</REG>", "evidence_targets": lesion["targets"]}])

    summary = builder.build_release(source, output)
    assert summary["exported_task_rows"] == {"vqa": 4, "vgrounding": 4}
    assert {path.name for path in output.glob("*.jsonl")} == {"canonical.jsonl", "paired_evidence.jsonl", "metastases.jsonl", "patient_splits.jsonl", "vqa.jsonl", "vgrounding.jsonl"}
    merged = [json.loads(line) for line in (output / "vqa.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["region_level"] for row in merged} == {"whole_body", "bone_region", "metastasis"}
    assert all(row["task_group"] == "vqa" for row in merged)
    grounded = [json.loads(line) for line in (output / "vgrounding.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["target"]["region"]["level"] in {"bone_region", "metastasis"} for row in grounded)
    assert all(row["target"]["classification"]["label"] == "benign" for row in grounded)
    assert all("question_voice" not in row for row in merged + grounded)


def test_r2_propagates_malignant_hotspot_to_region_and_whole_body(tmp_path: Path) -> None:
    builder = _load_builder()
    source = tmp_path / "r1"
    output = tmp_path / "r2"
    source.mkdir()
    canonical = []
    for view in ("ANT", "POST"):
        canonical.append(
            {
                "record_id": f"record-{view}",
                "patient_id": "p1",
                "image": f"{view}.png",
                "image_size": [100, 200],
                "view": view,
                "split": "test",
                "diagnosis": "normal",
                "target": {"kind": "region", "name": "head", "bbox": [1, 2, 20, 40]},
                "flags": {},
            }
        )
    _write(source / "canonical.jsonl", canonical)
    _write(
        source / "paired_evidence.jsonl",
        [
            {
                "evidence_id": "pair-1",
                "patient_id": "p1",
                "region": "head",
                "split": "test",
                "diagnosis_by_view": {"ANT": "normal", "POST": "normal"},
                "source_labels": {"ANT": "0", "POST": "0"},
                "images": [
                    {"record_id": "record-ANT", "view": "ANT", "image": "ANT.png", "image_size": [100, 200], "bbox": [1, 2, 20, 40]},
                    {"record_id": "record-POST", "view": "POST", "image": "POST.png", "image_size": [100, 200], "bbox": [1, 2, 20, 40]},
                ],
            }
        ],
    )
    _write(
        source / "metastases.jsonl",
        [
            {
                "metastasis_id": "m1",
                "patient_id": "p1",
                "view": "ANT",
                "bbox": [3, 4, 8, 9],
                "label": "Abnormal",
                "source_id": "p1:0",
                "assignment": "unique_region",
                "candidate_record_ids": ["record-ANT"],
            }
        ],
    )
    _write(source / "patient_splits.jsonl", [{"patient_id": "p1", "split": "test"}])
    common = {
        "evidence_id": "pair-1",
        "split": "test",
        "label": "head",
        "view_scope": "ANT",
        "source_labels": {"ANT": "0", "POST": "0"},
        "images": [
            {"view": "ANT", "image": "ANT.png", "image_size": [100, 200]},
            {"view": "POST", "image": "POST.png", "image_size": [100, 200]},
        ],
    }
    _write(source / "vqa.jsonl", [{**common, "task_id": "vqa-1", "question": "Is it abnormal?", "answer": "No", "answer_label": "no"}])
    _write(
        source / "grounding.jsonl",
        [
            {**common, "task_id": "ground-ant", "view_scope": "ANT", "query": "Locate it.", "targets": [{"view": "ANT", "bbox": [1, 2, 20, 40]}]},
            {**common, "task_id": "ground-post", "view_scope": "POST", "query": "Locate it.", "targets": [{"view": "POST", "bbox": [1, 2, 20, 40]}]},
            {**common, "task_id": "ground-both", "view_scope": "BOTH", "query": "Locate it.", "targets": [{"view": "ANT", "bbox": [1, 2, 20, 40]}, {"view": "POST", "bbox": [1, 2, 20, 40]}]},
        ],
    )
    _write(source / "grounded_vqa.jsonl", [])
    lesion = {
        **common,
        "metastasis_id": "m1",
        "source_hotspot_label": "Abnormal",
        "lesion_class": "bone_metastasis",
        "targets": [{"view": "ANT", "bbox": [3, 4, 8, 9]}],
    }
    _write(source / "lesion_vqa.jsonl", [{**lesion, "task_id": "lesion-vqa-1", "question": "Is it metastatic?", "answer": "Yes", "answer_label": "yes"}])
    _write(source / "lesion_grounding.jsonl", [{**lesion, "task_id": "lesion-ground-1", "query": "Locate it."}])
    _write(source / "lesion_grounded_vqa.jsonl", [])

    builder.build_release(source, output)
    vgrounding = [json.loads(line) for line in (output / "vgrounding.jsonl").read_text(encoding="utf-8").splitlines()]
    by_task = {row["task_id"]: row for row in vgrounding}
    assert by_task["ground-ant"]["target"]["classification"]["label"] == "malignant"
    assert by_task["ground-ant"]["source_region_label"] == "normal"
    assert by_task["ground-ant"]["effective_region_label"] == "malignant"
    assert by_task["ground-ant"]["label_conflict"] is True
    assert by_task["ground-post"]["target"]["classification"]["label"] == "benign"
    assert by_task["ground-post"]["source_region_label"] == "normal"
    assert by_task["ground-post"]["label_conflict"] is False
    assert by_task["ground-both"]["target"]["classification"]["label"] == "malignant"
    lesion_rows = [row for row in vgrounding if row.get("region_level") == "metastasis"]
    assert lesion_rows and all(row["target"]["classification"]["label"] == "malignant" for row in lesion_rows)
    assert all(row["source_region_label"] == "normal" for row in lesion_rows)
    assert all(row["effective_region_label"] == "malignant" for row in lesion_rows)
    assert all(row["label_conflict"] is True for row in lesion_rows)

    vqa = [json.loads(line) for line in (output / "vqa.jsonl").read_text(encoding="utf-8").splitlines()]
    bone_vqa = next(row for row in vqa if row.get("region_level") == "bone_region")
    assert bone_vqa["answer_label"] == "yes"
    assert bone_vqa["source_region_label"] == "normal"
    assert bone_vqa["effective_region_label"] == "malignant"
    assert bone_vqa["label_conflict"] is True
    whole_body = {row["view_scope"]: row for row in vqa if row.get("region_level") == "whole_body"}
    assert whole_body["ANT"]["labels"]["classification"]["label"] == "malignant"
    assert whole_body["ANT"]["source_region_label"] == "normal"
    assert whole_body["ANT"]["label_conflict"] is True
    assert whole_body["POST"]["labels"]["classification"]["label"] == "benign"
