import csv
import json
from pathlib import Path

from analysis.validate import validate_manifest
from preprocess.libs160k import LIBS_TEMPLATE_LAYOUT, TEXT_FILES
from preprocess.release import build_r0_release


def test_r0_release_builds_traceable_manifests(tmp_path: Path) -> None:
    root = tmp_path / "wbbs-dataset"
    _write_sources(root)
    output = tmp_path / "release"

    summary = build_r0_release(
        root,
        output,
        {
            "release": "r0-test",
            "exclude_low_precision_regions": True,
            "exclude_known_artifacts": True,
            "exclude_outlier_images": True,
            "exclude_corrupt_images": True,
            "exclude_missing_images": True,
        },
    )

    assert summary["verification"] == {
        "accepted": 4,
        "rejected": 3,
        "reasons": {"known_artifact": 1, "low_precision_region": 2},
    }
    assert summary["metastases"] == {"rows": 1, "assignments": {"unique_region": 1}}
    assert summary["paired_dataset"]["paired_records"] == 2
    assert summary["paired_dataset"]["tasks"] == {"vqa": 6, "grounding": 6, "grounded_vqa": 6}
    assert validate_manifest(output / "canonical.jsonl", "canonical")["rows"] == 4
    assert validate_manifest(output / "vqa.jsonl", "vqa")["rows"] == 6
    assert validate_manifest(output / "grounding.jsonl", "grounding")["rows"] == 6
    canonical = _read_jsonl(output / "canonical.jsonl")
    assert canonical[0]["hotspots"] == [{"assignment": "unique_region", "bbox": [10, 10, 20, 20], "label": "Normal", "source_id": "1:0"}]
    assert canonical[0]["captions"] == {"description": "template 37", "diagnosis": "template 39"}
    assert _read_jsonl(output / "grounding.jsonl")[0]["query"] == "Locate the left chest region in the anterior image."
    assert _read_jsonl(output / "vqa.jsonl")[0]["source_labels"] == {"ANT": "0", "POST": "0"}
    assert _read_jsonl(output / "grounding_bbx.jsonl")[0]["target"] == "<ANT><BBX>0,0,120,200</BBX></ANT>"
    assert _read_jsonl(output / "grounding_bbx.jsonl")[0]["label"] == "left chest"
    assert _read_jsonl(output / "grounded_vqa.jsonl")[0]["target"] == "<REG>left_chest</REG><CLS>normal</CLS><ANT><BBX>0,0,120,200</BBX></ANT>"
    assert {row["target"]["name"] for row in canonical} == {"left chest", "right chest"}
    inventory = json.loads((output / "source_inventory.json").read_text(encoding="utf-8"))
    assert inventory["sources"][0]["rows"] == 7
    assert _read_jsonl(output / "metastases.jsonl")[0]["assignment"] == "unique_region"
    assert (output / "quality_report.json").is_file()
    patient_splits = _read_jsonl(output / "patient_splits.jsonl")
    assert patient_splits == [{"patient_id": "1", "split": "train", "canonical_records": 4, "paired_evidence_records": 2}]
    split_report = json.loads((output / "split_report.json").read_text(encoding="utf-8"))
    assert split_report["leakage_checks"]["all_passed"]
    assert len(split_report["split_assignment_sha256"]) == 64
    assert summary["split_audit"]["leakage_checks"]["all_passed"]


def _write_sources(root: Path) -> None:
    _write_csv(
        root / "bs80k-bone_region-bb/bounding_boxes.csv",
        [
            {"component": "chestLANT", "id": "1", "x": "0", "y": "0", "width": "120", "height": "200", "label": "0", "diagnosis": "normal", "duplicate_of_patient_id": "", "duplicate_of_sibling_component": "", "match_score": "1.0"},
            {"component": "chestLPOST", "id": "1", "x": "1", "y": "0", "width": "120", "height": "200", "label": "0", "diagnosis": "normal", "duplicate_of_patient_id": "", "duplicate_of_sibling_component": "", "match_score": "1.0"},
            {"component": "chestRANT", "id": "1", "x": "120", "y": "0", "width": "136", "height": "200", "label": "1", "diagnosis": "abnormal", "duplicate_of_patient_id": "", "duplicate_of_sibling_component": "", "match_score": "1.0"},
            {"component": "chestRPOST", "id": "1", "x": "120", "y": "1", "width": "136", "height": "200", "label": "0", "diagnosis": "normal", "duplicate_of_patient_id": "", "duplicate_of_sibling_component": "", "match_score": "1.0"},
            {"component": "shoLANT", "id": "1", "x": "0", "y": "200", "width": "80", "height": "60", "label": "0", "diagnosis": "normal", "duplicate_of_patient_id": "", "duplicate_of_sibling_component": "", "match_score": "1.0"},
            {"component": "shoLPOST", "id": "1", "x": "0", "y": "200", "width": "80", "height": "60", "label": "0", "diagnosis": "normal", "duplicate_of_patient_id": "", "duplicate_of_sibling_component": "", "match_score": "1.0"},
            {"component": "elbowLPOST", "id": "1", "x": "0", "y": "0", "width": "80", "height": "60", "label": "0", "diagnosis": "normal", "duplicate_of_patient_id": "", "duplicate_of_sibling_component": "", "match_score": "1.0"},
        ],
    )
    _write_csv(
        root / "bs80k-wholebody-bb/bounding_boxes.csv",
        [
            {"id": "1", "view": "ANT", "x": "0", "y": "0", "width": "256", "height": "1024", "outlier": "False", "likely_corrupt_image": "False"},
            {"id": "1", "view": "POST", "x": "0", "y": "0", "width": "256", "height": "1024", "outlier": "False", "likely_corrupt_image": "False"},
        ],
    )
    xml = """<annotation><size><width>256</width><height>1024</height></size><object><name>Normal</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>20</xmax><ymax>20</ymax></bndbox></object></annotation>"""
    path = root / "bs80k-lesion-bb/ant/1.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")
    for view in ("ANT", "POST"):
        image = root / f"bs80k-imaging-raw/wholeBody{view}/1.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"placeholder")
    for relative in TEXT_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps({"text_id": text_id, "text": f"template {text_id}", "image_ids": []}) for text_id in LIBS_TEMPLATE_LAYOUT) + "\n",
            encoding="utf-8",
        )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
