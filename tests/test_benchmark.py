import json
from pathlib import Path

from analysis.benchmark import benchmark_release, benchmark_standard, compare_benchmark_reports, standardize_dataset, standardize_rows


def test_benchmark_release_reports_language_and_coverage(tmp_path: Path) -> None:
    release = tmp_path / "r-test"
    release.mkdir()
    _write(
        release / "canonical.jsonl",
        [
            {
                "record_id": "record-1",
                "patient_id": "patient-1",
                "split": "test",
                "view": "ANT",
                "diagnosis": "normal",
                "target": {"kind": "region", "name": "head", "bbox": [1, 2, 11, 22]},
                "image_size": [100, 100],
                "flags": {},
            }
        ],
    )
    _write(
        release / "paired_evidence.jsonl",
        [
            {
                "evidence_id": "pair-1",
                "patient_id": "patient-1",
                "split": "test",
                "region": "head",
            }
        ],
    )
    _write(
        release / "vqa.jsonl",
        [
            {
                "task_id": "vqa-1",
                "evidence_id": "pair-1",
                "split": "test",
                "question": "Is abnormal tracer uptake present in the head?",
                "answer": "No",
                "answer_label": "no",
                "template_id": "v1",
            },
            {
                "task_id": "vqa-whole-body-1",
                "evidence_id": "pair-1",
                "split": "test",
                "question": "Is abnormal tracer uptake present in this whole-body view?",
                "answer": "No",
                "answer_label": "no",
                "template_id": "v1",
                "region_level": "whole_body",
                "interaction_type": "classification",
            },
        ],
    )
    _write(
        release / "grounding.jsonl",
        [
            {
                "task_id": "ground-1",
                "evidence_id": "pair-1",
                "split": "test",
                "query": "Locate the head.",
                "label": "head",
                "targets": [{"view": "ANT", "bbox": [1, 2, 11, 22]}],
                "images": [{"view": "ANT", "image_size": [100, 100]}],
            }
        ],
    )

    report = benchmark_release(release, tmp_path / "audit")

    assert report["tasks"]["vqa"]["linguistic"]["unique_words"] > 0
    assert report["coverage"]["grounding_granularity"]["anatomical_region"]
    assert not report["coverage"]["grounding_granularity"]["whole_body"]
    assert report["coverage"]["whole_body_task_present"]
    assert report["integrity"]["canonical"]["bbox"]["invalid"] == 0
    assert (tmp_path / "audit" / "dataset_benchmark.md").is_file()


def test_standardize_rows_preserves_vqa_and_infers_task() -> None:
    rows = standardize_rows(
        [{"qid": "q1", "image_name": "img.png", "question": "Is there a lesion?", "answer": "No"}],
        dataset_name="toy",
    )

    assert len(rows) == 1
    assert rows[0]["record_id"] == "q1"
    assert rows[0]["task"] == "vqa"
    assert rows[0]["answer"] == "No"
    assert rows[0]["target_boxes"] == []
    assert rows[0]["split"] is None


def test_standardize_rows_accepts_explicit_field_map() -> None:
    rows = standardize_rows(
        [{"sample": "s1", "question_text": "What is shown?", "gt": ["bone"]}],
        dataset_name="toy",
        field_map={"record_id": "sample", "question": "question_text", "answer": "gt"},
    )
    assert rows[0]["record_id"] == "s1"
    assert rows[0]["question"] == "What is shown?"
    assert rows[0]["answers"] == ["bone"]


def test_standardize_dataset_converts_coco_xywh_and_benchmarks(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": 7,
                    "image_id": "scan-7.png",
                    "image_size": [100, 100],
                    "query": "Locate the lesion.",
                    "annotations": [{"bbox": [10, 20, 30, 40], "category": "lesion"}],
                    "split": "test",
                }
            ]
        ),
        encoding="utf-8",
    )
    standard = tmp_path / "standard.jsonl"
    summary = standardize_dataset(source, standard, dataset_name="toy", bbox_format="xywh")
    assert summary["rows"] == 1
    converted = json.loads(standard.read_text(encoding="utf-8").strip())
    assert converted["task"] == "grounding"
    assert converted["image_size"] == [100.0, 100.0]
    assert converted["target_boxes"][0]["bbox"] == [10.0, 20.0, 40.0, 60.0]
    report = benchmark_standard(standard, tmp_path / "audit", dataset_name="toy")
    assert report["format"] == "comparison-v1"
    assert report["coverage"]["grounding_granularity"]["lesion_or_metastasis"]
    assert report["integrity"]["canonical"]["bbox"]["invalid"] == 0


def test_compare_benchmark_reports_writes_combined_tables(tmp_path: Path) -> None:
    report = {
        "format": "comparison-v1",
        "release": "toy-a",
        "coverage": {"grounding_granularity": {"anatomical_region": True, "lesion_or_metastasis": False, "whole_body": False}},
        "warnings": [],
        "tasks": {
            "vqa": {
                "rows": 2,
                "linguistic": {
                    "unique_words": 3,
                    "unique_bigrams": 2,
                    "unique_trigrams": 1,
                    "row_type_token_ratio": {"mean": 0.75},
                    "exact_text": {"unique_normalized": 1, "duplicate_rate": 0.5},
                    "masked_skeleton": {"unique": 1, "top_share": 1.0},
                },
            }
        },
    }
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(report), encoding="utf-8")
    second.write_text(json.dumps({**report, "release": "toy-b"}), encoding="utf-8")

    summary = compare_benchmark_reports([first, second], tmp_path / "comparison")

    assert [row["dataset"] for row in summary["datasets"]] == ["toy-a", "toy-b"]
    assert (tmp_path / "comparison" / "dataset_comparison.md").is_file()
    assert (tmp_path / "comparison" / "dataset_comparison_metrics.csv").is_file()
    assert (tmp_path / "comparison" / "dataset_comparison_question_types.csv").is_file()


def test_standard_grounding_without_granularity_is_not_anatomical(tmp_path: Path) -> None:
    standard = tmp_path / "generic.standard.jsonl"
    _write(
        standard,
        [
            {
                "schema_version": "1.0",
                "record_id": "generic-1",
                "task": "grounding",
                "question": "Locate the corresponding image patch.",
                "target_boxes": [{"bbox": [1, 1, 10, 10], "label": "patch"}],
                "target_granularity": None,
                "split": "test",
            }
        ],
    )
    report = benchmark_standard(standard, tmp_path / "audit", dataset_name="generic")
    assert not report["coverage"]["grounding_granularity"]["anatomical_region"]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
