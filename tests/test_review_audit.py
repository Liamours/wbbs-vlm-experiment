from analysis.review import compare_reviews, validate_adjudications
from preprocess.gold import build_r01_gold_evidence


def test_review_comparison_and_gold_builder() -> None:
    queue = [_case()]
    first = [_response("reader_a", "normal", "normal")]
    second = [_response("reader_b", "normal", "abnormal")]

    report, adjudication = compare_reviews(queue, first, second)

    assert report["diagnosis"]["agreement_rate"] == 0.5
    assert report["adjudication_cases"] == 1
    assert adjudication[0]["differences"] == ["diagnosis:POST"]

    decisions = [_adjudication()]
    assert validate_adjudications(queue, decisions)["r01:pair-1"]["final_decision"] == "correct"
    accepted, excluded = build_r01_gold_evidence([_pair()], queue, decisions)
    assert not excluded
    assert accepted[0]["evidence_id"] == "pair-1:r01"
    assert accepted[0]["pair_diagnosis"] == "abnormal"


def _case() -> dict:
    return {
        "review_id": "r01:pair-1",
        "evidence_id": "pair-1",
        "region": "left chest",
        "images": [{"view": "ANT", "image": "ant.jpg", "image_size": [100, 100]}, {"view": "POST", "image": "post.jpg", "image_size": [100, 100]}],
        "target_bboxes": [{"view": "ANT", "bbox": [1, 1, 10, 10]}, {"view": "POST", "bbox": [2, 2, 11, 11]}],
    }


def _response(reviewer_id: str, ant: str, post: str) -> dict:
    return {
        "review_id": "r01:pair-1",
        "reviewer_id": reviewer_id,
        "bbox_status_by_view": {"ANT": "accept", "POST": "accept"},
        "corrected_bbox_by_view": {"ANT": None, "POST": None},
        "diagnosis_by_view": {"ANT": ant, "POST": post},
        "overall_decision": "accept",
        "notes": "",
    }


def _adjudication() -> dict:
    return {
        "review_id": "r01:pair-1",
        "adjudicator_id": "reader_c",
        "final_decision": "correct",
        "final_bbox_by_view": {"ANT": [1, 1, 10, 10], "POST": [2, 2, 11, 11]},
        "final_diagnosis_by_view": {"ANT": "normal", "POST": "abnormal"},
        "notes": "Resolved after joint review.",
    }


def _pair() -> dict:
    return {
        "evidence_id": "pair-1",
        "patient_id": "1",
        "region": "left chest",
        "split": "test",
        "images": [{"view": "ANT", "image": "ant.jpg", "image_size": [100, 100], "bbox": [1, 1, 10, 10]}, {"view": "POST", "image": "post.jpg", "image_size": [100, 100], "bbox": [2, 2, 11, 11]}],
        "diagnosis_by_view": {"ANT": "normal", "POST": "normal"},
        "pair_diagnosis": "normal",
        "source_labels": {"ANT": "0", "POST": "0"},
    }
