from preprocess.review import build_r01_review_package


def test_r01_review_package_selects_test_abnormal_and_normal_controls() -> None:
    rows = [
        _pair("left chest", "a", "abnormal", {"ANT": "abnormal", "POST": "normal"}),
        _pair("left chest", "b", "normal", {"ANT": "normal", "POST": "normal"}),
        _pair("left chest", "c", "normal", {"ANT": "normal", "POST": "normal"}),
        _pair("head", "d", "abnormal", {"ANT": "abnormal", "POST": "abnormal"}),
        {**_pair("head", "e", "abnormal", {"ANT": "abnormal", "POST": "abnormal"}), "split": "val"},
    ]

    queue, provenance, summary = build_r01_review_package(rows, normal_controls_per_region=1, seed=7)

    assert len(queue) == 3
    assert {"a", "d"} <= {case["evidence_id"] for case in queue}
    assert len({case["evidence_id"] for case in queue} & {"b", "c"}) == 1
    assert "source_diagnosis_by_view" not in queue[0]
    assert {row["evidence_id"] for row in provenance} == {case["evidence_id"] for case in queue}
    assert summary["selection_reasons"] == {"abnormal_test_pair": 2, "normal_control": 1}
    assert summary["view_diagnosis_disagreements"] == 1


def _pair(region: str, evidence_id: str, diagnosis: str, diagnosis_by_view: dict[str, str]) -> dict:
    return {
        "evidence_id": evidence_id,
        "patient_id": evidence_id,
        "region": region,
        "split": "test",
        "pair_diagnosis": diagnosis,
        "diagnosis_by_view": diagnosis_by_view,
        "source_labels": {"ANT": "1", "POST": "1"},
        "images": [
            {"view": "ANT", "image": "ant.jpg", "image_size": [100, 100], "bbox": [1, 1, 10, 10]},
            {"view": "POST", "image": "post.jpg", "image_size": [100, 100], "bbox": [1, 1, 10, 10]},
        ],
    }
