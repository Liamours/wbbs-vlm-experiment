from analysis.priors import run_prior_baseline


def test_question_anatomy_prior_reports_classification_and_grounding() -> None:
    train = [_row("train-1", "train", "normal", [1, 1, 5, 5]), _row("train-2", "train", "normal", [1, 1, 5, 5])]
    evaluation = [_row("test-1", "test", "normal", [1, 1, 5, 5])]

    predictions, report = run_prior_baseline(train, evaluation, "grounded_vqa")

    assert predictions[0]["prediction_label"] == "normal"
    assert predictions[0]["prediction_targets"] == [{"view": "ANT", "bbox": [1, 1, 5, 5]}]
    assert report["classification"]["accuracy"] == 1.0
    assert report["grounding"]["mean_iou"] == 1.0


def _row(task_id: str, split: str, diagnosis: str, bbox: list[int]) -> dict:
    return {
        "task_id": task_id,
        "split": split,
        "label": "left chest",
        "view_scope": "ANT",
        "answer_label": diagnosis,
        "images": [{"view": "ANT", "image_size": [10, 10]}, {"view": "POST", "image_size": [10, 10]}],
        "evidence_targets": [{"view": "ANT", "bbox": bbox}],
    }
