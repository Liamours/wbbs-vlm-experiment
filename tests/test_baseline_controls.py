from __future__ import annotations

from analysis.baseline_controls import build_control_predictions
from metric.qwen import evaluate_qwen_multitask_predictions
from postprocess.qwen import parse_qwen_multitask_json_prediction


def _row(task: str, task_id: str, prompt: str, label: str, answer_label: str, view: str = "ANT") -> dict:
    target = {"task": task}
    if task == "vqa":
        target.update({"answer": answer_label, "class": answer_label})
    elif task == "grounding":
        target.update({"region": label, "boxes": [{"view": view, "bbox": [1, 2, 8, 10]}]})
    else:
        target.update({"answer": answer_label, "region": label, "class": answer_label, "boxes": [{"view": view, "bbox": [1, 2, 8, 10]}]})
    return {
        "task": task,
        "task_id": task_id,
        "prompt": prompt,
        "label": label,
        "answer_label": answer_label,
        "targets": [{"view": "POST" if view == "PST" else view, "bbox": [1, 2, 8, 10]}],
        "target": __import__("json").dumps(target),
    }


def test_no_image_controls_cover_and_parse_every_multitask_row() -> None:
    train = [
        _row("vqa", "vqa-1", "Is the head metastatic?", "head", "no"),
        _row("vqa", "vqa-2", "Is the pelvis metastatic?", "pelvis", "yes"),
        _row("grounding", "ground-1", "Locate the head.", "head", "missing"),
        _row("grounded_vqa", "grounded-1", "Locate the pelvis and classify it.", "pelvis", "normal", "PST"),
    ]
    evaluation = [
        _row("vqa", "vqa-eval", "Is the head metastatic?", "head", "no"),
        _row("grounding", "ground-eval", "Locate the head.", "head", "missing"),
        _row("grounded_vqa", "grounded-eval", "Locate the pelvis and classify it.", "pelvis", "normal", "PST"),
    ]
    for control in ("majority", "prompt_only"):
        predictions = build_control_predictions(train, evaluation, control)
        assert [row["task_id"] for row in predictions] == [row["task_id"] for row in evaluation]
        for expected, prediction in zip(evaluation, predictions, strict=True):
            assert parse_qwen_multitask_json_prediction(prediction["prediction"])["task"] == expected["task"]
        assert evaluate_qwen_multitask_predictions(evaluation, predictions)["rows"] == len(evaluation)


def test_validation_only_controls_do_not_require_a_test_input(tmp_path) -> None:
    from analysis.baseline_controls import create_controls

    train = [
        _row("vqa", "train-vqa", "Is the head metastatic?", "head", "no"),
        _row("grounding", "train-ground", "Locate the head.", "head", "missing"),
    ]
    validation = [
        _row("vqa", "validation-vqa", "Is the head metastatic?", "head", "no"),
        _row("grounding", "validation-ground", "Locate the head.", "head", "missing"),
    ]
    train_path, validation_path = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    train_path.write_text("\n".join(__import__("json").dumps(row) for row in train) + "\n", encoding="utf-8")
    validation_path.write_text("\n".join(__import__("json").dumps(row) for row in validation) + "\n", encoding="utf-8")

    result = create_controls(train_path, validation_path, tmp_path / "controls")

    assert set(result["inputs"]) == {"validation"}
    assert set(result["controls"]["majority"]) == {"validation"}
