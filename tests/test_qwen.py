from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

import torch

from metric.qwen import evaluate_qwen_grounded_predictions, evaluate_qwen_multitask_predictions, patient_cluster_bootstrap_qwen_multitask
from analysis.qwen_schema import parse_strict_json_structure, parse_strict_structure, validate_grounded_row
from analysis.qwen_probe import build_balanced_probe
from postprocess.qwen import build_qwen_exports, export_qwen, parse_qwen_json_prediction, parse_qwen_multitask_json_prediction, parse_qwen_prediction
from preprocess.canonical import write_jsonl
from training.qwen import _append_prediction, _encode_prompt_batch, _format_seconds, _load_resume_checkpoint, _load_rows, _messages, _prepare_prediction_output, _save_resume_checkpoint, _scheduled_rows
from training.qwen_config import load_qwen_experiment


def _row() -> dict:
    return {
        "task_id": "paired-1",
        "split": "train",
        "images": [
            {"view": "ANT", "image": "ant.jpg", "image_size": [256, 1024]},
            {"view": "POST", "image": "post.jpg", "image_size": [256, 1024]},
        ],
        "question": "Locate the chest and determine uptake.",
        "query": "Locate the chest.",
        "answer": "The chest does not show abnormal tracer uptake.",
        "answer_label": "normal",
        "label": "chest",
        "target": "<REG>chest</REG><CLS>normal</CLS><ANT><BBX>10,20,30,40</BBX></ANT><PST><BBX>12,22,32,42</BBX></PST>",
        "evidence_targets": [
            {"view": "ANT", "bbox": [10, 20, 30, 40]},
            {"view": "POST", "bbox": [12, 22, 32, 42]},
        ],
        "view_scope": "both",
    }


def test_qwen_export_preserves_two_views_and_multiple_targets() -> None:
    exported = export_qwen([_row()], "grounded_vqa")
    assert [image["view"] for image in exported[0]["images"]] == ["ANT", "POST"]
    assert exported[0]["target"].startswith("The chest does not show abnormal tracer uptake.")
    assert len(exported[0]["targets"]) == 2


def test_qwen_prediction_parser_and_metric_support_two_views() -> None:
    expected = export_qwen([_row()], "grounded_vqa")
    prediction = "The chest does not show abnormal tracer uptake. <REG>chest</REG><CLS>normal</CLS><ANT><BBX>10,20,30,40</BBX></ANT><PST><BBX>12,22,32,42</BBX></PST>"
    parsed = parse_qwen_prediction(prediction)
    assert parsed["answer_label"] == "normal"
    assert len(parsed["boxes"]) == 2
    report = evaluate_qwen_grounded_predictions(expected, [{"task_id": "paired-1", "prediction": prediction}])
    assert report["mean_iou"] == 1.0
    assert report["strict_grounded_accuracy"] == 1.0


def test_qwen_json_export_parser_metric_and_audit_support_two_views() -> None:
    expected = export_qwen([_row()], "grounded_vqa", schema="json")
    prediction = expected[0]["target"]
    parsed = parse_qwen_json_prediction(prediction)
    assert parsed["region"] == "chest"
    assert parsed["answer_label"] == "normal"
    assert len(parsed["boxes"]) == 2
    assert parse_strict_json_structure(prediction)["valid"]
    assert validate_grounded_row(expected[0], schema="json")["valid"]
    report = evaluate_qwen_grounded_predictions(expected, [{"task_id": "paired-1", "prediction": prediction}], output_schema="json")
    assert report["mean_iou"] == 1.0
    assert report["strict_grounded_accuracy"] == 1.0


def test_qwen_json_audit_rejects_non_json_and_non_integer_boxes() -> None:
    assert "invalid_json" in parse_strict_json_structure("not json")["issues"]
    malformed = '{"answer":"normal","region":"chest","class":"normal","boxes":[{"view":"ANT","bbox":[1,2,3.5,4]}]}'
    assert "invalid_integer_bbox" in parse_strict_json_structure(malformed)["issues"]


def test_qwen_strict_schema_accepts_export_and_rejects_partial_tags() -> None:
    exported = export_qwen([_row()], "grounded_vqa")[0]
    assert validate_grounded_row(exported)["valid"]
    malformed = "The chest is normal. <REG>chest</REG><CLS>normal</CLS><ANT><BBX>10,20,30</BBX></ANT>"
    result = parse_strict_structure(malformed)
    assert not result["valid"]
    assert "invalid_integer_bbox" in result["issues"]


def test_qwen_strict_schema_allows_whitespace_between_blocks() -> None:
    prediction = "The chest is normal. <REG>chest</REG>\n<CLS>normal</CLS>\n<ANT><BBX>10,20,30,40</BBX></ANT>"
    assert parse_strict_structure(prediction)["valid"]
    assert not parse_strict_structure("<REG>chest</REG>")["format_valid"]


def test_qwen_metric_penalizes_invalid_classification() -> None:
    abnormal = deepcopy(_row())
    abnormal["task_id"] = "paired-2"
    abnormal["answer_label"] = "abnormal"
    abnormal["target"] = abnormal["target"].replace("<CLS>normal</CLS>", "<CLS>abnormal</CLS>")
    expected = export_qwen([_row(), abnormal], "grounded_vqa")
    normal_prediction = expected[0]["target"]
    invalid_prediction = "<REG>chest</REG>"
    report = evaluate_qwen_grounded_predictions(
        expected,
        [
            {"task_id": "paired-1", "prediction": normal_prediction},
            {"task_id": "paired-2", "prediction": invalid_prediction},
        ],
    )
    assert report["classification_accuracy"] == 0.5
    assert report["invalid_classifications"] == 1
    assert report["balanced_accuracy"] == 0.5


def test_qwen_grounded_metric_assigns_zero_iou_to_missing_view() -> None:
    expected = export_qwen([_row()], "grounded_vqa")
    prediction = "<REG>chest</REG><CLS>normal</CLS><ANT><BBX>10,20,30,40</BBX></ANT>"
    report = evaluate_qwen_grounded_predictions(expected, [{"task_id": "paired-1", "prediction": prediction}])
    assert report["mean_iou"] == 0.5
    assert report["iou_at_0_5"] == 0.5
    assert report["strict_grounded_accuracy"] == 0.0


def test_qwen_multitask_metrics_report_validity_and_zero_fill_missing_boxes() -> None:
    expected = [
        {
            "task_id": "ground-1",
            "task": "grounding",
            "label": "left knee",
            "targets": [{"view": "ANT", "bbox": [1, 2, 20, 40]}, {"view": "POST", "bbox": [3, 4, 22, 42]}],
        }
    ]
    prediction = '{"task":"grounding","region":"left_knee","boxes":[{"view":"ANT","bbox":[1,2,20,40]}]}'
    report = evaluate_qwen_multitask_predictions(expected, [{"task_id": "ground-1", "prediction": prediction}])["grounding"]
    assert report["valid_output_rate"] == 1.0
    assert report["mean_iou"] == 0.5
    assert report["iou_at_0_5"] == 0.5
    assert report["strict_grounding_accuracy"] == 0.0


def test_qwen_multitask_validity_rejects_duplicate_view_boxes() -> None:
    expected = [
        {
            "task_id": "ground-1",
            "task": "grounding",
            "label": "left knee",
            "targets": [{"view": "ANT", "bbox": [1, 2, 20, 40]}],
        }
    ]
    prediction = '{"task":"grounding","region":"left_knee","boxes":[{"view":"ANT","bbox":[1,2,20,40]},{"view":"ANT","bbox":[1,2,20,40]}]}'
    report = evaluate_qwen_multitask_predictions(expected, [{"task_id": "ground-1", "prediction": prediction}])["grounding"]
    assert report["valid_output_rate"] == 0.0
    assert report["mean_iou"] == 0.0
    assert report["strict_grounding_accuracy"] == 0.0


def test_qwen_messages_use_absolute_image_paths(tmp_path) -> None:
    row = export_qwen([_row()], "grounded_vqa")[0]
    row["_root"] = str(tmp_path)
    messages = _messages(row)
    images = messages[1]["content"][:2]
    assert all(item["image"].startswith(str(tmp_path)) for item in images)
    assert all(not item["image"].startswith("file:") for item in images)


def test_qwen_prompt_batch_uses_left_padding_and_restores_processor(tmp_path) -> None:
    class Tokenizer:
        padding_side = "right"

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            self.padding_side_during_call = self.tokenizer.padding_side
            return {"input_ids": "encoded"}

    rows = export_qwen([_row(), _row()], "grounded_vqa")
    rows[0]["_root"] = str(tmp_path)
    rows[1]["_root"] = str(tmp_path)
    processor = Processor()
    config = load_qwen_experiment(str(Path(__file__).resolve().parents[1] / "configs/qwen3vl_r1_grounded_vqa_4050.json"))
    assert _encode_prompt_batch(processor, rows, config) == {"input_ids": "encoded"}
    assert len(processor.messages) == 2
    assert processor.kwargs["processor_kwargs"] == {"padding": True}
    assert processor.padding_side_during_call == "left"
    assert processor.tokenizer.padding_side == "right"


def test_qwen_loader_samples_deterministically(tmp_path) -> None:
    rows = []
    for index in range(4):
        row = _row()
        row["task_id"] = f"paired-{index}"
        rows.extend(export_qwen([row], "grounded_vqa"))
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, rows)
    first = _load_rows(path, "grounded_vqa", 3, sample_seed=4050)
    second = _load_rows(path, "grounded_vqa", 3, sample_seed=4050)
    assert [row["task_id"] for row in first] == [row["task_id"] for row in second]


def test_qwen_balanced_schedule_is_deterministic_and_balances_each_update() -> None:
    rows = []
    for family, prefix in (("anatomy", "libs160k"), ("lesion", "bs80k")):
        for label in ("normal", "abnormal"):
            for index in range(2):
                rows.append({"task_id": f"{prefix}:{label}:{index}", "answer_label": label, "family": family})
    first = _scheduled_rows(rows, 8, "balanced_family_class", 4050)
    second = _scheduled_rows(rows, 8, "balanced_family_class", 4050)
    assert [row["task_id"] for row in first] == [row["task_id"] for row in second]
    first_update = [("lesion" if row["task_id"].startswith("bs80k:") else "anatomy", row["answer_label"]) for row in first]
    assert {key: first_update.count(key) for key in set(first_update)} == {
        ("anatomy", "normal"): 2,
        ("anatomy", "abnormal"): 2,
        ("lesion", "normal"): 2,
        ("lesion", "abnormal"): 2,
    }


def test_qwen_balanced_probe_has_each_family_and_class() -> None:
    rows = []
    for prefix in ("wbbs", "bs80k"):
        for label in ("normal", "abnormal"):
            for index in range(2):
                rows.append({"task_id": f"{prefix}:{label}:{index}", "task": "grounded_vqa", "answer_label": label})
    probe = build_balanced_probe(rows, per_stratum=2, seed=4050)
    observed = [("lesion" if row["task_id"].startswith("bs80k:") else "anatomy", row["answer_label"]) for row in probe]
    assert {key: observed.count(key) for key in set(observed)} == {
        ("anatomy", "normal"): 2,
        ("anatomy", "abnormal"): 2,
        ("lesion", "normal"): 2,
        ("lesion", "abnormal"): 2,
    }


def test_qwen_prediction_output_appends_and_resumes_in_task_order(tmp_path) -> None:
    output = tmp_path / "predictions.jsonl"
    rows = [{"task_id": "first"}, {"task_id": "second"}]
    assert _prepare_prediction_output(output, rows, resume=False) == 0
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        _append_prediction(handle, {"task_id": "first", "prediction": "one", "parsed": {}})
    assert _prepare_prediction_output(output, rows, resume=True) == 1
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        _append_prediction(handle, {"task_id": "second", "prediction": "two", "parsed": {}})
    assert _prepare_prediction_output(output, rows, resume=True) == 2


def test_qwen_prediction_resume_rejects_misaligned_task_ids(tmp_path) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text('{"task_id":"wrong"}\n', encoding="utf-8")
    try:
        _prepare_prediction_output(output, [{"task_id": "expected"}], resume=True)
    except ValueError as error:
        assert "task_id mismatch" in str(error)
    else:
        raise AssertionError("Expected resume task-id validation to fail")


def test_qwen_config_rejects_large_batch(tmp_path) -> None:
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "model_id": "Qwen/Qwen3-VL-2B-Instruct",
                "task": "grounded_vqa",
                "train_jsonl": "data/train.jsonl",
                "adapter": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj"]},
                "runtime": {
                    "output_dir": "runs/test",
                    "batch_size": 2,
                    "gradient_accumulation": 1,
                    "learning_rate": 0.00002,
                    "max_steps": 1,
                    "max_target_tokens": 64,
                    "min_pixels": 50176,
                    "max_pixels": 100352,
                    "gradient_checkpointing": True,
                    "seed": 7,
                    "eval_batches": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    try:
        load_qwen_experiment(path)
    except ValueError as error:
        assert "batch_size=1" in str(error)
    else:
        raise AssertionError("Expected RTX 4050 batch-size validation to fail")


def test_qwen_progress_eta_format_is_stable() -> None:
    assert _format_seconds(0) == "00:00:00"
    assert _format_seconds(3661.2) == "01:01:01"


def test_qwen_durable_checkpoint_round_trip(tmp_path) -> None:
    class AdapterOnlyModel:
        def save_pretrained(self, path) -> None:
            path.mkdir(parents=True)
            (path / "adapter.bin").write_bytes(b"adapter")

    config_path = tmp_path / "config.json"
    train = tmp_path / "train.jsonl"
    config_path.write_text("{}", encoding="utf-8")
    train.write_text('{"task_id":"one"}\n', encoding="utf-8")
    config = SimpleNamespace(
        config_path=config_path,
        train_jsonl=train,
        eval_jsonl=None,
        runtime=SimpleNamespace(output_dir=tmp_path / "run"),
    )
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    _save_resume_checkpoint(config, AdapterOnlyModel(), optimizer, scheduler, 1, 8, 12.5, 1, [{"task_id": "one"}])
    checkpoint, state = _load_resume_checkpoint(config, [{"task_id": "one"}])
    assert checkpoint.name == "slot-a"
    assert state["next_epoch"] == 1
    assert state["rows_completed_in_epoch"] == 8
    assert state["epoch_loss_sum"] == 12.5


def test_qwen_r2_export_uses_explicit_image_root_and_keeps_two_view_targets(tmp_path) -> None:
    release = tmp_path / "r2"
    images = tmp_path / "bs80k-imaging-raw"
    output = tmp_path / "qwen-r2"
    release.mkdir()
    for view in ("wholeBodyANT", "wholeBodyPOST"):
        directory = images / view
        directory.mkdir(parents=True)
        (directory / "1.jpg").write_bytes(b"image")
    common = {
        "split": "train",
        "images": [
            {"view": "ANT", "image": "bs80k/wholeBodyANT/1.jpg", "image_size": [100, 200]},
            {"view": "POST", "image": "bs80k/wholeBodyPOST/1.jpg", "image_size": [100, 200]},
        ],
        "label": "head",
        "target": {
            "classification": {"label": "benign"},
            "locations": [{"view": "ANT", "bbox": [1, 2, 20, 40]}, {"view": "POST", "bbox": [3, 4, 22, 42]}],
        },
        "targets": [{"view": "ANT", "bbox": [1, 2, 20, 40]}, {"view": "POST", "bbox": [3, 4, 22, 42]}],
    }
    rows = [
        {**common, "task_id": "wbbs:pair:1:head:grounding:both", "interaction_type": "localization", "query": "Locate the head."},
        {
            **common,
            "task_id": "wbbs:pair:1:head:grounded-vqa:both",
            "interaction_type": "localization_and_classification",
            "question": "Locate the head and determine whether it is abnormal.",
            "answer": "The head is normal.",
            "answer_label": "normal",
        },
    ]
    vqa = {
        "split": "train",
        "task_id": "wbbs:patient:1:whole-body:vqa:ant",
        "images": [{"view": "ANT", "image": "bs80k/wholeBodyANT/1.jpg", "image_size": [100, 200]}],
        "label": "whole_body",
        "question": "Is the scan abnormal?",
        "answer": "No.",
        "answer_label": "no",
    }
    write_jsonl(release / "vgrounding.jsonl", rows)
    write_jsonl(release / "vqa.jsonl", [vqa])

    summary = build_qwen_exports(release, output, schema="json", image_root=images)

    assert summary["primary_grounded_vqa"] == {"train": 1, "val": 0, "test": 0}
    exported = [json.loads(line) for line in (output / "grounded_vqa_train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(exported) == 1
    assert len(json.loads(exported[0]["target"])["boxes"]) == 2
    assert (output / exported[0]["images"][0]["image"]).resolve() == (images / "wholeBodyANT" / "1.jpg").resolve()
    assert (output / "SHA256SUMS.txt").is_file()


def test_qwen_r3_export_recovers_patient_id_from_paired_evidence(tmp_path) -> None:
    release = tmp_path / "r3"
    images = tmp_path / "bs80k-imaging-raw"
    output = tmp_path / "qwen-r3"
    release.mkdir()
    for directory in ("wholeBodyANT", "wholeBodyPOST"):
        (images / directory).mkdir(parents=True)
        (images / directory / "1.jpg").write_bytes(b"image")
    write_jsonl(release / "paired_evidence.jsonl", [{"evidence_id": "pair-1", "patient_id": "p1"}])
    write_jsonl(
        release / "multitask.jsonl",
        [
            {
                "task_id": "vqa-1",
                "task_type": "vqa",
                "evidence_id": "pair-1",
                "split": "train",
                "images": [
                    {"view": "ANT", "image": "bs80k/wholeBodyANT/1.jpg", "image_size": [100, 200]},
                    {"view": "POST", "image": "bs80k/wholeBodyPOST/1.jpg", "image_size": [100, 200]},
                ],
                "question": "Is the region abnormal?",
                "answer": "No.",
                "answer_label": "no",
                "label": "head",
            }
        ],
    )

    build_qwen_exports(release, output, schema="multitask_json", image_root=images)

    exported = [json.loads(line) for line in (output / "multitask_train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert exported[0]["patient_id"] == "p1"


def test_qwen_multitask_json_and_metric_cover_vqa_grounding_and_grounded_vqa() -> None:
    expected = [
        {
            "task_id": "vqa-1",
            "task": "vqa",
            "answer_label": "no",
            "target": '{"task":"vqa","answer":"No abnormal uptake.","class":"no"}',
        },
        {
            "task_id": "vqa-2",
            "task": "vqa",
            "answer_label": "yes",
            "target": '{"task":"vqa","answer":"Abnormal uptake.","class":"yes"}',
        },
        {
            "task_id": "ground-1",
            "task": "grounding",
            "label": "left knee",
            "targets": [{"view": "ANT", "bbox": [1, 2, 20, 40]}],
            "target": '{"task":"grounding","region":"left_knee","boxes":[{"view":"ANT","bbox":[1,2,20,40]}]}',
        },
        {
            "task_id": "grounded-1",
            "task": "grounded_vqa",
            "answer_label": "normal",
            "label": "head",
            "targets": [{"view": "PST", "bbox": [3, 4, 22, 42]}],
            "target": '{"task":"grounded_vqa","answer":"The head is normal.","region":"head","class":"normal","boxes":[{"view":"PST","bbox":[3,4,22,42]}]}',
        },
    ]
    predictions = [{"task_id": row["task_id"], "prediction": row["target"]} for row in expected]
    report = evaluate_qwen_multitask_predictions(expected, predictions)
    assert report["model_selection_score"] == 1.0
    assert report["vqa"]["macro_f1"] == 1.0
    assert report["vqa"]["per_class"]["yes"]["recall"] == 1.0
    assert parse_qwen_multitask_json_prediction(predictions[0]["prediction"])["answer_label"] == "no"


def test_qwen_multitask_patient_cluster_bootstrap_uses_patient_ids() -> None:
    expected = [
        {"task_id": "vqa-1", "patient_id": "p1", "task": "vqa", "answer_label": "yes"},
        {"task_id": "vqa-2", "patient_id": "p1", "task": "vqa", "answer_label": "no"},
        {"task_id": "vqa-3", "patient_id": "p2", "task": "vqa", "answer_label": "yes"},
        {"task_id": "vqa-4", "patient_id": "p2", "task": "vqa", "answer_label": "no"},
    ]
    predictions = [
        {"task_id": "vqa-1", "prediction": '{"task":"vqa","answer":"Abnormal uptake.","class":"yes"}'},
        {"task_id": "vqa-2", "prediction": '{"task":"vqa","answer":"No abnormal uptake.","class":"no"}'},
        {"task_id": "vqa-3", "prediction": '{"task":"vqa","answer":"Abnormal uptake.","class":"yes"}'},
        {"task_id": "vqa-4", "prediction": '{"task":"vqa","answer":"No abnormal uptake.","class":"no"}'},
    ]
    report = patient_cluster_bootstrap_qwen_multitask(expected, predictions, resamples=10, seed=7)
    assert report["patients"] == 2
    assert report["metrics"]["vqa.balanced_accuracy"]["valid_resamples"] == 10
    assert report["metrics"]["vqa.balanced_accuracy"]["lower_95"] == 1.0
