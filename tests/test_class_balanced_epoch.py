from __future__ import annotations

from collections import Counter

from training.sampling import class_balanced_epoch


def _rows():
    rows = []
    for i in range(100):
        rows.append({"task_id": f"v{i}", "task": "vqa", "answer_label": "yes" if i < 10 else "no"})
        rows.append({"task_id": f"g{i}", "task": "grounded_vqa", "answer_label": "abnormal" if i < 5 else "normal"})
        rows.append({"task_id": f"b{i}", "task": "grounding", "answer_label": "normal"})
    return rows


def test_epoch_has_the_source_size_and_half_positive_rows_per_task():
    rows = _rows()
    epoch = class_balanced_epoch(rows, 7)
    assert len(epoch) == len(rows)
    for task, positive in (("vqa", "yes"), ("grounded_vqa", "abnormal")):
        labels = Counter(r["answer_label"] for r in epoch if r["task"] == task)
        assert labels[positive] == 50 and sum(labels.values()) == 100
    assert sorted(r["task_id"] for r in epoch if r["task"] == "grounding") == sorted(r["task_id"] for r in rows if r["task"] == "grounding")


def test_positives_are_used_evenly_negatives_once_and_the_order_is_repeatable():
    rows = _rows()
    epoch = class_balanced_epoch(rows, 7)
    uses = Counter(r["task_id"] for r in epoch if r["task"] == "vqa" and r["answer_label"] == "yes")
    assert set(uses.values()) == {5}
    negatives = Counter(r["task_id"] for r in epoch if r["task"] == "vqa" and r["answer_label"] == "no")
    assert max(negatives.values()) == 1
    assert [r["task_id"] for r in epoch] == [r["task_id"] for r in class_balanced_epoch(rows, 7)]
    assert [r["task_id"] for r in epoch] != [r["task_id"] for r in class_balanced_epoch(rows, 8)]


def test_new_configs_load_and_stop_after_the_first_of_two_epochs():
    from training.qwen_config import load_qwen_experiment

    root = __import__("pathlib").Path(__file__).resolve().parents[1] / "configs"
    blind = load_qwen_experiment(root / "qwen3vl_r3_blind_image_epoch1_4050.json")
    balanced = load_qwen_experiment(root / "qwen3vl_r3_class_balanced_epoch1_4050.json")
    reported = load_qwen_experiment(root / "qwen3vl_r3_multitask_2epoch_5070_fast.json")
    for config in (blind, balanced):
        assert config.runtime.epochs == 2 and config.runtime.stop_after_epoch == 1 and config.runtime.seed == reported.runtime.seed == 5070
        assert config.runtime.learning_rate == reported.runtime.learning_rate and config.runtime.gradient_accumulation == reported.runtime.gradient_accumulation
    assert blind.sampling.mode == "epoch_uniform" and "blind" in str(blind.train_jsonl)
    assert balanced.sampling.mode == "class_balanced" and balanced.train_jsonl == reported.train_jsonl
