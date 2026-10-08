import json

import pytest

from training.config import load_experiment


def test_4050_profile_rejects_large_batch(tmp_path):
    config = {
        "model_id": "google/paligemma-3b-pt-224",
        "task": "vqa",
        "train_jsonl": "data/train.jsonl",
        "adapter": {"method": "qlora", "rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj"]},
        "runtime": {"output_dir": "runs/test", "batch_size": 2, "gradient_accumulation": 8, "learning_rate": 0.0001, "max_steps": 1, "max_target_tokens": 128, "gradient_checkpointing": True, "seed": 7},
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="batch_size=1"):
        load_experiment(path)


def test_legacy_qlora_maps_to_4bit_lora(tmp_path):
    config = {
        "model_id": "google/paligemma-3b-pt-224",
        "task": "vqa",
        "train_jsonl": "data/train.jsonl",
        "adapter": {"method": "qlora", "rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": "all-linear"},
        "runtime": {"output_dir": "runs/test", "batch_size": 1, "gradient_accumulation": 8, "learning_rate": 0.0001, "max_steps": 10, "max_target_tokens": 128, "gradient_checkpointing": True, "seed": 7},
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config))
    experiment = load_experiment(path)
    assert experiment.quantization == "4bit"
    assert experiment.adapter.method == "lora"
    assert experiment.adapter.target_modules == "all-linear"


def test_multitask_task_is_accepted(tmp_path):
    config = {
        "model_id": "google/paligemma-3b-pt-224",
        "task": "multitask",
        "train_jsonl": "data/train.jsonl",
        "adapter": {"method": "lora", "rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
        "runtime": {"output_dir": "runs/test", "batch_size": 1, "gradient_accumulation": 8, "learning_rate": 0.00005, "max_steps": 10, "max_target_tokens": 96, "gradient_checkpointing": True, "seed": 5070},
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config))
    experiment = load_experiment(path)
    assert experiment.task == "multitask"


def test_adalora_requires_rank_schedule(tmp_path):
    config = {
        "model_id": "google/paligemma-3b-pt-224",
        "task": "vqa",
        "train_jsonl": "data/train.jsonl",
        "adapter": {"method": "adalora", "rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj"]},
        "runtime": {"output_dir": "runs/test", "batch_size": 1, "gradient_accumulation": 1, "learning_rate": 0.0001, "max_steps": 2, "max_target_tokens": 128, "gradient_checkpointing": True, "seed": 7},
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="at least 3 optimizer steps"):
        load_experiment(path)
