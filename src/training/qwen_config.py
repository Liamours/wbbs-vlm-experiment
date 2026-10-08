from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from project_paths import PROJECT_ROOT
from typing import Literal


TaskName = Literal["vqa", "grounding", "grounded_vqa", "multitask"]
OutputSchema = Literal["tags", "json", "multitask_json"]
SamplingMode = Literal["uniform", "balanced_family_class", "epoch_uniform"]


@dataclass(frozen=True)
class QwenAdapterSettings:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class QwenRuntimeSettings:
    output_dir: Path
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    warmup_ratio: float
    max_grad_norm: float
    max_steps: int | None
    epochs: int | None
    max_target_tokens: int
    max_pixels: int
    min_pixels: int
    gradient_checkpointing: bool
    seed: int
    eval_batches: int
    validation_interval: int
    checkpoint_interval: int
    inference_batch_size: int
    logging_interval: int
    minimum_free_disk_gb: float


@dataclass(frozen=True)
class QwenSamplingSettings:
    mode: SamplingMode


@dataclass(frozen=True)
class QwenExperiment:
    config_path: Path
    model_id: str
    task: TaskName
    output_schema: OutputSchema
    train_jsonl: Path
    eval_jsonl: Path | None
    adapter: QwenAdapterSettings
    runtime: QwenRuntimeSettings
    sampling: QwenSamplingSettings
    selection_metric: str


def load_qwen_experiment(path: str | Path) -> QwenExperiment:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    root = PROJECT_ROOT
    adapter = payload["adapter"]
    runtime = payload["runtime"]
    result = QwenExperiment(
        config_path=config_path,
        model_id=payload["model_id"],
        task=payload["task"],
        output_schema=payload.get("output_schema", "tags"),
        train_jsonl=(root / payload["train_jsonl"]).resolve(),
        eval_jsonl=(root / payload["eval_jsonl"]).resolve() if payload.get("eval_jsonl") else None,
        adapter=QwenAdapterSettings(
            rank=adapter["rank"],
            alpha=adapter["alpha"],
            dropout=adapter["dropout"],
            target_modules=tuple(adapter["target_modules"]),
        ),
        runtime=QwenRuntimeSettings(
            output_dir=(root / runtime["output_dir"]).resolve(),
            batch_size=runtime["batch_size"],
            gradient_accumulation=runtime["gradient_accumulation"],
            learning_rate=runtime["learning_rate"],
            weight_decay=runtime.get("weight_decay", 0.01),
            adam_beta1=runtime.get("adam_beta1", 0.9),
            adam_beta2=runtime.get("adam_beta2", 0.999),
            adam_epsilon=runtime.get("adam_epsilon", 1e-8),
            warmup_ratio=runtime.get("warmup_ratio", 0.0),
            max_grad_norm=runtime.get("max_grad_norm", 1.0),
            max_steps=runtime.get("max_steps"),
            epochs=runtime.get("epochs"),
            max_target_tokens=runtime["max_target_tokens"],
            max_pixels=runtime["max_pixels"],
            min_pixels=runtime["min_pixels"],
            gradient_checkpointing=runtime["gradient_checkpointing"],
            seed=runtime["seed"],
            eval_batches=runtime["eval_batches"],
            validation_interval=runtime.get("validation_interval", 0),
            checkpoint_interval=runtime.get("checkpoint_interval", 0),
            inference_batch_size=runtime.get("inference_batch_size", 1),
            logging_interval=runtime.get("logging_interval", 1),
            minimum_free_disk_gb=runtime.get("minimum_free_disk_gb", 10.0),
        ),
        sampling=QwenSamplingSettings(mode=payload.get("sampling", {}).get("mode", "uniform")),
        selection_metric=payload.get("selection_metric", "validation_loss"),
    )
    _validate(result)
    return result


_SUPPORTED_MODEL_FAMILIES = ("qwen3-vl", "medgemma")


def _validate(config: QwenExperiment) -> None:
    if not any(family in config.model_id.lower() for family in _SUPPORTED_MODEL_FAMILIES):
        raise ValueError(f"model_id must be one of {_SUPPORTED_MODEL_FAMILIES} (both load via AutoModelForImageTextToText)")
    if config.task not in {"vqa", "grounding", "grounded_vqa", "multitask"}:
        raise ValueError(f"Unsupported task: {config.task}")
    if config.output_schema not in {"tags", "json", "multitask_json"}:
        raise ValueError("output_schema must be tags, json, or multitask_json")
    if config.sampling.mode not in {"uniform", "balanced_family_class", "epoch_uniform"}:
        raise ValueError("sampling.mode must be uniform, balanced_family_class, or epoch_uniform")
    if config.task == "multitask" and config.output_schema != "multitask_json":
        raise ValueError("multitask Qwen experiments require output_schema=multitask_json")
    if config.task != "multitask" and config.output_schema == "multitask_json":
        raise ValueError("multitask_json is only valid for task=multitask")
    if config.task == "multitask" and config.sampling.mode != "epoch_uniform":
        raise ValueError("multitask Qwen experiments require sampling.mode=epoch_uniform")
    if config.sampling.mode == "balanced_family_class" and config.runtime.gradient_accumulation % 4:
        raise ValueError("balanced_family_class requires gradient_accumulation divisible by 4")
    if config.runtime.batch_size != 1:
        raise ValueError("Qwen3-VL paired-image training currently requires batch_size=1")
    if config.adapter.rank < 1:
        raise ValueError("adapter.rank must be positive")
    if not config.adapter.target_modules:
        raise ValueError("adapter.target_modules must not be empty")
    if not 0 <= config.adapter.dropout < 1:
        raise ValueError("adapter.dropout must be in [0, 1)")
    if config.runtime.gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    if (config.runtime.max_steps is None) == (config.runtime.epochs is None):
        raise ValueError("Configure exactly one of runtime.max_steps or runtime.epochs")
    if config.runtime.max_steps is not None and config.runtime.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if config.runtime.epochs is not None and config.runtime.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.runtime.validation_interval < 0 or config.runtime.checkpoint_interval < 0:
        raise ValueError("validation_interval and checkpoint_interval must not be negative")
    if config.runtime.inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive")
    if config.runtime.logging_interval < 1:
        raise ValueError("logging_interval must be positive")
    if config.runtime.minimum_free_disk_gb <= 0:
        raise ValueError("minimum_free_disk_gb must be positive")
    if config.runtime.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= config.runtime.weight_decay <= 1:
        raise ValueError("weight_decay must be between 0 and 1")
    if not 0 <= config.runtime.adam_beta1 < 1 or not 0 <= config.runtime.adam_beta2 < 1:
        raise ValueError("Adam beta values must be in [0, 1)")
    if config.runtime.adam_epsilon <= 0:
        raise ValueError("adam_epsilon must be positive")
    if not 0 <= config.runtime.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if config.runtime.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if config.runtime.max_target_tokens < 1:
        raise ValueError("max_target_tokens must be positive")
    if config.runtime.min_pixels < 28 * 28 or config.runtime.min_pixels > config.runtime.max_pixels:
        raise ValueError("min_pixels must be at least 784 and no greater than max_pixels")
    if not config.runtime.gradient_checkpointing:
        raise ValueError("Qwen3-VL paired-image QLoRA requires gradient_checkpointing")
    if config.selection_metric not in {"validation_loss", "multitask_macro"}:
        raise ValueError("selection_metric must be validation_loss or multitask_macro")
    if config.task == "multitask" and config.selection_metric != "multitask_macro":
        raise ValueError("multitask Qwen experiments require selection_metric=multitask_macro")
