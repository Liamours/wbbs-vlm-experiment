from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from project_paths import PROJECT_ROOT
from typing import Literal


TaskName = Literal["caption", "vqa", "grounding", "grounded_vqa", "multitask"]
AdapterName = Literal["lora", "rslora", "dora", "adalora"]
QuantizationName = Literal["none", "4bit"]


@dataclass(frozen=True)
class AdapterSettings:
    method: AdapterName
    rank: int
    alpha: int
    dropout: float
    target_modules: str | tuple[str, ...]


@dataclass(frozen=True)
class RuntimeSettings:
    output_dir: Path
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    max_steps: int
    max_prompt_tokens: int
    max_target_tokens: int
    gradient_checkpointing: bool
    seed: int
    eval_batches: int
    checkpoint_interval: int


@dataclass(frozen=True)
class Experiment:
    model_id: str
    task: TaskName
    train_jsonl: Path
    eval_jsonl: Path | None
    quantization: QuantizationName
    adapter: AdapterSettings
    runtime: RuntimeSettings


def load_experiment(path: str | Path) -> Experiment:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    root = PROJECT_ROOT
    raw_adapter = payload["adapter"]
    raw_method = raw_adapter["method"]
    # `qlora` was used by the first scaffold. Keep that shorthand compatible,
    # but model quantization and adapter type are independent settings now.
    method = "lora" if raw_method == "qlora" else raw_method
    target_modules = raw_adapter["target_modules"]
    if isinstance(target_modules, list):
        target_modules = tuple(target_modules)
    if not isinstance(target_modules, (str, tuple)):
        raise ValueError("adapter.target_modules must be a string or list of strings")
    adapter = AdapterSettings(
        method=method,
        rank=raw_adapter["rank"],
        alpha=raw_adapter["alpha"],
        dropout=raw_adapter["dropout"],
        target_modules=target_modules,
    )
    runtime = RuntimeSettings(
        output_dir=(root / payload["runtime"]["output_dir"]).resolve(),
        batch_size=payload["runtime"]["batch_size"],
        gradient_accumulation=payload["runtime"]["gradient_accumulation"],
        learning_rate=payload["runtime"]["learning_rate"],
        max_steps=payload["runtime"]["max_steps"],
        max_prompt_tokens=payload["runtime"].get("max_prompt_tokens", 64),
        max_target_tokens=payload["runtime"]["max_target_tokens"],
        gradient_checkpointing=payload["runtime"]["gradient_checkpointing"],
        seed=payload["runtime"]["seed"],
        eval_batches=payload["runtime"].get("eval_batches", 32),
        checkpoint_interval=payload["runtime"].get("checkpoint_interval", 0),
    )
    experiment = Experiment(
        model_id=payload["model_id"],
        task=payload["task"],
        train_jsonl=(root / payload["train_jsonl"]).resolve(),
        eval_jsonl=(root / payload["eval_jsonl"]).resolve() if payload.get("eval_jsonl") else None,
        quantization=payload.get("quantization", "4bit" if raw_method == "qlora" else "none"),
        adapter=adapter,
        runtime=runtime,
    )
    _validate(experiment)
    return experiment


def _validate(config: Experiment) -> None:
    if config.task not in {"caption", "vqa", "grounding", "grounded_vqa", "multitask"}:
        raise ValueError(f"Unsupported task: {config.task}")
    if "paligemma" not in config.model_id.lower():
        raise ValueError("This standalone recipe currently supports PaliGemma checkpoints only")
    if config.quantization not in {"none", "4bit"}:
        raise ValueError(f"Unsupported quantization: {config.quantization}")
    if config.adapter.method not in {"lora", "rslora", "dora", "adalora"}:
        raise ValueError(f"Unsupported adapter: {config.adapter.method}")
    if config.adapter.rank < 1:
        raise ValueError("Adapter rank must be positive")
    if not 0 <= config.adapter.dropout < 1:
        raise ValueError("Adapter dropout must be in [0, 1)")
    if config.runtime.batch_size != 1:
        raise ValueError("The RTX 4050 profile intentionally uses batch_size=1")
    if not 1 <= config.runtime.max_prompt_tokens <= 128:
        raise ValueError("Use 1–128 prompt tokens in the RTX 4050 profile")
    if not 1 <= config.runtime.max_target_tokens <= 256:
        raise ValueError("Use at most 256 target tokens in the RTX 4050 profile")
    if config.runtime.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if config.runtime.gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    if config.runtime.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.runtime.eval_batches < 1:
        raise ValueError("eval_batches must be positive")
    if config.runtime.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must not be negative")
    if config.quantization == "4bit" and not config.runtime.gradient_checkpointing:
        raise ValueError("QLoRA profile requires gradient checkpointing")
    if config.adapter.method == "adalora" and config.runtime.max_steps < 3:
        raise ValueError("AdaLoRA needs at least 3 optimizer steps for its rank schedule")
