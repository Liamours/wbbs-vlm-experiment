from __future__ import annotations

from pathlib import Path

import torch
from peft import AdaLoraConfig, LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, PaliGemmaForConditionalGeneration

from .config import Experiment


def load_model(config: Experiment, resume_adapter: Path | None = None):
    base = _load_base_model(config)
    if resume_adapter is not None:
        return PeftModel.from_pretrained(base, resume_adapter, is_trainable=True)
    return get_peft_model(base, _peft_config(config))


def _load_base_model(config: Experiment):
    if not torch.cuda.is_available():
        raise RuntimeError("This baseline requires a CUDA GPU")

    load_args: dict[str, object] = {"device_map": {"": 0}}
    if config.quantization == "4bit":
        load_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        load_args["torch_dtype"] = torch.float16

    model = PaliGemmaForConditionalGeneration.from_pretrained(config.model_id, **load_args)
    model.config.use_cache = False
    if config.runtime.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    # transformers moved PaliGemma's vision tower under `.model` in newer releases.
    vision_tower = model.vision_tower if hasattr(model, "vision_tower") else model.model.vision_tower
    for parameter in vision_tower.parameters():
        parameter.requires_grad = False
    if config.quantization == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.runtime.gradient_checkpointing)
    return model


def _peft_config(config: Experiment):
    target_modules = list(config.adapter.target_modules) if isinstance(config.adapter.target_modules, tuple) else config.adapter.target_modules
    common = dict(
        r=config.adapter.rank,
        lora_alpha=config.adapter.alpha,
        lora_dropout=config.adapter.dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    if config.adapter.method == "lora":
        return LoraConfig(**common)
    if config.adapter.method == "rslora":
        return LoraConfig(**common, use_rslora=True)
    if config.adapter.method == "dora":
        return LoraConfig(**common, use_dora=True)
    if config.adapter.method == "adalora":
        tinit, tfinal, delta_t = _adalora_schedule(config.runtime.max_steps)
        return AdaLoraConfig(
            lora_alpha=config.adapter.alpha,
            lora_dropout=config.adapter.dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            init_r=config.adapter.rank,
            target_r=max(1, config.adapter.rank // 2),
            tinit=tinit,
            tfinal=tfinal,
            deltaT=delta_t,
            total_step=config.runtime.max_steps,
        )
    raise ValueError(f"Unsupported adapter: {config.adapter.method}")


def _adalora_schedule(total_steps: int) -> tuple[int, int, int]:
    tinit = max(1, min(10, total_steps // 10))
    tfinal = max(1, min(20, total_steps // 5))
    delta_t = max(1, (total_steps - tinit - tfinal) // 10)
    return tinit, tfinal, delta_t
