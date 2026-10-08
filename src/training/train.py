from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import replace
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from .adapters import load_model
from .config import Experiment, load_experiment
from .data import Example, load_examples, open_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune PaliGemma adapters on a small GPU")
    parser.add_argument("--config", required=True, help="Path to an experiment JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Validate the configuration and dataset without loading a model")
    parser.add_argument("--max-steps", type=int, help="Override max_steps for a smoke test without changing the experiment config")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest durable checkpoint under output_dir/checkpoints")
    args = parser.parse_args()
    config = load_experiment(args.config)
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        config = replace(config, runtime=replace(config.runtime, max_steps=args.max_steps))
    if args.dry_run:
        _dry_run(config)
        return
    train(config, args.resume)


def train(config: Experiment, resume: bool = False) -> None:
    _seed(config.runtime.seed)
    try:
        processor = AutoProcessor.from_pretrained(config.model_id)
    except OSError as error:
        if "gated repo" in str(error).lower() or "restricted" in str(error).lower():
            raise RuntimeError(
                f"Model access is required for {config.model_id}. Accept its Hugging Face terms and authenticate with an authorized account before training."
            ) from error
        raise
    examples = load_examples(config.train_jsonl, config.task)
    loader = _loader(examples, processor, config, shuffle=True)
    eval_examples = load_examples(config.eval_jsonl, config.task) if config.eval_jsonl else None
    eval_loader = _loader(eval_examples, processor, config, shuffle=False) if eval_examples else None

    checkpoint = _latest_checkpoint(config) if resume else None
    if resume and checkpoint is None:
        raise FileNotFoundError(f"Cannot resume: no checkpoint under {_checkpoint_root(config)}")
    model = load_model(config, resume_adapter=checkpoint)
    model.print_trainable_parameters()
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config.runtime.learning_rate)
    optimizer_steps = 0
    if checkpoint is not None:
        optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cuda", weights_only=False))
        optimizer_steps = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))["optimizer_steps"]
        print(f"resumed_checkpoint={checkpoint} optimizer_steps={optimizer_steps}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    micro_steps = 0
    loss_value = 0.0
    # One full pass over `examples` is one epoch; at batch_size=1 that's
    # len(examples) micro-steps. Used only to label epoch-boundary snapshots
    # (checkpoints/epoch-NN) for validation-based checkpoint selection,
    # mirroring the Qwen3-VL multitask trainer's per-epoch checkpoints.
    steps_per_epoch = _steps_per_epoch(len(examples), config.runtime.gradient_accumulation)
    with tqdm(total=config.runtime.max_steps, initial=optimizer_steps, desc="Training", unit="step") as progress:
        while optimizer_steps < config.runtime.max_steps:
            for batch in loader:
                batch = {key: value.to("cuda") for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(**batch).loss / config.runtime.gradient_accumulation
                loss.backward()
                micro_steps += 1
                loss_value = loss.item() * config.runtime.gradient_accumulation
                if micro_steps % config.runtime.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer_steps += 1
                    if config.adapter.method == "adalora":
                        model.base_model.update_and_allocate(optimizer_steps - 1)
                    optimizer.zero_grad(set_to_none=True)
                    allocated = torch.cuda.max_memory_allocated() / 1024**3
                    progress.set_postfix(loss=f"{loss_value:.4f}", peak_vram=f"{allocated:.2f}GB")
                    progress.update(1)
                    if config.runtime.checkpoint_interval and optimizer_steps % config.runtime.checkpoint_interval == 0:
                        _save_checkpoint(model, optimizer, config, optimizer_steps)
                    if optimizer_steps % steps_per_epoch == 0:
                        _save_epoch_checkpoint(model, config, optimizer_steps // steps_per_epoch)
                if optimizer_steps >= config.runtime.max_steps:
                    break

    if eval_loader is not None:
        print(f"eval_loss={_evaluate(model, eval_loader, config.runtime.eval_batches):.4f}")
    _save_adapter(model, config)


def _loader(examples: list[Example], processor, config: Experiment, shuffle: bool) -> DataLoader:
    return DataLoader(
        examples,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=lambda batch: _collate(
            processor,
            batch,
            config.runtime.max_prompt_tokens,
            config.runtime.max_target_tokens,
        ),
    )


def _collate(processor, examples: list[Example], max_prompt_tokens: int, max_target_tokens: int):
    return processor(
        text=[_cap_text(processor, example.prompt, max_prompt_tokens) for example in examples],
        images=open_images(examples),
        suffix=[_cap_text(processor, example.target, max_target_tokens) for example in examples],
        return_tensors="pt",
        padding="longest",
        # Prompt and suffix are capped independently above. Never truncate the
        # combined sequence because that can drop the image placeholder tokens.
        truncation=False,
    )


def _cap_text(processor, text: str, max_tokens: int) -> str:
    token_ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"][:max_tokens]
    return processor.tokenizer.decode(token_ids, skip_special_tokens=False)


@torch.no_grad()
def _evaluate(model, loader: DataLoader, max_batches: int) -> float:
    model.eval()
    losses: list[float] = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = {key: value.to("cuda") for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            losses.append(model(**batch).loss.item())
    model.train()
    return sum(losses) / len(losses)


def _checkpoint_root(config: Experiment) -> Path:
    return config.runtime.output_dir / "checkpoints"


def _latest_checkpoint(config: Experiment) -> Path | None:
    root = _checkpoint_root(config)
    if not root.is_dir():
        return None
    checkpoints = sorted(root.glob("step-*"), key=lambda path: int(path.name.removeprefix("step-")))
    return checkpoints[-1] if checkpoints else None


def _save_checkpoint(model, optimizer, config: Experiment, step: int) -> None:
    checkpoint = _checkpoint_root(config) / f"step-{step:06d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint)
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    (checkpoint / "state.json").write_text(json.dumps({"optimizer_steps": step}, indent=2), encoding="utf-8")
    print(f"saved_checkpoint={checkpoint}")


def _steps_per_epoch(example_count: int, gradient_accumulation: int) -> int:
    return math.ceil(example_count / gradient_accumulation)


def _save_epoch_checkpoint(model, config: Experiment, epoch: int) -> None:
    checkpoint = _checkpoint_root(config) / f"epoch-{epoch:02d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint)
    print(f"saved_epoch_checkpoint={checkpoint}")


def _save_adapter(model, config: Experiment) -> None:
    output_dir = config.runtime.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    state = {
        "model_id": config.model_id,
        "task": config.task,
        "quantization": config.quantization,
        "adapter": config.adapter.method,
        "max_steps": config.runtime.max_steps,
    }
    (output_dir / "training_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Saved adapter to {output_dir}")


def _dry_run(config: Experiment) -> None:
    examples = load_examples(config.train_jsonl, config.task)
    print(
        f"ready task={config.task} examples={len(examples)} model={config.model_id} "
        f"quantization={config.quantization} adapter={config.adapter.method}"
    )


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
