from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from postprocess.qwen import parse_qwen_json_prediction, parse_qwen_multitask_json_prediction, parse_qwen_prediction
from preprocess.canonical import read_jsonl

from .qwen_config import QwenExperiment, load_qwen_experiment


TAG_SYSTEM_PROMPT = (
    "You are a nuclear medicine imaging assistant. The user provides paired whole-body bone scans. "
    "The first image is anterior (ANT) and the second is posterior (PST). "
    "Use only the images. Give a concise clinical answer, then preserve any requested <REG>, <CLS>, "
    "<ANT>, <PST>, and <BBX> structures exactly."
)
JSON_SYSTEM_PROMPT = (
    "You are a nuclear medicine imaging assistant. The user provides paired whole-body bone scans. "
    "The first image is anterior (ANT) and the second is posterior (PST). "
    "Use only the images. Return only a compact JSON object with answer, region, class, and boxes fields. "
    "Each box must contain view (ANT or PST) and bbox ([x1,y1,x2,y2])."
)
MULTITASK_JSON_SYSTEM_PROMPT = (
    "You are a nuclear medicine imaging assistant. The user provides one or two whole-body bone scans. "
    "Use only the images. Return only the compact JSON object requested by the task: VQA requires task, answer, and class; "
    "grounding requires task, region, and boxes; grounded VQA requires task, answer, region, class, and boxes. "
    "Each box must contain view (ANT or PST) and bbox ([x1,y1,x2,y2])."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL adapters on paired WBBS tasks.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-examples", type=int)
    parser.add_argument("--resume", action="store_true", help="Resume a conventional epoch run from its latest durable checkpoint.")
    args = parser.parse_args()
    config = load_qwen_experiment(args.config)
    if args.max_steps is not None:
        if args.resume:
            parser.error("--resume is only supported for conventional epoch runs")
        config = replace(config, runtime=replace(config.runtime, max_steps=args.max_steps, epochs=None))
    if args.dry_run:
        dry_run(config, args.limit_examples)
        return
    train(config, args.limit_examples, args.resume)


def predict_main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic Qwen3-VL predictions for paired WBBS tasks.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-examples", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_qwen_experiment(args.config)
    predict(
        config,
        args.adapter,
        args.input,
        args.output,
        args.limit_examples,
        args.max_new_tokens,
        args.sample_seed,
        args.batch_size or config.runtime.inference_batch_size,
        args.resume,
    )


def dry_run(config: QwenExperiment, limit_examples: int | None = None) -> None:
    rows = _load_rows(config.train_jsonl, config.task, limit_examples)
    _validate_images(rows, config.train_jsonl.parent)
    image_counts = {str(count): sum(len(row["images"]) == count for row in rows) for count in (1, 2)}
    print(f"ready task={config.task} examples={len(rows)} model={config.model_id} image_counts={json.dumps(image_counts, sort_keys=True)}")


def train(config: QwenExperiment, limit_examples: int | None = None, resume: bool = False) -> None:
    if config.runtime.epochs is not None:
        _train_epochs(config, limit_examples, resume)
        return
    if resume:
        raise ValueError("Resume is only supported for conventional epoch runs")
    _require_cuda()
    _seed(config.runtime.seed)
    source_rows = _load_rows(config.train_jsonl, config.task, limit_examples)
    _validate_images(source_rows, config.train_jsonl.parent)
    _initialize_new_run(config, source_rows)
    eval_rows = _load_rows(config.eval_jsonl, config.task, limit_examples) if config.eval_jsonl else []
    random.Random(config.runtime.seed + 1).shuffle(eval_rows)
    assert config.runtime.max_steps is not None
    rows = _scheduled_rows(source_rows, config.runtime.max_steps * config.runtime.gradient_accumulation, config.sampling.mode, config.runtime.seed)
    print(f"sampling_mode={config.sampling.mode} scheduled_counts={json.dumps(_sampling_counts(rows), sort_keys=True)}")
    processor = _load_processor(config)
    model = _load_train_model(config)
    optimizer, scheduler = _optimizer_and_scheduler(config, model, config.runtime.max_steps)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    micro_steps = 0
    optimizer_steps = 0
    index = 0
    running_loss = 0.0
    evaluation = None
    validation_history: list[dict[str, float | int]] = []
    with tqdm(total=config.runtime.max_steps, desc="Training", unit="step") as progress:
        while optimizer_steps < config.runtime.max_steps:
            row = rows[index]
            index += 1
            batch = _to_cuda(_encode_training_example(processor, row, config))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw_loss = model(**batch, use_cache=False).loss
            (raw_loss / config.runtime.gradient_accumulation).backward()
            micro_steps += 1
            running_loss += raw_loss.item()
            if micro_steps % config.runtime.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.runtime.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                peak = torch.cuda.max_memory_allocated() / 1024**3
                progress.set_postfix(loss=f"{running_loss / micro_steps:.4f}", peak_vram=f"{peak:.2f}GB")
                progress.update(1)
                if optimizer_steps % config.runtime.logging_interval == 0:
                    tqdm.write(
                        f"step={optimizer_steps}/{config.runtime.max_steps} "
                        f"loss_mean={running_loss / micro_steps:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} peak_vram={peak:.2f}GB"
                    )
                if eval_rows and config.runtime.validation_interval and optimizer_steps % config.runtime.validation_interval == 0:
                    evaluation = _evaluate(model, processor, eval_rows, config)
                    validation_history.append({"step": optimizer_steps, "loss": evaluation})
                    _write_progress(config, optimizer_steps, validation_history)
                    tqdm.write(f"validation_step={optimizer_steps} loss={evaluation:.4f}")
                if config.runtime.checkpoint_interval and optimizer_steps % config.runtime.checkpoint_interval == 0:
                    _save_checkpoint(model, processor, config, optimizer_steps, evaluation)
    if eval_rows and evaluation is None:
        evaluation = _evaluate(model, processor, eval_rows, config)
        validation_history.append({"step": optimizer_steps, "loss": evaluation})
        _write_progress(config, optimizer_steps, validation_history)
    _save(model, processor, config, evaluation, rows, eval_rows[: config.runtime.eval_batches], validation_history)


def _train_epochs(config: QwenExperiment, limit_examples: int | None, resume: bool = False) -> None:
    """Run conventional full-data epochs without oversampling or dropped microbatches."""

    _require_cuda()
    assert config.runtime.epochs is not None
    source_rows = _load_rows(config.train_jsonl, config.task, limit_examples)
    _validate_images(source_rows, config.train_jsonl.parent)
    resume_state = _load_resume_checkpoint(config, source_rows) if resume else None
    if resume_state is None:
        _seed(config.runtime.seed)
        _initialize_new_run(config, source_rows)
    processor = _load_processor(config)
    steps_per_epoch = math.ceil(len(source_rows) / config.runtime.gradient_accumulation)
    total_optimizer_steps = steps_per_epoch * config.runtime.epochs
    model = _load_train_model(config, resume_state[0] if resume_state else None)
    optimizer, scheduler = _optimizer_and_scheduler(config, model, total_optimizer_steps)
    if resume_state:
        checkpoint, state = resume_state
        optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False))
        scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=False))
        _restore_rng(torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=False))
        start_epoch = int(state["next_epoch"])
        start_row = int(state["rows_completed_in_epoch"])
        optimizer_steps = int(state["optimizer_steps"])
        epoch_loss_at_resume = float(state["epoch_loss_sum"])
        epoch_records = _load_epoch_records(config.runtime.output_dir)
        print(f"resumed_checkpoint={checkpoint} next_epoch={start_epoch} rows_completed_in_epoch={start_row} optimizer_steps={optimizer_steps}")
    else:
        start_epoch = 1
        start_row = 0
        optimizer_steps = 0
        epoch_loss_at_resume = 0.0
        epoch_records = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    resumed_rows = (start_epoch - 1) * len(source_rows) + start_row
    total_rows = len(source_rows) * config.runtime.epochs
    with tqdm(total=total_rows, initial=resumed_rows, desc="Training", unit="row") as progress:
        for epoch in range(start_epoch, config.runtime.epochs + 1):
            rows = list(source_rows)
            random.Random(config.runtime.seed + epoch - 1).shuffle(rows)
            accumulated = 0
            row_start = start_row if epoch == start_epoch else 0
            epoch_loss = epoch_loss_at_resume if epoch == start_epoch else 0.0
            epoch_started = time.perf_counter()
            for index in range(row_start + 1, len(rows) + 1):
                row = rows[index - 1]
                batch = _to_cuda(_encode_training_example(processor, row, config))
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(**batch, use_cache=False).loss
                loss.backward()
                accumulated += 1
                epoch_loss += loss.item()
                progress.set_postfix(loss=f"{epoch_loss / index:.4f}", epoch=f"{epoch}/{config.runtime.epochs}")
                progress.update(1)
                if accumulated == config.runtime.gradient_accumulation or index == len(rows):
                    _average_gradients(model, accumulated)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.runtime.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    if config.runtime.checkpoint_interval and optimizer_steps % config.runtime.checkpoint_interval == 0:
                        _save_resume_checkpoint(config, model, optimizer, scheduler, epoch, index, epoch_loss, optimizer_steps, source_rows)
                    if optimizer_steps % config.runtime.logging_interval == 0 or index == len(rows):
                        peak = torch.cuda.max_memory_allocated() / 1024**3
                        elapsed_epoch = time.perf_counter() - epoch_started
                        completed_rows = (epoch - 1) * len(rows) + index
                        elapsed_total = time.perf_counter() - training_started
                        rows_per_second = (completed_rows - resumed_rows) / elapsed_total
                        remaining_rows = len(rows) * config.runtime.epochs - completed_rows
                        estimated_remaining = remaining_rows / rows_per_second
                        tqdm.write(
                            f"epoch={epoch}/{config.runtime.epochs} step={optimizer_steps}/{total_optimizer_steps} "
                            f"rows={index}/{len(rows)} ({completed_rows}/{len(rows) * config.runtime.epochs}) "
                            f"loss_mean={epoch_loss / index:.4f} lr={optimizer.param_groups[0]['lr']:.2e} "
                            f"epoch_rate={index / elapsed_epoch:.2f}rows/s eta={_format_seconds(estimated_remaining)} "
                            f"peak_vram={peak:.2f}GB"
                        )
                        _write_progress(
                            config,
                            optimizer_steps,
                            [],
                            {
                                "status": "training",
                                "epoch": epoch,
                                "total_epochs": config.runtime.epochs,
                                "completed_rows": completed_rows,
                                "total_rows": len(rows) * config.runtime.epochs,
                                "rows_per_second": round(rows_per_second, 6),
                                "estimated_remaining_seconds": round(estimated_remaining, 1),
                                "peak_vram_gb": round(peak, 3),
                            },
                        )
                    accumulated = 0
            average_loss = epoch_loss / len(rows)
            checkpoint = _save_epoch_checkpoint(model, processor, config, epoch, optimizer_steps, average_loss, rows)
            epoch_records.append(
                {
                    "epoch": epoch,
                    "rows": len(rows),
                    "optimizer_steps": optimizer_steps,
                    "mean_training_loss": round(average_loss, 6),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "checkpoint": str(checkpoint),
                    "order_sha256": _task_id_hash(rows),
                }
            )
            _write_epoch_progress(config, epoch_records)
            if epoch < config.runtime.epochs:
                _save_resume_checkpoint(config, model, optimizer, scheduler, epoch + 1, 0, 0.0, optimizer_steps, source_rows)
        start_row = 0
        epoch_loss_at_resume = 0.0
    latest = config.runtime.output_dir / "latest"
    _copy_adapter(config.runtime.output_dir / "checkpoints" / f"epoch-{config.runtime.epochs:02d}", latest)
    _write_epoch_state(config, source_rows, epoch_records)
    _mark_resume_complete(config, optimizer_steps)
    print(f"saved_latest_adapter={latest}", flush=True)


def predict(
    config: QwenExperiment,
    adapter: str | Path,
    input_path: str | Path,
    output: str | Path,
    limit_examples: int | None = None,
    max_new_tokens: int = 128,
    sample_seed: int | None = None,
    batch_size: int = 1,
    resume: bool = False,
) -> None:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    _require_cuda()
    _seed(config.runtime.seed)
    rows = _load_rows(Path(input_path), config.task, limit_examples, sample_seed)
    _validate_images(rows, Path(input_path).parent)
    output_path = Path(output)
    completed = _prepare_prediction_output(output_path, rows, resume)
    pending_rows = rows[completed:]
    _write_prediction_progress(output_path, config, adapter, input_path, len(rows), completed, batch_size, max_new_tokens, sample_seed, None, False)
    if not pending_rows:
        if not output_path.with_suffix(".metadata.json").exists():
            _write_prediction_metadata(output_path, config, adapter, input_path, len(rows), completed, batch_size, max_new_tokens, sample_seed, 0.0, 0.0, 0.0)
        _write_prediction_progress(output_path, config, adapter, input_path, len(rows), completed, batch_size, max_new_tokens, sample_seed, 0.0, True)
        print(f"prediction already complete rows={completed}")
        return
    processor = _load_processor(config)
    model = _load_inference_model(config, adapter)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    parser = _prediction_parser(config)
    from tqdm import tqdm

    with output_path.open("a", encoding="utf-8", newline="\n") as output_handle:
        with tqdm(
            total=len(rows),
            initial=completed,
            desc="Inference",
            unit="row",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=not sys.stderr.isatty(),
        ) as progress:
            for start in range(0, len(pending_rows), batch_size):
                batch_rows = pending_rows[start : start + batch_size]
                inputs = _to_cuda(_encode_prompt_batch(processor, batch_rows, config))
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                suffix = generated[:, inputs["input_ids"].shape[1] :]
                decoded = processor.batch_decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for row, prediction in zip(batch_rows, decoded, strict=True):
                    prediction = prediction.strip()
                    _append_prediction(output_handle, {"task_id": row["task_id"], "prediction": prediction, "parsed": parser(prediction)})
                    completed += 1
                    elapsed = time.perf_counter() - started
                    _write_prediction_progress(
                        output_path,
                        config,
                        adapter,
                        input_path,
                        len(rows),
                        completed,
                        batch_size,
                        max_new_tokens,
                        sample_seed,
                        elapsed,
                        False,
                    )
                    progress.update(1)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 1024**3
    _write_prediction_metadata(output_path, config, adapter, input_path, len(rows), completed, batch_size, max_new_tokens, sample_seed, elapsed, len(pending_rows) / elapsed, peak)
    _write_prediction_progress(output_path, config, adapter, input_path, len(rows), completed, batch_size, max_new_tokens, sample_seed, elapsed, True)
    print(f"prediction_seconds={elapsed:.3f} rows_per_second={len(pending_rows) / elapsed:.4f} peak_vram={peak:.2f}GB")


def _prepare_prediction_output(output: Path, rows: list[dict[str, Any]], resume: bool) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        output.write_text("", encoding="utf-8")
        return 0
    if not output.exists():
        raise FileNotFoundError(f"Cannot resume: prediction file not found: {output}")
    expected_ids = [str(row["task_id"]) for row in rows]
    completed = 0
    with output.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Cannot resume: invalid JSON at {output}:{line_number}") from error
            if completed >= len(expected_ids):
                raise ValueError(f"Cannot resume: prediction file has more rows than input: {output}")
            if not isinstance(record, dict) or record.get("task_id") != expected_ids[completed]:
                raise ValueError(f"Cannot resume: task_id mismatch at {output}:{line_number}")
            completed += 1
    return completed


def _append_prediction(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _write_prediction_progress(
    output: Path,
    config: QwenExperiment,
    adapter: str | Path,
    input_path: str | Path,
    rows: int,
    completed: int,
    batch_size: int,
    max_new_tokens: int,
    sample_seed: int | None,
    elapsed: float | None,
    complete: bool,
) -> None:
    payload = {
        **_prediction_provenance(config, adapter, input_path, rows, batch_size, max_new_tokens, sample_seed),
        "rows_completed": completed,
        "complete": complete,
        "elapsed_seconds": round(elapsed, 3) if elapsed is not None else None,
    }
    _write_json_atomic(output.with_suffix(".progress.json"), payload)


def _write_prediction_metadata(
    output: Path,
    config: QwenExperiment,
    adapter: str | Path,
    input_path: str | Path,
    rows: int,
    completed: int,
    batch_size: int,
    max_new_tokens: int,
    sample_seed: int | None,
    elapsed: float,
    rows_per_second: float,
    peak: float,
) -> None:
    payload = {
        **_prediction_provenance(config, adapter, input_path, rows, batch_size, max_new_tokens, sample_seed),
        "rows_completed": completed,
        "complete": completed == rows,
        "prediction_seconds": round(elapsed, 3),
        "rows_per_second": round(rows_per_second, 4),
        "peak_vram_gb": round(peak, 4),
    }
    _write_json_atomic(output.with_suffix(".metadata.json"), payload)


def _prediction_provenance(
    config: QwenExperiment,
    adapter: str | Path,
    input_path: str | Path,
    rows: int,
    batch_size: int,
    max_new_tokens: int,
    sample_seed: int | None,
) -> dict[str, Any]:
    return {
        "model_id": config.model_id,
        "adapter": str(Path(adapter).resolve()),
        "input": str(Path(input_path).resolve()),
        "output_schema": config.output_schema,
        "rows": rows,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "sample_seed": sample_seed,
        "inference_seed": config.runtime.seed,
        "do_sample": False,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_rows(path: Path, task: str, limit_examples: int | None, sample_seed: int | None = None) -> list[dict[str, Any]]:
    if limit_examples is not None and limit_examples < 1:
        raise ValueError("limit_examples must be positive")
    rows = read_jsonl(path)
    result = [{**row, "_root": str(path.parent)} for row in rows if task == "multitask" or row.get("task") == task]
    if sample_seed is not None:
        random.Random(sample_seed).shuffle(result)
    if limit_examples is not None:
        result = result[:limit_examples]
    if not result:
        raise ValueError(f"No {task} rows found in {path}")
    return result


def _scheduled_rows(rows: list[dict[str, Any]], count: int, mode: str, seed: int) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    if mode in {"uniform", "epoch_uniform"}:
        return _cycled_shuffle(rows, count, random.Random(seed))
    if count % 4:
        raise ValueError("balanced_family_class requires a schedule length divisible by 4")
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (_task_family(row), str(row["answer_label"]))
        strata.setdefault(key, []).append(row)
    keys = (("anatomy", "normal"), ("anatomy", "abnormal"), ("lesion", "normal"), ("lesion", "abnormal"))
    if set(strata) != set(keys):
        raise ValueError("balanced_family_class requires anatomy/lesion and normal/abnormal rows")
    rng = random.Random(seed)
    pools = {key: _cycled_shuffle(strata[key], count, rng) for key in keys}
    positions = {key: 0 for key in keys}
    scheduled = []
    while len(scheduled) < count:
        for key in keys:
            if len(scheduled) == count:
                break
            scheduled.append(pools[key][positions[key]])
            positions[key] += 1
    return scheduled


def _cycled_shuffle(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot sample empty rows")
    result = []
    while len(result) < count:
        cycle = list(rows)
        rng.shuffle(cycle)
        result.extend(cycle)
    return result[:count]


def _task_family(row: dict[str, Any]) -> str:
    return "lesion" if str(row["task_id"]).startswith("bs80k:") else "anatomy"


def _prediction_parser(config: QwenExperiment):
    if config.output_schema == "multitask_json":
        return parse_qwen_multitask_json_prediction
    return parse_qwen_json_prediction if config.output_schema == "json" else parse_qwen_prediction


def _validate_images(rows: list[dict[str, Any]], root: Path) -> None:
    for row in rows:
        images = row.get("images")
        views = [item.get("view") for item in images] if isinstance(images, list) else []
        if views not in (["ANT"], ["POST"], ["ANT", "POST"]):
            raise ValueError("Qwen rows must contain one ANT/POST image or ordered ANT and POST images")
        for image in images:
            if not (root / image["image"]).resolve().exists():
                raise FileNotFoundError(f"Image not found: {image['image']}")


def _load_processor(config: QwenExperiment):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(config.model_id, local_files_only=True)
    image_processor = processor.image_processor
    if hasattr(image_processor, "min_pixels"):
        # Qwen-VL dynamic-resolution processor: budgets pixels directly.
        # Fixed-resolution processors (e.g. Gemma3ImageProcessor) are left at
        # their native size: their vision tower has a fixed, learned position
        # embedding table (no interpolate_pos_encoding support in this
        # transformers version), so shrinking the input crashes the model
        # rather than reducing cost. min_pixels/max_pixels has no effect there.
        image_processor.min_pixels = config.runtime.min_pixels
        image_processor.max_pixels = config.runtime.max_pixels
    return processor


def _load_train_model(config: QwenExperiment, adapter_checkpoint: Path | None = None):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

    model = _load_base_model(config)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.runtime.gradient_checkpointing)
    if config.runtime.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if adapter_checkpoint is not None:
        model = PeftModel.from_pretrained(model, adapter_checkpoint / "adapter", is_trainable=True)
    else:
        adapter = LoraConfig(
            r=config.adapter.rank,
            lora_alpha=config.adapter.alpha,
            lora_dropout=config.adapter.dropout,
            target_modules=list(config.adapter.target_modules),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, adapter)
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Qwen3-VL LoRA configuration produced no trainable parameters")
    model.print_trainable_parameters()
    return model


def _load_inference_model(config: QwenExperiment, adapter: str | Path):
    from peft import PeftModel

    model = _load_base_model(config)
    model.config.use_cache = True
    return PeftModel.from_pretrained(model, adapter)


def _load_base_model(config: QwenExperiment):
    from transformers import BitsAndBytesConfig

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    )


def _optimizer_and_scheduler(config: QwenExperiment, model, total_steps: int):
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.runtime.learning_rate,
        weight_decay=config.runtime.weight_decay,
        betas=(config.runtime.adam_beta1, config.runtime.adam_beta2),
        eps=config.runtime.adam_epsilon,
    )
    warmup_steps = round(total_steps * config.runtime.warmup_ratio)

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _format_seconds(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _encode_training_example(processor, row: dict[str, Any], config: QwenExperiment) -> dict[str, torch.Tensor]:
    target = _truncate_target(processor, row["target"], config.runtime.max_target_tokens)
    prompt_messages = _messages(row, config.output_schema)
    full_messages = [*prompt_messages, {"role": "assistant", "content": [{"type": "text", "text": target}]}]
    prompt = processor.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    full = processor.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt")
    prefix_length = _shared_prefix_length(prompt["input_ids"][0], full["input_ids"][0])
    if prefix_length == 0 or prefix_length >= full["input_ids"].shape[1]:
        raise RuntimeError("Could not isolate the assistant target tokens")
    labels = full["input_ids"].clone()
    labels[:, :prefix_length] = -100
    labels[full["attention_mask"] == 0] = -100
    full["labels"] = labels
    return full


def _encode_prompt(processor, row: dict[str, Any], config: QwenExperiment) -> dict[str, torch.Tensor]:
    return _encode_prompt_batch(processor, [row], config)


def _encode_prompt_batch(processor, rows: list[dict[str, Any]], config: QwenExperiment) -> dict[str, torch.Tensor]:
    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        return processor.apply_chat_template(
            [_messages(row, config.output_schema) for row in rows],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
    finally:
        processor.tokenizer.padding_side = original_padding_side


def _messages(row: dict[str, Any], output_schema: str = "tags") -> list[dict[str, Any]]:
    root = Path(row["_root"])
    images = row["images"]
    content = [{"type": "image", "image": str((root / image["image"]).resolve())} for image in images]
    image_context = "ANT is the first image and PST is the second image." if len(images) == 2 else f"The provided image is {images[0]['view']}."
    task_context = f" Task type: {row['task']}." if output_schema == "multitask_json" else ""
    content.append({"type": "text", "text": f"{image_context}{task_context} {row['prompt']}"})
    return [
        {"role": "system", "content": [{"type": "text", "text": _system_prompt(output_schema)}]},
        {"role": "user", "content": content},
    ]


def _system_prompt(output_schema: str) -> str:
    if output_schema == "tags":
        return TAG_SYSTEM_PROMPT
    if output_schema == "json":
        return JSON_SYSTEM_PROMPT
    if output_schema == "multitask_json":
        return MULTITASK_JSON_SYSTEM_PROMPT
    raise ValueError("output_schema must be tags, json, or multitask_json")


def _shared_prefix_length(left: torch.Tensor, right: torch.Tensor) -> int:
    length = min(left.shape[0], right.shape[0])
    for index in range(length):
        if left[index] != right[index]:
            return index
    return length


def _truncate_target(processor, target: str, max_tokens: int) -> str:
    token_ids = processor.tokenizer(target, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= max_tokens:
        return target
    raise ValueError(
        f"Structured target has {len(token_ids)} tokens, exceeding max_target_tokens={max_tokens}; "
        "increase the configured limit rather than training on truncated JSON."
    )


def _to_cuda(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.to("cuda") if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _evaluate(model, processor, rows: list[dict[str, Any]], config: QwenExperiment) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for row in rows[: config.runtime.eval_batches]:
            batch = _to_cuda(_encode_training_example(processor, row, config))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses.append(model(**batch, use_cache=False).loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def _save(
    model,
    processor,
    config: QwenExperiment,
    evaluation: float | None,
    training_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    validation_history: list[dict[str, float | int]],
) -> None:
    output = config.runtime.output_dir
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    state = {
        "model_id": config.model_id,
        "task": config.task,
        "output_schema": config.output_schema,
        "sampling_mode": config.sampling.mode,
        "sampling_counts": _sampling_counts(training_rows),
        "max_steps": config.runtime.max_steps,
        "evaluation_loss": evaluation,
        "validation_history": validation_history,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
    }
    (output / "training_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    (output / "training_schedule.json").write_text(
        json.dumps(
            {
                "train_task_ids": [row["task_id"] for row in training_rows],
                "evaluation_task_ids": [row["task_id"] for row in evaluation_rows],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved_adapter={output}")


def _average_gradients(model, count: int) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(count)


def _task_id_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["task_id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _save_epoch_checkpoint(model, processor, config: QwenExperiment, epoch: int, optimizer_steps: int, mean_loss: float, rows: list[dict[str, Any]]) -> Path:
    output = config.runtime.output_dir / "checkpoints" / f"epoch-{epoch:02d}"
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    (output / "checkpoint_state.json").write_text(
        json.dumps(
            {
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "rows": len(rows),
                "mean_training_loss": round(mean_loss, 6),
                "order_sha256": _task_id_hash(rows),
                "selection_metric": config.selection_metric,
                "output_schema": config.output_schema,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved_epoch_checkpoint={output}")
    return output


def _copy_adapter(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite adapter directory: {target}")
    shutil.copytree(source, target)


def _write_epoch_progress(config: QwenExperiment, epochs: list[dict[str, Any]]) -> None:
    output = config.runtime.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_progress.json").write_text(json.dumps({"epochs": epochs}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_epoch_state(config: QwenExperiment, source_rows: list[dict[str, Any]], epochs: list[dict[str, Any]]) -> None:
    output = config.runtime.output_dir
    state = {
        "model_id": config.model_id,
        "task": config.task,
        "output_schema": config.output_schema,
        "sampling_mode": config.sampling.mode,
        "selection_metric": config.selection_metric,
        "epochs": epochs,
        "training_rows": len(source_rows),
        "training_task_counts": _task_counts(source_rows),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": config.runtime.learning_rate,
            "weight_decay": config.runtime.weight_decay,
            "betas": [config.runtime.adam_beta1, config.runtime.adam_beta2],
            "epsilon": config.runtime.adam_epsilon,
            "max_grad_norm": config.runtime.max_grad_norm,
            "schedule": "linear_warmup_cosine_decay",
            "warmup_ratio": config.runtime.warmup_ratio,
        },
        "train_jsonl": str(config.train_jsonl),
        "eval_jsonl": str(config.eval_jsonl) if config.eval_jsonl else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
    }
    (output / "training_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "training_schedule.json").write_text(
        json.dumps({"epoch_seeds": [config.runtime.seed + index for index in range(len(epochs))], "rows_per_epoch": len(source_rows)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _save_checkpoint(model, processor, config: QwenExperiment, step: int, evaluation: float | None) -> None:
    output = config.runtime.output_dir / "checkpoints" / f"step-{step:04d}"
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    (output / "checkpoint_state.json").write_text(
        json.dumps({"step": step, "evaluation_loss": evaluation, "output_schema": config.output_schema}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved_checkpoint={output}")


def _initialize_new_run(config: QwenExperiment, source_rows: list[dict[str, Any]]) -> None:
    output = config.runtime.output_dir
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output}")
    output.mkdir(parents=True)
    manifest_paths = {"train_jsonl": config.train_jsonl, "eval_jsonl": config.eval_jsonl}
    manifest_hashes = {name: _sha256(path) if path is not None else None for name, path in manifest_paths.items()}
    export_sums = config.train_jsonl.parent / "SHA256SUMS.txt"
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": str(config.config_path),
        "config_sha256": _sha256(config.config_path),
        "model_id": config.model_id,
        "task": config.task,
        "output_schema": config.output_schema,
        "training_rows": len(source_rows),
        "input_manifests": {name: {"path": str(path) if path is not None else None, "sha256": manifest_hashes[name]} for name, path in manifest_paths.items()},
        "export_sha256sums": {"path": str(export_sums), "sha256": _sha256(export_sums)} if export_sums.is_file() else None,
        "runtime": json.loads(config.config_path.read_text(encoding="utf-8"))["runtime"],
        "adapter": json.loads(config.config_path.read_text(encoding="utf-8"))["adapter"],
        "evaluation_protocol": {
            "checkpoint_selection": "complete validation split only",
            "selection_metric": config.selection_metric,
            "report": "evaluate and audit latest and validation-selected best adapters on complete validation and test splits",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 4),
        },
    }
    _write_json_atomic(output / "run_manifest.json", payload)


def _run_identity(config: QwenExperiment) -> dict[str, str | None]:
    return {
        "config_sha256": _sha256(config.config_path),
        "train_jsonl_sha256": _sha256(config.train_jsonl),
        "eval_jsonl_sha256": _sha256(config.eval_jsonl) if config.eval_jsonl else None,
    }


def _resume_pointer_path(output: Path) -> Path:
    return output / "resume" / "latest.json"


def _load_resume_checkpoint(config: QwenExperiment, source_rows: list[dict[str, Any]]) -> tuple[Path, dict[str, Any]]:
    output = config.runtime.output_dir
    if not output.is_dir():
        raise FileNotFoundError(f"Cannot resume because the run directory is missing: {output}")
    pointer_path = _resume_pointer_path(output)
    if not pointer_path.is_file():
        raise FileNotFoundError(f"Cannot resume because no durable checkpoint pointer exists: {pointer_path}")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("complete"):
        raise RuntimeError("This run already completed; resume is not valid")
    slot = pointer.get("slot")
    if slot not in {"slot-a", "slot-b"}:
        raise ValueError(f"Invalid durable checkpoint slot: {slot}")
    checkpoint = (output / "resume" / slot).resolve()
    if checkpoint.parent != (output / "resume").resolve() or not checkpoint.is_dir():
        raise FileNotFoundError(f"Durable checkpoint is missing: {checkpoint}")
    state_path = checkpoint / "resume_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Durable checkpoint state is missing: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("identity") != _run_identity(config):
        raise ValueError("Resume checkpoint does not match the selected configuration and manifests")
    if state.get("training_rows") != len(source_rows):
        raise ValueError("Resume checkpoint training-row count does not match the current export")
    return checkpoint, state


def _save_resume_checkpoint(
    config: QwenExperiment,
    model,
    optimizer,
    scheduler,
    next_epoch: int,
    rows_completed_in_epoch: int,
    epoch_loss_sum: float,
    optimizer_steps: int,
    source_rows: list[dict[str, Any]],
) -> None:
    root = config.runtime.output_dir / "resume"
    root.mkdir(parents=True, exist_ok=True)
    pointer_path = _resume_pointer_path(config.runtime.output_dir)
    previous = json.loads(pointer_path.read_text(encoding="utf-8")) if pointer_path.is_file() else {}
    target_name = "slot-b" if previous.get("slot") == "slot-a" else "slot-a"
    target = root / target_name
    temporary = root / f".{target_name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    adapter = temporary / "adapter"
    model.save_pretrained(adapter)
    torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
    torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
    torch.save(_capture_rng(), temporary / "rng.pt")
    state = {
        "identity": _run_identity(config),
        "next_epoch": next_epoch,
        "rows_completed_in_epoch": rows_completed_in_epoch,
        "epoch_loss_sum": epoch_loss_sum,
        "optimizer_steps": optimizer_steps,
        "training_rows": len(source_rows),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (temporary / "resume_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if target.exists():
        shutil.rmtree(target)
    os.replace(temporary, target)
    _write_json_atomic(pointer_path, {"complete": False, "slot": target_name, "optimizer_steps": optimizer_steps, "saved_at": state["saved_at"]})
    print(f"saved_resume_checkpoint={target} next_epoch={next_epoch} rows_completed_in_epoch={rows_completed_in_epoch}")


def _mark_resume_complete(config: QwenExperiment, optimizer_steps: int) -> None:
    _write_json_atomic(
        _resume_pointer_path(config.runtime.output_dir),
        {"complete": True, "optimizer_steps": optimizer_steps, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    )


def _capture_rng() -> dict[str, Any]:
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def _load_epoch_records(output: Path) -> list[dict[str, Any]]:
    path = output / "training_progress.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("epochs", [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid completed-epoch record: {path}")
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_progress(
    config: QwenExperiment,
    step: int,
    validation_history: list[dict[str, float | int]],
    details: dict[str, Any] | None = None,
) -> None:
    output = config.runtime.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_progress.json").write_text(
        json.dumps({"last_step": step, "validation_history": validation_history, **(details or {})}, indent=2) + "\n",
        encoding="utf-8",
    )


def _sampling_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        family, label = _task_family(row), row["answer_label"]
        key = f"{family}_{label}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _task_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        task = str(row.get("task"))
        counts[task] = counts.get(task, 0) + 1
    return dict(sorted(counts.items()))


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-VL training requires CUDA-enabled PyTorch and an NVIDIA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The Qwen3-VL RTX 4050 profile requires bfloat16 support")


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
