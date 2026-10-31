from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, PaliGemmaForConditionalGeneration

from preprocess.canonical import read_jsonl

from .config import Experiment, load_experiment
from .data import load_examples, open_images


def predict(config: Experiment, adapter: str | Path, input_path: str | Path, output: str | Path, max_new_tokens: int = 128, batch_size: int = 1, resume: bool = False) -> None:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = read_jsonl(input_path)
    examples = load_examples(Path(input_path), config.task)
    if len(rows) != len(examples):
        raise ValueError("Prediction rows and examples differ")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_rows(output_path, rows) if resume else 0
    if completed == len(rows):
        print(f"prediction already complete rows={completed}")
        return
    processor = _load_processor(config.model_id)
    model = _load_adapter(config, adapter)
    model.eval()
    with output_path.open("a" if resume else "w", encoding="utf-8", newline="\n") as handle:
        pairs = list(zip(rows, examples, strict=True))[completed:]
        with tqdm(total=len(rows), initial=completed, desc="Inference", unit="row") as progress:
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                predictions = _generate_batch(model, processor, [example for _, example in batch], max_new_tokens)
                for (row, _), prediction in zip(batch, predictions, strict=True):
                    handle.write(json.dumps({"task_id": row["task_id"], "prediction": prediction}) + "\n")
                handle.flush()
                progress.update(len(batch))
    print(f"peak_vram={torch.cuda.max_memory_allocated() / 1024**3:.2f}GB", flush=True)


def _completed_rows(output_path: Path, rows: list[dict]) -> int:
    """Count the leading rows already written, checking each task id against the input order."""
    if not output_path.exists():
        return 0
    expected_ids = [str(row["task_id"]) for row in rows]
    completed = 0
    with output_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Cannot resume: invalid JSON at {output_path}:{line_number}") from error
            if completed >= len(expected_ids) or record.get("task_id") != expected_ids[completed]:
                raise ValueError(f"Cannot resume: task_id mismatch at {output_path}:{line_number}")
            completed += 1
    return completed


def _generate_batch(model, processor, examples: list, max_new_tokens: int) -> list[str]:
    # Left-padding is required so every sequence's next-token position lines
    # up at the same index for batched causal-LM generation (same reasoning
    # as the Qwen3-VL trainer's _encode_prompt_batch).
    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        images = open_images(examples)
        inputs = processor(text=[example.prompt for example in examples], images=images, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        suffix = generated[:, inputs["input_ids"].shape[1] :]
        return [prediction.strip() for prediction in processor.batch_decode(suffix, skip_special_tokens=True)]
    finally:
        processor.tokenizer.padding_side = original_padding_side


def _load_processor(model_id: str):
    try:
        return AutoProcessor.from_pretrained(model_id)
    except OSError as error:
        if "gated repo" in str(error).lower() or "restricted" in str(error).lower():
            raise RuntimeError(f"Model access is required for {model_id}. Accept its Hugging Face terms and authenticate before inference.") from error
        raise


def _load_adapter(config: Experiment, adapter: str | Path):
    if not torch.cuda.is_available():
        raise RuntimeError("Inference requires a CUDA GPU")
    load_args: dict[str, object] = {"device_map": {"": 0}}
    if config.quantization == "4bit":
        load_args["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    else:
        load_args["torch_dtype"] = torch.float16
    try:
        model = PaliGemmaForConditionalGeneration.from_pretrained(config.model_id, **load_args)
    except OSError as error:
        if "gated repo" in str(error).lower() or "restricted" in str(error).lower():
            raise RuntimeError(f"Model access is required for {config.model_id}. Accept its Hugging Face terms and authenticate before inference.") from error
        raise
    return PeftModel.from_pretrained(model, adapter)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic predictions from a trained PaliGemma adapter.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run from the rows already written to --output.")
    args = parser.parse_args()
    predict(load_experiment(args.config), args.adapter, args.input, args.output, args.max_new_tokens, args.batch_size, args.resume)


if __name__ == "__main__":
    main()
