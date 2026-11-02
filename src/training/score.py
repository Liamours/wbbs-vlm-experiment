"""Score every candidate answer of VQA and grounded VQA instances with a trained adapter, for the score-based AUC.

For each instance the adapter's log-probability is computed for the output prefix of every answer sentence of the training
bank, under both class values (analysis.candidate_bank). The prompt and image are encoded once per instance and all
candidates run in one batch; only the last positions are projected to the vocabulary (logits_to_keep). Records are appended
per instance, so a rerun continues where it stopped. Run the validation split with --limit first: it reports the speed, the
projected time of the full split, and how often the score agrees with the stored greedy output of the same adapter.

    uv run --no-sync python -m training.score --config configs/score_auc_261102.json --model qwen3vl --split validation --limit 64
    uv run --no-sync python -m training.score --config configs/score_auc_261102.json --model qwen3vl --split test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm import tqdm

from analysis.candidate_bank import build_bank, candidates, check_bank, class_values, coverage
from analysis.score_auc import agreement, class_score, read_reference
from preprocess.canonical import read_jsonl
from project_paths import PROJECT_ROOT


def sequence_logprobs(logits: torch.Tensor, tokens: Sequence[Sequence[int]], trailing: Sequence[int]) -> list[float]:
    """Summed log-probability of each row's tokens from logits over the last positions of a left-padded batch.

    logits is (rows, window, vocabulary); row k's tokens end `trailing[k]` positions before the end of the window, and the
    logits at window index i predict the token at index i + 1.
    """

    window = logits.shape[1]
    result = []
    for row, (ids, tail) in enumerate(zip(tokens, trailing, strict=True)):
        first = window - tail - len(ids)
        if first < 1:
            raise ValueError("The logits window is shorter than the candidate")
        steps = torch.log_softmax(logits[row, first - 1 : first - 1 + len(ids)].float(), dim=-1)
        result.append(float(steps.gather(1, torch.tensor(ids, device=steps.device).unsqueeze(1)).sum()))
    return result


def left_pad(sequences: Sequence[torch.Tensor], value: int) -> torch.Tensor:
    length = max(len(sequence) for sequence in sequences)
    return torch.stack([torch.cat([sequence.new_full((length - len(sequence),), value), sequence]) for sequence in sequences])


class QwenBackend:
    def __init__(self, entry: dict[str, Any]) -> None:
        from training.qwen import _encode_training_example, _load_inference_model, _load_processor, _require_cuda
        from training.qwen_config import load_qwen_experiment

        _require_cuda()
        self.config = load_qwen_experiment(_path(entry["config"]))
        self.processor = _load_processor(self.config)
        self.model = _load_inference_model(self.config, _path(entry["adapter"])).eval()
        self.encode = _encode_training_example
        self.tokenizer = self.processor.tokenizer

    def score(self, row: dict[str, Any], prefixes: list[str]) -> list[float]:
        """Prefill the prompt and image once, then score every candidate against copies of that cache."""

        tokens = [self.tokenizer(text, add_special_tokens=False)["input_ids"] for text in prefixes]
        full = self.encode(self.processor, {**row, "target": prefixes[0]}, self.config)
        ids = full["input_ids"][0]
        head = int((full["labels"][0] == -100).sum())
        if ids[head : head + len(tokens[0])].tolist() != tokens[0]:
            raise RuntimeError(f"Target tokens differ in context for {row['task_id']}")
        prompt: dict[str, Any] = {"input_ids": ids[None, :head], "attention_mask": torch.ones(1, head, dtype=torch.long)}
        for key, value in full.items():
            if key not in ("input_ids", "attention_mask", "labels"):
                prompt[key] = value[:, :head] if value.shape == full["input_ids"].shape else value
        prompt = {key: value.to("cuda") for key, value in prompt.items()}
        count, width = len(tokens), max(len(t) for t in tokens)
        candidate_ids = torch.full((count, width), self.tokenizer.pad_token_id, dtype=torch.long)
        for k, t in enumerate(tokens):
            candidate_ids[k, : len(t)] = torch.tensor(t)
        lengths = torch.tensor([len(t) for t in tokens])
        mask = torch.cat([torch.ones(count, head, dtype=torch.long), (torch.arange(width)[None, :] < lengths[:, None]).long()], dim=1)
        with torch.inference_mode():
            out = self.model(**prompt, use_cache=True, logits_to_keep=1)
            first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            cache = out.past_key_values
            cache.batch_repeat_interleave(count)
            delta = next(m.rope_deltas for m in self.model.modules() if getattr(m, "rope_deltas", None) is not None)
            positions = (torch.arange(head, head + width, device="cuda") + delta.reshape(-1)[0]).view(1, 1, -1).expand(3, count, -1)
            logits = self.model(input_ids=candidate_ids.to("cuda"), attention_mask=mask.to("cuda"), position_ids=positions, past_key_values=cache, cache_position=torch.arange(head, head + width, device="cuda"), use_cache=True).logits
            result = []
            for k, t in enumerate(tokens):
                total = float(first[t[0]])
                if len(t) > 1:
                    steps = torch.log_softmax(logits[k, : len(t) - 1].float(), dim=-1)
                    total += float(steps.gather(1, torch.tensor(t[1:], device="cuda").unsqueeze(1)).sum())
                result.append(total)
        return result

    def score_batch(self, row: dict[str, Any], prefixes: list[str]) -> list[float]:
        """Reference scoring without a cache: every candidate runs the whole prompt. Slower; used to check the cached path."""

        tokens = [self.tokenizer(text, add_special_tokens=False)["input_ids"] for text in prefixes]
        full = self.encode(self.processor, {**row, "target": prefixes[0]}, self.config)
        ids = full["input_ids"][0]
        head = int((full["labels"][0] == -100).sum())
        if ids[head : head + len(tokens[0])].tolist() != tokens[0]:
            raise RuntimeError(f"Target tokens differ in context for {row['task_id']}")
        pad = self.tokenizer.pad_token_id
        sequences = [torch.cat([ids[:head], torch.tensor(t)]) for t in tokens]
        batch: dict[str, Any] = {"input_ids": left_pad(sequences, pad)}
        batch["attention_mask"] = left_pad([torch.ones_like(s) for s in sequences], 0)
        for key, value in full.items():
            if key in ("input_ids", "attention_mask", "labels"):
                continue
            if value.shape == full["input_ids"].shape:
                batch[key] = left_pad([torch.cat([value[0, :head], value.new_zeros(len(t))]) for t in tokens], 0)
            else:
                batch[key] = torch.cat([value] * len(tokens), dim=0)
        batch = {key: value.to("cuda") for key, value in batch.items()}
        with torch.inference_mode():
            logits = self.model(**batch, logits_to_keep=max(len(t) for t in tokens) + 1).logits
        return sequence_logprobs(logits, tokens, [0] * len(tokens))


class PaliGemmaBackend:
    def __init__(self, entry: dict[str, Any]) -> None:
        from training.config import load_experiment
        from training.predict import _load_adapter, _load_processor

        self.config = load_experiment(_path(entry["config"]))
        self.processor = _load_processor(self.config.model_id)
        self.processor.tokenizer.padding_side = "left"
        self.model = _load_adapter(self.config, _path(entry["adapter"])).eval()
        self.tokenizer = self.processor.tokenizer

    def score(self, row: dict[str, Any], prefixes: list[str]) -> list[float]:
        """Prefill the image and prompt once (bidirectional prefix), then score every candidate against copies of that cache."""

        from training.data import _to_example, open_images
        from training.train import _cap_text

        example = _to_example(row, self.config.task, Path(row["_root"]))
        prompt = _cap_text(self.processor, example.prompt, self.config.runtime.max_prompt_tokens)
        image = open_images([example])[0]
        tokens = [self.tokenizer(text, add_special_tokens=False)["input_ids"] for text in prefixes]
        batch = self.processor(text=[prompt], images=[image], suffix=[prefixes[0]], return_tensors="pt", padding="longest", truncation=False)
        head = int((batch["token_type_ids"][0] == 0).sum())
        if batch["input_ids"][0, head : head + len(tokens[0])].tolist() != tokens[0]:
            raise RuntimeError(f"Suffix tokens differ from the candidate for {row['task_id']}")
        count, width = len(tokens), max(len(t) for t in tokens)
        candidate_ids = torch.full((count, width), self.tokenizer.pad_token_id, dtype=torch.long)
        for k, t in enumerate(tokens):
            candidate_ids[k, : len(t)] = torch.tensor(t)
        lengths = torch.tensor([len(t) for t in tokens])
        mask = torch.cat([torch.ones(count, head, dtype=torch.long), (torch.arange(width)[None, :] < lengths[:, None]).long()], dim=1)
        with torch.inference_mode():
            out = self.model(input_ids=batch["input_ids"][:, :head].to("cuda"), pixel_values=batch["pixel_values"].to("cuda"), attention_mask=torch.ones(1, head, dtype=torch.long, device="cuda"), token_type_ids=torch.zeros(1, head, dtype=torch.long, device="cuda"), use_cache=True, logits_to_keep=1)
            first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            cache = out.past_key_values
            cache.batch_repeat_interleave(count)
            logits = self.model(input_ids=candidate_ids.to("cuda"), attention_mask=mask.to("cuda"), past_key_values=cache, use_cache=True).logits
            result = []
            for k, t in enumerate(tokens):
                total = float(first[t[0]])
                if len(t) > 1:
                    steps = torch.log_softmax(logits[k, : len(t) - 1].float(), dim=-1)
                    total += float(steps.gather(1, torch.tensor(t[1:], device="cuda").unsqueeze(1)).sum())
                result.append(total)
        return result

    def score_batch(self, row: dict[str, Any], prefixes: list[str]) -> list[float]:
        """Reference scoring without a cache: every candidate runs the whole prompt and image. Slower; used to check the cached path."""

        from training.data import _to_example, open_images
        from training.train import _cap_text

        example = _to_example(row, self.config.task, Path(row["_root"]))
        prompt = _cap_text(self.processor, example.prompt, self.config.runtime.max_prompt_tokens)
        image = open_images([example])[0]
        batch = self.processor(text=[prompt] * len(prefixes), images=[image] * len(prefixes), suffix=prefixes, return_tensors="pt", padding="longest", truncation=False)
        labels = batch.pop("labels")
        tokens = [self.tokenizer(text, add_special_tokens=False)["input_ids"] for text in prefixes]
        trailing = []
        for k, ids in enumerate(tokens):
            labeled = labels[k][labels[k] != -100].tolist()
            tail = len(labeled) - len(ids)
            if tail not in (0, 1) or labeled[: len(ids)] != ids:
                raise RuntimeError(f"Suffix tokens differ from the candidate for {row['task_id']}")
            trailing.append(tail)
        batch = batch.to("cuda")
        with torch.inference_mode():
            logits = self.model(**batch, logits_to_keep=max(len(t) + e for t, e in zip(tokens, trailing)) + 1).logits
        return sequence_logprobs(logits, tokens, trailing)


BACKENDS = {"qwen3vl": QwenBackend, "paligemma": PaliGemmaBackend}


def score_split(config: dict[str, Any], model: str, split: str, limit: int | None, backend=None, dry_run: bool = False) -> Path | None:
    spec = config["splits"][split]
    manifest = _path(spec["manifest"])
    _check_checksum(manifest, _path(config["checksums"]))
    train_rows = read_jsonl(_path(config["train_manifest"]))
    bank = build_bank(train_rows)
    print(f"bank rebuilt for {check_bank(train_rows)} training rows; templates {({'/'.join(k): len(v) for k, v in sorted(bank.items())})}")
    max_templates = config["max_templates"]
    rows = [{**row, "_root": str(manifest.parent)} for row in read_jsonl(manifest) if row["task"] in config["tasks"]]
    if len(rows) != spec["scored_rows"]:
        raise ValueError(f"{manifest.name} has {len(rows)} scored rows, expected {spec['scored_rows']}")
    total = len(rows) if limit is None else min(limit, len(rows))
    rows = rows[:total]

    if dry_run:
        sizes = [len(candidates(row, bank, max_templates)) for row in rows]
        print(f"{model} {split}: {len(rows)} instances, {sum(sizes)} candidate sequences ({sum(sizes) / len(sizes):.1f} per instance, at most {max(sizes)}); coverage {coverage(bank, max_templates)}")
        return None
    out = _path(config["output_dir"]) / model / split
    out.mkdir(parents=True, exist_ok=True)
    scores_path = out / ("scores.jsonl" if limit is None else "scores.smoke.jsonl")
    done = {json.loads(line)["task_id"] for line in scores_path.read_text(encoding="utf-8").splitlines() if line.strip()} if scores_path.exists() else set()
    log = _logger(_path(config["log_dir"]) / f"score-{model}-{split}-{time.strftime('%y%m%d-%H%M%S')}.log")
    log.info("model %s split %s rows %d done %d candidates per instance up to %d coverage %s", model, split, total, len(done), 2 * max(len(v) for v in bank.values()), coverage(bank, max_templates))
    pending = [row for row in rows if row["task_id"] not in done]
    if pending:
        _require_free_resources(config)
        backend = backend or BACKENDS[config["models"][model].get("backend", model)](config["models"][model])
        torch.cuda.reset_peak_memory_stats()
        if limit is not None and hasattr(backend, "score_batch"):
            for row in pending[:2]:
                found = candidates(row, bank, max_templates)
                cached, reference = backend.score(row, [c["prefix"] for c in found]), backend.score_batch(row, [c["prefix"] for c in found])
                gap = max(abs(a - b) for a, b in zip(cached, reference))
                log.info("cached against uncached scoring, largest gap %.4f over %d candidates", gap, len(found))
                if gap > 0.5:
                    raise SystemExit(f"The cached scoring differs from the reference by {gap:.3f}")
        started = time.perf_counter()
        last_log = started
        with scores_path.open("a", encoding="utf-8", newline="\n") as handle, tqdm(total=total, initial=len(done), desc=f"{model} {split}", unit="row", dynamic_ncols=True) as progress:
            for count, row in enumerate(pending, start=1):
                found = candidates(row, bank, max_templates)
                logps: list[float] = []
                for start in range(0, len(found), config["candidate_batch"]):
                    logps += backend.score(row, [c["prefix"] for c in found[start : start + config["candidate_batch"]]])
                best = max(range(len(found)), key=lambda i: logps[i])
                record = {"task_id": row["task_id"], "task": row["task"], "region_level": row["region_level"], "patient_id": row["patient_id"], "answer_label": row["answer_label"], "score": class_score(logps, [c["class"] for c in found], class_values(row["task"])[1]), "top_answer": found[best]["answer"], "top_class": found[best]["class"], "candidates": [[c["class"], round(lp, 5)] for c, lp in zip(found, logps)]}
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                progress.update(1)
                now = time.perf_counter()
                if now - last_log >= config["log_seconds"] or count == len(pending):
                    rate = count / (now - started)
                    log.info("%d/%d rows, %.3f rows/s, ETA %.1f min, peak VRAM %.2f GB", len(done) + count, total, rate, (len(pending) - count) / rate / 60, torch.cuda.max_memory_allocated() / 1024**3)
                    last_log = now
        seconds = time.perf_counter() - started
        full_rows = spec["scored_rows"]
        log.info("finished %d rows in %.1f min; projected full %s split %.1f min", len(pending), seconds / 60, split, full_rows / (len(pending) / seconds) / 60)
    records = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reference = config["models"][model].get("reference_predictions", {}).get(split)
    if reference:
        result = agreement(records, read_reference(_path(reference)))
        log.info("agreement with the stored greedy output: %s", result)
        threshold = config["models"][model].get("min_label_agreement", config["min_label_agreement"])
        if threshold is not None and result["label_agreement"] < threshold:
            raise SystemExit(f"Label agreement {result['label_agreement']:.3f} is below {threshold}; check the prompt, tokenization and window before the full run")
    return scores_path


def _check_checksum(path: Path, sums: Path) -> None:
    listed = [line.split()[0] for line in sums.read_text(encoding="utf-8").splitlines() if line.strip().endswith(path.name)]
    provenance = path.with_name(path.stem + ".provenance.json")
    expected = listed[0] if listed else json.loads(provenance.read_text(encoding="utf-8"))["manifest_sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path}")


def _require_free_resources(config: dict[str, Any]) -> None:
    running = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, check=True).stdout.lower()
    busy = sorted({name for name in config["blocked_processes"] if f'"{name.lower()}.exe"' in running})
    if busy:
        raise SystemExit(f"Close these apps first, they compete for RAM and the GPU: {', '.join(busy)}")
    free = torch.cuda.mem_get_info()[0] / 1024**2
    if free < config["min_free_vram_mb"]:
        raise SystemExit(f"GPU has {free:.0f} MiB free, the run needs {config['min_free_vram_mb']} MiB")


def _logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("score")
    log.setLevel(logging.INFO)
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        log.addHandler(handler)
    return log


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Key of a model in the config.")
    parser.add_argument("--split", required=True, help="Key of a split in the config: validation, test or test_blind.")
    parser.add_argument("--dry-run", action="store_true", help="Check the checksum, rebuild the bank, count the candidates and stop before loading a model.")
    parser.add_argument("--limit", type=int, help="Score only the first rows into scores.smoke.jsonl, for the speed and agreement check.")
    args = parser.parse_args()
    score_split(json.loads(args.config.read_text(encoding="utf-8")), args.model, args.split, args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
