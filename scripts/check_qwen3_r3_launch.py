"""Validate a local R3 Qwen training launch without starting training."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

from training.qwen_config import load_qwen_experiment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=("4050", "5070"), required=True)
    parser.add_argument("--skip-gpu", action="store_true", help="Use only for a non-training preflight on another machine.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_qwen_experiment(args.config)
    output = config.runtime.output_dir
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output}")
    if args.resume and not output.is_dir():
        raise FileNotFoundError(f"Cannot resume because the run directory is missing: {output}")
    disk_check_path = output.parent
    while not disk_check_path.exists():
        disk_check_path = disk_check_path.parent
    free_bytes = shutil.disk_usage(disk_check_path).free
    required_bytes = int(config.runtime.minimum_free_disk_gb * 1024**3)
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Insufficient free disk space at {output.parent}: {free_bytes / 1024**3:.2f} GiB available, "
            f"{config.runtime.minimum_free_disk_gb:.2f} GiB required."
        )
    for manifest in (config.train_jsonl, config.eval_jsonl):
        if manifest is None or not manifest.is_file():
            raise FileNotFoundError(f"Missing configured manifest: {manifest}")

    from huggingface_hub import snapshot_download

    model_cache = Path(snapshot_download(config.model_id, local_files_only=True))
    if not any(model_cache.glob("*.safetensors")):
        raise FileNotFoundError(f"Model cache has no safetensor weights: {model_cache}")
    report = {
        "config": str(config.config_path),
        "config_sha256": _sha256(config.config_path),
        "profile": args.profile,
        "model_id": config.model_id,
        "model_cache": str(model_cache),
        "run_directory": str(output),
        "resume": args.resume,
        "minimum_free_disk_gb": config.runtime.minimum_free_disk_gb,
        "available_disk_gb": round(free_bytes / 1024**3, 3),
        "train_jsonl_sha256": _sha256(config.train_jsonl),
        "eval_jsonl_sha256": _sha256(config.eval_jsonl) if config.eval_jsonl else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if args.skip_gpu:
        report["gpu_check"] = "skipped"
    else:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        name = torch.cuda.get_device_name(0)
        if f"RTX {args.profile}" not in name.upper():
            raise RuntimeError(f"Selected profile RTX {args.profile} does not match CUDA device: {name}")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"CUDA device does not support bfloat16: {name}")
        properties = torch.cuda.get_device_properties(0)
        report["gpu"] = {
            "name": name,
            "total_memory_gb": round(properties.total_memory / 1024**3, 3),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "bfloat16": True,
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
