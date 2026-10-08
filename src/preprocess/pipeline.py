from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .canonical import canonicalize, read_jsonl, write_jsonl
from .tasks import build_task_manifests


def build_dataset(source: str | Path, output: str | Path, seed: int = 4050) -> dict[str, Any]:
    canonical = canonicalize(read_jsonl(source), seed=seed)
    manifests = build_task_manifests(canonical)
    target = Path(output)
    write_jsonl(target / "canonical.jsonl", canonical)
    for name, rows in manifests.items():
        write_jsonl(target / f"{name}.jsonl", rows)
    summary = {
        "source": str(Path(source)),
        "records": len(canonical),
        "patients": len({record["patient_id"] for record in canonical}),
        "splits": dict(Counter(record["split"] for record in canonical)),
        "tasks": {name: len(rows) for name, rows in manifests.items()},
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical, VQA, and grounding manifests.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4050)
    args = parser.parse_args()
    summary = build_dataset(args.source, args.output, seed=args.seed)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

