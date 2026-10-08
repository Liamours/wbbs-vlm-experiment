from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from preprocess.canonical import read_jsonl, write_jsonl


STRATA = (("anatomy", "normal"), ("anatomy", "abnormal"), ("lesion", "normal"), ("lesion", "abnormal"))


def build_balanced_probe(rows: Iterable[dict[str, Any]], per_stratum: int, seed: int) -> list[dict[str, Any]]:
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in STRATA}
    for row in rows:
        if row.get("task") != "grounded_vqa":
            continue
        key = (_family(row), str(row.get("answer_label")))
        if key in groups:
            groups[key].append(row)
    selected = []
    for index, key in enumerate(STRATA):
        group = list(groups[key])
        if len(group) < per_stratum:
            raise ValueError(f"Insufficient rows for {key}: need {per_stratum}, found {len(group)}")
        random.Random(seed + index).shuffle(group)
        selected.extend(group[:per_stratum])
    return selected


def _family(row: dict[str, Any]) -> str:
    return "lesion" if str(row.get("task_id", "")).startswith("bs80k:") else "anatomy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic balanced grounded-VQA evaluation probe.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4050)
    args = parser.parse_args()
    probe = build_balanced_probe(read_jsonl(args.input), args.per_stratum, args.seed)
    write_jsonl(args.output, probe)
    metadata = {
        "input": str(args.input),
        "seed": args.seed,
        "per_stratum": args.per_stratum,
        "rows": len(probe),
        "strata": {f"{family}_{label}": args.per_stratum for family, label in STRATA},
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
