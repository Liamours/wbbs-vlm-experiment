from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from postprocess.paligemma import export_paligemma_with_rejections
from preprocess.canonical import read_jsonl, write_jsonl


EXPORTS = {
    "anatomy_vqa": ("vqa.jsonl", "vqa"),
    "anatomy_grounding": ("grounding.jsonl", "grounding"),
    "anatomy_grounded_vqa": ("grounded_vqa.jsonl", "grounded_vqa"),
    "lesion_vqa": ("lesion_vqa.jsonl", "vqa"),
    "lesion_grounding": ("lesion_grounding.jsonl", "grounding"),
    "lesion_grounded_vqa": ("lesion_grounded_vqa.jsonl", "grounded_vqa"),
}


def _write_sha256sums(output: Path, paths: list[Path]) -> None:
    entries = []
    for path in sorted(paths, key=lambda item: item.name):
        entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (output / "SHA256SUMS.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")


def build_baseline_exports(release: str | Path, output: str | Path) -> dict[str, Any]:
    source, target = Path(release), Path(output)
    target.mkdir(parents=True, exist_ok=True)
    if (source / "multitask.jsonl").exists():
        return _build_r3_multitask_baseline_exports(source, target)
    if (source / "vgrounding.jsonl").exists():
        return _build_r2_baseline_exports(source, target)
    summary: dict[str, Any] = {"release": str(source), "exports": {}}
    for name, (filename, task) in EXPORTS.items():
        rows = read_jsonl(source / filename)
        counts = {}
        for split in ("train", "val", "test"):
            exported, rejected = export_paligemma_with_rejections(rows, task, split=split)
            write_jsonl(target / f"{name}_{split}.jsonl", exported)
            if rejected:
                write_jsonl(target / f"{name}_{split}_rejected.jsonl", rejected)
            else:
                (target / f"{name}_{split}_rejected.jsonl").unlink(missing_ok=True)
            counts[split] = {"rows": len(exported), "rejected": len(rejected)}
        summary["exports"][name] = {"source": filename, "task": task, "splits": counts}
    (target / "baseline_export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_sha256sums(target, [*target.glob("*.jsonl"), target / "baseline_export_summary.json"])
    return summary


def _build_r2_baseline_exports(source: Path, target: Path) -> dict[str, Any]:
    """Export the merged R2 files without restoring anatomy/lesion splits."""

    vqa_rows = read_jsonl(source / "vqa.jsonl")
    merged_grounding = read_jsonl(source / "vgrounding.jsonl")
    groups = {
        "vqa": (vqa_rows, "vqa"),
        "vgrounding_localization": ([row for row in merged_grounding if row.get("interaction_type") == "localization"], "grounding"),
        "vgrounding_grounded_vqa": (
            [row for row in merged_grounding if row.get("interaction_type") == "localization_and_classification"],
            "grounded_vqa",
        ),
    }
    return _build_merged_baseline_exports(source, target, groups)


def _build_r3_multitask_baseline_exports(source: Path, target: Path) -> dict[str, Any]:
    rows = read_jsonl(source / "multitask.jsonl")
    groups = {
        "vqa": ([row for row in rows if row.get("task_type") == "vqa"], "vqa"),
        "vgrounding_localization": ([row for row in rows if row.get("task_type") == "grounding"], "grounding"),
        "vgrounding_grounded_vqa": ([row for row in rows if row.get("task_type") == "grounded_vqa"], "grounded_vqa"),
    }
    return _build_merged_baseline_exports(source, target, groups)


def _build_merged_baseline_exports(source: Path, target: Path, groups: dict[str, tuple[list[dict[str, Any]], str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"release": str(source), "format": "wbbs-merged-v1", "exports": {}}
    for name, (rows, task) in groups.items():
        counts = {}
        for split in ("train", "val", "test"):
            exported, rejected = export_paligemma_with_rejections(rows, task, split=split)
            write_jsonl(target / f"{name}_{split}.jsonl", exported)
            rejected_path = target / f"{name}_{split}_rejected.jsonl"
            if rejected:
                write_jsonl(rejected_path, rejected)
            else:
                rejected_path.unlink(missing_ok=True)
            counts[split] = {"rows": len(exported), "rejected": len(rejected)}
        summary["exports"][name] = {"task": task, "rows": len(rows), "splits": counts}
    (target / "baseline_export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_sha256sums(target, [*target.glob("*.jsonl"), target / "baseline_export_summary.json"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create split-safe baseline exports from an R1 or merged R2 release.")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_baseline_exports(args.release, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
