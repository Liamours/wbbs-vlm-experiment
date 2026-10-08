from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from analysis.review import validate_adjudications
from postprocess.bbx import export_bbx_grounding

from .canonical import read_jsonl, write_jsonl
from .pairs import build_paired_task_manifests


def build_r01_gold_evidence(
    source_pairs: Iterable[dict[str, Any]],
    review_queue: Iterable[dict[str, Any]],
    adjudications: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = {row["evidence_id"]: row for row in source_pairs}
    queue = list(review_queue)
    decisions = validate_adjudications(queue, adjudications)
    accepted, excluded = [], []
    for case in queue:
        decision = decisions[case["review_id"]]
        source = pairs.get(case["evidence_id"])
        if source is None:
            raise ValueError(f"Source pair not found for {case['evidence_id']}")
        if decision["final_decision"] == "exclude":
            excluded.append({"review_id": case["review_id"], "evidence_id": case["evidence_id"], "adjudication": decision})
            continue
        by_view = {view: bbox for view, bbox in decision["final_bbox_by_view"].items()}
        diagnosis_by_view = decision["final_diagnosis_by_view"]
        accepted.append(
            {
                **source,
                "evidence_id": f"{source['evidence_id']}:r01",
                "source_evidence_id": source["evidence_id"],
                "images": [{**image, "bbox": by_view[image["view"]]} for image in source["images"]],
                "diagnosis_by_view": diagnosis_by_view,
                "pair_diagnosis": "abnormal" if "abnormal" in diagnosis_by_view.values() else "normal",
                "review": {"review_id": case["review_id"], "adjudication": decision},
            }
        )
    return accepted, excluded


def write_r01_gold_release(
    source_pairs_path: str | Path,
    review_queue_path: str | Path,
    adjudications_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    evidence, excluded = build_r01_gold_evidence(
        read_jsonl(source_pairs_path), read_jsonl(review_queue_path), read_jsonl(adjudications_path)
    )
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    write_jsonl(target / "gold_evidence.jsonl", evidence)
    write_jsonl(target / "gold_excluded.jsonl", excluded)
    manifests = build_paired_task_manifests(evidence)
    for name, rows in manifests.items():
        write_jsonl(target / f"{name}.jsonl", rows)
    bbx_rows = export_bbx_grounding(target / "grounding.jsonl", target / "grounding_bbx.jsonl")
    summary = {
        "release": "r0.1-gold",
        "reviewed_cases": len(evidence) + len(excluded),
        "accepted_cases": len(evidence),
        "excluded_cases": len(excluded),
        "tasks": {name: len(rows) for name, rows in manifests.items()},
        "bbx_grounding_rows": bbx_rows,
    }
    (target / "gold_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the R0.1 gold release from completed adjudications.")
    parser.add_argument("--source-pairs", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_r01_gold_release(args.source_pairs, args.review_queue, args.adjudications, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
