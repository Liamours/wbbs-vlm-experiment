from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .canonical import read_jsonl, write_jsonl


def build_r01_review_package(
    paired_evidence: Iterable[dict[str, Any]],
    normal_controls_per_region: int = 20,
    seed: int = 4050,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if normal_controls_per_region < 0:
        raise ValueError("normal_controls_per_region must be non-negative")
    test_pairs = [pair for pair in paired_evidence if pair["split"] == "test"]
    selected: dict[str, set[str]] = defaultdict(set)
    for pair in test_pairs:
        if pair["pair_diagnosis"] == "abnormal":
            selected[pair["evidence_id"]].add("abnormal_test_pair")
    normals_by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in test_pairs:
        if pair["pair_diagnosis"] == "normal":
            normals_by_region[pair["region"]].append(pair)
    for region, rows in normals_by_region.items():
        for pair in sorted(rows, key=lambda row: _stable_key(seed, row["evidence_id"]))[:normal_controls_per_region]:
            selected[pair["evidence_id"]].add("normal_control")

    chosen = [pair for pair in test_pairs if pair["evidence_id"] in selected]
    queue = [_blinded_case(pair) for pair in sorted(chosen, key=lambda row: row["evidence_id"])]
    provenance = [_provenance_case(pair, selected[pair["evidence_id"]]) for pair in sorted(chosen, key=lambda row: row["evidence_id"])]
    summary = {
        "package": "r0.1-review-candidate",
        "source_split": "test",
        "seed": seed,
        "normal_controls_per_region": normal_controls_per_region,
        "test_pairs_available": len(test_pairs),
        "review_cases": len(queue),
        "selection_reasons": dict(sorted(Counter(reason for reasons in selected.values() for reason in reasons).items())),
        "cases_by_region": dict(sorted(Counter(case["region"] for case in queue).items())),
        "view_diagnosis_disagreements": sum(len(set(pair["diagnosis_by_view"].values())) > 1 for pair in chosen),
    }
    return queue, provenance, summary


def write_r01_review_package(
    paired_evidence_path: str | Path,
    output: str | Path,
    normal_controls_per_region: int = 20,
    seed: int = 4050,
) -> dict[str, Any]:
    queue, provenance, summary = build_r01_review_package(
        read_jsonl(paired_evidence_path),
        normal_controls_per_region=normal_controls_per_region,
        seed=seed,
    )
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    write_jsonl(target / "review_queue_blinded.jsonl", queue)
    write_jsonl(target / "review_provenance.jsonl", provenance)
    (target / "review_response_schema.json").write_text(
        json.dumps(_response_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (target / "review_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _blinded_case(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_id": f"r01:{pair['evidence_id']}",
        "evidence_id": pair["evidence_id"],
        "region": pair["region"],
        "images": [{key: image[key] for key in ("view", "image", "image_size")} for image in pair["images"]],
        "target_bboxes": [{"view": image["view"], "bbox": image["bbox"]} for image in pair["images"]],
        "review_tasks": ["verify_region_bbox", "classify_uptake_by_view", "mark_exclusion_if_not_assessable"],
    }


def _provenance_case(pair: dict[str, Any], reasons: set[str]) -> dict[str, Any]:
    return {
        "review_id": f"r01:{pair['evidence_id']}",
        "evidence_id": pair["evidence_id"],
        "source_diagnosis_by_view": pair["diagnosis_by_view"],
        "source_labels": pair["source_labels"],
        "selection_reasons": sorted(reasons),
    }


def _response_schema() -> dict[str, Any]:
    return {
        "package": "r0.1-review-candidate",
        "instructions": "One reviewer response per blinded review_id. Keep source labels hidden until independent reviews are complete.",
        "required_fields": {
            "review_id": "string",
            "reviewer_id": "string",
            "bbox_status_by_view": {"ANT": "accept|correct|not_assessable", "POST": "accept|correct|not_assessable"},
            "corrected_bbox_by_view": {"ANT": "[x1,y1,x2,y2]|null", "POST": "[x1,y1,x2,y2]|null"},
            "diagnosis_by_view": {"ANT": "normal|abnormal|indeterminate", "POST": "normal|abnormal|indeterminate"},
            "overall_decision": "accept|correct|exclude|indeterminate",
            "notes": "string",
        },
    }


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a blinded R0.1 clinician-review package from the locked test split.")
    parser.add_argument("--paired-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normal-controls-per-region", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4050)
    args = parser.parse_args()
    print(
        json.dumps(
            write_r01_review_package(
                args.paired_evidence,
                args.output,
                normal_controls_per_region=args.normal_controls_per_region,
                seed=args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
