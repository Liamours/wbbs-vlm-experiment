from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from analysis.report import build_quality_report, write_quality_report
from analysis.contact_sheet import create_region_contact_sheet
from analysis.splits import build_patient_split_manifest, build_split_report
from postprocess.bbx import export_bbx_grounding

from .bs80k import build_bs80k_evidence
from .canonical import read_jsonl, write_jsonl
from .inventory import write_source_inventory
from .lesions import build_lesion_task_manifests
from .libs160k import read_caption_templates
from .pairs import build_paired_evidence, build_paired_task_manifests, paired_summary
from .pipeline import build_dataset
from .templates import apply_r0_templates
from .verify import verify_records


def build_r0_release(dataset_root: str | Path, output: str | Path, policy: dict[str, Any]) -> dict[str, Any]:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    inventory = write_source_inventory(dataset_root, target / "source_inventory.json")
    evidence, metastases = build_bs80k_evidence(dataset_root, target)
    write_jsonl(target / "metastases.jsonl", metastases)
    templated = apply_r0_templates(evidence, read_caption_templates(dataset_root))
    write_jsonl(target / "evidence_candidates.jsonl", templated)
    accepted, rejected, verification = verify_records(templated, policy)
    if not accepted:
        raise ValueError("No records accepted by release policy")
    accepted_path = target / "evidence_accepted.jsonl"
    write_jsonl(accepted_path, accepted)
    write_jsonl(target / "evidence_rejected.jsonl", rejected)
    build_summary = build_dataset(accepted_path, target, seed=int(policy.get("split_seed", 4050)))
    canonical = read_jsonl(target / "canonical.jsonl")
    paired_evidence = build_paired_evidence(canonical)
    if not paired_evidence:
        raise ValueError("No complete anterior/posterior evidence pairs")
    write_jsonl(target / "paired_evidence.jsonl", paired_evidence)
    paired_manifests = build_paired_task_manifests(paired_evidence)
    for name, rows in paired_manifests.items():
        write_jsonl(target / f"{name}.jsonl", rows)
    paired_dataset = paired_summary(paired_evidence, paired_manifests)
    visual_qa_contact_sheet = None
    if policy.get("write_visual_qa_contact_sheet", False):
        visual_qa_contact_sheet = create_region_contact_sheet(target / "paired_evidence.jsonl", target / "qa_region_contact_sheet_test.png")
    lesion_summary = None
    if policy.get("include_lesion_tasks", False):
        lesion_manifests, lesion_rejected, lesion_summary = build_lesion_task_manifests(canonical, paired_evidence, metastases)
        write_jsonl(target / "lesion_rejected.jsonl", lesion_rejected)
        for name, rows in lesion_manifests.items():
            write_jsonl(target / f"{name}.jsonl", rows)
        export_bbx_grounding(target / "lesion_grounding.jsonl", target / "lesion_grounding_bbx.jsonl")
    (target / "summary.json").write_text(json.dumps(paired_dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    patient_splits = build_patient_split_manifest(canonical, paired_evidence)
    write_jsonl(target / "patient_splits.jsonl", patient_splits)
    split_report = build_split_report(
        canonical,
        paired_evidence,
        minimum_eval_abnormal_pairs=int(policy.get("evaluation_warning_minimum_abnormal_pairs", 5)),
    )
    (target / "split_report.json").write_text(json.dumps(split_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (target / "grounding_det.jsonl").unlink(missing_ok=True)
    bbx_grounding_rows = export_bbx_grounding(target / "grounding.jsonl", target / "grounding_bbx.jsonl")
    manifest_names = ["vqa", "grounding", "grounded_vqa"]
    if lesion_summary is not None:
        manifest_names.extend(["lesion_vqa", "lesion_grounding", "lesion_grounded_vqa"])
    manifests = {name: read_jsonl(target / f"{name}.jsonl") for name in manifest_names}
    report = build_quality_report(canonical, manifests, rejected)
    write_quality_report(report, target / "quality_report.json")
    summary = {
        "release": policy.get("release", "r0"),
        "inventory_sources": len(inventory["sources"]),
        "metastases": {"rows": len(metastases), "assignments": dict(sorted(Counter(row["assignment"] for row in metastases).items()))},
        "verification": verification,
        "single_view_evidence": build_summary,
        "paired_dataset": paired_dataset,
        "visual_qa_contact_sheet": str(visual_qa_contact_sheet) if visual_qa_contact_sheet else None,
        "lesion_dataset": lesion_summary,
        "split_audit": {
            "split_assignment_sha256": split_report["split_assignment_sha256"],
            "leakage_checks": split_report["leakage_checks"],
            "evaluation_warnings": split_report["evaluation_warnings"],
        },
        "bbx_grounding_rows": bbx_grounding_rows,
    }
    (target / "release_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def load_policy(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Release policy must be a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the conservative WBBS R0 release from upstream sources.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Read-only source dataset directory, normally datasets/sources")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("configs/dataset/r0.json"))
    args = parser.parse_args()
    print(json.dumps(build_r0_release(args.dataset_root, args.output, load_policy(args.policy)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
