from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from preprocess.canonical import read_jsonl

from .validate import validate_manifest


R1_MANIFESTS = ("canonical", "vqa", "grounding", "grounded_vqa", "lesion_vqa", "lesion_grounding", "lesion_grounded_vqa")
R1_FILES = ("source_inventory.json", "release_summary.json", "split_report.json", "patient_splits.jsonl", "quality_report.json", "lesion_rejected.jsonl", "qa_region_contact_sheet_test.png")
REPO_DOCUMENTS = ("DATASET_CARD.md", "LICENSE_REVIEW.md", "CHANGELOG.md", "CONTRIBUTING.md", "CITATION.cff.template", "BASELINES.md")


def audit_publish_readiness(release: str | Path, repository: str | Path) -> dict[str, Any]:
    release_path, repository_path = Path(release), Path(repository)
    missing_files = [name for name in R1_FILES if not (release_path / name).is_file()]
    missing_documents = [name for name in REPO_DOCUMENTS if not (repository_path / name).is_file()]
    manifests = {name: validate_manifest(release_path / f"{name}.jsonl", name) for name in R1_MANIFESTS if (release_path / f"{name}.jsonl").is_file()}
    missing_manifests = sorted(set(R1_MANIFESTS) - set(manifests))
    split_report = json.loads((release_path / "split_report.json").read_text(encoding="utf-8")) if not missing_files else {}
    release_summary = json.loads((release_path / "release_summary.json").read_text(encoding="utf-8")) if not missing_files else {}
    technical_ready = not missing_files and not missing_manifests and not missing_documents and split_report.get("leakage_checks", {}).get("all_passed") is True
    return {
        "release": str(release_path),
        "technical_ready": technical_ready,
        "missing_release_files": missing_files,
        "missing_manifests": missing_manifests,
        "missing_repository_documents": missing_documents,
        "manifest_validation": manifests,
        "split_assignment_sha256": split_report.get("split_assignment_sha256"),
        "leakage_checks": split_report.get("leakage_checks"),
        "release_name": release_summary.get("release"),
        "external_publication_gates": [
            "Independent clinician review, agreement analysis, and adjudication are required before R0.1 can be called gold.",
            "Explicit BS-80K redistribution rights or a manifests-only distribution decision is required.",
            "Project author metadata, code-license selection, and archival DOI are required for final publication metadata.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit technical WBBS R1 publication readiness.")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_publish_readiness(args.release, args.repository)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
