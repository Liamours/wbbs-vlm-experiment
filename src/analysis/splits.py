from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any, Iterable


def build_patient_split_manifest(canonical: Iterable[dict[str, Any]], paired_evidence: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = list(canonical)
    pairs = list(paired_evidence)
    by_patient: dict[str, dict[str, Any]] = {}
    for record in records:
        patient = record["patient_id"]
        entry = by_patient.setdefault(
            patient,
            {"patient_id": patient, "split": record["split"], "canonical_records": 0, "paired_evidence_records": 0},
        )
        if entry["split"] != record["split"]:
            raise ValueError(f"Patient crosses splits: {patient}")
        entry["canonical_records"] += 1
    for pair in pairs:
        patient = pair["patient_id"]
        if patient not in by_patient:
            raise ValueError(f"Paired evidence has no canonical patient: {patient}")
        if by_patient[patient]["split"] != pair["split"]:
            raise ValueError(f"Paired evidence crosses split: {pair['evidence_id']}")
        by_patient[patient]["paired_evidence_records"] += 1
    return [by_patient[patient] for patient in sorted(by_patient)]


def build_split_report(
    canonical: Iterable[dict[str, Any]],
    paired_evidence: Iterable[dict[str, Any]],
    minimum_eval_abnormal_pairs: int = 5,
) -> dict[str, Any]:
    if minimum_eval_abnormal_pairs < 0:
        raise ValueError("minimum_eval_abnormal_pairs must be non-negative")
    records = list(canonical)
    pairs = list(paired_evidence)
    manifest = build_patient_split_manifest(records, pairs)
    patient_splits = {row["patient_id"]: row["split"] for row in manifest}
    canonical_counts = Counter(record["split"] for record in records)
    paired_counts = Counter(pair["split"] for pair in pairs)
    paired_patients = {split: {pair["patient_id"] for pair in pairs if pair["split"] == split} for split in ("train", "val", "test")}
    pair_labels: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    view_labels: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    for pair in pairs:
        split, region = pair["split"], pair["region"]
        pair_labels[split][region][pair["pair_diagnosis"]] += 1
        for view, diagnosis in pair["diagnosis_by_view"].items():
            view_labels[split][region][view][diagnosis] += 1

    duplicate_link_conflicts = _duplicate_link_conflicts(records, patient_splits)
    pair_split_conflicts = _pair_split_conflicts(pairs, records)
    warnings = _evaluation_warnings(pair_labels, minimum_eval_abnormal_pairs)
    total_patients = len(manifest)
    split_counts = {
        split: {
            "canonical_records": canonical_counts[split],
            "canonical_patients": sum(row["split"] == split for row in manifest),
            "paired_evidence_records": paired_counts[split],
            "paired_evidence_patients": len(paired_patients[split]),
            "patient_fraction": round(sum(row["split"] == split for row in manifest) / total_patients, 6) if total_patients else 0.0,
        }
        for split in ("train", "val", "test")
    }
    return {
        "split_assignment_sha256": _split_fingerprint(manifest),
        "split_counts": split_counts,
        "paired_pair_diagnosis_by_split_region": {
            split: {region: dict(sorted(counts.items())) for region, counts in sorted(regions.items())}
            for split, regions in sorted(pair_labels.items())
        },
        "paired_view_diagnosis_by_split_region": {
            split: {
                region: {view: dict(sorted(counts.items())) for view, counts in sorted(views.items())}
                for region, views in sorted(regions.items())
            }
            for split, regions in sorted(view_labels.items())
        },
        "leakage_checks": {
            "patient_cross_split": 0,
            "duplicate_link_cross_split": duplicate_link_conflicts,
            "pair_cross_split": pair_split_conflicts,
            "all_passed": not duplicate_link_conflicts and not pair_split_conflicts,
        },
        "evaluation_warnings": warnings,
    }


def _duplicate_link_conflicts(records: list[dict[str, Any]], patient_splits: dict[str, str]) -> int:
    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for key in ("duplicate_group_id", "duplicate_of_sibling_component"):
            if value := record.get(key):
                groups[f"{key}:{value}"].add(record["split"])
        linked = record.get("duplicate_of_patient_id")
        if linked in patient_splits:
            groups[f"duplicate_of_patient_id:{min(record['patient_id'], linked)}:{max(record['patient_id'], linked)}"].update(
                {record["split"], patient_splits[linked]}
            )
    return sum(len(splits) > 1 for splits in groups.values())


def _pair_split_conflicts(pairs: list[dict[str, Any]], records: list[dict[str, Any]]) -> int:
    record_splits = {record["record_id"]: record["split"] for record in records}
    conflicts = 0
    for pair in pairs:
        image_splits = {record_splits.get(image["record_id"]) for image in pair["images"]}
        if image_splits != {pair["split"]}:
            conflicts += 1
    return conflicts


def _evaluation_warnings(pair_labels: dict[str, dict[str, Counter[str]]], minimum: int) -> list[dict[str, Any]]:
    warnings = []
    for split in ("val", "test"):
        for region, counts in sorted(pair_labels[split].items()):
            abnormal = counts["abnormal"]
            if abnormal < minimum:
                warnings.append(
                    {
                        "split": split,
                        "region": region,
                        "abnormal_pairs": abnormal,
                        "minimum_expected": minimum,
                        "reason": "rare_abnormal_class",
                    }
                )
    return warnings


def _split_fingerprint(manifest: list[dict[str, Any]]) -> str:
    payload = "".join(f"{row['patient_id']}\t{row['split']}\n" for row in manifest)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
