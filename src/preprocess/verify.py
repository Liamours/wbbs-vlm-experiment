from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


DEFAULT_POLICY = {
    "exclude_low_precision_regions": True,
    "exclude_known_artifacts": True,
    "exclude_outlier_images": True,
    "exclude_corrupt_images": True,
    "exclude_missing_images": True,
}


def verify_records(records: Iterable[dict[str, Any]], policy: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rules = {**DEFAULT_POLICY, **(policy or {})}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons = Counter()
    for record in records:
        record_reasons = _reasons(record, rules)
        if record_reasons:
            reasons.update(record_reasons)
            rejected.append({"record_id": record["record_id"], "reasons": record_reasons})
        else:
            accepted.append(record)
    return accepted, rejected, {"accepted": len(accepted), "rejected": len(rejected), "reasons": dict(sorted(reasons.items()))}


def _reasons(record: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    flags = record.get("flags", {})
    reasons = []
    if rules["exclude_missing_images"] and flags.get("missing_image"):
        reasons.append("missing_image")
    if rules["exclude_low_precision_regions"] and flags.get("low_precision_region"):
        reasons.append("low_precision_region")
    if rules["exclude_known_artifacts"] and flags.get("known_artifact"):
        reasons.append("known_artifact")
    if rules["exclude_outlier_images"] and flags.get("outlier"):
        reasons.append("outlier_image")
    if rules["exclude_corrupt_images"] and flags.get("likely_corrupt_image"):
        reasons.append("likely_corrupt_image")
    if record.get("target", {}).get("kind") == "region" and record.get("diagnosis") not in {"normal", "abnormal"}:
        reasons.append("unsupported_diagnosis")
    if not record.get("qa"):
        reasons.append("no_verified_vqa_template")
    if not record.get("grounding"):
        reasons.append("no_verified_grounding_template")
    return reasons
