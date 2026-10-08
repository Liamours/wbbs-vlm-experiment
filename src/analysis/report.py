from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def build_quality_report(canonical: Iterable[dict[str, Any]], manifests: dict[str, Iterable[dict[str, Any]]], rejected: Iterable[dict[str, Any]]) -> dict[str, Any]:
    evidence = list(canonical)
    rejected_rows = list(rejected)
    return {
        "records": len(evidence),
        "patients": len({row["patient_id"] for row in evidence}),
        "by_split": dict(sorted(Counter(row["split"] for row in evidence).items())),
        "by_diagnosis": dict(sorted(Counter(row.get("diagnosis") for row in evidence).items())),
        "by_region": dict(sorted(Counter(row["target"]["name"] for row in evidence).items())),
        "flagged_records": dict(sorted(Counter(flag for row in evidence for flag, value in row.get("flags", {}).items() if value).items())),
        "tasks": {
            name: {
                "rows": len(rows := list(items)),
                "templates": dict(sorted(Counter(row.get("template_id") for row in rows if row.get("template_id")).items())),
            }
            for name, items in manifests.items()
        },
        "rejections": dict(sorted(Counter(reason for row in rejected_rows for reason in row["reasons"]).items())),
    }


def write_quality_report(report: dict[str, Any], output: str | Path) -> None:
    target = Path(output)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
