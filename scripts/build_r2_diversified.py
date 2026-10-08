"""Build a controlled-language R2 release from the frozen R1 manifests.

R2 changes the natural-language surface forms and publishes a portable image
reference/structured-target schema. Evidence IDs, boxes, source labels, and
patient splits remain compatible with R1 so that benchmark differences are not
caused by changed annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TASK_FILES = (
    "vqa.jsonl",
    "grounding.jsonl",
    "grounded_vqa.jsonl",
    "lesion_vqa.jsonl",
    "lesion_grounding.jsonl",
    "lesion_grounded_vqa.jsonl",
)
MERGED_TASK_FILES = ("vqa.jsonl", "vgrounding.jsonl")
CORE_FILES = ("canonical.jsonl", "paired_evidence.jsonl", "metastases.jsonl", "patient_splits.jsonl")
REGION_REFERENCES = {
    "left ankle": "left ankle joint",
    "right ankle": "right ankle joint",
    "left chest": "left chest region",
    "right chest": "right chest region",
    "left elbow": "left elbow joint",
    "right elbow": "right elbow joint",
    "head": "head",
    "left knee": "left knee joint",
    "right knee": "right knee joint",
    "pelvis": "pelvis",
    "left shoulder": "left shoulder joint",
    "right shoulder": "right shoulder joint",
    "vertebra": "vertebral column",
}
VIEW_NAMES = {"ANT": "anterior", "POST": "posterior"}


ANATOMY_VQA_QUESTIONS = {
    "ANT": [
        "Is abnormal tracer uptake visible in the {ref} on the anterior bone scan?",
        "Does the anterior bone scan show abnormal tracer uptake at the {ref}?",
        "For the {ref}, is abnormal tracer uptake present on the anterior view?",
        "On the anterior projection, is the {ref} abnormal?",
        "Is the {ref} positive for abnormal tracer uptake on the anterior scan?",
        "Check the anterior scan: does the {ref} contain abnormal tracer uptake?",
        "Looking at the anterior image, is there abnormal tracer uptake in the {ref}?",
        "What is the abnormal-uptake status of the {ref} on the anterior view?",
    ],
    "POST": [
        "Is abnormal tracer uptake visible in the {ref} on the posterior bone scan?",
        "Does the posterior bone scan show abnormal tracer uptake at the {ref}?",
        "For the {ref}, is abnormal tracer uptake present on the posterior view?",
        "On the posterior projection, is the {ref} abnormal?",
        "Is the {ref} positive for abnormal tracer uptake on the posterior scan?",
        "Check the posterior scan: does the {ref} contain abnormal tracer uptake?",
        "Looking at the posterior image, is there abnormal tracer uptake in the {ref}?",
        "What is the abnormal-uptake status of the {ref} on the posterior view?",
    ],
    "BOTH": [
        "Do either bone-scan views show abnormal tracer uptake in the {ref}?",
        "Across the anterior and posterior scans, is the {ref} abnormal?",
        "Is abnormal tracer uptake seen in the {ref} on one or both views?",
        "Considering both projections, is the {ref} positive for abnormal tracer uptake?",
        "Do the paired scans indicate abnormal uptake in the {ref}?",
        "Check both views: is there abnormal tracer uptake in the {ref}?",
        "Looking across the two images, does the {ref} show abnormal tracer activity?",
        "What is the abnormal-uptake status of the {ref} across both views?",
    ],
}
ANATOMY_GROUNDING_QUESTIONS = {
    "ANT": [
        "Locate the {ref} on the anterior image.",
        "Find the {ref} in the anterior bone scan.",
        "Please mark the {ref} on the anterior projection.",
        "Where is the {ref} in the anterior view?",
        "Identify the {ref} on the anterior scan.",
        "Indicate the bounding box for the {ref} in the anterior image.",
        "Point to the {ref} on the anterior bone scan.",
        "Select the {ref} in the anterior projection.",
    ],
    "POST": [
        "Locate the {ref} on the posterior image.",
        "Find the {ref} in the posterior bone scan.",
        "Please mark the {ref} on the posterior projection.",
        "Where is the {ref} in the posterior view?",
        "Identify the {ref} on the posterior scan.",
        "Indicate the bounding box for the {ref} in the posterior image.",
        "Point to the {ref} on the posterior bone scan.",
        "Select the {ref} in the posterior projection.",
    ],
    "BOTH": [
        "Locate the {ref} in both bone-scan views.",
        "Find the {ref} across the anterior and posterior images.",
        "Please mark the {ref} in each projection.",
        "Where is the {ref} in the paired scans?",
        "Identify the {ref} on both views.",
        "Indicate the bounding boxes for the {ref} in the two images.",
        "Point to the {ref} across the paired bone scans.",
        "Select the {ref} in each projection.",
    ],
}
ANATOMY_GROUNDED_QUESTIONS = {
    "ANT": [
        "Locate the {ref} on the anterior image and determine whether it shows abnormal tracer uptake.",
        "Find the {ref} in the anterior scan, then classify its tracer uptake as normal or abnormal.",
        "Mark the {ref} on the anterior projection and report its abnormal-uptake status.",
        "Where is the {ref} in the anterior view, and is it abnormal?",
        "Identify the {ref} on the anterior bone scan and say whether abnormal uptake is present.",
        "Give the box for the {ref} in the anterior image and classify the uptake.",
        "Point to the {ref} on the anterior scan; does it show abnormal tracer activity?",
        "Select the {ref} in the anterior projection and state its diagnosis.",
    ],
    "POST": [
        "Locate the {ref} on the posterior image and determine whether it shows abnormal tracer uptake.",
        "Find the {ref} in the posterior scan, then classify its tracer uptake as normal or abnormal.",
        "Mark the {ref} on the posterior projection and report its abnormal-uptake status.",
        "Where is the {ref} in the posterior view, and is it abnormal?",
        "Identify the {ref} on the posterior bone scan and say whether abnormal uptake is present.",
        "Give the box for the {ref} in the posterior image and classify the uptake.",
        "Point to the {ref} on the posterior scan; does it show abnormal tracer activity?",
        "Select the {ref} in the posterior projection and state its diagnosis.",
    ],
    "BOTH": [
        "Locate the {ref} in both bone scans and determine whether either view shows abnormal uptake.",
        "Find the {ref} across the paired scans, then classify its tracer uptake as normal or abnormal.",
        "Mark the {ref} in each projection and report the combined abnormal-uptake status.",
        "Where is the {ref} in the two views, and is it abnormal on either one?",
        "Identify the {ref} on both scans and say whether abnormal uptake is present.",
        "Give the boxes for the {ref} in the paired images and classify the uptake.",
        "Point to the {ref} across both scans; does it show abnormal tracer activity?",
        "Select the {ref} in each projection and state the combined diagnosis.",
    ],
}

YES_ANSWERS = {
    "single": [
        "Yes—the {ref} shows abnormal tracer uptake.",
        "Yes. Abnormal tracer uptake is visible in the {ref}.",
        "The {ref} is positive for abnormal tracer uptake.",
        "Abnormal uptake is present in the {ref}.",
        "There is abnormal tracer activity in the {ref}.",
        "The scan indicates abnormal tracer uptake in the {ref}.",
    ],
    "both": [
        "Yes—at least one view shows abnormal tracer uptake in the {ref}.",
        "Yes. The paired scans show abnormal uptake in the {ref}.",
        "The paired views indicate abnormal tracer uptake in the {ref}.",
        "Abnormal tracer activity is seen in the {ref} in one or both views.",
        "The {ref} is abnormal on the paired bone scans.",
        "One of the two projections is positive for abnormal uptake in the {ref}.",
    ],
}
NO_ANSWERS = {
    "single": [
        "No—the {ref} does not show abnormal tracer uptake.",
        "No. The {ref} appears free of abnormal tracer uptake.",
        "The {ref} is negative for abnormal tracer uptake.",
        "No abnormal uptake is seen in the {ref}.",
        "Abnormal tracer activity is absent from the {ref}.",
        "The scan shows no abnormal tracer uptake in the {ref}.",
    ],
    "both": [
        "No—neither view shows abnormal tracer uptake in the {ref}.",
        "No. The paired scans show no abnormal uptake in the {ref}.",
        "The {ref} is negative for abnormal tracer uptake on both views.",
        "Abnormal tracer activity is absent from the {ref} in both views.",
        "Both views show no abnormal tracer uptake in the {ref}.",
        "The two projections are negative for abnormal uptake in the {ref}.",
    ],
}

LESION_VQA_QUESTIONS = [
    "Is the annotated hotspot in the {ref} on the {view} image labelled as a bone metastasis?",
    "On the {view} projection, does the marked hotspot in the {ref} represent bone metastasis?",
    "Does the annotated {ref} hotspot in the {view} scan carry a metastatic label?",
    "Review the hotspot in the {ref} on the {view} image: is it metastatic?",
    "Is the marked hotspot at the {ref} classified as a bone metastasis on the {view} view?",
    "For the annotated hotspot in the {ref}, is the {view} scan label metastatic?",
    "Does the {view} bone scan label the annotated {ref} hotspot as metastatic?",
    "What is the metastasis label for the annotated hotspot in the {ref} on the {view} image?",
]
LESION_GROUNDING_QUESTIONS = [
    "Locate the annotated {descriptor} in the {ref} on the {view} image.",
    "Find the marked {descriptor} at the {ref} in the {view} scan.",
    "Please draw a box around the annotated {descriptor} in the {ref} on the {view} projection.",
    "Where is the {descriptor} hotspot in the {ref} on the {view} view?",
    "Identify the annotated {descriptor} at the {ref} in the {view} bone scan.",
    "Indicate the bounding box for the {descriptor} in the {ref} on the {view} image.",
    "Point to the marked {descriptor} in the {ref} on the {view} projection.",
    "Select the annotated {descriptor} hotspot in the {ref} on the {view} scan.",
]
LESION_GROUNDED_QUESTIONS = [
    "Locate the annotated hotspot in the {ref} on the {view} image and determine whether it is labelled as bone metastasis.",
    "Find the marked hotspot in the {ref} on the {view} scan, then classify its metastasis label.",
    "Box the annotated hotspot at the {ref} on the {view} projection and report whether it is metastatic.",
    "Where is the hotspot in the {ref} on the {view} view, and does its label indicate bone metastasis?",
    "Identify the annotated hotspot in the {ref} on the {view} image and state its metastasis class.",
    "Give the bounding box for the hotspot in the {ref} on the {view} scan and classify it.",
    "Point to the marked hotspot at the {ref} on the {view} projection; is it metastatic?",
    "Select the hotspot in the {ref} on the {view} image and report its label.",
]
LESION_YES_ANSWERS = [
    "Yes—the annotated hotspot in the {ref} is labelled as a bone metastasis.",
    "Yes. The marked hotspot at the {ref} carries a bone-metastasis label.",
    "The hotspot in the {ref} is classified as metastatic.",
    "The annotation identifies the {ref} hotspot as a bone metastasis.",
    "This hotspot is positive for the bone-metastasis label at the {ref}.",
    "A metastatic label is assigned to the annotated hotspot in the {ref}.",
]
LESION_NO_ANSWERS = [
    "No—the annotated hotspot in the {ref} is labelled as non-metastatic.",
    "No. The marked hotspot at the {ref} carries a non-metastatic label.",
    "The hotspot in the {ref} is classified as non-metastatic.",
    "The annotation identifies the {ref} hotspot as non-metastatic.",
    "This hotspot is negative for the bone-metastasis label at the {ref}.",
    "No metastatic label is assigned to the annotated hotspot in the {ref}.",
]
LESION_GROUNDED_YES_ANSWERS = [
    "The annotated hotspot in the {ref} is labelled as a bone metastasis.",
    "The marked hotspot at the {ref} carries a bone-metastasis label.",
    "The hotspot in the {ref} is classified as metastatic.",
    "The annotation identifies a bone metastasis at the {ref} hotspot.",
    "This hotspot is positive for the bone-metastasis class at the {ref}.",
    "A metastatic class is assigned to the hotspot in the {ref}.",
]
LESION_GROUNDED_NO_ANSWERS = [
    "The annotated hotspot in the {ref} is labelled as non-metastatic.",
    "The marked hotspot at the {ref} carries a non-metastatic label.",
    "The hotspot in the {ref} is classified as non-metastatic.",
    "The annotation identifies a non-metastatic hotspot at the {ref}.",
    "This hotspot is negative for the bone-metastasis class at the {ref}.",
    "A non-metastatic class is assigned to the hotspot in the {ref}.",
]

WHOLE_BODY_QUESTIONS = {
    "ANT": [
        "Does the anterior whole-body bone scan show abnormal tracer uptake anywhere?",
        "Is abnormal tracer activity present anywhere on the anterior whole-body image?",
        "What is the overall tracer-uptake status of the anterior bone scan?",
        "Looking at the anterior projection, is the whole-body scan abnormal?",
        "How would you classify the anterior whole-body scan: normal or abnormal?",
        "Check the anterior image for any abnormal tracer uptake.",
        "Is the anterior view positive for abnormal tracer uptake at any site?",
        "Report the whole-body diagnosis for the anterior scan.",
    ],
    "POST": [
        "Does the posterior whole-body bone scan show abnormal tracer uptake anywhere?",
        "Is abnormal tracer activity present anywhere on the posterior whole-body image?",
        "What is the overall tracer-uptake status of the posterior bone scan?",
        "Looking at the posterior projection, is the whole-body scan abnormal?",
        "How would you classify the posterior whole-body scan: normal or abnormal?",
        "Check the posterior image for any abnormal tracer uptake.",
        "Is the posterior view positive for abnormal tracer uptake at any site?",
        "Report the whole-body diagnosis for the posterior scan.",
    ],
}
WHOLE_BODY_YES_ANSWERS = {
    "ANT": [
        "Yes—abnormal tracer uptake is present somewhere on the anterior whole-body scan.",
        "Yes. The anterior whole-body image contains an abnormal uptake focus.",
        "The anterior scan is positive for abnormal tracer uptake.",
        "Abnormal tracer activity is detected on the anterior whole-body view.",
        "The anterior whole-body diagnosis is abnormal.",
    ],
    "POST": [
        "Yes—abnormal tracer uptake is present somewhere on the posterior whole-body scan.",
        "Yes. The posterior whole-body image contains an abnormal uptake focus.",
        "The posterior scan is positive for abnormal tracer uptake.",
        "Abnormal tracer activity is detected on the posterior whole-body view.",
        "The posterior whole-body diagnosis is abnormal.",
    ],
}
WHOLE_BODY_NO_ANSWERS = {
    "ANT": [
        "No—there is no abnormal tracer uptake on the anterior whole-body scan.",
        "No. The anterior whole-body image shows no abnormal uptake focus.",
        "The anterior scan is negative for abnormal tracer uptake.",
        "No abnormal tracer activity is detected on the anterior whole-body view.",
        "The anterior whole-body diagnosis is normal.",
    ],
    "POST": [
        "No—there is no abnormal tracer uptake on the posterior whole-body scan.",
        "No. The posterior whole-body image shows no abnormal uptake focus.",
        "The posterior scan is negative for abnormal tracer uptake.",
        "No abnormal tracer activity is detected on the posterior whole-body view.",
        "The posterior whole-body diagnosis is normal.",
    ],
}


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _ref(label: str) -> str:
    return REGION_REFERENCES.get(label, label)


def _normalise_image_reference(value: Any) -> Any:
    """Use a portable BS80K-rooted image reference in the R2 release."""

    if not isinstance(value, str):
        return value
    path = value.replace("\\", "/")
    marker = "bs80k-imaging-raw/"
    if marker in path:
        return "bs80k/" + path.split(marker, 1)[1]
    return path


def _normalise_images(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    if isinstance(updated.get("image"), str):
        updated["image"] = _normalise_image_reference(updated["image"])
    if isinstance(updated.get("images"), list):
        updated["images"] = [
            {**image, "image": _normalise_image_reference(image.get("image"))}
            if isinstance(image, dict)
            else image
            for image in updated["images"]
        ]
    return updated


def _normalise_core_row(row: dict[str, Any]) -> dict[str, Any]:
    return _normalise_images(row)


def _scope(row: dict[str, Any]) -> str:
    value = str(row.get("view_scope") or row.get("view") or "BOTH").upper()
    return value if value in {"ANT", "POST", "BOTH"} else "BOTH"


def _view(scope: str) -> str:
    if scope in VIEW_NAMES:
        return VIEW_NAMES[scope]
    return "paired"


def _choice(row: dict[str, Any], family: str, values: list[str]) -> tuple[int, str]:
    key = f"{family}|{row.get('task_id', '')}|{row.get('evidence_id', '')}|{row.get('metastasis_id', '')}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % len(values)
    return index, values[index]


def _metadata(row: dict[str, Any], family: str, index: int) -> dict[str, Any]:
    updated = _normalise_images(row)
    updated.update(
        {
            "template_id": f"r2.{family}.s{index + 1:02d}",
            "template_family": family,
            "surface_form_id": f"{family}.s{index + 1:02d}",
            "template_version": "r2-controlled-template-v2",
            "generation_source": "r1-manifest-language-diversification",
            "language": "en",
        }
    )
    return updated


def _benign_malignant(value: Any) -> str | None:
    normalized = str(value).strip().lower()
    if normalized in {"abnormal", "abnormality", "malignant", "metastatic", "yes", "1"}:
        return "malignant"
    if normalized in {"normal", "benign", "non-metastatic", "non_metastatic", "no", "0"}:
        return "benign"
    return None


def _is_malignant(value: Any) -> bool:
    return _benign_malignant(value) == "malignant"


def _source_region_class(value: Any) -> str | None:
    mapped = _benign_malignant(value)
    if mapped == "malignant":
        return "abnormal"
    if mapped == "benign":
        return "normal"
    return None


def _build_hierarchy(
    canonical: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    metastases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build effective labels from the lesion -> region -> whole-body hierarchy.

    A malignant, uniquely assigned XML hotspot upgrades its containing bone
    region for that view, and any upgraded/abnormal region upgrades that
    patient's whole-body view.  The raw source labels remain in each row; this
    map only supplies the effective classification used by R2 targets.
    """

    records = {str(row.get("record_id")): row for row in canonical if row.get("record_id")}
    malignant_records: set[str] = set()
    for metastasis in metastases:
        if metastasis.get("assignment") != "unique_region" or _benign_malignant(metastasis.get("label")) != "malignant":
            continue
        candidates = metastasis.get("candidate_record_ids")
        if isinstance(candidates, list) and len(candidates) == 1 and str(candidates[0]) in records:
            malignant_records.add(str(candidates[0]))

    canonical_effective: dict[str, bool] = {}
    canonical_raw: dict[str, bool] = {}
    patient_view_effective: defaultdict[tuple[str, str], bool] = defaultdict(bool)
    patient_view_raw: defaultdict[tuple[str, str], bool] = defaultdict(bool)
    for record_id, row in records.items():
        raw = _is_malignant(row.get("diagnosis"))
        effective = bool(raw or record_id in malignant_records)
        canonical_raw[record_id] = raw
        canonical_effective[record_id] = effective
        patient_view_raw[(str(row.get("patient_id")), str(row.get("view")))] = bool(
            patient_view_raw[(str(row.get("patient_id")), str(row.get("view")))] or raw
        )
        patient_view_effective[(str(row.get("patient_id")), str(row.get("view")))] = bool(
            patient_view_effective[(str(row.get("patient_id")), str(row.get("view")))] or effective
        )

    pair_effective: dict[str, dict[str, bool]] = {}
    pair_raw: dict[str, dict[str, str | None]] = {}
    for pair in paired:
        evidence_id = pair.get("evidence_id")
        if not evidence_id:
            continue
        by_view: dict[str, bool] = {}
        raw_by_view: dict[str, str | None] = {}
        diagnosis_by_view = pair.get("diagnosis_by_view") if isinstance(pair.get("diagnosis_by_view"), dict) else {}
        source_labels = pair.get("source_labels") if isinstance(pair.get("source_labels"), dict) else {}
        for image in pair.get("images", []):
            if not isinstance(image, dict) or image.get("view") not in VIEW_NAMES:
                continue
            view = str(image["view"])
            record_id = str(image.get("record_id") or "")
            base = diagnosis_by_view.get(view, source_labels.get(view))
            raw_by_view[view] = _source_region_class(base)
            effective = bool(_is_malignant(base) or canonical_effective.get(record_id, False))
            by_view[view] = effective
            patient_view_effective[(str(pair.get("patient_id")), view)] = bool(
                patient_view_effective[(str(pair.get("patient_id")), view)] or effective
            )
        pair_effective[str(evidence_id)] = by_view
        pair_raw[str(evidence_id)] = raw_by_view

    return {
        "canonical": canonical_effective,
        "canonical_raw": canonical_raw,
        "pair": pair_effective,
        "pair_raw": pair_raw,
        "patient_view_raw": dict(patient_view_raw),
        "patient_view": dict(patient_view_effective),
    }


def _row_views(row: dict[str, Any]) -> list[str]:
    scope = _scope(row)
    return ["ANT", "POST"] if scope == "BOTH" else [scope]


def _select_view_values(by_view: dict[str, Any], row: dict[str, Any]) -> Any:
    selected = {view: by_view[view] for view in _row_views(row) if view in by_view}
    if len(selected) == 1:
        return next(iter(selected.values()))
    return selected


def _label_provenance(
    row: dict[str, Any],
    level: str,
    hierarchy: dict[str, Any] | None,
    effective_label: str | None,
) -> dict[str, Any]:
    """Expose raw-vs-effective labels without overwriting source fields."""

    raw_by_view: dict[str, Any] = {}
    effective_by_view: dict[str, Any] = {}
    if hierarchy:
        if level == "whole_body":
            patient_id = str(row.get("patient_id") or "")
            raw_by_view = {
                view: "abnormal" if hierarchy.get("patient_view_raw", {}).get((patient_id, view), False) else "normal"
                for view in _row_views(row)
            }
            effective_by_view = {
                view: bool(hierarchy.get("patient_view", {}).get((patient_id, view), False))
                for view in _row_views(row)
            }
        else:
            evidence_id = str(row.get("evidence_id") or "")
            raw_by_view = dict(hierarchy.get("pair_raw", {}).get(evidence_id, {}))
            if level == "metastasis":
                hotspot_class = _benign_malignant(row.get("source_hotspot_label"))
                effective_by_view = {view: hotspot_class == "malignant" for view in _row_views(row)}
            else:
                effective_by_view = dict(hierarchy.get("pair", {}).get(evidence_id, {}))

    if not raw_by_view:
        raw = row.get("diagnosis_by_view") or row.get("source_labels")
        if isinstance(raw, dict):
            raw_by_view = {str(view): _source_region_class(value) for view, value in raw.items()}
        elif raw is not None:
            raw_by_view = {view: _source_region_class(raw) for view in _row_views(row)}
    if not effective_by_view:
        effective_by_view = {view: effective_label == "malignant" for view in _row_views(row)}

    raw_selected = _select_view_values(raw_by_view, row)
    conflicts = []
    for view in _row_views(row):
        raw_class = _source_region_class(raw_by_view.get(view))
        effective = effective_by_view.get(view)
        if raw_class is not None and effective is not None and raw_class != ("abnormal" if effective else "normal"):
            conflicts.append(view)
    return {
        "source_region_label": raw_selected,
        "effective_region_label": effective_label,
        "label_conflict": bool(conflicts),
    }


def _structured_classification(
    row: dict[str, Any],
    level: str,
    hierarchy: dict[str, Any] | None,
    locations: list[dict[str, Any]],
) -> tuple[str | None, Any]:
    # A metastasis target is classified by its own XML hotspot label.  Region
    # labels must not override a benign hotspot in an otherwise abnormal region.
    if level == "metastasis":
        source = row.get("source_hotspot_label")
        class_label = _benign_malignant(source)
        if class_label is None:
            for key in ("lesion_class", "answer_label"):
                class_label = _benign_malignant(row.get(key))
                if class_label:
                    source = row.get(key)
                    break
        return class_label, source

    if hierarchy:
        if level == "whole_body":
            patient_id = str(row.get("patient_id") or "")
            status = hierarchy.get("patient_view", {})
            values = [status.get((patient_id, view)) for view in _row_views(row)]
            if any(value is not None for value in values):
                source = row.get("answer_label") or row.get("diagnosis")
                return ("malignant" if any(values) else "benign"), source
        elif level == "bone_region":
            evidence_id = str(row.get("evidence_id") or "")
            by_view = hierarchy.get("pair", {}).get(evidence_id)
            if isinstance(by_view, dict) and by_view:
                values = [by_view.get(view) for view in _row_views(row)]
                if any(value is not None for value in values):
                    source = row.get("source_labels") or row.get("diagnosis_by_view") or row.get("answer_label")
                    return ("malignant" if any(values) else "benign"), source

    raw_values: list[Any] = []
    for key in ("answer_label", "diagnosis"):
        if row.get(key) is not None:
            raw_values.append(row.get(key))
    source_labels = row.get("source_labels")
    if isinstance(source_labels, dict):
        raw_values.extend(source_labels.get(item.get("view")) for item in locations if item.get("view") in source_labels)
    mapped = [_benign_malignant(value) for value in raw_values]
    mapped = [value for value in mapped if value]
    primary_source = row.get("answer_label") or row.get("diagnosis")
    if primary_source is None:
        primary_source = source_labels or raw_values
    return ("malignant" if "malignant" in mapped else "benign" if mapped else None), primary_source


def _structured_target(
    row: dict[str, Any], level: str, locations: list[dict[str, Any]], hierarchy: dict[str, Any] | None = None
) -> dict[str, Any]:
    region_label = "metastasis" if level == "metastasis" else "whole_body" if level == "whole_body" else str(row.get("label") or "unknown")
    class_label, source_label = _structured_classification(row, level, hierarchy, locations)
    region: dict[str, Any] = {"label": region_label, "level": level}
    if level == "metastasis":
        region["anatomical_region"] = row.get("label")
    classification: dict[str, Any] = {"label": class_label, "source_label": source_label}
    return {
        "region": region,
        "classification": classification,
        "locations": [
            {"view": item.get("view"), "bbox": item.get("bbox")}
            for item in locations
            if isinstance(item, dict) and item.get("view") is not None and item.get("bbox") is not None
        ],
        "schema_version": "r2-target-v1",
    }


def _question_form(text: str) -> str:
    value = text.strip().lower()
    first = value.split(maxsplit=1)[0] if value else ""
    if first in {"what"}:
        return "what"
    if first in {"where"}:
        return "where"
    if first in {"does", "do"}:
        return "does"
    if first in {"is", "are", "can", "has", "have"}:
        return "is"
    if first in {"locate", "find", "mark", "identify", "indicate", "point", "select", "give", "box"}:
        return "where"
    return "other"


def _merge_metadata(
    row: dict[str, Any],
    task_group: str,
    level: str,
    interaction: str,
    hierarchy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = _normalise_images(row)
    text = str(updated.get("question") or updated.get("query") or "")
    updated.update(
        {
            "dataset_name": "wbbs-r2",
            "task_group": task_group,
            "region_level": level,
            "grounding_level": level,
            "interaction_type": interaction,
            "question_form": _question_form(text),
            "answer_mode": "binary" if task_group == "vqa" else "grounded",
        }
    )
    locations = updated.get("evidence_targets") or updated.get("targets") or []
    structured = _structured_target(updated, level, locations, hierarchy)
    updated.update(_label_provenance(updated, level, hierarchy, structured["classification"].get("label")))
    updated["labels"] = {
        "region": structured["region"],
        "classification": structured["classification"],
        "schema_version": "r2-labels-v1",
    }
    if task_group == "vgrounding":
        updated["target"] = structured
    return updated


def _whole_body_rows(
    canonical_rows: Iterable[dict[str, Any]], hierarchy: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Create image-level ANT/POST classification rows from region labels."""

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in canonical_rows:
        key = (str(row.get("patient_id")), str(row.get("view")), str(row.get("image")))
        item = grouped.setdefault(
            key,
            {
                "patient_id": row.get("patient_id"),
                "view": row.get("view"),
                "image": row.get("image"),
                "image_size": row.get("image_size"),
                "split": row.get("split"),
                "abnormal": False,
            },
        )
        effective = bool(str(row.get("diagnosis", "")).lower() == "abnormal")
        if hierarchy:
            effective = bool(effective or hierarchy.get("canonical", {}).get(str(row.get("record_id")), False))
        item["abnormal"] = bool(item["abnormal"] or effective)
    output: list[dict[str, Any]] = []
    for item in sorted(grouped.values(), key=lambda value: (str(value["patient_id"]), str(value["view"]), str(value["image"]))):
        view = str(item["view"])
        if view not in VIEW_NAMES:
            continue
        family = f"whole_body_vqa_{view.lower()}"
        task_id = f"wbbs:patient:{item['patient_id']}:whole-body:vqa:{view.lower()}"
        seed = {"task_id": task_id, "patient_id": item["patient_id"], "view": view}
        index, question = _choice(seed, family, WHOLE_BODY_QUESTIONS[view])
        bank = WHOLE_BODY_YES_ANSWERS if item["abnormal"] else WHOLE_BODY_NO_ANSWERS
        _, answer = _choice(seed, family + ":answer", bank[view])
        row = {
            "task_id": task_id,
            "patient_id": item["patient_id"],
            "evidence_id": None,
            "images": [{"view": view, "image": _normalise_image_reference(item["image"]), "image_size": item["image_size"]}],
            "question": question,
            "answer": answer,
            "answer_label": "yes" if item["abnormal"] else "no",
            "answer_rule": "any_region_abnormality",
            "diagnosis": "abnormal" if item["abnormal"] else "normal",
            "label": "whole_body",
            "view_scope": view,
            "targets": [],
            "split": item["split"],
            "template_id": f"r2.{family}.s{index + 1:02d}",
            "template_family": family,
            "surface_form_id": f"{family}.s{index + 1:02d}",
            "template_version": "r2-controlled-template-v2",
            "generation_source": "canonical-region-label-aggregation",
            "language": "en",
        }
        row = _merge_metadata(row, "vqa", "whole_body", "classification")
        row["localization_mode"] = "image_level"
        output.append(row)
    return output


def _anatomy_status(
    row: dict[str, Any], task: str, scope: str, hierarchy: dict[str, Any] | None = None
) -> str:
    if hierarchy:
        by_view = hierarchy.get("pair", {}).get(str(row.get("evidence_id") or ""))
        if isinstance(by_view, dict) and by_view:
            values = [by_view.get(view) for view in (["ANT", "POST"] if scope == "BOTH" else [scope])]
            if any(value is not None for value in values):
                return "abnormal" if any(values) else "normal"
    if task == "vqa":
        return "abnormal" if str(row.get("answer_label", "")).lower() == "yes" else "normal"
    return "abnormal" if str(row.get("answer_label", "")).lower() == "abnormal" else "normal"


def _diversify_anatomy(
    row: dict[str, Any], task: str, hierarchy: dict[str, Any] | None = None
) -> dict[str, Any]:
    scope = _scope(row)
    ref = _ref(str(row.get("label") or "region"))
    family = f"anatomy_{task}_{scope.lower()}"
    if task == "vqa":
        index, template = _choice(row, family, ANATOMY_VQA_QUESTIONS[scope])
        status = _anatomy_status(row, task, scope, hierarchy)
        answer_bank = (YES_ANSWERS if status == "abnormal" else NO_ANSWERS)["both" if scope == "BOTH" else "single"]
        _, answer = _choice(row, family + ":answer", answer_bank)
        updated = _metadata(row, family, index)
        updated["question"] = template.format(ref=ref)
        updated["answer"] = answer.format(ref=ref)
        updated["answer_label"] = "yes" if status == "abnormal" else "no"
        return updated
    if task == "grounding":
        index, template = _choice(row, family, ANATOMY_GROUNDING_QUESTIONS[scope])
        updated = _metadata(row, family, index)
        updated["query"] = template.format(ref=ref)
        return updated
    index, template = _choice(row, family, ANATOMY_GROUNDED_QUESTIONS[scope])
    status = _anatomy_status(row, task, scope, hierarchy)
    answer_bank = (YES_ANSWERS if status == "abnormal" else NO_ANSWERS)["both" if scope == "BOTH" else "single"]
    _, answer = _choice(row, family + ":answer", answer_bank)
    updated = _metadata(row, family, index)
    updated["question"] = template.format(ref=ref)
    updated["answer"] = answer.format(ref=ref)
    updated["answer_label"] = status
    return updated


def _lesion_label(row: dict[str, Any]) -> bool:
    return str(row.get("source_hotspot_label", "")).lower() == "abnormal" or str(row.get("lesion_class", "")) == "bone_metastasis"


def _lesion_descriptor(row: dict[str, Any]) -> str:
    return "bone-metastasis hotspot" if _lesion_label(row) else "non-metastatic hotspot"


def _diversify_lesion(row: dict[str, Any], task: str) -> dict[str, Any]:
    scope = _scope(row)
    view = _view(scope)
    ref = _ref(str(row.get("label") or "region"))
    family = f"lesion_{task}_{scope.lower()}"
    positive = _lesion_label(row)
    if task == "vqa":
        index, template = _choice(row, family, LESION_VQA_QUESTIONS)
        bank = LESION_YES_ANSWERS if positive else LESION_NO_ANSWERS
        _, answer = _choice(row, family + ":answer", bank)
        updated = _metadata(row, family, index)
        updated["question"] = template.format(ref=ref, view=view)
        updated["answer"] = answer.format(ref=ref)
        return updated
    if task == "grounding":
        index, template = _choice(row, family, LESION_GROUNDING_QUESTIONS)
        updated = _metadata(row, family, index)
        updated["query"] = template.format(ref=ref, view=view, descriptor=_lesion_descriptor(row))
        return updated
    index, template = _choice(row, family, LESION_GROUNDED_QUESTIONS)
    bank = LESION_GROUNDED_YES_ANSWERS if positive else LESION_GROUNDED_NO_ANSWERS
    _, answer = _choice(row, family + ":answer", bank)
    updated = _metadata(row, family, index)
    updated["question"] = template.format(ref=ref, view=view)
    updated["answer"] = answer.format(ref=ref)
    return updated


def diversify_row(row: dict[str, Any], task: str, hierarchy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one R2 row while preserving all evidence and label fields."""

    if task.startswith("lesion_"):
        return _diversify_lesion(row, task.removeprefix("lesion_"))
    return _diversify_anatomy(row, task, hierarchy)


def _select_task_rows(
    source: Path, limit: int | None, hierarchy: dict[str, Any] | None = None
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    original_counts: dict[str, int] = {}
    for filename in TASK_FILES:
        task = filename.removesuffix(".jsonl")
        rows = []
        for row in _jsonl(source / filename):
            original_counts[task] = original_counts.get(task, 0) + 1
            if limit is None or len(rows) < limit:
                rows.append(diversify_row(row, task, hierarchy))
        selected[task] = rows
    return selected, original_counts


def _merged_rows(
    source: Path,
    selected: dict[str, list[dict[str, Any]]],
    limit: int | None,
    hierarchy: dict[str, Any] | None = None,
    canonical: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Decorate legacy families and merge them into the two R2 task groups."""

    canonical = canonical if canonical is not None else list(_jsonl(source / "canonical.jsonl"))
    whole_body_all = _whole_body_rows(canonical, hierarchy)
    whole_body = whole_body_all if limit is None else whole_body_all[:limit]
    original_counts = {"whole_body_vqa": len(whole_body_all)}
    original_counts.update({key: len(value) for key, value in selected.items()})

    vqa: list[dict[str, Any]] = []
    vgrounding: list[dict[str, Any]] = []
    vqa.extend(_merge_metadata(row, "vqa", "whole_body", "classification", hierarchy) for row in whole_body)
    vqa.extend(_merge_metadata(row, "vqa", "bone_region", "classification", hierarchy) for row in selected["vqa"])
    vqa.extend(_merge_metadata(row, "vqa", "metastasis", "classification", hierarchy) for row in selected["lesion_vqa"])
    vgrounding.extend(_merge_metadata(row, "vgrounding", "bone_region", "localization", hierarchy) for row in selected["grounding"])
    vgrounding.extend(_merge_metadata(row, "vgrounding", "bone_region", "localization_and_classification", hierarchy) for row in selected["grounded_vqa"])
    vgrounding.extend(_merge_metadata(row, "vgrounding", "metastasis", "localization", hierarchy) for row in selected["lesion_grounding"])
    vgrounding.extend(_merge_metadata(row, "vgrounding", "metastasis", "localization_and_classification", hierarchy) for row in selected["lesion_grounded_vqa"])
    # Interleave families deterministically so a training reader sees all
    # levels throughout the file rather than one long contiguous block.
    return {"vqa": _interleave(vqa), "vgrounding": _interleave(vgrounding)}, original_counts


def _interleave(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row.get('region_level')}|{row.get('interaction_type')}"
        buckets.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    keys = sorted(buckets)
    position = 0
    while True:
        emitted = False
        for key in keys:
            bucket = buckets[key]
            if position < len(bucket):
                output.append(bucket[position])
                emitted = True
        if not emitted:
            return output
        position += 1


def _filter_core(source: Path, destination: Path, selected: dict[str, list[dict[str, Any]]], sample: bool) -> None:
    if not sample:
        for filename in CORE_FILES:
            path = source / filename
            if path.exists():
                _write_jsonl(destination / filename, (_normalise_core_row(row) for row in _jsonl(path)))
        return
    evidence_ids = {row.get("evidence_id") for rows in selected.values() for row in rows if row.get("evidence_id")}
    metastasis_ids = {row.get("metastasis_id") for rows in selected.values() for row in rows if row.get("metastasis_id")}
    image_paths = {
        image.get("image")
        for rows in selected.values()
        for row in rows
        for image in row.get("images", [])
        if isinstance(image, dict) and image.get("image")
    }
    patient_ids = {row.get("patient_id") for rows in selected.values() for row in rows if row.get("patient_id")}
    selected_pairs: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    for row in _jsonl(source / "paired_evidence.jsonl"):
        if row.get("evidence_id") in evidence_ids:
            row = _normalise_core_row(row)
            selected_pairs.append(row)
            record_ids.update(str(image["record_id"]) for image in row.get("images", []) if image.get("record_id"))
    _write_jsonl(destination / "paired_evidence.jsonl", selected_pairs)
    _write_jsonl(
        destination / "canonical.jsonl",
        (_normalise_core_row(row) for row in _jsonl(source / "canonical.jsonl") if row.get("record_id") in record_ids or row.get("image") in image_paths),
    )
    _write_jsonl(destination / "metastases.jsonl", (_normalise_core_row(row) for row in _jsonl(source / "metastases.jsonl") if row.get("metastasis_id") in metastasis_ids))
    if (source / "patient_splits.jsonl").exists():
        patient_ids.update(row.get("patient_id") for row in selected_pairs if row.get("patient_id"))
        _write_jsonl(destination / "patient_splits.jsonl", (_normalise_core_row(row) for row in _jsonl(source / "patient_splits.jsonl") if row.get("patient_id") in patient_ids))


def build_release(source: Path, destination: Path, limit: int | None = None) -> dict[str, Any]:
    if not (source / "canonical.jsonl").exists() or not (source / "paired_evidence.jsonl").exists():
        raise FileNotFoundError(f"R1 release is missing canonical/paired manifests: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    canonical = list(_jsonl(source / "canonical.jsonl"))
    paired = list(_jsonl(source / "paired_evidence.jsonl"))
    metastases = list(_jsonl(source / "metastases.jsonl")) if (source / "metastases.jsonl").exists() else []
    hierarchy = _build_hierarchy(canonical, paired, metastases)
    selected, original_counts = _select_task_rows(source, limit, hierarchy)
    merged, merged_original_counts = _merged_rows(source, selected, limit, hierarchy, canonical)
    # For sample core closure we need the legacy rows plus the generated
    # whole-body rows; for a complete release, core files are copied intact.
    core_selection = dict(selected)
    whole_body_rows = _whole_body_rows(canonical, hierarchy)
    core_selection["whole_body_vqa"] = whole_body_rows if limit is None else whole_body_rows[:limit]
    _filter_core(source, destination, core_selection, sample=limit is not None)
    for filename in (*TASK_FILES, *MERGED_TASK_FILES):
        stale = destination / filename
        if stale.exists():
            stale.unlink()
    counts: dict[str, int] = {task: _write_jsonl(destination / f"{task}.jsonl", rows) for task, rows in merged.items()}
    variants: dict[str, Counter[str]] = {task: Counter(str(row.get("surface_form_id")) for row in rows) for task, rows in merged.items()}
    summary = {
        "release": "wbbs-r2-sample" if limit is not None else "wbbs-r2",
        "derived_from": str(source.resolve()),
        "source_release": "r1",
        "sample_limit_per_task": limit,
        "deterministic": True,
        "language": "en",
        "template_version": "r2-controlled-template-v2",
        "image_reference_root": "bs80k",
        "image_reference_format": "bs80k/<wholeBodyANT|wholeBodyPOST>/<filename>",
        "target_schema": "r2-target-v1",
        "classification_labels": ["benign", "malignant"],
        "label_hierarchy": {
            "metastasis": "source_hotspot_label",
            "bone_region": "raw_region_diagnosis_or_malignant_unique_hotspot_in_view",
            "whole_body": "any_effective_malignant_bone_region_in_view",
            "conflict_field": "label_conflict",
        },
        "source_task_rows": merged_original_counts,
        "exported_task_rows": counts,
        "surface_forms_per_task": {task: len(counter) for task, counter in variants.items()},
        "surface_form_counts": {task: dict(sorted(counter.items())) for task, counter in variants.items()},
        "semantic_fields_preserved": [
            "task_id",
            "evidence_id",
            "split",
            "images",
            "targets",
            "evidence_targets",
            "target",
            "answer_label",
            "label",
            "lesion_class",
            "source_hotspot_label",
            "target.region",
            "target.classification",
            "target.locations",
            "labels.region",
            "labels.classification",
            "source_region_label",
            "effective_region_label",
            "label_conflict",
        ],
    }
    (destination / "release_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "diversification_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-release", type=Path, default=Path("datasets/releases/r1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Export at most this many rows per task (sample smoke release).")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    summary = build_release(args.input_release.resolve(), args.output_dir.resolve(), args.limit)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
