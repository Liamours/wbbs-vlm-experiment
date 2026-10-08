from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .templates import REGION_REFERENCES
from .tokens import format_grounded_target


VIEWS = ("ANT", "POST")


def build_paired_evidence(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record["target"]["kind"] != "region" or record["view"] not in VIEWS:
            continue
        key = (record["patient_id"], record["target"]["name"])
        if record["view"] in grouped[key]:
            raise ValueError(f"Duplicate view for paired evidence: {key}:{record['view']}")
        grouped[key][record["view"]] = record
    pairs = []
    for (patient_id, region), views in sorted(grouped.items()):
        if set(views) != set(VIEWS):
            continue
        ant, post = views["ANT"], views["POST"]
        if ant["split"] != post["split"]:
            raise ValueError(f"Cross-split image pair: {patient_id}:{region}")
        diagnosis_by_view = {view: views[view]["diagnosis"] for view in VIEWS}
        if not set(diagnosis_by_view.values()) <= {"normal", "abnormal"}:
            raise ValueError(f"Unsupported pair diagnosis: {patient_id}:{region}")
        pairs.append(
            {
                "evidence_id": f"wbbs:pair:{patient_id}:{region.replace(' ', '_')}",
                "patient_id": patient_id,
                "region": region,
                "split": ant["split"],
                "images": [
                    {
                        "view": view,
                        "image": views[view]["image"],
                        "image_size": views[view]["image_size"],
                        "bbox": views[view]["target"]["bbox"],
                        "record_id": views[view]["record_id"],
                    }
                    for view in VIEWS
                ],
                "diagnosis_by_view": diagnosis_by_view,
                "source_labels": {view: views[view].get("source_label") for view in VIEWS},
                "pair_diagnosis": "abnormal" if "abnormal" in diagnosis_by_view.values() else "normal",
            }
        )
    return pairs


def build_paired_task_manifests(pairs: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    manifests = {"vqa": [], "grounding": [], "grounded_vqa": []}
    for pair in pairs:
        for scope in (*VIEWS, "BOTH"):
            diagnosis = pair["diagnosis_by_view"][scope] if scope in VIEWS else pair["pair_diagnosis"]
            targets = [image for image in pair["images"] if scope == "BOTH" or image["view"] == scope]
            manifests["vqa"].append(_vqa_row(pair, scope, diagnosis))
            manifests["grounding"].append(_grounding_row(pair, scope, targets))
            manifests["grounded_vqa"].append(_grounded_vqa_row(pair, scope, diagnosis, targets))
    return manifests


def paired_summary(pairs: Iterable[dict[str, Any]], manifests: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = list(pairs)
    return {
        "paired_records": len(rows),
        "patients": len({row["patient_id"] for row in rows}),
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "tasks": {name: len(items) for name, items in manifests.items()},
        "diagnosis_disagreements": sum(len(set(row["diagnosis_by_view"].values())) > 1 for row in rows),
    }


def _vqa_row(pair: dict[str, Any], scope: str, diagnosis: str) -> dict[str, Any]:
    reference = _reference(pair["region"])
    suffix = _scope_suffix(scope)
    return {
        "task_id": f"{pair['evidence_id']}:vqa:{scope.lower()}",
        "images": _images(pair),
        "question": f"Is abnormal tracer uptake present in the {reference}{suffix}?",
        "answer": _vqa_answer(reference, diagnosis),
        "answer_label": "yes" if diagnosis == "abnormal" else "no",
        "label": pair["region"],
        "view_scope": scope,
        "diagnosis_by_view": pair["diagnosis_by_view"],
        "source_labels": pair["source_labels"],
        "template_id": f"r0.paired_vqa.{scope.lower()}.v1",
        "answer_rule": "view_diagnosis" if scope in VIEWS else "any_view_abnormal",
        "evidence_id": pair["evidence_id"],
        "split": pair["split"],
    }


def _grounding_row(pair: dict[str, Any], scope: str, targets: list[dict[str, Any]]) -> dict[str, Any]:
    reference = _reference(pair["region"])
    query = f"Locate the {reference}{_grounding_suffix(scope)}."
    return {
        "task_id": f"{pair['evidence_id']}:grounding:{scope.lower()}",
        "images": _images(pair),
        "query": query,
        "label": pair["region"],
        "view_scope": scope,
        "targets": [{"view": item["view"], "bbox": item["bbox"]} for item in targets],
        "source_labels": pair["source_labels"],
        "source": "wbbs-r0-paired-grounding/v1",
        "evidence_id": pair["evidence_id"],
        "split": pair["split"],
    }


def _grounded_vqa_row(pair: dict[str, Any], scope: str, diagnosis: str, targets: list[dict[str, Any]]) -> dict[str, Any]:
    reference = _reference(pair["region"])
    question = f"Locate the {reference}{_grounding_suffix(scope)} and determine whether it shows abnormal tracer uptake."
    target_boxes = [{"view": item["view"], "bbox": item["bbox"]} for item in targets]
    return {
        "task_id": f"{pair['evidence_id']}:grounded-vqa:{scope.lower()}",
        "images": _images(pair),
        "question": question,
        "answer": _grounded_answer(reference, diagnosis),
        "answer_label": diagnosis,
        "target": format_grounded_target(pair["region"], diagnosis, target_boxes),
        "label": pair["region"],
        "view_scope": scope,
        "diagnosis_by_view": pair["diagnosis_by_view"],
        "source_labels": pair["source_labels"],
        "template_id": f"r0.paired_grounded_vqa.{scope.lower()}.v1",
        "answer_rule": "view_diagnosis" if scope in VIEWS else "any_view_abnormal",
        "evidence_targets": target_boxes,
        "evidence_id": pair["evidence_id"],
        "split": pair["split"],
    }


def _images(pair: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"view": item["view"], "image": item["image"], "image_size": item["image_size"]} for item in pair["images"]]


def _reference(region: str) -> str:
    if region not in REGION_REFERENCES:
        raise ValueError(f"No natural-language reference for {region}")
    return REGION_REFERENCES[region]


def _scope_suffix(scope: str) -> str:
    return "" if scope == "BOTH" else f" on the {_view_name(scope)} bone scan"


def _grounding_suffix(scope: str) -> str:
    return " in these bone scans" if scope == "BOTH" else f" in the {_view_name(scope)} image"


def _view_name(view: str) -> str:
    return {"ANT": "anterior", "POST": "posterior"}[view]


def _vqa_answer(reference: str, diagnosis: str) -> str:
    if diagnosis == "abnormal":
        return f"Yes, abnormal tracer uptake is present in the {reference}."
    return f"No, abnormal tracer uptake is not present in the {reference}."


def _grounded_answer(reference: str, diagnosis: str) -> str:
    if diagnosis == "abnormal":
        return f"The {reference} shows abnormal tracer uptake."
    return f"The {reference} does not show abnormal tracer uptake."
