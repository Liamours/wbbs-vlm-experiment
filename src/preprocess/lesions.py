from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .templates import REGION_REFERENCES
from .tokens import format_grounded_target


LESION_CLASSES = {"Abnormal": "bone_metastasis", "Normal": "non_metastatic_hotspot"}


def build_lesion_task_manifests(
    canonical: Iterable[dict[str, Any]], paired_evidence: Iterable[dict[str, Any]], metastases: Iterable[dict[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    records = {record["record_id"]: record for record in canonical}
    pairs = {pair["evidence_id"]: pair for pair in paired_evidence}
    manifests = {"lesion_vqa": [], "lesion_grounding": [], "lesion_grounded_vqa": []}
    rejected = []
    for metastasis in metastases:
        eligible, reason = _resolve_lesion(metastasis, records, pairs)
        if eligible is None:
            rejected.append({"metastasis_id": metastasis["metastasis_id"], "reason": reason})
            continue
        manifests["lesion_vqa"].append(_lesion_vqa(eligible))
        manifests["lesion_grounding"].append(_lesion_grounding(eligible))
        manifests["lesion_grounded_vqa"].append(_lesion_grounded_vqa(eligible))
    summary = {
        "eligible_lesions": len(manifests["lesion_vqa"]),
        "rejected_lesions": len(rejected),
        "reasons": dict(sorted(Counter(row["reason"] for row in rejected).items())),
        "source_labels": dict(sorted(Counter(row["source_hotspot_label"] for row in manifests["lesion_vqa"]).items())),
        "tasks": {name: len(rows) for name, rows in manifests.items()},
    }
    return manifests, rejected, summary


def _resolve_lesion(
    metastasis: dict[str, Any], records: dict[str, dict[str, Any]], pairs: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any] | None, str | None]:
    if metastasis.get("assignment") != "unique_region":
        return None, metastasis.get("assignment", "invalid_assignment")
    candidates = metastasis.get("candidate_record_ids", [])
    if not isinstance(candidates, list) or len(candidates) != 1:
        return None, "invalid_candidate_record"
    source = records.get(candidates[0])
    if source is None:
        return None, "candidate_not_accepted"
    if source["view"] != metastasis.get("view"):
        return None, "view_mismatch"
    region = source["target"]["name"]
    pair_id = f"wbbs:pair:{source['patient_id']}:{region.replace(' ', '_')}"
    pair = pairs.get(pair_id)
    if pair is None:
        return None, "accepted_but_unpaired"
    source_label = metastasis.get("label")
    lesion_class = LESION_CLASSES.get(source_label)
    if lesion_class is None:
        return None, "unsupported_source_label"
    return {
        "metastasis_id": metastasis["metastasis_id"],
        "pair": pair,
        "region": region,
        "view": source["view"],
        "bbox": metastasis["bbox"],
        "source_hotspot_label": source_label,
        "lesion_class": lesion_class,
        "source_id": metastasis.get("source_id"),
    }, None


def _lesion_vqa(lesion: dict[str, Any]) -> dict[str, Any]:
    pair, reference = lesion["pair"], _reference(lesion["region"])
    label = lesion["source_hotspot_label"]
    metastatic = label == "Abnormal"
    return {
        "task_id": f"{lesion['metastasis_id']}:lesion-vqa",
        "images": _images(pair),
        "question": f"Is the annotated hotspot in the {reference} on the {_view_name(lesion['view'])} image labelled as a bone metastasis?",
        "answer": (
            f"Yes, the annotated hotspot in the {reference} is labelled as a bone metastasis."
            if metastatic
            else f"No, the annotated hotspot in the {reference} is labelled as non-metastatic."
        ),
        "answer_label": "yes" if metastatic else "no",
        "label": lesion["region"],
        "lesion_class": lesion["lesion_class"],
        "source_hotspot_label": label,
        "view_scope": lesion["view"],
        "targets": [{"view": lesion["view"], "bbox": lesion["bbox"]}],
        "source_labels": pair["source_labels"],
        "metastasis_id": lesion["metastasis_id"],
        "source_id": lesion["source_id"],
        "template_id": "r1.lesion_vqa.v1",
        "answer_rule": "xml_hotspot_label",
        "evidence_id": pair["evidence_id"],
        "split": pair["split"],
    }


def _lesion_grounding(lesion: dict[str, Any]) -> dict[str, Any]:
    pair, reference = lesion["pair"], _reference(lesion["region"])
    lesion_name = "bone-metastasis hotspot" if lesion["source_hotspot_label"] == "Abnormal" else "non-metastatic hotspot"
    return {
        "task_id": f"{lesion['metastasis_id']}:lesion-grounding",
        "images": _images(pair),
        "query": f"Locate the annotated {lesion_name} in the {reference} on the {_view_name(lesion['view'])} image.",
        "label": lesion["region"],
        "lesion_class": lesion["lesion_class"],
        "source_hotspot_label": lesion["source_hotspot_label"],
        "view_scope": lesion["view"],
        "targets": [{"view": lesion["view"], "bbox": lesion["bbox"]}],
        "source_labels": pair["source_labels"],
        "source": "bs80k-xml-hotspot/v1",
        "metastasis_id": lesion["metastasis_id"],
        "source_id": lesion["source_id"],
        "evidence_id": pair["evidence_id"],
        "split": pair["split"],
    }


def _lesion_grounded_vqa(lesion: dict[str, Any]) -> dict[str, Any]:
    pair, reference = lesion["pair"], _reference(lesion["region"])
    classification = "bone metastasis" if lesion["source_hotspot_label"] == "Abnormal" else "non-metastatic"
    targets = [{"view": lesion["view"], "bbox": lesion["bbox"]}]
    return {
        "task_id": f"{lesion['metastasis_id']}:lesion-grounded-vqa",
        "images": _images(pair),
        "question": f"Locate the annotated hotspot in the {reference} on the {_view_name(lesion['view'])} image and identify whether it is labelled as bone metastasis.",
        "answer": f"The annotated hotspot in the {reference} is labelled as {classification}.",
        "answer_label": "abnormal" if lesion["source_hotspot_label"] == "Abnormal" else "normal",
        "target": format_grounded_target(lesion["region"], "abnormal" if lesion["source_hotspot_label"] == "Abnormal" else "normal", targets),
        "label": lesion["region"],
        "lesion_class": lesion["lesion_class"],
        "source_hotspot_label": lesion["source_hotspot_label"],
        "view_scope": lesion["view"],
        "evidence_targets": targets,
        "source_labels": pair["source_labels"],
        "metastasis_id": lesion["metastasis_id"],
        "source_id": lesion["source_id"],
        "template_id": "r1.lesion_grounded_vqa.v1",
        "answer_rule": "xml_hotspot_label",
        "evidence_id": pair["evidence_id"],
        "split": pair["split"],
    }


def _images(pair: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"view": item["view"], "image": item["image"], "image_size": item["image_size"]} for item in pair["images"]]


def _reference(region: str) -> str:
    if region not in REGION_REFERENCES:
        raise ValueError(f"No natural-language reference for {region}")
    return REGION_REFERENCES[region]


def _view_name(view: str) -> str:
    return {"ANT": "anterior", "POST": "posterior"}[view]
