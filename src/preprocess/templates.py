from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .tokens import format_grounded_target


TEMPLATE_VERSION = "wbbs-r0/v1"
GROUNDING_TEMPLATE_VERSION = "wbbs-r0-controlled-grounding/v1"
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


def apply_r0_templates(records: Iterable[dict[str, Any]], captions: dict[str, dict[str, dict[str, str]]]) -> list[dict[str, Any]]:
    expanded = []
    for source in records:
        record = deepcopy(source)
        if record["target"]["kind"] != "region":
            expanded.append(record)
            continue
        region = record["target"]["name"]
        if region not in captions:
            raise ValueError(f"No LIBS caption template for {region}")
        view = _view_name(record["view"])
        reference = _region_reference(region)
        description = captions[region]["description"]
        diagnosis = record.get("diagnosis")
        if diagnosis not in {"normal", "abnormal"}:
            expanded.append(record)
            continue
        diagnosis_caption = captions[region][diagnosis]
        record["caption_description"] = description["text"]
        record["caption_diagnosis"] = diagnosis_caption["text"]
        record["grounding"] = [
            {
                "query": f"Locate the {reference} in this {view} bone scan.",
                "label": region,
                "bbox": record["target"]["bbox"],
                "source": GROUNDING_TEMPLATE_VERSION,
            },
        ]
        record["qa"] = [
            {
                "question": f"Is abnormal tracer uptake present in the {reference} on this {view} bone scan?",
                "answer": _vqa_answer(reference, diagnosis),
                "answer_label": "yes" if diagnosis == "abnormal" else "no",
                "template_id": "r0.vqa.region_abnormality.v2",
                "template_arguments": {"region": region, "view": record["view"]},
                "answer_rule": "diagnosis_is_abnormal",
                "evidence_bbox": record["target"]["bbox"],
            }
        ]
        record["grounded_qa"] = [
            {
                "question": f"Locate the {reference} and determine whether it shows abnormal tracer uptake.",
                "answer": _grounded_answer(reference, diagnosis),
                "answer_label": diagnosis,
                "target": format_grounded_target(region, diagnosis, record["target"]["bbox"]),
                "template_id": "r0.grounded_vqa.region_abnormality.v1",
                "template_arguments": {"region": region},
                "answer_rule": "region_and_diagnosis",
                "evidence_bbox": record["target"]["bbox"],
            }
        ]
        record["provenance"]["libs_templates"] = {
            "description": description["template_id"],
            "diagnosis": diagnosis_caption["template_id"],
            "library_version": TEMPLATE_VERSION,
        }
        expanded.append(record)
    return expanded


def _view_name(view: str) -> str:
    return {"ANT": "anterior", "POST": "posterior"}.get(view, view.lower())


def _region_reference(region: str) -> str:
    if region not in REGION_REFERENCES:
        raise ValueError(f"No natural-language reference for {region}")
    return REGION_REFERENCES[region]


def _vqa_answer(reference: str, diagnosis: str) -> str:
    if diagnosis == "abnormal":
        return f"Yes, abnormal tracer uptake is present in the {reference}."
    return f"No, abnormal tracer uptake is not present in the {reference}."


def _grounded_answer(reference: str, diagnosis: str) -> str:
    if diagnosis == "abnormal":
        return f"The {reference} shows abnormal tracer uptake."
    return f"The {reference} does not show abnormal tracer uptake."
