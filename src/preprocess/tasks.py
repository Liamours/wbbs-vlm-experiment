from __future__ import annotations

from typing import Any, Iterable


def build_task_manifests(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    manifests = {"vqa": [], "grounding": [], "grounded_vqa": []}
    for record in records:
        manifests["vqa"].extend(_vqa_rows(record))
        manifests["grounding"].extend(_grounding_rows(record))
        manifests["grounded_vqa"].extend(_grounded_vqa_rows(record))
    return manifests


def _vqa_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(record["qa"]):
        rows.append(
            {
                "task_id": f"{record['record_id']}:vqa:{index}",
                "image": record["image"],
                "image_size": record["image_size"],
                "question": item["question"],
                "answer": item["answer"],
                "template_id": item["template_id"],
                "template_arguments": item["template_arguments"],
                "answer_rule": item["answer_rule"],
                "evidence_id": record["record_id"],
                "split": record["split"],
                **({"answer_label": item["answer_label"]} if "answer_label" in item else {}),
                **({"source_label": record["source_label"]} if record.get("source_label") is not None else {}),
            }
        )
    return rows


def _grounding_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = [
        {
            "query": item["query"],
            "label": item["label"],
            "bbox": item["bbox"],
            "source": item["source"],
        }
        for item in record["grounding"]
    ]
    captions = [
        {
            "query": caption,
            "label": record["target"]["name"],
            "bbox": record["target"]["bbox"],
            "source": f"caption_{kind}",
        }
        for kind, caption in record["captions"].items()
    ]
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(explicit or captions):
        rows.append(
            {
                "task_id": f"{record['record_id']}:grounding:{index}",
                "image": record["image"],
                "image_size": record["image_size"],
                "query": item["query"],
                "label": item["label"],
                "bbox": item["bbox"],
                "source": item["source"],
                "evidence_id": record["record_id"],
                "split": record["split"],
                **({"source_label": record["source_label"]} if record.get("source_label") is not None else {}),
            }
        )
    return rows


def _grounded_vqa_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(record["grounded_qa"] or record["qa"]):
        rows.append(
            {
                "task_id": f"{record['record_id']}:grounded-vqa:{index}",
                "image": record["image"],
                "image_size": record["image_size"],
                "question": item["question"],
                "answer": item["answer"],
                "template_id": item["template_id"],
                "template_arguments": item["template_arguments"],
                "answer_rule": item["answer_rule"],
                "evidence_bbox": item.get("evidence_bbox", record["target"]["bbox"]),
                "evidence_id": record["record_id"],
                "split": record["split"],
                **({"answer_label": item["answer_label"]} if "answer_label" in item else {}),
                **({"target": item["target"]} if "target" in item else {}),
                **({"source_label": record["source_label"]} if record.get("source_label") is not None else {}),
            }
        )
    return rows
