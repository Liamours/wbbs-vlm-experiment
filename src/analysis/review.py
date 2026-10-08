from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from preprocess.canonical import read_jsonl, validate_box, write_jsonl


BBOX_STATUSES = {"accept", "correct", "not_assessable"}
DIAGNOSES = {"normal", "abnormal", "indeterminate"}
OVERALL_DECISIONS = {"accept", "correct", "exclude", "indeterminate"}
VIEWS = ("ANT", "POST")


def validate_review_responses(queue: Iterable[dict[str, Any]], responses: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cases = {row["review_id"]: row for row in queue}
    rows = list(responses)
    response_ids = [row.get("review_id") for row in rows]
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("Review responses contain duplicate review_id values")
    if set(response_ids) != set(cases):
        raise ValueError("Review responses must cover the blinded queue exactly")
    reviewers = {row.get("reviewer_id") for row in rows}
    if len(reviewers) != 1 or not isinstance(next(iter(reviewers)), str) or not next(iter(reviewers)).strip():
        raise ValueError("Each response file must contain one non-empty reviewer_id")
    for row in rows:
        _validate_response(row, cases[row["review_id"]])
    return {row["review_id"]: row for row in rows}


def compare_reviews(
    queue: Iterable[dict[str, Any]],
    first_responses: Iterable[dict[str, Any]],
    second_responses: Iterable[dict[str, Any]],
    bbox_iou_threshold: float = 0.9,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not 0 < bbox_iou_threshold <= 1:
        raise ValueError("bbox_iou_threshold must be in (0, 1]")
    cases = {row["review_id"]: row for row in queue}
    first = validate_review_responses(cases.values(), first_responses)
    second = validate_review_responses(cases.values(), second_responses)
    first_reviewer = next(iter({row["reviewer_id"] for row in first.values()}))
    second_reviewer = next(iter({row["reviewer_id"] for row in second.values()}))
    if first_reviewer == second_reviewer:
        raise ValueError("Independent review files must have different reviewer_id values")

    diagnosis_pairs: list[tuple[str, str]] = []
    box_ious: list[float] = []
    box_status_agreements = 0
    box_status_total = 0
    overall_agreements = 0
    adjudication = []
    for review_id in sorted(cases):
        case, left, right = cases[review_id], first[review_id], second[review_id]
        differences = []
        iou_by_view: dict[str, float | None] = {}
        for view in VIEWS:
            left_diagnosis = left["diagnosis_by_view"][view]
            right_diagnosis = right["diagnosis_by_view"][view]
            diagnosis_pairs.append((left_diagnosis, right_diagnosis))
            if left_diagnosis != right_diagnosis:
                differences.append(f"diagnosis:{view}")
            left_status = left["bbox_status_by_view"][view]
            right_status = right["bbox_status_by_view"][view]
            box_status_total += 1
            if left_status == right_status:
                box_status_agreements += 1
            if left_status == "not_assessable" or right_status == "not_assessable":
                iou_by_view[view] = None
                if left_status != right_status:
                    differences.append(f"bbox_status:{view}")
            else:
                iou = _iou(_effective_bbox(case, left, view), _effective_bbox(case, right, view))
                iou_by_view[view] = iou
                box_ious.append(iou)
                if left_status != right_status or iou < bbox_iou_threshold:
                    differences.append(f"bbox:{view}")
        if left["overall_decision"] == right["overall_decision"]:
            overall_agreements += 1
        else:
            differences.append("overall_decision")
        if differences:
            adjudication.append(
                {
                    "review_id": review_id,
                    "evidence_id": case["evidence_id"],
                    "region": case["region"],
                    "images": case["images"],
                    "target_bboxes": case["target_bboxes"],
                    "differences": sorted(set(differences)),
                    "bbox_iou_by_view": iou_by_view,
                    "reviewer_responses": {first_reviewer: left, second_reviewer: right},
                }
            )
    total_cases = len(cases)
    diagnosis_agreements = sum(left == right for left, right in diagnosis_pairs)
    report = {
        "reviewers": [first_reviewer, second_reviewer],
        "cases": total_cases,
        "bbox_iou_threshold": bbox_iou_threshold,
        "diagnosis": {
            "comparisons": len(diagnosis_pairs),
            "agreements": diagnosis_agreements,
            "agreement_rate": _rate(diagnosis_agreements, len(diagnosis_pairs)),
            "cohen_kappa": _cohen_kappa(diagnosis_pairs),
        },
        "bbox": {
            "status_comparisons": box_status_total,
            "status_agreements": box_status_agreements,
            "status_agreement_rate": _rate(box_status_agreements, box_status_total),
            "iou_comparisons": len(box_ious),
            "mean_iou": round(sum(box_ious) / len(box_ious), 6) if box_ious else None,
            "iou_at_threshold": sum(iou >= bbox_iou_threshold for iou in box_ious),
        },
        "overall_decision": {
            "agreements": overall_agreements,
            "agreement_rate": _rate(overall_agreements, total_cases),
        },
        "adjudication_cases": len(adjudication),
        "fully_agreed_cases": total_cases - len(adjudication),
    }
    return report, adjudication


def write_review_audit(
    queue_path: str | Path,
    first_response_path: str | Path,
    second_response_path: str | Path,
    output: str | Path,
    bbox_iou_threshold: float = 0.9,
) -> dict[str, Any]:
    report, adjudication = compare_reviews(
        read_jsonl(queue_path),
        read_jsonl(first_response_path),
        read_jsonl(second_response_path),
        bbox_iou_threshold=bbox_iou_threshold,
    )
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    (target / "review_agreement.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(target / "review_adjudication_queue.jsonl", adjudication)
    return report


def validate_adjudications(queue: Iterable[dict[str, Any]], adjudications: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cases = {row["review_id"]: row for row in queue}
    rows = list(adjudications)
    if {row.get("review_id") for row in rows} != set(cases) or len(rows) != len(cases):
        raise ValueError("Adjudications must cover the blinded queue exactly once")
    result = {}
    for row in rows:
        review_id = row["review_id"]
        if not isinstance(row.get("adjudicator_id"), str) or not row["adjudicator_id"].strip():
            raise ValueError("Adjudications require adjudicator_id")
        decision = row.get("final_decision")
        if decision not in {"accept", "correct", "exclude"}:
            raise ValueError("final_decision must be accept, correct, or exclude")
        if not isinstance(row.get("notes"), str):
            raise ValueError("Adjudications require text notes")
        if decision != "exclude":
            _validate_final_boxes(row.get("final_bbox_by_view"), cases[review_id])
            _validate_final_diagnoses(row.get("final_diagnosis_by_view"))
        result[review_id] = row
    return result


def _validate_response(row: dict[str, Any], case: dict[str, Any]) -> None:
    if not isinstance(row.get("notes"), str):
        raise ValueError("Review responses require text notes")
    statuses = row.get("bbox_status_by_view")
    corrected = row.get("corrected_bbox_by_view")
    diagnoses = row.get("diagnosis_by_view")
    if not isinstance(statuses, dict) or set(statuses) != set(VIEWS):
        raise ValueError("bbox_status_by_view must contain ANT and POST")
    if not isinstance(corrected, dict) or set(corrected) != set(VIEWS):
        raise ValueError("corrected_bbox_by_view must contain ANT and POST")
    _validate_final_diagnoses(diagnoses, allow_indeterminate=True)
    if row.get("overall_decision") not in OVERALL_DECISIONS:
        raise ValueError("Invalid overall_decision")
    sizes = {image["view"]: image["image_size"] for image in case["images"]}
    for view in VIEWS:
        status = statuses[view]
        bbox = corrected[view]
        if status not in BBOX_STATUSES:
            raise ValueError("Invalid bbox status")
        if status == "correct":
            _validate_bbox_in_image(bbox, sizes[view])
        elif bbox is not None:
            raise ValueError("corrected bbox must be null unless status is correct")


def _validate_final_boxes(boxes: Any, case: dict[str, Any]) -> None:
    if not isinstance(boxes, dict) or set(boxes) != set(VIEWS):
        raise ValueError("final_bbox_by_view must contain ANT and POST")
    sizes = {image["view"]: image["image_size"] for image in case["images"]}
    for view in VIEWS:
        _validate_bbox_in_image(boxes[view], sizes[view])


def _validate_final_diagnoses(diagnoses: Any, allow_indeterminate: bool = False) -> None:
    allowed = DIAGNOSES if allow_indeterminate else {"normal", "abnormal"}
    if not isinstance(diagnoses, dict) or set(diagnoses) != set(VIEWS) or not set(diagnoses.values()) <= allowed:
        raise ValueError("diagnosis_by_view has invalid values")


def _validate_bbox_in_image(value: Any, image_size: list[int]) -> None:
    _, _, x2, y2 = validate_box(value)
    if x2 > image_size[0] or y2 > image_size[1]:
        raise ValueError("bbox exceeds image bounds")


def _effective_bbox(case: dict[str, Any], response: dict[str, Any], view: str) -> list[float]:
    if response["bbox_status_by_view"][view] == "correct":
        return response["corrected_bbox_by_view"][view]
    return next(target["bbox"] for target in case["target_bboxes"] if target["view"] == view)


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_counts, right_counts = Counter(left for left, _ in pairs), Counter(right for _, right in pairs)
    expected = sum((left_counts[label] / len(pairs)) * (right_counts[label] / len(pairs)) for label in set(left_counts) | set(right_counts))
    return None if expected == 1 else round((observed - expected) / (1 - expected), 6)


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate independent clinician reviews and create an adjudication queue.")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--first-response", type=Path, required=True)
    parser.add_argument("--second-response", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bbox-iou-threshold", type=float, default=0.9)
    args = parser.parse_args()
    print(
        json.dumps(
            write_review_audit(
                args.queue,
                args.first_response,
                args.second_response,
                args.output,
                bbox_iou_threshold=args.bbox_iou_threshold,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
