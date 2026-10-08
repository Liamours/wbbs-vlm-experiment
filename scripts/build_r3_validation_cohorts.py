"""Build validation-only WBBS paired-view and perturbation evaluation cohorts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from preprocess.canonical import read_jsonl, write_jsonl


def _view(view: str) -> str:
    return "PST" if view == "POST" else view


def _target(row: dict[str, Any], task: str, targets: list[dict[str, Any]]) -> str:
    if task == "vqa":
        payload = {"task": task, "answer": row["answer"], "class": row["answer_label"]}
    else:
        payload = {
            "task": task,
            "region": str(row["label"]).replace(" ", "_"),
            "boxes": [{"view": _view(target["view"]), "bbox": [int(value) for value in target["bbox"]]} for target in targets],
        }
        if task == "grounded_vqa":
            payload.update({"answer": row["answer"], "class": row["answer_label"]})
    return json.dumps(payload, separators=(",", ":"))


def _prompt(task: str, label: str) -> str:
    if task == "vqa":
        return f"Is the {label} abnormal in the provided bone scan?"
    if task == "grounding":
        return f"Locate the {label} in the provided bone scan."
    return f"Locate the {label} in the provided bone scan and classify it as normal or abnormal."


def _paired_patient_ids(release: Path) -> dict[str, str]:
    return {
        str(row["evidence_id"]): str(row["patient_id"])
        for row in read_jsonl(release / "paired_evidence.jsonl")
        if row.get("evidence_id") and row.get("patient_id")
    }


def _eligible(row: dict[str, Any]) -> bool:
    images = row.get("images", [])
    return row.get("split") == "val" and row.get("view_scope") == "BOTH" and {image.get("view") for image in images if isinstance(image, dict)} == {"ANT", "POST"}


def _controlled_rows(rows: list[dict[str, Any]], patients: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    output = {"paired": [], "ant": [], "post": []}
    for row in rows:
        if not _eligible(row):
            continue
        task = row.get("task_type")
        if task not in {"vqa", "grounding", "grounded_vqa"}:
            continue
        targets = list(row.get("targets") or row.get("evidence_targets") or [])
        if task != "vqa" and {target.get("view") for target in targets} != {"ANT", "POST"}:
            continue
        evidence_id = str(row.get("evidence_id"))
        patient_id = patients.get(evidence_id)
        if patient_id is None:
            continue
        for variant in output:
            selected_images = row["images"] if variant == "paired" else [image for image in row["images"] if image.get("view") == variant.upper()]
            selected_targets = targets if variant == "paired" else [target for target in targets if target.get("view") == variant.upper()]
            expected_targets = 2 if variant == "paired" else 1
            if task != "vqa" and len(selected_targets) != expected_targets:
                continue
            output[variant].append(
                {
                    "task_id": f"{row['task_id']}:paired-view:{variant}",
                    "cohort_id": row["task_id"],
                    "variant": variant,
                    "source_task_id": row["task_id"],
                    "patient_id": patient_id,
                    "split": "val",
                    "task": task,
                    "images": selected_images,
                    "prompt": _prompt(task, str(row["label"])),
                    "target": _target(row, task, selected_targets),
                    "answer_label": row.get("answer_label"),
                    "label": row["label"],
                    "targets": selected_targets,
                    "region_level": row.get("region_level"),
                    "view_scope": variant.upper() if variant != "paired" else "BOTH",
                }
            )
    return output


def _text_perturbations(rows: list[dict[str, Any]], patients: dict[str, str]) -> list[dict[str, Any]]:
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _eligible(row) and row.get("task_type") == "vqa" and row.get("answer_label") in {"yes", "no"}:
            patient_id = patients.get(str(row.get("evidence_id")))
            if patient_id:
                by_patient[patient_id].append(row)
    records = []
    for patient_id, values in sorted(by_patient.items()):
        positive = sorted((row for row in values if row["answer_label"] == "yes"), key=lambda row: str(row["task_id"]))
        negative = sorted((row for row in values if row["answer_label"] == "no"), key=lambda row: str(row["task_id"]))
        if not positive or not negative:
            continue
        for row in positive:
            alternate = next((candidate for candidate in negative if candidate["label"] != row["label"]), None)
            if alternate is None:
                continue
            records.append(
                {
                    "perturbation_id": f"{row['task_id']}:tpt",
                    "kind": "textual",
                    "patient_id": patient_id,
                    "source_task_id": row["task_id"],
                    "baseline_label": row["label"],
                    "perturbed_label": alternate["label"],
                    "baseline_prompt": _prompt("vqa", str(row["label"])),
                    "perturbed_prompt": _prompt("vqa", str(alternate["label"])),
                    "baseline_expected_class": "yes",
                    "perturbed_expected_class": "no",
                    "images": row["images"],
                }
            )
    return records


def _visual_perturbations(rows: list[dict[str, Any]], patients: dict[str, str]) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        targets = list(row.get("targets") or row.get("evidence_targets") or [])
        evidence_id = str(row.get("evidence_id"))
        if not _eligible(row) or row.get("task_type") != "grounded_vqa" or row.get("answer_label") != "abnormal" or {target.get("view") for target in targets} != {"ANT", "POST"} or evidence_id not in patients:
            continue
        records.append(
            {
                "perturbation_id": f"{row['task_id']}:vpt",
                "kind": "visual_target_mask",
                "patient_id": patients[evidence_id],
                "source_task_id": row["task_id"],
                "prompt": _prompt("grounded_vqa", str(row["label"])),
                "images": row["images"],
                "mask_boxes": targets,
                "expected_class_after_mask": "normal",
                "control": "mask a same-area non-target box in each view",
            }
        )
    return records


def build(release: Path, output: Path) -> dict[str, int]:
    rows = read_jsonl(release / "multitask.jsonl")
    patients = _paired_patient_ids(release)
    output.mkdir(parents=True, exist_ok=False)
    variants = _controlled_rows(rows, patients)
    for variant, values in variants.items():
        write_jsonl(output / f"paired_view_{variant}.jsonl", values)
    text = _text_perturbations(rows, patients)
    visual = _visual_perturbations(rows, patients)
    write_jsonl(output / "text_perturbations.jsonl", text)
    write_jsonl(output / "visual_perturbation_specs.jsonl", visual)
    summary = {"paired": len(variants["paired"]), "ant": len(variants["ant"]), "post": len(variants["post"]), "text_perturbations": len(text), "visual_perturbation_specs": len(visual)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validation-only paired-view and perturbation cohorts.")
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.release, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
