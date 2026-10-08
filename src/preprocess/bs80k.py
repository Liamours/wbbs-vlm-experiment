from __future__ import annotations

import csv
import os
import xml.etree.ElementTree as xml
from collections import defaultdict
from pathlib import Path
from typing import Any


LOW_PRECISION_COMPONENTS = {"shoL", "shoR"}
KNOWN_ARTIFACT_COMPONENTS = {"elbowLPOST", "elbowRPOST"}
REGION_NAMES = {
    "ankleL": "left ankle",
    "ankleR": "right ankle",
    "chestL": "left chest",
    "chestR": "right chest",
    "elbowL": "left elbow",
    "elbowR": "right elbow",
    "head": "head",
    "kneeL": "left knee",
    "kneeR": "right knee",
    "pelvis": "pelvis",
    "shoL": "left shoulder",
    "shoR": "right shoulder",
    "vertbra": "vertebra",
}


def build_region_evidence(dataset_root: str | Path, manifest_dir: str | Path) -> list[dict[str, Any]]:
    return build_bs80k_evidence(dataset_root, manifest_dir)[0]


def build_bs80k_evidence(dataset_root: str | Path, manifest_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(dataset_root).resolve()
    region_rows = _read_csv(root / "bs80k-bone_region-bb/bounding_boxes.csv")
    image_sizes, niduses = read_nidus_annotations(root)
    whole_body_flags = {(row["patient_id"], row["view"]): row["flags"] for row in read_whole_body_boxes(root)}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in region_rows:
        component, view = _component_and_view(row["component"])
        patient_id = str(row["id"])
        target = _box_from_xywh(row, f"region box for {patient_id}:{row['component']}")
        image_path = root / "bs80k-imaging-raw" / f"wholeBody{view}" / f"{patient_id}.jpg"
        image_size = image_sizes.get((patient_id, view), [256, 1024])
        flags = {
            "low_precision_region": component in LOW_PRECISION_COMPONENTS,
            "known_artifact": row["component"] in KNOWN_ARTIFACT_COMPONENTS,
            "missing_image": not image_path.is_file(),
            **whole_body_flags.get((patient_id, view), {}),
        }
        record = {
            "record_id": f"bs80k:region:{patient_id}:{row['component']}",
            "patient_id": patient_id,
            "image": _relative_image(image_path, Path(manifest_dir)),
            "image_size": image_size,
            "view": view,
            "target": {"kind": "region", "name": _region_name(component), "bbox": target},
            "diagnosis": row["diagnosis"].strip().lower(),
            "source_label": row["label"].strip(),
            "hotspots": [],
            "qa": [],
            "grounding": [],
            "flags": flags,
            "duplicate_of_patient_id": _optional(row.get("duplicate_of_patient_id")),
            "duplicate_of_sibling_component": _optional(row.get("duplicate_of_sibling_component")),
            "provenance": {
                "source": "bs80k-bone_region-bb/bounding_boxes.csv",
                "source_row": {key: row[key] for key in ("id", "component", "label", "diagnosis", "match_score") if key in row},
                "component": row["component"],
            },
        }
        grouped[(patient_id, view)].append(record)

    metastases: list[dict[str, Any]] = []
    for key, records in grouped.items():
        for index, nidus in enumerate(niduses.get(key, [])):
            containing = [record for record in records if _contains(record["target"]["bbox"], nidus["bbox"])]
            metastasis = {
                "metastasis_id": f"bs80k:nidus:{key[0]}:{key[1]}:{index}",
                "patient_id": key[0],
                "view": key[1],
                "image_size": records[0]["image_size"],
                **nidus,
                "candidate_record_ids": [record["record_id"] for record in containing],
            }
            if len(containing) == 1:
                containing[0]["hotspots"].append({**nidus, "assignment": "unique_region"})
                metastasis["assignment"] = "unique_region"
            elif len(containing) > 1:
                for record in containing:
                    record["flags"]["ambiguous_hotspot_assignment"] = True
                    record["hotspots"].append({**nidus, "assignment": "ambiguous_regions"})
                metastasis["assignment"] = "ambiguous_regions"
            else:
                metastasis["assignment"] = "unassigned"
            metastases.append(metastasis)
    return [record for records in grouped.values() for record in records], metastases


def read_whole_body_boxes(dataset_root: str | Path) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    rows = _read_csv(root / "bs80k-wholebody-bb/bounding_boxes.csv")
    result = []
    for row in rows:
        result.append(
            {
                "patient_id": str(row["id"]),
                "view": row["view"],
                "bbox": _box_from_xywh(row, f"whole-body box for {row['id']}:{row['view']}"),
                "flags": {key: _bool(row[key]) for key in ("outlier", "likely_corrupt_image") if key in row},
            }
        )
    return result


def read_nidus_annotations(dataset_root: str | Path) -> tuple[dict[tuple[str, str], list[int]], dict[tuple[str, str], list[dict[str, Any]]]]:
    root = Path(dataset_root)
    image_sizes: dict[tuple[str, str], list[int]] = {}
    niduses: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for directory, view in (("ant", "ANT"), ("post", "POST")):
        for path in sorted((root / "bs80k-lesion-bb" / directory).glob("*.xml")):
            patient_id = path.stem
            parsed = _parse_nidus_xml(path)
            image_sizes[(patient_id, view)] = parsed["image_size"]
            niduses[(patient_id, view)] = parsed["hotspots"]
    return image_sizes, niduses


def _parse_nidus_xml(path: Path) -> dict[str, Any]:
    root = xml.parse(path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing image size in {path}")
    image_size = [int(_xml_text(size, "width", path)), int(_xml_text(size, "height", path))]
    hotspots = []
    for index, item in enumerate(root.findall("object")):
        box = item.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bndbox in {path} object {index}")
        bbox = [
            int(_xml_text(box, "xmin", path)),
            int(_xml_text(box, "ymin", path)),
            int(_xml_text(box, "xmax", path)),
            int(_xml_text(box, "ymax", path)),
        ]
        if not (0 <= bbox[0] < bbox[2] <= image_size[0] and 0 <= bbox[1] < bbox[3] <= image_size[1]):
            raise ValueError(f"Invalid nidus box in {path} object {index}")
        hotspots.append({"bbox": bbox, "label": _xml_text(item, "name", path), "source_id": f"{path.stem}:{index}"})
    return {"image_size": image_size, "hotspots": hotspots}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required BS80K source is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _component_and_view(value: str) -> tuple[str, str]:
    for view in ("ANT", "POST"):
        if value.endswith(view):
            return value[: -len(view)], view
    raise ValueError(f"Unknown BS80K component view: {value}")


def _region_name(component: str) -> str:
    if component not in REGION_NAMES:
        raise ValueError(f"Unknown BS80K region component: {component}")
    return REGION_NAMES[component]


def _box_from_xywh(row: dict[str, str], field: str) -> list[int]:
    try:
        x, y, width, height = (int(float(row[key])) for key in ("x", "y", "width", "height"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field}") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid {field}: width and height must be positive")
    return [x, y, x + width, y + height]


def _relative_image(image: Path, manifest_dir: Path) -> str:
    return Path(os.path.relpath(image, manifest_dir)).as_posix()


def _contains(outer: list[int], inner: list[int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def _optional(value: str | None) -> str | None:
    return value.strip() if value and value.strip() else None


def _bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def _xml_text(parent: xml.Element, tag: str, path: Path) -> str:
    value = parent.findtext(tag)
    if not value or not value.strip():
        raise ValueError(f"Missing {tag} in {path}")
    return value.strip()
