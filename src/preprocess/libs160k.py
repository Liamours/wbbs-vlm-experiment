from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LIBS_TEMPLATE_LAYOUT = {
    1: ("right chest", "description"),
    2: ("right chest", "abnormal"),
    3: ("right chest", "normal"),
    4: ("left shoulder", "description"),
    5: ("left shoulder", "abnormal"),
    6: ("left shoulder", "normal"),
    7: ("left knee", "description"),
    8: ("left knee", "abnormal"),
    9: ("left knee", "normal"),
    10: ("right elbow", "description"),
    11: ("right elbow", "abnormal"),
    12: ("right elbow", "normal"),
    13: ("right knee", "description"),
    14: ("right knee", "abnormal"),
    15: ("right knee", "normal"),
    16: ("right shoulder", "description"),
    17: ("right shoulder", "abnormal"),
    18: ("right shoulder", "normal"),
    19: ("right ankle", "description"),
    20: ("right ankle", "abnormal"),
    21: ("right ankle", "normal"),
    22: ("pelvis", "description"),
    23: ("pelvis", "abnormal"),
    24: ("pelvis", "normal"),
    25: ("left elbow", "description"),
    26: ("left elbow", "abnormal"),
    27: ("left elbow", "normal"),
    28: ("left ankle", "description"),
    29: ("left ankle", "abnormal"),
    30: ("left ankle", "normal"),
    31: ("vertebra", "description"),
    32: ("vertebra", "abnormal"),
    33: ("vertebra", "normal"),
    34: ("head", "description"),
    35: ("head", "abnormal"),
    36: ("head", "normal"),
    37: ("left chest", "description"),
    38: ("left chest", "abnormal"),
    39: ("left chest", "normal"),
}
TEXT_FILES = (
    "libs160k-imaging-raw/LIBS-160K-EN/train/train_texts.jsonl",
    "libs160k-imaging-raw/LIBS-160K-EN/train/test_texts.jsonl",
    "libs160k-imaging-raw/LIBS-160K-EN/valid/valid_texts.jsonl",
)


def read_caption_templates(dataset_root: str | Path) -> dict[str, dict[str, dict[str, str]]]:
    root = Path(dataset_root)
    expected: dict[int, str] = {}
    for relative in TEXT_FILES:
        path = root / relative
        rows = _read_jsonl(path)
        by_id = {row.get("text_id"): row.get("text") for row in rows}
        if set(by_id) != set(LIBS_TEMPLATE_LAYOUT):
            raise ValueError(f"Expected LIBS template ids 1..39 in {path}")
        if not all(isinstance(text, str) and text.strip() for text in by_id.values()):
            raise ValueError(f"LIBS template text must be non-empty in {path}")
        for text_id, text in by_id.items():
            if text_id in expected and expected[text_id] != text:
                raise ValueError(f"LIBS template {text_id} differs across splits")
            expected[text_id] = text.strip()
    templates: dict[str, dict[str, dict[str, str]]] = {}
    for text_id, (region, kind) in LIBS_TEMPLATE_LAYOUT.items():
        templates.setdefault(region, {})[kind] = {
            "text": expected[text_id],
            "template_id": f"libs160k-en:{text_id}",
        }
    return templates


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required LIBS text source is missing: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line.rstrip().removesuffix(","))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid LIBS JSON at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(value)
    return rows
