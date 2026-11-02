from __future__ import annotations

import json

import pytest
from PIL import Image

from analysis.blind_manifests import build


def test_blind_copy_keeps_every_field_but_the_image_and_is_black(tmp_path):
    source = tmp_path / "multitask_train.jsonl"
    rows = [{"task_id": f"t{i}", "task": "vqa", "prompt": "p", "target": "x", "images": [{"image": "a.jpg", "image_size": [8, 16], "view": "ANT"}]} for i in range(3)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    record = build(source, "blind_train", 3, output=tmp_path / "out")
    copied = [json.loads(line) for line in (tmp_path / "out" / "manifests" / "blind_train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["task_id"] for r in copied] == ["t0", "t1", "t2"] and all(r["target"] == "x" for r in copied)
    assert copied[0]["images"][0]["image"] == "../images/blank_8x16.png"
    with Image.open(tmp_path / "out" / "images" / "blank_8x16.png") as blank:
        assert blank.size == (8, 16) and blank.convert("RGB").getcolors() == [(128, (0, 0, 0))]
    assert record["rows"] == 3


def test_blind_copy_refuses_a_wrong_row_count(tmp_path):
    source = tmp_path / "x.jsonl"
    source.write_text(json.dumps({"task_id": "a", "images": [{"image": "a", "image_size": [2, 2], "view": "ANT"}]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build(source, "n", 2, output=tmp_path / "out")
