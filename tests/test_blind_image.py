import json
from pathlib import Path

import pytest
from PIL import Image

from analysis import blind_image
from analysis.blind_image import BlindImageConfig, build_blind_manifest


def _row(task_id: str, split: str = "val", size: tuple[int, int] = (8, 32), views=("ANT", "POST")) -> dict:
    return {
        "answer_label": "no",
        "images": [{"image": f"scans/{task_id.replace(':', '_')}_{view}.jpg", "image_size": list(size), "view": view} for view in views],
        "label": "right chest",
        "patient_id": "1",
        "prompt": "Is the hotspot labelled as a bone metastasis?",
        "split": split,
        "target": "{\"task\":\"vqa\"}",
        "targets": [{"bbox": [1, 2, 3, 4], "view": "ANT"}],
        "task": "vqa",
        "task_id": task_id,
    }


def _write_source(directory: Path, rows: list[dict], name: str = "multitask_val.jsonl", make_images: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    if make_images:
        (directory / "scans").mkdir(exist_ok=True)
        for row in rows:
            for image in row["images"]:
                Image.new("RGB", tuple(image["image_size"]), color=(200, 200, 200)).save(directory / image["image"])
    return path


def _config(tmp_path: Path, source: Path, **overrides) -> BlindImageConfig:
    values = dict(split="val", source_manifest=source, expected_rows=3, output_dir=tmp_path / "out", blank_fill=(0, 0, 0), smoke_rows=2, seed=42)
    values.update(overrides)
    return BlindImageConfig(**values)


def _rows() -> list[dict]:
    return [_row("a:1"), _row("a:2", views=("ANT",)), _row("a:3")]


def test_manifest_keeps_ids_prompts_and_labels_and_swaps_images(tmp_path):
    rows = _rows()
    source = _write_source(tmp_path / "export", rows)
    provenance = build_blind_manifest(_config(tmp_path, source))
    blind = [json.loads(line) for line in (tmp_path / "out/manifests/multitask_val_blind.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["task_id"] for row in blind] == [row["task_id"] for row in rows]
    for original, changed in zip(rows, blind):
        assert {key: value for key, value in changed.items() if key != "images"} == {key: value for key, value in original.items() if key != "images"}
        assert [image["view"] for image in changed["images"]] == [image["view"] for image in original["images"]]
        for image in changed["images"]:
            resolved = (tmp_path / "out/manifests" / image["image"]).resolve()
            with Image.open(resolved) as blank:
                assert list(blank.size) == image["image_size"]
                assert blank.convert("RGB").getcolors(maxcolors=1)[0][1] == (0, 0, 0)
    assert provenance["splits_in_manifest"] == ["val"]
    assert provenance["rows"] == 3 and provenance["smoke_rows"] == 2


def test_only_the_val_split_is_accepted(tmp_path):
    rows = [_row("a:1", split="test"), _row("a:2", split="test"), _row("a:3", split="test")]
    source = _write_source(tmp_path / "export", rows)
    with pytest.raises(ValueError, match="only the 'val' split"):
        build_blind_manifest(_config(tmp_path, source))
    with pytest.raises(ValueError, match="validation split only"):
        build_blind_manifest(_config(tmp_path, source, split="test"))


def test_a_manifest_named_like_the_test_split_is_refused(tmp_path):
    source = _write_source(tmp_path / "export", _rows(), name="multitask_test.jsonl")
    with pytest.raises(ValueError, match="locked test split"):
        build_blind_manifest(_config(tmp_path, source))


def test_output_under_datasets_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(blind_image, "PROJECT_ROOT", tmp_path)
    source = _write_source(tmp_path / "export", _rows())
    with pytest.raises(ValueError, match="datasets"):
        build_blind_manifest(_config(tmp_path, source, output_dir=tmp_path / "datasets/blind"))


def test_a_wrong_image_size_is_caught(tmp_path):
    rows = _rows()
    source = _write_source(tmp_path / "export", rows)
    Image.new("RGB", (9, 32)).save(tmp_path / "export" / rows[0]["images"][0]["image"])
    with pytest.raises(ValueError, match="image problems"):
        build_blind_manifest(_config(tmp_path, source))


def test_row_count_must_match(tmp_path):
    source = _write_source(tmp_path / "export", _rows())
    with pytest.raises(ValueError, match="expected 4"):
        build_blind_manifest(_config(tmp_path, source, expected_rows=4))


def test_second_build_is_identical_and_a_changed_source_is_refused(tmp_path):
    source = _write_source(tmp_path / "export", _rows())
    first = build_blind_manifest(_config(tmp_path, source))
    second = build_blind_manifest(_config(tmp_path, source))
    assert first["manifest_sha256"] == second["manifest_sha256"]
    changed = _rows()
    changed[0]["prompt"] = "A different question?"
    source = _write_source(tmp_path / "export", changed)
    with pytest.raises(FileExistsError):
        build_blind_manifest(_config(tmp_path, source))
