import json
from pathlib import Path

import pytest
from PIL import Image

from training.data import load_examples, open_images


def test_grounding_uses_paligemma_coordinate_order(tmp_path: Path):
    Image.new("RGB", (4, 4)).save(tmp_path / "image.png")
    (tmp_path / "rows.jsonl").write_text(
        '{"image":"image.png","query":"cat","label":"cat","box_1024":[10,20,30,40]}\n',
        encoding="utf-8",
    )
    example = load_examples(tmp_path / "rows.jsonl", "grounding")[0]
    assert example.prompt == "detect cat"
    assert example.target == "<loc0020><loc0010><loc0040><loc0030> cat"


def test_paired_examples_compose_ant_and_post_horizontally(tmp_path: Path):
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(tmp_path / "ant.png")
    Image.new("RGB", (4, 4), color=(0, 0, 255)).save(tmp_path / "post.png")
    (tmp_path / "rows.jsonl").write_text(
        '{"images":[{"view":"ANT","image":"ant.png"},{"view":"POST","image":"post.png"}],"prompt":"question","target":"answer"}\n',
        encoding="utf-8",
    )
    example = load_examples(tmp_path / "rows.jsonl", "vqa")[0]
    image = open_images([example])[0]
    assert image.size == (8, 4)
    assert image.getpixel((0, 0)) == (255, 0, 0)
    assert image.getpixel((7, 0)) == (0, 0, 255)


def test_multitask_dispatches_prompt_by_row_task(tmp_path: Path):
    Image.new("RGB", (4, 4)).save(tmp_path / "ant.png")
    Image.new("RGB", (4, 4)).save(tmp_path / "post.png")
    rows = [
        {"task": "vqa", "images": [{"view": "ANT", "image": "ant.png"}, {"view": "POST", "image": "post.png"}], "prompt": "Is it abnormal?", "target": '{"task":"vqa","answer":"no","class":"normal"}'},
        {"task": "grounding", "images": [{"view": "ANT", "image": "ant.png"}, {"view": "POST", "image": "post.png"}], "prompt": "Locate the hotspot.", "target": '{"task":"grounding","region":"pelvis","boxes":[]}'},
        {"task": "grounded_vqa", "images": [{"view": "ANT", "image": "ant.png"}, {"view": "POST", "image": "post.png"}], "prompt": "Locate and classify.", "target": '{"task":"grounded_vqa","answer":"no","region":"pelvis","class":"normal","boxes":[]}'},
    ]
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    examples = load_examples(tmp_path / "rows.jsonl", "multitask")
    assert [example.prompt for example in examples] == [
        "<image> ANT is the left panel and POST is the right panel. Task type: vqa. Is it abnormal?",
        "<image> ANT is the left panel and POST is the right panel. Task type: grounding. Locate the hotspot.",
        "<image> ANT is the left panel and POST is the right panel. Task type: grounded_vqa. Locate and classify.",
    ]
    assert examples[0].target == rows[0]["target"]
    assert examples[1].target == rows[1]["target"]
    assert examples[2].target == rows[2]["target"]


def test_multitask_rejects_unknown_row_task(tmp_path: Path):
    Image.new("RGB", (4, 4)).save(tmp_path / "image.png")
    (tmp_path / "rows.jsonl").write_text(
        '{"task":"caption","images":[{"view":"ANT","image":"image.png"}],"prompt":"x","target":"y"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported multitask row task"):
        load_examples(tmp_path / "rows.jsonl", "multitask")


def test_multitask_accepts_single_view_rows(tmp_path: Path):
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(tmp_path / "ant.png")
    (tmp_path / "rows.jsonl").write_text(
        '{"task":"vqa","images":[{"view":"ANT","image":"ant.png"}],"prompt":"Is it abnormal?","target":"{\\"task\\":\\"vqa\\",\\"answer\\":\\"no\\"}"}\n',
        encoding="utf-8",
    )
    example = load_examples(tmp_path / "rows.jsonl", "multitask")[0]
    assert example.prompt == "<image> Task type: vqa. Is it abnormal?"
    image = open_images([example])[0]
    assert image.size == (4, 4)
