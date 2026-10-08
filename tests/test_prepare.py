import base64
import io
import json
from pathlib import Path

from PIL import Image

from analysis.audit import run_preparation


def _jpeg_payload(value: int) -> str:
    image = Image.new("L", (16, 32), value)
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _write_libs_split(root: Path, folder: str, split: str, image_value: int) -> None:
    target = root / "LIBS-160K-EN" / folder
    target.mkdir(parents=True, exist_ok=True)
    rows = [
        {"text_id": 1, "text": "bone scan head", "image_ids": [1]},
        {"text_id": 2, "text": "tracer uptake head", "image_ids": [1]},
    ]
    (target / f"{split}_texts.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (target / f"{split}_imgs.tsv").write_text(f"1\t{_jpeg_payload(image_value)}\n", encoding="utf-8")


def _write_whole(root: Path, view: str, label: str, name: str, value: int) -> None:
    path = root / view / label
    path.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 64), value).save(path / name, format="JPEG")


def test_preparation_writes_audits_and_grouped_manifests(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "derived"
    _write_libs_split(raw, "train", "train", 10)
    _write_libs_split(raw, "train", "test", 20)
    _write_libs_split(raw, "valid", "valid", 30)
    _write_whole(raw, "wholeANT", "Normal", "1.jpg", 10)
    _write_whole(raw, "wholeANT", "Abnormal", "2.jpg", 20)
    _write_whole(raw, "wholePOST", "Normal", "1.jpg", 30)
    _write_whole(raw, "wholePOST", "Abnormal", "2.jpg", 40)

    summary = run_preparation(raw, output, libs_image_sample_size=1, skip_unsupervised=True)

    assert summary["libs"]["multi_label_structure"]["image_label_cardinality"] == {"2": 1}
    assert summary["whole"]["pair_audit"]["matched_filenames"] == 2
    assert (output / "analysis" / "summary.json").exists()
    assert (output / "analysis" / "libs_text_labels.csv").exists()
    assert (output / "manifests" / "wholeANT.jsonl").exists()
    assert (output / "reports" / "dataset_audit.md").exists()
