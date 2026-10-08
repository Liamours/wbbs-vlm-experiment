"""Rebuild R0 to R3, the Qwen export, and the full local dataset from datasets/sources, then compare each with its frozen copy.

Reports (.json) record the build path and are skipped in the comparison; manifests, images, and the contact sheet must match byte for byte
once the image-root prefix inside manifests is normalized.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from project_paths import DATASETS, PROJECT_ROOT, SOURCES  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
CONFIGS = SCRIPTS.parent / "configs/dataset"
IMAGE_PREFIX = re.compile(rb'(?:\.\./)+(?:[^"\\]*?/)?bs80k-imaging-raw/')
COMPARED = {".jsonl", ".png"}


def steps(stage: Path, frozen_export: Path) -> list[tuple[str, list[str]]]:
    py = sys.executable
    release = [py, "-m", "preprocess.release", "--dataset-root", str(SOURCES)]
    return [
        ("labels", [py, str(SCRIPTS / "import_reformatted_labels.py")]),
        ("r0", release + ["--output", str(stage / "r0"), "--policy", str(CONFIGS / "r0.json")]),
        ("r1", release + ["--output", str(stage / "r1"), "--policy", str(CONFIGS / "r1.json")]),
        ("r2", [py, str(SCRIPTS / "build_r2_diversified.py"), "--input-release", str(stage / "r1"), "--output-dir", str(stage / "r2")]),
        ("r3", [py, str(SCRIPTS / "build_r3_compact.py"), "--input-release", str(stage / "r2"), "--output-dir", str(stage / "r3")]),
        ("export", [py, "-m", "postprocess.qwen", "--release", str(stage / "r3"), "--output", str(stage / "export"), "--schema", "multitask_json", "--image-root", str(SOURCES / "bs80k-imaging-raw")]),
        ("full", [py, str(SCRIPTS / "build_full_local_dataset.py"), "--release", str(stage / "r3"), "--image-root", str(SOURCES / "bs80k-imaging-raw"), "--training-export", str(frozen_export), "--output", str(stage / "scintiground-38k")]),
    ]


def normalized(path: Path) -> bytes:
    return IMAGE_PREFIX.sub(b"IMG/", path.read_bytes()).replace(b"\r\n", b"\n")


def compare(built: Path, frozen: Path) -> list[str]:
    problems = []
    for file in sorted(frozen.rglob("*")):
        if not file.is_file() or file.suffix not in COMPARED:
            continue
        name = file.relative_to(frozen)
        other = built / name
        if not other.is_file():
            problems.append(f"{built.name}/{name}: missing")
        elif file.suffix == ".png" and file.read_bytes() != other.read_bytes() or file.suffix == ".jsonl" and normalized(file) != normalized(other):
            problems.append(f"{built.name}/{name}: differs")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=PROJECT_ROOT / "temp/regenerated")
    parser.add_argument("--frozen-export", type=Path, default=DATASETS / "model_exports/qwen3_json/r3")
    args = parser.parse_args()
    if args.staging.exists():
        raise SystemExit(f"Staging already exists: {args.staging}")
    for label, command in tqdm(steps(args.staging, args.frozen_export), desc="regenerate", unit="step"):
        result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")
        if result.returncode:
            raise SystemExit(f"{label} failed:\n{result.stderr[-2000:]}")

    problems = []
    for name in ("r0", "r1", "r2", "r3"):
        problems += compare(args.staging / name, DATASETS / "releases" / name)
    problems += compare(args.staging / "export", args.frozen_export)
    problems += compare(args.staging / "scintiground-38k", DATASETS / "scintiground-38k")
    frozen_sums = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in (DATASETS / "scintiground-38k/SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()}
    built_sums = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in (args.staging / "scintiground-38k/SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()}
    problems += [f"scintiground-38k/{name}: checksum differs" for name, digest in frozen_sums.items() if not name.endswith(".json") and built_sums.get(name) != digest]
    print("\n".join(problems) if problems else "all releases, the export, and the full dataset match their frozen copies")
    if problems:
        raise SystemExit(1)
    shutil.rmtree(args.staging)


if __name__ == "__main__":
    main()
