"""Per-region recall and the question-only control decomposition for the R3 locked test.

Uses the same config as compute_r3_statistics.py. CPU only, seconds.
Run:
  uv run --project repositories\\wbbs-vqa_vgrounding --extra analysis python repositories\\wbbs-vqa_vgrounding\\scripts\\compute_r3_supplementary.py
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from analysis.r3_statistics import load_config
from analysis.r3_supplementary import run

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "statistics" / "r3.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    config.log_directory.mkdir(parents=True, exist_ok=True)
    log = config.log_directory / f"compute_r3_supplementary-{datetime.now():%Y%m%d-%H%M}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(log, encoding="utf-8"), logging.StreamHandler()])
    summary = run(config)
    print(json.dumps({**summary, "log": str(log)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
