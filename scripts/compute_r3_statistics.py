"""Phase 1 statistics for the R3 locked test: patient-cluster bootstrap intervals, paired tests, controls.

Reads the archived predictions and the no-image controls, checks that the vectorized
point estimates reproduce the archived metric files, and writes intervals, paired
differences with Holm correction, confusion counts and class counts to the output
directory of the config. A re-run continues from state.jsonl.

Run (CPU only, about a minute):
  uv run --project repositories\\wbbs-vqa_vgrounding --extra analysis python repositories\\wbbs-vqa_vgrounding\\scripts\\compute_r3_statistics.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from analysis.r3_statistics import InputMismatchError, ReproductionError, StateMismatchError, load_config, run_statistics

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "statistics" / "r3.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-only", action="store_true", help="Read the inputs and verify the point estimates against the archived metrics, then stop.")
    parser.add_argument("--restart", action="store_true", help="Move the existing results aside and recompute everything.")
    args = parser.parse_args()

    config = load_config(args.config)
    config.log_directory.mkdir(parents=True, exist_ok=True)
    log_path = config.log_directory / f"compute_r3_statistics-{datetime.now():%Y%m%d-%H%M}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
    )
    logging.getLogger("r3_statistics").addHandler(_console_handler())
    try:
        summary = run_statistics(config, restart=args.restart, check_only=args.check_only)
    except (InputMismatchError, ReproductionError, StateMismatchError) as error:
        logging.getLogger("r3_statistics").error("%s: %s", type(error).__name__, error)
        print(f"FAILED {type(error).__name__}\n{error}\nlog: {log_path}", file=sys.stderr)
        return 2
    print(json.dumps({**summary, "log": str(log_path)}, sort_keys=True))
    return 0


def _console_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    return handler


if __name__ == "__main__":
    raise SystemExit(main())
