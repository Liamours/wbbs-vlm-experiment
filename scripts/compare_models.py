"""Paired comparison of two models from patient-cluster bootstrap replicates written by score_locked_test.py --bootstrap-replicates.

Both files must come from the same test manifest, seed, and resample count, so replicate i resamples the same patients for both models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist, fmean, stdev

from metric.qwen import _bootstrap_interval

NORMAL = NormalDist()


def compare(first: list[float | None], second: list[float | None]) -> dict:
    diffs = [a - b for a, b in zip(first, second) if a is not None and b is not None]
    mean, sd, n = fmean(diffs), stdev(diffs), len(diffs)
    below = sum(d <= 0 for d in diffs)
    above = sum(d >= 0 for d in diffs)
    return {
        "valid_pairs": n,
        "mean_difference": round(mean, 6),
        "difference_ci_95": [_bootstrap_interval(diffs)["lower_95"], _bootstrap_interval(diffs)["upper_95"]],
        "p_bootstrap": round(min(1.0, 2 * (min(below, above) + 1) / (n + 1)), 6),
        "p_z_difference_over_sd": round(2 * (1 - NORMAL.cdf(abs(mean / sd))), 6),
        "p_paired_t_on_replicates": round(2 * (1 - NORMAL.cdf(abs(mean / (sd / n**0.5)))), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first, second = (json.loads(path.read_text(encoding="utf-8")) for path in (args.first, args.second))
    for key in ("seed", "resamples", "patients"):
        if first[key] != second[key]:
            raise SystemExit(f"{key} differs: {first[key]} vs {second[key]}")
    result = {
        "difference": f"{args.first.stem} minus {args.second.stem}",
        "seed": first["seed"],
        "resamples": first["resamples"],
        "patients": first["patients"],
        "metrics": {path: compare(values, second["replicates"][path]) for path, values in first["replicates"].items()},
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path, row in result["metrics"].items():
        print(f"{path:45} diff {row['mean_difference']:+.4f}  CI {row['difference_ci_95']}  p_boot {row['p_bootstrap']}  p_t {row['p_paired_t_on_replicates']}")


if __name__ == "__main__":
    main()
