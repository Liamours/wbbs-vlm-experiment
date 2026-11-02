"""Epoch row order for Qwen3-VL training."""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

CLASS_TASKS = ("vqa", "grounded_vqa")
POSITIVE_LABELS = frozenset({"yes", "abnormal"})


def class_balanced_epoch(rows: Sequence[Mapping[str, Any]], seed: int) -> list[Mapping[str, Any]]:
    """One epoch with as many rows as the source, where half of the VQA rows and half of the grounded VQA rows are positive.

    Positive rows (answer yes or abnormal) are drawn in repeated shuffled passes, so each is used a near-equal number of
    times; the negative rows of the same task are sampled without repetition. Rows of other tasks are kept once. The epoch is
    shuffled. The result depends only on the rows and the seed.
    """

    rng = random.Random(seed)
    chosen = [row for row in rows if row["task"] not in CLASS_TASKS]
    for task in CLASS_TASKS:
        group = [row for row in rows if row["task"] == task]
        if not group:
            continue
        positive = [row for row in group if row["answer_label"] in POSITIVE_LABELS]
        negative = [row for row in group if row["answer_label"] not in POSITIVE_LABELS]
        if not positive or not negative:
            raise ValueError(f"{task} needs rows of both classes to balance")
        wanted = len(group) // 2
        draws: list[Mapping[str, Any]] = []
        while len(draws) < wanted:
            cycle = list(positive)
            rng.shuffle(cycle)
            draws.extend(cycle[: wanted - len(draws)])
        chosen += draws + rng.sample(negative, len(group) - wanted)
    rng.shuffle(chosen)
    if len(chosen) != len(rows):
        raise ValueError("The balanced epoch changed the number of rows")
    return chosen
