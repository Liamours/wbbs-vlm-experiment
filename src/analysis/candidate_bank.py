"""Answer-text bank and candidate prefixes for the score-based AUC.

The adapters write `{"task":"vqa","answer":"<sentence>","class":"<label>"}`. The sentence commits to the label before the
class field, so the probability of the class token alone is degenerate. The probability of each label is instead the sum
over every answer sentence the training split uses for that label, each scored as the text of the output up to and
including the class value. The bank is read from the training split only; a sentence is stored with its region phrase
replaced by a slot, so one entry covers every region of that region level.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable, Mapping

from preprocess.templates import REGION_REFERENCES

SLOT = "{region}"
CLASS_VALUES = {"vqa": ("no", "yes"), "grounded_vqa": ("normal", "abnormal")}
Key = tuple[str, str, str]


def region_reference(label: str) -> str:
    return REGION_REFERENCES.get(label, label)


def class_values(task: str) -> tuple[str, str]:
    """(negative, positive) class value of a task; the positive class is yes or abnormal."""

    return CLASS_VALUES[task]


def normalise(answer: str, label: str) -> str:
    reference = region_reference(label)
    return answer.replace(reference, SLOT) if reference in answer else answer


def build_bank(train_rows: Iterable[Mapping[str, Any]]) -> dict[Key, Counter]:
    """Counts of normalised answer sentences by (task, region level, class value), from the training split."""

    bank: dict[Key, Counter] = {}
    for row in train_rows:
        if row["task"] not in CLASS_VALUES:
            continue
        target = json.loads(row["target"])
        key = (row["task"], row["region_level"], target["class"])
        bank.setdefault(key, Counter())[normalise(target["answer"], row["label"])] += 1
    return bank


def prefix_text(skeleton: Mapping[str, Any], answer: str, class_value: str) -> str:
    """The output text up to and including the class value, without its closing quote: the model writes the quote and the next character as one token."""

    text = json.dumps({**skeleton, "answer": answer, "class": class_value}, separators=(",", ":"))
    marker = f'"class":"{class_value}'
    return text[: text.index(marker) + len(marker)]


def candidates(row: Mapping[str, Any], bank: Mapping[Key, Counter], max_templates: int | None = None) -> list[dict[str, Any]]:
    """Every (class value, answer sentence) candidate of one instance with its output prefix.

    The skeleton of the instance's own target supplies the key order and the region; its answer and class are replaced,
    and the boxes lie after the class value, so no label information of the instance enters the prefix.
    """

    skeleton = json.loads(row["target"])
    reference = region_reference(row["label"])
    result = []
    for class_value in class_values(row["task"]):
        key = (row["task"], row["region_level"], class_value)
        if key not in bank:
            raise KeyError(f"No training answers for {key}")
        for template, _ in bank[key].most_common(max_templates):
            answer = template.replace(SLOT, reference)
            result.append({"class": class_value, "answer": answer, "prefix": prefix_text(skeleton, answer, class_value)})
    return result


def coverage(bank: Mapping[Key, Counter], max_templates: int | None) -> dict[str, float]:
    """Share of the training answers of each (task, level, class value) that the kept templates cover."""

    return {"/".join(key): sum(count for _, count in counter.most_common(max_templates)) / sum(counter.values()) for key, counter in sorted(bank.items())}


def check_bank(train_rows: Iterable[Mapping[str, Any]]) -> int:
    """Rebuild the prefix of every training row from its slot form and compare it with the row's own target."""

    checked, problems = 0, []
    for row in train_rows:
        if row["task"] not in CLASS_VALUES:
            continue
        target = json.loads(row["target"])
        rebuilt = prefix_text(target, normalise(target["answer"], row["label"]).replace(SLOT, region_reference(row["label"])), target["class"])
        if rebuilt != prefix_text(target, target["answer"], target["class"]):
            problems.append(row["task_id"])
        checked += 1
    if problems:
        raise ValueError(f"{len(problems)} of {checked} training prefixes do not rebuild from the bank, first: {problems[:3]}")
    return checked
