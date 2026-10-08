from __future__ import annotations

import re
from collections.abc import Sequence


def exact_match(prediction: str, answer: str | Sequence[str]) -> float:
    expected = [answer] if isinstance(answer, str) else answer
    normalized = _normalize(prediction)
    return float(any(normalized == _normalize(item) for item in expected))


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())

