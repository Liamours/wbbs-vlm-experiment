from __future__ import annotations

from .canonical import validate_box


VIEW_TOKENS = {"ANT": "ANT", "POST": "PST"}


def format_bbx(bbox: list[int | float]) -> str:
    coordinates = ",".join(_coordinate(value) for value in validate_box(bbox))
    return f"<BBX>{coordinates}</BBX>"


def format_view_bbx(targets: list[dict[str, object]]) -> str:
    parts = []
    for target in targets:
        view = target.get("view")
        bbox = target.get("bbox")
        if view not in VIEW_TOKENS or not isinstance(bbox, list):
            raise ValueError("Each target requires ANT or POST view and a bbox")
        token = VIEW_TOKENS[view]
        parts.append(f"<{token}>{format_bbx(bbox)}</{token}>")
    return "".join(parts)


def format_grounded_target(region: str, diagnosis: str, targets: list[dict[str, object]] | list[int | float]) -> str:
    region_value = region.replace(" ", "_")
    coordinates = format_view_bbx(targets) if targets and isinstance(targets[0], dict) else format_bbx(targets)
    return f"<REG>{region_value}</REG><CLS>{diagnosis}</CLS>{coordinates}"


def _coordinate(value: int | float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)
