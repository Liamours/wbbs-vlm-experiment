from __future__ import annotations

from typing import Sequence


IOU_THRESHOLDS: tuple[float, ...] = (0.3, 0.5, 0.75, 0.9)


def iou(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        raise ValueError("Boxes must be [x1, y1, x2, y2]")
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def iou_threshold_sweep(ious: Sequence[float], thresholds: Sequence[float] = IOU_THRESHOLDS) -> dict[str, float | None]:
    """Detection-rate-at-threshold sweep.

    This is one predicted box per query, not ranked multi-object detections, so
    it reports accuracy@threshold rather than COCO-style mAP (which needs a
    confidence-ranked precision/recall curve per class).
    """

    if not ious:
        return {f"accuracy_at_{threshold:.2f}": None for threshold in thresholds} | {"mean_average_accuracy": None}
    rates = {f"accuracy_at_{threshold:.2f}": round(sum(score >= threshold for score in ious) / len(ious), 6) for threshold in thresholds}
    rates["mean_average_accuracy"] = round(sum(rates.values()) / len(rates), 6)
    return rates

