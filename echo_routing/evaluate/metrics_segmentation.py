"""Segment F1: same-class one-to-one temporal-IoU matching."""
from __future__ import annotations

import numpy as np

from echo_routing.temporal.segments import Segment


def temporal_iou(a: Segment, b: Segment) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = a.length + b.length - inter
    return inter / union if union > 0 else 0.0


def segment_f1(pred: list[Segment], true: list[Segment], iou_threshold: float) -> dict:
    """Same-class one-to-one matching at a temporal IoU threshold."""
    cands = sorted(((temporal_iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(true)
                    if p.label == t.label and temporal_iou(p, t) >= iou_threshold), reverse=True)
    used_p, used_t, m = set(), set(), 0
    for _, i, j in cands:
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j); m += 1
    precision = m / len(pred) if pred else (1.0 if not true else 0.0)
    recall = m / len(true) if true else (1.0 if not pred else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"iou": iou_threshold, "precision": precision, "recall": recall, "f1": f1, "matched": m}
