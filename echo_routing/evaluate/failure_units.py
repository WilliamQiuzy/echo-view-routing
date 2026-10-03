"""Per-unit failure analysis of a segmentation (round 4, configs/experiments/failure_analysis_v1.yaml). Diagnostic only.

A unit is a reference segment of a multi-fragment stream: a run of fragments with the same class (a same-class join is
not a boundary). A unit is wrong for a system when no predicted segment of its class is matched to it by the
one-to-one matching of segment F1@50 (same class, IoU >= 0.5, greedy by IoU). Predicted label changes are matched
one-to-one to reference changes within `near` samples, with as many matches as possible and then the smallest total
distance. A wrong unit gets the first cause that applies:
- label: a predicted segment overlaps the unit at IoU >= 0.5 but has another class;
- missed_cut: a reference change at one of the unit's edges has no matched predicted change;
- extra_cut: an unmatched predicted change lies inside the unit;
- boundary_shift: everything else (the edge changes are matched, at least one beyond `hit` samples, and nothing cuts
  the interior).
Unmatched predicted segments are spurious; each is hosted by the unit it overlaps most.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from echo_routing.evaluate.metrics_segmentation import temporal_iou
from echo_routing.temporal.segments import Segment, boundaries_from_labels, labels_to_segments

CAUSES = ("label", "missed_cut", "extra_cut", "boundary_shift")
NUM_CLASSES = 5


@dataclass(frozen=True)
class Unit:
    stream: str
    index: int
    start: int
    end: int
    label: int
    left: int | None                                  # class of the previous unit; None at the stream start
    right: int | None                                 # class of the next unit; None at the stream end
    sources: tuple[tuple[str, str, int, int], ...]    # (video_id, variant, start, end) in source-clip samples

    @property
    def length(self) -> int:
        return self.end - self.start


def session_of(video_id: str) -> str:
    """The protocol's split unit: the clip id without its last '_k'."""
    return video_id.rsplit("_", 1)[0]


def stream_units(recipe) -> list[Unit]:
    """Units of one recipe in stream order."""
    runs, pos = [], 0
    for fr in recipe.fragments:
        n = fr.end - fr.start; src = (fr.video_id, fr.variant, int(fr.start), int(fr.end))
        if runs and runs[-1][2] == int(fr.label):
            runs[-1][1] += n; runs[-1][3].append(src)
        else:
            runs.append([pos, pos + n, int(fr.label), [src]])
        pos += n
    return [Unit(recipe.recipe_id, i, s, e, lab, runs[i - 1][2] if i else None, runs[i + 1][2] if i + 1 < len(runs) else None,
                 tuple(src)) for i, (s, e, lab, src) in enumerate(runs)]


def match_segments(pred: list[Segment], true: list[Segment], iou: float = 0.5) -> dict[int, int]:
    """true index -> predicted index under the matching of segment_f1 (same class, IoU >= iou, greedy by IoU)."""
    cands = sorted(((temporal_iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(true)
                    if p.label == t.label and temporal_iou(p, t) >= iou), reverse=True)
    used, out = set(), {}
    for _, i, j in cands:
        if i not in used and j not in out:
            used.add(i); out[j] = i
    return out


def match_changes(pred: np.ndarray, true: np.ndarray, near: int) -> dict[int, int]:
    """true change -> predicted change: one-to-one within `near` samples, most matches, then least total distance."""
    pred, true = np.asarray(pred, int), np.asarray(true, int)
    if not pred.size or not true.size:
        return {}
    dist = np.abs(true[:, None] - pred[None, :]).astype(float)
    big = float(near + 1) * (true.size + pred.size + 1)          # any invalid pair costs more than all valid ones
    rows, cols = linear_sum_assignment(np.where(dist <= near, dist, big))
    return {int(true[r]): int(pred[c]) for r, c in zip(rows, cols) if dist[r, c] <= near}


def segment_counts(pred: np.ndarray, units: list[Unit], iou: float = 0.5) -> tuple[int, int, int]:
    """(matched, predicted segments, reference segments) of one stream, as pooled by segment F1@50."""
    ps = labels_to_segments(np.asarray(pred, int))
    return len(match_segments(ps, [Segment(u.start, u.end, u.label) for u in units], iou)), len(ps), len(units)


def unit_outcomes(pred: np.ndarray, units: list[Unit], iou: float = 0.5, hit: int = 5, near: int = 10) -> tuple[list[dict], list[dict]]:
    """Per-unit outcome rows and the spurious predicted segments of one stream (`hit` only feeds edges_within_hit).

    The label rule looks at the best-overlapping predicted segment. For a wrong unit no segment of its own class
    reaches IoU 0.5 (it would be matched), so any segment that does reach it has another class and is the best."""
    pred = np.asarray(pred, int)
    ps = labels_to_segments(pred); ts = [Segment(u.start, u.end, u.label) for u in units]
    matched = match_segments(ps, ts, iou)
    pc = boundaries_from_labels(pred); cm = match_changes(pc, np.array([u.start for u in units[1:]], int), near)
    used = set(cm.values()); rows = []
    for j, u in enumerate(units):
        inside = pred[u.start:u.end]
        best_iou, best_label = max(((temporal_iou(p, ts[j]), p.label) for p in ps if p.end > u.start and p.start < u.end),
                                   key=lambda x: (x[0], x[1] == u.label))
        dist = [abs(cm[e] - e) if e in cm else None for e, nb in ((u.start, u.left), (u.end, u.right)) if nb is not None]
        extra = [int(p) for p in pc if u.start < p < u.end and int(p) not in used]
        if j in matched:
            cause = None
        elif best_iou >= iou and best_label != u.label:
            cause = "label"
        elif any(d is None for d in dist):
            cause = "missed_cut"
        elif extra:
            cause = "extra_cut"
        else:
            cause = "boundary_shift"
        rows.append({"index": j, "matched": j in matched, "cause": cause, "best_iou": float(best_iou), "best_label": int(best_label),
                     "frac_label": float((inside == u.label).mean()), "majority": int(np.bincount(inside, minlength=NUM_CLASSES).argmax()),
                     "edge_dist": dist, "edges_within_hit": sum(d is not None and d <= hit for d in dist), "extra_cuts": len(extra)})
    hits = set(matched.values()); spurious = []
    for i, p in enumerate(ps):
        if i not in hits:
            _, neg = max((min(p.end, u.end) - max(p.start, u.start), -j) for j, u in enumerate(units)); host = units[-neg]
            spurious.append({"start": p.start, "end": p.end, "label": p.label, "host": host.index,
                             "inside": host.start <= p.start and p.end <= host.end})
    return rows, spurious


def repair(pred: np.ndarray, ref: np.ndarray, units: list[Unit], which: list[int], mode: str,
           scores: np.ndarray | None = None) -> np.ndarray:
    """A repaired copy of one stream's labels (diagnostic only). ideal: each chosen unit takes its class on its interval;
    segmentation: its interval takes the argmax of `scores` averaged over it; class: every predicted segment that
    overlaps it takes the majority reference class over its own span, so the predicted intervals are kept."""
    pred = np.asarray(pred, int); ref = np.asarray(ref, int); out = pred.copy()
    for j in which:
        u = units[j]
        if mode == "ideal":
            out[u.start:u.end] = u.label
        elif mode == "segmentation":
            out[u.start:u.end] = int(np.asarray(scores[u.start:u.end], float).mean(0).argmax())
        elif mode == "class":
            for p in labels_to_segments(pred):
                if p.end > u.start and p.start < u.end:
                    out[p.start:p.end] = int(np.bincount(ref[p.start:p.end], minlength=NUM_CLASSES).argmax())
        else:
            raise ValueError(f"unknown repair mode {mode!r}")
    return out
