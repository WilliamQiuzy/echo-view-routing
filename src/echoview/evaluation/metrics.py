"""The development metrics, as the project defines them.

native recordings
  A recording's view is the majority of its decoded sample labels. `native_macro_f1` is the mean F1 over the five
  views of that recording-level decision. Internal cuts are label changes inside a recording (there should be none).
streams (pair and multi)
  `segment_f1_50`     Predicted and reference segments of the same view are matched one to one when their temporal
                      IoU is at least 0.5 (best IoU first). F1 = 2 * matched / (predicted + reference), pooled over
                      all streams.
  `boundary_*`        Predicted label changes are matched one to one to reference view changes at most 0.5 s away
                      (nearest first). Recall, precision and F1 are pooled over all streams.
  `same_class_cut_rate`  Of the joins where two fragments of the same view meet, the share with a predicted change
                      within 0.5 s. A model that cuts at every splice instead of at view changes scores high here.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from .benchmark import Sequence, of_kind

HZ = 10.0
BOUNDARY_TOLERANCE_S = 0.5
IOU = 0.5
CLASSES = 5


class Segment(NamedTuple):
    start: int  # first sample
    end: int  # one past the last sample
    label: int

    @property
    def length(self) -> int:
        return self.end - self.start


def labels_to_segments(labels: np.ndarray) -> list[Segment]:
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    change = np.flatnonzero(np.diff(labels)) + 1
    starts, ends = np.concatenate([[0], change]), np.concatenate([change, [labels.size]])
    return [Segment(int(s), int(e), int(labels[s])) for s, e in zip(starts, ends)]


def boundaries_from_labels(labels: np.ndarray) -> np.ndarray:
    """Positions i where labels[i] differs from labels[i - 1]."""
    labels = np.asarray(labels)
    return np.flatnonzero(np.diff(labels)) + 1 if labels.size > 1 else np.array([], dtype=int)


def temporal_iou(a: Segment, b: Segment) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = a.length + b.length - inter
    return inter / union if union > 0 else 0.0


def matched_segments(pred: list[Segment], true: list[Segment], iou_threshold: float = IOU) -> int:
    """The number of same-view one-to-one matches at the IoU threshold, best IoU first."""
    candidates = sorted(((temporal_iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(true)
                         if p.label == t.label and temporal_iou(p, t) >= iou_threshold), reverse=True)
    used_pred, used_true, matched = set(), set(), 0
    for _, i, j in candidates:
        if i not in used_pred and j not in used_true:
            used_pred.add(i)
            used_true.add(j)
            matched += 1
    return matched


def match_boundaries(pred: np.ndarray, true: np.ndarray, tolerance: float) -> list[tuple[int, int]]:
    """One-to-one matching of predicted to reference positions within the tolerance, nearest first."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    if pred.size == 0 or true.size == 0:
        return []
    distance = np.abs(pred[:, None] - true[None, :])
    pairs = sorted((distance[i, j], i, j) for i in range(pred.size) for j in range(true.size) if distance[i, j] <= tolerance)
    used_pred, used_true, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_pred and j not in used_true:
            used_pred.add(i)
            used_true.add(j)
            out.append((i, j))
    return out


def stream_counts(sequence: Sequence, labels: np.ndarray, tolerance: float) -> dict[str, int]:
    """Segment and boundary counts of one decoded stream."""
    labels = np.asarray(labels)
    predicted, reference = labels_to_segments(labels), labels_to_segments(sequence.labels)
    cuts = boundaries_from_labels(labels)
    changes = sequence.view_changes
    same_joins = [int(j) for j in sequence.joins if j not in set(changes.tolist())]
    return {
        "seg_matched": matched_segments(predicted, reference), "seg_pred": len(predicted), "seg_true": len(reference),
        "n": int(labels.size), "correct": int((labels == sequence.labels).sum()),
        "t_true": int(changes.size), "t_matched": len(match_boundaries(cuts, changes, tolerance)), "p_pred": int(cuts.size),
        "same_join": len(same_joins), "false_split": sum(bool(np.any(np.abs(cuts - j) <= tolerance)) for j in same_joins),
    }


def _pooled(sequences: tuple[Sequence, ...], preds: Mapping[str, np.ndarray], tolerance: float) -> dict[str, int]:
    total: dict[str, int] = {}
    for sequence in sequences:
        for key, value in stream_counts(sequence, preds[sequence.id], tolerance).items():
            total[key] = total.get(key, 0) + value
    return total


def _ratio(a: float, b: float) -> float | None:
    return float(a / b) if b else None


def native_summary(sequences: tuple[Sequence, ...], preds: Mapping[str, np.ndarray], hz: float = HZ, classes: int = CLASSES) -> dict:
    """Recording-level decisions (majority of the decoded labels) and the cuts inside recordings."""
    confusion = np.zeros((classes, classes), int)
    cuts, samples, correct = [], 0, 0
    for sequence in sequences:
        labels = np.asarray(preds[sequence.id])
        truth = int(sequence.labels[0])
        confusion[truth, int(np.bincount(labels, minlength=classes).argmax())] += 1
        cuts.append(int(boundaries_from_labels(labels).size))
        samples += int(labels.size)
        correct += int((labels == truth).sum())
    tp = np.diag(confusion)
    fp, fn = confusion.sum(0) - tp, confusion.sum(1) - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    recall = np.where(confusion.sum(1) > 0, tp / np.maximum(confusion.sum(1), 1), np.nan)
    cuts_array = np.asarray(cuts)
    return {
        "native_macro_f1": float(f1.mean()), "native_per_class_recall": [None if np.isnan(r) else float(r) for r in recall],
        "native_confusion": confusion.tolist(), "native_accuracy": float(tp.sum() / max(confusion.sum(), 1)),
        "native_frame_accuracy": float(correct / samples),
        "native_internal_cuts_per_clip": float(cuts_array.mean()),
        "native_internal_cuts_per_min": float(cuts_array.sum() / (samples / (60.0 * hz))),
        "native_share_cut": float((cuts_array > 0).mean()),
    }


def evaluate(preds: Mapping[str, np.ndarray], sequences: tuple[Sequence, ...], hz: float = HZ,
             tolerance_s: float = BOUNDARY_TOLERANCE_S, classes: int = CLASSES) -> dict:
    """All development metrics of one system on one half; `preds` maps a sequence id to its decoded sample labels."""
    natives, pairs, multi = (of_kind(sequences, kind) for kind in ("native", "pairs", "multi"))
    for sequence in (*natives, *pairs, *multi):
        labels = np.asarray(preds[sequence.id])
        if labels.size and (labels.min() < 0 or labels.max() >= classes):
            raise ValueError("every sample needs one of the views; labels with uncertain samples go to evaluation/selective.py")
    m, p = _pooled(multi, preds, tolerance_s * hz), _pooled(pairs, preds, tolerance_s * hz)
    return {
        **native_summary(natives, preds, hz, classes),
        "segment_f1_50_multi": float(2 * m["seg_matched"] / (m["seg_pred"] + m["seg_true"])),
        "boundary_recall_multi": _ratio(m["t_matched"], m["t_true"]),
        "boundary_precision_multi": _ratio(m["t_matched"], m["p_pred"]),
        "boundary_f1_multi": _ratio(2 * m["t_matched"], m["p_pred"] + m["t_true"]),
        "same_class_cut_rate_multi": _ratio(m["false_split"], m["same_join"]),
        "multi_frame_accuracy": float(m["correct"] / m["n"]),
        "segment_f1_50_pairs": float(2 * p["seg_matched"] / (p["seg_pred"] + p["seg_true"])),
        "boundary_recall_pairs": _ratio(p["t_matched"], p["t_true"]),
        "counts": {"native_clips": len(natives), "multi_streams": len(multi), "pair_streams": len(pairs),
                   "multi_true_changes": int(m["t_true"]), "multi_same_joins": int(m["same_join"])},
    }
