"""Descriptive stream metrics for decoded labels: where the errors are, not only how many.

A stream is a sequence of 10 Hz samples with a reference label per sample, a source piece per sample (consecutive
samples from one recording) and a predicted label per sample. Every function takes plain integer arrays.

  A1  midpoint correctness in two directions: per source piece (is the middle of every recording read right?) and
      per predicted segment (does the middle of every piece the system returns carry its label?)
  A3  every wrong sample put in one of four classes by the two midpoints: edge (both midpoints right: the cut is a
      little off), spurious (an extra segment), absorbed (a piece swallowed by its neighbour), misread (a whole piece
      given another view)
  A4  signed offsets of matched cuts (predicted minus reference, at most 1 s apart)
  A5  purity and recovery per view, plain and with 0.5 s removed at both ends of every predicted segment
  A6  segment F1 at overlaps 0.10, 0.25 and 0.50 and the Edit score, as defined by MS-TCN (Farha and Gall 2019)

The sampling-grid floor is the error of a perfect model that decides every `stride` samples and fills by the nearest
centre. `summarise` needs SciPy (pip install scipy).
"""
from __future__ import annotations

import numpy as np

HZ = 10.0
TRIM = 5  # samples removed at each end of a predicted segment in the trimmed purity (0.5 s)
NEAR = 10  # a predicted cut can match a reference cut at most this many samples away (1 s)
CATEGORIES = ("edge", "spurious", "absorbed", "misread")
OVERLAPS = (0.10, 0.25, 0.50)


def runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    """(start, end, value) of every run of equal values, end-exclusive."""
    values = np.asarray(values)
    if values.size == 0:
        return []
    cuts = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts, ends = np.concatenate([[0], cuts]), np.concatenate([cuts, [values.size]])
    return [(int(a), int(b), int(values[a])) for a, b in zip(starts, ends)]


def midpoint(start: int, end: int) -> int:
    """The middle sample of [start, end); the earlier one when the length is even."""
    return start + (end - start - 1) // 2


def grid_floor_labels(truth: np.ndarray, stride: int = 5) -> np.ndarray:
    """Labels of a perfect model that is read only at the window centres and filled by the nearest centre."""
    n = len(truth)
    centres = np.arange(stride // 2, n, stride)
    if centres.size == 0 or centres[-1] < n - 1 - stride // 2:
        centres = np.append(centres, n - 1)
    nearest = np.abs(np.arange(n)[:, None] - centres[None, :]).argmin(1)
    return np.asarray(truth)[centres[nearest]]


def _midpoint_tables(truth: np.ndarray, piece: np.ndarray, pred: np.ndarray, classes: int) -> tuple:
    """Per-sample flags (its piece's midpoint is read right; its segment's midpoint carries the label) and the tables."""
    n = len(truth)
    piece_ok, segment_ok = np.zeros(n, bool), np.zeros(n, bool)
    piece_mid, segment_mid = np.zeros((classes, classes), np.int64), np.zeros((classes, classes), np.int64)
    pieces, segments = runs(piece), runs(pred)
    for a, b, _ in pieces:
        m = midpoint(a, b)
        piece_ok[a:b] = pred[m] == truth[m]
        piece_mid[truth[m], pred[m]] += 1
    for a, b, label in segments:
        m = midpoint(a, b)
        segment_ok[a:b] = truth[m] == label
        segment_mid[truth[m], label] += 1
    return piece_ok, segment_ok, piece_mid, segment_mid, pieces, segments


def stream_counts(truth: np.ndarray, piece: np.ndarray, pred: np.ndarray, classes: int = 5) -> dict:
    """Counts of one stream; summing the dictionaries of many streams gives the pooled numbers."""
    truth, piece, pred = np.asarray(truth, int), np.asarray(piece, int), np.asarray(pred, int)
    n = len(truth)
    piece_ok, segment_ok, piece_mid, segment_mid, pieces, segments = _midpoint_tables(truth, piece, pred, classes)
    wrong = pred != truth
    split = {"edge": int((wrong & piece_ok & segment_ok).sum()), "spurious": int((wrong & piece_ok & ~segment_ok).sum()),
             "absorbed": int((wrong & ~piece_ok & segment_ok).sum()), "misread": int((wrong & ~piece_ok & ~segment_ok).sum())}
    frame = np.zeros((classes, classes), np.int64)
    np.add.at(frame, (truth, pred), 1)
    keep = np.zeros(n, bool)
    dropped = 0
    for a, b, _ in segments:
        lo, hi = (a if a == 0 else a + TRIM), (b if b == n else b - TRIM)
        if hi > lo:
            keep[lo:hi] = True
        else:
            dropped += 1
    trimmed = np.zeros((classes, classes), np.int64)
    np.add.at(trimmed, (truth[keep], pred[keep]), 1)
    return {"samples": n, "wrong": int(wrong.sum()), "floor_wrong": int((grid_floor_labels(truth) != truth).sum()), "split": split,
            "pieces": len(pieces), "piece_midpoints": piece_mid, "segments": len(segments), "segment_midpoints": segment_mid,
            "frame": frame, "trimmed": trimmed, "trimmed_dropped_segments": dropped}


def match_changes(pred: np.ndarray, true: np.ndarray, near: int) -> dict[int, int]:
    """Reference change -> predicted change: one to one within `near` samples, most matches, then least total distance."""
    from scipy.optimize import linear_sum_assignment

    pred, true = np.asarray(pred, int), np.asarray(true, int)
    if not pred.size or not true.size:
        return {}
    distance = np.abs(true[:, None] - pred[None, :]).astype(float)
    big = float(near + 1) * (true.size + pred.size + 1)  # any invalid pair costs more than all valid ones
    rows, cols = linear_sum_assignment(np.where(distance <= near, distance, big))
    return {int(true[r]): int(pred[c]) for r, c in zip(rows, cols) if distance[r, c] <= near}


def cut_offsets(truth: np.ndarray, pred: np.ndarray) -> tuple[list[int], int]:
    """Signed offsets (predicted minus reference, samples) of the matched cuts, and the number of reference cuts."""
    true_cuts = np.flatnonzero(np.asarray(truth)[1:] != np.asarray(truth)[:-1]) + 1
    pred_cuts = np.flatnonzero(np.asarray(pred)[1:] != np.asarray(pred)[:-1]) + 1
    matched = match_changes(pred_cuts, true_cuts, NEAR)
    return [p - t for t, p in matched.items()], int(true_cuts.size)


def _levenshtein(a: list[int], b: list[int]) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


def edit_score(truth: np.ndarray, pred: np.ndarray) -> float:
    """MS-TCN's Edit: 100 * (1 - Levenshtein distance of the two label sequences / the longer sequence length)."""
    t, p = [v for _, _, v in runs(truth)], [v for _, _, v in runs(pred)]
    return 100.0 * (1.0 - _levenshtein(p, t) / max(len(p), len(t), 1))


def overlap_counts(truth: np.ndarray, pred: np.ndarray, overlap: float) -> tuple[int, int, int]:
    """MS-TCN's (tp, fp, fn) of one stream at an IoU threshold: each predicted segment takes the reference segment of
    its own label with the largest IoU; it is a true positive when that IoU reaches the threshold and the reference
    segment is still free."""
    t, p = runs(truth), runs(pred)
    used = [False] * len(t)
    tp = fp = 0
    for a, b, label in p:
        best, where = 0.0, -1
        for k, (c, d, other) in enumerate(t):
            if other != label:
                continue
            iou = max(0, min(b, d) - max(a, c)) / (max(b, d) - min(a, c))
            if iou > best:
                best, where = iou, k
        if where >= 0 and best >= overlap and not used[where]:
            tp, used[where] = tp + 1, True
        else:
            fp += 1
    return tp, fp, len(t) - sum(used)


def _ratio(a: float, b: float) -> float | None:
    return float(a / b) if b else None


def _pool(streams: list[tuple[np.ndarray, np.ndarray, np.ndarray]], classes: int) -> tuple[dict, list[int], int, list[float], dict]:
    total: dict | None = None
    offsets: list[int] = []
    cuts, edits = 0, []
    f1 = {k: np.zeros(3, np.int64) for k in OVERLAPS}
    for truth, piece, pred in streams:
        counts = stream_counts(truth, piece, pred, classes)
        if total is None:
            total = counts
        else:
            for key, value in counts.items():
                total[key] = {k: total[key][k] + value[k] for k in value} if isinstance(value, dict) else total[key] + value
        found, n = cut_offsets(truth, pred)
        offsets.extend(found)
        cuts += n
        edits.append(edit_score(truth, pred))
        for k in f1:
            f1[k] += np.array(overlap_counts(truth, pred, k))
    if total is None:
        raise ValueError("no stream to summarise")
    return total, offsets, cuts, edits, f1


def summarise(streams: list[tuple[np.ndarray, np.ndarray, np.ndarray]], classes: int = 5) -> dict:
    """Pooled numbers of a set of streams; each stream is (reference labels, source piece, predicted labels)."""
    total, offsets, cuts, edits, f1 = _pool(streams, classes)
    pm, sm, fr, tr = total["piece_midpoints"], total["segment_midpoints"], total["frame"], total["trimmed"]
    off = np.asarray(offsets, float) / HZ
    return {
        "streams": len(streams), "samples": int(total["samples"]),
        "frame_error": _ratio(total["wrong"], total["samples"]), "grid_floor": _ratio(total["floor_wrong"], total["samples"]),
        "wrong_split": {k: _ratio(total["split"][k], total["samples"]) for k in CATEGORIES},
        "wrong_split_counts": total["split"],
        "A1_piece": {"n": int(pm.sum()), "accuracy": _ratio(np.diag(pm).sum(), pm.sum()),
                     "by_class": [_ratio(pm[k, k], pm[k].sum()) for k in range(classes)], "confusion": pm.tolist()},
        "A1_segment": {"n": int(sm.sum()), "accuracy": _ratio(np.diag(sm).sum(), sm.sum()),
                       "by_class": [_ratio(sm[k, k], sm[:, k].sum()) for k in range(classes)], "confusion": sm.tolist()},
        "A4_cuts": {"true": cuts, "matched_share": _ratio(len(offsets), cuts), "median_s": float(np.median(off)) if off.size else None,
                    "late_share": _ratio(int((off > 0).sum()), off.size), "early_share": _ratio(int((off < 0).sum()), off.size),
                    "abs_median_s": float(np.median(np.abs(off))) if off.size else None,
                    "abs_p90_s": float(np.percentile(np.abs(off), 90)) if off.size else None},
        "A5_purity": [_ratio(fr[k, k], fr[:, k].sum()) for k in range(classes)],
        "A5_recovery": [_ratio(fr[k, k], fr[k].sum()) for k in range(classes)],
        "A5_purity_trimmed": [_ratio(tr[k, k], tr[:, k].sum()) for k in range(classes)],
        "A5_recovery_trimmed": [_ratio(tr[k, k], fr[k].sum()) for k in range(classes)],
        "A5_trimmed_dropped_segments": int(total["trimmed_dropped_segments"]),
        "A6_f1": {f"{int(k * 100)}": _ratio(2 * v[0], 2 * v[0] + v[1] + v[2]) for k, v in f1.items()},
        "A6_edit": float(np.mean(edits)) if edits else None,
    }
