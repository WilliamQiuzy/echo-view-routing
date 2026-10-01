"""Problem A (docs/research_plan.md): does comparing the same cardiac phase across adjacent cycles separate label
changes from within-recording fluctuations better than ordinary pooling of the same frames?

The methods see only the stream's own images and the recognizer's per-sample probabilities. Source files, labels and
join positions are used for scoring only.

- `local_period`: cardiac period per 30-fps frame from image motion in sliding windows (nan where unreliable).
- `change_score`: compares the K cycles after sample t with the K cycles before it. The three variants use exactly the
  same samples:
    phase     the after- and before-samples at the same phase are compared (mean over phases of the TV distance);
    pool      the two windows are pooled first (TV distance between the window means): same frames, no matching;
    scramble  as phase, but each before-cycle is re-paired at its own random phase offset (drawn per cycle).
  Where no period is available, every variant uses the same fixed two-window comparison (FALLBACK_SAMPLES per side).
- `percentile_table` / `to_percentile`: scores become percentiles among validation single-label clips, separately
  for samples with and without a period, so one threshold means the same false-alarm level in both.
- `decode_cuts`: cuts at percentile peaks above a threshold (non-maximum suppression), each segment labelled by its
  mean probability, equal neighbours merged.
- `cut_counts`: matches cuts to different-label joins within a tolerance and counts cuts at same-label joins and
  extra cuts away from every join.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echo_routing.evaluate.metrics_boundary import match_boundaries
from echo_routing.realvideo.cycles import estimate_cycle
from echo_routing.temporal.segments import boundaries_from_labels

WINDOW_S, HOP_S = 2.4, 0.3          # sliding windows for the period (about 2.5 cycles)
MIN_PEAK, MIN_DIP = 0.3, 0.6        # reliability, chosen against the MIMIC ECG (round 1)
FALLBACK_SAMPLES = 9                # fixed two-window comparison where no period is known (0.9 s at 10 Hz)
SCRAMBLE_RANGE = (0.25, 0.75)       # per-cycle phase offset of the scramble control, in cycles
MODES = ("phase", "pool", "scramble")


def local_period(grey: np.ndarray, fps: float, window_s: float = WINDOW_S, hop_s: float = HOP_S,
                 min_peak: float = MIN_PEAK, min_dip: float = MIN_DIP) -> np.ndarray:
    """Period in frames for each frame of a (T, H, W) stream: mean over reliable windows covering it, else nan."""
    t = len(grey); w = int(round(window_s * fps)); hop = max(1, int(round(hop_s * fps)))
    total, count = np.zeros(t), np.zeros(t)
    if t >= w:
        starts = list(range(0, t - w + 1, hop))
        if starts[-1] != t - w:
            starts.append(t - w)
        for s in starts:
            est = estimate_cycle(grey[s:s + w], fps)
            if est.reliable(min_peak, min_dip):
                total[s:s + w] += est.period_s * fps; count[s:s + w] += 1
    out = np.full(t, np.nan)
    out[count > 0] = total[count > 0] / count[count > 0]
    return out


def _tv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(a - b).sum(-1)


def _window_tv(prob: np.ndarray, t: int, length: int) -> float:
    n = len(prob); a, b = prob[t:min(n, t + length)], prob[max(0, t - length):t]
    return float(_tv(a.mean(0), b.mean(0))) if len(a) and len(b) else 0.0


def _phase_tv(prob: np.ndarray, t: int, period: float, k_cycles: int, shifts: np.ndarray | None) -> float:
    """Mean over phases u of TV(mean of after-samples at u, mean of before-samples at u, or at a shifted u)."""
    n = len(prob); p = max(1, int(round(period))); u = np.arange(p)
    after = np.rint(t + u[:, None] + period * np.arange(k_cycles)[None, :]).astype(int)            # (p, K)
    ub = u[:, None] if shifts is None else (u[:, None] + np.rint(shifts[None, :] * period)) % p
    before = np.rint(t + ub - period * np.arange(1, k_cycles + 1)[None, :]).astype(int)           # (p, K)
    ok_a, ok_b = (after >= 0) & (after < n), (before >= 0) & (before < n)
    keep = ok_a.any(1) & ok_b.any(1)
    if not keep.any():
        return 0.0
    pa = np.where(ok_a[..., None], prob[np.clip(after, 0, n - 1)], 0.0).sum(1) / np.maximum(ok_a.sum(1), 1)[:, None]
    pb = np.where(ok_b[..., None], prob[np.clip(before, 0, n - 1)], 0.0).sum(1) / np.maximum(ok_b.sum(1), 1)[:, None]
    return float(_tv(pa[keep], pb[keep]).mean())


def change_score(prob: np.ndarray, period_samples: np.ndarray, mode: str, k_cycles: int = 1,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample change score and a mask of samples where a period was available (else the fixed fallback)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    prob = np.asarray(prob, dtype=float); n = len(prob)
    period_samples = np.asarray(period_samples, dtype=float)
    avail = np.isfinite(period_samples) & (period_samples >= 2)
    rng = np.random.default_rng(seed)
    cycle_shift: dict[tuple[int, int], float] = {}
    score = np.zeros(n)
    for t in range(1, n):
        if not avail[t]:
            score[t] = _window_tv(prob, t, FALLBACK_SAMPLES)
        elif mode == "pool":
            score[t] = _window_tv(prob, t, int(round(k_cycles * period_samples[t])))
        else:
            shifts = None
            if mode == "scramble":
                c = int(t // period_samples[t])  # cycle index; one independent offset per before-cycle and cycle
                shifts = np.array([cycle_shift.setdefault((c - k, k), rng.uniform(*SCRAMBLE_RANGE)) for k in range(1, k_cycles + 1)])
            score[t] = _phase_tv(prob, t, period_samples[t], k_cycles, shifts)
    return score, avail


def percentile_table(scores: np.ndarray, n_points: int = 2001) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    return np.quantile(s, np.linspace(0, 1, n_points)) if s.size else np.array([np.inf])


def to_percentile(scores: np.ndarray, table: np.ndarray) -> np.ndarray:
    return np.searchsorted(table, np.asarray(scores, dtype=float), side="right") / len(table)


def decode_cuts(prob: np.ndarray, pct: np.ndarray, threshold: float, nms: int = 5) -> np.ndarray:
    """Labels from cuts at percentile peaks >= threshold (greedy NMS within `nms` samples)."""
    prob = np.asarray(prob, dtype=float); n = len(prob)
    cand = [t for t in range(1, n) if pct[t] >= threshold
            and pct[t] >= pct[max(1, t - 1)] and pct[t] >= pct[min(n - 1, t + 1)]]
    accepted: list[int] = []
    for t in sorted(cand, key=lambda i: -pct[i]):
        if all(abs(t - a) > nms for a in accepted):
            accepted.append(t)
    edges = [0] + sorted(accepted) + [n]
    labels = np.empty(n, dtype=int)
    for a, b in zip(edges[:-1], edges[1:]):
        labels[a:b] = int(prob[a:b].mean(0).argmax())
    return labels


@dataclass(frozen=True)
class CutCounts:
    n_diff_joins: int
    matched: int
    n_same_joins: int
    same_join_cuts: int        # same-label joins with a cut within the tolerance
    extra_internal: int        # cuts farther than the tolerance from every join
    extra_near_join: int       # additional cuts within the tolerance of a different-label join (double cuts)
    loc_error_samples: tuple[float, ...]
    n_samples: int


def cut_counts(labels: np.ndarray, joins: np.ndarray, diff_joins: np.ndarray, tol: int = 5) -> CutCounts:
    pred = boundaries_from_labels(labels)
    joins, diff_joins = np.asarray(joins, dtype=int), np.asarray(diff_joins, dtype=int)
    same = np.array(sorted(set(joins.tolist()) - set(diff_joins.tolist())), dtype=int)
    pairs = match_boundaries(pred, diff_joins, tol)
    matched_pred = {i for i, _ in pairs}
    rest = [int(pred[i]) for i in range(pred.size) if i not in matched_pred]
    same_hit = sum(1 for j in same if any(abs(r - j) <= tol for r in rest))
    near_same = [r for r in rest if same.size and np.min(np.abs(same - r)) <= tol]
    near_diff = [r for r in rest if r not in near_same and diff_joins.size and np.min(np.abs(diff_joins - r)) <= tol]
    internal = len(rest) - len(near_same) - len(near_diff)
    errs = tuple(float(abs(pred[i] - diff_joins[j])) for i, j in pairs)
    return CutCounts(int(diff_joins.size), len(pairs), int(same.size), same_hit, internal, len(near_diff), errs, len(labels))
