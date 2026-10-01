"""Screening scores for one clip from per-sample class probabilities (no clip label is used).

- dominant class D = most frequent top-1 class;
- excursion = maximal run of samples whose top-1 is not D, starting after the first sample and ending before the last,
  lasting from `min_run` samples to `max_exc` seconds (it returns to D);
- split contrast = largest total-variation distance between the mean probability vectors before and after a split
  point with at least `min_side` seconds on each side.
Strata: repeated (>= rep_k excursions), sustained (contrast >= theta and the two sides' top-1 classes differ), mixed
(both), none; clips shorter than 2 * min_side are "short".
"""
from __future__ import annotations

import re

import numpy as np

DEFAULT = {"min_run": 2, "max_exc": 1.0, "theta": 0.5, "rep_k": 2}
MIN_SIDE_S = 1.0
_SESSION = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")


def session_of(video_id: str) -> str:
    """EV9V export timestamp; the official split is disjoint at this level."""
    m = _SESSION.match(video_id)
    return m.group(1) if m else video_id


def patient_of(video_id: str) -> str:
    """EV9V patient: the id without its clip index (703 groups, the paper's patient count)."""
    return video_id.rsplit("_", 1)[0]


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True as (start, end_exclusive)."""
    d = np.diff(np.r_[0, np.asarray(mask).astype(int), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def interior_runs(top: np.ndarray, dom: int) -> list[tuple[int, int]]:
    n = len(top)
    return [(a, b) for a, b in runs(top != dom) if a > 0 and b < n]


def split_contrast(p: np.ndarray, hz: float, min_side_s: float = MIN_SIDE_S) -> tuple[float, int, int, int]:
    """Max TV distance between side means over split points: (tv, split sample, top1 left, top1 right); nan if short."""
    n, k = len(p), int(round(min_side_s * hz))
    if n < 2 * k:
        return float("nan"), -1, -1, -1
    c = np.cumsum(p, 0)
    s = np.arange(k, n - k + 1)
    left = c[s - 1] / s[:, None]
    right = (c[-1] - c[s - 1]) / (n - s)[:, None]
    tv = 0.5 * np.abs(left - right).sum(1)
    j = int(tv.argmax())
    return float(tv[j]), int(s[j]), int(left[j].argmax()), int(right[j].argmax())


def stratum(n_ret: int, sustained: bool, short: bool, rep_k: int) -> str:
    if short:
        return "short"
    rep = n_ret >= rep_k
    return "mixed" if rep and sustained else "repeated" if rep else "sustained" if sustained else "none"


def clip_scores(p: np.ndarray, names: list[str], hz: float, prm: dict | None = None) -> dict:
    """Continuous scores and stratum for one clip; p is (n_samples, n_classes)."""
    prm = {**DEFAULT, **(prm or {})}
    p = np.asarray(p, dtype=float)
    top = p.argmax(1)
    dom = int(np.bincount(top, minlength=p.shape[1]).argmax())
    inner = interior_runs(top, dom)
    exc = [(a, b) for a, b in inner if prm["min_run"] <= b - a <= prm["max_exc"] * hz]
    longer = [(a, b) for a, b in inner if b - a > prm["max_exc"] * hz]
    tv, s, cl, cr = split_contrast(p, hz)
    short = len(top) < 2 * MIN_SIDE_S * hz
    sustained = (not short) and tv >= prm["theta"] and cl != cr
    return {"dominant": names[dom], "f_off": float((top != dom).mean()), "n_ret": len(exc),
            "t_ret": sum(b - a for a, b in exc) / hz, "n_long_interior": len(longer), "contrast": tv,
            "split_s": s / hz if s >= 0 else float("nan"), "sides": f"{names[cl]}>{names[cr]}" if s >= 0 else "",
            "stratum": stratum(len(exc), sustained, short, prm["rep_k"])}


def cohen_kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b)); ix = {c: i for i, c in enumerate(cats)}
    t = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        t[ix[x], ix[y]] += 1
    n = t.sum(); po = np.trace(t) / n; pe = (t.sum(0) * t.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")
