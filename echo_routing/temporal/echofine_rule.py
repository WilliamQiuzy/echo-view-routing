"""A35 (configs/experiments/a5c_echofine_evidence_v1.yaml): the single fixed acceptance rule for frozen EchoFine evidence.

For a Base run with class A4C or A5C, q is the equal-weight mean of the full 47-class EchoFine softmax over the run's
samples. The run becomes A4C if q's unique top class is a4ch-full, A5C if it is a5ch-full; any other top class (including
a5ch-outflow and the focused / colour / Doppler / M-mode classes) or a tie keeps Base. Runs of other classes never change.
"""
from __future__ import annotations

import numpy as np

A4C, A5C = 2, 3
COL_A4C_FULL, COL_A5C_FULL, COL_A5C_OUTFLOW = 7, 13, 14
COLS_FOCUSED_A4C = (8, 9, 10, 11, 12)          # a4ch-ias, a4ch-la, a4ch-lv, a4ch-ra, a4ch-rv


def decide(q: np.ndarray, base: int) -> int:
    if int(base) not in (A4C, A5C):
        return int(base)
    q = np.asarray(q, np.float64); top = np.flatnonzero(q == q.max())
    if top.size != 1:
        return int(base)
    return A4C if top[0] == COL_A4C_FULL else (A5C if top[0] == COL_A5C_FULL else int(base))


def top_category(q: np.ndarray) -> str:
    q = np.asarray(q, np.float64); top = np.flatnonzero(q == q.max())
    if top.size != 1:
        return "tie"
    t = int(top[0])
    return {COL_A4C_FULL: "a4ch-full", COL_A5C_FULL: "a5ch-full", COL_A5C_OUTFLOW: "a5ch-outflow"}.get(t, "focused A4C" if t in COLS_FOCUSED_A4C else "other")


def relabel(base: np.ndarray, p47: np.ndarray, runs) -> np.ndarray:
    """Every Base run takes decide(mean of its samples' 47-class probabilities, Base class); equal neighbours merge by RLE."""
    out = np.asarray(base, np.int64).copy()
    for a, b, c in runs(np.asarray(base)):
        out[a:b] = decide(np.asarray(p47[a:b], np.float64).mean(0), c)
    return out
