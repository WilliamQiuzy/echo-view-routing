"""Does a window model's output answer "which view at this time"? Diagnostics for B7's 1.6 s windows
(configs/experiments/b7_window_target_v1.yaml; 2026-10-05). Pure functions; scripts/b7_window_target.py does the I/O.

- Windows are the runner's own (scripts/run_windowed_echoviewclip.py): centres every `stride` samples, `size` samples
  clamped to the sequence, `frames` frames spread evenly over them. Every sample takes the nearest centre's output.
- A window is single-recording, cross-recording with one view, or mixed-views, by its frames' recordings and labels.
- The paired probe keeps a target recording's centre frames and fills the other positions with another recording's
  frames at the same positions.
- The ideal repair gives chosen windows the one-hot of their centre label (labels serve diagnosis only).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

CATEGORIES = ("single recording", "cross-recording, one view", "mixed views")
SHARE_BINS = ("<= 7 of 16", "8 of 16", "9-11 of 16", "12-15 of 16")


def window_frames(centre: int, n: int, size: int, frames: int) -> np.ndarray:
    """Sample indices of the frames the runner feeds for the window centred at `centre`."""
    start = max(0, min(centre - size // 2, n - size))
    samp = np.arange(start, min(n, start + size), dtype=int)
    return samp[np.linspace(0, len(samp) - 1, frames).round().astype(int)]


def nearest_centre(n: int, centres: np.ndarray) -> np.ndarray:
    """Index into `centres` of the centre nearest to each sample (the runner's piecewise-constant assignment)."""
    return np.abs(np.arange(n)[:, None] - np.asarray(centres)[None, :]).argmin(1)


def category(recordings: Sequence[str], labels: Sequence[int]) -> str:
    if len(set(int(x) for x in labels)) > 1:
        return CATEGORIES[2]
    return CATEGORIES[0] if len(set(recordings)) == 1 else CATEGORIES[1]


def share_bin(centre_frames: int) -> str:
    """Bin of the number of frames (of 16) that carry the centre sample's label."""
    k = int(centre_frames)
    return SHARE_BINS[0] if k <= 7 else SHARE_BINS[1] if k == 8 else SHARE_BINS[2] if k <= 11 else SHARE_BINS[3]


def middle(refs: Sequence[Sequence], size: int = 16) -> list[list]:
    """The `size` consecutive references around the middle of a recording, starting at n // 2 - size // 2."""
    if len(refs) < size:
        raise ValueError(f"{len(refs)} samples, need {size}")
    s = len(refs) // 2 - size // 2
    return [list(r) for r in refs[s:s + size]]


def probe_window(target: Sequence[Sequence], context: Sequence[Sequence], centre_frames: int, size: int = 16) -> list[list]:
    """`size` references: the target's at positions [size/2 - m/2, size/2 + m/2), the context's at all others."""
    m = int(centre_frames)
    if len(target) != size or len(context) != size or not 0 < m <= size or m % 2:
        raise ValueError(f"need two windows of {size} references and an even centre size in (0, {size}], got {m}")
    lo, hi = size // 2 - m // 2, size // 2 + m // 2
    return [list(target[i]) if lo <= i < hi else list(context[i]) for i in range(size)]


def pick_context(candidates: Sequence[str], exclude_sessions: set[str], rng: np.random.Generator,
                 session_of: Callable[[str], str]) -> str | None:
    """One candidate recording from another session (sorted pool, seeded draw); None when no candidate remains."""
    pool = [c for c in sorted(candidates) if session_of(c) not in exclude_sessions]
    return pool[int(rng.integers(len(pool)))] if pool else None


def repair(prob: np.ndarray, centres: np.ndarray, which: Sequence[int], labels: np.ndarray) -> np.ndarray:
    """Copy of `prob` in which every sample assigned to a chosen centre (index into `centres`) gets the one-hot of that
    centre sample's label. Other rows are unchanged."""
    out = np.array(prob, dtype=float, copy=True)
    assign = nearest_centre(len(out), centres)
    for j in which:
        rows = assign == int(j)
        out[rows] = 0.0
        out[rows, int(labels[int(centres[int(j)])])] = 1.0
    return out
