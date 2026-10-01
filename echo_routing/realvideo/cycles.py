"""Cardiac cycle from image motion alone; view predictions are never used.

Each frame becomes the vector of its moving pixels (every pixel centred over time, so strongly moving structures weigh
most; static overlays dropped), and frames are compared by Pearson correlation. The mean correlation of frames L apart, A(L), peaks at the cardiac period P.
Reliability is judged by the peak height A(P) and the dip A(P) - min_{0<L<P} A(L). Same-phase frames ("boundaries")
are the correlation peaks with one reference frame; phase in [0, 1) is linear between boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

MIN_PERIOD_S, MAX_PERIOD_S = 0.3, 2.0  # 200 to 30 beats per minute
MIN_CYCLES = 1.5                       # periods are searched only up to duration / 1.5
PEAK_FRACTION = 0.6                    # shortest lag whose peak reaches 60% of the best one (avoids 2P)
HIGHPASS_S = 1.5                       # per-pixel moving-average removal: respiration and probe drift


@dataclass(frozen=True)
class CycleEstimate:
    period_s: float
    peak: float
    dip: float
    boundaries: np.ndarray  # frame indices of same-phase frames
    cv: float               # coefficient of variation of boundary spacing (nan with < 3 boundaries)
    n_cycles: float         # clip length / period

    def reliable(self, min_peak: float, min_dip: float) -> bool:
        return bool(np.isfinite(self.period_s) and self.peak >= min_peak and self.dip >= min_dip)


NO_CYCLE = CycleEstimate(float("nan"), float("nan"), float("nan"), np.array([], dtype=int), float("nan"), float("nan"))


def frame_similarity(frames: np.ndarray, min_std: float = 1.0, fps: float | None = None) -> np.ndarray:
    """(T, H, W) grey frames -> (T, T) Pearson correlation of their moving pixels (nan if nothing moves).

    With fps, each pixel is first high-passed (its HIGHPASS_S moving average removed) so that slow drift does not
    make distant beats look alike.
    """
    x = np.asarray(frames, dtype=np.float64).reshape(len(frames), -1)
    w = int(round(HIGHPASS_S * fps)) if fps else 0
    if 3 <= w < len(x):
        x = x - uniform_filter1d(x, size=w, axis=0, mode="nearest")
    sd = x.std(0)
    keep = sd > min_std
    if keep.sum() < 10:
        return np.full((len(x), len(x)), np.nan)
    z = x[:, keep] - x[:, keep].mean(0)
    z -= z.mean(1, keepdims=True)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    return z @ z.T


def lag_profile(g: np.ndarray) -> np.ndarray:
    """A(L), L = 0..T-1: mean of the L-th off-diagonal of the similarity matrix."""
    return np.array([np.diagonal(g, k).mean() for k in range(len(g))])


def same_phase_frames(g: np.ndarray, period: float) -> np.ndarray:
    """Frames in the same phase as the reference frame that best matches its periodic successors."""
    t = len(g)

    def score(r: int) -> float:
        ks = np.round(np.arange(r + period, t, period)).astype(int)
        ks = ks[ks < t]
        return float(g[r, ks].mean()) if ks.size else -np.inf

    r = max(range(max(1, min(t, int(round(period))))), key=score)
    s = g[r]
    pk, _ = find_peaks(s, distance=max(1, int(0.6 * period)))
    keep = [k for k in pk.tolist() if s[k] > 0 and abs(k - r) >= 0.6 * period]
    return np.array(sorted(keep + [r]), dtype=int)


def estimate_cycle(frames: np.ndarray, fps: float, min_std: float = 1.0) -> CycleEstimate:
    g = frame_similarity(frames, min_std, fps)
    if not np.isfinite(g).all():
        return NO_CYCLE
    t = len(g)
    a = lag_profile(g)
    lo = max(2, int(np.floor(MIN_PERIOD_S * fps)))
    hi = min(int(np.ceil(MAX_PERIOD_S * fps)), int(np.floor(t / MIN_CYCLES)), t - 2)
    cand = [k for k in range(lo, hi + 1) if a[k] >= a[k - 1] and a[k] >= a[k + 1]]
    if not cand:
        return NO_CYCLE
    best = max(a[k] for k in cand)
    lag = min(k for k in cand if a[k] >= best - (1 - PEAK_FRACTION) * abs(best))  # shortest strong peak, not 2P
    y0, y1, y2 = a[lag - 1], a[lag], a[lag + 1]
    den = y0 - 2 * y1 + y2
    per = lag + (0.5 * (y0 - y2) / den if den < 0 else 0.0)
    bnd = same_phase_frames(g, per)
    d = np.diff(bnd)
    cv = float(d.std() / d.mean()) if d.size >= 2 else float("nan")
    return CycleEstimate(per / fps, float(a[lag]), float(a[lag] - a[1:lag].min()), bnd, cv, t / per)


def phase_from_boundaries(n_frames: int, boundaries: np.ndarray, period_frames: float) -> np.ndarray:
    """Phase in [0, 1) per frame: linear between boundaries, extrapolated by the period outside them."""
    ph = np.full(n_frames, np.nan)
    b = np.asarray(boundaries, dtype=float)
    if b.size == 0 or not np.isfinite(period_frames) or period_frames <= 0:
        return ph
    f = np.arange(n_frames, dtype=float)
    ph = np.interp(f, b, np.arange(b.size, dtype=float))
    lead, tail = f < b[0], f > b[-1]
    ph[lead] = (f[lead] - b[0]) / period_frames
    ph[tail] = (b.size - 1) + (f[tail] - b[-1]) / period_frames
    return np.mod(ph, 1.0)


def _random_placement(lengths: list[int], n: int, rng: np.random.Generator) -> np.ndarray:
    """Mask with runs of the given lengths placed uniformly at random without overlap."""
    k, free = len(lengths), n - sum(lengths)
    cuts = np.sort(rng.choice(free + k, size=k, replace=False))
    gaps = np.diff(np.r_[-1, cuts]) - 1
    m = np.zeros(n, dtype=bool); pos = 0
    for g, ln in zip(gaps, rng.permutation(lengths)):
        pos += int(g); m[pos:pos + ln] = True; pos += int(ln)
    return m


def phase_locking(phase: np.ndarray, off: np.ndarray, n_perm: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Resultant length of the phases of `off` samples, and a permutation p-value.

    The null keeps the number and lengths of off runs and places them uniformly at random. (A circular shift is not a
    valid null here: it keeps a periodic mask periodic, and the resultant length ignores which phase it locks to.)
    """
    ok = np.isfinite(phase)
    ph, m = np.asarray(phase)[ok], np.asarray(off, dtype=bool)[ok]
    if m.sum() < 2 or m.all():
        return float("nan"), float("nan")
    z = np.exp(2j * np.pi * ph)
    obs = abs(z[m].mean())
    lengths = [b - a for a, b in _runs(m)]
    rng = np.random.default_rng(seed)
    null = np.array([abs(z[_random_placement(lengths, m.size, rng)].mean()) for _ in range(n_perm)])
    return float(obs), float((1 + (null >= obs - 1e-12).sum()) / (1 + n_perm))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))
