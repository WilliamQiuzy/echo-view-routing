"""Cardiac phase from the ECG drawn on MIMIC-IV-Echo cine frames; independent of any view prediction.

Each stored loop shows its ECG as a coloured trace near the bottom of the frame and a red cursor that moves with the
displayed frame. The cursor gives a linear map frame -> x, and the R waves are the tallest upward spikes of the trace.
The parse is checked against the DICOM HeartRate tag by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from echo_routing.realvideo.cycles import phase_from_boundaries


R_HEIGHT = 0.6


@dataclass(frozen=True)
class EcgParse:
    ok: bool
    reason: str
    r_frames: np.ndarray        # fractional frame indices of R waves
    rr_s: float                 # median R-R interval
    rr_cv: float                # nan with < 3 R waves
    px_per_frame: float
    cursor_resid_px: float

    def phase(self, n_frames: int) -> np.ndarray:
        if not self.ok:
            return np.full(n_frames, np.nan)
        rr_f = float(np.median(np.diff(self.r_frames))) if self.r_frames.size >= 2 else np.nan
        return phase_from_boundaries(n_frames, self.r_frames, rr_f)


def _fail(reason: str) -> EcgParse:
    return EcgParse(False, reason, np.array([]), float("nan"), float("nan"), float("nan"), float("nan"))


def _trace_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    teal = (g > 120) & (b > 110) & (r < 90)
    green = (g > 150) & (r < 100) & (b < 100)
    return teal if teal.sum() >= green.sum() else green


def _cursor_mask(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] > 170) & (rgb[..., 1] < 90) & (rgb[..., 2] < 90)


def trace_top_row(frame: np.ndarray, search_from: float = 0.6) -> int | None:
    """First row of the ECG trace in one RGB frame (None if no trace), to keep the trace out of image-motion crops."""
    y0 = int(frame.shape[0] * search_from)
    rows = np.flatnonzero(_trace_mask(frame[y0:].astype(np.int16)).any(1))
    return int(y0 + rows[0]) if rows.size else None


def parse_burned_ecg(frames: np.ndarray, frame_time_s: float, search_from: float = 0.6) -> EcgParse:
    """frames: (T, H, W, 3) uint8 RGB cine. Searches the lower part of the frame (rows >= search_from * H)."""
    t, h, w, _ = frames.shape
    low = frames[:, int(h * search_from):].astype(np.int16)
    trace = _trace_mask(low).any(0)
    if trace.sum() < 50:
        return _fail("no trace")
    red = _cursor_mask(low)
    cx = np.array([np.flatnonzero(red[k].any(0)).mean() if red[k].any() else np.nan for k in range(t)])
    good = np.isfinite(cx)
    if good.sum() < max(3, t // 2):
        return _fail("no cursor")
    f = np.flatnonzero(good)
    slope, icpt = np.polyfit(f, cx[good], 1)
    resid = float(np.percentile(np.abs(cx[good] - (icpt + slope * f)), 95))
    if slope <= 0 or resid > max(3.0, 2 * slope):
        return _fail("cursor not linear")
    cols = np.flatnonzero(trace.any(0))
    top = np.array([np.flatnonzero(trace[:, c])[0] for c in cols], dtype=float)
    span = np.arange(cols.min(), cols.max() + 1)
    height = np.interp(span, cols, -top)
    base = float(np.median(height))
    px_per_s = slope / frame_time_s
    padded = np.r_[base, height, base]  # lets a spike at either end of the trace count
    # R waves: spikes reaching >= 60% of the tallest excursion above baseline (heights, not prominences: a deep S wave
    # inflates prominence and width)
    pk, _ = find_peaks(padded, height=base + max(4.0, R_HEIGHT * (height.max() - base)), prominence=4.0,
                       distance=max(1, int(0.25 * px_per_s)))
    r_x = span[np.clip(pk - 1, 0, span.size - 1)].astype(float)
    if r_x.size == 0:
        return _fail("no R wave")
    r_frames = (r_x - icpt) / slope
    rr = np.diff(r_x) / px_per_s
    rr_s = float(np.median(rr)) if rr.size else float("nan")
    rr_cv = float(rr.std() / rr.mean()) if rr.size >= 2 else float("nan")
    return EcgParse(True, "", r_frames, rr_s, rr_cv, float(slope), resid)
