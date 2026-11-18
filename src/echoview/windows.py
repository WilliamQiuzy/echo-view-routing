"""Windows over a 10 Hz sample sequence: one prediction every 5 samples (0.5 s), each read from 16 samples."""
from __future__ import annotations

import numpy as np

STRIDE = 5
SIZE = 16


def window_centres(n_samples: int, stride: int = STRIDE) -> np.ndarray:
    """Centre samples, `stride` apart, with the last sample's neighbourhood always covered."""
    if n_samples <= 0:
        return np.array([], dtype=int)
    centres = np.arange(stride // 2, n_samples, stride, dtype=int)
    if centres.size == 0 or centres[-1] < n_samples - 1 - stride // 2:
        centres = np.append(centres, n_samples - 1)
    return centres


def window_frames(centre: int, n_samples: int, size: int = SIZE) -> np.ndarray:
    """The 16 sample indices shown for one centre, clamped at the ends; a shorter sequence repeats samples."""
    start = max(0, min(centre - size // 2, n_samples - size))
    samples = np.arange(start, min(n_samples, start + size), dtype=int)
    return samples[np.linspace(0, len(samples) - 1, size).round().astype(int)]


def window_plan(n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """centres (W,), sample indices (W, 16) and the marker position (W,) of every window of a sequence."""
    if n_samples < 1:
        raise ValueError("an empty sequence has no prediction windows")
    centres = window_centres(n_samples)
    windows = np.stack([window_frames(int(c), n_samples) for c in centres])
    slots = []
    for centre, window in zip(centres, windows):
        at_centre = np.flatnonzero(window == centre)
        slots.append(at_centre[len(at_centre) // 2])
    return centres, windows, np.array(slots, np.int64)


def nearest_centre(n_samples: int, centres: np.ndarray) -> np.ndarray:
    """For every sample, the index of the window whose centre is nearest (the earlier one on a tie).

    `centres` must rise, as window_centres gives them. A sample belongs to the first centre whose midpoint to the next
    centre is not before it, which is the nearest centre without a table of all distances.
    """
    centres = np.asarray(centres)
    if centres.size == 0:
        raise ValueError("no window centre to assign samples to")
    midpoints = (centres[:-1] + centres[1:]) / 2.0
    return np.searchsorted(midpoints, np.arange(n_samples), side="left")
