"""Clip index policies for clip-level video models (pure NumPy; unit-tested).

A clip is `n_frames` source frames taken every `step` 30-fps frames (step 3 = the 10 Hz routing grid), so a
16-frame clip spans the same 1.6 s as the sliding windows used to evaluate STFM / EchoViewCLIP. Cines shorter than
a clip repeat their last frame (never loop, never stretch).
"""
from __future__ import annotations

import numpy as np

from echo_routing.errors import FrameError


def clip_indices(n_source: int, n_frames: int, step: int, start: int) -> np.ndarray:
    """Indices start, start+step, ... clamped to the last source frame."""
    if n_source <= 0 or n_frames <= 0 or step <= 0:
        raise FrameError("n_source, n_frames and step must be positive")
    if not 0 <= start < n_source:
        raise FrameError(f"start {start} outside [0, {n_source})")
    return np.minimum(start + step * np.arange(n_frames), n_source - 1).astype(int)


def max_start(n_source: int, n_frames: int, step: int) -> int:
    """Largest start whose clip fits entirely inside the cine (0 when the cine is shorter than a clip)."""
    return max(0, n_source - 1 - step * (n_frames - 1))


def random_clip(n_source: int, n_frames: int, step: int, rng: np.random.Generator) -> np.ndarray:
    return clip_indices(n_source, n_frames, step, int(rng.integers(0, max_start(n_source, n_frames, step) + 1)))


def spaced_clips(n_source: int, n_frames: int, step: int, k: int) -> list[np.ndarray]:
    """k evenly spaced clips covering the cine (deterministic evaluation); fewer if the cine is short."""
    if k <= 0:
        raise FrameError("k must be positive")
    top = max_start(n_source, n_frames, step)
    starts = sorted(set(int(round(s)) for s in np.linspace(0, top, k)))
    return [clip_indices(n_source, n_frames, step, s) for s in starts]


def pad_window(samples: np.ndarray, size: int) -> np.ndarray:
    """Right-pad a window of sample indices to `size` by repeating its last sample (short streams / cines)."""
    samples = np.asarray(samples, dtype=int)
    if samples.size == 0:
        raise FrameError("empty window")
    if samples.size >= size:
        return samples[:size]
    return np.concatenate([samples, np.full(size - samples.size, samples[-1], dtype=int)])
