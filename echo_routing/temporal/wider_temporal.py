"""A37 (configs/experiments/video_wider_temporal_evidence_v1.yaml): wider-span request windows on the complete constructed
sequences, and the residual head of the S / W comparison.

S = window_frames(c, N, 16, 16) are A08's 16 sample positions of a request centred at c. W doubles every time offset,
W = clip(c + 2 * (S - c), 0, N - 1), on the same sample axis; both keep S's marker slot, where W equals c. The mapping
reads only positions and lengths, never labels; frames come from the sequence's own refs at those positions.
"""
from __future__ import annotations

import numpy as np

import echo_routing.temporal.spatial_detail as sd

FACTOR, SEED = 2, 20261026 + 37


def wide_positions(s: np.ndarray, c: np.ndarray, n: np.ndarray, factor: int = FACTOR) -> np.ndarray:
    """Element-wise clip(c + factor * (s - c), 0, n - 1); s is (..., 16), c and n broadcast over the last axis."""
    s = np.asarray(s, dtype=np.int64); c = np.asarray(c, dtype=np.int64)[..., None]; n = np.asarray(n, dtype=np.int64)[..., None]
    return np.clip(c + factor * (s - c), 0, n - 1)


def pool_positions(seqs: list[dict], W: dict, factor: int = FACTOR) -> np.ndarray:
    """W positions of every request window, in the order of W = pool_W(seqs); checks the marker slot lands on c."""
    n = np.asarray([len(q["refs"]) for q in seqs], dtype=np.int64)[W["seq"]]; pos = wide_positions(W["s"], W["centre"], n, factor)
    if not np.array_equal(pos[np.arange(len(pos)), W["marker"]], W["centre"]):
        raise ValueError("a wider window's marker slot is not its centre")
    return pos


def window_refs(seqs: list[dict], seq_idx: np.ndarray, pos: np.ndarray) -> list[list[tuple]]:
    return [[tuple(seqs[int(i)]["refs"][int(p)]) for p in row] for i, row in zip(seq_idx, pos)]


def head(fold: int) -> sd.Residual:
    """The S / W residual head of a fit fold (bit-identical for both arms)."""
    return sd.Residual(SEED + int(fold))
