"""A38 (configs/experiments/video_predicted_run_context_v1.yaml): A37's wider auxiliary window W limited to the frozen
Base's predicted run around the request centre.

For a request centred at c, [a(c), b(c)) is the run of the Base's run-length encoding (consecutive equal predicted labels
of the sequence) that contains c, and C = clip(W, a(c), b(c) - 1) element-wise. The mapping takes only positions,
sequence lengths and the Base's predicted labels as arguments; it never sees reference labels or interval targets. One
mapping per window serves every batch and the readout. The marker slot keeps c because a(c) <= c < b(c).
"""
from __future__ import annotations

import numpy as np

import echo_routing.temporal.wider_temporal as wt


def run_bounds(base: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[a, b) of the Base run that contains each centre c; base is one sequence's predicted labels."""
    base = np.asarray(base); starts = np.flatnonzero(np.r_[True, base[1:] != base[:-1]]); ends = np.r_[starts[1:], len(base)]
    k = np.searchsorted(starts, np.asarray(c, dtype=np.int64), side="right") - 1; return starts[k], ends[k]


def split_base(base_labels: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    """Per-sequence Base labels from A31's pool-order concatenation."""
    if int(np.sum(lengths)) != len(base_labels):
        raise ValueError("Base labels do not cover the pool's samples")
    return np.split(np.asarray(base_labels), np.cumsum(lengths)[:-1])


def limited_positions(seq: np.ndarray, centre: np.ndarray, s: np.ndarray, marker: np.ndarray, lengths: np.ndarray, base_labels: np.ndarray) -> dict:
    """W (A37) and C of every request window; seq / centre / s / marker are pool_W's index arrays, lengths the sequence lengths."""
    seq, centre = np.asarray(seq, dtype=np.int64), np.asarray(centre, dtype=np.int64); n = np.asarray(lengths, dtype=np.int64)
    w = wt.wide_positions(s, centre, n[seq]); per = split_base(base_labels, n); a, b = np.empty_like(centre), np.empty_like(centre)
    for i in np.unique(seq):
        m = seq == i; a[m], b[m] = run_bounds(per[int(i)], centre[m])
    cpos = np.clip(w, a[:, None], b[:, None] - 1); rows = np.arange(len(centre))
    if not (np.array_equal(w[rows, marker], centre) and np.array_equal(cpos[rows, marker], centre)):
        raise ValueError("a window's marker slot is not its centre")
    return {"W": w, "C": cpos, "a": a, "b": b}


def pool_limited(seqs: list[dict], W: dict, base_labels: np.ndarray) -> dict:
    """limited_positions on a pool: reads only the index arrays of W = pool_W(seqs) and the sequences' ref counts."""
    return limited_positions(W["seq"], W["centre"], W["s"], W["marker"], np.asarray([len(q["refs"]) for q in seqs]), base_labels)


def same_refs(seqs: list[dict], seq_idx: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Windows whose complete ref sequences (source, frame, gamma) at positions p and q are identical."""
    out = (np.asarray(p) == np.asarray(q)).all(1)
    for k in np.flatnonzero(~out):
        out[k] = wt.window_refs(seqs, [seq_idx[k]], p[k][None])[0] == wt.window_refs(seqs, [seq_idx[k]], q[k][None])[0]
    return out


def head(fold: int):
    """A37's residual head of a fit fold (bit-identical for the W and C arms and for A37)."""
    return wt.head(fold)
