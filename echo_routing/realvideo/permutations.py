"""Within-fragment reorderings with fixed seeds, and the temporal structure of label deviations
(docs/research_plan.md, round of 2026-10-01).

Every reordering returns an index array over one fragment's samples; the set of samples never changes.
- true: identity;
- name: frame IDs sorted as text (the legacy order);
- random: a seeded permutation;
- reverse: last sample first;
- block-b: consecutive blocks of b samples keep their internal order and are shuffled as units. A trailing partial
  block stays at the end. A fragment with fewer than two full blocks is left unchanged and counted as not perturbed.
The permutation of a fragment depends only on (condition seed, recipe id, fragment index), so every method and every
experiment sees exactly the same permutation.
"""
from __future__ import annotations

import zlib

import numpy as np

KINDS = ("true", "name", "random", "reverse", "block")


def fragment_rng(seed: int, recipe_id: str, frag_index: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(recipe_id.encode()), int(frag_index)])


def order_index(n: int, kind: str, seed: int = 0, recipe_id: str = "", frag_index: int = 0,
                ids: np.ndarray | None = None, block: int | None = None) -> tuple[np.ndarray, bool]:
    """(index array, whether the order actually changed)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    ident = np.arange(n)
    if kind == "true" or n < 2:
        return ident, False
    if kind == "name":
        if ids is None:
            raise ValueError("name order needs frame ids")
        idx = np.array(sorted(range(n), key=lambda i: str(int(ids[i]))), dtype=int)
    elif kind == "reverse":
        idx = ident[::-1].copy()
    elif kind == "random":
        idx = fragment_rng(seed, recipe_id, frag_index).permutation(n)
    else:
        if not block or block < 1:
            raise ValueError("block order needs a positive block length")
        full = n // block
        if full < 2:
            return ident, False
        order = fragment_rng(seed, recipe_id, frag_index).permutation(full)
        idx = np.concatenate([ident[b * block:(b + 1) * block] for b in order] + [ident[full * block:]])
    return idx, bool((idx != ident).any())


def error_runs(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Lengths of maximal runs of consecutive samples whose label differs from the reference."""
    wrong = np.asarray(pred) != np.asarray(truth)
    d = np.diff(np.r_[0, wrong.astype(int), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def lag1_autocorrelation(prob: np.ndarray) -> float:
    """Mean over classes of the lag-1 autocorrelation of the probability sequence (nan if too short or constant)."""
    p = np.asarray(prob, dtype=float)
    if len(p) < 3:
        return float("nan")
    vals = []
    for c in range(p.shape[1]):
        x = p[:, c] - p[:, c].mean()
        den = float((x * x).sum())
        if den > 1e-12:
            vals.append(float((x[1:] * x[:-1]).sum() / den))
    return float(np.mean(vals)) if vals else float("nan")
