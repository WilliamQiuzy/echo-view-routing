"""Constructed multi-clip streams (proposal §5.2-5.3) assembled from cached per-cine arrays.

Two-by-two design: {same, diff} family x {none, edit}. Each recipe records provenance
(native video ids, fragment ranges, edit variant) so a stream can be regenerated exactly.
Semantic boundary = position where the family label differs across a join.
Physical join = every fragment boundary (never a model input).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

CELLS = ("same_none", "same_edit", "diff_none", "diff_edit")


@dataclass(frozen=True)
class Fragment:
    video_id: str
    label: int
    variant: str  # feature-cache variant used for this fragment ("orig" or an edit)
    start: int    # sample index into the cached arrays (inclusive)
    end: int      # exclusive


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    cell: str
    fragments: tuple[Fragment, ...]


@dataclass
class Pool:
    """Candidate cines grouped by label, each with its cached sample count."""
    by_label: dict[int, list[tuple[str, int]]] = field(default_factory=dict)

    @classmethod
    def from_index(cls, video_ids: Sequence[str], labels: Sequence[int], n_samples: Sequence[int]) -> "Pool":
        pool = cls()
        for v, l, n in zip(video_ids, labels, n_samples):
            pool.by_label.setdefault(int(l), []).append((str(v), int(n)))
        return pool


def _fragment(rng, video_id: str, n: int, label: int, variant: str, min_len: int, max_len: int) -> Fragment:
    length = int(rng.integers(min_len, min(max_len, n) + 1))
    start = int(rng.integers(0, n - length + 1))
    return Fragment(video_id, label, variant, start, start + length)


def _pairs_same(chunk: list[tuple[str, int, int]]) -> list[tuple[tuple, tuple]]:
    by_label: dict[int, list] = {}
    for item in chunk:
        by_label.setdefault(item[2], []).append(item)
    pairs = []
    for items in by_label.values():
        pairs.extend(zip(items[0::2], items[1::2]))
    return pairs


def _pairs_diff(chunk: list[tuple[str, int, int]]) -> list[tuple[tuple, tuple]]:
    """Greedy cross-label pairing: repeatedly pair the two most populous remaining labels."""
    by_label: dict[int, list] = {}
    for item in chunk:
        by_label.setdefault(item[2], []).append(item)
    pairs = []
    while True:
        ranked = sorted((lab for lab in by_label if by_label[lab]), key=lambda l: -len(by_label[l]))
        if len(ranked) < 2:
            return pairs
        a, b = by_label[ranked[0]].pop(), by_label[ranked[1]].pop()
        pairs.append((a, b))


def make_pair_recipes(pool: Pool, n_per_cell: int, rng: np.random.Generator, edit_variant: str = "gamma090",
                      min_len: int = 10, max_len: int = 60, reuse: bool = False) -> list[Recipe]:
    """Two-fragment recipes for all four cells, balanced by construction.

    Eligible cines (n >= min_len) are shuffled and dealt round-robin to the four cells so every cell
    draws from an equally sized, disjoint pool; each native cine is used at most once (reuse=False).
    With reuse=True the same shuffled pool is offered to every cell (training banks only).
    """
    if len(pool.by_label) < 2:
        raise ValueError("need at least two labels to build different-family pairs")
    eligible = [(v, n, lab) for lab, items in pool.by_label.items() for v, n in items if n >= min_len]
    order = rng.permutation(len(eligible))
    chunks = [[eligible[i] for i in order[k::4]] for k in range(4)] if not reuse else [[eligible[i] for i in order]] * 4
    recipes: list[Recipe] = []
    for cell, chunk in zip(CELLS, chunks):
        same, edited = cell.startswith("same"), cell.endswith("edit")
        chunk = [chunk[i] for i in rng.permutation(len(chunk))]
        pairs = (_pairs_same if same else _pairs_diff)(chunk)[:n_per_cell]
        for k, ((va, na, la), (vb, nb, lb)) in enumerate(pairs):
            fa = _fragment(rng, va, na, la, "orig", min_len, max_len)
            fb = _fragment(rng, vb, nb, lb, edit_variant if edited else "orig", min_len, max_len)
            recipes.append(Recipe(f"{cell}_{k:04d}", cell, (fa, fb)))
    return recipes


def recipe_to_dict(r: Recipe) -> dict:
    return {"recipe_id": r.recipe_id, "cell": r.cell,
            "fragments": [{"video_id": f.video_id, "label": f.label, "variant": f.variant, "start": f.start, "end": f.end}
                          for f in r.fragments]}


def recipe_from_dict(d: dict) -> Recipe:
    return Recipe(d["recipe_id"], d["cell"], tuple(Fragment(**f) for f in d["fragments"]))
