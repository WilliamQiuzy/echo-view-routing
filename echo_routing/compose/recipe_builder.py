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


def _draw(rng: np.random.Generator, items: list[tuple[str, int]], used: set[str], min_len: int):
    cands = [(v, n) for v, n in items if v not in used and n >= min_len]
    if not cands:
        return None
    v, n = cands[rng.integers(len(cands))]
    used.add(v)
    return v, n


def _fragment(rng, video_id: str, n: int, label: int, variant: str, min_len: int, max_len: int) -> Fragment:
    length = int(rng.integers(min_len, min(max_len, n) + 1))
    start = int(rng.integers(0, n - length + 1))
    return Fragment(video_id, label, variant, start, start + length)


def make_pair_recipes(pool: Pool, n_per_cell: int, rng: np.random.Generator, edit_variant: str = "gamma090",
                      min_len: int = 10, max_len: int = 60, reuse: bool = False) -> list[Recipe]:
    """Two-fragment recipes for all four cells. Each native cine used at most once unless reuse=True."""
    labels = sorted(pool.by_label)
    if len(labels) < 2:
        raise ValueError("need at least two labels to build different-family pairs")
    used: set[str] = set(); recipes: list[Recipe] = []
    for cell in CELLS:
        same, edited = cell.startswith("same"), cell.endswith("edit")
        for k in range(n_per_cell):
            la = labels[rng.integers(len(labels))]
            lb = la if same else labels[(labels.index(la) + 1 + rng.integers(len(labels) - 1)) % len(labels)]
            a = _draw(rng, pool.by_label[la], used if not reuse else set(), min_len)
            b = _draw(rng, pool.by_label[lb], used if not reuse else set(), min_len)
            if a is None or b is None:
                break
            fa = _fragment(rng, a[0], a[1], la, "orig", min_len, max_len)
            fb = _fragment(rng, b[0], b[1], lb, edit_variant if edited else "orig", min_len, max_len)
            recipes.append(Recipe(f"{cell}_{k:04d}", cell, (fa, fb)))
    return recipes


def recipe_to_dict(r: Recipe) -> dict:
    return {"recipe_id": r.recipe_id, "cell": r.cell,
            "fragments": [{"video_id": f.video_id, "label": f.label, "variant": f.variant, "start": f.start, "end": f.end}
                          for f in r.fragments]}


def recipe_from_dict(d: dict) -> Recipe:
    return Recipe(d["recipe_id"], d["cell"], tuple(Fragment(**f) for f in d["fragments"]))
