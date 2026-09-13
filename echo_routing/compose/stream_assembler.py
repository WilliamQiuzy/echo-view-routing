"""Assemble a constructed stream from cached per-cine arrays (feature-space composition, ADR-0001)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from echo_routing.compose.recipe_builder import Recipe
from echo_routing.errors import RecipeError
from echo_routing.features.cache import VideoFeatures


@dataclass(frozen=True)
class Stream:
    recipe: Recipe
    prob: np.ndarray
    feat: np.ndarray
    labels: np.ndarray            # per-sample family label (clip label propagated)
    joins: np.ndarray             # physical join positions (sample index of first sample after a join)
    semantic_boundaries: np.ndarray  # subset of joins where the label changes


def render(recipe: Recipe, loader: Callable[[str, str], VideoFeatures]) -> Stream:
    """Concatenate cached arrays per fragment. loader(video_id, variant) -> VideoFeatures."""
    probs, feats, labels, joins = [], [], [], []
    pos = 0
    for i, fr in enumerate(recipe.fragments):
        vf = loader(fr.video_id, fr.variant)
        if fr.end > vf.n:
            raise RecipeError(f"fragment {fr} exceeds cached length {vf.n}")
        probs.append(vf.prob[fr.start:fr.end]); feats.append(vf.feat[fr.start:fr.end])
        labels.append(np.full(fr.end - fr.start, fr.label, dtype=int))
        if i > 0:
            joins.append(pos)
        pos += fr.end - fr.start
    labels_arr = np.concatenate(labels)
    joins_arr = np.array(joins, dtype=int)
    sem = np.array([j for j in joins_arr if labels_arr[j] != labels_arr[j - 1]], dtype=int)
    return Stream(recipe, np.concatenate(probs), np.concatenate(feats), labels_arr, joins_arr, sem)


