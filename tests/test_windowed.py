from __future__ import annotations

import numpy as np
import pytest

from echo_routing.compose.recipe_builder import Fragment, Recipe
from echo_routing.features.cache import VideoFeatures
from echo_routing.temporal.windowed import (
    ECHOPRIME_TO_FAMILY5, ECHOPRIME_VIEWS, RAW9_ORDER, RAW9_TO_FAMILY5, assign_to_samples, frame_indices_30fps,
    map_probs, stream_plan, window_centres, window_samples,
)


def vf(video_id, n):
    return VideoFeatures(video_id, np.arange(n) * 3, np.arange(n) / 10, np.zeros((n, 4)), np.full((n, 5), 0.2), np.zeros((n, 5)))


def test_stream_plan_follows_fragments_and_edits():
    r = Recipe("x", "diff_edit", (Fragment("a", 0, "orig", 2, 5), Fragment("b", 2, "gamma090", 0, 2)))
    refs = stream_plan(r, lambda v, var: vf(v, 10))
    assert [(x.video_id, x.frame_idx, x.gamma) for x in refs] == [("a", 6, 1.0), ("a", 9, 1.0), ("a", 12, 1.0), ("b", 0, 0.9), ("b", 3, 0.9)]
    assert frame_indices_30fps(refs, np.array([3]))[:3] == [("b", 0, 0.9), ("b", 1, 0.9), ("b", 2, 0.9)]


def test_window_centres_and_samples():
    c = window_centres(23, 5)
    assert c[0] == 2 and c[-1] == 22 and np.all(np.diff(c) > 0)
    assert window_samples(0, 23, 16).tolist() == list(range(16))
    assert window_samples(22, 23, 16).tolist() == list(range(7, 23))
    assert window_samples(10, 23, 16).tolist() == list(range(2, 18))
    assert window_samples(3, 8, 16).tolist() == list(range(8))  # short stream: whole stream


def test_assign_nearest_centre():
    out = assign_to_samples(6, np.array([1, 4]), np.array([[1, 0], [0, 1]]))
    assert out.tolist() == [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]
    with pytest.raises(ValueError):
        assign_to_samples(3, np.array([]), np.zeros((0, 2)))


def test_map_probs_drops_unmapped_mass():
    raw = np.zeros((1, 9)); raw[0, RAW9_ORDER.index("PMASA")] = 0.5; raw[0, RAW9_ORDER.index("PPMLSA")] = 0.2; raw[0, RAW9_ORDER.index("PMPALA")] = 0.3
    fam = map_probs(raw, RAW9_ORDER, RAW9_TO_FAMILY5)
    assert fam[0].tolist() == pytest.approx([0, 0.7, 0, 0, 0])
    ep = np.zeros((1, 11)); ep[0, ECHOPRIME_VIEWS.index("Subcostal")] = 0.9; ep[0, ECHOPRIME_VIEWS.index("SSN")] = 0.1
    assert map_probs(ep, ECHOPRIME_VIEWS, ECHOPRIME_TO_FAMILY5)[0].tolist() == pytest.approx([0, 0, 0, 0, 0.9])
