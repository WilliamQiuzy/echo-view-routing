from __future__ import annotations

import numpy as np
import pytest

from echo_routing.features.cache import VideoFeatures
from echo_routing.compose.recipe_builder import CELLS, Pool, make_pair_recipes, recipe_from_dict, recipe_to_dict
from echo_routing.compose.stream_assembler import render
from echo_routing.errors import RecipeError


def fake_features(video_id: str, label: int, n: int, c: int = 3, variant: str = "orig") -> VideoFeatures:
    conf = 0.9 if variant == "orig" else 0.8
    prob = np.full((n, c), (1 - conf) / (c - 1)); prob[:, label] = conf
    return VideoFeatures(video_id, np.arange(n) * 3, np.arange(n) * 0.1, np.full((n, 4), float(label)), prob, np.log(prob))


@pytest.fixture
def pool():
    ids = [f"v{i}" for i in range(12)]; labels = [i % 3 for i in range(12)]; n = [40] * 12
    return Pool.from_index(ids, labels, n), dict(zip(ids, labels))


def test_recipes_cover_all_cells_without_reuse(pool):
    p, lab = pool
    rec = make_pair_recipes(p, n_per_cell=1, rng=np.random.default_rng(0), min_len=10, max_len=20)
    assert [r.cell for r in rec] == list(CELLS)
    used = [f.video_id for r in rec for f in r.fragments]
    assert len(used) == len(set(used))
    for r in rec:
        a, b = r.fragments
        assert (a.label == b.label) == r.cell.startswith("same")
        assert (b.variant == "gamma090") == r.cell.endswith("edit")
        assert 10 <= a.end - a.start <= 20


def test_render_marks_only_semantic_boundaries(pool):
    p, lab = pool
    rec = make_pair_recipes(p, 1, np.random.default_rng(1), min_len=10, max_len=20)
    loader = lambda vid, variant: fake_features(vid, lab[vid], 40, variant=variant)
    for r in rec:
        s = render(r, loader)
        assert s.joins.tolist() == [r.fragments[0].end - r.fragments[0].start]
        assert s.prob.shape[0] == sum(f.end - f.start for f in r.fragments)
        if r.cell.startswith("same"):
            assert s.semantic_boundaries.size == 0
        else:
            assert s.semantic_boundaries.tolist() == s.joins.tolist()
        assert np.array_equal(s.labels[: s.joins[0]], np.full(s.joins[0], r.fragments[0].label))


def test_recipe_dict_roundtrip(pool):
    p, _ = pool
    rec = make_pair_recipes(p, 1, np.random.default_rng(2))[0]
    assert recipe_from_dict(recipe_to_dict(rec)) == rec


def test_render_rejects_out_of_range_fragment(pool):
    p, lab = pool
    rec = make_pair_recipes(p, 1, np.random.default_rng(3), min_len=10, max_len=20)[0]
    with pytest.raises(RecipeError):
        render(rec, lambda vid, variant: fake_features(vid, lab[vid], 5))


def test_pool_needs_two_labels():
    with pytest.raises(ValueError):
        make_pair_recipes(Pool.from_index(["a"], [0], [40]), 1, np.random.default_rng(0))
