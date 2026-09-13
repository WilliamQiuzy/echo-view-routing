from __future__ import annotations

import numpy as np
import pytest

from echo_routing.features.cache import VideoFeatures
from echo_routing.compose.recipe_builder import CELLS, Pool, make_multi_recipes, make_pair_recipes, recipe_from_dict, recipe_to_dict
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
        assert r.fragments[0].label == lab[r.fragments[0].video_id] and r.fragments[1].label == lab[r.fragments[1].video_id]
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


def test_recipes_balanced_across_cells_with_small_classes():
    ids = [f"v{i}" for i in range(200)]
    labels = [0] * 120 + [1] * 60 + [2] * 20  # imbalanced like EV9V
    pool = Pool.from_index(ids, labels, [40] * 200)
    rec = make_pair_recipes(pool, n_per_cell=100, rng=np.random.default_rng(0), min_len=10, max_len=20)
    counts = {c: sum(r.cell == c for r in rec) for c in CELLS}
    assert all(counts[c] >= 15 for c in CELLS), counts  # every cell populated, incl. the last one
    used = [f.video_id for r in rec for f in r.fragments]
    assert len(used) == len(set(used))
    assert all((r.fragments[0].label == r.fragments[1].label) == r.cell.startswith("same") for r in rec)


def test_recipes_skip_short_cines_and_allow_reuse():
    pool = Pool.from_index(["a", "b", "c", "d"], [0, 0, 1, 1], [5, 40, 40, 40])
    rec = make_pair_recipes(pool, 5, np.random.default_rng(0), min_len=10, max_len=20, reuse=True)
    assert "a" not in {f.video_id for r in rec for f in r.fragments}
    assert {r.cell for r in rec} == set(CELLS)


def test_multi_recipes_class_balanced_and_bounded():
    ids = [f"v{i}" for i in range(300)]; labels = [0] * 200 + [1] * 60 + [2] * 30 + [3] * 10
    pool = Pool.from_index(ids, labels, [40] * 300)
    rec = make_multi_recipes(pool, 200, np.random.default_rng(0), n_fragments=(4, 8), min_len=10, max_len=20, reuse=True)
    assert len(rec) == 200 and all(4 <= len(r.fragments) <= 8 for r in rec)
    counts = np.bincount([f.label for r in rec for f in r.fragments], minlength=4)
    assert counts.min() > 0.6 * counts.max()  # near-uniform over classes despite 20:1 availability
    assert any(f.variant == "gamma090" for r in rec for f in r.fragments)
    lens = [f.end - f.start for r in rec for f in r.fragments]; assert min(lens) >= 10 and max(lens) <= 20


def test_multi_recipes_no_reuse_uses_each_cine_once():
    pool = Pool.from_index([f"v{i}" for i in range(40)], [i % 2 for i in range(40)], [40] * 40)
    rec = make_multi_recipes(pool, 10, np.random.default_rng(1), n_fragments=(4, 4), reuse=False)
    used = [f.video_id for r in rec for f in r.fragments]
    assert len(used) == len(set(used)) and len(rec) == 10


def test_banks_are_deterministic_and_disjoint_per_split():
    import pandas as pd
    from echo_routing.compose.banks import multi_bank, native_bank, pair_bank
    from echo_routing.config import load_config
    cfg = load_config("configs/base.yaml", overrides={"recipes": {"n_per_cell": 5, "min_len": 10, "max_len": 20}, "multi": {"n_eval": 6, "n_train": 8}})
    idx = pd.DataFrame({"video_id": [f"v{i}" for i in range(120)], "split": ["train"] * 60 + ["validation"] * 30 + ["test"] * 30,
                        "label_index": [i % 5 for i in range(120)], "n_samples": [40] * 120})
    a, b = pair_bank(idx, "test", cfg), pair_bank(idx, "test", cfg)
    assert a.recipes == b.recipes and a.bank_id == "pairs_test"
    assert all(r.recipe_id.startswith("pairs_test_") for r in a.recipes)
    v = pair_bank(idx, "validation", cfg); assert not ({r.recipe_id for r in v.recipes} & {r.recipe_id for r in a.recipes})
    m = multi_bank(idx, "validation", cfg); assert len(m.recipes) == 6 and all(r.recipe_id.startswith("multi_validation_") for r in m.recipes)
    n = native_bank(idx, "test"); assert len(n.recipes) == 30 and n.recipes[0].recipe_id == "v90"
    ids = {f.video_id for r in m.recipes for f in r.fragments}; assert ids <= set(idx[idx.split == "validation"].video_id)
    assert all(len({f.video_id for f in r.fragments}) == len(r.fragments) for r in m.recipes)  # no cine twice within a stream
