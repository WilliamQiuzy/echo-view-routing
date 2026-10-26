"""A38: Base-run-limited wider positions on the constructed sequence axis."""
import numpy as np
import pytest
import torch

import echo_routing.temporal.predicted_run_context as prc
import echo_routing.temporal.wider_temporal as wt
import echo_routing.vit.target_mvit as tm
from echo_routing.evaluate.window_target import window_frames
from echo_routing.temporal.windowed import window_centres


def fake_pool(lengths):
    seqs = [{"refs": [(f"v{i}", k, 1.0) for k in range(n)]} for i, n in enumerate(lengths)]; seq, cen, ss, mk = [], [], [], []
    for i, n in enumerate(lengths):
        for c in window_centres(n, 5):
            s = window_frames(int(c), n, 16, 16); seq.append(i); cen.append(int(c)); ss.append(s); mk.append(tm.marker_slot(s, int(c)))
    return seqs, {"seq": np.asarray(seq), "centre": np.asarray(cen), "s": np.asarray(ss), "marker": np.asarray(mk)}


def loop_c(c: int, n: int, base: np.ndarray) -> list[int]:
    a, b = c, c + 1
    while a > 0 and base[a - 1] == base[c]:
        a -= 1
    while b < n and base[b] == base[c]:
        b += 1
    return [min(max(min(max(c + 2 * (int(x) - c), 0), n - 1), a), b - 1) for x in window_frames(c, n, 16, 16)]


@pytest.mark.parametrize("lengths", [[1], [15], [16, 40], [31, 200, 7]])
def test_vectorised_equals_loop_marker_and_bounds(lengths):
    seqs, W = fake_pool(lengths); rng = np.random.default_rng(0); base = rng.integers(0, 3, sum(lengths)) // 2     # long and short runs
    out = prc.pool_limited(seqs, W, base); per = prc.split_base(base, np.asarray(lengths))
    for k, (i, c) in enumerate(zip(W["seq"], W["centre"])):
        n = lengths[int(i)]; row = out["C"][k]; assert row.tolist() == loop_c(int(c), n, per[int(i)])
        assert row[W["marker"][k]] == c and (np.diff(row) >= 0).all() and out["a"][k] <= row.min() and row.max() <= out["b"][k] - 1
        assert np.array_equal(out["W"][k], wt.pool_positions(seqs, W)[k])


def test_one_long_run_gives_w_and_unlimited_windows_keep_w():
    seqs, W = fake_pool([200]); out = prc.pool_limited(seqs, W, np.zeros(200, int)); assert np.array_equal(out["C"], out["W"])
    base = np.r_[np.zeros(100, int), np.ones(100, int)]; out = prc.pool_limited(seqs, W, base)
    inside = (out["W"] >= out["a"][:, None]).all(1) & (out["W"] <= out["b"][:, None] - 1).all(1); assert inside.any() and (~inside).any()
    assert np.array_equal(out["C"][inside], out["W"][inside]) and not np.array_equal(out["C"][~inside], out["W"][~inside])
    k = int(np.flatnonzero(W["centre"] == 102)[0]); assert out["C"][k].min() == 100 and out["a"][k] == 100 and out["b"][k] == 200


def test_mapping_ignores_everything_but_positions_lengths_and_base():
    seqs, W = fake_pool([60, 33]); base = np.r_[np.zeros(20, int), np.full(40, 3), np.full(33, 1)]
    a = prc.pool_limited(seqs, W, base); seqs2 = [dict(q, labels=np.full(len(q["refs"]), 4)) for q in seqs]; W2 = dict(W, label=np.zeros(len(W["seq"])), labels16=np.ones((len(W["seq"]), 16)))
    b = prc.pool_limited(seqs2, W2, base); assert all(np.array_equal(a[k], b[k]) for k in a)
    with pytest.raises(ValueError):
        prc.pool_limited(seqs, W, base[:-1])


def test_same_refs_uses_the_complete_ref_sequence():
    seqs = [{"refs": [("v0", 0, 1.0), ("v0", 0, 1.0), ("v0", 1, 1.0), ("v1", 0, 0.8)]}]; p = np.asarray([[0, 2], [0, 2], [2, 3]]); q = np.asarray([[1, 2], [0, 3], [2, 3]])
    assert prc.same_refs(seqs, np.zeros(3, int), p, q).tolist() == [True, False, True]


def test_head_is_a37s_and_starts_at_q():
    a, b = prc.head(0), wt.head(0); assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    q = torch.randn(3, 5); e = torch.randn(3, 512); assert torch.equal(a(q, e, e), q)
