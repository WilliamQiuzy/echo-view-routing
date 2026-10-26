"""A37: wider-span positions on the constructed sequence axis and the S / W head."""
import numpy as np
import pytest
import torch

import echo_routing.temporal.wider_temporal as wt
import echo_routing.vit.target_mvit as tm
from echo_routing.evaluate.window_target import window_frames
from echo_routing.temporal.windowed import window_centres


def loop_positions(c: int, n: int, factor: int = 2) -> list[int]:
    s = window_frames(c, n, 16, 16); return [min(max(c + factor * (int(x) - c), 0), n - 1) for x in s]


def fake_pool(lengths):
    seqs = [{"refs": [(f"v{i}", k, 1.0) for k in range(n)]} for i, n in enumerate(lengths)]; seq, cen, ss, mk = [], [], [], []
    for i, n in enumerate(lengths):
        for c in window_centres(n, 5):
            s = window_frames(int(c), n, 16, 16); seq.append(i); cen.append(int(c)); ss.append(s); mk.append(tm.marker_slot(s, int(c)))
    return seqs, {"seq": np.asarray(seq), "centre": np.asarray(cen), "s": np.asarray(ss), "marker": np.asarray(mk)}


@pytest.mark.parametrize("n", [1, 5, 15, 16, 17, 31, 40, 200])
def test_vectorised_equals_loop_and_marker_is_centre(n):
    seqs, W = fake_pool([n]); pos = wt.pool_positions(seqs, W)
    for k, c in enumerate(W["centre"]):
        assert pos[k].tolist() == loop_positions(int(c), n)
        assert pos[k][W["marker"][k]] == c and (np.diff(pos[k]) >= 0).all() and pos[k].min() >= 0 and pos[k].max() <= n - 1


def test_interior_offsets_double_and_factor_one_is_s():
    seqs, W = fake_pool([200]); pos = wt.pool_positions(seqs, W); k = int(np.flatnonzero(W["centre"] == 102)[0])
    assert (pos[k] - 102).tolist() == list(range(-16, 15, 2)) and (W["s"][k] - 102).tolist() == list(range(-8, 8))
    assert np.array_equal(wt.pool_positions(seqs, W, factor=1), W["s"])


def test_edges_repeat_and_refs_stay_inside_the_sequence():
    seqs, W = fake_pool([10, 40]); pos = wt.pool_positions(seqs, W); refs = wt.window_refs(seqs, W["seq"], pos)
    short = W["seq"] == 0; assert (pos[short] <= 9).all() and any(len(set(r)) < 16 for r, s_ in zip(refs, short) if s_)
    assert all(r[0][0] == f"v{int(i)}" for r, i in zip(refs, W["seq"]))
    first = int(np.flatnonzero(W["seq"] == 1)[0]); assert pos[first][0] == 0      # the first centre clamps at the start


def test_head_seed_identical_and_starts_at_q():
    a, b = wt.head(0), wt.head(0); assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert not torch.equal(a.fc1.weight, wt.head(1).fc1.weight)
    q = torch.randn(4, 5); e = torch.randn(4, 512); assert torch.equal(a(q, e, e), q) and sum(p.numel() for p in a.parameters()) == 32965
