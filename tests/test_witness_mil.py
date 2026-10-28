"""CPU tests for A43's witness module (echo_routing/temporal/witness_mil.py): bags, plan, local tokens, head, losses, edit."""
import numpy as np
import pytest
import torch

import echo_routing.temporal.interval_recognition as ir
import echo_routing.temporal.witness_mil as wm
from echo_routing.temporal.windowed import window_centres


def test_duration_weights_nearest_centre_ties_to_earlier():
    c = window_centres(12, 5)
    assert c.tolist() == [2, 7, 11]
    assert (wm.duration_weights(12, c) * 12).round().astype(int).tolist() == [5, 5, 2]       # sample 9 ties 7 / 11 -> 7


def _pool():
    seqs = [{"kind": "native", "labels": [2] * 12}, {"kind": "multi", "labels": [2] * 6 + [3] * 6}, {"kind": "native", "labels": [3] * 7},
            {"kind": "native", "labels": [0] * 9}, {"kind": "native", "labels": [2] * 3}]
    seq, cen = [], []
    for i, q in enumerate(seqs):
        for c in window_centres(len(q["labels"]), 5):
            seq.append(i); cen.append(int(c))
    return seqs, {"seq": np.asarray(seq), "centre": np.asarray(cen)}


def test_native_bags_whole_windows_and_weights():
    seqs, W = _pool(); B = wm.native_bags(seqs, W)
    assert B["a4c"] == [0, 4] and B["a5c"] == [2]
    for i in B["a4c"] + B["a5c"]:
        assert B["rows"][i].tolist() == np.flatnonzero(W["seq"] == i).tolist()
        assert abs(B["weights"][i].sum() - 1) < 1e-12
    with pytest.raises(ValueError):
        wm.native_bags([{"kind": "native", "labels": [2, 3]}], {"seq": np.zeros(1, int), "centre": np.zeros(1, int)})


def test_plan_is_stateless_and_explicit():
    seqs, W = _pool(); B = wm.native_bags(seqs, W); p = wm.plan(B, 0, 1, 50)
    assert np.array_equal(p, wm.plan(B, 0, 1, 50)) and not np.array_equal(p, wm.plan(B, 2, 1, 50))
    for u in (0, 17, 49):
        g = np.random.default_rng([20261027, 43, 0, 1, u]); a = int(g.integers(2)); b = int(g.integers(1))
        assert tuple(p[u]) == (B["a4c"][a], B["a5c"][b])


def test_local_tokens_marker_time():
    u = torch.randn(3, wm.N_TOKENS, 8); slots = torch.tensor([3, 15, 4]); h = wm.local_tokens(u, slots)
    for b, t in enumerate((1, 7, 2)):
        assert torch.equal(h[b], u[b, 1 + t * 49:1 + (t + 1) * 49])
    assert [wm.token_time(k) for k in range(16)] == [k // 2 for k in range(16)]


def test_head_parameters_init_and_first_gradient():
    torch.manual_seed(0); h = wm.GatedWitness(); x = torch.randn(4, 49, 768, requires_grad=True); s, a = h(x)
    assert sum(p.numel() for p in h.parameters()) == 197_761 and h.w.bias is None
    assert torch.equal(s, torch.full((4,), -2.0)) and torch.allclose(a.sum(1), torch.ones(4))
    s.sum().backward()
    assert (x.grad == 0).all() and (h.V.weight.grad == 0).all() and h.scorer.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("positive", ["dense", "max"])
@pytest.mark.parametrize("ties", [0, 2, 3])
def test_bag_loss_matches_float64_reference(positive, ties):
    g = np.random.default_rng(ties); s4, s5 = g.normal(size=6) * 3, g.normal(size=9) * 3; v4, v5 = g.dirichlet(np.ones(6)), g.dirichlet(np.ones(9))
    if ties:
        s5[g.choice(9, ties, replace=False)] = s5.max() + 1.0
    a, b = torch.tensor(s4, requires_grad=True), torch.tensor(s5, requires_grad=True); L, l4, l5 = wm.bag_loss(a, torch.tensor(v4), b, torch.tensor(v5), 0.7, 0.3, positive); L.backward()
    r = wm.reference_loss(s4, v4, s5, v5, 0.7, 0.3, positive)
    assert abs(float(L) - r["total"]) < 1e-12 and abs(float(l5) - r["l5"]) < 1e-12
    assert np.abs(a.grad.numpy() - r["g4"]).max() < 1e-12 and np.abs(b.grad.numpy() - r["g5"]).max() < 1e-12
    if positive == "max":
        assert (b.grad.numpy() != 0).sum() == max(ties, 1)


def test_one_instance_max_equals_dense():
    s4, v4, s5 = torch.tensor([0.3, -1.0]), torch.tensor([0.5, 0.5]), torch.tensor([0.8])
    d = wm.bag_loss(s4, v4, s5, torch.tensor([1.0]), 0.6, 0.4, "dense")[0]; m = wm.bag_loss(s4, v4, s5, torch.tensor([1.0]), 0.6, 0.4, "max")[0]
    assert abs(float(d) - float(m)) < 1e-7


def test_one_way_edit_and_merge():
    base = np.asarray([0, 1, 2, 3, 4, 2, 2]); out = wm.one_way(base, np.asarray([9.0, 9, 0.5, 9, 9, 0.0, -3]))
    assert out.tolist() == [0, 1, 3, 3, 4, 2, 2] and base.tolist() == [0, 1, 2, 3, 4, 2, 2]
    seq = np.asarray([0, 0, 2, 2, 3, 3, 2, 2, 2, 1]); pk = {"seq": np.zeros(5, int), "a": np.asarray([0, 2, 4, 6, 9]), "b": np.asarray([2, 4, 6, 9, 10]), "base": np.asarray([0, 2, 3, 2, 1])}
    ed = wm.edit_sequences({"q": seq}, pk, [{"id": "q"}], np.asarray([0, 3, 3, 2, 1]))["q"]
    assert ed.tolist() == [0, 0, 3, 3, 3, 3, 2, 2, 2, 1] and len(ir.runs(ed)) == 4 and seq.tolist()[2] == 2


def test_run_max_unique_rows_and_ineligible():
    pk = {"labels": np.zeros(3, int), "offsets": np.asarray([0, 2, 5, 6]), "rows": np.asarray([10, 11, 12, 12, 13, 14])}
    sc = np.asarray([-1.0, 0.5, -0.2, 2.0, -5.0]); r = wm.run_max(pk, sc, 10, np.asarray([True, True, False]))
    assert r.tolist() == [0.5, 2.0, -np.inf]
