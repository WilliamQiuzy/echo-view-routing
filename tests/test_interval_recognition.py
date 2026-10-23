"""Numerical and structural tests of A27's interval recognition (echo_routing/temporal/interval_recognition.py)."""
from __future__ import annotations

import numpy as np
import torch

import echo_routing.temporal.interval_recognition as ir


def np_objective(arm, z, q, items):
    """Independent numpy restatement over a list of (rows, weights, label)."""
    def lsm(x):
        m = x.max(1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))
    lp, lq = lsm(z), lsm(q); cls, kl = [], []
    for rows, w, y in items:
        if arm == "W":
            cls.append(-float(np.sum(w * lp[rows, y])))
        else:
            cls.append(-float(np.log(np.sum(w * np.exp(lp[rows, y])))))
        kl.append(float(np.sum(w * np.sum(np.exp(lq[rows]) * (lq[rows] - lp[rows]), 1))))
    return np.mean(cls), np.mean(kl)


def batch(items):
    rows = torch.from_numpy(np.concatenate([r for r, _, _ in items])); w = torch.from_numpy(np.concatenate([x for _, x, _ in items])).double()
    ix = torch.from_numpy(np.concatenate([np.full(len(r), j) for j, (r, _, _) in enumerate(items)])); y = torch.tensor([c for _, _, c in items])
    return rows, w, ix, y


def test_interval_rows_uneven_and_short():
    nearest = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    r, w = ir.interval_rows(nearest, 1, 6)                      # samples 1-5 -> rows 0 (x2), 1 (x2), 2 (x1)
    assert r.tolist() == [0, 1, 2] and np.allclose(w, [0.4, 0.4, 0.2]) and abs(w.sum() - 1) < 1e-12
    r, w = ir.interval_rows(nearest, 6, 8)                      # a short interval with no centre inside still maps
    assert r.tolist() == [2] and np.allclose(w, [1.0])


def test_objectives_match_numpy_and_differ():
    rng = np.random.default_rng(0); z = rng.normal(size=(9, 5)); q = rng.normal(size=(9, 5))
    items = [(np.array([0, 1, 2]), np.array([0.5, 0.3, 0.2]), 2), (np.array([3]), np.array([1.0]), 0), (np.array([4, 5, 6, 7, 8]), np.full(5, 0.2), 4)]
    rows, w, ix, y = batch(items); zt, qt = torch.from_numpy(z[rows.numpy()]), torch.from_numpy(q[rows.numpy()])
    remap = [(np.arange(len(r)) + sum(len(x) for x, _, _ in items[:j]), ww, c) for j, (r, ww, c) in enumerate(items)]
    for arm in ("W", "I"):
        o = ir.objective(arm, zt, qt, w, ix, y); c, k = np_objective(arm, z[rows.numpy()], q[rows.numpy()], remap)
        assert abs(float(o["cls"]) - c) < 1e-10 and abs(float(o["kl"]) - k) < 1e-10
    assert abs(float(ir.objective("W", zt, qt, w, ix, y)["cls"]) - float(ir.objective("I", zt, qt, w, ix, y)["cls"])) > 1e-3


def test_equal_predictions_give_equal_ce():
    z = torch.tensor([[1.0, 0.2, -0.5, 0.0, 0.3]] * 4).double(); q = torch.zeros(4, 5).double()
    w = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64); ix = torch.zeros(4, dtype=torch.long); y = torch.tensor([1])
    assert abs(float(ir.objective("W", z, q, w, ix, y)["cls"]) - float(ir.objective("I", z, q, w, ix, y)["cls"])) < 1e-12


def test_gradients_only_in_head():
    torch.manual_seed(0); emb = torch.randn(6, 512); frozen = torch.nn.Linear(512, 5)
    for p in frozen.parameters():
        p.requires_grad_(False)
    head = ir.copied_head(frozen.weight, frozen.bias); q = frozen(emb)
    o = ir.objective("I", head(emb), q, torch.full((6,), 1 / 3), torch.tensor([0, 0, 0, 1, 1, 1]), torch.tensor([2, 3])); o["loss"].backward()
    assert head.weight.grad is not None and frozen.weight.grad is None and sum(p.numel() for p in head.parameters()) == 2565
    zero = ir.copied_head(frozen.weight, frozen.bias)(emb)
    assert torch.equal(zero, q)


def test_relabel_merges_and_reproduces_base():
    base = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    p = np.zeros((8, 5)); p[:3, 0] = 1; p[3:5, 2] = 1; p[5:, 2] = 1        # the middle run relabelled to the right neighbour's class
    out = ir.relabel(base, p)
    assert out.tolist() == [0, 0, 0, 2, 2, 2, 2, 2] and len(ir.runs(out)) == 2  # its boundary with the third run disappears
    q = np.eye(5)[base] * 0.6 + 0.08                                          # zero update: p = q reproduces Base
    assert np.array_equal(ir.relabel(base, q), base)


def test_build_intervals_majority_and_ties():
    ref = np.array([0, 0, 1, 1, 1, 1]); base = np.array([0, 0, 0, 0, 1, 1]); nearest = np.array([0, 0, 1, 1, 2, 2])
    d = ir.build_intervals(ref, base, nearest)
    assert [x["label"] for x in d["T"]] == [0, 1] and [x["label"] for x in d["P"]] == [0, 1]
    tie = ir.build_intervals(np.array([1, 1, 0, 0]), np.array([3, 3, 3, 3]), np.array([0, 0, 1, 1]))["P"][0]
    assert tie["label"] == 0 and tie["purity"] == 0.5
