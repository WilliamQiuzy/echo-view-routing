"""CPU tests for A44's module (echo_routing/temporal/risk_constrained.py): lists, plan, membership, head, objectives, dual, deploy."""
import numpy as np
import pytest
import torch

import echo_routing.temporal.risk_constrained as rc


def _pk():
    # 6 P runs: base / reference labels and memberships (global rows with offset 100)
    rows = np.asarray([100, 101, 101, 102, 103, 104, 105, 106, 107, 107, 108, 109])
    offsets = np.asarray([0, 2, 3, 5, 7, 9, 12])
    weights = np.asarray([0.5, 0.5, 1.0, 0.25, 0.75, 0.5, 0.5, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 3])
    return {"rows": rows, "offsets": offsets, "weights": weights, "labels": np.asarray([3, 2, 2, 2, 0, 2]), "base": np.asarray([2, 2, 2, 3, 2, 2]),
            "a": np.asarray([0, 4, 6, 10, 14, 18]), "b": np.asarray([4, 6, 10, 14, 18, 24])}


def test_training_lists_in_p_order():
    L = rc.training_lists(_pk())
    assert L["eligible"].tolist() == [0, 1, 2, 4, 5] and L["positives"].tolist() == [0] and L["negatives"].tolist() == [1, 2, 5] and L["other"].tolist() == [4]


def test_plan_is_explicit_covers_negatives_without_same_update_duplicates():
    pos, neg = np.asarray([3, 8, 11]), np.arange(20, 31)
    pl = rc.plan(pos, neg, 0, 1, 50)
    assert pl.shape == (50, 4) and pl.dtype == np.int64 and np.array_equal(pl, rc.plan(pos, neg, 0, 1, 50)) and not np.array_equal(pl, rc.plan(pos, neg, 2, 1, 50))
    perm = np.random.default_rng([20261028, 44, 0, 1, 1000]).permutation(neg)
    for u in (0, 3, 49):
        assert pl[u, 0] == int(np.random.default_rng([20261028, 44, 0, 1, u]).choice(pos))
        assert pl[u, 1:].tolist() == perm[(3 * u + np.arange(3)) % 11].tolist()
    assert set(pl[:, 1:].ravel().tolist()) == set(neg.tolist()) and all(len(set(r[1:].tolist())) == 3 for r in pl)
    with pytest.raises(ValueError):
        rc.plan(pos, neg[:2], 0, 0, 5)


def test_membership_and_explicit_loop():
    pk = _pk(); r, w = rc.members(pk, 2, 100)
    assert r.tolist() == [2, 3] and w.tolist() == [0.25, 0.75]
    nearest = np.asarray([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 9, 10, 11, 11, 11])
    er, ew = rc.explicit_membership(nearest, 7, 10)
    assert er.tolist() == [3] and ew.tolist() == [1.0]
    er, ew = rc.explicit_membership(nearest, 2, 6)
    assert er.tolist() == [1, 2] and np.allclose(ew, [0.5, 0.5])
    rows, ws, seg = rc.occurrences(pk, [0, 1, 2, 5], 100)
    assert rows.tolist() == [0, 1, 1, 2, 3, 7, 8, 9] and seg.tolist() == [0, 0, 1, 2, 2, 3, 3, 3] and abs(ws.sum() - 4) < 1e-12
    ex = rc.exposure(pk, np.asarray([[0, 1, 2, 5], [0, 2, 5, 1]]))
    assert ex["occurrences_total"] == 16 and ex["occurrences_max_update"] == 8 and ex["same_update_duplicate_negatives"] == 0


def test_head_init_scores_minus_2_and_first_gradient():
    torch.manual_seed(0); h = rc.ScalarHead(); e = torch.randn(5, 512, requires_grad=True); s = h(e)
    assert sum(p.numel() for p in h.parameters()) == 513 and torch.equal(s, torch.full((5,), -2.0))
    s.sum().backward()
    assert (e.grad == 0).all() and h.fc.weight.grad.abs().sum() > 0 and float(h.fc.bias.grad) == 5.0
    with pytest.raises(ValueError):
        h(torch.randn(2, 768))


@pytest.mark.parametrize("arm", ["C", "N"])
def test_objective_matches_float64_reference(arm):
    g = np.random.default_rng(7); n = [3, 1, 4, 2]; seg = np.concatenate([np.full(k, i) for i, k in enumerate(n)]); s = g.normal(size=10) * 2
    v = np.concatenate([g.dirichlet(np.ones(k)) for k in n]); lam = 2.5
    st = torch.tensor(s, requires_grad=True); r = rc.run_means(st, torch.tensor(v), torch.tensor(seg), 4); L, A, B, H = rc.objective(r, arm, lam); L.backward()
    ref = rc.reference(s, v, seg, arm, lam)
    assert np.abs(r.detach().numpy() - ref["r"]).max() < 1e-12 and abs(L.item() - ref["loss"]) < 1e-12 and abs(H.item() - ref["H"]) < 1e-12 and abs(B.item() - ref["B"]) < 1e-12
    assert np.abs(st.grad.numpy() - ref["g"]).max() < 1e-12


def test_hinge_kinks_and_dual_recurrence():
    # r = -1 and r = 0: ReLU derivative 0 at the kink, 1 above; huge scores stay finite
    r = torch.tensor([0.0, -1.0, 0.0, -1e3], requires_grad=True); L, A, B, H = rc.objective(r, "N", 1.0); L.backward()
    assert float(H) == pytest.approx(1.0 / 3) and r.grad.tolist() == pytest.approx([-0.5, 0.0, 1.0 / 3, 0.0])
    r2 = torch.tensor([1e3, 1e3, -1e3, 5.0], requires_grad=True); L2, A2, B2, H2 = rc.objective(r2, "C", 0.0); L2.backward()
    assert torch.isfinite(L2) and float(A2) == 0.0 and torch.isfinite(r2.grad).all()
    assert rc.dual_update(1.0, 0.01) == 1.0 and rc.dual_update(1.0, 0.51) == pytest.approx(1.05) and rc.dual_update(0.02, 0.0) == pytest.approx(0.019)
    assert rc.dual_update(0.0, 0.0) == 0.0 and rc.dual_update(99.995, 1.0) == 100.0 and rc.dual_update(100.0, 0.0) == pytest.approx(99.999)
    ref = rc.reference([0.0, -1.0, 0.0, -2.0], [1, 1, 1, 1], [0, 1, 2, 3], "N", 3.0)
    assert ref["g"].tolist() == pytest.approx([-0.5, 0.0, 1.0, 0.0]) and ref["lambda_next"] == pytest.approx(3.0 + 0.1 * (1 / 3 - 0.01))


def test_weighted_mean_deploy_explicit_and_one_way():
    pk = _pk(); sc = np.arange(12, dtype=np.float64) - 3.0; el = pk["base"] == 2
    r = rc.run_weighted_mean(pk, sc, 100, el)
    assert r[3] == -np.inf and r[2] == pytest.approx(0.25 * (-1) + 0.75 * 0) and r[0] == pytest.approx(0.5 * -3 + 0.5 * -2)
    nearest = np.asarray([100, 100, 101, 101, 101, 101, 102, 103, 103, 103, 104, 104, 105, 105, 106, 106, 107, 107, 107, 107, 108, 108, 109, 109])
    for i in (0, 1, 2, 4, 5):
        assert rc.run_weighted_mean_explicit(pk, sc, 100, i, nearest) == pytest.approx(r[i])
    assert r[4] == pytest.approx(3.5) and r[5] == pytest.approx(5.0)
    out = rc.one_way(pk["base"], r)
    assert out.tolist() == [2, 2, 2, 3, 3, 3] and pk["base"].tolist() == [2, 2, 2, 3, 2, 2]
    fe = rc.feasibility(r, np.asarray([1, 2, 5]), np.asarray([0]))
    assert fe["false_promotions"] == 1 and fe["positives_recovered"] == 0 and fe["H_full_fit"] == pytest.approx(np.mean([max(0, 1 + r[1]), max(0, 1 + r[2]), max(0, 1 + r[5])]))
