"""A34: power-mean bag objective against NumPy, A29's mean-probability CE, and its invariances."""
import numpy as np
import pytest
import torch

import echo_routing.temporal.bag_objective as bo
from scripts.train_native_window_recognition import pooled_ce


def bags(seed=0, n=5, C=5):
    g = np.random.default_rng(seed); zs, ws, ys = [], [], []
    for k in range(n):
        t = int(g.integers(1, 9)); zs.append(g.normal(0, 2, (t, C))); w = g.random(t); ws.append(w / w.sum()); ys.append(int(g.integers(0, C)))
    ix = np.concatenate([np.full(len(w), k) for k, w in enumerate(ws)])
    return zs, ws, ys, torch.tensor(np.concatenate(zs)), torch.tensor(np.concatenate(ws)), torch.tensor(ix), torch.tensor(ys)


@pytest.mark.parametrize("r", [1.0, 2.0, 4.0])
def test_matches_numpy(r):
    zs, ws, ys, z, w, ix, y = bags()
    got = bo.bag_loss(z, w, ix, y, r).numpy(); ref = [bo.numpy_loss(a, b, c, r) for a, b, c in zip(zs, ws, ys)]
    assert np.allclose(got, ref, atol=1e-10)


def test_r1_is_mean_probability_ce():
    _, _, _, z, w, ix, y = bags(1)
    assert torch.allclose(bo.bag_loss(z, w, ix, y, 1.0), pooled_ce(z, w, ix, y), atol=1e-12)


@pytest.mark.parametrize("r", [1.0, 4.0])
def test_single_window_bag_is_plain_ce(r):
    z = torch.randn(3, 5, dtype=torch.float64); y = torch.tensor([0, 3, 4])
    assert torch.allclose(bo.bag_loss(z, torch.ones(3, dtype=torch.float64), torch.arange(3), y, r), torch.nn.functional.cross_entropy(z, y, reduction="none"), atol=1e-12)


def test_zero_weights_order_and_duplicates():
    _, _, _, z, w, ix, y = bags(2); base = bo.bag_loss(z, w, ix, y, 4.0)
    zp = torch.cat([z, torch.randn(4, 5, dtype=torch.float64)]); wp = torch.cat([w, torch.zeros(4, dtype=torch.float64)]); ip = torch.cat([ix, torch.tensor([0, 1, 2, 3])])
    assert torch.allclose(bo.bag_loss(zp, wp, ip, y, 4.0), base, atol=1e-12)                                  # zero-weight padding
    perm = torch.randperm(len(w)); assert torch.allclose(bo.bag_loss(z[perm], w[perm], ix[perm], y, 4.0), base, atol=1e-12)
    zd = torch.cat([z, z[:1]]); wd = torch.cat([w[:1] / 2, w[1:], w[:1] / 2]); idd = torch.cat([ix, ix[:1]])   # one window split in two copies
    assert torch.allclose(bo.bag_loss(zd, wd, idd, y, 4.0), base, atol=1e-12)


@pytest.mark.parametrize("z", [torch.zeros(4, 5, dtype=torch.float64), torch.tensor([[30.0, -30, -30, -30, -30], [-30, 30, -30, -30, -30]], dtype=torch.float64)])
def test_gradients_finite_uniform_and_two_class(z):
    z = z.clone().requires_grad_(True); n = z.shape[0]; w = torch.full((n,), 1.0 / n, dtype=torch.float64)
    bo.bag_loss(z, w, torch.zeros(n, dtype=torch.long), torch.tensor([1]), 4.0).sum().backward()
    assert torch.isfinite(z.grad).all()


def test_relabel_r1_equals_mean_argmax():
    from echo_routing.temporal.interval_recognition import relabel, runs
    g = np.random.default_rng(3); base = np.repeat([0, 2, 3, 2], [5, 7, 3, 9]); p = g.dirichlet(np.ones(5), len(base))
    assert np.array_equal(bo.relabel_r(base, p, 1.0, runs), relabel(base, p))
