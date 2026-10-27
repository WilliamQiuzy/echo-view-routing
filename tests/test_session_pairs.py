"""A42 CPU tests: native-only pair bank, extra draw stream, S / C partner maps, rank loss against the float64 reference,
gather max pool against nn.MaxPool3d and the skip-pool replacement on torchvision's MViTv2-S."""
import numpy as np
import pytest
import torch
import torch.nn as nn

import echo_routing.temporal.session_pairs as sp


def session_of(v: str) -> str:
    return v.rsplit("_", 1)[0]


def _seqs() -> list[dict]:
    nat = lambda v, c, n=7: {"id": v, "kind": "native", "labels": np.full(n, c)}  # noqa: E731
    return [{"id": "multi_0", "kind": "multi", "labels": np.array([2, 2, 3, 3])},       # constructed: never a candidate
            nat("sA_1", 2), nat("sA_2", 3), nat("sA_3", 3), nat("sB_1", 3), nat("sB_2", 2), nat("sC_1", 2),
            nat("sD_1", 0), nat("sE_1", 2), nat("sE_2", 3), nat("sF_1", 3), nat("sF_2", 2), nat("sF_3", 2), nat("sG_1", 4)]


def test_pair_bank_native_only_sorted_and_counted():
    seqs = _seqs(); bank = sp.pair_bank(seqs, session_of)
    assert bank["sessions"] == ["sA", "sB", "sE", "sF"]
    assert bank["a4c"] == [[1], [5], [8], [11, 12]] and bank["a5c"] == [[2, 3], [4], [9], [10]]
    assert sp.bank_counts(bank, seqs) == {"native_recordings": 13, "sessions": 4, "A4C_recordings": 5, "A5C_recordings": 5, "pairs": 2 + 1 + 1 + 2}


def test_pair_bank_rejects_mixed_native_codes():
    with pytest.raises(ValueError):
        sp.pair_bank([{"id": "s_1", "kind": "native", "labels": np.array([2, 3])}], session_of)


def test_extra_draw_stream_and_structure():
    seqs = _seqs(); bank = sp.pair_bank(seqs, session_of); rows_of = [np.arange(10 * i, 10 * i + 4) for i in range(len(seqs))]
    rows, ch = sp.extra_draw(bank, rows_of, 0, 1, 17)
    rng = np.random.default_rng([20261027, 42, 0, 1, 17, 2]); exp_ch = rng.choice(4, 4, replace=False); exp = [], []
    for k in exp_ch:
        i4 = bank["a4c"][k][int(rng.integers(len(bank["a4c"][k])))]; i5 = bank["a5c"][k][int(rng.integers(len(bank["a5c"][k])))]
        exp[0].append(int(rows_of[i4][int(rng.integers(4))])); exp[1].append(int(rows_of[i5][int(rng.integers(4))]))
    assert np.array_equal(ch, exp_ch) and rows.tolist() == exp[0] + exp[1]
    seq_of = rows // 10; sess = [session_of(seqs[i]["id"]) for i in seq_of]
    assert len(set(sess[:4])) == 4 and all(sess[4 + i] == sess[i] for i in range(4))
    assert all(seqs[i]["labels"][0] == sp.A4C for i in seq_of[:4]) and all(seqs[i]["labels"][0] == sp.A5C for i in seq_of[4:])
    assert all(sess[4 + i] != sess[sp.PARTNER["C"][i]] for i in range(4))
    assert np.array_equal(sp.extra_draw(bank, rows_of, 0, 1, 17)[0], rows) and not np.array_equal(sp.extra_draw(bank, rows_of, 0, 1, 18)[0], rows)


@pytest.mark.parametrize("arm", ["S", "C"])
@pytest.mark.parametrize("scale", [1.0, 3.0, 1e3])
def test_total_loss_matches_float64_reference(arm, scale):
    g = torch.Generator().manual_seed(3); z0 = torch.randn(16, 5, generator=g, dtype=torch.float64) * scale; y = torch.tensor([0, 1, 2, 3, 4, 2, 3, 1] + [2] * 4 + [3] * 4)
    z = z0.clone().requires_grad_(True); tot, ce, rk = sp.total_loss(z, y, sp.PARTNER[arm]); tot.backward(); r = sp.reference(z0.numpy(), y.numpy(), sp.PARTNER[arm])
    assert abs(float(ce) - r["ce"]) < 1e-9 and abs(float(rk) - r["rank"]) < 1e-8 and abs(float(tot) - r["total"]) < 1e-8
    assert np.abs(z.grad.numpy() - r["g_total"]).max() < 1e-8 and torch.isfinite(z.grad).all()


def test_partner_maps_and_rank_pairs():
    z = torch.zeros(16, 5, dtype=torch.float64); z[8:12, sp.A5C] = torch.tensor([0.0, 1.0, 2.0, 3.0]); z[12:16, sp.A5C] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    d5 = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64)
    assert torch.allclose(sp.rank_loss(z, sp.PARTNER["S"]), torch.nn.functional.softplus(-(d5 - torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64))).mean())
    assert torch.allclose(sp.rank_loss(z, sp.PARTNER["C"]), torch.nn.functional.softplus(-(d5 - torch.tensor([1.0, 2.0, 3.0, 0.0], dtype=torch.float64))).mean())
    assert sp.PARTNER["S"] == (0, 1, 2, 3) and sp.PARTNER["C"] == (1, 2, 3, 0)


def test_rank_gradient_touches_only_a4c_a5c_logits_of_extra_windows():
    z = torch.randn(16, 5, dtype=torch.float64, requires_grad=True); sp.rank_loss(z, sp.PARTNER["C"]).backward(); g = z.grad
    assert float(g[:8].abs().sum()) == 0 and float(g[:, [0, 1, 4]].abs().sum()) == 0 and float(g[8:, [2, 3]].abs().min()) > 0


def test_total_loss_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        sp.total_loss(torch.zeros(8, 5), torch.zeros(8, dtype=torch.long), sp.PARTNER["S"])


def test_gather_max_pool_equals_maxpool3d():
    mp = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)); gm = sp.GatherMaxPool3d(mp)
    x = torch.randn(6, 4, 3, 14, 14, dtype=torch.float64, requires_grad=True); g = torch.randn(6, 4, 3, 7, 7, dtype=torch.float64)
    ya = mp(x); (ya * g).sum().backward(); ga = x.grad.clone(); x.grad = None
    yb = gm(x); (yb * g).sum().backward()
    assert torch.equal(ya, yb) and torch.allclose(ga, x.grad, rtol=0, atol=1e-12) and not list(gm.parameters()) and not list(gm.buffers())


def test_skip_pool_replacement_on_mvit_keeps_state():
    from torchvision.models.video import mvit_v2_s
    net = mvit_v2_s(); keys = sorted(net.state_dict()); done = sp.deterministic_skip_pools(net)
    assert done == [1, 3, 14] and sorted(net.state_dict()) == keys and all(isinstance(net.blocks[i].pool_skip.pool, sp.GatherMaxPool3d) for i in done)
