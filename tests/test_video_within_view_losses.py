"""A08 loss (configs/experiments/video_within_view_v1.yaml, loss; task sheet section 3): the truncated, earlier-endpoint
detached squared log-probability penalty on all neighbours (A) or on equal-label neighbours with per-group normalisation
(M); native and mixed groups, X-Y-X, same-class joins, zero and all-one masks, clipping, finite gradients, the detach on
independent log-probability tensors, float64 finite differences of a current endpoint with its previous endpoint held,
the coefficient-0 identity with A07's frame loss, the safe denominator and the monitor. CPU only; no data."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from echo_routing.temporal.losses import segment_objective as so
from echo_routing.temporal.losses import within_view as wv

ONES = np.ones(5)


def logits(seed: int, scale: float = 2.0, dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(2, 4, 5, generator=g, dtype=torch.float64) * scale).to(dtype).requires_grad_(True)


def expected(z: torch.Tensor, labels) -> tuple[float, float, np.ndarray]:
    """The task sheet's S_A and S_M written out with numpy loops (float64 from the fp32 log-softmax)."""
    ell = F.log_softmax(z.detach(), dim=-1).to(torch.float64).numpy(); lab = np.asarray(labels)
    d = np.minimum((ell[:, 1:] - ell[:, :-1]) ** 2, 16.0)
    s_a = np.mean([d[n].sum() / 15.0 for n in range(2)])
    per = []
    for n in range(2):
        m = (lab[n, 1:] == lab[n, :-1]).astype(float)
        per.append((m[:, None] * d[n]).sum() / (5.0 * m.sum()) if m.sum() > 0 else 0.0)
    return float(s_a), float(np.mean(per)), d


def regs(z, labels):
    ell = wv.log_probabilities(z); d = wv.edge_penalties(ell); mask = wv.equal_label_mask(labels)
    return wv.all_neighbours(d), wv.equal_label(d, mask), d, mask


# ----------------------------------------------------------------------------------------------- the equations
@pytest.mark.parametrize("labels", [[[1, 1, 1, 1], [0, 0, 3, 3]], [[2, 2, 2, 2], [0, 3, 3, 0]], [[4, 4, 4, 4], [1, 1, 1, 2]]])
def test_both_regularisers_follow_the_task_sheet_equations(labels):
    z = logits(1); s_a, s_m, d = expected(z, labels); a, m, d_t, _ = regs(z, labels)
    assert np.allclose(d_t.detach().numpy(), d, rtol=0, atol=1e-15)
    assert float(a) == pytest.approx(s_a, abs=1e-14) and float(m) == pytest.approx(s_m, abs=1e-14)


def test_the_mask_marks_equal_label_neighbours_only():
    m = wv.equal_label_mask(np.array([[3, 3, 3, 3], [0, 1, 1, 0]]))
    assert m.dtype == torch.float64 and m.tolist() == [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]          # native group; X-Y-X
    assert wv.equal_label_mask(np.array([[2, 2, 2, 2], [2, 2, 2, 2]])).tolist() == [[1.0] * 3] * 2  # a same-class join has no change


def test_masked_equals_all_on_groups_whose_labels_are_equal():
    z = logits(2, dtype=torch.float64); lab = [[1, 1, 1, 1], [4, 4, 4, 4]]
    a, m, _, _ = regs(z, lab)
    ga, = torch.autograd.grad(a, z, retain_graph=True); gm, = torch.autograd.grad(m, z)
    assert abs(float(a - m)) <= 1e-12 and float((ga - gm).abs().max()) <= 1e-12


def test_group_normalisation_keeps_the_native_weight_and_reweights_mixed_edges():
    z = logits(3, dtype=torch.float64); lab = [[0, 0, 0, 0], [1, 1, 2, 2]]; _, _, d, _ = regs(z, lab)
    g0 = d[0].sum() / 15.0; a = wv.all_neighbours(d); m = wv.equal_label(d, wv.equal_label_mask(lab))
    assert float(a) == pytest.approx(float((g0 + d[1].sum() / 15.0) / 2), abs=1e-15)
    assert float(m) == pytest.approx(float((g0 + (d[1, 0].sum() + d[1, 2].sum()) / 10.0) / 2), abs=1e-15)   # two selected edges of five classes


def test_a_group_without_selected_edges_adds_exact_zero_and_no_gradient():
    z = logits(4, dtype=torch.float64); lab = [[2, 2, 2, 2], [0, 1, 0, 1]]
    _, m, d, mask = regs(z, lab); g, = torch.autograd.grad(m, z)
    assert mask[1].sum() == 0 and float(m) == pytest.approx(float(d[0].sum() / 15.0 / 2), abs=1e-15)
    assert torch.equal(g[1], torch.zeros_like(g[1])) and float(g[0].abs().max()) > 0
    z2 = logits(5, dtype=torch.float64); _, m2, _, _ = regs(z2, [[0, 1, 0, 1], [3, 4, 3, 4]]); g2, = torch.autograd.grad(m2, z2)
    assert float(m2) == 0.0 and torch.equal(g2, torch.zeros_like(g2))


def test_the_safe_denominator_leaves_no_nan_even_at_coefficient_zero():
    z = logits(6); lab = np.array([[0, 1, 0, 1], [2, 3, 2, 3]])
    for coeff in (0.0, 0.15):
        loss, _ = wv.objective("masked", z, lab, coeff); g, = torch.autograd.grad(loss, z)
        assert torch.isfinite(loss) and bool(torch.isfinite(g).all())


# ----------------------------------------------------------------------------------------------- truncation, detach, finite differences
def test_truncation_caps_the_value_at_16_and_stops_its_gradient():
    ell = torch.zeros(1, 2, 5, dtype=torch.float64); ell[0, 1, 0] = -10.0; ell[0, 1, 1] = -1.0; ell.requires_grad_(True)
    d = wv.edge_penalties(ell); (g,) = torch.autograd.grad(d.sum(), ell)
    assert float(d[0, 0, 0]) == 16.0 and float(g[0, 1, 0]) == 0.0                              # (-10)^2 = 100 truncated
    assert float(d[0, 0, 1]) == 1.0 and float(g[0, 1, 1]) == pytest.approx(-2.0)               # 2 (ell_t - ell_t-1) below the cap


def test_the_earlier_endpoint_is_detached_on_independent_log_probabilities():
    g = torch.Generator().manual_seed(7); ell = torch.randn(2, 4, 5, generator=g, dtype=torch.float64).requires_grad_(True)
    d = wv.edge_penalties(ell); (grad,) = torch.autograd.grad(wv.all_neighbours(d), ell)
    assert torch.equal(grad[:, 0], torch.zeros_like(grad[:, 0]))                               # position 0 is only an earlier endpoint
    want = 2 * (ell[:, 1:] - ell[:, :-1]).detach() * (d.detach() < 16).double() / 15.0 / 2
    assert torch.allclose(grad[:, 1:], want, rtol=0, atol=1e-15)                               # current endpoints only


def test_cross_entropy_still_trains_the_earlier_outputs():
    z = logits(8); loss, _ = wv.objective("all", z, np.array([[1, 1, 1, 1], [0, 0, 2, 2]])); (g,) = torch.autograd.grad(loss, z)
    assert float(g[:, 0].abs().max()) > 0 and bool(torch.isfinite(g).all())


def test_finite_differences_of_a_current_endpoint_with_its_previous_endpoint_held():
    g = torch.Generator().manual_seed(9); base = (torch.randn(2, 4, 5, generator=g, dtype=torch.float64) * 0.5)
    lab = np.array([[1, 1, 1, 1], [2, 2, 3, 3]])

    def value(x: torch.Tensor, arm: str) -> torch.Tensor:
        d = wv.edge_penalties(x)
        return wv.all_neighbours(d) if arm == "all" else wv.equal_label(d, wv.equal_label_mask(lab))
    for arm in wv.ARMS:
        x = base.clone().requires_grad_(True); (grad,) = torch.autograd.grad(value(x, arm), x)
        for n, c in ((0, 0), (1, 3), (1, 4)):
            h = 1e-6; xp, xm = base.clone(), base.clone(); xp[n, 3, c] += h; xm[n, 3, c] -= h    # t = 3 is only a current endpoint
            fd = float((value(xp, arm) - value(xm, arm)) / (2 * h))
            assert fd == pytest.approx(float(grad[n, 3, c]), rel=1e-6, abs=1e-9)


# ----------------------------------------------------------------------------------------------- the objective and coefficient 0
@pytest.mark.parametrize("labels", [[[1, 1, 1, 1], [3, 3, 3, 3]], [[2, 2, 2, 2], [0, 0, 3, 3]], [[4, 4, 4, 4], [0, 1, 1, 0]]])
def test_coefficient_zero_equals_a07_frame_loss_in_value_and_logit_gradient(labels):
    lab = np.array(labels); runs = [so.reference_runs(x) for x in lab]
    for arm in wv.ARMS:
        z1, z2 = logits(10), logits(10)
        ce, _ = so.objective("frame", z1, runs, ONES); (g1,) = torch.autograd.grad(ce, z1)
        loss, parts = wv.objective(arm, z2, lab, 0.0); (g2,) = torch.autograd.grad(loss, z2)
        assert float(loss) == float(ce) and torch.equal(g1, g2) and parts["coefficient"] == 0.0


def test_the_objective_is_ce_plus_the_coefficient_times_the_arms_regulariser():
    lab = np.array([[2, 2, 2, 2], [0, 0, 3, 3]]); runs = [so.reference_runs(x) for x in lab]
    for arm in wv.ARMS:
        z = logits(11); loss, parts = wv.objective(arm, z, lab); (g,) = torch.autograd.grad(loss, z)
        z1 = logits(11); ce, _ = so.objective("frame", z1, runs, ONES)
        _, _, d, mask = regs(z1, lab); reg = wv.all_neighbours(d) if arm == "all" else wv.equal_label(d, mask)
        (gw,) = torch.autograd.grad(ce + 0.15 * reg, z1)
        assert wv.COEFFICIENT == 0.15 and wv.TRUNCATION == 16.0
        assert float(loss) == pytest.approx(float(ce) + 0.15 * float(reg), abs=1e-14) and torch.allclose(g, gw, rtol=0, atol=1e-7)
        assert parts["ce"] == pytest.approx(float(ce)) and parts["regulariser"] == pytest.approx(float(reg))
    with pytest.raises(ValueError):
        wv.objective("both", logits(12), lab)


def test_the_monitor_logs_both_regularisers_and_the_edge_sums_without_gradient_or_rng():
    z = logits(13); lab = np.array([[1, 1, 1, 1], [0, 0, 3, 3]]); state = torch.get_rng_state()
    mon = wv.monitor(z, lab); a, m, d, mask = regs(z, lab); edge = d.detach().mean(-1)
    assert torch.equal(torch.get_rng_state(), state)
    assert mon["S_A"] == pytest.approx(float(a)) and mon["S_M"] == pytest.approx(float(m))
    assert (mon["same_edges"], mon["diff_edges"], mon["zero_edge_groups"]) == (5, 1, 0)
    assert mon["same_sum"] == pytest.approx(float((edge * mask).sum())) and mon["diff_sum"] == pytest.approx(float((edge * (1 - mask)).sum()))
    assert z.grad is None
