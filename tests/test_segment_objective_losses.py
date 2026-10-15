"""Mathematics of the segment objective round's losses (configs/experiments/segment_objective_v1.yaml).

The transcript-forward term is compared with an independent enumeration of every positive-duration stay/advance path
(T <= 6, X-Y-X included) in CPU float64, values and gradients within 1e-9. The other tests cover K = 1, T = K, the
probability bound, the rejections, B == O on all-native batches, F against masked weighted cross-entropy, the B
invariance to repeated identical-loss samples and padding.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from echo_routing.temporal.losses import segment_objective as so

TOL = 1e-9
ALPHA = np.array([0.7340, 0.9877, 1.0, 1.2529, 1.4164])


# ----------------------------------------------------------------------------------------------- an independent reference
def compositions(total: int, parts: int):
    """Every way of writing total as an ordered sum of `parts` positive integers."""
    for cuts in itertools.combinations(range(1, total), parts - 1):
        bounds = (0, *cuts, total)
        yield [bounds[i + 1] - bounds[i] for i in range(parts)]


def enumerated_nll(logp: torch.Tensor, y) -> torch.Tensor:
    """-log P(y | x) / T by listing every path: state k lasts d_k >= 1 samples, the d_k sum to T."""
    T = logp.shape[0]; scores = []
    for d in compositions(T, len(y)):
        states = np.repeat(np.arange(len(y)), d); cls = torch.as_tensor(np.asarray(y)[states])
        scores.append(logp[torch.arange(T), cls].sum())
    return -torch.logsumexp(torch.stack(scores), 0) / T


def rand_logits(T: int, seed: int, scale: float = 2.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(T, 5, generator=g, dtype=torch.float64) * scale).requires_grad_(True)


CASES = [((2,), 1), ((2,), 4), ((2, 3), 2), ((2, 3), 5), ((2, 3, 2), 3), ((2, 3, 2), 6), ((3, 2, 3), 5), ((0, 1, 2, 3, 4), 5),
         ((0, 1, 2, 3, 4), 6), ((4, 0, 4, 0), 6), ((1, 0), 6), ((3, 2, 3, 2, 3), 6)]


# ----------------------------------------------------------------------------------------------- runs and transcripts
def test_runs_merge_adjacent_equal_classes_and_keep_xyx():
    r = so.reference_runs(np.array([2, 2, 3, 3, 3, 2]))
    assert r.starts.tolist() == [0, 2, 5] and r.ends.tolist() == [2, 5, 6] and r.classes.tolist() == [2, 3, 2]
    assert r.transcript == (2, 3, 2) and r.length == 6
    same_view_join = so.reference_runs(np.array([1, 1, 1, 1]))          # two fragments of the same view: one run
    assert same_view_join.transcript == (1,) and same_view_join.ends.tolist() == [4]


def test_runs_reject_empty_and_out_of_range_labels():
    for bad in (np.array([], np.int64), np.array([0, 5]), np.array([-100, 1]), np.array([[0, 1]])):
        with pytest.raises(ValueError):
            so.reference_runs(bad)


# ----------------------------------------------------------------------------------------------- the forward recursion
@pytest.mark.parametrize("y,T", CASES)
def test_forward_equals_enumeration_value_and_gradient(y, T):
    for seed in range(3):
        z1 = rand_logits(T, seed); z2 = z1.detach().clone().requires_grad_(True)
        q = so.transcript_nll(F.log_softmax(z1, -1), y); q_ref = enumerated_nll(F.log_softmax(z2, -1), y)
        assert abs(float(q) - float(q_ref)) <= TOL
        (g1,) = torch.autograd.grad(q, z1); (g2,) = torch.autograd.grad(q_ref, z2)
        assert float((g1 - g2).abs().max()) <= TOL
        assert torch.isfinite(g1).all()


def test_single_state_is_mean_cross_entropy():
    z = rand_logits(6, 11); lp = F.log_softmax(z, -1)
    assert abs(float(so.transcript_nll(lp, (3,))) - float(-lp[:, 3].mean())) <= TOL


def test_t_equal_k_has_one_alignment():
    y = (0, 2, 1, 3); z = rand_logits(4, 5); lp = F.log_softmax(z, -1)
    want = -lp[torch.arange(4), torch.tensor(y)].sum() / 4
    assert abs(float(so.transcript_nll(lp, y)) - float(want)) <= TOL


def test_probability_at_most_one_and_q_at_most_the_reference_alignment_ce():
    rng = np.random.default_rng(3)
    for k in range(40):
        T = int(rng.integers(3, 30)); labels = np.sort(rng.integers(0, 3, T)); labels = np.array([(2, 3, 2)[i] for i in labels])
        r = so.reference_runs(labels); lp = F.log_softmax(rand_logits(T, 100 + k, 3.0), -1)
        q = float(so.transcript_nll(lp, r.transcript))
        assert q >= -TOL                                                       # P(y | x) <= 1
        mean_ce = float(-lp[torch.arange(T), torch.as_tensor(labels)].mean())  # the reference labels are one path
        assert q <= mean_ce + TOL


def test_rejections_are_explicit():
    lp = F.log_softmax(rand_logits(4, 0), -1)
    for y in ((), (1, 1), (0, 1, 0, 1, 0), (5,), (2, 2, 3)):
        with pytest.raises(ValueError):
            so.transcript_nll(lp, y)
    with pytest.raises(ValueError):
        so.transcript_nll(F.log_softmax(torch.zeros(4, 6, dtype=torch.float64), -1), (0,))


def test_long_sequences_stay_finite():
    T = 3000; z = rand_logits(T, 9, 6.0); y = (0, 1, 2, 3, 4, 3, 2)
    q = so.transcript_nll(F.log_softmax(z, -1), y); (g,) = torch.autograd.grad(q, z)
    assert torch.isfinite(q) and torch.isfinite(g).all()


# ----------------------------------------------------------------------------------------------- batch objectives
def batch(lengths, labels_list, seed=0, pad_value=0.0):
    B, T = len(lengths), max(lengths); g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, T, 5, generator=g, dtype=torch.float64) * 2
    for i, n in enumerate(lengths):
        logits[i, n:] = pad_value
    runs = [so.reference_runs(np.asarray(lab)) for lab in labels_list]
    return logits.requires_grad_(True), runs


def grads(loss, x):
    (g,) = torch.autograd.grad(loss, x)
    return g


def test_segment_and_order_are_equal_on_all_native_batches():
    lengths = [7, 3, 10, 5]; labels = [[2] * 7, [0] * 3, [4] * 10, [3] * 5]
    z, runs = batch(lengths, labels, 1)
    lb, _ = so.objective("segment", z, runs, ALPHA); lo, parts = so.objective("order", z, runs, ALPHA)
    assert abs(float(lb) - float(lo)) <= TOL and abs(parts["order"] - parts["segment"]) <= TOL
    assert float((grads(lb, z) - grads(lo, z)).abs().max()) <= TOL


def test_frame_term_is_masked_weighted_cross_entropy():
    lengths = [6, 4, 9]; labels = [[2, 2, 3, 3, 3, 2], [1] * 4, [0, 0, 4, 4, 4, 4, 4, 1, 1]]
    z, runs = batch(lengths, labels, 2, pad_value=7.0)
    lf, parts = so.objective("frame", z, runs, ALPHA)
    tgt = torch.full((3, 9), -100, dtype=torch.long)
    for i, lab in enumerate(labels):
        tgt[i, :len(lab)] = torch.as_tensor(lab)
    ce = F.cross_entropy(z.reshape(-1, 5), tgt.reshape(-1), weight=torch.as_tensor(ALPHA), ignore_index=-100)
    assert abs(float(lf) - float(ce)) <= 1e-12 and abs(parts["frame"] - float(ce)) <= 1e-12
    assert float((grads(lf, z) - grads(ce, z)).abs().max()) <= 1e-12


def test_segment_term_matches_its_definition():
    lengths = [6, 9]; labels = [[2, 2, 3, 3, 3, 2], [0, 0, 4, 4, 4, 4, 4, 1, 1]]
    z, runs = batch(lengths, labels, 4)
    lp = F.log_softmax(z, -1); num = den = 0.0
    for i, r in enumerate(runs):
        for s, t, c in zip(r.starts, r.ends, r.classes):
            num += ALPHA[c] * float(-lp[i, s:t, c].mean()); den += ALPHA[c]
    lb, parts = so.objective("segment", z, runs, ALPHA)
    assert abs(float(lb) - num / den) <= TOL and abs(parts["segment"] - num / den) <= TOL


def test_order_objective_matches_its_definition():
    lengths = [6, 9, 4]; labels = [[2, 2, 3, 3, 3, 2], [0, 0, 4, 4, 4, 4, 4, 1, 1], [3] * 4]
    z, runs = batch(lengths, labels, 5)
    lp = F.log_softmax(z, -1)
    D = np.array([ALPHA[r.classes].sum() for r in runs])
    q = np.array([float(enumerated_nll(lp[i, :lengths[i]], r.transcript)) for i, r in enumerate(runs)])
    lb, _ = so.objective("segment", z, runs, ALPHA); lo, parts = so.objective("order", z, runs, ALPHA)
    order = float((D * q).sum() / D.sum())
    assert abs(parts["order"] - order) <= TOL
    assert abs(float(lo) - (float(lb) + 0.5 * order) / 1.5) <= TOL


def test_segment_balance_ignores_repeated_identical_samples_but_frame_does_not():
    base = torch.tensor([[2.0, 0.1, -1.0, 0.3, 0.0], [0.5, 1.5, 0.2, -0.3, 0.0]], dtype=torch.float64)
    def seq(n_a):   # run of class 0 with n_a identical samples, then 3 identical samples of class 1
        logits = torch.cat([base[0].repeat(n_a, 1), base[1].repeat(3, 1)])[None]
        return logits, [so.reference_runs(np.array([0] * n_a + [1] * 3))]
    (z2, r2), (z5, r5) = seq(2), seq(5)
    lb2, _ = so.objective("segment", z2, r2, ALPHA); lb5, _ = so.objective("segment", z5, r5, ALPHA)
    lf2, _ = so.objective("frame", z2, r2, ALPHA); lf5, _ = so.objective("frame", z5, r5, ALPHA)
    assert abs(float(lb2) - float(lb5)) <= 1e-12
    assert abs(float(lf2) - float(lf5)) > 1e-3


@pytest.mark.parametrize("arm", so.ARMS)
def test_padding_values_never_reach_a_loss_or_gradient(arm):
    lengths = [6, 3, 9]; labels = [[2, 2, 3, 3, 3, 2], [1] * 3, [0, 0, 4, 4, 4, 4, 4, 1, 1]]
    z, runs = batch(lengths, labels, 6)
    z_alt = z.detach().clone()
    for i, n in enumerate(lengths):
        z_alt[i, n:] = torch.randn(9 - n, 5, dtype=torch.float64) * 50
    z_alt.requires_grad_(True)
    l1, _ = so.objective(arm, z, runs, ALPHA); l2, _ = so.objective(arm, z_alt, runs, ALPHA)
    assert float(l1) == float(l2)
    g1, g2 = grads(l1, z), grads(l2, z_alt)
    for i, n in enumerate(lengths):
        assert torch.equal(g1[i, :n], g2[i, :n]) and not g2[i, n:].any()


def test_runs_must_fit_the_batch():
    z, runs = batch([4, 4], [[1] * 4, [2] * 4], 7)
    with pytest.raises(ValueError):
        so.objective("frame", z, runs[:1], ALPHA)
    with pytest.raises(ValueError):
        so.objective("frame", z[:, :3], runs, ALPHA)
    with pytest.raises(ValueError):
        so.objective("ctc", z, runs, ALPHA)


def test_all_terms_report_every_arm_term():
    lengths = [6, 4]; labels = [[2, 2, 3, 3, 3, 2], [1] * 4]
    z, runs = batch(lengths, labels, 8)
    t = so.all_terms(z.detach(), runs, ALPHA)
    for arm in so.ARMS:
        assert abs(t[{"frame": "frame", "segment": "segment", "order": "objective_order"}[arm]] - float(so.objective(arm, z, runs, ALPHA)[0])) <= TOL
    lp = F.log_softmax(z.detach(), -1)
    means = [float(-lp[i, s:e, c].mean()) for i, r in enumerate(runs) for s, e, c in zip(r.starts, r.ends, r.classes)]
    assert abs(t["segment_mean_ce"] - float(np.mean(means))) <= TOL


def test_class_weights_must_be_five_finite_positive_values():
    assert so.check_class_weights([0.7, 1.0, 1.0, 1.2, 1.4]).dtype == np.float64
    for bad in ([1.0] * 4, [1.0, 1.0, 0.0, 1.0, 1.0], [1.0, 1.0, -1.0, 1.0, 1.0], [1.0, np.nan, 1.0, 1.0, 1.0], [1.0, np.inf, 1.0, 1.0, 1.0]):
        with pytest.raises(ValueError):
            so.check_class_weights(bad)


def test_float32_logits_are_accumulated_in_float64():
    z = torch.randn(2, 5, 5, dtype=torch.float32, requires_grad=True); runs = [so.reference_runs(np.array([0, 0, 1, 1, 1]))] * 2
    for arm in so.ARMS:
        loss, _ = so.objective(arm, z, runs, ALPHA)
        assert loss.dtype == torch.float64
        (g,) = torch.autograd.grad(loss, z)
        assert g.dtype == torch.float32 and torch.isfinite(g).all()
