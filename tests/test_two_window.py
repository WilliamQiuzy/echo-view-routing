"""Two-window component audit: the B3 configuration must reproduce B3 exactly; windows and scores behave as stated."""
from __future__ import annotations

import numpy as np
import pytest

from echo_routing.realvideo.two_window import detect_labels, quantile_thresholds, two_window_score, window_lengths
from echo_routing.temporal.baselines.b3_js_divergence import boundary_scores, detect_boundaries, labels_from_boundaries


def noisy_stream(seed: int = 0, n: int = 120) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lab = np.r_[np.zeros(40, int), np.ones(50, int), np.full(30, 2)]
    p = np.eye(4)[lab] * 0.7 + rng.dirichlet(np.ones(4), size=n) * 0.3
    return p / p.sum(1, keepdims=True)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_b3_configuration_reproduces_b3(seed):
    prob = noisy_stream(seed); lengths, _ = window_lengths(len(prob), fixed=5)
    score = two_window_score(prob, lengths, "js")
    assert np.allclose(score, boundary_scores(prob, 5))
    for thr in (0.05, 0.15, 0.3):
        ours = detect_labels(prob, score, thr, lengths)
        theirs = labels_from_boundaries(prob, detect_boundaries(prob, 5, thr, 5))
        assert np.array_equal(ours, theirs)


def test_tv_score_is_total_variation_of_window_means():
    prob = noisy_stream(); lengths, _ = window_lengths(len(prob), fixed=9)
    s = two_window_score(prob, lengths, "tv")
    t = 40
    assert s[t] == pytest.approx(0.5 * np.abs(prob[t - 9:t].mean(0) - prob[t:t + 9].mean(0)).sum())
    with pytest.raises(ValueError):
        two_window_score(prob, lengths, "kl")


def test_period_adaptive_windows_and_fallback():
    per = np.r_[np.full(5, np.nan), np.full(5, 8.0)]
    lengths, avail = window_lengths(10, period_samples=per, k=0.5, fallback=9)
    assert lengths.tolist() == [9] * 5 + [4] * 5 and avail.tolist() == [False] * 5 + [True] * 5
    fixed, none = window_lengths(10, fixed=7)
    assert fixed.tolist() == [7] * 10 and not none.any()


def test_quantile_thresholds_are_monotone():
    th = quantile_thresholds(np.arange(1000.0), (0.9, 0.99))
    assert th[0] < th[1] and th[0] == pytest.approx(899.1)
