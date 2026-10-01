"""Problem A helpers: local period, phase/pool/scramble change scores, percentile decoding, cut accounting."""
from __future__ import annotations

import numpy as np
import pytest

from echo_routing.realvideo.phase_change import (FALLBACK_SAMPLES, change_score, cut_counts, decode_cuts, local_period,
                                                 percentile_table, to_percentile)

P = 9  # samples per cycle


def dropout_stream(n: int = 200, change_at: int = 100) -> np.ndarray:
    """View A then view B; at phases 3-4 of every cycle the recognizer says C (evidence that recurs with the cycle)."""
    lab = np.where(np.arange(n) < change_at, 0, 1)
    lab[np.isin(np.arange(n) % P, (3, 4))] = 2
    return np.eye(3)[lab] * 0.9 + 0.1 / 3


def test_phase_score_ignores_recurring_dropouts_but_sees_the_change():
    prob = dropout_stream(); per = np.full(len(prob), float(P))
    phase, avail = change_score(prob, per, "phase")
    assert avail[1:].all()
    assert phase[20:80].max() < 0.05 and phase[100] > 0.6  # the 2 dropout phases carry no view evidence: 7/9 of 0.9
    assert int(np.argmax(phase)) in range(97, 104)


def test_pooling_whole_cycles_also_cancels_recurring_dropouts():
    prob = dropout_stream(); per = np.full(len(prob), float(P))
    pool, _ = change_score(prob, per, "pool")
    assert pool[20:80].max() < 0.05 and pool[100] > 0.6


def test_scrambled_correspondence_turns_dropouts_into_change_evidence():
    prob = dropout_stream(); per = np.full(len(prob), float(P))
    phase, _ = change_score(prob, per, "phase")
    scr, _ = change_score(prob, per, "scramble", seed=1)
    assert scr[20:80].mean() > phase[20:80].mean() + 0.1


def test_fallback_without_period_is_a_fixed_window_comparison():
    prob = dropout_stream(); per = np.full(len(prob), np.nan)
    s_phase, avail = change_score(prob, per, "phase")
    s_pool, _ = change_score(prob, per, "pool")
    assert not avail.any() and np.allclose(s_phase, s_pool)
    t = 100
    a, b = prob[t:t + FALLBACK_SAMPLES].mean(0), prob[t - FALLBACK_SAMPLES:t].mean(0)
    assert s_phase[t] == pytest.approx(0.5 * np.abs(a - b).sum())


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        change_score(np.ones((5, 2)) / 2, np.full(5, 3.0), "median")


def test_percentiles_are_monotone_and_bounded():
    table = percentile_table(np.arange(100.0))
    p = to_percentile(np.array([-1.0, 10.0, 50.0, 1000.0]), table)
    assert np.all(np.diff(p) >= 0) and p[0] == 0.0 and p[-1] == 1.0


def test_decode_cuts_merges_segments_with_the_same_label():
    prob = np.eye(2)[np.r_[np.zeros(30, int), np.ones(30, int)]]
    pct = np.zeros(60); pct[30] = 0.99; pct[10] = 0.995  # the cut at 10 splits two segments of the same view
    labels = decode_cuts(prob, pct, threshold=0.98)
    assert np.flatnonzero(np.diff(labels)).tolist() == [29]
    assert decode_cuts(prob, pct, threshold=0.999).tolist() == [0] * 60  # no cut above threshold: majority label


def test_cut_counts_classifies_every_cut():
    labels = np.zeros(100, int)
    labels[20:] = 1      # cut at 20, near the different-label join at 21: matched
    labels[23:] = 2      # cut at 23, also near that join: a double cut
    labels[50:] = 3      # cut at 50, at a same-label join
    labels[80:] = 4      # cut at 80, far from every join: internal extra
    c = cut_counts(labels, joins=np.array([21, 50]), diff_joins=np.array([21]), tol=5)
    assert (c.n_diff_joins, c.matched, c.n_same_joins, c.same_join_cuts, c.extra_internal, c.extra_near_join) == (1, 1, 1, 1, 1, 1)
    assert c.loc_error_samples == (1.0,)


def beating_blob(n: int = 150, period: float = 25.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[:48, :64]
    out = []
    for t in range(n):
        r = 8 + 4 * np.sin(2 * np.pi * t / period)
        out.append(np.where((yy - 24) ** 2 + (xx - 32) ** 2 < r ** 2, 200.0, 20.0) + rng.normal(0, 5, (48, 64)))
    return np.stack(out)


def test_local_period_from_image_motion():
    per = local_period(beating_blob(), fps=30)
    assert np.nanmedian(per) == pytest.approx(25, rel=0.05) and np.isfinite(per).mean() > 0.9
    noise = np.random.default_rng(2).normal(100, 20, (150, 48, 64))
    assert np.isnan(local_period(noise, fps=30)).all()
