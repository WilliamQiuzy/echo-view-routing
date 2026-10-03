"""Round-5 A4C/A5C decider: features, span decisions, training spans, weights and the classifier."""
from __future__ import annotations

import numpy as np
import pytest

from echo_routing.compose.recipe_builder import Fragment, Recipe
from echo_routing.temporal.evidence_decider import (A4C, A5C, MIN_SPAN, Decider, decide_spans, distribution_features,
                                                    mean_rule, span_weights, training_spans)


def _prob(n, cls, p=0.9):
    out = np.full((n, 5), (1 - p) / 4); out[:, cls] = p
    return out


def test_features_and_mean_rule():
    prob = _prob(10, A4C); prob[::3, :] = _prob(4, A5C, 0.95)          # intermittent A5C frames
    f = distribution_features(prob)
    assert f.shape == (17,) and np.isclose(f[:5].sum(), 1.0) and f[10] == np.float64(0.95) and 0 < f[15] < 1
    assert mean_rule(prob) == A4C and mean_rule(_prob(5, A5C)) == A5C


def test_decide_spans_changes_only_a4c_a5c_spans_and_keeps_limits():
    labels = np.array([0] * 5 + [2] * 6 + [2] * 4 + [1] * 5); cuts = [5, 11, 15]
    prob = np.vstack([_prob(5, 0), _prob(6, A4C), _prob(4, A5C), _prob(5, 1)])
    out = decide_spans(labels, cuts, prob, mean_rule)
    assert out.tolist() == [0] * 5 + [2] * 6 + [3] * 4 + [1] * 5     # the A4C-voted span of A5C evidence turns A5C
    assert decide_spans(labels, cuts, prob, lambda p: A5C)[:5].tolist() == [0] * 5
    with pytest.raises(ValueError):                                    # spans that do not match the labels stop the run
        decide_spans(labels, [11, 15], prob, mean_rule)                # span 0-11 holds PLAX and A4C


def test_training_spans_use_reference_units_jitter_and_clips():
    recipe = Recipe("t", "multi", (Fragment("a_1", 0, "orig", 0, 12), Fragment("b_1", A4C, "orig", 0, 15),
                                    Fragment("c_1", A5C, "orig", 0, 20), Fragment("d_1", 4, "orig", 0, 12)))
    prob = np.vstack([_prob(12, 0), _prob(15, A4C), _prob(20, A5C), _prob(12, 4)])
    spans = training_spans([(recipe, prob)], [("e_1", A5C, _prob(30, A5C)), ("f_1", 0, _prob(30, 0))], seed=0)
    labels = [lab for _, lab, _ in spans]
    assert set(labels) <= {A4C, A5C} and labels.count(A4C) >= 1 and labels.count(A5C) >= 2
    assert all(len(p) >= MIN_SPAN for p, _, _ in spans) and ("e_1" in {c for *_, c in spans}) and ("f_1" not in {c for *_, c in spans})
    assert training_spans([(recipe, prob)], [], seed=0)[0][0].shape == (15, 5)


def test_weights_equalise_clips_then_classes():
    w = span_weights(["a", "a", "a", "b", "c"], np.array([A4C, A4C, A4C, A4C, A5C]))
    assert np.isclose(w[:3].sum(), w[3]) and np.isclose(w[:4].sum(), w[4]) and np.isclose(w.mean(), 1.0)


def test_decider_learns_a_separable_rule_and_is_seeded():
    rng = np.random.default_rng(0); x = rng.normal(size=(200, 3)); y = np.where(x[:, 0] > 0, A5C, A4C); w = np.ones(200)
    d1 = Decider(seed=1).fit(x, y, w); d2 = Decider(seed=1).fit(x, y, w)
    assert ((d1.prob_a5c(x) >= 0.5) == (y == A5C)).mean() > 0.95 and np.allclose(d1.prob_a5c(x), d2.prob_a5c(x))
