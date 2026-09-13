from __future__ import annotations

import numpy as np
import pytest

from echo_routing.temporal.baselines.b0_frame_argmax import clip_majority, frame_argmax
from echo_routing.temporal.baselines.b1_smoothing import hysteresis, majority_filter, moving_average, smooth_labels
from echo_routing.temporal.baselines.b2_hmm import estimate_transitions, hmm_decode, sticky_transitions, viterbi
from echo_routing.temporal.baselines.b3_js_divergence import boundary_scores, detect_boundaries, js_divergence, labels_from_boundaries
from echo_routing.temporal.baselines.b_file import decide, file_score
from echo_routing.temporal.segments import (
    boundaries_from_labels, count_fragments, labels_to_segments, non_max_suppression, segments_to_labels,
)
from echo_routing.temporal.decode_select import route, segment_score, select_threshold


def onehot_seq(labels, c=3, conf=0.9):
    p = np.full((len(labels), c), (1 - conf) / (c - 1)); p[np.arange(len(labels)), labels] = conf
    return p


def test_segments_roundtrip():
    labels = np.array([0, 0, 1, 1, 1, 2])
    segs = labels_to_segments(labels)
    assert [(s.start, s.end, s.label) for s in segs] == [(0, 2, 0), (2, 5, 1), (5, 6, 2)]
    assert np.array_equal(segments_to_labels(segs, 6), labels)
    assert boundaries_from_labels(labels).tolist() == [2, 5]
    assert count_fragments(labels) == 2 and count_fragments(np.zeros(5, int)) == 0
    assert labels_to_segments(np.array([])) == []


def test_nms_keeps_strongest_within_radius():
    s = np.array([0, 0.9, 0.8, 0, 0, 0.5, 0.95, 0])
    assert non_max_suppression(s, 0.4, radius=1).tolist() == [1, 6]


def test_b0_argmax_and_majority():
    p = onehot_seq([0, 0, 1])
    assert frame_argmax(p).tolist() == [0, 0, 1] and clip_majority(p) == 0
    tie = np.array([[0.6, 0.4], [0.3, 0.7]])
    assert clip_majority(tie) == 1  # tie broken by mean prob (0.45 vs 0.55)


def test_bfile_scores_and_decision():
    p = onehot_seq([1, 1, 1, 0], conf=0.8)
    view, score = file_score(p)
    assert view == 1 and 0.6 < score < 0.7
    assert decide(p, tau=0.5).accepted and not decide(p, tau=0.9).accepted
    v2, s2 = file_score(p, "p10_agree"); assert v2 == 1 and s2 < score
    with pytest.raises(ValueError):
        file_score(p, "nope")


def test_moving_average_edges_and_window_validation():
    p = np.array([[1, 0], [0, 1], [0, 1], [0, 1]], float)
    ma = moving_average(p, 3)
    assert ma[0].tolist() == [0.5, 0.5] and ma[2].tolist() == [0, 1]
    assert np.array_equal(moving_average(p, 1), p)
    with pytest.raises(ValueError):
        moving_average(p, 2)


def test_majority_filter_removes_isolated_flicker():
    labels = np.array([0, 0, 1, 0, 0, 2, 2, 2])
    assert majority_filter(labels, 3).tolist() == [0, 0, 0, 0, 0, 2, 2, 2]


def test_hysteresis_requires_persistence():
    labels = np.array([0, 0, 1, 0, 0, 1, 1, 1, 0])
    assert hysteresis(labels, min_run=2).tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1]
    assert hysteresis(labels, min_run=1).tolist() == labels.tolist()


def test_smooth_labels_modes():
    p = onehot_seq([0, 0, 1, 0, 0, 1, 1, 1])
    assert smooth_labels(p, 3, 2, "prob").tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    assert smooth_labels(p, 3, 2, "majority").tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    with pytest.raises(ValueError):
        smooth_labels(p, 3, 2, "x")


def test_hmm_transitions_and_viterbi():
    t = estimate_transitions([np.array([0, 0, 0, 1, 1])], 2, pseudo=0.0)
    assert t[0, 0] == pytest.approx(2 / 3) and t[1, 1] == 1.0
    st = sticky_transitions(3, 0.9); assert np.allclose(st.sum(1), 1) and st[0, 0] == pytest.approx(0.9)
    p = onehot_seq([0, 0, 1, 0, 0, 1, 1, 1], conf=0.7)
    assert hmm_decode(p, sticky_transitions(3, 0.95)).tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    # explicit viterbi with no transition penalty reproduces argmax
    le = np.log(p); lt = np.log(np.full((3, 3), 1 / 3)); li = np.log(np.full(3, 1 / 3))
    assert viterbi(le, lt, li).tolist() == p.argmax(1).tolist()
    with pytest.raises(ValueError):
        sticky_transitions(3, 1.0)


def test_jsd_properties_and_boundaries():
    assert js_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(np.log(2))
    p = onehot_seq([0] * 6 + [1] * 6, conf=0.95)
    s = boundary_scores(p, half_window=3)
    assert s.argmax() == 6 and s[0] == 0
    assert detect_boundaries(p, 3, threshold=0.2, nms_radius=2).tolist() == [6]
    same = onehot_seq([0] * 12, conf=0.95)
    assert detect_boundaries(same, 3, 0.2, 2).size == 0
    assert labels_from_boundaries(p, np.array([6])).tolist() == [0] * 6 + [1] * 6


def test_segment_score_and_route():
    p = onehot_seq([0, 0, 0, 1, 1], conf=0.9)
    labels = p.argmax(1)
    segs = labels_to_segments(labels)
    assert segment_score(p, segs[0], "mean") == pytest.approx(0.9)
    assert segment_score(p, segs[0], "p10_agree") == pytest.approx(0.9)
    assert segment_score(p, segs[0], "max_softmax") == pytest.approx(0.9)
    routed = route(p, labels, tau=0.5, min_len=3)
    assert [r.accepted for r in routed] == [True, False]
    with pytest.raises(ValueError):
        segment_score(p, segs[0], "nope")


def test_select_threshold_targets_risk():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5]); wrong = np.array([0, 0, 0, 1, 1], bool); w = np.ones(5)
    assert select_threshold(scores, wrong, w, target_risk=0.0) == pytest.approx(0.7)
    assert select_threshold(scores, wrong, w, target_risk=0.25) == pytest.approx(0.6)
    assert select_threshold(scores, np.ones(5, bool), w, 0.05) is None


def test_select_threshold_is_max_coverage_not_first_violation():
    # one confident wrong sample at the top must not force an over-conservative threshold
    scores = np.array([0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]); wrong = np.array([1, 0, 0, 0, 0, 0, 0], bool); w = np.ones(7)
    assert select_threshold(scores, wrong, w, target_risk=0.2) == pytest.approx(0.4)  # 1/7 wrong at full coverage
    assert select_threshold(scores, wrong, w, target_risk=0.0) is None
    assert select_threshold(np.array([-np.inf, 0.5]), np.array([0, 0], bool), np.ones(2), 0.05) == pytest.approx(0.5)
