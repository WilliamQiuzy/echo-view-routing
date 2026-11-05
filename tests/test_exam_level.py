"""EX1 pure-logic tests. Synthetic inputs only; no media, labels or model files are read."""
import itertools

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from exam_level import clip_models as cm
from exam_level import metrics as mx
from exam_level import selective as sel
from exam_level import streams as st


def native(video_id, label, length):
    return {'id': video_id, 'kind': 'native', 'labels': np.full(length, label, np.int64),
            'refs': [(video_id, i * 3, 1.0) for i in range(length)], 'joins': np.zeros(0, np.int64)}


def session_bank():
    clips = []
    for s in range(6):
        clips += [native(f'S{s}_{k}', k % 5, 4 + k) for k in [0, 1, 2, 10]]
    return clips


# --- streams ---

def test_declared_banks_include_three_distinct_rewiring_draws():
    bank = session_bank()
    draws = [st.exam_streams(bank, mode, 'selection', seed=seed) for name, (mode, seed) in st.BANKS.items() if name.startswith('rewired')]
    assert len(draws) == 3
    assert draws[0][0]['clips'] != draws[1][0]['clips'] or draws[0][1]['clips'] != draws[1][1]['clips']


def test_clip_index_sorts_numerically_not_by_name():
    clips = [native('S_10', 0, 3), native('S_2', 1, 3), native('S_1', 2, 3)]
    stream = st.exam_streams(clips, 'exam', 'selection')[0]
    assert [c[0] for c in stream['clips']] == ['S_1', 'S_2', 'S_10']


def test_session_and_timestamp_identity():
    assert st.session_of('2022-01-02_03-04-05_320x240_7') == '2022-01-02_03-04-05_320x240'
    assert st.clip_index('2022-01-02_03-04-05_320x240_17') == 17
    assert st.timestamp_of('2022-01-02_03-04-05_001_320x240_7') == '2022-01-02_03-04-05'


def test_exam_stream_concatenates_whole_clips_with_correct_joins():
    clips = [native('S_0', 0, 4), native('S_1', 3, 6)]
    stream = st.exam_streams(clips, 'exam', 'selection')[0]
    assert stream['joins'].tolist() == [4]
    assert stream['labels'].tolist() == [0] * 4 + [3] * 6
    assert stream['refs'][4] == ('S_1', 0, 1.0)
    assert stream['clips'] == [('S_0', 0, 4), ('S_1', 4, 10)]


@pytest.mark.parametrize('mode', ['exam', 'shuffled', 'rewired'])
def test_every_mode_keeps_the_same_clip_multiset_and_lengths(mode):
    bank = session_bank()
    streams = st.exam_streams(bank, mode, 'selection')
    used = sorted(c[0] for s in streams for c in s['clips'])
    assert used == sorted(q['id'] for q in bank)
    assert sorted(len(s['clips']) for s in streams) == [4] * 6


def test_shuffled_stays_inside_session():
    for stream in st.exam_streams(session_bank(), 'shuffled', 'selection'):
        assert len({st.session_of(c[0]) for c in stream['clips']}) == 1


def test_rewired_has_no_same_session_join():
    for stream in st.exam_streams(session_bank(), 'rewired', 'selection'):
        sessions = [st.session_of(c[0]) for c in stream['clips']]
        assert all(a != b for a, b in zip(sessions, sessions[1:]))


def test_streams_are_deterministic_and_differ_by_half():
    bank = session_bank()
    first = st.exam_streams(bank, 'rewired', 'selection')
    assert [s['clips'] for s in first] == [s['clips'] for s in st.exam_streams(bank, 'rewired', 'selection')]
    assert [s['clips'] for s in first] != [s['clips'] for s in st.exam_streams(bank, 'rewired', 'calibration')]


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        st.exam_streams(session_bank(), 'random', 'selection')


def test_expand_clip_labels_follows_spans():
    spans = [('a', 0, 2), ('b', 2, 5)]
    assert st.expand_clip_labels(spans, [3, 1]).tolist() == [3, 3, 1, 1, 1]
    with pytest.raises(ValueError):
        st.expand_clip_labels(spans, [3])


# --- metrics ---

def test_stream_counts_on_a_perfect_prediction():
    y = np.array([0] * 20 + [3] * 8 + [2] * 20)
    row = mx.stream_counts(y, y)
    assert (row['M'], row['P'], row['T']) == (3, 3, 3)
    assert row['short16_true'] == 1 and row['short16_fn'] == 0 and row['short16_fp'] == 0
    assert row['change_true'] == 2 and row['change_matched'] == 2
    assert row['correct'] == row['n'] == 48


def test_stream_counts_counts_missed_short_and_false_short():
    y = np.array([0] * 20 + [3] * 8 + [0] * 20)
    missed = mx.stream_counts(y, np.zeros(48, int))
    assert missed['short16_fn'] == 1 and missed['a5c_to_a4c'] == 0 and missed['change_matched'] == 0
    false = mx.stream_counts(np.zeros(48, int), y)
    assert false['short16_fp'] == 1


def test_a4c_a5c_position_confusion_directions():
    y = np.array([2, 2, 3, 3, 3])
    p = np.array([3, 2, 2, 2, 3])
    row = mx.stream_counts(y, p)
    assert row['a4c_to_a5c'] == 1 and row['a5c_to_a4c'] == 2


def test_merge_removes_a4c_a5c_distinction():
    y = np.array([2] * 30 + [3] * 30)
    p = np.array([2] * 60)
    assert mx.segment_f1(mx.stream_counts(y, p)) < 1
    assert mx.segment_f1(mx.stream_counts(mx.merge_apical(y), mx.merge_apical(p))) == 1


def test_macro_f1_matches_hand_value():
    conf = np.array([[2, 0], [1, 1]])
    assert mx.macro_f1(conf) == pytest.approx((0.8 + 2 / 3) / 2)


def test_macro_f1_on_stacked_confusions_matches_single():
    conf = np.random.default_rng(0).integers(0, 9, (5, 5))
    stacked = np.stack([conf, conf * 2])
    assert np.allclose(mx.macro_f1(stacked), mx.macro_f1(conf))


def test_cluster_bootstrap_identical_inputs_give_zero_interval():
    rows = np.random.default_rng(1).integers(1, 9, (20, 3)).astype(float)
    clusters = [f'c{i // 2}' for i in range(20)]
    out = mx.paired_delta(rows, rows, clusters, lambda x: x[..., 0] / x[..., 1], n_boot=200, seed=3)
    assert out['delta'] == 0 and out['ci95'] == [0.0, 0.0]


def test_cluster_bootstrap_resamples_whole_clusters():
    a = np.array([[1.0], [1.0], [0.0], [0.0]])
    b = np.zeros((4, 1))
    values = mx.paired_replicates(a, b, ['x', 'x', 'y', 'y'], lambda v: v[..., 0], n_boot=500, seed=1)
    assert np.unique(values).tolist() == [0.0, 2.0, 4.0]


# --- chain model ---

def random_chain(seed, n=5):
    rng = np.random.default_rng(seed)
    start = np.log(rng.dirichlet(np.ones(n)))
    trans = np.log(rng.dirichlet(np.ones(n), size=n))
    return rng, start, trans


def path_score(E, y, start, trans):
    return E[np.arange(len(y)), y].sum() + start[y[0]] + trans[y[:-1], y[1:]].sum()


@pytest.mark.parametrize('seed', range(6))
def test_viterbi_matches_brute_force(seed):
    rng, start, trans = random_chain(seed, 3)
    E = rng.normal(size=(4, 3))
    best = max(itertools.product(range(3), repeat=4), key=lambda y: path_score(E, np.array(y), start, trans))
    assert tuple(cm.viterbi(E, start, trans).tolist()) == best


@pytest.mark.parametrize('seed', range(4))
def test_conditional_probabilities_sum_to_one(seed):
    rng, start, trans = random_chain(seed, 3)
    E = rng.normal(size=(4, 3))
    total = sum(np.exp(cm.conditional_ll(E, np.array(y), start, trans)) for y in itertools.product(range(3), repeat=4))
    assert total == pytest.approx(1.0)


def test_fit_chain_uses_order_and_smoothing():
    start, trans = cm.fit_chain([np.array([0, 1, 1])], n=2, alpha=1.0)
    assert np.exp(start).tolist() == pytest.approx([2 / 3, 1 / 3])
    assert np.exp(trans[0]).tolist() == pytest.approx([1 / 3, 2 / 3])
    assert np.exp(trans[1]).tolist() == pytest.approx([1 / 3, 2 / 3])


def test_bag_chain_has_no_order_information():
    seqs = [np.array([0, 0, 0, 1, 1, 1])] * 4
    _, trans = cm.fit_bag_chain(seqs, n=2, seed=0, n_perm=400)
    assert abs(np.exp(trans[0, 1]) - np.exp(trans[1, 0])) < 0.05


def test_temperature_grid_prefers_flatter_scale_for_overconfident_evidence():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 600)
    wrong = rng.random(600) < .3
    shown = np.where(wrong, 1 - y, y)
    E = np.zeros((600, 2))
    E[np.arange(600), shown] = 20.0
    start = np.log([.5, .5])
    trans = np.log(np.full((2, 2), .5))
    tau = cm.fit_temperature([E[i:i + 1] for i in range(600)], [y[i:i + 1] for i in range(600)], start, trans)
    assert tau == 32


def test_decode_permuted_returns_labels_in_original_clip_order():
    E = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [5.0, 0.0, 0.0]])
    start, trans = cm.prior_chain(np.full(3, 1 / 3))
    for p in ([0, 1, 2, 3], [3, 1, 0, 2], [2, 3, 1, 0]):
        assert cm.decode_permuted(E, p, 1.0, start, trans).tolist() == [0, 1, 2, 0]


def test_prior_chain_decodes_like_independent_argmax():
    pi = np.array([.2, .8])
    start, trans = cm.prior_chain(pi)
    E = np.array([[3.0, 0.0], [0.0, 3.0], [0.1, 0.0]])
    assert cm.viterbi(E, start, trans).tolist() == [0, 1, 1]


# --- offset model ---

@pytest.mark.parametrize('tau', [0.0, 0.7, 3.0])
def test_offset_loglik_matches_multivariate_normal(tau):
    p = cm.OffsetParams(-2.0, 1.5, 0.8, 1.3, tau)
    d = np.array([-1.0, 0.2, 2.0])
    bits = np.array([0, 1, 1])
    m = np.array([-2.0, 1.5, 1.5])
    cov = np.diag([.8 ** 2, 1.3 ** 2, 1.3 ** 2]) + tau ** 2
    assert cm.offset_loglik(d, bits, p) == pytest.approx(multivariate_normal(m, cov).logpdf(d))


def test_fit_offset_recovers_simulated_parameters():
    rng = np.random.default_rng(5)
    ds, ys = [], []
    for _ in range(1500):
        n = rng.integers(2, 5)
        bits = rng.integers(0, 2, n)
        ds.append(np.array([-2.0, 2.0])[bits] + rng.normal(0, 1.5) + rng.normal(0, np.array([.5, 1.0])[bits]))
        ys.append(bits)
    p = cm.fit_offset(ds, ys)
    assert p.mu5 - p.mu4 == pytest.approx(4.0, abs=.2)
    assert p.tau == pytest.approx(1.5, abs=.2)
    assert p.s4 == pytest.approx(.5, abs=.1) and p.s5 == pytest.approx(1.0, abs=.1)
    assert p.converged
    assert cm.fit_offset(ds, ys, with_offset=False).tau == 0.0


def test_session_offset_flips_a_relative_call():
    # Both clips look A4C-like in absolute terms. A large shared offset explains that, so the higher one is A5C.
    p = cm.OffsetParams(-1.0, 1.0, 0.2, 0.2, 3.0)
    L = np.zeros((2, 5))
    L[:, 2] = [2.0, 0.0]
    L[:, 3] = [0.0, 0.0]
    base = np.array([2, 2])
    labels, fallback = cm.decode_apical(L, base, p, log_pi=np.log([.5, .5]))
    assert labels.tolist() == [2, 3] and not fallback
    independent, _ = cm.decode_apical(L, base, cm.OffsetParams(-1.0, 1.0, 0.2, 0.2, 0.0), log_pi=np.log([.5, .5]))
    assert independent[0] == 2


def test_decode_apical_leaves_other_views_alone_and_matches_enumeration():
    rng = np.random.default_rng(7)
    p = cm.OffsetParams(-1.0, 1.0, 1.0, 1.0, 1.0)
    L = rng.normal(size=(6, 5))
    base = np.array([0, 2, 3, 1, 2, 3])
    start, trans = cm.fit_chain([np.array([0, 1, 2, 3, 2, 4])], n=5, alpha=1.0)
    labels, _ = cm.decode_apical(L, base, p, chain=(start, trans))
    assert labels[0] == 0 and labels[3] == 1
    idx = np.array([1, 2, 4, 5])
    d = L[idx, 3] - L[idx, 2]

    def score(bits):
        y = base.copy()
        y[idx] = 2 + np.array(bits)
        return cm.offset_loglik(d, np.array(bits), p) + start[y[0]] + trans[y[:-1], y[1:]].sum()
    best = max(itertools.product([0, 1], repeat=4), key=score)
    assert labels[idx].tolist() == [2 + b for b in best]


def test_decode_apical_falls_back_beyond_cap_and_reports_it():
    p = cm.OffsetParams(-1.0, 1.0, 1.0, 1.0, 1.0)
    L = np.random.default_rng(1).normal(size=(5, 5))
    labels, fallback = cm.decode_apical(L, np.full(5, 2), p, log_pi=np.log([.5, .5]), cap=3)
    assert fallback and set(labels.tolist()) <= {2, 3}


def test_clip_evidence_is_mean_log_probability_with_floor():
    prob = np.array([[1.0, 0.0], [0.5, 0.5]])
    out = cm.clip_evidence(prob)
    assert out[0] == pytest.approx(np.log(.5) / 2)
    assert out[1] == pytest.approx((np.log(1e-8) + np.log(.5)) / 2)


def test_within_and_cross_session_auc_separate_the_offset_signal():
    # Within a session A5C is always higher, but sessions have large offsets.
    d = np.array([10.0, 11.0, -10.0, -9.0])
    y = np.array([0, 1, 0, 1])
    s = np.array(['a', 'a', 'b', 'b'])
    out = cm.session_auc(d, y, s)
    assert out['within_auc'] == 1.0 and out['cross_auc'] == 0.5
    assert out['within_pairs'] == 2 and out['cross_pairs'] == 2


# --- selective ---

def test_threshold_keeps_largest_coverage_meeting_risk():
    conf = np.array([.9, .8, .7, .6, .5])
    correct = np.array([1, 1, 1, 0, 1], bool)
    assert sel.threshold_for_risk(conf, correct, .0) == .7
    assert sel.threshold_for_risk(conf, correct, .2) == .5


def test_threshold_is_infinite_when_nothing_meets_risk():
    assert sel.threshold_for_risk(np.array([.9, .8]), np.array([0, 0], bool), .1) == np.inf


def test_threshold_treats_tied_scores_as_one_decision():
    conf = np.array([.9, .9, .5])
    correct = np.array([1, 0, 1], bool)
    assert sel.threshold_for_risk(conf, correct, .0) == np.inf
    assert sel.threshold_for_risk(conf, correct, .5) == .5


def test_apply_threshold_reports_coverage_and_risk():
    out = sel.apply_threshold(np.array([.9, .8, .1]), np.array([1, 0, 1], bool), .5)
    assert out == {'n': 3, 'accepted': 2, 'errors': 1, 'coverage': 2 / 3, 'risk': .5}
    empty = sel.apply_threshold(np.array([.1]), np.array([1], bool), np.inf)
    assert empty['accepted'] == 0 and empty['risk'] is None


def test_merged_view_sums_apical_probabilities():
    p = np.array([[.1, .1, .4, .3, .1]])
    assert sel.merged_view(p)[0].tolist() == pytest.approx([.1, .1, .7, .1])
    assert sel.merged_labels(np.array([0, 1, 2, 3, 4])).tolist() == [0, 1, 2, 2, 3]


def test_decisions_are_consistent_between_label_spaces():
    p = np.array([[.05, .05, .35, .45, .10], [.8, .1, .05, .03, .02]])
    five = sel.clip_decision(p, merged=False)
    four = sel.clip_decision(p, merged=True)
    assert five['label'].tolist() == [3, 0] and five['confidence'].tolist() == pytest.approx([.45, .8])
    assert four['label'].tolist() == [2, 0] and four['confidence'].tolist() == pytest.approx([.8, .8])


def test_risk_upper_bound_matches_rule_of_three_and_handles_edges():
    assert sel.risk_upper_bound(0, 300) == pytest.approx(3 / 300, abs=.0005)
    assert sel.risk_upper_bound(0, 0) is None
    assert sel.risk_upper_bound(5, 5) == 1.0
    assert sel.risk_upper_bound(2, 100) > .02


def test_destination_error_shares():
    pred = np.array([2, 2, 3, 0])
    truth = np.array([2, 3, 3, 0])
    out = sel.destination_errors(pred, truth, np.array([1, 1, 1, 0], bool), n_classes=5)
    assert out[2] == {'routed': 2, 'errors': 1}
    assert out[3] == {'routed': 1, 'errors': 0}
    assert out[0] == {'routed': 0, 'errors': 0}
