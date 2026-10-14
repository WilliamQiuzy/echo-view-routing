"""Checks of the token stability attribution round's pure parts (configs/experiments/token_stability_attribution_v1.yaml),
written before the module: the four reference-conditioned cells and their endpoints, the exact greedy segment matching
and its tie order, segment status and unmatched predicted segments, the request-to-change distance, the overlapping-anchor
pairs with their dedupe, neighbourhood, marker and weight rules, the bootstrap that must equal the existing paired
bootstrap, and the fixed nonexclusive readings. Synthetic data only."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

import scripts.token_stability_attribution_tables as tb
from echo_routing.evaluate.metrics_segmentation import segment_f1
from echo_routing.temporal.segments import Segment, labels_to_segments
from scripts.report_model_development import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/experiments/token_stability_attribution_v1.yaml").read_text())["token_stability_attribution"]


# ----------------------------------------------------------------------------------------------- the four cells
def test_the_cells_follow_the_mask_and_keep_the_inputs():
    rng = np.random.default_rng(0); pc = rng.dirichlet(np.ones(5), 6); pt = rng.dirichlet(np.ones(5), 6)
    mixed = np.array([1, 0, 1, 0, 0, 1], bool); keep = (pc.copy(), pt.copy(), mixed.copy())
    cells = tb.cell_probs(pc, pt, mixed)
    assert np.array_equal(cells["CC"], pc) and np.array_equal(cells["TT"], pt)
    assert np.array_equal(cells["TC"][mixed], pt[mixed]) and np.array_equal(cells["TC"][~mixed], pc[~mixed])
    assert np.array_equal(cells["CT"][mixed], pc[mixed]) and np.array_equal(cells["CT"][~mixed], pt[~mixed])
    assert all(np.array_equal(a, b) for a, b in zip(keep, (pc, pt, mixed)))
    none = tb.cell_probs(pc, pt, np.zeros(6, bool)); every = tb.cell_probs(pc, pt, np.ones(6, bool))
    assert np.array_equal(none["TC"], pc) and np.array_equal(none["CT"], pt)          # no mixed window: TC = CC, CT = TT
    assert np.array_equal(every["TC"], pt) and np.array_equal(every["CT"], pc)        # every window mixed: TC = TT, CT = CC
    serve = np.array([0, 0, 0, 2, 2, 5, 5])                                           # windows 0, 2 and 5 serve several samples
    mapped = cells["TC"][serve]
    assert all(np.array_equal(mapped[k], (pt if mixed[w] else pc)[w]) for k, w in enumerate(serve))
    with pytest.raises(ValueError):
        tb.diagnostic_probs(pc, pt[:5], mixed, 1, 0)


# ----------------------------------------------------------------------------------------------- matching
def test_matching_uses_the_evaluators_greedy_order():
    pred = [Segment(0, 10, 1), Segment(0, 10, 1)]; ref = [Segment(0, 10, 1)]
    assert tb.match_pairs(pred, ref) == [(1, 0)]                                      # an equal IoU goes to the higher prediction index
    labels = np.array([0] * 10 + [1] * 4 + [0] * 10)
    for pred_labels in (labels, np.zeros(24, int), np.array([0] * 12 + [1] * 2 + [0] * 10), np.array([0] * 5 + [2] * 5 + [1] * 4 + [0] * 10)):
        p, r = labels_to_segments(pred_labels), labels_to_segments(labels)
        c = tb.seg_counts(p, r)
        assert c["tp"] == segment_f1(p, r, 0.5)["matched"] and c["fp"] == len(p) - c["tp"] and c["fn"] == len(r) - c["tp"]


def test_segment_status_and_unmatched_predicted_segments():
    labels = np.array([0] * 10 + [1] * 4 + [0] * 10)
    a = tb.compare_segments(labels, labels.copy(), np.zeros(24, int))                 # token right everywhere; CLS one long segment
    assert a["status"] == ["token only"] * 3 and a["new_unmatched"] == set() and a["removed_unmatched"] == {(0, 24, 0)}
    b = tb.compare_segments(labels, np.zeros(24, int), np.array([0] * 12 + [1] * 2 + [0] * 10))
    assert b["status"] == ["cls only"] * 3 and b["new_unmatched"] == {(0, 24, 0)} and b["removed_unmatched"] == set()
    c = tb.compare_segments(labels, np.array([0] * 10 + [2] * 4 + [0] * 10), labels.copy())
    assert c["status"] == ["both", "cls only", "both"] and c["new_unmatched"] == {(10, 14, 2)}


# ----------------------------------------------------------------------------------------------- distances
def test_request_distance_counts_samples_between_the_request_and_the_change():
    ch = tb.view_changes(np.array([0] * 10 + [1] * 4 + [0] * 20))
    assert ch.tolist() == [10, 14]
    for c, d, b in ((9, 0, "0-5"), (10, 0, "0-5"), (12, 1, "0-5"), (4, 5, "0-5"), (3, 6, "6-10"), (20, 6, "6-10"), (25, 11, ">10")):
        assert tb.request_distance(c, ch) == d and tb.distance_bin(d) == b
    assert tb.request_distance(3, np.array([], int)) is None and tb.distance_bin(None) == "none"


# ----------------------------------------------------------------------------------------------- overlapping anchors
def test_a_sample_held_by_several_anchors_of_one_window_keeps_the_one_nearest_the_marker():
    s = np.array([[0] * 6 + [1] * 5 + [2] * 5]); m = np.array([8])
    i, j, t = tb.window_observations(s, m)
    assert list(zip(t.tolist(), j.tolist())) == [(0, 2), (1, 4), (2, 6)] and i.tolist() == [0, 0, 0]


def test_pairs_neighbourhoods_markers_selection_and_weights():
    S = np.array([np.arange(16), np.arange(16), np.arange(4, 20)]); C = np.array([2, 7, 12]); Mk = np.array([2, 7, 8])
    sel = np.array([1, 3, 4]); seq = np.zeros(3, int)
    P = tb.anchor_pairs(S, C, Mk, seq, sel)
    key = {(int(t), int(a), int(b)): k for k, (t, a, b) in enumerate(zip(P["t"], P["i1"], P["i2"]))}
    k = key[(2, 0, 1)]                                                                # sample 2 in windows A and B, anchor 1 in both
    assert P["complete"][k] and P["rgb_equal"][k] and not P["marker_equal"][k] and P["sel1"][k] and not P["sel2"][k]
    k = key[(0, 0, 1)]
    assert not P["complete"][k]                                                       # anchor 0 lacks slot -1
    k = key[(6, 0, 2)]                                                                # windows A (anchor 3) and C (anchor 1)
    assert P["complete"][k] and P["rgb_equal"][k] and P["marker_equal"][k] and P["centre_distance"][k] == 10
    k = key[(6, 1, 2)]
    assert P["main"][k] and not P["marker_equal"][k] and P["sel1"][k]                 # B's selected anchor 3 holds sample 6
    for subset in ("main", "broad"):
        w = tb.group_weights(P["group"], P[subset])
        for g in np.unique(P["group"][P[subset]]):
            assert abs(w[P[subset] & (P["group"] == g)].sum() - 1.0) < 1e-12
        assert (w[~P[subset]] == 0).all()
    assert not any(P["i1"] == P["i2"]) and (P["t"] == S[P["i1"], 2 * P["j1"]]).all() and (P["t"] == S[P["i2"], 2 * P["j2"]]).all()


def test_divergence_and_rates():
    p = np.array([[1.0, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0]]); q = np.array([[0, 1.0, 0, 0, 0], [0.5, 0.5, 0, 0, 0]])
    assert np.allclose(tb.js_divergence(p, q), [1.0, 0.0])
    assert tb.weighted_rate(np.array([1, 0, 1.0]), np.array([0.5, 0.5, 1.0])) == pytest.approx(1.5 / 2.0)


# ----------------------------------------------------------------------------------------------- bootstrap
def tables(seed: int) -> dict:
    rng = np.random.default_rng(seed); vids = ["p1_0", "p1_1", "p2_0", "p3_0", "p4_0", "p4_1"]
    nat = pd.DataFrame({"video_id": vids, "label": [0, 1, 2, 3, 4, 2], "majority": rng.integers(0, 5, 6), "cuts": rng.integers(0, 3, 6)})
    seg = pd.DataFrame({"stream": ["m0", "m1", "m2", "m3"], "seg_matched": rng.integers(3, 6, 4), "seg_pred": rng.integers(6, 9, 4), "seg_true": [6, 7, 6, 8]})
    frag = pd.DataFrame({"stream": ["m0", "m0", "m1", "m2", "m3", "m3"], "t_matched": rng.integers(0, 2, 6), "t_true": [1, 1, 1, 1, 1, 0],
                         "p_pred": rng.integers(0, 3, 6), "false_split": rng.integers(0, 2, 6), "same_join": [1, 0, 1, 1, 0, 1]})
    return {"native": nat, "multi_seg": seg, "multi_frag": frag}


def test_the_contrast_bootstrap_equals_the_paired_bootstrap():
    a, b = tables(1), tables(2)
    got = tb.contrast_bootstrap({"A": a, "B": b}, {"A": 1, "B": -1}); want = paired_bootstrap(a, b)
    assert json.dumps(got, sort_keys=True, default=float) == json.dumps(want, sort_keys=True, default=float)   # NaN-aware, exact
    z = tb.contrast_bootstrap({"TT": a, "TC": a, "CT": b, "CC": b}, {"TT": 1, "TC": -1, "CT": -1, "CC": 1})
    assert all(v["delta"] == 0 and v["ci95"] == [0.0, 0.0] for v in z.values())


# ----------------------------------------------------------------------------------------------- readings
def test_the_fixed_readings():
    P, N, NEG = tb.POS, tb.NONE, tb.NEG
    trade = {"selection": {"tc": [0.01, 0.02, 0.01], "ct": [-0.02, -0.01, -0.03], "tc_verdict": P, "ct_verdict": NEG},
             "calibration": {"tc": [0.01, -0.01, 0.0], "ct": [-0.01, -0.01, -0.01], "tc_verdict": N, "ct_verdict": N}}
    d = tb.readings([0.7, 0.8, 0.6, 0.9, 0.55, 0.51], trade, [0, 3, 0, 0, 0, 0], {"no selected-output pairs": True}, CFG["readings"])
    assert d["applied"] == ["local_correction_without_segment_recovery", "mixed_vs_pure_tradeoff", "anchor_sensitivity_observed", "unresolved", "end"]
    assert d["rows"]["mixed_vs_pure_tradeoff"]["halves"] == {"selection": "consistent", "calibration": "not observed"}
    d = tb.readings([0.7, None, 0.6, 0.9, 0.55, 0.5], trade | {"selection": trade["calibration"]}, [0] * 6, {}, CFG["readings"])
    assert d["applied"] == ["end"]
    trade2 = {h: {"tc": [0.01, 0.02, 0.01], "ct": [-0.02, -0.01, -0.03], "tc_verdict": N, "ct_verdict": NEG} for h in tb.HALVES}
    d = tb.readings([0.4] * 6, trade2, [0] * 6, {}, CFG["readings"])
    assert d["rows"]["mixed_vs_pure_tradeoff"]["halves"] == {"selection": "observed", "calibration": "observed"}
