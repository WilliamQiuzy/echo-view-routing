"""Pure helpers of scripts/short_segment_transfer.py on synthetic data: verified join types, reasons outside the binary
task, the tallies, the rebuild reproduction check, the decision order, and candidates and tables on a toy half."""
from types import SimpleNamespace

import numpy as np

import scripts.short_segment_transfer as tr
from echo_routing.compose.recipe_builder import Fragment, Recipe
from echo_routing.compose.stream_assembler import render
from echo_routing.features.cache import VideoFeatures
from echo_routing.temporal.dev_data import EvalSet, native_seq
from echo_routing.temporal.segments import Segment


def test_join_types_check_both_sides():
    ref = np.array([2] * 10 + [2] * 10 + [3] * 10)                 # joins at 10 (same view) and 20 (A4C -> A5C)
    assert tr.join_type(ref, np.array([10, 20]), 30, 31, 5) == "none"
    assert tr.join_type(ref, np.array([10, 20]), 8, 12, 2) == "same-view"
    assert tr.join_type(ref, np.array([10, 20]), 12, 18, 5) == "view-change"   # both joins within 5: one of them changes the view
    assert tr.join_type(ref, np.zeros(0, int), 3, 6, 5) == "none"


def _s(a, b, lab):
    return Segment(a, b, lab)


def test_reasons_outside_the_binary_task():
    left, mid, right = _s(0, 10, 2), _s(10, 14, 3), _s(14, 24, 2)
    assert tr.other_reason(left, mid, right, np.full(24, 0)) == "sides not X"
    assert tr.other_reason(left, mid, right, np.array([2] * 10 + [4] * 4 + [2] * 10)) == "a third class under the middle"
    assert tr.other_reason(left, mid, right, np.array([2] * 8 + [3] * 10 + [2] * 6)) == "part of a Y unit (IoU < 0.5)"   # Y unit 8-18, IoU 0.4
    assert tr.other_reason(left, mid, right, np.array([2] * 12 + [3] * 2 + [2] * 10)) == "a reference change under the middle"


def test_tally_counts_recordings_and_sessions():
    rows = [{"cat": "kept", "mid_videos": ["a_1"]}, {"cat": "kept", "mid_videos": ["a_2"]}, {"cat": "removed", "mid_videos": ["b_1"]},
            {"cat": "removed", "mid_videos": ["b_1"]}, {"cat": "other", "mid_videos": ["c_1"]}]
    t = tr.tally(rows, np.array([0.9, 0.1, 0.2, 0.8, 0.5]), 0.5)
    assert t["kept"] == {"n": 2, "kept": 1, "wrongly_removed": 1, "recordings": 2, "sessions": 1}
    assert t["removed"] == {"n": 2, "missed": 1, "removed": 1, "recordings": 1, "sessions": 1}
    assert tr.rates_of(t) == {"K": 0.5, "R": 0.5}


def test_reproduction_check():
    pts = {"0.95": {"all": {"keep_rate_kept": 0.97, "removal_rate_removed": 1.0, "n_kept": 10, "n_removed": 5}}}
    stored = {"thresholds": {"0.95": 0.6, "0.9": 0.9}, "held_out_auc": 0.99, "points": pts}
    assert tr.reproduce(stored, {"thresholds": {"0.95": 0.6, "0.9": 0.9}, "held_out_auc": 0.99, "points": pts})["ok"]
    assert not tr.reproduce(stored, {"thresholds": {"0.95": 0.61, "0.9": 0.9}, "held_out_auc": 0.99, "points": pts})["ok"]
    other = {"0.95": {"all": {**pts["0.95"]["all"], "n_removed": 6}}}
    assert not tr.reproduce(stored, {"thresholds": {"0.95": 0.6, "0.9": 0.9}, "held_out_auc": 0.99, "points": other})["ok"]


def _seed(q0, q1, ci_low=-0.1, k_delta=0.0, raa_delta=0.0):
    keys = ("AUC", "K_all", "R_all", "K_aa", "R_aa")
    return {"Q0": dict(zip(keys, q0)), "Q1": dict(zip(keys, q1)), "Q1 - Q0": {"R_all_ci_low": ci_low, "K_all_delta": k_delta, "R_aa_delta": raa_delta}}


def test_decision_order():
    bad, good = (0.7, 0.8, 0.3, 0.8, 0.3), (0.97, 0.96, 0.85, 0.95, 0.8)
    assert tr.branch([_seed(bad, bad)] * 3)["case"] == "both fail"
    assert tr.branch([_seed(good, good, ci_low=0.01)] * 3)["case"] == "Q1 extra value"
    assert tr.branch([_seed(good, good)] * 3)["case"] == "Q0 practical value"
    aa_problem = (0.97, 0.96, 0.85, 0.75, 0.8)
    assert tr.branch([_seed(aa_problem, aa_problem)] * 3)["case"] == "Q0 high overall but removes correct A4C/A5C middles"
    middling = (0.9, 0.92, 0.55, 0.85, 0.6)                                         # not failing, not practical (K_aa), not high (AUC)
    assert tr.branch([_seed(middling, bad)] * 3)["case"] == "no branch"
    assert tr.branch([_seed((0.9, 0.92, 0.4, 0.95, 0.6), bad)] * 3)["case"] == "both fail"    # R_all 0.4 < 0.5 is a failure
    assert tr.branch([_seed(good, good, ci_low=0.01, raa_delta=None)] * 3)["case"] == "Q1 extra value"     # no A4C/A5C wrong middles to judge
    assert tr.branch([_seed(good, good, ci_low=0.01), _seed(good, good, ci_low=0.01), _seed(good, good)])["case"] == "Q0 practical value"


def _vf(vid, n, label, noise=()):
    prob = np.full((n, 5), 0.02, np.float32); prob[:, label] = 0.92
    for i in noise:
        prob[i] = 0.02; prob[i, (label + 1) % 5] = 0.92
    return VideoFeatures(vid, np.arange(n) * 3, np.arange(n) * 0.1, np.random.default_rng(len(vid) + n).standard_normal((n, 4)).astype(np.float32),
                         prob, np.zeros((n, 5), np.float32))


class _Store:
    def __init__(self):
        self.d = {"a_1": _vf("a_1", 40, 2, (5, 6)), "b_1": _vf("b_1", 40, 3), "c_2": _vf("c_2", 40, 2, (10, 11, 12)), "d_3": _vf("d_3", 40, 1)}

    def get(self, vid, variant):
        return self.d[vid]


def test_candidates_and_tables_on_a_toy_half():
    st = _Store()
    natives = tuple(native_seq(v, lab, st.get(v, "orig")) for v, lab in (("a_1", 2), ("c_2", 2)))
    pairs = (render(Recipe("p1", "cell", (Fragment("a_1", 2, "orig", 10, 30), Fragment("c_2", 2, "orig", 10, 30))), st.get),)
    multi = (render(Recipe("m1", "cell", (Fragment("a_1", 2, "orig", 0, 15), Fragment("b_1", 3, "orig", 5, 25), Fragment("c_2", 2, "orig", 0, 15))), st.get),)
    inp = SimpleNamespace(halves={"selection": EvalSet("toy", natives, pairs, multi)}, store=st)
    conf = {"generator": {"eps": 1e-6, "kappa": 0.0}, "features": {"side_window": 16}, "candidates": {"tolerance_samples": 5}}
    arrays, rows = tr.candidates(conf, inp, "selection")
    assert arrays["q0"].shape == (len(rows), 33) and arrays["vis"].shape == (len(rows), 12) and arrays["sim"].shape == (len(rows), 2)
    kinds = {}
    for r in rows:
        kinds[(r["kind"], r["cat"], r["join"])] = kinds.get((r["kind"], r["cat"], r["join"]), 0) + 1
    assert kinds[("native clips", "removed", "none")] == 2                         # the noisy frames of a_1 (5-6) and c_2 (20-22)
    assert kinds[("multi streams", "kept", "view-change")] == 1                     # the true A5C fragment between A4C fragments
    assert any(r["kind"] == "pair streams" and r["cat"] == "removed" and r["join"] == "same-view" for r in rows)   # c_2 frames 10-12 sit at the same-view join (stream 20)
    scores = {"Q0": np.linspace(0, 1, len(rows)), "Q1": np.linspace(1, 0, len(rows))}
    t = tr.half_tables(rows, scores, {"Q0": {"0.95": 0.5}, "Q1": {"0.95": 0.5}})
    assert t["candidates"] == len(rows) and t["Q0"]["0.95"]["all"]["kept"]["n"] + t["Q0"]["0.95"]["all"]["removed"]["n"] + t["other"]["n"] == len(rows)
    assert set(t["Q0"]["0.95"]["subsets"]) == set(tr.SUBSETS)
