"""Pure helpers of scripts/short_segment_pair_threshold.py on synthetic data: the predicted-pair rule, the probe's
threshold rule on few candidates, the library thresholds, tallies on decisions, transitions, the decision cases and the
decision bootstrap."""
import json

import numpy as np
import pytest

import scripts.short_segment_pair_threshold as pt
import scripts.short_segment_probe as sp
import scripts.short_segment_transfer as tr
from scripts.ev9v_complementarity import md_table

PAIR = frozenset((sp.A4C, sp.A5C))


def _row(cat, x, y, vids, **kw):
    return {"cat": cat, "x": x, "y": y, "mid_videos": vids, "half": "selection", "group": "selection|stream:r1", "kind": "multi streams",
            "join": "view-change", "len_mid": 20, **kw}


def test_pair_rule_reads_only_the_predicted_classes():
    rows = [_row("kept", 2, 3, ["a_1"]), _row("removed", 3, 2, ["a_2"]), _row("other", 2, 3, ["b_1"]), _row("kept", 2, 4, ["c_1"])]
    assert [pt.in_pair(r, PAIR) for r in rows] == [True, True, True, False]
    assert pt.rule_thresholds(rows, PAIR, 0.8, 0.3).tolist() == [0.3, 0.3, 0.3, 0.8]
    assert pt.rule_thresholds(rows, PAIR, 0.8, None).tolist() == [0.8] * 4
    flipped = [{**r, "cat": "removed"} for r in rows]                       # the reference category never changes it
    assert pt.rule_thresholds(flipped, PAIR, 0.8, 0.3).tolist() == pt.rule_thresholds(rows, PAIR, 0.8, 0.3).tolist()
    assert pt.pair_of(["A5C", "A4C"]) == PAIR


def test_pair_threshold_keeps_all_below_twenty_candidates():
    r = pt.pair_threshold(np.linspace(0.2, 0.9, 13), 0.95)
    assert r["threshold"] == pytest.approx(0.2) and r["kept"] == 13 and r["keeps_all"]
    assert r["keep_lower_bound_95"] == pytest.approx(0.05 ** (1 / 13))
    r40 = pt.pair_threshold(np.arange(40) / 40, 0.95)                       # 38 of 40 must stay: two may fall below
    assert r40["kept"] == 38 and not r40["keeps_all"] and r40["threshold"] == pytest.approx(2 / 40)


class _Pipe:
    def __init__(self, scores):
        self.s = np.asarray(scores, float)

    def predict_proba(self, x):
        assert len(x) == len(self.s)
        return np.stack([1 - self.s, self.s], axis=1)


def test_library_thresholds_use_only_calibration_correct_pair_candidates():
    meta, scores = [], []

    def add(part, cat, x, y, s, vid):
        meta.append({"part": part, "cat": cat, "x": x, "y": y, "mid_videos": [vid]}); scores.append(s)
    for k in range(40):                                                    # other pairs set the original threshold at 0.6
        add("calibration", "kept", 0, 1, 0.6 + k * 0.009, f"o{k}_1")
    add("calibration", "kept", 2, 3, 0.35, "a_1"); add("calibration", "kept", 3, 2, 0.45, "a_2")         # both directions
    add("calibration", "removed", 2, 3, 0.10, "w_1"); add("calibration", "removed", 2, 3, 0.40, "w_2")
    add("train", "kept", 2, 3, 0.01, "t_1"); add("held_out", "kept", 2, 3, 0.05, "h_1"); add("calibration", "other", 2, 3, 0.02, "x_1")
    arrays = {"q0": np.zeros((len(meta), 33), np.float32)}
    out = pt.seed_thresholds(arrays, meta, {"Q0": _Pipe(scores)}, {"Q0": 0.6}, PAIR, 0.95)["Q0"]
    assert out["pair threshold"] == pytest.approx(0.35) and out["pair vs original"] == "lower"
    cal = out["calibration"]
    assert cal["pair rule"]["n"] == 2 and cal["pair rule"]["keeps_all"]
    assert cal["correct by direction"] == {"A4C -> A5C -> A4C": 1, "A5C -> A4C -> A5C": 1}
    assert cal["at the original threshold"]["kept"]["kept"] == 0 and cal["at the original threshold"]["removed"]["removed"] == 2
    assert cal["at the pair threshold"]["kept"]["kept"] == 2 and cal["at the pair threshold"]["removed"]["removed"] == 1
    assert out["held-out, supplementary"]["pair"]["kept"] == {"n": 1, "kept": 0, "wrongly_removed": 1, "recordings": 1, "sessions": 1}
    with pytest.raises(SystemExit):
        pt.seed_thresholds(arrays, meta, {"Q0": _Pipe(scores)}, {"Q0": 0.5}, PAIR, 0.95)


def test_tally_on_decisions_matches_the_transfer_tally():
    rng = np.random.default_rng(0)
    rows = [_row(str(c), 2, 3, [f"p{k % 5}_{k}"]) for k, c in enumerate(rng.choice(["kept", "removed", "other"], 30))]
    s = rng.random(30)
    assert pt.tally_kept(rows, s >= 0.4) == tr.tally(rows, s, 0.4)


def test_transitions_and_changed_cases():
    rows = [_row("kept", 2, 3, ["a_1"]), _row("kept", 3, 2, ["a_2"]), _row("removed", 2, 3, ["b_1"]), _row("removed", 2, 3, ["b_2"]),
            _row("other", 2, 3, ["c_1"]), _row("kept", 0, 1, ["d_1"])]
    s = np.array([0.5, 0.9, 0.5, 0.2, 0.5, 0.1])
    old, new = pt.decisions(rows, s, PAIR, 0.8, 0.4)
    assert old.tolist() == [False, True, False, False, False, False] and new.tolist() == [True, True, True, False, True, False]
    t = pt.transitions(rows, old, new, PAIR)
    assert t["correct rescued"] == {"n": 1, "recordings": 1, "sessions": 1} and t["wrong let through"]["n"] == 1 and t["outside, now kept"]["n"] == 1
    assert t["correct lost"]["n"] == 0 and t["wrong newly removed"]["n"] == 0 and t["outside, now removed"]["n"] == 0
    ch = pt.changed(rows, s, old, new, PAIR)
    assert [(c["_i"], c["category"], c["now"], c["direction"]) for c in ch] == [(0, "correct", "kept", "A4C -> A5C -> A4C"),
                                                                              (2, "wrong", "kept", "A4C -> A5C -> A4C"),
                                                                              (4, "outside", "kept", "A4C -> A5C -> A4C")]
    tl = pt.split_tallies(rows, new, PAIR)                                 # the other pair keeps the original threshold
    assert tl["other pairs"]["kept"] == {"n": 1, "kept": 0, "wrongly_removed": 1, "recordings": 1, "sessions": 1}
    assert tl["A4C/A5C pairs"]["kept"]["kept"] == 2 and tl["A4C/A5C pairs"]["removed"]["missed"] == 1


def test_unit_key_uses_the_own_view_recordings():
    view = {"a_1": 2, "a_2": 3, "a_3": 3}
    assert pt.unit_key({"half": "selection", "group": "g", "y": 3, "mid_videos": ["a_1", "a_3", "a_2"]}, view) == ("selection", "g", ("a_2", "a_3"))


def test_model_cases_and_decision():
    meets = [{"K_aa": 0.92, "R_aa": 0.6}, {"K_aa": 0.95, "R_aa": 0.55}, {"K_aa": 0.7, "R_aa": 0.9}]
    lets = [{"K_aa": 1.0, "R_aa": 0.1}, {"K_aa": 0.92, "R_aa": 0.3}, {"K_aa": 0.92, "R_aa": 0.6}]
    short = [{"K_aa": 0.85, "R_aa": 0.9}, {"K_aa": 0.92, "R_aa": 0.6}, {"K_aa": 0.80, "R_aa": 0.2}]
    assert pt.model_case(meets, 0.90, 0.50) == "meets"
    assert pt.model_case(lets, 0.90, 0.50) == "keeps correct, lets wrong through"
    assert pt.model_case(short, 0.90, 0.50) == "keep not reached"
    assert pt.model_case([{"K_aa": None, "R_aa": None}] * 3, 0.90, 0.50) == "keep not reached"
    d = pt.decide({"Q0": lets, "Q1": meets}, 0.90, 0.50)
    assert d["meets"] == ["Q1"] and "operating point" in d["outcome"]
    d = pt.decide({"Q0": lets, "Q1": short}, 0.90, 0.50)
    assert d["meets"] == [] and "ends" in d["outcome"]


def test_bootstrap_on_decisions_matches_the_probe_bootstrap():
    rng = np.random.default_rng(1); n = 60
    sa, sb, y = rng.random(n), rng.random(n), rng.integers(0, 2, n); groups = np.array([f"g{k % 12}" for k in range(n)])
    want = sp.bootstrap_diff(sa, sb, y, 0.4, 0.6, groups, 200, 7)
    got = pt.bootstrap_decisions(sa >= 0.4, sb >= 0.6, y, groups, 200, 7)
    for k in ("removal", "keep"):
        assert got[k]["delta"] == pytest.approx(want[k]["delta"]) and got[k]["ci95"] == pytest.approx(want[k]["ci95"])
    same = pt.bootstrap_decisions(sa >= 0.5, sa >= 0.5, y, groups, 50, 7)
    assert same["keep"]["delta"] == 0 and same["removal"]["ci95"] == [0, 0]
    empty = pt.bootstrap_decisions(np.zeros(0, bool), np.zeros(0, bool), np.zeros(0, int), np.zeros(0), 10, 7)
    assert empty["keep"] == {"delta": None, "ci95": [None, None]}


def test_no_calibration_pair_candidate_stops_with_a_message():
    meta = [{"part": "calibration", "cat": "kept", "x": 0, "y": 1, "mid_videos": [f"o{k}_1"]} for k in range(5)]
    meta.append({"part": "calibration", "cat": "removed", "x": 2, "y": 3, "mid_videos": ["w_1"]})
    with pytest.raises(SystemExit, match="no correct A4C/A5C candidate"):
        pt.seed_thresholds({"q0": np.zeros((6, 33))}, meta, {"Q0": _Pipe([0.9, 0.8, 0.7, 0.6, 0.5, 0.1])}, {"Q0": 0.5}, PAIR, 0.95, "seed 0")


def test_a_higher_pair_threshold_loses_and_newly_removes():
    rows = [_row("kept", 2, 3, ["a_1"]), _row("removed", 3, 2, ["b_1"]), _row("other", 2, 3, ["c_1"])]
    old, new = pt.decisions(rows, np.array([0.5, 0.5, 0.5]), PAIR, 0.4, 0.6)
    t = pt.transitions(rows, old, new, PAIR)
    assert (t["correct lost"]["n"], t["wrong newly removed"]["n"], t["outside, now removed"]["n"]) == (1, 1, 1)
    assert (t["correct rescued"]["n"], t["wrong let through"]["n"]) == (0, 0)
    assert [c["now"] for c in pt.changed(rows, np.array([0.5, 0.5, 0.5]), old, new, PAIR)] == ["removed"] * 3


def test_case_listing_carries_no_ids():
    view = {"sessA_1": 2, "sessA_2": 3, "sessB_1": 3, "sessC_1": 2}
    rows = [_row("kept", 2, 3, ["sessA_1", "sessA_2"], group="selection|stream:recX"), _row("kept", 2, 3, ["sessB_1"], group="selection|stream:recY"),
            _row("removed", 2, 3, ["sessC_1"], group="selection|session:sessC")]
    s = np.array([0.5, 0.9, 0.5]); dec = {"Q0": pt.decisions(rows, s, PAIR, 0.8, 0.4)}
    cases, changed_by_seed = {}, {"0": {"Q0": pt.attach_keys(pt.changed(rows, s, *dec["Q0"], PAIR), rows, view)}}
    pt.add_cases(cases, rows, dec, PAIR, view, "0"); pt.add_cases(cases, rows[1:2], {"Q0": (np.array([True]), np.array([True]))}, PAIR, view, "1")
    listing = pt.number_cases(cases, changed_by_seed)
    text = json.dumps({"cases": listing, "changed": changed_by_seed})
    assert not any(token in text for token in ("sessA", "sessB", "sessC", "recX", "recY", "_key", '"_i"'))
    assert [c["case"] for c in listing] == [1, 2] and listing[1]["seeds"]["1"] == [{"Q0": {"original": "kept", "pair": "kept"}}]
    assert [(e["category"], e["case"]) for e in changed_by_seed["0"]["Q0"]] == [("correct", 1), ("wrong", None)]


def test_case_table_keeps_every_seed_column():
    cases = [{"case": 1, "half": "selection", "direction": "A4C -> A5C -> A4C", "seeds": {"0": [{"Q0": {"original": "removed", "pair": "kept"}, "Q1": {"original": "kept", "pair": "kept"}}]}},
             {"case": 2, "half": "selection", "direction": "A5C -> A4C -> A5C", "seeds": {"0": [{"Q0": {"original": "kept", "pair": "kept"}, "Q1": {"original": "kept", "pair": "kept"}}],
                                                                                          "1": [{"Q0": {"original": "removed", "pair": "removed"}, "Q1": {"original": "kept", "pair": "kept"}}]}}]
    rows = pt.case_rows(cases)
    header = md_table(rows, pt.columns(rows)).splitlines()[0]
    assert "s1 Q0" in header and "s1 Q1" in header and rows[1]["s1 Q0"] == "removed -> removed"


def test_only_this_rounds_results_directory_is_written():
    p = {"results_dir": "docs/results/2026-10-07_short_segment_pair_threshold"}
    pt.check_out("docs/results/2026-10-07_short_segment_pair_threshold", p)
    with pytest.raises(SystemExit):
        pt.check_out("docs/results/2026-10-07_short_segment_transfer", p)
