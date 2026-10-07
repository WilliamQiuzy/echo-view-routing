"""Pure helpers of scripts/joint_boundary_decoding.py on synthetic data: the grid order and the J0/J1 choice, ASRF's
candidate peaks, the alignment profile, change attribution to recordings, merged-change recovery, the stable rule and
reading one half only."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.joint_boundary_decoding as jb
from echo_routing.compose.recipe_builder import Fragment, Recipe
from scripts.report_l1_asrf import classify_changes

JD = {"grid": {"lambda": [0.0, 0.25, 0.5, 1.0, 2.0], "kappa": [0.0, 1.0, 2.0, 4.0, 8.0]}}


def test_grid_order_puts_lambda_zero_first():
    g = jb.grid_of(JD)
    assert len(g) == 25 and g[0] == (0.0, 0.0) and g[4] == (0.0, 8.0) and g[5] == (0.25, 0.0) and g[-1] == (2.0, 8.0)


def _rec(order, lam, kap, native, recall, seg):
    return {"order": order, "params": {"lambda": lam, "kappa": kap}, "native_macro_f1": native, "boundary_recall_multi": recall,
            "segment_f1_50_multi": seg, **{k: 0.0 for k in jb.FLAT if k not in ("native_macro_f1", "boundary_recall_multi", "segment_f1_50_multi")}}


def test_choice_of_j0_and_j1():
    recs = [_rec(0, 0.0, 0.0, 0.99, 0.96, 0.90), _rec(1, 0.0, 1.0, 0.99, 0.96, 0.95), _rec(5, 0.25, 0.0, 0.99, 0.96, 0.95),
            _rec(6, 0.25, 1.0, 0.97, 0.99, 0.99), _rec(7, 0.25, 2.0, 0.99, 0.94, 0.99)]
    ch = jb.choose_systems(recs, 0.980, 0.95)
    assert ch["J0"]["params"] == {"lambda": 0.0, "kappa": 1.0} and ch["J0"]["n_feasible"] == 2
    assert ch["J1"]["params"] == {"lambda": 0.0, "kappa": 1.0}            # a tie at 0.95 goes to the lower order: J1 = J0
    recs.append(_rec(8, 0.25, 4.0, 0.99, 0.96, 0.96))
    assert jb.choose_systems(recs, 0.980, 0.95)["J1"]["params"] == {"lambda": 0.25, "kappa": 4.0}
    none = jb.choose_systems([_rec(0, 0.0, 0.0, 0.97, 0.99, 0.99), _rec(5, 0.25, 0.0, 0.99, 0.90, 0.99)], 0.980, 0.95)
    assert none["J0"]["status"] == none["J1"]["status"] == "infeasible" and none["J1"]["params"] is None   # no fallback


def _official_argrelmax(prob, threshold):
    prob = prob.copy(); prob[prob < threshold] = 0.0
    peak = np.concatenate([np.ones(1, bool), (prob[:-2] < prob[1:-1]) & (prob[2:] < prob[1:-1]), np.zeros(1, bool)])
    return np.where(peak)[0].tolist()


def test_candidates_follow_the_official_peak_rule():
    rng = np.random.default_rng(0)
    for _ in range(50):
        b = rng.uniform(size=int(rng.integers(3, 60))).astype(np.float32)
        assert jb.candidates(b) == [i for i in _official_argrelmax(b, 0.5) if i > 0]
    assert jb.candidates(np.array([0.9, 0.2])) == [] and jb.candidates(np.array([0.2, 0.9, 0.9, 0.2])) == []   # plateaus are no peak


def test_boundary_profile_peaks_where_the_change_starts():
    b = np.full(30, 0.05); b[10] = 0.9; b[20] = 0.8
    assert jb.boundary_profile([b], [np.array([10, 20])])["peak_offset"] == 0
    assert jb.boundary_profile([np.roll(b, 1)], [np.array([10, 20])])["peak_offset"] == 1


def _stream(labels_per_fragment, lengths, videos):
    frs = tuple(Fragment(v, lab, "orig", 0, n) for v, lab, n in zip(videos, labels_per_fragment, lengths))
    labels = np.concatenate([np.full(n, lab) for lab, n in zip(labels_per_fragment, lengths)])
    joins = np.cumsum(lengths)[:-1]; sem = np.array([j for j in joins if labels[j] != labels[j - 1]], int)
    return SimpleNamespace(recipe=Recipe("r", "cell", frs), labels=labels, joins=joins, semantic_boundaries=sem)


def test_change_records_name_the_recordings_on_both_sides():
    s = _stream([0, 2, 2, 3], [10, 10, 20, 10], ["a_1", "b_1", "c_1", "d_2"])     # true changes at 10 and 40; same-view join at 20
    final = s.labels.copy(); final[24:28] = 4                                      # extra changes at 24 (4 from the join) and 28 (inside c_1)
    changes, extras = jb.change_records(final, [10, 40], s)
    assert [(c["r"], c["cat"], c["videos"]) for c in changes] == [(10, "found, both views right", ["a_1", "b_1"]), (40, "found, both views right", ["c_1", "d_2"])]
    assert [(e["pos"], e["kind"], e["videos"]) for e in extras] == [(24, "at a same-view join", ["b_1", "c_1"]), (28, "inside a clip", ["c_1"])]
    near = s.labels.copy(); near[36:38] = 4                                        # within 5 samples of the true change at 40: not extra
    assert jb.change_records(near, [10, 40], s)[1] == []
    assert jb.dedup(extras) == {"count": 2, "recordings": 2, "sessions": 2}


def test_extra_positions_match_classify_changes_counts():
    rng = np.random.default_rng(1)
    for _ in range(40):
        s = _stream(list(rng.integers(0, 5, size=5)), list(rng.integers(8, 20, size=5)), [f"v{i}_1" for i in range(5)])
        final = s.labels.copy(); k = int(rng.integers(0, 4))
        for _ in range(k):
            a = int(rng.integers(0, final.size - 3)); final[a:a + int(rng.integers(1, 4))] = int(rng.integers(0, 5))
        _, extra = classify_changes(final, [], s.labels, s.joins, s.semantic_boundaries)
        pos = jb.extra_positions(final, s.joins, s.semantic_boundaries)
        assert (sum(k == "inside a clip" for _, k in pos), sum(k == "at a same-view join" for _, k in pos)) == (extra["extra_inside_clip"], extra["extra_at_same_view_join"])


def test_merged_recovery_counts():
    ref = [{"stream": "r", "r": 10, "cat": "merged", "videos": ["a_1", "b_1"]}, {"stream": "r", "r": 30, "cat": "merged", "videos": ["c_1", "d_2"]},
           {"stream": "r", "r": 50, "cat": "found, both views right", "videos": ["d_2", "e_3"]}]
    now = [{"stream": "r", "r": 10, "cat": "found, a view wrong", "videos": ["a_1", "b_1"]}, {"stream": "r", "r": 30, "cat": "merged", "videos": ["c_1", "d_2"]},
           {"stream": "r", "r": 50, "cat": "found, both views right", "videos": ["d_2", "e_3"]}]
    out = jb.merged_recovery(ref, now)
    assert out["merged_in_reference"] == {"count": 2, "recordings": 4, "sessions": 4}                     # sessions a, b, c, d
    assert out["found"]["count"] == 1 and out["found_a_view_wrong"] == 1 and out["found_both_views_right"] == 0 and out["still_missed"] == {"merged": 1}


def test_native_reads_and_differences():
    a = pd.DataFrame({"video_id": ["p_1", "p_2", "q_1"], "label": [3, 3, 2], "majority": [3, 2, 2]})
    b = pd.DataFrame({"video_id": ["p_1", "p_2", "q_1"], "label": [3, 3, 2], "majority": [2, 3, 2]})
    assert jb.clip_reads(a, ["p_1", "p_2"]) == {"clips": 2, "right": 1, "sessions_right": 1}
    d = jb.native_diff(a, b)
    assert (d["fixed"], d["new"], d["fixed_sessions"], d["new_sessions"]) == (1, 1, 1, 1) and d["_ids"] == {"fixed": ["p_1"], "new": ["p_2"]}


def _cmp(sel_lo, cal_hi):
    return {"selection": {"bootstrap": {"multi segment F1@50": {"ci95": [sel_lo, sel_lo + 0.02]}}},
            "calibration": {"bootstrap": {"multi segment F1@50": {"ci95": [cal_hi - 0.02, cal_hi]}}}}


def test_stable_rule():
    assert jb.stable([_cmp(0.001, 0.01)] * 3)["stable"]
    assert not jb.stable([_cmp(0.001, 0.01), _cmp(0.0, 0.01), _cmp(0.001, 0.01)])["stable"]       # an interval touching zero
    assert not jb.stable([_cmp(0.001, 0.01), _cmp(0.001, -0.001), _cmp(0.001, 0.01)])["stable"]   # calibration below zero
    assert not jb.stable([_cmp(0.001, 0.01), None, _cmp(0.001, 0.01)])["stable"]                 # an infeasible seed


def test_three_clips_must_repeat_in_every_seed():
    def det(ids):
        return {"comparisons": {f"s{s}|B - (L1 + B3)": {"selection": {"_ids": {"A4C read as A5C": {"new": ids[:1]}, "A5C read as A4C": {"new": ids[1:]}}}}
                                for s in range(3)}}
    assert jb.three_clips(det(["x_1", "y_1", "z_1"]), [0, 1, 2]) == ["x_1", "y_1", "z_1"]
    bad = det(["x_1", "y_1", "z_1"]); bad["comparisons"]["s2|B - (L1 + B3)"]["selection"]["_ids"]["A5C read as A4C"]["new"] = ["y_1"]
    with pytest.raises(SystemExit):
        jb.three_clips(bad, [0, 1, 2])


def test_visual_loss_counts_correct_fragments_in_unmatched_units():
    transfer = [{"stream": "s", "unit": 0, "video": "a_1", "majority_category": jb.RR}, {"stream": "s", "unit": 1, "video": "a_1", "majority_category": jb.RR},
                {"stream": "s", "unit": 1, "video": "b_2", "majority_category": "recording right, fragment wrong"}]
    units = [{"stream": "s", "unit": 0, "cause": "label"}, {"stream": "s", "unit": 1, "cause": "matched"}]
    assert jb.visual_loss(transfer, units) == {"fragments": 1, "recordings": 1, "sessions": 1, "of_fragments": 2}


def test_load_half_reads_one_half(tmp_path):
    p = tmp_path / "x.npz"; np.savez(p, **{"calibration|a": np.ones(3), "selection|a": np.zeros(3), "calibration|b": np.ones(2)})
    assert sorted(jb.load_half(p, "calibration")) == ["a", "b"] and jb.load_half(p, "selection")["a"].sum() == 0
