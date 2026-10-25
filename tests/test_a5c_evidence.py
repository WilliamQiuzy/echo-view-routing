"""A32: fixed set definitions, run counts and the review-package selection rule."""
from collections import Counter

import numpy as np

import scripts.a5c_review_package as pkg
import scripts.diagnose_a5c_evidence as d


def fold(y, b, arms):
    return {"pk": {"labels": np.asarray(y), "base": np.asarray(b)}, "arms": {k: np.asarray(v) for k, v in arms.items()}}


def test_set_masks_follow_the_protocol():
    #   run:      0  1  2  3  4  5
    y = [3, 3, 3, 3, 2, 2]
    b = [2, 2, 3, 3, 3, 3]
    arms = {"F_U": [2, 3, 3, 3, 3, 3], "F_E": [2, 3, 2, 3, 2, 3], "T_U": [2, 2, 3, 3, 3, 3], "T_E": [2, 2, 3, 3, 3, 3]}
    m = d.set_masks(fold(y, b, arms))
    assert m["S1"].tolist() == [1, 1, 0, 0, 0, 0]
    assert m["S1_persistent"].tolist() == [1, 0, 0, 0, 0, 0]       # still A4C under all four arms
    assert m["S2"].tolist() == [0, 0, 1, 0, 0, 0] and m["S2_F_E"].tolist() == m["S2"].tolist() and not m["S2_T_E"].any()
    assert m["S3"].tolist() == [0, 0, 0, 1, 0, 0]
    assert m["S4"].tolist() == [0, 0, 0, 0, 1, 0]


def test_run_counts_and_change_table():
    y, b, p = np.array([3, 3, 2, 2, 0]), np.array([2, 3, 3, 2, 0]), np.array([3, 2, 2, 3, 0])
    r = d.run_counts(y, b, p)
    assert (r["base_wrong"], r["fixed"], r["base_correct"], r["broken"], r["net_fixed"]) == (2, 2, 3, 2, 0)
    t = d.change_table(y, b, p)
    assert t["fixed"] == {"2<-3": 1, "3<-2": 1} and t["broken"] == {"2->3": 1, "3->2": 1}


def test_set_stats_split_multi_source_runs():
    D = {"src": [{"r1": 0.5, "r2": 0.5}, {"r1": 1.0}], "true_src": [{"r1": 0.5}, {"r1": 1.0}], "kind": np.array(["multi", "native"]), "length": np.array([10, 20]),
         "pure": np.array([False, True]), "n_sources": np.array([2, 1])}
    st = d.set_stats([(D, 0), (D, 1)])
    assert st["by_src"]["recordings"] == 2 and st["by_src"]["top_recording_share"]["1"] == 0.75 and st["multi_source"] == 1 and st["short_le16"] == 1
    assert st["by_true_src"]["recordings"] == 1 and st["by_true_src"]["weight"] == 1.5 and st["by_true_src"]["top_recording_share"]["1"] == 1.0


def test_top_share():
    assert d.top_share(Counter({"a": 3, "b": 1}), 1, 4) == 0.75 and d.top_share(Counter(), 1, 0) is None


def test_choose_prefers_distinct_sessions_and_is_fixed():
    cands = {n: {f"s{i // 2}_{i % 2}": None for i in range(20)} for n in pkg.SETS}
    a, b = pkg.choose(cands), pkg.choose(cands)
    assert a == b
    for n in pkg.SETS:
        assert len(a[n]) == pkg.PER_SET and len({v.rsplit("_", 1)[0] for v in a[n]}) == pkg.PER_SET


def test_choose_fills_after_sessions_run_out():
    cands = {n: {f"s0_{i}": None for i in range(4)} for n in pkg.SETS}
    out = pkg.choose(cands)
    assert all(sorted(out[n]) == sorted(cands[n]) for n in pkg.SETS)
