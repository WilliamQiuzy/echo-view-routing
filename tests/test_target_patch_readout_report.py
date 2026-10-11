"""CPU checks of the target-anchored readout report on synthetic data: the group bootstrap and its not-estimable rule,
the stream-sensitivity verdict, the mixed-window subsets and support counts, the fixed decision table, the window metadata
and the table rendering."""
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.target_patch_readout_tables as pt
from scripts.raw_video_context_tables import NEG, NONE, POS

ROOT = Path(__file__).resolve().parents[1]
TP = yaml.safe_load((ROOT / "configs/experiments/target_patch_readout_v1.yaml").read_text())["target_patch_readout"]


def test_group_bootstrap_is_paired_seeded_and_knows_its_limits():
    rng = np.random.default_rng(0); n = 120; g = rng.integers(0, 20, n); a = rng.random(n) < 0.8; b = rng.random(n) < 0.5
    x, y = pt.group_bootstrap(a, b, g, draws=300), pt.group_bootstrap(a, b, g, draws=300)
    assert x == y and x["estimable"] and x["delta"] == pytest.approx(a.mean() - b.mean()) and x["ci95"][0] < x["delta"] < x["ci95"][1] and x["units"] == 20
    same = pt.group_bootstrap(a, a, g, draws=100); assert same["delta"] == 0 and same["ci95"] == [0.0, 0.0]
    assert not pt.group_bootstrap(a[:0], b[:0], g[:0])["estimable"] and not pt.group_bootstrap(a, b, np.zeros(n))["estimable"]
    assert pt.group_bootstrap(a, b, g % 5, draws=50)["few_units"]
    with pytest.raises(ValueError):
        pt.group_bootstrap(a, b[:-1], g)


def d(lo, hi, ok=True):
    return {"delta": (lo + hi) / 2, "ci95": [lo, hi], "estimable": ok}


def test_verdict_uses_sessions_and_the_stream_conflict_rule():
    pos, neg, zero = d(0.01, 0.05), d(-0.05, -0.01), d(-0.01, 0.02)
    assert pt.verdict([pos] * 3, [zero] * 3) == POS and pt.verdict([pos] * 3, [neg] * 3) == pt.UNCERTAIN
    assert pt.verdict([pos, zero, pos], [pos] * 3) == NONE and pt.verdict([neg] * 3, [neg] * 3) == NEG
    assert pt.verdict([pos, d(0, 0, ok=False), pos], [pos] * 3) == "not estimable"


def test_mixed_subsets_and_support():
    labels16 = np.array([[0] * 8 + [2] * 8, [2] * 9 + [3] * 7, [3] * 12 + [2] * 4, [1] * 16, [0] * 4 + [3] * 12])
    label = np.array([0, 3, 2, 1, 0]); mixed = np.array([True, True, True, False, True])
    s = pt.mixed_subsets(labels16, label, mixed)
    assert s["8 of 16"].tolist() == [0] and s["<= 7 of 16"].tolist() == [1, 2, 4] and s["all mixed"].tolist() == [0, 1, 2, 4]
    assert s["A4C/A5C pair windows"].tolist() == [1, 2] and s["12-15 of 16"].size == 0
    rows = np.array([0, 1, 2, 4]); rec = np.array(["r1", "r1", "r2", "r9", "r3"]); ses = np.array(["a", "a", "b", "z", "a"]); run = np.array(["x", "x", "y", "w", "z"])
    sup = pt.subset_support(rows, rec, ses, run, label)
    assert sup["windows"] == 4 and sup["recordings"] == 3 and sup["sessions"] == 2 and sup["reference_runs"] == 3 and sup["largest_run_share"] == 0.5
    assert sup["per_class"]["0"] == {"windows": 2, "recordings": 2, "sessions": 1}


def v(sel, cal):
    return {"selection": sel, "calibration": cal}


def test_decision_table():
    rules = TP["decision_rules"]; goals = {"target_patches": {"met": False}}
    native = {c: v(NONE, NONE) for c in ("Target - CLS", "Target - Global")}
    sysd = lambda a, b: {"Target - CLS": a, "Target - Ordered": b}  # noqa: E731
    rule = lambda x: [r["rule"] for r in x["rows_that_apply"]]  # noqa: E731
    both_pos = {"Target - Global": v(POS, POS), "Target - CLS": v(POS, POS)}
    out = pt.decide(rules, both_pos, native, sysd(v(POS, POS), v(POS, POS)), goals)
    assert rule(out) == ["target_position"] and out["beats_existing"]
    out = pt.decide(rules, both_pos, native, sysd(v(POS, NONE), v(NONE, NONE)), goals)
    assert rule(out) == ["target_position", "tradeoff"] and not out["beats_existing"]
    one = {"Target - Global": v(POS, NONE), "Target - CLS": v(POS, POS)}
    assert rule(pt.decide(rules, one, native, sysd(v(POS, POS), v(POS, POS)), goals)) == ["target_position_one_half", "patch_not_position"]
    cls_only = {"Target - Global": v(NONE, NONE), "Target - CLS": v(POS, POS)}
    assert rule(pt.decide(rules, cls_only, native, sysd(v(NONE, NONE), v(NONE, NONE)), goals)) == ["patch_not_position", "no_gain"]
    none = {"Target - Global": v(NONE, pt.UNCERTAIN), "Target - CLS": v(NONE, NONE)}
    out = pt.decide(rules, none, native, sysd(v(NONE, NONE), v(NONE, NONE)), {"target_patches": {"met": True}})
    assert rule(out) == ["no_gain"] and out["goal_met_groups"] == ["target_patches"]
    worse = {c: v(NEG, NONE) for c in ("Target - CLS", "Target - Global")}
    assert rule(pt.decide(rules, both_pos, worse, sysd(v(POS, POS), v(POS, POS)), goals)) == ["target_position", "tradeoff"]


def test_window_meta_runs_and_mixed_flags():
    from scripts.report_target_patch_readout import window_meta
    seqs = [{"id": "s0", "kind": "multi", "labels": np.array([0] * 10 + [2] * 10), "refs": [("rA" if i < 10 else "rB_1", i, 1.0) for i in range(20)]},
            {"id": "n0", "kind": "native", "labels": np.array([3] * 20), "refs": [("rC_2", i, 1.0) for i in range(20)]}]
    W = {"seq": np.array([0, 0, 1]), "centre": np.array([2, 15, 7]), "labels16": np.array([[0] * 10 + [2] * 6, [0] * 4 + [2] * 12, [3] * 16])}
    m = window_meta(W, seqs)
    assert m["mixed"].tolist() == [True, True, False] and m["recording"].tolist() == ["rA", "rB_1", "rC_2"] and m["session"].tolist() == ["rA", "rB", "rC"]
    assert m["run"].tolist() == ["0:0", "0:10", "1:0"] and m["stream"].tolist() == ["0", "0", "1"]


def test_tables_render_partial_results():
    res = {"mixed": {"accuracy": {"target_patches_s0": {"selection": {"8 of 16": 0.7, "all mixed": 0.9}}},
                     "comparisons": {"s0|Target - Global|8 of 16|selection": {"session": d(0.01, 0.05) | {"units": 12, "windows": 40}, "stream": d(0.0, 0.06) | {"units": 30, "windows": 40}}}}}
    md = pt.tables_md(res)
    assert "Mixed-window accuracy" in md and "+0.0300 [+0.0100, +0.0500]" in md and "pending" in md
