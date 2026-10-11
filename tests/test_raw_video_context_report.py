"""CPU checks of the raw-video context report on synthetic data: recognition metrics against brute force, the
session-paired bootstrap, mixed-window, end and offset tables, the goal, the fixed decision readings, the evaluation
windows and the nearest-centre mapping."""
from pathlib import Path

import numpy as np
import pytest
import yaml

import echo_routing.vit.target_mvit as tm
import scripts.raw_video_context_tables as rt

ROOT = Path(__file__).resolve().parents[1]
RV = yaml.safe_load((ROOT / "configs/experiments/raw_video_context_v1.yaml").read_text())["raw_video_context"]


def test_recognition_metrics_match_brute_force():
    rng = np.random.default_rng(0); y = rng.choice([0, 2, 3], 200); pred = np.where(rng.random(200) < 0.7, y, rng.choice([2, 3, 1], 200))
    margin = rng.normal(size=200) + (y == 3) * 0.8; margin[:20] = 0.5                     # some ties
    pair = np.isin(y, (2, 3))
    want_ba = 0.5 * ((pred[y == 2] == 2).mean() + (pred[y == 3] == 3).mean())
    pos, neg = margin[pair & (y == 3)], margin[pair & (y == 2)]
    want_auc = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    r = rt.recognition(pred, margin, y)
    assert r["A4C/A5C balanced accuracy"] == pytest.approx(want_ba) and r["A5C-A4C margin AUC"] == pytest.approx(want_auc)
    assert r["native window accuracy"] == pytest.approx((pred == y).mean()) and r["A5C windows"] == int((y == 3).sum())
    assert rt.balanced_accuracy(np.array([2, 2]), np.array([2, 2])) is None and rt.margin_auc(np.zeros(3), np.array([2, 2, 2])) is None


def test_window_bootstrap_is_paired_and_seeded():
    rng = np.random.default_rng(1); n = 300; y = rng.choice([2, 3, 0], n); ses = rng.choice([f"s{i}" for i in range(30)], n)
    good = {"pred": y.copy(), "margin": (y == 3).astype(float)}; bad = {"pred": np.where(rng.random(n) < 0.5, y, 4), "margin": rng.normal(size=n)}
    same = rt.window_bootstrap(good, good, y, ses, draws=200)
    assert all(v["delta"] == 0 and v["ci95"] == [0.0, 0.0] for v in same.values())
    a = rt.window_bootstrap(good, bad, y, ses, draws=200); b = rt.window_bootstrap(good, bad, y, ses, draws=200)
    assert a == b and a["A4C/A5C balanced accuracy"]["delta"] > 0 and a["A4C/A5C balanced accuracy"]["ci95"][0] > 0
    assert rt.consistency([a["A4C/A5C balanced accuracy"]] * 3) == rt.POS and rt.consistency([a["A4C/A5C balanced accuracy"], same["A4C/A5C balanced accuracy"]]) == rt.NONE
    with pytest.raises(ValueError):
        rt.window_bootstrap(good, {"pred": y[:-1], "margin": y[:-1]}, y, ses)


def test_window_tables_and_offsets():
    m = rt.mixed_table(np.array([1, 0, 1, 1]), np.array([4, 8, 10, 14]))
    assert m["accuracy"] == 0.75 and m["by_share"]["8 of 16"] == {"windows": 1, "accuracy": 0.0} and m["by_share"]["12-15 of 16"]["accuracy"] == 1.0
    c = rt.changes(np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0]), np.array(["r1", "r1", "r2", "r3"]), np.array(["a", "a", "b", "b"]))
    assert c == {"fixed": {"windows": 1, "recordings": 1, "sessions": 1}, "new": {"windows": 1, "recordings": 1, "sessions": 1}}
    e = rt.end_table(np.array([1, 0, 1, 1]), np.array([0, 8, 8, 15]))
    assert e == {"end_windows": 2, "end_accuracy": 1.0, "middle_windows": 2, "middle_accuracy": 0.5}
    assert sorted(rt.match_offsets(np.array([10, 30]), np.array([12, 29, 31, 60]), 5)) == [-1, 2]       # one-to-one, closest first
    assert rt.offset_summary([0, 2, -2])["mean_abs"] == pytest.approx(4 / 3) and rt.offset_summary([])["matched"] == 0


def _row(seg, rec=0.96, nat=0.99):
    return {h: {"segment_f1_50_multi": seg, "boundary_recall_multi": rec, "native_macro_f1": nat} for h in rt.HALVES}


def test_goal_needs_means_floors_and_feasible_b3():
    g = RV["goal"]
    assert rt.goal([_row(0.99)] * 3, [True] * 3, g)["met"]
    assert not rt.goal([_row(0.99)] * 3, [True, False, True], g)["met"]
    assert not rt.goal([_row(0.99, nat=0.975)] * 3, [True] * 3, g)["met"]                       # calibration floor 0.980
    r = rt.goal([_row(0.95)] * 3, [True] * 3, g); assert not r["met"] and r["margin"]["calibration"] < 0


def _verdicts(ba=None, auc=None, sysv=None):
    within = ("Ordered - Shuffled", "Ordered - Target-only", "Shuffled - Target-only")
    rec = {c: {m: {h: rt.NONE for h in rt.HALVES} for m in rt.REC_METRICS} for c in within}
    for c, v in (ba or {}).items():
        rec[c]["A4C/A5C balanced accuracy"] = {h: v for h in rt.HALVES} if isinstance(v, str) else v
    for c, v in (auc or {}).items():
        rec[c]["A5C-A4C margin AUC"] = {h: v for h in rt.HALVES}
    s = {c: {h: rt.NONE for h in rt.HALVES} for c in (*within, "Ordered - (E1 + B3)")}
    for c, v in (sysv or {}).items():
        s[c] = {h: v for h in rt.HALVES} if isinstance(v, str) else v
    return rec, s


def test_decision_readings():
    rules, no_goal = RV["decision_rules"], {g: {"met": False} for g in tm.GROUPS}
    rules_of = lambda d: [r["rule"] for r in d["rows_that_apply"]]  # noqa: E731
    rec, s = _verdicts(ba={"Ordered - Shuffled": rt.POS, "Ordered - Target-only": rt.POS})
    assert rules_of(rt.decide(rules, rec, s, no_goal)) == ["order_gain"]
    rec, s = _verdicts(ba={"Ordered - Target-only": rt.POS, "Shuffled - Target-only": rt.POS})
    assert rules_of(rt.decide(rules, rec, s, no_goal)) == ["multiframe_gain"]
    rec, s = _verdicts(ba={"Ordered - Shuffled": {"selection": rt.POS, "calibration": rt.NONE}})
    assert rules_of(rt.decide(rules, rec, s, no_goal)) == ["order_gain_one_half", "no_gain"]
    rec, s = _verdicts(auc={"Ordered - Shuffled": rt.POS})
    assert rules_of(rt.decide(rules, rec, s, no_goal)) == ["ranking_only", "no_gain"]
    rec, s = _verdicts(ba={"Ordered - Shuffled": rt.POS}, sysv={"Ordered - Shuffled": {"selection": rt.NEG, "calibration": rt.NONE}})
    d = rt.decide(rules, rec, s, no_goal); assert rules_of(d) == ["order_gain", "recognition_vs_system", "control_wins"]
    rec, s = _verdicts(sysv={"Ordered - (E1 + B3)": rt.POS})
    assert rules_of(rt.decide(rules, rec, s, no_goal)) == ["system_gain"]
    rec, s = _verdicts(ba={"Ordered - Target-only": rt.NEG})
    assert rules_of(rt.decide(rules, rec, s, {**no_goal, "Ordered context": {"met": True}})) == ["no_gain", "control_wins", "goal_met"]


def _toy_seqs():
    nat = {"id": "clip_a", "kind": "native", "labels": np.full(23, 2), "refs": [("clip_a", i, 1.0) for i in range(23)]}
    lab = np.r_[np.zeros(12, int), np.full(10, 3), np.zeros(8, int)]
    st = {"id": "stream_x", "kind": "multi", "labels": lab, "refs": [("clip_b" if i < 12 or i >= 22 else "clip_c", i, 1.0) for i in range(30)]}
    return [nat, st]


def test_evaluation_windows_orders_and_mapping():
    import scripts.report_raw_video_context as rr
    from echo_routing.evaluate.window_target import nearest_centre, window_frames
    from echo_routing.temporal.windowed import window_centres
    seqs = _toy_seqs(); refs = sorted({r for q in seqs for r in q["refs"]}); at = {r: i for i, r in enumerate(refs)}
    W = rr.half_windows(seqs, at, 5)
    assert np.array_equal(W["centre"][W["seq"] == 0], window_centres(23, 5)) and W["s"].shape[1] == 16
    i = int(np.flatnonzero(W["seq"] == 1)[3]); c = int(W["centre"][i])
    assert np.array_equal(W["s"][i], window_frames(c, 30, 16, 16)) and W["marker"][i] == tm.marker_slot(W["s"][i], c)
    assert W["label"][i] == seqs[1]["labels"][c] and [refs[j] for j in W["ref"][i]] == [seqs[1]["refs"][k] for k in W["s"][i]]
    o_sh = rr.slot_orders("Shuffled context", 1, W, seqs); o_or = rr.slot_orders("Ordered context", 1, W, seqs); o_t = rr.slot_orders("Target-only", 1, W, seqs)
    for k in range(len(W["seq"])):
        s, c = W["s"][k], int(W["centre"][k]); keep = tm.target_slots(s, c)
        assert np.array_equal(o_sh[k][keep], keep) and sorted(o_sh[k]) == list(range(16)) and np.all(o_t[k] == W["marker"][k])
        want = tm.shuffle_order(tm.inference_rng(1, seqs[W["seq"][k]]["id"], c), s, c)
        assert np.array_equal(o_sh[k], want) and np.array_equal(o_or[k], np.arange(16))
    assert not np.array_equal(rr.slot_orders("Shuffled context", 2, W, seqs), o_sh)
    logits = np.zeros((len(W["seq"]), 5)); logits[np.arange(len(logits)), np.arange(len(logits)) % 5] = 5.0
    probs = rr.per_sample(W, logits, seqs)
    for qi, q in enumerate(seqs):
        rows = np.flatnonzero(W["seq"] == qi); want = rows[nearest_centre(len(q["labels"]), W["centre"][rows])]
        assert np.array_equal(probs[q["id"]].argmax(1), want % 5)
    assert rr.native_rows(W, seqs).tolist() == np.flatnonzero(W["seq"] == 0).tolist()
    mixed = rr.mixed_rows(W, seqs); assert mixed.size > 0 and all(len(set(W["labels16"][k].tolist())) >= 2 and W["seq"][k] == 1 for k in mixed)


def test_tables_render_partial_results():
    md = rt.tables_md({"recognition": {"ordered_s0": {"selection": rt.recognition(np.array([2, 3]), np.array([0.1, 0.9]), np.array([2, 3]))}}})
    assert "Native window recognition" in md and "pending" in md
