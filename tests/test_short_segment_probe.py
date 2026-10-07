"""Pure helpers of scripts/short_segment_probe.py on synthetic data: the session partition and per-part streams, the
X -> Y -> X candidates and their categories, the features, the operating point, the bootstrap and the decision."""
import numpy as np
import pytest

import scripts.short_segment_probe as sp
from echo_routing.temporal.segments import Segment


def test_partition_is_by_session_and_seeded():
    sessions = [f"p{i}" for i in range(50)]
    a = sp.partition_sessions(sessions + sessions[:5], {"train": 0.6, "calibration": 0.2, "held_out": 0.2}, 7)
    assert set(a) == set(sessions) and a == sp.partition_sessions(sessions, {"train": 0.6, "calibration": 0.2, "held_out": 0.2}, 7)
    counts = {p: sum(v == p for v in a.values()) for p in sp.PARTS}
    assert counts == {"train": 30, "calibration": 10, "held_out": 10}
    assert a != sp.partition_sessions(sessions, {"train": 0.6, "calibration": 0.2, "held_out": 0.2}, 8)


def test_banks_never_cross_parts():
    clips = [f"s{i}_{k}" for i in range(40) for k in range(1, 4)]
    labels = {v: (int(v.split("_")[0][1:]) + int(v.split("_")[1])) % 5 for v in clips}; n = {v: 45 for v in clips}
    cfg = {"multi": {"n_fragments": [4, 8], "edit_prob": 0.5, "p_same_next": 0.35}, "recipes": {"edit_variant": "gamma090", "min_len": 10, "max_len": 40}}
    conf = {"partition": {"fractions": {"train": 0.6, "calibration": 0.2, "held_out": 0.2}, "seed": 3},
            "streams": {"n_streams": {"train": 30, "calibration": 10, "held_out": 10}, "seeds": {"train": 1, "calibration": 2, "held_out": 3}}}
    part_of, banks = sp.make_banks(cfg, conf, None, {"train": {"clips": clips}}, labels, n)
    for part, b in banks.items():
        assert b["recipes"] and all(part_of[f["video_id"].rsplit("_", 1)[0]] == part for r in b["recipes"] for f in r["fragments"])
        assert all(part_of[v.rsplit("_", 1)[0]] == part for v in b["natives"])
    assert sorted(v for b in banks.values() for v in b["natives"]) == sorted(clips)


def test_sandwiches():
    pred = np.array([2] * 5 + [3] * 2 + [2] * 4 + [0] * 3 + [1] * 2 + [0] * 3)
    got = [(l.label, m.label, r.label, m.start, m.end) for l, m, r in sp.sandwiches(pred)]
    assert got == [(2, 3, 2, 5, 7), (0, 1, 0, 14, 16)]
    assert sp.sandwiches(np.array([1, 1, 2, 2, 3, 3])) == []


def _seg(a, b, lab):
    return Segment(a, b, lab)


def test_categories_and_splice():
    ref = np.array([2] * 10 + [3] * 12 + [2] * 10)                       # a true A5C unit between A4C units, joins at 10 and 22
    joins = np.array([10, 22])
    assert sp.categorise(_seg(0, 11, 2), _seg(11, 22, 3), _seg(22, 32, 2), ref, joins, 5) == ("kept", True)
    ref_b = np.full(32, 2)                                               # one A4C view; the middle is a wrong A5C run
    assert sp.categorise(_seg(0, 14, 2), _seg(14, 17, 3), _seg(17, 32, 2), ref_b, np.zeros(0, int), 5) == ("removed", False)
    assert sp.categorise(_seg(0, 14, 2), _seg(14, 17, 3), _seg(17, 32, 2), ref_b, np.array([12]), 5) == ("removed", True)   # same-view splice nearby
    ref_c = np.array([2] * 14 + [4] * 3 + [2] * 15)                      # a third class under the middle
    assert sp.categorise(_seg(0, 14, 2), _seg(14, 17, 3), _seg(17, 32, 2), ref_c, np.zeros(0, int), 5)[0] == "other"
    ref_d = np.full(32, 0)                                               # the outer runs are wrong
    assert sp.categorise(_seg(0, 14, 2), _seg(14, 17, 3), _seg(17, 32, 2), ref_d, np.zeros(0, int), 5)[0] == "other"


def test_features():
    rng = np.random.default_rng(0); prob = rng.dirichlet(np.ones(5), size=40).astype(np.float32); feat = rng.standard_normal((40, 8)).astype(np.float32)
    left, mid, right = _seg(0, 20, 2), _seg(20, 24, 3), _seg(24, 40, 2)
    jsd = np.linspace(0, 1, 40)
    q0 = sp.q0_features(prob, jsd, left, mid, right, 16, 1e-6)
    assert q0.shape == (3 + 5 + 5 + 15 + 5,) and q0[0] == pytest.approx(np.log(4)) and q0[3 + 2] == 1 and q0[8 + 3] == 1
    assert q0[-2] == pytest.approx(jsd[20]) and q0[-1] == pytest.approx(jsd[24])
    v, c = sp.visual_features(feat, left, mid, right, 16)
    assert v.shape == (24,) and np.allclose(v[:8], feat[20:24].mean(0)) and np.allclose(v[8:16], feat[4:20].mean(0)) and np.allclose(v[16:], feat[24:40].mean(0))
    assert -1 <= c[0] <= 1 and -1 <= c[1] <= 1
    arrays = {"q0": np.zeros((3, 33)), "vis": np.zeros((3, 24)), "sim": np.zeros((3, 2))}
    assert sp.design(arrays, "Q0").shape == (3, 33) and sp.design(arrays, "Q1").shape == (3, 59) and sp.design(arrays, "Q0+sim").shape == (3, 35)


def test_operating_point_and_rates():
    kept = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    t = sp.threshold_for(kept, 0.9)
    assert t == 0.1 and (kept >= t).mean() >= 0.9 and (kept >= 0.2).mean() < 0.9            # the highest qualifying threshold
    r = sp.rates(np.array([0.9, 0.05, 0.02, 0.3]), np.array([1, 1, 0, 0]), 0.1)
    assert r == {"keep_rate_kept": 0.5, "removal_rate_removed": 0.5, "n_kept": 2, "n_removed": 2}


def test_bootstrap_is_paired_and_zero_for_identical_scores():
    rng = np.random.default_rng(1); s = rng.uniform(size=60); y = (rng.uniform(size=60) > 0.5).astype(int); g = np.repeat(np.arange(20), 3)
    b = sp.bootstrap_diff(s, s, y, 0.4, 0.4, g, 200, 5)
    assert b["removal"] == {"delta": 0.0, "ci95": [0.0, 0.0]} and b["keep"]["delta"] == 0.0
    assert sp.bootstrap_diff(s, s - 0.2, y, 0.4, 0.4, g, 200, 5)["removal"]["delta"] < 0     # lower scores remove more


def _seed_result(lo, keep, splice):
    return {"Q1 - Q0": {"0.95": {"bootstrap": {"removal": {"ci95": [lo, lo + 0.1]}, "keep": {"delta": keep}}, "splice_removed_delta": splice}}}


def test_sufficiency_and_decision():
    meta = [{"cat": "kept", "mid_sessions": [f"a{i}"]} for i in range(30)] + [{"cat": "removed", "mid_sessions": [f"b{i % 10}"]} for i in range(40)]
    s = sp.sufficient(meta, {"candidates": 30, "sessions": 15})
    assert s["kept"] == {"candidates": 30, "sessions": 30} and s["removed"]["sessions"] == 10 and not s["ok"]
    assert sp.decide([_seed_result(0.01, 0.0, 0.0)] * 3)["extra_value"]
    assert not sp.decide([_seed_result(0.01, 0.0, 0.0), _seed_result(0.0, 0.0, 0.0), _seed_result(0.01, 0.0, 0.0)])["extra_value"]
    assert not sp.decide([_seed_result(0.01, -0.03, 0.0)] * 3)["extra_value"]                 # keeps too few true segments
    assert not sp.decide([_seed_result(0.01, 0.0, -0.1)] * 3)["extra_value"]                  # worse on spliced wrong segments
    assert sp.decide([_seed_result(0.01, 0.0, None)] * 3)["extra_value"]                      # no spliced (b) to judge
