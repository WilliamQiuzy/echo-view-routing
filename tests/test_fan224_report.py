"""Pure helpers of scripts/report_fan224_visual.py on synthetic tables and sequences (no model, no data)."""
import numpy as np
import pandas as pd
import pytest

import scripts.report_fan224_visual as rp
from echo_routing.evaluate.metrics_segmentation import segment_f1
from echo_routing.temporal.segments import labels_to_segments
from scripts.report_model_development import paired_bootstrap


def _tables(seed: int, n_sessions: int = 12, n_streams: int = 10) -> dict:
    rng = np.random.default_rng(seed); nat = []
    for s in range(n_sessions):
        for k in range(2):
            y = int(rng.integers(5)); nat.append({"video_id": f"sess{s:02d}_{k}", "label": y, "n": 30,
                                                  "majority": y if rng.random() < 0.85 else int((y + 1) % 5), "cuts": int(rng.integers(0, 3)), "frame_correct": 25})
    seg = [{"stream": f"st{i}", "seg_matched": int(rng.integers(3, 7)), "seg_pred": int(rng.integers(6, 9)), "seg_true": 7, "n": 300, "correct": 250} for i in range(n_streams)]
    frag = [{"stream": f"st{i}", "t_matched": int(rng.integers(3, 6)), "t_true": 6, "p_pred": int(rng.integers(5, 9)), "false_split": int(rng.integers(0, 2)),
             "same_join": 2} for i in range(n_streams) for _ in range(2)]
    return {"native": pd.DataFrame(nat), "multi_seg": pd.DataFrame(seg), "multi_frag": pd.DataFrame(frag)}


def test_two_system_contrast_equals_the_paired_bootstrap():
    a, b = _tables(1), _tables(2)
    b["native"]["video_id"] = a["native"]["video_id"]
    pb = paired_bootstrap(a, b); cb = rp.contrast_bootstrap({"a": a, "b": b}, {"a": 1.0, "b": -1.0})
    assert set(pb) == set(cb)
    for k in pb:
        assert pb[k]["delta"] == pytest.approx(cb[k]["delta"], abs=1e-12) and np.allclose(pb[k]["ci95"], cb[k]["ci95"], atol=1e-12)


def test_interaction_of_equal_differences_is_zero():
    a, b = _tables(3), _tables(4); b["native"]["video_id"] = a["native"]["video_id"]
    res = rp.contrast_bootstrap({"F1": a, "F0": b, "L1": a, "L0": b}, rp.INTERACTION)
    for v in res.values():
        assert v["delta"] == pytest.approx(0.0, abs=1e-12) and v["ci95"] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_same_bootstrap_treats_matching_nans_as_equal():
    a, b = _tables(8), _tables(9); b["native"]["video_id"] = a["native"]["video_id"]
    for t in (a, b):
        t["multi_frag"]["same_join"] = 0; t["multi_frag"]["false_split"] = 0     # same-class cut rate is 0/0
    with np.errstate(all="ignore"):
        chk = rp.same_bootstrap(paired_bootstrap(a, b), rp.contrast_bootstrap({"a": a, "b": b}, {"a": 1.0, "b": -1.0}))
    assert all(chk.values())
    bad = {k: {"delta": 1.0, "ci95": [0.0, 2.0]} for k in chk}; good = {k: {"delta": 1.0, "ci95": [0.0, 2.1]} for k in chk}
    assert not any(rp.same_bootstrap(bad, good).values())


def test_contrast_refuses_different_units():
    a, b = _tables(5), _tables(6); b["native"].loc[0, "video_id"] = "other_9"
    with pytest.raises(ValueError, match="native clips"):
        rp.contrast_bootstrap({"a": a, "b": b}, {"a": 1.0, "b": -1.0})


def test_per_segment_matching_agrees_with_segment_f1():
    rng = np.random.default_rng(7)
    for _ in range(200):
        true = np.repeat(rng.integers(0, 5, 6), rng.integers(3, 12, 6))
        pred = true.copy(); flips = rng.integers(0, true.size, 5); pred[flips] = rng.integers(0, 5, 5)
        p, t = labels_to_segments(pred), labels_to_segments(true)
        assert len(rp.matched_true(p, t)) == segment_f1(p, t, 0.5)["matched"]


def test_segment_changes_count_segments_recordings_and_sessions():
    labels = np.array([0] * 10 + [2] * 10 + [3] * 10)
    refs = [("sA_1", i, 1.0) for i in range(10)] + [("sB_1", i, 1.0) for i in range(10)] + [("sB_2", i, 1.0) for i in range(5)] + [("sC_1", i, 0.9) for i in range(5)]
    seqs = [{"id": "m1", "kind": "multi", "labels": labels, "refs": refs}, {"id": "n1", "kind": "native", "labels": np.zeros(5, int), "refs": []}]
    good = {"m1": labels.copy(), "n1": np.zeros(5, int)}
    bad = {"m1": np.array([0] * 10 + [3] * 20), "n1": np.zeros(5, int)}   # the A4C segment read as A5C
    ch = rp.segment_changes(good, bad, seqs)
    # the 20-sample run of 3s still matches the 10-sample 3 segment (IoU 0.5); only the 2 segment (one recording) is lost
    assert ch["fixed"] == {"segments": 1, "recordings": 1, "sessions": 1} and ch["new"] == {"segments": 0, "recordings": 0, "sessions": 0}
    assert rp.segment_changes(bad, good, seqs)["new"]["segments"] == 1
    other = {"m1": np.array([0] * 10 + [2] * 10 + [4] * 10), "n1": np.zeros(5, int)}   # the last segment, built from two recordings
    assert rp.segment_changes(good, other, seqs)["fixed"] == {"segments": 1, "recordings": 2, "sessions": 2}


def test_goal_requires_feasible_b3_and_every_floor():
    def row(sel, cal, feasible=True, rs=0.96, rc=0.98, ns=0.98, nc=0.99):
        return {"b3_choice": {"status": "feasible" if feasible else "infeasible (diagnostic setting shown)"},
                "b3|selection": {"segment_f1_50_multi": sel, "boundary_recall_multi": rs, "native_macro_f1": ns},
                "b3|calibration": {"segment_f1_50_multi": cal, "boundary_recall_multi": rc, "native_macro_f1": nc}}
    fv = {"goal": {"selection_min": 0.944409687184662, "calibration_min": 0.9821421709894332, "calibration_sensitivity": 0.9738935784747763}}
    assert rp.goal(fv, [row(0.95, 0.99)] * 3)["met"]
    assert not rp.goal(fv, [row(0.95, 0.99), row(0.95, 0.99), row(0.95, 0.99, ns=0.97)])["met"]
    assert not rp.goal(fv, [row(0.95, 0.99), row(0.95, 0.99), row(0.95, 0.99, feasible=False)])["met"]
    assert not rp.goal(fv, [row(0.94, 0.99)] * 3)["met"]
    g = rp.goal(fv, [row(0.95, 0.99)] * 3)
    assert g["sensitivity_calibration"] == pytest.approx(0.99 - 0.9838935784747763)


def test_strip_ids_and_frames_kind():
    assert rp.strip_ids({"a": 1, "_ids": [1], "b": [{"_x": 2, "c": 3}]}) == {"a": 1, "b": [{"c": 3}]}
    with pytest.raises(ValueError):
        rp.Frames([], None, "other")
