"""Report helpers of the dense target inference round: offsets, unit flags (hit 5 versus near 10), de-duplication,
native stages, the 10 Hz sampling table, the goal with its feasibility recheck, the fixed readings and the tables."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.dense_target_inference_tables as dtt  # noqa: E402
from echo_routing.evaluate.failure_units import Unit, unit_outcomes  # noqa: E402

POS, NEG, NONE = "consistent positive", "consistent negative", "no consistent difference detected"


# ----------------------------------------------------------------------------------------------- offsets
def test_offset_stats_in_samples_and_seconds():
    s = dtt.offset_stats([-2, 0, 3, 1, -4], hz=10.0)
    assert s["matched"] == 5
    assert s["mean_abs"] == pytest.approx(2.0) and s["median_abs"] == pytest.approx(2.0)
    assert s["p90_abs"] == pytest.approx(3.6)                      # linear percentile of 0, 1, 2, 3, 4
    assert s["mean_abs_s"] == pytest.approx(0.2) and s["median_abs_s"] == pytest.approx(0.2) and s["p90_abs_s"] == pytest.approx(0.36)
    assert s["mean_signed"] == pytest.approx(-0.4) and s["exact_share"] == pytest.approx(0.2)
    e = dtt.offset_stats([], hz=10.0)
    assert e["matched"] == 0 and e["mean_abs"] is None and e["p90_abs_s"] is None


# ----------------------------------------------------------------------------------------------- units
def _units(bounds, labels, recs):
    out = []
    for i, ((s, e), y, r) in enumerate(zip(bounds, labels, recs)):
        out.append(Unit("st", i, s, e, y, labels[i - 1] if i else None, labels[i + 1] if i + 1 < len(labels) else None, ((r, "orig", 0, e - s),)))
    return out


def _labels(spans, n):
    out = np.empty(n, int)
    for s, e, y in spans:
        out[s:e] = y
    return out


def test_an_edge_matched_within_near_but_beyond_hit_is_not_a_hit():
    units = _units([(0, 30), (30, 40), (40, 90)], [0, 2, 0], ["r_1", "r_2", "r_3"])
    pred = _labels([(0, 37, 0), (37, 46, 2), (46, 90, 0)], 90)       # changes 7 and 6 samples late
    rows, spurious = unit_outcomes(pred, units, iou=0.5, hit=5, near=10)
    own_raw = _labels([(0, 40, 0), (40, 90, 0)], 90)                # the grid assigns the neighbours' class to the short unit
    dense_raw = _labels([(0, 30, 0), (30, 40, 2), (40, 90, 0)], 90)  # the model's own output at every sample is right
    own_prob = np.eye(5)[own_raw]
    f = dtt.unit_flags(rows[1], units[1], own_raw, own_prob, dense_raw, hit=5, near=10)
    assert f["cause"] == "boundary_shift" and not f["matched"]
    assert (f["edges"], f["hit_edges"], f["near_only_edges"], f["missed_edges"]) == (2, 0, 2, 0)
    assert f["all_edges_hit"] is False and f["extra_cuts"] == 0
    assert f["own_majority_ok"] is False and f["own_mean_ok"] is False and f["dense_majority_ok"] is True
    assert f["readings"] == {"raw classification": False, "sampling position": True, "decoding loss": False, "interval mismatch": True}
    assert f["recordings"] == ["r_2"] and f["sessions"] == ["r"]


def test_a_matched_unit_can_still_have_an_edge_beyond_hit():
    units = _units([(0, 30), (30, 60), (60, 90)], [0, 2, 0], ["r_1", "r_2", "r_3"])
    pred = _labels([(0, 37, 0), (37, 60, 2), (60, 90, 0)], 90)
    rows, _ = unit_outcomes(pred, units, iou=0.5, hit=5, near=10)
    f = dtt.unit_flags(rows[1], units[1], pred, np.eye(5)[pred], pred, hit=5, near=10)
    assert f["matched"] and f["cause"] is None
    assert (f["hit_edges"], f["near_only_edges"]) == (1, 1) and f["all_edges_hit"] is False
    assert not any(f["readings"].values())                         # readings describe wrong units only


def test_unit_table_counts_causes_flags_and_distinct_sources():
    base = {"matched": False, "edges": 2, "hit_edges": 0, "near_only_edges": 0, "missed_edges": 1, "all_edges_hit": False,
            "extra_cuts": 0, "own_majority_ok": True, "own_mean_ok": True, "dense_majority_ok": True}
    rd = {"raw classification": False, "sampling position": False, "decoding loss": True, "interval mismatch": False}
    flags = [{**base, "cause": "missed_cut", "readings": rd, "recordings": ["a_1"], "sessions": ["a"]},
             {**base, "cause": "missed_cut", "readings": rd, "recordings": ["a_1", "a_2"], "sessions": ["a"]},
             {**base, "cause": None, "matched": True, "missed_edges": 0, "hit_edges": 2, "all_edges_hit": True,
              "readings": dict.fromkeys(rd, False), "recordings": ["b_1"], "sessions": ["b"]}]
    t = dtt.unit_table(flags, [{"inside": True}, {"inside": False}])
    assert (t["units"], t["matched"], t["wrong"]) == (3, 1, 2)
    assert t["by_cause"]["missed_cut"] == {"units": 2, "recordings": 2, "sessions": 1}
    assert t["by_cause"]["label"] == {"units": 0, "recordings": 0, "sessions": 0}
    assert t["readings"]["decoding loss"] == {"units": 2, "recordings": 2, "sessions": 1}
    assert t["flags_wrong"]["missed edge"]["units"] == 2 and t["flags_wrong"]["all edges within hit"]["units"] == 0
    assert t["matched_edges"] == {"units": 1, "all edges within hit": 1, "an edge only within near": 0}
    assert t["spurious"] == {"segments": 2, "inside_host": 1}


# ----------------------------------------------------------------------------------------------- natives
def test_native_stages_separate_raw_and_decode_and_count_sessions():
    clips = [{"video_id": "x_1", "label": 2, "raw": 2, "mean": 2, "decode": 3, "dense_raw": 2},   # raw right, decode wrong
             {"video_id": "x_2", "label": 2, "raw": 3, "mean": 3, "decode": 3, "dense_raw": 3},   # both wrong, the model itself
             {"video_id": "y_1", "label": 3, "raw": 2, "mean": 3, "decode": 3, "dense_raw": 3},   # raw wrong, decode right
             {"video_id": "z_1", "label": 4, "raw": 0, "mean": 0, "decode": 0, "dense_raw": 4},   # sampling: own raw wrong, dense right
             {"video_id": "w_1", "label": 0, "raw": 0, "mean": 0, "decode": 0, "dense_raw": 0}]
    t = dtt.native_stages(clips)
    assert t["clips"] == 5
    assert t["decode wrong"]["clips"] == 3 and t["decode wrong"]["sessions"] == 2
    assert t["decode wrong"]["pairs"] == {"A4C->A5C": 2, "SC4C->PLAX": 1}
    assert t["raw right, decode wrong"]["clips"] == 1 and t["raw right, decode wrong"]["pairs"] == {"A4C->A5C": 1}
    assert t["both wrong"]["clips"] == 2 and t["raw wrong, decode right"]["clips"] == 1
    assert t["mean label right among decode wrong"] == 1
    assert t["readings"]["raw classification"]["clips"] == 1
    assert t["readings"]["sampling position"]["clips"] == 1
    assert t["readings"]["decoding loss"]["clips"] == 1
    assert dtt.localisation(t) == "mixed"                                   # one decoding, one sampling, one recognition clip
    assert dtt.localisation(dtt.native_stages(clips[:1])) == "decoding"
    assert dtt.localisation(dtt.native_stages(clips[1:2])) == "recognition"
    assert dtt.localisation(dtt.native_stages(clips[3:4])) == "sampling"
    assert dtt.localisation(dtt.native_stages(clips[4:])) == "no errors"
    other = {"video_id": "v_1", "label": 1, "raw": 1, "mean": 1, "decode": 0, "dense_raw": 1}
    assert dtt.localisation(dtt.native_stages(clips[:2] + [other])) == "decoding"      # 2 of 3 recordings with a right raw majority
    u = dtt.native_union([clips[:1], clips[:2], []])                                    # x_1 wrong in two seeds counts once
    assert u["decode wrong"]["recordings"] == 2 and u["decode wrong"]["sessions"] == 1
    assert u["decode wrong"]["pairs"]["A4C->A5C"] == {"recordings": 2, "sessions": 1}
    assert u["localisation"] == "mixed"


def test_union_over_seeds_counts_each_recording_and_session_once():
    assert dtt.union_ids([["a_1", "a_2"], ["a_1", "b_1"], []]) == {"recordings": 3, "sessions": 2}


def test_native_changes_by_pair_with_sessions():
    a = [{"video_id": "x_1", "label": 2, "decode": 2}, {"video_id": "x_2", "label": 2, "decode": 2}, {"video_id": "y_1", "label": 3, "decode": 2}]
    b = [{"video_id": "x_1", "label": 2, "decode": 3}, {"video_id": "x_2", "label": 2, "decode": 3}, {"video_id": "y_1", "label": 3, "decode": 3}]
    c = dtt.native_pair_changes(a, b)            # a fixes what b gets wrong, and the reverse
    assert c["A4C->A5C"] == {"fixed": 2, "fixed_recordings": 2, "fixed_sessions": 1, "new": 0, "new_recordings": 0, "new_sessions": 0}
    assert c["A5C->A4C"]["new"] == 1 and c["A5C->A4C"]["new_sessions"] == 1
    assert c["all clips"]["fixed"] == 2 and c["all clips"]["new"] == 1


# ----------------------------------------------------------------------------------------------- 10 Hz sampling
def test_sampling_table_compares_conditions_sample_by_sample():
    y = np.array([0, 0, 1, 1, 2]); sparse = np.array([0, 1, 1, 0, 0]); dense = np.array([0, 0, 1, 1, 1])
    recs = np.array(["r_1", "r_1", "r_2", "q_1", "q_1"])
    t = dtt.sampling_pair(y, dense, sparse, recs)
    assert t["samples"] == 5 and t["a right"] == 4 and t["b right"] == 2
    assert t["a right, b wrong"] == {"samples": 2, "recordings": 2, "sessions": 2}
    assert t["b right, a wrong"] == {"samples": 0, "recordings": 0, "sessions": 0}
    assert t["both wrong"] == {"samples": 1, "recordings": 1, "sessions": 1}


# ----------------------------------------------------------------------------------------------- goal and readings
G = {"selection_min": 0.944409687184662, "calibration_min": 0.9821421709894332, "calibration_sensitivity": 0.9838935784747763,
     "boundary_recall_min": 0.95, "native_macro_f1_min": {"selection": 0.973, "calibration": 0.980}}


def _row(f1, rec=0.97, nat=0.986):
    return {"segment_f1_50_multi": f1, "boundary_recall_multi": rec, "native_macro_f1": nat}


def test_goal_rechecks_the_fixed_setting_on_calibration():
    rows = [{"selection": _row(0.95), "calibration": _row(0.985)}, {"selection": _row(0.95), "calibration": _row(0.985, nat=0.979)},
            {"selection": _row(0.95), "calibration": _row(0.985)}]
    g = dtt.goal_check(rows, G)
    assert g["feasible"] == [True, False, True] and not g["met"]
    assert any(f["seed"] == 1 and f["half"] == "calibration" for f in g["constraint_failures"])
    rows[1]["calibration"] = _row(0.985)
    g = dtt.goal_check(rows, G)
    assert g["feasible"] == [True, True, True] and g["met"]


def _ver(f1=NONE, nat=NONE, cuts=NONE, scr=NONE):
    return {"multi segment F1@50": f1, "native macro-F1": nat, "native cuts per clip": cuts, "same-class cut rate": scr,
            "multi boundary recall": NONE, "multi boundary F1": NONE}


def _goals(dense_ok=True, met=False):
    fail = [] if dense_ok else [{"seed": 2, "half": "selection"}]
    return {c: {"constraint_failures": fail if c == "Dense" else [], "met": met} for c in ("Sparse", "Interpolated", "Dense")}


def _offsets(dense_lower=True):
    return {f"{c}|s{s}|{h}": (1.0 if (c == "Dense") == dense_lower else 1.5) for c in ("Sparse", "Interpolated", "Dense")
            for s in (0, 1, 2) for h in ("selection", "calibration")}


def test_readings_no_gain_with_finer_offsets():
    ver = {"b3|Dense - Sparse": {h: _ver() for h in ("selection", "calibration")},
           "b3|Dense - Interpolated": {h: _ver() for h in ("selection", "calibration")}}
    d = dtt.decide(ver, _offsets(True), _goals(), {"Dense|selection": "recognition"})
    assert d["applied"] == ["offset_only", "interpolation_close", "native_localisation", "end"]
    assert d["rows"]["interpolation_close"]["halves"] == ["selection", "calibration"]


def test_readings_gain_needs_both_halves_and_every_constraint():
    ver = {"b3|Dense - Sparse": {h: _ver(f1=POS) for h in ("selection", "calibration")},
           "b3|Dense - Interpolated": {h: _ver(f1=POS) for h in ("selection", "calibration")}}
    d = dtt.decide(ver, _offsets(True), _goals(dense_ok=True, met=True), {})
    assert d["applied"] == ["dense_system_gain", "goal_met", "end"]
    d = dtt.decide(ver, _offsets(True), _goals(dense_ok=False), {})
    assert d["applied"] == ["gain_without_constraints", "end"]
    ver["b3|Dense - Sparse"]["calibration"] = _ver(f1=POS, cuts=POS)
    ver["b3|Dense - Sparse"]["selection"] = _ver(f1=NONE)
    d = dtt.decide(ver, _offsets(False), _goals(), {})
    assert d["applied"] == ["dense_costs", "end"] and d["rows"]["dense_costs"]["halves"] == ["calibration"]


# ----------------------------------------------------------------------------------------------- tables
CONDS = ("Sparse", "Interpolated", "Dense")
HALVES = ("selection", "calibration")
NAMES = ("Dense - Sparse", "Dense - Interpolated", "Interpolated - Sparse")
METRICS = ("native macro-F1", "native cuts per clip", "multi segment F1@50", "multi boundary recall", "multi boundary F1", "same-class cut rate")


def _sysrow(f1):
    return {"segment_f1_50_multi": f1, "boundary_precision_multi": 0.94, "boundary_recall_multi": 0.97, "boundary_f1_multi": 0.955,
            "native_macro_f1": 0.98, "native_internal_cuts_per_clip": 0.04, "same_class_cut_rate_multi": 0.02, "segment_f1_50_pairs": 0.97,
            "boundary_recall_pairs": 0.98, "native_recall": dict.fromkeys(("PLAX", "PSAX", "A4C", "A5C", "SC4C"), 0.97),
            "A4C read as A5C": 2, "A5C read as A4C": 0, "multi_frame_accuracy": 0.97, "native_frame_accuracy": 0.98}


def _res():
    clip = [{"video_id": "x_1", "label": 2, "raw": 2, "mean": 2, "decode": 3, "dense_raw": 2}]
    mech = {k: {"segments": 7, "matched": 5, "wrong view": 0, "no segment at IoU 0.5": 2, "taken by another reference": 0, "recall": 5 / 7,
                "missed_recordings": 2, "missed_sessions": 2} for k in ("short segments", "short A4C or A5C segments", "X-Y-X middles", "A4C/A5C X-Y-X middles")}
    rows = [{h: _sysrow(0.96) for h in HALVES} for _ in range(3)]
    ver = {f"{r}|{n}": {h: dict.fromkeys(METRICS, NONE) for h in HALVES} for r in ("b3", "argmax") for n in NAMES}
    boot = {m: {"delta": 0.001, "ci95": [-0.002, 0.004]} for m in METRICS}
    y = np.array([0, 1, 2]); recs = np.array(["r_1", "r_1", "q_1"])
    seg = {k: {"segments": 1, "recordings": 1, "sessions": 1} for k in ("matched by a only", "matched by b only")}
    bnd = {k: {"changes": 2, "recordings": 2, "sessions": 1} for k in ("found by a only", "found by b only")}
    return {
        "counts": {h: {"sequences": {"native": 3, "pairs": 2, "multi": 4}, "samples": 900, "old_windows": 190, "new_windows": 710} for h in HALVES},
        "systems": {f"{c}|s{s}": {"argmax": {h: _sysrow(0.9) for h in HALVES}, "b3": {h: _sysrow(0.96) for h in HALVES}, "feasible": True}
                    for c in CONDS for s in range(3)},
        "comparisons": {f"s{s}|{r}|{n}|{h}": {"bootstrap": boot, "segment_changes": seg, "boundary_changes": bnd}
                        for s in range(3) for r in ("b3", "argmax") for n in NAMES for h in HALVES},
        "verdicts": ver,
        "offsets": {f"{c}|{r}|s{s}|{h}": dtt.offset_stats([1, -2, 0], 10.0) for c in CONDS for r in ("b3", "argmax") for s in range(3) for h in HALVES},
        "mechanism": {f"{c}|b3|s{s}|{h}": mech for c in CONDS for s in range(3) for h in HALVES},
        "units": {f"{c}|s{s}|{h}|{k}": dtt.unit_table([], []) for c in CONDS for s in range(3) for h in HALVES for k in ("multi", "pairs")},
        "natives": {f"{c}|s{s}|{h}": dtt.native_stages(clip) for c in CONDS for s in range(3) for h in HALVES},
        "natives_union": {f"{c}|{h}": dtt.native_union([clip, clip, []]) for c in CONDS for h in HALVES},
        "native_changes": {f"s{s}|{n}|{h}": dtt.native_pair_changes(clip, clip) for s in range(3) for n in NAMES for h in HALVES},
        "sampling": {f"s{s}|{h}|{k}|{p}": dtt.sampling_pair(y, y, y, recs) for s in range(3) for h in HALVES for k in ("native", "pairs", "multi")
                     for p in NAMES},
        "goal": {c: dtt.goal_check(rows, G) for c in CONDS},
        "decision": dtt.decide({"b3|Dense - Sparse": ver["b3|Dense - Sparse"], "b3|Dense - Interpolated": ver["b3|Dense - Interpolated"]},
                               _offsets(True), _goals(), {"Dense|selection": "decoding"}),
        "cases": {"native floor": {"clips": 3, "recordings": 3, "sessions": 2}, "calibration": {"clips": 4, "units": 9, "recordings": 11, "sessions": 8}},
        "cost": {"new_windows": 4260, "ms_per_new_window": 16.2, "gpu_hours": {"checks": 0.1, "run": 1.2, "total": 1.3}},
    }


def test_tables_render_every_section_without_ids():
    md = dtt.tables_md(_res())
    for head in ("## Systems with the fixed B3 numbers", "## Direct argmax", "## Comparisons", "## Matched boundary offsets",
                 "## Reference units", "## Native clips", "## The 10 Hz axis", "## Goal", "## Decision"):
        assert head in md
    assert "| Sparse | 0 |" in md and "Dense - Sparse" in md and "pending" not in md
    assert "x_1" not in md and "r_1" not in md                 # counts only, never a recording
    for block in md.split("\n\n"):                            # every table row has its header's number of columns
        rows = [x for x in block.splitlines() if x.startswith("|")]
        assert len({x.count("|") for x in rows}) <= 1, rows[0][:80]
    assert "pending" in dtt.tables_md({})                      # missing sections render as pending, not as a crash
