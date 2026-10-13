"""Report helpers of the video feature temporal round: ASRF edge proposal and merge status, unit and native readings,
the frozen diagnostic list, the overlap with Linear + B3, the fixed readings and the tables. Expected values are worked
out by hand."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.video_feature_temporal_tables as vtt  # noqa: E402
from echo_routing.evaluate.failure_units import Unit, unit_outcomes  # noqa: E402

POS, NEG, NONE = "consistent positive", "consistent negative", "no consistent difference detected"


def _labels(spans, n):
    out = np.empty(n, int)
    for s, e, y in spans:
        out[s:e] = y
    return out


def _units(bounds, labels, recs):
    return [Unit("st", i, s, e, y, labels[i - 1] if i else None, labels[i + 1] if i + 1 < len(labels) else None, ((r, "orig", 0, e - s),))
            for i, ((s, e), y, r) in enumerate(zip(bounds, labels, recs))]


# ----------------------------------------------------------------------------------------------- edges
def test_edge_status_separates_proposed_merged_and_missing_edges():
    st = vtt.edge_status([30, 60], peaks=np.array([0, 33, 90]), refined_changes=np.array([33]), hit=5)
    assert st == {"edges": 2, "proposed": 1, "merged": 0, "not_proposed": 1}
    st = vtt.edge_status([30], peaks=np.array([0, 31]), refined_changes=np.array([], int), hit=5)
    assert st == {"edges": 1, "proposed": 1, "merged": 1, "not_proposed": 0}
    assert vtt.edge_status([30], peaks=np.array([0, 36]), refined_changes=np.array([36]), hit=5)["not_proposed"] == 1   # 6 samples away


def test_peaks_follow_the_official_rule_and_leave_the_input_unchanged():
    b = np.array([0.9, 0.2, 0.7, 0.4, 0.3, 0.6, 0.6, 0.1], np.float32); keep = b.copy()
    assert vtt.boundary_peaks(b, 0.5).tolist() == [0, 2]       # 0.6, 0.6 is a plateau, not a strict maximum
    assert np.array_equal(b, keep)


# ----------------------------------------------------------------------------------------------- ASRF readings
def test_asrf_unit_readings_name_where_an_unmatched_unit_fails():
    units = _units([(0, 30), (30, 40), (40, 90)], [0, 2, 0], ["r_1", "r_2", "r_3"])
    refined = _labels([(0, 90, 0)], 90)                          # the refinement merged the short unit away
    rows, _ = unit_outcomes(refined, units, iou=0.5, hit=5, near=10)
    frame = _labels([(0, 30, 0), (30, 40, 2), (40, 90, 0)], 90)  # the class branch had it right
    source = _labels([(0, 90, 0)], 90)                           # the source classifier did not
    u = vtt.asrf_unit(rows[1], units[1], frame, source, peaks=np.array([0, 31, 41]), refined_changes=np.array([], int), hit=5)
    assert not u["matched"] and u["class_branch_right"] and not u["source_right"]
    assert (u["edges"], u["proposed"], u["merged"], u["not_proposed"]) == (2, 2, 2, 0)
    assert u["readings"] == {"source right, class branch wrong": False, "class branch right, refinement wrong": True,
                             "edge not proposed": False, "proposed but merged": True}
    assert u["recordings"] == ["r_2"] and u["sessions"] == ["r"]


def test_asrf_native_readings():
    a = vtt.asrf_native({"video_id": "x_1", "label": 2, "decode": 3, "frame": 2, "source": 2})
    assert a["readings"] == {"source right, class branch wrong": False, "class branch right, refinement wrong": True}
    b = vtt.asrf_native({"video_id": "x_2", "label": 2, "decode": 3, "frame": 3, "source": 2})
    assert b["readings"] == {"source right, class branch wrong": True, "class branch right, refinement wrong": False}
    c = vtt.asrf_native({"video_id": "x_3", "label": 2, "decode": 2, "frame": 3, "source": 3})
    assert not any(c["readings"].values())                      # a right clip has no reading


# ----------------------------------------------------------------------------------------------- the frozen list and overlaps
def test_the_frozen_list_counts_fixed_units_and_new_misses_with_sources():
    src = {("a", 0): ["r_1"], ("a", 1): ["r_2"], ("b", 0): ["q_1"], ("b", 3): ["q_2"]}
    listed = {("a", 0), ("a", 1)}; ordered_matched = {("b", 0), ("b", 3)}; new_matched = {("a", 1), ("b", 0)}
    out = vtt.list_outcomes(listed, ordered_matched, new_matched, src.__getitem__)
    assert out["listed"] == {"units": 2, "recordings": 2, "sessions": 1}
    assert out["fixed"] == {"units": 1, "recordings": 1, "sessions": 1}
    assert out["new_misses"] == {"units": 1, "recordings": 1, "sessions": 1}


def test_overlap_with_linear():
    assert vtt.overlap({1, 2, 3, 4}, {2, 3, 9}) == {"fixed": 4, "also_by_linear": 2, "share": 0.5}
    assert vtt.overlap(set(), {1})["share"] is None


# ----------------------------------------------------------------------------------------------- readings
def _v(f1=NONE, nat=NONE, cuts=NONE, scr=NONE):
    return {"multi segment F1@50": f1, "native macro-F1": nat, "native cuts per clip": cuts, "same-class cut rate": scr,
            "multi boundary recall": NONE, "multi boundary F1": NONE}


def _ver(asrf_lin=NONE, asrf_ord=NONE, asrf_ord_cal=None, asrf_ms=NONE, nat_cal=NONE, ms_ord=NONE, cuts=NONE):
    out = {}
    for name, sel, cal in (("ASRF - (Linear + B3)", asrf_lin, asrf_lin), ("ASRF - (frozen Ordered + B3)", asrf_ord, asrf_ord if asrf_ord_cal is None else asrf_ord_cal),
                           ("ASRF - MS-TCN", asrf_ms, asrf_ms), ("MS-TCN - (frozen Ordered + B3)", ms_ord, ms_ord)):
        out[name] = {"selection": _v(f1=sel, cuts=cuts), "calibration": _v(f1=cal, nat=nat_cal if name.startswith("ASRF - (frozen") else NONE, cuts=cuts)}
    return out


GOALS_OK = {"ASRF": {"constraint_failures": [], "met": False}}
GOALS_BAD = {"ASRF": {"constraint_failures": [{"seed": 2}], "met": False}}


def test_readings_system_gain_needs_both_comparators_both_halves_and_the_constraints():
    d = vtt.decide(_ver(asrf_lin=POS, asrf_ord=POS), GOALS_OK, overlap_share=0.2, trade={})
    assert d["applied"] == ["system_gain", "end"]
    d = vtt.decide(_ver(asrf_lin=POS, asrf_ord=POS), GOALS_BAD, overlap_share=0.2, trade={})
    assert "system_gain" not in d["applied"]


def test_readings_selection_only_linear_explains_tradeoff_and_no_gain():
    d = vtt.decide(_ver(asrf_ord=POS, asrf_ord_cal=NONE), GOALS_OK, overlap_share=0.8, trade={"ASRF|calibration": True})
    assert d["applied"] == ["selection_only", "linear_explains", "tradeoff", "end"]
    d = vtt.decide(_ver(asrf_ms=NEG), GOALS_OK, overlap_share=None, trade={})
    assert d["applied"] == ["no_gain", "end"] and d["rows"]["no_gain"]["mstcn_leads_in"] == ["selection", "calibration"]


# ----------------------------------------------------------------------------------------------- tables
def test_tables_render_pending_sections_and_no_ids():
    md = vtt.tables_md({})
    assert "pending" in md and "## Decision" in md
