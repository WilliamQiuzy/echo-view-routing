"""Pure helpers of scripts/report_l1_asrf.py on synthetic data: fragment transfer, the fault classes of true view
changes, the goal and the visual-against-system counts."""
import numpy as np
import pandas as pd
import pytest

import scripts.report_l1_asrf as rp
from echo_routing.evaluate.failure_units import Unit


def _onehot(labels, p=0.9):
    out = np.full((len(labels), 5), (1 - p) / 4)
    out[np.arange(len(labels)), labels] = p
    return out


def test_majority_and_mean_rules_can_disagree():
    prob = np.array([[0.4, 0.6, 0, 0, 0], [0.4, 0.6, 0, 0, 0], [1.0, 0, 0, 0, 0]])
    assert rp.classify_rule(prob, "majority") == 1 and rp.classify_rule(prob, "mean") == 0
    with pytest.raises(ValueError):
        rp.classify_rule(prob, "median")


def test_transfer_categories():
    assert rp.transfer_category(True, True) == "recording right, fragment right"
    assert rp.transfer_category(True, False) == "recording right, fragment wrong"
    assert rp.transfer_category(False, True) == "recording wrong, fragment right"
    assert rp.transfer_category(False, False) == "recording wrong, fragment wrong"


def test_fragment_transfer_uses_the_same_variant_and_rule_for_fragment_and_recording():
    rec = {("v1", "orig"): _onehot([3] * 10 + [2] * 4), ("v1", "gamma090"): _onehot([2] * 14), ("v2", "orig"): _onehot([3] * 6)}
    units = {"s": [Unit("s", 0, 0, 9, 3, None, 2, (("v1", "orig", 10, 14), ("v2", "orig", 0, 5))),
                   Unit("s", 1, 9, 13, 2, 3, None, (("v1", "gamma090", 0, 4),))]}
    rows = rp.fragment_transfer(units, lambda v, var: rec[(v, var)])
    assert len(rows) == 3 and [r["unit_fragments"] for r in rows] == [2, 2, 1]
    a, b, c = rows
    assert a["majority_category"] == "recording right, fragment wrong"        # the clip reads A5C, its last 4 samples A4C
    assert b["majority_category"] == "recording right, fragment right"
    assert c["majority_category"] == "recording right, fragment right" and c["variant"] == "gamma090"


def test_transfer_table_counts_recordings_sessions_and_system_outcomes():
    rows = [{"stream": "s", "unit": 0, "video": "p1_1", "majority_category": "recording right, fragment wrong", "mean_category": "recording right, fragment right"},
            {"stream": "s", "unit": 1, "video": "p1_1", "majority_category": "recording right, fragment wrong", "mean_category": "recording right, fragment wrong"},
            {"stream": "t", "unit": 0, "video": "p1_2", "majority_category": "recording right, fragment right", "mean_category": "recording right, fragment right"},
            {"stream": "t", "unit": 0, "video": "p2_1", "majority_category": "recording right, fragment right", "mean_category": "recording right, fragment right"}]
    units = [{"stream": "s", "unit": 0, "cause": "label"}, {"stream": "s", "unit": 1, "cause": "matched"}, {"stream": "t", "unit": 0, "cause": "missed_cut"}]
    out = rp.transfer_table(rows, units)
    wrong = out["majority"]["recording right, fragment wrong"]
    assert (wrong["fragments"], wrong["recordings"], wrong["sessions"]) == (2, 1, 1)          # one recording repeated in two fragments
    assert wrong["by_outcome"] == {"label": {"fragments": 1, "recordings": 1, "sessions": 1}, "matched": {"fragments": 1, "recordings": 1, "sessions": 1}}
    right = out["majority"]["recording right, fragment right"]
    assert (right["fragments"], right["recordings"], right["sessions"]) == (2, 2, 2)
    assert right["by_outcome"] == {"missed_cut": {"fragments": 2, "recordings": 2, "sessions": 2}}   # both right, the system misses the unit
    assert out["mean"]["recording right, fragment right"]["fragments"] == 3


def test_change_faults():
    labels = np.array([0] * 10 + [2] * 10 + [3] * 10 + [3] * 10)
    joins = np.array([10, 20, 30]); semantic = np.array([10, 20])
    final = labels.copy()
    rows, extra = rp.classify_changes(final, [10, 20], labels, joins, semantic)
    assert [r["cat"] for r in rows] == ["found, both views right", "found, both views right"] and extra == {"extra_inside_clip": 0, "extra_at_same_view_join": 0}
    merged = np.array([0] * 10 + [2] * 30)                                   # candidate at 20, both sides read 2
    rows, _ = rp.classify_changes(merged, [10, 20], labels, joins, semantic)
    assert [r["cat"] for r in rows] == ["found, both views right", "merged"]  # the change at 10 has both of its fragments right
    shifted_view = np.array([0] * 10 + [1] * 10 + [3] * 20)                  # found at 10 and 20, but the middle fragment is wrong
    rows, _ = rp.classify_changes(shifted_view, [10, 20], labels, joins, semantic)
    assert [r["cat"] for r in rows] == ["found, a view wrong", "found, a view wrong"]
    displaced = np.array([0] * 10 + [2] * 18 + [3] * 12)                     # the change at 20 is predicted at 28
    rows, _ = rp.classify_changes(displaced, [10, 28], labels, joins, semantic)
    assert rows[1]["cat"] == "displaced"
    none = np.array([0] * 10 + [2] * 30)
    rows, _ = rp.classify_changes(none, [10], labels, joins, semantic)
    assert rows[1]["cat"] == "no candidate"
    extra_cuts = labels.copy(); extra_cuts[3:6] = 1; extra_cuts[30:33] = 4      # cuts inside a clip and at the same-view join
    _, extra = rp.classify_changes(extra_cuts, [], labels, joins, semantic)
    assert extra["extra_inside_clip"] >= 1 and extra["extra_at_same_view_join"] >= 1


def test_goal_needs_feasible_checkpoints_and_floors():
    def row(sel, cal, status="feasible", rs=0.96, nc=0.99):
        return {"chosen": {"status": status}, "selection": {"segment_f1_50_multi": sel, "boundary_recall_multi": rs, "native_macro_f1": 0.98},
                "calibration": {"segment_f1_50_multi": cal, "boundary_recall_multi": 0.97, "native_macro_f1": nc}}
    la = {"goal": {"selection_min": 0.944409687184662, "calibration_min": 0.9821421709894332, "calibration_sensitivity": 0.9738935784747763}}
    assert rp.goal(la, [row(0.95, 0.99)] * 3)["met"]
    assert not rp.goal(la, [row(0.95, 0.99), row(0.95, 0.99), row(0.95, 0.99, status="infeasible (final epoch shown as diagnostic)")])["met"]
    assert not rp.goal(la, [row(0.95, 0.99), row(0.95, 0.99, rs=0.94), row(0.95, 0.99)])["met"]
    assert not rp.goal(la, [row(0.95, 0.99), row(0.95, 0.99), row(0.95, 0.99, nc=0.97)])["met"]
    assert rp.goal(la, [row(0.95, 0.99)] * 3)["sensitivity_calibration"] == pytest.approx(0.99 - 0.9838935784747763)


def test_visual_against_system_counts():
    native = pd.DataFrame({"video_id": ["a_1", "b_1", "c_1"], "label": [3, 3, 2], "majority": [2, 3, 2]})
    visual = {"a_1": True, "b_1": False, "c_1": True}
    units = [{"visual_majority": "all fragments right", "matched": False, "cause": "label", "sources": ["a_1"]},
             {"visual_majority": "some fragment wrong", "matched": True, "cause": "matched", "sources": ["b_1", "c_1"]},
             {"visual_majority": "all fragments right", "matched": True, "cause": "matched", "sources": ["c_1"]}]
    out = rp.visual_vs_system(visual, native, units)
    assert out["native_visual_right_system_wrong"] == {"clips": 1, "sessions": 1}
    assert out["native_visual_wrong_system_right"] == {"clips": 1, "sessions": 1}
    assert out["units_visual_right_system_unmatched"] == {"units": 1, "causes": {"label": 1}, "source_recordings": 1}
    assert out["units_visual_wrong_system_matched"] == {"units": 1, "source_recordings": 2}


def test_counts_by():
    rows = [{"c": "x", "g": 1}, {"c": "y", "g": 1}, {"c": "x", "g": 2}]
    assert rp.counts_by(rows, "c") == {"x": 2, "y": 1} and rp.counts_by(rows, "c", "g") == {"1": {"x": 1, "y": 1}, "2": {"x": 1}}
