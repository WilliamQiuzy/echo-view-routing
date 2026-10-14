"""Checks of the video dense supervision round's fixed readings and diagnostics (configs/experiments/video_dense_supervision_v1.yaml,
Task 4), written before any development output: native or recall failure, an infeasible B3 setting, a win over the
shuffled control only, a mechanism signal without a system gain, the stream-sensitivity conflict, full success, the
anchor accuracy and class-wise recall, and the change-point counts with their sources."""
from pathlib import Path

import numpy as np
import yaml

import scripts.raw_video_context_tables as rt
import scripts.video_dense_supervision_tables as vt
from scripts.target_patch_readout_tables import UNCERTAIN

ROOT = Path(__file__).resolve().parents[1]
VD = yaml.safe_load((ROOT / "configs/experiments/video_dense_supervision_v1.yaml").read_text())["video_dense_supervision"]
P, N, NEG = vt.POS, vt.NONE, vt.NEG


def v(sel: str, cal: str) -> dict:
    return {"selection": sel, "calibration": cal}


def readings(sys_ac=(N, N), sys_ao=(N, N), sys_as=(N, N), mix_ac=(N, N), mix_as=(N, N), constraints=True, goal=False) -> dict:
    sysv = {"aligned - center_only": v(*sys_ac), "aligned - Ordered": v(*sys_ao), "aligned - shuffled": v(*sys_as)}
    mixv = {"aligned - center_only": v(*mix_ac), "aligned - shuffled": v(*mix_as)}
    return vt.decide(sysv, mixv, constraints, goal, VD["decision_rules"])


def applied(d: dict) -> list[str]:
    return d["applied"]


def test_rule_names_match_the_config():
    assert set(vt.RULE_ORDER) == {k for k in VD["decision_rules"] if k != "several"}


def test_full_success_is_model_progress_and_mechanism_support():
    d = readings(sys_ac=(P, P), sys_ao=(P, P), sys_as=(P, P), mix_ac=(P, P), mix_as=(P, P), constraints=True, goal=True)
    assert applied(d) == ["model_progress", "mechanism_support", "end"]


def test_native_or_recall_failure_blocks_progress_and_mechanism():
    d = readings(sys_ac=(P, P), sys_ao=(P, P), mix_ac=(P, P), mix_as=(P, P), constraints=False, goal=False)
    assert applied(d) == ["system_only", "tradeoff", "end"] and not d["rows"]["mechanism_support"]["applies"]


def test_an_infeasible_b3_setting_is_a_constraint_failure():
    rows = [{h: {"segment_f1_50_multi": 0.99, "boundary_recall_multi": 0.99, "native_macro_f1": 0.99} for h in vt.HALVES} for _ in range(3)]
    g = VD["goal"]
    ok = rt.goal(rows, [True, True, True], g); bad = rt.goal(rows, [True, False, True], g)
    assert ok["met"] and not ok["constraint_failures"] and not bad["met"] and bad["constraint_failures"]
    d = readings(sys_ac=(P, P), sys_ao=(P, P), mix_ac=(P, P), mix_as=(P, P), constraints=not bad["constraint_failures"], goal=bad["met"])
    assert "model_progress" not in applied(d) and "mechanism_support" not in applied(d)


def test_beating_only_the_shuffled_control_is_no_practical_gain():
    d = readings(sys_as=(P, P), mix_as=(P, P))
    assert applied(d) == ["shuffled_only", "no_gain", "end"]


def test_a_mechanism_signal_without_a_system_gain_is_a_tradeoff():
    d = readings(mix_ac=(P, P), mix_as=(P, P), sys_ac=(N, N))
    assert applied(d) == ["tradeoff", "end"] and not d["rows"]["mechanism_support"]["applies"] and not d["rows"]["no_gain"]["applies"]


def test_a_stream_conflict_makes_the_mechanism_uncertain():
    d = readings(sys_ac=(P, P), mix_ac=(UNCERTAIN, P), mix_as=(P, P), constraints=True)
    assert not d["rows"]["mechanism_support"]["applies"] and d["mechanism_uncertain_halves"] == ["selection"] and "system_only" in applied(d)


def test_no_signal_anywhere_is_no_gain():
    d = readings(sys_ac=(N, NEG), mix_ac=(N, N))
    assert applied(d) == ["no_gain", "end"]
    d = readings(sys_ao=(P, N))
    assert "no_gain" not in applied(d) and applied(d) == ["end"]


def test_anchor_accuracy_and_class_recall():
    labels = np.array([[2] * 8, [2, 2, 2, 3, 3, 3, 3, 3], [0] * 8]); pred = labels.copy(); pred[1, 3] = 2; pred[2, 0] = 1
    a = vt.anchor_accuracy(pred, labels, np.array([0, 1]))
    assert a["windows"] == 2 and a["anchors"] == 16 and a["accuracy"] == 15 / 16
    assert a["recall"]["A4C"] == {"anchors": 11, "recall": 1.0} and a["recall"]["A5C"] == {"anchors": 5, "recall": 0.8} and a["recall"]["PLAX"]["recall"] is None


def test_change_points_and_both_sides_with_sources():
    labels = np.array([[2] * 8, [2, 2, 2, 3, 3, 3, 3, 3], [0, 0, 1, 1, 0, 0, 0, 0]]); pred = labels.copy(); pred[2, 2] = 0
    pts = vt.change_points(labels, np.array([0, 1, 2]))
    assert pts == [(1, 2), (2, 1), (2, 3)]
    src = {(1, 2): ({"r1"}, {"s1"}), (2, 1): ({"r2", "r3"}, {"s2"}), (2, 3): ({"r3"}, {"s2"})}
    s = vt.change_summary(pred, labels, pts, lambda i, j: src[(i, j)])
    assert s["all"] == {"changes": 3, "recordings": 3, "sessions": 2, "few_units": True}
    assert s["both_right"]["changes"] == 2 and s["both_right_share"] == 2 / 3          # (1, 2) and (2, 3); (2, 1) misses anchor 2
    assert vt.change_summary(pred, labels, [], lambda i, j: (set(), set()))["both_right_share"] is None


def test_anchor_labels_take_the_configured_slots():
    labels16 = np.arange(32).reshape(2, 16) % 5
    assert np.array_equal(vt.anchor_labels(labels16, VD["model"]["anchors"]), labels16[:, ::2])
