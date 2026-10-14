"""Checks of the trained token readout's fixed readings and summaries (configs/experiments/trained_token_readout_v1.yaml,
Task 2), written before any new token metric exists: the same 40-setting budget for the CLS and token systems, no
feasible setting, a win over the unsupervised or shuffled controls only, a classification gain without a segment gain,
one half only, native or recall failures, the stream-sensitivity conflict, full success, and few changed cases that must
carry their source counts and a flag."""
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.raw_video_context_tables as rt
import scripts.trained_token_readout_tables as tb
from scripts.target_patch_readout_tables import UNCERTAIN

ROOT = Path(__file__).resolve().parents[1]
TT = yaml.safe_load((ROOT / "configs/experiments/trained_token_readout_v1.yaml").read_text())["trained_token_readout"]
P, N, NEG = tb.POS, tb.NONE, tb.NEG


def v(sel: str, cal: str) -> dict:
    return {"selection": sel, "calibration": cal}


def read(cls=(N, N), ordered=(N, N), cls_only=(N, N), shuffled=(N, N), mixed=(N, N), constraints=True, goal=False) -> dict:
    sysv = {"token - CLS": v(*cls), "token - Ordered": v(*ordered), "token - CLS-only token": v(*cls_only), "token - Shuffled token": v(*shuffled)}
    return tb.decide(sysv, v(*mixed), constraints, goal, TT["decision_rules"])


def test_rule_names_match_the_config():
    assert set(tb.RULE_ORDER) == {k for k in TT["decision_rules"] if k != "several"}


def test_full_success():
    d = read(cls=(P, P), ordered=(P, P), cls_only=(P, P), shuffled=(P, P), mixed=(P, P), constraints=True, goal=True)
    assert d["applied"] == ["complete_progress", "useful_readout", "end"] and not d["notes"]["only_controls"]


def test_useful_readout_without_the_goal_or_the_strong_reference():
    d = read(cls=(P, P), mixed=(P, P), constraints=True, goal=False)
    assert d["applied"] == ["useful_readout", "end"]


def test_no_feasible_setting_or_a_native_or_recall_failure_blocks_every_gain_reading():
    d = read(cls=(P, P), ordered=(P, P), cls_only=(P, P), shuffled=(P, P), mixed=(P, P), constraints=False, goal=False)
    assert d["applied"] == ["labels_learned_no_usable_gain", "end"] and d["notes"]["constraints_failed"]
    rows = [{h: {"segment_f1_50_multi": 0.99, "boundary_recall_multi": 0.99, "native_macro_f1": 0.99} for h in tb.HALVES} for _ in range(3)]
    g = rt.goal(rows, [True, False, True], TT["goal"])
    assert not g["met"] and g["constraint_failures"]


def test_beating_only_the_controls_is_no_progress():
    d = read(cls_only=(P, P), shuffled=(P, P))
    assert d["applied"] == ["labels_learned_no_usable_gain", "end"] and d["notes"]["only_controls"]


def test_a_classification_gain_without_a_segment_gain():
    d = read(mixed=(P, P), cls=(N, N))
    assert d["applied"] == ["labels_learned_no_usable_gain", "end"] and d["notes"]["classification_without_segments"]


def test_one_half_only_is_reported_as_such():
    d = read(cls=(P, N), mixed=(P, N))
    assert d["applied"] == ["labels_learned_no_usable_gain", "end"] and d["notes"]["one_half_only"] == ["selection"]


def test_a_stream_conflict_makes_that_half_uncertain():
    d = read(cls=(P, P), mixed=(UNCERTAIN, P), constraints=True)
    assert d["applied"] == ["labels_learned_no_usable_gain", "end"] and d["notes"]["uncertain_halves"] == ["selection"]


def test_negative_results_are_named():
    d = read(cls=(NEG, NEG), mixed=(NEG, N))
    assert d["applied"] == ["labels_learned_no_usable_gain", "end"] and d["notes"]["negative"] == {"segment F1 vs CLS": ["selection", "calibration"], "classification vs CLS": ["selection"]}


def test_choice_records_use_the_same_forty_setting_budget():
    grid = [{"order": k, "params": {"half_window": h, "threshold_quantile": q, "nms_radius": 5}} for k, (h, q) in
            enumerate((h, q) for h in (5, 7, 9, 11, 15) for q in (0.80, 0.85, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999))]
    tb.check_choice_record({"status": "feasible", "params": grid[3]["params"], "grid": grid}, TT["b3"]["grid"])
    with pytest.raises(SystemExit):
        tb.check_choice_record({"status": "feasible", "params": grid[3]["params"], "grid": grid[:39]}, TT["b3"]["grid"])
    with pytest.raises(SystemExit):
        tb.check_choice_record({"status": "feasible", "params": {"half_window": 13, "threshold_quantile": 0.9, "nms_radius": 5}, "grid": grid}, TT["b3"]["grid"])
    tb.check_choice_record({"status": "infeasible (diagnostic setting shown)", "params": grid[20]["params"], "grid": grid}, TT["b3"]["grid"])


def test_changed_cases_carry_sources_and_a_few_unit_flag():
    a = np.array([1, 1, 0, 1, 0, 1], bool); b = np.array([0, 1, 1, 1, 0, 0], bool)
    rec = np.array(["r1", "r1", "r2", "r3", "r4", "r5"]); ses = np.array(["s1", "s1", "s2", "s3", "s3", "s4"])
    c = tb.changes(a, b, rec, ses)
    assert c["fixed"] == {"windows": 2, "recordings": 2, "sessions": 2, "few_units": True}
    assert c["new"] == {"windows": 1, "recordings": 1, "sessions": 1, "few_units": True}


def test_anchor_offset_effects_are_split_by_exactness():
    off = np.array([0, 0, -1, -1, 0]); a = np.array([1, 0, 1, 1, 1], bool); b = np.array([1, 1, 0, 1, 1], bool)
    s = tb.offset_split(off, a, b)
    assert s["exact"] == {"windows": 3, "accuracy_token": 2 / 3, "accuracy_cls": 1.0} and s["non_exact"] == {"windows": 2, "accuracy_token": 1.0, "accuracy_cls": 0.5}
