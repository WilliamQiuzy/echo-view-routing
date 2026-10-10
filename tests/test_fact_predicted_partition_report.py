"""Checks of the predicted-partition report on synthetic records: stage tracking (earliest unmatched readout, recovery,
runs lost at i or u, minority runs of the partition), interval and head-cut counts, the head + frame-classifier
control, the fixed decision rows, and the queue's budget projection."""
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.fact_predicted_partition_tables as pt
from echo_routing.temporal.dev_data import Seq

ROOT = Path(__file__).resolve().parents[1]
FP = yaml.safe_load((ROOT / "configs/experiments/fact_predicted_partition_v1.yaml").read_text())["fact_predicted_partition"]


def record(ref, stages, partition, frags, native=False, cuts=None, sid="q"):
    return {"id": sid, "native": native, "ref": list(ref), "partition": partition, "frags": frags, "head_cuts": cuts,
            "stages": {k: list(v) for k, v in stages.items()}}


def test_stage_tracking_counts_the_earliest_loss_and_recovery():
    ref = [0] * 10 + [2] * 4 + [0] * 10                                    # an X -> Y -> X middle (short)
    good, bad = ref, [0] * 24
    r = record(ref, {"E1": good, "i": bad, "u": bad, "U": good, "final": bad}, [[0, 24]], [["p1_1", 0, 12], ["p2_1", 12, 24]])
    us = pt.stage_units([r], 5, 16)
    mid = next(u for u in us if "xyx" in u["groups"])
    assert mid["m"] == {"E1": True, "i": False, "u": False, "U": True, "final": False} and mid["own_share"] == pytest.approx(4 / 24)
    assert pt.stage_table(us)["stream:short"]["segments"] == 3                 # all three runs have at most 16 samples
    t = pt.stage_table(us)["stream:xyx"]                                        # the middle alone
    assert t["matched"] == {"E1": 1, "i": 0, "u": 0, "U": 1, "final": 0} and t["first_unmatched"]["i"]["segments"] == 1
    assert t["transitions"]["u -> U"]["recovered"]["segments"] == 1 and t["transitions"]["U -> final"]["lost"]["segments"] == 1
    assert t["minority_in_every_interval"]["segments"] == 1 and t["minority_in_every_interval"]["matched_by_the_official_output"] == 0
    lost, missed = pt.lost_at_i_u(us)
    assert missed == 3 and lost == 3                                       # all three runs are first lost at the i block
    r2 = record(ref, {n: ref for n in pt.CHAIN} | {"i": bad}, [[0, 10], [10, 14], [14, 24]], [["p1_1", 0, 24]])
    t2 = pt.stage_table(pt.stage_units([r2], 5, 16))["stream:all"]
    assert t2["first_unmatched"]["i"] == {"segments": 0, "recordings": 0, "sessions": 0, "matched_again_by_the_official_output": 3}


def test_interval_and_head_cut_counts():
    recs = [record([0] * 10 + [1] * 10, {}, [[0, 8], [8, 10], [10, 20]], [], cuts=[10, 15], sid="s"), record([3] * 16, {}, [[0, 8], [8, 16]], [], native=True, cuts=[4], sid="n")]
    c = pt.interval_counts(recs)
    assert c["stream"]["intervals_mean"] == 3 and c["native"]["intervals_per_100_samples"] == pytest.approx(100 * 2 / 16)
    h = pt.head_cut_stats(recs, {"s": [5, 10]}, 5)                         # a same-view join at 5, a view change at 10
    assert h["stream_cut_precision"] == 0.5 and h["stream_cut_recall"] == 1.0 and h["same_view_join_cut_rate"] == 1.0
    assert h["cuts_per_native_clip"] == 1.0 and h["native_clips_with_a_cut"] == 1


def test_head_frame_classifier_names_each_head_interval():
    from scripts.report_fact_predicted_partition import head_frame_classifier
    prob = np.full((12, 5), 0.01, np.float32); prob[:6, 2] = 0.9; prob[6:, 3] = 0.9; prob[3, 3] = 0.95   # one stray frame
    s = Seq("q", False, np.zeros((12, 4), np.float16), np.array([2] * 6 + [3] * 6), np.zeros(0, int), prob)
    out = head_frame_classifier(lambda x: [6], [s], "cpu")
    assert out["q"]["labels"].tolist() == [2] * 6 + [3] * 6 and out["q"]["head_cuts"] == [6]   # the stray frame is averaged away
    assert head_frame_classifier(lambda x: [], [s], "cpu")["q"]["labels"].tolist() == [3] * 12        # one interval: one mean, one view


def verdict(pos):
    return {h: ("consistent positive" if p else "no consistent difference detected") for h, p in zip(("selection", "calibration"), pos)}


def rows(value, recall=0.96, native=0.99):
    return {f"{g} s{s}": {h: {"segment_f1_50_multi": value, "boundary_recall_multi": recall, "native_macro_f1": native} for h in ("selection", "calibration")}
            for g in ("Uniform", "Predicted") for s in (0, 1, 2)}


def iu(new, orig, missed=10):
    return {g: {s: {h: ((orig if g == "Original FACT" else new), missed) for h in ("selection", "calibration")} for s in (0, 1, 2)}
            for g in ("Original FACT", "Uniform", "Predicted")}


def test_decision_rows_follow_the_fixed_readings():
    names = ["Predicted - Uniform", "Predicted - Original FACT", "Uniform - Original FACT", "Predicted - (Boundary head + frame classifier)"]
    pos = {s: {"selection": 0.01, "calibration": 0.02} for s in (0, 1, 2)}
    d = pt.decide(FP, {n: verdict((n == "Uniform - Original FACT",) * 2) for n in names}, rows(0.95), iu(7, 8), pos)
    assert d["rows_that_apply"][0] == FP["decision"][0] and FP["decision"][3] in d["rows_that_apply"]        # uniform helps; primary uncertain
    d = pt.decide(FP, {n: verdict((True, True)) for n in names}, rows(0.99), iu(3, 8), pos)
    assert d["rows_that_apply"] == [FP["decision"][1]] and d["groups"]["Predicted"]["goal_met"] and d["position_control_stable_direction"]
    mixed = {0: {"selection": 0.01, "calibration": -0.01}, 1: pos[1], 2: pos[2]}
    d = pt.decide(FP, {n: verdict((True, True)) for n in names}, rows(0.99, recall=0.9), iu(9, 8), mixed)
    assert not d["position_control_stable_direction"] and FP["decision"][1] not in d["rows_that_apply"] and d["groups"]["Uniform"]["constraint_failures"]
    d = pt.decide(FP, {n: verdict((False, False)) for n in names}, rows(0.9), iu(9, 8), pos)
    assert FP["decision"][2] in d["rows_that_apply"] and not d["groups"]["Predicted"]["full_system_gain"] and not d["groups"]["Uniform"]["goal_met"]


def test_queue_projection_counts_everything_before_and_after():
    from scripts.queue_fact_predicted_partition import projection
    order = ["Uniform_s0", "Predicted_s0", "Uniform_s1", "Predicted_s1"]
    assert projection(order, {}, {"Uniform": 3.0, "Predicted": 3.5}, 1.0, 1.5) == pytest.approx(1.0 + 13.0 + 1.5)
    assert projection(order, {"Uniform_s0": 2.0}, {"Uniform": 3.0, "Predicted": 3.5}, 1.0, 1.5) == pytest.approx(1.0 + 2.0 + 2.0 + 7.0 + 1.5)
