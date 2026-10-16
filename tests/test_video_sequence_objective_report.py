"""A07 report side (configs/experiments/video_sequence_objective_v1.yaml, task sheet sections 3-4 of the evaluation): the
fixed readings on synthetic verdict tables (full success, objective signal, balancing only, continuation only, trade-off,
consistent losses, no qualifying gain, end), the comparison map, the half-bounded reader and access guard, the eager
loaders forbidden, the choice record rules, the per-sample mapping and the paths of the barrier. CPU only; no data."""
from __future__ import annotations

import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import scripts.report_raw_video_context as rr
import scripts.report_video_sequence_objective as rp

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/experiments/video_sequence_objective_v1.yaml").read_text())
VSO = CFG["video_sequence_objective"]
POS, NEG, NONE = rp.POS, rp.NEG, rp.NONE
CMPS = ("O - B", "B - F", "O - F", "O - frozen Ordered", "B - frozen Ordered", "F - frozen Ordered")


def verdicts() -> dict:
    return {c: {"selection": NONE, "calibration": NONE} for c in CMPS}


def goal(met: bool = True, failures: list | None = None) -> dict:
    return {"met": met, "constraint_failures": failures or [], "mean_segment_f1": {"selection": 0.95, "calibration": 0.985}}


def both(x):
    return {"selection": x, "calibration": x}


def test_full_success_needs_three_two_half_comparisons_constraints_and_goal():
    v = verdicts(); v["O - B"] = v["O - F"] = v["O - frozen Ordered"] = both(POS)
    r = rp.decide(v, goal(), [])
    assert r["applied"][0] == "full_success" and "end" in r["applied"] and "no_qualifying_gain" not in r["applied"]
    assert "full_success" not in rp.decide(v, goal(met=False), [])["applied"]
    assert "full_success" not in rp.decide(v, goal(failures=[{"seed": 0, "half": "selection"}]), [])["applied"]


def test_objective_signal_without_full_success():
    v = verdicts(); v["O - B"] = {"selection": POS, "calibration": NONE}; v["O - F"] = {"selection": POS, "calibration": NONE}
    r = rp.decide(v, goal(), [])
    assert "objective_signal" in r["applied"] and "full_success" not in r["applied"] and "no_qualifying_gain" not in r["applied"]
    assert r["notes"]["objective_signal"]["o_minus_frozen_ordered_positive_anywhere"] is False
    v["O - F"] = {"selection": POS, "calibration": NEG}
    assert "objective_signal" not in rp.decide(v, goal(), [])["applied"]          # consistent negative in the other half
    v["O - F"] = {"selection": NONE, "calibration": POS}
    assert "objective_signal" not in rp.decide(v, goal(), [])["applied"]          # O - B and O - F positive in different halves


def test_balancing_only_and_continuation_only():
    v = verdicts(); v["B - F"] = {"selection": POS, "calibration": NONE}
    r = rp.decide(v, goal(), [])
    assert "balancing_only" in r["applied"] and "no_qualifying_gain" not in r["applied"]     # the config: none of the four gain readings
    v["O - B"] = both(POS)
    assert "balancing_only" not in rp.decide(v, goal(), [])["applied"]
    v = verdicts(); v["F - frozen Ordered"] = {"selection": NONE, "calibration": POS}
    assert "continuation_only" in rp.decide(v, goal(), [])["applied"]
    v["O - F"] = {"selection": POS, "calibration": NONE}
    assert "continuation_only" not in rp.decide(v, goal(), [])["applied"]


def test_trade_off_and_consistent_losses_are_never_hidden():
    v = verdicts(); v["O - frozen Ordered"] = {"selection": NONE, "calibration": NEG}; v["F - frozen Ordered"] = {"selection": NEG, "calibration": NONE}
    cells = [{"comparison": "O - B", "half": "selection", "recovered": 4, "reason": "more multi FP segments"}]
    r = rp.decide(v, goal(), cells)
    assert {"trade_off", "consistent_losses", "no_qualifying_gain", "end"} <= set(r["applied"])
    assert {(x["comparison"], x["half"]) for x in r["notes"]["consistent_losses"]} == {("O - frozen Ordered", "calibration"), ("F - frozen Ordered", "selection")}
    assert r["notes"]["trade_off"]["cells"] == cells
    assert r["applied"] == [x for x in rp.READING_ORDER if x in r["applied"]]


def test_no_qualifying_gain_reports_the_primary_direction():
    r = rp.decide(verdicts(), goal(), [])
    assert r["applied"] == ["no_qualifying_gain", "end"] and r["notes"]["no_qualifying_gain"]["primary"] == {"selection": NONE, "calibration": NONE}


def test_qualifies_rule():
    assert rp.qualifies({"selection": POS, "calibration": NONE}) and rp.qualifies(both(POS))
    assert not rp.qualifies({"selection": POS, "calibration": NEG}) and not rp.qualifies(both(NONE)) and not rp.qualifies(both(NEG))


def test_the_comparison_map_follows_the_config():
    ev = VSO["evaluation"]["comparisons"]
    assert rp.PRIMARY == "O - B" == ev["primary"] and set(ev["mandatory"]) | set(ev["also"]) | {ev["primary"]} == set(rp.COMPARISONS)
    assert set(rp.DECODINGS) == {"b3", "argmax", "ordered_setting"}
    assert all(rp.CMP[d][c] == (rp.system_name(c.split(" - ")[0], d), rp.system_name(c.split(" - ")[1], d)) for d in rp.DECODINGS for c in rp.COMPARISONS)


def test_the_half_reader_needs_access_and_reads_one_half(monkeypatch):
    access = rp.DevAccess({"calibration"})
    access.require("calibration")
    with pytest.raises(SystemExit):
        access.require("selection")
    seqs = [{"id": "a", "refs": [("v1", 3, 1.0), ("v1", 6, 1.0)], "labels": np.array([0, 0])}, {"id": "b", "refs": [("v2", 0, 0.8), ("v1", 3, 1.0)], "labels": np.array([1, 1])}]
    refs = rp.half_refs(seqs)
    assert refs == sorted({("v1", 3, 1.0), ("v1", 6, 1.0), ("v2", 0, 0.8)})


def test_eager_development_loaders_are_forbidden_during_the_report():
    import scripts.report_center_supervision as rcs
    undo = rp.forbid_eager_loaders()
    try:
        with pytest.raises(RuntimeError):
            rcs.validation(None, None, None)
    finally:
        undo()
    assert rcs.validation.__name__ == "validation"


def test_per_sample_mapping_is_v01s():
    W = {"seq": np.array([0, 0, 0]), "centre": np.array([2, 7, 11])}; lg = np.array([[2.0, 0, 0, 0, 0], [0, 3.0, 0, 0, 0], [0, 0, 0, 0, 4.0]])
    p = rr.per_sample(W, lg, [{"id": "x", "labels": np.zeros(13)}])["x"]
    assert p.shape == (13, 5) and p[:5].argmax(1).tolist() == [0] * 5 and p[5:10].argmax(1).tolist() == [1] * 5 and p[10:].argmax(1).tolist() == [4] * 3


def test_choice_records_are_complete_and_follow_the_rule():
    grid = []
    for h in VSO["evaluation"]["b3"]["grid"]["half_window"]:
        for q in VSO["evaluation"]["b3"]["grid"]["quantile"]:
            grid.append({"order": len(grid), "params": {"half_window": h, "threshold_quantile": q, "threshold": 0.3, "nms_radius": 5}, "segment_f1_50_multi": 0.9 + h / 1000 + q / 100,
                         "native_macro_f1": 0.99, "boundary_recall_multi": 0.96})
    best = max(grid, key=lambda r: r["segment_f1_50_multi"])
    rec = {"status": "feasible", "params": best["params"], "n_feasible": len(grid), "grid": grid}
    keys = [f"{a}_s{s}" for a in rp.ARMS for s in (0, 1, 2)]
    choices = {"systems": {k: copy.deepcopy(rec) for k in keys}, "choices": {k: {x: rec[x] for x in ("status", "params", "n_feasible")} for k in keys}}
    ctx = SimpleNamespace(seeds=[0, 1, 2], vso=VSO)
    assert rp.load_choices(ctx, choices) is choices
    bad = copy.deepcopy(choices); bad["systems"]["order_s1"]["params"] = grid[0]["params"]; bad["choices"]["order_s1"]["params"] = grid[0]["params"]
    with pytest.raises(SystemExit):
        rp.load_choices(ctx, bad)
    bad = copy.deepcopy(choices); bad["systems"].pop("frame_s2")
    with pytest.raises(SystemExit):
        rp.load_choices(ctx, bad)


def test_the_barrier_paths_and_stages_follow_the_config():
    assert VSO["outputs"]["choices"].startswith(VSO["outputs"]["results_dir"]) and VSO["outputs"]["choices_proof"].startswith(VSO["outputs"]["root"])
    src = inspect.getsource(rp)
    assert "barrier(" in src and "open_selection(" in src and "verify_proof" in src and "DevAccess({\"calibration\"})" in src
    assert set(rp.STAGES) == {"calibrate", "dry_run", "selection"}


def test_the_report_never_reads_test_or_pediatric_data():
    src = inspect.getsource(rp).lower()
    assert not any(b in src for b in ("pediatric", "'test'", "split_test", "rcs.validation(", "dev_run.inputs("))
