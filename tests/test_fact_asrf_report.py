"""Checks of the paired FACT-ASRF report on synthetic data: every row of the fixed decision table, the constraints, the
goal and sensitivity margins, short-segment and X -> Y -> X recall with the reasons of the misses, and the token runs."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import scripts.report_fact_asrf as rp

ROOT = Path(__file__).resolve().parents[1]
FA = yaml.safe_load((ROOT / "configs/experiments/fact_asrf_paired_v1.yaml").read_text())["fact_asrf_paired"]
POS, NONE = "consistent positive", "no consistent difference detected"


def rows(seg: float, rec: float = 0.97, nat: dict | None = None) -> list[dict]:
    nat = nat or {"selection": 0.98, "calibration": 0.99}
    return [{h: {"segment_f1_50_multi": seg, "boundary_recall_multi": rec, "native_macro_f1": nat[h]} for h in ("selection", "calibration")} for _ in range(3)]


def both(v: str) -> dict:
    return {"selection": v, "calibration": v}


def test_decision_rows_in_order():
    d = FA["decision"]
    assert rp.decide({"selection": POS, "calibration": NONE}, both(POS), rows(0.99), FA)["action"] == d[0]["action"]
    assert rp.decide(both(POS), {"selection": POS, "calibration": NONE}, rows(0.99), FA)["action"] == d[1]["action"]
    failed = rp.decide(both(POS), both(POS), rows(0.99, rec=0.94), FA)
    assert failed["action"] == d[2]["action"] and len(failed["constraint_failures"]) == 6
    assert rp.decide(both(POS), both(POS), rows(0.95), FA)["action"] == d[3]["action"]
    top = rp.decide(both(POS), both(POS), rows(0.99), FA)
    assert top["action"] == d[4]["action"] and top["goal_met"]
    assert top["calibration_sensitivity_margin"] == pytest.approx(0.99 - 0.9838935784747763)


def test_constraint_floors_differ_by_half():
    r = {"boundary_recall_multi": 0.96, "native_macro_f1": 0.975}
    assert rp.constraints_met(r, "selection", FA["constraints"]) and not rp.constraints_met(r, "calibration", FA["constraints"])


def stream(labels: list[int], frag_lens: list[int], sid: str = "r1"):
    vids = [f"2022-01-01_00-00-0{i}_320x240_{i}" for i in range(len(frag_lens))]
    frags = [SimpleNamespace(video_id=v, start=0, end=n) for v, n in zip(vids, frag_lens)]
    return SimpleNamespace(labels=np.asarray(labels), recipe=SimpleNamespace(recipe_id=sid, fragments=frags))


def test_short_segment_recall_and_miss_reasons():
    A4C, A5C, PSAX = 2, 3, 1
    truth = [A4C] * 20 + [A5C] * 8 + [A4C] * 20; ev = SimpleNamespace(multi=[stream(truth, [20, 8, 20])])
    exact = rp.mechanism(ev, {"r1": np.asarray(truth)}, 16)
    assert exact["short segments"]["recall"] == 1.0 and exact["A4C/A5C X-Y-X middles"]["recall"] == 1.0
    merged = rp.mechanism(ev, {"r1": np.asarray([A4C] * 48)}, 16)["short A4C or A5C segments"]
    assert merged["matched"] == 0 and merged["no segment at IoU 0.5"] == 1 and merged["missed_recordings"] == 1 and merged["missed_sessions"] == 1
    wrong = rp.mechanism(ev, {"r1": np.asarray([A4C] * 20 + [PSAX] * 8 + [A4C] * 20)}, 16)
    assert wrong["X-Y-X middles"]["wrong view"] == 1 and wrong["A4C/A5C X-Y-X middles"]["wrong view"] == 1


def test_xyx_middle_needs_equal_outer_runs():
    seg = lambda labels: [SimpleNamespace(label=x) for x in labels]   # noqa: E731
    assert rp.xyx_middle()(seg([2, 3, 2]), 1) and not rp.xyx_middle()(seg([2, 3, 1]), 1) and not rp.xyx_middle()(seg([2, 3, 2]), 0)
    assert rp.xyx_middle((2, 3))(seg([2, 3, 2]), 1) and not rp.xyx_middle((2, 3))(seg([1, 0, 1]), 1)


def test_token_runs_flag_a_token_on_separate_intervals():
    assert rp.token_runs(np.array([3, 3, 1, 1, 3]))["separate_intervals"]
    assert not rp.token_runs(np.array([3, 3, 1, 1]))["separate_intervals"]
    assert rp.token_runs(None) == {"runs": 0, "tokens": 0, "separate_intervals": False}
    ev = SimpleNamespace(multi=[stream([2] * 5 + [3] * 5, [5, 5])])
    t = rp.token_table(ev, {"r1": {"tokens_used": 3, "separate_intervals": True}, "n1": {"tokens_used": 1, "separate_intervals": False}})
    assert (t["more_tokens"], t["outputs_with_a_token_on_separate_intervals"], t["sequences"]) == (1, 1, 2)


def test_tables_render_at_every_stage():
    first = rp.tables_md({"runs": {}, "frozen": {}, "references": "pending", "decision": "pending"})
    assert "## Decision" in first and "pending" in first
    row = {"segment_f1_50_multi": 0.9, "boundary_recall_multi": 0.95, "native_macro_f1": 0.97}
    partial = rp.tables_md({"systems": {"G1-like": {h: {"row": row} for h in ("selection", "calibration")}}, "comparisons": {}, "decision": "not computed"})
    assert "G1-like" in partial and "not computed" in partial
