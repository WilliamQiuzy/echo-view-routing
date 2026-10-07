"""Pure helpers of scripts/probe_single_frame_controls.py (no model, no data)."""
import numpy as np
import pandas as pd
import pytest

from scripts.probe_single_frame_controls import analyse, block_key, changes, reproduces, target_refs


def _win(target: str, other: str, frame: int = 30) -> list:
    return [(other, i, 1.0) for i in range(8)] + [(target, frame, 1.0)] + [(other, 9 + i, 1.0) for i in range(7)]


def _meta() -> pd.DataFrame:
    rows = []
    for t, s, lab in (("a", "sa", 2), ("b", "sb", 3)):
        rows.append({"id": f"{t}|own", "target": t, "session": s, "label": lab, "m": np.nan, "condition": "own", "context_label": lab})
        rows.append({"id": f"{t}|same|4", "target": t, "session": s, "label": lab, "m": 4.0, "condition": "same view", "context_label": lab})
        for j in range(2):
            rows.append({"id": f"{t}|other|{j}|4", "target": t, "session": s, "label": lab, "m": 4.0, "condition": "other view", "context_label": 0})
    return pd.DataFrame(rows)


def test_target_refs_takes_position_8_and_checks_agreement():
    meta = _meta(); spec = {k: _win(t, "ctx") for k, t in zip(meta["id"], meta["target"])}
    assert target_refs(spec, meta) == {"a": ("a", 30, 1.0), "b": ("b", 30, 1.0)}
    spec["a|same|4"] = _win("a", "ctx", frame=31)
    with pytest.raises(ValueError, match="disagree"):
        target_refs(spec, meta)


def test_target_refs_rejects_a_foreign_target_frame_and_short_windows():
    meta = _meta(); spec = {k: _win(t, "ctx") for k, t in zip(meta["id"], meta["target"])}
    spec["b|own"] = _win("zz", "ctx")
    with pytest.raises(ValueError, match="not from the target"):
        target_refs(spec, meta)
    spec = {k: _win(t, "ctx")[:15] for k, t in zip(meta["id"], meta["target"])}
    with pytest.raises(ValueError, match="15 references"):
        target_refs(spec, meta)


def test_changes_counts_windows_recordings_sessions_and_context_naming():
    df = pd.DataFrame({"target": ["a", "a", "b", "b"], "session": ["s1", "s1", "s2", "s2"],
                       "final": [True, True, False, False], "head": [False, False, True, False],
                       "names_context": [False, False, True, True]})
    c = changes(df, "head")
    assert c["fixed"] == {"windows": 2, "recordings": 1, "sessions": 1}
    assert c["new"] == {"windows": 1, "recordings": 1, "sessions": 1, "names_context": 1}


def test_analyse_blocks_and_paired_difference():
    meta = _meta()
    # final: a right everywhere; b wrong in its two other-view windows, right otherwise
    final = np.array([True, True, True, True, True, True, False, False])
    names = np.array([False, False, False, False, False, False, True, True])
    out = analyse(meta, final, names, e1={"a": True, "b": True}, head={"a": False, "b": True})
    assert set(out) == {"own window, 16 of 16", "same view, 4 of 16", "other view, 4 of 16"}
    oth = out["other view, 4 of 16"]
    assert oth["windows"] == 4 and oth["targets"] == 2 and oth["final_accuracy"] == pytest.approx(0.5)
    assert oth["head"]["accuracy"] == pytest.approx(0.5) and oth["head"]["final_minus_control"] == pytest.approx(0.0)
    assert oth["head"]["changes"]["fixed"]["recordings"] == 1 and oth["head"]["changes"]["new"]["names_context"] == 2
    assert oth["e1"]["final_minus_control"] == pytest.approx(-0.5)
    assert out["same view, 4 of 16"]["head"]["changes"]["new"]["names_context"] == 0


def test_block_key():
    assert block_key("own", 16) == "own window, 16 of 16" and block_key("other view", 4.0) == "other view, 4 of 16"


def _meta_all_shares() -> pd.DataFrame:
    rows = [{"id": f"a|{c}|{m}", "target": "a", "session": "s", "label": 1, "m": float(m), "condition": c, "context_label": 0}
            for m in (4, 8, 12, 14) for c in ("same view", "other view")]
    return pd.DataFrame(rows)


def test_reproduces_measures_the_largest_accuracy_gap():
    meta = _meta_all_shares()
    final = np.array([c == "same view" for c in meta["condition"]])   # same view right, other view wrong
    stored = {str(m): {"same_view_accuracy": 1.0, "other_view_accuracy": 0.0} for m in (4, 8, 12, 14)}
    assert reproduces(meta, final, stored) == 0.0
    stored["12"]["other_view_accuracy"] = 0.25
    assert reproduces(meta, final, stored) == pytest.approx(0.25)


def test_reproduces_pairs_other_view_windows_with_a_same_view_window():
    meta = _meta_all_shares()
    extra = pd.DataFrame([{"id": "b|other view|4", "target": "b", "session": "s2", "label": 1, "m": 4.0, "condition": "other view", "context_label": 0}])
    meta = pd.concat([meta, extra], ignore_index=True)       # target b has no same-view window: the stored probe skips it
    final = np.array([c == "same view" for c in meta["condition"]]); final[-1] = True
    stored = {str(m): {"same_view_accuracy": 1.0, "other_view_accuracy": 0.0} for m in (4, 8, 12, 14)}
    assert reproduces(meta, final, stored) == 0.0


def test_analyse_refuses_a_control_without_every_target():
    meta = _meta(); final = np.ones(len(meta), dtype=bool)
    with pytest.raises(ValueError, match="misses probe targets"):
        analyse(meta, final, np.zeros(len(meta), dtype=bool), e1={"a": True}, head={"a": True, "b": True})


def test_reproduces_refuses_a_missing_share():
    meta = _meta_all_shares(); meta = meta[meta["m"] != 8]
    final = np.ones(len(meta), dtype=bool)
    stored = {str(m): {"same_view_accuracy": 1.0, "other_view_accuracy": 1.0} for m in (4, 8, 12, 14)}
    with pytest.raises(ValueError, match="8 of 16"):
        reproduces(meta, final, stored)
