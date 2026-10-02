"""Model development protocol: session-level validation halves and per-half banks (2026-10-02 plan, Task 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.model_development_protocol import half_banks, session_of, split_sessions

CFG = {"recipes": {"n_per_cell": 4, "min_len": 10, "max_len": 40, "edit_variant": "gamma090", "seed": 1},
       "multi": {"n_eval": 6, "n_fragments": [4, 8], "edit_prob": 0.5, "p_same_next": 0.35}}


def _val(n_sessions: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed); rows = []
    for s in range(n_sessions):
        for c in range(6):
            rows.append({"video_id": f"2022-01-{s:02d}_10-00-00_320x240_{c}", "split": "validation",
                         "label_index": int(rng.integers(0, 5)) if c else s % 5, "n_samples": 60})
    df = pd.DataFrame(rows); df["session"] = df["video_id"].map(session_of)
    return df


def test_session_of_strips_the_clip_number():
    assert session_of("2022-07-27_14-06-42_320x240_2") == "2022-07-27_14-06-42_320x240"


def test_halves_are_disjoint_complete_and_cover_all_classes():
    val = _val()
    halves, attempts = split_sessions(val, seed=20261002, max_attempts=50)
    a, b = set(halves["selection"]), set(halves["calibration"])
    assert not (a & b) and a | b == set(val["session"]) and attempts >= 1
    for h in halves.values():
        assert set(val[val["session"].isin(h)]["label_index"]) == set(range(5))
    assert split_sessions(val, 20261002, 50) == (halves, attempts)  # deterministic


def test_impossible_coverage_stops():
    val = _val(n_sessions=4)
    val.loc[val["session"] != val["session"].iloc[0], "label_index"] = 0
    with pytest.raises(SystemExit):
        split_sessions(val, seed=1, max_attempts=5)


def test_half_banks_use_only_their_own_clips_and_prefixed_ids():
    val = _val()
    halves, _ = split_sessions(val, 20261002, 50)
    sub = val[val["session"].isin(halves["selection"])]
    banks = half_banks(sub, "selection", CFG, seed=7)
    own = set(sub["video_id"])
    for kind in ("pairs", "multi"):
        assert banks[kind].recipes and all(r.recipe_id.startswith(f"{kind}_selection_") for r in banks[kind].recipes)
        assert all(f.video_id in own for r in banks[kind].recipes for f in r.fragments)
    assert {r.recipe_id for r in banks["native"].recipes} == own
