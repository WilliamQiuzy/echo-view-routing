"""Fixed-interval relabelling helpers (round-2 diagnostic)."""
from __future__ import annotations

import numpy as np

from scripts.model_development_relabel_diagnostic import interval_table, intervals, relabel


def _p(labels, p=0.8):
    out = np.full((len(labels), 5), (1 - p) / 4); out[np.arange(len(labels)), labels] = p
    return out


def test_intervals_are_runs_of_the_default_output():
    assert intervals(np.array([2, 2, 3, 3, 3, 2])) == [(0, 2), (2, 5), (5, 6)]
    assert intervals(np.array([], int)) == []


def test_relabel_uses_the_mean_probability_inside_each_interval():
    prob = _p([3, 3, 2, 2, 2, 2])                     # one interval: mean favours class 2
    lab, per = relabel(prob, [(0, 6)])
    assert per == [2] and lab.tolist() == [2] * 6


def test_interval_table_counts_fixes_breaks_and_source_clips_once():
    default = {"m": np.array([3] * 10 + [1] * 10)}     # model intervals 0-10 and 10-20
    truth = {"m": np.array([2] * 10 + [1] * 10)}
    own = {"m": _p([3] * 10 + [1] * 10)}               # own: 3 then 1 -> first interval wrong
    vis = {"m": _p([2] * 10 + [4] * 10)}               # visual: 2 then 4 -> fixes the first, breaks the second
    spans = {"m": [(0, 6, "a"), (6, 10, "b"), (10, 20, "c")]}
    out = interval_table(["m"], default, own, vis, truth, spans)
    assert out["intervals"] == 2 and out["label_changed"] == 2
    assert out["own_wrong_intervals"] == 1 and out["visual_wrong_intervals"] == 1
    assert out["visual_fixes"] == 1 and out["visual_breaks"] == 1
    assert out["own_wrong_source_clips"] == 2 and out["visual_wrong_source_clips"] == 1   # a, b versus c
