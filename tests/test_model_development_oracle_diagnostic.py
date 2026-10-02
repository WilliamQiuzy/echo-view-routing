"""ORACLE_DIAGNOSTIC helpers (round 2, Task 2) on hand-made streams."""
from __future__ import annotations

import numpy as np

from echo_routing.compose.recipe_builder import Fragment, Recipe
from echo_routing.compose.stream_assembler import Stream
from scripts.model_development_oracle_diagnostic import (error_runs, native_summary, oracle_labels, oracle_summary,
                                                         reference_segments)


def _stream(frags, rid="m1") -> Stream:
    fr = tuple(Fragment(v, lab, "orig", 0, n) for v, lab, n in frags)
    labels = np.concatenate([np.full(n, lab) for _, lab, n in frags])
    joins = np.cumsum([n for _, _, n in frags])[:-1]
    sem = np.array([j for j in joins if labels[j] != labels[j - 1]], int)
    return Stream(Recipe(rid, "multi", fr), np.zeros((labels.size, 5)), np.zeros((labels.size, 4)), labels, joins, sem)


def _onehot(labels, n_classes=5, p=0.9):
    out = np.full((len(labels), n_classes), (1 - p) / (n_classes - 1))
    out[np.arange(len(labels)), labels] = p
    return out


def test_same_class_joins_are_not_reference_boundaries():
    s = _stream([("a", 0, 10), ("b", 1, 10), ("c", 1, 10), ("d", 2, 10)])
    segs = reference_segments(s)
    assert [(g["start"], g["end"], g["label"]) for g in segs] == [(0, 10, 0), (10, 30, 1), (30, 40, 2)]
    assert segs[1]["videos"] == ["b", "c"]


def test_oracle_averages_inside_each_reference_segment():
    s = _stream([("a", 0, 10), ("b", 1, 10)])
    prob = _onehot(np.array([0] * 10 + [1] * 4 + [3] * 6))      # second segment: mean favours class 3
    lab, seg_pred = oracle_labels(prob, reference_segments(s))
    assert seg_pred == [0, 3] and lab.tolist() == [0] * 10 + [3] * 10


def test_oracle_summary_counts_segments_samples_and_source_clips():
    s1 = _stream([("a", 0, 10), ("b", 1, 10), ("c", 1, 10)], "m1")    # the 20-sample PSAX segment is read as A5C
    s2 = _stream([("b", 1, 10), ("d", 2, 10)], "m2")                   # clip b again, read as A5C
    probs = {"m1": _onehot(np.array([0] * 10 + [3] * 20)), "m2": _onehot(np.array([3] * 10 + [2] * 10))}
    out = oracle_summary(probs, [s1, s2])
    assert out["reference_segments"] == 4 and out["wrong_segments"] == 2 and out["segment_error_rate"] == 0.5
    assert out["sample_error_rate"] == 30 / 50
    assert out["wrong_source_videos"] == 2 and out["wrong_source_occurrences"] == 3   # b twice, c once
    assert out["confusion"][1][3] == 2 and out["label"] == "ORACLE_DIAGNOSTIC"


def test_default_error_runs_by_type():
    s = _stream([("a", 0, 20), ("b", 2, 20), ("c", 1, 20)])
    lab = s.labels.copy()
    lab[20:23] = 0          # late change: timing run at a true change
    lab[30:32] = 4          # interior run of 2 samples
    lab[40:60] = 3          # whole third fragment wrong
    out = error_runs({"m1": lab}, [s])
    assert out["runs"] == {"timing": 1, "whole_fragment": 1, "interior_short": 1, "interior_long": 0}
    assert out["whole_fragment_source_videos"] == 1


def test_native_summary_keeps_numerators_and_denominators():
    class C:
        def __init__(self, sid, y, n=10):
            self.seq_id, self.labels = sid, np.full(n, y)
    natives = [C("x1", 3), C("x2", 3), C("x3", 2)]
    labels = {"x1": np.full(10, 3), "x2": np.array([2] * 6 + [3] * 4), "x3": np.full(10, 2)}
    out = native_summary(labels, natives)
    assert out["recall"]["3"] == [1, 2] and out["recall"]["2"] == [1, 1] and out["wrong_clips"] == ["x2"]
