from __future__ import annotations

import numpy as np
import pytest

from echo_routing.temporal.segments import Segment
from echo_routing.evaluate.metrics import (
    boundary_f1, classification_summary, coverage_and_risk, fragments_per_minute, match_boundaries,
    risk_coverage_curve, segment_f1, temporal_iou,
)


def test_classification_summary_basic():
    s = classification_summary([0, 1, 1, 2], [0, 1, 0, 2], 3)
    assert s["n"] == 4 and s["accuracy"] == 0.75
    assert 0 < s["macro_f1"] < 1 and len(s["confusion"]) == 3


def test_fragments_per_minute():
    labels = np.array([0] * 300 + [1] * 300)  # 60 s at 10 Hz, one change
    assert fragments_per_minute(labels, hz=10) == pytest.approx(1.0)
    assert fragments_per_minute(np.array([0]), 10) == 0.0


def test_boundary_matching_is_one_to_one():
    pairs = match_boundaries(np.array([10, 11, 30]), np.array([10, 31]), tol=2)
    assert sorted(pairs) == [(0, 0), (2, 1)]
    r = boundary_f1(np.array([10, 11, 30]), np.array([10, 31]), tol=2)
    assert r.matched == 2 and r.precision == pytest.approx(2 / 3) and r.recall == 1.0
    assert r.median_abs_error == pytest.approx(0.5)


def test_boundary_f1_edge_cases():
    assert boundary_f1(np.array([]), np.array([]), 1).f1 == 1.0
    assert boundary_f1(np.array([5]), np.array([]), 1).precision == 0.0
    assert boundary_f1(np.array([]), np.array([5]), 1).recall == 0.0


def test_segment_iou_and_f1():
    a, b = Segment(0, 10, 1), Segment(5, 15, 1)
    assert temporal_iou(a, b) == pytest.approx(1 / 3)
    r = segment_f1([Segment(0, 10, 1), Segment(10, 20, 2)], [Segment(0, 9, 1), Segment(9, 20, 2)], 0.5)
    assert r["f1"] == 1.0
    assert segment_f1([Segment(0, 10, 1)], [Segment(0, 10, 2)], 0.5)["f1"] == 0.0


def test_coverage_and_risk():
    acc = np.array([1, 1, 0, 1], bool); wrong = np.array([0, 1, 1, 0], bool); w = np.array([1, 1, 1, 2.0])
    r = coverage_and_risk(acc, wrong, w)
    assert r["coverage"] == pytest.approx(4 / 5) and r["risk"] == pytest.approx(1 / 4)
    assert coverage_and_risk(np.zeros(3, bool), wrong[:3])["risk"] is None


def test_risk_coverage_curve_monotone_coverage():
    curve = risk_coverage_curve(np.array([0.9, 0.5, 0.1]), np.array([0, 0, 1], bool))
    assert [p["coverage"] for p in curve] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert curve[-1]["risk"] == pytest.approx(1 / 3)
