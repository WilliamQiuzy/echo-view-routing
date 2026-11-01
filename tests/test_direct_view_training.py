import numpy as np
import pytest

from echo_routing.temporal import direct_view_training as dt


def test_same_plan_covers_every_request_and_balances_directories():
    directory = np.array([0] * 19 + [1] * 13)
    plan = dt.source_plan(directory, updates=20)
    assert plan.shape == (20, 8)
    assert set(plan.ravel()) == set(range(32))
    assert np.all(directory[plan].sum(1) == 4)
    assert np.array_equal(plan, dt.source_plan(directory, updates=20))
    with pytest.raises(ValueError, match="coverage"):
        dt.source_plan(np.array([0] * 81 + [1] * 2), updates=20)


def test_durable_rows_resume_saved_prefix_and_reject_unsaved_forward(tmp_path):
    identity = {"model": "abc", "inputs": "xyz"}
    cell = dt.open_cell(tmp_path, "L", identity, 3)
    assert dt.claim_row(cell) == 0
    dt.finish_row(cell, 0, .2)
    cell = dt.open_cell(tmp_path, "L", identity, 3)
    assert dt.claim_row(cell) == 1
    with pytest.raises(RuntimeError, match="inflight"):
        dt.open_cell(tmp_path, "L", identity, 3)
    dt.finish_row(cell, 1, -.3)
    assert dt.claim_row(cell) == 2
    dt.finish_row(cell, 2, 0.)
    assert dt.claim_row(cell) is None
    receipt = dt.seal_cell(cell)
    assert receipt["rows"] == 3
    assert np.array_equal(np.load(tmp_path / "L.npy"), np.array([.2, -.3, 0.]))
    assert dt.claim_row(dt.open_cell(tmp_path, "L", identity, 3)) is None
    with pytest.raises(ValueError, match="identity"):
        dt.open_cell(tmp_path, "L", {**identity, "model": "other"}, 3)


def test_source_metrics_use_true_view_and_strict_positive_threshold():
    y = np.array([0, 0, 1, 1])
    s = np.array([0., 1., 1., -1.])
    out = dt.source_metrics(y, s, np.array([False, True, True, False]))
    assert out["balanced_accuracy"] == .5
    assert out["a4c_fpr"] == .5
    assert out["discordant_accuracy"] == .5
    assert out["tp"] == out["tn"] == out["fp"] == out["fn"] == 1
