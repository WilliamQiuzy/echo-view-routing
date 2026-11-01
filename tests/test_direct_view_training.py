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
    dt.close_cell(cell)
    cell = dt.open_cell(tmp_path, "L", identity, 3)
    assert dt.claim_row(cell) == 1
    dt.close_cell(cell)
    with pytest.raises(RuntimeError, match="inflight"):
        dt.open_cell(tmp_path, "L", identity, 3)
    # A different fresh cell exercises successful completion; the interrupted
    # cell remains blocked and is never repaired by clearing its claim.
    cell = dt.open_cell(tmp_path, "completed", identity, 3)
    assert dt.claim_row(cell) == 0
    dt.finish_row(cell, 0, .2)
    assert dt.claim_row(cell) == 1
    dt.finish_row(cell, 1, -.3)
    assert dt.claim_row(cell) == 2
    dt.finish_row(cell, 2, 0.)
    assert dt.claim_row(cell) is None
    receipt = dt.seal_cell(cell)
    assert receipt["rows"] == 3
    assert np.array_equal(np.load(tmp_path / "completed.npy"), np.array([.2, -.3, 0.]))
    dt.close_cell(cell)
    cell = dt.open_cell(tmp_path, "completed", identity, 3)
    assert dt.claim_row(cell) is None
    dt.close_cell(cell)
    with pytest.raises(ValueError, match="identity"):
        dt.open_cell(tmp_path, "completed", {**identity, "model": "other"}, 3)


def test_existing_score_cell_has_exclusive_owner(tmp_path):
    cell = dt.open_cell(tmp_path, "L", {"model": "one"}, 2)
    with pytest.raises(RuntimeError, match="owned"):
        dt.open_cell(tmp_path, "L", {"model": "one"}, 2)
    dt.close_cell(cell)
    reopened = dt.open_cell(tmp_path, "L", {"model": "one"}, 2)
    dt.close_cell(reopened)


def test_update_crash_after_checkpoint_cannot_repeat_possibly_executed_step(tmp_path):
    seen = []
    def side_effect_then_crash():
        seen.append(150)
        raise RuntimeError("simulated process loss after optimizer work")
    dt.write_json(tmp_path / "update_guard.json", {"pending": None, "last_committed": 149})
    with pytest.raises(RuntimeError, match="simulated"):
        dt.claimed_update(tmp_path, 150, side_effect_then_crash, lambda rec: None)
    assert seen == [150]
    with pytest.raises(RuntimeError, match="possibly executed"):
        dt.validate_recovery(tmp_path, checkpoint_step=150, journal_length=150)
    with pytest.raises(RuntimeError, match="possibly executed"):
        dt.claimed_update(tmp_path, 150, side_effect_then_crash, lambda rec: None)
    assert seen == [150]


def test_committed_source_update_requires_matching_checkpoint_and_journal(tmp_path):
    rows = []
    rec = dt.claimed_update(tmp_path, 0, lambda: {"u": 0}, rows.append)
    assert rec == {"u": 0} and rows == [rec]
    dt.validate_recovery(tmp_path, checkpoint_step=1, journal_length=1)
    with pytest.raises(RuntimeError, match="unsaved"):
        dt.validate_recovery(tmp_path, checkpoint_step=0, journal_length=1)


def test_source_metrics_use_true_view_and_strict_positive_threshold():
    y = np.array([0, 0, 1, 1])
    s = np.array([0., 1., 1., -1.])
    out = dt.source_metrics(y, s, np.array([False, True, True, False]))
    assert out["balanced_accuracy"] == .5
    assert out["a4c_fpr"] == .5
    assert out["discordant_accuracy"] == .5
    assert out["tp"] == out["tn"] == out["fp"] == out["fn"] == 1
