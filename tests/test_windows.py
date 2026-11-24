"""The window grid must be the research code's, for every sequence length."""
import numpy as np
import pytest

from echoview import windows


def test_plan_equals_research_code_for_every_stored_length(parity):
    for n in parity["window_lengths"].tolist():
        centres, frames, slots = windows.window_plan(n)
        assert np.array_equal(centres, parity[f"w{n}_centres"]), n
        assert np.array_equal(frames, parity[f"w{n}_frames"]), n
        assert np.array_equal(slots, parity[f"w{n}_slots"]), n
        assert np.array_equal(windows.nearest_centre(n, centres), parity[f"w{n}_nearest"]), n


def test_every_window_shows_16_samples_and_marks_its_centre():
    for n in (1, 3, 16, 17, 101):
        centres, frames, slots = windows.window_plan(n)
        assert frames.shape == (len(centres), 16)
        assert frames.min() >= 0 and frames.max() < n
        assert all(window[slot] == centre for window, slot, centre in zip(frames, slots, centres))


def test_one_prediction_every_half_second_and_the_end_is_covered():
    centres, _, _ = windows.window_plan(103)
    assert np.all(np.diff(centres[:-1]) == 5)
    assert centres[-1] >= 103 - 1 - 2


def test_an_empty_sequence_has_no_plan():
    with pytest.raises(ValueError, match="empty"):
        windows.window_plan(0)
    assert windows.window_centres(0).size == 0


def test_another_step_asks_more_often_for_the_same_windows():
    for n in (1, 7, 16, 40, 103):
        dense, frames, slots = windows.window_plan(n, 1)
        assert dense.tolist() == list(range(n))  # an output for every sample
        for stride in (2, 5):
            centres, sparse_frames, sparse_slots = windows.window_plan(n, stride)
            assert np.array_equal(centres, windows.window_centres(n, stride))
            assert np.array_equal(sparse_frames, frames[centres]) and np.array_equal(sparse_slots, slots[centres])
    for n in (1, 16, 103):
        assert all(np.array_equal(a, b) for a, b in zip(windows.window_plan(n), windows.window_plan(n, windows.STRIDE)))


def test_a_step_below_one_sample_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        windows.window_plan(10, 0)
