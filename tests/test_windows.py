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
