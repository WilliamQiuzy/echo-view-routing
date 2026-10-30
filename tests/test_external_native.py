import numpy as np
import pytest

from echo_routing.evaluate.external_native import (
    majority_label, native_metrics, preserve_unknown_map, sample_indices, window_plan,
)


def test_time_sampling_preserves_real_source_rate():
    np.testing.assert_array_equal(sample_indices(171, 50), np.arange(0, 171, 5))
    np.testing.assert_array_equal(sample_indices(31, 30), np.arange(0, 31, 3))
    assert sample_indices(1, 30).tolist() == [0]
    with pytest.raises(ValueError):
        sample_indices(31, 0)


def test_target_marker_moves_at_edges_and_windows_keep_real_support():
    centres, windows, slots = window_plan(80)
    assert centres.tolist() == list(range(2, 80, 5))
    assert windows.shape == (16, 16)
    assert windows[4].tolist() == list(range(14, 30))
    assert slots[4] == 8
    assert slots[0] == 2 and slots[-1] == 13
    c, w, s = window_plan(3)
    assert w.shape[1] == 16
    assert w.min() == 0 and w.max() == 2
    assert w[0, s[0]] == c[0]


def test_unmapped_top_prediction_is_an_error_not_a_forced_supported_choice():
    p = np.array([[.8, .1, .1], [.1, .3, .6]])
    # Raw class zero is not in the five-family task.
    assert preserve_unknown_map(p, {1: 2, 2: 3}).tolist() == [-1, 3]
    assert majority_label(np.array([-1, -1, 2])) == -1


def test_native_metrics_weight_each_recording_once_and_keep_unknown_errors():
    rows = [
        {'true': 2, 'pred': 2, 'n_samples': 1000, 'cuts': 1, 'duration_s': 100},
        {'true': 2, 'pred': -1, 'n_samples': 10, 'cuts': 0, 'duration_s': 1},
        {'true': 3, 'pred': 2, 'n_samples': 20, 'cuts': 2, 'duration_s': 2},
    ]
    m = native_metrics(rows)
    assert m['n_clips'] == 3
    assert m['accuracy'] == pytest.approx(1/3)
    assert m['class_recall']['A4C'] == .5
    assert m['class_recall']['A5C'] == 0
    assert m['class_recall']['PLAX'] is None
    assert m['confusion'][2][5] == 1
    assert m['internal_cuts'] == 3
    assert m['five_class_macro_f1'] is None


def test_five_class_macro_requires_all_true_classes():
    rows = [{'true': i, 'pred': i, 'n_samples': 10, 'cuts': 0, 'duration_s': 1} for i in range(5)]
    assert native_metrics(rows)['five_class_macro_f1'] == 1
