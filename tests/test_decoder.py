"""The change-point decoder must give the research code's labels on the stored cases."""
import numpy as np
import pytest

from echoview import decoder


def test_labels_equal_research_code_on_every_stored_case(parity):
    for case, (half, threshold, radius) in enumerate(parity["decoder_settings"]):
        labels = decoder.decode(parity[f"d{case}_prob"], int(half), float(threshold), int(radius))
        assert np.array_equal(labels, parity[f"d{case}_labels"]), case


def test_scores_equal_research_code_on_every_stored_case(parity):
    for case, (half, _threshold, _radius) in enumerate(parity["decoder_settings"]):
        scores = decoder.boundary_scores(parity[f"d{case}_prob"], int(half))
        np.testing.assert_allclose(scores, parity[f"d{case}_scores"], rtol=1e-10, atol=1e-13)


def one_hot_stream(views: list[int], length: int) -> np.ndarray:
    prob = np.full((len(views) * length, 5), 0.01)
    for k, view in enumerate(views):
        prob[k * length:(k + 1) * length, view] = 0.96
    return prob


def test_a_clear_change_is_cut_where_it_happens():
    labels = decoder.decode(one_hot_stream([2, 0], 40), half_window=7, threshold=0.17, nms_radius=5)
    assert decoder.segments(labels) == [(0, 40, 2), (40, 80, 0)]


def test_a_stream_of_one_view_is_one_piece():
    labels = decoder.decode(one_hot_stream([3], 60), half_window=7, threshold=0.17, nms_radius=5)
    assert decoder.segments(labels) == [(0, 60, 3)]


def test_a_peak_whose_two_sides_name_the_same_view_is_no_cut():
    prob = one_hot_stream([1, 1], 30)
    prob[30:, 1], prob[30:, 4] = 0.55, 0.42  # the probabilities move, the winner does not
    labels = decoder.decode(prob, half_window=7, threshold=0.01, nms_radius=5)
    assert decoder.segments(labels) == [(0, 60, 1)]


def test_weaker_peaks_inside_the_radius_are_dropped():
    kept = decoder.non_max_suppression(np.array([0.0, 0.5, 0.9, 0.6, 0.0, 0.0, 0.0, 0.0, 0.7]), threshold=0.4, radius=5)
    assert kept.tolist() == [2, 8]


def test_segments_of_an_empty_label_sequence():
    assert decoder.segments(np.array([], int)) == []


def test_scores_computed_beforehand_give_the_same_labels():
    rng = np.random.default_rng(3)
    prob = rng.dirichlet(np.ones(5) * 0.3, 120)
    prob[40:80] = rng.dirichlet(np.array([8.0, 0.2, 0.2, 0.2, 0.2]), 40)
    for half_window, threshold in ((5, 0.05), (7, 0.2), (15, 0.4)):
        scores = decoder.boundary_scores(prob, half_window)
        plain = decoder.decode(prob, half_window, threshold, 5)
        assert np.array_equal(decoder.decode(prob, half_window, threshold, 5, scores=scores), plain)
        assert np.array_equal(decoder.detect_boundaries(prob, half_window, threshold, 5, scores=scores),
                              decoder.detect_boundaries(prob, half_window, threshold, 5))


def test_scores_of_another_stream_are_refused():
    prob = np.full((30, 5), 0.2)
    with pytest.raises(ValueError, match="scores"):
        decoder.decode(prob, 5, 0.1, 5, scores=np.zeros(29))
