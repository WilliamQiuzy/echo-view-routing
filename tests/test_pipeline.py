"""The ensemble's rules, with stand-in window models whose answers are known."""
import numpy as np
import pytest

from echoview import Ensemble, weights
from echoview.pipeline import softmax
from echoview.reference import B3

from .conftest import BrightnessModel, flat_frames


def system(*shifts: float) -> Ensemble:
    return Ensemble.from_models([BrightnessModel(shift) for shift in shifts], B3, device="cpu")


def test_a_recording_takes_the_view_of_its_grey_level():
    result = system(0.0).classify(flat_frames([100] * 30))
    assert result.view == "A4C" and result.probability > 0.99
    assert result.n_windows == 6 and result.seeds_agree


def test_the_ensemble_averages_the_seeds_mean_window_probabilities():
    ensemble = system(0.0, 20.0, -20.0)
    frames = flat_frames([100] * 12 + [150] * 8)
    logits = ensemble.window_logits(frames)
    assert logits.shape == (4, 3, 5) and logits.dtype == np.float32
    by_seed = softmax(logits).mean(0)
    result = ensemble.classify(frames)
    np.testing.assert_allclose(result.by_seed, by_seed)
    np.testing.assert_allclose(result.probabilities, by_seed.mean(0))
    assert abs(result.probabilities.sum() - 1.0) < 1e-9


def test_seeds_that_name_different_views_are_reported():
    result = system(0.0, 50.0).classify(flat_frames([85] * 20))  # one seed reads 85 (A4C), the other 135 (A5C)
    assert result.seeds_agree is False


def test_a_stream_is_cut_where_the_view_changes():
    result = system(0.0, 5.0, -5.0).segment(flat_frames([0] * 40 + [100] * 40))
    assert [(start, end, view) for start, end, view, _ in result.segments] == [(0.0, 4.0, "PLAX"), (4.0, 8.0, "A4C")]
    assert result.labels.tolist() == [0] * 40 + [2] * 40
    assert result.probabilities.shape == (80, 5)


def test_every_sample_takes_the_probability_of_the_nearest_window():
    ensemble = system(0.0)
    prob = ensemble.sample_probabilities(flat_frames([0] * 13 + [200] * 12))
    assert prob[:10].argmax(1).tolist() == [0] * 10  # the windows centred at samples 2 and 7
    assert prob[15:].argmax(1).tolist() == [4] * 10  # the windows centred at 17 and 22


def test_frames_of_another_shape_or_type_are_refused():
    ensemble = system(0.0)
    with pytest.raises(ValueError, match="uint8 frames"):
        ensemble.window_logits(np.zeros((4, 200, 200, 3), np.uint8))
    with pytest.raises(ValueError, match="uint8 frames"):
        ensemble.window_logits(np.zeros((4, 224, 224, 3), np.float32))


def test_models_of_your_own_are_never_called_the_reference():
    ensemble = system(0.0, 1.0)
    assert ensemble.is_reference is False and ensemble.version == "custom ensemble of 2 model(s)"
    with pytest.raises(ValueError, match="at least one model"):
        Ensemble.from_models([], B3)


def test_unknown_or_repeated_seeds_are_refused(tmp_path):
    for seeds in ((), (0, 0), (5,)):
        with pytest.raises(ValueError, match="seed"):
            Ensemble(tmp_path, seeds=seeds)


def test_changed_weight_files_are_not_loaded_as_the_reference(tmp_path):
    for seed in (0, 1, 2):
        (tmp_path / f"C2_s{seed}").mkdir()
        (tmp_path / f"C2_s{seed}" / "final.pt").write_bytes(b"not the published file")
    with pytest.raises(weights.WeightsError, match="not a published file"):
        Ensemble(tmp_path)
