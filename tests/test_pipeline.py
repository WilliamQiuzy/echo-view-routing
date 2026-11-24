"""The ensemble's rules, with stand-in window models whose answers are known."""
import numpy as np
import pytest

from echoview import Ensemble, reference, uncertain, weights, windows
from echoview.pipeline import probabilities_from_logits, softmax
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


def test_a_rule_lets_the_segmentation_decline_samples_and_changes_nothing_else():
    from echoview import uncertain

    frames = flat_frames([0] * 40 + [100] * 40)
    plain = system(0.0, 5.0, -5.0).segment(frames)
    assert plain.uncertain.sum() == 0 and (plain.labels >= 0).all()
    marked = system(0.0, 5.0, -5.0).segment(frames, uncertain.Rule(band=3))
    assert np.flatnonzero(marked.uncertain).tolist() == [37, 38, 39, 40, 41, 42] and (marked.labels[marked.uncertain] == -1).all()
    assert np.array_equal(marked.labels[~marked.uncertain], plain.labels[~marked.uncertain])  # the other samples keep their view
    assert [(start, end, view) for start, end, view, _ in marked.segments] == [(0.0, 3.7, "PLAX"), (3.7, 4.3, "uncertain"), (4.3, 8.0, "A4C")]
    assert all(0.0 <= probability <= 1.0 for *_, probability in marked.segments)
    # an uncertain piece reports the mean probability of its likeliest view, which is low where two views meet
    across = marked.probabilities[37:43]
    assert marked.segments[1][3] == pytest.approx(across.mean(0).max()) and marked.segments[1][3] < across.max(1).mean() - 0.2
    assert np.array_equal(system(0.0, 5.0, -5.0).segment(frames, uncertain.Rule(seeds=True)).labels, plain.labels)  # the seeds agree here


def test_a_segmentation_built_without_the_uncertain_mask_derives_it():
    """Code written before the uncertain output existed builds a Segmentation from three values."""
    from echoview import Segmentation

    plain = Segmentation(labels=np.array([0, 0, 2, 2]), probabilities=np.zeros((4, 5)), segments=())
    assert plain.uncertain.dtype == bool and not plain.uncertain.any()
    assert Segmentation(np.array([0, -1, -1, 2]), np.zeros((4, 5)), ()).uncertain.tolist() == [False, True, True, False]


def test_a_denser_read_out_gives_every_sample_its_own_output():
    ensemble = system(0.0, 5.0)
    frames = flat_frames([0] * 13 + [200] * 12)
    dense = ensemble.window_logits(frames, stride=1)
    assert dense.shape == (25, 2, 5)
    np.testing.assert_allclose(dense[windows.window_centres(25)], ensemble.window_logits(frames), atol=1e-5)  # the published outputs are among them
    prob = probabilities_from_logits(dense, 25, stride=1)
    assert prob.argmax(1).tolist() == [0] * 13 + [4] * 12  # exact to the sample, where the 0.5 s grid is not
    assert ensemble.sample_probabilities(frames).argmax(1).tolist() != prob.argmax(1).tolist()
    with pytest.raises(ValueError, match="has 5 windows, not 25"):
        probabilities_from_logits(dense, 25)


def published(monkeypatch, *shifts: float) -> Ensemble:
    """A stand-in that passes for the published system, so that its denser read-outs can be tried."""
    monkeypatch.setattr(Ensemble, "is_reference", property(lambda self: True))
    return system(*shifts)


def test_a_denser_read_out_cuts_nearer_and_says_what_it_is(monkeypatch):
    ensemble = published(monkeypatch, 0.0, 5.0, -5.0)
    frames = flat_frames([0] * 44 + [100] * 36)  # the view changes at sample 44, between two outputs of the 0.5 s grid
    coarse, fifth, tenth = ensemble.segment(frames), ensemble.segment(frames, stride=2), ensemble.segment(frames, stride=1)
    assert coarse.labels.tolist() == [0] * 45 + [2] * 35  # the published read-out cuts at 4.5 s
    assert tenth.labels.tolist() == [0] * 44 + [2] * 36  # an output for every sample: the cut is at 4.4 s
    assert [(start, end, view) for start, end, view, _ in tenth.segments] == [(0.0, 4.4, "PLAX"), (4.4, 8.0, "A4C")]
    assert fifth.labels.tolist() == [0] * 45 + [2] * 35  # sample 44 lies midway between two outputs and takes the earlier one
    odd = flat_frames([0] * 43 + [100] * 37)  # a change at an odd sample has an output of its own every 0.2 s, and none every 0.5 s
    assert ensemble.segment(odd, stride=2).labels.tolist() == [0] * 43 + [2] * 37 and ensemble.segment(odd).labels.tolist() == [0] * 45 + [2] * 35
    assert tenth.probabilities.shape == (80, 5) and not tenth.uncertain.any()
    np.testing.assert_array_equal(probabilities_from_logits(ensemble.window_logits(frames, 1), 80, 1), tenth.probabilities)
    np.testing.assert_array_equal(ensemble.sample_probabilities(frames), coarse.probabilities)  # this method is the published read-out
    assert ensemble.decoder_setting() == B3 and ensemble.decoder_setting(5) == B3
    assert ensemble.decoder_setting(2) == reference.DENSER_READ_OUT[2] and ensemble.decoder_setting(1) == reference.DENSER_READ_OUT[1]
    assert ensemble.version_at(5) == ensemble.version
    assert ensemble.version_at(1) == f"{ensemble.version}, read every 0.1 s" and ensemble.version_at(2) == f"{ensemble.version}, read every 0.2 s"


def test_the_decoder_setting_of_a_denser_read_out_is_the_one_it_decodes_with(monkeypatch):
    ensemble = published(monkeypatch, 0.0)
    frames = flat_frames([0] * 30 + [100] * 30)
    monkeypatch.setitem(reference.DENSER_READ_OUT, 1, {"half_window": 5, "threshold": 10.0, "nms_radius": 5})  # no score reaches this
    assert ensemble.segment(frames, stride=1).labels.tolist() == [0] * 60  # so nothing is cut, and the piece takes the larger mean
    assert len(ensemble.segment(frames, stride=2).segments) == 2 and len(ensemble.segment(frames).segments) == 2
    monkeypatch.setitem(reference.DENSER_READ_OUT, 2, {"half_window": 5, "threshold": 10.0, "nms_radius": 5})
    assert len(ensemble.segment(frames, stride=2).segments) == 1 and len(ensemble.segment(frames).segments) == 2  # each step has its own


def test_a_denser_read_out_is_for_the_published_system_without_the_uncertain_rule(monkeypatch):
    frames = flat_frames([0] * 40 + [100] * 40)
    for stride in (1, 2):
        with pytest.raises(ValueError, match="published system"):
            system(0.0).segment(frames, stride=stride)
    assert len(system(0.0).segment(frames, stride=5).segments) == 2  # the published step needs no such check
    ensemble = published(monkeypatch, 0.0)
    with pytest.raises(ValueError, match="was not studied"):
        ensemble.segment(frames, stride=3)
    for stride in (1, 2):
        for rule in (uncertain.released(), uncertain.Rule(seeds=True), uncertain.Rule(margin=0.1)):
            with pytest.raises(ValueError, match="uncertain rule was studied with the published read-out"):
                ensemble.segment(frames, rule, stride=stride)
    with pytest.raises(TypeError):
        ensemble.sample_probabilities(frames, stride=1)  # a denser read-out comes from segment, which checks and names it
    assert ensemble.segment(frames, uncertain.OFF, stride=1).labels.tolist() == [0] * 40 + [2] * 40
    assert ensemble.segment(frames, uncertain.released()).labels.shape == (80,)
