"""The published weights on the sample clips. Needs the weights: pytest -m weights."""
import pytest

from echoview import cli, load, read_recording, weights

pytestmark = pytest.mark.weights


def test_the_published_files_are_found_and_verified():
    folder = weights.resolve()
    assert all(weights.verify(folder).values())


def test_the_self_check_passes(capsys):
    assert cli.main(["selfcheck", "--device", "cpu"]) == 0
    assert "SELF-CHECK PASSED" in capsys.readouterr().out


def test_the_reference_ensemble_names_a_sample_clip():
    system = load(device="cpu")
    assert system.is_reference and system.version == "C2_ens_v1"
    clip = sorted(weights.examples_folder(system.weights_dir).glob("SC4C*.mp4"))[0]
    result = system.classify(read_recording(clip).frames[:20])
    assert result.view == "SC4C" and result.seeds_agree
