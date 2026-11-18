"""Behaviours fixed after the independent review of 2026-11-17: each test failed before its fix."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from echoview import Ensemble, cli, selfcheck, weights, windows
from echoview import reference as ref
from echoview.evaluation import benchmark, ev9v
from echoview.evaluation.ev9v import recording_key
from echoview.reference import B3
from echoview.training import cache, plans, staged, train

from .conftest import BrightnessModel, write_video

REPOSITORY = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------ which system is the reference
@pytest.fixture(scope="module")
def three_seeds(tmp_path_factory) -> tuple[Path, dict]:
    """A weights folder with three loadable seeds (one random network, three files) and their checksums."""
    folder = tmp_path_factory.mktemp("weights")
    torch.manual_seed(0)
    state = train.initial_model(None, 0).state_dict()
    sums = {}
    for seed in (0, 1, 2):
        (folder / f"C2_s{seed}").mkdir()
        target = folder / f"C2_s{seed}" / "final.pt"
        if seed == 0:
            torch.save(state, target)
        else:
            os.link(folder / "C2_s0" / "final.pt", target)
        sums[seed] = {"final.pt": weights.sha256(target), "model.safetensors": ""}
    return folder, sums


def test_files_with_the_published_checksums_are_the_reference(three_seeds, monkeypatch):
    folder, sums = three_seeds
    monkeypatch.setattr(weights, "WEIGHT_SHA256", sums)
    system = Ensemble(folder, device="cpu")
    assert system.published and system.is_reference and system.version == ref.VERSION
    assert Ensemble(folder, device="cpu", seeds=(0, 2)).is_reference is False  # fewer seeds are not the reference


def test_the_label_follows_the_checksums_not_the_switch(three_seeds, monkeypatch):
    folder, sums = three_seeds
    monkeypatch.setattr(weights, "WEIGHT_SHA256", sums)
    assert Ensemble(folder, device="cpu", verify=False).is_reference is True  # the published files, whatever the switch
    monkeypatch.setattr(weights, "WEIGHT_SHA256", {seed: {"final.pt": "0" * 64} for seed in (0, 1, 2)})
    own = Ensemble(folder, device="cpu", verify=False)
    assert own.published is False and own.is_reference is False and "not the reference" in own.version
    with pytest.raises(weights.WeightsError, match="--unverified"):
        Ensemble(folder, device="cpu")


def test_a_decoder_setting_must_make_sense():
    with pytest.raises(ValueError, match="half_window"):
        Ensemble.from_models([BrightnessModel()], {**B3, "half_window": 0}, device="cpu")


# ------------------------------------------------------------------------------------------------- the Hub
@pytest.mark.parametrize("error", [OSError("no network"), ValueError("offline and not in the cache")])
def test_any_failed_download_is_a_weights_error_with_advice(monkeypatch, error):
    import huggingface_hub

    def fail(**_kwargs):
        raise error

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    with pytest.raises(weights.WeightsError, match="ECHOVIEW_WEIGHTS"):
        weights.download()


def test_start_weights_are_found_beside_the_weights_and_checked(tmp_path, monkeypatch):
    (tmp_path / "training").mkdir()
    (tmp_path / "training" / "stage2_s1.pt").write_bytes(b"the stage-2 model of seed 1")
    monkeypatch.setattr(weights, "START_SHA256", {1: hashlib.sha256(b"the stage-2 model of seed 1").hexdigest()})
    assert weights.start_weights(1, tmp_path) == tmp_path / "training" / "stage2_s1.pt"
    (tmp_path / "training" / "stage2_s1.pt").write_bytes(b"something else")
    with pytest.raises(weights.WeightsError, match=r"stage2_s1\.pt"):
        weights.start_weights(1, tmp_path)


def test_start_weights_have_published_checksums():
    assert set(ref.START_SHA256) == {0, 1, 2} and all(len(value) == 64 for value in ref.START_SHA256.values())


# ---------------------------------------------------------------------------------------------- command line
def test_repeated_seeds_and_unknown_devices_end_with_a_message(tmp_path):
    with pytest.raises(SystemExit, match="once"):
        cli.parse_seeds("0,0")
    args = cli.parser().parse_args(["classify", "x", "--weights", str(tmp_path), "--unverified", "--device", "bogus"])
    for seed in (0, 1, 2):
        (tmp_path / f"C2_s{seed}").mkdir()
        (tmp_path / f"C2_s{seed}" / "final.pt").write_bytes(b"x")
    with pytest.raises(SystemExit, match="device"):
        cli.build(args)


def test_evaluate_says_at_once_when_scipy_is_missing(tmp_path, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "scipy" else real(name, *a, **k))
    with pytest.raises(SystemExit, match="evaluation"):
        cli.main(["evaluate", "--images", str(tmp_path / "nowhere")])


def test_evaluate_asks_for_the_weights_before_it_decodes_frames(tmp_path):
    (tmp_path / "weights").mkdir()
    with pytest.raises(SystemExit, match="C2_s0"):  # the weights problem, not the missing dataset
        cli.main(["evaluate", "--images", str(tmp_path / "nowhere"), "--weights", str(tmp_path / "weights")])


def test_segment_samples_name_the_whole_path_of_each_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "build", lambda args, verify=True: Ensemble.from_models([BrightnessModel()], B3, device="cpu"))
    for folder, level in (("a", 100), ("b", 0)):
        (tmp_path / folder).mkdir()
        write_video(tmp_path / folder / "clip.mp4", [level] * 60, fps=30.0)
    samples = tmp_path / "samples.csv"
    assert cli.main(["segment", str(tmp_path / "a" / "clip.mp4"), str(tmp_path / "b" / "clip.mp4"), "--samples", str(samples)]) == 0
    files = {line.split(",")[2] for line in samples.read_text().splitlines()[1:]}
    assert files == {str(tmp_path / "a" / "clip.mp4"), str(tmp_path / "b" / "clip.mp4")}


# ------------------------------------------------------------------------------------------------ self-check
@pytest.fixture()
def sample_folder(tmp_path) -> Path:
    folder = tmp_path / "examples"
    folder.mkdir()
    for name, level in (("one.mp4", 0), ("two.mp4", 100)):
        write_video(folder / name, [level] * 90, fps=30.0)
    (folder / "views.json").write_text(json.dumps({"one.mp4": "PLAX", "two.mp4": "A4C"}))
    return folder


def test_the_self_check_writes_and_then_passes_its_stored_results(sample_folder, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build", lambda args, verify=True: Ensemble.from_models([BrightnessModel()], B3, device="cpu"))
    monkeypatch.setattr(weights, "verify", lambda folder, seeds: {})
    assert cli.main(["selfcheck", "--clips", str(sample_folder), "--write-expected"]) == 0  # the first expected.json
    stored = json.loads((sample_folder / "expected.json").read_text())
    assert [stored["clips"][name]["view"] for name in ("one.mp4", "two.mp4")] == ["PLAX", "A4C"]
    assert cli.main(["selfcheck", "--clips", str(sample_folder)]) == 0
    assert "SELF-CHECK PASSED" in capsys.readouterr().out
    stored["clips"]["two.mp4"]["view"] = "PSAX"
    (sample_folder / "expected.json").write_text(json.dumps(stored))
    assert cli.main(["selfcheck", "--clips", str(sample_folder)]) == 1
    assert "SELF-CHECK FAILED: two.mp4" in capsys.readouterr().out


def test_a_sample_folder_without_clips_is_named(tmp_path):
    (tmp_path / "expected.json").write_text("{}")
    with pytest.raises(SystemExit, match="no sample clip"):
        selfcheck.sample_clips(tmp_path)


# --------------------------------------------------------------------------------------------------- windows
def test_the_nearest_window_of_a_long_stream_needs_little_memory():
    n = 400_000  # eleven hours at 10 Hz; the old table of all distances would need hundreds of GB
    centres = windows.window_centres(n)
    nearest = windows.nearest_centre(n, centres)
    assert nearest.shape == (n,) and nearest[0] == 0 and nearest[-1] == len(centres) - 1
    for sample in (0, 4, 5, 7, 12345, n - 3, n - 1):  # the earlier centre wins a tie
        assert nearest[sample] == int(np.abs(sample - centres).argmin())


def test_the_nearest_window_equals_the_table_of_distances_for_short_streams():
    for n in (1, 2, 3, 5, 6, 7, 8, 9, 11, 16, 23, 104, 257):
        centres = windows.window_centres(n)
        table = np.abs(np.arange(n)[:, None] - centres[None, :]).argmin(1)
        assert np.array_equal(windows.nearest_centre(n, centres), table), n


# ---------------------------------------------------------------------------------------- the frame cache file
def test_a_damaged_frame_cache_file_is_rebuilt(tmp_path):
    images = tmp_path / "Images"
    (images / "rec").mkdir(parents=True)
    for k in range(4):
        Image.fromarray(np.full((240, 320, 3), 50 + k, np.uint8)).save(images / "rec" / f"{k}.jpg", quality=95)
    sequence = benchmark.Sequence("s", "native", (benchmark.Fragment(recording_key("rec"), 0, 1.0, (0, 1, 2, 3)),))
    first = ev9v.FrameTable.build([sequence], images, workers=1, cache=tmp_path / "cache")
    stored = next((tmp_path / "cache").glob("*.npy"))
    np.save(stored, np.zeros_like(first.grey))  # the right shape, the wrong content
    again = ev9v.FrameTable.build([sequence], images, workers=1, cache=tmp_path / "cache")
    assert np.array_equal(again.grey, first.grey) and int(again.grey.max()) > 0


# --------------------------------------------------------------------------------------------------- training
def recipe_file(tmp_path: Path, **values) -> str:
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(values))
    return str(path)


@pytest.mark.parametrize("values, word", [({"objective": "within_viw"}, "within_viw"), ({"frames": "cpu"}, "cpu"), ({"updates": "many"}, "many")])
def test_a_recipe_value_is_checked_like_a_command_line_value(tmp_path, values, word):
    with pytest.raises(SystemExit):
        train.parse_args(["--config", recipe_file(tmp_path, **values), "--cache", "c", "--out", "o", "--init", "x"])
    args = train.parse_args(["--config", recipe_file(tmp_path, updates=20, objective="ce"), "--cache", "c", "--out", "o", "--init", "x", "--updates", "7"])
    assert args.updates == 7 and isinstance(args.updates, int) and args.objective == "ce"


def test_a_stage_recipe_value_is_checked_too(tmp_path):
    with pytest.raises(SystemExit):
        staged.parse_args(["--config", recipe_file(tmp_path, loop="plan", objective="sideways"), "--images", "i", "--out", "o"])


def test_training_refuses_a_missing_start_file_and_a_broken_batch_before_it_loads_anything(tmp_path):
    base = ["--cache", str(tmp_path / "no_cache"), "--out", str(tmp_path / "out"), "--device", "cpu"]
    with pytest.raises(SystemExit, match=r"start\.pt"):
        train.train(train.parse_args([*base, "--init", str(tmp_path / "start.pt")]))
    (tmp_path / "start.pt").write_bytes(b"x")
    with pytest.raises(SystemExit, match="whole groups"):
        train.train(train.parse_args([*base, "--init", str(tmp_path / "start.pt"), "--objective", "within_view", "--batch", "6"]))
    with pytest.raises(SystemExit, match="--encoder"):
        train.train(train.parse_args([*base, "--init", "encoder"]))


class StubCache:
    """The layout of the frame cache without its frames: enough to resolve the plans."""

    def __init__(self) -> None:
        self.clips = cache.layout(cache.recordings())
        self.by_key = {clip["key"]: clip for clip in self.clips}

    def rows(self, split: str) -> list[dict]:
        return [clip for clip in self.clips if clip["split"] == split]

    @property
    def train_end(self) -> int:
        last = self.rows("train")[-1]
        return last["start"] + last["count"]


def test_the_shipped_plans_resolve_against_the_shipped_list_of_recordings():
    stub = StubCache()
    groups = plans.GroupPool(stub, train.packaged_plans("within_view"))
    singles = plans.Pool(stub, train.packaged_plans("ce"))
    assert groups.rows.shape == (22500, 4, 16) and int(groups.rows.max()) < stub.train_end == 207614
    assert len(singles.rows) == 45000 and [len(table) for table in singles.by_class] == [1120, 974, 858, 217, 195]


def test_the_fast_loop_runs_two_updates_on_the_cpu_and_writes_its_files(tiny_training, tmp_path):
    folder, plan, start = tiny_training
    argv = ["--config", str(REPOSITORY / "configs" / "stage3.json"), "--cache", str(folder), "--plans", str(plan), "--init", str(start),
            "--device", "cpu", "--updates", "2", "--warmup", "1", "--batch", "4", "--compile", "0", "--log-every", "1", "--frames", "gpu"]
    out = train.train(train.parse_args([*argv, "--out", str(tmp_path / "run")]))
    assert {p.name for p in out.iterdir()} == {"config.json", "train_log.jsonl", "final.pt", "ema.pt", "done.json"}
    final, ema, before = (torch.load(p, weights_only=True) for p in (out / "final.pt", out / "ema.pt", start))
    assert not torch.equal(final["classifier.weight"], before["classifier.weight"])
    assert not torch.equal(final["encoder.conv_proj.weight"], before["encoder.conv_proj.weight"])  # freeze 0: the encoder trains at once
    assert not torch.equal(ema["classifier.weight"], final["classifier.weight"]) and set(ema) == set(final)
    log = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert [r["update"] for r in log] == [1, 2] and all(np.isfinite(r["loss"]) for r in log)
    assert log[0]["lr_enc"] == pytest.approx(1e-5) and log[0]["lr_new"] == pytest.approx(1e-4)
    config = json.loads((out / "config.json").read_text())
    assert config["objective"] == "within_view" and config["pool_rows"] == 8 and config["strength_ranges"]["shared"] == 1.0
