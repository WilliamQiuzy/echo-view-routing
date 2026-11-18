"""Stage 3 on a machine that can be switched off: checkpoints, and a cut run that continues as the uncut run would.

Two runs of this loop are not equal bit for bit, on the CPU as little as on a GPU. The test of a continued run
therefore compares it with an uncut run by distance. Its limits sit between sizes measured on this tiny run. Two uncut
runs differ by 2e-8 of the weights' length, by 6e-10 for the moving average and by 2e-7 in the loss. A continuation
without the optimiser's state ends 1e-4 away; without the moving average, the average ends 5e-4 away; with a wrong
random stream (input changes or dropout) the next loss differs in its first decimal and the weights end 6e-5 away.
The sampler's position is checked directly: the continued run must be fed the windows the uncut run was fed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from echoview.training import cache, fast, plans, train

REPOSITORY = Path(__file__).resolve().parents[1]


def distance(a: dict, b: dict) -> float:
    """The length of the difference of two weight files, as a share of the first one's length."""
    assert set(a) == set(b)
    gap = sum(float((a[key].double() - b[key].double()).pow(2).sum()) for key in a) ** 0.5
    return gap / sum(float(a[key].double().pow(2).sum()) for key in a) ** 0.5


def arguments(tiny_training: tuple[Path, Path, Path]) -> list[str]:
    folder, plan, start = tiny_training
    return ["--config", str(REPOSITORY / "configs" / "stage3.json"), "--cache", str(folder), "--plans", str(plan), "--init", str(start),
            "--device", "cpu", "--updates", "4", "--warmup", "1", "--batch", "4", "--compile", "0", "--log-every", "1", "--frames", "gpu",
            "--seed", "1"]  # with seed 1 the third and fourth updates draw other groups than the first two


def log_of(folder: Path) -> list[dict]:
    return [json.loads(line) for line in (folder / "train_log.jsonl").read_text().splitlines()]


def test_a_cut_run_continues_as_the_uncut_run_would(tiny_training, tmp_path, monkeypatch, capsys):
    argv = arguments(tiny_training)
    real, fed = train.augment.augment, {"whole": [], "cut": [], "continued": []}

    def watch(name: str, stop_at: int | None = None):
        def run(source, fragment, *args, **kwargs):
            if stop_at is not None and len(fed[name]) == stop_at:
                raise KeyboardInterrupt  # the machine is switched off
            fed[name].append(hashlib.sha256(source.numpy().tobytes() + fragment.numpy().tobytes()).hexdigest())
            return real(source, fragment, *args, **kwargs)
        return run

    monkeypatch.setattr(train.augment, "augment", watch("whole"))
    whole = train.train(train.parse_args([*argv, "--out", str(tmp_path / "whole")]))
    assert {p.name for p in whole.iterdir()} == {"config.json", "train_log.jsonl", "final.pt", "ema.pt", "done.json"}
    assert fed["whole"][2] != fed["whole"][0] and fed["whole"][3] != fed["whole"][1]  # a sampler that began again would show

    # the same run with a checkpoint every two updates, switched off during its fourth update
    cut = tmp_path / "cut"
    monkeypatch.setattr(train.augment, "augment", watch("cut", stop_at=3))
    with pytest.raises(KeyboardInterrupt):
        train.train(train.parse_args([*argv, "--out", str(cut), "--checkpoint-every", "2"]))
    assert (cut / "checkpoint.pt").is_file() and not (cut / "final.pt").exists() and not list(cut.glob("*.tmp"))
    assert [r["update"] for r in log_of(cut)] == [1, 2, 3]  # one record ahead of the checkpoint

    # it continues without its start weights: everything it needs is in the checkpoint
    start_file = Path(argv[argv.index("--init") + 1])
    start_file.rename(start_file.with_name("moved_away.pt"))
    monkeypatch.setattr(train.augment, "augment", watch("continued"))
    capsys.readouterr()
    out = train.train(train.parse_args([*argv, "--out", str(cut), "--checkpoint-every", "2", "--resume"]))
    assert "continuing after update 2 of 4" in capsys.readouterr().err
    start_file.with_name("moved_away.pt").rename(start_file)
    assert fed["continued"] == fed["whole"][2:]  # the windows of the third and fourth update, not of the first two
    for name, limit in (("final.pt", 1e-5), ("ema.pt", 1e-7)):
        a, b = (torch.load(folder / name, weights_only=True) for folder in (whole, out))
        assert distance(a, b) < limit, name
    assert [(r["update"], r["accuracy"], r["lr_enc"]) for r in log_of(out)] == [(r["update"], r["accuracy"], r["lr_enc"]) for r in log_of(whole)]
    assert max(abs(x["loss"] - y["loss"]) for x, y in zip(log_of(out), log_of(whole))) < 1e-4
    assert {p.name for p in out.iterdir()} == {"config.json", "train_log.jsonl", "final.pt", "ema.pt", "done.json"}
    assert json.loads((out / "done.json").read_text())["windows"] == 16

    # a finished run is left alone, a checkpoint left beside it is removed, and other options are refused
    def never(*args, **kwargs):
        raise AssertionError("a finished run trained again")

    monkeypatch.setattr(train.augment, "augment", never)
    (out / "checkpoint.pt").write_bytes(b"left over")
    assert train.train(train.parse_args([*argv, "--out", str(cut), "--resume"])) == out
    assert not (out / "checkpoint.pt").exists() and "finished" in capsys.readouterr().err
    with pytest.raises(SystemExit, match=r"updates: 4 -> 9"):
        train.train(train.parse_args([*argv, "--out", str(cut), "--resume", "--updates", "9"]))


def test_a_run_says_how_it_starts(tmp_path):
    args = train.parse_args(["--cache", "c", "--out", str(tmp_path), "--init", "x", "--updates", "1500", "--resume"])
    assert "continuing after update 300 of 1500" in train.resume_note({"updates_done": 300}, args, tmp_path)
    fresh = train.resume_note(None, args, tmp_path)
    assert "no checkpoint" in fresh and "first update" in fresh and "--checkpoint-every" in fresh  # nothing will be written to continue from
    args.checkpoint_every = 100
    assert "--checkpoint-every" not in train.resume_note(None, args, tmp_path)


def test_the_sampler_goes_on_at_the_draw_after_the_last_finished_update(tiny_training):
    folder, plan, _ = tiny_training
    pool = plans.GroupPool(cache.Cache(folder), [plan])
    uncut = np.random.default_rng([train.SAMPLING_STREAM, 0])
    draws = [pool.batch(uncut, 4, 0.5) for _ in range(40)]
    assert len({draw[0].tobytes() for draw in draws}) > 1  # the draws are not all the same group
    continued = np.random.default_rng([train.SAMPLING_STREAM, 0])
    train.skip_draws(pool, continued, 37, 4, 0.5)
    for want in draws[37:]:
        got = pool.batch(continued, 4, 0.5)
        assert all(np.array_equal(x, y) for x, y in zip(got, want))


def test_a_checkpoint_holds_every_part_of_the_loop_and_reads_back_equal(tmp_path):
    torch.manual_seed(5)
    model = train.initial_model(None, 0)
    optimiser = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": 1e-5}, {"params": model.new_parameters(), "lr": 1e-4}])
    for parameter in model.new_parameters():
        parameter.grad = torch.ones_like(parameter)
    optimiser.step()
    average = fast.Average(model)
    average.update(0.5)
    generator = torch.Generator().manual_seed(9)
    torch.rand(3, generator=generator)
    train.save_checkpoint(tmp_path, 100, {"seed": 0}, model, optimiser, average, generator, 3200, 12.5, torch.device("cpu"))
    state = train.load_checkpoint(tmp_path / "checkpoint.pt")
    assert (state["updates_done"], state["seen"], state["elapsed_s"], state["identity"]) == (100, 3200, 12.5, {"seed": 0})
    assert state["cuda_rng"] is None and not list(tmp_path.glob("*.tmp"))  # on a GPU: the stream of the training device alone
    assert all(torch.equal(state["model"][key], value) for key, value in model.state_dict().items())
    assert all(torch.equal(a, b) for a, b in zip(state["average"], average.mean)) and len(state["average"]) == len(average.mean)
    stored = [entry["exp_avg"] for entry in state["optimiser"]["state"].values()]
    assert len(stored) == len(model.new_parameters()) and all(float(moment.abs().sum()) > 0 for moment in stored)
    want = torch.rand(4, generator=generator)
    again = torch.Generator()
    again.set_state(state["generator"])
    assert torch.equal(torch.rand(4, generator=again), want)
    after = torch.rand(4)
    torch.set_rng_state(state["torch_rng"])
    assert torch.equal(torch.rand(4), after)
    (tmp_path / "checkpoint.pt").write_bytes(b"cut short")
    with pytest.raises(SystemExit, match=r"cannot be read: \w+") as damaged:
        train.load_checkpoint(tmp_path / "checkpoint.pt")
    assert damaged.value.__cause__ is not None  # the reason is kept
    torch.save({"model": {}}, tmp_path / "checkpoint.pt")
    with pytest.raises(SystemExit, match="not a checkpoint of this program"):
        train.load_checkpoint(tmp_path / "checkpoint.pt")


def test_records_written_after_the_last_checkpoint_are_dropped_when_a_run_continues(tmp_path):
    log = tmp_path / "train_log.jsonl"
    log.write_text("".join(json.dumps({"update": u, "loss": 1.0}) + "\n" for u in (50, 100, 150)) + '{"update": 200, "lo')
    train.keep_log_until(log, 100)
    assert [json.loads(line)["update"] for line in log.read_text().splitlines()] == [50, 100]
    assert [p.name for p in tmp_path.iterdir()] == ["train_log.jsonl"]  # swapped in whole, nothing left beside it


def test_a_checkpoint_of_a_run_with_other_options_is_refused(tiny_training, tmp_path):
    argv = arguments(tiny_training)
    args = train.parse_args([*argv, "--out", str(tmp_path / "run"), "--checkpoint-every", "1"])
    other = train.parse_args([*argv, "--out", str(tmp_path / "run"), "--checkpoint-every", "1", "--seed", "2", "--penalty", "0.3"])
    moved = train.parse_args([*argv, "--out", str(tmp_path / "elsewhere"), "--checkpoint-every", "2", "--resume"])
    assert train.run_identity(args) == train.run_identity(moved)
    with pytest.raises(SystemExit, match=r"penalty: 0.15 -> 0.3.*seed: 1 -> 2"):
        train.same_run(train.run_identity(args), train.run_identity(other), tmp_path / "run" / "checkpoint.pt")
    # the same files reached by another path (a new machine, another mount) are the same run
    elsewhere = [("/mnt/new/" + Path(a).name if a.endswith((".pt", ".npz")) else a) for a in argv]
    assert train.run_identity(train.parse_args([*elsewhere, "--out", "o"])) == train.run_identity(args)


def test_checkpoints_fall_on_log_records(tiny_training, tmp_path):
    argv = arguments(tiny_training)
    with pytest.raises(SystemExit, match="--checkpoint-every"):
        train.train(train.parse_args([*argv, "--out", str(tmp_path / "run"), "--log-every", "2", "--checkpoint-every", "3"]))
    with pytest.raises(SystemExit):
        train.parse_args([*argv, "--out", str(tmp_path / "run"), "--checkpoint-every", "-1"])


def test_the_moving_average_can_be_stored_and_restored():
    model = torch.nn.Linear(3, 2)
    average = fast.Average(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    average.update(0.5)
    stored = average.state()
    other = fast.Average(torch.nn.Linear(3, 2))
    other.load(stored)
    assert all(torch.equal(a, b) for a, b in zip(other.mean, average.mean)) and stored[0] is not average.mean[0]
    with pytest.raises(ValueError):
        other.load(stored[:1])
