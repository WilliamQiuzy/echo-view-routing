"""Stages 1 and 2: the objective, the schedule, the frames of a plan row, and a few updates on the CPU."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from echoview.evaluation.ev9v import recording_key
from echoview.training import cli, staged, train

REPOSITORY = Path(__file__).resolve().parents[1]
NAMES = ("rec_a", "rec_b")


def test_the_stage_2_objective_is_cross_entropy_plus_the_smoothing_penalty():
    torch.manual_seed(0)
    logits = (torch.randn(8, 5) * 2.0).requires_grad_()
    target = torch.tensor([0, 0, 1, 1, 3, 3, 3, 2])
    loss, parts = staged.within_view_loss(logits, target, penalty=0.15)
    assert loss.dtype == torch.float64
    want = F.cross_entropy(logits.double(), target) + 0.15 * train.neighbour_penalty(logits.double())
    assert abs(float(loss) - float(want)) < 1e-6
    assert abs(parts["ce"] + 0.15 * parts["smoothing"] - float(loss)) < 1e-12
    loss.backward()
    assert logits.grad is not None and logits.grad.dtype == torch.float32 and torch.isfinite(logits.grad).all()


def test_equal_neighbours_add_nothing_and_a_huge_jump_is_cut():
    same = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0]]).repeat(4, 1)
    _, parts = staged.within_view_loss(same, torch.zeros(4, dtype=torch.long), penalty=0.15)
    assert parts["smoothing"] == 0.0
    jump = torch.zeros(4, 5)
    jump[1, 0] = 200.0
    _, parts = staged.within_view_loss(jump, torch.zeros(4, dtype=torch.long), penalty=0.15)
    assert parts["smoothing"] <= 16.0 * 2 * 5 / 15 + 1e-9


def test_the_encoder_rests_during_warm_up():
    assert staged.schedule(0, 375, 3750, 1e-5, 1e-4) == (0.0, 1e-4 / 375)
    assert staged.schedule(374, 375, 3750, 1e-5, 1e-4) == (0.0, 1e-4)
    encoder, new = staged.schedule(375, 375, 3750, 1e-5, 1e-4)
    assert encoder == 1e-5 and new == 1e-4
    encoder, new = staged.schedule(3749, 375, 3750, 1e-5, 1e-4)
    assert 0 < encoder < 1e-9 and 0 < new < 1e-8


# The learning rates at the last update of every tenth of the published stage-1 run of seed 0 (its train_log.jsonl).
PUBLISHED_RATES = {375: (0.0, 0.0001), 750: (9.700052902196541e-06, 9.700052902196541e-05), 1125: (8.833212225163124e-06, 8.833212225163124e-05),
                   1500: (7.5040295815877e-06, 7.5040295815877e-05), 1875: (5.8728240149320505e-06, 5.8728240149320505e-05),
                   2250: (4.136342990565055e-06, 4.1363429905650545e-05), 2625: (2.504031747755862e-06, 2.5040317477558617e-05),
                   3000: (1.1727711127355118e-06, 1.1727711127355118e-05), 3375: (3.0313076540193e-07, 3.0313076540192998e-06),
                   3750: (2.1661681620654963e-12, 2.1661681620654962e-11)}


def test_the_schedule_gives_the_learning_rates_of_the_published_log_to_the_last_digit():
    for done, rates in PUBLISHED_RATES.items():
        assert staged.schedule(done - 1, 375, 3750, 1e-5, 1e-4) == rates, done


@pytest.fixture()
def tiny(tmp_path) -> tuple[Path, Path]:
    """Two recordings of 30 frames whose grey level is 100 * recording + frame, and a plan of two updates of four
    windows: three windows inside rec_a, the fourth changing to rec_b with a gamma of 0.9."""
    images = tmp_path / "Images"
    for r, name in enumerate(NAMES):
        (images / name).mkdir(parents=True)
        for k in range(30):
            Image.fromarray(np.full((240, 320, 3), 100 * r + k, np.uint8)).save(images / name / f"{k}.png")
    vid = np.zeros((2, 4, 16), np.int32)
    frame = np.tile(np.arange(16, dtype=np.int32), (2, 4, 1))
    gamma = np.ones((2, 4, 16), np.float32)
    vid[:, 3, 8:], gamma[:, 3, 8:] = 1, 0.9
    frame[1] += 20  # the second update reads frames 20 to 35: past the end of the recording
    target = np.zeros((2, 4), np.int8)
    target[:, 3] = 1
    plan = tmp_path / "plan.npz"
    np.savez_compressed(plan, recordings=np.array([recording_key(n) for n in NAMES]), vid=vid, frame=frame, gamma=gamma,
                        marker=np.full((2, 4), 8, np.int8), target=target, kind=np.zeros((2, 4), np.int8))
    return images, plan


def test_a_plan_row_becomes_the_frames_of_its_windows(tiny):
    images, plan = tiny
    data = staged.PlanFrames(staged.read_plan(plan), images, [0, 1])
    update, frames, marker, target = data[0]
    assert update == 0 and frames.shape == (4, 16, 224, 224, 3) and frames.dtype == torch.uint8
    assert marker.tolist() == [8] * 4 and target.tolist() == [0, 0, 0, 1]
    assert [int(frames[0, slot, 0, 0, 0]) for slot in (0, 7, 15)] == [0, 7, 15]
    assert int(frames[3, 7, 0, 0, 0]) == 7  # still rec_a
    assert int(frames[3, 8, 0, 0, 0]) == round(255 * (108 / 255) ** 0.9)  # rec_b, frame 8, with the gamma
    _, later, _, _ = data[1]
    assert [int(later[0, slot, 0, 0, 0]) for slot in (0, 9, 15)] == [20, 29, 29]  # a frame past the end is the last one


def test_a_recording_missing_from_the_dataset_is_reported(tiny, tmp_path):
    images, plan = tiny
    (images / "rec_b").rename(tmp_path / "elsewhere")
    with pytest.raises(staged.DataError, match="1 of the 2 recordings"):
        staged.PlanFrames(staged.read_plan(plan), images, [0])


def test_two_updates_run_on_the_cpu_and_repeat_exactly(tiny, tmp_path):
    images, plan = tiny
    start = tmp_path / "start.pt"
    torch.manual_seed(1)
    torch.save(train.initial_model(None, 0).state_dict(), start)
    common = ["--config", str(REPOSITORY / "configs" / "stage2.json"), "--images", str(images), "--plan-file", str(plan), "--init", str(start),
              "--device", "cpu", "--workers", "0", "--updates", "2", "--warmup", "1", "--log-every", "1"]
    for name in ("first", "again"):
        assert cli.main([*common, "--out", str(tmp_path / name)]) == 0
    first, again = (torch.load(tmp_path / name / "final.pt", weights_only=True) for name in ("first", "again"))
    before = torch.load(start, weights_only=True)
    assert all(torch.equal(first[key], again[key]) for key in first)
    assert not torch.equal(first["classifier.weight"], before["classifier.weight"])
    assert not torch.equal(first["encoder.conv_proj.weight"], before["encoder.conv_proj.weight"])  # the encoder joined at update 1
    log = [json.loads(line) for line in (tmp_path / "first" / "train_log.jsonl").read_text().splitlines()]
    assert [r["updates"] for r in log] == [1, 2] and log[0]["lr_encoder_last"] == 0.0 and log[1]["lr_encoder_last"] > 0.0


def test_the_command_picks_the_loop_named_by_the_recipe(tmp_path):
    assert cli.loop_of(["--config", str(REPOSITORY / "configs" / "stage1.json")]) == "plan"
    assert cli.loop_of(["--config", str(REPOSITORY / "configs" / "stage2.json")]) == "plan"
    assert cli.loop_of(["--config", str(REPOSITORY / "configs" / "stage3.json")]) == "fast"
    assert cli.loop_of(["--cache", "x"]) == "fast"
    (tmp_path / "odd.json").write_text(json.dumps({"loop": "sideways"}))
    with pytest.raises(SystemExit, match="sideways"):
        cli.loop_of(["--config", str(tmp_path / "odd.json")])


def test_the_stage_recipes_are_the_published_ones():
    one = staged.parse_args(["--config", str(REPOSITORY / "configs" / "stage1.json"), "--images", "i", "--out", "o", "--encoder", "e.pt"])
    two = staged.parse_args(["--config", str(REPOSITORY / "configs" / "stage2.json"), "--images", "i", "--out", "o", "--init", "s.pt"])
    assert (one.plan, one.objective, one.updates, one.warmup) == ("windows", "ce", 3750, 375)
    assert (one.lr_enc, one.lr_new, one.weight_decay, one.grad_clip) == (1e-5, 1e-4, 1e-4, 1.0)
    assert (two.plan, two.objective, two.penalty, two.updates, two.warmup) == ("groups", "within_view", 0.15, 3750, 375)
    assert staged.packaged_plan("windows", 2).is_file() and staged.packaged_plan("groups", 0).is_file()
