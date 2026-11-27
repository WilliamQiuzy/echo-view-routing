"""Training: the augmentation, the hook-free forward, the sampling plans, the frame cache and the loop's helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from echoview import model as md
from echoview.evaluation.ev9v import recording_key
from echoview.frames import crop_uint8
from echoview.training import augment, cache, fast, plans, train


def blocky(seed: int = 0, b: int = 2, h: int = 240, w: int = 320) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (b, 16, h // 8, w // 8)).astype(np.uint8)
    return np.repeat(np.repeat(small, 8, 2), 8, 3)


# -------------------------------------------------------------------------------------------- augmentation
def test_strength_zero_is_the_plain_input():
    frames = blocky()
    out = augment.warp(torch.from_numpy(frames), torch.zeros((2, 16), dtype=torch.long),
                       augment.draw(2, augment.Strength(0.0), torch.Generator().manual_seed(0), "cpu"))
    for b in range(2):
        for t in (0, 7, 15):
            plain = crop_uint8(Image.fromarray(np.repeat(frames[b, t][..., None], 3, -1)))[..., 0].astype(np.float32)
            assert np.abs(out[b, t].numpy() * 255.0 - plain).max() <= 1.01


def test_augment_shape_range_and_reproducibility():
    frames = torch.from_numpy(blocky(2, 4))
    fragment = torch.zeros((4, 16), dtype=torch.long)
    fragment[2:, 9:] = 1
    a = augment.augment(frames, fragment, augment.Strength(1.0), torch.Generator().manual_seed(5))
    b = augment.augment(frames, fragment, augment.Strength(1.0), torch.Generator().manual_seed(5))
    c = augment.augment(frames, fragment, augment.Strength(1.0), torch.Generator().manual_seed(6))
    assert a.shape == (4, 3, 16, 224, 224) and torch.isfinite(a).all()
    assert torch.equal(a, b) and not torch.equal(a, c)
    low, high = (0.0 - max(md.MEAN)) / min(md.STD), (255.0 - min(md.MEAN)) / min(md.STD)
    assert float(a.min()) >= low - 1e-3 and float(a.max()) <= high + 1e-3


def test_shared_parameters_copy_fragment_zero():
    params = augment.draw(64, augment.Strength(1.0, shared=1.0), torch.Generator().manual_seed(1), "cpu")
    assert torch.equal(params["gain"][:, 0], params["gain"][:, 2]) and torch.equal(params["zoom"][:, 0], params["zoom"][:, 1])
    free = augment.draw(64, augment.Strength(1.0, shared=0.0), torch.Generator().manual_seed(1), "cpu")
    assert not torch.equal(free["gain"][:, 0], free["gain"][:, 1])


def test_draw_ranges():
    s = augment.Strength(1.0)
    p = augment.draw(4000, s, torch.Generator().manual_seed(2), "cpu")
    assert float(p["gain"].min()) >= np.exp(-s.gain_low) - 1e-6 and float(p["gain"].max()) <= np.exp(s.gain_high) + 1e-6
    assert float(p["zoom"].min()) >= np.exp(-s.zoom) - 1e-6 and float(p["zoom"].max()) <= np.exp(s.zoom) + 1e-6
    assert 0.10 < float((p["zoom"] == 1.0).float().mean()) < 0.60  # no geometric change, alone or copied from fragment 0
    assert float(p["border"].max()) <= s.border + 1e-6 and float(p["tint"].min()) >= 1 - s.tint - 1e-6


def test_tone_curve_is_monotone_and_bounded():
    params = augment.draw(8, augment.Strength(1.0, tone_p=1.0), torch.Generator().manual_seed(3), "cpu")
    assert torch.allclose(params["tone_w"].sum(-1), torch.ones(8, augment.MAX_FRAGMENTS), atol=1e-5)
    ramp = torch.linspace(0, 1, 224).view(1, 1, 1, 224).expand(8, 16, 224, 224).contiguous()
    neutral = {**params, "gamma": torch.ones_like(params["gamma"]), "gain": torch.ones_like(params["gain"]), "ramp": torch.zeros_like(params["ramp"]),
               "blur": torch.zeros(8), "sharpen": torch.zeros(8), "speckle": torch.zeros(8), "additive": torch.zeros(8), "border": torch.zeros(8, 4)}
    out = augment.grey_levels(ramp, torch.zeros((8, 16), dtype=torch.long), neutral, torch.Generator().manual_seed(0))
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0 + 1e-6
    assert bool((out[:, :, :, 1:] - out[:, :, :, :-1] >= -1e-6).all())
    assert not torch.allclose(out, ramp, atol=1e-3)


def test_group_multipliers_switch_groups_off():
    generator = lambda: torch.Generator().manual_seed(9)
    levels_only = augment.draw(256, augment.Strength(1.0, geometry=0.0, degrade=0.0), generator(), "cpu")
    assert bool((levels_only["zoom"] == 1).all()) and bool((levels_only["angle"] == 0).all()) and bool((levels_only["shift_x"] == 0).all())
    assert float(levels_only["blur"].sum() + levels_only["speckle"].sum() + levels_only["additive"].sum() + levels_only["border"].sum()) == 0.0
    assert float(levels_only["gain"].std()) > 0.05 and float(levels_only["tone"].max()) > 0.1
    geometry_only = augment.draw(256, augment.Strength(1.0, levels=0.0, degrade=0.0), generator(), "cpu")
    assert bool((geometry_only["gain"] == 1).all()) and bool((geometry_only["gamma"] == 1).all()) and bool((geometry_only["tone"] == 0).all())
    assert bool((geometry_only["tint"] == 1).all()) and float(geometry_only["zoom"].std()) > 0.05
    frames = torch.from_numpy(blocky(3, 2))
    plain = augment.augment(frames, torch.zeros((2, 16), dtype=torch.long), augment.Strength(1.0, geometry=0.0, levels=0.0, degrade=0.0), generator())
    untouched = augment.augment(frames, torch.zeros((2, 16), dtype=torch.long), augment.Strength(0.0), generator())
    assert torch.allclose(plain, untouched, atol=1e-5)


def test_strength_zero_leaves_no_random_change_whatever_the_generator_draws():
    frames, fragment = torch.from_numpy(blocky(5, 8)), torch.zeros((8, 16), dtype=torch.long)
    fragment[:, 8:] = 1  # two fragments in every window
    off = augment.Strength(0.0, shared=1.0)
    first = augment.augment(frames, fragment, off, torch.Generator().manual_seed(1), group=4)
    second = augment.augment(frames, fragment, off, torch.Generator().manual_seed(2), group=4)
    assert torch.equal(first, second)  # two generators, one input: the switch is off for every change
    on = augment.augment(frames, fragment, augment.Strength(1.0, shared=1.0), torch.Generator().manual_seed(1), group=4)
    assert not torch.allclose(first, on, atol=1e-3)


def test_a_mirrored_window_is_the_plain_window_seen_from_the_other_side():
    frames, fragment = torch.from_numpy(blocky(6, 4)), torch.zeros((4, 16), dtype=torch.long)
    for strength in (augment.Strength(0.0), augment.Strength(1.0, levels=0.0, degrade=0.0)):  # the plain input, and random geometry
        params = augment.draw(4, strength, torch.Generator().manual_seed(3), "cpu")
        plain = augment.warp(frames, fragment, params)
        every = augment.warp(frames, fragment, params, torch.ones(4, dtype=torch.bool))
        assert torch.allclose(every, torch.flip(plain, dims=[-1]), atol=1e-5) and not torch.allclose(every, plain, atol=1e-2)
        some = augment.warp(frames, fragment, params, torch.tensor([True, False, False, True]))
        assert torch.allclose(some[0], torch.flip(plain[0], dims=[-1]), atol=1e-5) and torch.allclose(some[3], torch.flip(plain[3], dims=[-1]), atol=1e-5)
        assert torch.equal(some[1], plain[1]) and torch.equal(some[2], plain[2])  # a window that is not mirrored is the window it was, bit for bit
        assert torch.equal(augment.warp(frames, fragment, params, torch.zeros(4, dtype=torch.bool)), plain)
    # through the whole augmentation: the mirror changes nothing else, and without it the output is the published one
    full = augment.Strength(1.0, shared=1.0)
    published = augment.augment(frames, fragment, full, torch.Generator().manual_seed(8), group=4)
    assert torch.equal(augment.augment(frames, fragment, full, torch.Generator().manual_seed(8), group=4, mirror=None), published)
    assert torch.equal(augment.augment(frames, fragment, full, torch.Generator().manual_seed(8), group=4, mirror=torch.zeros(4, dtype=torch.bool)), published)
    # a mirrored group takes no number more from the generator of the other input changes: the next draw is the one it would have been
    states = []
    for mirror in (None, torch.tensor([True, False, False, True]), torch.ones(4, dtype=torch.bool)):
        generator = torch.Generator().manual_seed(8)
        out = augment.augment(frames, fragment, full, generator, group=4, mirror=mirror)
        states.append((generator.get_state().clone(), out))
    assert torch.equal(states[0][0], states[1][0]) and torch.equal(states[0][0], states[2][0])
    assert torch.equal(states[1][1][1], states[0][1][1]) and not torch.equal(states[1][1][0], states[0][1][0])  # only the mirrored windows differ


def test_the_mirror_is_drawn_from_a_stream_of_its_own():
    torch.manual_seed(5)  # the generator of the whole program, from which the network draws its dropout
    before = torch.get_rng_state().clone()
    drawn = augment.mirrored(seed=1, update=7, groups=8, probability=0.5)
    assert torch.equal(torch.get_rng_state(), before) and drawn.shape == (8,) and drawn.dtype == torch.bool
    # the seed and the update each count: seed 1 at update 7 is not seed 2 at update 6, though their numbers add up alike
    pairs = [(seed, update) for seed in range(3) for update in range(40)]
    assert len({tuple(augment.mirrored(seed, update, 64, 0.5).tolist()) for seed, update in pairs}) == len(pairs)
    assert torch.equal(drawn, augment.mirrored(1, 7, 8, 0.5))  # the same seed and update give the same groups: a run that goes on draws them again
    updates = torch.stack([augment.mirrored(1, update, 8, 0.5) for update in range(200)])
    assert 0.45 < updates.float().mean() < 0.55 and len({tuple(row.tolist()) for row in updates}) > 100
    assert not torch.equal(updates, torch.stack([augment.mirrored(2, update, 8, 0.5) for update in range(200)]))  # another seed, other groups
    assert not augment.mirrored(1, 7, 8, 0.0).any() and augment.mirrored(1, 7, 8, 1.0).all()
    assert augment.MIRROR_STREAM not in (train.SAMPLING_STREAM, train.SAMPLING_STREAM + 1, train.SAMPLING_STREAM + 2)


def test_group_draw_repeats_parameters():
    params = augment.draw_groups(16, 4, augment.Strength(1.0, shared=1.0), torch.Generator().manual_seed(4), "cpu")
    assert params["gain"].shape == (16, augment.MAX_FRAGMENTS)
    assert torch.equal(params["gain"][0], params["gain"][3]) and not torch.equal(params["gain"][0], params["gain"][4])
    assert torch.equal(params["border"][4], params["border"][7])
    with pytest.raises(ValueError):
        augment.draw_groups(10, 4, augment.Strength(1.0), torch.Generator().manual_seed(4), "cpu")


# ------------------------------------------------------------------------------------------- fast forward
def test_direct_forward_equals_hook_forward():
    torch.manual_seed(0)
    model = train.initial_model(None, 0)
    torch.nn.init.normal_(model.marker_stem.weight, std=0.05)  # a non-zero marker stem, or the marker path is untested
    model.eval()
    rgb, marker = torch.randn(2, 3, 16, 224, 224), md.marker_planes(torch.tensor([3, 8]))
    with torch.no_grad():
        reference, direct = model(rgb, marker), fast.Direct(model)(rgb, marker)
        unmarked = fast.Direct(model)(rgb, torch.zeros_like(marker))
    assert torch.allclose(reference, direct, atol=1e-5)
    assert not torch.allclose(reference, unmarked, atol=1e-4)


def test_average_tracks_parameters():
    model = torch.nn.Linear(4, 2)
    average = fast.Average(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    average.update(0.5)
    assert torch.allclose(average.mean[0], model.weight - 0.5)
    average.update(0.0)
    assert torch.allclose(average.mean[0], model.weight)
    state = average.state_dict(model)
    assert set(state) == set(model.state_dict()) and torch.allclose(state["weight"], model.weight)


# ------------------------------------------------------------------------------------------------ helpers
def test_learning_rate_warms_up_then_follows_a_cosine():
    assert train.lr_factor(0, 10, 100) == 0.1 and train.lr_factor(9, 10, 100) == 1.0 and abs(train.lr_factor(99, 10, 100)) < 1e-3


def test_neighbour_penalty_matches_its_definition():
    same = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]]).repeat(8, 1)
    assert float(train.neighbour_penalty(same)) == 0.0
    logits = torch.zeros(4, 5)
    logits[2, 0] = 3.0  # one window of the group differs
    ell = torch.log_softmax(logits, -1)
    want = ((ell[2] - ell[1]) ** 2).sum() + ((ell[3] - ell[2]) ** 2).sum()
    assert abs(float(train.neighbour_penalty(logits)) - float(want) / 15) < 1e-6
    huge = torch.zeros(4, 5)
    huge[1, 0] = 100.0
    assert float(train.neighbour_penalty(huge)) <= 16.0 * 2 * 5 / 15 + 1e-6  # every term is cut at 16
    probe = torch.zeros(4, 5, requires_grad=True)
    train.neighbour_penalty(probe + logits).backward()
    assert probe.grad is not None and torch.isfinite(probe.grad).all()


def test_a_model_starts_with_a_silent_marker_and_a_seeded_head():
    first, again, other = train.initial_model(None, 3), train.initial_model(None, 3), train.initial_model(None, 4)
    assert float(first.marker_stem.weight.abs().sum()) == 0.0 and float(first.classifier.bias.abs().sum()) == 0.0
    assert torch.equal(first.classifier.weight, again.classifier.weight) and not torch.equal(first.classifier.weight, other.classifier.weight)
    assert 0.015 < float(first.classifier.weight.std()) < 0.025
    assert len(first.new_parameters()) == 3
    first.set_encoder_trainable(False)
    assert not any(p.requires_grad for p in first.encoder.parameters()) and all(p.requires_grad for p in first.new_parameters())


def test_a_checkpoint_is_loaded_whole_and_the_encoder_start_needs_its_file(tmp_path):
    source = train.initial_model(None, 1)
    torch.nn.init.normal_(source.marker_stem.weight, std=0.1)
    torch.save(source.state_dict(), tmp_path / "start.pt")
    loaded = train.build_model(tmp_path / "start.pt", None, seed=9)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in source.state_dict().items())
    with pytest.raises(SystemExit, match="--encoder"):
        train.build_model("encoder", None, seed=0)


def test_the_stage_3_recipe_is_read_and_the_command_line_wins():
    recipe = Path(__file__).resolve().parents[1] / "configs" / "stage3.json"
    args = train.parse_args(["--config", str(recipe), "--cache", "c", "--out", "o", "--init", "start.pt", "--batch", "16"])
    assert (args.objective, args.updates, args.lr_enc, args.warmup, args.freeze, args.frames) == ("within_view", 1500, 1e-5, 100, 0, "host")
    assert args.batch == 16 and args.plans is None and args.seed == 0
    assert [p.name for p in train.packaged_plans("within_view")] == ["groups_s0.npz", "groups_s1.npz", "groups_s2.npz"]
    assert all(p.is_file() for p in train.packaged_plans("ce"))


def test_an_unknown_option_in_a_recipe_is_refused(tmp_path):
    (tmp_path / "bad.json").write_text(json.dumps({"updates": 10, "learning_rate": 1.0}))
    with pytest.raises(SystemExit, match="learning_rate"):
        train.parse_args(["--config", str(tmp_path / "bad.json"), "--cache", "c", "--out", "o", "--init", "x"])


# ------------------------------------------------------------------------------------- cache and the plans
def test_fragment_numbers_and_native_window():
    vid = np.array([[3] * 16, [3] * 6 + [9] * 10, [1] * 4 + [2] * 8 + [1] * 4])
    f = plans.fragment_numbers(vid)
    assert f[0].tolist() == [0] * 16 and f[1].tolist() == [0] * 6 + [1] * 10 and f[2].tolist() == [0] * 4 + [1] * 8 + [2] * 4
    rows, marker = plans.native_window(100, 40, 20)
    assert rows.tolist() == list(range(112, 128)) and marker == 8
    rows, marker = plans.native_window(100, 40, 39)
    assert rows.tolist() == list(range(124, 140)) and marker == 15
    rows, marker = plans.native_window(50, 9, 4)
    assert len(rows) == 16 and rows.min() == 50 and rows.max() == 58 and rows[marker] == 54
    with pytest.raises(ValueError):
        plans.native_window(0, 9, 9)


NAMES = ("rec_a", "rec_b", "rec_c", "rec_d")


@pytest.fixture()
def tiny_cache(tmp_path) -> cache.Cache:
    """Four recordings of 60 source frames (20 samples): two for training, one per development half. Every frame is
    filled with 10 * recording + sample, so a test can tell where a row came from."""
    images = tmp_path / "Images"
    items = []
    for r, (name, split) in enumerate(zip(NAMES, ("train", "train", "selection", "calibration"))):
        (images / name).mkdir(parents=True)
        for k in range(20):
            Image.fromarray(np.full((240, 320, 3), 10 * r + k, np.uint8)).save(images / name / f"{3 * k}.png")
            (images / name / f"{3 * k}.png").rename(images / name / f"{3 * k}.jpg")
        items.append({"key": recording_key(name), "split": split, "label": r % 5, "n_frames": 60})
    index = cache.build(images, tmp_path / "cache", workers=1, items=items)
    assert index["total"] == 80 and [c["start"] for c in index["clips"]] == [0, 20, 40, 60]
    return cache.Cache(tmp_path / "cache")


def test_the_cache_holds_every_tenth_of_a_second_in_grey(tiny_cache):
    assert tiny_cache.frames.shape == (80, 240, 320) and tiny_cache.train_end == 40
    assert int(tiny_cache.frames[0, 0, 0]) == 0 and int(tiny_cache.frames[25, 100, 100]) == 15 and int(tiny_cache.frames[79, 0, 0]) == 49
    assert [c["split"] for c in tiny_cache.rows("train")] == ["train", "train"]


def test_the_packaged_list_names_the_published_split():
    items = cache.recordings()
    assert {split: sum(i["split"] == split for i in items) for split in cache.SPLITS} == {"train": 3364, "selection": 284, "calibration": 237}
    rows = cache.layout(items)
    assert rows[-1]["start"] + rows[-1]["count"] == 238572 and len({i["key"] for i in items}) == len(items)


def write_plan(path: Path, groups: int = 4) -> Path:
    """A plan of `groups` pairs of groups over the two training recordings: one group inside rec_a, one that changes
    from rec_a to rec_b in its last window."""
    vid = np.zeros((groups, 8, 16), np.int32)
    frame = np.tile(np.arange(16, dtype=np.int32) * 3, (groups, 8, 1))
    vid[:, 7, 8:] = 1
    kind = np.zeros((groups, 8), np.int8)
    kind[:, 7] = 1
    target = np.zeros((groups, 8), np.int8)
    target[:, 7] = 1
    np.savez_compressed(path, recordings=np.array([recording_key(n) for n in NAMES[:2]]), vid=vid, frame=frame,
                        gamma=np.ones((groups, 8, 16), np.float32), marker=np.full((groups, 8), 8, np.int8), target=target, kind=kind)
    return path


def test_group_batches_are_whole_groups_of_training_rows(tiny_cache, tmp_path):
    pool = plans.GroupPool(tiny_cache, [write_plan(tmp_path / "a.npz"), write_plan(tmp_path / "b.npz")])
    assert pool.rows.shape == (16, 4, 16) and pool.train_end == 40
    rows, marker, target, fragment = pool.batch(np.random.default_rng(0), 8)
    assert rows.shape == (8, 16) and marker.tolist() == [8] * 8 and rows.max() < 40
    crossing = np.flatnonzero(fragment.max(1) > 0)
    assert all(rows[i, 8] >= 20 and target[i] == 1 for i in crossing)  # the second fragment is rec_b
    with pytest.raises(ValueError, match="whole groups"):
        pool.batch(np.random.default_rng(0), 6)


def test_single_window_batches_mix_fresh_and_planned_windows(tiny_cache, tmp_path):
    pool = plans.Pool(tiny_cache, [write_plan(tmp_path / "a.npz")], classes=2)  # the tiny cache trains on two views
    assert len(pool.rows) == 4  # the mixed windows of the plan
    rows, marker, target, fragment = pool.batch(np.random.default_rng(1), 8, native_share=0.5)
    assert rows.shape == (8, 16) and fragment[:4].max() == 0 and fragment[4:].max() == 1
    assert set(target[:4].tolist()) <= {0, 1} and all(rows[i, marker[i]] < 40 for i in range(8))


def test_a_plan_that_reads_outside_the_training_recordings_is_refused(tiny_cache, tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, recordings=np.array([recording_key("rec_c")]), vid=np.zeros((1, 8, 16), np.int32),
                        frame=np.tile(np.arange(16, dtype=np.int32) * 3, (1, 8, 1)), gamma=np.ones((1, 8, 16), np.float32),
                        marker=np.full((1, 8), 8, np.int8), target=np.zeros((1, 8), np.int8), kind=np.zeros((1, 8), np.int8))
    with pytest.raises(ValueError, match="outside the training recordings"):
        plans.GroupPool(tiny_cache, [path])
    np.savez_compressed(path, recordings=np.array(["0000000000000000"]), vid=np.zeros((1, 8, 16), np.int32),
                        frame=np.zeros((1, 8, 16), np.int32), gamma=np.ones((1, 8, 16), np.float32),
                        marker=np.full((1, 8), 8, np.int8), target=np.zeros((1, 8), np.int8), kind=np.zeros((1, 8), np.int8))
    with pytest.raises(ValueError, match="outside the frame cache"):
        plans.GroupPool(tiny_cache, [path])
