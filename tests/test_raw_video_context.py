"""CPU checks of the raw-video context round (configs/experiments/raw_video_context_v1.yaml): target slots and markers,
the three input conditions and their permutations, pixel digests shared by preparation and training, the model's
marker branch and hook, the learning-rate schedule, check-window selection and the queue's budget projection. The
EchoPrime checkpoint is not needed: the encoder is randomly initialised here."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

import echo_routing.vit.target_mvit as tm
from echo_routing.evaluate.window_target import window_frames

ROOT = Path(__file__).resolve().parents[1]
RV = yaml.safe_load((ROOT / "configs/experiments/raw_video_context_v1.yaml").read_text())["raw_video_context"]


def test_target_slots_and_marker_follow_s_and_c_only():
    s = window_frames(3, 10, 16, 16)                       # a 10-sample recording: repeated samples
    slots = tm.target_slots(s, 3)
    assert slots.size >= 1 and all(s[i] == 3 for i in slots) and tm.marker_slot(s, 3) == int(slots[slots.size // 2])
    two = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]); assert tm.marker_slot(two, 2) == 5   # [4, 5] -> index 1
    assert tm.marker_slot(window_frames(0, 100, 16, 16), 0) == 0 and tm.marker_slot(window_frames(99, 100, 16, 16), 99) == 15
    assert tm.marker_slot(window_frames(50, 100, 16, 16), 50) == 8
    with pytest.raises(ValueError):
        tm.target_slots(np.arange(16), 40)


def test_shuffle_keeps_target_slots_and_uses_the_fixed_rngs():
    s = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]); c = 3
    o1 = tm.shuffle_order(tm.train_rng(1, 7, 2), s, c); o2 = tm.shuffle_order(np.random.default_rng([20261010, 1, 7, 2]), s, c)
    assert np.array_equal(o1, o2) and sorted(o1.tolist()) == list(range(16)) and o1[6] == 6 and o1[7] == 7
    other = np.flatnonzero(s != c); ref = np.arange(16); ref[other] = other[np.random.default_rng([20261010, 1, 7, 2]).permutation(other.size)]
    assert np.array_equal(o1, ref)
    seed = int.from_bytes(hashlib.sha256(b"2|seq-x|45").digest()[:8], "big")
    assert np.array_equal(tm.shuffle_order(tm.inference_rng(2, "seq-x", 45), s, c), tm.shuffle_order(np.random.default_rng(seed), s, c))
    assert not np.array_equal(tm.shuffle_order(tm.inference_rng(2, "seq-x", 45), np.arange(16), 8), tm.shuffle_order(tm.inference_rng(1, "seq-x", 45), np.arange(16), 8))


def test_group_orders():
    s = np.arange(16) + 10; c = 18; sh = tm.shuffle_order(tm.train_rng(0, 0, 0), s, c)
    assert np.array_equal(tm.group_order("Ordered context", s, c, sh), np.arange(16))
    assert np.array_equal(tm.group_order("Shuffled context", s, c, sh), sh) and sh[8] == 8
    assert np.array_equal(tm.group_order("Target-only", s, c, sh), np.full(16, 8))
    bad = sh.copy(); i = int(np.flatnonzero(bad != 8)[0]); bad[8], bad[i] = bad[i], 8
    with pytest.raises(ValueError):
        tm.group_order("Shuffled context", s, c, bad)


def test_marker_planes_and_rgb_normalisation():
    m = tm.marker_planes(torch.tensor([0, 15, 8]), size=8)
    assert m.shape == (3, 1, 16, 8, 8) and float(m.sum()) == 3 * 64
    assert all(float(m[b, 0, k].sum()) == 64 for b, k in enumerate((0, 15, 8)))
    with pytest.raises(ValueError):
        tm.marker_planes(torch.tensor([16]))
    u8 = torch.zeros(1, 2, 4, 4, 3, dtype=torch.uint8); u8[..., 1] = 255
    x = tm.to_rgb(u8)
    assert x.shape == (1, 3, 2, 4, 4) and torch.allclose(x[0, 0], torch.full((2, 4, 4), -tm.MEAN[0] / tm.STD[0]))
    assert torch.allclose(x[0, 1], torch.full((2, 4, 4), (255 - tm.MEAN[1]) / tm.STD[1]))
    with pytest.raises(ValueError):
        tm.to_rgb(u8.float())


def test_initial_state_is_per_seed_and_shared():
    a, b, c = tm.initial_new_state(0), tm.initial_new_state(0), tm.initial_new_state(1)
    assert torch.equal(a["classifier"]["weight"], b["classifier"]["weight"]) and not torch.equal(a["classifier"]["weight"], c["classifier"]["weight"])
    assert abs(float(a["classifier"]["weight"].std()) - 0.02) < 0.003 and float(a["classifier"]["bias"].abs().max()) == 0
    assert a["marker_stem"]["weight"].shape == (96, 1, 3, 7, 7) and float(a["marker_stem"]["weight"].abs().max()) == 0


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tm.build_model(None, tm.initial_new_state(0)).eval()


def test_zero_marker_branch_equals_the_encoder_and_the_hook_never_stays(model):
    rgb = torch.randn(1, 3, 16, 224, 224); mk = tm.marker_planes(torch.tensor([3]))
    with torch.no_grad():
        a = model.encoder(rgb); b = model.embed(rgb, mk)
    assert torch.equal(a, b) and not model.encoder.conv_proj._forward_hooks
    blk = model.encoder.blocks[0]
    blk.forward = lambda *x, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        with pytest.raises(RuntimeError):
            model(rgb, mk)
    finally:
        del blk.forward
    assert not model.encoder.conv_proj._forward_hooks
    with torch.no_grad():
        assert torch.equal(model.encoder(rgb), a)
    assert list(__import__("inspect").signature(model.forward).parameters) == ["rgb", "marker"]


def test_frozen_encoder_still_passes_gradient_to_the_marker(model):
    model.set_encoder_trainable(False); model.zero_grad(set_to_none=True)
    rgb = torch.randn(1, 3, 16, 224, 224)
    loss = torch.nn.functional.cross_entropy(model(rgb, tm.marker_planes(torch.tensor([8]))), torch.tensor([2]))
    loss.backward()
    assert float(model.marker_stem.weight.grad.abs().max()) > 0 and model.classifier.weight.grad is not None
    assert all(p.grad is None for p in model.encoder_parameters())
    model.set_encoder_trainable(True)


def test_schedule_freezes_the_encoder_for_the_warmup():
    from scripts.train_raw_video_context import schedule
    o = RV["training"]["optimiser"]
    assert schedule(0, 375, 3750, o) == (0.0, pytest.approx(1e-4 / 375))
    assert schedule(374, 375, 3750, o) == (0.0, pytest.approx(1e-4))
    assert schedule(375, 375, 3750, o) == (pytest.approx(1e-5), pytest.approx(1e-4))
    enc, new = schedule(3749, 375, 3750, o); assert 0 < enc < 1e-11 and 0 < new < 1e-10 and new == pytest.approx(10 * enc)


def _jpeg(path: Path, value: int) -> None:
    arr = np.full((240, 320, 3), value, np.uint8); arr[::7, ::5] = (value * 3) % 256
    Image.fromarray(arr).save(path, quality=95)


def _toy_manifest(tmp_path: Path) -> tuple[dict, dict]:
    """Two recordings of 20 frames, one update of two windows (native c=0 and c=10)."""
    dirs = {}
    for v, base in (("rec_a", 10), ("rec_b", 120)):
        d = tmp_path / v; d.mkdir()
        for i in range(20):
            _jpeg(d / f"{i}.jpg", base + 3 * i)
        dirs[v] = str(d)
    s = np.stack([window_frames(0, 20, 16, 16), window_frames(10, 20, 16, 16)])[None]
    c = np.array([[0, 10]]); m = {"videos": np.array(["rec_a", "rec_b"]), "vid": np.stack([np.zeros((2, 16), np.int32) + [[0], [1]]]),
                                  "frame": s.astype(np.int32), "gamma": np.ones((1, 2, 16)), "s": s.astype(np.int32), "c": c.astype(np.int32),
                                  "target": np.array([[2, 3]], np.int8), "kind": np.array([[0, 1]], np.int8), "seq": np.array([[-1, 4]], np.int32),
                                  "repeats": np.array([[0, 0]], np.int8)}
    return m, dirs


def test_prepared_digests_equal_the_decoded_training_inputs(tmp_path):
    import scripts.prepare_raw_video_context as pr
    import scripts.train_raw_video_context as tr
    from scripts.train_joint_visual_temporal import FramePaths
    m, dirs = _toy_manifest(tmp_path)
    marker, n_t, orders = pr.window_orders(0, m["s"], m["c"])
    assert marker.tolist() == [[0, 8]] and n_t.tolist() == [[1, 1]]
    refs = {(str(m["videos"][v]), int(f), 1.0) for v, f in zip(m["vid"].ravel(), m["frame"].ravel())}
    hashes = pr.pixel_hashes(refs, dirs, workers=1); dig = pr.group_digests(m, orders, hashes)
    m |= {"marker": marker, "n_target_slots": n_t} | {f"order_{tm.KEYS[g]}": orders[g] for g in tm.GROUPS} | {f"digest_{tm.KEYS[g]}": dig[g] for g in tm.GROUPS}
    fp = FramePaths(dirs); out = {}
    for g in tm.GROUPS:
        u8, slots, target, d = tr.window_inputs(m, m[f"order_{tm.KEYS[g]}"], fp, [(0, 0), (0, 1)])
        tr.check_digests(m, g, 0, d); out[g] = u8.numpy()
        assert slots.tolist() == [0, 8] and target.tolist() == [2, 3]
    o, sh, t = out["Ordered context"], out["Shuffled context"], out["Target-only"]
    for k, mk in enumerate((0, 8)):
        ho = sorted(tm.frame_hash(x) for x in o[k]); hs = sorted(tm.frame_hash(x) for x in sh[k])
        assert ho == hs and np.array_equal(sh[k, mk], o[k, mk]) and all(np.array_equal(t[k, i], o[k, mk]) for i in range(16))
    assert not np.array_equal(o[1], sh[1])
    bad = dict(m); bad["digest_ordered"] = m["digest_ordered"].copy(); bad["digest_ordered"][0, 0, 0] ^= 1
    with pytest.raises(SystemExit):
        tr.check_digests(bad, "Ordered context", 0, torch.from_numpy(np.stack([m["digest_ordered"][0, 0], m["digest_ordered"][0, 1]])))


def test_inputs_ignore_labels_names_and_native_flags(tmp_path):
    import scripts.prepare_raw_video_context as pr
    import scripts.train_raw_video_context as tr
    from scripts.train_joint_visual_temporal import FramePaths
    m, dirs = _toy_manifest(tmp_path); marker, n_t, orders = pr.window_orders(0, m["s"], m["c"])
    m |= {"marker": marker, "n_target_slots": n_t}
    a = tr.window_inputs(m, orders["Shuffled context"], FramePaths(dirs), [(0, 0), (0, 1)])
    m2 = dict(m, videos=np.array(["x1", "x2"]), kind=1 - m["kind"], target=(m["target"] + 1) % 5)
    b = tr.window_inputs(m2, orders["Shuffled context"], FramePaths({"x1": dirs["rec_a"], "x2": dirs["rec_b"]}), [(0, 0), (0, 1)])
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[3], b[3]) and not torch.equal(a[2], b[2])


def test_check_windows_use_manifest_order_and_coverage_only():
    from scripts.check_raw_video_context import check_windows
    u = 6; kind = np.tile([0, 1], (u, 4)).astype(np.int8); mk = np.full((u, 8), 8, np.int8); nt = np.ones((u, 8), np.int8); rep = np.zeros((u, 8), np.int8)
    rep[1, 2] = 3; mk[2, 0] = 0; mk[2, 2] = 15; mk[3, 1] = 0; mk[3, 3] = 15; nt[4, 5] = 2
    out = check_windows({"kind": kind, "marker": mk, "n_target_slots": nt, "repeats": rep})
    assert [(c["u"], c["w"]) for c in out[:8]] == [(1, 2), (2, 0), (2, 2), (3, 1), (3, 3), (4, 5), (0, 0), (0, 1)]
    assert len(out) == 16 and len({(c["u"], c["w"]) for c in out}) == 16 and [(c["u"], c["w"]) for c in out[8:]] == [(0, w) for w in range(2, 8)] + [(1, 0), (1, 1)]


def test_queue_projection_and_order():
    from scripts.queue_raw_video_context import projection
    order = RV["queue"]
    assert order == [f"{g}_s{s}" for s in (0, 1, 2) for g in ("target_only", "shuffled", "ordered")]
    assert projection(order, {}, 0.3, 0.5, 2.0) == pytest.approx(0.3 + 9 * 0.5 + 2.0)
    assert projection(order, {"target_only_s0": 0.6, "shuffled_s0": 0.4}, 0.3, 0.6, 2.0) == pytest.approx(0.3 + 1.0 + 7 * 0.6 + 2.0)


def test_config_fixes_the_plan_values():
    tr_ = RV["training"]; o = tr_["optimiser"]
    assert tr_["total_updates"] == 3750 and tr_["warmup_updates"] == 375 and tr_["batch_windows"] == 8 and tr_["seeds"] == [0, 1, 2]
    assert (o["lr_encoder"], o["lr_new"], o["weight_decay"], o["grad_clip"]) == (1e-5, 1e-4, 1e-4, 1.0)
    assert RV["inputs"]["mean"] == list(tm.MEAN) and RV["inputs"]["std"] == list(tm.STD)
    assert RV["budget"]["gpu_hours_max"] == 24.0 and RV["checks"]["parity_max_abs"] == 1e-4
    g = yaml.safe_load((ROOT / "configs/experiments/model_development_v1.yaml").read_text())["decode_grids"]["b3"]
    e = RV["evaluation"]["b3"]
    assert g["half_window"] == e["half_window"] and g["threshold_quantile"] == e["quantile"] and g["nms_radius"] == [e["nms_radius"]]
    assert RV["goal"]["selection_min"] == 0.944409687184662 and RV["goal"]["calibration_min"] == 0.9821421709894332
