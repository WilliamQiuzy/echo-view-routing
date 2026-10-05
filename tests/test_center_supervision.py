"""Target-time supervision round (configs/experiments/ev9v_center_supervision_v1.yaml): the input contract of the
training windows and the model and loss checks of the plan (Tasks 1-2), on small synthetic sequences."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from echo_routing.config import CACHE_MARKER, FRAME_ORDER_VERSION, REPO_ROOT, require_current_cache
from echo_routing.errors import ConfigError
from echo_routing.ingest.frames import list_frame_paths
from echo_routing.vit.target_video import TargetTimeHead, TargetVideoModel, losses, window_target
from echo_routing.vit.target_windows import (Track, build_manifests, check_train_only, frame_shares, is_mixed, mixed_index,
                                             native_track, window)

CS = yaml.safe_load((REPO_ROOT / "configs/experiments/ev9v_center_supervision_v1.yaml").read_text())["center_supervision"]


def stream(seq_id, parts):
    """parts: [(video, label, n, gamma, clip_start)] -> a training-stream Track."""
    vids, frames, gam, labs, clip = [], [], [], [], []
    for v, lab, n, g, s0 in parts:
        vids += [v] * n; frames += list(range(3 * s0, 3 * (s0 + n), 3)); gam += [g] * n; labs += [lab] * n; clip += list(range(s0, s0 + n))
    return Track(seq_id, tuple(vids), np.array(frames), np.array(gam, float), np.array(labs), np.array(clip))


def test_frame_order_and_cache_version_are_enforced(tmp_path: Path):
    d = tmp_path / "clip"; d.mkdir()
    for k in (0, 1, 10, 2):
        (d / f"{k}.jpg").write_bytes(b"x")
    assert [p.stem for p in list_frame_paths(d)] == ["0", "1", "2", "10"]
    cache = tmp_path / "cache"; (cache / "features" / "k").mkdir(parents=True)
    with pytest.raises(ConfigError):
        require_current_cache(cache / "features" / "k")
    (cache / CACHE_MARKER).write_text("name-order\n")
    with pytest.raises(ConfigError):
        require_current_cache(cache / "features" / "k")
    (cache / CACHE_MARKER).write_text(FRAME_ORDER_VERSION + "\n")
    require_current_cache(cache / "features" / "k")


def test_training_windows_refuse_non_training_sources():
    check_train_only({"a_1", "b_2"}, {"a_1", "b_2", "c_3"})
    with pytest.raises(ValueError):
        check_train_only({"a_1", "val_9"}, {"a_1", "b_2"})


def test_targets_follow_the_actual_target_sample_at_edges_and_in_short_sequences():
    t = stream("x", [("a_1", 2, 20, 1.0, 0), ("b_1", 3, 20, 0.9, 5)])
    for c, rel in ((0, range(0, 16)), (39, range(-15, 1)), (20, range(-8, 8))):
        w = window(t, c)
        assert w["rel"].tolist() == list(rel) and w["target_label"] == t.labels[c] and len(w["frames"]) == 16
        assert w["labels"][w["rel"] == 0][0] == t.labels[c] and w["frames"][w["rel"] == 0][0] == t.frames[c]
    short = native_track("s_1", np.arange(0, 27, 3), 4)
    for c in range(9):
        w = window(short, c)
        assert c in w["s"].tolist() and w["repeats"] == 7 and np.abs(w["rel"]).max() <= 8 and w["target_label"] == 4
    with pytest.raises(IndexError):
        window(short, 9)


def test_s_and_c_targets_agree_on_one_view_and_shares_sum_to_one():
    one = torch.full((3, 16), 2); mixed = torch.tensor([[0] * 6 + [3] * 10, [3] * 15 + [1]])
    assert torch.equal(window_target("S", one, one[:, 7]), window_target("C", one, one[:, 7]))
    s = window_target("S", mixed, mixed[:, 7])
    assert torch.allclose(s.sum(1), torch.ones(2)) and torch.allclose(s, torch.tensor(frame_shares(mixed.numpy()), dtype=torch.float32))
    assert not torch.equal(s, window_target("C", mixed, mixed[:, 7])) and is_mixed(mixed[0].tolist()) and not is_mixed(one[0].tolist())


def test_manifests_pair_the_arms_and_keep_targets():
    natives = {k: [native_track(f"v{k}_{i}", np.arange(0, 3 * (30 + i), 3), k) for i in range(2)] for k in range(5)}
    by = {t.seq_id: t for k in natives for t in natives[k]}
    streams = []
    for j in range(6):
        a, b = j % 5, (j + 1) % 5
        streams.append(stream(f"m{j}", [(f"v{a}_0", a, 12, 0.9, 3), (f"v{b}_1", b, 14, 1.0, 1), (f"v{a}_1", a, 10, 1.0, 0)]))
    for t in streams:   # stream samples point at real native frames of their source recording
        assert all(by[v].frames[k] == f for v, k, f in zip(t.videos, t.clip_index, t.frames))
    mix = mixed_index(streams); assert sorted(mix) == list(range(5))
    sc, n_arm = build_manifests(0, 7, natives, streams, mix, 5)
    sc2, _ = build_manifests(0, 7, natives, streams, mix, 5)
    assert all(np.array_equal(sc[k], sc2[k]) for k in sc)
    assert sc["vid"].shape == (5, 8, 16) and (sc["rel"] == 0).any(-1).all()
    for key in ("frame", "gamma", "rel", "labels", "target", "c"):
        assert np.array_equal(sc[key][:, :4], n_arm[key][:, :4])
    assert (sc["kind"][:, 4:] == 1).all() and (n_arm["kind"][:, 4:] == 2).all()
    assert all(is_mixed(x) for x in sc["labels"][:, 4:].reshape(-1, 16))
    assert np.array_equal(sc["target"][:, 4:], n_arm["target"][:, 4:])
    assert np.array_equal(sc["videos"][sc["target_vid"][:, 4:]], n_arm["videos"][n_arm["target_vid"][:, 4:]])
    for m in (sc, n_arm):
        at0 = m["rel"][:, 4:] == 0
        tf = np.where(at0, m["frame"][:, 4:], -1).max(-1); tg = np.where(at0, m["gamma"][:, 4:], -1).max(-1)
        m["_tf"], m["_tg"] = tf, tg
    assert np.array_equal(sc["_tf"], n_arm["_tf"]) and np.allclose(sc["_tg"], n_arm["_tg"])
    assert all(not is_mixed(x) for x in n_arm["labels"].reshape(-1, 16))


def test_head_shapes_offsets_and_order():
    torch.manual_seed(0); head = TargetTimeHead(512, 2, 8, 2048, 0.1, 15).eval()
    emb = torch.randn(2, 16, 512); rel = torch.arange(-8, 8).repeat(2, 1)
    logits, frame_logits = head(emb, rel)
    assert logits.shape == (2, 5) and frame_logits.shape == (2, 16, 5)
    assert not torch.allclose(head(emb, rel + 3)[0], logits, atol=1e-5)          # moving the target changes the output
    perm = torch.randperm(16)
    assert torch.allclose(head(emb[:, perm], rel[:, perm])[0], logits, atol=1e-5)  # order of (token, offset) pairs does not
    with pytest.raises(ValueError):
        head(emb, rel * 3)
    with pytest.raises(TypeError):
        head(emb, rel.float())

    class Stub(nn.Module):
        def forward(self, u8):
            return u8.float().mean((1, 2, 3))[:, None].expand(-1, 512)
    model = TargetVideoModel(Stub(), head)
    out = model(torch.zeros(32, 4, 4, 3, dtype=torch.uint8), rel)
    assert out[0].shape == (2, 5)
    with pytest.raises(ValueError):
        model(torch.zeros(30, 4, 4, 3, dtype=torch.uint8), rel)
    probe = {"weight": torch.randn(5, 512), "bias": torch.randn(5)}; head.init_from_probe(probe)
    assert torch.equal(head.head.weight, probe["weight"]) and torch.equal(head.frame_head.bias, probe["bias"])


def test_loss_formula_and_s_c_pairing():
    torch.manual_seed(1)
    logits = torch.randn(3, 5, requires_grad=True); frame_logits = torch.randn(3, 16, 5, requires_grad=True)
    labels = torch.tensor([[1] * 16, [4] * 16, [2] * 10 + [3] * 6]); target = labels[:, 12]
    loss_c, parts = losses("C", logits, frame_logits, labels, target, CS["loss"]["aux_weight"])
    main = -(torch.nn.functional.one_hot(target, 5) * torch.log_softmax(logits, -1)).sum(-1).mean()
    aux = torch.nn.functional.cross_entropy(frame_logits.reshape(-1, 5), labels.reshape(-1))
    assert torch.allclose(loss_c, main + 0.5 * aux) and abs(parts["main"] - float(main)) < 1e-6
    single = slice(0, 2)
    gs = torch.autograd.grad(losses("S", logits[single], frame_logits[single], labels[single], target[single], 0.5)[0], [logits, frame_logits])
    gc = torch.autograd.grad(losses("C", logits[single], frame_logits[single], labels[single], target[single], 0.5)[0], [logits, frame_logits])
    assert all(torch.equal(a, b) for a, b in zip(gs, gc))
    loss_s, _ = losses("S", logits, frame_logits, labels, target, 0.5)
    shares = torch.tensor([[0, 1, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 10 / 16, 6 / 16, 0]])
    assert torch.allclose(loss_s, -(shares * torch.log_softmax(logits, -1)).sum(-1).mean() + 0.5 * aux)
    with pytest.raises(ValueError):
        window_target("X", labels, target)
