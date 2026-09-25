from __future__ import annotations

import numpy as np
import pytest

from echo_routing.errors import CacheMissError, FrameError
from echo_routing.features.cache import VideoFeatures
from echo_routing.vit.clips import clip_indices, max_start, pad_window, random_clip, spaced_clips
from echo_routing.vit.gpu_frames import gamma_lut, letterbox_geometry
from echo_routing.vit.infer import assemble, sample_rows, variant_of, windowed_probs, windows
from echo_routing.vit.optim import param_groups, vit_layer_id, warmup_cosine

torch = pytest.importorskip("torch")


def vf(video_id, n, d=3):
    feat = np.arange(n * d, dtype=float).reshape(n, d)
    return VideoFeatures(video_id, np.arange(n) * 3, np.arange(n) / 10, feat, np.full((n, 5), 0.2), np.zeros((n, 5)))


# ---- clips -------------------------------------------------------------------------------------------------------

def test_clip_indices_clamp_short_cine():
    assert clip_indices(100, 4, 3, 10).tolist() == [10, 13, 16, 19]
    assert clip_indices(7, 4, 3, 0).tolist() == [0, 3, 6, 6]   # repeats last frame, never loops
    with pytest.raises(FrameError):
        clip_indices(10, 4, 3, 10)


def test_random_and_spaced_clips_stay_inside():
    rng = np.random.default_rng(0)
    for n in (25, 48, 200):
        assert max_start(n, 16, 3) == max(0, n - 46)
        for _ in range(20):
            c = random_clip(n, 16, 3, rng)
            assert c.size == 16 and c.min() >= 0 and c.max() <= n - 1
    cl = spaced_clips(200, 16, 3, 3)
    assert [c[0] for c in cl] == [0, 77, 154] and all(c[-1] <= 199 for c in cl)
    assert len(spaced_clips(30, 16, 3, 3)) == 1                 # short cine: a single clip


def test_pad_window():
    assert pad_window(np.array([4, 5, 6]), 5).tolist() == [4, 5, 6, 6, 6]
    assert pad_window(np.arange(20), 16).tolist() == list(range(16))
    with pytest.raises(FrameError):
        pad_window(np.array([], dtype=int), 4)


# ---- preprocessing geometry --------------------------------------------------------------------------------------

def test_letterbox_geometry_matches_pil_pad():
    from PIL import Image, ImageOps
    for w, h in ((320, 240), (240, 320), (300, 300), (640, 181)):
        im = Image.new("RGB", (w, h), (255, 255, 255))
        out = np.asarray(ImageOps.pad(im, (224, 224), color=(0, 0, 0), method=Image.BILINEAR))[..., 0]
        nh, nw, top, left = letterbox_geometry(h, w, 224)
        rows, cols = np.where(out > 128)
        assert (rows.min(), rows.max() + 1, cols.min(), cols.max() + 1) == (top, top + nh, left, left + nw)


def test_gamma_lut_matches_apply_gamma():
    from PIL import Image
    from echo_routing.ingest.frames import apply_gamma
    ramp = Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (2, 1))).convert("RGB")
    ref = np.asarray(apply_gamma(ramp, 0.9))[0, :, 0]
    assert gamma_lut(0.9).numpy().tolist() == ref.tolist()
    assert gamma_lut(1.0).numpy().tolist() == list(range(256))


def test_cpu_decoder_letterbox_and_gamma(tmp_path):
    from PIL import Image
    from echo_routing.vit.gpu_frames import IMAGENET, GpuDecoder, read_bytes
    p = tmp_path / "f.jpg"
    Image.new("RGB", (320, 240), (128, 128, 128)).save(p, quality=95)
    dec = GpuDecoder(224, IMAGENET, torch.device("cpu"))
    x = dec([read_bytes(p), read_bytes(p)], [1.0, 0.9])
    assert x.shape == (2, 3, 224, 224)
    pad_val = (0 - 0.485) / 0.229
    assert torch.allclose(x[0, 0, 0, 0], torch.tensor(pad_val), atol=1e-4)            # black letterbox band
    assert x[1, 0, 112, 112] > x[0, 0, 112, 112]                                        # gamma 0.9 brightens mid-grey


# ---- optimiser ---------------------------------------------------------------------------------------------------

def test_vit_layer_ids_and_decay():
    assert vit_layer_id("vit.patch_embed.proj.weight", "vit", 12) == 0
    assert vit_layer_id("vit.cls_token", "vit", 12) == 0
    assert vit_layer_id("vit.blocks.0.attn.qkv.weight", "vit", 12) == 1
    assert vit_layer_id("vit.blocks.11.mlp.fc1.weight", "vit", 12) == 12
    assert vit_layer_id("vit.norm.weight", "vit", 12) == 13
    assert vit_layer_id("head.weight", "vit", 12) == 13
    net = torch.nn.ModuleDict({"vit": torch.nn.ModuleDict({"blocks": torch.nn.ModuleList([torch.nn.Linear(2, 2)])}),
                               "head": torch.nn.Linear(2, 2)})
    groups = param_groups(net, 1e-3, 0.05, 0.5, "vit", 1)
    by = {(g["layer_id"], g["weight_decay"]): g for g in groups}
    assert by[(1, 0.05)]["lr"] == pytest.approx(1e-3 * 0.5)       # block 0 weight
    assert by[(1, 0.0)]["lr"] == pytest.approx(1e-3 * 0.5)        # block 0 bias: no weight decay
    assert by[(2, 0.05)]["lr"] == pytest.approx(1e-3)             # head


def test_warmup_cosine_shape():
    f = [warmup_cosine(s, 100, 10) for s in range(100)]
    assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0)
    assert f[10] == pytest.approx(1.0) and f[-1] < 0.01
    assert all(a >= b for a, b in zip(f[10:], f[11:]))


# ---- stream assembly and windows ---------------------------------------------------------------------------------

def test_sample_rows_and_assemble_runs():
    a, b = vf("a", 10), vf("b", 6)
    assert sample_rows(a, [0, 9, 27]).tolist() == [0, 3, 9]
    with pytest.raises(CacheMissError):
        sample_rows(a, [1])
    refs = [["a", 6, 1.0], ["a", 9, 1.0], ["b", 0, 0.9], ["b", 3, 0.9], ["a", 0, 1.0]]
    calls = []

    def loader(v, var):
        calls.append((v, var)); return a if v == "a" else b
    feat = assemble(refs, loader, "feat")
    assert feat[:, 0].tolist() == [6.0, 9.0, 0.0, 3.0, 0.0]
    assert calls == [("a", "orig"), ("b", "gamma090"), ("a", "orig")]
    assert variant_of(1.1) == "gamma110"
    with pytest.raises(CacheMissError):
        variant_of(0.7)


def test_windowed_probs_cover_every_sample():
    centres, wins = windows(23)
    assert all(w.size == 16 for w in wins) and centres[-1] == 22
    _, short = windows(7)
    assert len(short) == 2 and short[0].tolist() == list(range(7)) + [6] * 9

    def fn(ws):  # probability of class 1 = mean sample index of the window / 100
        m = np.array([w.mean() / 100 for w in ws]); return np.stack([1 - m, m], 1)
    per_sample, per_window = windowed_probs(23, fn)
    assert per_sample.shape == (23, 2) and per_window.shape == (len(centres), 2)
    assert np.all(np.diff(per_sample[:, 1]) >= 0)


# ---- router ------------------------------------------------------------------------------------------------------

def test_router_band_mask_and_training_step():
    from echo_routing.vit.router import ViTRouter, band_mask, router_loss
    m = band_mask(6, 3)
    assert not m[2, 1] and not m[2, 3] and m[2, 4] and m[0, 2]
    torch.manual_seed(0)
    net = ViTRouter(in_dim=8, n_classes=3, dim=16, heads=2, n_layers=3, n_refine=1, refine_dim=8, refine_layers=2)
    x = torch.randn(1, 40, 8); y = torch.cat([torch.zeros(20), torch.full((20,), 2)]).long()[None]
    outs = net(x)
    assert len(outs) == 2 and all(o.shape == (1, 40, 3) for o in outs)
    opt = torch.optim.Adam(net.parameters(), 1e-2)
    first = router_loss(net(x), y).item()
    for _ in range(30):
        opt.zero_grad(); loss = router_loss(net(x), y); loss.backward(); opt.step()
    assert loss.item() < first
    net.eval()
    assert net(torch.randn(1, 1, 8))[-1].shape == (1, 1, 3)       # single-sample stream


# ---- models (tiny, no downloads) -----------------------------------------------------------------------------------

def test_factorized_video_vit_shapes_and_checkpoint_roundtrip(tmp_path):
    pytest.importorskip("timm")
    from echo_routing.vit.models import FactorizedVideoViT, ModelSpec, build_model, load_vit_checkpoint, save_vit_checkpoint
    net = FactorizedVideoViT(4, arch="vit_tiny_patch16_224", pretrained=False, n_frames=3, depth=1, heads=3)
    clip_logits, frame_logits = net(torch.randn(2, 3, 3, 224, 224))
    assert clip_logits.shape == (2, 4) and frame_logits.shape == (2, 3, 4)
    assert net.temporal_logits(torch.randn(1, 2, net.feature_dim)).shape == (1, 4)    # shorter window is allowed
    with pytest.raises(ValueError):
        net.temporal_logits(torch.randn(1, 4, net.feature_dim))
    spec = ModelSpec("vit_frame", 5, "family5", ("a", "b", "c", "d", "e"), arch="vit_tiny_patch16_224")
    model = build_model(spec, pretrained=False)
    path = save_vit_checkpoint(tmp_path / "m.pt", model, spec, {"epoch": 3})
    loaded, spec2, meta = load_vit_checkpoint(path)
    assert spec2 == spec and meta["epoch"] == 3 and spec2.kind == "frame" and spec2.norm == "imagenet"
    x = torch.randn(1, 3, 224, 224)
    model.eval(); loaded.eval()
    assert torch.allclose(model(x), loaded(x))


# ---- transition-aware clips ---------------------------------------------------------------------------------------

def _records(n_per_class=6, classes=3):
    from pathlib import Path
    from echo_routing.ingest.frames import VideoRecord
    return [VideoRecord(f"v{c}_{i}", Path("/nonexistent"), 60 + 7 * i, c, "train") for c in range(classes) for i in range(n_per_class)]


def test_mixed_clips_structure_and_plain_sampler_unchanged():
    from echo_routing.vit.datasets import EpochClips
    recs = _records()
    plain = EpochClips(recs, 16, 3, None, seed=5, mix_prob=0.0)
    assert all(len(it.segments) == 1 and len(it.segments[0].frames) == 16 for it in plain.items)
    # mix_prob = 0 must consume the random stream exactly as the original sampler (clip, then gamma per draw)
    rng = np.random.default_rng(5)
    from echo_routing.features.sampling import balanced_video_weights
    draws = rng.choice(len(recs), size=len(recs), replace=True, p=balanced_video_weights([r.label_index for r in recs]))
    first = random_clip(recs[draws[0]].n_frames, 16, 3, rng)
    assert plain.items[0].segments[0].video_idx == draws[0] and list(plain.items[0].segments[0].frames) == first.tolist()
    mixed = EpochClips(recs, 16, 3, None, seed=5, mix_prob=1.0)
    for it in mixed.items:
        assert len(it.segments) == 2 and sum(len(s.frames) for s in it.segments) == 16
        assert all(len(s.frames) >= 1 for s in it.segments)
    half = EpochClips(recs, 16, 3, 400, seed=1, mix_prob=0.5)
    share = np.mean([len(it.segments) == 2 for it in half.items])
    assert 0.35 < share < 0.65


def test_collate_and_soft_targets():
    from echo_routing.vit.datasets import collate_bytes, soft_targets
    batch = [([b"a"] * 4, [2, 2, 0, 0], 7, [1.0, 1.0, 0.9, 0.9]), ([b"b"] * 4, [1, 1, 1, 1], 3, [1.1] * 4)]
    b = collate_bytes(batch)
    assert len(b.datas) == 8 and b.t == 4 and b.labels.tolist() == [2, 1] and b.vids.tolist() == [7, 3]
    assert b.frame_labels.shape == (2, 4) and b.gammas[2] == 0.9
    st = soft_targets(b.frame_labels, 3)
    assert torch.allclose(st, torch.tensor([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]))


def test_soft_cross_entropy_matches_weighted_ce_on_one_hot():
    import torch.nn.functional as F
    from echo_routing.vit.train import soft_cross_entropy
    torch.manual_seed(0)
    logits = torch.randn(6, 4); y = torch.tensor([0, 1, 3, 3, 2, 1]); w = torch.tensor([1.0, 2.0, 0.5, 3.0])
    one_hot = F.one_hot(y, 4).float()
    assert torch.allclose(soft_cross_entropy(logits, one_hot, w), F.cross_entropy(logits, y, weight=w))
    assert torch.allclose(soft_cross_entropy(logits, one_hot, None), F.cross_entropy(logits, y))
    mixed = torch.tensor([[0.25, 0.75, 0.0, 0.0]])
    manual = -(mixed * F.log_softmax(logits[:1], 1)).sum()
    assert torch.allclose(soft_cross_entropy(logits[:1], mixed, None), manual)


def test_centre_target_equals_clip_label_for_single_view_and_centre_for_mixed():
    import torch.nn.functional as F
    from echo_routing.vit.train import ViTTrainConfig, _forward_loss

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.lin = torch.nn.Linear(3 * 4 * 4, 3)
        def forward(self, clips):
            return self.lin(clips.flatten(2).mean(1)), None
    torch.manual_seed(0)
    net = Toy(); x = torch.randn(2 * 16, 3, 4, 4)
    fl = torch.tensor([[1] * 16, [0] * 10 + [2] * 6]); y = fl[:, 0]
    logits = net(x.view(2, 16, 3, 4, 4))[0]
    centre = ViTTrainConfig(model="vivit_fe", task="raw9", mix_prob=0.5, mix_target="centre")
    plain = ViTTrainConfig(model="vivit_fe", task="raw9")
    got = _forward_loss(net, "clip", x, y, fl, 16, None, centre, 3)
    assert torch.allclose(got, F.cross_entropy(logits, torch.tensor([1, 0])))        # position 8 of [0]*10+[2]*6 is 0
    fl2 = torch.tensor([[1] * 16, [0] * 5 + [2] * 11])
    assert torch.allclose(_forward_loss(net, "clip", x, y, fl2, 16, None, centre, 3), F.cross_entropy(logits, torch.tensor([1, 2])))
    assert torch.allclose(_forward_loss(net, "clip", x, y, fl, 16, None, plain, 3), F.cross_entropy(logits, y))
    with pytest.raises(ValueError):
        ViTTrainConfig.from_dict({"model": "vivit_fe", "task": "raw9", "mix_target": "middle"})


def test_registry_resolves_seeded_vit_runs(tmp_path):
    from echo_routing.temporal.baselines.registry import decode_with_prob
    prob = np.tile(np.array([[0.1, 0.7, 0.1, 0.05, 0.05]]), (6, 1))
    np.savez_compressed(tmp_path / "s.npz", prob=prob.astype(np.float32))
    labels, p = decode_with_prob("vivit_fe@s1", np.full((6, 5), 0.2), {"vivit_fe@s1": {"pred_dir": str(tmp_path)}}, stream_id="s")
    assert labels.tolist() == [1] * 6 and np.allclose(p, prob)
    with pytest.raises(KeyError):
        decode_with_prob("b9@s1", np.full((6, 5), 0.2), {}, stream_id="s")
