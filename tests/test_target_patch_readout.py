"""CPU checks of the target-anchored readout (configs/experiments/target_patch_readout_v1.yaml): the marker's time weights
under the RGB stem's temporal geometry, the Global and Target patch readouts on tokens of known content, the token order of
the real torchvision Pool, the absence of label inputs, and the wrapper (no new parameters, Continued CLS equal to the source
model, hand-recomputed readouts, hooks removed on every path). The encoder is randomly initialised here."""
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import echo_routing.vit.target_mvit as tm
import echo_routing.vit.target_patch_readout as tp

ROOT = Path(__file__).resolve().parents[1]
TP = yaml.safe_load((ROOT / "configs/experiments/target_patch_readout_v1.yaml").read_text())["target_patch_readout"]
GRID = (8, 7, 7)


def one_hot16(slots) -> torch.Tensor:
    m = torch.zeros(len(slots), 16)
    m[torch.arange(len(slots)), torch.as_tensor(slots)] = 1.0
    return m


def test_time_weights_follow_the_stem_geometry():
    w = tp.time_weights(one_hot16([0, 8, 9, 15, 3]))
    want = torch.zeros(5, 8)
    want[0, 0] = 1; want[1, 4] = 1; want[2, 4] = want[2, 5] = 0.5; want[3, 7] = 1; want[4, 1] = want[4, 2] = 0.5
    assert w.shape == (5, 8) and torch.allclose(w, want) and torch.allclose(w.sum(1), torch.ones(5)) and bool((w >= 0).all())


def test_marker_vector_requires_one_plane():
    planes = tm.marker_planes(torch.tensor([2, 15]), size=6)
    assert torch.equal(tp.marker_vector(planes), one_hot16([2, 15]))
    two = planes.clone(); two[0, 0, 5] = 1.0
    with pytest.raises(ValueError):
        tp.marker_vector(two)
    with pytest.raises(ValueError):
        tp.marker_vector(torch.zeros(1, 1, 16, 6, 6))


def test_repeated_targets_use_the_single_marker_plane():
    s = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]); c = 3          # target slots 6 and 7
    slot = tm.marker_slot(s, c); w = tp.time_weights(tp.marker_vector(tm.marker_planes(torch.tensor([slot]), size=4)))
    assert slot == 7 and torch.allclose(w, tp.time_weights(one_hot16([7]))) and not torch.allclose(w, tp.time_weights(one_hot16([6])))


def tokens_with_slot_values(b: int = 2, c: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """Patch tokens of time slot t all equal v[:, t]; the CLS token is a large constant that must never leak."""
    g = torch.Generator().manual_seed(0); v = torch.randn(b, GRID[0], c, generator=g)
    patches = v[:, :, None, None, :].expand(b, *GRID, c).reshape(b, -1, c)
    return torch.cat([torch.full((b, 1, c), 1e6), patches], 1), v


def test_global_and_target_readouts_on_tokens_of_known_content():
    x, v = tokens_with_slot_values(); w = tp.time_weights(one_hot16([0, 9]))
    assert torch.allclose(tp.readout_vector(x, "global_patches", w, GRID), v.mean(1), atol=1e-6)
    assert torch.allclose(tp.readout_vector(x, "target_patches", w, GRID), (w[:, :, None] * v).sum(1), atol=1e-6)
    assert torch.allclose(tp.readout_vector(x, "target_patches", w, GRID)[0], v[0, 0], atol=1e-6)
    y = tp.replace_token0(x, "target_patches", w, GRID)
    assert torch.equal(y[:, 1:], x[:, 1:]) and torch.allclose(y[:, 0], (w[:, :, None] * v).sum(1), atol=1e-6) and x[0, 0, 0] == 1e6
    assert tp.replace_token0(x, "cls", w, GRID) is x
    with pytest.raises(ValueError):
        tp.readout_vector(x[:, :-1], "global_patches", w, GRID)


def test_different_markers_select_different_positions():
    x, v = tokens_with_slot_values(b=1)
    a = tp.readout_vector(x, "target_patches", tp.time_weights(one_hot16([2])), GRID)
    b = tp.readout_vector(x, "target_patches", tp.time_weights(one_hot16([12])), GRID)
    assert torch.allclose(a, v[:, 1]) and torch.allclose(b, v[:, 6]) and not torch.allclose(a, b)


def test_token_order_matches_conv_and_pool_flattening():
    """Encode (t, h, w) in the channels, flatten as conv_proj does, pool as a stride-(1,2,2) block does, and decode."""
    from torchvision.models.video.mvit import Pool
    t, h, w = 4, 6, 6; grid = torch.stack(torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")).float()
    tokens = grid[None].flatten(2).transpose(1, 2)                                   # conv_proj: (B, C, T, H, W) -> (B, THW, C)
    x = torch.cat([torch.full((1, 1, 3), -1.0), tokens], 1)[:, None]                  # (B, heads=1, 1 + THW, C)
    pool = Pool(torch.nn.MaxPool3d((1, 1, 1), stride=(1, 2, 2)), None)
    y, thw = pool(x, (t, h, w)); patches = y[0, 0, 1:].reshape(*thw, 3)
    assert thw == (4, 3, 3)
    for ti in range(4):
        for hi in range(3):
            for wi in range(3):
                assert patches[ti, hi, wi].tolist() == [ti, 2 * hi, 2 * wi]


def test_readout_functions_take_no_labels():
    allowed = {"self", "tokens", "mode", "weights", "grid", "m16", "marker", "rgb", "kernel", "stride", "padding", "dilation", "encoder", "readout", "num_classes"}
    for fn in (tp.time_weights, tp.marker_vector, tp.readout_vector, tp.replace_token0, tp.ReadoutMViT.forward, tp.ReadoutMViT.embed, tp.ReadoutMViT.__init__):
        assert set(inspect.signature(fn).parameters) <= allowed, fn.__name__


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    base = tm.build_model(None, tm.initial_new_state(0)).eval()
    with torch.no_grad():
        base.marker_stem.weight.normal_(0, 0.01)                                     # a learned-looking marker
    state = {k: v.clone() for k, v in base.state_dict().items()}
    out = {}
    for r in tp.READOUTS:
        m = tp.ReadoutMViT(tm.build_source_encoder(None), r); m.load_state_dict(state, strict=True); out[r] = m.eval()
    return base, out


def test_wrapper_adds_nothing_and_cls_equals_the_source(models):
    base, ms = models
    keys = list(base.state_dict())
    for m in ms.values():
        assert list(m.state_dict()) == keys and sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in base.parameters())
        assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), base.state_dict().values()))
    assert ms["target_patches"].grid == GRID and all(s[0] == 1 for s in ms["target_patches"].query_strides)
    rgb = torch.randn(2, 3, 16, 224, 224); mk = tm.marker_planes(torch.tensor([8, 15]))
    with torch.no_grad():
        assert torch.equal(ms["cls"](rgb, mk), base(rgb, mk))


def test_patch_readouts_equal_a_recomputation_and_hooks_never_stay(models):
    base, ms = models
    rgb = torch.randn(2, 3, 16, 224, 224); mk = tm.marker_planes(torch.tensor([9, 0])); seen = {}
    h = base.encoder.norm.register_forward_hook(lambda _m, _i, o: seen.__setitem__("x", o.detach().clone()))
    try:
        with torch.no_grad():
            base.embed(rgb, mk)
    finally:
        h.remove()
    x = seen["x"]; assert x.shape == (2, 1 + 8 * 7 * 7, 768)
    p = x[:, 1:].reshape(2, *GRID, 768).mean((2, 3)); w = torch.tensor([[0, 0, 0, 0, 0.5, 0.5, 0, 0], [1.0, 0, 0, 0, 0, 0, 0, 0]])
    for r, vec in (("global_patches", p.mean(1)), ("target_patches", (w[:, :, None] * p).sum(1))):
        with torch.no_grad():
            want = base.classifier(base.encoder.head(vec)); got = ms[r](rgb, mk); emb = ms[r].embed(rgb, mk)
        assert float((got - want).abs().max()) <= 1e-5 and float((emb - base.encoder.head(vec)).abs().max()) <= 1e-5
        assert not ms[r].encoder.norm._forward_hooks and not ms[r].encoder.conv_proj._forward_hooks
    m = ms["target_patches"]; blk = m.encoder.blocks[3]
    blk.forward = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        with pytest.raises(RuntimeError):
            m(rgb, mk)
    finally:
        del blk.forward
    assert not m.encoder.norm._forward_hooks and not m.encoder.conv_proj._forward_hooks


def test_gradients_reach_marker_head_and_projection(models):
    _, ms = models; m = ms["target_patches"]; m.train(); m.set_encoder_trainable(False); m.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(m(torch.randn(1, 3, 16, 224, 224), tm.marker_planes(torch.tensor([4]))), torch.tensor([3]))
    loss.backward()
    assert float(m.marker_stem.weight.grad.abs().max()) > 0 and m.classifier.weight.grad is not None
    assert all(p.grad is None for p in m.encoder_parameters())
    m.set_encoder_trainable(True); m.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(m(torch.randn(1, 3, 16, 224, 224), tm.marker_planes(torch.tensor([4]))), torch.tensor([3])).backward()
    assert float(m.encoder.head[-1].weight.grad.abs().max()) > 0 and m.eval() is m


def test_grid_mismatch_stops():
    enc = tm.build_source_encoder(None); enc.blocks[2].attn.pool_q.pool.stride = (2, 1, 1)
    with pytest.raises(SystemExit):
        tp.ReadoutMViT(enc, "target_patches")


def test_config_keeps_the_source_recipe():
    tr_ = TP["training"]; o = tr_["optimiser"]
    assert TP["readouts"] == list(tp.READOUTS) and TP["primary"] == "target_patches" and TP["budget"]["gpu_hours_max"] == 12.0
    assert (tr_["total_updates"], tr_["warmup_updates"], tr_["batch_windows"], tr_["seeds"]) == (3750, 375, 8, [0, 1, 2])
    assert (o["lr_encoder"], o["lr_new"], o["weight_decay"], o["grad_clip"]) == (1e-5, 1e-4, 1e-4, 1.0)
    assert TP["queue"] == [f"{r}_s{s}" for s in (0, 1, 2) for r in tp.READOUTS]
    assert TP["evaluation"]["fixed_ordered_setting"]["thresholds"] == {0: 0.29269428317686846, 1: 0.2845063773784935, 2: 0.30845250187026374}
