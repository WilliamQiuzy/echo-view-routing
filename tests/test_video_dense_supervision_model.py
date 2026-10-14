"""CPU checks of the video dense supervision model (configs/experiments/video_dense_supervision_v1.yaml, Task 2): the 8 time
readouts on tokens of known content, no added parameter, the CLS path equal to the source TargetMViT (no token-0
replacement), one capture per forward with the hook removed on every path, no label input, the loss formula (lambda 0
equals the CLS cross-entropy in value and gradients), identical native losses for aligned and shuffled labels, auxiliary
gradients reaching the shared head, projection, marker and (unfrozen) encoder, and the geometry stops. Random encoder."""
import inspect
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

import echo_routing.vit.target_mvit as tm
import echo_routing.vit.video_dense_supervision as vd
import scripts.prepare_video_dense_supervision as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs/experiments/video_dense_supervision_v1.yaml").read_text())["video_dense_supervision"]
GRID = (8, 7, 7)


def tokens_with_time_values(b: int = 2, c: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """Patch tokens of time slot t all equal v[:, t]; token 0 is a huge constant that must never leak."""
    g = torch.Generator().manual_seed(0); v = torch.randn(b, GRID[0], c, generator=g)
    patches = v[:, :, None, None, :].expand(b, *GRID, c).reshape(b, -1, c)
    return torch.cat([torch.full((b, 1, c), 1e6), patches], 1), v


def test_constants_match_the_config():
    assert list(vd.ARMS) == CFG["arms"] and vd.NAMES == CFG["names"] and vd.LAMBDA == CFG["aux"]["lambda"]
    assert list(vd.ANCHORS) == CFG["model"]["anchors"] and list(vd.STEM_TIME) == CFG["model"]["stem_time"] and vd.EXPECTED_GRID == GRID
    assert vd.anchor_slots(*vd.STEM_TIME, 8) == pd.stem_anchors(*vd.STEM_TIME, 8) == list(vd.ANCHORS)


def test_time_vectors_average_space_per_time_slot():
    x, v = tokens_with_time_values()
    p = vd.time_vectors(x, GRID)
    assert p.shape == (2, 8, 6) and torch.allclose(p, v, atol=1e-6) and float(p.abs().max()) < 1e3
    with pytest.raises(ValueError):
        vd.time_vectors(x[:, :-1], GRID)


def test_model_functions_take_no_labels():
    allowed = {"self", "rgb", "marker", "encoder", "num_classes", "tokens", "grid", "kernel", "stride", "padding", "dilation", "n_time", "state"}
    for fn in (vd.DenseMViT.__init__, vd.DenseMViT.forward_with_time, vd.time_vectors, vd.anchor_slots, vd.build_dense_model):
        assert set(inspect.signature(fn).parameters) <= allowed, fn.__name__


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    base = tm.build_model(None, tm.initial_new_state(0)).eval()
    with torch.no_grad():
        base.marker_stem.weight.normal_(0, 0.01)
    state = {k: v.clone() for k, v in base.state_dict().items()}
    return base, state


def test_no_parameter_is_added_and_the_state_loads_strictly(models):
    base, state = models; m = vd.build_dense_model(state)
    assert list(m.state_dict()) == list(base.state_dict()) and sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in base.parameters())
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), base.state_dict().values()))
    assert m.grid == GRID and m.anchors == list(vd.ANCHORS) and m.stem_time == vd.STEM_TIME


def test_cls_path_equals_the_source_and_time_logits_equal_a_recomputation(models):
    base, state = models; m = vd.build_dense_model(state).eval()
    rgb = torch.randn(2, 3, 16, 224, 224); mk = tm.marker_planes(torch.tensor([8, 15])); seen = {}
    h = base.encoder.norm.register_forward_hook(lambda _m, _i, o: seen.__setitem__("x", o.detach().clone()))
    try:
        with torch.no_grad():
            want_cls = base(rgb, mk)
    finally:
        h.remove()
    with torch.no_grad():
        z_cls, z_time = m.forward_with_time(rgb, mk); plain = m(rgb, mk)
        want_time = base.classifier(base.encoder.head(seen["x"][:, 1:].reshape(2, *GRID, 768).mean((2, 3))))
    assert float((z_cls - want_cls).abs().max()) == 0.0 and torch.equal(plain, want_cls)
    assert z_time.shape == (2, 8, 5) and float((z_time - want_time).abs().max()) <= 1e-5
    assert not m.encoder.norm._forward_hooks and not m.encoder.conv_proj._forward_hooks


def test_hooks_never_stay_and_capture_exactly_once(models):
    _, state = models; m = vd.build_dense_model(state).eval(); rgb = torch.randn(1, 3, 16, 224, 224); mk = tm.marker_planes(torch.tensor([4]))
    blk = m.encoder.blocks[3]; blk.forward = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        with pytest.raises(RuntimeError):
            m.forward_with_time(rgb, mk)
    finally:
        del blk.forward
    assert not m.encoder.norm._forward_hooks and not m.encoder.conv_proj._forward_hooks
    other = m.encoder.norm.register_forward_hook(lambda *_: None)
    try:
        with pytest.raises(RuntimeError):
            m.forward_with_time(rgb, mk)
    finally:
        other.remove()
    twice = vd.build_dense_model(state).eval()
    twice.forward = lambda r, k: (tm.TargetMViT.forward(twice, r, k), tm.TargetMViT.forward(twice, r, k))[1]
    with torch.no_grad(), pytest.raises(RuntimeError):
        twice.forward_with_time(rgb, mk)
    assert not twice.encoder.norm._forward_hooks


def test_the_loss_formula_and_label_isolation():
    g = torch.Generator().manual_seed(1); z_cls = torch.randn(4, 5, generator=g); z_time = torch.randn(4, 8, 5, generator=g)
    y = torch.tensor([0, 2, 3, 4]); a = torch.randint(0, 5, (4, 8), generator=g); b = (a + 1) % 5
    lc = F.cross_entropy(z_cls, y); lt = F.cross_entropy(z_time.reshape(-1, 5), a.reshape(-1), reduction="none").reshape(4, 8).mean(1).mean()
    loss, parts = vd.dense_loss(z_cls, z_time, y, a, 0.5)
    assert torch.allclose(loss, (lc + 0.5 * lt) / 1.5) and abs(parts["center"] - float(lc)) < 1e-7 and abs(parts["time"] - float(lt)) < 1e-7
    zero, _ = vd.dense_loss(z_cls, z_time, y, a, 0.0)
    assert torch.equal(zero, lc + 0.0 * lt) and float(zero) == float(lc)
    assert float(vd.dense_loss(z_cls, z_time, y, b, 0.5)[0]) != float(loss)          # labels change the loss only


def test_lambda_zero_gives_the_cls_cross_entropy_gradients(models):
    base, state = models; plain = tm.build_model(None, tm.initial_new_state(0)); plain.load_state_dict(state, strict=True); m = vd.build_dense_model(state)
    for net in (plain, m):
        net.train(); net.set_encoder_trainable(True); net.zero_grad(set_to_none=True)
    rgb = torch.randn(2, 3, 16, 224, 224); mk = tm.marker_planes(torch.tensor([8, 3])); y = torch.tensor([1, 4]); aux = torch.randint(0, 5, (2, 8))
    torch.manual_seed(7); want = F.cross_entropy(plain(rgb, mk), y); want.backward()
    torch.manual_seed(7); z_cls, z_time = m.forward_with_time(rgb, mk); got, _ = vd.dense_loss(z_cls, z_time, y, aux, 0.0); got.backward()
    assert float(got) == float(want)
    gp, gm = dict(plain.named_parameters()), dict(m.named_parameters())
    assert all((gp[k].grad is None and gm[k].grad is None) or torch.equal(gp[k].grad, gm[k].grad) for k in gp)


def test_native_windows_give_identical_aligned_and_shuffled_losses_and_gradients(models):
    _, state = models; rgb = torch.randn(2, 3, 16, 224, 224); mk = tm.marker_planes(torch.tensor([8, 8])); y = torch.tensor([2, 2])
    aligned = torch.full((2, 8), 2); shuffled = aligned[:, torch.tensor([3, 1, 7, 0, 2, 6, 5, 4])]
    out = []
    for labels in (aligned, shuffled):
        m = vd.build_dense_model(state); m.train(); m.set_encoder_trainable(False); m.zero_grad(set_to_none=True)
        torch.manual_seed(3); z_cls, z_time = m.forward_with_time(rgb, mk); loss, _ = vd.dense_loss(z_cls, z_time, y, labels, 0.5); loss.backward()
        out.append((float(loss), {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}))
    assert out[0][0] == out[1][0] and out[0][1].keys() == out[1][1].keys() and all(torch.equal(out[0][1][k], out[1][1][k]) for k in out[0][1])


def test_aux_gradients_reach_the_shared_head_projection_marker_and_encoder(models):
    _, state = models; m = vd.build_dense_model(state); m.train(); rgb = torch.randn(1, 3, 16, 224, 224); mk = tm.marker_planes(torch.tensor([5]))
    aux = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]])
    m.set_encoder_trainable(False); m.zero_grad(set_to_none=True)
    _, z_time = m.forward_with_time(rgb, mk); F.cross_entropy(z_time.reshape(-1, 5), aux.reshape(-1)).backward()
    assert float(m.classifier.weight.grad.abs().max()) > 0 and float(m.marker_stem.weight.grad.abs().max()) > 0
    assert all(p.grad is None for p in m.encoder_parameters())
    m.set_encoder_trainable(True); m.zero_grad(set_to_none=True)
    _, z_time = m.forward_with_time(rgb, mk); F.cross_entropy(z_time.reshape(-1, 5), aux.reshape(-1)).backward()
    assert float(m.encoder.head[-1].weight.grad.abs().max()) > 0 and float(m.encoder.blocks[0].attn.qkv.weight.grad.abs().max()) > 0


def test_train_mode_rng_use_does_not_depend_on_labels_or_lambda(models):
    _, state = models; rgb = torch.randn(2, 3, 16, 224, 224); mk = tm.marker_planes(torch.tensor([8, 2])); y = torch.tensor([0, 3]); res = []
    for lam, labels in ((0.0, torch.zeros(2, 8, dtype=torch.long)), (0.5, torch.ones(2, 8, dtype=torch.long)), (0.5, torch.full((2, 8), 4))):
        m = vd.build_dense_model(state); m.train(); m.set_encoder_trainable(False)
        torch.manual_seed(11); z_cls, z_time = m.forward_with_time(rgb, mk); vd.dense_loss(z_cls, z_time, y, labels, lam)[0].backward()
        res.append((z_cls.detach(), z_time.detach(), torch.get_rng_state()))
    assert all(torch.equal(res[0][i], r[i]) for r in res[1:] for i in range(3))


def test_geometry_mismatches_stop():
    enc = tm.build_source_encoder(None); enc.blocks[2].attn.pool_q.pool.stride = (2, 1, 1)
    with pytest.raises(SystemExit):
        vd.DenseMViT(enc)
    enc = tm.build_source_encoder(None); enc.conv_proj.padding = (0, 3, 3)
    with pytest.raises(SystemExit):
        vd.DenseMViT(enc)
