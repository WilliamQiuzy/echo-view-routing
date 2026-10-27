"""A40: HOG target, tube mask, token layout, decoder and the grad-keeping capture (random encoder, CPU)."""
import math

import numpy as np
import pytest
import torch

import echo_routing.temporal.masked_feature as mf
import echo_routing.vit.target_mvit as tm


def _vectorised(gray: np.ndarray) -> np.ndarray:
    return mf.block_features(mf.cell_histograms(torch.from_numpy(gray.astype(np.float32))[None]))[0].double().numpy()


def test_hog_matches_pixel_loop():
    rng = np.random.default_rng(40); u8 = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8); g = mf.gray_of(torch.from_numpy(u8)).numpy()
    assert np.allclose(g, u8.astype(np.float32).mean(-1) / 255, atol=1e-7)
    assert np.abs(_vectorised(g) - mf.reference_block_hog(g)).max() < 1e-5


def test_constant_and_single_orientation():
    assert np.array_equal(_vectorised(np.full((224, 224), 0.5, np.float32)), np.zeros((49, 144)))
    ramp = np.tile(np.arange(224, dtype=np.float32) / 224, (224, 1)); h = _vectorised(ramp).reshape(49, 16, 9)   # gx > 0, gy = 0: theta 0
    assert np.allclose(h[..., 1:8], 0) and np.allclose(h[..., 0], h[..., 8]) and np.allclose(h[..., 0], math.sqrt(0.5), atol=1e-3)
    h = _vectorised(ramp.T.copy()).reshape(49, 16, 9)                                                                 # theta pi / 2: bin 4
    assert np.allclose(np.delete(h, 4, -1), 0, atol=1e-6) and np.allclose(h[..., 4], 1.0, atol=1e-3)
    norms = np.linalg.norm(_vectorised(np.random.default_rng(1).random((224, 224)).astype(np.float32)).reshape(49, 16, 9), axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_block_layout():
    cells = torch.arange(28 * 28 * 9, dtype=torch.float32).reshape(1, 28, 28, 9); b = mf.block_features(cells)[0]
    assert torch.equal(b[2 * 7 + 5], cells[0, 8:12, 20:24].reshape(144))


def test_targets_use_even_slots():
    u8 = torch.randint(0, 256, (2, 16, 224, 224, 3), dtype=torch.uint8); t = mf.hog_targets(u8)
    assert t.shape == (2, 8, 49, 144) and not t.requires_grad
    one = mf.block_features(mf.cell_histograms(mf.gray_of(u8[1, 6])[None]))[0]
    assert torch.equal(t[1, 3], one)


def test_mask_draw_and_pixels():
    a, b = mf.mask_blocks(0, 1, 7, 8), mf.mask_blocks(0, 1, 7, 8)
    assert np.array_equal(a, b) and (a.sum(1) == 25).all() and not np.array_equal(a[0], a[1])
    exp = np.zeros(49, bool); exp[np.random.default_rng([20261026, 40, 0, 1, 7, 3]).choice(49, 25, replace=False)] = True
    assert np.array_equal(a[3], exp)
    rgb = torch.randn(2, 3, 16, 224, 224); sel = torch.from_numpy(a[:2]); out = mf.masked_rgb(rgb, sel)
    for w in range(2):
        for k in range(49):
            y, x = divmod(k, 7); blk = out[w, :, :, 32 * y:32 * y + 32, 32 * x:32 * x + 32]
            assert bool((blk == 0).all()) if sel[w, k] else torch.equal(blk, rgb[w, :, :, 32 * y:32 * y + 32, 32 * x:32 * x + 32])


def test_loss_ignores_unmasked_positions():
    sel = torch.from_numpy(mf.mask_blocks(0, 0, 0, 2)); t = torch.randn(2, 8, 49, 144); p = torch.randn(2, 8, 49, 144, requires_grad=True)
    loss = mf.hog_loss(p, t, sel); loss.backward()
    ref = ((p.detach() - t) ** 2)[sel[:, None, :, None].expand_as(t)].mean()
    assert torch.allclose(loss, ref) and bool((p.grad[~sel[:, None, :].expand(2, 8, 49)] == 0).all())
    q = p.detach().clone(); q[~sel[:, None, :].expand(2, 8, 49)] += 5.0
    assert torch.allclose(mf.hog_loss(q, t, sel), loss.detach())


def test_token_grid_order():
    u = torch.arange(2 * 393, dtype=torch.float32).reshape(2, 393, 1).expand(2, 393, 3); g = mf.token_grid(u)
    for t, y, x in ((0, 0, 0), (3, 2, 5), (7, 6, 6)):
        assert torch.equal(g[1, t, y * 7 + x], u[1, 1 + t * 49 + y * 7 + x])


def test_decoder_init_is_fixed_and_isolated():
    torch.manual_seed(123); s0 = torch.random.get_rng_state(); a = mf.decoder(0, 1); b = mf.decoder(0, 1); c = mf.decoder(0, 0)
    assert torch.equal(torch.random.get_rng_state(), s0) and torch.equal(a.weight, b.weight) and not torch.equal(a.weight, c.weight)
    assert bool((a.bias == 0).all()) and 0.015 < float(a.weight.std()) < 0.025


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0); m = tm.build_model(None, tm.initial_new_state(0)).eval()
    with torch.no_grad():
        m.marker_stem.weight.normal_(0, 0.02)
    return m


def test_capture_keeps_autograd_and_output(model):
    rgb, mk = torch.randn(1, 3, 16, 224, 224), tm.marker_planes(torch.tensor([7]))
    with torch.no_grad():
        e0 = model.embed(rgb, mk)
    e, u = mf.embed_with_tokens_grad(model, rgb, mk)
    assert torch.equal(e.detach(), e0) and u.shape == (1, 393, 768) and u.requires_grad and u.grad_fn is not None
    assert torch.allclose(model.encoder.head(u[:, 0]).detach(), e0, atol=1e-5)
    mf.hog_loss(mf.decoder(0, 0)(mf.token_grid(u)), torch.zeros(1, 8, 49, 144), torch.ones(1, 49, dtype=torch.bool)).backward()
    assert model.marker_stem.weight.grad is not None and float(model.marker_stem.weight.grad.abs().sum()) > 0
    assert model.encoder.head[-1].weight.grad is None and not model.encoder.norm._forward_hooks and not model.encoder.conv_proj._forward_hooks
    model.zero_grad(set_to_none=True)


def test_capture_removed_after_error(model):
    with pytest.raises(ValueError):
        mf.embed_with_tokens_grad(model, torch.randn(1, 3, 16, 224, 224), torch.zeros(1, 1, 8, 224, 224))
    assert not model.encoder.norm._forward_hooks and not model.encoder.conv_proj._forward_hooks
