"""A36: matched 320 inputs, the 320-grid encoder bookkeeping and the zero-initialised residual head."""
import numpy as np
import pytest
import torch
from PIL import Image

import echo_routing.temporal.spatial_detail as sd
import echo_routing.vit.target_mvit as tm


def test_constant_image_identical_and_box():
    im = Image.new("RGB", (320, 240), (90, 120, 30)); H, L = sd.pair(im, 1.0)
    assert H.shape == L.shape == (320, 320, 3) and np.array_equal(H, L)
    assert sd.content_box(320, 240) == (0, 40, 320, 280) and (H[:40] == sd.FILL).all() and (H[280:] == sd.FILL).all()


def test_h_content_is_the_gamma_original_and_l_loses_detail():
    g = np.random.default_rng(0); arr = g.integers(0, 256, (240, 320, 3), dtype=np.uint8); im = Image.fromarray(arr)
    H, L = sd.pair(im, 0.9); ref = np.asarray(sd.gamma_rgb(im, 0.9))
    assert np.array_equal(H[40:280], ref) and not np.array_equal(L[40:280], ref)
    cb = (np.indices((240, 320)).sum(0) % 2 * 255).astype(np.uint8); cbi = Image.fromarray(np.stack([cb] * 3, -1)); Hc, Lc = sd.pair(cbi, 1.0)
    assert np.abs(Hc[40:280].astype(int) - 127.5).mean() > np.abs(Lc[40:280].astype(int) - 127.5).mean() + 50     # the checkerboard is smoothed in L


@pytest.mark.parametrize("size", [(300, 200), (321, 239), (200, 300)])
def test_other_sizes_and_odd_padding(size):
    im = Image.new("RGB", size, (10, 20, 30)); base = sd.long_side_320(im); w, h = base.size
    assert max(w, h) == 320; l, t, r, b = sd.content_box(w, h); assert (l, t) == ((320 - w) // 2, (320 - h) // 2) and 320 - r - l in (0, 1) and 320 - b - t in (0, 1)
    H, L = sd.pair(im, 1.0); assert H.shape == L.shape == (320, 320, 3)


def test_residual_head_starts_at_q_and_layers():
    r = sd.Residual(5); q = torch.randn(7, 5); e = torch.randn(7, 512); out = r(q, e, e)
    assert torch.equal(out, q) and sum(p.numel() for p in r.parameters()) == 32965
    out.sum().backward(); assert float(r.fc1.weight.grad.abs().sum()) == 0 and float(r.fc2.weight.grad.abs().sum()) > 0
    a, b = sd.Residual(5), sd.Residual(5); assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_encoder320_grid_and_materialised_equivalence():
    torch.manual_seed(0); m = tm.TargetMViT(tm.build_source_encoder(None)).eval()
    with torch.no_grad():
        m.marker_stem.weight.normal_(0, 0.01)
    e = sd.encoder320(m); x = torch.randn(1, 3, 16, 320, 320); mk = tm.marker_planes(torch.tensor([8]), size=320)
    grids = sd.block_grids(e, x, mk); assert grids[0] == (96, 8, 80, 80)[1:] or grids[0] == (8, 80, 80); assert grids[1] == (8, 80, 80) and grids[-1] == (8, 10, 10)
    mm, changed = sd.materialised320(m); assert changed and all(k.endswith(("rel_pos_h", "rel_pos_w")) for k, _, _ in changed)
    with torch.no_grad():
        assert torch.allclose(e(x, mk), mm(x, mk), atol=1e-5)
    assert m.encoder.pos_encoding.spatial_size == (56, 56) and not any(p.requires_grad for p in e.parameters())
