"""A39: final-token capture, projection-then-pooling and the matched residual heads (random encoder, CPU)."""
import pytest
import torch

import echo_routing.temporal.token_pool_evidence as tpe
import echo_routing.vit.target_mvit as tm
import echo_routing.vit.target_patch_readout as tp


def test_pool_on_synthetic_tokens():
    v = torch.tensor([[[1.0, -2.0], [3.0, 0.0], [-1.0, 5.0]]]); d = tpe.pool(v)
    assert torch.equal(d["G"], torch.tensor([[1.0, 1.0]])) and torch.equal(d["P"], torch.tensor([[3.0, 5.0]]))
    r = torch.randn(4, 392, 16); d = tpe.pool(r)
    assert bool((d["P"] >= d["G"]).all())
    same = r[:, :1].expand(4, 392, 16); d = tpe.pool(same)
    assert torch.allclose(d["P"], d["G"]) and torch.equal(d["P"], r[:, 0])


def test_projection_before_pooling():
    """Mean commutes with the linear projection, max does not: the order is part of the contract."""
    torch.manual_seed(0); lin = torch.nn.Linear(8, 4).eval(); u = torch.randn(2, 1 + 12, 8)
    with torch.no_grad():
        v = tpe.project_tokens(lin, u); d = tpe.pool(v)
        assert v.shape == (2, 12, 4) and torch.allclose(d["G"], lin(u[:, 1:].mean(1)), atol=1e-6)
        assert not torch.allclose(d["P"], lin(u[:, 1:].amax(1)))
    lin.train()
    with pytest.raises(RuntimeError):
        tpe.project_tokens(lin, u)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0); m = tm.build_model(None, tm.initial_new_state(0)).eval()
    with torch.no_grad():
        m.marker_stem.weight.normal_(0, 0.01)
    return m


def test_capture_leaves_the_forward_unchanged(model):
    rgb = torch.randn(2, 3, 16, 224, 224); mk = tm.marker_planes(torch.tensor([8, 3]))
    with torch.no_grad():
        plain = model.embed(rgb, mk); e, u = tpe.embed_with_tokens(model, rgb, mk); cls = model.encoder.head(u[:, 0])
        d = tpe.descriptors(model, rgb, mk)
    assert torch.equal(e, plain) and torch.equal(cls, e) and u.shape == (2, tpe.N_TOKENS, 768) and tp.token_grid(model.encoder)[0] == tpe.GRID
    assert torch.equal(d["e224"], plain) and torch.equal(d["q"], model.classifier(plain)) and d["G"].shape == d["P"].shape == (2, 512)
    assert not model.encoder.norm._forward_hooks and not model.encoder.conv_proj._forward_hooks
    other = model.encoder.norm.register_forward_hook(lambda *_: None)
    try:
        with pytest.raises(RuntimeError):
            tpe.embed_with_tokens(model, rgb, mk)
    finally:
        other.remove()


def test_heads_are_identical_and_start_at_q():
    a, b = tpe.head(0), tpe.head(0)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values())) and sum(p.numel() for p in a.parameters()) == 32965
    assert not torch.equal(tpe.head(0).fc1.weight, tpe.head(1).fc1.weight) and float(a.fc2.weight.abs().sum()) == 0
    q, e = torch.randn(3, 5), torch.randn(3, 512)
    assert torch.equal(a(q, e, torch.randn(3, 512)), q)
