"""Round-7 readouts (echo_routing/temporal/local_readout.py)."""
import torch
from torch import nn

from echo_routing.temporal.joint_visual import VisualEncoder
from echo_routing.temporal.local_readout import (GatedAttention, ResidualFrameModule, all_tokens, build_pair, n_params,
                                                 pad_spans)
from echo_routing.temporal.span_classifier import SpanHead


class _Block(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__(); self.lin = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.lin(x))


class _TinyViT(nn.Module):
    """The attributes VisualEncoder uses, at a small width."""

    def __init__(self, d: int = 32, out: int = 16, blocks: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, d, kernel_size=16, stride=16, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(d)); self.positional_embedding = nn.Parameter(torch.randn(197, d))
        self.ln_pre, self.ln_post = nn.LayerNorm(d), nn.LayerNorm(d)
        self.transformer = nn.Module(); self.transformer.resblocks = nn.ModuleList([_Block(d) for _ in range(blocks)])
        self.proj = nn.Parameter(torch.randn(d, out))


def test_token0_is_the_global_embedding():
    torch.manual_seed(0); enc = VisualEncoder(_TinyViT(), trainable_blocks=1, chunk=2).eval()
    u8 = torch.randint(0, 256, (3, 224, 224, 3), dtype=torch.uint8)
    tok = all_tokens(enc, u8)
    assert tok.shape == (3, 197, 16)
    with torch.no_grad():
        assert torch.allclose(tok[:, 0], enc(u8), atol=1e-6)


def test_gated_attention_ignores_padding_and_sums_to_one():
    torch.manual_seed(0); att = GatedAttention(dim=8, hidden=4)
    x = torch.randn(2, 5, 8); mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    pooled, a = att(x, mask)
    assert torch.allclose(a.sum(1), torch.ones(2)) and torch.all(a[0, 3:] == 0)
    x2 = x.clone(); x2[0, 3:] = 100.0                                  # padded rows must not change anything
    assert torch.allclose(att(x2, mask)[0][0], pooled[0])


def test_pad_spans_keeps_order_and_marks_presence():
    frames = torch.arange(12.0).view(6, 2); x, mask = pad_spans(frames, [2, 3, 1])
    assert x.shape == (3, 3, 2) and mask.tolist() == [[True, True, False], [True, True, True], [True, False, False]]
    assert torch.equal(x[1, :3], frames[2:5]) and torch.equal(x[2, 0], frames[5])


def test_frame_module_starts_as_identity():
    x = torch.randn(4, 512)
    assert torch.equal(ResidualFrameModule()(x), x)


def test_capacities_match_the_config():
    p1, p2 = build_pair(0, SpanHead().state_dict())
    assert n_params(SpanHead()) == 65_922
    assert n_params(p1.frame) == 131_712 and n_params(p2.spatial) == 131_457
    assert n_params(p1) == 329_091 and n_params(p2) == 328_836
    assert abs(n_params(p1) - n_params(p2)) <= 0.01 * n_params(p2)


def test_pair_shares_head_and_temporal_initialisation():
    torch.manual_seed(1000); head = SpanHead().state_dict()
    p1, p2 = build_pair(3, head)
    for a, b in ((p1.head, p2.head), (p1.temporal, p2.temporal)):
        assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert all(torch.equal(p1.head.state_dict()[n], v) for n, v in head.items())
    q1, q2 = build_pair(3, head)                                       # deterministic per seed
    assert all(torch.equal(x, y) for x, y in zip(p2.state_dict().values(), q2.state_dict().values()))


def test_readouts_give_span_logits():
    torch.manual_seed(0); p1, p2 = build_pair(0, SpanHead().state_dict()); counts = [3, 1, 2]
    assert p1(torch.randn(6, 512), counts).shape == (3, 2)
    assert p2(torch.randn(6, 197, 512), counts).shape == (3, 2)
