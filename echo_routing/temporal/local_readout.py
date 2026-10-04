"""Round 7: global against local readouts of the frozen E1 encoder for span-level A4C/A5C decisions
(configs/experiments/local_readout_v1.yaml).

- P0 is round 6's V0: the mean of the span's global frame embeddings into the span head.
- P1 reads the global token of each frame through a residual frame module, then weighs frames with gated attention.
- P2 reads all 197 tokens of each frame (the global token and the 196 patch tokens) with gated attention over the
  tokens, then weighs frames with the same temporal module.

Gated attention pooling follows Ilse, Tomczak and Welling (ICML 2018); it is an established method, not a
contribution here. P1's frame module has the same number of parameters as P2's token attention, so the two arms
differ only in what the frame vector reads.
"""
from __future__ import annotations

import torch
from torch import nn

from echo_routing.temporal.span_classifier import SpanHead


@torch.no_grad()
def all_tokens(enc, u8: torch.Tensor) -> torch.Tensor:
    """(N, 224, 224, 3) uint8 -> (N, 197, 512): the last block's output for every token, through ln_post and proj, as
    VisualEncoder does for the global token (token 0)."""
    vt, outs = enc.vt, []
    for i in range(0, u8.shape[0], enc.chunk):
        x = enc.prefix(enc.normalise(u8[i:i + enc.chunk]))
        for blk in vt.transformer.resblocks[enc.split:]:
            x = blk(x)
        outs.append(vt.ln_post(x.permute(1, 0, 2)) @ vt.proj)
    return torch.cat(outs, 0)


class GatedAttention(nn.Module):
    """Weights softmax(w . (tanh(V x) * sigmoid(U x))) over the set axis, and the weighted mean of the set."""

    def __init__(self, dim: int = 512, hidden: int = 128) -> None:
        super().__init__()
        self.V, self.U, self.w = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, S, D), mask (B, S) True where present -> pooled (B, D) and weights (B, S)."""
        s = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=1)
        return torch.einsum("bs,bsd->bd", a, x), a


class ResidualFrameModule(nn.Module):
    """x + W2 relu(W1 x), with W2 starting at zero, so the module starts as the identity."""

    def __init__(self, dim: int = 512, hidden: int = 128) -> None:
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(torch.relu(self.fc1(x)))


def pad_spans(frames: torch.Tensor, counts: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(sum(counts), D) frame vectors -> (B, T_max, D) padded with zeros and the (B, T_max) presence mask."""
    t_max = max(counts); out = frames.new_zeros(len(counts), t_max, frames.shape[-1])
    mask = torch.zeros(len(counts), t_max, dtype=torch.bool, device=frames.device)
    for b, chunk in enumerate(torch.split(frames, counts)):
        out[b, :len(chunk)] = chunk; mask[b, :len(chunk)] = True
    return out, mask


class GlobalReadout(nn.Module):
    """P1: global token -> residual frame module -> temporal gated attention -> span head."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.frame, self.temporal, self.head = ResidualFrameModule(hidden=hidden), GatedAttention(hidden=hidden), SpanHead()

    def forward(self, global_tokens: torch.Tensor, counts: list[int]) -> torch.Tensor:
        x, mask = pad_spans(self.frame(global_tokens), counts)
        return self.head(self.temporal(x, mask)[0])


class LocalReadout(nn.Module):
    """P2: all tokens -> token gated attention per frame -> temporal gated attention -> span head."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.spatial, self.temporal, self.head = GatedAttention(hidden=hidden), GatedAttention(hidden=hidden), SpanHead()

    def frame_vectors(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.spatial(tokens)

    def forward(self, tokens: torch.Tensor, counts: list[int]) -> torch.Tensor:
        f, _ = self.spatial(tokens)
        x, mask = pad_spans(f, counts)
        return self.head(self.temporal(x, mask)[0])


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_pair(seed: int, head_state: dict, hidden: int = 128) -> tuple[GlobalReadout, LocalReadout]:
    """P1 and P2 with round 6's head initialisation, one temporal initialisation copied into both, and frame modules
    initialised from their own seed."""
    p1, p2 = GlobalReadout(hidden), LocalReadout(hidden)
    torch.manual_seed(2000 + seed); temporal = GatedAttention(hidden=hidden).state_dict()
    torch.manual_seed(3000 + seed); spatial = GatedAttention(hidden=hidden).state_dict()
    torch.manual_seed(3000 + seed); frame = ResidualFrameModule(hidden=hidden).state_dict()
    for m in (p1, p2):
        m.head.load_state_dict(head_state); m.temporal.load_state_dict(temporal)
    p2.spatial.load_state_dict(spatial); p1.frame.load_state_dict(frame)
    return p1, p2
