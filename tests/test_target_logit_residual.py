"""Target-logit residual round (configs/experiments/target_logit_residual_v1.yaml): the target index, the Z/R readout
rule, the zero-initialised output layer and its gradients, and the run identity."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from echo_routing.config import REPO_ROOT, load_config
from echo_routing.evaluate.window_target import window_frames
from echo_routing.vit.residual_target import (ResidualTargetModel, readout, target_logits, temporal_named,
                                              zero_output_state)
from echo_routing.vit.target_video import TargetTimeHead, losses

CFG_PATH = REPO_ROOT / "configs/experiments/target_logit_residual_v1.yaml"


class Stub(nn.Module):
    """Stands in for the encoder: a frame's 'features' are a fixed projection of its mean pixel."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(1, dim)

    def forward(self, u8: torch.Tensor) -> torch.Tensor:
        return self.proj(u8.float().mean((1, 2, 3))[:, None])

    def trainable(self):
        return list(self.named_parameters())


def small_head(seed: int = 0) -> TargetTimeHead:
    torch.manual_seed(seed)
    return TargetTimeHead(dim=32, layers=2, heads=4, ffn=64, dropout=0.1, max_offset=15)


def test_target_logits_follow_the_offset_zero_positions():
    fl = torch.randn(3, 16, 5)
    rel = torch.stack([torch.arange(16) - 8, torch.arange(16), torch.arange(16) - 15])
    assert torch.equal(target_logits(fl, rel), torch.stack([fl[0, 8], fl[1, 0], fl[2, 15]]))
    s = window_frames(3, 6, 16, 16); rel_short = torch.from_numpy(s - 3)[None]          # a 6-sample clip repeats frames
    assert int(rel_short.eq(0).sum()) > 1
    fl_short = torch.randn(1, 16, 5); fl_short[0, rel_short[0] == 0] = fl_short[0, rel_short[0] == 0][0]   # same frame -> same logits
    assert torch.allclose(target_logits(fl_short, rel_short), fl_short[0, rel_short[0] == 0][0][None])
    with pytest.raises(ValueError):
        target_logits(fl[:1], (torch.arange(16) + 1)[None])


def test_readout_adds_logits_and_z_reads_the_correction_alone():
    fl = torch.randn(2, 16, 5); rel = (torch.arange(16) - 8).repeat(2, 1); delta = torch.randn(2, 5)
    r, _ = readout(delta, fl, rel, "R"); z, _ = readout(delta, fl, rel, "Z")
    assert torch.equal(r, fl[:, 8] + delta) and torch.equal(z, delta)
    avg = torch.log((torch.softmax(fl[:, 8], -1) + torch.softmax(delta, -1)) / 2)
    assert not torch.allclose(torch.log_softmax(r, -1), avg, atol=1e-4)                  # not a probability average
    with pytest.raises(ValueError):
        readout(delta, fl, rel, "C")


def test_zero_output_state_and_shared_parameters():
    head = small_head(); state = head.state_dict(); z = zero_output_state(state)
    assert sorted(k for k in state if not torch.equal(state[k], z[k])) == ["head.bias", "head.weight"]
    assert float(z["head.weight"].abs().sum()) == 0 and float(z["head.bias"].abs().sum()) == 0
    hz, hr = small_head(), small_head(); hz.load_state_dict(z); hr.load_state_dict(z)
    mz, mr = ResidualTargetModel(Stub(32), hz, "Z"), ResidualTargetModel(Stub(32), hr, "R")
    assert [(n, p.shape) for n, p in mz.head.named_parameters()] == [(n, p.shape) for n, p in mr.head.named_parameters()]
    names = {n for n, _ in temporal_named(hz)}
    assert not any(n.startswith(("head.", "frame_head.")) for n in names) and any(n.startswith("temporal.") for n in names) and {"cls", "cls_pos", "rel_pos"} <= names
    with pytest.raises(ValueError):
        ResidualTargetModel(Stub(32), hz, "X")


def test_start_values_and_zero_init_gradients():
    head = small_head(); head.load_state_dict(zero_output_state(head.state_dict()))
    model = ResidualTargetModel(Stub(32), head, "R")
    u8 = torch.randint(0, 255, (32, 4, 4, 3), dtype=torch.uint8); rel = (torch.arange(16) - 8).repeat(2, 1)
    labels = torch.tensor([[2] * 8 + [3] * 8, [1] * 16]); target = labels[:, 8]
    model.eval()
    with torch.no_grad():
        final, fl = model(u8, rel)
        assert torch.equal(final, target_logits(fl, rel))                                   # R starts at the frame logits
        model.mode = "Z"; assert float(model(u8, rel)[0].abs().max()) == 0.0; model.mode = "R"   # Z starts at zero
    model.train(); opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss, _ = losses("C", *model(u8, rel), labels, target, 0.5); loss.backward()
    assert all(float(p.grad.abs().max()) == 0.0 for _, p in temporal_named(head))           # expected with W = 0
    assert float(head.head.weight.grad.abs().max()) > 0 and float(head.frame_head.weight.grad.abs().max()) > 0
    assert float(model.encoder.proj.weight.grad.abs().max()) > 0
    opt.step(); opt.zero_grad()
    loss, _ = losses("C", *model(u8, rel), labels, target, 0.5); loss.backward()
    assert all(p.grad is not None and float(p.grad.abs().max()) > 0 for _, p in temporal_named(head))


def test_identity_tracks_arm_seed_inputs_and_code():
    from scripts.train_target_logit_residual import identity
    cfg = load_config(CFG_PATH); src = {"training_config": "t" * 64}
    base = identity(cfg, "R", 0, "m" * 64, "i" * 64, src, REPO_ROOT)
    assert base == identity(cfg, "R", 0, "m" * 64, "i" * 64, src, REPO_ROOT)
    others = [identity(cfg, "Z", 0, "m" * 64, "i" * 64, src, REPO_ROOT), identity(cfg, "R", 1, "m" * 64, "i" * 64, src, REPO_ROOT),
              identity(cfg, "R", 0, "n" * 64, "i" * 64, src, REPO_ROOT), identity(cfg, "R", 0, "m" * 64, "j" * 64, src, REPO_ROOT),
              identity(cfg, "R", 0, "m" * 64, "i" * 64, {"training_config": "u" * 64}, REPO_ROOT)]
    assert len({base, *others}) == 6
    raw = yaml.safe_load(CFG_PATH.read_text())["target_logit_residual"]
    assert raw["goal"]["selection_min"] == 0.934409687184662 + 0.01 or abs(raw["goal"]["selection_min"] - 0.944409687184662) < 1e-15
    assert np.isclose(raw["goal"]["calibration_min"], 0.9721421709894332 + 0.01, atol=1e-15)
