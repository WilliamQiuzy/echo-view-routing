from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from echo_routing.temporal.models.mstcn import MultiStageTCN, mstcn_loss  # noqa: E402


def test_mstcn_shapes_and_loss_decreases():
    torch.manual_seed(0)
    model = MultiStageTCN(num_stages=2, num_layers=3, channels=16, in_dim=8, num_classes=3)
    x = torch.randn(2, 8, 30); m = torch.ones(2, 1, 30); y = torch.randint(0, 3, (2, 30))
    outs = model(x, m)
    assert len(outs) == 2 and outs[0].shape == (2, 3, 30)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(15):
        loss = mstcn_loss(model(x, m), y, m); opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first


def test_mask_zeroes_padded_positions():
    model = MultiStageTCN(1, 2, 8, 4, 3).eval()
    x = torch.randn(1, 4, 10); m = torch.ones(1, 1, 10); m[0, 0, 7:] = 0
    out = model(x, m)[-1]
    assert torch.all(out[0, :, 7:] == 0)
