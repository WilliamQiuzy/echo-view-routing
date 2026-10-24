"""Clip rule and branch structure tests of A28 (echo_routing/temporal/interval_clips.py)."""
from __future__ import annotations

import numpy as np
import torch

import echo_routing.temporal.interval_clips as ic
import echo_routing.vit.target_mvit as tm


def test_clip_positions_cases():
    assert ic.clip_positions(5, 6).tolist() == [5] * 16 and ic.clip_marker(ic.clip_positions(5, 6)) == 8
    s = ic.clip_positions(0, 10)
    assert s.tolist() == [0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 6, 7, 7, 8, 9, 9] and ic.clip_marker(s) == 9      # repeats in order; marker mid of s[8]'s slots
    assert ic.clip_positions(3, 19).tolist() == list(range(3, 19)) and ic.clip_marker(ic.clip_positions(3, 19)) == 8
    s = ic.clip_positions(100, 140)
    assert s.min() >= 100 and s.max() <= 139 and np.all(np.diff(s) > 0) and s[0] == 101 and s[-1] == 138    # even coverage inside the run
    for a, b in ((0, 1), (0, 2), (7, 23), (0, 333)):
        s = ic.clip_positions(a, b); assert s.min() >= a and s.max() < b and np.all(np.diff(s) >= 0)


def test_branch_trainable_sets_and_split():
    torch.manual_seed(0); enc = tm.build_source_encoder(None); m = tm.TargetMViT(enc).eval()
    f, a = ic.branch(m, False), ic.branch(m, True)
    assert [n for n, _ in ic.trainable(f)] == ["classifier.weight", "classifier.bias"] and sum(p.numel() for _, p in ic.trainable(f)) == 2565
    names = {n.split(".")[0] + "." + n.split(".")[1] for n, _ in ic.trainable(a)}
    assert sum(p.numel() for _, p in ic.trainable(a)) == 7497989 and names <= {"encoder.blocks", "encoder.norm", "encoder.head", "classifier.weight", "classifier.bias"}
    assert all(p.requires_grad for p in m.parameters())                                   # copying leaves the source model's flags untouched
    rgb = torch.randn(1, 3, 16, 224, 224); mk = tm.marker_planes(torch.tensor([8]))
    with torch.no_grad():
        x, thw = ic.prefix(a, rgb, mk); assert torch.allclose(ic.suffix(a, x, thw), a.embed(rgb, mk), atol=1e-5)
