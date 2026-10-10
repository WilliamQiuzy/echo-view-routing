"""Partition strategies for the last U block of the official FACT, and the frozen boundary head of the predicted
partition (docs/superpowers/plans/2026-10-09-fact-predicted-partition.md).

The official UpdateBlockTDU groups the frames into the runs of the previous (u) block's argmax, averages each group,
and runs a GRU, cross-attention, upsampling and its frame branch over the groups. Here the last block may refine that
partition before the averaging.
- original: the official runs, unchanged;
- uniform: every run [s, e) also cut at s + 8, s + 16, ... < e;
- predicted: every run also cut where the frozen boundary head proposes a cut strictly inside it;
- predicted_even (an evaluation control on frozen weights): in every run, as many cuts as the head proposes inside it,
  placed evenly at s + floor(j (e - s) / (k + 1)), j = 1..k.
A refinement only adds boundaries. The official averaging, GRU, attention, upsampling, frame branch, losses and decode
are unchanged. Only positions enter the partition: a segment's action is the u block's argmax at its start, and no
reference label, nor anything derived from one, is read in the forward pass. Native windows and constructed streams
follow the same rule.

The boundary head reads the raw E1 features that enter FACT, before FACT's channel and time masking. It is trained
beforehand on its own loss and then frozen. It is not a submodule of the FACT model, so it is in neither FACT's state
dict nor its optimiser, and no FACT loss reaches it.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from echo_routing.temporal.segments import labels_to_segments, non_max_suppression

POLICIES = ("original", "uniform", "predicted", "predicted_even")
GRID = 8


# ----------------------------------------------------------------------------------------------- boundary head
class BoundaryHead(nn.Module):
    """Linear(3D, 64) - ReLU - Conv1d(64, 64, 3, padding 1) - ReLU - Conv1d(64, 1, 1) over one unpadded sequence; no
    dropout, normalisation or residual."""

    def __init__(self, in_dim: int = 512, hidden: int = 64) -> None:
        super().__init__()
        self.inp = nn.Linear(3 * in_dim, hidden)
        self.conv = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.out = nn.Conv1d(hidden, 1, 1)

    @staticmethod
    def features(x: torch.Tensor) -> torch.Tensor:
        """(T, D) -> (T, 3D): [x_(t-1), x_t, x_t - x_(t-1)] with x_(-1) = x_0."""
        prev = torch.cat([x[:1], x[:-1]], 0)
        return torch.cat([prev, x, x - prev], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One unpadded sequence (T, D) -> boundary logits (T,)."""
        h = torch.relu(self.inp(self.features(x))).T.unsqueeze(0)   # (1, H, T)
        return self.out(torch.relu(self.conv(h)))[0, 0]


def head_targets(labels) -> np.ndarray:
    """z_t = 1[y_t != y_(t-1)] for t >= 1, z_0 = 0 (position 0 is never scored). A join of two fragments of one view
    and every position of a native window are 0."""
    y = np.asarray(labels)
    z = np.zeros(y.size, np.float32)
    if y.size > 1:
        z[1:] = (y[1:] != y[:-1]).astype(np.float32)
    return z


def boundary_counts(label_seqs) -> tuple[int, int]:
    """(positives, negatives) over the scored positions t >= 1 of the given sequences."""
    pos = neg = 0
    for y in label_seqs:
        z = head_targets(y)[1:]
        pos += int(z.sum()); neg += int(z.size - z.sum())
    return pos, neg


def head_pos_weight(label_seqs) -> float:
    """Negatives / positives over the scored positions; no positive or a non-finite ratio stops the run."""
    pos, neg = boundary_counts(label_seqs)
    w = neg / pos if pos else float("nan")
    if not pos or not np.isfinite(w):
        raise ValueError(f"the head's pos_weight is undefined ({pos} positives, {neg} negatives)")
    return float(w)


def head_loss(head: BoundaryHead, xs: list[torch.Tensor], labels: list[torch.Tensor], pos_weight: torch.Tensor) -> torch.Tensor:
    """BCE with logits and pos_weight over the scored positions t >= 1 of each unpadded sequence, averaged within each
    sequence and then over the sequences. A sequence of one sample has no scored position and is left out."""
    losses = []
    for x, y in zip(xs, labels):
        if x.shape[0] < 2:
            continue
        z = (y[1:] != y[:-1]).float()
        losses.append(F.binary_cross_entropy_with_logits(head(x)[1:], z, pos_weight=pos_weight))
    if not losses:
        raise ValueError("no sequence has a scored position")
    return torch.stack(losses).mean()


@torch.no_grad()
def head_cuts(head: BoundaryHead, x: torch.Tensor, threshold: float = 0.5, radius: int = 2) -> list[int]:
    """The head's cut positions: the sigmoid (in float64) at least `threshold`, peaks kept by the shared
    non_max_suppression with `radius` (ties keep the earlier position). Position 0 is never a candidate, so it neither
    becomes a cut nor suppresses one."""
    p = torch.sigmoid(head(x).double()).cpu().numpy()
    if p.size == 0:
        return []
    p[0] = -np.inf
    return [int(c) for c in non_max_suppression(p, threshold, radius) if 0 < c < p.size]


def head_cut_fn(head: BoundaryHead, threshold: float, radius: int) -> Callable[[torch.Tensor], list[int]]:
    """The frozen head as a plain function of the raw features (it stays outside the FACT module tree)."""
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return lambda x: head_cuts(head, x, threshold, radius)


# ----------------------------------------------------------------------------------------------- partitions (half-open)
def runs(pred) -> list[tuple[int, int]]:
    """The runs of a label sequence as half-open intervals (the official utils.parse_label, half-open)."""
    return [(g.start, g.end) for g in labels_to_segments(np.asarray(pred))]


def check_tiling(intervals: list, n: int) -> None:
    if not intervals or intervals[0][0] != 0 or intervals[-1][1] != n or any(s >= e for s, e in intervals) \
            or any(a[1] != b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("the intervals do not tile the sequence")


def add_cuts(intervals: list, cuts) -> list[tuple[int, int]]:
    """The intervals split at every cut strictly inside one of them. A cut on an existing boundary or outside the
    sequence adds nothing, and repeated cuts count once."""
    cs = sorted({int(c) for c in cuts}); out = []
    for s, e in intervals:
        pts = [s, *[c for c in cs if s < c < e], e]
        out += list(zip(pts[:-1], pts[1:]))
    return out


def grid_cuts(intervals: list, step: int = GRID) -> list[int]:
    """s + step, s + 2 step, ... strictly below e in every interval [s, e); an interval of at most `step` samples gets none."""
    return [c for s, e in intervals for c in range(s + step, e, step)]


def even_cuts(intervals: list, cuts) -> list[int]:
    """In every interval [s, e), as many cuts as `cuts` has strictly inside it (k), at s + floor(j (e - s) / (k + 1))."""
    cs = sorted({int(c) for c in cuts}); out = []
    for s, e in intervals:
        k = sum(1 for c in cs if s < c < e)
        out += [s + (j * (e - s)) // (k + 1) for j in range(1, k + 1)]
    return out


def refine(base: list, policy: str, cuts=None, step: int = GRID) -> list[tuple[int, int]]:
    """The partition of one policy from the official runs `base` (and the head's cuts for the predicted policies)."""
    if policy == "original":
        return list(base)
    if policy == "uniform":
        return add_cuts(base, grid_cuts(base, step))
    if cuts is None:
        raise ValueError(f"the {policy} policy needs the head's cuts")
    if policy == "predicted":
        return add_cuts(base, cuts)
    if policy == "predicted_even":
        return add_cuts(base, even_cuts(base, cuts))
    raise ValueError(f"unknown policy {policy}")


def closed_segments(intervals: list, action, segment_cls) -> list:
    """Half-open [s, e) as the official closed Segment(action, s, e - 1), whose len is e - s. The action is the u block's
    argmax at s. The segments are built directly, so adjacent intervals of one class stay separate."""
    check_tiling(intervals, len(action))
    return [segment_cls(int(action[s]), int(s), int(e) - 1) for s, e in intervals]


# ----------------------------------------------------------------------------------------------- FACT adapter
_CLASSES: dict[int, tuple] = {}


def adapter_classes(mods: dict) -> tuple:
    """Subclasses of the official FACT and UpdateBlockTDU (cached per imported checkout). Only the last block's segment
    list changes; every other line of temporal_downsample is the official one, and the forward keeps gradients."""
    blocks = mods["blocks"]
    if id(blocks) in _CLASSES:
        return _CLASSES[id(blocks)]
    utils, basic = blocks.utils, mods["basic"]

    class RefinedUpdateBlockTDU(blocks.UpdateBlockTDU):
        def temporal_downsample(self, frame_feature):
            cprob = frame_feature[:, :, -self.nclass:]
            _, pred = cprob[:, 0].max(dim=-1)
            pred = utils.to_numpy(pred)
            intervals = refine(runs(pred), self.policy, self.pending_cuts, self.grid)
            segs = closed_segments(intervals, pred, utils.Segment)
            tdu = basic.TemporalDownsampleUpsample(segs)
            tdu.to(cprob.device)
            seg_feature = tdu.feature_frame2seg(frame_feature)
            seg_feature, _ = self.seg_update(seg_feature)
            seg_feature = torch.relu(seg_feature)
            seg_feature = self.seg_combine(seg_feature)
            seg_feature, seg_clogit = self.process_feature(seg_feature, self.nclass)
            return tdu, seg_feature, seg_clogit

    class PartitionedFACT(blocks.FACT):
        def _forward_one_video(self, seq, transcript=None):
            last = self.block_list[-1]
            last.pending_cuts = self.cut_fn(seq[:, 0]) if last.policy in ("predicted", "predicted_even") else None   # raw features
            try:
                return super()._forward_one_video(seq, transcript)
            finally:
                last.pending_cuts = None

    _CLASSES[id(blocks)] = (RefinedUpdateBlockTDU, PartitionedFACT)
    return _CLASSES[id(blocks)]


def partitioned(model: nn.Module, mods: dict, policy: str, cut_fn: Callable | None = None, step: int = GRID) -> nn.Module:
    """The official FACT model turned, in place, into one that refines its last block's partition by `policy`. The
    module tree and state dict are unchanged. The predicted policies need the frozen head's cut function."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy}")
    if policy in ("predicted", "predicted_even") and cut_fn is None:
        raise ValueError(f"the {policy} policy needs the frozen head's cut function")
    tdu_cls, fact_cls = adapter_classes(mods)
    last = model.block_list[-1]
    if not isinstance(last, mods["blocks"].UpdateBlockTDU):
        raise TypeError("the last FACT block is not an UpdateBlockTDU")
    last.__class__, model.__class__ = tdu_cls, fact_cls
    last.policy, last.grid, last.pending_cuts, model.cut_fn = policy, int(step), None, cut_fn
    return model


def set_policy(model: nn.Module, policy: str) -> None:
    """Switch an adapted model's policy (the evaluation controls on frozen weights)."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy}")
    if policy in ("predicted", "predicted_even") and getattr(model, "cut_fn", None) is None:
        raise ValueError(f"the {policy} policy needs the frozen head's cut function")
    model.block_list[-1].policy = policy


def last_partition(model: nn.Module) -> list[tuple[int, int]]:
    """The half-open intervals the last block used in its latest forward."""
    return [(int(g.start), int(g.end) + 1) for g in model.block_list[-1].tdu.segs]
