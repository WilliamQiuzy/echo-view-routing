"""A34 (configs/experiments/video_native_bag_evidence_v1.yaml): power-mean bag objective and readout.

For a bag of windows t with weights w_t >= 0 (sum 1) and window class probabilities p_tc = softmax(z_t)_c:
    g_c(r) = (sum_t w_t p_tc^r)^(1/r),   B_c(r) = g_c(r) / sum_k g_k(r),   L_r = -log B_y(r).
r = 1 is the weighted mean-probability CE; larger r lets a few windows with high p_tc carry the bag. Everything is
computed in log space: log g_c = (1/r) logsumexp_t(log w_t + r log p_tc), with zero-weight windows removed.
"""
from __future__ import annotations

import numpy as np
import torch


def bag_log_g(z: torch.Tensor, w: torch.Tensor, ix: torch.Tensor, n_bags: int, r: float) -> torch.Tensor:
    """(n_bags, C) log g_c(r) from window logits z (T, C), weights w (T,) and bag indices ix (T,)."""
    keep = w > 0
    z, w, ix = z[keep], w[keep], ix[keep]
    s = torch.log(w).unsqueeze(1) + float(r) * torch.log_softmax(z, -1)
    m = torch.full((n_bags, z.shape[1]), -torch.inf, dtype=s.dtype, device=s.device).scatter_reduce(0, ix.unsqueeze(1).expand_as(s), s, reduce="amax", include_self=True)
    tot = torch.zeros(n_bags, z.shape[1], dtype=s.dtype, device=s.device).index_add(0, ix, torch.exp(s - m[ix].detach()))
    return (m.detach() + torch.log(tot)) / float(r)


def bag_loss(z: torch.Tensor, w: torch.Tensor, ix: torch.Tensor, y: torch.Tensor, r: float) -> torch.Tensor:
    """Per-bag L_r = -log B_y(r)."""
    lg = bag_log_g(z, w, ix, int(y.shape[0]), r)
    return -(lg - torch.logsumexp(lg, -1, keepdim=True)).gather(1, y.unsqueeze(1)).squeeze(1)


def bag_scores(z: torch.Tensor, w: torch.Tensor, ix: torch.Tensor, n_bags: int, r: float) -> torch.Tensor:
    """(n_bags, C) normalised B(r): the readout of a bag or an interval under aggregation r."""
    with torch.no_grad():
        lg = bag_log_g(z.double(), w.double(), ix, n_bags, r); return torch.softmax(lg, -1)


def numpy_loss(z: np.ndarray, w: np.ndarray, y: int, r: float) -> float:
    """Independent restatement for one bag (direct powers, float64)."""
    e = np.exp(z - z.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True); g = (w[:, None] * p ** r).sum(0) ** (1.0 / r)
    return float(-np.log(g[y] / g.sum()))


def relabel_r(base: np.ndarray, p: np.ndarray, r: float, runs) -> np.ndarray:
    """Each Base run takes argmax_c (mean over its samples of p_sc^r)^(1/r); equal neighbours merge by RLE."""
    out = np.empty(len(base), np.int64)
    for a, b, _ in runs(base):
        out[a:b] = int((np.asarray(p[a:b], np.float64) ** r).mean(0).argmax())
    return out
