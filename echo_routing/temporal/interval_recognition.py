"""Interval recognition (A27): interval pooling, the window (W) and interval (I) objectives with the shared own-teacher
retention term, the copied 512->5 recognition head and the Base-interval relabelling adapter.

An interval is a list of window rows with multiplicity weights that sum to 1 (each sample maps to its nearest request
centre). W averages the per-window log loss inside the interval; I classifies the interval's pooled probability. Both add
the weighted KL from the frozen original probabilities q to the new head's p.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from echo_routing.temporal.segments import labels_to_segments

N_CLASSES = 5


def runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """(start, end exclusive, class) of maximal equal-class runs."""
    return [(s.start, s.end, int(s.label)) for s in labels_to_segments(np.asarray(labels))]


def majority(ref: np.ndarray) -> tuple[int, float]:
    """Majority reference class (ties: smallest index) and its purity."""
    c = np.bincount(np.asarray(ref, int), minlength=N_CLASSES); k = int(c.argmax()); return k, float(c[k] / c.sum())


def interval_rows(nearest: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Unique window rows of samples [start, end) under the nearest-centre map, with multiplicity weights summing to 1."""
    rows, counts = np.unique(np.asarray(nearest[start:end]), return_counts=True); return rows.astype(np.int64), (counts / counts.sum()).astype(np.float64)


def build_intervals(ref: np.ndarray, base: np.ndarray, nearest: np.ndarray) -> dict:
    """T (reference runs) and P (Base runs labelled by the majority reference class) intervals of one sequence."""
    out = {}
    for name, lab in (("T", ref), ("P", base)):
        items = []
        for a, b, c in runs(lab):
            y, purity = (c, 1.0) if name == "T" else majority(ref[a:b]); r, w = interval_rows(nearest, a, b)
            items.append({"rows": r, "weights": w, "label": int(y), "length": b - a, "purity": purity})
        out[name] = items
    return out


def pack(items: list[dict]) -> dict:
    """Flat arrays of a list of intervals: concatenated rows/weights with offsets, labels, lengths, purities."""
    off = np.cumsum([0, *[len(x["rows"]) for x in items]]).astype(np.int64)
    return {"offsets": off, "rows": np.concatenate([x["rows"] for x in items]).astype(np.int64), "weights": np.concatenate([x["weights"] for x in items]),
            "labels": np.asarray([x["label"] for x in items], np.int64), "lengths": np.asarray([x["length"] for x in items], np.int64),
            "purity": np.asarray([x["purity"] for x in items], np.float64)}


def gather(packed: dict, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows, weights, interval index and labels of the chosen intervals (duplicates allowed)."""
    off = packed["offsets"]; rows, ws, ix = [], [], []
    for j, i in enumerate(np.asarray(idx, int)):
        a, b = off[i], off[i + 1]; rows.append(packed["rows"][a:b]); ws.append(packed["weights"][a:b]); ix.append(np.full(b - a, j))
    return (torch.from_numpy(np.concatenate(rows)), torch.from_numpy(np.concatenate(ws)).float(), torch.from_numpy(np.concatenate(ix)),
            torch.from_numpy(packed["labels"][np.asarray(idx, int)]))


def copied_head(weight: torch.Tensor, bias: torch.Tensor) -> nn.Linear:
    h = nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        h.weight.copy_(weight); h.bias.copy_(bias)
    return h


def objective(arm: str, z: torch.Tensor, q_logits: torch.Tensor, w: torch.Tensor, ix: torch.Tensor, y: torch.Tensor) -> dict:
    """z, q_logits: (rows, 5) new-head and original logits of the gathered rows; w their weights; ix their interval index;
    y the interval labels. Returns the classification term (W or I), the retention KL and their sum."""
    n = int(y.shape[0]); logp = F.log_softmax(z, -1); logq = F.log_softmax(q_logits.detach(), -1); lp_y = logp.gather(1, y[ix].unsqueeze(1)).squeeze(1)
    if arm == "W":
        cls = -torch.zeros(n, dtype=z.dtype).index_add(0, ix, w * lp_y)
    elif arm == "I":
        s = torch.log(w) + lp_y; m = torch.full((n,), -torch.inf, dtype=z.dtype).scatter_reduce(0, ix, s, reduce="amax", include_self=True)
        cls = -(m + torch.log(torch.zeros(n, dtype=z.dtype).index_add(0, ix, torch.exp(s - m[ix]))))
    else:
        raise ValueError(f"unknown arm {arm!r}")
    kl_t = (logq.exp() * (logq - logp)).sum(-1); kl = torch.zeros(n, dtype=z.dtype).index_add(0, ix, w * kl_t)
    return {"cls": cls.mean(), "kl": kl.mean(), "loss": cls.mean() + kl.mean()}


def relabel(base: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Each Base run takes the argmax of the mean new-head probability over its samples; equal neighbours merge by RLE."""
    out = np.empty(len(base), np.int64)
    for a, b, _ in runs(base):
        out[a:b] = int(np.asarray(p[a:b]).mean(0).argmax())
    return out
