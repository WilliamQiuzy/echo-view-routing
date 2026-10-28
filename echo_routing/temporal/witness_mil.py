"""A43 (configs/experiments/video_a5c_witness_mil_v1.yaml): native-recording bags of a fit pool, the fixed 600-update draw
plan, the binary A5C witness specialist read from A08's final local tokens, the dense (D) and multiple-instance (M) bag
objectives with float64 references, and the one-way Base-A4C -> A5C deployment edit.

Bags. Every native sequence whose single clip code is A4C or A5C, two lists in pool order; a bag is all its stride-5
request windows in pool order. v_j = (samples whose nearest centre is window j) / (samples). Only kinds, clip codes and
the window grid are read.
Plan. Update u draws, from default_rng([20261027, 43, s, f, u]), an A4C index then an A5C index (uniform); one A4C bag then
one A5C bag. Nothing reads a model output, a Base mistake or a held-out pool.
Specialist. The 49 spatial tokens of encoder.norm's output at temporal index marker_slot // 2 (token centres 0, 2, ..., 14,
ties to the earlier); a = softmax_i(w_a^T [tanh(V h_i + b_v) * sigmoid(U h_i + b_u)]), e = sum_i a_i h_i, s = w_c^T e + b_c.
The scorer starts at weight 0, bias -2: every initial score is -2.
Losses. A4C: sum_j v_j softplus(s_j) (both arms). A5C: D sum_j v_j softplus(-s_j), M softplus(-amax_j s_j).
Total pi4 * A4C + pi5 * A5C.
Deployment. A Base run labelled A4C with r = max score over its unique rows > 0 becomes A5C; every other label is copied.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from echo_routing.evaluate.window_target import nearest_centre

BASE, ROUND = 20261027, 43
A4C, A5C = 2, 3
GRID, DIM, HID, INIT_BIAS = (8, 7, 7), 768, 128, -2.0
N_SPATIAL, N_TOKENS = GRID[1] * GRID[2], 1 + GRID[0] * GRID[1] * GRID[2]
POSITIVE = {"D": "dense", "M": "max"}
CONTRACT = ("native A4C / A5C bags = all stride-5 windows, v_j = nearest-centre samples / samples; plan rng [20261027, 43, s, f, u]: "
            "A4C index then A5C index; tokens = encoder.norm[:, 1:] -> (8, 49) at t = marker // 2; gated attention V, U 768 -> 128, "
            "w_a 128 -> 1 no bias, scorer 768 -> 1 init 0 / -2; L = pi4 sum v softplus(s4) + pi5 (D: sum v softplus(-s5), M: softplus(-amax s5)); "
            "deploy: Base A4C run with max score over its unique rows > 0 -> A5C")


# ----------------------------------------------------------------------------------------------- bags and plan
def duration_weights(n: int, centres: np.ndarray) -> np.ndarray:
    """Per window: the samples whose nearest centre it is, over all n samples (a window without samples keeps weight 0)."""
    return np.bincount(nearest_centre(int(n), np.asarray(centres)), minlength=len(centres)).astype(np.float64) / int(n)


def native_bags(seqs: list[dict], W: dict) -> dict:
    """Pool-order lists of native A4C and A5C sequences; per sequence its pool_W rows (pool order) and duration weights."""
    out = {"a4c": [], "a5c": [], "rows": {}, "weights": {}}
    for i, q in enumerate(seqs):
        if q["kind"] != "native":
            continue
        codes = set(np.asarray(q["labels"]).tolist())
        if len(codes) != 1:
            raise ValueError("a native sequence carries more than one clip code")
        c = codes.pop()
        if c not in (A4C, A5C):
            continue
        r = np.flatnonzero(W["seq"] == i); out["a4c" if c == A4C else "a5c"].append(i)
        out["rows"][i], out["weights"][i] = r, duration_weights(len(q["labels"]), W["centre"][r])
    return out


def draw(n4: int, n5: int, s: int, f: int, u: int) -> tuple[int, int]:
    rng = np.random.default_rng([BASE, ROUND, int(s), int(f), int(u)]); i4 = int(rng.integers(n4)); return i4, int(rng.integers(n5))


def plan(bags: dict, s: int, f: int, updates: int) -> np.ndarray:
    """(updates, 2) pool sequence indices: the A4C bag and the A5C bag of each update (the same for both arms)."""
    a4, a5 = bags["a4c"], bags["a5c"]
    return np.asarray([(a4[i], a5[j]) for i, j in (draw(len(a4), len(a5), s, f, u) for u in range(updates))], np.int64)


# ----------------------------------------------------------------------------------------------- specialist
def token_time(slot: int) -> int:
    """Temporal token index of a marker slot: nearest token centre 2t, ties to the earlier."""
    return int(slot) // 2


def local_tokens(u: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """(B, 393, C) norm output and (B,) marker slots -> (B, 49, C) spatial tokens at t = slot // 2 (graph kept)."""
    if u.ndim != 3 or u.shape[1] != N_TOKENS or slots.shape != (u.shape[0],):
        raise ValueError(f"tokens {tuple(u.shape)} / slots {tuple(slots.shape)} do not fit")
    g = u[:, 1:].reshape(u.shape[0], GRID[0], N_SPATIAL, u.shape[-1])
    return g[torch.arange(u.shape[0], device=u.device), torch.div(slots.to(u.device), 2, rounding_mode="floor")]


class GatedWitness(nn.Module):
    """Gated attention over the 49 local tokens and a scalar A5C witness score; 197,761 parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.V, self.U, self.w = nn.Linear(DIM, HID), nn.Linear(DIM, HID), nn.Linear(HID, 1, bias=False)
        self.scorer = nn.Linear(DIM, 1)
        with torch.no_grad():
            self.scorer.weight.zero_(); self.scorer.bias.fill_(INIT_BIAS)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 49, 768) -> scores (B,) and attention (B, 49)."""
        a = torch.softmax(self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1), dim=1)
        return self.scorer((a.unsqueeze(-1) * h).sum(1)).squeeze(-1), a


def score(m, head: GatedWitness, rgb: torch.Tensor, marker: torch.Tensor, slots: torch.Tensor, embed_with_tokens) -> torch.Tensor:
    """Witness scores (B,) of one full original marker forward; `embed_with_tokens` is A40's hooked embed (hook removed in finally)."""
    _, u = embed_with_tokens(m, rgb, marker); return head(local_tokens(u, slots))[0]


# ----------------------------------------------------------------------------------------------- losses
def bag_loss(s4: torch.Tensor, v4: torch.Tensor, s5: torch.Tensor, v5: torch.Tensor, pi4: float, pi5: float, positive: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(total, A4C loss, A5C loss) of one A4C bag and one A5C bag; positive 'dense' (D) or 'max' (M)."""
    if s4.ndim != 1 or s5.ndim != 1 or s4.shape != v4.shape or s5.shape != v5.shape or not len(s4) or not len(s5):
        raise ValueError("expected non-empty score / weight vectors of equal length")
    l4 = (v4 * F.softplus(s4)).sum()
    if positive == "dense":
        l5 = (v5 * F.softplus(-s5)).sum()
    elif positive == "max":
        l5 = F.softplus(-torch.amax(s5))
    else:
        raise ValueError(f"unknown positive supervision {positive!r}")
    return pi4 * l4 + pi5 * l5, l4, l5


def reference_loss(s4, v4, s5, v5, pi4: float, pi5: float, positive: str) -> dict:
    """Independent float64 loop: both bag losses, the total and dL/ds of every window (exact max ties share equally)."""
    s4, v4, s5, v5 = (np.asarray(x, np.float64) for x in (s4, v4, s5, v5)); sig = lambda x: 0.5 * (1.0 + np.tanh(0.5 * x))  # noqa: E731
    l4, g4 = 0.0, np.zeros(len(s4))
    for j in range(len(s4)):
        l4 += v4[j] * np.logaddexp(0.0, s4[j]); g4[j] = pi4 * v4[j] * sig(s4[j])
    l5, g5 = 0.0, np.zeros(len(s5))
    if positive == "dense":
        for j in range(len(s5)):
            l5 += v5[j] * np.logaddexp(0.0, -s5[j]); g5[j] = -pi5 * v5[j] * sig(-s5[j])
    else:
        mx = max(float(x) for x in s5); ties = [j for j in range(len(s5)) if s5[j] == mx]; l5 = np.logaddexp(0.0, -mx)
        for j in ties:
            g5[j] = -pi5 * sig(-mx) / len(ties)
    return {"l4": l4, "l5": l5, "total": pi4 * l4 + pi5 * l5, "g4": g4, "g5": g5}


def reference_head(h: np.ndarray, sd: dict) -> tuple[float, np.ndarray]:
    """Float64 loop of the gated attention and score for one (49, 768) token set; sd = the head state as numpy arrays."""
    h = np.asarray(h, np.float64); logits = np.zeros(len(h))
    for i in range(len(h)):
        a_ = np.tanh(sd["V.weight"] @ h[i] + sd["V.bias"]); b_ = 1.0 / (1.0 + np.exp(-(sd["U.weight"] @ h[i] + sd["U.bias"]))); logits[i] = float(sd["w.weight"][0] @ (a_ * b_))
    a = np.exp(logits - logits.max()); a /= a.sum(); e = (a[:, None] * h).sum(0)
    return float(sd["scorer.weight"][0] @ e + sd["scorer.bias"][0]), a


# ----------------------------------------------------------------------------------------------- deployment
def run_max(pk: dict, scores: np.ndarray, off: int, eligible: np.ndarray) -> np.ndarray:
    """r per P run: the max specialist score over the run's unique request rows (fold-local = global - off); -inf if not eligible."""
    r = np.full(len(pk["labels"]), -np.inf); o = pk["offsets"]
    for i in np.flatnonzero(eligible):
        rows = np.unique(pk["rows"][o[i]:o[i + 1]]) - off; r[i] = float(np.max(scores[rows]))
    return r


def one_way(base: np.ndarray, r: np.ndarray) -> np.ndarray:
    """New run labels: Base A4C runs with r > 0 become A5C; every other label is copied exactly."""
    out = np.asarray(base).copy(); out[(out == A4C) & (np.asarray(r) > 0)] = A5C; return out


def edit_sequences(base_preds: dict, pk: dict, seqs: list[dict], pred: np.ndarray) -> dict:
    """Per-sample sequence labels: Base's, with each changed run's samples [a, b) set to its new label (RLE merge is implicit)."""
    out = {k: np.asarray(v).copy() for k, v in base_preds.items()}
    for i in np.flatnonzero(np.asarray(pred) != pk["base"]):
        out[seqs[int(pk["seq"][i])]["id"]][int(pk["a"][i]):int(pk["b"][i])] = int(pred[i])
    return out
