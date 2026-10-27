"""A42 (configs/experiments/video_session_matched_discrimination_v1.yaml): the native-only same-session A4C / A5C pair
bank of a fit pool, the extra 8 windows of an update, the S / C partner maps, the logistic margin-ranking loss added to
CE, float64 references for the checks, and a gather max pool with a deterministic backward.

Bank. A session qualifies if the pool holds at least one native A4C and one native A5C recording of it. Sessions are in
sorted id order, recordings of a session and class in pool order. Only kinds, recording ids and clip codes are read.

Update. Slots 0-7 are A40's original batch; slots 8-11 the A4C windows of four distinct chosen sessions, slots 12-15
their A5C windows in the same session order. d(x) = logit_A5C - logit_A4C. S pairs d5_i with d4_i (same session), C
with d4_((i + 1) mod 4) (another chosen session). L = mean_16 CE + 0.5 x mean_i softplus(-(d5_i - d4_partner(i))).
Nothing here reads a model output, a Base mistake or a held-out pool.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE, ROUND = 20261027, 42
A4C, A5C = 2, 3
N_PAIRS, PER, COEF = 4, 16, 0.5
EXTRA4, EXTRA5 = slice(8, 12), slice(12, 16)
PARTNER = {"S": (0, 1, 2, 3), "C": (1, 2, 3, 0)}
CONTRACT = ("slots 0-7 A40 batch (rng [20261027, 42, s, f, u, 1]); slots 8-11 A4C / 12-15 A5C of 4 distinct bank sessions "
            "(rng [20261027, 42, s, f, u, 2]: sessions, then per session A4C recording, A5C recording, A4C centre, A5C centre); "
            "L = mean_16 CE + 0.5 mean_i softplus(-(d5_i - d4_partner(i))), d = z[3] - z[2], S partner i, C partner (i + 1) mod 4")


# ----------------------------------------------------------------------------------------------- bank and draw
def pair_bank(seqs: list[dict], session_of) -> dict:
    """Sorted sessions with >= 1 native A4C and >= 1 native A5C sequence; per session the pool indices of each class."""
    per: dict = {}
    for i, q in enumerate(seqs):
        if q["kind"] != "native":
            continue
        codes = set(np.asarray(q["labels"]).tolist())
        if len(codes) != 1:
            raise ValueError("a native sequence carries more than one clip code")
        c = codes.pop()
        if c in (A4C, A5C):
            per.setdefault(session_of(q["id"]), {A4C: [], A5C: []})[c].append(i)
    sess = sorted(s for s, d in per.items() if d[A4C] and d[A5C])
    return {"sessions": sess, "a4c": [list(per[s][A4C]) for s in sess], "a5c": [list(per[s][A5C]) for s in sess]}


def bank_counts(bank: dict, seqs: list[dict]) -> dict:
    return {"native_recordings": int(sum(q["kind"] == "native" for q in seqs)), "sessions": len(bank["sessions"]),
            "A4C_recordings": int(sum(len(x) for x in bank["a4c"])), "A5C_recordings": int(sum(len(x) for x in bank["a5c"])),
            "pairs": int(sum(len(a) * len(b) for a, b in zip(bank["a4c"], bank["a5c"])))}


def extra_draw(bank: dict, rows_of: list, s: int, f: int, u: int) -> tuple[np.ndarray, np.ndarray]:
    """Pool_W rows of slots 8-15 (four A4C windows, then their same-session A5C windows) and the chosen bank sessions."""
    rng = np.random.default_rng([BASE, ROUND, int(s), int(f), int(u), 2]); ch = rng.choice(len(bank["sessions"]), N_PAIRS, replace=False); w4, w5 = [], []
    for k in ch:
        i4 = bank["a4c"][k][int(rng.integers(len(bank["a4c"][k])))]; i5 = bank["a5c"][k][int(rng.integers(len(bank["a5c"][k])))]
        r4, r5 = rows_of[i4], rows_of[i5]; w4.append(int(r4[int(rng.integers(len(r4)))])); w5.append(int(r5[int(rng.integers(len(r5)))]))
    return np.asarray(w4 + w5, np.int64), np.asarray(ch, np.int64)


# ----------------------------------------------------------------------------------------------- losses
def margins(z: torch.Tensor) -> torch.Tensor:
    return z[:, A5C] - z[:, A4C]


def rank_loss(z: torch.Tensor, partner) -> torch.Tensor:
    """mean_i softplus(-(d5_i - d4_partner(i))) over the 16-window logits z; partner indexes by basic selection."""
    d4, d5 = margins(z[EXTRA4]), margins(z[EXTRA5]); p = torch.stack([d4[int(j)] for j in partner])
    return F.softplus(-(d5 - p)).mean()


def total_loss(z: torch.Tensor, y: torch.Tensor, partner) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if z.shape != (PER, 5) or y.shape != (PER,):
        raise ValueError(f"expected 16 x 5 logits and 16 labels, got {tuple(z.shape)} / {tuple(y.shape)}")
    ce = F.cross_entropy(z, y); rk = rank_loss(z, partner); return ce + COEF * rk, ce, rk


def reference(z: np.ndarray, y: np.ndarray, partner) -> dict:
    """Independent float64 loop: CE (denominator 16), rank and total, and the gradient of each w.r.t. the logits."""
    z = np.asarray(z, np.float64); n = len(z); gce, grk = np.zeros_like(z), np.zeros_like(z); ce = 0.0; rk = 0.0
    for w in range(n):
        m = z[w].max(); lse = m + np.log(np.exp(z[w] - m).sum()); ce += (lse - z[w, int(y[w])]) / n
        gce[w] = np.exp(z[w] - lse) / n; gce[w, int(y[w])] -= 1.0 / n
    for i in range(N_PAIRS):
        a, b = 12 + i, 8 + int(partner[i]); x = -((z[a, A5C] - z[a, A4C]) - (z[b, A5C] - z[b, A4C]))
        rk += np.logaddexp(0.0, x) / N_PAIRS; sg = 0.5 * (1.0 + np.tanh(0.5 * x)) / N_PAIRS
        grk[a, A5C] -= sg; grk[a, A4C] += sg; grk[b, A5C] += sg; grk[b, A4C] -= sg
    return {"ce": ce, "rank": rk, "total": ce + COEF * rk, "g_ce": gce, "g_rank": grk, "g_total": gce + COEF * grk}


# ----------------------------------------------------------------------------------------------- deterministic skip pool
class GatherMaxPool3d(nn.Module):
    """nn.MaxPool3d's output and argmax routing with a deterministic backward: indices from F.max_pool3d under no_grad,
    values gathered from the input. No parameters or buffers."""

    def __init__(self, mp: nn.MaxPool3d) -> None:
        super().__init__()
        if mp.return_indices:
            raise ValueError("the skip pool must not return indices")
        self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode = mp.kernel_size, mp.stride, mp.padding, mp.dilation, mp.ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, idx = F.max_pool3d(x, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, return_indices=True)
        return x.flatten(2).gather(2, idx.flatten(2)).view(idx.shape)


def deterministic_skip_pools(encoder: nn.Module) -> list[int]:
    """Replace every MultiscaleBlock's residual nn.MaxPool3d (pool_skip.pool) by GatherMaxPool3d on this training copy."""
    done = []
    for i, b in enumerate(encoder.blocks):
        if b.pool_skip is not None:
            if not isinstance(b.pool_skip.pool, nn.MaxPool3d):
                raise TypeError(f"block {i}: unexpected skip pool {type(b.pool_skip.pool).__name__}")
            b.pool_skip.pool = GatherMaxPool3d(b.pool_skip.pool); done.append(i)
    return done
