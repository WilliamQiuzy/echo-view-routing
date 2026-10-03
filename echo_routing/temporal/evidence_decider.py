"""Round 5: an A4C/A5C decision on fixed spans from a frozen frame classifier's evidence
(configs/experiments/evidence_pooling_v1.yaml, section decider).

The visual encoder, the frame classifier and the temporal system stay frozen. A span keeps its label unless that label
is A4C or A5C; then a rule picks A4C or A5C from the classifier's per-sample probabilities inside the span:
- mean: the larger of the mean p(A4C) and the mean p(A5C);
- trained_mean: a small classifier on the five mean class probabilities;
- trained_distribution: the same classifier on fixed statistics of the span's probabilities.
Training spans come from the training split only, with every source clip carrying the same total weight and the two
classes balanced.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from echo_routing.evaluate.failure_units import stream_units

A4C, A5C, NUM_CLASSES = 2, 3, 5
JITTER, MIN_SPAN = 3, 5


def mean_features(prob: np.ndarray) -> np.ndarray:
    return np.asarray(prob, float).mean(0)


def distribution_features(prob: np.ndarray) -> np.ndarray:
    """The five mean class probabilities; percentiles 10/25/50/75/90 and the maximum of p(A5C); percentiles 10/50/90
    of p(A4C); the standard deviation of p(A5C); the shares of samples with p(A5C) above 0.5 and above 0.8."""
    prob = np.asarray(prob, float); p5, p4 = prob[:, A5C], prob[:, A4C]
    return np.concatenate([prob.mean(0), np.quantile(p5, [0.1, 0.25, 0.5, 0.75, 0.9]), [p5.max()], np.quantile(p4, [0.1, 0.5, 0.9]),
                           [p5.std(), (p5 > 0.5).mean(), (p5 > 0.8).mean()]])


FEATURES = {"trained_mean": mean_features, "trained_distribution": distribution_features}


def mean_rule(prob: np.ndarray) -> int:
    m = np.asarray(prob, float).mean(0)
    return A5C if m[A5C] > m[A4C] else A4C


def decide_spans(labels: np.ndarray, cuts: list[int], prob: np.ndarray, rule) -> np.ndarray:
    """Labels after the rule: each span between consecutive cuts whose (constant) label is A4C or A5C takes rule(prob
    of the span); every other span keeps its label. Equal neighbours merge by construction (labels are per sample)."""
    out = np.asarray(labels, int).copy(); edges = [0, *sorted(cuts), out.size]
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a and len(set(out[a:b].tolist())) != 1:   # the spans must be those of the decoding that produced labels
            raise ValueError(f"labels vary inside the span {a}-{b}: the spans do not match the decoding")
        if b > a and out[a] in (A4C, A5C):
            out[a:b] = int(rule(prob[a:b]))
    return out


def training_spans(streams: list[tuple], clips: list[tuple], seed: int) -> list[tuple[np.ndarray, int, str]]:
    """(probabilities, label, first source clip) of every training span.

    streams: (recipe, prob) of the training-bank streams; every A4C or A5C reference unit is used once exactly and once
    with each limit moved by an integer in [-JITTER, JITTER] (kept if at least MIN_SPAN samples remain and the reference
    majority is A4C or A5C, which becomes its label). clips: (video_id, label, prob) of training clips, one span each."""
    rng = np.random.default_rng(int(seed)); out = []
    for recipe, prob in streams:
        units = stream_units(recipe); ref = np.concatenate([np.full(u.length, u.label) for u in units]); n = ref.size
        for u in units:
            if u.label not in (A4C, A5C):
                continue
            out.append((prob[u.start:u.end], u.label, u.sources[0][0]))
            a = int(np.clip(u.start + rng.integers(-JITTER, JITTER + 1), 0, n)); b = int(np.clip(u.end + rng.integers(-JITTER, JITTER + 1), 0, n))
            if b - a >= MIN_SPAN:
                m = int(np.bincount(ref[a:b], minlength=NUM_CLASSES).argmax())
                if m in (A4C, A5C):
                    out.append((prob[a:b], m, u.sources[0][0]))
    out += [(prob, int(lab), vid) for vid, lab, prob in clips if int(lab) in (A4C, A5C)]
    return out


def span_weights(clip_ids: list[str], labels: np.ndarray) -> np.ndarray:
    """Every source clip carries the same total weight; then the two classes get equal totals (mean weight 1)."""
    ids = np.asarray(clip_ids); _, inv, cnt = np.unique(ids, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]; labels = np.asarray(labels)
    for c in (A4C, A5C):
        w[labels == c] *= 0.5 / w[labels == c].sum()
    return w * len(w)


class Decider:
    """One hidden layer of 32 ReLU units on standardised inputs; Adam (lr 0.01, weight decay 0.001), 300 full-batch
    epochs, weighted binary cross-entropy; A5C when the output probability is at least 0.5."""

    def __init__(self, seed: int, hidden: int = 32, epochs: int = 300, lr: float = 0.01, weight_decay: float = 0.001) -> None:
        self.seed, self.hidden, self.epochs, self.lr, self.weight_decay = int(seed), hidden, epochs, lr, weight_decay

    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> "Decider":
        x = np.asarray(x, np.float64); self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        torch.manual_seed(self.seed)
        self.net = nn.Sequential(nn.Linear(x.shape[1], self.hidden), nn.ReLU(), nn.Linear(self.hidden, 1))
        xt = torch.tensor((x - self.mu) / self.sd, dtype=torch.float32); yt = torch.tensor(np.asarray(y) == A5C, dtype=torch.float32)
        wt = torch.tensor(np.asarray(w), dtype=torch.float32); opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = (nn.functional.binary_cross_entropy_with_logits(self.net(xt)[:, 0], yt, reduction="none") * wt).mean()
            loss.backward(); opt.step()
        self.final_loss = float(loss.detach())
        return self

    @torch.no_grad()
    def prob_a5c(self, x: np.ndarray) -> np.ndarray:
        xt = torch.tensor((np.atleast_2d(np.asarray(x, np.float64)) - self.mu) / self.sd, dtype=torch.float32)
        return torch.sigmoid(self.net(xt)[:, 0]).numpy()

    def rule(self, features):
        return lambda prob: A5C if self.prob_a5c(features(prob))[0] >= 0.5 else A4C
