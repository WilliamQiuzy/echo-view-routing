"""Round 6: an A4C/A5C classifier trained on actual short spans (configs/experiments/span_classifier_v1.yaml).

Spans come from a frozen temporal system (E1 ASRF candidate spans). A span whose vote is A4C or A5C is classified from
the mean of its frames' visual embeddings by a small head; every other span keeps its vote, and equal neighbours merge
afterwards. V0 trains the head on a frozen encoder; V1 also trains the encoder's last block, ln_post and proj.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from echo_routing.temporal.windowed import SampleRef

A4C, A5C, NUM_CLASSES = 2, 3, 5
CLASSES2 = (A4C, A5C)


@dataclass(frozen=True)
class SpanRecord:
    seq_id: str
    start: int
    end: int
    vote: int                       # the frozen system's class for the span (A4C or A5C)
    label: int                      # reference majority class, or -1 when it is neither A4C nor A5C
    purity: float                   # share of the span's samples with the majority class
    source: str                     # the clip covering most of the span's samples
    refs: tuple[SampleRef, ...]     # the frames used: at most `cap`, evenly spaced

    @property
    def length(self) -> int:
        return self.end - self.start


def even_cap(n: int, cap: int) -> np.ndarray:
    """Indices of at most `cap` evenly spaced samples of n (all of them when n <= cap)."""
    return np.arange(n) if n <= cap else np.unique(np.linspace(0, n - 1, cap).round().astype(int))


def span_records(seq_id: str, final: np.ndarray, cuts: list[int], ref: np.ndarray, refs: list[SampleRef], cap: int) -> list[SpanRecord]:
    """Records of the spans between consecutive cuts whose (constant) vote is A4C or A5C."""
    final = np.asarray(final, int); ref = np.asarray(ref, int); edges = [0, *sorted(cuts), final.size]; out = []
    if len(refs) != final.size or ref.size != final.size:
        raise ValueError(f"{seq_id}: labels, references and frames differ in length")
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        if len(set(final[a:b].tolist())) != 1:
            raise ValueError(f"{seq_id}: the vote varies inside the span {a}-{b}")
        if final[a] not in CLASSES2:
            continue
        counts = np.bincount(ref[a:b], minlength=NUM_CLASSES); maj = int(counts.argmax())
        source = Counter(r.video_id for r in refs[a:b]).most_common(1)[0][0]
        picks = even_cap(b - a, cap) + a
        out.append(SpanRecord(seq_id, a, b, int(final[a]), maj if maj in CLASSES2 else -1, float(counts.max() / (b - a)), source,
                              tuple(refs[i] for i in picks)))
    return out


def apply_decisions(final: np.ndarray, records: list[SpanRecord], decisions: list[int]) -> np.ndarray:
    """Labels after the classifier: each record's span takes its decision (A4C or A5C); nothing else changes."""
    out = np.asarray(final, int).copy()
    for r, d in zip(records, decisions):
        if d not in CLASSES2:
            raise ValueError(f"decision {d} is not A4C or A5C")
        out[r.start:r.end] = int(d)
    return out


class SpanHead(nn.Module):
    """Linear(512, 128), ReLU, Linear(128, 2): logits for (A4C, A5C)."""

    def __init__(self, dim: int = 512, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def span_means(emb: torch.Tensor, counts: list[int]) -> torch.Tensor:
    """(sum(counts), D) frame embeddings -> (len(counts), D) span means, in order."""
    return torch.stack([c.mean(0) for c in torch.split(emb, counts)])


def epoch_order(n: int, seed: int, epoch: int) -> np.ndarray:
    """The span order of one epoch; it depends only on (seed, epoch), so V0 and V1 see the same batches."""
    return np.random.default_rng([int(seed), int(epoch), 6]).permutation(n)


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warm-up over `warmup` updates, then cosine to zero at `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def target_index(labels: np.ndarray) -> np.ndarray:
    """A4C -> 0, A5C -> 1."""
    labels = np.asarray(labels)
    if not np.isin(labels, CLASSES2).all():
        raise ValueError("training labels must be A4C or A5C")
    return (labels == A5C).astype(np.int64)
