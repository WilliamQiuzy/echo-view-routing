"""Sequences, per-epoch batches and validation sets for the round-1 temporal heads (2026-10-02 plan, Tasks 3-4).

Every head (M0-M4) trains on the same batches. Each update reads `streams_per_batch` constructed training streams
(the fixed 1,500-stream bank, reshuffled each epoch) and `natives_per_batch` native-clip windows (clips re-drawn each
epoch; at most `native_window` samples with a uniform start in longer clips). The batch plan depends only on
(seed, epoch), never on the model. Recipe joins and labels are training targets and metric inputs only; models
receive features and a padding mask.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from echo_routing.compose.recipe_builder import Recipe, recipe_from_dict
from echo_routing.compose.stream_assembler import Stream, render
from echo_routing.features.cache import VideoFeatures, load_video_features, variant_dir
from echo_routing.temporal.losses.boundary_context import boundary_targets

PAD_LABEL = -100


@dataclass(frozen=True)
class Seq:
    seq_id: str
    is_native: bool
    feat: np.ndarray      # (T, D)
    labels: np.ndarray    # (T,) proxy (stream) or clip label (native)
    joins: np.ndarray     # recipe joins; empty for native clips and windows
    prob: np.ndarray | None = None   # (T, C) the frozen frame-level classifier's probabilities (round 3); never a model input


class FeatureStore:
    """Memoised cached arrays of one encoder (feature-cache key)."""

    def __init__(self, cache_root: Path, ckpt_hash: str) -> None:
        self.cache_root, self.ckpt_hash = Path(cache_root), ckpt_hash
        self._cache: dict[tuple[str, str], VideoFeatures] = {}

    def get(self, video_id: str, variant: str) -> VideoFeatures:
        key = (video_id, variant)
        if key not in self._cache:
            self._cache[key] = load_video_features(variant_dir(self.cache_root, self.ckpt_hash, variant) / f"{video_id}.npz")
        return self._cache[key]


def stream_seq(stream: Stream) -> Seq:
    return Seq(stream.recipe.recipe_id, False, stream.feat.astype(np.float16), stream.labels.astype(np.int64), stream.joins.astype(int),
               stream.prob.astype(np.float32))


def native_seq(video_id: str, label: int, vf: VideoFeatures, start: int = 0, length: int | None = None, seq_id: str | None = None) -> Seq:
    end = vf.n if length is None else start + length
    return Seq(seq_id or video_id, True, vf.feat[start:end].astype(np.float16), np.full(end - start, int(label), np.int64), np.zeros(0, int),
               vf.prob[start:end].astype(np.float32))


@dataclass(frozen=True)
class TrainData:
    streams: tuple[Seq, ...]
    clips: tuple[tuple[str, int], ...]   # (video_id, label) of training clips
    store: FeatureStore

    def epoch_batches(self, seed: int, epoch: int, n_native: int, window: int, streams_per_batch: int,
                      natives_per_batch: int) -> list[list[Seq]]:
        """Deterministic batches of one epoch: all streams in a fresh order plus `n_native` native windows."""
        rng = np.random.default_rng([int(seed), int(epoch)])
        order = rng.permutation(len(self.streams))
        picks = rng.choice(len(self.clips), size=n_native, replace=n_native > len(self.clips))
        natives = []
        for i in picks:
            vid, lab = self.clips[int(i)]
            vf = self.store.get(vid, "orig"); length = min(window, vf.n)
            start = int(rng.integers(0, vf.n - length + 1))
            natives.append(native_seq(vid, lab, vf, start, length, f"{vid}@{start}"))
        n_batches = max(math.ceil(len(order) / streams_per_batch), math.ceil(n_native / natives_per_batch))
        return [[self.streams[int(k)] for k in order[b * streams_per_batch:(b + 1) * streams_per_batch]]
                + natives[b * natives_per_batch:(b + 1) * natives_per_batch] for b in range(n_batches)]


def collate(seqs: list[Seq], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Pad to the longest sequence. Returns x (B, D, T), mask (B, 1, T), valid (B, T), labels (B, T) with PAD_LABEL at
    padding, is_native (B,), the boundary target/weight/counted arrays (B, T) and lengths (B,)."""
    n = len(seqs); t = max(s.labels.size for s in seqs); d = seqs[0].feat.shape[1]
    x = np.zeros((n, d, t), np.float32); labels = np.full((n, t), PAD_LABEL, np.int64)
    valid = np.zeros((n, t), np.float32)
    bt = np.zeros((n, t), np.float32); bw = np.zeros((n, t), np.float32); bc = np.zeros((n, t), bool)
    for i, s in enumerate(seqs):
        m = s.labels.size
        x[i, :, :m] = s.feat.T; labels[i, :m] = s.labels; valid[i, :m] = 1.0
        bt[i, :m], bw[i, :m], bc[i, :m] = boundary_targets(s.labels, s.joins)
    out = {"x": x, "labels": labels, "valid": valid, "b_target": bt, "b_weight": bw, "b_counted": bc,
           "is_native": np.array([s.is_native for s in seqs]), "lengths": np.array([s.labels.size for s in seqs])}
    out = {k: torch.from_numpy(v).to(device) for k, v in out.items()}
    out["mask"] = out["valid"][:, None, :]
    return out


@dataclass(frozen=True)
class EvalSet:
    """One validation half: native clips (whole clips) and the half's pair and multi-fragment streams."""
    name: str
    natives: tuple[Seq, ...]
    pairs: tuple[Stream, ...]
    multi: tuple[Stream, ...]

    def sequences(self) -> list[Seq]:
        return [*self.natives, *(stream_seq(s) for s in self.pairs), *(stream_seq(s) for s in self.multi)]


def recipes_of(bank_record: dict) -> list[Recipe]:
    return [recipe_from_dict(r) for r in bank_record["recipes"]]


def eval_set(proto: dict, half: str, labels: dict[str, int], store: FeatureStore) -> EvalSet:
    """Validation half `half` of a protocol.json: its native clips and its own pair and multi banks."""
    banks = proto["validation"][half]["banks"]
    natives = tuple(native_seq(v, labels[v], store.get(v, "orig")) for v in proto["validation"][half]["clips"])
    pairs = tuple(render(r, store.get) for r in recipes_of(banks["pairs"]))
    multi = tuple(render(r, store.get) for r in recipes_of(banks["multi"]))
    return EvalSet(half, natives, pairs, multi)


def feature_mask_augment(stream_prob: float, span: tuple[int, int], max_fraction: float, min_gap_to_change: int):
    """Training-only occlusion (Task 4, second step): with probability `stream_prob` per constructed stream, zero the
    features of one contiguous span of span[0]..span[1] samples inside one source fragment, covering at most
    `max_fraction` of that fragment and staying `min_gap_to_change` samples away from every join where the label
    changes. Native windows and all evaluation inputs are left unchanged."""
    lo, hi = int(span[0]), int(span[1])

    def apply(seq: Seq, rng: np.random.Generator) -> Seq:
        if seq.is_native or rng.random() >= stream_prob:
            return seq
        n = seq.labels.size
        edges = [0, *[int(j) for j in seq.joins], n]
        changes = [int(j) for j in seq.joins if seq.labels[j] != seq.labels[j - 1]]
        options = []
        for a, b in zip(edges[:-1], edges[1:]):
            top = min(hi, int(np.floor(max_fraction * (b - a))))
            for length in range(lo, top + 1):
                for start in range(a, b - length + 1):
                    if all(start + length <= c - min_gap_to_change or start >= c + min_gap_to_change for c in changes):
                        options.append((start, length))
        if not options:
            return seq
        start, length = options[int(rng.integers(len(options)))]
        feat = seq.feat.copy(); feat[start:start + length] = 0
        return Seq(seq.seq_id, seq.is_native, feat, seq.labels, seq.joins, seq.prob)

    return apply
