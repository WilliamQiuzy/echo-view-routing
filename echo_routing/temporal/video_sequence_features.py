"""Whole-sequence video features for temporal heads (configs/experiments/video_feature_temporal_v1.yaml).

A sequence is a list of 10 Hz references (recording, 30-fps frame index, gamma). Native clips are their whole orig
recording; constructed streams are assembled from their recipe first, so stride-5 windows are taken on the whole stream
and cross its joins with each fragment's own gamma. Every window's 16 references and marker go through a frozen video
model's embed; every sample then takes its nearest centre's 512-d window feature, in float32. Labels, joins and the native
flag never enter a feature. The batch adapter reproduces TrainData.epoch_batches (the same RNG, order and crops) with
float32 features, and native crops are slices of the whole recording's features.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from echo_routing.evaluate.window_target import nearest_centre, window_frames
from echo_routing.features.extract import VARIANT_GAMMA
from echo_routing.temporal.dev_data import Seq
from echo_routing.temporal.windowed import window_centres
from echo_routing.vit.target_mvit import marker_slot

STRIDE, SIZE, FRAMES = 5, 16, 16
Ref = tuple[str, int, float]

__all__ = ["Seq", "native_refs", "stream_refs", "stream_labels_joins", "label_changes", "windows", "map_to_samples",
           "sequence_features", "refs_digest", "Batches", "training_index"]


# ----------------------------------------------------------------------------------------------- references
def native_refs(video_id: str, frame_idx: Sequence[int]) -> list[Ref]:
    """A whole orig recording: one reference per cached 10 Hz sample."""
    return [(str(video_id), int(f), 1.0) for f in frame_idx]


def stream_refs(fragments, frame_idx_of: Callable[[str], np.ndarray]) -> list[Ref]:
    """A constructed stream's per-sample references, fragment by fragment (as windowed.stream_plan builds them)."""
    out: list[Ref] = []
    for fr in fragments:
        fidx = frame_idx_of(fr.video_id); g = float(VARIANT_GAMMA[fr.variant])
        out.extend((str(fr.video_id), int(fidx[s]), g) for s in range(int(fr.start), int(fr.end)))
    return out


def stream_labels_joins(fragments) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample labels and the join positions (every fragment start after the first, same-view joins included)."""
    labels = np.concatenate([np.full(int(fr.end) - int(fr.start), int(fr.label), np.int64) for fr in fragments])
    joins = np.cumsum([int(fr.end) - int(fr.start) for fr in fragments])[:-1].astype(np.int64)
    return labels, joins


def label_changes(labels: np.ndarray) -> np.ndarray:
    """Positions where the label changes: the true boundaries (a same-view join is not one)."""
    lab = np.asarray(labels)
    return np.flatnonzero(lab[1:] != lab[:-1]) + 1


def refs_digest(refs: Sequence[Ref]) -> str:
    h = hashlib.sha256()
    for v, f, g in refs:
        h.update(f"{v}|{int(f)}|{float(g):.6f};".encode())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- windows and mapping
def windows(n: int) -> dict[str, np.ndarray]:
    """The source round's stride-5 windows of an n-sample sequence: centres, the 16 sample indices and the marker slot."""
    centres = window_centres(int(n), STRIDE)
    samples = np.stack([window_frames(int(c), int(n), SIZE, FRAMES) for c in centres]) if centres.size else np.zeros((0, FRAMES), int)
    markers = np.array([marker_slot(s, int(c)) for s, c in zip(samples, centres)], dtype=np.int64)
    return {"centres": centres.astype(np.int64), "samples": samples.astype(np.int64), "markers": markers}


def map_to_samples(n: int, centres: np.ndarray, wfeat: np.ndarray) -> np.ndarray:
    """(n, D) float32: every sample takes its nearest centre's window feature (ties to the earlier centre)."""
    return np.ascontiguousarray(np.asarray(wfeat, np.float32)[nearest_centre(int(n), np.asarray(centres))])


def sequence_features(refs: Sequence[Ref], embed: Callable[[list[list[Ref]], np.ndarray], np.ndarray]) -> dict[str, np.ndarray]:
    """Window features of one whole sequence from its references only, and their nearest-centre (T, D) mapping.
    `embed(window_refs, markers)` returns one float row per window."""
    w = windows(len(refs)); wrefs = [[refs[i] for i in s] for s in w["samples"]]
    wfeat = np.asarray(embed(wrefs, w["markers"]), np.float32)
    if wfeat.shape[0] != len(w["centres"]):
        raise ValueError("one feature row per window")
    return {"centres": w["centres"], "markers": w["markers"], "window": wfeat, "samples": map_to_samples(len(refs), w["centres"], wfeat)}


# ----------------------------------------------------------------------------------------------- float32 batches
class Batches:
    """TrainData.epoch_batches with float32 features: the same default_rng([seed, epoch]), the bank in a fresh
    permutation, clips drawn in protocol order and a crop of min(window, n) samples at an RNG start. Native crops are
    slices of the whole recording's features."""

    def __init__(self, streams: Sequence[Seq], clips: Sequence[tuple[str, int, int]], native_feat: dict[str, np.ndarray]) -> None:
        self.streams, self.clips, self.native_feat = tuple(streams), tuple(clips), native_feat
        for s in self.streams:
            if s.feat.dtype != np.float32:
                raise TypeError(f"{s.seq_id}: stream features must be float32")
        for vid, _, n in self.clips:
            f = native_feat[vid]
            if f.dtype != np.float32 or f.shape[0] != int(n):
                raise ValueError(f"{vid}: native features must be float32 with one row per sample")

    def crop(self, vid: str, label: int, start: int, length: int) -> Seq:
        f = self.native_feat[vid][start:start + length]
        return Seq(f"{vid}@{start}", True, f, np.full(length, int(label), np.int64), np.zeros(0, np.int64))

    def epoch(self, seed: int, epoch: int, n_native: int, window: int, streams_per_batch: int, natives_per_batch: int) -> list[list[Seq]]:
        rng = np.random.default_rng([int(seed), int(epoch)])
        order = rng.permutation(len(self.streams))
        picks = rng.choice(len(self.clips), size=n_native, replace=n_native > len(self.clips))
        natives = []
        for i in picks:
            vid, lab, n = self.clips[int(i)]; length = min(int(window), int(n))
            start = int(rng.integers(0, int(n) - length + 1))
            natives.append(self.crop(vid, lab, start, length))
        n_batches = max(math.ceil(len(order) / streams_per_batch), math.ceil(n_native / natives_per_batch))
        return [[self.streams[int(k)] for k in order[b * streams_per_batch:(b + 1) * streams_per_batch]]
                + natives[b * natives_per_batch:(b + 1) * natives_per_batch] for b in range(n_batches)]


# ----------------------------------------------------------------------------------------------- training-only index
def training_index(path: Path) -> pd.DataFrame:
    """The training rows of a feature-cache index.csv: video_id, split and n_samples are read for every row; label_index
    is then parsed for the training rows only (other rows are skipped unread); raw_label and path are never read."""
    meta = pd.read_csv(path, usecols=["video_id", "split", "n_samples"], dtype={"video_id": str, "split": str})
    keep = set(np.flatnonzero(meta["split"].to_numpy() == "train").tolist())
    lab = pd.read_csv(path, usecols=["label_index"], skiprows=lambda i: i > 0 and (i - 1) not in keep)
    out = meta[meta["split"] == "train"].reset_index(drop=True)
    if len(lab) != len(out):
        raise SystemExit("the training label rows do not line up with the training index rows")
    return out.assign(label_index=lab["label_index"].astype(int).to_numpy())
