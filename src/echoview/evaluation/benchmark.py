"""The EV9V benchmarks as data: which samples make up every sequence.

EV9V holds single-view recordings. A benchmark builds its sequences from them:

  native  one whole recording (284 and 237 recordings in the two halves);
  pairs   two fragments joined, the same view or two different views;
  multi   four to eight fragments joined: the streams behind segment F1 and the boundary metrics;
  exam    every recording of one examination joined in the order of acquisition ("real order").

A fragment is a run of 10 Hz samples from one recording, given by the JPEG frame number of every sample. `gamma` is
the grey-level change of the fragment's input variant (1.0 for none).

There are two files, both written once from the research protocol (tools/research/) and part of the package:

  development  from EV9V's validation split, in two halves that are disjoint in examinations: `calibration` sets
               the decoder, `selection` is the half the development numbers are read from;
  test         the project's frozen test benchmark, from EV9V's test split: native recordings, pair streams and
               multi streams. Nothing is chosen on it.

Neither holds a recording name: a recording is named by a key, the first 16 hex digits of the SHA-256 of its EV9V
folder name. The frame reader (ev9v.py) finds the folders of a local EV9V copy by hashing their names. In the test
file a native recording also carries `group`, the number of its examination: the bootstrap resamples examinations.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from functools import cached_property
from importlib import resources
from pathlib import Path

import numpy as np

HALVES = ("calibration", "selection")
KINDS = ("native", "pairs", "multi", "exam")
EV9V_DEV = "ev9v_dev.json.gz"
EV9V_TEST = "ev9v_test.json.gz"
TEST = "test"
FORMAT = "echo-view-routing benchmark v2"

Ref = tuple[str, int, float]  # recording key, JPEG frame number, gamma


@dataclass(frozen=True)
class Fragment:
    video: str  # the recording's key
    label: int
    gamma: float
    frames: tuple[int, ...]


@dataclass(frozen=True)
class Sequence:
    id: str
    kind: str
    fragments: tuple[Fragment, ...]
    cell: str = ""
    group: int = -1  # test benchmark, native recordings: the examination, numbered in the order the bootstrap uses

    def __len__(self) -> int:
        return sum(len(fragment.frames) for fragment in self.fragments)

    @cached_property
    def labels(self) -> np.ndarray:
        """The reference view of every sample."""
        return np.concatenate([np.full(len(f.frames), f.label, dtype=int) for f in self.fragments])

    @cached_property
    def joins(self) -> np.ndarray:
        """The first sample after every join of two fragments."""
        return np.cumsum([len(f.frames) for f in self.fragments])[:-1].astype(int)

    @cached_property
    def view_changes(self) -> np.ndarray:
        """The joins at which the reference view changes."""
        labels = self.labels
        return np.array([j for j in self.joins if labels[j] != labels[j - 1]], dtype=int)

    @cached_property
    def refs(self) -> tuple[Ref, ...]:
        """(recording key, frame number, gamma) of every sample."""
        return tuple((f.video, frame, f.gamma) for f in self.fragments for frame in f.frames)

    @cached_property
    def pieces(self) -> np.ndarray:
        """For every sample, the index of its source piece: a run of consecutive samples from one recording."""
        video = np.array([f.video for f in self.fragments for _ in f.frames])
        return np.concatenate([[0], np.cumsum(video[1:] != video[:-1])]).astype(int)


def _sequence(item: dict) -> Sequence:
    fragments = tuple(Fragment(str(f["video"]), int(f["label"]), float(f["gamma"]), tuple(int(n) for n in f["frames"]))
                      for f in item["fragments"])
    if item["kind"] not in KINDS or not fragments or any(not f.frames for f in fragments):
        raise ValueError(f"sequence {item.get('id')!r} is not a sequence of this benchmark format")
    return Sequence(str(item["id"]), item["kind"], fragments, str(item.get("cell", "")), int(item.get("group", -1)))


def _parts(path: str | Path | None, packaged: str, parts: tuple[str, ...]) -> dict[str, tuple[Sequence, ...]]:
    raw = resources.files("echoview").joinpath("data", packaged).read_bytes() if path is None else Path(path).read_bytes()
    document = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    if document.get("format") != FORMAT or any(part not in document.get("halves", {}) for part in parts):
        raise ValueError(f"not a benchmark file of this package with the parts {', '.join(parts)}")
    return {part: tuple(_sequence(item) for item in document["halves"][part]["sequences"]) for part in parts}


def load(path: str | Path | None = None) -> dict[str, tuple[Sequence, ...]]:
    """The sequences of both halves, in the benchmark's order. Without a path: the EV9V development benchmark."""
    return _parts(path, EV9V_DEV, HALVES)


def load_test(path: str | Path | None = None) -> tuple[Sequence, ...]:
    """The sequences of the frozen test benchmark, in its order. Without a path: the EV9V test benchmark."""
    return _parts(path, EV9V_TEST, (TEST,))[TEST]


def of_kind(sequences: tuple[Sequence, ...], kind: str) -> tuple[Sequence, ...]:
    return tuple(s for s in sequences if s.kind == kind)
