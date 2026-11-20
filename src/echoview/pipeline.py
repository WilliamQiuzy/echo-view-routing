"""The reference system: three trained seeds of one window model, run as one probability ensemble.

Recording level.  Each seed's probability for a recording is the mean of its windows' softmax outputs; the ensemble
                  averages the three seeds. The view is the largest of the five probabilities.
Stream level.     Each sample takes the softmax of the window whose centre is nearest; the three seeds are averaged per
                  sample; the change-point decoder (decoder.py) cuts the stream with the setting of the ensemble.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import decoder, uncertain, weights
from .model import CLASSES, WindowModel, load_model, marker_planes, normalise
from .reference import B3, SEEDS, VERSION
from .windows import nearest_centre, window_plan

HZ = 10.0


@dataclass(frozen=True)
class Classification:
    view: str
    probability: float
    probabilities: np.ndarray  # (5,) ensemble
    by_seed: np.ndarray  # (seeds, 5)
    seeds_agree: bool
    n_windows: int


@dataclass(frozen=True)
class Segmentation:
    labels: np.ndarray  # (samples,) view index of every 0.1 s sample; -1 where the system declines (uncertain)
    probabilities: np.ndarray  # (samples, 5) ensemble
    segments: tuple  # (start_s, end_s, view or "uncertain", mean probability of that view; uncertain: of the likeliest view)
    uncertain: np.ndarray  # (samples,) True where the system declines; all False without a rule


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the last axis in float64."""
    shifted = np.asarray(logits, np.float64)
    shifted = shifted - shifted.max(-1, keepdims=True)
    prob = np.exp(shifted)
    return prob / prob.sum(-1, keepdims=True)


def probabilities_from_logits(logits: np.ndarray, n_samples: int) -> np.ndarray:
    """Window logits (windows, models, 5) of a stream of `n_samples` -> per-sample probabilities (samples, 5) float64.

    Every sample takes the softmax of the window whose centre is nearest; the models are averaged per sample.
    """
    centres, _, _ = window_plan(n_samples)
    if len(logits) != len(centres):
        raise ValueError(f"a stream of {n_samples} samples has {len(centres)} windows, not {len(logits)}")
    window_prob = torch.softmax(torch.from_numpy(np.asarray(logits, np.float32)), -1).numpy()
    return window_prob[nearest_centre(n_samples, centres)].astype(np.float64).mean(1)


class Ensemble:
    """Window models on one device whose probabilities are averaged.

    `Ensemble()` is the reference system: seeds 0, 1 and 2 of the published weights with the published decoder
    setting. `seeds` may name fewer (faster, and no longer the reference system). Whether a system is the reference
    follows from the checksums of its weight files alone; `verify=False` only allows other files to be loaded.
    `Ensemble.from_models` wraps window models of your own, with a decoder setting of your own.
    """

    def __init__(self, weights_dir: str | Path | None = None, seeds: tuple = SEEDS, device: str = "auto", batch: int = 0,
                 verify: bool = True) -> None:
        self.seeds = tuple(int(s) for s in seeds)
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or not set(self.seeds) <= set(SEEDS):
            raise ValueError(f"name at least one seed among {SEEDS}, each once")
        self.weights_dir = weights.resolve(weights_dir, self.seeds)
        files = [weights.seed_file(self.weights_dir, seed) for seed in self.seeds]
        self.published = all(weights.is_published(path, seed) for path, seed in zip(files, self.seeds))
        if verify and not self.published:
            raise weights.WeightsError(
                f"a weight file in {self.weights_dir} is not a published file of this release (checksum differs). "
                "Download the weights again. To run files of your own, pass --unverified on the command line or verify=False in Python."
            )
        self.device = pick_device(device)
        self.batch = batch or (8 if self.device.type == "cuda" else 4)
        self.models = [load_model(path, self.device) for path in files]
        self.b3 = dict(B3)

    @classmethod
    def from_models(cls, models: list[WindowModel], b3: dict, device: str = "auto", batch: int = 0) -> Ensemble:
        """Window models of your own (already loaded) as one ensemble, with your own decoder setting."""
        if not models:
            raise ValueError("an ensemble needs at least one model")
        self = cls.__new__(cls)
        self.seeds, self.weights_dir, self.published = (), None, False
        self.device = pick_device(device)
        self.batch = batch or (8 if self.device.type == "cuda" else 4)
        self.models = [model.eval().to(self.device) for model in models]
        self.b3 = {key: b3[key] for key in ("half_window", "threshold", "nms_radius")}
        if int(self.b3["half_window"]) < 1 or int(self.b3["nms_radius"]) < 0 or not np.isfinite(float(self.b3["threshold"])):
            raise ValueError("a decoder setting needs half_window of 1 or more, nms_radius of 0 or more and a finite threshold")
        return self

    @property
    def is_reference(self) -> bool:
        """True for the three published seeds, checked against their checksums, with the published decoder setting."""
        return self.seeds == SEEDS and self.published and self.b3 == B3

    @property
    def version(self) -> str:
        if self.is_reference:
            return VERSION
        if self.seeds:
            return f"C2 seeds {','.join(str(s) for s in self.seeds)} (not the reference ensemble)"
        return f"custom ensemble of {len(self.models)} model(s)"

    @torch.no_grad()
    def window_logits(self, frames: np.ndarray) -> np.ndarray:
        """frames (n, 224, 224, 3) uint8 on the 10 Hz grid -> logits (windows, models, 5) float32."""
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[1:] != (224, 224, 3) or frames.dtype != np.uint8:
            raise ValueError(f"expected uint8 frames (n, 224, 224, 3), got {frames.shape} {frames.dtype}")
        _, windows, slots = window_plan(len(frames))
        pixels = torch.from_numpy(frames).to(self.device)
        index = torch.from_numpy(windows).to(self.device)
        marks = torch.from_numpy(slots).to(self.device)
        out = []
        for begin in range(0, len(windows), self.batch):
            rgb = normalise(pixels[index[begin:begin + self.batch]])
            marker = marker_planes(marks[begin:begin + self.batch])
            out.append(torch.stack([model(rgb, marker).float() for model in self.models], 1).cpu())
        return torch.cat(out).numpy()

    def classify(self, frames: np.ndarray) -> Classification:
        """One recording -> its view."""
        prob = softmax(self.window_logits(frames))
        by_seed = prob.mean(0)
        mean = by_seed.mean(0)
        winners = by_seed.argmax(-1)
        return Classification(view=CLASSES[int(mean.argmax())], probability=float(mean.max()), probabilities=mean, by_seed=by_seed,
                              seeds_agree=bool(winners.min() == winners.max()), n_windows=len(prob))

    def sample_probabilities(self, frames: np.ndarray) -> np.ndarray:
        """One stream -> the ensemble's probabilities for every 0.1 s sample, (samples, 5) float64."""
        return probabilities_from_logits(self.window_logits(frames), len(frames))

    def segment(self, frames: np.ndarray, rule: uncertain.Rule | None = None) -> Segmentation:
        """One stream (a long recording, or several recordings joined) -> a view for every 0.1 s and the pieces.

        With a `rule` (echoview.uncertain.Rule) the system may decline: such samples get -1 and form pieces called
        "uncertain". Every other sample keeps the view it has without the rule.
        """
        logits = self.window_logits(frames)
        prob = probabilities_from_logits(logits, len(frames))
        rule = uncertain.OFF if rule is None else rule
        by_seed = np.stack([probabilities_from_logits(logits[:, [k]], len(frames)) for k in range(logits.shape[1])]) if rule.seeds else None
        marked = uncertain.mark(prob, self.b3, rule, by_seed)
        pieces = tuple((start / HZ, end / HZ, CLASSES[view] if view >= 0 else "uncertain",
                        float(prob[start:end, view].mean() if view >= 0 else prob[start:end].max(1).mean()))
                       for start, end, view in decoder.segments(marked.labels))
        return Segmentation(labels=marked.labels, probabilities=prob, segments=pieces, uncertain=marked.mask)
