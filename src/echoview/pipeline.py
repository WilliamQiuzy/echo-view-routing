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
from .reference import B3, DENSER_READ_OUT, SEEDS, VERSION
from .windows import STRIDE, nearest_centre, window_plan

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
    uncertain: np.ndarray | None = None  # (samples,) True where the system declines; all False without a rule

    def __post_init__(self) -> None:
        if self.uncertain is None:  # a Segmentation built without it, as before the uncertain output existed
            object.__setattr__(self, "uncertain", np.asarray(self.labels) < 0)


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


def probabilities_from_logits(logits: np.ndarray, n_samples: int, stride: int = STRIDE) -> np.ndarray:
    """Window logits (windows, models, 5) of a stream of `n_samples` -> per-sample probabilities (samples, 5) float64.

    Every sample takes the softmax of the window whose centre is nearest; the models are averaged per sample.
    `stride` is the step at which the logits were read (window_logits), 5 samples in the published system.
    """
    centres, _, _ = window_plan(n_samples, stride)
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

    def version_at(self, stride: int = STRIDE) -> str:
        """The name of what `segment` returns at a read-out step: the system's version, and the step when it is not the published one."""
        return self.version if stride == STRIDE else f"{self.version}, read every {stride / HZ:.1f} s"

    def decoder_setting(self, stride: int = STRIDE) -> dict:
        """The decoder setting for a read-out every `stride` samples: the system's own at 5, the published step.

        The published system also has a setting for 2 and for 1 sample (reference.DENSER_READ_OUT). Those read-outs
        were chosen on development data; on the test benchmark they have nearer cuts at a cost (docs/FOLLOWUP.md, section 9).
        """
        if stride == STRIDE:
            return self.b3
        if not self.is_reference:
            raise ValueError("a denser read-out has a decoder setting for the published system only")
        if stride not in DENSER_READ_OUT:
            raise ValueError(f"a read-out every {stride} samples was not studied; the steps are {STRIDE} (published), 2 and 1")
        return dict(DENSER_READ_OUT[stride])

    @torch.no_grad()
    def window_logits(self, frames: np.ndarray, stride: int = STRIDE) -> np.ndarray:
        """frames (n, 224, 224, 3) uint8 on the 10 Hz grid -> logits (windows, models, 5) float32.

        One window every `stride` samples. The published system, and every other method of this class, reads every
        5 samples; another step is for studies of the read-out (docs/FOLLOWUP.md).
        """
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[1:] != (224, 224, 3) or frames.dtype != np.uint8:
            raise ValueError(f"expected uint8 frames (n, 224, 224, 3), got {frames.shape} {frames.dtype}")
        _, windows, slots = window_plan(len(frames), stride)
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

    def segment(self, frames: np.ndarray, rule: uncertain.Rule | None = None, stride: int = STRIDE) -> Segmentation:
        """One stream (a long recording, or several recordings joined) -> a view for every 0.1 s and the pieces.

        With a `rule` (echoview.uncertain.Rule) the system may decline: such samples get -1 and form pieces called
        "uncertain". Every other sample keeps the view it has without the rule.

        `stride` is the number of samples between two outputs of the models. 5 is the published system. With 2 or 1
        the published models are read more often and decoded with the setting of `decoder_setting(stride)`: the cuts
        lie nearer to the view changes on development data and on the test benchmark, where one single recording
        more of 800 is misread, and the uncertain rule was not studied with it (docs/FOLLOWUP.md).
        """
        rule = uncertain.OFF if rule is None else rule
        if stride != STRIDE and rule != uncertain.OFF:
            raise ValueError("the uncertain rule was studied with the published read-out only (stride 5)")
        setting = self.decoder_setting(stride)
        logits = self.window_logits(frames, stride)
        prob = probabilities_from_logits(logits, len(frames), stride)
        by_seed = np.stack([probabilities_from_logits(logits[:, [k]], len(frames), stride) for k in range(logits.shape[1])]) if rule.seeds else None
        marked = uncertain.mark(prob, setting, rule, by_seed)
        pieces = tuple((start / HZ, end / HZ, CLASSES[view] if view >= 0 else "uncertain",
                        float(prob[start:end, view].mean() if view >= 0 else prob[start:end].mean(0).max()))
                       for start, end, view in decoder.segments(marked.labels))
        return Segmentation(labels=marked.labels, probabilities=prob, segments=pieces, uncertain=marked.mask)
