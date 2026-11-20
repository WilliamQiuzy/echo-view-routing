"""The uncertain output: the samples of a decoded stream the system declines to name.

The published decoder runs first and gives every 0.1 s sample a view. Four tests then mark samples as uncertain; a
sample is uncertain when any test marks it. A sample that is not uncertain keeps the view the decoder gave it.

  band      the `band` samples before and the `band` samples after every cut: the system decides once every 0.5 s
            from 1.6 s of video and knows the position of a cut only to a few tenths of a second
  weak cut  the samples whose view changes when the stream is decoded again with the threshold divided by `f_low`
            or multiplied by `f_high`: a piece that owes its name to a cut another threshold would not make
  margin    every sample of a segment whose mean probability favours its view by less than `margin` over the next
  seeds     every sample of a segment that one of the seeds, by its own mean probability, would name differently

Uncertain is not "other": it does not say that a sample shows none of the five views. docs/UNCERTAIN.md tells how
the rule was chosen and how it is evaluated.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from . import decoder

UNCERTAIN = -1
SHORTEST_RUN = 5  # samples: a labelled run shorter than this beside an uncertain sample is not kept (0.5 s)


@dataclass(frozen=True)
class Rule:
    band: int = 0
    f_low: float = 1.0
    f_high: float = 1.0
    margin: float = 0.0
    seeds: bool = False

    def __post_init__(self) -> None:
        whole = isinstance(self.band, (int, np.integer)) and not isinstance(self.band, bool)
        factors = all(isinstance(f, (int, float, np.number)) and math.isfinite(f) and f >= 1.0 for f in (self.f_low, self.f_high))
        if not whole or self.band < 0 or not factors or not 0.0 <= self.margin <= 1.0 or not isinstance(self.seeds, (bool, np.bool_)):
            raise ValueError("a rule needs a whole band of 0 or more, finite f_low and f_high of 1 or more, margin between 0 and 1 "
                             "and seeds on or off")

    @property
    def off(self) -> bool:
        return self.band == 0 and self.f_low == 1.0 and self.f_high == 1.0 and self.margin == 0.0 and not self.seeds


OFF = Rule()


@dataclass(frozen=True)
class Marked:
    labels: np.ndarray  # (samples,) the view of every sample, UNCERTAIN where the system declines
    mask: np.ndarray  # (samples,) True where uncertain
    decided: np.ndarray  # (samples,) the published decoder's view of every sample: what a forced answer would be


def segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Runs of equal labels as (first sample, one past the last sample, view index or UNCERTAIN)."""
    return decoder.segments(labels)


def band_mask(labels: np.ndarray, band: int) -> np.ndarray:
    """True for the `band` samples before and after every cut. A cut is a sample whose view differs from the one before."""
    labels = np.asarray(labels)
    mask = np.zeros(labels.size, bool)
    if band > 0:
        for cut in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
            mask[max(0, cut - band):cut + band] = True
    return mask


def weak_cut_mask(prob: np.ndarray, labels: np.ndarray, b3: Mapping, f_low: float, f_high: float, scores: np.ndarray | None = None) -> np.ndarray:
    """True where decoding with the threshold divided by `f_low`, or multiplied by `f_high`, gives another view than
    `labels`, the decoding at the threshold itself. `scores` as in decoder.detect_boundaries."""
    labels = np.asarray(labels)
    mask = np.zeros(labels.size, bool)
    half_window, threshold, radius = int(b3["half_window"]), float(b3["threshold"]), int(b3["nms_radius"])
    if (f_low != 1.0 or f_high != 1.0) and scores is None:
        scores = decoder.boundary_scores(prob, half_window)
    for factor in (1.0 / f_low, f_high):
        if factor != 1.0:
            mask |= decoder.decode(prob, half_window, threshold * factor, radius, scores) != labels
    return mask


def margin_mask(prob: np.ndarray, labels: np.ndarray, margin: float) -> np.ndarray:
    """True for every sample of a segment whose mean probability favours its first view by less than `margin`."""
    prob, labels = np.asarray(prob, float), np.asarray(labels)
    mask = np.zeros(labels.size, bool)
    if margin > 0:
        for start, end, _ in decoder.segments(labels):
            first, second = np.sort(prob[start:end].mean(0))[[-1, -2]]
            mask[start:end] = first - second < margin
    return mask


def seed_mask(seed_prob: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """True for every sample of a segment that a seed names differently. `seed_prob` is (seeds, samples, views)."""
    seed_prob, labels = np.asarray(seed_prob, float), np.asarray(labels)
    if seed_prob.ndim != 3 or seed_prob.shape[1] != labels.size:
        raise ValueError(f"the seeds' probabilities must be (seeds, {labels.size} samples, views); got {seed_prob.shape}")
    mask = np.zeros(labels.size, bool)
    for start, end, view in decoder.segments(labels):
        mask[start:end] = bool((seed_prob[:, start:end].mean(1).argmax(-1) != view).any())
    return mask


def tidy(mask: np.ndarray) -> np.ndarray:
    """A labelled run shorter than SHORTEST_RUN samples that has an uncertain sample next to it becomes uncertain."""
    mask = np.asarray(mask, bool).copy()
    for start, end, uncertain in decoder.segments(mask.astype(int)):
        if not uncertain and end - start < SHORTEST_RUN and (start > 0 or end < mask.size):
            mask[start:end] = True
    return mask


def mark(prob: np.ndarray, b3: Mapping, rule: Rule, seed_prob: np.ndarray | None = None, scores: np.ndarray | None = None) -> Marked:
    """Decodes per-sample probabilities (samples, views) with the setting `b3` and marks the uncertain samples.

    `seed_prob` holds the same probabilities for every seed, (seeds, samples, views); the seeds test needs it.
    """
    prob = np.asarray(prob, float)
    half_window = int(b3["half_window"])
    if scores is None:
        scores = decoder.boundary_scores(prob, half_window)
    decided = decoder.decode(prob, half_window, float(b3["threshold"]), int(b3["nms_radius"]), scores)
    if rule.seeds and seed_prob is None:
        raise ValueError("the seeds test needs the probabilities of every seed")
    mask = band_mask(decided, rule.band) | weak_cut_mask(prob, decided, b3, rule.f_low, rule.f_high, scores) | margin_mask(prob, decided, rule.margin)
    if rule.seeds:
        mask |= seed_mask(seed_prob, decided)
    mask = tidy(mask)
    return Marked(labels=np.where(mask, UNCERTAIN, decided), mask=mask, decided=decided)
