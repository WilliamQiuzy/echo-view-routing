"""The intervals of the test benchmark: a percentile bootstrap over whole units, as the project computes it.

A unit is what is drawn again with replacement: an examination for the recording-level numbers, a stream for
segment F1. Every number is a function of counts summed over units, so a resample is a weighted sum of the units'
counts. One generator with a fixed seed serves the tables in a fixed order: native recordings, multi streams, pair
streams. With the same units in the same order this gives the intervals of the project's reports to the last digit;
`tests/test_evaluation_test_benchmark.py` holds them against the output of the research code.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from . import metrics
from .benchmark import Sequence, of_kind

DRAWS = 2000
SEED = 20261001
CLASSES = metrics.CLASSES


def resampling_weights(generator: np.random.Generator, units: int, draws: int = DRAWS) -> np.ndarray:
    """(draws, units): how often every unit is in every resample."""
    drawn = generator.integers(0, units, size=(draws, units))
    return np.stack([np.bincount(row, minlength=units) for row in drawn]).astype(float)


def percentile_interval(weights: np.ndarray, sums: np.ndarray, statistic: Callable[[np.ndarray], np.ndarray]) -> list[float]:
    """The 2.5th and 97.5th percentile of the statistic over the resamples. `sums` is (units, columns); the statistic
    gets the summed columns with the resamples on the last axis."""
    with np.errstate(divide="ignore", invalid="ignore"):
        replicates = statistic(np.moveaxis(weights @ sums, -1, 0))
    return np.nanpercentile(replicates, [2.5, 97.5]).tolist()


def macro_f1(cells: np.ndarray) -> np.ndarray:
    """Mean F1 over the views from the 25 confusion counts (row: reference view, column: decided view)."""
    table = cells.reshape((CLASSES, CLASSES, *cells.shape[1:]))
    hit = np.stack([table[k, k] for k in range(CLASSES)])
    wrongly_decided, missed = table.sum(0) - hit, table.sum(1) - hit
    denominator = 2 * hit + wrongly_decided + missed
    return np.where(denominator > 0, 2 * hit / np.where(denominator > 0, denominator, 1), 0.0).mean(0)


def accuracy(cells: np.ndarray) -> np.ndarray:
    return sum(cells[(CLASSES + 1) * k] for k in range(CLASSES)) / sum(cells)


def segment_f1(counts: np.ndarray) -> np.ndarray:
    """From the summed columns matched, predicted, reference."""
    return 2 * counts[0] / (counts[1] + counts[2])


def examinations(natives: tuple[Sequence, ...]) -> int:
    """The number of examinations the native recordings come from; they must be numbered from 0 without a gap."""
    groups = sorted({sequence.group for sequence in natives})
    if not groups or groups != list(range(len(groups))):
        raise ValueError("the native recordings do not carry their examinations, numbered from 0 without a gap")
    return len(groups)


def native_sums(natives: tuple[Sequence, ...], decoded: Mapping[str, np.ndarray]) -> np.ndarray:
    """(examinations, 25): the confusion counts of every examination's recordings, examinations in their numbered order."""
    sums = np.zeros((examinations(natives), CLASSES * CLASSES))
    for sequence in natives:
        decided = int(np.bincount(np.asarray(decoded[sequence.id]), minlength=CLASSES).argmax())
        sums[sequence.group, CLASSES * int(sequence.labels[0]) + decided] += 1
    return sums


def stream_sums(streams: tuple[Sequence, ...], decoded: Mapping[str, np.ndarray]) -> np.ndarray:
    """(streams, 3): matched, predicted and reference segments of every stream, streams in the order of their names."""
    rows = []
    for sequence in sorted(streams, key=lambda s: s.id):
        predicted, reference = metrics.labels_to_segments(decoded[sequence.id]), metrics.labels_to_segments(sequence.labels)
        rows.append([metrics.matched_segments(predicted, reference), len(predicted), len(reference)])
    return np.asarray(rows, float)


def intervals(sequences: tuple[Sequence, ...], decoded: Mapping[str, np.ndarray], draws: int = DRAWS, seed: int = SEED) -> dict[str, list[float]]:
    """95% intervals of the headline numbers of a benchmark with native recordings, multi streams and pair streams."""
    generator = np.random.default_rng(seed)
    native, multi, pairs = (of_kind(sequences, kind) for kind in ("native", "multi", "pairs"))
    cells = native_sums(native, decoded)
    weights = resampling_weights(generator, len(cells), draws)
    out = {"native_macro_f1": percentile_interval(weights, cells, macro_f1),
           "native_accuracy": percentile_interval(weights, cells, accuracy)}
    for name, streams in (("segment_f1_50_multi", multi), ("segment_f1_50_pairs", pairs)):
        sums = stream_sums(streams, decoded)
        out[name] = percentile_interval(resampling_weights(generator, len(sums), draws), sums, segment_f1)
    return out
