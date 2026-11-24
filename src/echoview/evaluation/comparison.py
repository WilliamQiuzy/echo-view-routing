"""The comparison with other approaches, as planned in docs/COMPARISON.md.

  python -m echoview.evaluation.comparison inventory --images EV9V/Images --store NAME=FOLDER ... --out inventory.json
  python -m echoview.evaluation.comparison settings --images EV9V/Images --logits DEV_LOGITS.npz [--store NAME=FOLDER ...] --out settings.json
  python -m echoview.evaluation.comparison score --benchmark test --images EV9V/Images --logits LOGITS.npz --store NAME=FOLDER ...
        --settings settings.json --out comparison.json

A store is a folder as the research code wrote it: one .npz per sequence with the array `prob` (samples x 5 views).
The file of a single recording is named by the recording's EV9V folder name, the file of a stream by the stream's id.
NAME is one of EchoViewCLIP, STFM, ASFormer, EchoPrime.

`inventory` lists what the stores hold and computes no metric. `settings` chooses decoder settings on the calibration
half of the development benchmark: for each seed of EchoRouter and for each store given. `score` builds every row of
the plan and scores it; with `--benchmark test` it first checks that the stored outputs give the published numbers and
stops when one differs. No command replaces an output file. The outputs hold pooled counts only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..model import CLASSES
from ..reference import B3
from . import benchmark, bootstrap, calibrate, metrics, scorecard
from .benchmark import Sequence, of_kind
from .ev9v import folders_by_key
from .run import member_probabilities

OTHERS = ("EchoViewCLIP", "STFM", "ASFormer", "EchoPrime")
ROUTER = "EchoRouter"
# The decoder setting chosen for EchoPrime's classifier on validation data in October 2026 (its published row).
ECHOPRIME_OCTOBER = {"half_window": 9, "threshold": 0.3351074947858375, "nms_radius": 5}
# What the stored outputs must give before anything is scored (docs/COMPARISON.md, section 3).
PUBLISHED = {
    ROUTER: {"segment_f1_50_multi": 0.908909, "segment_f1_50_pairs": 0.984701, "native_accuracy": 0.981250, "native_macro_f1": 0.956141},
    "EchoViewCLIP": {"segment_f1_50_multi": 0.866731, "segment_f1_50_pairs": 0.988920, "native_accuracy": 0.988750, "native_macro_f1": 0.971936},
    "STFM": {"segment_f1_50_multi": 0.859086, "segment_f1_50_pairs": 0.970954, "native_accuracy": 0.982500, "native_macro_f1": 0.959627},
    "ASFormer": {"segment_f1_50_multi": 0.856311, "segment_f1_50_pairs": 0.980716, "native_accuracy": 0.982500, "native_macro_f1": 0.955852},
    "EchoPrime + decoder": {"segment_f1_50_multi": 0.851020, "segment_f1_50_pairs": 0.987586, "native_accuracy": 0.978750, "native_macro_f1": 0.943459},
    "EchoPrime per frame": {"segment_f1_50_multi": 0.495768, "segment_f1_50_pairs": 0.742734, "native_accuracy": 0.978750, "native_macro_f1": 0.943459},
}
PUBLISHED_SELECTION_F1 = 0.9594  # EchoRouter on the multi streams of the selection half
TEST_LOGITS_SHA256 = "c5df435ef3022dd55f711c4efa72c74f19aad7f40193f179902c57dca97e288d"  # the file the plan names
RELEASED_QUANTILE, THRESHOLD_CLOSE = 0.999, 1e-3  # the grid position of the released setting; logits of another machine move the threshold a little
CLOSE = 5e-7
DRAWS, SEED = 10_000, 20261123
SHORT = 16  # samples: a segment of at most 1.6 s
TOLERANCES = (1, 2, 3, 5, 10)  # samples within which a cut may match a view change
OVERLAPS = (10, 25, 50)
N = len(CLASSES)
NAMED, DESCRIBED = 98.75, 95.0

Item = tuple[str, int, float]  # (column, the sample at which it is counted, amount)


class Row(NamedTuple):
    name: str
    system: str
    role: str  # A as published, D with the project's decoder, C released numbers carried over, N no decoder, S seed, P perfect on the grid
    labels: dict[str, np.ndarray]


class Stat(NamedTuple):
    name: str
    direction: str  # "up", "down" or "" when neither is better
    value: Callable[[Callable[[str], np.ndarray]], np.ndarray]
    support: Callable[[Callable[[str], np.ndarray]], np.ndarray]  # the count the number rests on


# ------------------------------------------------------------------------------------------------------- stores
def store_file(folder: Path, sequence: Sequence, names: Mapping[str, str]) -> Path:
    stem = names[sequence.fragments[0].video] if sequence.kind == "native" else sequence.id
    return folder / f"{stem}.npz"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_store(folder: Path, sequences: tuple[Sequence, ...], names: Mapping[str, str]) -> tuple[dict[str, np.ndarray], str]:
    """The stored rows of every sequence, as stored, and a digest over (sequence id, file SHA-256) in id order."""
    rows, digest = {}, hashlib.sha256()
    for sequence in sorted(sequences, key=lambda s: s.id):
        path = store_file(folder, sequence, names)
        if not path.is_file():
            raise SystemExit(f"{folder}: no stored output for a sequence of kind {sequence.kind}")
        with np.load(path) as stored:
            prob = stored["prob"].astype(float)
        if prob.shape != (len(sequence), N) or not np.isfinite(prob).all():
            raise SystemExit(f"{folder}: a stored output has the wrong shape or values that are not finite")
        rows[sequence.id] = prob
        digest.update(sequence.id.encode() + b"\0" + file_sha256(path).encode() + b"\n")
    return rows, digest.hexdigest()


def coverage(folder: Path, sequences: tuple[Sequence, ...], names: Mapping[str, str]) -> dict:
    """What a store holds for a set of sequences: counts, array names, shapes, the range of the row sums and the
    digest that read_store gives. Nothing here stops on a file that is not as expected: it is counted."""
    have = sorted(((s, store_file(folder, s, names)) for s in sequences if store_file(folder, s, names).is_file()), key=lambda pair: pair[0].id)
    out: dict = {"sequences": len(sequences), "stored": len(have)}
    if have and len(have) == len(sequences):
        arrays, wrong, low, high, digest = set(), 0, np.inf, -np.inf, hashlib.sha256()
        for sequence, path in have:
            with np.load(path) as stored:
                arrays |= set(stored.files)
                prob = stored["prob"] if "prob" in stored.files else None
            if prob is None or prob.shape != (len(sequence), N):
                wrong += 1
            else:
                low, high = min(low, float(prob.sum(1).min())), max(high, float(prob.sum(1).max()))
            digest.update(sequence.id.encode() + b"\0" + file_sha256(path).encode() + b"\n")
        out.update({"arrays": sorted(arrays), "wrong_shape": wrong, "row_sums": [round(low, 4), round(high, 4)], "digest": digest.hexdigest()})
    return out


# ------------------------------------------------------------------------------------------------- counted items
def matched_pairs(pred: list[metrics.Segment], true: list[metrics.Segment]) -> list[tuple[int, int]]:
    """The pairs behind metrics.matched_segments: same view, IoU at least 0.5, best IoU first, one to one."""
    candidates = sorted(((metrics.temporal_iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(true)
                         if p.label == t.label and metrics.temporal_iou(p, t) >= metrics.IOU), reverse=True)
    used_pred, used_true, pairs = set(), set(), []
    for _, i, j in candidates:
        if i not in used_pred and j not in used_true:
            used_pred.add(i)
            used_true.add(j)
            pairs.append((i, j))
    return pairs


def _middle(segment: metrics.Segment) -> int:
    return scorecard.midpoint(segment.start, segment.end)


def segment_items(truth: np.ndarray, pred: np.ndarray) -> list[Item]:
    """Reference segments, returned segments and their matches, each counted at its middle sample."""
    reference, returned = metrics.labels_to_segments(truth), metrics.labels_to_segments(pred)
    pairs = matched_pairs(returned, reference)
    hit_pred, hit_true = {i for i, _ in pairs}, {j for _, j in pairs}
    items: list[Item] = []
    for _, j in pairs:
        items += [("seg_matched", _middle(reference[j]), 1.0), (f"seg_matched_v{reference[j].label}", _middle(reference[j]), 1.0)]
    for i, segment in enumerate(returned):
        items += [("seg_pred", _middle(segment), 1.0), (f"seg_pred_v{segment.label}", _middle(segment), 1.0)]
        if i not in hit_pred:
            items.append(("extra_short" if segment.length <= SHORT else "extra_long", _middle(segment), 1.0))
    for j, segment in enumerate(reference):
        items += [("seg_true", _middle(segment), 1.0), (f"seg_true_v{segment.label}", _middle(segment), 1.0)]
        if j not in hit_true:
            items.append(("missed_short" if segment.length <= SHORT else "missed_long", _middle(segment), 1.0))
    return items


def overlap_items(truth: np.ndarray, pred: np.ndarray) -> list[Item]:
    """MS-TCN's true positives, false positives and false negatives (scorecard.overlap_counts), each at its segment's
    middle sample."""
    reference, returned = metrics.labels_to_segments(truth), metrics.labels_to_segments(pred)
    items: list[Item] = []
    for overlap in OVERLAPS:
        used = [False] * len(reference)
        for segment in returned:
            best, where = 0.0, -1
            for k, other in enumerate(reference):
                if other.label != segment.label:
                    continue
                union = max(segment.end, other.end) - min(segment.start, other.start)
                iou = max(0, min(segment.end, other.end) - max(segment.start, other.start)) / union
                if iou > best:
                    best, where = iou, k
            if where >= 0 and best >= overlap / 100 and not used[where]:
                used[where] = True
                items.append((f"tp{overlap}", _middle(segment), 1.0))
            else:
                items.append((f"fp{overlap}", _middle(segment), 1.0))
        items += [(f"fn{overlap}", _middle(other), 1.0) for k, other in enumerate(reference) if not used[k]]
    return items


def sample_items(truth: np.ndarray, piece: np.ndarray, pred: np.ndarray) -> list[Item]:
    """Every sample in the table of reference view by returned view; wrong samples by kind; pieces and returned
    segments by whether their middle sample is right."""
    piece_ok, segment_ok, _, _, pieces, segments = scorecard._midpoint_tables(truth, piece, pred, N)
    wrong, floor = pred != truth, scorecard.grid_floor_labels(truth) != truth
    kinds = np.where(piece_ok, np.where(segment_ok, "edge", "spurious"), np.where(segment_ok, "absorbed", "misread"))
    items: list[Item] = [(f"conf_{truth[k]}_{pred[k]}", k, 1.0) for k in range(len(truth))]
    items += [(str(kinds[k]), int(k), 1.0) for k in np.flatnonzero(wrong)]
    items += [("floor_wrong", int(k), 1.0) for k in np.flatnonzero(floor)]
    for start, end, _ in pieces:
        m, length = scorecard.midpoint(start, end), "short" if end - start <= SHORT else "long"
        right = float(pred[m] == truth[m])
        for column in ("piece", f"piece_v{truth[m]}", f"piece_{length}"):
            items += [(column, m, 1.0), (column.replace("piece", "piece_right", 1), m, right)]
    for start, end, label in segments:
        m = scorecard.midpoint(start, end)
        items += [("segment", m, 1.0), ("segment_right", m, float(truth[m] == label))]
    return items


def cut_items(truth: np.ndarray, pred: np.ndarray, joins: np.ndarray) -> list[Item]:
    """Cuts and view changes. A match is counted at the view change, a cut at the cut, a join at the join."""
    cuts, changes = metrics.boundaries_from_labels(pred), metrics.boundaries_from_labels(truth)
    items: list[Item] = [("cut_pred", int(c), 1.0) for c in cuts] + [("change", int(v), 1.0) for v in changes]
    for tolerance in TOLERANCES:
        items += [(f"cut_hit_{tolerance}", int(changes[j]), 1.0) for _, j in metrics.match_boundaries(cuts, changes, tolerance)]
    for change, cut in scorecard.match_changes(cuts, changes, scorecard.NEAR).items():
        items += [("near_matched", change, 1.0), ("near_abs", change, abs(cut - change) / metrics.HZ), ("near_late", change, float(cut > change))]
    near = metrics.BOUNDARY_TOLERANCE_S * metrics.HZ
    for join in (int(j) for j in joins if j not in set(changes.tolist())):
        items += [("same_join", join, 1.0), ("false_split", join, float(np.any(np.abs(cuts - join) <= near)))]
    return items


def stream_items(truth: np.ndarray, piece: np.ndarray, pred: np.ndarray, joins: np.ndarray) -> list[Item]:
    truth, piece, pred = np.asarray(truth, int), np.asarray(piece, int), np.asarray(pred, int)
    if pred.shape != truth.shape or (pred.size and (pred.min() < 0 or pred.max() >= N)):
        raise ValueError("every sample of a stream needs one of the views")
    return segment_items(truth, pred) + overlap_items(truth, pred) + sample_items(truth, piece, pred) + cut_items(truth, pred, np.asarray(joins, int))


def _stream_columns() -> tuple[str, ...]:
    views = range(N)
    names = ["seg_matched", "seg_pred", "seg_true", "extra_short", "extra_long", "missed_short", "missed_long",
             "edge", "spurious", "absorbed", "misread", "floor_wrong", "segment", "segment_right", "cut_pred", "change",
             "near_matched", "near_abs", "near_late", "same_join", "false_split", "edit", "streams"]
    names += [f"{kind}_v{v}" for kind in ("seg_matched", "seg_pred", "seg_true", "piece", "piece_right") for v in views]
    names += [f"{kind}{overlap}" for kind in ("tp", "fp", "fn") for overlap in OVERLAPS]
    names += [f"conf_{t}_{p}" for t in views for p in views]
    names += ["piece", "piece_right", "piece_short", "piece_right_short", "piece_long", "piece_right_long"]
    names += [f"cut_hit_{tolerance}" for tolerance in TOLERANCES]
    return tuple(names)


STREAM_COLUMNS = _stream_columns()
STREAM_INDEX = {name: k for k, name in enumerate(STREAM_COLUMNS)}
NATIVE_COLUMNS = tuple([f"conf_{t}_{p}" for t in range(N) for p in range(N)] + ["cuts", "recordings"])
NATIVE_INDEX = {name: k for k, name in enumerate(NATIVE_COLUMNS)}


def stream_tables(streams: tuple[Sequence, ...], labels: Mapping[str, np.ndarray], examination: Mapping[str, int] | None) -> dict[str, np.ndarray]:
    """The counts of a row by unit: one line per stream, and one line per examination when the recordings carry
    theirs. An item goes to the examination of the recording at its sample. The Edit score exists per stream only."""
    tables = {"streams": np.zeros((len(streams), len(STREAM_COLUMNS)))}
    if examination is not None:
        tables["examinations"] = np.zeros((max(examination.values()) + 1, len(STREAM_COLUMNS)))
    for k, sequence in enumerate(streams):
        pred = np.asarray(labels[sequence.id], int)
        where = None if examination is None else np.concatenate([np.full(len(f.frames), examination[f.video]) for f in sequence.fragments])
        for column, sample, amount in stream_items(sequence.labels, sequence.pieces, pred, sequence.joins):
            tables["streams"][k, STREAM_INDEX[column]] += amount
            if where is not None:
                tables["examinations"][where[sample], STREAM_INDEX[column]] += amount
        tables["streams"][k, STREAM_INDEX["edit"]] = scorecard.edit_score(sequence.labels, pred)
        tables["streams"][k, STREAM_INDEX["streams"]] = 1.0
    return tables


def native_tables(natives: tuple[Sequence, ...], labels: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The counts of the single recordings by examination, or by recording where the benchmark names no examination."""
    grouped = all(s.group >= 0 for s in natives)
    table = np.zeros(((max(s.group for s in natives) + 1) if grouped else len(natives), len(NATIVE_COLUMNS)))
    for k, sequence in enumerate(natives):
        pred, unit = np.asarray(labels[sequence.id], int), sequence.group if grouped else k
        decided = int(np.bincount(pred, minlength=N).argmax())
        table[unit, NATIVE_INDEX[f"conf_{int(sequence.labels[0])}_{decided}"]] += 1
        table[unit, NATIVE_INDEX["cuts"]] += metrics.boundaries_from_labels(pred).size
        table[unit, NATIVE_INDEX["recordings"]] += 1
    return {"examinations" if grouped else "recordings": table}


# --------------------------------------------------------------------------------------------------- the numbers
def _ratio(top: str, bottom: str, name: str, direction: str) -> Stat:
    return Stat(name, direction, lambda g: g(top) / g(bottom), lambda g: g(bottom))


def _confusion(g: Callable[[str], np.ndarray]) -> np.ndarray:
    """(reference view, returned view, ...) with the resamples on the last axes."""
    return np.stack([np.stack([g(f"conf_{t}_{p}") for p in range(N)]) for t in range(N)])


def _macro_f1(g: Callable[[str], np.ndarray]) -> np.ndarray:
    cells = _confusion(g)
    tp = np.stack([cells[v, v] for v in range(N)])
    den = cells.sum(0) + cells.sum(1)  # per view: 2 tp + fp + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0).mean(0)


def _samples(g: Callable[[str], np.ndarray]) -> np.ndarray:
    return _confusion(g).sum((0, 1))


def _right(g: Callable[[str], np.ndarray]) -> np.ndarray:
    return sum(g(f"conf_{v}_{v}") for v in range(N))


def stream_stats() -> tuple[Stat, ...]:
    stats = [Stat("segment_f1_50", "up", lambda g: 2 * g("seg_matched") / (g("seg_pred") + g("seg_true")), lambda g: g("seg_true"))]
    for v, view in enumerate(CLASSES):
        stats.append(Stat(f"segment_f1_50_{view}", "up", lambda g, v=v: 2 * g(f"seg_matched_v{v}") / (g(f"seg_pred_v{v}") + g(f"seg_true_v{v}")),
                          lambda g, v=v: g(f"seg_true_v{v}")))
    for k in OVERLAPS:
        stats.append(Stat(f"mstcn_f1_{k}", "up", lambda g, k=k: 2 * g(f"tp{k}") / (2 * g(f"tp{k}") + g(f"fp{k}") + g(f"fn{k}")),
                          lambda g, k=k: g(f"tp{k}") + g(f"fn{k}")))
    stats += [_ratio("edit", "streams", "edit", "up"),
              Stat("extra_segments", "down", lambda g: g("extra_short") + g("extra_long"), lambda g: g("extra_short") + g("extra_long")),
              Stat("extra_segments_short", "down", lambda g: g("extra_short"), lambda g: g("extra_short")),
              Stat("missed_segments", "down", lambda g: g("missed_short") + g("missed_long"), lambda g: g("missed_short") + g("missed_long")),
              Stat("missed_segments_short", "down", lambda g: g("missed_short"), lambda g: g("missed_short")),
              _ratio("seg_pred", "seg_true", "segments_per_reference", ""),
              Stat("wrong_samples", "down", lambda g: 1 - _right(g) / _samples(g), _samples),
              Stat("grid_floor", "", lambda g: g("floor_wrong") / _samples(g), _samples),
              _ratio("piece_right", "piece", "piece_midpoint", "up"),
              _ratio("piece_right_short", "piece_short", "piece_midpoint_short", "up"),
              _ratio("piece_right_long", "piece_long", "piece_midpoint_long", "up"),
              _ratio("segment_right", "segment", "segment_midpoint", "up")]
    stats += [_ratio(f"piece_right_v{v}", f"piece_v{v}", f"piece_midpoint_{view}", "up") for v, view in enumerate(CLASSES)]
    stats += [Stat(f"wrong_{kind}", "", lambda g, kind=kind: g(kind) / _samples(g), _samples) for kind in scorecard.CATEGORIES]
    stats += [_ratio("cut_hit_5", "cut_pred", "cut_precision", "up"), _ratio("cut_hit_5", "change", "cut_recall", "up")]
    stats += [_ratio(f"cut_hit_{t}", "change", f"cut_recall_{t / metrics.HZ:.1f}s", "up") for t in TOLERANCES if t != 5]
    stats += [_ratio("near_abs", "near_matched", "cut_distance_mean_s", "down"), _ratio("near_matched", "change", "cut_matched_share", ""),
              _ratio("near_late", "near_matched", "cut_late_share", ""), _ratio("false_split", "same_join", "same_view_cut_rate", "down")]
    for v, view in enumerate(CLASSES):
        stats.append(Stat(f"recovery_{view}", "up", lambda g, v=v: g(f"conf_{v}_{v}") / sum(g(f"conf_{v}_{p}") for p in range(N)),
                          lambda g, v=v: sum(g(f"conf_{v}_{p}") for p in range(N))))
        stats.append(Stat(f"purity_{view}", "up", lambda g, v=v: g(f"conf_{v}_{v}") / sum(g(f"conf_{t}_{v}") for t in range(N)),
                          lambda g, v=v: sum(g(f"conf_{t}_{v}") for t in range(N))))
    return tuple(stats)


def native_stats() -> tuple[Stat, ...]:
    a4c, a5c = CLASSES.index("A4C"), CLASSES.index("A5C")
    stats = [Stat("accuracy", "up", lambda g: _right(g) / g("recordings"), lambda g: g("recordings")),
             Stat("macro_f1", "up", _macro_f1, lambda g: g("recordings")),
             Stat("a4c_as_a5c", "down", lambda g: g(f"conf_{a4c}_{a5c}"), lambda g: g(f"conf_{a4c}_{a5c}")),
             Stat("a5c_as_a4c", "down", lambda g: g(f"conf_{a5c}_{a4c}"), lambda g: g(f"conf_{a5c}_{a4c}")),
             _ratio("cuts", "recordings", "cuts_per_recording", "down")]
    for v, view in enumerate(CLASSES):
        stats.append(Stat(f"recall_{view}", "up", lambda g, v=v: g(f"conf_{v}_{v}") / sum(g(f"conf_{v}_{p}") for p in range(N)),
                          lambda g, v=v: sum(g(f"conf_{v}_{p}") for p in range(N))))
    return tuple(stats)


STREAM_STATS, NATIVE_STATS = stream_stats(), native_stats()


def evaluate_stats(stats: tuple[Stat, ...], sums: np.ndarray, index: Mapping[str, int]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Every number and its support from summed counts; `sums` has the columns on its last axis."""
    def column(name: str) -> np.ndarray:
        return sums[..., index[name]]

    with np.errstate(divide="ignore", invalid="ignore"):
        return {s.name: np.asarray(s.value(column), float) for s in stats}, {s.name: np.asarray(s.support(column), float) for s in stats}


def percentiles(values: np.ndarray, level: float) -> list[float] | None:
    values = np.where(np.isfinite(values), values, np.nan)  # a resample without the thing counted gives no number
    if np.isnan(values).all():
        return None
    tail = (100.0 - level) / 2
    return [float(x) for x in np.nanpercentile(values, [tail, 100.0 - tail])]


def wider(intervals: Mapping[str, list[float] | None]) -> list[float] | None:
    """Of the intervals by unit, the wider one: the one the plan reports and lets decide."""
    known = [i for i in intervals.values() if i is not None]
    return max(known, key=lambda i: i[1] - i[0]) if known else None


def hull(intervals: Mapping[str, list[float] | None]) -> list[float] | None:
    """From the lowest lower bound to the highest upper bound of the intervals by unit. A word of the named
    comparisons rests on this: it needs every unit to agree."""
    known = [i for i in intervals.values() if i is not None]
    return [min(i[0] for i in known), max(i[1] for i in known)] if known else None


class Scored(NamedTuple):
    values: dict[str, float]
    support: dict[str, float]
    replicates: dict[str, dict[str, np.ndarray]]  # unit -> number -> one value per resample


def score_tables(stats: tuple[Stat, ...], tables: Mapping[str, np.ndarray], index: Mapping[str, int], weights: Mapping[str, np.ndarray]) -> Scored:
    first = next(iter(tables.values()))
    values, support = evaluate_stats(stats, first.sum(0), index)
    replicates = {unit: evaluate_stats(stats, weights[unit] @ table, index)[0] for unit, table in tables.items()}
    return Scored({k: float(v) for k, v in values.items()}, {k: float(v) for k, v in support.items()}, replicates)


def report(scored: Scored, level: float = DESCRIBED) -> dict:
    """Every number with its interval by unit and the wider one. A count of 5 or fewer has no interval; the Edit score
    has the stream interval only."""
    out = {}
    for name, value in scored.values.items():
        if not np.isfinite(value):
            out[name] = {"value": None, "support": scored.support[name]}
            continue
        by_unit = {unit: (percentiles(numbers[name], level) if scored.support[name] > 5 else None) for unit, numbers in scored.replicates.items()}
        out[name] = {"value": value, "support": scored.support[name], **by_unit, "interval": wider(by_unit)}
    return out


def difference(first: Scored, second: Scored, name: str, level: float) -> dict:
    """First minus second for one number, with the paired interval by unit, the wider one and their hull. As in
    `report`, there is no interval when either number rests on a count of 5 or fewer."""
    value = first.values[name] - second.values[name]
    counted = min(first.support[name], second.support[name]) > 5
    by_unit = {unit: (percentiles(first.replicates[unit][name] - second.replicates[unit][name], level) if counted else None) for unit in first.replicates}
    return {"value": float(value) if np.isfinite(value) else None, "level": level, **by_unit, "interval": wider(by_unit), "hull": hull(by_unit)}


def word(interval: list[float] | None) -> str:
    if interval is None:
        return "not answered"
    return "ahead of" if interval[0] > 0 else "behind" if interval[1] < 0 else "not shown to differ from"


# ---------------------------------------------------------------------------------------------------------- rows
def likeliest(rows: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: p.argmax(1) for key, p in rows.items()}


def decoded(rows: Mapping[str, np.ndarray], setting: Mapping) -> dict[str, np.ndarray]:
    return calibrate.decode_all({k: setting[k] for k in ("half_window", "threshold", "nms_radius")}, rows)


def build_rows(sequences: tuple[Sequence, ...], logits: Mapping[str, np.ndarray], stores: Mapping[str, Mapping[str, np.ndarray]],
               settings: Mapping) -> list[Row]:
    """Every row of the plan that the given outputs allow."""
    members = tuple(range(next(iter(logits.values())).shape[1]))
    ensemble = member_probabilities(dict(logits), sequences, members)
    rows = [Row(ROUTER, ROUTER, "A", decoded(ensemble, B3)), Row(f"{ROUTER}, no decoder", ROUTER, "N", likeliest(ensemble)),
            Row("Perfect on the 0.5 s grid", "", "P", {s.id: scorecard.grid_floor_labels(s.labels) for s in sequences})]
    for member in members:
        if str(member) in settings.get("seeds", {}):
            alone = member_probabilities(dict(logits), sequences, (member,))
            rows.append(Row(f"{ROUTER} seed {member}", ROUTER, "S", decoded(alone, settings["seeds"][str(member)]["params"])))
    for system, stored in stores.items():
        if system == "EchoPrime":
            rows += [Row("EchoPrime per frame", system, "A", likeliest(stored)), Row("EchoPrime + decoder", system, "D", decoded(stored, ECHOPRIME_OCTOBER))]
        else:
            rows.append(Row(system, system, "A", likeliest(stored)))
            if system in settings.get("systems", {}):
                rows.append(Row(f"{system} + decoder", system, "D", decoded(stored, settings["systems"][system]["params"])))
        rows.append(Row(f"{system} + released numbers", system, "C", decoded(stored, B3)))
    return rows


def published_check(rows: list[Row], sequences: tuple[Sequence, ...]) -> dict[str, dict[str, float]]:
    """The published numbers of every published row, from the stored outputs. Stops when one differs."""
    got, off = {}, {}
    for row in rows:
        if row.name in PUBLISHED:
            numbers = metrics.evaluate(row.labels, sequences)
            got[row.name] = {key: float(numbers[key]) for key in PUBLISHED[row.name]}
            off.update({f"{row.name}: {key}": got[row.name][key] for key, value in PUBLISHED[row.name].items() if abs(got[row.name][key] - value) > CLOSE})
    missing = [name for name in PUBLISHED if name not in got]
    if off or missing:
        raise SystemExit(f"the stored outputs do not give the published numbers; nothing was scored. Differing: {off}. Rows without outputs: {missing}")
    return got


def discordant(natives: tuple[Sequence, ...], first: Row, second: Row) -> dict[str, int]:
    """Single recordings that only one of two rows reads right."""
    def right(row: Row) -> np.ndarray:
        return np.array([int(np.bincount(row.labels[s.id], minlength=N).argmax()) == int(s.labels[0]) for s in natives])

    a, b = right(first), right(second)
    return {"only_first_right": int((a & ~b).sum()), "only_second_right": int((~a & b).sum())}


# ------------------------------------------------------------------------------------------------------ scoring
def score_rows(rows: list[Row], sequences: tuple[Sequence, ...], draws: int, seed: int) -> dict[str, dict[str, Scored]]:
    """Every row scored on the multi streams, the pair streams and the single recordings, with shared resamples."""
    natives = of_kind(sequences, "native")
    grouped = all(s.group >= 0 for s in natives)
    examination = {s.fragments[0].video: s.group for s in natives} if grouped else None
    generator = np.random.default_rng(seed)
    parts: dict[str, tuple] = {}
    for kind in ("multi", "pairs"):
        streams = of_kind(sequences, kind)
        units = {"streams": len(streams), **({"examinations": max(examination.values()) + 1} if examination else {})}
        parts[kind] = (streams, {unit: bootstrap.resampling_weights(generator, n, draws) for unit, n in units.items()})
    native_unit = "examinations" if grouped else "recordings"
    native_units = (max(s.group for s in natives) + 1) if grouped else len(natives)
    native_weights = {native_unit: bootstrap.resampling_weights(generator, native_units, draws)}
    out: dict[str, dict[str, Scored]] = {}
    for row in rows:
        out[row.name] = {kind: score_tables(STREAM_STATS, stream_tables(streams, row.labels, examination), STREAM_INDEX, weights)
                         for kind, (streams, weights) in parts.items()}
        out[row.name]["recordings"] = score_tables(NATIVE_STATS, native_tables(natives, row.labels), NATIVE_INDEX, native_weights)
    return out


def named_comparisons(rows: list[Row], scored: Mapping[str, Mapping[str, Scored]], first: str = ROUTER) -> list[dict]:
    """Segment F1 at 50% on multi streams: `first` minus each other system with the project's decoder, and the
    published gap in two parts. The word rests on the hull of the intervals by unit, so every unit must agree. What
    the decoder adds to a system is measured against its row without the decoder: the published row, or for
    EchoPrime's classifier, whose published row has the decoder, the row per frame."""
    out = []
    for system in OTHERS:
        as_published = "EchoPrime + decoder" if system == "EchoPrime" else system
        without_decoder = "EchoPrime per frame" if system == "EchoPrime" else system
        with_decoder = next((r.name for r in rows if r.system == system and r.role == "D"), None)
        if as_published not in scored:
            continue
        entry: dict = {"system": system, "row_as_published": as_published, "row_without_decoder": without_decoder, "row_with_decoder": with_decoder}
        gap = difference(scored[first]["multi"], scored[as_published]["multi"], "segment_f1_50", DESCRIBED)
        entry["published_gap"] = gap
        if with_decoder is None:
            entry["word"] = "not answered"
        else:
            decides = difference(scored[first]["multi"], scored[with_decoder]["multi"], "segment_f1_50", NAMED)
            entry["named"] = decides
            entry["remains"] = difference(scored[first]["multi"], scored[with_decoder]["multi"], "segment_f1_50", DESCRIBED)
            entry["decoder_adds"] = difference(scored[with_decoder]["multi"], scored[without_decoder]["multi"], "segment_f1_50", DESCRIBED)
            said = word(decides["hull"])
            if said == "ahead of" and not (gap["hull"] and gap["hull"][0] > 0):
                said = "not shown to differ from"
            entry["word"] = said
        out.append(entry)
    return out


def best_rows(rows: list[Row], scored: Mapping[str, Mapping[str, Scored]]) -> dict[str, dict[str, list[str]]]:
    """For every number with a direction, the rows of roles A, D and N with the best value; equal values together."""
    out: dict[str, dict[str, list[str]]] = {}
    candidates = [r.name for r in rows if r.role in ("A", "D", "N")]
    for part, stats in (("multi", STREAM_STATS), ("pairs", STREAM_STATS), ("recordings", NATIVE_STATS)):
        out[part] = {}
        for stat in stats:
            values = {name: scored[name][part].values[stat.name] for name in candidates if np.isfinite(scored[name][part].values[stat.name])}
            if stat.direction and values:
                best = max(values.values()) if stat.direction == "up" else min(values.values())
                out[part][stat.name] = sorted(name for name, value in values.items() if abs(value - best) <= 1e-12)
    return out


def compare(sequences: tuple[Sequence, ...], rows: list[Row], draws: int = DRAWS, seed: int = SEED) -> dict:
    """The whole comparison of the given rows: every number with intervals, differences from EchoRouter, the named
    comparisons, the seeds, the decoder on EchoRouter and the best rows."""
    scored = score_rows(rows, sequences, draws, seed)
    natives = of_kind(sequences, "native")
    by_name = {row.name: row for row in rows}
    stats_of = {"multi": STREAM_STATS, "pairs": STREAM_STATS, "recordings": NATIVE_STATS}
    result: dict = {"draws": draws, "seed": seed, "rows": {}, "differences_from_echorouter": {}}
    for row in rows:
        result["rows"][row.name] = {"system": row.system, "role": row.role, **{part: report(s) for part, s in scored[row.name].items()}}
        if row.name != ROUTER:
            result["differences_from_echorouter"][row.name] = {
                part: {stat.name: difference(scored[ROUTER][part], scored[row.name][part], stat.name, DESCRIBED) for stat in stats_of[part]}
                for part in stats_of}
            result["differences_from_echorouter"][row.name]["recordings_only_one_reads_right"] = discordant(natives, by_name[ROUTER], row)
    result["named_comparisons"] = named_comparisons(rows, scored)
    result["seeds"] = {row.name: named_comparisons(rows, scored, row.name) for row in rows if row.role == "S"}
    no_decoder = f"{ROUTER}, no decoder"
    result["decoder_on_echorouter"] = {part: difference(scored[ROUTER][part], scored[no_decoder][part], "segment_f1_50", DESCRIBED)
                                       for part in ("multi", "pairs")}
    result["best_rows"] = best_rows(rows, scored)
    return result


def reproduced(earlier: Mapping, result: Mapping) -> dict:
    """Every number of an earlier output of `score` must come out again in `result`, which may hold more rows.

    A later read that adds a row (the plan's addendum) is held to this: the rows, the differences from EchoRouter,
    the named comparisons, the seeds and the decoder on EchoRouter of the earlier output are compared as written,
    number by number. A comparison that had no row with the decoder then is held to its published gap. Which rows
    are best is taken again and is not compared. Stops when anything differs."""
    new = json.loads(json.dumps(_clean(result), allow_nan=False))
    problems = [key for key in ("benchmark", "sequences", "draws", "seed", "published_numbers_reproduced", "decoder_on_echorouter")
                if earlier.get(key) != new.get(key)]
    for part in ("rows", "differences_from_echorouter"):
        problems += [f"{part}: {name}" for name, numbers in earlier[part].items() if new[part].get(name) != numbers]
    listed = [("named_comparisons", earlier["named_comparisons"], new["named_comparisons"])]
    listed += [(f"seeds: {name}", entries, new["seeds"].get(name, [])) for name, entries in earlier["seeds"].items()]
    for label, before, after in listed:
        now = {entry["system"]: entry for entry in after}
        for entry in before:
            found = now.get(entry["system"])
            same = found is not None and (found == entry if entry["row_with_decoder"] else found["published_gap"] == entry["published_gap"])
            if not same:
                problems.append(f"{label}: {entry['system']}")
    if problems:
        raise SystemExit("the earlier output does not come out again: " + "; ".join(problems[:10]))
    return {"rows": len(earlier["rows"]), "new_rows": sorted(set(new["rows"]) - set(earlier["rows"]))}


# ----------------------------------------------------------------------------------------------------- commands
def _stores(given: list[str]) -> dict[str, Path]:
    stores = {}
    for text in given:
        name, _, folder = text.partition("=")
        if name not in OTHERS or not folder:
            raise SystemExit(f"--store takes NAME=FOLDER with NAME one of {', '.join(OTHERS)}")
        stores[name] = Path(folder)
    return stores


def _names(images: Path, sequences: tuple[Sequence, ...]) -> dict[str, str]:
    return folders_by_key(images, {s.fragments[0].video for s in sequences if s.kind == "native"})


def _logits(path: Path, sequences: tuple[Sequence, ...]) -> dict[str, np.ndarray]:
    with np.load(path) as stored:
        missing = sum(s.id not in stored.files for s in sequences)
        if missing:
            raise SystemExit(f"{path} lacks the logits of {missing} of the {len(sequences)} sequences")
        return {s.id: stored[s.id] for s in sequences}


def _scored_kinds(sequences: tuple[Sequence, ...]) -> tuple[Sequence, ...]:
    return tuple(s for s in sequences if s.kind in ("native", "pairs", "multi"))


def _clean(value: object) -> object:
    """The value as plain JSON types; a number that is not finite becomes null."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write(result: dict, out: Path) -> None:
    text = json.dumps(_clean(result), indent=1, allow_nan=False)  # built whole before the file exists
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:  # never replaces a file
        handle.write(text + "\n")


def inventory(args: argparse.Namespace) -> dict:
    halves, test = benchmark.load(), _scored_kinds(benchmark.load_test())
    sets = {"test": test, **{half: _scored_kinds(halves[half]) for half in benchmark.HALVES}}
    names = _names(args.images, tuple(s for group in sets.values() for s in group))
    return {name: {"files": sum(1 for _ in folder.glob("*.npz")), **{label: coverage(folder, group, names) for label, group in sets.items()}}
            for name, folder in _stores(args.store).items()}


def choose(probabilities: Mapping[str, np.ndarray], sequences: tuple[Sequence, ...], drop_floors: bool) -> dict:
    """The rule of calibrate.py on the calibration half. With `drop_floors`, a system for which no setting meets the
    floors gets the setting with the highest segment F1@50 on the multi streams instead of the diagnostic one."""
    chosen = calibrate.choose(probabilities, sequences)
    if chosen["status"] == "feasible" or not drop_floors:
        params = chosen["params"]
    else:
        params = chosen["best_unconstrained"]["params"]
    numbers = metrics.evaluate(decoded(probabilities, params), sequences)
    return {"status": chosen["status"], "n_feasible": chosen["n_feasible"], "floors_dropped": chosen["status"] != "feasible" and drop_floors,
            "params": params, "calibration": {key: numbers[key] for key in calibrate.REPORTED}}


def settings(args: argparse.Namespace) -> dict:
    stores = _stores(args.store)
    if "EchoPrime" in stores:
        raise SystemExit("EchoPrime's classifier keeps the setting chosen for it in October; give no store for it here")
    calibration = _scored_kinds(benchmark.load()["calibration"])
    logits = _logits(args.logits, calibration)
    members = tuple(range(next(iter(logits.values())).shape[1]))
    ensemble = choose(member_probabilities(logits, calibration, members), calibration, drop_floors=False)
    found = ensemble["params"]
    released = (ensemble["status"] == "feasible" and found["half_window"] == B3["half_window"] and found["nms_radius"] == B3["nms_radius"]
                and found["threshold_quantile"] == RELEASED_QUANTILE and abs(found["threshold"] - B3["threshold"]) <= THRESHOLD_CLOSE)
    if not released:
        raise SystemExit(f"these logits do not give the released decoder setting on the calibration half: {found}")
    out: dict = {"half": "calibration", "ensemble": ensemble, "systems": {}, "stores": {},
                 "without_development_outputs": sorted(set(OTHERS) - set(stores) - {"EchoPrime"}),
                 "seeds": {str(m): choose(member_probabilities(logits, calibration, (m,)), calibration, drop_floors=False) for m in members}}
    names = _names(args.images, calibration)
    for name, folder in stores.items():
        stored, digest = read_store(folder, calibration, names)
        out["systems"][name] = choose(stored, calibration, drop_floors=True)
        out["stores"][name] = {"sequences": len(calibration), "digest": digest}
    return out


def check_test_setup(args: argparse.Namespace, chosen: Mapping, stores: Mapping[str, Path], logits_digest: str) -> None:
    """The one read of the test benchmark is made with everything the plan fixes, or not at all."""
    problems = []
    if args.draws != DRAWS:
        problems.append(f"the plan fixes {DRAWS} draws")
    if set(stores) != set(OTHERS):
        problems.append(f"all four stores are needed: {', '.join(OTHERS)}")
    if logits_digest != TEST_LOGITS_SHA256:
        problems.append("the logits file is not the one the plan names")
    if set(chosen.get("seeds", {})) != {"0", "1", "2"}:
        problems.append("the settings file must hold a setting for each of the three seeds")
    decided = set(chosen.get("systems", {})) | set(chosen.get("without_development_outputs", [])) | {"EchoPrime"}
    if decided != set(OTHERS) or set(chosen.get("systems", {})) & set(chosen.get("without_development_outputs", [])):
        problems.append("the settings file must say for every other system either its setting or that it has no development outputs")
    if problems:
        raise SystemExit("the test benchmark is not read: " + "; ".join(problems))


def score(args: argparse.Namespace) -> dict:
    test = args.benchmark == "test"
    sequences = _scored_kinds(benchmark.load_test() if test else benchmark.load()[args.benchmark])
    chosen = json.loads(args.settings.read_text())
    names = _names(args.images, sequences)
    folders, stores, digests = _stores(args.store), {}, {"logits": file_sha256(args.logits), "settings": file_sha256(args.settings)}
    if test:
        check_test_setup(args, chosen, folders, digests["logits"])
    for name, folder in folders.items():
        stores[name], digests[name] = read_store(folder, sequences, names)
    rows = build_rows(sequences, _logits(args.logits, sequences), stores, chosen)
    if test:
        checked = published_check(rows, sequences)
    else:
        f1 = metrics.evaluate(rows[0].labels, sequences)["segment_f1_50_multi"]
        if round(f1, 4) != PUBLISHED_SELECTION_F1 or args.benchmark != "selection":
            raise SystemExit(f"{args.benchmark}: only the selection half is scored here, and its logits must give {PUBLISHED_SELECTION_F1}; got {f1:.4f}")
        checked = {ROUTER: {"segment_f1_50_multi": float(f1)}}
    result = {"benchmark": args.benchmark, "sequences": len(sequences), "sha256": digests, "settings": chosen, "published_numbers_reproduced": checked,
              **compare(sequences, rows, args.draws, SEED)}
    if args.earlier:
        result["earlier_output"] = {"sha256": file_sha256(args.earlier), **reproduced(json.loads(args.earlier.read_text()), result)}
    return result


def main(argv: list[str] | None = None) -> int:
    top = argparse.ArgumentParser(prog="python -m echoview.evaluation.comparison", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = top.add_subparsers(dest="command", required=True)
    for name, run in (("inventory", inventory), ("settings", settings), ("score", score)):
        p = commands.add_parser(name)
        p.add_argument("--images", type=Path, required=True, help="the Images folder of EV9V; it gives the stored files of single recordings their names")
        p.add_argument("--store", action="append", default=[], help="NAME=FOLDER, the stored outputs of another system; repeat for every system")
        p.add_argument("--out", type=Path, required=True, help="JSON file to write; an existing file is never replaced")
        if name != "inventory":
            p.add_argument("--logits", type=Path, required=True, help="the window logits of EchoRouter's three seeds for the benchmark")
        if name == "score":
            p.add_argument("--benchmark", choices=("test", "selection"), required=True)
            p.add_argument("--settings", type=Path, required=True, help="the output of the settings command")
            p.add_argument("--draws", type=int, default=DRAWS)
            p.add_argument("--earlier", type=Path, help="an earlier output of this command for the same benchmark: every number of it must "
                                                        "come out again, or nothing is written")
        p.set_defaults(run=run)
    args = top.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; this command does not replace an output")
    _write(args.run(args), args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
