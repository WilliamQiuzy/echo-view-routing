"""Which windows a training update reads: the sampling plans.

A plan file holds windows as (recording, frame number) per slot, the marker slot and the label at the marker. The
plans were drawn once, during the project's development, from the training recordings and from streams built by
joining training recordings with hard cuts. They are published with the weights (`training/plans/` on the Hub) and
name recordings by key. A window is 16 rows of the frame cache.

  GroupPool   Groups of four neighbouring windows (stages 2 and 3). Half of the groups lie inside one recording, half
              cross at least one view change. The smoothing penalty compares the outputs of neighbouring windows.
  Pool        Single windows (stage 1 and plain cross-entropy recipes): mixed windows from the plan, and windows inside
              one recording drawn fresh: class first, then recording, then centre.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SLOTS = 16
GROUP = 4  # neighbouring windows in one group
MIXED = 1  # the plans' kind of a window cut from more than one recording


def fragment_numbers(vid: np.ndarray) -> np.ndarray:
    """(M, 16) recording index per slot -> (M, 16) fragment number: 0 until the first change, then 1, 2, ..."""
    change = np.concatenate([np.zeros((len(vid), 1), np.int64), (vid[:, 1:] != vid[:, :-1]).astype(np.int64)], 1)
    return np.cumsum(change, 1)


def native_window(start: int, count: int, centre: int) -> tuple[np.ndarray, int]:
    """Rows and marker slot of the window centred at `centre` of a recording that starts at row `start`."""
    if not 0 <= centre < count:
        raise ValueError("centre outside the recording")
    first = max(0, min(centre - SLOTS // 2, count - SLOTS))
    samples = np.arange(first, min(count, first + SLOTS))
    samples = samples[np.linspace(0, len(samples) - 1, SLOTS).round().astype(int)]
    slots = np.flatnonzero(samples == centre)
    return start + samples, int(slots[len(slots) // 2])


def read_plan(path: str | Path, cache, step: int = 3) -> dict[str, np.ndarray]:
    """One plan file as cache rows: rows (W, 16), marker (W,), target (W,), kind (W,) and vid (W, 16)."""
    with np.load(path, allow_pickle=False) as stored:
        keys = [str(key) for key in stored["recordings"]]
        vid, frame = stored["vid"].reshape(-1, SLOTS), stored["frame"].reshape(-1, SLOTS)
        marker, target, kind = (stored[name].reshape(-1).astype(np.int64) for name in ("marker", "target", "kind"))
    starts = np.array([cache.by_key[key]["start"] if key in cache.by_key else -1 for key in keys], np.int64)
    counts = np.array([cache.by_key[key]["count"] if key in cache.by_key else 0 for key in keys], np.int64)
    if (frame % step).any() or (starts[vid] < 0).any() or (frame // step >= counts[vid]).any():
        raise ValueError(f"{path}: a window refers to a frame outside the frame cache")
    return {"rows": starts[vid] + frame // step, "marker": marker, "target": target, "kind": kind, "vid": vid.astype(np.int64)}


class GroupPool:
    """Groups of four neighbouring windows from the given plan files; a batch is a random draw of whole groups."""

    def __init__(self, cache, plans: list[str | Path]) -> None:
        self.train_end = cache.train_end
        rows, marker, target, vids = [], [], [], []
        offset = 0
        for path in plans:
            plan = read_plan(path, cache)
            if len(plan["rows"]) % (2 * GROUP):
                raise ValueError(f"{path}: the windows do not form whole groups")
            rows.append(plan["rows"].reshape(-1, GROUP, SLOTS))
            marker.append(plan["marker"].reshape(-1, GROUP))
            target.append(plan["target"].reshape(-1, GROUP))
            vids.append((plan["vid"] + offset).reshape(-1, GROUP, SLOTS))
            offset += int(plan["vid"].max()) + 1
        self.rows, self.marker, self.target = np.concatenate(rows), np.concatenate(marker), np.concatenate(target)
        self.fragment = fragment_numbers(np.concatenate(vids).reshape(-1, SLOTS)).reshape(-1, GROUP, SLOTS)
        if self.rows.max() >= self.train_end:
            raise ValueError("a group reads a row outside the training recordings")

    def batch(self, rng: np.random.Generator, n: int, native_share: float = 0.5) -> tuple[np.ndarray, ...]:
        """`n` windows as n / 4 whole groups: rows, marker, target, fragment (native_share is unused: the plans hold
        both kinds of group one to one)."""
        if n % GROUP:
            raise ValueError("the batch must hold whole groups")
        pick = rng.integers(0, len(self.rows), n // GROUP)
        return tuple(a[pick].reshape(n, *a.shape[2:]) for a in (self.rows, self.marker, self.target, self.fragment))


class Pool:
    """Single windows: the mixed windows of the plan files, and fresh windows inside one training recording."""

    def __init__(self, cache, plans: list[str | Path], classes: int = 5) -> None:
        train = cache.rows("train")
        self.train_end = cache.train_end
        self.by_class = [np.array([(c["start"], c["count"]) for c in train if c["label"] == k], np.int64) for k in range(classes)]
        rows, marker, target, vids = [], [], [], []
        offset = 0
        for path in plans:
            plan = read_plan(path, cache)
            mixed = plan["kind"] == MIXED
            rows.append(plan["rows"][mixed])
            marker.append(plan["marker"][mixed])
            target.append(plan["target"][mixed])
            vids.append(plan["vid"][mixed] + offset)
            offset += int(plan["vid"].max()) + 1
        self.rows, self.marker, self.target = np.concatenate(rows), np.concatenate(marker), np.concatenate(target)
        self.fragment = fragment_numbers(np.concatenate(vids))
        if self.rows.max() >= self.train_end:
            raise ValueError("a mixed window reads a row outside the training recordings")

    def mixed(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, ...]:
        pick = rng.integers(0, len(self.rows), n)
        return self.rows[pick], self.marker[pick], self.target[pick], self.fragment[pick]

    def native(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, ...]:
        rows, marker, target = np.empty((n, SLOTS), np.int64), np.empty(n, np.int64), rng.integers(0, len(self.by_class), n)
        for i, k in enumerate(target):
            start, count = self.by_class[k][rng.integers(0, len(self.by_class[k]))]
            rows[i], marker[i] = native_window(int(start), int(count), int(rng.integers(0, count)))
        return rows, marker, target.astype(np.int64), np.zeros((n, SLOTS), np.int64)

    def batch(self, rng: np.random.Generator, n: int, native_share: float = 0.5) -> tuple[np.ndarray, ...]:
        k = round(n * native_share)
        parts = [self.native(rng, k), self.mixed(rng, n - k)]
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))
