"""Training windows for target-time supervision (configs/experiments/ev9v_center_supervision_v1.yaml, 2026-10-05).

A window holds the 16 frames that B7's rule takes around a target sample c of a 10 Hz sequence: window_samples
clamped to the sequence, then 16 evenly spaced frames (echo_routing.evaluate.window_target.window_frames). It records
c, the sample indices s[0:16], the frame references, the 16 frame labels and the relative positions s[i] - c. The
positions come from the sampling time axis, never from labels or joins. Labels are the public recording labels and
serve only the losses.

Every update holds 4 native windows (class, recording and target sample drawn uniformly), shared by all arms, and 4
more windows. For S and C these are mixed-view windows of the training streams (target class, target recording and
window drawn uniformly). For N each is replaced by the target recording's own native window at the same target frame
and contrast variant. The data generator has its own RNG, so S and C read one identical manifest.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from echo_routing.evaluate.window_target import window_frames

N_FRAMES = 16
NUM_CLASSES = 5
KIND = {"native": 0, "mixed": 1, "replacement": 2}
GAMMA = {"orig": 1.0, "gamma090": 0.9}


@dataclass(frozen=True)
class Track:
    """One 10 Hz sequence of the training data: a native clip or a constructed training stream."""
    seq_id: str
    videos: tuple[str, ...]     # source recording of each sample
    frames: np.ndarray          # actual frame index of each sample (cache_v2, numeric order)
    gammas: np.ndarray          # contrast gamma of each sample
    labels: np.ndarray          # public recording label of each sample
    clip_index: np.ndarray      # index of each sample inside its source recording


def native_track(video: str, frames: np.ndarray, label: int, gamma: float = 1.0) -> Track:
    n = len(frames)
    return Track(video, (video,) * n, np.asarray(frames, int), np.full(n, float(gamma)), np.full(n, int(label)), np.arange(n))


def window(track: Track, c: int) -> dict:
    """The window B7 would feed for target sample c, with its relative positions and labels."""
    n = len(track.labels)
    if not 0 <= int(c) < n:
        raise IndexError(f"{track.seq_id}: target {c} outside 0..{n - 1}")
    s = window_frames(int(c), n, N_FRAMES, N_FRAMES)
    rel = s - int(c)
    if int(np.abs(rel).max()) > N_FRAMES - 1 or int(c) not in set(s.tolist()):
        raise ValueError(f"{track.seq_id}: window around {c} breaks the position contract")
    return {"c": int(c), "s": s, "rel": rel, "labels": track.labels[s].astype(int), "videos": [track.videos[i] for i in s],
            "frames": track.frames[s].astype(int), "gammas": track.gammas[s].astype(float), "target_video": track.videos[int(c)],
            "target_label": int(track.labels[int(c)]), "repeats": N_FRAMES - len(set(s.tolist()))}


def is_mixed(labels: Sequence[int]) -> bool:
    """At least two labels among the window's frames. Several recordings of one label do not count."""
    return len({int(x) for x in labels}) >= 2


def mixed_index(streams: Sequence[Track]) -> dict[int, dict[str, list[tuple[int, int]]]]:
    """class -> target recording -> [(stream index, c)] of every mixed-view window, keyed by its target sample."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for i, t in enumerate(streams):
        n = len(t.labels)
        for c in range(n):
            if is_mixed(t.labels[window_frames(c, n, N_FRAMES, N_FRAMES)]):
                out[int(t.labels[c])][t.videos[c]].append((i, c))
    return {k: {v: list(w) for v, w in d.items()} for k, d in out.items()}


def check_train_only(videos: set[str], train_clips: set[str]) -> None:
    bad = sorted(set(videos) - set(train_clips))
    if bad:
        raise ValueError(f"{len(bad)} recordings outside the protocol's training clips, e.g. {bad[:3]}")


def draw_native(rng: np.random.Generator, natives: dict[int, list[Track]]) -> tuple[Track, int]:
    k = int(rng.integers(NUM_CLASSES)); pool = natives[k]; t = pool[int(rng.integers(len(pool)))]
    c = int(rng.integers(len(t.labels))); g = GAMMA["orig"] if rng.random() < 0.5 else GAMMA["gamma090"]
    return replace(t, gammas=np.full(len(t.labels), g)), c


def draw_mixed(rng: np.random.Generator, mix: dict[int, dict[str, list[tuple[int, int]]]]) -> tuple[int, int]:
    k = int(rng.integers(NUM_CLASSES)); vids = sorted(mix[k]); v = vids[int(rng.integers(len(vids)))]
    cands = mix[k][v]
    return cands[int(rng.integers(len(cands)))]


def replacement(stream: Track, c: int, natives_by_video: dict[str, Track]) -> tuple[Track, int]:
    """N's window for a mixed window: the target recording's native window at the same frame and contrast variant."""
    v = stream.videos[c]; base = natives_by_video[v]
    return replace(base, gammas=np.full(len(base.labels), float(stream.gammas[c]))), int(stream.clip_index[c])


def build_manifests(seed: int, data_seed_base: int, natives: dict[int, list[Track]], streams: Sequence[Track],
                    mix: dict, n_updates: int, n_native: int = 4, n_mixed: int = 4) -> tuple[dict, dict]:
    """Arrays of every update's windows for S/C (shared) and for N. Both use the same draws, from one data RNG."""
    rng = np.random.default_rng([int(data_seed_base), int(seed)])
    by_video = {t.seq_id: t for k in natives for t in natives[k]}
    sc, n_arm = [], []
    for _ in range(n_updates):
        nat = [window(*draw_native(rng, natives)) | {"kind": KIND["native"], "seq": -1} for _ in range(n_native)]
        picks = [draw_mixed(rng, mix) for _ in range(n_mixed)]
        mixed = [window(streams[i], c) | {"kind": KIND["mixed"], "seq": i} for i, c in picks]
        repl = [window(*replacement(streams[i], c, by_video)) | {"kind": KIND["replacement"], "seq": -1} for i, c in picks]
        for w, r in zip(mixed, repl):
            if w["target_video"] != r["target_video"] or w["target_label"] != r["target_label"] or w["frames"][w["s"] == w["c"]][0] != r["frames"][r["s"] == r["c"]][0]:
                raise ValueError("an N replacement does not keep the mixed window's target frame")
        sc.append(nat + mixed); n_arm.append(nat + repl)
    return pack(sc), pack(n_arm)


def pack(updates: list[list[dict]]) -> dict:
    videos = sorted({v for u in updates for w in u for v in w["videos"]})
    at = {v: i for i, v in enumerate(videos)}
    arr = lambda key, dtype: np.asarray([[w[key] for w in u] for u in updates], dtype=dtype)  # noqa: E731
    return {"videos": np.asarray(videos), "vid": np.asarray([[[at[v] for v in w["videos"]] for w in u] for u in updates], np.int32),
            "frame": arr("frames", np.int32), "gamma": arr("gammas", np.float64), "rel": arr("rel", np.int8),
            "labels": arr("labels", np.int8), "s": arr("s", np.int32), "target": arr("target_label", np.int8),
            "target_vid": np.asarray([[at[w["target_video"]] for w in u] for u in updates], np.int32), "c": arr("c", np.int32),
            "kind": arr("kind", np.int8), "seq": arr("seq", np.int32), "repeats": arr("repeats", np.int8)}


def epoch_digest(m: dict, start: int, stop: int) -> str:
    """Digest of the inputs and targets of updates [start, stop)."""
    h = hashlib.sha256(); h.update("|".join(m["videos"].tolist()).encode())
    for key in ("vid", "frame", "gamma", "rel", "labels", "target", "kind"):
        h.update(np.ascontiguousarray(m[key][start:stop]).tobytes())
    return h.hexdigest()


def frame_shares(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """S's window target: the share of each class among the window's 16 input positions (repeats counted as fed)."""
    labels = np.asarray(labels, int)
    return np.stack([(labels == k).mean(-1) for k in range(num_classes)], -1)
