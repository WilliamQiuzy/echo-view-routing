#!/usr/bin/env python
"""Write the frozen inputs of the raw-video context round (plan Task 1; configs/experiments/raw_video_context_v1.yaml).

Checks every pinned source: the task sheet, the protocol, the SC manifests and their index, the EchoPrime commit,
loader and checkpoint, torchvision's MViT source and the environment versions. The protocol's training clips are the
allowlist. Every recording of a seed's SC manifest must be on it, and frame paths come from the native manifest's
video_id and frames_dir columns of allowlisted rows only.

Per seed it writes RV_s{seed}.npz:
  vid, frame, gamma, s, c, target, kind, seq, repeats, videos   copied from SC_s{seed}.npz (frame labels are not copied)
  marker (U, 8)              the marker slot, the middle of the positions with s == c
  n_target_slots (U, 8)
  order_<group> (U, 8, 16)   index into the time-ordered slots for each input slot (Ordered: identity; Shuffled: the
                             training permutation; Target-only: the marker slot)
  digest_<group> (U, 8, 32)  SHA256 over the 16 slot pixel digests in input order
and init_s{seed}.pt, the common initial state of the classifier and the marker stem. Every distinct (recording, frame,
gamma) is decoded once with load_frame_u8 for its pixel digest. index.json holds the files' digests, the per-epoch input
digests and the statistics. The id-free record goes to the results directory.

  python scripts/prepare_raw_video_context.py --config configs/experiments/raw_video_context_v1.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402
from echo_routing.temporal.joint_visual import load_frame_u8  # noqa: E402

COPIED = ("vid", "frame", "gamma", "s", "c", "target", "kind", "seq", "repeats")
WINDOWS, SLOTS = 8, tm.N_SLOTS


# ----------------------------------------------------------------------------------------------- sources
def check_sources(rv: dict, root: Path) -> dict:
    """Every pinned file digest, the EchoPrime commit, torchvision's MViT source and the environment versions."""
    import torchvision
    import torchvision.models.video.mvit as tv_mvit
    src = rv["source"]; out = {}
    pins = {"task_sheet": rv["task_sheet"], "protocol": src["protocol"], "sc_index": src["sc_manifests"]["index"],
            "sc_config": src["sc_manifests"]["config"], "echoprime_loader": src["echoprime"]["loader"], "echoprime_checkpoint": src["echoprime"]["checkpoint"],
            **{f"sc_s{k}": v for k, v in src["sc_manifests"]["files"].items()}}
    for name, pin in pins.items():
        got = sha256_file(root / pin["path"])
        if got != pin["sha256"]:
            raise SystemExit(f"{name} {pin['path']}: sha256 {got[:12]} != {pin['sha256'][:12]}")
        out[name] = got
    index = json.loads((root / src["sc_manifests"]["index"]["path"]).read_text())
    for k, pin in src["sc_manifests"]["files"].items():
        if index["seeds"][str(k)]["files"]["SC"]["sha256"] != pin["sha256"]:
            raise SystemExit(f"SC_s{k}: the pin differs from the source index")
    head = subprocess.run(["git", "-C", str(root / src["echoprime"]["repo"]), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != src["echoprime"]["commit"]:
        raise SystemExit(f"EchoPrime commit {head} != {src['echoprime']['commit']}")
    env = environment()
    want = src["environment"]
    if env["torchvision_mvit_sha256"] != want["torchvision_mvit_sha256"] or any(str(env[k]) != str(want[k]) for k in ("torch", "torchvision", "cuda", "cudnn")):
        raise SystemExit(f"environment {env} differs from the pinned {want}")
    return {**out, "echoprime_commit": head, "environment": env, "torchvision_file": str(Path(tv_mvit.__file__).name), "torchvision_version": torchvision.__version__}


def environment() -> dict:
    import torchvision
    import torchvision.models.video.mvit as tv_mvit
    return {"torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "torchvision_mvit_sha256": sha256_file(Path(tv_mvit.__file__))}


def allowlist(rv: dict, root: Path) -> list[str]:
    return sorted(json.loads((root / rv["source"]["protocol"]["path"]).read_text())["train"]["clips"])


def frame_dirs(paths, rv: dict, keep: set[str]) -> dict[str, str]:
    """video_id -> frames_dir for allowlisted recordings, reading only the two whitelisted columns."""
    cols = list(rv["source"]["frame_manifest_columns"])
    if cols != ["video_id", "frames_dir"]:
        raise SystemExit(f"frame_manifest_columns must be [video_id, frames_dir], got {cols}")
    df = pd.read_csv(paths.manifest_path, usecols=cols, dtype=str)
    df = df[df["video_id"].isin(keep)]
    out = dict(zip(df["video_id"], df["frames_dir"]))
    missing = sorted(keep - set(out))
    if missing:
        raise SystemExit(f"{len(missing)} allowlisted recordings have no frames directory")
    return out


# ----------------------------------------------------------------------------------------------- windows
def window_orders(seed: int, s: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Marker slots, target-slot counts and the three groups' slot orders of every window of one seed."""
    n_upd, n_win, _ = s.shape
    marker = np.empty((n_upd, n_win), np.int8); n_t = np.empty((n_upd, n_win), np.int8)
    orders = {g: np.empty((n_upd, n_win, SLOTS), np.int8) for g in tm.GROUPS}
    for u in range(n_upd):
        for w in range(n_win):
            sw, cw = s[u, w], int(c[u, w])
            marker[u, w] = tm.marker_slot(sw, cw); n_t[u, w] = tm.target_slots(sw, cw).size
            shuffled = tm.shuffle_order(tm.train_rng(seed, u, w), sw, cw)
            for g in tm.GROUPS:
                orders[g][u, w] = tm.group_order(g, sw, cw, shuffled)
    return marker, n_t, orders


def _hash_video(args: tuple) -> list[tuple[int, float, bytes]]:
    frames_dir, wanted = args
    paths = list_frame_paths(Path(frames_dir)); out = []
    for f, g in wanted:
        out.append((int(f), float(g), tm.frame_hash(load_frame_u8(paths[min(int(f), len(paths) - 1)], round(float(g), 6)))))
    return out


def pixel_hashes(refs: set[tuple[str, int, float]], dirs: dict[str, str], workers: int) -> dict[tuple[str, int, float], bytes]:
    """Pixel digest of every distinct (recording, frame, gamma), decoded once with the training preprocessing."""
    by_video: dict[str, list] = defaultdict(list)
    for v, f, g in sorted(refs):
        by_video[v].append((f, g))
    vids = sorted(by_video); out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for v, res in zip(vids, ex.map(_hash_video, [(dirs[v], by_video[v]) for v in vids], chunksize=8)):
            for f, g, h in res:
                out[(v, f, g)] = h
    return out


def group_digests(m: dict, orders: dict, hashes: dict) -> dict[str, np.ndarray]:
    videos = m["videos"]; n_upd, n_win, _ = m["vid"].shape
    out = {g: np.empty((n_upd, n_win, 32), np.uint8) for g in tm.GROUPS}
    for u in range(n_upd):
        for w in range(n_win):
            slot = [hashes[(str(videos[m["vid"][u, w, i]]), int(m["frame"][u, w, i]), round(float(m["gamma"][u, w, i]), 6))] for i in range(SLOTS)]
            for g in tm.GROUPS:
                out[g][u, w] = np.frombuffer(tm.window_digest([slot[j] for j in orders[g][u, w]]), np.uint8)
    return out


def shuffle_stats(m: dict, orders: dict, hashes: dict, n_t: np.ndarray) -> dict:
    """How often the shuffle left pixels unchanged: identity permutations, unchanged non-target slots, unchanged windows."""
    videos = m["videos"]; o = orders["Shuffled context"]; n_upd, n_win, _ = o.shape
    same_win = ident = moved_same = nontarget = 0
    for u in range(n_upd):
        for w in range(n_win):
            h = [hashes[(str(videos[m["vid"][u, w, i]]), int(m["frame"][u, w, i]), round(float(m["gamma"][u, w, i]), 6))] for i in range(SLOTS)]
            other = np.flatnonzero(m["s"][u, w] != m["c"][u, w])
            nontarget += other.size
            ident += int(np.array_equal(o[u, w], np.arange(SLOTS)))
            moved_same += sum(h[i] == h[o[u, w, i]] for i in other)
            same_win += int(all(h[i] == h[o[u, w, i]] for i in range(SLOTS)))
    total = n_upd * n_win
    return {"windows": total, "identity_permutations": ident, "windows_with_unchanged_pixels": same_win,
            "share_windows_unchanged": same_win / total, "nontarget_slots": nontarget, "nontarget_slots_with_unchanged_pixels": moved_same,
            "share_nontarget_slots_unchanged": moved_same / nontarget, "windows_with_two_or_more_target_slots": int((n_t >= 2).sum())}


def distinct_frames(m: dict, order: np.ndarray) -> dict:
    """Distinct (recording, frame, gamma) fed by one group: over the manifest and per window."""
    take = lambda key: np.take_along_axis(m[key], order.astype(int), -1)  # noqa: E731
    v, f, g = take("vid"), take("frame"), np.round(take("gamma"), 6)
    per_window = [len(set(zip(a.tolist(), b.tolist(), c.tolist()))) for a, b, c in zip(v.reshape(-1, SLOTS), f.reshape(-1, SLOTS), g.reshape(-1, SLOTS))]
    return {"manifest": len(set(zip(v.ravel().tolist(), f.ravel().tolist(), g.ravel().tolist()))), "per_window_mean": float(np.mean(per_window)),
            "per_window_min": int(min(per_window)), "per_window_max": int(max(per_window))}


def epoch_digests(m: dict, marker: np.ndarray, orders: dict, digests: dict, upe: int) -> dict[str, list[str]]:
    import hashlib
    out = {}
    for g in tm.GROUPS:
        out[g] = []
        for e in range(m["vid"].shape[0] // upe):
            h = hashlib.sha256(); h.update("|".join(map(str, m["videos"].tolist())).encode()); sl = slice(e * upe, (e + 1) * upe)
            for arr in (m["vid"][sl], m["frame"][sl], m["gamma"][sl], marker[sl], m["target"][sl], orders[g][sl], digests[g][sl]):
                h.update(np.ascontiguousarray(arr).tobytes())
            out[g].append(h.hexdigest())
    return out


# ----------------------------------------------------------------------------------------------- main
def load_sc(path: Path, train: set[str]) -> dict:
    with np.load(path) as z:
        m = {k: z[k] for k in (*COPIED, "videos", "labels")}
    bad = sorted(set(map(str, m["videos"].tolist())) - train)
    if bad:
        raise SystemExit(f"{path.name}: {len(bad)} recordings are not training clips")
    if m["vid"].shape[1:] != (WINDOWS, SLOTS):
        raise SystemExit(f"{path.name}: windows of shape {m['vid'].shape}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); rv = cfg["raw_video_context"]; paths = load_paths(); root = paths.repo_root
    t0 = time.time(); src = check_sources(rv, root); train = set(allowlist(rv, root))
    out = root / rv["outputs"]["root"] / "manifests"; out.mkdir(parents=True, exist_ok=True)
    tr = rv["training"]; upe = int(tr["total_updates"]) // 10
    ms = {int(k): load_sc(root / v["path"], train) for k, v in rv["source"]["sc_manifests"]["files"].items()}
    for seed, m in ms.items():
        if m["vid"].shape[0] != int(tr["total_updates"]):
            raise SystemExit(f"SC_s{seed}: {m['vid'].shape[0]} updates, expected {tr['total_updates']}")
    dirs = frame_dirs(paths, rv, {str(v) for m in ms.values() for v in m["videos"].tolist()})
    refs = {(str(m["videos"][vi]), int(f), round(float(g), 6)) for m in ms.values()
            for vi, f, g in zip(m["vid"].ravel(), m["frame"].ravel(), m["gamma"].ravel())}
    print(f"sources checked; {len(refs)} distinct frames to digest", flush=True)
    hashes = pixel_hashes(refs, dirs, a.workers); print(f"pixel digests done ({time.time() - t0:.0f} s)", flush=True)
    index = {"config": a.config, "sources": src, "allowlist_clips": len(train), "frames_digested": len(hashes), "seeds": {}}
    for seed, m in ms.items():
        marker, n_t, orders = window_orders(seed, m["s"], m["c"])
        lab_at_marker = np.take_along_axis(m["labels"], marker[..., None].astype(int), -1)[..., 0]
        if not np.array_equal(lab_at_marker, m["target"]):
            raise SystemExit(f"seed {seed}: a target differs from the label of its requested sample")
        digests = group_digests(m, orders, hashes); stats = shuffle_stats(m, orders, hashes, n_t)
        arrays = {k: m[k] for k in COPIED} | {"videos": m["videos"], "marker": marker, "n_target_slots": n_t}
        arrays |= {f"order_{tm.KEYS[g]}": orders[g] for g in tm.GROUPS} | {f"digest_{tm.KEYS[g]}": digests[g] for g in tm.GROUPS}
        np.savez(out / f"RV_s{seed}.npz", **arrays)
        init = tm.initial_new_state(seed); torch.save(init, out / f"init_s{seed}.pt")
        index["seeds"][str(seed)] = {
            "files": {"manifest": {"file": f"RV_s{seed}.npz", "sha256": sha256_file(out / f"RV_s{seed}.npz")},
                      "init": {"file": f"init_s{seed}.pt", "sha256": sha256_file(out / f"init_s{seed}.pt")}},
            "source_sc_sha256": src[f"sc_s{seed}"], "epoch_digests": epoch_digests(m, marker, orders, digests, upe),
            "windows": int(m["kind"].size), "native_windows": int((m["kind"] == 0).sum()), "mixed_windows": int((m["kind"] == 1).sum()),
            "marker_slot_counts": np.bincount(marker.ravel(), minlength=SLOTS).tolist(), "shuffle": stats,
            "distinct_frames_per_group": {g: distinct_frames(m, orders[g]) for g in tm.GROUPS},
            "windows_with_repeated_samples": int((m["repeats"] > 0).sum()),
            "classifier_init": {"weight_std": float(init["classifier"]["weight"].std()), "bias_abs_max": float(init["classifier"]["bias"].abs().max()),
                                "marker_abs_max": float(init["marker_stem"]["weight"].abs().max())}}
        print(f"seed {seed}: {json.dumps(stats)}", flush=True)
    index["seconds"] = time.time() - t0
    ids = train | {session_of(v) for v in train}; assert_no_ids(index, ids, "index")
    (out / "index.json").write_text(json.dumps(index, indent=1))
    res = root / rv["outputs"]["results_dir"]; res.mkdir(parents=True, exist_ok=True)
    (res / "preparation.json").write_text(json.dumps(index, indent=1)); print(f"wrote {out} and {res / 'preparation.json'}")


if __name__ == "__main__":
    main()
