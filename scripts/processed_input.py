#!/usr/bin/env python
"""A12 input layer (configs/experiments/video_processed_input_v1.yaml): map our allowlisted EV9V videos to the Pro group's
processed videos (read only), decide the fixed fallback, spot-check the time alignment, cache the needed 10 Hz Pro frames
under runs/video_processed_input_v1/cache, and write Pro window digests for A10's training manifests.

Never reads a test or Pediatric row, file or label: provider rows are filtered by our allowlisted video_path first.

  python scripts/processed_input.py --config configs/experiments/video_processed_input_v1.yaml --prepare
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.prepare_video_sequence_objective as vpr  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.report_video_within_view as rw  # noqa: E402
import scripts.report_segment_objective as rso  # noqa: E402
import scripts.train_video_discriminative as td  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.temporal.joint_visual import gamma_lut, load_frame_u8  # noqa: E402

NATIVE_CSV = "data/manifests/ev9v_native.csv"


# ----------------------------------------------------------------------------------------------- the cache
class ProCache:
    """Pro frames of the mapped videos (identity time mapping), with the same gamma LUT as the original path; videos
    without a usable Pro output fall back to V01's original loader."""

    def __init__(self, root: Path, raw_fp=None) -> None:
        self.root = Path(root); self.index = json.loads((self.root / "index.json").read_text()); self.raw_fp = raw_fp; self._arr = collections.OrderedDict()

    def is_pro(self, video: str) -> bool:
        return self.index["videos"].get(video, {}).get("fallback") is None and video in self.index["videos"]

    def _frames(self, video: str):
        """A bounded LRU of memory maps (64 videos), so the open-file limit is never reached."""
        if video in self._arr:
            self._arr.move_to_end(video)
        else:
            e = self.index["videos"][video]; self._arr[video] = (np.load(self.root / e["npy"], mmap_mode="r"), {int(f): i for i, f in enumerate(e["frames"])})
            while len(self._arr) > 64:
                self._arr.popitem(last=False)
        return self._arr[video]

    def frame(self, video: str, frame: int, gamma: float) -> np.ndarray:
        if not self.is_pro(video):
            paths = self.raw_fp(video); return load_frame_u8(paths[min(int(frame), len(paths) - 1)], gamma)
        arr, pos = self._frames(video); n = int(self.index["videos"][video]["n_frames"]); f = min(int(frame), n - 1)
        img = np.ascontiguousarray(arr[pos[f]])
        if abs(gamma - 1.0) > 1e-9:
            img = np.asarray(Image.fromarray(img).point(gamma_lut(gamma)), dtype=np.uint8)
        return img


def window_inputs(m: dict, order: np.ndarray, cache: ProCache, pairs: list[tuple[int, int]]):
    """V01's window_inputs with Pro pixels (fallback per video): frames, marker slots, targets and pixel digests."""
    import torch
    frames = np.empty((len(pairs), tm.N_SLOTS, 224, 224, 3), np.uint8); digests = np.empty((len(pairs), 32), np.uint8); memo = {}
    for k, (u, w) in enumerate(pairs):
        hashes = []
        for slot in range(tm.N_SLOTS):
            j = int(order[u, w, slot]); ref = (str(m["videos"][int(m["vid"][u, w, j])]), int(m["frame"][u, w, j]), round(float(m["gamma"][u, w, j]), 6))
            if ref not in memo:
                img = cache.frame(*ref); memo[ref] = (img, tm.frame_hash(img))
            frames[k, slot], h = memo[ref]; hashes.append(h)
        digests[k] = np.frombuffer(tm.window_digest(hashes), np.uint8)
    us, ws = [p[0] for p in pairs], [p[1] for p in pairs]
    return (torch.from_numpy(frames), torch.from_numpy(m["marker"][us, ws].astype(np.int64)), torch.from_numpy(m["target"][us, ws].astype(np.int64)), torch.from_numpy(digests))


def frames_for_refs(refs: list[tuple], cache: ProCache) -> np.ndarray:
    out = np.empty((len(refs), 224, 224, 3), np.uint8)
    for i, (v, f, g) in enumerate(refs):
        out[i] = cache.frame(str(v), int(f), float(g))
    return out


# ----------------------------------------------------------------------------------------------- decoding (workers)
def _decode(job: tuple) -> dict:
    video, path, frames, n_expected, out = job
    cap = cv2.VideoCapture(path); keep = set(frames); got, n = {}, 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if n in keep:
            got[n] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        n += 1
    cap.release()
    rec = {"video": video, "decoded_frames": n, "shape": None, "fallback": None}
    if n != n_expected:
        rec["fallback"] = "decoded_frame_count_mismatch"; return rec
    arr = np.stack([got[f] for f in frames]) if frames else np.zeros((0, 224, 224, 3), np.uint8); rec["shape"] = list(arr.shape)
    if arr.shape[1:] != (224, 224, 3):
        rec["fallback"] = "output_shape_mismatch"; return rec
    np.save(out, arr); rec["npy"] = Path(out).name
    return rec


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel(); a -= a.mean(); b -= b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def raw_in_pro_geometry(img: np.ndarray, row: dict) -> np.ndarray:
    """The original frame cropped with the provider's recorded crop box and padding, resized to 224 (INTER_AREA)."""
    t, l, b, r = (int(float(row[k])) for k in ("crop_top", "crop_left", "crop_bottom", "crop_right")); h, w = img.shape[:2]
    canvas = np.zeros((b - t, r - l, 3), np.uint8); y0, x0 = max(0, t), max(0, l); y1, x1 = min(h, b), min(w, r)
    canvas[y0 - t:y1 - t, x0 - l:x1 - l] = img[y0:y1, x0:x1]
    return cv2.resize(canvas, (224, 224), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------------------------- preparation
def prepare(cfg: dict, config: str, workers: int) -> None:
    vp = cfg["video_processed_input"]; out = ROOT / vp["outputs"]["root"]; ledger = ROOT / vp["budget"]["ledger"]; cdir = out / "cache"
    t0, started, ok, note, cpu_workers = time.time(), vsh.now(), False, "", 0.0
    try:
        workers = min(int(workers), int(vp["budget"]["decode_workers_max"]))
        a10cfg = "configs/experiments/video_switch_head_v1.yaml"; a10 = vsh.Ctx(load_cfg(SimpleNamespace(config=a10cfg, set=[])), a10cfg); a8 = rsh.a08_ctx()
        train = set(prv.allowlist(a10.rv, ROOT)); dev, dev_refs = set(), {}
        for h in ("calibration", "selection"):
            _, seqs = a8.half(h, rso.DevAccess({h})); refs = rw.half_refs(seqs); dev_refs[h] = refs; dev |= {r[0] for r in refs}
        allow = train | dev
        need = {}                                                                     # video -> needed source frame indices
        for s in (0, 1, 2):
            m = a10.manifest(s)[0]; U = a10.total
            for v, f in zip(m["videos"][m["vid"][:U]].ravel().tolist(), m["frame"][:U].ravel().tolist()):
                need.setdefault(str(v), set()).add(int(f))
        for refs in dev_refs.values():
            for v, f, _ in refs:
                need.setdefault(str(v), set()).add(int(f))
        if set(need) - allow:
            raise SystemExit("a needed video is outside the train/development allowlist")
        nat = pd.read_csv(ROOT / NATIVE_CSV, usecols=["video_id", "split", "n_frames", "video_path"], dtype=str); nat = nat[nat["video_id"].isin(allow)]
        if set(nat["video_id"]) != allow or set(nat["split"]) - {"train", "validation"}:
            raise SystemExit("the native manifest does not cover exactly the allowlist, or holds another split")
        ours = {r.video_id: {"split": r.split, "n_frames": int(r.n_frames), "path": r.video_path} for r in nat.itertuples()}; by_path = {v["path"]: k for k, v in ours.items()}
        prov = {}
        with open(Path(vp["pro"]["root"]) / "manifests" / "ev9v_oct16_model_ready.csv") as fh:
            for row in csv.DictReader(fh):
                vid = by_path.get(row["input_path"])                                  # every non-allowlisted row is skipped before any other field is used
                if vid is not None:
                    prov[vid] = {k: row[k] for k in ("frame_count", "fps", "analysis_status", "render_status", "audit_status", "output_path", "number_of_segments",
                                                     "crop_top", "crop_left", "crop_bottom", "crop_right", "fallback_source")}
        fails = set()
        with open(Path(vp["pro"]["root"]) / "manifests" / "ev9v_oct16_failures.csv") as fh:
            for row in csv.DictReader(fh):
                p = row.get("input_path") or row.get("relative_input_path") or ""; vid = by_path.get(p) or by_path.get(str(Path(vp["pro"]["root"]).parents[1] / "echo-view-routing" / p))
                if vid is not None:
                    fails.add(vid)
        status = {}
        for v, o in ours.items():
            r = prov.get(v)
            if r is None:
                status[v] = "no_provider_row"
            elif not all(r[k] == "complete" for k in ("analysis_status", "render_status", "audit_status")):
                status[v] = "provider_not_complete"
            elif not Path(r["output_path"]).is_file():
                status[v] = "output_missing"
            elif int(r["frame_count"]) != o["n_frames"]:
                status[v] = "frame_count_mismatch"
            else:
                status[v] = None
        jobs = [(v, prov[v]["output_path"], sorted(need[v]), ours[v]["n_frames"], str(cdir / f"{hashlib.sha256(v.encode()).hexdigest()[:20]}.npy"))
                for v in sorted(need) if status[v] is None]
        cdir.mkdir(parents=True, exist_ok=True); t1 = time.time()
        with Pool(workers) as pool:
            recs = pool.map(_decode, jobs, chunksize=4)
        cpu_workers = (time.time() - t1) * workers
        index = {"pro_root": vp["pro"]["root"], "mapping": "identity frame index (provider frame_count == our n_frames == decoded count)", "videos": {}}
        for (v, p, frames, n, _), r in zip(jobs, recs):
            if r["fallback"] is not None:
                status[v] = r["fallback"]
            else:
                index["videos"][v] = {"npy": r["npy"], "frames": frames, "n_frames": n, "fallback": None, "segments": prov[v]["number_of_segments"]}
        for v in need:
            if status[v] is not None:
                index["videos"][v] = {"fallback": status[v]}
        vsh.write_json(cdir / "index.json", index)
        # spot alignment check on training videos
        rng = np.random.default_rng([20261021, 0]); cand = sorted(v for v in index["videos"] if index["videos"][v]["fallback"] is None and ours[v]["split"] == "train")
        fp = FramePaths(prv.frame_dirs(a10.paths, a10.rv, set(cand))); spot = []
        for v in [cand[i] for i in rng.choice(len(cand), size=min(20, len(cand)), replace=False)]:
            n = ours[v]["n_frames"]; cap = cv2.VideoCapture(prov[v]["output_path"]); pro = []
            while True:
                okr, bgr = cap.read()
                if not okr:
                    break
                pro.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            cap.release(); paths = fp(v)
            for f in (2, n // 2, n - 3):
                raw = raw_in_pro_geometry(np.asarray(Image.open(paths[f]).convert("RGB")), prov[v])
                scores = {d: _ncc(raw, pro[f + d]) for d in range(-2, 3) if 0 <= f + d < len(pro)}; best = max(scores, key=scores.get)
                spot.append({"best_offset": int(best), "ncc0": scores[0], "zero_within_tie": bool(scores[0] >= scores[best] - 1e-3), "segments": prov[v]["number_of_segments"]})
        aligned = sum(s["best_offset"] == 0 or s["zero_within_tie"] for s in spot)          # a 1e-3 tie tolerance for near-static clips, fixed before any reading
        if aligned != len(spot):
            raise SystemExit(f"time alignment spot check failed: {aligned} of {len(spot)} frames peak at offset 0")
        # Pro window digests for A10's rows 0-1,349 (all seeds) and the Pro manifests
        cache = ProCache(cdir, FramePaths(prv.frame_dirs(a10.paths, a10.rv, set(need)))); man = {}
        for s in (0, 1, 2):
            m10 = a10.manifest(s)[0]; U = a10.total; m = {k: (v if k == "videos" else v[:U]) for k, v in m10.items()}
            dg = np.empty((U, 12, 32), np.uint8); fb = np.zeros((U, 12), bool)
            for u in range(U):
                _, _, _, d = window_inputs(m, m["order_ordered"], cache, [(u, w) for w in range(12)]); dg[u] = d.numpy()
                fb[u] = [not cache.is_pro(str(m["videos"][int(m["vid"][u, w, 0])])) for w in range(12)]
            m["digest_pro"] = dg; m["fallback_window"] = fb; path = out / "manifests" / f"VPI_s{s}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                raise SystemExit(f"{path.name} exists; never overwritten")
            np.savez(path, **m); man[str(s)] = {"file": path.name, "sha256": sha256_file(path), "arrays_digest": vpr.arrays_digest(m), "fallback_windows": int(fb.sum()),
                                               "a10_manifest_sha256": a10.manifest(s)[1]}
        vsh.write_json(out / "manifests" / "index.json", {"seeds": man})
        def split_counts(vs):
            c = {}
            for v in vs:
                key = f"{ours[v]['split']}|{status[v] or 'pro'}"; c[key] = c.get(key, 0) + 1
            return dict(sorted(c.items()))
        rec = {"config_sha256": sha256_file(ROOT / config), "started": started, "allowlist": {"train": len(train), "dev_videos": len(dev), "total": len(allow)},
               "status_all_allowlisted": split_counts(allow), "status_needed": split_counts(need), "provider_failures_in_allowlist": len(fails & allow),
               "needed_frames": int(sum(len(v) for v in need.values())), "cache": {"videos": sum(1 for v in index["videos"].values() if v.get("fallback") is None),
               "bytes": int(sum((cdir / e["npy"]).stat().st_size for e in index["videos"].values() if e.get("fallback") is None))},
               "multi_segment_videos_needed": sum(1 for v in need if status[v] is None and prov[v]["number_of_segments"] not in ("1", "")),
               "spot_check": {"frames": len(spot), "peak_at_zero": aligned, "ncc0_median": float(np.median([s["ncc0"] for s in spot]))},
               "manifests": man, "decode_worker_seconds": cpu_workers, "finished": vsh.now(), "seconds": time.time() - t0}
        ids = set(allow) | {Path(p).stem for p in by_path}; flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "preparation")
        res = ROOT / vp["outputs"]["results_dir"]; res.mkdir(parents=True, exist_ok=True)
        for p in (out / "preparation.json", res / "preparation.json"):
            vsh.write_json(p, flat)
        print(json.dumps(flat, indent=1), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        vsh.ledger_append(ledger, "prepare: mapping, decoding (worker seconds), spot check, digests", "cpu", started, time.time() - t0 + cpu_workers, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--prepare", action="store_true", required=True)
    a = ap.parse_args(); a.set = []; prepare(load_cfg(a), a.config, a.workers)


if __name__ == "__main__":
    main()
