#!/usr/bin/env python
"""A22 (configs/experiments/video_temporal_density_v1.yaml): A08 all_s{k} continued with A10's batches and A18's cached-own
step, with the student's 16 slots taken from the 8 old 10 Hz samples a = window_frames(c, N, 8, 8): Repeated (each sample
twice) against Interleaved (one real source frame between neighbours inside the left sample's own interval).

  --prepare          CPU: per-seed Repeated / Interleaved manifests rebuilt from the complete training sequences, Long10
                     rebuilt and compared with A10's, every source plan checked, pixel digests (failed intermediates fall
                     back to the left frame), input-difference, teacher-error and label-purity summaries
  --check            synthetic window cases; GPU: teacher parity, Long10 reproduction of A18 own / A10 R, forced
                     Interleaved = Repeated, optimiser and encoder states, resume, timing and budget projection
  --queue K [K ...]  Repeated_s0, Interleaved_s0, Repeated_s2, Interleaved_s2 (seed 1 only after a met rule)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.prepare_video_sequence_objective as vpr  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_raw_video_context as rvc  # noqa: E402
import scripts.train_video_switch_head as vsh  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.window_target import window_frames  # noqa: E402
from echo_routing.temporal.joint_visual import load_frame_u8  # noqa: E402

A10_CONFIG = "configs/experiments/video_switch_head_v1.yaml"
ARMS, WINDOWS, SLOTS, OLD = ("Repeated", "Interleaved"), 12, 16, 8
now, write_json, ledger_hours, ledger_append, determinism, max_diff = vsh.now, vsh.write_json, vsh.ledger_hours, vsh.ledger_append, vsh.determinism, vsh.max_diff


# ----------------------------------------------------------------------------------------------- windows
class Plans:
    """Per source: its 10 Hz plan (cache frame_idx), ordinal -> plan position, and the raw frame count (FramePaths)."""

    def __init__(self, fidx_of, n_raw_of) -> None:
        self.fidx_of, self.n_raw_of, self._p = fidx_of, n_raw_of, {}

    def plan(self, v: str):
        if v not in self._p:
            f = np.asarray(self.fidx_of(v), np.int64); n = int(self.n_raw_of(v)); step = int(f[1] - f[0]) if f.size > 1 else None
            regular = bool(f.size >= 1 and f[0] == 0 and (np.array_equal(f, np.arange(0, n, step)) if step else n >= 1))
            self._p[v] = (f, {int(x): i for i, x in enumerate(f)}, n, regular, step)
        return self._p[v]

    def next_after(self, v: str, f: int):
        fidx, pos, n, _, _ = self.plan(v); i = pos.get(int(f))
        if i is None:
            return None
        return int(fidx[i + 1]) if i + 1 < fidx.size else n


def coords(a: np.ndarray) -> np.ndarray:
    """Planned slot coordinates in half-sample units: 2 a_i at even slots, a_i + a_{i+1} at odd slots, 2 a_7 at slot 15."""
    a = np.asarray(a, np.int64); out = np.empty(SLOTS, np.int64); out[0::2] = 2 * a; out[1:15:2] = a[:-1] + a[1:]; out[15] = 2 * a[-1]
    return out


def build(refs: list, c: int, plans: Plans) -> dict:
    """Both arms' 16 slot references, the planned coordinates, the shared marker and per-slot flags for one request c."""
    n = len(refs); a = window_frames(int(c), n, OLD, OLD); co = coords(a); marker = tm.marker_slot(co, 2 * int(c))
    rep = [tuple(refs[int(x)]) for x in np.repeat(a, 2)]; inter = list(rep); new = np.zeros(SLOTS, bool); why = Counter()
    for i in range(OLD - 1):
        if a[i] == a[i + 1]:
            why["repeat position"] += 1; continue
        v, f, g = refs[int(a[i])]; nxt = plans.next_after(v, int(f))
        if nxt is None:
            why["left frame outside its plan"] += 1; continue
        mid = min((int(f) + int(nxt)) // 2, plans.plan(v)[2] - 1)
        if mid <= int(f):
            why["no raw frame inside the interval"] += 1; continue
        inter[2 * i + 1] = (v, int(mid), g); new[2 * i + 1] = True
    return {"a": a, "coords": co, "marker": int(marker), "Repeated": rep, "Interleaved": inter, "new": new, "why": why}


def reference_slots(refs: list, c: int, nxt_of, n_raw_of) -> list:
    """An independent, loop-only restatement of the Interleaved rule, used by the checks."""
    n = len(refs); start = max(0, min(c - 4, n - 8)); samp = list(range(start, min(n, start + 8))); a = [samp[int(round(k * (len(samp) - 1) / 7))] for k in range(8)]
    out = []
    for i in range(8):
        out.append(tuple(refs[a[i]]))
        if i == 7:
            out.append(tuple(refs[a[7]])); break
        if a[i] == a[i + 1]:
            out.append(tuple(refs[a[i]])); continue
        v, f, g = refs[a[i]]; e = nxt_of(v, f); m = (f + e) // 2
        m = min(m, n_raw_of(v) - 1)
        out.append((v, m, g) if m > f else tuple(refs[a[i]]))
    return out


def synthetic_cases() -> dict:
    """Window cases on synthetic sources: boundaries, fragments, gamma, ends, N < 8, repeats, raw end, odd steps."""
    srcs = {"A": (np.arange(0, 120, 3), 120), "B": (np.arange(0, 60, 3), 60), "C": (np.arange(0, 50, 5), 50), "D": (np.arange(0, 20, 2), 20), "E": (np.arange(0, 31, 3), 32),
            "F": (np.arange(0, 28, 3), 28)}
    plans = Plans(lambda v: srcs[v][0], lambda v: srcs[v][1]); nxt_of = lambda v, f: plans.next_after(v, f); nraw = lambda v: srcs[v][1]  # noqa: E731
    seqs = {"one source": [("A", int(x), 1.0) for x in srcs["A"][0][:30]],
            "source boundary": [("A", int(x), 1.0) for x in srcs["A"][0][10:15]] + [("B", int(x), 1.0) for x in srcs["B"][0][3:12]],
            "same source, two fragments": [("A", int(x), 1.0) for x in srcs["A"][0][10:15]] + [("A", int(x), 1.0) for x in srcs["A"][0][30:38]],
            "gamma change": [("A", int(x), 1.0) for x in srcs["A"][0][0:6]] + [("B", int(x), 0.8) for x in srcs["B"][0][0:8]],
            "short (N = 5)": [("A", int(x), 1.0) for x in srcs["A"][0][3:8]],
            "raw end (n = 32, last plan 30)": [("E", int(x), 1.0) for x in srcs["E"][0][8:]] + [("B", int(x), 1.0) for x in srcs["B"][0][0:6]],
            "raw end (n = 28, last plan 27)": [("F", int(x), 1.0) for x in srcs["F"][0][7:]] + [("B", int(x), 1.0) for x in srcs["B"][0][0:6]],
            "step 5": [("C", int(x), 1.0) for x in srcs["C"][0]], "step 2": [("D", int(x), 1.0) for x in srcs["D"][0]]}
    out, ok = {}, True
    for name, refs in seqs.items():
        res = []
        for c in sorted({0, len(refs) // 2, len(refs) - 1}):
            w = build(refs, c, plans); ref = reference_slots(refs, c, nxt_of, nraw); src = {(r[0], r[1]): r for r in refs}
            even_eq = all(w["Repeated"][k] == w["Interleaved"][k] for k in list(range(0, SLOTS, 2)) + [15])
            inside = all((not w["new"][k]) or (w["Interleaved"][k][0] == w["Interleaved"][k - 1][0] and w["Interleaved"][k - 1][1] < w["Interleaved"][k][1] < nxt_of(*w["Interleaved"][k - 1][:2])
                                               and w["Interleaved"][k][2] == w["Interleaved"][k - 1][2]) for k in range(1, 15, 2))
            mk_ok = w["coords"][w["marker"]] == 2 * c and w["marker"] == int(np.flatnonzero(w["coords"] == 2 * c)[len(np.flatnonzero(w["coords"] == 2 * c)) // 2])
            r = {"c": c, "a": w["a"].tolist(), "marker": w["marker"], "new": int(w["new"].sum()), "matches_reference": w["Interleaved"] == ref, "even_and_last_equal": even_eq,
                 "intermediates_inside_left_interval": inside, "marker_ok": bool(mk_ok), "old_refs_kept": all(src.get((x[0], x[1])) == x for x in w["Repeated"])}
            ok = ok and all(r[k] for k in ("matches_reference", "even_and_last_equal", "intermediates_inside_left_interval", "marker_ok", "old_refs_kept")); res.append(r)
        out[name] = res
    b = build(seqs["source boundary"], 6, plans); k = int(np.flatnonzero(b["a"] == 4)[0]) * 2 + 1
    out["boundary intermediate from the left source"] = bool(b["Interleaved"][k][0] == "A" and b["Interleaved"][k][1] == 43)
    f2 = build(seqs["same source, two fragments"], 4, plans); k = 2 * int(np.flatnonzero(f2["a"] == 4)[0]) + 1
    out["fragment intermediate from the left fragment's interval"] = bool(f2["Interleaved"][k] == ("A", 43, 1.0))
    g = build(seqs["gamma change"], 6, plans); k = 2 * int(np.flatnonzero(g["a"] == 5)[0]) + 1
    out["gamma from the left fragment"] = bool(g["Interleaved"][k] == ("A", 16, 1.0))
    e = build(seqs["raw end (n = 32, last plan 30)"], 4, plans); out["raw end uses the raw frame count"] = bool(e["Interleaved"][5] == ("E", 31, 1.0) and all(x[1] <= 31 for x in e["Interleaved"] if x[0] == "E"))
    e2 = build(seqs["raw end (n = 28, last plan 27)"], 4, plans); out["raw end without a raw frame repeats the left"] = bool(e2["Interleaved"][5] == ("F", 27, 1.0) and not e2["new"][5] and e2["why"]["no raw frame inside the interval"] >= 1)
    out["fallback replaces with the left frame"] = fallback_slots([("A", 0, 1.0), ("A", 1, 1.0)] + [("A", 3, 1.0)] * 14, np.array([False, True] + [False] * 14), {("A", 1, 1.0)})[0][1] == ("A", 0, 1.0)
    out["ok"] = bool(ok and all(v for k2, v in out.items() if isinstance(v, bool)))
    return out


def fallback_slots(slots: list, new: np.ndarray, failed: set) -> tuple[list, int]:
    """Fixed rule: a new intermediate whose JPEG failed to decode is replaced by the left old frame (slot - 1)."""
    out, n = list(slots), 0
    for k in np.flatnonzero(new):
        if out[k] in failed:
            out[k] = out[k - 1]; n += 1
    return out, n


# ----------------------------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vt = cfg, config, cfg["video_temporal_density"]; p = self.vt["pinned"]
        for k in ("A10_config", "A10_manifest_index", "A08_choices", "A15_pool_manifest", "A19_screen", "A18_screen", "A20_screen", "A19_logits_index", "crop_code"):
            if sha256_file(ROOT / p[k]["path"]) != p[k]["sha256"]:
                raise SystemExit(f"{k}: differs from its pin")
        self.a10 = vsh.Ctx(load_cfg(SimpleNamespace(config=A10_CONFIG, set=[])), A10_CONFIG); self.total, self.warmup, self.o = self.a10.total, self.a10.warmup, self.a10.o
        o = self.vt["outputs"]; self.out, self.ckpt_root, self.results, self.ledger = ROOT / o["root"], ROOT / o["checkpoints"], ROOT / o["results_dir"], ROOT / self.vt["budget"]["ledger"]
        self._m = {}

    def manifest(self, arm: str, seed: int) -> tuple[dict, str]:
        if arm == "Long10":
            return self.a10.manifest(seed)
        key = f"{arm}_s{seed}"
        if key not in self._m:
            idx = json.loads((self.out / "manifests" / "index.json").read_text())[key]; p = self.out / "manifests" / idx["file"]
            if (got := sha256_file(p)) != idx["sha256"]:
                raise SystemExit(f"{key}: the manifest differs from its index")
            with np.load(p) as z:
                self._m[key] = ({k: z[k] for k in z.files}, got)
        return self._m[key]

    def start(self, seed: int) -> tuple[dict, str]:
        state, ssha = self.a10.start_state(seed)
        if ssha != self.vt["pinned"]["A08_weights"][seed]:
            raise SystemExit(f"seed {seed}: A08's start weights differ from the pin")
        return state, ssha

    def teacher(self, seed: int) -> dict:
        path = ROOT / "runs/video_ensemble_teacher_v1/cache" / f"teachers_s{seed}.npz"; csha = sha256_file(path); m, msha = self.a10.manifest(seed)
        if csha != self.vt["pinned"]["A18_teacher_cache"][seed]:
            raise SystemExit(f"seed {seed}: A18's teacher cache differs from its pin")
        with np.load(path) as z:
            if str(z["manifest_sha256"]) != msha or not np.array_equal(z["digests"][:self.total], m["digest_ordered"][:self.total]):
                raise SystemExit(f"seed {seed}: A18's cached requests differ from A10's windows")
            return {"logits": z["logits"][:, :self.total].astype(np.float32), "digests": z["digests"][:self.total], "sha256": csha}


# ----------------------------------------------------------------------------------------------- preparation (CPU)
def _decode_video(job: tuple) -> tuple[str, dict, list, list]:
    v, paths, wanted, pairs = job; out, failed, imgs = {}, [], {}
    for f, g in wanted:
        try:
            img = load_frame_u8(Path(paths[min(int(f), len(paths) - 1)]), round(float(g), 6)); out[(int(f), float(g))] = tm.frame_hash(img); imgs[(int(f), float(g))] = img
        except Exception as e:  # noqa: BLE001 - recorded and handled by the fixed fallback rule
            failed.append((int(f), float(g), type(e).__name__))
    diffs = [(mid, left, g, float(np.abs(imgs[(mid, g)].astype(np.int16) - imgs[(left, g)].astype(np.int16)).mean())) for mid, left, g in pairs if (mid, g) in imgs and (left, g) in imgs]
    return v, out, failed, diffs


def prepare(ctx: Ctx, seeds: list[int], workers: int) -> None:
    t0, started, ok, note, rec = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config), "started": now(), "seeds": {}}
    try:
        data = vpr.training_data(ctx.a10.a07p); seqs = data.seqs; fp = ctx.a10.frame_paths(); plans = Plans(data.frame_idx, lambda v: len(fp(v))); built = {}
        for s in seeds:
            m10, msha = ctx.a10.manifest(s); vix = {str(v): i for i, v in enumerate(m10["videos"].tolist())}; U = ctx.total; why = Counter(); bad = Counter()
            arr = {arm: {"vid": np.empty((U, WINDOWS, SLOTS), np.int32), "frame": np.empty((U, WINDOWS, SLOTS), np.int32), "gamma": np.empty((U, WINDOWS, SLOTS), np.float32)} for arm in ARMS}
            common = {"marker": np.empty((U, WINDOWS), np.int8), "a": np.empty((U, WINDOWS, OLD), np.int32), "coords": np.empty((U, WINDOWS, SLOTS), np.int32), "new": np.zeros((U, WINDOWS, SLOTS), bool),
                      "purity": np.empty((U, WINDOWS), np.float32), "purity_long": np.empty((U, WINDOWS), np.float32)}
            for u in range(U):
                for w in range(WINDOWS):
                    qi, c = int(m10["seq"][u, w]), int(m10["c"][u, w]); q = seqs[qi]; refs = [tuple(r) for r in q["refs"]]; lab = np.asarray(q["labels"]); n = len(refs)
                    s16 = window_frames(c, n, 16, 16)
                    if not np.array_equal(s16, m10["s"][u, w]) or tm.marker_slot(s16, c) != int(m10["marker"][u, w]) or int(lab[c]) != int(m10["target"][u, w]):
                        bad["Long10 window, marker or target differs from A10"] += 1
                    old = [(str(m10["videos"][int(m10["vid"][u, w, j])]), int(m10["frame"][u, w, j]), round(float(m10["gamma"][u, w, j]), 6)) for j in range(SLOTS)]
                    if [(r[0], int(r[1]), round(float(r[2]), 6)) for r in (refs[int(x)] for x in s16)] != old:
                        bad["Long10 references differ from A10"] += 1
                    b = build(refs, c, plans); why.update(b["why"])
                    for arm in ARMS:
                        arr[arm]["vid"][u, w] = [vix[str(r[0])] for r in b[arm]]; arr[arm]["frame"][u, w] = [int(r[1]) for r in b[arm]]; arr[arm]["gamma"][u, w] = [float(r[2]) for r in b[arm]]
                    common["marker"][u, w], common["a"][u, w], common["coords"][u, w], common["new"][u, w] = b["marker"], b["a"], b["coords"], b["new"]
                    common["purity"][u, w] = float((lab[np.repeat(b["a"], 2)] == lab[c]).mean()); common["purity_long"][u, w] = float((lab[s16] == lab[c]).mean())
            if bad:
                raise SystemExit(f"seed {s}: the rebuilt windows differ from A10's ({dict(bad)})")
            built[s] = (m10, msha, arr, common, why)
        refs = defaultdict(set); pairs = defaultdict(set)
        for s, (m10, _, arr, common, _) in built.items():
            V = m10["videos"]
            for arm in ARMS:
                for v, f, g in zip(arr[arm]["vid"].ravel(), arr[arm]["frame"].ravel(), arr[arm]["gamma"].ravel()):
                    refs[str(V[v])].add((int(f), round(float(g), 6)))
            I = arr["Interleaved"]; uu, ww, kk = np.nonzero(common["new"])
            for u, w, k in zip(uu, ww, kk):
                pairs[str(V[I["vid"][u, w, k]])].add((int(I["frame"][u, w, k]), int(I["frame"][u, w, k - 1]), round(float(I["gamma"][u, w, k]), 6)))
        sizes = Counter(); irregular = []
        for v in refs:
            f, _, n, regular, step = plans.plan(v); sizes[step] += 1
            if not regular:
                irregular.append(v)
        t1 = time.time(); hashes, failed, diffs = {}, set(), []
        jobs = [(v, list(map(str, fp(v))), sorted(refs[v]), sorted(pairs.get(v, ()))) for v in sorted(refs)]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for v, h, fl, d in ex.map(_decode_video, jobs, chunksize=8):
                hashes.update({(v, f, g): x for (f, g), x in h.items()}); failed |= {(v, f, round(g, 6)) for f, g, _ in fl}; diffs += [x[3] for x in d]
        dsec = time.time() - t1; rec["decode"] = {"recordings": len(refs), "distinct_frames": len(hashes), "failed": len(failed), "seconds": dsec, "workers": workers,
                                                 "plan_steps": {str(k): n for k, n in sizes.items()}, "irregular_plans": len(irregular)}
        old_failed = set()
        mdir = ctx.out / "manifests"; mdir.mkdir(parents=True, exist_ok=True); index = {}
        for s, (m10, msha, arr, common, why) in built.items():
            V = m10["videos"]; I = arr["Interleaved"]; nfb = 0
            for u, w in zip(*np.nonzero(common["new"].any(-1))):
                slots = [(str(V[I["vid"][u, w, k]]), int(I["frame"][u, w, k]), round(float(I["gamma"][u, w, k]), 6)) for k in range(SLOTS)]
                fixed, n = fallback_slots(slots, common["new"][u, w], failed)
                if n:
                    nfb += n; I["frame"][u, w] = [x[1] for x in fixed]; common["new"][u, w] = common["new"][u, w] & np.asarray([x not in failed for x in slots])
            for arm in ARMS:
                A = arr[arm]; dig = np.empty((ctx.total, WINDOWS, 32), np.uint8)
                for u in range(ctx.total):
                    for w in range(WINDOWS):
                        keys = [(str(V[A["vid"][u, w, k]]), int(A["frame"][u, w, k]), round(float(A["gamma"][u, w, k]), 6)) for k in range(SLOTS)]
                        if any(x not in hashes for x in keys):
                            old_failed |= {x for x in keys if x not in hashes}; dig[u, w] = 0; continue
                        dig[u, w] = np.frombuffer(tm.window_digest([hashes[x] for x in keys]), np.uint8)
                m = {**A, "marker": common["marker"], "target": m10["target"][:ctx.total].astype(np.int8), "order_ordered": np.broadcast_to(np.arange(SLOTS, dtype=np.int8), (ctx.total, WINDOWS, SLOTS)).copy(),
                     "videos": V, "digest_ordered": dig, "seq": m10["seq"][:ctx.total], "c": m10["c"][:ctx.total], "a": common["a"], "coords": common["coords"], "new": common["new"] if arm == "Interleaved" else np.zeros_like(common["new"])}
                p = mdir / f"{arm}_s{s}.npz"
                if p.is_file():
                    raise SystemExit(f"{p.name} exists; manifests are never overwritten")
                np.savez(p, **m); index[f"{arm}_s{s}"] = {"file": p.name, "sha256": sha256_file(p), "a10_manifest_sha256": msha}
            tl = ctx.teacher(s)["logits"][s]; terr = tl.argmax(-1) != m10["target"][:ctx.total]; newc = common["new"].sum(-1); same = np.all(arr["Repeated"]["frame"] == I["frame"], axis=-1)
            pur = common["purity"].ravel(); purl = common["purity_long"].ravel(); bins = lambda x: {"1.0": int((x == 1).sum()), "0.75-1": int(((x >= 0.75) & (x < 1)).sum()), "0.5-0.75": int(((x >= 0.5) & (x < 0.75)).sum()), "<0.5": int((x < 0.5).sum())}  # noqa: E731
            rec["seeds"][str(s)] = {"a10_manifest_sha256": msha, "windows": int(ctx.total * WINDOWS), "windows_with_new_frames": int((newc > 0).sum()), "windows_identical_across_arms": int(same.sum()),
                                    "new_frames_per_window": {"mean": float(newc.mean()), "distribution": {str(k): int((newc == k).sum()) for k in range(8)}}, "no_new_frame_reasons": dict(why), "fallbacks": nfb,
                                    "teacher_error_rate": float(terr.mean()), "teacher_error_rate_by_group": {g: float(terr[:, 4 * i:4 * i + 4].mean()) for i, g in enumerate(("native", "change", "same-label join"))},
                                    "target_label_share_short": {"mean": float(pur.mean()), "bins": bins(pur)}, "target_label_share_long": {"mean": float(purl.mean()), "bins": bins(purl)},
                                    "short_positions_repeated": int(sum(len(set(x.tolist())) < OLD for x in common["a"].reshape(-1, OLD)))}
        if old_failed:
            raise SystemExit(f"{len(old_failed)} old frames failed to decode; the old handling is not changed by this round")
        rec["new_vs_left_pixel_difference"] = {"pairs": len(diffs), "mean_abs_0_255": float(np.mean(diffs)) if diffs else None, "median": float(np.median(diffs)) if diffs else None,
                                               "zero_share": float(np.mean(np.asarray(diffs) == 0)) if diffs else None}
        rec["met"] = bool(all(v["windows_with_new_frames"] > 0 for v in rec["seeds"].values())); rec["finished"] = now()
        write_json(mdir / "index.json", index)
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "seeds"}, indent=1), json.dumps(rec["seeds"], indent=1), flush=True)
        if not rec["met"]:
            raise SystemExit("no intermediate frame is available: preparation not met")
        ok = True; note = f"decode {dsec:.0f} s"
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, f"prepare manifests (CPU: main process + {workers} decode workers during the decode)", "cpu", started, (time.time() - t0) + rec.get("decode", {}).get("seconds", 0.0) * workers, ok, note)


# ----------------------------------------------------------------------------------------------- training
def code_digest() -> str:
    h = hashlib.sha256()
    for p in (Path(__file__), Path(tet.__file__), Path(vsh.__file__), Path(rvc.__file__), Path(tm.__file__)):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def identity(ctx: Ctx, arm: str, seed: int, msha: str, ssha: str, csha: str) -> str:
    payload = {"experiment": ctx.cfg.get("experiment"), "sections": {k: ctx.vt[k] for k in ("predecessors", "arms", "windows", "data", "loss", "training")}, "arm": arm, "seed": int(seed),
               "manifest_sha256": msha, "start_sha256": ssha, "teacher_cache_sha256": csha, "environment": prv.environment(), "total_updates": int(ctx.total), "code": code_digest()}
    return hashlib.sha256(json.dumps(json.loads(json.dumps(payload, default=str)), sort_keys=True).encode()).hexdigest()


def student(ctx: Ctx, seed: int, device):
    state, ssha = ctx.start(seed); return tet.make_model(state, device), vsh.build_heads(seed).to(device), ssha


def check_batch(m: dict, b, cache: dict | None = None) -> None:
    u = int(b[0])
    if not np.array_equal(b[4].numpy(), m["digest_ordered"][u]):
        raise SystemExit(f"update {u}: the decoded pixels differ from the manifest digests")
    if not np.array_equal(b[3].numpy(), m["target"][u]) or not np.array_equal(b[2].numpy(), m["marker"][u]):
        raise SystemExit(f"update {u}: targets or markers differ from the manifest")


def train(ctx: Ctx, arm: str, seed: int, device, workers: int) -> dict:
    m, msha = ctx.manifest(arm, seed); cache = ctx.teacher(seed); model, heads, ssha = student(ctx, seed, device); ident = identity(ctx, arm, seed, msha, ssha, cache["sha256"])
    key = f"{arm}_s{seed}"; rd, ck = ctx.out / "train" / key, ctx.ckpt_root / key; rd.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)
    if (rd / "run_manifest.json").is_file() and json.loads((rd / "run_manifest.json").read_text()).get("identity") == ident and (ck / "final.pt").is_file():
        return {"key": key, "status": "finished"}
    opt = vsh.make_optimizer(model, heads, ctx.o)
    if not vsh.bound(opt, model, heads):
        raise SystemExit("the optimiser is not bound")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); done, log = 0, []; ul = rd / "updates.jsonl"
    if (ck / "resume.pt").is_file():
        done, log = vsh.load_state(ck / "resume.pt", model, heads, opt, ident); lines = ul.read_text().splitlines()[:done] if ul.is_file() else []; ul.write_text("".join(x + "\n" for x in lines))
        with (rd / "resumes.jsonl").open("a") as fh:
            fh.write(json.dumps({"at_update": done, "time": now()}) + "\n")
    else:
        for p in (ul, rd / "train_log.jsonl", rd / "resumes.jsonl"):
            p.unlink(missing_ok=True)
    print(f"[{key}] identity {ident[:12]}; updates {done} -> {ctx.total}", flush=True); t0 = time.time(); acc = []; torch.cuda.reset_peak_memory_stats(device)
    for b in vsh.loader(m, ctx.a10.frame_paths(), list(range(done, ctx.total)), workers):
        check_batch(m, b); u = int(b[0]); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        r = tet.train_step(model, heads, opt, b, torch.from_numpy(cache["logits"][seed, u].copy()), "own", lrs, ctx.o, device, keep=True); z = r.pop("logits"); t = b[3]
        ok_w = (z.argmax(1) == t).numpy(); r.update({f"acc_{g}": float(ok_w[4 * i:4 * i + 4].mean()) for i, g in enumerate(("native", "change", "join"))}); acc.append(r); done = u + 1
        with ul.open("a") as fh:
            fh.write(json.dumps({"u": u, **{k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}}) + "\n")
        if done % 25 == 0:
            write_json(rd / "progress.json", {"stage": f"{arm} student training (backward, GPU)", "identity": ident, "updates": done, "total": ctx.total, "seconds": time.time() - t0, "loss": r["loss"]})
        if done % ctx.warmup == 0:
            mean = lambda k: float(np.mean([a[k] for a in acc]))  # noqa: E731
            rec = {"updates": done, **{k: mean(k) for k in ("loss", "ce", "kl", "grad_norm", "acc_native", "acc_change", "acc_join")}, "accuracy": sum(a["correct"] for a in acc) / (WINDOWS * len(acc)),
                   "lr_encoder_last": acc[-1]["lr_encoder"], "seconds_this_process": time.time() - t0, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
            log.append(rec); vsh.save_state(ck / "resume.pt", model, heads, opt, done, ident, log)
            with (rd / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{key}] {done}/{ctx.total} loss {rec['loss']:.4f} CE {rec['ce']:.4f} KL {rec['kl']:.4f} acc {rec['accuracy']:.3f} {rec['seconds_this_process']:.0f}s", flush=True); acc = []
    if done != ctx.total:
        raise SystemExit(f"stopped at update {done}")
    final = ck / "final.pt"; torch.save({"model": model.state_dict(), "heads": heads.state_dict()}, final)
    man = {"identity": ident, "arm": arm, "key": key, "seed": seed, "config_sha256": sha256_file(ROOT / ctx.config), "manifest_sha256": msha, "cache_sha256": cache["sha256"], "start_sha256": ssha,
           "environment": prv.environment(), "code": code_digest(), "updates": done, "epochs": log, "resumes": (rd / "resumes.jsonl").read_text().splitlines() if (rd / "resumes.jsonl").is_file() else [],
           "final_checkpoint": str(final), "final_sha256": sha256_file(final), "parameters": int(sum(p.numel() for p in model.parameters())), "encoder_forwards_per_window": 1,
           "seconds_last_process": time.time() - t0}
    write_json(rd / "run_manifest.json", man); (ck / "resume.pt").unlink(missing_ok=True); print(f"[{key}] done: final {man['final_sha256'][:12]}", flush=True)
    return {"key": key, "status": "finished", "final_sha256": man["final_sha256"]}


def queue(ctx: Ctx, keys: list[str], device, workers: int) -> None:
    chk = json.loads((ctx.out / "checks.json").read_text())
    if not chk["ok"]:
        raise SystemExit("the checks did not pass")
    spu = float(chk["timing"]["seconds_per_update"]); ev = float(chk["budget"]["evaluation_per_model_hours"])
    for i, key in enumerate(keys):
        spent = ledger_hours(ctx.ledger, "gpu"); proj = spent + len(keys[i:]) * (ctx.total * spu / 3600 + ev)
        print(f"[queue] before {key}: spent {spent:.3f}, projection with evaluation {proj:.3f} of {ctx.vt['budget']['gpu_hours_max']}", flush=True)
        if proj > float(ctx.vt["budget"]["gpu_hours_max"]):
            raise SystemExit("the projection exceeds the cap; nothing new starts")
        arm, s = key.rsplit("_s", 1); t0, started, ok, note = time.time(), now(), False, ""
        try:
            r = train(ctx, arm, int(s), device, workers); ok = True; note = json.dumps(r)
        except BaseException as e:  # noqa: BLE001 - recorded, then raised again
            note = f"{type(e).__name__}: {str(e)[:200]}"; raise
        finally:
            ledger_append(ctx.ledger, f"train {key}", "gpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- checks
def batch_of(ctx: Ctx, arm: str, seed: int, u: int):
    m, _ = ctx.manifest(arm, seed); b = vsh.Inputs(m, ctx.a10.frame_paths(), [u])[0]; check_batch(m, b); return b


def run(ctx: Ctx, arm: str, seed: int, us: list[int], device, state_file: Path | None = None, inputs_of: str | None = None) -> dict:
    """A fresh student over the given updates through train()'s exact step (A18's cached-own step)."""
    cache = ctx.teacher(seed); model, heads, _ = student(ctx, seed, device); opt = vsh.make_optimizer(model, heads, ctx.o)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); losses, secs, grads = [], [], []
    if state_file is not None and state_file.is_file():
        vsh.load_state(state_file, model, heads, opt, "check")
    for u in us:
        b = batch_of(ctx, inputs_of or arm, seed, u); lrs = vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o); torch.cuda.synchronize(); a = time.time()
        r = tet.train_step(model, heads, opt, b, torch.from_numpy(cache["logits"][seed, u].copy()), "own", lrs, ctx.o, device); torch.cuda.synchronize(); secs.append(time.time() - a)
        losses.append(r["loss"]); grads.append(r["grad_norm"])
    if state_file is not None and not state_file.is_file():
        vsh.save_state(state_file, model, heads, opt, len(us), "check", [])
    return {"model": model, "heads": heads, "losses": losses, "grad_norms": grads, "seconds": secs}


def logged(path: Path, us: list[int]) -> list[float]:
    rows = {r["u"]: r for r in (json.loads(x) for x in path.read_text().splitlines())}; return [float(rows[u]["loss"]) for u in us]


def check(ctx: Ctx, device, workers: int) -> None:
    t0, started, ok, note, res = time.time(), now(), False, "", {"config_sha256": sha256_file(ROOT / ctx.config), "started": now()}
    try:
        res["synthetic_windows"] = synthetic_cases()
        for s in (0, 1, 2):
            ctx.teacher(s); ctx.start(s)
        state, _ = ctx.start(0); cache = ctx.teacher(0); a08 = tet.make_model(state, device).eval(); d0 = tet.state_digest(a08); par = []
        with torch.no_grad():
            for u in range(6):
                b = batch_of(ctx, "Long10", 0, u); par.append(float((a08(tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device))).float().cpu() - torch.from_numpy(cache["logits"][0, u])).abs().max()))
        res["teacher"] = {"cached_against_online_max_abs": max(par), "windows": 72, "unchanged": tet.state_digest(a08) == d0, "ok": max(par) <= 1e-4 and tet.state_digest(a08) == d0}; del a08
        arms_eq = {}
        for u in (0, 1):
            bR, bI, bL = (batch_of(ctx, a, 0, u) for a in ("Repeated", "Interleaved", "Long10"))
            even = list(range(0, SLOTS, 2)) + [15]; odd = list(range(1, 15, 2))
            arms_eq[u] = {"even_and_last_slots_equal": bool(torch.equal(bR[1][:, even], bI[1][:, even])), "odd_slots_differ_somewhere": bool(not torch.equal(bR[1][:, odd], bI[1][:, odd])),
                          "markers_equal": bool(torch.equal(bR[2], bI[2])), "targets_equal_Long10": bool(torch.equal(bR[3], bL[3]) and torch.equal(bI[3], bL[3]))}
        res["arm_inputs"] = {"updates": arms_eq, "ok": all(all(v.values()) for v in arms_eq.values())}
        rep = {}
        for s, ref_log in ((1, ROOT / "runs/video_ensemble_teacher_v1/train/R_s1/updates.jsonl"), (0, ROOT / "runs/video_switch_head_v1/train/R_s0/updates.jsonl")):
            us = list(range(0, ctx.warmup + 2)) if s == 1 else [0]; r1 = run(ctx, "Long10", s, us, device); lg = logged(ref_log, us)
            rep[f"s{s}"] = {"reference": "A18 R_s1 (cached own teacher)" if s == 1 else "A10 R_s0 (online teacher)", "updates": [us[0], us[-1]], "loss_max_abs": float(max(abs(a - b) for a, b in zip(r1["losses"], lg))),
                            "after_unfreezing_loss_abs": [float(abs(a - b)) for a, b in zip(r1["losses"][-2:], lg[-2:])] if s == 1 else None}
            del r1; torch.cuda.empty_cache()
        for u in (0, ctx.warmup):
            st = {}
            for path_name in ("mine", "A18"):
                model, heads, _ = student(ctx, 0, device); opt = vsh.make_optimizer(model, heads, ctx.o); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
                b = batch_of(ctx, "Long10", 0, u) if path_name == "mine" else vsh.Inputs(ctx.a10.manifest(0)[0], ctx.a10.frame_paths(), [u])[0]
                r = tet.train_step(model, heads, opt, b, torch.from_numpy(cache["logits"][0, u].copy()), "own", vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o), ctx.o, device)
                st[path_name] = (r["loss"], r["grad_norm"], model)
            rep[f"single update u{u} (determinism: the same step twice)"] = {"loss_abs": abs(st["mine"][0] - st["A18"][0]), "grad_norm_abs": abs(st["mine"][1] - st["A18"][1]), "params_max_abs": max_diff(st["mine"][2], st["A18"][2])}
            del st; torch.cuda.empty_cache()
        rep["ok"] = bool(rep["s1"]["loss_max_abs"] <= 1e-4 and rep["s0"]["loss_max_abs"] <= 1e-4 and all(rep[f"single update u{u} (determinism: the same step twice)"]["loss_abs"] <= 1e-4 and rep[f"single update u{u} (determinism: the same step twice)"]["params_max_abs"] <= 1e-5 for u in (0, ctx.warmup)))
        res["Long10_reproduction"] = rep
        f1 = run(ctx, "Interleaved", 0, [0, 1], device, inputs_of="Repeated"); r1 = run(ctx, "Repeated", 0, [0, 1], device)
        res["forced_interleaved_equals_repeated"] = {"loss_max_abs": float(max(abs(a - b) for a, b in zip(f1["losses"], r1["losses"]))), "params_max_abs": max_diff(f1["model"], r1["model"])}
        res["forced_interleaved_equals_repeated"]["ok"] = res["forced_interleaved_equals_repeated"]["loss_max_abs"] <= 1e-5 and res["forced_interleaved_equals_repeated"]["params_max_abs"] <= 1e-5
        del f1, r1; torch.cuda.empty_cache()
        model, heads, _ = student(ctx, 0, device); opt = vsh.make_optimizer(model, heads, ctx.o); enc = {}
        for u in (0, ctx.warmup):
            b = batch_of(ctx, "Interleaved", 0, u); tet.train_step(model, heads, opt, b, torch.from_numpy(cache["logits"][0, u].copy()), "own", vsh.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o), ctx.o, device)
            enc[f"u{u}"] = {"encoder_grads": int(sum(p.grad is not None for p in model.encoder_parameters())), "head_grads": int(sum(p.grad is not None for p in heads.parameters()))}
        res["optimiser_and_encoder"] = {"bound": vsh.bound(opt, model, heads), **enc, "ok": bool(vsh.bound(opt, model, heads) and enc["u0"]["encoder_grads"] == 0 and enc[f"u{ctx.warmup}"]["encoder_grads"] > 0
                                                                                          and enc["u0"]["head_grads"] == 0)}
        del model, heads, opt; torch.cuda.empty_cache()
        us = list(range(ctx.warmup - 2, ctx.warmup + 4)); sf = ctx.out / "check_resume.pt"; sf.unlink(missing_ok=True); straight = run(ctx, "Interleaved", 0, us, device)
        repeat = run(ctx, "Interleaved", 0, us, device); run(ctx, "Interleaved", 0, us[:3], device, sf); second = run(ctx, "Interleaved", 0, us[3:], device, sf)
        rp = max_diff(straight["model"], repeat["model"]); dif = max_diff(straight["model"], second["model"]); ld = float(max(abs(a - b) for a, b in zip(straight["losses"][3:], second["losses"])))
        res["resume"] = {"updates": us, "loss_max_abs": ld, "params_max_abs": dif, "repeat_params_max_abs": rp, "ok": bool(ld <= 1e-4 and dif <= max(1e-5, 2 * rp))}
        sf.unlink(missing_ok=True); spu = float(np.mean(straight["seconds"][2:])); del straight, repeat, second; torch.cuda.empty_cache()
        res["timing"] = {"seconds_per_update": spu, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
        a08 = tet.make_model(state, device).eval(); x = torch.randint(0, 256, (32, 16, 224, 224, 3), dtype=torch.uint8); mk = tm.marker_planes(torch.zeros(32, dtype=torch.long).to(device))
        with torch.no_grad():
            a08(tm.to_rgb(x.to(device)), mk); torch.cuda.synchronize(); a = time.time()
            for _ in range(3):
                a08(tm.to_rgb(x.to(device)), mk)
            torch.cuda.synchronize(); spw = (time.time() - a) / 96
        ev = (7666 + 7021 + 18175) * spw / 3600; spent = ledger_hours(ctx.ledger, "gpu"); per_run = ctx.total * spu / 3600
        res["budget"] = {"spent": spent, "inference_seconds_per_window": spw, "evaluation_per_model_hours": ev, "per_run_hours": per_run, "projection_two_seeds": spent + 4 * (per_run + ev),
                         "seed1_reserve": 2 * (per_run + ev), "cap": ctx.vt["budget"]["gpu_hours_max"]}
        res["budget"]["fits_with_seed1"] = bool(res["budget"]["projection_two_seeds"] + res["budget"]["seed1_reserve"] <= res["budget"]["cap"])
        res["ok"] = bool(all(res[k]["ok"] for k in ("synthetic_windows", "teacher", "arm_inputs", "Long10_reproduction", "forced_interleaved_equals_repeated", "optimiser_and_encoder", "resume"))
                         and res["budget"]["projection_two_seeds"] <= res["budget"]["cap"]); res["finished"] = now()
        flat = json.loads(json.dumps(res, default=str))
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)
        print(json.dumps({k: (v.get("ok") if isinstance(v, dict) and "ok" in v else v) for k, v in flat.items()}, indent=1), json.dumps(flat["Long10_reproduction"], indent=1), json.dumps(flat["budget"]), flush=True)
        if not res["ok"]:
            raise SystemExit("a check failed")
        ok = True; note = f"{spu:.3f} s/update, projection {res['budget']['projection_two_seeds']:.3f}"
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "check (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=7); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--prepare", action="store_true"); g.add_argument("--check", action="store_true"); g.add_argument("--queue", nargs="+")
    g.add_argument("--synthetic", action="store_true")
    a = ap.parse_args(); a.set = []
    if a.synthetic:
        print(json.dumps(synthetic_cases(), indent=1, default=str)); return
    ctx = Ctx(load_cfg(a), a.config)
    if a.prepare:
        return prepare(ctx, a.seeds, a.workers)
    device = torch.device("cuda"); determinism()
    if a.check:
        return check(ctx, device, a.workers)
    queue(ctx, a.queue, device, a.workers)


if __name__ == "__main__":
    main()
