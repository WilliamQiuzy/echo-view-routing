#!/usr/bin/env python
"""Prepare, check, train and queue the A09 cross-session discriminative round (configs/experiments/video_discriminative_v1.yaml).

Every run strictly loads its seed's V01 Ordered final state into the unchanged TargetMViT. Update u reads row u of this
round's seed manifest through V01's loader: 10 native windows (each class from two sessions, drawn by a fixed rule) and
A07's 4-window constructed group of update u, 14 windows in one forward. C trains plain CE; D adds lambda times SupCon on
the existing 512-d embedding of the native windows. Optimiser, schedule, warm-up, clipping, precision and model RNG are
A07's (V01's). Only training data is read.

  --prepare                 CPU: pins, the three manifests, constructed-window equality with A07, pixel digests, counts
  --check                   GPU: the necessary checks, the lambda rule (training batches only), timing and projection
  --queue K [K ...]         runs in order (keys plain_s0, contrast_s2, ...); --stop-at N pauses each at update N
  --arm A --seed S          one run

  python scripts/train_video_discriminative.py --config configs/experiments/video_discriminative_v1.yaml --prepare
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.prepare_video_feature_temporal as pvft  # noqa: E402
import scripts.prepare_video_sequence_objective as vpr  # noqa: E402
import scripts.train_raw_video_context as rvc  # noqa: E402
import scripts.train_video_sequence_objective as vso  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths, params_digest  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.temporal.video_sequence_features as vsf  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.losses import supcon as sc  # noqa: E402

NS = "video_discriminative_v1"
GROUP_NAME = vso.GROUP_NAME
NAT, CON = 10, vpr.GROUP
WINDOWS, SLOTS = NAT + CON, vpr.SLOTS
RNG_BASE = 20261018
LADDER = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
ARMS = ("plain", "contrast")
TRAINING_KEYS = ("predecessors", "source", "arms", "model", "data", "loss", "training", "determinism")
CODE_FILES = (Path(__file__), Path(sc.__file__), Path(rvc.__file__), Path(vso.__file__), Path(vpr.__file__), Path(tm.__file__), Path(prv.__file__),
              Path(pvft.__file__), Path(vsf.__file__))
make_optimizer, apply_schedule, save_resume, load_resume, determinism = rvc.make_optimizer, rvc.apply_schedule, rvc.save_resume, rvc.load_resume, rvc.determinism
make_model, optimiser_bound, max_diff, now, write_json, ledger_hours = vso.make_model, vso.optimiser_bound, vso.max_diff, vso.now, vso.write_json, vso.ledger_hours
ledger_append = vpr.ledger_append


def pinned(pin: dict) -> Path:
    return vpr.pinned(ROOT, pin)


# ----------------------------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vd = cfg, config, cfg["video_discriminative"]; self.src = self.vd["source"]
        a07 = self.src["a07_config"]; pinned(a07); pinned(self.src["a08_config"])
        a07cfg = load_cfg(SimpleNamespace(config=a07["path"], set=[]))
        self.a07 = vso.Ctx(a07cfg, a07["path"]); self.a07p = vpr.Ctx(a07cfg, a07["path"]); self.rv, self.paths = self.a07.rv, self.a07.paths
        tr = self.vd["training"]; a7 = self.a07.vso["training"]
        for k in ("total_updates", "updates_per_epoch", "optimiser", "warmup_updates"):
            if tr[k] != a7[k]:
                raise SystemExit(f"training.{k} differs from A07's")
        self.seeds = [int(s) for s in tr["seeds"]]; self.total, self.upe, self.warmup, self.o = int(tr["total_updates"]), int(tr["updates_per_epoch"]), int(tr["warmup_updates"]), tr["optimiser"]
        o = self.vd["outputs"]; self.out, self.ckpt_root, self.results = ROOT / o["root"], ROOT / o["checkpoints"], ROOT / o["results_dir"]
        self.mdir = self.out / "manifests"; self.ledger = ROOT / self.vd["budget"]["ledger"]; self.tau = float(self.vd["loss"]["tau"]); self._m, self._fp = {}, None

    def a07_manifest(self, seed: int) -> dict:
        m, sha = self.a07.manifest(seed)
        if sha != self.src["a07_manifests"][seed]["sha256"]:
            raise SystemExit(f"seed {seed}: A07's manifest differs from this round's pin")
        return m

    def manifest(self, seed: int) -> tuple[dict, str]:
        if seed not in self._m:
            e = json.loads((self.mdir / "index.json").read_text())["seeds"][str(seed)]; path = self.mdir / e["file"]
            if (got := sha256_file(path)) != e["sha256"]:
                raise SystemExit(f"{path.name}: sha256 {got[:12]} differs from index.json")
            with np.load(path) as z:
                m = {k: z[k] for k in z.files}
            if vpr.arrays_digest(m) != e["arrays_digest"]:
                raise SystemExit(f"{path.name}: its arrays differ from index.json")
            self._m[seed] = (m, got)
        return self._m[seed]

    def frame_paths(self) -> FramePaths:
        if self._fp is None:
            vids = {str(v) for s in self.seeds for v in self.manifest(s)[0]["videos"].tolist()}
            if vids - set(prv.allowlist(self.rv, ROOT)):
                raise SystemExit("a manifest recording is not a training clip")
            self._fp = FramePaths(prv.frame_dirs(self.paths, self.rv, vids))
        return self._fp

    def lam(self) -> float:
        rec = json.loads((self.out / "lambda.json").read_text())
        if rec["config_sha256"] != sha256_file(ROOT / self.config):
            raise SystemExit("lambda.json was fixed under another configuration")
        return float(rec["lambda"])


# ----------------------------------------------------------------------------------------------- manifests (CPU)
def native_pool(seqs: list[dict]) -> dict[int, list[tuple[str, list[int]]]]:
    """Per class: (session, native sequence indices of that class) in sorted session order."""
    by: dict[int, dict[str, list[int]]] = {c: {} for c in range(len(vpr.CLASSES))}
    for i, q in enumerate(seqs):
        if q["kind"] == "native":
            by[int(np.asarray(q["labels"])[0])].setdefault(session_of(q["id"]), []).append(i)
    return {c: [(s, sorted(v)) for s, v in sorted(d.items())] for c, d in by.items()}


def draw_natives(pool: dict, info: list[dict], seed: int, u: int) -> list[tuple[int, int]]:
    """The fixed draw of update u: per class two distinct sessions, a clip of the class in each, a centre of the clip."""
    rng = np.random.default_rng([RNG_BASE, int(seed), int(u)]); out = []
    for c in range(len(vpr.CLASSES)):
        sess = pool[c]; a, b = rng.choice(len(sess), size=2, replace=False)
        for j in (int(a), int(b)):
            clips = sess[j][1]; qi = clips[int(rng.integers(len(clips)))]; out.append((qi, int(rng.integers(len(info[qi]["windows"]["centres"])))))
    return out


def plan(pool: dict, info: list[dict], a07m: dict, seed: int, total: int) -> list[list[tuple[int, int]]]:
    out = []
    for u in range(int(total)):
        qi = int(a07m["group_seq"][u, 1]); idx = vpr.group_indices(len(info[qi]["windows"]["centres"]), int(a07m["group_start"][u, 1]))
        out.append(draw_natives(pool, info, seed, u) + [(qi, int(ci)) for ci in idx])
    return out


def build_manifest(seqs: list[dict], info: list[dict], windows: list[list[tuple[int, int]]], a07m: dict) -> dict:
    """V01's manifest layout with 14 windows per update; the constructed windows keep A07's repeat flags."""
    U = len(windows); refs_of = lambda qi, ci: [seqs[qi]["refs"][int(x)] for x in info[qi]["windows"]["samples"][int(ci)]]  # noqa: E731
    videos = sorted({r[0] for ws in windows for qi, ci in ws for r in refs_of(qi, ci)}); vix = {v: i for i, v in enumerate(videos)}
    sessions = sorted({session_of(seqs[qi]["id"]) for ws in windows for qi, _ in ws[:NAT]}); six = {s: i for i, s in enumerate(sessions)}
    m = {"vid": np.empty((U, WINDOWS, SLOTS), np.int32), "frame": np.empty((U, WINDOWS, SLOTS), np.int32), "gamma": np.empty((U, WINDOWS, SLOTS), np.float32),
         "s": np.empty((U, WINDOWS, SLOTS), np.int32), "c": np.empty((U, WINDOWS), np.int32), "target": np.empty((U, WINDOWS), np.int8),
         "kind": np.empty((U, WINDOWS), np.int8), "seq": np.empty((U, WINDOWS), np.int32), "centre_index": np.empty((U, WINDOWS), np.int32),
         "marker": np.empty((U, WINDOWS), np.int8), "repeats": np.zeros((U, WINDOWS), bool), "sess": np.full((U, WINDOWS), -1, np.int32)}
    for u, ws in enumerate(windows):
        for w, (qi, ci) in enumerate(ws):
            W = info[qi]["windows"]; s = W["samples"][int(ci)]; c = int(W["centres"][int(ci)]); refs = refs_of(qi, ci)
            m["vid"][u, w] = [vix[r[0]] for r in refs]; m["frame"][u, w] = [int(r[1]) for r in refs]; m["gamma"][u, w] = [float(r[2]) for r in refs]
            m["s"][u, w], m["c"][u, w], m["target"][u, w], m["kind"][u, w] = s, c, int(np.asarray(seqs[qi]["labels"])[c]), int(w >= NAT)
            m["seq"][u, w], m["centre_index"][u, w], m["marker"][u, w] = qi, int(ci), int(W["markers"][int(ci)])
            if w < NAT:
                m["sess"][u, w] = six[session_of(seqs[qi]["id"])]
        m["repeats"][u, NAT:] = a07m["repeats"][u, CON:]
    m["order_ordered"] = np.broadcast_to(np.arange(SLOTS, dtype=np.int8), (U, WINDOWS, SLOTS)).copy()
    m["videos"], m["sessions"] = np.asarray(videos), np.asarray(sessions)
    return m


def constructed_equal(m: dict, a07m: dict) -> bool:
    """Windows 10-13 of every update equal A07's windows 4-7 reference by reference."""
    names, a_names = m["videos"][m["vid"][:, NAT:]], a07m["videos"][a07m["vid"][:, CON:]]
    same = np.array_equal(names, a_names)
    for k in ("frame", "gamma", "s", "c", "target", "marker", "centre_index", "seq"):
        same = same and np.array_equal(m[k][:, NAT:], a07m[k][:, CON:])
    return bool(same)


def window_digests(m: dict, hashes: dict, ws: range) -> np.ndarray:
    videos = m["videos"]; U = m["vid"].shape[0]; out = np.empty((U, len(ws), 32), np.uint8)
    for u in range(U):
        for k, w in enumerate(ws):
            slot = [hashes[(str(videos[m["vid"][u, w, i]]), int(m["frame"][u, w, i]), round(float(m["gamma"][u, w, i]), 6))] for i in range(SLOTS)]
            out[u, k] = np.frombuffer(tm.window_digest([slot[j] for j in m["order_ordered"][u, w]]), np.uint8)
    return out


def manifest_counts(m: dict, info: list[dict]) -> dict:
    nat = m["seq"][:, :NAT]; pairs = Counter(zip(nat.ravel().tolist(), m["centre_index"][:, :NAT].ravel().tolist()))
    sess = m["sess"][:, :NAT]; short = np.array([len(info[int(q)]["windows"]["centres"]) < CON for q in nat.ravel()])
    shared = int(sum(len(set(r.tolist())) < NAT for r in sess)); within = all(sess[u, 2 * c] != sess[u, 2 * c + 1] for u in range(len(sess)) for c in range(5))
    tgt = m["target"][:, :NAT]
    return {"updates": int(m["vid"].shape[0]), "native_windows": int(nat.size), "native_targets_by_class": {vpr.CLASSES[k]: int(v) for k, v in sorted(Counter(tgt.ravel().tolist()).items())},
            "class_layout_ok": bool(all(np.array_equal(tgt[u], np.repeat(np.arange(5), 2)) for u in range(len(tgt)))), "pair_sessions_distinct": bool(within),
            "updates_with_a_session_shared_across_classes": shared, "distinct_native_clips": int(len(set(nat.ravel().tolist()))),
            "distinct_sessions": int(len(set(sess.ravel().tolist()))), "repeated_native_windows": int(sum(v - 1 for v in pairs.values() if v > 1)),
            "max_draws_per_native_window": int(max(pairs.values())), "max_draws_per_clip": int(max(Counter(nat.ravel().tolist()).values())),
            "max_draws_per_session": int(max(Counter(sess.ravel().tolist()).values())), "windows_from_clips_under_4_centres": int(short.sum()),
            "constructed_repeated_targets": int(m["repeats"][:, NAT:].sum()), "recordings_in_windows": int(len(m["videos"]))}


def prepare(ctx: Ctx, workers: int) -> None:
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        rec = {"config": ctx.config, "config_sha256": sha256_file(ROOT / ctx.config), "started": started,
               "pins": {"a07_config": sha256_file(pinned(ctx.src["a07_config"])), "a08_config": sha256_file(pinned(ctx.src["a08_config"])),
                        "a08_choices": sha256_file(pinned(ctx.src["a08_choices"])), "a07_inherited": vpr.verify_sources(ctx.a07p)}}
        data = vpr.training_data(ctx.a07p); seqs = data.seqs; info = vpr.sequence_info(seqs); pool = native_pool(seqs)
        rec["pool"] = {vpr.CLASSES[c]: {"sessions": len(v), "clips": int(sum(len(x[1]) for x in v))} for c, v in pool.items()}
        ms, plans = {}, {}
        for s in ctx.seeds:
            a07m = ctx.a07_manifest(s); plans[s] = plan(pool, info, a07m, s, ctx.total)
            if plan(pool, info, a07m, s, ctx.total) != plans[s]:
                raise SystemExit(f"seed {s}: a second draw differs")
            m = build_manifest(seqs, info, plans[s], a07m)
            if not constructed_equal(m, a07m):
                raise SystemExit(f"seed {s}: the constructed windows differ from A07's")
            ms[s] = (m, a07m)
        print("plans drawn twice, constructed windows equal A07's", flush=True)
        refs = {(str(m["videos"][v]), int(f), round(float(g), 6)) for m, _ in ms.values() for v, f, g in
                zip(m["vid"][:, :NAT].ravel(), m["frame"][:, :NAT].ravel(), m["gamma"][:, :NAT].ravel())}
        vids = {r[0] for r in refs} | {str(v) for m, _ in ms.values() for v in m["videos"].tolist()}
        if vids - set(prv.allowlist(ctx.rv, ROOT)):
            raise SystemExit("a recording lies outside the training allowlist")
        dirs = prv.frame_dirs(ctx.paths, ctx.rv, {r[0] for r in refs}); t1 = time.time(); hashes = prv.pixel_hashes(refs, dirs, workers)
        rec["pixels"] = {"distinct_native_frames": len(hashes), "seconds": time.time() - t1}
        ctx.mdir.mkdir(parents=True, exist_ok=True); index = {"config": ctx.config, "seeds": {}}; rec["seeds"] = {}
        for s, (m, a07m) in ms.items():
            m["digest_ordered"] = np.concatenate([window_digests(m, hashes, range(NAT)), a07m["digest_ordered"][:, CON:]], axis=1)
            path = ctx.mdir / f"VDS_s{s}.npz"
            if path.is_file():
                raise SystemExit(f"{path.name} exists; manifests are never overwritten")
            np.savez(path, **m); sha = sha256_file(path)
            index["seeds"][str(s)] = {"file": path.name, "sha256": sha, "arrays_digest": vpr.arrays_digest(m), "epoch_digests": vpr.epoch_digests(m, ctx.upe)}
            rec["seeds"][str(s)] = {"sha256": sha, "counts": manifest_counts(m, info)}
        write_json(ctx.mdir / "index.json", index); rec["index_sha256"] = sha256_file(ctx.mdir / "index.json")
        ids = {str(v) for m, _ in ms.values() for v in m["videos"].tolist()} | {str(x) for m, _ in ms.values() for x in m["sessions"].tolist()}
        ids |= {session_of(v) for v in ids}
        rec["finished"] = now(); rec["seconds"] = time.time() - t0; flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "preparation")
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, flat)
        print(json.dumps({k: rec[k] for k in ("pool", "pixels")}, indent=1), json.dumps(rec["seeds"], indent=1), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "prepare", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- inputs, identity
class Inputs(Dataset):
    def __init__(self, m: dict, fp, updates: list[int]) -> None:
        self.m, self.order, self.fp, self.updates = m, m[f"order_{tm.KEYS[GROUP_NAME]}"], fp, list(updates)

    def __len__(self) -> int:
        return len(self.updates)

    def __getitem__(self, i: int):
        u = self.updates[i]
        return (u, *rvc.window_inputs(self.m, self.order, self.fp, [(u, w) for w in range(WINDOWS)]))


def loader(m: dict, fp, updates: list[int], workers: int) -> DataLoader:
    return DataLoader(Inputs(m, fp, updates), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None, pin_memory=True)


def code_digest() -> str:
    h = hashlib.sha256()
    for p in CODE_FILES:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def identity(ctx: Ctx, arm: str, seed: int, msha: str, ssha: str, lam: float) -> str:
    payload = {"experiment": ctx.cfg.get("experiment"), "video_discriminative": {k: ctx.vd[k] for k in TRAINING_KEYS}, "arm": arm, "seed": int(seed),
               "lambda": float(lam) if arm == "contrast" else 0.0, "manifest_sha256": msha, "source_weights_sha256": ssha,
               "epoch_digests": json.loads((ctx.mdir / "index.json").read_text())["seeds"][str(seed)]["epoch_digests"], "environment": prv.environment(),
               "total_updates": int(ctx.total), "code": code_digest()}
    canon = json.loads(json.dumps(payload, default=str))
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------- one update
def train_step(model, opt, batch, sess: np.ndarray, arm: str, lam: float, lrs: tuple[float, float], o: dict, device, tau: float, keep: bool = False) -> dict:
    """One forward of the 14 windows; C: mean CE; D: mean CE + lam x SupCon of the 10 native embeddings; S and the
    embedding geometry are logged from the same forward in both arms; clipping and the AdamW step."""
    u8, slots, target = batch[1], batch[2], batch[3]; t = target.to(device); sn = torch.as_tensor(np.asarray(sess[:NAT]), device=device)
    model.train()
    e = model.embed(tm.to_rgb(u8.to(device, non_blocking=True)), tm.marker_planes(slots.to(device))); z = model.classifier(e)
    ce = F.cross_entropy(z, t)
    if arm == "contrast":
        s, k = sc.supcon(e[:NAT], t[:NAT], sn, tau); loss = ce + float(lam) * s
    else:
        with torch.no_grad():
            s, k = sc.supcon(e[:NAT].detach(), t[:NAT], sn, tau)
        loss = ce
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    geo = sc.geometry(e[:NAT], t[:NAT], sn)
    opt.zero_grad(set_to_none=True); loss.backward()
    params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    gn = float(torch.nn.utils.clip_grad_norm_(params, float(o["grad_clip"])))
    if not np.isfinite(gn):
        raise FloatingPointError("non-finite gradient norm")
    opt.step()
    pred = z.detach().argmax(1).cpu().numpy(); tt = target.numpy()
    rec = {"loss": float(loss.detach()), "ce": float(ce.detach()), "supcon": float(s.detach()), "anchors": k, **geo, "grad_norm": gn,
           "lr_encoder": lrs[0], "lr_new": lrs[1], "correct_native": int((pred[:NAT] == tt[:NAT]).sum()), "correct_constructed": int((pred[NAT:] == tt[NAT:]).sum()),
           "a4c_a5c_native_errors": int(sum((tt[i], pred[i]) in ((2, 3), (3, 2)) for i in range(NAT)))}
    if keep:
        rec["logits"] = z.detach().cpu()
    return rec


def epoch_record(acc: list[dict], done: int, t0: float, device, partial: bool) -> dict:
    mean = lambda k: float(np.nanmean([a[k] for a in acc]))  # noqa: E731
    return {"updates": done, "n": len(acc), "partial": partial, **{k: mean(k) for k in ("loss", "ce", "supcon", "cos_pos", "cos_neg", "cos_a4c_a5c", "grad_norm")},
            "grad_norm_max": float(max(a["grad_norm"] for a in acc)), "native_accuracy": sum(a["correct_native"] for a in acc) / (NAT * len(acc)),
            "constructed_accuracy": sum(a["correct_constructed"] for a in acc) / (CON * len(acc)), "a4c_a5c_native_errors": int(sum(a["a4c_a5c_native_errors"] for a in acc)),
            "lr_encoder_last": acc[-1]["lr_encoder"], "lr_new_last": acc[-1]["lr_new"], "seconds_this_process": time.time() - t0,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if torch.cuda.is_available() else None}


# ----------------------------------------------------------------------------------------------- formal run
def run_key(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


def train(ctx: Ctx, arm: str, seed: int, device, workers: int, stop_at: int | None) -> dict:
    lam = ctx.lam() if arm == "contrast" else 0.0
    m, msha = ctx.manifest(seed); state, ssha = ctx.a07.source_state(seed); ident = identity(ctx, arm, seed, msha, ssha, lam); key = run_key(arm, seed)
    run_dir, ckpt = ctx.out / "train" / key, ctx.ckpt_root / key; run_dir.mkdir(parents=True, exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)
    if rvc.finished(run_dir / "run_manifest.json", ident):
        print(f"[{key}] already finished", flush=True); return {"key": key, "status": "finished"}
    model = make_model(state, device); opt = make_optimizer(model, ctx.o)
    if not optimiser_bound(opt, model):
        raise SystemExit("the optimiser is not bound to the model's parameters")
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); done, log = 0, []
    upd_log = run_dir / "updates.jsonl"
    if (ckpt / "resume.pt").is_file():
        done, log = load_resume(ckpt / "resume.pt", model, opt, ident)
        lines = upd_log.read_text().splitlines()[:done] if upd_log.is_file() else []; upd_log.write_text("".join(x + "\n" for x in lines))
        with (run_dir / "resumes.jsonl").open("a") as fh:
            fh.write(json.dumps({"at_update": done, "time": now()}) + "\n")
    else:
        for p in (upd_log, run_dir / "resumes.jsonl", run_dir / "train_log.jsonl"):
            p.unlink(missing_ok=True)
    end = min(ctx.total, int(stop_at)) if stop_at else ctx.total
    if done >= end:
        print(f"[{key}] already at update {done} (pause point {end})", flush=True); return {"key": key, "status": "at_pause", "updates": done}
    print(f"[{key}] identity {ident[:12]} lambda {lam}; updates {done} -> {end} of {ctx.total}", flush=True)
    sessions = m["sess"]; t0 = time.time(); acc = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader(m, ctx.frame_paths(), list(range(done, end)), workers):
        u = int(batch[0]); rvc.check_digests(m, GROUP_NAME, u, batch[4])
        if not np.array_equal(batch[3].numpy(), m["target"][u]) or not np.array_equal(batch[2].numpy(), m["marker"][u]):
            raise SystemExit(f"update {u}: the loader's targets or markers differ from the manifest")
        lrs = apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        r = train_step(model, opt, batch, sessions[u], arm, lam, lrs, ctx.o, device, ctx.tau); acc.append(r); done = u + 1
        with upd_log.open("a") as fh:
            fh.write(json.dumps({"u": u, **{k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}}) + "\n")
        if done % 25 == 0:
            write_json(run_dir / "progress.json", {"identity": ident, "updates": done, "end": end, "total": ctx.total, "seconds": time.time() - t0, "loss": r["loss"]})
        if done % ctx.upe == 0 or done == end:
            rec = epoch_record(acc, done, t0, device, partial=done % ctx.upe != 0); log.append(rec); save_resume(ckpt / "resume.pt", model, opt, done, ident, log)
            with (run_dir / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{key}] {done}/{ctx.total} loss {rec['loss']:.4f} CE {rec['ce']:.4f} S {rec['supcon']:.4f} cos+ {rec['cos_pos']:.3f} cos- {rec['cos_neg']:.3f} "
                  f"a45 {rec['cos_a4c_a5c']:.3f} nat {rec['native_accuracy']:.3f} grad {rec['grad_norm']:.2f} {rec['seconds_this_process']:.0f}s", flush=True)
            acc = []
    if done < ctx.total:
        snap = ckpt / f"pilot_u{done}.pt"; torch.save(model.state_dict(), snap)
        out = {"identity": ident, "key": key, "updates": done, "weights": str(snap), "sha256": sha256_file(snap), "lambda": lam, "seconds_this_process": time.time() - t0, "time": now()}
        write_json(run_dir / f"pilot_u{done}.json", out); print(f"[{key}] paused at {done}; snapshot {out['sha256'][:12]}", flush=True)
        return {"key": key, "status": "paused", **out}
    final = ckpt / "final.pt"; torch.save(model.state_dict(), final)
    man = {"identity": ident, "arm": arm, "key": key, "seed": seed, "lambda": lam, "config": ctx.config, "config_sha256": sha256_file(ROOT / ctx.config),
           "manifest_sha256": msha, "source_weights_sha256": ssha, "environment": prv.environment(), "code": code_digest(), "updates": done, "epochs": log,
           "resumes": (run_dir / "resumes.jsonl").read_text().splitlines() if (run_dir / "resumes.jsonl").is_file() else [],
           "final_checkpoint": str(final), "final_sha256": sha256_file(final), "seconds_last_process": time.time() - t0}
    write_json(run_dir / "run_manifest.json", man); (ckpt / "resume.pt").unlink(missing_ok=True)
    print(f"[{key}] done: final {man['final_sha256'][:12]}", flush=True)
    return {"key": key, "status": "finished", "final_sha256": man["final_sha256"], "seconds_this_process": time.time() - t0}


# ----------------------------------------------------------------------------------------------- checks
def batch_of(ctx: Ctx, seed: int, u: int):
    m, _ = ctx.manifest(seed); b = Inputs(m, ctx.frame_paths(), [u])[0]
    rvc.check_digests(m, GROUP_NAME, u, b[4])
    return b, m["sess"][u]


def grad_norms(model, batch, sess, device, tau: float) -> dict:
    """||grad CE|| and ||grad S|| over the encoder and marker-stem parameters from one train-mode forward (encoder trainable)."""
    model.train(); model.set_encoder_trainable(True); params = [*model.encoder_parameters(), *model.marker_stem.parameters()]
    t = batch[3].to(device); sn = torch.as_tensor(np.asarray(sess[:NAT]), device=device)
    e = model.embed(tm.to_rgb(batch[1].to(device)), tm.marker_planes(batch[2].to(device))); z = model.classifier(e)
    ce = F.cross_entropy(z, t); s, k = sc.supcon(e[:NAT], t[:NAT], sn, tau)
    gc = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True); gs = torch.autograd.grad(s, params, retain_graph=True, allow_unused=True)
    gcls = torch.autograd.grad(s, list(model.classifier.parameters()), allow_unused=True)
    norm = lambda gs_: float(torch.sqrt(sum((g.double() ** 2).sum() for g in gs_ if g is not None)))  # noqa: E731
    by_id = {id(p): g for p, g in zip(params, gs)}
    reach = {n: (None if by_id[id(p)] is None else float(by_id[id(p)].abs().max())) for n, p in
             (("projection", model.encoder.head[-1].weight), ("conv_proj", model.encoder.conv_proj.weight), ("marker_stem", model.marker_stem.weight))}
    return {"ce": float(ce), "supcon": float(s), "anchors": k, "grad_ce": norm(gc), "grad_s": norm(gs), "ratio": norm(gs) / max(norm(gc), 1e-30),
            "s_without_gradient": int(sum(g is None for g in gs)), "s_nonfinite": int(sum(g is not None and not bool(torch.isfinite(g).all()) for g in gs)),
            "s_reach": reach, "s_classifier_gradient": [None if g is None else float(g.abs().max()) for g in gcls]}


def run_updates(ctx: Ctx, arm: str, lam: float, us: list[int], device, state_file: Path | None = None, keep: bool = False) -> dict:
    state, _ = ctx.a07.source_state(0); model = make_model(state, device); opt = make_optimizer(model, ctx.o); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    if state_file is not None:
        load_resume(state_file, model, opt, "check")
    recs = []
    for u in us:
        b, sess = batch_of(ctx, 0, u); lrs = apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        recs.append(train_step(model, opt, b, sess, arm, lam, lrs, ctx.o, device, ctx.tau, keep=keep))
    return {"model": model, "opt": opt, "records": recs}


def check(ctx: Ctx, device, workers: int) -> None:
    t0, started, ok, note = time.time(), now(), False, ""; out = {"config_sha256": sha256_file(ROOT / ctx.config), "started": started}
    try:
        determinism()
        # pairing and masks (every update of every seed; CPU)
        out["masks"] = {}
        for s in ctx.seeds:
            m, _ = ctx.manifest(s); t = m["target"][:, :NAT].astype(np.int64); se = m["sess"][:, :NAT]; bad_pos = bad_same = 0
            for u in range(len(t)):
                pos, cand = sc.masks(torch.as_tensor(t[u]), torch.as_tensor(se[u]))
                bad_pos += int((pos.sum(1) == 0).sum()); bad_same += int((pos & (torch.as_tensor(se[u])[:, None] == torch.as_tensor(se[u])[None, :])).sum())
            out["masks"][str(s)] = {"anchors_without_positive": bad_pos, "same_session_positives": bad_same, "ok": bad_pos == 0 and bad_same == 0}
        if not all(v["ok"] for v in out["masks"].values()):
            raise SystemExit("an anchor lacks a cross-session positive or a positive shares its session")
        # lambda rule (training batches only)
        rows = []
        for s in (0, 2):
            state, _ = ctx.a07.source_state(s); model = make_model(state, device); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
            for u in range(4):
                b, sess = batch_of(ctx, s, u); r = grad_norms(model, b, sess, device, ctx.tau); r.update(seed=s, u=u); rows.append(r)
            del model; torch.cuda.empty_cache()
        r_med = float(np.median([x["ratio"] for x in rows])); fit = [x for x in LADDER if x * r_med <= 1.0]; lam = fit[0] if fit else LADDER[-1]
        out["lambda_rule"] = {"batches": rows, "median_ratio": r_med, "ladder": list(LADDER), "lambda": lam, "flag_none_fits": not fit}
        reach_ok = all(x["s_nonfinite"] == 0 and all(v is not None and v > 0 for v in x["s_reach"].values()) and all(g in (None, 0.0) for g in x["s_classifier_gradient"]) for x in rows)
        out["gradient_reach_ok"] = bool(reach_ok)
        if not reach_ok:
            raise SystemExit("S does not reach the projection, conv_proj and marker stem, reaches the classifier or is not finite")
        print(f"lambda rule: median ratio {r_med:.4g} -> lambda {lam}", flush=True)
        # lambda 0 equals C (encoder trainable: updates 375-380 of the formal schedule)
        us = list(range(ctx.warmup, ctx.warmup + 6))
        c6 = run_updates(ctx, "plain", 0.0, us, device, keep=True); d6 = run_updates(ctx, "contrast", 0.0, us, device, keep=True)
        out["lambda_zero"] = {"first_logits_identical": bool(torch.equal(c6["records"][0]["logits"], d6["records"][0]["logits"])),
                              "first_loss_equal": c6["records"][0]["loss"] == d6["records"][0]["loss"],
                              "loss_max_abs_6": float(max(abs(a["loss"] - b["loss"]) for a, b in zip(c6["records"], d6["records"]))), "params_max_abs_6": max_diff(c6["model"], d6["model"])}
        out["lambda_zero"]["ok"] = bool(out["lambda_zero"]["loss_max_abs_6"] <= 1e-4 and out["lambda_zero"]["params_max_abs_6"] <= 1e-5)
        out["optimiser_bound"] = bool(optimiser_bound(c6["opt"], c6["model"])); del d6; torch.cuda.empty_cache()
        # resume 3 + 3 against 6 (arm D at the fixed lambda)
        full = run_updates(ctx, "contrast", lam, us, device); a3 = run_updates(ctx, "contrast", lam, us[:3], device); tmp = ctx.out / "check_resume.pt"
        save_resume(tmp, a3["model"], a3["opt"], 3, "check", []); del a3; b3 = run_updates(ctx, "contrast", lam, us[3:], device, state_file=tmp); tmp.unlink()
        out["resume"] = {"loss_max_abs": float(max(abs(a["loss"] - b["loss"]) for a, b in zip(full["records"][3:], b3["records"]))), "params_max_abs": max_diff(full["model"], b3["model"])}
        out["resume"]["ok"] = bool(out["resume"]["loss_max_abs"] <= 1e-4 and out["resume"]["params_max_abs"] <= 1e-5); del full, b3, c6; torch.cuda.empty_cache()
        # label isolation and source state
        state, ssha = ctx.a07.source_state(0); model = make_model(state, device); model.eval(); b, sess = batch_of(ctx, 0, 0)
        with torch.no_grad():
            z1 = model(tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device))); z2 = model(tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device)))
        out["label_isolation"] = {"forward_inputs": "rgb, marker", "eval_logits_identical": bool(torch.equal(z1, z2))}
        out["source"] = {"sha256": ssha, "pin": ctx.a07.vso["source"]["weights"][0]["sha256"], "strict_load": True, "ok": ssha == ctx.a07.vso["source"]["weights"][0]["sha256"]}
        del model; torch.cuda.empty_cache()
        # timing: 20 formal updates of D (encoder trainable) after 3 warm-up updates
        state, _ = ctx.a07.source_state(0); model = make_model(state, device); opt = make_optimizer(model, ctx.o); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats(device); m, _ = ctx.manifest(0); ts = []
        for i, batch in enumerate(loader(m, ctx.frame_paths(), list(range(ctx.warmup, ctx.warmup + 23)), workers)):
            torch.cuda.synchronize(); a = time.time(); u = int(batch[0]); lrs = apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
            train_step(model, opt, batch, m["sess"][u], "contrast", lam, lrs, ctx.o, device, ctx.tau); torch.cuda.synchronize()
            if i >= 3:
                ts.append(time.time() - a)
        spu = float(np.mean(ts)); out["timing"] = {"seconds_per_update_compute": spu, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20}
        del model, opt; torch.cuda.empty_cache()
        out["ok"] = bool(out["lambda_zero"]["ok"] and out["resume"]["ok"] and out["optimiser_bound"] and out["source"]["ok"] and out["label_isolation"]["eval_logits_identical"])
        if out["ok"]:
            write_json(ctx.out / "lambda.json", {"config_sha256": out["config_sha256"], "lambda": lam, "median_ratio": r_med, "time": now()})
        out["finished"] = now(); out["seconds"] = time.time() - t0; ok = out["ok"]; note = f"lambda {lam}, ratio {r_med:.4g}, {spu:.3f} s/update"
        for p in (ctx.out / "checks.json", ctx.results / "checks.json"):
            p.parent.mkdir(parents=True, exist_ok=True); write_json(p, out)
        print(json.dumps({k: out[k] for k in ("masks", "gradient_reach_ok", "lambda_zero", "optimiser_bound", "resume", "label_isolation", "source", "timing", "ok")}, indent=1, default=str), flush=True)
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "check", "gpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- queue
def start_of(ctx: Ctx, key: str) -> int:
    """The update a run would resume from: its latest pilot pause, else 0 (a projection input only)."""
    got = [json.loads(p.read_text())["updates"] for p in (ctx.out / "train" / key).glob("pilot_u*.json")] if (ctx.out / "train" / key).is_dir() else []
    return int(max(got, default=0))


def queue(ctx: Ctx, keys: list[str], device, workers: int, stop_at: int | None) -> None:
    b = ctx.vd["budget"]; chk = json.loads((ctx.out / "checks.json").read_text())
    if not chk["ok"]:
        raise SystemExit("the checks did not pass")
    spu = float(chk["timing"]["seconds_per_update_compute"]) * 1.10            # 10% for loading and logging
    for i, key in enumerate(keys):
        arm, s = key.rsplit("_s", 1); s = int(s); end = min(ctx.total, stop_at) if stop_at else ctx.total
        rem = keys[i:]; spent = ledger_hours(ctx.ledger, "gpu"); proj = spent + sum(end - start_of(ctx, k) for k in rem) * spu / 3600 + 0.5
        cap = float(b["pilot_gpu_hours_max"]) if stop_at else float(b["gpu_hours_max"])
        print(f"[queue] before {key}: spent {spent:.3f} + remaining {len(rem)} runs -> {proj:.3f} of {cap}", flush=True)
        if proj > cap:
            raise SystemExit(f"projection {proj:.3f} exceeds the cap {cap}; nothing new starts")
        t0, started, ok, note = time.time(), now(), False, ""
        try:
            r = train(ctx, arm, s, device, workers, stop_at); ok = True; note = json.dumps({k: r.get(k) for k in ("status", "updates", "sha256", "final_sha256")})
        except BaseException as e:  # noqa: BLE001 - recorded, then raised again
            note = f"{type(e).__name__}: {str(e)[:200]}"; raise
        finally:
            ledger_append(ctx.ledger, f"train {key}" + (f" to {stop_at}" if stop_at else ""), "gpu", started, time.time() - t0, ok, note)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--stop-at", type=int, default=None)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--prepare", action="store_true"); g.add_argument("--check", action="store_true")
    g.add_argument("--queue", nargs="+"); g.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    if a.prepare:
        return prepare(ctx, a.workers)
    determinism(); device = torch.device("cuda")
    if a.check:
        return check(ctx, device, a.workers)
    if a.queue:
        return queue(ctx, a.queue, device, a.workers, a.stop_at)
    train(ctx, a.arm, int(a.seed), device, a.workers, a.stop_at)


if __name__ == "__main__":
    main()
