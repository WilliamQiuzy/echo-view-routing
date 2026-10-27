#!/usr/bin/env python
"""A40 (configs/experiments/video_masked_feature_recognition_v1.yaml): masked-region HOG auxiliary (H) against the matched
coefficient-0 continuation (R); both continue the whole A08 (encoder, projection, marker stem, classifier) for 900 updates.

  --stage check    groups / pixels / HOG / mask / token / hook checks on fit frames, Init = A31 cache and Base, identical
                   arms, accumulation and microbatch, unfreeze / binding, aux-only gradient path, forced lambda 0, repeat
                   and 3 + 3 resume, throughput / peak memory and the launch projection (GPU)
  --stage lambda   --seed s: the fit-only coefficient of each fit fold from updates 0-3 (GPU)
  --stage train    --seed s --lambda-sha h: R_f0, H_f0, R_f1, H_f1, resumable (GPU)
  --stage freeze   --seed s --commit c --manifest-sha h: record that weights_s{s}.json (sha h) was committed in c
  --stage readout  --seed s: after the weights are committed, one opposite-fold online readout (+ the source-2 rule) (GPU);
                   --dry-run reads Init (z = q) through the same tables and publishes nothing (CPU)
The frozen A08 + B3 path supplies q, Base and its runs; nothing here changes it.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # A27's CPU-only trainer import would otherwise hide the GPU

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b3_coverage as bc  # noqa: E402
import scripts.diagnose_a5c_evidence as d32  # noqa: E402
import scripts.prepare_video_sequence_objective as a07  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.train_ensemble_teacher as tet  # noqa: E402
import scripts.train_interval_visual_adaptation as t28  # noqa: E402
import scripts.train_native_spatial_detail as t36  # noqa: E402
import scripts.train_native_window_recognition as t29  # noqa: E402
import scripts.train_patch_evidence as t39  # noqa: E402
import scripts.train_predicted_run_context as t38  # noqa: E402
import scripts.train_session_correction as t31  # noqa: E402
import scripts.train_wider_temporal_evidence as t37  # noqa: E402
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.temporal.interval_recognition as ir  # noqa: E402
import echo_routing.temporal.joint_visual as jv  # noqa: E402
import echo_routing.temporal.masked_feature as mf  # noqa: E402
import echo_routing.temporal.video_sequence_features as vsf  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
import echo_routing.vit.target_patch_readout as tp  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402

A31_CONFIG, A32_CONFIG = t36.A31_CONFIG, t36.A32_CONFIG
FOLDS, ARMS, U, WARM, PER, SAVE_EVERY, SPLIT_EVERY = (0, 1), ("R", "H"), 900, 90, 8, 150, 50
LR_ENC, LR_NEW, CLIP, BATCH = 1e-5, 1e-4, 1.0, 32
SCH = {"updates": U, "warmup": WARM}
LIMITS = {"loss": 1e-4, "params": 1e-5}
TOL = {"parity": 1e-4, "hog_reference": 1e-4, "exact_float": 1e-5, "microbatch_rel": 1e-5}
A4C, A5C = d32.A4C, d32.A5C
now, write_json, ledger_append, tables, a5c_extras, ceil_pct = t31.now, t31.write_json, t31.ledger_append, t36.tables, t39.a5c_extras, t39.ceil_pct
A30_ZERO = ("encoder.blocks.15.attn.pool_q.pool.weight", "encoder.blocks.15.attn.rel_pos_h", "encoder.blocks.15.attn.rel_pos_t", "encoder.blocks.15.attn.rel_pos_w")


def model_seed(s: int, f: int) -> int:
    return 2026102640 + 10 * int(s) + int(f)


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.config, self.v = config, cfg["masked_feature"]; o = self.v["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.v["budget"]["ledger"]
        self.a31 = t31.Ctx(load_cfg(SimpleNamespace(config=A31_CONFIG, set=[])), A31_CONFIG); self.d = d32.Ctx(load_cfg(SimpleNamespace(config=A32_CONFIG, set=[])), A32_CONFIG)


def staged(ctx: Ctx, job: str, kind: str, fn) -> None:
    t36.staged(ctx, job, kind, fn)


def publish(ctx: Ctx, name: str, rec: dict) -> None:
    t36.publish(ctx, name, rec)


# ----------------------------------------------------------------------------------------------- groups and batches
class Groups:
    """A07's groups on one fit pool's pool_W centres: native starts and eligible constructed starts."""

    def __init__(self, seqs: list[dict], W: dict) -> None:
        self.rows = [np.flatnonzero(W["seq"] == i) for i in range(len(seqs))]; self.nat, self.con = [], []
        for i, q in enumerate(seqs):
            r = self.rows[i]
            if q["kind"] == "native":
                self.nat.append((i, a07.native_starts(len(r))))
            else:
                st = a07.eligible_starts(np.asarray(q["labels"])[W["centre"][r]])
                if len(st):
                    self.con.append((i, st))
        if not self.nat or not self.con:
            raise SystemExit("no legal native or constructed group: engineering stop")

    def draw(self, s: int, f: int, u: int) -> np.ndarray:
        """The 8 pool_W rows of update u: native group, then constructed group (A07's draw order)."""
        rng = np.random.default_rng([mf.BASE, mf.ROUND, int(s), int(f), int(u), 1])
        i = int(rng.integers(len(self.nat))); j = int(rng.integers(len(self.nat[i][1]))); k = int(rng.integers(len(self.con))); l_ = int(rng.integers(len(self.con[k][1])))
        (ni, ns), (ci, cs) = self.nat[i], self.con[k]; rn, rc = self.rows[ni], self.rows[ci]
        return np.concatenate([rn[a07.group_indices(len(rn), int(ns[j]))], rc[a07.group_indices(len(rc), int(cs[l_]))]])


class Fold:
    def __init__(self, ctx: Ctx, f: int) -> None:
        self.f, self.seqs = f, t31.load_pool(ctx.a31, f); self.W = t31.pool_W(self.seqs); self.G = Groups(self.seqs, self.W); self.fp = t28.frame_paths(self.seqs)


class Batches(Dataset):
    """Item = (update, rows, clean uint8 frames (8, 16, 224, 224, 3), marker slots, centre labels) of a fit fold."""

    def __init__(self, F_: Fold, s: int, us) -> None:
        self.seqs, self.W, self.G, self.fp, self.s, self.f, self.us = F_.seqs, F_.W, F_.G, F_.fp, s, F_.f, list(us)

    def __len__(self) -> int:
        return len(self.us)

    def __getitem__(self, i: int):
        u = self.us[i]; rows = self.G.draw(self.s, self.f, u); out = np.empty((PER, tm.N_SLOTS, 224, 224, 3), np.uint8); memo = {}
        for w, r in enumerate(rows):
            q = self.seqs[int(self.W["seq"][r])]
            for k, p in enumerate(self.W["s"][r]):
                ref = tuple(q["refs"][int(p)])
                if ref not in memo:
                    v, fi, g = ref; pth = self.fp(v); memo[ref] = jv.load_frame_u8(pth[min(int(fi), len(pth) - 1)], round(float(g), 6))
                out[w, k] = memo[ref]
        return u, torch.from_numpy(rows), torch.from_numpy(out), torch.from_numpy(self.W["marker"][rows].astype(np.int64)), torch.from_numpy(self.W["label"][rows].astype(np.int64))


def loader(F_: Fold, s: int, us, workers: int):
    return DataLoader(Batches(F_, s, us), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None, pin_memory=True)


def rows_sha(F_: Fold, s: int, us) -> str:
    h = hashlib.sha256()
    for u in us:
        h.update(F_.G.draw(s, F_.f, u).astype(np.int64).tobytes())
    return h.hexdigest()


def masks_sha(s: int, f: int, us) -> str:
    h = hashlib.sha256()
    for u in us:
        h.update(np.packbits(mf.mask_blocks(s, f, u, PER)).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- model, update
def build(ctx: Ctx, s: int, f: int, device):
    """The A08 copy, every parameter explicitly trainable, train mode; the fold's decoder."""
    m = ctx.a31.a08(s, device); m.set_encoder_trainable(True)
    for p in m.parameters():
        p.requires_grad_(True)
    return m.train(), mf.decoder(s, f).to(device).train()


def make_opt(m, dec):
    return t31.opt_of([("encoder", list(m.encoder.parameters()), LR_ENC), ("new", [*m.marker_stem.parameters(), *m.classifier.parameters(), *dec.parameters()], LR_NEW)])


def all_params(m, dec) -> list:
    return [*m.parameters(), *dec.parameters()]


def on_device(b, device) -> tuple:
    u, rows, u8, mk, y = b
    return int(u), rows.numpy(), u8.to(device, non_blocking=True), tm.marker_planes(mk.to(device)), y.to(device)


def update(m, dec, opt, b, s: int, f: int, coef: float, device, split: bool = False) -> dict:
    """One real update: clean CE backward, then masked coef x L_HOG backward, one clip and one AdamW step."""
    u, _, u8, mk, y = on_device(b, device); t31.sched(opt, u, SCH); opt.zero_grad(set_to_none=True); sel = torch.from_numpy(mf.mask_blocks(s, f, u, PER)).to(device)
    target = mf.hog_targets(u8); rgb = tm.to_rgb(u8); ce = F.cross_entropy(m(rgb, mk), y); ce.backward(); rec = {"u": u, "ce": float(ce.detach())}
    trunk = [p for _, p in mf.shared_trunk(m)]
    if split:
        g_ce = [None if p.grad is None else p.grad.detach().clone() for p in trunk]; rec["trunk_grad_ce"] = mf.grad_norm(trunk)
    lh = mf.masked_loss(m, dec, rgb, mk, target, sel); (coef * lh).backward(); rec["hog"] = float(lh.detach())
    if split:
        d2 = sum(float((p.grad.double() - (0 if g is None else g.double())).pow(2).sum()) for p, g in zip(trunk, g_ce) if p.grad is not None); rec["trunk_grad_coef_hog"] = d2 ** 0.5; del g_ce
    rec["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(all_params(m, dec), CLIP)); opt.step(); return rec


def run_updates(m, dec, opt, F_: Fold, s: int, us, coef: float, device, workers: int, log: list, split_every: int = 0, hook=None) -> float:
    gsec = 0.0
    for b in loader(F_, s, us, workers):
        torch.cuda.synchronize(); a = time.time(); u = int(b[0]); rec = update(m, dec, opt, b, s, F_.f, coef, device, bool(split_every) and (u % split_every == 0 or u == U - 1))
        torch.cuda.synchronize(); gsec += time.time() - a; log.append(rec)
        if hook is not None:
            hook(u, rec)
    return gsec


def state_of(m, dec, opt) -> dict:
    return {"model": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, "dec": {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()}, "opt": opt.state_dict(),
            "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}


def load_state(m, dec, opt, st: dict) -> None:
    m.load_state_dict(st["model"]); dec.load_state_dict(st["dec"]); opt.load_state_dict(st["opt"]); torch.set_rng_state(st["cpu_rng"]); torch.cuda.set_rng_state_all(st["cuda_rng"])


def params_max_diff(a: dict, b: dict) -> float:
    return float(max((a[k].float() - b[k].float()).abs().max() for k in a))


def flat_state(m, dec) -> dict:
    return {**{f"m.{k}": v.detach().cpu() for k, v in m.state_dict().items()}, **{f"d.{k}": v.detach().cpu() for k, v in dec.state_dict().items()}}


@torch.no_grad()
def fit_monitor(m, dec, F_: Fold, s: int, device) -> dict:
    """Training inputs only: CE, accuracy and L_HOG of updates 0-3's batches in eval mode."""
    was = m.training; m.eval(); dec.eval(); ce, hog, acc = [], [], []
    for b in (Batches(F_, s, range(4))[i] for i in range(4)):
        u, _, u8, mk, y = on_device(b, device); rgb = tm.to_rgb(u8); z = m(rgb, mk); sel = torch.from_numpy(mf.mask_blocks(s, F_.f, u, PER)).to(device)
        ce.append(float(F.cross_entropy(z, y))); acc.append(float((z.argmax(1) == y).float().mean())); hog.append(float(mf.masked_loss(m, dec, rgb, mk, mf.hog_targets(u8), sel)))
    m.train(was); dec.train(was); return {"ce": float(np.mean(ce)), "accuracy": float(np.mean(acc)), "hog": float(np.mean(hog)), "batches": "updates 0-3 (eval)"}


def drift(m, s: int) -> dict:
    a08 = torch.load(ROOT / f"models_frozen/video_within_view_v1/all_s{s}/final.pt", map_location="cpu", weights_only=True); cur = {k: v.detach().cpu() for k, v in m.state_dict().items()}; out = {}
    for name, pre in (("encoder_trunk", "encoder."), ("encoder_head", "encoder.head."), ("marker_stem", "marker_stem."), ("classifier", "classifier.")):
        ks = [k for k in cur if k.startswith(pre) and cur[k].is_floating_point() and not (name == "encoder_trunk" and k.startswith("encoder.head."))]
        d = sum(float((cur[k].double() - a08[k].double()).pow(2).sum()) for k in ks) ** 0.5; n0 = sum(float(a08[k].double().pow(2).sum()) for k in ks) ** 0.5
        out[name] = {"l2": d, "relative": d / max(n0, 1e-12), "max_abs": float(max((cur[k] - a08[k]).abs().max() for k in ks))}
    return out


# ----------------------------------------------------------------------------------------------- lambda (fit only)
def calibrate(ctx: Ctx, s: int, F_: Fold, device) -> dict:
    m, dec = build(ctx, s, F_.f, device); d0 = tet.state_digest(m); dd0 = tet.state_digest(dec); cpu0, cuda0 = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    torch.manual_seed(model_seed(s, F_.f)); trunk = [p for _, p in mf.shared_trunk(m)]; per = []
    for b in (Batches(F_, s, range(4))[i] for i in range(4)):
        u, _, u8, mk, y = on_device(b, device); sel = torch.from_numpy(mf.mask_blocks(s, F_.f, u, PER)).to(device); target = mf.hog_targets(u8); rgb = tm.to_rgb(u8)
        m.zero_grad(set_to_none=True); dec.zero_grad(set_to_none=True); ce = F.cross_entropy(m(rgb, mk), y); ce.backward(); gce = mf.grad_norm(trunk)
        m.zero_grad(set_to_none=True); dec.zero_grad(set_to_none=True); lh = mf.masked_loss(m, dec, rgb, mk, target, sel); lh.backward(); ghog = mf.grad_norm(trunk)
        per.append({"u": u, "ce": float(ce.detach()), "hog": float(lh.detach()), "trunk_grad_ce": gce, "trunk_grad_hog": ghog, "ratio": gce / ghog if ghog > 0 else float("nan")})
    m.zero_grad(set_to_none=True); dec.zero_grad(set_to_none=True); torch.set_rng_state(cpu0); torch.cuda.set_rng_state_all(cuda0)
    ratios = np.asarray([p["ratio"] for p in per]); lam = 0.25 * float(np.median(ratios)); vals = [*ratios.tolist(), lam, *(p["trunk_grad_ce"] for p in per), *(p["trunk_grad_hog"] for p in per)]
    rec = {"batches": per, "median_ratio": float(np.median(ratios)), "lambda": lam, "all_finite_nonzero": bool(all(np.isfinite(v) and v != 0 for v in vals)),
           "parameters_unchanged": tet.state_digest(m) == d0 and tet.state_digest(dec) == dd0, "rng_restored": bool(torch.equal(torch.get_rng_state(), cpu0) and all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda0))),
           "rows_sha256_updates_0_3": rows_sha(F_, s, range(4)), "masks_sha256_updates_0_3": masks_sha(s, F_.f, range(4)),
           "shared_trunk": {"tensors": len(trunk), "parameters": int(sum(p.numel() for p in trunk)), "rule": "encoder.* except encoder.head.*, plus marker_stem.weight; None gradient = 0"}}
    del m, dec; torch.cuda.empty_cache(); return rec


def lambda_stage(ctx: Ctx, s: int, device) -> str:
    rec = {"seed": s, "config_sha256": sha256_file(ROOT / ctx.config), "rule": "lambda = 0.25 x median(||g_CE|| / ||g_HOG||) over updates 0-3, train mode, model RNG as in training, no step", "folds": {}}
    for f in FOLDS:
        rec["folds"][str(f)] = calibrate(ctx, s, Fold(ctx, f), device)
        if not (rec["folds"][str(f)]["all_finite_nonzero"] and rec["folds"][str(f)]["parameters_unchanged"] and rec["folds"][str(f)]["rng_restored"]):
            raise SystemExit(f"fold {f}: lambda not finite / nonzero, parameters changed or RNG not restored: engineering stop")
    rec["loaded_code_sha256"] = t38.loaded_code(); rec["written"] = now(); publish(ctx, f"lambda_s{s}.json", rec)
    return "; ".join(f"f{f} lambda {rec['folds'][str(f)]['lambda']:.4g}" for f in FOLDS)


def lambdas(ctx: Ctx, s: int, sha: str) -> dict:
    p = ctx.results / f"lambda_s{s}.json"
    if sha256_file(p) != sha or sha256_file(ctx.out / f"lambda_s{s}.json") != sha:
        raise SystemExit("the lambda record differs from the committed one")
    return {int(f): v["lambda"] for f, v in json.loads(p.read_text())["folds"].items()}


# ----------------------------------------------------------------------------------------------- train
def identity(ctx: Ctx, arm: str, s: int, f: int, lam_sha: str) -> str:
    h = hashlib.sha256()
    for p in (Path(mf.__file__), Path(tm.__file__), Path(__file__)):
        h.update(p.read_bytes())
    sp = json.loads((ctx.a31.out / "split.json").read_text())
    h.update(json.dumps({"arm": arm, "seed": s, "fold": f, "a08": t31.A08[s], "pools": sp["pool_sha256"], "lambda_sha256": lam_sha, "config": sha256_file(ROOT / ctx.config)}, sort_keys=True).encode()); return h.hexdigest()


def train_one(ctx: Ctx, arm: str, s: int, F_: Fold, lam: float, ident: str, device, workers: int) -> dict:
    key = f"{arm}_s{s}_f{F_.f}"; rd = ctx.out / "train" / key; rd.mkdir(parents=True, exist_ok=True); rp = rd / "resume.pt"; coef = lam if arm == "H" else 0.0
    m, dec = build(ctx, s, F_.f, device); opt = make_opt(m, dec); before = fit_monitor(m, dec, F_, s, device); torch.manual_seed(model_seed(s, F_.f)); start, log, resumes = 0, [], []
    if rp.exists():
        st = torch.load(rp, map_location="cpu", weights_only=False)
        if st["identity"] != ident:
            raise SystemExit(f"{key}: the resume file belongs to another identity")
        load_state(m, dec, opt, st); start, log, before = st["update"], st["log"], st["fit_before"]; resumes.append({"from_update": start, "at": now()})
    prog = ctx.out / f"progress_s{s}.json"; t0 = time.time()

    def hook(u: int, rec: dict) -> None:
        if (u + 1) % SPLIT_EVERY == 0:
            last = log[-SPLIT_EVERY:]; print(f"[{key}] update {u + 1}/{U}: CE {np.mean([x['ce'] for x in last]):.4f} HOG {np.mean([x['hog'] for x in last]):.4f} grad {np.mean([x['grad_norm'] for x in last]):.3f}  {time.time() - t0:.0f} s", flush=True)
            write_json(prog, {"seed": s, "model": key, "update": u + 1, "total": U, "updated_at": now(), "pid": os.getpid(), "device": torch.cuda.get_device_name(0)})
        if (u + 1) % SAVE_EVERY == 0 and u + 1 < U:
            tmp = rp.with_suffix(".tmp"); torch.save({**state_of(m, dec, opt), "update": u + 1, "log": log, "identity": ident, "fit_before": before}, tmp); tmp.replace(rp)

    gsec = run_updates(m, dec, opt, F_, s, range(start, U), coef, device, workers, log, SPLIT_EVERY, hook)
    steps = [int(st_["step"]) for st_ in opt.state_dict()["state"].values()]
    if len(log) != U or [x["u"] for x in log] != list(range(U)) or not steps or set(steps) != {U}:
        raise SystemExit(f"{key}: {len(log)} logged updates, optimiser steps {sorted(set(steps))}")
    after = fit_monitor(m, dec, F_, s, device); dr = drift(m, s)
    torch.save({k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, rd / "final.pt"); torch.save({k: v.detach().cpu().clone() for k, v in dec.state_dict().items()}, rd / "decoder.pt")
    (rd / "updates.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log)); sp = [x for x in log if "trunk_grad_ce" in x]
    man = {"arm": arm, "seed": s, "fit_fold": F_.f, "coefficient": coef, "lambda": lam, "updates": len(log), "optimizer_steps": U, "identity": ident, "final_sha256": sha256_file(rd / "final.pt"),
           "decoder_sha256": sha256_file(rd / "decoder.pt"), "updates_sha256": sha256_file(rd / "updates.jsonl"), "rows_sha256": rows_sha(F_, s, range(U)), "masks_sha256": masks_sha(s, F_.f, range(U)),
           "fit_before": before, "fit_after": after, "drift_from_A08": dr, "resumes": resumes, "gpu_seconds_this_process": gsec, "seconds_this_process": time.time() - t0,
           "loss": {k: {"first_90": float(np.mean([x[k] for x in log[:WARM]])), "last_90": float(np.mean([x[k] for x in log[-WARM:]]))} for k in ("ce", "hog", "grad_norm")},
           "grad_split_every_50": [{k: x[k] for k in ("u", "trunk_grad_ce", "trunk_grad_coef_hog")} for x in sp], "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
    write_json(rd / "run_manifest.json", man); return man


def train(ctx: Ctx, s: int, lam_sha: str, device, workers: int) -> None:
    lam = lambdas(ctx, s, lam_sha)
    for f in FOLDS:
        F_ = Fold(ctx, f)
        for arm in ARMS:
            key = f"{arm}_s{s}_f{f}"; rd = ctx.out / "train" / key; ident = identity(ctx, arm, s, f, lam_sha); mp = rd / "run_manifest.json"
            if mp.exists() and json.loads(mp.read_text())["identity"] == ident and sha256_file(rd / "final.pt") == json.loads(mp.read_text())["final_sha256"]:
                print(f"[{key}] finished earlier", flush=True); continue
            t0, started, ok, note = time.time(), now(), False, ""
            try:
                man = train_one(ctx, arm, s, F_, lam[f], ident, device, workers); torch.cuda.empty_cache(); ok = True; note = f"final {man['final_sha256'][:12]}"
                if sha256_file(rd / "final.pt") == man["final_sha256"] and (rd / "resume.pt").exists():
                    (rd / "resume.pt").unlink()                                  # regenerable; the final weights are verified
                print(json.dumps({k: man[k] for k in ("arm", "seed", "fit_fold", "updates", "coefficient", "final_sha256", "fit_before", "fit_after")}), flush=True)
            except BaseException as e:  # noqa: BLE001 - recorded, then raised again
                note = f"{type(e).__name__}: {str(e)[:200]}"; raise
            finally:
                ledger_append(ctx.ledger, f"train {key} (GPU job wall time)", "gpu", started, time.time() - t0, ok, note)
    man = {f"{arm}_s{s}_f{f}": json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text()) for f in FOLDS for arm in ARMS}
    publish(ctx, f"weights_s{s}.json", {"seed": s, "models": man, "lambda_sha256": lam_sha, "config_sha256": sha256_file(ROOT / ctx.config), "loaded_code_sha256": t38.loaded_code(), "written": now()})


def freeze(ctx: Ctx, s: int, commit: str, manifest_sha: str) -> str:
    p = ctx.results / f"weights_s{s}.json"
    if sha256_file(p) != manifest_sha or sha256_file(ctx.out / f"weights_s{s}.json") != manifest_sha:
        raise SystemExit("the committed weight manifest differs from the server's")
    models = json.loads(p.read_text())["models"]; com = ctx.out / "weights_committed.json"; cur = json.loads(com.read_text()) if com.exists() else {}
    for k, man in models.items():
        if sha256_file(ctx.out / "train" / k / "final.pt") != man["final_sha256"] or sha256_file(ctx.out / "train" / k / "decoder.pt") != man["decoder_sha256"]:
            raise SystemExit(f"{k}: weights differ from the manifest")
        cur[k] = man["final_sha256"]
    cur[f"commit_s{s}"] = commit; write_json(com, cur); return f"{len(models)} models pinned at {commit}"


# ----------------------------------------------------------------------------------------------- check
def group_checks(ctx: Ctx, s: int) -> dict:
    out = {}
    for f in FOLDS:
        F_ = Fold(ctx, f); grid_ok = True
        for i, q in enumerate(F_.seqs):
            w = vsf.windows(len(q["refs"])); r = F_.G.rows[i]
            grid_ok &= bool(np.array_equal(w["centres"], F_.W["centre"][r]) and np.array_equal(w["samples"], F_.W["s"][r]) and np.array_equal(w["markers"], F_.W["marker"][r]))
        valid, lab = True, Counter()
        for u in range(U):
            rows = F_.G.draw(s, f, u); sq = F_.W["seq"][rows]; c = F_.W["centre"][rows]; y = F_.W["label"][rows]; lab.update(y.tolist())
            nat, con = F_.seqs[int(sq[0])], F_.seqs[int(sq[4])]
            valid &= bool(len(set(sq[:4].tolist())) == 1 and len(set(sq[4:].tolist())) == 1 and nat["kind"] == "native" and con["kind"] == "multi" and len(set(y[4:].tolist())) > 1
                          and np.all(np.diff(c[4:]) > 0) and np.all(np.diff(c[:4]) >= 0) and np.array_equal(y, np.asarray([F_.seqs[int(a)]["labels"][int(b)] for a, b in zip(sq, c)])))
        out[str(f)] = {"grid_equals_A07_windows": grid_ok, "native_sequences": len(F_.G.nat), "eligible_streams": len(F_.G.con), "eligible_starts": int(sum(len(x[1]) for x in F_.G.con)),
                       "short_natives_repeating": int(sum(1 for _, st in F_.G.nat if len(F_.G.rows[_]) < 4)), "draws_valid_900": valid, "label_counts_900x8": {str(k): int(v) for k, v in sorted(lab.items())},
                       "draw_repeat_equal": rows_sha(F_, s, range(U)) == rows_sha(F_, s, range(U)), "rows_sha256": rows_sha(F_, s, range(U)), "masks_sha256": masks_sha(s, f, range(U))}
    return out


def hog_checks(u8: torch.Tensor, device) -> dict:
    t = mf.hog_targets(u8.to(device)); ref = [mf.reference_block_hog(u8[w, 2 * k].numpy().astype(np.float64).mean(-1) / 255.0) for w, k in ((0, 0), (3, 5), (7, 7))]
    vec = [t[w, k].double().cpu().numpy() for w, k in ((0, 0), (3, 5), (7, 7))]; sep = mf.block_features(mf.cell_histograms(mf.gray_of(u8[5, 6].to(device))[None]))[0]
    const = torch.full((1, 16, 224, 224, 3), 117, dtype=torch.uint8, device=device); ramp = torch.arange(224, device=device).float().mul(255 / 223).round().to(torch.uint8)
    hz = ramp[None, None, None, :, None].expand(1, 16, 224, 224, 3).contiguous(); vt = ramp[None, None, :, None, None].expand(1, 16, 224, 224, 3).contiguous()
    th, tv = mf.hog_targets(hz).reshape(8, 49, 16, 9), mf.hog_targets(vt).reshape(8, 49, 16, 9); cn = mf.cell_histograms(mf.gray_of(u8[:, 0].to(device))).pow(2).sum(-1).sqrt()
    return {"vectorised_vs_pixel_loop_max_abs": float(max(np.abs(a - b).max() for a, b in zip(vec, ref))), "slot_2t_alignment": bool(torch.equal(t[5, 3], sep)), "target_requires_grad": bool(t.requires_grad), "target_grad_fn": t.grad_fn is not None,
            "constant_image_zero": bool((mf.hog_targets(const) == 0).all()), "horizontal_ramp_only_bins_0_8": bool((th[..., 1:8].abs() < 1e-6).all() and (th[..., 0] > 0.5).all()),
            "vertical_ramp_only_bin_4": bool((tv[..., [0, 1, 2, 3, 5, 6, 7, 8]].abs() < 1e-3).all() and (tv[..., 4] > 0.99).all()), "cell_norm_max": float(cn.max()), "cell_norm_share_above_0.99": float((cn > 0.99).float().mean()), "cell_zero_share": float((cn == 0).float().mean()),
            "finite": bool(torch.isfinite(t).all()), "shape": list(t.shape), "mean_target_value": float(t.mean())}


def mask_checks(rgb: torch.Tensor, s: int, f: int, u: int) -> dict:
    sel_np = mf.mask_blocks(s, f, u, PER); sel = torch.from_numpy(sel_np).to(rgb.device); out = mf.masked_rgb(rgb, sel); cover, keep = True, True
    for w in range(PER):
        exp = np.zeros(49, bool); exp[np.random.default_rng([20261026, 40, s, f, u, w]).choice(49, 25, replace=False)] = True; cover &= bool(np.array_equal(exp, sel_np[w]))
        for k in range(49):
            y, x = divmod(k, 7); blk, org = out[w, :, :, 32 * y:32 * y + 32, 32 * x:32 * x + 32], rgb[w, :, :, 32 * y:32 * y + 32, 32 * x:32 * x + 32]
            cover &= bool((blk == 0).all()) if sel_np[w, k] else True; keep &= bool(torch.equal(blk, org)) if not sel_np[w, k] else True
    return {"draw_equals_explicit_rng": cover, "selected_blocks_zero_all_slots_channels": cover, "unselected_unchanged": keep, "per_window_masked": sel_np.sum(1).tolist(), "input_not_modified": out.data_ptr() != rgb.data_ptr()}


def token_checks(a08, rgb: torch.Tensor, mk: torch.Tensor) -> dict:
    enc = a08.encoder; seen = {}; h2 = enc.blocks[0].register_forward_pre_hook(lambda _m, a: seen.__setitem__("b0", (a[0].detach().clone(), a[1])))   # embed refuses a conv_proj hook
    try:
        with torch.no_grad():
            e0 = a08.embed(rgb, mk); q0 = a08.classifier(e0); conv = enc.conv_proj(rgb); exp = (conv + a08.marker_stem(mk.to(conv.dtype))).flatten(2).transpose(1, 2)
    finally:
        h2.remove()
    e, u = mf.embed_with_tokens_grad(a08, rgb, mk); g = mf.token_grid(u)
    loop = max(float((g[:, t, y * 7 + x] - u[:, 1 + t * 49 + y * 7 + x]).abs().max()) for t in range(8) for y in range(7) for x in range(7))
    err = False
    try:
        mf.embed_with_tokens_grad(a08, rgb, mk[:, :, :8])
    except ValueError:
        err = True
    return {"first_stage_tokens_equal_conv_flatten_THW": float((seen["b0"][0][:, 1:] - exp).abs().max()), "first_stage_thw": list(seen["b0"][1]), "final_grid_from_network": list(tp.token_grid(enc)[0]),
            "token_grid_vs_loop_max_abs": loop, "U_shape": list(u.shape), "U_keeps_autograd": bool(u.requires_grad and u.grad_fn is not None), "e_equals_plain_embed": bool(torch.equal(e.detach(), e0)),
            "q_unchanged_by_hook": bool(torch.equal(a08.classifier(e).detach(), q0)), "hook_removed_after_error": err and not enc.norm._forward_hooks and not enc.conv_proj._forward_hooks,
            "no_hook_left": not enc.norm._forward_hooks and not enc.conv_proj._forward_hooks and not enc.blocks[0]._forward_pre_hooks}


def grads_of(m, dec) -> dict:
    return {**{f"m.{n}": p.grad for n, p in m.named_parameters()}, **{f"d.{n}": p.grad for n, p in dec.named_parameters()}}


def path_checks(ctx: Ctx, s: int, F_: Fold, device) -> dict:
    """Eval mode (dropout / stochastic depth off): aux-only and CE-only gradient paths, accumulation = joint, 4 + 4 microbatch."""
    m, dec = build(ctx, s, F_.f, device); m.eval(); dec.eval(); b = Batches(F_, s, [7])[0]; u, _, u8, mk, y = on_device(b, device); sel = torch.from_numpy(mf.mask_blocks(s, F_.f, u, PER)).to(device)
    target = mf.hog_targets(u8); rgb = tm.to_rgb(u8); out = {}
    mf.masked_loss(m, dec, rgb, mk, target, sel).backward(); g = grads_of(m, dec)
    zero = sorted(k for k, v in g.items() if v is None or float(v.abs().sum()) == 0); nz = lambda pre: any(v is not None and float(v.abs().sum()) > 0 for k, v in g.items() if k.startswith(pre))  # noqa: E731
    out["aux_only"] = {"finite": all(v is None or bool(torch.isfinite(v).all()) for v in g.values()), "marker_stem": nz("m.marker_stem."), "conv_proj_patch_stem": nz("m.encoder.conv_proj."), "block_0": nz("m.encoder.blocks.0."),
                       "block_15": nz("m.encoder.blocks.15."), "decoder": nz("d."), "zero_gradient_tensors": zero, "A30_structural_four_zero": {k: f"m.{k}" in zero for k in A30_ZERO}}
    m.zero_grad(set_to_none=True); dec.zero_grad(set_to_none=True); F.cross_entropy(m(rgb, mk), y).backward(); g = grads_of(m, dec)
    out["ce_only"] = {"zero_gradient_tensors": sorted(k for k, v in g.items() if v is None or float(v.abs().sum()) == 0)}
    m.zero_grad(set_to_none=True); dec.zero_grad(set_to_none=True); h = 4; c = 0.7
    F.cross_entropy(m(rgb[:h], mk[:h]), y[:h]).backward(); (c * mf.masked_loss(m, dec, rgb[:h], mk[:h], target[:h], sel[:h])).backward(); ga = {k: v.clone() for k, v in grads_of(m, dec).items() if v is not None}
    m.zero_grad(set_to_none=True); dec.zero_grad(set_to_none=True)
    (F.cross_entropy(m(rgb[:h], mk[:h]), y[:h]) + c * mf.masked_loss(m, dec, rgb[:h], mk[:h], target[:h], sel[:h])).backward(); gj = {k: v.clone() for k, v in grads_of(m, dec).items() if v is not None}
    gmax = max(float(v.abs().max()) for v in gj.values()); out["accumulated_vs_joint"] = {"max_abs": float(max((ga[k] - gj[k]).abs().max() for k in gj)), "grad_max": gmax, "same_tensors": sorted(ga) == sorted(gj)}
    m.zero_grad(set_to_none=True); dec.zero_grad(set_to_none=True); F.cross_entropy(m(rgb, mk), y).backward(); (c * mf.masked_loss(m, dec, rgb, mk, target, sel)).backward(); gf = {k: v.clone() for k, v in grads_of(m, dec).items() if v is not None}
    m.zero_grad(set_to_none=True); dec.zero_grad(set_to_none=True)
    for a in (0, h):
        sl = slice(a, a + h); (0.5 * F.cross_entropy(m(rgb[sl], mk[sl]), y[sl])).backward(); (0.5 * c * mf.masked_loss(m, dec, rgb[sl], mk[sl], target[sl], sel[sl])).backward()
    gm = {k: v.clone() for k, v in grads_of(m, dec).items() if v is not None}; gmax = max(float(v.abs().max()) for v in gf.values())
    out["microbatch_4_plus_4"] = {"max_abs": float(max((gf[k] - gm[k]).abs().max() for k in gf)), "grad_max": gmax}
    del m, dec; torch.cuda.empty_cache(); return out


def mech_run(ctx: Ctx, s: int, F_: Fold, arm: str, coef: float, us, device, workers: int, state: dict | None = None):
    m, dec = build(ctx, s, F_.f, device); opt = make_opt(m, dec); torch.manual_seed(model_seed(s, F_.f))
    if state is not None:
        load_state(m, dec, opt, state)
    log = []; g = run_updates(m, dec, opt, F_, s, us, coef if arm == "H" else 0.0, device, workers, log); return m, dec, opt, log, g


def training_checks(ctx: Ctx, s: int, F_: Fold, device, workers: int) -> dict:
    us = range(85, 91); res = {}
    m, dec = build(ctx, s, F_.f, device); opt = make_opt(m, dec); m2, dec2 = build(ctx, s, F_.f, device)
    ids = sorted(id(p) for g in opt.param_groups for p in g["params"]); grp = {g["name"]: {id(p) for p in g["params"]} for g in opt.param_groups}
    res["binding"] = {"all_trainable": all(p.requires_grad for p in all_params(m, dec)), "binds_exactly": ids == sorted(id(p) for p in all_params(m, dec)) and len(ids) == len(set(ids)),
                      "encoder_group_is_encoder_incl_projection": grp["encoder"] == {id(p) for p in m.encoder.parameters()}, "train_mode": m.training and dec.training and m.encoder.head.training,
                      "parameters": {"model": int(sum(p.numel() for p in m.parameters())), "decoder": int(sum(p.numel() for p in dec.parameters()))}, "lr": {g["name"]: g["base"] for g in opt.param_groups},
                      "weight_decay": [g["weight_decay"] for g in opt.param_groups], "betas_eps": [list(opt.defaults["betas"]), opt.defaults["eps"]]}
    res["identical_arm_start"] = {"model": tet.state_digest(m) == tet.state_digest(m2), "decoder": tet.state_digest(dec) == tet.state_digest(dec2), "decoder_nonzero": bool(dec.weight.abs().sum() > 0)}
    del m, dec, opt, m2, dec2; torch.cuda.empty_cache()
    ra = mech_run(ctx, s, F_, "R", 0.0, us, device, workers); rb = mech_run(ctx, s, F_, "R", 0.0, us, device, workers); hz = mech_run(ctx, s, F_, "H", 0.0, us, device, workers)
    L = lambda a, b: max(max(abs(x["ce"] - y["ce"]), abs(x["hog"] - y["hog"])) for x, y in zip(a[3], b[3]))  # noqa: E731
    P = lambda a, b: params_max_diff(flat_state(a[0], a[1]), flat_state(b[0], b[1]))  # noqa: E731
    res["repeat_6_R"] = {"loss_max_abs": L(ra, rb), "params_max_abs": P(ra, rb)}; res["forced_lambda0_H_vs_R"] = {"loss_max_abs": L(hz, ra), "params_max_abs": P(hz, ra)}
    res["real_update"] = {"params_moved_max_abs": params_max_diff(flat_state(ra[0], ra[1]), flat_state(*build(ctx, s, F_.f, device))), "optimizer_steps": sorted({int(v["step"]) for v in ra[2].state_dict()["state"].values()})}
    del ra, rb, hz; torch.cuda.empty_cache()
    h6 = mech_run(ctx, s, F_, "H", 1.0, us, device, workers); h3 = mech_run(ctx, s, F_, "H", 1.0, range(85, 88), device, workers); p = ctx.out / "check_resume.pt"; torch.save(state_of(h3[0], h3[1], h3[2]), p); log3 = h3[3]; del h3
    st = torch.load(p, map_location="cpu", weights_only=False); hr = mech_run(ctx, s, F_, "H", 1.0, range(88, 91), device, workers, st); p.unlink()
    res["resume_3_plus_3_H_coef1"] = {"loss_max_abs": max(max(abs(x["ce"] - y["ce"]), abs(x["hog"] - y["hog"])) for x, y in zip(h6[3], log3 + hr[3])), "params_max_abs": P(h6, hr)}
    res["timing"] = {"seconds_per_update_gpu_H": h6[4] / len(us), "updates": len(us)}; del h6, hr; torch.cuda.empty_cache()
    t0 = time.time(); m, dec, opt, log, gsec = mech_run(ctx, s, F_, "H", 1.0, range(200, 230), device, workers); wall = time.time() - t0
    res["timing"].update({"seconds_per_update_gpu_H_30": gsec / 30, "seconds_per_update_wall_H_30_incl_build_and_loader": wall / 30, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}); del m, dec, opt
    return res


def check(ctx: Ctx, device, workers: int) -> str:
    s = 0; t_start = time.time(); res = {"config_sha256": sha256_file(ROOT / ctx.config), "tolerances": TOL, "limits": LIMITS, "contract": mf.CONTRACT, "numerics": {"tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
           "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark, "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}}
    sp = json.loads((ctx.a31.out / "split.json").read_text()); res["isolation"] = sp["isolation"]; res["pool_sha256"] = sp["pool_sha256"]; res["groups"] = group_checks(ctx, s)
    F0 = Fold(ctx, 0); b = Batches(F0, s, [0])[0]; u, rows, u8, mk_slots, y = b; rows = rows.numpy()
    fr, idx = t37.chunk_frames(F0.seqs, {"seq": F0.W["seq"][rows]}, F0.W["s"][rows], F0.fp, 0); res["pixels_equal_A31_path"] = bool(np.array_equal(fr[idx], u8.numpy()))
    b2 = Batches(F0, s, [0])[0]; res["identical_arm_pixels_masks"] = bool(torch.equal(b2[2], u8) and np.array_equal(mf.mask_blocks(s, 0, 0, PER), mf.mask_blocks(s, 0, 0, PER)))
    res["hog"] = hog_checks(u8, device); a08 = ctx.a31.a08(s, device).eval(); d0 = tet.state_digest(a08); u8d = u8.to(device); rgb, mk = tm.to_rgb(u8d), tm.marker_planes(mk_slots.to(device))
    mk0 = mk.clone(); res["mask"] = mask_checks(rgb, s, 0, 0); res["tokens"] = token_checks(a08, rgb, mk); res["marker_untouched_by_mask"] = bool(torch.equal(mk, mk0))
    O = t31.outputs(ctx.a31, s, 0)
    with torch.no_grad():
        qb = a08(rgb, mk).float().cpu().numpy()
    pick = np.sort(np.random.default_rng([20261026, 40, 1]).choice(len(F0.W["seq"]), 32, replace=False)); fr, idx = t37.chunk_frames(F0.seqs, {"seq": F0.W["seq"][pick]}, F0.W["s"][pick], F0.fp, 0)
    with torch.no_grad():
        e32 = a08.embed(tm.to_rgb(torch.from_numpy(fr[idx]).to(device)), tm.marker_planes(torch.from_numpy(F0.W["marker"][pick]).long().to(device))); q32 = a08.classifier(e32).float().cpu().numpy()
    res["init_vs_A31_cache"] = {"training_batch_logits_max_abs": float(np.abs(qb - O["logits"][rows]).max()), "random_32_logits_max_abs": float(np.abs(q32 - O["logits"][pick]).max()),
                                "random_32_emb_max_abs": float(np.abs(e32.float().cpu().numpy() - O["emb"][pick]).max())}
    res["a08_unchanged"] = tet.state_digest(a08) == d0; del a08; torch.cuda.empty_cache()
    res["init_equals_base"] = init_base(ctx, s); res["paths"] = path_checks(ctx, s, F0, device); res["training"] = training_checks(ctx, s, F0, device, workers)
    # launch projection: 4 x 900 updates at the measured rate, lambda (2 x 4 batches), the readout (2 models on every window) and the reserve
    spu = max(res["training"]["timing"]["seconds_per_update_gpu_H_30"], res["training"]["timing"]["seconds_per_update_wall_H_30_incl_build_and_loader"]); tm_ = readout_timing(ctx, s, device, workers); nw = sum(len(t31.pool_W(t31.load_pool(ctx.a31, f))["seq"]) for f in FOLDS)
    spent = t31.vsh.ledger_hours(ctx.ledger, "gpu") + (time.time() - t_start) / 3600; train_h = 4 * U * spu * 1.15 / 3600; lam_h = 0.05; read_h = nw * tm_["seconds_per_window_two_models"] * 1.15 / 3600
    proj = spent + train_h + lam_h + read_h + 0.6; res["projection"] = {"spent_gpu_h_incl_this_check": spent, "train_4x900_h": train_h, "lambda_h": lam_h, "readout_h": read_h, "reserve_h": 0.6, "source0_total_h": proj,
                                                                       "source2_if_triggered_h": train_h + lam_h + read_h, "cap_h": ctx.v["budget"]["gpu_hours_max"], "windows": nw, **tm_}
    res["disk"] = {"free_gb": shutil.disk_usage(ctx.out).free / 1e9, "per_model_final_gb": 4 * res["training"]["binding"]["parameters"]["model"] / 1e9, "resume_gb": 12 * res["training"]["binding"]["parameters"]["model"] / 1e9}
    res["loaded_code_sha256"] = t38.loaded_code(); res["ok"] = check_ok(res); res["finished"] = now(); publish(ctx, "checks.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("loaded_code_sha256", "groups")}, indent=1, default=str)[:9000], flush=True)
    if not res["ok"]:
        raise SystemExit("a check failed")
    return f"{spu:.2f} s per update; source-0 projection {proj:.2f} GPU h"


def readout_timing(ctx: Ctx, s: int, device, workers: int) -> dict:
    F_ = Fold(ctx, 0); rows = np.flatnonzero(F_.W["seq"] < 30); t0 = time.time(); fr, idx = t37.chunk_frames(F_.seqs, {"seq": F_.W["seq"][rows]}, F_.W["s"][rows], F_.fp, workers); td = time.time() - t0
    a, b_ = ctx.a31.a08(s, device).eval(), ctx.a31.a08(s, device).eval(); g = 0.0
    for b in range(0, len(rows), BATCH):
        ks = np.arange(b, min(b + BATCH, len(rows))); torch.cuda.synchronize(); t1 = time.time(); [t37.embed(x, fr, idx[ks], F_.W["marker"][rows[ks]], device) for x in (a, b_)]; torch.cuda.synchronize(); g += time.time() - t1
    return {"seconds_per_window_two_models": (g + td) / len(rows), "gpu_ms_per_window_one_model": 1e3 * g / 2 / len(rows), "decode_ms_per_window": 1e3 * td / len(rows)}


def init_base(ctx: Ctx, s: int) -> dict:
    """Init (the A08 copy) gives q; Init = Base on every P run, complete sequence and A32 Base row."""
    X = full_q(ctx, s); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); old = json.loads((ROOT / "docs/results/2026-10-25_a5c_source_external_evidence/readout.json").read_text())["sequences"]; out = {}
    for h in FOLDS:
        D = d32.fold_data(ctx.d, s, h); pk = D["pk"]; off = X["offset"][h]; Wf = t31.pool_W(D["seqs"]); pp = t29.pooled_probs(X["q"], pk, np.arange(len(pk["labels"])))
        ps = rr.per_sample(Wf, X["q"][off:off + len(Wf["seq"])].numpy(), D["seqs"]); preds = {q["id"]: ir.relabel(D["base_preds"][q["id"]], ps[q["id"]]) for q in D["seqs"]}
        P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; sr = t31.seq_rows(a8.cfg, P, preds, bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{h}")); ob = old[f"s{s}_heldout_f{h}"]["A08 Base"]
        out[str(h)] = {"runs_equal_base": bool(np.array_equal(pp.argmax(1), pk["base"])), "sequences_equal_base": all(np.array_equal(preds[q["id"]], D["base_preds"][q["id"]]) for q in D["seqs"]),
                       "rows_equal_A32_base": bool(sr["row"] == ob["sequence_row"] and rsh.counts_of(D["seqs"], sr) == ob["counts"]), "P_runs": int(len(pk["labels"]))}
    return out


def check_ok(r: dict) -> bool:
    H, M, T, I, Pa, Tr, Pr = r["hog"], r["mask"], r["tokens"], r["init_vs_A31_cache"], r["paths"], r["training"], r["projection"]; G = r["groups"].values(); lim = lambda d: d["loss_max_abs"] <= LIMITS["loss"] and d["params_max_abs"] <= LIMITS["params"]  # noqa: E731
    return bool(not any(r["isolation"].values()) and all(g["grid_equals_A07_windows"] and g["draws_valid_900"] and g["draw_repeat_equal"] and g["eligible_streams"] > 0 for g in G)
                and not r["numerics"]["tf32_matmul"] and not r["numerics"]["tf32_cudnn"] and r["numerics"]["cudnn_deterministic"] and not r["numerics"]["cudnn_benchmark"] and r["numerics"]["deterministic_warn_only"]
                and r["pixels_equal_A31_path"] and r["identical_arm_pixels_masks"] and r["marker_untouched_by_mask"]
                and H["vectorised_vs_pixel_loop_max_abs"] <= TOL["hog_reference"] and H["slot_2t_alignment"] and not H["target_requires_grad"] and not H["target_grad_fn"] and H["constant_image_zero"]
                and H["horizontal_ramp_only_bins_0_8"] and H["vertical_ramp_only_bin_4"] and H["cell_norm_max"] <= 1 + 1e-5 and H["finite"]
                and M["draw_equals_explicit_rng"] and M["unselected_unchanged"] and all(x == 25 for x in M["per_window_masked"]) and M["input_not_modified"]
                and T["first_stage_tokens_equal_conv_flatten_THW"] <= TOL["exact_float"] and T["first_stage_thw"] == [8, 56, 56] and T["final_grid_from_network"] == list(mf.GRID) and T["token_grid_vs_loop_max_abs"] == 0
                and T["U_shape"][1:] == [mf.N_TOKENS, mf.DIM] and T["U_keeps_autograd"] and T["e_equals_plain_embed"] and T["q_unchanged_by_hook"] and T["hook_removed_after_error"] and T["no_hook_left"]
                and all(v <= TOL["parity"] for v in I.values()) and r["a08_unchanged"] and all(all(v[k] for k in ("runs_equal_base", "sequences_equal_base", "rows_equal_A32_base")) for v in r["init_equals_base"].values())
                and Pa["aux_only"]["finite"] and all(Pa["aux_only"][k] for k in ("marker_stem", "conv_proj_patch_stem", "block_0", "block_15", "decoder"))
                and Pa["accumulated_vs_joint"]["max_abs"] <= TOL["microbatch_rel"] * max(1.0, Pa["accumulated_vs_joint"]["grad_max"]) and Pa["accumulated_vs_joint"]["same_tensors"]
                and Pa["microbatch_4_plus_4"]["max_abs"] <= TOL["microbatch_rel"] * max(1.0, Pa["microbatch_4_plus_4"]["grad_max"])
                and Tr["binding"]["all_trainable"] and Tr["binding"]["binds_exactly"] and Tr["binding"]["encoder_group_is_encoder_incl_projection"] and Tr["binding"]["train_mode"] and all(Tr["identical_arm_start"].values())
                and lim(Tr["repeat_6_R"]) and lim(Tr["forced_lambda0_H_vs_R"]) and lim(Tr["resume_3_plus_3_H_coef1"]) and Tr["real_update"]["params_moved_max_abs"] > 0 and Tr["real_update"]["optimizer_steps"] == [6]
                and Pr["source0_total_h"] <= Pr["cap_h"] and r["disk"]["free_gb"] > 10)


# ----------------------------------------------------------------------------------------------- readout
def full_q(ctx: Ctx, s: int) -> dict:
    out, o = {"q": [], "offset": {}}, 0
    for f in FOLDS:
        O = t31.outputs(ctx.a31, s, f); out["q"].append(O["logits"]); out["offset"][f] = o; o += len(O["logits"])
        if int(O["offset"]) != out["offset"][f]:
            raise SystemExit("fold offsets differ from A31's")
    return {"q": torch.from_numpy(np.concatenate(out["q"])), "offset": out["offset"]}


def load_model(ctx: Ctx, arm: str, s: int, f: int, device):
    rd = ctx.out / "train" / f"{arm}_s{s}_f{f}"; man = json.loads((rd / "run_manifest.json").read_text()); com = json.loads((ctx.out / "weights_committed.json").read_text())
    if sha256_file(rd / "final.pt") != man["final_sha256"] or com.get(f"{arm}_s{s}_f{f}") != man["final_sha256"]:
        raise SystemExit(f"{arm}_s{s}_f{f}: weights not committed or changed")
    m = ctx.a31.a08(s, device); m.load_state_dict(torch.load(rd / "final.pt", map_location="cpu", weights_only=True), strict=True); return m.eval()


def heldout_logits(ctx: Ctx, s: int, h: int, models: dict, device, workers: int, acc: dict) -> dict:
    """Online forward of the fit-(1 - h) models on every original request window of held-out fold h (decoded once)."""
    seqs = t31.load_pool(ctx.a31, h); W = t31.pool_W(seqs); fp = t28.frame_paths(seqs); n = len(W["seq"]); Z = {a: np.zeros((n, 5), np.float32) for a in models}
    for c0 in range(0, len(seqs), 150):
        rows = np.flatnonzero((W["seq"] >= c0) & (W["seq"] < c0 + 150)); a = time.time(); fr, idx = t37.chunk_frames(seqs, {"seq": W["seq"][rows]}, W["s"][rows], fp, workers); acc["decode_seconds"] += time.time() - a
        for b in range(0, len(rows), BATCH):
            ks = np.arange(b, min(b + BATCH, len(rows))); r = rows[ks]
            for arm, m in models.items():
                torch.cuda.synchronize(); a = time.time(); Z[arm][r] = t37.embed(m, fr, idx[ks], W["marker"][r], device)[1].cpu().numpy(); torch.cuda.synchronize(); acc["gpu_seconds"] += time.time() - a
        del fr; print(f"seed {s} held-out fold {h}: {int(rows.max()) + 1}/{n} windows; GPU {acc['gpu_seconds']:.0f} s", flush=True)
    if not all(np.isfinite(z).all() for z in Z.values()):
        raise SystemExit("a held-out logit is not finite")
    return Z


def readout(ctx: Ctx, s: int, device, workers: int, dry: bool = False) -> str:
    name = f"readout_s{s}.json"
    if (ctx.out / name).exists() and not dry:
        raise SystemExit("this seed's readout exists; it runs once")
    X = full_q(ctx, s); a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a31.a15.a10.a07p); rec = {"seed": s, "cells": {}}; acc = {m_: [] for m_ in ("Base", *ARMS)}; ys, bs = [], []
    cov = {a: [] for a in ARMS}; s1recs = {a: [] for a in ARMS}; old = json.loads((ROOT / "docs/results/2026-10-25_a5c_source_external_evidence/readout.json").read_text())["sequences"]; tim = {"gpu_seconds": 0.0, "decode_seconds": 0.0}
    a39 = json.loads((ROOT / "docs/results/2026-10-26_local_token_a5c_evidence/readout_s0.json").read_text())["cells"] if s == 0 else None
    for h in FOLDS:
        f = 1 - h; D = d32.fold_data(ctx.d, s, h); D["M"] = d32.set_masks(D); off = X["offset"][h]; P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; ev = bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{h}")
        if dry:
            Z = {a: X["q"][off:off + len(t31.pool_W(D["seqs"])["seq"])].numpy() for a in ARMS}
        else:
            Z = heldout_logits(ctx, s, h, {a: load_model(ctx, a, s, f, device) for a in ARMS}, device, workers, tim); torch.cuda.empty_cache()
        cell = {}
        for arm in ARMS:
            z = X["q"].clone(); z[off:off + len(Z[arm])] = torch.from_numpy(Z[arm]); t = tables(z, D, off, a8.cfg, P, ev); p = t.pop("pred"); acc[arm].append(p); cell[arm] = t
            cell[arm]["coverage"], sets = t37.coverage(D, D["pk"]["labels"], D["pk"]["base"], p); cov[arm].append(sets); cell[arm]["boundaries"] = t38.boundaries(z, D, off)
            ex, recs = a5c_extras(D, p); cell[arm].update(ex); s1recs[arm].append(recs)
        t = tables(X["q"], D, off, a8.cfg, P, ev); p = t.pop("pred"); acc["Base"].append(p); cell["Base"] = t; cell["Base"]["boundaries"] = t38.boundaries(X["q"], D, off); cell["Base"].update(a5c_extras(D, p)[0])
        s1 = np.flatnonzero(D["M"]["S1"]); cell["S1_denominator"] = {"runs": int(len(s1)), **d32.concentration([(D, int(i)) for i in s1], "true_src", None)}
        ob = old[f"s{s}_heldout_f{h}"]["A08 Base"]; cell["Base_reproduced"] = {"A32_row": cell["Base"]["sequence_row"] == ob["sequence_row"] and cell["Base"]["counts"] == ob["counts"]}
        if a39 is not None:
            js = lambda x: json.loads(json.dumps(x, default=float))  # noqa: E731
            cell["Base_reproduced"]["A39_Base_cell"] = all(js(cell["Base"][k]) == a39[f"heldout_f{h}"]["Base"][k] for k in a39[f"heldout_f{h}"]["Base"] if k in cell["Base"])
        row = lambda m_: cell[m_]["sequence_row"]  # noqa: E731
        cell["comparisons"] = {f"{a} - {bb}": {"net_fixed": cell[a]["runs"]["all"]["net_fixed"] - (cell[bb]["runs"]["all"]["net_fixed"] if bb != "Base" else 0), "S1_fixed": cell[a]["S1_fixed"]["fixed"] - cell[bb]["S1_fixed"]["fixed"],
                                               "correct_A5C_kept": cell[a]["correct_A5C_kept"]["kept"] - cell[bb]["correct_A5C_kept"]["kept"], "correct_A4C_kept": cell[a]["correct_A4C_kept"]["kept"] - cell[bb]["correct_A4C_kept"]["kept"],
                                               **{k: row(a)[k] - row(bb)[k] for k in ("segment_f1_50_multi", "native_macro_f1", "boundary_recall_multi", "boundary_precision_multi")},
                                               **{k: cell[a]["counts"][k] - cell[bb]["counts"][k] for k in ("fp", "fp_short", "fn", "fn_short")}} for a, bb in (("H", "R"), ("H", "Base"), ("R", "Base"))}
        if not dry:
            mans = {arm: json.loads((ctx.out / "train" / f"{arm}_s{s}_f{f}" / "run_manifest.json").read_text()) for arm in ARMS}
            cell["fit_only"] = {arm: {k: mans[arm][k] for k in ("lambda", "coefficient", "fit_before", "fit_after", "drift_from_A08", "loss")} for arm in ARMS}
        rec["cells"][f"heldout_f{h}"] = cell; ys.append(D["pk"]["labels"]); bs.append(D["pk"]["base"])
    y, b = np.concatenate(ys), np.concatenate(bs); wr = b != y; rec["oof"] = {}
    for m_ in ARMS:
        p = np.concatenate(acc[m_]); fx, br = int((wr & (p == y)).sum()), int((~wr & (p != y)).sum()); un = {k: set().union(*[c[k] for c in cov[m_]]) for k in ("fixed", "broken")}
        rec["oof"][m_] = {"runs": int(len(y)), "fixed": fx, "broken": br, "net_fixed": fx - br, **{f"{k}_recordings": len(v) for k, v in un.items()}, **{f"{k}_sessions": len({session_of(x) for x in v}) for k, v in un.items()},
                          "S1_fixed_true_class_recordings": len(set().union(*s1recs[m_]))}
    rec["seed2_rule"], rec["values"], rec["summary"] = rule_of(rec, s1recs)
    if dry:
        print(json.dumps({"dry_run_init": True, "Base_reproduced": {k: v["Base_reproduced"] for k, v in rec["cells"].items()}, "summary": rec["summary"], "oof": rec["oof"]}, indent=1)); return "dry run (Init); nothing published"
    nw = sum(len(t31.pool_W(t31.load_pool(ctx.a31, f_))["seq"]) for f_ in FOLDS); pr = json.loads((ctx.a31.out / "cache" / f"prefix_s{s}.json").read_text())
    rec["deployment_cost"] = {"forwards": "Base's one 224 forward per window plus one forward of the new recognition branch (a second full A08-size encoder)", "parameters": "A08 34.6M (Base) + 34.6M (new branch); the decoder is training only",
                              "gpu_ms_per_window_new_branch": 1e3 * tim["gpu_seconds"] / (2 * nw), "cpu_decode_ms_per_window_shared": 1e3 * tim["decode_seconds"] / nw, "gpu_ms_per_window_A31_base_pass": 1e3 * pr["gpu_seconds"] / sum(pr["rows"].values()),
                              "readout_gpu_seconds": tim["gpu_seconds"], "readout_decode_seconds": tim["decode_seconds"], "note": "two models (R, H) per held-out window share one decode; the A31 pass also wrote its prefix cache"}
    rec["written"] = now(); publish(ctx, name, rec)
    print(json.dumps({"summary": rec["summary"], "oof": rec["oof"], "rule": rec["seed2_rule"], "values": rec["values"], "Base_reproduced": {k: v["Base_reproduced"] for k, v in rec["cells"].items()}}, indent=1)); return f"rule met: {rec['seed2_rule']['met']}"


def rule_of(rec: dict, s1recs: dict) -> tuple[dict, dict, dict]:
    C = lambda h: rec["cells"][f"heldout_f{h}"]  # noqa: E731
    tot = lambda m_, k, sub: sum(C(h)[m_][k][sub] for h in FOLDS)  # noqa: E731
    d_ = lambda a, bb, h, k: C(h)[a]["sequence_row"][k] - C(h)[bb]["sequence_row"][k]  # noqa: E731
    net = lambda a, h: C(h)[a]["runs"]["all"]["net_fixed"]  # noqa: E731
    n1, n5, n4 = tot("H", "S1_fixed", "runs"), tot("H", "correct_A5C_kept", "runs"), tot("H", "correct_A4C_kept", "runs"); s1H, s1R = tot("H", "S1_fixed", "fixed"), tot("R", "S1_fixed", "fixed")
    k5H, k5R, k4H, k4R = tot("H", "correct_A5C_kept", "kept"), tot("R", "correct_A5C_kept", "kept"), tot("H", "correct_A4C_kept", "kept"), tot("R", "correct_A4C_kept", "kept")
    recH = [len(r) for r in s1recs["H"]]; recH_all = len(set().union(*s1recs["H"]))
    rule = {"target_each_fold_H_minus_R_S1_ge_1": all(C(h)["H"]["S1_fixed"]["fixed"] - C(h)["R"]["S1_fixed"]["fixed"] >= 1 for h in FOLDS), "target_sum_H_minus_R_S1_ge_5": s1H - s1R >= 5,
            "S1_H": f"{s1H}/{n1}", "S1_R": f"{s1R}/{n1}", "S1_needed": ceil_pct(10, n1), "signal_S1_runs_ok": s1H >= ceil_pct(10, n1),
            "S1_H_true_class_recordings_by_fold": recH, "S1_H_true_class_recordings_total": recH_all, "signal_recordings_ok": recH_all >= 3 and all(r >= 1 for r in recH),
            "correct_A5C_H": f"{k5H}/{n5}", "correct_A5C_R": f"{k5R}/{n5}", "A5C_needed": ceil_pct(98, n5), "retain_A5C_ok": k5H >= ceil_pct(98, n5) and k5H >= k5R - 3,
            "correct_A4C_H": f"{k4H}/{n4}", "correct_A4C_R": f"{k4R}/{n4}", "A4C_needed": ceil_pct(98, n4), "retain_A4C_ok": k4H >= ceil_pct(98, n4) and k4H >= k4R - 3,
            "net_H_minus_Base_each_fold_ge_0": all(net("H", h) >= 0 for h in FOLDS), "net_H_minus_R_each_fold_ge_0": all(net("H", h) - net("R", h) >= 0 for h in FOLDS),
            "dF_H_minus_Base_each_fold_ge_0": all(d_("H", "Base", h, "segment_f1_50_multi") >= 0 for h in FOLDS), "dF_H_minus_R_each_fold_ge_0": all(d_("H", "R", h, "segment_f1_50_multi") >= 0 for h in FOLDS),
            "native_H_minus_Base_each_fold_ge_-0.003": all(d_("H", "Base", h, "native_macro_f1") >= -0.003 for h in FOLDS), "boundary_recall_H_minus_Base_each_fold_ge_-0.005": all(d_("H", "Base", h, "boundary_recall_multi") >= -0.005 for h in FOLDS),
            "missed_short_H_le_Base_plus_3": tot("H", "counts", "fn_short") <= tot("Base", "counts", "fn_short") + 3}
    rule["met"] = all(v for v in rule.values() if isinstance(v, bool))
    values = {"H_minus_R_S1": [C(h)["H"]["S1_fixed"]["fixed"] - C(h)["R"]["S1_fixed"]["fixed"] for h in FOLDS], "H_minus_R_net": [net("H", h) - net("R", h) for h in FOLDS], "H_minus_Base_net": [net("H", h) for h in FOLDS],
              "H_minus_R_dF": [d_("H", "R", h, "segment_f1_50_multi") for h in FOLDS], "H_minus_Base_dF": [d_("H", "Base", h, "segment_f1_50_multi") for h in FOLDS],
              "H_minus_Base_native": [d_("H", "Base", h, "native_macro_f1") for h in FOLDS], "H_minus_Base_boundary_recall": [d_("H", "Base", h, "boundary_recall_multi") for h in FOLDS],
              "missed_short": {m_: tot(m_, "counts", "fn_short") for m_ in ("Base", *ARMS)}}
    summary = {f"heldout_f{h}": {m_: {"net": net(m_, h) if m_ != "Base" else 0, "F1": C(h)[m_]["sequence_row"]["segment_f1_50_multi"], "S1": C(h)[m_]["S1_fixed"]["fixed"], "A5C_kept": C(h)[m_]["correct_A5C_kept"]["kept"],
                                       "A4C_kept": C(h)[m_]["correct_A4C_kept"]["kept"]} for m_ in ("Base", *ARMS)} for h in FOLDS}
    return rule, values, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--stage", choices=("check", "lambda", "train", "freeze", "readout"), required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--commit", default=""); ap.add_argument("--manifest-sha", default=""); ap.add_argument("--lambda-sha", default=""); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "freeze":
        return staged(ctx, f"freeze s{a.seed} (CPU)", "cpu", lambda: freeze(ctx, a.seed, a.commit, a.manifest_sha))
    if a.stage == "readout" and a.dry_run:
        return staged(ctx, f"readout dry run s{a.seed} (Init, CPU)", "cpu", lambda: readout(ctx, a.seed, None, a.workers, True))
    t31.vsh.determinism(); device = torch.device("cuda")
    if a.stage == "train":
        return train(ctx, a.seed, a.lambda_sha, device, a.workers)
    fn = {"check": lambda: check(ctx, device, a.workers), "lambda": lambda: lambda_stage(ctx, a.seed, device), "readout": lambda: readout(ctx, a.seed, device, a.workers)}[a.stage]
    staged(ctx, f"{a.stage}{'' if a.stage == 'check' else ' s' + str(a.seed)} (GPU job wall time)", "gpu", fn)


if __name__ == "__main__":
    main()
