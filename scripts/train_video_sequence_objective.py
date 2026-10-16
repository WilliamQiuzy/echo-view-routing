#!/usr/bin/env python
"""Train, check and queue the A07 sequence-objective round (docs/superpowers/plans/2026-10-15-end-to-end-sequence-objective.md,
Tasks 1-2; configs/experiments/video_sequence_objective_v1.yaml).

Every run strictly loads its seed's V01 Ordered final state into an unchanged TargetMViT (no parameter added). Update u
reads row u of the seed's VSO manifest through V01's loader (8 windows of 16 frames decoded with load_frame_u8, whose
pixel digests must equal the manifest's) and computes one forward. The (8, 5) CLS logits are viewed as two groups of
four targets for the pinned A05 objective of the arm: frame (F, target CE), segment (B, run-balanced CE) or order (O,
B plus the no-blank transcript-forward term), with class weights 1. Every arm logs all objective terms of the same
detached logits. The recipe is V01's: AdamW, the encoder frozen (lr 0, requires_grad False, autograd still through it)
for 375 updates and then 1e-5 x the cosine factor, the marker stem and classifier at 1e-4 x lr_factor, clipping at 1,
fp32, TF32 off, deterministic algorithms with warnings, final weights after 3,750 updates; torch.manual_seed(seed) and the
CUDA seeds precede the first update of every arm. The full resume state is saved after every epoch. Only training data
is read.

  --check        the Task 1-2 checks before the queue (GPU), with the budget projection
  --queue        the nine runs in order with a projection before each, then the freeze
  --arm A --seed S   one formal run

  python scripts/train_video_sequence_objective.py --config configs/experiments/video_sequence_objective_v1.yaml --check
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.prepare_raw_video_context as prv  # noqa: E402
import scripts.prepare_video_feature_temporal as pvft  # noqa: E402
import scripts.prepare_video_sequence_objective as pr  # noqa: E402
import scripts.train_raw_video_context as rvc  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths, params_digest  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import echo_routing.temporal.video_sequence_features as vsf  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.config import gpu_memory_fraction  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402
from echo_routing.temporal.joint_visual import crop_uint8, gamma_lut, load_frame_u8  # noqa: E402
from echo_routing.temporal.losses import segment_objective as so  # noqa: E402

NS = "video_sequence_objective_v1"
GROUP_NAME = "Ordered context"                  # V01's image order; slot order = time order
ARMS = so.ARMS
ALPHA = np.ones(5)
WINDOWS = pr.WINDOWS
SEED = 0                                        # the check's seed
TRAINING_KEYS = ("task_sheet", "source", "arms", "model", "groups", "objectives", "data", "training", "determinism", "budget")
CODE_FILES = (Path(__file__), Path(pr.__file__), Path(so.__file__), Path(tm.__file__), Path(rvc.__file__), Path(prv.__file__), Path(pvft.__file__), Path(vsf.__file__))
finished, make_optimizer, apply_schedule, schedule = rvc.finished, rvc.make_optimizer, rvc.apply_schedule, rvc.schedule
save_resume, load_resume, determinism, loader, check_digests = rvc.save_resume, rvc.load_resume, rvc.determinism, rvc.loader, rvc.check_digests
ledger_append = pr.ledger_append


# ----------------------------------------------------------------------------------------------- context
class Ctx(pr.Ctx):
    """The round's configuration, V01's (read only), the prepared manifests and the output directories."""

    def __init__(self, cfg: dict, config: str) -> None:
        super().__init__(cfg, config)
        tr = self.vso["training"]; self.ckpt_root = self.root / self.vso["outputs"]["checkpoints"]; self.mdir = self.out / "manifests"
        self.warmup, self.o = int(tr["warmup_updates"]), tr["optimiser"]
        if self.o != self.rv["training"]["optimiser"] or (self.total, self.warmup) != (int(self.rv["training"]["total_updates"]), int(self.rv["training"]["warmup_updates"])):
            raise SystemExit("the optimiser or schedule differs from the source round")
        self._index, self._fp, self._m = None, None, {}

    def index(self) -> dict:
        if self._index is None:
            self._index = json.loads((self.mdir / "index.json").read_text())
        return self._index

    def manifest(self, seed: int) -> tuple[dict, str]:
        if seed not in self._m:
            e = self.index()["seeds"][str(seed)]; path = self.mdir / e["file"]
            if (got := sha256_file(path)) != e["sha256"]:
                raise SystemExit(f"{path.name}: sha256 {got[:12]} differs from index.json")
            with np.load(path) as z:
                m = {k: z[k] for k in z.files}
            if pr.arrays_digest(m) != e["arrays_digest"]:
                raise SystemExit(f"{path.name}: its arrays differ from index.json")
            self._m[seed] = (m, got)
        return self._m[seed]

    def epoch_digests(self, seed: int) -> list[str]:
        return self.index()["seeds"][str(seed)]["epoch_digests"]

    def source_state(self, seed: int) -> tuple[dict, str]:
        path = pr.pinned(self.root, self.src["weights"][seed])
        return torch.load(path, map_location="cpu", weights_only=True), sha256_file(path)

    def frame_paths(self) -> FramePaths:
        if self._fp is None:
            train = set(prv.allowlist(self.rv, self.root)); vids = {str(v) for s in self.seeds for v in self.manifest(s)[0]["videos"].tolist()}
            if vids - train:
                raise SystemExit("a manifest recording is not a training clip")
            self._fp = FramePaths(prv.frame_dirs(self.paths, self.rv, vids))
        return self._fp


# ----------------------------------------------------------------------------------------------- identity, ledger, budget
def code_digest() -> str:
    """The round's preparation and training scripts, the loss and model modules, V01's trainer and preparation and A01's
    preparation and sequence modules in full, plus the shared frame, path and schedule functions."""
    h = hashlib.sha256()
    for p in CODE_FILES:
        h.update(Path(p).read_bytes())
    for fn in (load_frame_u8, crop_uint8, gamma_lut, lr_factor, FramePaths, list_frame_paths, prv.frame_dirs):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()


def identity(ctx, arm: str, seed: int, manifest_sha: str, source_sha: str) -> str:
    payload = {"experiment": ctx.cfg.get("experiment"), "video_sequence_objective": {k: ctx.vso[k] for k in TRAINING_KEYS},
               "source_training": {k: ctx.rv[k] for k in rvc.TRAINING_KEYS}, "arm": arm, "seed": int(seed), "manifest_sha256": manifest_sha,
               "source_weights_sha256": source_sha, "epoch_digests": ctx.epoch_digests(seed), "environment": prv.environment(),
               "total_updates": int(ctx.total), "code": code_digest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def run_key(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


def identities(ctx) -> dict:
    return {k: identity(ctx, k.rsplit("_s", 1)[0], int(k.rsplit("_s", 1)[1]), ctx.manifest(int(k.rsplit("_s", 1)[1]))[1],
                        ctx.src["weights"][int(k.rsplit("_s", 1)[1])]["sha256"]) for k in ctx.vso["queue"]}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ledger_hours(path: Path, kind: str) -> float:
    if not Path(path).is_file():
        return 0.0
    return float(sum(x["hours"] for x in (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()) if x.get("kind") == kind))


def launch_projection(spent: float, per_arm: dict, b: dict) -> dict:
    """Before the queue: spent + 3 x each arm's measured hours per run + the evaluation and overhead reserves."""
    runs = float(sum(3 * float(per_arm[a]) for a in ARMS)); res = float(b["evaluation_reserve_hours"]) + float(b["overhead_reserve_hours"])
    total = float(spent) + runs + res
    return {"spent": float(spent), "runs": runs, "reserves": res, "total": total, "cap": float(b["gpu_hours_max"]), "within_cap": bool(total <= float(b["gpu_hours_max"]))}


def queue_projection(spent: float, remaining: list[str], per_arm: dict, finished_hours: list[float], b: dict) -> dict:
    """Before every run: spent + each unfinished run at the larger of its arm's measured hours and the slowest finished run
    + the reserves; finished runs enter only through the ledger."""
    slow = max([float(x) for x in finished_hours], default=0.0); runs = float(sum(max(float(per_arm[k.rsplit("_s", 1)[0]]), slow) for k in remaining))
    res = float(b["evaluation_reserve_hours"]) + float(b["overhead_reserve_hours"]); total = float(spent) + runs + res
    return {"spent": float(spent), "remaining": list(remaining), "runs": runs, "reserves": res, "total": total, "cap": float(b["gpu_hours_max"]),
            "within_cap": bool(total <= float(b["gpu_hours_max"]))}


# ----------------------------------------------------------------------------------------------- model, objective and one update
def make_model(state: dict, device) -> tm.TargetMViT:
    """An unchanged TargetMViT strictly loaded with the full source state (encoder and projection, marker stem, classifier)."""
    m = tm.build_model(None, tm.initial_new_state(0)); m.load_state_dict(state, strict=True)
    return m.to(device)


def group_runs(targets) -> list:
    t = np.asarray(targets).astype(np.int64).ravel()
    if t.size != WINDOWS:
        raise ValueError(f"an update holds {WINDOWS} targets, got {t.size}")
    return [so.reference_runs(t[:pr.GROUP]), so.reference_runs(t[pr.GROUP:])]


def optimiser_bound(opt, model) -> bool:
    return ([id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in model.encoder_parameters()]
            and [id(p) for p in opt.param_groups[1]["params"]] == [id(p) for p in model.new_parameters()] and len(opt.param_groups) == 2)


def parity(a: np.ndarray, b: np.ndarray, tol: float = 1e-4) -> dict:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64); d = float(np.abs(a - b).max()); eq = bool(np.array_equal(a.argmax(1), b.argmax(1)))
    return {"windows": int(len(a)), "max_abs": d, "argmax_equal": eq, "ok": bool(d <= float(tol) and eq)}


def train_step(model, opt, batch, arm: str, lrs: tuple[float, float], o: dict, device, keep: bool = False) -> dict:
    """One update: one forward of the eight windows, the arm's objective on the (2, 4, 5) view, every term logged from the
    same detached logits, clipping at grad_clip and the AdamW step."""
    u8, slots, target = batch[1], batch[2], batch[3]; t = np.asarray(target).astype(np.int64)
    model.train()
    z = model(tm.to_rgb(u8.to(device, non_blocking=True)), tm.marker_planes(slots.to(device)))
    runs = group_runs(t); z3 = z.view(2, pr.GROUP, 5)
    loss, parts = so.objective(arm, z3, runs, ALPHA)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    terms = so.all_terms(z3.detach(), runs, ALPHA)
    opt.zero_grad(set_to_none=True); loss.backward()
    params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    gn = float(torch.nn.utils.clip_grad_norm_(params, float(o["grad_clip"])))
    if not np.isfinite(gn):
        raise FloatingPointError("non-finite gradient norm")
    opt.step()
    pred = z.detach().argmax(1).cpu().numpy().astype(np.int64)
    rec = {"loss": float(loss.detach()), "parts": parts, "terms": terms, "grad_norm": gn, "lr_encoder": lrs[0], "lr_new": lrs[1],
           "correct": int((pred == t).sum()), "correct_native": int((pred[:pr.GROUP] == t[:pr.GROUP]).sum()),
           "correct_constructed": int((pred[pr.GROUP:] == t[pr.GROUP:]).sum()), "windows": int(t.size),
           "constructed_transcript_exact": int(so.reference_runs(pred[pr.GROUP:]).transcript == runs[1].transcript)}
    if keep:
        rec["outputs"] = z.detach().cpu()
    return rec


def order_gradients(model, batch, device) -> dict:
    """The transcript term alone: which parameters receive a gradient (the model's mode and trainability are the caller's)."""
    model.zero_grad(set_to_none=True)
    z = model(tm.to_rgb(batch[1].to(device)), tm.marker_planes(batch[2].to(device))).view(2, pr.GROUP, 5)
    lo = so.order_term(F.log_softmax(z, dim=-1).to(torch.float64), group_runs(np.asarray(batch[3])), ALPHA); lo.backward()
    enc = [p.grad for p in model.encoder_parameters()]; new = [p.grad for p in model.new_parameters()]
    mx = lambda p: float(p.grad.abs().max()) if p.grad is not None else 0.0  # noqa: E731
    return {"loss_order": float(lo.detach()), "encoder_without_gradient": int(sum(g is None for g in enc)),
            "nonfinite": int(sum(g is not None and not bool(torch.isfinite(g).all()) for g in enc + new)),
            "projection": mx(model.encoder.head[-1].weight), "conv_proj": mx(model.encoder.conv_proj.weight), "marker": mx(model.marker_stem.weight),
            "classifier": mx(model.classifier.weight)}


def epoch_record(acc: list[dict], done: int, upe: int, t0: float, device, checked: int, digest: str, warm_end: dict | None) -> dict:
    s = lambda k: sum(a[k] for a in acc)  # noqa: E731
    mean = lambda k: float(np.mean([a["terms"][k] for a in acc]))  # noqa: E731
    return {"epoch": done // upe - 1, "updates": done, "loss": float(np.mean([a["loss"] for a in acc])),
            "terms": {k: mean(k) for k in ("frame", "segment", "order", "objective_order", "segment_mean_ce")},
            "train_accuracy": s("correct") / s("windows"), "native_accuracy": s("correct_native") / (pr.GROUP * len(acc)),
            "constructed_accuracy": s("correct_constructed") / (pr.GROUP * len(acc)), "constructed_transcript_exact": s("constructed_transcript_exact") / len(acc),
            "grad_norm_mean": float(np.mean([a["grad_norm"] for a in acc])), "grad_norm_max": float(max(a["grad_norm"] for a in acc)),
            "lr_encoder_last": acc[-1]["lr_encoder"], "lr_new_last": acc[-1]["lr_new"], "seconds_this_process": time.time() - t0,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if torch.cuda.is_available() else None,
            "windows_digest_checked": checked, "input_digest": digest, **({"warmup_end": warm_end} if warm_end else {})}


# ----------------------------------------------------------------------------------------------- formal run
def train(ctx: Ctx, arm: str, seed: int, device, workers: int) -> None:
    m, msha = ctx.manifest(seed); state, ssha = ctx.source_state(seed); ident = identity(ctx, arm, seed, msha, ssha); key = run_key(arm, seed)
    run_dir = ctx.out / "train" / key; ckpt = ctx.ckpt_root / key; run_dir.mkdir(parents=True, exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)
    if finished(run_dir / "run_manifest.json", ident):
        print(f"[{key}] already finished with identity {ident[:12]}", flush=True); return
    model = make_model(state, device); opt = make_optimizer(model, ctx.o)
    if not optimiser_bound(opt, model):
        raise SystemExit("the optimiser is not bound to the model's parameters")
    start = {"encoder": params_digest(model.encoder.named_parameters()), "all": params_digest(model.named_parameters())}
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    resumes, done, log = [], 0, []
    if (ckpt / "resume.pt").is_file():                       # bookkeeping continues only with a resume of this identity
        done, log = load_resume(ckpt / "resume.pt", model, opt, ident)
        resumes = json.loads((run_dir / "resumes.json").read_text()) if (run_dir / "resumes.json").is_file() else []
        resumes.append({"at_update": done, "time": now()}); (run_dir / "resumes.json").write_text(json.dumps(resumes))
        (run_dir / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    else:
        (run_dir / "resumes.json").unlink(missing_ok=True); (run_dir / "train_log.jsonl").unlink(missing_ok=True)
    print(f"[{key}] identity {ident[:12]}; start at update {done}/{ctx.total}", flush=True)
    fp = ctx.frame_paths(); upe = ctx.upe; digests = ctx.epoch_digests(seed); t0 = time.time(); acc = []; checked = 0; warm_end = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader(m, GROUP_NAME, fp, list(range(done, ctx.total)), workers):
        u = int(batch[0]); check_digests(m, GROUP_NAME, u, batch[4]); checked += WINDOWS
        if not np.array_equal(batch[3].numpy(), m["target"][u]) or not np.array_equal(batch[2].numpy(), m["marker"][u]):
            raise SystemExit(f"update {u}: the loader's targets or markers differ from the manifest")
        lrs = apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        acc.append(train_step(model, opt, batch, arm, lrs, ctx.o, device)); done = u + 1
        if done == ctx.warmup:
            warm_end = {"encoder_unchanged_through_warmup": params_digest(model.encoder.named_parameters()) == start["encoder"],
                        "marker_abs_max": float(model.marker_stem.weight.detach().abs().max())}
        if done % 25 == 0:
            (run_dir / "progress.json").write_text(json.dumps({"identity": ident, "updates": done, "total": ctx.total, "seconds": time.time() - t0,
                                                               "loss": acc[-1]["loss"], "lr_encoder": lrs[0], "lr_new": lrs[1]}))
        if done % upe == 0:
            rec = epoch_record(acc, done, upe, t0, device, checked, digests[done // upe - 1], warm_end)
            log.append(rec); save_resume(ckpt / "resume.pt", model, opt, done, ident, log)
            with (run_dir / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{key}] epoch {rec['epoch']} {done}/{ctx.total} loss {rec['loss']:.4f} (F {rec['terms']['frame']:.4f} B {rec['terms']['segment']:.4f} "
                  f"order {rec['terms']['order']:.4f}) acc {rec['train_accuracy']:.3f} constructed {rec['constructed_accuracy']:.3f} "
                  f"grad {rec['grad_norm_mean']:.2f}/{rec['grad_norm_max']:.2f} {rec['seconds_this_process']:.0f}s", flush=True)
            acc, checked, warm_end = [], 0, None
    if done != ctx.total:
        raise SystemExit(f"stopped at update {done} of {ctx.total}")
    final = ckpt / "final.pt"; torch.save(model.state_dict(), final)
    man = {"identity": ident, "arm": arm, "key": key, "seed": seed, "config": ctx.config, "config_sha256": sha256_file(ctx.root / ctx.config),
           "manifest_sha256": msha, "source_weights_sha256": ssha, "environment": prv.environment(), "code": code_digest(), "updates": done,
           "epochs": log, "start": start, "end": {"encoder": params_digest(model.encoder.named_parameters())}, "interrupted": bool(resumes), "resumes": resumes,
           "final_checkpoint": str(final), "final_sha256": sha256_file(final),
           "parameters": {"encoder": int(sum(p.numel() for p in model.encoder_parameters())), "new": int(sum(p.numel() for p in model.new_parameters()))},
           "seconds_last_process": time.time() - t0}
    tmp = run_dir / "run_manifest.tmp"; tmp.write_text(json.dumps(man, indent=1, default=str)); tmp.replace(run_dir / "run_manifest.json")
    (ckpt / "resume.pt").unlink(missing_ok=True)
    print(f"[{key}] done: {done} updates, final {man['final_sha256'][:12]}", flush=True)


# ----------------------------------------------------------------------------------------------- checks (Tasks 1-2)
class Updates:
    """Update u's decoded batch (digest-checked against the manifest), cached."""

    def __init__(self, m: dict, fp) -> None:
        self.m, self.fp, self.cache = m, fp, {}

    def __call__(self, u: int):
        if u not in self.cache:
            out = rvc.window_inputs(self.m, self.m["order_ordered"], self.fp, [(u, w) for w in range(WINDOWS)])
            check_digests(self.m, GROUP_NAME, u, out[3]); self.cache[u] = (u, *out)
        return self.cache[u]


def run_updates(ctx: Ctx, upd: Updates, arm: str, us: list[int], warmup: int, total: int, device, state_file=None) -> dict:
    state, _ = ctx.source_state(SEED); model = make_model(state, device); opt = make_optimizer(model, ctx.o); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    if state_file is not None:
        load_resume(state_file, model, opt, "check")
    enc0 = params_digest(model.encoder.named_parameters()); recs = []
    for u in us:
        lrs = apply_schedule(model, opt, u, warmup, total, ctx.o)
        r = train_step(model, opt, upd(u), arm, lrs, ctx.o, device); r["encoder_digest"] = params_digest(model.encoder.named_parameters()); recs.append(r)
    return {"model": model, "opt": opt, "records": recs, "encoder_start": enc0}


def max_diff(a, b) -> float:
    sa, sb = a.state_dict(), b.state_dict()
    return float(max((sa[k].float() - sb[k].float()).abs().max() for k in sa))


def group_diff(a, b) -> dict:
    out = {}
    for name, pa, pb in (("encoder", a.encoder_parameters(), b.encoder_parameters()), ("new", a.new_parameters(), b.new_parameters())):
        d = [float((x.detach().float() - y.detach().float()).abs().max()) for x, y in zip(pa, pb)]
        n = sum(int(((x.detach().float() - y.detach().float()).abs() > 1e-6).sum()) for x, y in zip(pa, pb))
        out[name] = {"max_abs": max(d), "elements_over_1e-6": n}
    return out


def roundtrip(ctx: Ctx, run: dict, tmp: Path, device) -> dict:
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all(); save_resume(tmp, run["model"], run["opt"], 0, "check", [])
    state, _ = ctx.source_state(SEED); model = make_model(state, device); opt = make_optimizer(model, ctx.o)
    torch.manual_seed(12345); torch.cuda.manual_seed_all(12345); load_resume(tmp, model, opt, "check"); tmp.unlink(missing_ok=True)
    a, b, sa, sb = run["model"].state_dict(), model.state_dict(), run["opt"].state_dict(), opt.state_dict()
    weights = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    optim = sa["param_groups"] == sb["param_groups"] and sa["state"].keys() == sb["state"].keys() and all(
        torch.equal(torch.as_tensor(sa["state"][i][f]).cpu(), torch.as_tensor(sb["state"][i][f]).cpu()) for i in sa["state"] for f in sa["state"][i])
    rng = torch.equal(torch.get_rng_state(), cpu) and all(torch.equal(x, y) for x, y in zip(torch.cuda.get_rng_state_all(), cuda))
    return {"weights": bool(weights), "optimiser": bool(optim), "rng": bool(rng), "ok": bool(weights and optim and rng)}


def reproducibility(ctx: Ctx, upd: Updates, arm: str, device, tmp: Path) -> dict:
    c = ctx.vso["checks"]; us = list(range(int(c["reproducibility"]["updates"])))
    a = run_updates(ctx, upd, arm, us, ctx.warmup, ctx.total, device); b = run_updates(ctx, upd, arm, us, ctx.warmup, ctx.total, device)
    half = run_updates(ctx, upd, arm, us[:3], ctx.warmup, ctx.total, device); save_resume(tmp, half["model"], half["opt"], 3, "check", [])
    rest = run_updates(ctx, upd, arm, us[3:], ctx.warmup, ctx.total, device, state_file=tmp); tmp.unlink(missing_ok=True)
    la, lb, lc = ([r["loss"] for r in x] for x in (a["records"], b["records"], half["records"] + rest["records"]))
    rep = {"loss_max_abs": max(abs(x - y) for x, y in zip(la, lb)), "params_max_abs": max_diff(a["model"], b["model"])}
    res = {"loss_max_abs": max(abs(x - y) for x, y in zip(la, lc)), "params_max_abs": max_diff(a["model"], rest["model"])}
    enc_same = all(r["encoder_digest"] == a["encoder_start"] for r in a["records"]); bound = optimiser_bound(a["opt"], a["model"])
    ok = (rep["loss_max_abs"] <= float(c["reproducibility"]["loss_abs"]) and rep["params_max_abs"] <= float(c["reproducibility"]["params_abs"])
          and res["loss_max_abs"] <= float(c["resume"]["loss_abs"]) and res["params_max_abs"] <= float(c["resume"]["params_abs"]) and enc_same and bound)
    return {"losses": la, "reproducibility": rep, "resume": res, "encoder_unchanged_in_warmup": enc_same, "optimiser_bound": bound, "ok": bool(ok)}


def boundary(ctx: Ctx, upd: Updates, arm: str, device, tmp: Path) -> dict:
    b = ctx.vso["checks"]["boundary"]; w, t, k = int(b["warmup"]), int(b["total"]), int(b["resume_at"]); us = list(range(t))
    d = run_updates(ctx, upd, arm, us, w, t, device); f = run_updates(ctx, upd, arm, us, w, t, device)
    half = run_updates(ctx, upd, arm, us[:k], w, t, device); save_resume(tmp, half["model"], half["opt"], k, "check", [])
    e = run_updates(ctx, upd, arm, us[k:], w, t, device, state_file=tmp); tmp.unlink(missing_ok=True)
    want = [schedule(u, w, t, ctx.o) for u in us]; got = [(r["lr_encoder"], r["lr_new"]) for r in d["records"]]
    frozen = [r["encoder_digest"] == d["encoder_start"] for r in d["records"]]
    ld, le, lf = ([r["loss"] for r in x] for x in (d["records"], half["records"] + e["records"], f["records"]))
    res = {"loss_max_abs": max(abs(x - y) for x, y in zip(ld, le)), "by_group": group_diff(d["model"], e["model"])}
    rep = {"loss_max_abs": max(abs(x - y) for x, y in zip(ld, lf)), "by_group": group_diff(d["model"], f["model"])}
    trip = roundtrip(ctx, d, tmp, device)
    bound = {"encoder": float(b["adam_step_bound"]) * sum(x[0] for x in want), "new": float(b["adam_step_bound"]) * sum(x[1] for x in want)}
    within = all(x["by_group"][g]["max_abs"] <= bound[g] for x in (res, rep) for g in bound)
    tol = float(ctx.vso["checks"]["resume"]["loss_abs"])
    ok = got == want and all(frozen[:w]) and not any(frozen[w:]) and trip["ok"] and res["loss_max_abs"] <= tol and rep["loss_max_abs"] <= tol and within
    return {"lrs": got, "encoder_unchanged_after_update": frozen, "state_roundtrip": trip, "resume": res, "repeat": rep, "parameter_bound": bound,
            "within_bound": within, "ok": bool(ok)}


def smoke(ctx: Ctx, upd: Updates, arm: str, device) -> dict:
    sc = ctx.vso["checks"]["smoke"]; state, _ = ctx.source_state(SEED); m = make_model(state, device); m.set_encoder_trainable(False)
    opt = torch.optim.AdamW(m.new_parameters(), lr=float(sc["lr_new"]), betas=tuple(float(x) for x in ctx.o["betas"]), eps=float(ctx.o["eps"]),
                            weight_decay=float(ctx.o["weight_decay"]))
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); enc0 = params_digest(m.encoder.named_parameters()); new0 = [p.detach().clone() for p in m.new_parameters()]
    curve, t0 = [], time.time()
    for i in range(int(sc["updates"])):
        b = upd(i % 2); m.train(); z = m(tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device))); runs = group_runs(b[3].numpy()); z3 = z.view(2, pr.GROUP, 5)
        loss, _ = so.objective(arm, z3, runs, ALPHA); terms = so.all_terms(z3.detach(), runs, ALPHA)
        opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(m.new_parameters(), float(ctx.o["grad_clip"]))); opt.step()
        curve.append({"update": i + 1, "loss": float(loss.detach()), "grad_norm": gn, **terms,
                      "correct": int((z.detach().argmax(1).cpu() == b[3]).sum())})
    enc_same = params_digest(m.encoder.named_parameters()) == enc0; changed = all(not torch.equal(p.detach(), q) for p, q in zip(m.new_parameters(), new0))
    finite = bool(np.isfinite([c["loss"] for c in curve]).all() and np.isfinite([c["grad_norm"] for c in curve]).all())
    return {"arm": arm, "curve": curve, "encoder_unchanged": enc_same, "marker_and_classifier_changed": changed, "finite": finite,
            "seconds": time.time() - t0, "ok": bool(finite and enc_same and changed)}


def timing(ctx: Ctx, m: dict, fp, arm: str, device, workers: int) -> dict:
    tc = ctx.vso["checks"]["timing"]; n_w, n_t = int(tc["warmup_updates"]), int(tc["timed_updates"]); state, _ = ctx.source_state(SEED)
    model = make_model(state, device); opt = make_optimizer(model, ctx.o); model.set_encoder_trainable(True)
    lrs = (float(ctx.o["lr_encoder"]), float(ctx.o["lr_new"])); opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = lrs
    torch.manual_seed(SEED); torch.cuda.reset_peak_memory_stats(device); stamps = []; torch.cuda.synchronize(); prev = time.time()
    for batch in loader(m, GROUP_NAME, fp, list(range(n_w + n_t)), workers):
        u = int(batch[0]); check_digests(m, GROUP_NAME, u, batch[4]); train_step(model, opt, batch, arm, lrs, ctx.o, device)
        torch.cuda.synchronize(); t = time.time(); stamps.append(t - prev); prev = t
    timed = stamps[n_w:]
    return {"arm": arm, "seconds_per_update_mean": float(np.mean(timed)), "seconds_per_update_max": float(np.max(timed)),
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20, "hours_per_run": float(np.mean(timed)) * ctx.total / 3600}


def source_parity(ctx: Ctx, inp: tuple, device) -> dict:
    """Zero-update model against V01's own build of the source (EchoPrime build, initial new state, final state), eval."""
    a01 = ctx.a01(); out = {}; tol = float(ctx.vso["checks"]["source_parity_max_abs"])
    rgb, mk = tm.to_rgb(inp[0].to(device)), tm.marker_planes(inp[1].to(device))
    for s in ctx.seeds:
        state, _ = ctx.source_state(s); mine = make_model(state, device).eval(); ref = pvft.build_source(a01, s, device)
        with torch.no_grad():
            a = torch.cat([mine(rgb[i:i + 8], mk[i:i + 8]) for i in range(0, len(rgb), 8)]).cpu().numpy()
            b = torch.cat([ref(rgb[i:i + 8], mk[i:i + 8]) for i in range(0, len(rgb), 8)]).cpu().numpy()
        out[str(s)] = parity(a, b, tol); out[str(s)]["parameters"] = int(sum(p.numel() for p in mine.parameters()))
        del mine, ref; torch.cuda.empty_cache()
    return {"seeds": out, "ok": all(v["ok"] for v in out.values())}


def isolation_and_pairing(ctx: Ctx, upd: Updates, device) -> dict:
    """Labels never reach the forward; the arms' logits and RNG use are identical; pure groups give equal values and logit
    gradients in every arm; the transcript term reaches every encoder parameter, the marker and the classifier."""
    state, _ = ctx.source_state(SEED); b0, b1 = upd(0), upd(1); tol = float(ctx.vso["checks"]["single_class_tolerance"])
    m = make_model(state, device).eval()
    with torch.no_grad():
        z1 = m(tm.to_rgb(b0[1].to(device)), tm.marker_planes(b0[2].to(device)))
        _ = [so.objective(a, z1.view(2, 4, 5), group_runs(np.roll(b0[3].numpy(), 3)), ALPHA) for a in ARMS]
        z2 = m(tm.to_rgb(b0[1].to(device)), tm.marker_planes(b0[2].to(device)))
    isolation = bool(torch.equal(z1, z2))
    res = {}
    for arm in ARMS:
        mm = make_model(state, device); opt = make_optimizer(mm, ctx.o); lrs = apply_schedule(mm, opt, 0, ctx.warmup, ctx.total, ctx.o)
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        rec = train_step(mm, opt, b0, arm, lrs, ctx.o, device, keep=True)
        res[arm] = (rec["outputs"], torch.get_rng_state(), torch.cuda.get_rng_state_all(), rec["loss"]); del mm, opt; torch.cuda.empty_cache()
    ref = res["frame"]
    pairing = all(torch.equal(ref[0], res[a][0]) and torch.equal(ref[1], res[a][1]) and all(torch.equal(x, y) for x, y in zip(ref[2], res[a][2])) for a in ARMS)
    pure_u8 = torch.cat([b0[1][:4], b1[1][:4]]); pure_mk = torch.cat([b0[2][:4], b1[2][:4]]); pure_t = np.concatenate([b0[3][:4].numpy(), b1[3][:4].numpy()])
    with torch.no_grad():
        zp = m(tm.to_rgb(pure_u8.to(device)), tm.marker_planes(pure_mk.to(device))).double()
    vals, grads = [], []
    for arm in ARMS:
        x = zp.clone().requires_grad_(True); loss, _ = so.objective(arm, x.view(2, 4, 5), group_runs(pure_t), ALPHA); g, = torch.autograd.grad(loss, x)
        vals.append(float(loss)); grads.append(g)
    single = {"targets": pure_t.tolist(), "value_spread": max(vals) - min(vals), "gradient_max_abs_difference": float(max((g - grads[0]).abs().max() for g in grads))}
    single["ok"] = bool(len(set(pure_t[:4].tolist())) == 1 and len(set(pure_t[4:].tolist())) == 1 and single["value_spread"] <= tol and single["gradient_max_abs_difference"] <= tol)
    gm = make_model(state, device); gm.train(); gm.set_encoder_trainable(True); og = order_gradients(gm, b0, device); del gm
    og["ok"] = bool(og["encoder_without_gradient"] == 0 and og["nonfinite"] == 0 and og["projection"] > 0 and og["marker"] > 0 and og["classifier"] > 0)
    del m; torch.cuda.empty_cache()
    return {"label_isolation": isolation, "arm_pairing": {"logits_and_rng_identical": bool(pairing), "losses": {a: res[a][3] for a in ARMS}},
            "single_class": single, "order_gradients": og, "ok": bool(isolation and pairing and single["ok"] and og["ok"])}


def input_checks(m: dict) -> dict:
    fails = {"marker_slot": 0, "group_layout": 0, "constructed_without_change": 0}
    for u in (0, 1):
        for w in range(WINDOWS):
            fails["marker_slot"] += int(int(m["marker"][u, w]) != tm.marker_slot(m["s"][u, w], int(m["c"][u, w])))
        fails["group_layout"] += int(not ((m["kind"][u, :4] == 0).all() and (m["kind"][u, 4:] == 1).all() and np.all(np.diff(m["c"][u, 4:]) > 0)))
        fails["constructed_without_change"] += int(len(set(m["target"][u, 4:].tolist())) < 2)
    return {"windows": 16, "failures": fails, "ok": not any(fails.values())}


def smoke_figure(smokes: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4)); col = {"frame": "#8172b2", "segment": "#dd8452", "order": "#4c72b0"}
    for arm, s in smokes.items():
        x = [c["update"] for c in s["curve"]]
        ax[0].plot(x, [c["loss"] for c in s["curve"]], color=col[arm], label=f"{arm} (its own objective)")
        ax[1].plot(x, [c["order"] for c in s["curve"]], color=col[arm], label=arm); ax[2].plot(x, [c["correct"] / 8 for c in s["curve"]], color=col[arm], label=arm)
    for a, t in zip(ax, ("training objective", "transcript term L_order (logged)", "accuracy of the 8 targets of the batch")):
        a.set_title(t, fontsize=9); a.set_xlabel("updates (two fixed training batches alternating; encoder frozen)"); a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.suptitle("100-update finite-value checks on training data only; not an efficacy test and nothing is chosen from them", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def run_checks(ctx: Ctx, device, workers: int) -> None:
    """Every attempt, finished or not, adds its hours to the round's ledger as a GPU stage."""
    t0 = time.time(); started = now(); ok = False; note = ""
    try:
        ok = checks(ctx, device, workers, t0)
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "check", "gpu", started, time.time() - t0, ok, note)
    if not ok:
        raise SystemExit("a hard check failed; see check.json")


def checks(ctx: Ctx, device, workers: int, t0: float) -> bool:
    out = ctx.out / "checks"; out.mkdir(parents=True, exist_ok=True); ctx.results.mkdir(parents=True, exist_ok=True); rec: dict = {"started": now()}
    rec["sources"] = pr.verify_sources(ctx)
    rec["manifests"] = {str(s): {"sha256": ctx.manifest(s)[1], "arrays_digest": ctx.index()["seeds"][str(s)]["arrays_digest"]} for s in ctx.seeds}
    m, _ = ctx.manifest(SEED); fp = ctx.frame_paths(); upd = Updates(m, fp)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        b0, b1 = upd(0), upd(1); inp = (torch.cat([b0[1], b1[1]]), torch.cat([b0[2], b1[2]]))
        rec["inputs"] = input_checks(m); rec["source_parity"] = source_parity(ctx, inp, device)
        rec["isolation_and_pairing"] = isolation_and_pairing(ctx, upd, device); print("inputs, parity, isolation, pairing and gradients done", flush=True)
        rec["smoke"] = {a: smoke(ctx, upd, a, device) for a in ARMS}; print("smokes done", flush=True)
        rec["reproducibility"] = {a: reproducibility(ctx, upd, a, device, out / "tmp_resume.pt") for a in ARMS}
        rec["boundary"] = {a: boundary(ctx, upd, a, device, out / "tmp_resume.pt") for a in ARMS}; print("reproducibility, resume and boundary done", flush=True)
        rec["timing"] = {a: timing(ctx, m, fp, a, device, workers) for a in ARMS}
    rec["nondeterministic_operator_warnings"] = sorted({str(w.message).split("\n")[0][:300] for w in caught if "determinis" in str(w.message).lower()})
    spent = ledger_hours(ctx.ledger, "gpu") + (time.time() - t0) / 3600; per_arm = {a: rec["timing"][a]["hours_per_run"] for a in ARMS}
    rec["budget"] = launch_projection(spent, per_arm, ctx.vso["budget"]); rec["budget"]["ok"] = rec["budget"]["within_cap"]
    parts = ("inputs", "source_parity", "isolation_and_pairing", "budget")
    rec["ok"] = bool(all(rec[p]["ok"] for p in parts) and all(v["ok"] for v in rec["smoke"].values()) and all(v["ok"] for v in rec["reproducibility"].values())
                     and all(v["ok"] for v in rec["boundary"].values()))
    rec["identities"] = identities(ctx); rec["code"] = code_digest(); rec["config_sha256"] = sha256_file(ctx.root / ctx.config); rec["seconds"] = time.time() - t0
    smoke_figure(rec["smoke"], ctx.results / "smoke_curves.png")
    ids = {str(v) for v in m["videos"].tolist()}; ids |= {session_of(v) for v in ids}; flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "check")
    for p in (out / "check.json", ctx.results / "checks.json"):
        p.write_text(json.dumps(flat, indent=1))
    print(json.dumps({k: v.get("ok") for k, v in rec.items() if isinstance(v, dict) and "ok" in v}, indent=1)); print(json.dumps(rec["budget"], indent=1))
    print(f"check ok: {rec['ok']}", flush=True)
    return rec["ok"]


# ----------------------------------------------------------------------------------------------- queue and freeze
def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(obj, indent=1, default=str)); tmp.replace(path)


def alive(pid, script: str) -> bool:
    """A live process of this script (on Linux its command line is read, so a pid reused after a reboot does not count)."""
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    cmdline = Path(f"/proc/{int(pid)}/cmdline")
    return script in cmdline.read_bytes().decode(errors="replace") if cmdline.exists() else True


def acquire_lock(path: Path) -> None:
    if path.exists():
        held = path.read_text().split()
        if held and alive(held[0], "train_video_sequence_objective.py"):
            raise SystemExit(f"another queue (pid {held[0]}) holds {path.name}")
    path.write_text(f"{os.getpid()} {now()}\n")


def on_signal(signum, _frame):
    raise SystemExit(f"queue stopped by signal {signum}")


def reconcile(ctx, status: dict, idents: dict) -> None:
    """Charge every attempt a dead queue left 'running': until its run manifest (finished) or its log's last write."""
    for key, job in status["jobs"].items():
        if job.get("state") != "running":
            continue
        if alive(job.get("pid"), "--arm"):
            raise SystemExit(f"{key}: its trainer (pid {job['pid']}) is still running; wait for it or stop it first")
        man = ctx.out / "train" / key / "run_manifest.json"; done = finished(man, idents[key]); log = Path(job.get("log", ""))
        end_path = man if done else log
        start = dt.datetime.fromisoformat(job["started"].replace("Z", "+00:00"))
        end = dt.datetime.fromtimestamp(end_path.stat().st_mtime, dt.timezone.utc) if end_path.is_file() else start
        hours = max(0.0, (end - start).total_seconds() / 3600)
        ledger_append(ctx.ledger, f"train {key}", "gpu", job["started"], hours * 3600, done, "attempt left running by a stopped queue; charged at the next queue start "
                      + ("until its run manifest" if done else "until its log's last write"))
        job.update({"state": "done" if done else "interrupted", "hours": hours, "charged_at_restart": True})


def freeze(ctx: Ctx, idents: dict) -> dict:
    out = {}
    for key, ident in idents.items():
        man_path = ctx.out / "train" / key / "run_manifest.json"; man = json.loads(man_path.read_text()); dest = ROOT / "models_frozen" / NS / key
        if man["identity"] != ident or int(man["updates"]) != ctx.total or not finished(man_path, ident):
            raise SystemExit(f"{key}: the run manifest does not match its identity or update count")
        if not dest.exists():
            files = [man["final_checkpoint"], str(man_path), str(ROOT / ctx.config), str(ctx.mdir / f"VSO_s{man['seed']}.npz")]
            subprocess.run([sys.executable, "scripts/freeze_model.py", NS, key, *[x for f in files for x in ("--file", f)], "--meta", f"identity={ident}",
                            "--meta", f"arm={man['arm']}", "--meta", f"seed={man['seed']}", "--meta", f"source_weights_sha256={man['source_weights_sha256']}",
                            "--notes", "A07 sequence-objective round (2026-10-15); final weights after 3,750 updates from the seed's V01 Ordered model."],
                           cwd=ROOT, check=True, capture_output=True)
        if sha256_file(dest / "final.pt") != man["final_sha256"]:
            raise SystemExit(f"{key}: the frozen copy differs from the final weights")
        out[key] = {"path": str(dest.relative_to(ROOT)), "final_sha256": man["final_sha256"], "updates": man["updates"], "interrupted": man["interrupted"]}
    for pin in ctx.src["weights"].values():
        pr.pinned(ROOT, pin)                                       # the V01 Ordered sources are unchanged
    return out


def queue(ctx: Ctx, config: str, logs: str) -> None:
    check = json.loads((ctx.out / "checks" / "check.json").read_text()); idents = identities(ctx)
    if not check.get("ok") or check.get("identities") != idents:
        raise SystemExit("no passing check record for the current code and inputs; rerun --check")
    lock = ctx.out / "queue.lock"; acquire_lock(lock); signal.signal(signal.SIGTERM, on_signal); signal.signal(signal.SIGHUP, on_signal)
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        run_queue(ctx, config, logs, check, idents); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        lock.unlink(missing_ok=True)
        run_h = sum(float(x["hours"]) for x in (json.loads(line) for line in ctx.ledger.read_text().splitlines() if line.strip())
                    if x["job"].startswith("train ") and x["started"] >= started) if ctx.ledger.is_file() else 0.0
        ledger_append(ctx.ledger, "queue overhead and freeze", "cpu", started, max(0.0, time.time() - t0 - run_h * 3600), ok, note)


def run_queue(ctx: Ctx, config: str, logs: str, check: dict, idents: dict) -> None:
    b = ctx.vso["budget"]; status_path = ctx.out / "queue_status.json"; status = json.loads(status_path.read_text()) if status_path.exists() else {}
    foreign = [k for k, v in status.get("jobs", {}).items() if v.get("identity") not in (None, idents.get(k))]
    if foreign:
        raise SystemExit(f"the status file belongs to other identities ({foreign})")
    status.setdefault("jobs", {}); status["cap_hours"] = float(b["gpu_hours_max"]); per_arm = {a: float(check["timing"][a]["hours_per_run"]) for a in ARMS}
    for k in idents:
        status["jobs"].setdefault(k, {"state": "pending", "identity": idents[k]})
    reconcile(ctx, status, idents); write_json(status_path, status)
    for key in ctx.vso["queue"]:
        arm, seed = key.rsplit("_s", 1)[0], int(key.rsplit("_s", 1)[1]); man = ctx.out / "train" / key / "run_manifest.json"
        if finished(man, idents[key]):
            status["jobs"][key]["state"] = "done"; write_json(status_path, status); print(f"{key}: done, skipped", flush=True); continue
        done_h = [v["hours"] for v in status["jobs"].values() if v.get("state") == "done" and "hours" in v]
        remaining = [k for k in ctx.vso["queue"] if not finished(ctx.out / "train" / k / "run_manifest.json", idents[k])]
        proj = queue_projection(ledger_hours(ctx.ledger, "gpu"), remaining, per_arm, done_h, b); status["projection"] = {"at": now(), "before": key, **proj}
        status.setdefault("projections", []).append(status["projection"])
        if not proj["within_cap"]:
            status["stopped"] = {"at": now(), "reason": f"projected {proj['total']:.3f} GPU hours exceed the cap {proj['cap']}", "before": key}
            write_json(status_path, status); raise SystemExit(status["stopped"]["reason"])
        log = ROOT / logs / f"{NS}_{key}.log"; log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "scripts/train_video_sequence_objective.py", "--config", config, "--arm", arm, "--seed", str(seed)]
        t0 = dt.datetime.now(dt.timezone.utc); started = now(); rc, ok, note = None, False, ""
        with log.open("a") as fh:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            status["jobs"][key] = {"state": "running", "identity": idents[key], "arm": arm, "seed": seed, "log": str(log), "started": started, "pid": proc.pid}
            write_json(status_path, status)
            try:
                rc = proc.wait(); ok = rc == 0 and finished(man, idents[key]); note = "" if ok else f"returncode {rc}"
            except BaseException as e:  # noqa: BLE001 - the queue is stopping: stop the trainer and charge the attempt
                proc.terminate()
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                note = f"queue stopped ({type(e).__name__}: {e}); trainer terminated"; raise
            finally:
                hours = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds() / 3600
                ledger_append(ctx.ledger, f"train {key}", "gpu", started, hours * 3600, ok, note)
                status["jobs"][key] = {**status["jobs"][key], "state": "done" if ok else ("failed" if rc is not None else "interrupted"), "returncode": rc,
                                       "finished": now(), "hours": hours}
                write_json(status_path, status)
        if ok:
            mj = json.loads(man.read_text()); status["jobs"][key].update({"updates": mj["updates"], "final_sha256": mj["final_sha256"], "interrupted": mj["interrupted"]})
        else:
            status["jobs"][key]["log_tail"] = log.read_text()[-2000:]
        write_json(status_path, status); print(f"{key}: {status['jobs'][key]['state']} ({hours:.3f} h)", flush=True)
        if not ok:
            raise SystemExit(f"{key} failed; locate the cause and rerun the queue (finished runs are skipped, this one resumes under its identity)")
    status["frozen"] = freeze(ctx, idents); status["finished"] = now(); write_json(status_path, status); print("queue finished and frozen", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--arm", choices=ARMS); ap.add_argument("--seed", type=int)
    g = ap.add_mutually_exclusive_group(); g.add_argument("--check", action="store_true"); g.add_argument("--queue", action="store_true")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--logs", default="logs")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    if a.queue:
        queue(ctx, a.config, a.logs); return
    determinism(); device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    if a.check:
        run_checks(ctx, device, a.workers); return
    if a.arm is None or a.seed not in ctx.seeds:
        raise SystemExit("--arm and a configured --seed are required")
    train(ctx, a.arm, a.seed, device, a.workers)


if __name__ == "__main__":
    main()
