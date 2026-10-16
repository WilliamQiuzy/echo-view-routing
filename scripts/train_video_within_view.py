#!/usr/bin/env python
"""Prepare, check, train and queue the A08 within-view regularisation round
(docs/superpowers/plans/2026-10-16-video-within-view-regularization.md, Tasks 1-2; configs/experiments/video_within_view_v1.yaml).

Every new run strictly loads its seed's V01 Ordered final state (never an A07-trained state) into an unchanged
TargetMViT. Update u reads row u of A07's seed manifest through V01's loader (8 windows of 16 frames; every window's pixel
digest must equal the manifest's) and makes one forward. The (8, 5) CLS logits are viewed as two groups of four targets:
A07 F's plain cross-entropy plus 0.15 times the arm's truncated neighbour penalty (all neighbours, A; equal-label
neighbours, M). Both arms log both penalties and the equal- and different-label edge sums of the same detached logits.
Optimiser, schedule, warm-up, clipping, precision, determinism and model RNG are A07's (V01's). A07's frozen plain-CE
runs F are the control and are not touched here. Only training data is read.

  --prepare      CPU: every training-side pin, the manifests' file, array and epoch digests, every window's pixel digest
                 and the edge-mask counts
  --check        GPU: the Task 1-2 checks before the queue, with the budget projection
  --queue        the six runs in order with a projection before each, then the freeze
  --arm A --seed S   one formal run

  python scripts/train_video_within_view.py --config configs/experiments/video_within_view_v1.yaml --prepare
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
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

import echo_routing.temporal.video_sequence_features as vsf  # noqa: E402
import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.config import gpu_memory_fraction  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402
from echo_routing.temporal.joint_visual import crop_uint8, gamma_lut, load_frame_u8  # noqa: E402
from echo_routing.temporal.losses import segment_objective as so  # noqa: E402
from echo_routing.temporal.losses import within_view as wv  # noqa: E402

NS = "video_within_view_v1"
GROUP_NAME = vso.GROUP_NAME
ARMS = wv.ARMS
NAMES = {"all": "A", "masked": "M"}
WINDOWS, GROUP = vso.WINDOWS, vpr.GROUP
SEED = 0                                        # the check's seed
TRAINING_KEYS = ("task_sheet", "source", "arms", "control", "model", "data", "loss", "training", "determinism", "budget")
CODE_FILES = (Path(__file__), Path(wv.__file__), Path(vso.__file__), Path(vpr.__file__), Path(so.__file__), Path(tm.__file__), Path(rvc.__file__),
              Path(prv.__file__), Path(pvft.__file__), Path(vsf.__file__))
OPS = ("precision", "tf32", "cudnn_deterministic", "cudnn_benchmark")
F_TRAINING_PINS = ("weights", "frozen_manifest", "run_manifest")
finished, make_optimizer, apply_schedule, schedule = rvc.finished, rvc.make_optimizer, rvc.apply_schedule, rvc.schedule
save_resume, load_resume, determinism, loader, check_digests = rvc.save_resume, rvc.load_resume, rvc.determinism, rvc.loader, rvc.check_digests
make_model, optimiser_bound, parity, max_diff, group_diff, Updates = vso.make_model, vso.optimiser_bound, vso.parity, vso.max_diff, vso.group_diff, vso.Updates
ledger_append, ledger_hours, now, write_json, queue_projection, alive = vpr.ledger_append, vso.ledger_hours, vso.now, vso.write_json, vso.queue_projection, vso.alive


def pinned(root: Path, pin: dict) -> Path:
    return vpr.pinned(root, pin)


# ----------------------------------------------------------------------------------------------- context
class Ctx:
    """This round's configuration, A07's (read only: its manifests, the V01 start states and the frame paths) and the
    output directories of this round."""

    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.www = cfg, config, cfg["video_within_view"]; self.root = ROOT; self.src = self.www["source"]
        a07 = self.src["a07_config"]; pinned(ROOT, a07)
        self.a07 = vso.Ctx(load_cfg(SimpleNamespace(config=a07["path"], set=[])), a07["path"])
        self.a07vso, self.rv, self.paths = self.a07.vso, self.a07.rv, self.a07.paths
        if self.www["training"] != self.a07vso["training"]:
            raise SystemExit("the training section differs from A07's")
        if any(self.www["determinism"][k] != self.a07vso["determinism"][k] for k in OPS):
            raise SystemExit("the determinism settings differ from A07's")
        tr = self.www["training"]; self.seeds = [int(s) for s in tr["seeds"]]; self.total, self.upe = int(tr["total_updates"]), int(tr["updates_per_epoch"])
        self.warmup, self.o = int(tr["warmup_updates"]), tr["optimiser"]
        o = self.www["outputs"]; self.out, self.ckpt_root, self.results = ROOT / o["root"], ROOT / o["checkpoints"], ROOT / o["results_dir"]
        self.ledger = ROOT / self.www["budget"]["ledger"]; self._m = {}

    def manifest(self, seed: int) -> tuple[dict, str]:
        """A07's manifest of the seed (its own sha256 and array checks against index.json), equal to this round's pins."""
        if seed not in self._m:
            pinned(ROOT, self.src["manifests"]["index"]); m, sha = self.a07.manifest(seed)
            if sha != self.src["manifests"][seed]["sha256"]:
                raise SystemExit(f"seed {seed}: the manifest differs from this round's pin")
            self._m[seed] = (m, sha)
        return self._m[seed]

    def epoch_digests(self, seed: int) -> list[str]:
        return self.a07.epoch_digests(seed)

    def source_state(self, seed: int) -> tuple[dict, str]:
        return self.a07.source_state(seed)

    def frame_paths(self) -> FramePaths:
        return self.a07.frame_paths()


# ----------------------------------------------------------------------------------------------- identity, budget
def code_digest() -> str:
    """This round's training script and loss module, A07's training and preparation scripts, A05's loss module, the model
    module, V01's trainer and preparation and A01's preparation and sequence modules in full, plus the shared frame, path
    and schedule functions."""
    h = hashlib.sha256()
    for p in CODE_FILES:
        h.update(Path(p).read_bytes())
    for fn in (load_frame_u8, crop_uint8, gamma_lut, lr_factor, FramePaths, list_frame_paths, prv.frame_dirs):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()


def identity(ctx, arm: str, seed: int, manifest_sha: str, source_sha: str) -> str:
    payload = {"experiment": ctx.cfg.get("experiment"), "video_within_view": {k: ctx.www[k] for k in TRAINING_KEYS},
               "a07_training": {k: ctx.a07vso[k] for k in vso.TRAINING_KEYS}, "source_training": {k: ctx.rv[k] for k in rvc.TRAINING_KEYS},
               "arm": arm, "seed": int(seed), "manifest_sha256": manifest_sha, "source_weights_sha256": source_sha,
               "epoch_digests": ctx.epoch_digests(seed), "environment": prv.environment(), "total_updates": int(ctx.total), "code": code_digest()}
    canon = json.loads(json.dumps(payload, default=str))            # integer and string keys (the seed pins) become strings first
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()


def run_key(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


def split_key(key: str) -> tuple[str, int]:
    arm, s = key.rsplit("_s", 1)
    return arm, int(s)


def identities(ctx: Ctx) -> dict:
    out = {}
    for key in ctx.www["queue"]:
        arm, s = split_key(key); out[key] = identity(ctx, arm, s, ctx.manifest(s)[1], ctx.a07vso["source"]["weights"][s]["sha256"])
    return out


def launch_projection(spent: float, per_arm: dict, b: dict) -> dict:
    """Before the queue: spent + 3 x each arm's measured hours per run + the evaluation and overhead reserves."""
    runs = float(sum(3 * float(per_arm[a]) for a in ARMS)); res = float(b["evaluation_reserve_hours"]) + float(b["overhead_reserve_hours"])
    total = float(spent) + runs + res
    return {"spent": float(spent), "runs": runs, "reserves": res, "total": total, "cap": float(b["gpu_hours_max"]), "within_cap": bool(total <= float(b["gpu_hours_max"]))}


# ----------------------------------------------------------------------------------------------- sources, manifests, masks (preparation)
def verify_manifest(m: dict, file_sha: str, pin_sha: str, index_entry: dict, prep_entry: dict, arrays=vpr.arrays_digest, epochs=None) -> dict:
    """File, array and epoch digests of one A07 manifest against this round's pin, A07's index.json and its preparation."""
    if not (file_sha == pin_sha == index_entry["sha256"] == prep_entry["sha256"]):
        raise SystemExit("a manifest file differs from its pin, index.json or A07's preparation")
    a = arrays(m)
    if not (a == index_entry["arrays_digest"] == prep_entry["arrays_digest"]):
        raise SystemExit("a manifest's arrays differ from index.json or A07's preparation")
    e = epochs(m)
    if not (list(e) == list(index_entry["epoch_digests"]) == list(prep_entry["epoch_digests"])):
        raise SystemExit("a manifest's epoch digests differ from index.json or A07's preparation")
    return {"sha256": file_sha, "arrays_digest": a, "epochs": len(e), "ok": True}


def mask_counts(targets) -> dict:
    """Equal-label edges of the native groups (windows 0-3) and the constructed groups (windows 4-7) of every update."""
    t = np.asarray(targets).astype(np.int64); out = {}
    for kind, sl in (("native", slice(0, GROUP)), ("constructed", slice(GROUP, 2 * GROUP))):
        g = t[:, sl]; m = g[:, 1:] == g[:, :-1]
        out[kind] = {"groups": int(len(g)), "selected_edges": int(m.sum()), "unselected_edges": int((~m).sum()), "zero_edge_groups": int((m.sum(1) == 0).sum())}
        out[f"{kind}_patterns"] = dict(sorted(Counter("".join("1" if x else "0" for x in row) for row in m).items()))
    return out


def verify_sources(ctx: Ctx) -> dict:
    """Every training-side pin: this round's task sheet and A07 records, scripts and loss module, the manifests, F's
    weights and run records (its saved development outputs are not opened here), and A07's own pins through A07's check."""
    src = ctx.src; out = {"task_sheet": sha256_file(pinned(ROOT, ctx.www["task_sheet"]))}
    for k in ("a07_config", "a07_preparation", "a07_summary", "a07_choices", "loss_module"):
        out[k] = sha256_file(pinned(ROOT, src[k]))
    out["a07_scripts"] = {k: sha256_file(pinned(ROOT, p)) for k, p in src["a07_scripts"].items()}
    out["manifests"] = {str(k): sha256_file(pinned(ROOT, p)) for k, p in src["manifests"].items()}
    summary = json.loads((ROOT / src["a07_summary"]["path"]).read_text())["runs"]; out["f_control"] = {}
    for s in ctx.seeds:
        f = src["f_control"][s]; got = {k: sha256_file(pinned(ROOT, f[k])) for k in F_TRAINING_PINS}
        man = json.loads((ROOT / f["run_manifest"]["path"]).read_text()); key = run_key(ctx.www["control"]["a07_arm"], s)
        if not (man["identity"] == f["identity"] == summary[key]["identity"]) or not (man["final_sha256"] == f["weights"]["sha256"] == summary[key]["final_sha256"]):
            raise SystemExit(f"seed {s}: F's identity or final weights differ between the pin, its run manifest and A07's summary")
        if int(man["updates"]) != ctx.total or man["interrupted"]:
            raise SystemExit(f"seed {s}: F is not a complete uninterrupted run")
        out["f_control"][str(s)] = {**got, "identity": f["identity"]}
    out["a07_inherited"] = vpr.verify_sources(ctx.a07)
    env = vpr.environment()
    if any(str(env[k]) != str(v) for k, v in src["environment"].items()):
        raise SystemExit(f"environment {env} differs from the pinned {src['environment']}")
    out["environment"] = env
    return out


def recheck_pixels(ctx: Ctx, workers: int) -> dict:
    """Every window's pixel digest recomputed from the allowlisted training frames (V01's preprocessing)."""
    ms = {s: ctx.manifest(s)[0] for s in ctx.seeds}; refs = set().union(*(vpr.slot_refs(m) for m in ms.values()))
    vids = {r[0] for r in refs}
    if vids - set(prv.allowlist(ctx.rv, ROOT)):
        raise SystemExit("a manifest recording is not a training clip")
    dirs = prv.frame_dirs(ctx.paths, ctx.rv, vids); t0 = time.time(); hashes = prv.pixel_hashes(refs, dirs, workers)
    out = {"distinct_frames": len(hashes), "seconds": time.time() - t0, "seeds": {}}
    for s, m in ms.items():
        got = vpr.window_digests(m, hashes); out["seeds"][str(s)] = {"windows": int(got.shape[0] * got.shape[1]), "equal": bool(np.array_equal(got, m["digest_ordered"]))}
    out["ok"] = all(v["equal"] for v in out["seeds"].values())
    return out


def prepare(ctx: Ctx, workers: int) -> None:
    t0, started, ok, note = time.time(), now(), False, ""
    try:
        rec = {"config": ctx.config, "config_sha256": sha256_file(ROOT / ctx.config), "started": started, "sources": verify_sources(ctx)}
        prep = json.loads((ROOT / ctx.src["a07_preparation"]["path"]).read_text())["seeds"]; idx = json.loads((ROOT / ctx.src["manifests"]["index"]["path"]).read_text())["seeds"]
        rec["manifests"], rec["masks"] = {}, {}
        for s in ctx.seeds:
            m, sha = ctx.manifest(s)
            rec["manifests"][str(s)] = verify_manifest(m, sha, ctx.src["manifests"][s]["sha256"], idx[str(s)], prep[str(s)], epochs=lambda x: vpr.epoch_digests(x, ctx.upe))
            rep = int(np.asarray(m["repeats"])[:, :GROUP].sum()); want = int(prep[str(s)]["counts"]["repeated_native_targets"])
            if rep != want:
                raise SystemExit(f"seed {s}: {rep} repeated native targets, A07's preparation counted {want}")
            rec["masks"][str(s)] = {**mask_counts(m["target"]), "repeated_native_targets": rep}
        print("pins, manifests and masks checked", flush=True)
        rec["pixels"] = recheck_pixels(ctx, workers)
        if not rec["pixels"]["ok"]:
            raise SystemExit("a window's pixel digest differs from the manifest")
        ids = {str(v) for s in ctx.seeds for v in ctx.manifest(s)[0]["videos"].tolist()}; ids |= {session_of(v) for v in ids}
        rec["finished"] = now(); rec["seconds"] = time.time() - t0; flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "preparation")
        for p in (ctx.out / "preparation.json", ctx.results / "preparation.json"):
            p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(flat, indent=1))
        print(json.dumps(rec["masks"], indent=1), flush=True); ok = True
    except BaseException as e:  # noqa: BLE001 - recorded, then raised again
        note = f"{type(e).__name__}: {str(e)[:200]}"; raise
    finally:
        ledger_append(ctx.ledger, "prepare", "cpu", started, time.time() - t0, ok, note)


# ----------------------------------------------------------------------------------------------- one update
def train_step(model, opt, batch, arm: str, lrs: tuple[float, float], o: dict, device, coefficient: float = wv.COEFFICIENT, keep: bool = False) -> dict:
    """One update: one forward of the eight windows, L_CE + coefficient x S_arm on the (2, 4, 5) view, both penalties and
    the equal-/different-label edge sums logged from the same detached logits, clipping at grad_clip and the AdamW step;
    the operations and random draws of A07's F update plus the penalty terms."""
    u8, slots, target = batch[1], batch[2], batch[3]; t = np.asarray(target).astype(np.int64); lab = t.reshape(2, GROUP)
    model.train()
    z = model(tm.to_rgb(u8.to(device, non_blocking=True)), tm.marker_planes(slots.to(device)))
    z3 = z.view(2, GROUP, 5)
    loss, parts = wv.objective(arm, z3, lab, coefficient)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    mon = wv.monitor(z3, lab)
    opt.zero_grad(set_to_none=True); loss.backward()
    params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    gn = float(torch.nn.utils.clip_grad_norm_(params, float(o["grad_clip"])))
    if not np.isfinite(gn):
        raise FloatingPointError("non-finite gradient norm")
    opt.step()
    pred = z.detach().argmax(1).cpu().numpy().astype(np.int64)
    rec = {"loss": float(loss.detach()), "ce": parts["ce"], "regulariser": parts["regulariser"], "monitor": mon, "grad_norm": gn,
           "lr_encoder": lrs[0], "lr_new": lrs[1], "correct": int((pred == t).sum()), "correct_native": int((pred[:GROUP] == t[:GROUP]).sum()),
           "correct_constructed": int((pred[GROUP:] == t[GROUP:]).sum()), "windows": int(t.size),
           "native_switches": int((np.diff(pred[:GROUP]) != 0).sum()),
           "constructed_within_view_switches": int(((np.diff(pred[GROUP:]) != 0) & (np.diff(t[GROUP:]) == 0)).sum())}
    if keep:
        rec["logits"] = z.detach().cpu()
    return rec


def epoch_record(acc: list[dict], done: int, upe: int, t0: float, device, checked: int, digest: str, warm_end: dict | None) -> dict:
    s = lambda k: sum(a[k] for a in acc)  # noqa: E731
    mon = lambda k: sum(a["monitor"][k] for a in acc)  # noqa: E731
    mean = lambda k: float(np.mean([a[k] for a in acc]))  # noqa: E731
    return {"epoch": done // upe - 1, "updates": done, "loss": mean("loss"), "ce": mean("ce"), "regulariser": mean("regulariser"),
            "S_A": float(np.mean([a["monitor"]["S_A"] for a in acc])), "S_M": float(np.mean([a["monitor"]["S_M"] for a in acc])),
            "same_label_penalty": mon("same_sum") / max(1, mon("same_edges")), "different_label_penalty": mon("diff_sum") / max(1, mon("diff_edges")),
            "same_edges": mon("same_edges"), "different_edges": mon("diff_edges"), "zero_edge_groups": mon("zero_edge_groups"),
            "train_accuracy": s("correct") / s("windows"), "native_accuracy": s("correct_native") / (GROUP * len(acc)),
            "constructed_accuracy": s("correct_constructed") / (GROUP * len(acc)), "native_switches": s("native_switches"),
            "constructed_within_view_switches": s("constructed_within_view_switches"),
            "grad_norm_mean": mean("grad_norm"), "grad_norm_max": float(max(a["grad_norm"] for a in acc)),
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
            print(f"[{key}] epoch {rec['epoch']} {done}/{ctx.total} loss {rec['loss']:.4f} (CE {rec['ce']:.4f} S_A {rec['S_A']:.4f} S_M {rec['S_M']:.4f} "
                  f"same {rec['same_label_penalty']:.4f} diff {rec['different_label_penalty']:.4f}) acc {rec['train_accuracy']:.3f} "
                  f"grad {rec['grad_norm_mean']:.2f}/{rec['grad_norm_max']:.2f} {rec['seconds_this_process']:.0f}s", flush=True)
            acc, checked, warm_end = [], 0, None
    if done != ctx.total:
        raise SystemExit(f"stopped at update {done} of {ctx.total}")
    final = ckpt / "final.pt"; torch.save(model.state_dict(), final)
    man = {"identity": ident, "arm": arm, "key": key, "seed": seed, "config": ctx.config, "config_sha256": sha256_file(ROOT / ctx.config),
           "manifest_sha256": msha, "source_weights_sha256": ssha, "environment": prv.environment(), "code": code_digest(), "updates": done,
           "epochs": log, "start": start, "end": {"encoder": params_digest(model.encoder.named_parameters())}, "interrupted": bool(resumes), "resumes": resumes,
           "final_checkpoint": str(final), "final_sha256": sha256_file(final),
           "parameters": {"encoder": int(sum(p.numel() for p in model.encoder_parameters())), "new": int(sum(p.numel() for p in model.new_parameters()))},
           "seconds_last_process": time.time() - t0}
    tmp = run_dir / "run_manifest.tmp"; tmp.write_text(json.dumps(man, indent=1, default=str)); tmp.replace(run_dir / "run_manifest.json")
    (ckpt / "resume.pt").unlink(missing_ok=True)
    print(f"[{key}] done: {done} updates, final {man['final_sha256'][:12]}", flush=True)


# ----------------------------------------------------------------------------------------------- checks (Tasks 1-2)
def hook_counts(model) -> tuple[int, int, int, int]:
    mods = list(model.modules())
    return (sum(len(m._forward_hooks) for m in mods), sum(len(m._forward_pre_hooks) for m in mods), sum(len(m._backward_hooks) for m in mods),
            sum(len(getattr(m, "_backward_pre_hooks", {})) for m in mods))


def regulariser_gradients(model, batch, arm: str, device) -> dict:
    """coefficient x S of the arm alone: which parameters receive a gradient (the model's mode and trainability are the caller's)."""
    model.zero_grad(set_to_none=True); t = np.asarray(batch[3]).astype(np.int64).reshape(2, GROUP)
    z = model(tm.to_rgb(batch[1].to(device)), tm.marker_planes(batch[2].to(device))).view(2, GROUP, 5)
    reg = wv.regulariser(arm, wv.edge_penalties(wv.log_probabilities(z)), wv.equal_label_mask(t)); (wv.COEFFICIENT * reg).backward()
    enc = [p.grad for p in model.encoder_parameters()]; new = [p.grad for p in model.new_parameters()]
    mx = lambda p: float(p.grad.abs().max()) if p.grad is not None else 0.0  # noqa: E731
    out = {"regulariser": float(reg.detach()), "encoder_without_gradient": int(sum(g is None for g in enc)),
           "nonfinite": int(sum(g is not None and not bool(torch.isfinite(g).all()) for g in enc + new)),
           "projection": mx(model.encoder.head[-1].weight), "conv_proj": mx(model.encoder.conv_proj.weight), "marker": mx(model.marker_stem.weight),
           "classifier": mx(model.classifier.weight)}
    model.zero_grad(set_to_none=True)
    return out


def run_updates(ctx: Ctx, upd, arm: str, us: list[int], warmup: int, total: int, device, coefficient: float = wv.COEFFICIENT, state_file=None) -> dict:
    state, _ = ctx.source_state(SEED); model = make_model(state, device); opt = make_optimizer(model, ctx.o); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    if state_file is not None:
        load_resume(state_file, model, opt, "check")
    enc0 = params_digest(model.encoder.named_parameters()); recs = []
    for u in us:
        lrs = apply_schedule(model, opt, u, warmup, total, ctx.o)
        r = train_step(model, opt, upd(u), arm, lrs, ctx.o, device, coefficient); r["encoder_digest"] = params_digest(model.encoder.named_parameters()); recs.append(r)
    return {"model": model, "opt": opt, "records": recs, "encoder_start": enc0}


def one_update(ctx: Ctx, step, device) -> tuple:
    state, _ = ctx.source_state(SEED); model = make_model(state, device); opt = make_optimizer(model, ctx.o)
    lrs = apply_schedule(model, opt, 0, ctx.warmup, ctx.total, ctx.o); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    rec = step(model, opt, lrs)
    return model, rec, torch.get_rng_state(), torch.cuda.get_rng_state_all()


def lambda_zero(ctx: Ctx, upd, device, tmp: Path) -> dict:
    """This round's update at coefficient 0 against A07's original F update (vso.train_step, arm frame): one update, six
    updates and 3 + 3 with a resume, from identical states, RNG and optimiser."""
    c = ctx.www["checks"]["lambda_zero"]; la, pa = float(c["loss_abs"]), float(c["params_abs"]); us = list(range(int(ctx.www["checks"]["reproducibility"]["updates"])))
    mf, rf, cf, gf = one_update(ctx, lambda m, o, lrs: vso.train_step(m, o, upd(0), "frame", lrs, ctx.o, device, keep=True), device)
    single = {}
    for arm in ARMS:
        m, r, cpu, cuda = one_update(ctx, lambda mm, o, lrs, a=arm: train_step(mm, o, upd(0), a, lrs, ctx.o, device, 0.0, keep=True), device)
        same_rng = torch.equal(cpu, cf) and all(torch.equal(x, y) for x, y in zip(cuda, gf))
        single[arm] = {"logits_identical": bool(torch.equal(r["logits"], rf["outputs"])), "loss_identical": r["loss"] == rf["loss"], "rng_identical": bool(same_rng),
                       "params_max_abs": max_diff(m, mf)}
        single[arm]["ok"] = bool(single[arm]["logits_identical"] and single[arm]["loss_identical"] and same_rng and single[arm]["params_max_abs"] <= pa)
        del m; torch.cuda.empty_cache()
    del mf; f6 = vso.run_updates(ctx.a07, upd, "frame", us, ctx.warmup, ctx.total, device); lf = [r["loss"] for r in f6["records"]]; six, resume = {}, {}
    for arm in ARMS:
        a6 = run_updates(ctx, upd, arm, us, ctx.warmup, ctx.total, device, 0.0); la6 = [r["loss"] for r in a6["records"]]
        six[arm] = {"loss_max_abs": max(abs(x - y) for x, y in zip(la6, lf)), "params_max_abs": max_diff(a6["model"], f6["model"])}
        six[arm]["ok"] = bool(six[arm]["loss_max_abs"] <= la and six[arm]["params_max_abs"] <= pa); del a6
        half = run_updates(ctx, upd, arm, us[:3], ctx.warmup, ctx.total, device, 0.0); save_resume(tmp, half["model"], half["opt"], 3, "check", [])
        rest = run_updates(ctx, upd, arm, us[3:], ctx.warmup, ctx.total, device, 0.0, state_file=tmp); tmp.unlink(missing_ok=True)
        lr_ = [r["loss"] for r in half["records"] + rest["records"]]
        resume[arm] = {"loss_max_abs": max(abs(x - y) for x, y in zip(lr_, lf)), "params_max_abs": max_diff(rest["model"], f6["model"])}
        resume[arm]["ok"] = bool(resume[arm]["loss_max_abs"] <= la and resume[arm]["params_max_abs"] <= pa); del half, rest; torch.cuda.empty_cache()
    del f6; torch.cuda.empty_cache()
    ok = all(v["ok"] for d in (single, six, resume) for v in d.values())
    return {"single_update": single, "six_updates": six, "resume_3_plus_3": resume, "f_losses": lf, "ok": bool(ok)}


def isolation_pairing_gradients(ctx: Ctx, upd, device) -> dict:
    """Labels never reach the forward; A, M and A07's F update see identical logits and use the RNG identically; S_M
    equals S_A on pure groups; each arm's penalty alone reaches the encoder, projection, marker and classifier after
    unfreezing; no check leaves a hook behind."""
    state, _ = ctx.source_state(SEED); b0, b1 = upd(0), upd(1); tol = float(ctx.www["checks"]["pure_identity_tolerance"])
    m = make_model(state, device).eval(); hooks0 = hook_counts(m)
    with torch.no_grad():
        z1 = m(tm.to_rgb(b0[1].to(device)), tm.marker_planes(b0[2].to(device)))
        _ = [wv.objective(a, z1.view(2, 4, 5), np.roll(b0[3].numpy(), 3).reshape(2, 4)) for a in ARMS]
        z2 = m(tm.to_rgb(b0[1].to(device)), tm.marker_planes(b0[2].to(device)))
    isolation = bool(torch.equal(z1, z2)); res = {}
    for arm in ARMS:
        _, rec, cpu, cuda = one_update(ctx, lambda mm, o, lrs, a=arm: train_step(mm, o, b0, a, lrs, ctx.o, device, keep=True), device)
        res[arm] = (rec["logits"], cpu, cuda, rec["loss"], rec["monitor"]); torch.cuda.empty_cache()
    _, rf, cf, gf = one_update(ctx, lambda mm, o, lrs: vso.train_step(mm, o, b0, "frame", lrs, ctx.o, device, keep=True), device)
    pairing = all(torch.equal(res[a][0], rf["outputs"]) and torch.equal(res[a][1], cf) and all(torch.equal(x, y) for x, y in zip(res[a][2], gf)) for a in ARMS)
    monitors_equal = res["all"][4] == res["masked"][4]
    pure_u8 = torch.cat([b0[1][:4], b1[1][:4]]); pure_mk = torch.cat([b0[2][:4], b1[2][:4]]); pure_t = np.concatenate([b0[3][:4].numpy(), b1[3][:4].numpy()]).reshape(2, 4)
    with torch.no_grad():
        zp = m(tm.to_rgb(pure_u8.to(device)), tm.marker_planes(pure_mk.to(device))).double()
    vals, grads = [], []
    for arm in ARMS:
        x = zp.clone().view(2, 4, 5).requires_grad_(True); r = wv.regulariser(arm, wv.edge_penalties(wv.log_probabilities(x)), wv.equal_label_mask(pure_t))
        g, = torch.autograd.grad(r, x); vals.append(float(r)); grads.append(g)
    pure = {"targets": pure_t.tolist(), "value_difference": abs(vals[0] - vals[1]), "gradient_max_abs_difference": float((grads[0] - grads[1]).abs().max())}
    pure["ok"] = bool(all(len(set(row)) == 1 for row in pure_t.tolist()) and pure["value_difference"] <= tol and pure["gradient_max_abs_difference"] <= tol)
    gm = make_model(state, device); gm.train(); gm.set_encoder_trainable(True); hg0 = hook_counts(gm)
    grads_ = {a: regulariser_gradients(gm, b0, a, device) for a in ARMS}; hooks_ok = hook_counts(gm) == hg0 and hook_counts(m) == hooks0; del gm
    for g in grads_.values():
        g["ok"] = bool(g["encoder_without_gradient"] == 0 and g["nonfinite"] == 0 and g["projection"] > 0 and g["conv_proj"] > 0 and g["marker"] > 0 and g["classifier"] > 0)
    del m; torch.cuda.empty_cache()
    ok = isolation and pairing and monitors_equal and pure["ok"] and hooks_ok and all(g["ok"] for g in grads_.values())
    return {"label_isolation": isolation, "pairing": {"logits_and_rng_identical_to_f": bool(pairing), "monitors_equal": bool(monitors_equal),
                                                       "losses": {a: res[a][3] for a in ARMS}, "f_loss": rf["loss"]},
            "pure_groups": pure, "regulariser_gradients": grads_, "hooks_unchanged": bool(hooks_ok), "ok": bool(ok)}


def reproducibility(ctx: Ctx, upd, arm: str, device, tmp: Path) -> dict:
    c = ctx.www["checks"]; us = list(range(int(c["reproducibility"]["updates"])))
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


def boundary(ctx: Ctx, upd, arm: str, device, tmp: Path) -> dict:
    b = ctx.www["checks"]["boundary"]; w, t, k = int(b["warmup"]), int(b["total"]), int(b["resume_at"]); us = list(range(t))
    d = run_updates(ctx, upd, arm, us, w, t, device); f = run_updates(ctx, upd, arm, us, w, t, device)
    half = run_updates(ctx, upd, arm, us[:k], w, t, device); save_resume(tmp, half["model"], half["opt"], k, "check", [])
    e = run_updates(ctx, upd, arm, us[k:], w, t, device, state_file=tmp); tmp.unlink(missing_ok=True)
    want = [schedule(u, w, t, ctx.o) for u in us]; got = [(r["lr_encoder"], r["lr_new"]) for r in d["records"]]
    frozen = [r["encoder_digest"] == d["encoder_start"] for r in d["records"]]
    ld, le, lf = ([r["loss"] for r in x] for x in (d["records"], half["records"] + e["records"], f["records"]))
    res = {"loss_max_abs": max(abs(x - y) for x, y in zip(ld, le)), "by_group": group_diff(d["model"], e["model"])}
    rep = {"loss_max_abs": max(abs(x - y) for x, y in zip(ld, lf)), "by_group": group_diff(d["model"], f["model"])}
    trip = vso.roundtrip(ctx, d, tmp, device)
    bound = {"encoder": float(b["adam_step_bound"]) * sum(x[0] for x in want), "new": float(b["adam_step_bound"]) * sum(x[1] for x in want)}
    within = all(x["by_group"][g]["max_abs"] <= bound[g] for x in (res, rep) for g in bound)
    tol = float(ctx.www["checks"]["resume"]["loss_abs"])
    ok = got == want and all(frozen[:w]) and not any(frozen[w:]) and trip["ok"] and res["loss_max_abs"] <= tol and rep["loss_max_abs"] <= tol and within
    return {"lrs": got, "encoder_unchanged_after_update": frozen, "state_roundtrip": trip, "resume": res, "repeat": rep, "parameter_bound": bound,
            "within_bound": within, "ok": bool(ok)}


def smoke(ctx: Ctx, upd, arm: str, device) -> dict:
    sc = ctx.www["checks"]["smoke"]; state, _ = ctx.source_state(SEED); m = make_model(state, device); m.set_encoder_trainable(False)
    opt = torch.optim.AdamW(m.new_parameters(), lr=float(sc["lr_new"]), betas=tuple(float(x) for x in ctx.o["betas"]), eps=float(ctx.o["eps"]),
                            weight_decay=float(ctx.o["weight_decay"]))
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); enc0 = params_digest(m.encoder.named_parameters()); new0 = [p.detach().clone() for p in m.new_parameters()]
    curve, t0 = [], time.time()
    for i in range(int(sc["updates"])):
        b = upd(i % 2); m.train(); z = m(tm.to_rgb(b[1].to(device)), tm.marker_planes(b[2].to(device))); lab = b[3].numpy().reshape(2, GROUP); z3 = z.view(2, GROUP, 5)
        loss, parts = wv.objective(arm, z3, lab); mon = wv.monitor(z3, lab)
        opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(m.new_parameters(), float(ctx.o["grad_clip"]))); opt.step()
        curve.append({"update": i + 1, "loss": float(loss.detach()), "ce": parts["ce"], "regulariser": parts["regulariser"], "grad_norm": gn,
                      "S_A": mon["S_A"], "S_M": mon["S_M"], "same_label_penalty": mon["same_sum"] / max(1, mon["same_edges"]),
                      "different_label_penalty": mon["diff_sum"] / max(1, mon["diff_edges"]), "correct": int((z.detach().argmax(1).cpu() == b[3]).sum())})
    enc_same = params_digest(m.encoder.named_parameters()) == enc0; changed = all(not torch.equal(p.detach(), q) for p, q in zip(m.new_parameters(), new0))
    finite = bool(np.isfinite([[c["loss"], c["grad_norm"], c["S_A"], c["S_M"]] for c in curve]).all())
    return {"arm": arm, "curve": curve, "encoder_unchanged": enc_same, "marker_and_classifier_changed": changed, "finite": finite,
            "seconds": time.time() - t0, "ok": bool(finite and enc_same and changed)}


def timing(ctx: Ctx, m: dict, fp, arm: str, device, workers: int) -> dict:
    tc = ctx.www["checks"]["timing"]; n_w, n_t = int(tc["warmup_updates"]), int(tc["timed_updates"]); state, _ = ctx.source_state(SEED)
    model = make_model(state, device); opt = make_optimizer(model, ctx.o); model.set_encoder_trainable(True)
    lrs = (float(ctx.o["lr_encoder"]), float(ctx.o["lr_new"])); opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = lrs
    torch.manual_seed(SEED); torch.cuda.reset_peak_memory_stats(device); stamps = []; torch.cuda.synchronize(); prev = time.time()
    for batch in loader(m, GROUP_NAME, fp, list(range(n_w + n_t)), workers):
        u = int(batch[0]); check_digests(m, GROUP_NAME, u, batch[4]); train_step(model, opt, batch, arm, lrs, ctx.o, device)
        torch.cuda.synchronize(); t = time.time(); stamps.append(t - prev); prev = t
    timed = stamps[n_w:]
    return {"arm": arm, "seconds_per_update_mean": float(np.mean(timed)), "seconds_per_update_max": float(np.max(timed)),
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20, "hours_per_run": float(np.mean(timed)) * ctx.total / 3600}


def smoke_figure(smokes: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4)); col = {"all": "#dd8452", "masked": "#4c72b0"}
    for arm, s in smokes.items():
        x = [c["update"] for c in s["curve"]]; lab = f"{NAMES[arm]} ({arm})"
        ax[0].plot(x, [c["ce"] for c in s["curve"]], color=col[arm], label=lab)
        ax[1].plot(x, [c["same_label_penalty"] for c in s["curve"]], color=col[arm], label=lab)
        ax[2].plot(x, [c["different_label_penalty"] for c in s["curve"]], color=col[arm], label=lab)
        ax[3].plot(x, [c["correct"] / 8 for c in s["curve"]], color=col[arm], label=lab)
    for a, t in zip(ax, ("plain CE L_CE", "equal-label edge penalty (logged)", "different-label edge penalty (logged)", "accuracy of the 8 targets of the batch")):
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
    prep = ctx.out / "preparation.json"
    if not prep.is_file() or json.loads(prep.read_text()).get("config_sha256") != sha256_file(ROOT / ctx.config):
        raise SystemExit("run --prepare first with this configuration")
    out = ctx.out / "checks"; out.mkdir(parents=True, exist_ok=True); ctx.results.mkdir(parents=True, exist_ok=True); rec: dict = {"started": now()}
    rec["sources"] = verify_sources(ctx)
    m, _ = ctx.manifest(SEED); fp = ctx.frame_paths(); upd = Updates(m, fp)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        b0, b1 = upd(0), upd(1); inp = (torch.cat([b0[1], b1[1]]), torch.cat([b0[2], b1[2]]))
        rec["inputs"] = vso.input_checks(m); rec["source_parity"] = vso.source_parity(ctx.a07, inp, device)
        rec["isolation_pairing_gradients"] = isolation_pairing_gradients(ctx, upd, device); print("inputs, parity, isolation, pairing and gradients done", flush=True)
        rec["lambda_zero"] = lambda_zero(ctx, upd, device, out / "tmp_resume.pt"); print("coefficient-0 control path done", flush=True)
        rec["smoke"] = {a: smoke(ctx, upd, a, device) for a in ARMS}; print("smokes done", flush=True)
        rec["reproducibility"] = {a: reproducibility(ctx, upd, a, device, out / "tmp_resume.pt") for a in ARMS}
        rec["boundary"] = {a: boundary(ctx, upd, a, device, out / "tmp_resume.pt") for a in ARMS}; print("reproducibility, resume and boundary done", flush=True)
        rec["timing"] = {a: timing(ctx, m, fp, a, device, workers) for a in ARMS}
    rec["nondeterministic_operator_warnings"] = sorted({str(w.message).split("\n")[0][:300] for w in caught if "determinis" in str(w.message).lower()})
    spent = ledger_hours(ctx.ledger, "gpu") + (time.time() - t0) / 3600; per_arm = {a: rec["timing"][a]["hours_per_run"] for a in ARMS}
    rec["budget"] = launch_projection(spent, per_arm, ctx.www["budget"]); rec["budget"]["ok"] = rec["budget"]["within_cap"]
    parts = ("inputs", "source_parity", "isolation_pairing_gradients", "lambda_zero", "budget")
    rec["ok"] = bool(all(rec[p]["ok"] for p in parts) and all(v["ok"] for v in rec["smoke"].values()) and all(v["ok"] for v in rec["reproducibility"].values())
                     and all(v["ok"] for v in rec["boundary"].values()))
    rec["identities"] = identities(ctx); rec["code"] = code_digest(); rec["config_sha256"] = sha256_file(ROOT / ctx.config); rec["seconds"] = time.time() - t0
    smoke_figure(rec["smoke"], ctx.results / "smoke_curves.png")
    ids = {str(v) for v in m["videos"].tolist()}; ids |= {session_of(v) for v in ids}; flat = json.loads(json.dumps(rec, default=str)); assert_no_ids(flat, ids, "check")
    for p in (out / "check.json", ctx.results / "checks.json"):
        p.write_text(json.dumps(flat, indent=1))
    print(json.dumps({k: v.get("ok") for k, v in rec.items() if isinstance(v, dict) and "ok" in v}, indent=1)); print(json.dumps(rec["budget"], indent=1))
    print(f"check ok: {rec['ok']}", flush=True)
    return rec["ok"]


# ----------------------------------------------------------------------------------------------- queue and freeze
def acquire_lock(path: Path) -> None:
    if path.exists():
        held = path.read_text().split()
        if held and alive(held[0], "train_video_within_view.py"):
            raise SystemExit(f"another queue (pid {held[0]}) holds {path.name}")
    path.write_text(f"{os.getpid()} {now()}\n")


def on_signal(signum, _frame):
    raise SystemExit(f"queue stopped by signal {signum}")


def freeze(ctx: Ctx, idents: dict) -> dict:
    out = {}
    for key, ident in idents.items():
        man_path = ctx.out / "train" / key / "run_manifest.json"; man = json.loads(man_path.read_text()); dest = ROOT / "models_frozen" / NS / key
        if man["identity"] != ident or int(man["updates"]) != ctx.total or not finished(man_path, ident):
            raise SystemExit(f"{key}: the run manifest does not match its identity or update count")
        if not dest.exists():
            files = [man["final_checkpoint"], str(man_path), str(ROOT / ctx.config), str(ROOT / ctx.src["manifests"][int(man["seed"])]["path"])]
            subprocess.run([sys.executable, "scripts/freeze_model.py", NS, key, *[x for f in files for x in ("--file", f)], "--meta", f"identity={ident}",
                            "--meta", f"arm={man['arm']}", "--meta", f"seed={man['seed']}", "--meta", f"source_weights_sha256={man['source_weights_sha256']}",
                            "--notes", "A08 within-view regularisation round (2026-10-16); final weights after 3,750 updates from the seed's V01 Ordered model."],
                           cwd=ROOT, check=True, capture_output=True)
        if sha256_file(dest / "final.pt") != man["final_sha256"]:
            raise SystemExit(f"{key}: the frozen copy differs from the final weights")
        out[key] = {"path": str(dest.relative_to(ROOT)), "final_sha256": man["final_sha256"], "updates": man["updates"], "interrupted": man["interrupted"]}
    for pin in ctx.a07vso["source"]["weights"].values():
        pinned(ROOT, pin)                                          # the V01 Ordered sources are unchanged
    for s in ctx.seeds:
        pinned(ROOT, ctx.src["f_control"][s]["weights"])            # and so are A07's F controls
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
    import datetime as dt
    b = ctx.www["budget"]; status_path = ctx.out / "queue_status.json"; status = json.loads(status_path.read_text()) if status_path.exists() else {}
    foreign = [k for k, v in status.get("jobs", {}).items() if v.get("identity") not in (None, idents.get(k))]
    if foreign:
        raise SystemExit(f"the status file belongs to other identities ({foreign})")
    status.setdefault("jobs", {}); status["cap_hours"] = float(b["gpu_hours_max"]); per_arm = {a: float(check["timing"][a]["hours_per_run"]) for a in ARMS}
    for k in idents:
        status["jobs"].setdefault(k, {"state": "pending", "identity": idents[k]})
    vso.reconcile(ctx, status, idents); write_json(status_path, status)
    for key in ctx.www["queue"]:
        arm, seed = split_key(key); man = ctx.out / "train" / key / "run_manifest.json"
        if finished(man, idents[key]):
            status["jobs"][key]["state"] = "done"; write_json(status_path, status); print(f"{key}: done, skipped", flush=True); continue
        done_h = [v["hours"] for v in status["jobs"].values() if v.get("state") == "done" and "hours" in v]
        remaining = [k for k in ctx.www["queue"] if not finished(ctx.out / "train" / k / "run_manifest.json", idents[k])]
        proj = queue_projection(ledger_hours(ctx.ledger, "gpu"), remaining, per_arm, done_h, b); status["projection"] = {"at": now(), "before": key, **proj}
        status.setdefault("projections", []).append(status["projection"])
        if not proj["within_cap"]:
            status["stopped"] = {"at": now(), "reason": f"projected {proj['total']:.3f} GPU hours exceed the cap {proj['cap']}", "before": key}
            write_json(status_path, status); raise SystemExit(status["stopped"]["reason"])
        log = ROOT / logs / f"{NS}_{key}.log"; log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "scripts/train_video_within_view.py", "--config", config, "--arm", arm, "--seed", str(seed)]
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
    g = ap.add_mutually_exclusive_group(); g.add_argument("--prepare", action="store_true"); g.add_argument("--check", action="store_true"); g.add_argument("--queue", action="store_true")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--logs", default="logs")
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config)
    if a.prepare:
        prepare(ctx, a.workers); return
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
