#!/usr/bin/env python
"""Train one readout and seed of the target-anchored readout round, or run its checks before the queue
(docs/superpowers/plans/2026-10-11-target-patch-readout.md, Task 2; configs/experiments/target_patch_readout_v1.yaml).

Every run strictly loads its seed's Ordered context model of the raw-video context round (encoder with the 512-d
projection, learned marker stem and five-class head) into echo_routing.vit.target_patch_readout.ReadoutMViT with the
run's readout (cls, global_patches or target_patches); nothing is reinitialised and no parameter is added. Update u reads
row u of the source round's frozen manifest RV_s{seed}.npz in its Ordered image order: 8 windows of 16 frames decoded with
load_frame_u8, whose pixel digests must equal the manifest's. The recipe is the source round's: batch-mean cross-entropy,
AdamW with the encoder (lr 0 and requires_grad False for 375 updates, then 1e-5 x the cosine factor) and the marker stem
and head (1e-4 x lr_factor), clipping at 1, fp32 with TF32 off, deterministic algorithms with warnings, final weights
after 3,750 updates. A new optimiser and update count start at 0. The full resume state is saved after every epoch, and
the run manifest records every resume. Outputs go only to this round's directories; no development data is read.

  python scripts/train_target_patch_readout.py --config configs/experiments/target_patch_readout_v1.yaml --check
  python scripts/train_target_patch_readout.py --config configs/experiments/target_patch_readout_v1.yaml --readout target_patches --seed 0
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.train_raw_video_context as rvc  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_raw_video_context import allowlist, environment, frame_dirs  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths, params_digest  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
import echo_routing.vit.target_patch_readout as tp  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_config, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402
from echo_routing.temporal.joint_visual import crop_uint8, gamma_lut, load_frame_u8  # noqa: E402

GROUP = "Ordered context"                     # the only image order of this round
TRAINING_KEYS = ("task_sheet", "source", "readouts", "readout", "training", "determinism")   # sections a run identity covers


class Ctx:
    """This round's configuration, the source round's (read only) and this round's own output directories."""

    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.tp = cfg, config, cfg["target_patch_readout"]
        self.paths = load_paths(); self.root = self.paths.repo_root; src = self.tp["source"]
        if sha256_file(self.root / src["config"]["path"]) != src["config"]["sha256"]:
            raise SystemExit("the source round's configuration differs from its pin")
        self.rv = load_config(self.root / src["config"]["path"])["raw_video_context"]
        self.out = self.root / self.tp["outputs"]["root"]; self.ckpt_root = self.root / self.tp["outputs"]["checkpoints"]
        tr = self.tp["training"]; self.total, self.warmup, self.o = int(tr["total_updates"]), int(tr["warmup_updates"]), tr["optimiser"]
        if self.o != self.rv["training"]["optimiser"] or (self.total, self.warmup) != (int(self.rv["training"]["total_updates"]), int(self.rv["training"]["warmup_updates"])):
            raise SystemExit("the optimiser or schedule differs from the source round")
        self.seeds = [int(s) for s in tr["seeds"]]; self._fp = None

    def manifest(self, seed: int) -> tuple[dict, str]:
        pin = self.tp["source"]["manifests"]["files"][seed]; path = self.root / pin["path"]
        if (got := sha256_file(path)) != pin["sha256"]:
            raise SystemExit(f"{path.name}: sha256 {got[:12]} differs from the pin")
        with np.load(path) as z:
            return {k: z[k] for k in z.files}, got

    def source_state(self, seed: int) -> tuple[dict, str]:
        pin = self.tp["source"]["weights"][seed]; path = self.root / pin["path"]
        if (got := sha256_file(path)) != pin["sha256"]:
            raise SystemExit(f"{path}: sha256 {got[:12]} differs from the pin")
        return torch.load(path, map_location="cpu", weights_only=True), got

    def epoch_digests(self, seed: int) -> list[str]:
        pin = self.tp["source"]["manifests"]["index"]
        if sha256_file(self.root / pin["path"]) != pin["sha256"]:
            raise SystemExit("the source manifest index differs from its pin")
        return json.loads((self.root / pin["path"]).read_text())["seeds"][str(seed)]["epoch_digests"][GROUP]

    def frame_paths(self) -> FramePaths:
        if self._fp is None:
            train = set(allowlist(self.rv, self.root))
            vids = {str(v) for s in self.seeds for v in self.manifest(s)[0]["videos"].tolist()}
            if vids - train:
                raise SystemExit("a manifest recording is not a training clip")
            self._fp = FramePaths(frame_dirs(self.paths, self.rv, vids))
        return self._fp


def code_digest() -> str:
    """The readout and model modules and this script and the source trainer in full, plus the shared helpers they call."""
    h = hashlib.sha256()
    for p in (tp.__file__, tm.__file__, __file__, rvc.__file__):
        h.update(Path(p).read_bytes())
    for fn in (load_frame_u8, crop_uint8, gamma_lut, lr_factor, FramePaths, list_frame_paths, frame_dirs):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()


def identity(ctx: Ctx, readout: str, seed: int, manifest_sha: str, source_sha: str) -> str:
    payload = {"experiment": ctx.cfg.get("experiment"), "target_patch_readout": {k: ctx.tp[k] for k in TRAINING_KEYS},
               "source_training": {k: ctx.rv[k] for k in rvc.TRAINING_KEYS}, "readout": readout, "seed": int(seed), "manifest_sha256": manifest_sha,
               "source_weights_sha256": source_sha, "epoch_digests": ctx.epoch_digests(seed), "environment": environment(), "total_updates": ctx.total,
               "code": code_digest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def run_key(readout: str, seed: int) -> str:
    return f"{readout}_s{seed}"


def make_model(state: dict, readout: str, device) -> tp.ReadoutMViT:
    return tp.build_readout_model(readout, state).to(device)


def train(ctx: Ctx, readout: str, seed: int, device, workers: int) -> None:
    m, msha = ctx.manifest(seed); state, ssha = ctx.source_state(seed); ident = identity(ctx, readout, seed, msha, ssha); key = run_key(readout, seed)
    run_dir = ctx.out / "train" / key; ckpt = ctx.ckpt_root / key; run_dir.mkdir(parents=True, exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)
    if rvc.finished(run_dir / "run_manifest.json", ident):
        print(f"[{key}] already finished with identity {ident[:12]}", flush=True); return
    model = make_model(state, readout, device); opt = rvc.make_optimizer(model, ctx.o)
    start = {"encoder": params_digest(model.encoder.named_parameters()), "all": params_digest(model.named_parameters())}
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    resumes = json.loads((run_dir / "resumes.json").read_text()) if (run_dir / "resumes.json").is_file() else []
    done, log = (0, [])
    if (ckpt / "resume.pt").is_file():
        done, log = rvc.load_resume(ckpt / "resume.pt", model, opt, ident)
        resumes.append({"at_update": done, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}); (run_dir / "resumes.json").write_text(json.dumps(resumes))
    print(f"[{key}] identity {ident[:12]}; start at update {done}/{ctx.total}", flush=True)
    fp = ctx.frame_paths(); upe = ctx.total // 10; digests = ctx.epoch_digests(seed); t0 = time.time(); acc = []; checked = 0; warm_end = None
    torch.cuda.reset_peak_memory_stats(device)
    for batch in rvc.loader(m, GROUP, fp, list(range(done, ctx.total)), workers):
        u = int(batch[0]); rvc.check_digests(m, GROUP, u, batch[4]); checked += rvc.WINDOWS
        lrs = rvc.apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        acc.append(rvc.train_step(model, opt, batch, lrs, ctx.o, device)); done = u + 1
        if done == ctx.warmup:
            warm_end = {"encoder_unchanged_through_warmup": params_digest(model.encoder.named_parameters()) == start["encoder"]}
        if done % 25 == 0:
            (run_dir / "progress.json").write_text(json.dumps({"identity": ident, "updates": done, "total": ctx.total, "seconds": time.time() - t0,
                                                               "loss": acc[-1]["loss"], "lr_encoder": lrs[0], "lr_new": lrs[1]}))
        if done % upe == 0:
            rec = {"epoch": done // upe - 1, "updates": done, "loss": float(np.mean([a["loss"] for a in acc])),
                   "train_accuracy": sum(a["correct"] for a in acc) / sum(a["windows"] for a in acc),
                   "grad_norm_mean": float(np.mean([a["grad_norm"] for a in acc])), "grad_norm_max": float(max(a["grad_norm"] for a in acc)),
                   "lr_encoder_last": acc[-1]["lr_encoder"], "lr_new_last": acc[-1]["lr_new"], "seconds_this_process": time.time() - t0,
                   "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20, "windows_digest_checked": checked,
                   "input_digest": digests[done // upe - 1], **({"warmup_end": warm_end} if warm_end else {})}
            log.append(rec); rvc.save_resume(ckpt / "resume.pt", model, opt, done, ident, log)
            with (run_dir / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{key}] epoch {rec['epoch']} {done}/{ctx.total} loss {rec['loss']:.4f} acc {rec['train_accuracy']:.3f} "
                  f"grad {rec['grad_norm_mean']:.2f}/{rec['grad_norm_max']:.2f} {rec['seconds_this_process']:.0f}s", flush=True)
            acc, checked, warm_end = [], 0, None
    if done != ctx.total:
        raise SystemExit(f"stopped at update {done} of {ctx.total}")
    final = ckpt / "final.pt"; torch.save(model.state_dict(), final)
    man = {"identity": ident, "readout": readout, "key": key, "seed": seed, "config": ctx.config, "config_sha256": sha256_file(ctx.root / ctx.config),
           "manifest_sha256": msha, "source_weights_sha256": ssha, "environment": environment(), "code": code_digest(), "updates": done, "epochs": log,
           "start": start, "end": {"encoder": params_digest(model.encoder.named_parameters())}, "interrupted": bool(resumes), "resumes": resumes,
           "final_checkpoint": str(final), "final_sha256": sha256_file(final),
           "parameters": {"encoder": int(sum(p.numel() for p in model.encoder_parameters())), "new": int(sum(p.numel() for p in model.new_parameters()))},
           "seconds_last_process": time.time() - t0}
    (run_dir / "run_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(f"[{key}] done: {done} updates, final {man['final_sha256'][:12]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--readout", choices=tp.READOUTS); ap.add_argument("--seed", type=int)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); ctx = Ctx(cfg, a.config)
    rvc.determinism(); device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    if a.check:
        from scripts.check_target_patch_readout import run_checks
        run_checks(ctx, device, a.workers); return
    if a.readout is None or a.seed not in ctx.seeds:
        raise SystemExit("--readout and a configured --seed are required")
    train(ctx, a.readout, a.seed, device, a.workers)


if __name__ == "__main__":
    main()
