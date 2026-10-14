#!/usr/bin/env python
"""Train one arm and seed of the video dense supervision round, or run its checks before the queue
(docs/superpowers/plans/2026-10-13-video-dense-supervision.md, Task 3; configs/experiments/video_dense_supervision_v1.yaml).

Every run strictly loads its seed's V01 Ordered final weights into echo_routing.vit.video_dense_supervision.DenseMViT;
nothing is reinitialised and no parameter is added. Update u reads row u of V01's frozen manifest RV_s{seed}.npz in its
Ordered image order: 8 windows of 16 frames decoded with load_frame_u8, whose pixel digests must equal the manifest's,
and row u of this round's AUX_s{seed}.npz. Every arm runs forward_with_time (CLS logits first, then the 8 time logits) and
the loss (L_center + lambda L_time) / (1 + lambda): lambda 0 with the aligned labels (CLS continuation), 0.5 with the
aligned labels, or 0.5 with the shuffled labels. The recipe is V01's: AdamW with the encoder (lr 0 and requires_grad False
for 375 updates, then 1e-5 x the cosine factor) and the marker stem and head (1e-4 x lr_factor), clipping at 1, fp32 with
TF32 off, deterministic algorithms with warnings, final weights after 3,750 updates. torch.manual_seed(seed) precedes the
first update. The full resume state is saved after every epoch. No development data is read.

  python scripts/train_video_dense_supervision.py --config configs/experiments/video_dense_supervision_v1.yaml --check
  python scripts/train_video_dense_supervision.py --config configs/experiments/video_dense_supervision_v1.yaml --arm aligned_dense --seed 0
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
import scripts.prepare_video_dense_supervision as pd  # noqa: E402
import scripts.train_raw_video_context as rvc  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_raw_video_context import allowlist, environment, frame_dirs  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths, params_digest  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
import echo_routing.vit.target_patch_readout as tp  # noqa: E402
import echo_routing.vit.video_dense_supervision as vds  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_config, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402
from echo_routing.temporal.joint_visual import crop_uint8, gamma_lut, load_frame_u8  # noqa: E402

GROUP = "Ordered context"                     # V01's image order; the only one of this round
TRAINING_KEYS = ("task_sheet", "source", "arms", "model", "aux", "data", "training", "determinism", "budget")
CODE_FILES = (Path(vds.__file__), Path(tm.__file__), Path(tp.__file__), Path(__file__), Path(pd.__file__), Path(rvc.__file__))
finished, make_optimizer, apply_schedule, schedule = rvc.finished, rvc.make_optimizer, rvc.apply_schedule, rvc.schedule
save_resume, load_resume, determinism, loader, check_digests, WINDOWS = rvc.save_resume, rvc.load_resume, rvc.determinism, rvc.loader, rvc.check_digests, rvc.WINDOWS


# ----------------------------------------------------------------------------------------------- context
class Ctx:
    """This round's configuration, V01's (read only), the pinned training inputs and this round's output directories."""

    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.vd = cfg, config, cfg["video_dense_supervision"]
        self.paths = load_paths(); self.root = self.paths.repo_root; src = self.vd["source"]
        self.rv = load_config(pd.pinned(self.root, src["raw_video_config"]))["raw_video_context"]
        self.out = self.root / self.vd["outputs"]["root"]; self.ckpt_root = self.root / self.vd["outputs"]["checkpoints"]
        self.ledger = self.root / self.vd["budget"]["ledger"]
        tr = self.vd["training"]; self.total, self.warmup, self.o = int(tr["total_updates"]), int(tr["warmup_updates"]), tr["optimiser"]
        if self.o != self.rv["training"]["optimiser"] or (self.total, self.warmup) != (int(self.rv["training"]["total_updates"]), int(self.rv["training"]["warmup_updates"])):
            raise SystemExit("the optimiser or schedule differs from the source round")
        self.seeds = [int(s) for s in tr["seeds"]]; self._fp = None

    def manifest(self, seed: int) -> tuple[dict, str]:
        path = pd.pinned(self.root, self.vd["source"]["rv_manifests"]["files"][seed])
        return pd.load_npz(path), sha256_file(path)

    def aux(self, seed: int) -> tuple[dict, str]:
        """The seed's label file; its digest must equal the preparation index, whose RV and SC digests must equal the pins."""
        mdir = self.out / "manifests"; index = json.loads((mdir / "index.json").read_text()); e = index["seeds"][str(seed)]; src = self.vd["source"]
        if e["rv_sha256"] != src["rv_manifests"]["files"][seed]["sha256"] or e["sc_sha256"] != src["sc_manifests"]["files"][seed]["sha256"]:
            raise SystemExit(f"seed {seed}: the label file was prepared from other window lists")
        if index["anchors"] != list(self.vd["model"]["anchors"]) or (got := sha256_file(mdir / e["file"])) != e["sha256"]:
            raise SystemExit(f"seed {seed}: the label file differs from the preparation index")
        return pd.load_npz(mdir / e["file"]), got

    def source_state(self, seed: int) -> tuple[dict, str]:
        path = pd.pinned(self.root, self.vd["source"]["weights"][seed])
        return torch.load(path, map_location="cpu", weights_only=True), sha256_file(path)

    def epoch_digests(self, seed: int) -> list[str]:
        path = pd.pinned(self.root, self.vd["source"]["rv_manifests"]["index"])
        return json.loads(path.read_text())["seeds"][str(seed)]["epoch_digests"][GROUP]

    def frame_paths(self) -> FramePaths:
        if self._fp is None:
            train = set(allowlist(self.rv, self.root)); vids = {str(v) for s in self.seeds for v in self.manifest(s)[0]["videos"].tolist()}
            pd.check_allowlist(vids, train); self._fp = FramePaths(frame_dirs(self.paths, self.rv, vids))
        return self._fp


# ----------------------------------------------------------------------------------------------- identity, labels, ledger
def code_digest() -> str:
    """The dense, model and token-grid modules, the preparation, this script and V01's trainer in full, plus the shared helpers."""
    h = hashlib.sha256()
    for p in CODE_FILES:
        h.update(Path(p).read_bytes())
    for fn in (load_frame_u8, crop_uint8, gamma_lut, lr_factor, FramePaths, list_frame_paths, frame_dirs):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()


def identity(ctx, arm: str, seed: int, manifest_sha: str, aux_sha: str, source_sha: str) -> str:
    payload = {"experiment": ctx.cfg.get("experiment"), "video_dense_supervision": {k: ctx.vd[k] for k in TRAINING_KEYS},
               "source_training": {k: ctx.rv[k] for k in rvc.TRAINING_KEYS}, "arm": arm, "seed": int(seed), "manifest_sha256": manifest_sha,
               "aux_sha256": aux_sha, "source_weights_sha256": source_sha, "epoch_digests": ctx.epoch_digests(seed), "environment": environment(),
               "total_updates": ctx.total, "code": code_digest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def run_key(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


def arm_labels(aux: dict, arm: str) -> np.ndarray:
    """(U, W, 8) labels of an arm: the aligned labels for CLS continuation (weight 0) and aligned supervision, else the shuffled."""
    if arm in ("center_only", "aligned_dense"):
        return aux["aligned"]
    if arm == "shuffled_dense":
        return aux["shuffled"]
    raise ValueError(f"unknown arm {arm!r}")


def arm_lambda(vd: dict, arm: str) -> float:
    return float(vd["aux"]["lambda"][arm])


def ledger_hours(path: Path) -> float:
    """Every recorded GPU job of the round, passed or failed."""
    if not Path(path).is_file():
        return 0.0
    return float(sum(json.loads(x)["hours"] for x in Path(path).read_text().splitlines() if x.strip()))


def ledger_append(path: Path, job: str, started: str, hours: float, ok: bool, note: str = "") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as fh:
        fh.write(json.dumps({"job": job, "started": started, "finished": now(), "hours": float(hours), "ok": bool(ok), "note": note}) + "\n")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ----------------------------------------------------------------------------------------------- one update
def make_model(state: dict, device) -> vds.DenseMViT:
    return vds.build_dense_model(state).to(device)


def train_step(model: vds.DenseMViT, opt, batch, aux_u: np.ndarray, aligned_u: np.ndarray, lam: float, lrs: tuple[float, float], o: dict,
               device, keep: bool = False) -> dict:
    """One update: CLS logits and time logits, the arm's loss, clipping at grad_clip and the AdamW step."""
    u8, slots, target = batch[1], batch[2], batch[3]
    model.train()
    z_cls, z_time = model.forward_with_time(tm.to_rgb(u8.to(device, non_blocking=True)), tm.marker_planes(slots.to(device)))
    aux = torch.from_numpy(np.asarray(aux_u, np.int64)).to(device); y = target.to(device)
    loss, parts = vds.dense_loss(z_cls, z_time, y, aux, lam)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    opt.zero_grad(set_to_none=True); loss.backward()
    params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    gn = float(torch.nn.utils.clip_grad_norm_(params, float(o["grad_clip"])))
    if not np.isfinite(gn):
        raise FloatingPointError("non-finite gradient norm")
    opt.step()
    pt = z_time.detach().argmax(-1).cpu().numpy()
    rec = {"loss": float(loss.detach()), **parts, "grad_norm": gn, "lr_encoder": lrs[0], "lr_new": lrs[1],
           "correct": int((z_cls.detach().argmax(1).cpu() == target).sum()), "windows": int(target.numel()),
           "time_correct_own": int((pt == np.asarray(aux_u)).sum()), "time_correct_aligned": int((pt == np.asarray(aligned_u)).sum()), "anchors": int(pt.size)}
    if keep:
        rec["outputs"] = (z_cls.detach().cpu(), z_time.detach().cpu())
    return rec


def epoch_record(acc: list[dict], done: int, upe: int, t0: float, device, checked: int, digest: str, warm_end: dict | None) -> dict:
    s = lambda k: sum(a[k] for a in acc)  # noqa: E731
    return {"epoch": done // upe - 1, "updates": done, "loss": float(np.mean([a["loss"] for a in acc])), "loss_center": float(np.mean([a["center"] for a in acc])),
            "loss_time": float(np.mean([a["time"] for a in acc])), "train_accuracy": s("correct") / s("windows"),
            "time_accuracy_own_labels": s("time_correct_own") / s("anchors"), "time_accuracy_aligned_labels": s("time_correct_aligned") / s("anchors"),
            "grad_norm_mean": float(np.mean([a["grad_norm"] for a in acc])), "grad_norm_max": float(max(a["grad_norm"] for a in acc)),
            "lr_encoder_last": acc[-1]["lr_encoder"], "lr_new_last": acc[-1]["lr_new"], "seconds_this_process": time.time() - t0,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if torch.cuda.is_available() else None,
            "windows_digest_checked": checked, "input_digest": digest, **({"warmup_end": warm_end} if warm_end else {})}


# ----------------------------------------------------------------------------------------------- formal run
def train(ctx: Ctx, arm: str, seed: int, device, workers: int) -> None:
    m, msha = ctx.manifest(seed); aux, asha = ctx.aux(seed); state, ssha = ctx.source_state(seed); ident = identity(ctx, arm, seed, msha, asha, ssha)
    key = run_key(arm, seed); lam = arm_lambda(ctx.vd, arm); labels = arm_labels(aux, arm)
    if labels.shape[:2] != m["target"].shape:
        raise SystemExit(f"{key}: the label file does not cover the manifest's windows")
    run_dir = ctx.out / "train" / key; ckpt = ctx.ckpt_root / key; run_dir.mkdir(parents=True, exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)
    if finished(run_dir / "run_manifest.json", ident):
        print(f"[{key}] already finished with identity {ident[:12]}", flush=True); return
    model = make_model(state, device); opt = make_optimizer(model, ctx.o)
    start = {"encoder": params_digest(model.encoder.named_parameters()), "all": params_digest(model.named_parameters())}
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    resumes = json.loads((run_dir / "resumes.json").read_text()) if (run_dir / "resumes.json").is_file() else []
    done, log = 0, []
    if (ckpt / "resume.pt").is_file():
        done, log = load_resume(ckpt / "resume.pt", model, opt, ident)
        resumes.append({"at_update": done, "time": now()}); (run_dir / "resumes.json").write_text(json.dumps(resumes))
    print(f"[{key}] identity {ident[:12]}; lambda {lam}; start at update {done}/{ctx.total}", flush=True)
    fp = ctx.frame_paths(); upe = ctx.total // 10; digests = ctx.epoch_digests(seed); t0 = time.time(); acc = []; checked = 0; warm_end = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader(m, GROUP, fp, list(range(done, ctx.total)), workers):
        u = int(batch[0]); check_digests(m, GROUP, u, batch[4]); checked += WINDOWS
        lrs = apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        acc.append(train_step(model, opt, batch, labels[u], aux["aligned"][u], lam, lrs, ctx.o, device)); done = u + 1
        if done == ctx.warmup:
            warm_end = {"encoder_unchanged_through_warmup": params_digest(model.encoder.named_parameters()) == start["encoder"]}
        if done % 25 == 0:
            (run_dir / "progress.json").write_text(json.dumps({"identity": ident, "updates": done, "total": ctx.total, "seconds": time.time() - t0,
                                                               "loss": acc[-1]["loss"], "lr_encoder": lrs[0], "lr_new": lrs[1]}))
        if done % upe == 0:
            rec = epoch_record(acc, done, upe, t0, device, checked, digests[done // upe - 1], warm_end)
            log.append(rec); save_resume(ckpt / "resume.pt", model, opt, done, ident, log)
            with (run_dir / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{key}] epoch {rec['epoch']} {done}/{ctx.total} loss {rec['loss']:.4f} (centre {rec['loss_center']:.4f}, time {rec['loss_time']:.4f}) "
                  f"acc {rec['train_accuracy']:.3f} time acc {rec['time_accuracy_own_labels']:.3f} grad {rec['grad_norm_mean']:.2f}/{rec['grad_norm_max']:.2f} "
                  f"{rec['seconds_this_process']:.0f}s", flush=True)
            acc, checked, warm_end = [], 0, None
    if done != ctx.total:
        raise SystemExit(f"stopped at update {done} of {ctx.total}")
    final = ckpt / "final.pt"; torch.save(model.state_dict(), final)
    man = {"identity": ident, "arm": arm, "key": key, "seed": seed, "lambda": lam, "config": ctx.config, "config_sha256": sha256_file(ctx.root / ctx.config),
           "manifest_sha256": msha, "aux_sha256": asha, "source_weights_sha256": ssha, "environment": environment(), "code": code_digest(), "updates": done,
           "epochs": log, "start": start, "end": {"encoder": params_digest(model.encoder.named_parameters())}, "interrupted": bool(resumes), "resumes": resumes,
           "final_checkpoint": str(final), "final_sha256": sha256_file(final),
           "parameters": {"encoder": int(sum(p.numel() for p in model.encoder_parameters())), "new": int(sum(p.numel() for p in model.new_parameters()))},
           "seconds_last_process": time.time() - t0}
    (run_dir / "run_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(f"[{key}] done: {done} updates, final {man['final_sha256'][:12]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--arm", choices=vds.ARMS); ap.add_argument("--seed", type=int)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); ctx = Ctx(cfg, a.config)
    determinism(); device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    if a.check:
        from scripts.check_video_dense_supervision import run_checks
        run_checks(ctx, device, a.workers); return
    if a.arm is None or a.seed not in ctx.seeds:
        raise SystemExit("--arm and a configured --seed are required")
    train(ctx, a.arm, a.seed, device, a.workers)


if __name__ == "__main__":
    main()
