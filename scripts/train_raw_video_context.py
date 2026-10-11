#!/usr/bin/env python
"""Train one group and seed of the raw-video context round, or run the checks before the queue
(docs/superpowers/plans/2026-10-10-raw-video-recognition.md, Tasks 1-2; configs/experiments/raw_video_context_v1.yaml).

The model is echo_routing.vit.target_mvit.TargetMViT: EchoPrime's MViTv2-S with its 512-d projection, a new
Linear(512, 5) and a zero-initialised marker stem, started from the seed's saved initial state. Update u reads row u of
the seed's frozen manifest (RV_s{seed}.npz): 8 windows of 16 frames in the group's slot order, decoded with
load_frame_u8, normalised with EchoPrime's 0-255 mean and std. Every window's pixel digest, computed from the decoded
frames, must equal the prepared one. The loss is the batch-mean cross-entropy of the target class. AdamW holds two
groups from update 0: the encoder (lr 0 and requires_grad False for the first 375 updates, then 1e-5 x the cosine
factor) and the new parameters (1e-4 x lr_factor). Gradients are clipped to norm 1. fp32, TF32 off, cuDNN
deterministic, benchmark off, deterministic algorithms with warnings. No development data is read. After every epoch
the full resume state is saved; the final weights after 3,750 updates are the model.

  --check   the plan's Task 1-2 checks: encoder parity and strict load, the 16 check windows (inputs, pixels, markers,
            supervision), isolation, the marker branch, hook hygiene, reproducibility and resume, the unfreeze boundary
            on a short schedule, the 300-update fits, the formal timing and the input-token figure.

  python scripts/train_raw_video_context.py --config configs/experiments/raw_video_context_v1.yaml --check
  python scripts/train_raw_video_context.py --config configs/experiments/raw_video_context_v1.yaml --group ordered --seed 0
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
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_raw_video_context import allowlist, environment, frame_dirs  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths, params_digest  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import echo_routing.vit.target_mvit as tm  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402
from echo_routing.temporal.joint_visual import crop_uint8, gamma_lut, load_frame_u8  # noqa: E402

GROUP_OF = {v: k for k, v in tm.KEYS.items()}          # target_only -> Target-only, ...
WINDOWS = 8


# ----------------------------------------------------------------------------------------------- context
class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.cfg, self.config, self.rv = cfg, config, cfg["raw_video_context"]
        self.paths = load_paths(); self.root = self.paths.repo_root
        self.out = self.root / self.rv["outputs"]["root"]; self.ckpt_root = self.root / self.rv["outputs"]["checkpoints"]
        self.mdir = self.out / "manifests"; self.index = json.loads((self.mdir / "index.json").read_text())
        tr = self.rv["training"]; self.total, self.warmup = int(tr["total_updates"]), int(tr["warmup_updates"]); self.o = tr["optimiser"]
        self.checkpoint = self.root / self.rv["source"]["echoprime"]["checkpoint"]["path"]; self._fp = None

    def entry(self, seed: int) -> dict:
        return self.index["seeds"][str(seed)]

    def manifest(self, seed: int) -> tuple[dict, str]:
        e = self.entry(seed)["files"]["manifest"]; path = self.mdir / e["file"]
        if (got := sha256_file(path)) != e["sha256"]:
            raise SystemExit(f"{path.name}: sha256 {got[:12]} differs from index.json")
        with np.load(path) as z:
            return {k: z[k] for k in z.files}, got

    def init(self, seed: int) -> tuple[dict, str]:
        e = self.entry(seed)["files"]["init"]; path = self.mdir / e["file"]
        if (got := sha256_file(path)) != e["sha256"]:
            raise SystemExit(f"{path.name}: sha256 {got[:12]} differs from index.json")
        return torch.load(path, map_location="cpu", weights_only=True), got

    def frame_paths(self) -> FramePaths:
        if self._fp is None:
            train = set(allowlist(self.rv, self.root))
            vids = {str(v) for s in self.index["seeds"] for v in self.manifest(int(s))[0]["videos"].tolist()}
            if vids - train:
                raise SystemExit("a manifest recording is not a training clip")
            self._fp = FramePaths(frame_dirs(self.paths, self.rv, vids))
        return self._fp


def determinism() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest"); torch.use_deterministic_algorithms(True, warn_only=True)


def make_model(ctx: Ctx, init: dict, device) -> tm.TargetMViT:
    return tm.build_model(ctx.checkpoint, init).to(device)


def code_digest() -> str:
    """Sources of the model, the input functions and the training step (unrelated edits elsewhere do not count)."""
    h = hashlib.sha256(Path(tm.__file__).read_bytes())
    for fn in (load_frame_u8, crop_uint8, gamma_lut, lr_factor, window_inputs, make_optimizer, schedule, apply_schedule, train_step, save_resume, load_resume):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()


def identity(ctx: Ctx, group: str, seed: int, manifest_sha: str, init_sha: str) -> str:
    payload = {"experiment": ctx.cfg.get("experiment"), "raw_video_context": ctx.rv, "group": group, "seed": int(seed),
               "manifest_sha256": manifest_sha, "init_sha256": init_sha, "checkpoint_sha256": ctx.rv["source"]["echoprime"]["checkpoint"]["sha256"],
               "environment": environment(), "total_updates": ctx.total, "code": code_digest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------- inputs
def window_inputs(m: dict, order: np.ndarray, frame_paths, pairs: list[tuple[int, int]]):
    """Frames of the windows (u, w) in the group's slot order (uint8 (N, 16, 224, 224, 3)), their marker slots, targets
    and pixel digests computed from the decoded frames. Nothing but references, slot order and markers is read."""
    frames = np.empty((len(pairs), tm.N_SLOTS, 224, 224, 3), np.uint8); digests = np.empty((len(pairs), 32), np.uint8); cache = {}
    for k, (u, w) in enumerate(pairs):
        hashes = []
        for slot in range(tm.N_SLOTS):
            j = int(order[u, w, slot]); ref = (int(m["vid"][u, w, j]), int(m["frame"][u, w, j]), round(float(m["gamma"][u, w, j]), 6))
            if ref not in cache:
                paths = frame_paths(str(m["videos"][ref[0]])); img = load_frame_u8(paths[min(ref[1], len(paths) - 1)], ref[2])
                cache[ref] = (img, tm.frame_hash(img))
            frames[k, slot], h = cache[ref]
            hashes.append(h)
        digests[k] = np.frombuffer(tm.window_digest(hashes), np.uint8)
    us, ws = [p[0] for p in pairs], [p[1] for p in pairs]
    return (torch.from_numpy(frames), torch.from_numpy(m["marker"][us, ws].astype(np.int64)), torch.from_numpy(m["target"][us, ws].astype(np.int64)),
            torch.from_numpy(digests))


class UpdateInputs(Dataset):
    """Item i: update u's 8 windows (window_inputs) for one group."""

    def __init__(self, m: dict, group: str, frame_paths, updates: list[int]) -> None:
        self.m, self.order, self.fp, self.updates = m, m[f"order_{tm.KEYS[group]}"], frame_paths, list(updates)

    def __len__(self) -> int:
        return len(self.updates)

    def __getitem__(self, i: int):
        u = self.updates[i]
        return (u, *window_inputs(self.m, self.order, self.fp, [(u, w) for w in range(WINDOWS)]))


def loader(m: dict, group: str, frame_paths, updates: list[int], workers: int) -> DataLoader:
    return DataLoader(UpdateInputs(m, group, frame_paths, updates), batch_size=None, num_workers=workers,
                      prefetch_factor=2 if workers else None, pin_memory=True, persistent_workers=False)


def check_digests(m: dict, group: str, u: int, digests: torch.Tensor) -> None:
    if not np.array_equal(digests.numpy(), m[f"digest_{tm.KEYS[group]}"][u]):
        raise SystemExit(f"update {u}: the decoded pixels differ from the prepared digests ({group})")


# ----------------------------------------------------------------------------------------------- optimisation
def make_optimizer(model: tm.TargetMViT, o: dict) -> torch.optim.AdamW:
    return torch.optim.AdamW([{"params": model.encoder_parameters(), "lr": 0.0, "name": "encoder"},
                              {"params": model.new_parameters(), "lr": 0.0, "name": "new"}],
                             betas=tuple(float(b) for b in o["betas"]), eps=float(o["eps"]), weight_decay=float(o["weight_decay"]))


def schedule(u: int, warmup: int, total: int, o: dict) -> tuple[float, float]:
    """(encoder lr, new-parameter lr) of update u."""
    f = lr_factor(int(u), int(warmup), int(total))
    return (0.0 if u < warmup else float(o["lr_encoder"]) * f), float(o["lr_new"]) * f


def apply_schedule(model: tm.TargetMViT, opt, u: int, warmup: int, total: int, o: dict) -> tuple[float, float]:
    enc, new = schedule(u, warmup, total, o)
    model.set_encoder_trainable(u >= warmup)
    opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = enc, new
    return enc, new


def train_step(model: tm.TargetMViT, opt, batch, lrs: tuple[float, float], o: dict, device) -> dict:
    u8, slots, target = batch[1], batch[2], batch[3]
    model.train()
    logits = model(tm.to_rgb(u8.to(device, non_blocking=True)), tm.marker_planes(slots.to(device)))
    loss = F.cross_entropy(logits, target.to(device))
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    opt.zero_grad(set_to_none=True); loss.backward()
    params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
    gn = float(torch.nn.utils.clip_grad_norm_(params, float(o["grad_clip"])))
    if not np.isfinite(gn):
        raise FloatingPointError("non-finite gradient norm")
    opt.step()
    return {"loss": float(loss.detach()), "grad_norm": gn, "lr_encoder": lrs[0], "lr_new": lrs[1],
            "correct": int((logits.detach().argmax(1).cpu() == target).sum()), "windows": int(target.numel())}


def save_resume(path: Path, model, opt, done: int, ident: str, log: list) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"identity": ident, "updates": done, "model": model.state_dict(), "opt": opt.state_dict(), "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all(), "log": log}, tmp)
    tmp.replace(path)


def load_resume(path: Path, model, opt, ident: str) -> tuple[int, list]:
    st = torch.load(path, map_location="cpu", weights_only=False)
    if st["identity"] != ident:
        raise SystemExit(f"{path} belongs to another run identity")
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
    torch.set_rng_state(st["rng_cpu"]); torch.cuda.set_rng_state_all(st["rng_cuda"])
    return int(st["updates"]), list(st["log"])


# ----------------------------------------------------------------------------------------------- formal run
def run_key(group: str, seed: int) -> str:
    return f"{tm.KEYS[group]}_s{seed}"


def finished(man_path: Path, ident: str) -> bool:
    if not man_path.is_file():
        return False
    man = json.loads(man_path.read_text()); final = Path(man.get("final_checkpoint", "/nonexistent"))
    return man.get("identity") == ident and final.is_file() and sha256_file(final) == man.get("final_sha256")


def train(ctx: Ctx, group: str, seed: int, device, workers: int) -> None:
    m, msha = ctx.manifest(seed); init, isha = ctx.init(seed); ident = identity(ctx, group, seed, msha, isha); key = run_key(group, seed)
    run_dir = ctx.out / "train" / key; ckpt = ctx.ckpt_root / key; run_dir.mkdir(parents=True, exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)
    if finished(run_dir / "run_manifest.json", ident):
        print(f"[{key}] already finished with identity {ident[:12]}", flush=True); return
    model = make_model(ctx, init, device); opt = make_optimizer(model, ctx.o)
    start = {"encoder": params_digest(model.encoder.named_parameters()), "all": params_digest(model.named_parameters())}
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    done, log = load_resume(ckpt / "resume.pt", model, opt, ident) if (ckpt / "resume.pt").is_file() else (0, [])
    print(f"[{key}] identity {ident[:12]}; start at update {done}/{ctx.total}", flush=True)
    fp = ctx.frame_paths(); upe = ctx.total // 10; t0 = time.time(); acc = []; checked = 0; warm_end = None
    torch.cuda.reset_peak_memory_stats(device)
    for batch in loader(m, group, fp, list(range(done, ctx.total)), workers):
        u = int(batch[0]); check_digests(m, group, u, batch[4]); checked += WINDOWS
        lrs = apply_schedule(model, opt, u, ctx.warmup, ctx.total, ctx.o)
        acc.append(train_step(model, opt, batch, lrs, ctx.o, device)); done = u + 1
        if done == ctx.warmup:
            warm_end = {"encoder_unchanged_through_warmup": params_digest(model.encoder.named_parameters()) == start["encoder"],
                        "marker_abs_max": float(model.marker_stem.weight.detach().abs().max())}
        if done % 25 == 0:
            (run_dir / "progress.json").write_text(json.dumps({"identity": ident, "updates": done, "total": ctx.total, "seconds": time.time() - t0,
                                                               "loss": acc[-1]["loss"], "lr_encoder": lrs[0], "lr_new": lrs[1]}))
        if done % upe == 0:
            rec = {"epoch": done // upe - 1, "updates": done, "loss": float(np.mean([a["loss"] for a in acc])),
                   "train_accuracy": sum(a["correct"] for a in acc) / sum(a["windows"] for a in acc),
                   "grad_norm_mean": float(np.mean([a["grad_norm"] for a in acc])), "grad_norm_max": float(max(a["grad_norm"] for a in acc)),
                   "lr_encoder_last": acc[-1]["lr_encoder"], "lr_new_last": acc[-1]["lr_new"], "seconds_this_process": time.time() - t0,
                   "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20, "windows_digest_checked": checked,
                   "marker_abs_max": float(model.marker_stem.weight.detach().abs().max()), "input_digest": ctx.entry(seed)["epoch_digests"][group][done // upe - 1],
                   **({"warmup_end": warm_end} if warm_end else {})}
            log.append(rec); save_resume(ckpt / "resume.pt", model, opt, done, ident, log)
            with (run_dir / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{key}] epoch {rec['epoch']} {done}/{ctx.total} loss {rec['loss']:.4f} acc {rec['train_accuracy']:.3f} "
                  f"grad {rec['grad_norm_mean']:.2f}/{rec['grad_norm_max']:.2f} {rec['seconds_this_process']:.0f}s", flush=True)
            acc, checked, warm_end = [], 0, None
    if done != ctx.total:
        raise SystemExit(f"stopped at update {done} of {ctx.total}")
    final = ckpt / "final.pt"; torch.save(model.state_dict(), final)
    man = {"identity": ident, "group": group, "key": key, "seed": seed, "config": ctx.config, "config_sha256": sha256_file(ctx.root / ctx.config),
           "manifest_sha256": msha, "init_sha256": isha, "checkpoint_sha256": ctx.rv["source"]["echoprime"]["checkpoint"]["sha256"],
           "environment": environment(), "code": code_digest(), "updates": done, "epochs": log, "start": start,
           "end": {"encoder": params_digest(model.encoder.named_parameters())}, "final_checkpoint": str(final), "final_sha256": sha256_file(final),
           "parameters": {"encoder": int(sum(p.numel() for p in model.encoder_parameters())), "new": int(sum(p.numel() for p in model.new_parameters()))},
           "seconds_last_process": time.time() - t0}
    (run_dir / "run_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(f"[{key}] done: {done} updates, final {man['final_sha256'][:12]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--group", choices=sorted(GROUP_OF)); ap.add_argument("--seed", type=int)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); ctx = Ctx(cfg, a.config)
    determinism(); device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    if a.check:
        from scripts.check_raw_video_context import run_checks
        run_checks(ctx, device, a.workers); return
    if a.group is None or a.seed not in ctx.rv["training"]["seeds"]:
        raise SystemExit("--group and a configured --seed are required")
    train(ctx, GROUP_OF[a.group], a.seed, device, a.workers)


if __name__ == "__main__":
    main()
