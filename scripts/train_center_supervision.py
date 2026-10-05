#!/usr/bin/env python
"""Train one arm (N, S or C) and seed of the target-time supervision round
(configs/experiments/ev9v_center_supervision_v1.yaml; plan Tasks 2-3).

The model is the E1 encoder (last four blocks, ln_post and proj trainable) with the two-layer temporal head of
echo_routing.vit.target_video, started from the seed's saved initial state. Each update reads the 8 windows of the frozen
manifest (SC_s{seed}.npz for S and C, N_s{seed}.npz for N): 128 frames decoded with the E1 preprocessing. The loss is
the arm's window target plus 0.5 x the frame cross-entropy, from one forward pass. AdamW (encoder 1e-5, head 1e-4,
weight decay 1e-4), gradient clipping 1.0, 375 warm-up updates then cosine to 0, 10 epochs of 375 updates, fp32 with
TF32 off. No validation data is read and the final epoch is the model. After every epoch the resume state (weights,
optimiser, scheduler, update count, RNG) is saved; an interrupted run continues from it with the same run identity.

--smoke runs the plan's checks on the first training updates instead of training: encoder parity, the trainable set,
gradients, frozen weights, finite losses, same-seed reproducibility, resume, S/C input pairing, memory and time.

  python scripts/train_center_supervision.py --config configs/experiments/ev9v_center_supervision_v1.yaml --arm C \
      --seed 0 --manifest-root runs/center_supervision_v1/manifests [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths, params_digest  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402
from echo_routing.temporal.dev_train import lr_factor  # noqa: E402
from echo_routing.temporal.joint_visual import (VisualEncoder, load_frame_u8, load_vision_transformer,  # noqa: E402
                                                trainable_blocks_from)
from echo_routing.vit.target_video import TargetTimeHead, TargetVideoModel, losses, window_target  # noqa: E402
from echo_routing.vit.target_windows import epoch_digest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class UpdateFrames(Dataset):
    """Item u: the 8 x 16 frames of update u (uint8, window order), their offsets, frame labels and target labels."""

    def __init__(self, m: dict, frame_paths, updates: list[int]) -> None:
        self.m, self.frame_paths, self.updates = m, frame_paths, list(updates)

    def __len__(self) -> int:
        return len(self.updates)

    def __getitem__(self, i: int):
        u = self.updates[i]; m = self.m; frames = []
        for vid, fi, g in zip(m["vid"][u].ravel(), m["frame"][u].ravel(), m["gamma"][u].ravel()):
            paths = self.frame_paths(str(m["videos"][vid]))
            frames.append(load_frame_u8(paths[min(int(fi), len(paths) - 1)], round(float(g), 6)))
        return (u, torch.from_numpy(np.stack(frames)), torch.from_numpy(m["rel"][u].astype(np.int64)),
                torch.from_numpy(m["labels"][u].astype(np.int64)), torch.from_numpy(m["target"][u].astype(np.int64)))


def identity(cfg: dict, arm: str, seed: int, manifest_sha: str, init_sha: str) -> str:
    payload = {"experiment": cfg.get("experiment"), "center_supervision": cfg["center_supervision"], "arm": arm, "seed": int(seed),
               "manifest_sha256": manifest_sha, "init_sha256": init_sha}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def load_inputs(cs: dict, root: Path, arm: str, seed: int, manifest_root: Path) -> tuple[dict, dict, dict]:
    index = json.loads((manifest_root / "index.json").read_text()); entry = index["seeds"][str(seed)]
    files = {k: manifest_root / v["file"] for k, v in entry["files"].items()}
    which = "N" if arm == "N" else "SC"
    for k in (which, "init"):
        if (got := sha256_file(files[k])) != entry["files"][k]["sha256"]:
            raise SystemExit(f"{files[k]}: sha256 {got[:12]} differs from index.json")
    with np.load(files[which]) as z:
        m = {k: z[k] for k in z.files}
    return m, {"manifest": str(files[which]), "manifest_sha256": entry["files"][which]["sha256"], "init": str(files["init"]),
               "init_sha256": entry["files"]["init"]["sha256"], "epoch_digests": entry["epoch_digests"][which]}, index


def build_model(cs: dict, paths, init_path: Path, device) -> tuple[TargetVideoModel, dict]:
    vt, vinfo = load_vision_transformer(paths.repo_root / "third_party" / "EchoViewCLIP", paths.repo_root / cs["inputs"]["visual_export"]["path"])
    m, t = cs["model"], cs["model"]["temporal"]
    encoder = VisualEncoder(vt, trainable_blocks=trainable_blocks_from(m["unfreeze"], len(vt.transformer.resblocks)), chunk=int(m["encoder_chunk"]))
    head = TargetTimeHead(512, int(t["layers"]), int(t["heads"]), int(t["ffn"]), float(t["dropout"]), int(cs["window"]["max_offset"]),
                          len(cs["classes"]), float(m["position_init_std"]))
    head.load_state_dict(torch.load(init_path, map_location="cpu", weights_only=True), strict=True)
    return TargetVideoModel(encoder, head).to(device), vinfo


def check_trainable(model: TargetVideoModel) -> list[str]:
    names = [n for n, _ in model.encoder.trainable()]
    want = {f"transformer.resblocks.{b}." for b in (8, 9, 10, 11)}
    bad = [n for n in names if not (n.startswith(("ln_post.", "proj")) or any(n.startswith(w) for w in want))]
    blocks = {int(n.split(".")[2]) for n in names if n.startswith("transformer.resblocks.")}
    if bad or blocks != {8, 9, 10, 11} or not any(n.startswith("ln_post.") for n in names) or "proj" not in names:
        raise SystemExit(f"trainable encoder set is not blocks 8-11, ln_post and proj: {bad or sorted(blocks)}")
    return names


@torch.no_grad()
def parity(model: TargetVideoModel, cs: dict, paths, frame_paths, first_clip: str, device) -> dict:
    """The initial encoder against the export's reference embeddings (fp32) and the E1 cache (fp16) of a training clip."""
    model.encoder.eval(); ref = json.loads((paths.repo_root / cs["inputs"]["visual_reference"]["path"]).read_text())
    u8 = torch.from_numpy(np.stack([load_frame_u8(Path(p), 1.0) for p in ref["reference_frames"]])).to(device)
    d_ref = float((model.encoder(u8).cpu() - torch.tensor(ref["reference_embeddings"])).abs().max())
    with np.load(paths.cache_root / "features" / cs["inputs"]["cache_key"] / "orig" / f"{first_clip}.npz") as z:
        feat, fidx = z["feat"].astype(np.float32), z["frame_idx"]
    fp = frame_paths(first_clip); n = min(len(fidx), 32)
    u8 = torch.from_numpy(np.stack([load_frame_u8(fp[min(int(f), len(fp) - 1)], 1.0) for f in fidx[:n]])).to(device)
    d_cache = float(np.abs(model.encoder(u8).cpu().numpy() - feat[:n]).max())
    tol = cs["smoke"]["parity"]
    if d_ref > float(tol["reference_fp32"]) or d_cache > float(tol["cache_fp16"]):
        raise SystemExit(f"encoder parity failed: reference {d_ref:.2e}, cache {d_cache:.2e}")
    return {"reference_max_abs": d_ref, "cache_fp16_max_abs": d_cache, "cache_clip": first_clip, "cache_frames": n}


def make_optimiser(model: TargetVideoModel, cs: dict, total: int):
    o = cs["optimiser"]
    enc = [p for _, p in model.encoder.trainable()]; head = model.head_parameters()
    opt = torch.optim.AdamW([{"params": enc, "lr": float(o["lr_encoder"])}, {"params": head, "lr": float(o["lr_head"])}],
                            weight_decay=float(o["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, int(o["warmup_updates"]), total))
    return opt, sched, enc + head


def step(model, batch, arm: str, opt, sched, params, cs: dict, device) -> dict:
    _, u8, rel, labels, target = batch
    model.train()
    logits, frame_logits = model(u8.to(device, non_blocking=True), rel.to(device))
    loss, parts = losses(arm, logits, frame_logits, labels.to(device), target.to(device), float(cs["loss"]["aux_weight"]))
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(params, float(cs["optimiser"]["grad_clip"])))
    if not np.isfinite(gn):
        raise FloatingPointError("non-finite gradient norm")
    opt.step(); sched.step()
    return {"loss": float(loss.detach()), **parts, "grad_norm": gn}


def trainable_state(model: TargetVideoModel) -> dict:
    return {"encoder_trainable": {n: p.detach().cpu().clone() for n, p in model.encoder.trainable()}, "head": {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}}


def save_resume(path: Path, model, opt, sched, updates: int, ident: str, log: list) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"identity": ident, "updates": updates, **trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(), "log": log}, tmp)
    tmp.replace(path)


def load_resume(path: Path, model, opt, sched, ident: str) -> tuple[int, list]:
    st = torch.load(path, map_location="cpu", weights_only=False)
    if st["identity"] != ident:
        raise SystemExit(f"{path} belongs to another run identity")
    with torch.no_grad():
        for n, p in model.encoder.trainable():
            p.copy_(st["encoder_trainable"][n].to(p.device))
    model.head.load_state_dict(st["head"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    torch.set_rng_state(st["rng_cpu"]); torch.cuda.set_rng_state_all(st["rng_cuda"])
    return int(st["updates"]), list(st["log"])


def loader(m: dict, frame_paths, updates: list[int], workers: int) -> DataLoader:
    return DataLoader(UpdateFrames(m, frame_paths, updates), batch_size=None, num_workers=workers,
                      prefetch_factor=2 if workers else None, pin_memory=True)


def smoke(cfg, cs, paths, arm, seed, m, info, frame_paths, first_clip, device, workers, out: Path) -> None:
    sm = cs["smoke"]; k = int(sm["updates"]); total = int(cs["sampling"]["epochs"]) * int(cs["sampling"]["updates_per_epoch"])
    report = {"arm": arm, "seed": seed, **{x: info[x] for x in ("manifest", "manifest_sha256", "init", "init_sha256")}}
    runs = {}
    for name, n_upd in (("A", k), ("B", k), ("C2", k // 2)):
        model, _ = build_model(cs, paths, Path(info["init"]), device)
        if name == "A":
            report["trainable"] = check_trainable(model); report["parity"] = parity(model, cs, paths, frame_paths, first_clip, device)
            frozen = [(n, p) for n, p in model.encoder.vt.named_parameters() if not p.requires_grad]; frozen0 = params_digest(frozen)
        opt, sched, params = make_optimiser(model, cs, total); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device); stats, times = [], []
        for i, batch in enumerate(loader(m, frame_paths, list(range(n_upd)), workers)):
            t0 = time.time(); torch.cuda.synchronize()
            if name == "A" and i == 0:   # gradients of the first update, before the optimiser step
                model.train(); _, u8, rel, labels, target = batch
                logits, fl = model(u8.to(device), rel.to(device))
                loss, _ = losses(arm, logits, fl, labels.to(device), target.to(device), float(cs["loss"]["aux_weight"]))
                opt.zero_grad(set_to_none=True); loss.backward()
                named = [*model.encoder.trainable(), *(("head." + n, p) for n, p in model.head.named_parameters())]
                report["no_finite_nonzero_grad"] = [n for n, p in named if p.grad is None or not torch.isfinite(p.grad).all() or float(p.grad.abs().max()) == 0.0]
                s_t, c_t = window_target("S", labels, target), window_target("C", labels, target)
                single = labels.eq(labels[:, :1]).all(1)
                report["pairing_first_update"] = {"single_view_windows": int(single.sum()), "S_equals_C_on_single_view": bool(torch.equal(s_t[single], c_t[single])),
                                                  "mixed_windows": int((~single).sum()), "S_target_sums": [float(x) for x in s_t.sum(1)],
                                                  "S_differs_from_C_on_mixed": int((~torch.isclose(s_t[~single], c_t[~single]).all(1)).sum())}
                opt.zero_grad(set_to_none=True); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            stats.append(step(model, batch, arm, opt, sched, params, cs, device)); torch.cuda.synchronize(); times.append(time.time() - t0)
        if name == "C2":
            save_resume(out / "smoke_resume.pt", model, opt, sched, n_upd, "smoke", [])
            model2, _ = build_model(cs, paths, Path(info["init"]), device); opt2, sched2, params2 = make_optimiser(model2, cs, total)
            done, _ = load_resume(out / "smoke_resume.pt", model2, opt2, sched2, "smoke")
            for batch in loader(m, frame_paths, list(range(done, k)), workers):
                stats.append(step(model2, batch, arm, opt2, sched2, params2, cs, device))
            model = model2
        runs[name] = {"losses": [s["loss"] for s in stats], "state": trainable_state(model), "seconds": times,
                      "peak_mb": torch.cuda.max_memory_allocated(device) / 2**20, "grad_norms": [s["grad_norm"] for s in stats]}
        if name == "A":
            report["frozen_unchanged"] = params_digest([(n, p) for n, p in model.encoder.vt.named_parameters() if not p.requires_grad]) == frozen0
            report["trainable_changed"] = params_digest(model.encoder.trainable()) != params_digest(build_model(cs, paths, Path(info["init"]), "cpu")[0].encoder.trainable())
        del model; torch.cuda.empty_cache()

    def max_diff(a: dict, b: dict) -> float:
        return max(float((a[g][n] - b[g][n]).abs().max()) for g in ("encoder_trainable", "head") for n in a[g])
    tol = sm["reproducibility"]
    report["losses_A"] = runs["A"]["losses"]; report["grad_norms_A"] = runs["A"]["grad_norms"]
    report["reproducibility"] = {"loss_max_abs": max(abs(x - y) for x, y in zip(runs["A"]["losses"], runs["B"]["losses"])),
                                 "params_max_abs": max_diff(runs["A"]["state"], runs["B"]["state"])}
    report["reproducibility"]["ok"] = report["reproducibility"]["loss_max_abs"] <= float(tol["loss_abs"]) and report["reproducibility"]["params_max_abs"] <= float(tol["params_abs"])
    report["resume"] = {"params_max_abs": max_diff(runs["A"]["state"], runs["C2"]["state"]), "loss_max_abs": max(abs(x - y) for x, y in zip(runs["A"]["losses"], runs["C2"]["losses"]))}
    report["resume"]["ok"] = report["resume"]["params_max_abs"] <= float(sm["resume"]["params_abs"])
    report["seconds_per_update"] = float(np.median(runs["A"]["seconds"][1:])); report["peak_gpu_memory_mb"] = runs["A"]["peak_mb"]
    report["estimated_hours_per_model"] = report["seconds_per_update"] * total / 3600
    report["finite"] = all(np.isfinite(runs["A"]["losses"]))
    report["ok"] = bool(report["finite"] and not report["no_finite_nonzero_grad"] and report["frozen_unchanged"] and report["trainable_changed"]
                        and report["reproducibility"]["ok"] and report["resume"]["ok"] and report["pairing_first_update"]["S_equals_C_on_single_view"])
    (out / "smoke.json").write_text(json.dumps(report, indent=1, default=str)); print(json.dumps(report, indent=1, default=str))
    (out / "smoke_resume.pt").unlink(missing_ok=True)
    if not report["ok"]:
        raise SystemExit("smoke checks failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--arm", choices=("N", "S", "C"), required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--manifest-root", required=True)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); cs = cfg["center_supervision"]; paths = load_paths()
    if a.seed not in cs["seeds"]:
        raise SystemExit(f"seed {a.seed} is not one of {cs['seeds']}")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    m, info, index = load_inputs(cs, paths.repo_root, a.arm, a.seed, Path(a.manifest_root))
    sp = cs["sampling"]; upe = int(sp["updates_per_epoch"]); total = int(sp["epochs"]) * upe
    if m["vid"].shape[:2] != (total, int(sp["native_per_update"]) + int(sp["mixed_per_update"])):
        raise SystemExit(f"manifest holds {m['vid'].shape[:2]} windows, expected ({total}, 8)")
    digests = [epoch_digest(m, e * upe, (e + 1) * upe) for e in range(int(sp["epochs"]))]
    if digests != info["epoch_digests"]:
        raise SystemExit("epoch input digests differ from index.json")
    man = load_manifest(paths.manifest_path).set_index("video_id")
    frame_paths = FramePaths({str(v): str(man.loc[str(v), "frames_dir"]) for v in m["videos"]})
    proto = json.loads((paths.repo_root / cs["inputs"]["protocol"]["path"]).read_text()); first_clip = sorted(proto["train"]["clips"])[0]
    if a.smoke:
        out = paths.repo_root / "runs" / "center_supervision_v1" / "smoke" / f"{a.arm}_s{a.seed}"; out.mkdir(parents=True, exist_ok=True)
        smoke(cfg, cs, paths, a.arm, a.seed, m, info, frame_paths, first_clip, device, a.workers, out); return
    ident = identity(cfg, a.arm, a.seed, info["manifest_sha256"], info["init_sha256"])
    run_dir = paths.repo_root / "runs" / "center_supervision_v1" / "train" / f"{a.arm}_s{a.seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = paths.checkpoints_root / "center_supervision_v1" / f"{a.arm}_s{a.seed}"; ckpt.mkdir(parents=True, exist_ok=True)
    model, vinfo = build_model(cs, paths, Path(info["init"]), device); names = check_trainable(model)
    par = parity(model, cs, paths, frame_paths, first_clip, device)
    frozen = [(n, p) for n, p in model.encoder.vt.named_parameters() if not p.requires_grad]; frozen0 = params_digest(frozen)
    start = {"trainable_digest": params_digest(model.encoder.trainable()), "head_digest": params_digest(model.head.named_parameters()),
             "frozen_digest": frozen0}
    opt, sched, params = make_optimiser(model, cs, total); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    done, log = (load_resume(ckpt / "resume.pt", model, opt, sched, ident) if (ckpt / "resume.pt").is_file() else (0, []))
    print(f"[{a.arm} s{a.seed}] identity {ident[:12]} start at update {done}/{total}; parity {par['reference_max_abs']:.1e}/{par['cache_fp16_max_abs']:.1e}", flush=True)
    progress = run_dir / "progress.json"; t_start = time.time(); sums = defaultdict(float); n_ep = 0; gmax = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    for batch in loader(m, frame_paths, list(range(done, total)), a.workers):
        u = int(batch[0]); s = step(model, batch, a.arm, opt, sched, params, cs, device); done = u + 1; n_ep += 1
        for key in ("loss", "main", "aux", "grad_norm"):
            sums[key] += s[key]
        gmax = max(gmax, s["grad_norm"])
        if done % 25 == 0:
            progress.write_text(json.dumps({"identity": ident, "updates": done, "total": total, "seconds": time.time() - t_start,
                                            "loss": s["loss"], "lr_encoder": sched.get_last_lr()[0]}))
        if done % upe == 0:
            ep = done // upe - 1
            rec = {"epoch": ep, "updates": done, **{k: v / n_ep for k, v in sums.items()}, "grad_norm_max": gmax,
                   "lr_encoder": sched.get_last_lr()[0], "lr_head": sched.get_last_lr()[1], "seconds": time.time() - t_start,
                   "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20, "input_digest": digests[ep]}
            log.append(rec); save_resume(ckpt / "resume.pt", model, opt, sched, done, ident, log)
            with (run_dir / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{a.arm} s{a.seed}] epoch {ep} updates {done}/{total} loss {rec['loss']:.4f} (main {rec['main']:.4f}, aux {rec['aux']:.4f}) "
                  f"grad {rec['grad_norm']:.2f}/{gmax:.2f} {rec['seconds']:.0f}s", flush=True)
            sums, n_ep, gmax = defaultdict(float), 0, 0.0
    if done != total:
        raise SystemExit(f"stopped at update {done} of {total}")
    final = ckpt / "final.pt"; torch.save({"identity": ident, **trainable_state(model)}, final)
    frozen_end = params_digest([(n, p) for n, p in model.encoder.vt.named_parameters() if not p.requires_grad])
    manifest = {"identity": ident, "arm": a.arm, "seed": a.seed, "config": a.config, "config_sha256": sha256_file(paths.repo_root / a.config),
                **info, "visual": vinfo, "trainable_encoder": names, "parity": par, "start": start,
                "end": {"trainable_digest": params_digest(model.encoder.trainable()), "head_digest": params_digest(model.head.named_parameters()),
                        "frozen_digest": frozen_end}, "frozen_unchanged": frozen_end == frozen0, "updates": done, "epochs": log,
                "final_checkpoint": str(final), "final_sha256": sha256_file(final), "train_seconds_this_process": time.time() - t_start,
                "n_params": {"encoder_trainable": int(sum(p.numel() for _, p in model.encoder.trainable())),
                             "head": int(sum(p.numel() for p in model.head.parameters()))}}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"[{a.arm} s{a.seed}] done: {done} updates, final {manifest['final_sha256'][:12]}"); print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
