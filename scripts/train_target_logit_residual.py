#!/usr/bin/env python
"""Train one arm (Z or R) and seed of the target-logit residual round (configs/experiments/target_logit_residual_v1.yaml;
plan Tasks 1-4).

The model is the C model of the center-supervision round (E1 encoder with blocks 8-11, ln_post and proj trainable;
two-layer relative-position temporal head), started from the source round's untrained initial state of the same seed
with the window output layer set to zero (one file per seed, shared by Z and R). Z reads out the CLS correction alone;
R adds it to the frame logits at the target sample (echo_routing.vit.residual_target). Data, loss, optimiser, budget,
precision and checkpoint rule are the source round's: the SC manifest of the seed (4 native + 4 mixed windows per
update), CE on label[c] plus 0.5 x the frame cross-entropy, AdamW 1e-5 / 1e-4, 3,750 updates, fp32 without TF32, the
final epoch. No validation data is read. Resume state is saved after every epoch under the round's own namespace.

--smoke runs the plan's checks on the first training updates instead of training: encoder parity (reference and cache,
reported apart), the trainable set, R's start against the E1 probe on the same fp32 embedding, gradients with the
zero-initialised output layer (zero on the temporal path at the first backward, non-zero after one update),
same-seed reproducibility, resume, frozen weights, and the Z/R pairing of inputs and initial state.

  python scripts/train_target_logit_residual.py --config configs/experiments/target_logit_residual_v1.yaml --arm R --seed 0 [--smoke]
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
from scripts.train_center_supervision import (check_trainable, load_inputs, load_resume, loader, make_optimiser,  # noqa: E402
                                              parity, save_resume, trainable_state)
from scripts.train_joint_visual_temporal import FramePaths, params_digest  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402
from echo_routing.temporal.joint_visual import VisualEncoder, load_vision_transformer, trainable_blocks_from  # noqa: E402
from echo_routing.vit.residual_target import ResidualTargetModel, target_logits, temporal_named, zero_output_state  # noqa: E402
from echo_routing.vit.target_video import TargetTimeHead, losses  # noqa: E402
from echo_routing.vit.target_windows import epoch_digest  # noqa: E402

NS = "target_logit_residual_v1"
CODE = ("echo_routing/vit/residual_target.py", "scripts/train_target_logit_residual.py")


def check_source(rt: dict, root: Path) -> dict:
    out = {}
    for key in ("training_config", "manifest_index", "status"):
        spec = rt["source"][key]
        if (got := sha256_file(root / spec["path"])) != spec["sha256"]:
            raise SystemExit(f"source {key} {spec['path']}: sha256 {got[:12]} != {spec['sha256'][:12]}")
        out[key] = got
    return out


def prepare_inits(cfg: dict, paths) -> dict:
    """One zero-output initial state per seed from the source round's untrained init; written once, then verified."""
    rt, root = cfg["target_logit_residual"], paths.repo_root
    index = json.loads((root / rt["source"]["manifest_index"]["path"]).read_text()); src_root = (root / rt["source"]["manifest_index"]["path"]).parent
    out_dir = root / "runs" / NS / "inits"; out_dir.mkdir(parents=True, exist_ok=True); rec, temporal = {}, {}
    for seed in rt["seeds"]:
        entry = index["seeds"][str(seed)]["files"]["init"]; src = src_root / entry["file"]
        if sha256_file(src) != entry["sha256"]:
            raise SystemExit(f"{src} differs from the source index")
        state = torch.load(src, map_location="cpu", weights_only=True); z = zero_output_state(state)
        changed = sorted(k for k in state if not torch.equal(state[k], z[k]))
        if any(k not in ("head.weight", "head.bias") for k in changed):
            raise SystemExit(f"zeroing changed more than the output layer: {changed}")
        path = out_dir / f"init_zero_s{seed}.pt"
        if path.is_file():
            old = torch.load(path, map_location="cpu", weights_only=True)
            if set(old) != set(z) or any(not torch.equal(old[k], z[k]) for k in z):
                raise SystemExit(f"{path} exists with other content")
        else:
            torch.save(z, path)
        rec[str(seed)] = {"file": path.name, "sha256": sha256_file(path), "source": str(src), "source_sha256": entry["sha256"],
                          "zeroed": changed}
        temporal[seed] = {k: v for k, v in z.items() if k.startswith(("cls_pos", "rel_pos", "temporal."))}
    seeds = sorted(temporal)
    differ = all(any(not torch.equal(temporal[a][k], temporal[b][k]) for k in temporal[a]) for i, a in enumerate(seeds) for b in seeds[i + 1:])
    if not differ:
        raise SystemExit("two seeds share identical temporal parameters")
    (out_dir / "index.json").write_text(json.dumps({"seeds": rec, "temporal_parameters_differ_across_seeds": differ}, indent=1))
    return rec


def check_target_positions(m: dict) -> dict:
    """In every window all positions with offset 0 must be the same source frame with the same gamma (short clips repeat
    it); a window without its target is refused. Uses the manifest's real sample references, not a fixed slot."""
    at0 = m["rel"] == 0
    if not bool(at0.any(-1).all()):
        raise SystemExit("a window lacks its target sample")
    for key in ("vid", "frame", "gamma"):
        v = m[key].astype(float)
        if not np.array_equal(np.where(at0, v, np.inf).min(-1), np.where(at0, v, -np.inf).max(-1)):
            raise SystemExit(f"offset-0 positions disagree on {key}")
    n_at0 = at0.sum(-1)
    return {"windows": int(at0.shape[0] * at0.shape[1]), "with_repeated_target": int((n_at0 > 1).sum()), "max_target_repeats": int(n_at0.max()),
            "target_not_at_slot_8": int((~at0[..., 8]).sum())}


def build(cs: dict, paths, init_path: Path, mode: str, device) -> tuple[ResidualTargetModel, dict]:
    vt, vinfo = load_vision_transformer(paths.repo_root / "third_party" / "EchoViewCLIP", paths.repo_root / cs["inputs"]["visual_export"]["path"])
    m, t = cs["model"], cs["model"]["temporal"]
    encoder = VisualEncoder(vt, trainable_blocks=trainable_blocks_from(m["unfreeze"], len(vt.transformer.resblocks)), chunk=int(m["encoder_chunk"]))
    head = TargetTimeHead(512, int(t["layers"]), int(t["heads"]), int(t["ffn"]), float(t["dropout"]), int(cs["window"]["max_offset"]),
                          len(cs["classes"]), float(m["position_init_std"]))
    head.load_state_dict(torch.load(init_path, map_location="cpu", weights_only=True), strict=True)
    return ResidualTargetModel(encoder, head, mode).to(device), vinfo


def identity(cfg: dict, arm: str, seed: int, manifest_sha: str, init_sha: str, source: dict, root: Path) -> str:
    payload = {"experiment": cfg.get("experiment"), "arm": arm, "seed": int(seed), "config": {k: v for k, v in cfg.items() if k != "run_name"},
               "source_training_config_sha256": source["training_config"], "manifest_sha256": manifest_sha, "init_sha256": init_sha,
               "code": {c: sha256_file(root / c) for c in CODE}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def step(model: ResidualTargetModel, batch, opt, sched, params, cs: dict, device) -> dict:
    _, u8, rel, labels, target = batch
    model.train()
    final, frame_logits = model(u8.to(device, non_blocking=True), rel.to(device))
    loss, parts = losses("C", final, frame_logits, labels.to(device), target.to(device), float(cs["loss"]["aux_weight"]))
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(params, float(cs["optimiser"]["grad_clip"])))
    if not np.isfinite(gn):
        raise FloatingPointError("non-finite gradient norm")
    opt.step(); sched.step()
    return {"loss": float(loss.detach()), **parts, "grad_norm": gn}


def grad_groups(model: ResidualTargetModel) -> dict:
    def stat(named):
        g = [p.grad for _, p in named]
        finite = all(x is not None and bool(torch.isfinite(x).all()) for x in g)
        return {"finite": finite, "max_abs": max((float(x.abs().max()) for x in g if x is not None), default=0.0),
                "all_nonzero": all(x is not None and float(x.abs().max()) > 0 for x in g)}
    h = model.head
    return {"output_layer": stat([("head.weight", h.head.weight), ("head.bias", h.head.bias)]),
            "frame_head": stat(list(h.frame_head.named_parameters())), "encoder": stat(model.encoder.trainable()),
            "temporal_path": stat(temporal_named(h))}


def smoke(cfg, cs, rt, paths, arm, seed, m, info, init, frame_paths, first_clip, device, workers, out: Path) -> None:
    sm = rt["smoke"]; k = int(sm["updates"]); total = int(cs["sampling"]["epochs"]) * int(cs["sampling"]["updates_per_epoch"])
    report = {"arm": arm, "seed": seed, "manifest_sha256": info["manifest_sha256"], "init_sha256": init["sha256"],
              "epoch_digests": info["epoch_digests"], "probe_init": "frame head = E1 probe at the start"}
    probe = torch.load(paths.repo_root / cs["inputs"]["probe"]["path"], map_location="cpu", weights_only=False)["state_dict"]
    runs = {}
    for name, n_upd in (("A", k), ("B", k), ("C2", k // 2)):
        model, _ = build(cs, paths, paths.repo_root / "runs" / NS / "inits" / init["file"], arm, device)
        opt, sched, params = make_optimiser(model, cs, total); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        batches = list(loader(m, frame_paths, list(range(n_upd)), workers))
        if name == "A":
            report["trainable"] = check_trainable(model); report["parity"] = parity(model, cs, paths, frame_paths, first_clip, device)
            frozen = [(n, p) for n, p in model.encoder.vt.named_parameters() if not p.requires_grad]; frozen0 = params_digest(frozen)
            start_digest = params_digest(model.encoder.trainable())
            with torch.no_grad():   # the start: R = frame head at the target (E1 probe on the fp32 embedding), Z = 0
                model.eval(); _, u8, rel, _, _ = batches[0]; u8, rel = u8.to(device), rel.to(device)
                emb = model.encoder(u8).view(rel.shape[0], rel.shape[1], -1)
                expected = target_logits(F.linear(emb, probe["weight"].to(device).float(), probe["bias"].to(device).float()), rel)
                final, _ = model(u8, rel)
                report["start"] = {"R_vs_probe_max_abs": float((final - expected).abs().max()) if arm == "R" else None,
                                   "Z_max_abs": float(final.abs().max()) if arm == "Z" else None}
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model.train(); _, u8, rel, labels, target = batches[0]
            final, fl = model(u8.to(device), rel.to(device))
            loss, _ = losses("C", final, fl, labels.to(device), target.to(device), float(cs["loss"]["aux_weight"]))
            opt.zero_grad(set_to_none=True); loss.backward(); report["grad_first_backward"] = grad_groups(model)
            opt.zero_grad(set_to_none=True); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        stats, times = [], []
        for i, batch in enumerate(batches):
            torch.cuda.synchronize(); t0 = time.time()
            stats.append(step(model, batch, opt, sched, params, cs, device)); torch.cuda.synchronize(); times.append(time.time() - t0)
            if name == "A" and i == 0:   # after one update with a non-zero learning rate: gradients of the next batch
                model.train(); _, u8, rel, labels, target = batches[1]
                rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
                final, fl = model(u8.to(device), rel.to(device))
                loss, _ = losses("C", final, fl, labels.to(device), target.to(device), float(cs["loss"]["aux_weight"]))
                opt.zero_grad(set_to_none=True); loss.backward(); report["grad_after_first_update"] = grad_groups(model)
                opt.zero_grad(set_to_none=True); torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1])
        if name == "C2":
            save_resume(out / "smoke_resume.pt", model, opt, sched, n_upd, "smoke", [])
            model2, _ = build(cs, paths, paths.repo_root / "runs" / NS / "inits" / init["file"], arm, device)
            opt2, sched2, params2 = make_optimiser(model2, cs, total)
            done, _ = load_resume(out / "smoke_resume.pt", model2, opt2, sched2, "smoke")
            for batch in loader(m, frame_paths, list(range(done, k)), workers):
                stats.append(step(model2, batch, opt2, sched2, params2, cs, device))
            model = model2
        runs[name] = {"losses": [s["loss"] for s in stats], "state": trainable_state(model), "seconds": times,
                      "peak_mb": torch.cuda.max_memory_allocated(device) / 2**20}
        if name == "A":
            report["frozen_unchanged"] = params_digest([(n, p) for n, p in model.encoder.vt.named_parameters() if not p.requires_grad]) == frozen0
            report["trainable_changed"] = params_digest(model.encoder.trainable()) != start_digest
        del model; torch.cuda.empty_cache()

    def max_diff(a: dict, b: dict) -> float:
        return max(float((a[g][n] - b[g][n]).abs().max()) for g in ("encoder_trainable", "head") for n in a[g])
    report["losses_A"] = runs["A"]["losses"]
    report["reproducibility"] = {"loss_max_abs": max(abs(x - y) for x, y in zip(runs["A"]["losses"], runs["B"]["losses"])),
                                 "params_max_abs": max_diff(runs["A"]["state"], runs["B"]["state"])}
    report["resume"] = {"params_max_abs": max_diff(runs["A"]["state"], runs["C2"]["state"])}
    report["seconds_per_update"] = float(np.median(runs["A"]["seconds"][1:])); report["peak_gpu_memory_mb"] = runs["A"]["peak_mb"]
    g0, g1 = report["grad_first_backward"], report["grad_after_first_update"]
    checks = {
        "parity": True,   # parity() stops the run when a threshold is missed
        "start": (report["start"]["R_vs_probe_max_abs"] <= float(sm["init_equivalence"])) if arm == "R" else report["start"]["Z_max_abs"] == 0.0,
        "first_backward": all(g0[x]["finite"] and g0[x]["all_nonzero"] for x in ("output_layer", "frame_head", "encoder")) and g0["temporal_path"]["max_abs"] == 0.0,
        "after_first_update": all(g1[x]["finite"] and g1[x]["all_nonzero"] for x in ("output_layer", "frame_head", "encoder", "temporal_path")),
        "reproducibility": report["reproducibility"]["loss_max_abs"] <= float(sm["reproducibility"]["loss_abs"]) and report["reproducibility"]["params_max_abs"] <= float(sm["reproducibility"]["params_abs"]),
        "resume": report["resume"]["params_max_abs"] <= float(sm["resume"]["params_abs"]),
        "frozen_unchanged": report["frozen_unchanged"], "trainable_changed": report["trainable_changed"],
        "finite": bool(np.all(np.isfinite(runs["A"]["losses"])))}
    other = out.parent / f"{'Z' if arm == 'R' else 'R'}_s{seed}" / "smoke.json"
    if other.is_file():
        o = json.loads(other.read_text())
        checks["paired_with_other_arm"] = (o["manifest_sha256"], o["init_sha256"], o["epoch_digests"]) == (report["manifest_sha256"], report["init_sha256"], report["epoch_digests"])
    report["checks"] = checks; report["ok"] = all(checks.values())
    (out / "smoke.json").write_text(json.dumps(report, indent=1, default=str)); print(json.dumps(report, indent=1, default=str))
    (out / "smoke_resume.pt").unlink(missing_ok=True)
    if not report["ok"]:
        raise SystemExit(f"smoke checks failed: {[k for k, v in checks.items() if not v]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--arm", choices=("Z", "R"), required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); cs, rt = cfg["center_supervision"], cfg["target_logit_residual"]; paths = load_paths(); root = paths.repo_root
    if a.seed not in rt["seeds"]:
        raise SystemExit(f"seed {a.seed} is not one of {rt['seeds']}")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    source = check_source(rt, root); inits = prepare_inits(cfg, paths); init = inits[str(a.seed)]
    m, info, _ = load_inputs(cs, root, "C", a.seed, (root / rt["source"]["manifest_index"]["path"]).parent)   # the SC manifest
    sp = cs["sampling"]; upe = int(sp["updates_per_epoch"]); total = int(sp["epochs"]) * upe
    digests = [epoch_digest(m, e * upe, (e + 1) * upe) for e in range(int(sp["epochs"]))]
    if digests != info["epoch_digests"] or m["vid"].shape[:2] != (total, 8):
        raise SystemExit("the SC manifest does not match its index")
    targets = check_target_positions(m)
    man = load_manifest(paths.manifest_path).set_index("video_id")
    frame_paths = FramePaths({str(v): str(man.loc[str(v), "frames_dir"]) for v in m["videos"]})
    proto = json.loads((root / cs["inputs"]["protocol"]["path"]).read_text()); first_clip = sorted(proto["train"]["clips"])[0]
    if a.smoke:
        out = root / "runs" / NS / "smoke" / f"{a.arm}_s{a.seed}"; out.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"target_positions": targets}), flush=True)
        smoke(cfg, cs, rt, paths, a.arm, a.seed, m, info, init, frame_paths, first_clip, device, a.workers, out); return
    ident = identity(cfg, a.arm, a.seed, info["manifest_sha256"], init["sha256"], source, root)
    run_dir = root / "runs" / NS / "train" / f"{a.arm}_s{a.seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = paths.checkpoints_root / NS / f"{a.arm}_s{a.seed}"; ckpt.mkdir(parents=True, exist_ok=True)
    model, vinfo = build(cs, paths, root / "runs" / NS / "inits" / init["file"], a.arm, device); names = check_trainable(model)
    par = parity(model, cs, paths, frame_paths, first_clip, device)
    frozen0 = params_digest([(n, p) for n, p in model.encoder.vt.named_parameters() if not p.requires_grad])
    start = {"trainable_digest": params_digest(model.encoder.trainable()), "head_digest": params_digest(model.head.named_parameters()), "frozen_digest": frozen0}
    opt, sched, params = make_optimiser(model, cs, total); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    done, log = (load_resume(ckpt / "resume.pt", model, opt, sched, ident) if (ckpt / "resume.pt").is_file() else (0, []))
    print(f"[{a.arm} s{a.seed}] identity {ident[:12]} start at update {done}/{total}; parity {par['reference_max_abs']:.1e} / {par['cache_fp16_max_abs']:.1e}", flush=True)
    progress = run_dir / "progress.json"; t_start = time.time(); sums = defaultdict(float); n_ep = 0; gmax = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    for batch in loader(m, frame_paths, list(range(done, total)), a.workers):
        u = int(batch[0]); s = step(model, batch, opt, sched, params, cs, device); done = u + 1; n_ep += 1
        for key in ("loss", "main", "aux", "grad_norm"):
            sums[key] += s[key]
        gmax = max(gmax, s["grad_norm"])
        if done % 25 == 0:
            progress.write_text(json.dumps({"identity": ident, "updates": done, "total": total, "seconds": time.time() - t_start, "loss": s["loss"]}))
        if done % upe == 0:
            ep = done // upe - 1
            rec = {"epoch": ep, "updates": done, **{k: v / n_ep for k, v in sums.items()}, "grad_norm_max": gmax, "lr_encoder": sched.get_last_lr()[0],
                   "lr_head": sched.get_last_lr()[1], "seconds": time.time() - t_start, "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                   "input_digest": digests[ep]}
            log.append(rec); save_resume(ckpt / "resume.pt", model, opt, sched, done, ident, log)
            with (run_dir / "train_log.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{a.arm} s{a.seed}] epoch {ep} updates {done}/{total} loss {rec['loss']:.4f} (main {rec['main']:.4f}, aux {rec['aux']:.4f}) "
                  f"grad {rec['grad_norm']:.2f}/{gmax:.2f} {rec['seconds']:.0f}s", flush=True)
            sums, n_ep, gmax = defaultdict(float), 0, 0.0
    if done != total:
        raise SystemExit(f"stopped at update {done} of {total}")
    final = ckpt / "final.pt"; torch.save({"identity": ident, "mode": a.arm, **trainable_state(model)}, final)
    frozen_end = params_digest([(n, p) for n, p in model.encoder.vt.named_parameters() if not p.requires_grad])
    manifest = {"identity": ident, "arm": a.arm, "seed": a.seed, "config": a.config, "config_sha256": sha256_file(root / a.config), "source": source,
                "manifest": info["manifest"], "manifest_sha256": info["manifest_sha256"], "init": str(root / "runs" / NS / "inits" / init["file"]),
                "init_sha256": init["sha256"], "code": {c: sha256_file(root / c) for c in CODE}, "visual": vinfo, "trainable_encoder": names,
                "parity": par, "target_positions": targets, "start": start, "end": {"trainable_digest": params_digest(model.encoder.trainable()),
                                                     "head_digest": params_digest(model.head.named_parameters()), "frozen_digest": frozen_end},
                "frozen_unchanged": frozen_end == frozen0, "updates": done, "epochs": log, "final_checkpoint": str(final), "final_sha256": sha256_file(final),
                "train_seconds_this_process": time.time() - t_start}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(f"[{a.arm} s{a.seed}] done: {done} updates, final {manifest['final_sha256'][:12]}"); print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
