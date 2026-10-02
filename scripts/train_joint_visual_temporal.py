#!/usr/bin/env python
"""Paired joint fine-tuning of the E1 visual encoder with a temporal head (round 2, Task 5b).

Starts from the exported E1 image encoder (the frozen stage-1 weights) and from the E1-trained head of the same seed,
unfreezes the last ViT block, ln_post and proj, and trains on rendered frames with the round-1 batch plan (4 training
streams and 4 native windows per update, 375 updates per epoch): AdamW, backbone lr 1e-5, head lr 1e-4, weight decay
1e-4, warm-up then cosine, gradient clipping 1, at most 10 epochs, patience 3 (configs/experiments/model_development_v2.yaml,
joint_finetune). Checkpoints are chosen on the selection half by the round-1 rule; the calibration half is used only
after training, for the fixed B3 grid. Each update decodes exactly the frames of its streams and windows.

  python scripts/train_joint_visual_temporal.py -c configs/experiments/model_development_v2.yaml --set seed=0 \
      --protocol P --feature-manifest E1_MANIFEST --head-run E1_RUN_DIR --visual-export VISUAL_PT --row J_M5 [--sanity]
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402
from scripts.posterior_decoders import FLAT, apply, decoders  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from echo_routing.compose.banks import multi_bank  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.dev_metrics import evaluate_predictions, select_checkpoint, select_operating_point  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402
from echo_routing.temporal.dev_data import Seq, collate  # noqa: E402
from echo_routing.temporal.dev_heads import head_from_manifest  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs, decoded_digest, plot_curves, sha256_file  # noqa: E402
from echo_routing.temporal.dev_train import LOGGED_METRICS, lr_factor, seed_everything  # noqa: E402
from echo_routing.temporal.joint_visual import (FrameBatches, VisualEncoder, load_frame_u8, load_vision_transformer,  # noqa: E402
                                                native_window_refs, refs_digest, trainable_blocks_from)
from echo_routing.temporal.windowed import stream_plan  # noqa: E402


class FramePaths:
    """video id -> frame paths in temporal order (fork-safe, memoised per worker)."""

    def __init__(self, frames_dirs: dict[str, str]) -> None:
        self.dirs, self.cache = frames_dirs, {}

    def __call__(self, video_id: str):
        if video_id not in self.cache:
            self.cache[video_id] = list_frame_paths(Path(self.dirs[video_id]))
        return self.cache[video_id]


def params_digest(named) -> str:
    h = hashlib.sha256()
    for n, p in named:
        h.update(n.encode()); h.update(p.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()


def build_refs(seqs: list[Seq], recipes: dict, labels: dict, loader) -> list[list]:
    refs = []
    for s in seqs:
        if s.is_native:
            vid, _, start = s.seq_id.partition("@")
            refs.append(native_window_refs(vid, labels[vid], int(start or 0), s.labels.size, loader))
        else:
            refs.append(stream_plan(recipes[s.seq_id], loader))
        if len(refs[-1]) != s.labels.size:
            raise RuntimeError(f"{s.seq_id}: {len(refs[-1])} frames for {s.labels.size} samples")
    return refs


def padded(feats: torch.Tensor, counts: list[int]) -> torch.Tensor:
    """Concatenated per-frame features -> (B, D, T) zero-padded, as collate() builds x."""
    parts = torch.split(feats, counts)
    return torch.nn.utils.rnn.pad_sequence(parts, batch_first=True).transpose(1, 2).contiguous()


@torch.no_grad()
def features_for(encoder: VisualEncoder, refs: list[list], frame_paths, device, workers: int, per_item: int = 24) -> list[np.ndarray]:
    """Per-sequence features (float32 numpy) with the current encoder, in eval mode."""
    encoder.eval(); groups = [refs[i:i + per_item] for i in range(0, len(refs), per_item)]
    dl = DataLoader(FrameBatches(groups, frame_paths), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None)
    out = []
    for u8, counts in dl:
        f = encoder(u8.to(device, non_blocking=True)).float().cpu()
        out += [x.numpy() for x in torch.split(f, counts.tolist())]
    return out


def evaluate_half(head, head_model, encoder, ev, refs, frame_paths, device, workers, hz, tol_s):
    seqs = ev.sequences(); feats = features_for(encoder, refs, frame_paths, device, workers)
    seqs = [Seq(s.seq_id, s.is_native, f, s.labels, s.joins) for s, f in zip(seqs, feats)]
    decoded = head.decode(head_model, seqs, device)
    metrics, tables = evaluate_predictions({k: v["labels"] for k, v in decoded.items()}, ev, hz, tol_s)
    return metrics, tables, decoded


def main() -> None:
    ap = base_parser("Joint visual-temporal fine-tuning")
    ap.add_argument("--protocol", required=True); ap.add_argument("--feature-manifest", required=True)
    ap.add_argument("--head-run", required=True); ap.add_argument("--visual-export", required=True)
    ap.add_argument("--row", required=True); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-epochs", type=int, default=None); ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths(); jc = cfg["joint_finetune"]; tc = cfg["temporal_training"]
    seed = int(cfg.get("seed", 0)); hz = float(cfg["protocol"]["hz"]); sc = cfg["selection"]; tol_s = float(sc["boundary_tolerance_s"])
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    # full fp32: with TF32 the encoder departs from the E1 features by up to 4e-3; without it the match is about 3e-6
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    inputs = Inputs(cfg, Path(args.protocol), Path(args.feature_manifest), paths.cache_root)
    head_man = json.loads((Path(args.head_run) / "run_manifest.json").read_text())
    if head_man["seed"] != seed or head_man["feature_manifest_sha256"] != inputs.fman_sha:
        raise SystemExit("the head run must be the E1 run of the same seed")
    run_dir = new_run_dir(cfg, f"{args.row}{'_sanity' if args.sanity else ''}"); ckpt_dir = paths.checkpoints_root / run_dir.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    idx = read_index(variant_dir(paths.cache_root, inputs.cache_key, "orig"))
    recipes = {r.recipe_id: r for r in multi_bank(idx, "train", cfg).recipes}
    man = load_manifest(paths.manifest_path).set_index("video_id")
    frame_paths = FramePaths({v: str(man.loc[v, "frames_dir"]) for v in idx["video_id"]})
    loader = inputs.store.get

    # encoder: official class, exported weights; parity with the stored reference embeddings and with the E1 cache
    seed_everything(seed)
    vt, vinfo = load_vision_transformer(paths.repo_root / "third_party" / "EchoViewCLIP", Path(args.visual_export))
    if getattr(vt, "VPT_shallow", False):
        raise SystemExit("visual prompts are not part of the E1 encoder")
    encoder = VisualEncoder(vt, trainable_blocks=trainable_blocks_from(jc["unfreeze"], len(vt.transformer.resblocks))).to(device)
    ref = json.loads(Path(args.visual_export).with_suffix(".json").read_text())
    with torch.no_grad():
        encoder.eval()
        u8 = torch.from_numpy(np.stack([load_frame_u8(Path(p), 1.0) for p in ref["reference_frames"]])).to(device)
        parity_ref = float((encoder(u8).cpu() - torch.tensor(ref["reference_embeddings"])).abs().max())
        sel0 = inputs.halves["selection"].natives[0]
        r0 = build_refs([sel0], {}, inputs.labels, loader)[0]
        f0 = features_for(encoder, [r0], frame_paths, device, 0)[0]
        parity_cache = float(np.abs(f0 - inputs.store.get(sel0.seq_id, "orig").feat).max())
    if parity_ref > 1e-3 or parity_cache > 2e-2:
        raise SystemExit(f"encoder parity failed: reference {parity_ref}, cache {parity_cache}")
    print(f"[{args.row}] encoder parity: reference frames {parity_ref:.2e}, E1 cache (float16) {parity_cache:.2e}", flush=True)

    head = head_from_manifest(head_man, paths.repo_root, inputs.feature_dim, device)
    head_model = head.build().to(device)
    head_model.load_state_dict(torch.load(head_man["checkpoints"]["best"], map_location=device))
    trainable = encoder.trainable(); frozen = [(n, p) for n, p in encoder.vt.named_parameters() if not p.requires_grad]
    opt = torch.optim.AdamW([{"params": [p for _, p in trainable], "lr": float(jc["backbone_lr"])},
                             {"params": list(head_model.parameters()), "lr": float(jc["head_lr"])}], weight_decay=float(jc["weight_decay"]))
    st_spb, st_npb = int(tc["batch_size"]) // 2, int(tc["batch_size"]) - int(tc["batch_size"]) // 2
    max_epochs = int(args.max_epochs or jc["max_epochs"])
    plan0 = inputs.train.epoch_batches(seed, 0, int(tc["native_per_epoch"]), int(tc["native_window"]), st_spb, st_npb)
    total, warmup = len(plan0) * max_epochs, len(plan0) * int(tc["warmup_epochs"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warmup, total))
    baseline = inputs.frame_majority(inputs.halves["selection"], hz, tol_s)
    floor = baseline["native_macro_f1"] - float(sc["native_f1_tolerance"])
    start = {"visual_export_sha256": sha256_file(Path(args.visual_export)), "head_best_sha256": head_man["checkpoints"]["best_sha256"],
             "trainable_params": [n for n, _ in trainable], "n_trainable_visual": int(sum(p.numel() for _, p in trainable)),
             "trainable_digest_start": params_digest(trainable), "frozen_digest": params_digest(frozen)}
    eval_refs = {}
    for h in HALVES:   # validation streams come from the protocol's half banks
        ev = inputs.halves[h]; recs = {s.recipe.recipe_id: s.recipe for s in (*ev.pairs, *ev.multi)}
        eval_refs[h] = build_refs(ev.sequences(), recs, inputs.labels, loader)
    records, updates, log_path = [], 0, ckpt_dir / "train_log.jsonl"; log_path.write_text("")
    for epoch in range(max_epochs):
        torch.cuda.reset_peak_memory_stats(device); t0 = time.time()
        plan = inputs.train.epoch_batches(seed, epoch, int(tc["native_per_epoch"]), int(tc["native_window"]), st_spb, st_npb)
        refs = [build_refs(b, recipes, inputs.labels, loader) for b in plan]
        epoch_digest = refs_digest([r for b in refs for r in b])
        dl = DataLoader(FrameBatches(refs, frame_paths), batch_size=None, num_workers=args.workers, prefetch_factor=2, pin_memory=True)
        sums = defaultdict(float); n_b = 0
        for (u8, counts), seqs in zip(dl, plan):
            encoder.train(); head_model.train()
            feats = encoder(u8.to(device, non_blocking=True))
            batch = collate([Seq(s.seq_id, s.is_native, np.zeros((s.labels.size, 1), np.float16), s.labels, s.joins) for s in seqs], device)
            batch["x"] = padded(feats, counts.tolist())
            loss, parts = head.loss(head_model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, update {updates}")
            opt.zero_grad(set_to_none=True); loss.backward()
            if args.sanity:
                grads = {n: p.grad for n, p in trainable}
                bad = [n for n, g in grads.items() if g is None or not torch.isfinite(g).all() or float(g.abs().max()) == 0.0]
                before = params_digest(trainable)
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable] + list(head_model.parameters()), float(tc["grad_clip"]))
            opt.step(); sched.step(); updates += 1; n_b += 1
            sums["total"] += float(loss.detach())
            for k, v in parts.items():
                sums[k] += v
            if args.sanity:
                report = {"row": args.row, "seed": seed, "first_update_frames_digest": refs_digest(refs[0]), "n_frames": int(u8.shape[0]),
                          "trainable_without_finite_nonzero_grad": bad, "trainable_changed": params_digest(trainable) != before,
                          "frozen_unchanged": params_digest(frozen) == start["frozen_digest"], "loss": float(loss.detach()),
                          "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20, "seconds": time.time() - t0, **start}
                (run_dir / "sanity.json").write_text(json.dumps(report, indent=1)); print(json.dumps(report, indent=1)); return
        t_train = time.time() - t0
        m, _, decoded = evaluate_half(head, head_model, encoder, inputs.halves["selection"], eval_refs["selection"], frame_paths, device, args.workers, hz, tol_s)
        rec = {"epoch": epoch, "updates": updates, "lr_backbone": float(sched.get_last_lr()[0]), "lr_head": float(sched.get_last_lr()[1]),
               "loss": {k: v / n_b for k, v in sums.items()}, "seconds_train": t_train, "seconds_eval": time.time() - t0 - t_train,
               "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20, "frames_digest": epoch_digest,
               **{k: m[k] for k in LOGGED_METRICS}}
        records.append(rec)
        torch.save({"visual_trainable": {n: p.detach().cpu() for n, p in trainable}, "head": head_model.state_dict()}, ckpt_dir / f"epoch_{epoch:03d}.pt")
        chosen = select_checkpoint(records, floor, float(sc["tie_margin"]))["chosen"]["epoch"]; rec["chosen_epoch"] = chosen
        with log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"epoch {epoch} upd {updates} loss {rec['loss']['total']:.4f} | native F1 {rec['native_macro_f1']:.4f} segF1 "
              f"{rec['segment_f1_50_multi']:.4f} rec {rec['boundary_recall_multi']:.3f} cuts {rec['native_internal_cuts_per_clip']:.3f} | "
              f"chosen {chosen} ({t_train:.0f}s+{rec['seconds_eval']:.0f}s)", flush=True)
        if epoch - chosen >= int(jc["patience"]):
            print(f"early stop: chosen epoch {chosen}", flush=True); break
    final = select_checkpoint(records, floor, float(sc["tie_margin"]))
    best = torch.load(ckpt_dir / f"epoch_{final['chosen']['epoch']:03d}.pt", map_location=device)
    with torch.no_grad():
        for n, p in trainable:
            p.copy_(best["visual_trainable"][n].to(device))
    head_model.load_state_dict(best["head"])
    shutil.copyfile(ckpt_dir / f"epoch_{final['chosen']['epoch']:03d}.pt", ckpt_dir / "best.pt")
    shutil.copyfile(ckpt_dir / f"epoch_{records[-1]['epoch']:03d}.pt", ckpt_dir / "last.pt")
    out_metrics, post, digests = {}, {}, {}
    for h in HALVES:
        m, tables, decoded = evaluate_half(head, head_model, encoder, inputs.halves[h], eval_refs[h], frame_paths, device, args.workers, hz, tol_s)
        out_metrics[h] = m; post[h] = decoded; digests[h] = decoded_digest(decoded)
        np.savez_compressed(run_dir / f"predictions_{h}.npz", **{k: np.asarray(v["labels"], np.int8) for k, v in decoded.items()})
        for name, df in tables.items():
            df.to_csv(run_dir / f"rows_{h}_{name}.csv", index=False)
    # the same fixed B3 grid on this system's posteriors, chosen on the calibration half
    row = f"{args.row}_s{seed}"; base_cal = inputs.frame_majority(inputs.halves["calibration"], hz, tol_s)
    cal_floor = base_cal["native_macro_f1"] - float(sc["native_f1_tolerance"])
    cal_nat = [post["calibration"][s.seq_id]["prob"] for s in inputs.halves["calibration"].natives]
    recs = []
    for i, (fam, params) in enumerate(d for d in decoders(cfg, cal_nat, False) if d[0] == "b3"):
        mm, _ = evaluate_predictions(apply(fam, params, post["calibration"]), inputs.halves["calibration"], hz, tol_s)
        recs.append({"order": i, "family": fam, "params": params, **{k: mm[k] for k in FLAT}})
    op = select_operating_point(recs, cal_floor, float(sc["boundary_recall_min"]), float(sc["tie_margin"]))
    b3 = {"status": op["status"], "params": op["main"]["params"]}
    for h in HALVES:
        mm, tables = evaluate_predictions(apply("b3", op["main"]["params"], post[h]), inputs.halves[h], hz, tol_s)
        b3[h] = {k: mm[k] for k in (*FLAT, "native_per_class_recall")}
        for t in ("native", "multi_seg", "multi_frag"):
            tables[t].to_csv(run_dir / f"rows_{row}_b3_{h}_{t}.csv", index=False)
    (run_dir / "posterior_decoders.json").write_text(json.dumps({"floor_calibration": cal_floor, "results": {row: {"run_dir": str(run_dir),
                                                                 "source": "joint", "families": {"b3": b3}}}}, indent=1, default=str))
    shutil.copyfile(log_path, run_dir / "train_log.jsonl")
    plot_curves(records, final["chosen"]["epoch"], run_dir / "training_curves.png", f"{args.row} joint fine-tuning, seed {seed}")
    manifest = {"row": args.row, "head": head.name, "seed": seed, "experiment": cfg.get("experiment"), "chosen_epoch": final["chosen"]["epoch"],
                "checkpoint_status": final["status"], "epochs_run": len(records), "updates": updates, "native_floor_selection": floor,
                "visual": vinfo, "start": start, "trainable_digest_end": params_digest(trainable), "frozen_digest_end": params_digest(frozen),
                "frozen_unchanged": params_digest(frozen) == start["frozen_digest"], "joint_finetune": dict(jc),
                "protocol": args.protocol, "protocol_sha256": inputs.proto_sha, "feature_manifest": args.feature_manifest,
                "feature_manifest_sha256": inputs.fman_sha, "head_run": args.head_run, "model_cfg": head_man.get("model_cfg"),
                "boundary_weight": head_man.get("boundary_weight"), "upstream": head_man.get("upstream"),
                "class_weight": head_man.get("class_weight"), "pos_weight": head_man.get("pos_weight"),
                "checkpoints": {"best": str(ckpt_dir / "best.pt"), "best_sha256": sha256_file(ckpt_dir / "best.pt"),
                                "last": str(ckpt_dir / "last.pt")}, "predictions_digest": digests,
                "frames_digests": [r["frames_digest"] for r in records], "train_seconds": sum(r["seconds_train"] for r in records),
                "max_train_peak_gpu_memory_mb": max(r["peak_gpu_memory_mb"] for r in records)}
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    (run_dir / "validation_metrics.json").write_text(json.dumps({"method": out_metrics, "b3": b3, "encoder_frame_majority_selection": baseline},
                                                                indent=1, default=str))
    cal = out_metrics["calibration"]
    print(f"[{args.row}] chosen epoch {final['chosen']['epoch']} ({final['status']}); calibration: native F1 {cal['native_macro_f1']:.4f} "
          f"segF1 {cal['segment_f1_50_multi']:.4f} recall {cal['boundary_recall_multi']:.3f} cuts {cal['native_internal_cuts_per_clip']:.3f}")
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
