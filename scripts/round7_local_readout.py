#!/usr/bin/env python
"""Round 7: global against local readouts of the frozen E1 encoder on the round-6 spans, one seed per run
(configs/experiments/local_readout_v1.yaml).

For seed k the frozen E1 ASRF seed k decodes the training split and both validation halves into the round-6 span
records (at most 32 evenly spaced frames per span). Nothing in the encoder trains.
- P0 is round 6's V0, run with round 6's own functions; its validation predictions must equal the stored V0 ones.
- P1 and P2 train on the same span order and batches. Each update runs the frozen encoder once and feeds token 0 to P1
  and all 197 tokens to P2.
The systems ASRF default, mean rule, P0, P1 and P2 are then scored on both halves, with all clips and without the ten
sensitivity clips. Validation data are never used for training or for any choice.
Writes seed.json, rows, spans_<half>.csv (span margins for the report), predictions.npz and the head weights.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg, new_run_dir  # noqa: E402
from scripts.round4_failure_inventory import load_asrf  # noqa: E402
from scripts.round5_evidence_decider import flips, unit_states  # noqa: E402
from scripts.round6_span_classifier import (BATCH, CAP, CLIP, EPOCHS, FLAT, HEAD_LR, WD, classification, decode_records,  # noqa: E402
                                            encoder_with_parity, features, native_refs, schedule, span_stats, systems,
                                            train_v0, validation_records)
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from echo_routing.compose.banks import multi_bank  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.dev_metrics import evaluate_predictions  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402
from echo_routing.temporal.evidence_decider import span_weights  # noqa: E402
from echo_routing.temporal.joint_visual import FrameBatches, load_frame_u8  # noqa: E402
from echo_routing.temporal.local_readout import all_tokens, build_pair, n_params  # noqa: E402
from echo_routing.temporal.span_classifier import A4C, A5C, CLASSES2, SpanHead, epoch_order, target_index  # noqa: E402
from echo_routing.temporal.windowed import stream_plan  # noqa: E402

SYSTEMS = ("default", "mean", "P0", "P1", "P2")
TOKEN0_TOL = 1e-5


def round6_dir(seed: int) -> Path:
    runs = sorted(p for p in glob.glob(f"runs/*-md6_span_classifier_s{seed}-*") if "sanity" not in p)
    if len(runs) != 1:
        raise SystemExit(f"expected one round-6 run for seed {seed}, found {runs}")
    return Path(runs[0])


def token0_parity(enc, export: Path, device) -> float:
    """Token 0 of the all-token path against the existing global embedding, on round 6's parity frames."""
    ref = json.loads(export.with_suffix(".json").read_text())
    u8 = torch.from_numpy(np.stack([load_frame_u8(Path(p), 1.0) for p in ref["reference_frames"]])).to(device)
    with torch.no_grad():
        enc.eval()
        return float((all_tokens(enc, u8)[:, 0] - enc(u8)).abs().max())


def train_pair(enc, p1, p2, recs: list, t: np.ndarray, w: np.ndarray, seed: int, epochs: int, frame_paths, device, workers: int) -> dict:
    """P1 and P2 on the same batches: one frozen forward per update, token 0 to P1 and all tokens to P2."""
    opts = {"P1": torch.optim.AdamW(p1.parameters(), lr=HEAD_LR, weight_decay=WD),
            "P2": torch.optim.AdamW(p2.parameters(), lr=HEAD_LR, weight_decay=WD)}
    per = math.ceil(len(t) / BATCH); total, step, log = per * epochs, 0, {"P1": [], "P2": []}
    t_all = torch.tensor(t, device=device); w_all = torch.tensor(w, dtype=torch.float32, device=device)
    for epoch in range(epochs):
        order = epoch_order(len(t), seed, epoch); sums, t0 = {"P1": 0.0, "P2": 0.0}, time.time()
        plans = [[list(recs[i].refs) for i in order[b * BATCH:(b + 1) * BATCH]] for b in range(per)]
        dl = DataLoader(FrameBatches(plans, frame_paths), batch_size=None, num_workers=workers, prefetch_factor=2, pin_memory=True)
        for b, (u8, counts) in enumerate(dl):
            tok = all_tokens(enc, u8.to(device, non_blocking=True)); c = counts.tolist()
            idx = torch.tensor(order[b * BATCH:(b + 1) * BATCH], device=device)
            for name, m, x in (("P1", p1, tok[:, 0]), ("P2", p2, tok)):
                m.train(); opt = opts[name]; schedule(opt, [HEAD_LR], step, total, per)
                loss = (F.cross_entropy(m(x, c), t_all[idx], reduction="none") * w_all[idx]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{name}: non-finite loss at epoch {epoch}, update {b}")
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
                opt.step(); sums[name] += float(loss.detach())
            step += 1
        for name in log:
            log[name].append({"epoch": epoch, "loss": sums[name] / per, "seconds": time.time() - t0})
        print(f"epoch {epoch}: P1 {sums['P1'] / per:.4f}, P2 {sums['P2'] / per:.4f} ({time.time() - t0:.0f} s)", flush=True)
    return log


@torch.no_grad()
def readout_logits(enc, models: dict, recs: list, frame_paths, device, workers: int) -> dict[str, np.ndarray]:
    plans = [[list(r.refs) for r in recs[i:i + BATCH]] for i in range(0, len(recs), BATCH)]
    dl = DataLoader(FrameBatches(plans, frame_paths), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None, pin_memory=True)
    out = {k: [] for k in models}
    for u8, counts in dl:
        tok = all_tokens(enc, u8.to(device, non_blocking=True)); c = counts.tolist()
        for name, m in models.items():
            m.eval(); out[name].append(m(tok[:, 0] if name == "P1" else tok, c).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def decisions_of(logits: np.ndarray) -> list[int]:
    """A5C when its probability is at least 0.5, as round 6."""
    return [A5C if lg[1] >= lg[0] else A4C for lg in logits]


def directional(recs: list, d: list[int]) -> dict:
    """On spans with an A4C or A5C majority: A4C read as A5C and A5C read as A4C, by spans and source clips."""
    out = {}
    for name, lab, wrong in (("A4C read as A5C", A4C, A5C), ("A5C read as A4C", A5C, A4C)):
        ks = [i for i, r in enumerate(recs) if r.label == lab and d[i] == wrong]
        out[name] = {"spans": len(ks), "of": int(sum(r.label == lab for r in recs)), "clips": len({recs[i].source for i in ks})}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True); ap.add_argument("--protocol", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--visual-export", default="models_frozen/e1_echoviewclip_probe/v1/visual_encoder_fp32.pt")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--sanity", action="store_true")
    ap.add_argument("-c", "--config", default="configs/experiments/local_readout_v1.yaml")
    a = ap.parse_args(); a.set = [f"seed={a.seed}"]; cfg = load_cfg(a); paths = load_paths(); e = json.loads(Path(a.spec).read_text())["encoders"]["E1"]
    k, sc = a.seed, cfg["selection"]; hz, tol_s = float(cfg["protocol"]["hz"]), float(sc["boundary_tolerance_s"]); t_start = time.time()
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    inputs = Inputs(cfg, Path(a.protocol), Path(e["feature_manifest"]), paths.cache_root)
    idx = read_index(variant_dir(paths.cache_root, inputs.cache_key, "orig")); recipes = multi_bank(idx, "train", cfg).recipes
    if [r.recipe_id for r in recipes] != [s.seq_id for s in inputs.train.streams]:
        raise SystemExit("training-bank recipes and streams are out of order")
    man = load_manifest(paths.manifest_path).set_index("video_id"); frame_paths = FramePaths({v: str(man.loc[v, "frames_dir"]) for v in idx["video_id"]})
    rd = Path(e["asrf"][str(k)]); asrf = load_asrf(rd, paths.repo_root, inputs.feature_dim, device)
    items = [(s.seq_id, s.feat, s.labels.astype(int), stream_plan(r, inputs.store.get)) for r, s in zip(recipes, inputs.train.streams)]
    items += [(f"native:{v}", inputs.store.get(v, "orig").feat, np.full(inputs.store.get(v, "orig").n, int(lab)), native_refs(v, inputs.store))
              for v, lab in inputs.train.clips]
    train_all, _ = decode_records(items, *asrf, device); train = [r for r in train_all if r.label in CLASSES2]
    train = train[:256] if a.sanity else train
    t = target_index(np.array([r.label for r in train])); w = span_weights([r.source for r in train], np.array([r.label for r in train]))
    val = {h: validation_records(inputs, h, asrf, rd, set(cfg["sensitivity"]["exclude_clips"]), device) for h in HALVES}
    print(f"seed {k}: {len(train)} training spans, validation " + ", ".join(f"{h} {len(val[h]['recs'])}" for h in HALVES), flush=True)
    epochs = 1 if a.sanity else EPOCHS; torch.manual_seed(1000 + k); init = {n: v.clone() for n, v in SpanHead().state_dict().items()}
    enc, par = encoder_with_parity(paths, Path(a.visual_export), False, inputs, frame_paths, device)
    par["token0_vs_global"] = token0_parity(enc, Path(a.visual_export), device)
    if par["token0_vs_global"] > TOKEN0_TOL:
        raise SystemExit(f"token 0 differs from the global embedding by {par['token0_vs_global']}")
    # P0: round 6's V0, unchanged
    f_train = features(enc, train, frame_paths, device, a.workers); f_val0 = {h: features(enc, val[h]["recs"], frame_paths, device, a.workers) for h in HALVES}
    head0 = SpanHead().to(device); head0.load_state_dict(init); log0 = train_v0(head0, f_train, t, w, k, epochs, device)
    with torch.no_grad():
        head0.eval(); logits = {h: {"P0": head0(torch.tensor(f_val0[h], device=device)).cpu().numpy()} for h in HALVES}
    # P1 and P2
    p1, p2 = build_pair(k, init); p1, p2 = p1.to(device), p2.to(device)
    capacity = {"P0": sum(p.numel() for p in head0.parameters()), "P1": n_params(p1), "P2": n_params(p2)}
    if abs(capacity["P1"] - capacity["P2"]) > 0.01 * capacity["P2"]:
        raise SystemExit(f"P1 and P2 capacities differ by more than 1%: {capacity}")
    log12 = train_pair(enc, p1, p2, train, t, w, k, epochs, frame_paths, device, a.workers)
    for h in HALVES:
        logits[h].update(readout_logits(enc, {"P1": p1, "P2": p2}, val[h]["recs"], frame_paths, device, a.workers))
    run_dir = new_run_dir(cfg, f"md7_local_readout_s{k}{'_sanity' if a.sanity else ''}"); ckpt = paths.checkpoints_root / run_dir.name; ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(head0.state_dict(), ckpt / "P0_head.pt"); torch.save(p1.state_dict(), ckpt / "P1.pt"); torch.save(p2.state_dict(), ckpt / "P2.pt")
    res = {"seed": k, "sanity": a.sanity, "asrf_run": str(rd), "parity": par, "capacity": capacity,
           "span_stats": {"train": span_stats(train_all), **{h: span_stats(val[h]["recs"]) for h in HALVES}},
           "train_log": {"P0": log0, **log12}, "metrics": {}, "classification": {}, "directional": {}, "flips": {}, "pieces": {},
           "checkpoints": str(ckpt)}
    arrays, r6 = {}, None if a.sanity else np.load(round6_dir(k) / "predictions.npz")
    for h in HALVES:
        V = val[h]; dec = {n: decisions_of(logits[h][n]) for n in ("P0", "P1", "P2")}; preds = systems(V, dec)
        if r6 is not None:
            res.setdefault("p0_reproduces_v0", {})[h] = all(np.array_equal(preds["P0"][sid], r6[f"V0|{h}|{sid}"].astype(int)) for sid in preds["P0"])
        res["classification"][h] = classification(V["recs"], {"vote": [r.vote for r in V["recs"]], **dec})
        res["directional"][h] = {n: directional(V["recs"], d) for n, d in {"vote": [r.vote for r in V["recs"]], **dec}.items()}
        pd.DataFrame({"seq_id": [r.seq_id for r in V["recs"]], "start": [r.start for r in V["recs"]], "end": [r.end for r in V["recs"]],
                      "source": [r.source for r in V["recs"]], "label": [r.label for r in V["recs"]], "vote": [r.vote for r in V["recs"]],
                      **{f"margin_{n}": logits[h][n][:, 1] - logits[h][n][:, 0] for n in ("P0", "P1", "P2")}}).to_csv(run_dir / f"spans_{h}.csv", index=False)
        states = {}
        for s_name in SYSTEMS:
            for scope, evs in (("all", V["ev"]), ("without", V["red"])):
                m, tabs = evaluate_predictions(preds[s_name], evs, hz, tol_s)
                res["metrics"][f"{s_name}|{scope}|{h}"] = {**{x: m[x] for x in FLAT}, "predicted_segments": int(tabs["multi_seg"].seg_pred.sum())}
                for tn in ("native", "multi_seg", "multi_frag"):
                    tabs[tn].to_csv(run_dir / f"rows_{s_name}_{scope}_{h}_{tn}.csv", index=False)
            states[s_name], res["pieces"][f"{s_name}|{h}"] = unit_states(preds[s_name], V["ev"])
            arrays.update({f"{s_name}|{h}|{sid}": v.astype(np.int8) for sid, v in preds[s_name].items()})
        for new, ref in (("P2", "P1"), ("P1", "P0"), ("P2", "P0"), ("P0", "default"), ("P1", "default"), ("P2", "default")):
            res["flips"][f"{new} vs {ref}|{h}"] = flips(states[new], states[ref])
        print(f"seed {k} {h}: " + "; ".join(f"{s_name} {res['metrics'][f'{s_name}|all|{h}']['segment_f1_50_multi']:.4f}" for s_name in SYSTEMS)
              + " | span accuracy " + "; ".join(f"{n} {c['accuracy']:.3f}" for n, c in res["classification"][h].items()), flush=True)
    res["minutes"] = (time.time() - t_start) / 60
    np.savez_compressed(run_dir / "predictions.npz", **arrays); (run_dir / "seed.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
