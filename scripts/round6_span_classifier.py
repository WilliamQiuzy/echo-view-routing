#!/usr/bin/env python
"""Round 6: paired span-level A4C/A5C training, V0 (frozen encoder) against V1 (last block, ln_post and proj trained),
one seed per run (configs/experiments/span_classifier_v1.yaml).

For seed k the frozen E1 ASRF seed k decodes the training split (1,500 training-bank streams, every training clip as a
native recording) and both validation halves. Its candidate spans voted A4C or A5C become span records, with at most
32 evenly spaced frames each. V0 trains the head on span features computed once with the frozen encoder over those
frames. V1 trains the same head, from the same initialisation, together with the encoder's last block, ln_post and
proj, with the same span order, batches and budget. Both then classify the validation spans, and four systems are
scored on both halves, with all clips and without the ten sensitivity clips: ASRF's default, the round-5 mean rule, V0
and V1. Validation data are never used for training or for any choice.
Writes seed.json, rows, predictions.npz and checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg, new_run_dir  # noqa: E402
from scripts.round4_failure_inventory import load_asrf  # noqa: E402
from scripts.round5_evidence_decider import flips, unit_states  # noqa: E402
from scripts.round5_reachability import asrf_decode, inputs_of, without  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from echo_routing.compose.banks import multi_bank  # noqa: E402
from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.dev_metrics import evaluate_predictions  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402
from echo_routing.temporal.evidence_decider import decide_spans, mean_rule  # noqa: E402
from echo_routing.temporal.joint_visual import FrameBatches, VisualEncoder, load_frame_u8, load_vision_transformer  # noqa: E402
from echo_routing.temporal.span_classifier import (A5C, CLASSES2, SpanHead, apply_decisions, epoch_order, lr_factor,  # noqa: E402
                                                   span_means, span_records, target_index)
from echo_routing.temporal.evidence_decider import span_weights  # noqa: E402
from echo_routing.temporal.windowed import SampleRef, stream_plan  # noqa: E402

CAP, BATCH, EPOCHS = 32, 16, 8
HEAD_LR, ENC_LR, WD, CLIP = 1e-4, 1e-5, 1e-4, 1.0
SYSTEMS = ("default", "mean", "V0", "V1")
FLAT = ("segment_f1_50_multi", "boundary_recall_multi", "native_macro_f1")


def native_refs(vid: str, store) -> list[SampleRef]:
    vf = store.get(vid, "orig")
    return [SampleRef(vid, int(vf.frame_idx[s]), 1.0) for s in range(vf.n)]


def decode_records(items, head, model, device) -> tuple[list, dict]:
    """items: (seq_id, feat, reference labels, frame refs). Records of A4C/A5C-voted spans and seq_id -> (final, cuts)."""
    recs, decoded = [], {}
    for sid, feat, ref, refs in items:
        final, cuts = asrf_decode(head, model, np.asarray(feat).astype(np.float16), device); decoded[sid] = (final, cuts)
        recs += span_records(sid, final, cuts, ref, refs, CAP)
    return recs, decoded


@torch.no_grad()
def features(encoder, recs: list, frame_paths, device, workers: int) -> np.ndarray:
    encoder.eval(); plans = [[list(r.refs) for r in recs[i:i + BATCH]] for i in range(0, len(recs), BATCH)]
    dl = DataLoader(FrameBatches(plans, frame_paths), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None, pin_memory=True)
    return np.concatenate([span_means(encoder(u8.to(device, non_blocking=True)), c.tolist()).cpu().numpy() for u8, c in dl])


def schedule(opt, groups_lr: list[float], step: int, total: int, warmup: int) -> None:
    for g, lr in zip(opt.param_groups, groups_lr):
        g["lr"] = lr * lr_factor(step, warmup, total)


def train_v0(head, feats: np.ndarray, t: np.ndarray, w: np.ndarray, seed: int, epochs: int, device) -> list[dict]:
    opt = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=WD); per = math.ceil(len(t) / BATCH)
    total, step, log = per * epochs, 0, []
    x_all = torch.tensor(feats, device=device); t_all = torch.tensor(t, device=device); w_all = torch.tensor(w, dtype=torch.float32, device=device)
    for epoch in range(epochs):
        order = epoch_order(len(t), seed, epoch); s = 0.0
        for b in range(per):
            idx = torch.tensor(order[b * BATCH:(b + 1) * BATCH], device=device); head.train(); schedule(opt, [HEAD_LR], step, total, per)
            loss = (F.cross_entropy(head(x_all[idx]), t_all[idx], reduction="none") * w_all[idx]).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP); opt.step(); step += 1; s += float(loss.detach())
        log.append({"epoch": epoch, "loss": s / per})
    return log


def train_v1(encoder, head, recs: list, t: np.ndarray, w: np.ndarray, seed: int, epochs: int, frame_paths, device, workers: int) -> list[dict]:
    enc_params = [p for _, p in encoder.trainable()]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": ENC_LR}, {"params": list(head.parameters()), "lr": HEAD_LR}], weight_decay=WD)
    per = math.ceil(len(t) / BATCH); total, step, log = per * epochs, 0, []
    t_all = torch.tensor(t, device=device); w_all = torch.tensor(w, dtype=torch.float32, device=device)
    for epoch in range(epochs):
        order = epoch_order(len(t), seed, epoch); s, t0 = 0.0, time.time()
        plans = [[list(recs[i].refs) for i in order[b * BATCH:(b + 1) * BATCH]] for b in range(per)]
        dl = DataLoader(FrameBatches(plans, frame_paths), batch_size=None, num_workers=workers, prefetch_factor=2, pin_memory=True)
        for b, (u8, counts) in enumerate(dl):
            encoder.train(); head.train(); idx = torch.tensor(order[b * BATCH:(b + 1) * BATCH], device=device)
            schedule(opt, [ENC_LR, HEAD_LR], step, total, per)
            x = span_means(encoder(u8.to(device, non_blocking=True)), counts.tolist())
            loss = (F.cross_entropy(head(x), t_all[idx], reduction="none") * w_all[idx]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, update {b}")
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(enc_params + list(head.parameters()), CLIP)
            opt.step(); step += 1; s += float(loss.detach())
        log.append({"epoch": epoch, "loss": s / per, "seconds": time.time() - t0})
        print(f"V1 epoch {epoch}: loss {s / per:.4f} ({time.time() - t0:.0f} s)", flush=True)
    return log


@torch.no_grad()
def decide(head, feats: np.ndarray, device) -> list[int]:
    head.eval(); logits = head(torch.tensor(feats, device=device)).cpu().numpy()
    return [A5C if lg[1] >= lg[0] else CLASSES2[0] for lg in logits]


def span_stats(recs: list) -> dict:
    lab = np.array([r.label for r in recs]); keep = np.isin(lab, CLASSES2); ln = np.array([r.length for r in recs])[keep]
    return {"spans": int(len(recs)), "with_A4C_A5C_majority": int(keep.sum()), "A5C": int((lab == A5C).sum()),
            "length_median": float(np.median(ln)), "length_p90": float(np.percentile(ln, 90)), "length_max": int(ln.max()),
            "mixed_share": float(np.mean([r.purity < 1 for r, k in zip(recs, keep) if k])),
            "vote_differs_share": float(np.mean([r.vote != r.label for r, k in zip(recs, keep) if k]))}


def classification(recs: list, decisions: dict[str, list[int]]) -> dict:
    """Per rule: accuracy and per-class recall on A4C/A5C-majority spans, and the share of source clips whose spans are
    mostly right."""
    keep = [i for i, r in enumerate(recs) if r.label in CLASSES2]; out = {}
    for name, d in decisions.items():
        ok = np.array([d[i] == recs[i].label for i in keep]); lab = np.array([recs[i].label for i in keep])
        by_clip: dict[str, list[bool]] = {}
        for i, o in zip(keep, ok):
            by_clip.setdefault(recs[i].source, []).append(bool(o))
        out[name] = {"spans": int(len(keep)), "accuracy": float(ok.mean()), "recall_A4C": float(ok[lab == CLASSES2[0]].mean()),
                     "recall_A5C": float(ok[lab == A5C].mean()), "clips": len(by_clip),
                     "clips_mostly_right": float(np.mean([np.mean(v) > 0.5 for v in by_clip.values()]))}
    return out


def params_digest(named) -> str:
    h = hashlib.sha256()
    for n, p in sorted(named, key=lambda x: x[0]):
        h.update(n.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def encoder_with_parity(paths, export: Path, trainable: bool, inputs, frame_paths, device) -> tuple:
    """The exported E1 encoder, all frozen (V0) or with its last block, ln_post and proj trainable (V1); it must match
    the stored reference embeddings and the E1 cache before use."""
    vt, info = load_vision_transformer(paths.repo_root / "third_party" / "EchoViewCLIP", export)
    enc = VisualEncoder(vt, trainable_blocks=1).to(device)
    if not trainable:
        for p in enc.vt.parameters():
            p.requires_grad_(False)
    ref = json.loads(export.with_suffix(".json").read_text())
    with torch.no_grad():
        enc.eval()
        u8 = torch.from_numpy(np.stack([load_frame_u8(Path(p), 1.0) for p in ref["reference_frames"]])).to(device)
        parity_ref = float((enc(u8).cpu() - torch.tensor(ref["reference_embeddings"])).abs().max())
        sel0 = inputs.halves["selection"].natives[0]
        u8b, _ = FrameBatches([[native_refs(sel0.seq_id, inputs.store)]], frame_paths)[0]
        parity_cache = float(np.abs(enc(u8b.to(device)).cpu().numpy() - inputs.store.get(sel0.seq_id, "orig").feat).max())
    if parity_ref > 1e-3 or parity_cache > 2e-2:
        raise SystemExit(f"encoder parity failed: reference {parity_ref}, cache {parity_cache}")
    return enc, {"parity_reference": parity_ref, "parity_cache": parity_cache, "commit": info["commit"],
                 "trainable": [n for n, _ in enc.trainable()]}


def validation_records(inputs, h: str, asrf, rd: Path, clips_out: set, device) -> dict:
    ev = inputs.halves[h]; seqs = inputs_of(ev, inputs.store); recipes = {s.recipe.recipe_id: s.recipe for s in (*ev.pairs, *ev.multi)}
    items = [(sid, f, ref, stream_plan(recipes[sid], inputs.store.get) if sid in recipes else native_refs(sid, inputs.store))
             for sid, (f, _, ref) in seqs.items()]
    recs, decoded = decode_records(items, *asrf, device); stored = np.load(rd / f"predictions_{h}.npz")
    if any(not np.array_equal(decoded[sid][0], stored[sid].astype(int)) for sid in seqs):
        raise SystemExit(f"ASRF {rd.name} {h}: decoding does not reproduce the stored predictions")
    return {"ev": ev, "red": without(ev, clips_out), "seqs": seqs, "recs": recs, "decoded": decoded}


def systems(V: dict, decisions: dict[str, list[int]]) -> dict[str, dict]:
    by_seq: dict[str, list[int]] = {}
    for i, r in enumerate(V["recs"]):
        by_seq.setdefault(r.seq_id, []).append(i)
    out = {"default": {sid: f for sid, (f, _) in V["decoded"].items()},
           "mean": {sid: decide_spans(f, c, V["seqs"][sid][1], mean_rule) for sid, (f, c) in V["decoded"].items()}}
    for name, d in decisions.items():
        out[name] = {sid: apply_decisions(f, [V["recs"][i] for i in by_seq.get(sid, [])], [d[i] for i in by_seq.get(sid, [])])
                     for sid, (f, _) in V["decoded"].items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True); ap.add_argument("--protocol", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--visual-export", default="models_frozen/e1_echoviewclip_probe/v1/visual_encoder_fp32.pt")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--sanity", action="store_true")
    ap.add_argument("-c", "--config", default="configs/experiments/span_classifier_v1.yaml")
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
    enc0, par0 = encoder_with_parity(paths, Path(a.visual_export), False, inputs, frame_paths, device)
    f_train = features(enc0, train, frame_paths, device, a.workers); f_val0 = {h: features(enc0, val[h]["recs"], frame_paths, device, a.workers) for h in HALVES}
    del enc0; torch.cuda.empty_cache()
    head0 = SpanHead().to(device); head0.load_state_dict(init); log0 = train_v0(head0, f_train, t, w, k, epochs, device)
    enc1, par1 = encoder_with_parity(paths, Path(a.visual_export), True, inputs, frame_paths, device); digest0 = params_digest(enc1.trainable())
    head1 = SpanHead().to(device); head1.load_state_dict(init)
    log1 = train_v1(enc1, head1, train, t, w, k, epochs, frame_paths, device, a.workers)
    f_val1 = {h: features(enc1, val[h]["recs"], frame_paths, device, a.workers) for h in HALVES}
    dec = {"V0": {h: decide(head0, f_val0[h], device) for h in HALVES}, "V1": {h: decide(head1, f_val1[h], device) for h in HALVES}}
    run_dir = new_run_dir(cfg, f"md6_span_classifier_s{k}{'_sanity' if a.sanity else ''}"); ckpt = paths.checkpoints_root / run_dir.name; ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(head0.state_dict(), ckpt / "head_V0.pt"); torch.save(head1.state_dict(), ckpt / "head_V1.pt")
    torch.save({n: p.detach().cpu() for n, p in enc1.trainable()}, ckpt / "encoder_V1_trainable.pt")
    res = {"seed": k, "sanity": a.sanity, "asrf_run": str(rd), "parity": {"V0": par0, "V1": par1},
           "encoder_V1_trainable_digest": {"start": digest0, "end": params_digest(enc1.trainable())},
           "span_stats": {"train": span_stats(train_all), **{h: span_stats(val[h]["recs"]) for h in HALVES}},
           "train_log": {"V0": log0, "V1": log1}, "metrics": {}, "classification": {}, "flips": {}, "pieces": {}, "checkpoints": str(ckpt)}
    arrays = {}
    for h in HALVES:
        V = val[h]; preds = systems(V, {n: dec[n][h] for n in ("V0", "V1")})
        mean_dec = [mean_rule(V["seqs"][r.seq_id][1][r.start:r.end]) for r in V["recs"]]
        res["classification"][h] = classification(V["recs"], {"vote": [r.vote for r in V["recs"]], "mean": mean_dec, "V0": dec["V0"][h], "V1": dec["V1"][h]})
        states = {}
        for s_name in SYSTEMS:
            for scope, evs in (("all", V["ev"]), ("without", V["red"])):
                m, tabs = evaluate_predictions(preds[s_name], evs, hz, tol_s)
                res["metrics"][f"{s_name}|{scope}|{h}"] = {**{x: m[x] for x in FLAT}, "predicted_segments": int(tabs["multi_seg"].seg_pred.sum())}
                for tn in ("native", "multi_seg", "multi_frag"):
                    tabs[tn].to_csv(run_dir / f"rows_{s_name}_{scope}_{h}_{tn}.csv", index=False)
            states[s_name], res["pieces"][f"{s_name}|{h}"] = unit_states(preds[s_name], V["ev"])
            arrays.update({f"{s_name}|{h}|{sid}": v.astype(np.int8) for sid, v in preds[s_name].items()})
        for new, ref in (("V0", "default"), ("V1", "default"), ("V0", "mean"), ("V1", "mean"), ("V1", "V0")):
            res["flips"][f"{new} vs {ref}|{h}"] = flips(states[new], states[ref])
        print(f"seed {k} {h}: " + "; ".join(f"{s_name} {res['metrics'][f'{s_name}|all|{h}']['segment_f1_50_multi']:.4f}" for s_name in SYSTEMS)
              + " | span accuracy " + "; ".join(f"{n} {c['accuracy']:.3f}" for n, c in res["classification"][h].items()), flush=True)
    res["minutes"] = (time.time() - t_start) / 60
    np.savez_compressed(run_dir / "predictions.npz", **arrays); (run_dir / "seed.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
