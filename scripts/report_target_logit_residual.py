#!/usr/bin/env python
"""Evaluate, freeze and report the target-logit residual round (plan Task 5-6;
configs/experiments/target_logit_residual_v1.yaml).

1. Checks. The six new runs are done with their identities, final checkpoints and Z/R pairing (one manifest, one
   initial state and equal per-epoch input digests per seed). The three old C models are verified with their own
   training config. The reference systems (E0 + B3, E1 + B3, B7 + B3, E1 + ASRF) reproduce from their frozen outputs.
2. Outputs. Every model extracts its own features for every validation frame (B7's window plans; no test split). Its
   final logits are read on B7's windows (R: target-frame logits + the CLS correction; Z and C: the CLS readout), every
   sample takes its nearest window (argmax row), and B3 is chosen on calibration only and applied once to selection
   (the fixed 9/0.95/5 setting is shown as a diagnostic when nothing is feasible). Every R model is also read once with
   the correction set to zero, which must equal its J-grid path.
3. Results. Native confusion, macro-F1, per-class counts and fixes/new errors by recording; target-sample error in
   native clips, single-recording windows and mixed windows (each recording weighted once); the 474-target context
   probe; internal cuts and same-class join cuts; segment F1@50 and boundary precision/recall/F1; R - Z, R - C and
   Z - C with the paired bootstrap; the final goal for R + B3.
4. The six new models are frozen with scripts/freeze_model.py (skipped with --no-freeze). Per-sequence probabilities
   of every model stay on the server; the repository gets summaries without recording ids.

--dry-run uses only training data (a few training clips and streams) and the untrained initial states, prints no
metric, and checks that the old C models load and reproduce the old forward through the new readout (mode Z).

  python scripts/report_target_logit_residual.py --config configs/experiments/target_logit_residual_v1.yaml \
      --status runs/target_logit_residual_v1/status.json --out docs/results/2026-10-06_target_logit_residual
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.b7_window_target import session_ci  # noqa: E402
from scripts.diagnose_center_recognition import native_bootstrap, to_grid  # noqa: E402
from scripts.ev9v_complementarity import decode, md_table  # noqa: E402
from scripts.report_center_supervision import (PAIRS, b3_choose, evaluate, extract, native_changes, references, summary,  # noqa: E402
                                               validation, verify)
from scripts.report_model_development import paired_bootstrap  # noqa: E402
from scripts.train_center_supervision import build_model  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402
from scripts.train_target_logit_residual import NS, build, check_source, identity, prepare_inits  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_config, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.window_target import category, nearest_centre, window_frames  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402
from echo_routing.temporal.dev_run import HALVES  # noqa: E402
from echo_routing.temporal.windowed import window_centres  # noqa: E402
from echo_routing.vit.residual_target import readout, target_logits  # noqa: E402

STRIDE = 5


# ----------------------------------------------------------------------------------------------- model reading
class Reader:
    """Final window logits of one model: 'C' (old CLS readout), 'Z' or 'R' (new readout), optionally with delta = 0."""

    def __init__(self, model, mode: str, device) -> None:
        self.model, self.mode, self.device = model, mode, device

    @torch.no_grad()
    def logits(self, emb: np.ndarray, rel: np.ndarray, delta_zero: bool = False) -> torch.Tensor:
        e, r = torch.from_numpy(emb).to(self.device), torch.from_numpy(rel).to(self.device)
        delta, frame_logits = self.model.head(e, r)
        if delta_zero:
            return target_logits(frame_logits, r)
        if self.mode == "C":
            return delta
        return readout(delta, frame_logits, r, self.mode)[0]


def windows(reader: Reader, feats: np.ndarray, at: dict, seq: dict, delta_zero: bool = False) -> dict:
    n = len(seq["labels"]); centres = window_centres(n, STRIDE)
    s = np.stack([window_frames(int(c), n, 16, 16) for c in centres]); rel = s - centres[:, None]
    idx = np.vectorize(lambda i: at[seq["refs"][i]])(s)
    wprob = torch.softmax(reader.logits(feats[idx], rel, delta_zero), -1).cpu().numpy()
    return {"centres": centres, "frames": s, "wprob": wprob, "prob": wprob[nearest_centre(n, centres)]}


@torch.no_grad()
def frame_probs(model, feats: np.ndarray, at: dict, seq: dict, device) -> np.ndarray:
    x = torch.from_numpy(feats[[at[r] for r in seq["refs"]]]).to(device)
    return torch.softmax(model.head.frame_head(x), -1).cpu().numpy()


def census3(seqs_h: list, outs: dict) -> list[dict]:
    """Target-sample error in native clips, single-recording stream windows and mixed-view windows."""
    rows = []
    for q in seqs_h:
        o = outs[q["id"]]; vids = [r[0] for r in q["refs"]]
        for c, fr, wp in zip(o["centres"], o["frames"], o["wprob"]):
            cat = category([vids[i] for i in fr], q["labels"][fr])
            group = "native clips" if q["kind"] == "native" else ("mixed views" if cat == "mixed views" else "single-recording stream windows" if cat == "single recording" else "cross-recording, one view")
            rows.append({"group": group, "wrong": int(wp.argmax()) != int(q["labels"][c]), "recording": vids[c]})
    df = pd.DataFrame(rows)
    return [{"group": g, "windows": len(x), "error": float(x["wrong"].mean()), "recording_weighted": float(x.groupby("recording")["wrong"].mean().mean()),
             "recordings_with_error": int(x[x["wrong"]]["recording"].nunique())} for g, x in df.groupby("group")]


def probe(reader: Reader, feats: np.ndarray, at: dict, probe_dir: Path) -> dict:
    spec = json.loads((probe_dir / "windows.json").read_text())["windows"]; meta = pd.read_csv(probe_dir / "meta.csv").dropna(subset=["id"]).copy()
    idx = np.array([[at[(v, int(f), float(g))] for v, f, g in spec[k]] for k in meta["id"]]); rel = np.tile(np.arange(16) - 8, (len(meta), 1))
    p = np.concatenate([torch.softmax(reader.logits(feats[idx[i:i + 1024]], rel[i:i + 1024]), -1).cpu().numpy() for i in range(0, len(meta), 1024)])
    meta["pred"] = p.argmax(1); meta["correct"] = meta["pred"] == meta["label"]; meta["names_context"] = meta["pred"] == meta["context_label"]
    res = {}
    for m in sorted(meta["m"].dropna().unique()):
        same = meta[(meta["condition"] == "same view") & (meta["m"] == m)].set_index("target")
        oth = meta[(meta["condition"] == "other view") & (meta["m"] == m)].copy(); oth = oth[oth["target"].isin(same.index)]
        oth["d"] = oth["correct"].astype(float).to_numpy() - same.loc[oth["target"], "correct"].astype(float).to_numpy()
        new = oth[oth["d"] < 0]
        res[int(m)] = {"same_view_accuracy": float(same["correct"].mean()), "other_view_accuracy": float(oth["correct"].mean()),
                       "difference": float(oth["d"].mean()), "ci95": session_ci(oth, "d", 2000, 20261002), "additional_errors": len(new),
                       "naming_context": int(new["names_context"].sum())}
    return res


# ----------------------------------------------------------------------------------------------- checks
def verify_new(cfg: dict, rt: dict, cs: dict, paths, status_path: Path) -> dict:
    root = paths.repo_root; source = check_source(rt, root); inits = prepare_inits(cfg, paths)
    index = json.loads((root / rt["source"]["manifest_index"]["path"]).read_text()); status = json.loads(status_path.read_text())
    total = int(cs["sampling"]["epochs"]) * int(cs["sampling"]["updates_per_epoch"]); runs = {}
    for seed in rt["seeds"]:
        for arm in ("Z", "R"):
            key = f"{arm}_s{seed}"; st = status.get(key, {})
            ident = identity(cfg, arm, seed, index["seeds"][str(seed)]["files"]["SC"]["sha256"], inits[str(seed)]["sha256"], source, root)
            man = json.loads((root / "runs" / NS / "train" / key / "run_manifest.json").read_text()); final = Path(man["final_checkpoint"])
            if not (st.get("state") == "done" and st.get("identity") == ident == man["identity"] and man["updates"] == total and man["frozen_unchanged"]
                    and sha256_file(final) == man["final_sha256"] == st.get("final_sha256")):
                raise SystemExit(f"{key}: not a complete, verified run")
            runs[key] = man
        z, r = runs[f"Z_s{seed}"], runs[f"R_s{seed}"]
        if not (z["manifest_sha256"] == r["manifest_sha256"] and z["init_sha256"] == r["init_sha256"]
                and [e["input_digest"] for e in z["epochs"]] == [e["input_digest"] for e in r["epochs"]]):
            raise SystemExit(f"seed {seed}: Z and R are not paired")
    return runs


def load_new(cs: dict, paths, man: dict, device):
    model, _ = build(cs, paths, Path(man["init"]), man["arm"], device)
    st = torch.load(man["final_checkpoint"], map_location="cpu", weights_only=False)
    with torch.no_grad():
        for n, p in model.encoder.trainable():
            p.copy_(st["encoder_trainable"][n].to(device))
    model.head.load_state_dict(st["head"]); model.eval()
    return model


def load_old_c(cs: dict, paths, man: dict, device):
    model, _ = build_model(cs, paths, Path(man["init"]), device)
    st = torch.load(man["final_checkpoint"], map_location="cpu", weights_only=False)
    with torch.no_grad():
        for n, p in model.encoder.trainable():
            p.copy_(st["encoder_trainable"][n].to(device))
    model.head.load_state_dict(st["head"]); model.eval()
    return model


# ----------------------------------------------------------------------------------------------- evaluation
def evaluate_model(cfg: dict, cs: dict, reader: Reader, model, feats: np.ndarray, at: dict, inputs, seqs: dict, probe_dir: Path, device) -> tuple[dict, dict, dict]:
    outs = {h: {q["id"]: windows(reader, feats, at, q) for q in seqs[h]} for h in HALVES}
    rows, tables = {}, {}
    ch = b3_choose(cfg, cs, {k: o["prob"] for k, o in outs["calibration"].items()}, inputs.halves["calibration"])
    rows["b3_choice"] = {k: v for k, v in ch.items() if k != "grid"}; rows["b3_grid"] = ch["grid"]
    for row in ("argmax", "b3"):
        for h in HALVES:
            probs = {k: o["prob"] for k, o in outs[h].items()}
            preds = {k: v.argmax(1) for k, v in probs.items()} if row == "argmax" else decode(ch["params"], probs)
            m, t = evaluate(cfg, preds, inputs.halves[h]); rows[f"{row}|{h}"] = summary(m)
            tables[f"{row}|{h}"] = t
    rows["census"] = {h: census3(seqs[h], outs[h]) for h in HALVES}
    rows["probe"] = probe(reader, feats, at, probe_dir)
    if reader.mode == "R":   # the correction set to zero must equal J-grid; then what the correction keeps and breaks
        rows["delta_zero"] = {}
        for h in HALVES:
            z0 = {q["id"]: windows(reader, feats, at, q, delta_zero=True)["prob"] for q in seqs[h]}
            jg = {q["id"]: to_grid(frame_probs(model, feats, at, q, device)) for q in seqs[h]}
            diff = max(float(np.abs(z0[k] - jg[k]).max()) for k in z0)
            if diff > 1e-5:
                raise SystemExit(f"delta = 0 does not equal J-grid ({diff:.2e})")
            m0, t0 = evaluate(cfg, {k: v.argmax(1) for k, v in z0.items()}, inputs.halves[h])
            nc = native_changes(tables[f"argmax|{h}"]["native"], t0["native"])   # fixed / newly wrong by the correction
            rows["delta_zero"][h] = {"max_abs_vs_j_grid": diff, "argmax": summary(m0),
                                     "final_vs_delta_zero": {k: ({"fixed": len(v["fixed"]), "new": len(v["new"])} if isinstance(v.get("fixed"), list) else v) for k, v in nc.items()},
                                     "bootstrap": native_bootstrap(tables[f"argmax|{h}"]["native"], t0["native"], 2000, 20261002)}
    return rows, tables, outs


def compare(ta: dict, tb: dict) -> dict:
    out = {}
    for row in ("b3", "argmax"):
        for h in HALVES:
            a, b = ta[f"{row}|{h}"], tb[f"{row}|{h}"]
            ch = native_changes(a["native"], b["native"])
            out[f"{row}|{h}"] = {"bootstrap": paired_bootstrap(a, b), "native_session": native_bootstrap(a["native"], b["native"], 2000, 20261002),
                                 "native_changes": {k: ({"fixed": len(v["fixed"]), "new": len(v["new"]), "fixed_sessions": v["fixed_sessions"], "new_sessions": v["new_sessions"]}
                                                        if isinstance(v.get("fixed"), list) else v) for k, v in ch.items()},
                                 "_ids": {k: {"fixed": v["fixed"], "new": v["new"]} for k, v in ch.items() if isinstance(v.get("fixed"), list)}}
    return out


def goal(rt: dict, rows: list[dict]) -> dict:
    g = rt["goal"]; floors = {"selection": 0.973, "calibration": 0.980}
    mean = {h: float(np.mean([r[f"b3|{h}"]["segment_f1_50_multi"] for r in rows])) for h in HALVES}
    per_seed = [{h: {"feasible": r["b3_choice"]["status"] == "feasible", "recall": r[f"b3|{h}"]["boundary_recall_multi"], "native": r[f"b3|{h}"]["native_macro_f1"]}
                 for h in HALVES} for r in rows]
    ok = all(p[h]["feasible"] and (p[h]["recall"] or 0) >= 0.95 and p[h]["native"] >= floors[h] for p in per_seed for h in HALVES)
    return {"mean_segment_f1": mean, "required": {"selection": g["selection_min"], "calibration": g["calibration_min"]},
            "margin": {"selection": mean["selection"] - float(g["selection_min"]), "calibration": mean["calibration"] - float(g["calibration_min"])},
            "per_seed": per_seed, "met": bool(ok and mean["selection"] >= float(g["selection_min"]) and mean["calibration"] >= float(g["calibration_min"])),
            "sensitivity_calibration": mean["calibration"] - float(g["calibration_sensitivity"])}


def strip_ids(obj):
    if isinstance(obj, dict):
        return {k: strip_ids(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip_ids(v) for v in obj]
    return obj


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--status", default=f"runs/{NS}/status.json"); ap.add_argument("--out", required=True)
    ap.add_argument("--details", default=f"runs/{NS}/report"); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-freeze", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); cs, rt = cfg["center_supervision"], cfg["target_logit_residual"]; paths = load_paths(); root = paths.repo_root
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    train_cfg = load_config(root / rt["source"]["training_config"]["path"])
    if train_cfg["center_supervision"] != cs:
        raise SystemExit("the inherited center_supervision section differs from the source training config")
    old = verify(train_cfg, cs, paths, root / rt["source"]["status"]["path"], (root / rt["source"]["manifest_index"]["path"]).parent)
    c_runs = {k: v for k, v in old.items() if k.startswith("C_")}
    if {k: v["final_sha256"] for k, v in c_runs.items()} != rt["source"]["c_final"]:
        raise SystemExit("old C checkpoints differ from the config")
    if a.dry_run:
        dry_run(cfg, cs, rt, paths, c_runs, device, a.workers); return
    new = verify_new(cfg, rt, cs, paths, root / a.status); print("six new runs verified and paired; three C runs verified", flush=True)
    inputs, seqs = validation(cfg, cs, paths)
    triples = sorted({r for h in HALVES for q in seqs[h] for r in q["refs"]}); at = {t: i for i, t in enumerate(triples)}
    man = load_manifest(paths.manifest_path).set_index("video_id"); fp = FramePaths({v: str(man.loc[v, "frames_dir"]) for v in {t[0] for t in triples}})
    details = root / a.details; details.mkdir(parents=True, exist_ok=True); probe_dir = root / "runs/b7_window_target/probe"
    models, tables, timing = {}, {}, {}
    for key, run in [*c_runs.items(), *new.items()]:
        t0 = time.time()
        model = load_old_c(cs, paths, run, device) if key.startswith("C_") else load_new(cs, paths, run, device)
        reader = Reader(model, "C" if key.startswith("C_") else run["arm"], device)
        feats = extract(model, triples, fp, device, a.workers); t1 = time.time()
        rows, tabs, outs = evaluate_model(cfg, cs, reader, model, feats, at, inputs, seqs, probe_dir, device)
        np.savez_compressed(details / f"probs_{key}.npz", **{f"{h}|{k}": o["prob"].astype(np.float32) for h in HALVES for k, o in outs[h].items()})
        models[key], tables[key] = rows, tabs
        timing[key] = {"extract_s": t1 - t0, "total_s": time.time() - t0}
        print(f"{key}: B3 {rows['b3_choice']['status']} | sel {rows['b3|selection']['segment_f1_50_multi']:.4f} cal {rows['b3|calibration']['segment_f1_50_multi']:.4f} "
              f"| native argmax {rows['argmax|selection']['native_macro_f1']:.4f} / {rows['argmax|calibration']['native_macro_f1']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        del model, feats; torch.cuda.empty_cache()
    refs = references(cfg, paths, inputs, seqs); print("references reproduced", flush=True)
    comps = {}
    for seed in rt["seeds"]:
        for name, (x, y) in {"R - Z": ("R", "Z"), "R - C": ("R", "C"), "Z - C": ("Z", "C")}.items():
            comps[f"s{seed}|{name}"] = compare(tables[f"{x}_s{seed}"], tables[f"{y}_s{seed}"])
    res = {"models": models, "comparisons": comps, "goal_R": goal(rt, [models[f"R_s{s}"] for s in rt["seeds"]]),
           "goal_Z": goal(rt, [models[f"Z_s{s}"] for s in rt["seeds"]]), "goal_C": goal(rt, [models[f"C_s{s}"] for s in rt["seeds"]]),
           "references": {k: {h: v[h]["metrics"] for h in HALVES} for k, v in refs.items()}, "timing": timing,
           "runs": {k: {x: v[x] for x in ("identity", "arm", "seed", "final_sha256", "manifest_sha256", "init_sha256", "updates")} | {"epochs": v["epochs"]} for k, v in new.items()}}
    res["frozen"] = {} if a.no_freeze else freeze(paths, new, a.config)
    (details / "details.json").write_text(json.dumps(res, indent=1, default=str))
    out = root / a.out; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(strip_ids(res), indent=1, default=str)); (out / "report_tables.md").write_text(tables_md(strip_ids(res)))
    print(json.dumps(res["goal_R"], indent=1, default=str)); print(f"wrote {out}")


def freeze(paths, runs: dict, config: str) -> dict:
    out = {}
    for key, man in runs.items():
        files = [man["final_checkpoint"], str(paths.repo_root / "runs" / NS / "train" / key / "run_manifest.json"), man["init"], str(paths.repo_root / config)]
        dest = paths.repo_root / "models_frozen" / NS / key
        if not dest.exists():
            subprocess.run([sys.executable, "scripts/freeze_model.py", NS, key, *sum((["--file", f] for f in files), []), "--meta", f"identity={man['identity']}",
                            "--meta", f"arm={man['arm']}", "--meta", f"seed={man['seed']}",
                            "--notes", "Target-logit residual round (2026-10-06); final epoch."], cwd=paths.repo_root, check=True, capture_output=True)
        out[key] = str(dest)
    return out


def dry_run(cfg: dict, cs: dict, rt: dict, paths, c_runs: dict, device, workers: int) -> None:
    """Training data only: the evaluation steps on a few training clips and streams, the untrained Z and R states of seed
    0, and the old C model of seed 0 read through the old forward and the new readout (mode Z)."""
    from echo_routing.compose.banks import multi_bank
    from echo_routing.compose.stream_assembler import render
    from echo_routing.features.cache import read_index, variant_dir
    from echo_routing.temporal.dev_data import EvalSet, FeatureStore, native_seq
    from echo_routing.temporal.windowed import stream_plan
    from echo_routing.vit.residual_target import ResidualTargetModel
    root = paths.repo_root; key = cs["inputs"]["cache_key"]; store = FeatureStore(paths.cache_root, key)
    idx = read_index(variant_dir(paths.cache_root, key, "orig")); labels = dict(zip(idx["video_id"], idx["label_index"].astype(int)))
    proto = json.loads((root / cs["inputs"]["protocol"]["path"]).read_text()); clips = sorted(proto["train"]["clips"])[:6]
    recipes = multi_bank(idx, "train", cfg).recipes[:2]
    natives = [native_seq(v, labels[v], store.get(v, "orig")) for v in clips]; streams = [render(r, store.get) for r in recipes]
    ev = EvalSet("train-subset", tuple(natives), (streams[0],), (streams[1],))
    seqs = [{"id": v, "kind": "native", "labels": s.labels.astype(int), "refs": [(v, int(f), 1.0) for f in store.get(v, "orig").frame_idx]} for v, s in zip(clips, natives)]
    seqs += [{"id": r.recipe_id, "kind": kind, "labels": st.labels.astype(int), "refs": [(x.video_id, int(x.frame_idx), float(x.gamma)) for x in stream_plan(r, store.get)]}
             for r, st, kind in zip(recipes, streams, ("pairs", "multi"))]
    triples = sorted({r for q in seqs for r in q["refs"]}); at = {t: i for i, t in enumerate(triples)}
    man = load_manifest(paths.manifest_path).set_index("video_id"); fp = FramePaths({v: str(man.loc[v, "frames_dir"]) for v in {t[0] for t in triples}})
    inits = prepare_inits(cfg, paths); out = {"frames": len(triples)}
    for mode in ("Z", "R"):
        model, _ = build(cs, paths, root / "runs" / NS / "inits" / inits["0"]["file"], mode, device); model.eval()
        feats = extract(model, triples, fp, device, workers); reader = Reader(model, mode, device)
        outs = {q["id"]: windows(reader, feats, at, q) for q in seqs}
        z0 = {q["id"]: windows(reader, feats, at, q, delta_zero=True)["prob"] for q in seqs}
        jg = {q["id"]: to_grid(frame_probs(model, feats, at, q, device)) for q in seqs}
        ch = b3_choose(cfg, cs, {k: o["prob"] for k, o in outs.items()}, ev); evaluate(cfg, decode(ch["params"], {k: o["prob"] for k, o in outs.items()}), ev)
        out[mode] = {"delta_zero_vs_j_grid": max(float(np.abs(z0[k] - jg[k]).max()) for k in z0), "windows": int(sum(len(o["centres"]) for o in outs.values())),
                     "census_groups": [r["group"] for r in census3(seqs, outs)], "b3_settings": len(ch["grid"])}
        if mode == "R":   # untrained R reads exactly the frame head at the target
            out["R"]["start_equals_delta_zero"] = max(float(np.abs(outs[k]["prob"] - z0[k]).max()) for k in z0)
        del model; torch.cuda.empty_cache()
    old = load_old_c(cs, paths, c_runs["C_s0"], device); feats = extract(old, triples, fp, device, workers)
    new = ResidualTargetModel(old.encoder, old.head, "Z").eval()
    a = {q["id"]: windows(Reader(old, "C", device), feats, at, q)["wprob"] for q in seqs}
    b = {q["id"]: windows(Reader(new, "Z", device), feats, at, q)["wprob"] for q in seqs}
    out["old_C_through_new_readout_max_abs"] = max(float(np.abs(a[k] - b[k]).max()) for k in a)
    print(json.dumps({"dry_run": "ok (training data only, no metric)", **out}, indent=1))


def tables_md(res: dict) -> str:
    rows = []
    for key, r in res["models"].items():
        for row in ("b3", "argmax"):
            for h in HALVES:
                m = r[f"{row}|{h}"]
                rows.append({"model": key, "row": row, "half": h, "B3": r["b3_choice"]["status"] if row == "b3" else "", "segment F1@50": m["segment_f1_50_multi"],
                             "boundary P": m["boundary_precision_multi"], "boundary R": m["boundary_recall_multi"], "boundary F1": m["boundary_f1_multi"],
                             "native macro-F1": m["native_macro_f1"], "cuts per clip": m["native_internal_cuts_per_clip"], "same-class cut rate": m["same_class_cut_rate_multi"],
                             **{p: m[p] for p in PAIRS}})
    comp = []
    for k, v in res["comparisons"].items():
        for rh, c in v.items():
            b = c["bootstrap"]; ns = c["native_session"]
            comp.append({"comparison": k, "row": rh, **{f"d {m}": f"{b[m]['delta']:+.4f} [{b[m]['ci95'][0]:+.4f}, {b[m]['ci95'][1]:+.4f}]" for m in ("multi segment F1@50", "multi boundary recall")},
                         "d native macro-F1 (session)": f"{ns['d_macro_f1']:+.4f} [{ns['ci_macro_f1'][0]:+.4f}, {ns['ci_macro_f1'][1]:+.4f}]",
                         "native fixed/new": f"{c['native_changes']['all clips']['fixed']}/{c['native_changes']['all clips']['new']}"})
    cen = [{"model": k, "half": h, **{g["group"]: f"{g['error']:.4f} ({g['recording_weighted']:.4f})" for g in r["census"][h]}} for k, r in res["models"].items() for h in HALVES]
    prb = [{"model": k, **{f"{m}/16 other-view acc": f"{v['other_view_accuracy']:.3f}" for m, v in r["probe"].items()}} for k, r in res["models"].items()]
    dz = [{"model": k, "half": h, "delta=0 native macro-F1": v["argmax"]["native_macro_f1"], "delta=0 cuts": v["argmax"]["native_internal_cuts_per_clip"],
           "final vs delta=0 fixed/new": f"{v['final_vs_delta_zero']['all clips']['fixed']}/{v['final_vs_delta_zero']['all clips']['new']}"}
          for k, r in res["models"].items() if "delta_zero" in r for h, v in r["delta_zero"].items()]
    ref = [{"system": k, "half": h, "segment F1@50": v[h]["segment_f1_50_multi"], "boundary R": v[h]["boundary_recall_multi"], "native macro-F1": v[h]["native_macro_f1"]}
           for k, v in res["references"].items() for h in HALVES]
    return "\n".join(["# Target-logit residual round: tables\n", "## Models\n", md_table(rows), "## Comparisons\n", md_table(comp),
                      "## Target-sample error (recording-weighted)\n", md_table(cen), "## Context probe\n", md_table(prb), "## R with delta = 0\n", md_table(dz),
                      "## References\n", md_table(ref), "## Goal (R + B3)\n", "```\n" + json.dumps(res["goal_R"], indent=1, default=str) + "\n```\n"])


if __name__ == "__main__":
    main()
