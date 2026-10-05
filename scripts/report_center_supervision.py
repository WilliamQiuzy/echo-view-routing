#!/usr/bin/env python
"""Evaluate, freeze and report the nine models of the target-time supervision round (plan Task 4;
configs/experiments/ev9v_center_supervision_v1.yaml).

1. Checks. All nine runs are done with their identities, and their final checkpoints match the status and run manifests.
   S and C of a seed read the same manifest file and initial state with equal per-epoch input digests. N shares the
   initial state and the first four windows of every update.
2. Outputs. For every model the trained encoder (export plus final trainable blocks) extracts its own features for every
   frame of the two validation halves (the frames of B7's window plans; the test split is never read). The head scores
   B7's windows (centres every 5 samples, 16 frames, offsets s - c), and every sample takes its nearest window: the
   direct argmax row. B3 with the fixed 40-setting grid is chosen on the calibration half only and applied once to the
   selection half; with no feasible setting the fixed diagnostic setting is reported instead and passes nothing.
3. Results. Metrics per arm, seed and half; C - S paired bootstrap; native per-class precision and recall; A4C/A5C and
   SC4C -> PLAX fixes and new errors with their sources; the window-type census and the paired context probe; the
   reference systems reproduced from their frozen outputs; the final goal; end-to-end inference time.
4. Freezing. The nine models go through scripts/freeze_model.py (skipped with --no-freeze).

  python scripts/report_center_supervision.py --config configs/experiments/ev9v_center_supervision_v1.yaml \
      --status runs/center_supervision_v1/status.json --out docs/results/2026-10-05_center_supervision_training
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
from scripts.calibrate_simple_decoders import grid as simple_grid  # noqa: E402
from scripts.ev9v_complementarity import decode, md_table  # noqa: E402
from scripts.report_model_development import paired_bootstrap  # noqa: E402
from scripts.train_center_supervision import build_model, identity  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import select_setting, sha256_file  # noqa: E402
from echo_routing.evaluate.dev_metrics import evaluate_predictions  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.window_target import category, nearest_centre, window_frames  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402
from echo_routing.temporal.joint_visual import load_frame_u8  # noqa: E402
from echo_routing.temporal.windowed import stream_plan, window_centres  # noqa: E402

CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
PAIRS = {"A4C read as A5C": (2, 3), "A5C read as A4C": (3, 2), "SC4C read as PLAX": (4, 0)}
FLAT = ("native_macro_f1", "segment_f1_50_multi", "boundary_recall_multi", "boundary_precision_multi", "boundary_f1_multi",
        "native_internal_cuts_per_clip", "same_class_cut_rate_multi", "segment_f1_50_pairs", "boundary_recall_pairs")
ARM_RUN = "runs/center_supervision_v1/train/{key}"


# ----------------------------------------------------------------------------------------------- 1. checks
def verify(cfg: dict, cs: dict, paths, status_path: Path, manifest_root: Path) -> dict:
    status, index = json.loads(status_path.read_text()), json.loads((manifest_root / "index.json").read_text())
    total = int(cs["sampling"]["epochs"]) * int(cs["sampling"]["updates_per_epoch"]); runs = {}
    for seed in cs["seeds"]:
        files = index["seeds"][str(seed)]["files"]
        for arm in cs["arms"]:
            key = f"{arm}_s{seed}"; st = status.get(key, {})
            ident = identity(cfg, arm, seed, files["N" if arm == "N" else "SC"]["sha256"], files["init"]["sha256"])
            man = json.loads((paths.repo_root / ARM_RUN.format(key=key) / "run_manifest.json").read_text())
            final = Path(man["final_checkpoint"])
            if not (st.get("state") == "done" and st.get("identity") == ident == man["identity"] and man["updates"] == total
                    and man["frozen_unchanged"] and sha256_file(final) == man["final_sha256"] == st.get("final_sha256")):
                raise SystemExit(f"{key}: not a complete, verified run")
            runs[key] = man
        s, c, n = runs[f"S_s{seed}"], runs[f"C_s{seed}"], runs[f"N_s{seed}"]
        if not (s["manifest_sha256"] == c["manifest_sha256"] and s["init_sha256"] == c["init_sha256"] == n["init_sha256"]
                and [e["input_digest"] for e in s["epochs"]] == [e["input_digest"] for e in c["epochs"]]):
            raise SystemExit(f"seed {seed}: S and C are not paired")
        with np.load(s["manifest"]) as a, np.load(n["manifest"]) as b:
            same = all(np.array_equal(a[k][:, :4], b[k][:, :4]) for k in ("frame", "gamma", "rel", "labels", "target")) and \
                np.array_equal(a["videos"][a["vid"][:, :4]], b["videos"][b["vid"][:, :4]])
        if not same:
            raise SystemExit(f"seed {seed}: N does not share the first four windows of S/C")
    return runs


# ----------------------------------------------------------------------------------------------- 2. outputs
def validation(cfg: dict, cs: dict, paths) -> tuple[Inputs, dict]:
    """The two halves with every sequence's frame references (B7's plans must agree) and labels."""
    root = paths.repo_root
    inputs = Inputs(cfg, root / cs["inputs"]["protocol"]["path"], root / cs["inputs"]["feature_manifest"]["path"], paths.cache_root)
    plans = json.loads((root / "data/window_plans/beb36d80df31_protocol_a897483b776e/plans.json").read_text())["streams"]
    seqs = {}
    for h in HALVES:
        ev = inputs.halves[h]; seqs[h] = []
        for s in ev.natives:
            refs = [(s.seq_id, int(f), 1.0) for f in inputs.store.get(s.seq_id, "orig").frame_idx]
            seqs[h].append({"id": s.seq_id, "kind": "native", "labels": s.labels.astype(int), "refs": refs})
        for kind, coll in (("pairs", ev.pairs), ("multi", ev.multi)):
            for st in coll:
                refs = [(r.video_id, int(r.frame_idx), float(r.gamma)) for r in stream_plan(st.recipe, inputs.store.get)]
                seqs[h].append({"id": st.recipe.recipe_id, "kind": kind, "labels": st.labels.astype(int), "refs": refs})
        for q in seqs[h]:
            p = plans[q["id"]]["refs"]
            if len(p) != len(q["refs"]) or any(a[0] != b[0] or int(a[1]) != int(b[1]) or abs(float(a[2]) - float(b[2])) > 1e-9 for a, b in zip(q["refs"], p)):
                raise SystemExit(f"{q['id']}: frame references differ from B7's window plans")
    return inputs, seqs


class Frames(Dataset):
    def __init__(self, triples: list[tuple], frame_paths, chunk: int = 256) -> None:
        self.t, self.fp, self.chunk = triples, frame_paths, chunk

    def __len__(self) -> int:
        return (len(self.t) + self.chunk - 1) // self.chunk

    def __getitem__(self, i: int) -> torch.Tensor:
        out = []
        for v, f, g in self.t[i * self.chunk:(i + 1) * self.chunk]:
            p = self.fp(v); out.append(load_frame_u8(p[min(int(f), len(p) - 1)], float(g)))
        return torch.from_numpy(np.stack(out))


@torch.no_grad()
def extract(model, triples: list[tuple], frame_paths, device, workers: int) -> np.ndarray:
    model.eval(); feats = []
    for u8 in DataLoader(Frames(triples, frame_paths), batch_size=None, num_workers=workers, prefetch_factor=2 if workers else None):
        feats.append(model.encoder(u8.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(feats)


@torch.no_grad()
def windows_of(model, feats: np.ndarray, at: dict, seq: dict, stride: int, device) -> dict:
    n = len(seq["labels"]); centres = window_centres(n, stride)
    s = np.stack([window_frames(int(c), n, 16, 16) for c in centres]); rel = s - centres[:, None]
    idx = np.vectorize(lambda i: at[seq["refs"][i]])(s)
    logits, _ = model.head(torch.from_numpy(feats[idx]).to(device), torch.from_numpy(rel).to(device))
    wprob = torch.softmax(logits, -1).cpu().numpy()
    return {"centres": centres, "frames": s, "wprob": wprob, "prob": wprob[nearest_centre(n, centres)]}


def b3_choose(cfg: dict, cs: dict, probs_cal: dict, ev) -> dict:
    settings = [p for m, p in simple_grid(cfg, [probs_cal[s.seq_id] for s in ev.natives]) if m == "b3"]
    recs = []
    for order, params in enumerate(settings):
        m, _ = evaluate(cfg, decode(params, probs_cal), ev)
        recs.append({"order": order, "params": params, **{k: m[k] for k in FLAT}})
    sel = select_setting(recs, 0.980, 0.95)
    if sel["chosen"] is None:
        d = cs["evaluation"]["b3_infeasible_diagnostic"]
        diag = next(r for r in recs if r["params"]["half_window"] == int(d["half_window"]) and abs(r["params"]["threshold_quantile"] - float(d["quantile"])) < 1e-9)
        return {"status": "infeasible (diagnostic setting shown)", "params": diag["params"], "n_feasible": 0, "grid": recs}
    return {"status": "feasible", "params": sel["chosen"]["params"], "n_feasible": sel["n_feasible"], "grid": recs}


def evaluate(cfg: dict, preds: dict, ev) -> tuple[dict, dict]:
    return evaluate_predictions(preds, ev, float(cfg["protocol"]["hz"]), float(cfg["selection"]["boundary_tolerance_s"]))


def summary(m: dict) -> dict:
    conf = np.asarray(m["native_confusion"]); col = conf.sum(0); row = conf.sum(1)
    return {**{k: m[k] for k in FLAT}, "native_recall": {CLASSES[k]: float(conf[k, k] / row[k]) if row[k] else None for k in range(5)},
            "native_precision": {CLASSES[k]: float(conf[k, k] / col[k]) if col[k] else None for k in range(5)},
            "native_confusion": conf.tolist(), **{p: int(conf[a, b]) for p, (a, b) in PAIRS.items()}}


def native_changes(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Native clips fixed (b made the error, a is right) and newly wrong (a makes it, b was right), per error pair."""
    j = a.merge(b, on=["video_id", "label"], suffixes=("_a", "_b")); out = {}
    for name, (y, x) in PAIRS.items():
        fixed = j[(j["label"] == y) & (j["majority_b"] == x) & (j["majority_a"] == y)]["video_id"].tolist()
        new = j[(j["label"] == y) & (j["majority_a"] == x) & (j["majority_b"] == y)]["video_id"].tolist()
        out[name] = {"fixed": fixed, "new": new, "fixed_sessions": len({session_of(v) for v in fixed}), "new_sessions": len({session_of(v) for v in new})}
    out["all clips"] = {"fixed": int(((j["majority_b"] != j["label"]) & (j["majority_a"] == j["label"])).sum()),
                        "new": int(((j["majority_a"] != j["label"]) & (j["majority_b"] == j["label"])).sum())}
    return out


def census(seqs_h: list, outs: dict) -> list[dict]:
    rows = []
    for q in seqs_h:
        o = outs[q["id"]]; vids = [r[0] for r in q["refs"]]
        for c, fr, wp in zip(o["centres"], o["frames"], o["wprob"]):
            rows.append({"category": category([vids[i] for i in fr], q["labels"][fr]), "wrong": int(wp.argmax()) != int(q["labels"][c]),
                         "recording": vids[c]})
    df = pd.DataFrame(rows)
    return [{"category": k, "windows": len(g), "error": float(g["wrong"].mean()), "recording_averaged_error": float(g.groupby("recording")["wrong"].mean().mean()),
             "recordings_with_error": int(g[g["wrong"]]["recording"].nunique())} for k, g in df.groupby("category")]


@torch.no_grad()
def probe(model, feats: np.ndarray, at: dict, probe_dir: Path, device, bs: dict) -> dict:
    spec = json.loads((probe_dir / "windows.json").read_text())["windows"]; meta = pd.read_csv(probe_dir / "meta.csv").dropna(subset=["id"]).copy()
    ids = meta["id"].tolist(); idx = np.array([[at[(v, int(f), float(g))] for v, f, g in spec[k]] for k in ids])
    rel = torch.arange(16).repeat(len(ids), 1) - 8; out = []
    for i in range(0, len(ids), 1024):
        lg, _ = model.head(torch.from_numpy(feats[idx[i:i + 1024]]).to(device), rel[i:i + 1024].to(device)); out.append(torch.softmax(lg, -1).cpu().numpy())
    p = np.concatenate(out); meta["pred"] = p.argmax(1); meta["correct"] = meta["pred"] == meta["label"]
    meta["p_target"] = p[np.arange(len(p)), meta["label"].astype(int)]; meta["names_context"] = meta["pred"] == meta["context_label"]
    res = {}
    for m in sorted(meta["m"].dropna().unique()):
        same = meta[(meta["condition"] == "same view") & (meta["m"] == m)].set_index("target")
        oth = meta[(meta["condition"] == "other view") & (meta["m"] == m)].copy(); oth = oth[oth["target"].isin(same.index)]
        oth["d"] = oth["correct"].astype(float).to_numpy() - same.loc[oth["target"], "correct"].astype(float).to_numpy()
        new = oth[oth["d"] < 0]
        res[int(m)] = {"same_view_accuracy": float(same["correct"].mean()), "other_view_accuracy": float(oth["correct"].mean()),
                       "difference": float(oth["d"].mean()), "ci95": session_ci(oth, "d", int(bs["draws"]), int(bs["seed"])),
                       "additional_errors": len(new), "naming_context": int(new["names_context"].sum()), "targets": int(new["target"].nunique())}
    return res


# ----------------------------------------------------------------------------------------------- references
def references(cfg: dict, paths, inputs: Inputs, seqs: dict) -> dict:
    """E0 + B3, E1 + B3, B7 + B3 and E1 + ASRF seeds 0-2 from their frozen outputs; each must reproduce its reported rows."""
    root = paths.repo_root; out = {}
    e0 = root / "runs/20261002-061914-md_simple_decoders-8e834aa6-s0"; e1 = root / "runs/20261002-144641-md_simple_decoders-38bbcf7c-s0"
    b7 = root / "runs/20261002-145232-md_posterior_decoders_B7-38bbcf7c-s0"; pred_b7 = root / "cache_v2/echoviewclip_windowed/beb36d80df31_protocol_a897483b776e"
    asrf = {k: root / d for k, d in {0: "runs/20261002-142950-md_M1_lr0.0003-38bbcf7c-s0", 1: "runs/20261002-142955-md_M1_lr0.0003-77963b09-s1",
                                    2: "runs/20261002-143000-md_M1_lr0.0003-313524a1-s2"}.items()}
    e0_store = Inputs(cfg, root / cfg["center_supervision"]["inputs"]["protocol"]["path"],
                      root / "runs/20261002-053334-feature_manifest-6941483d-s0/feature_manifest.json", paths.cache_root)
    systems = {"E0 + B3": (json.loads((e0 / "selected.json").read_text())["selected"]["b3"]["main"], e0_store, e0, "rows_b3_tuned_main_{h}_multi_seg.csv"),
               "E1 + B3": (json.loads((e1 / "selected.json").read_text())["selected"]["b3"]["main"], inputs, e1, "rows_b3_tuned_main_{h}_multi_seg.csv")}
    for name, (params, inp, run, rows) in systems.items():
        out[name] = {}
        for h in HALVES:
            ev = inp.halves[h]; probs = {s.seq_id: inp.store.get(s.seq_id, "orig").prob for s in ev.natives}
            probs.update({s.recipe.recipe_id: s.prob for s in (*ev.pairs, *ev.multi)})
            m, t = evaluate(cfg, decode(params, probs), ev); check_rows(name, h, t, run / rows.format(h=h))
            out[name][h] = {"metrics": summary(m), "tables": t}
    params = json.loads((b7 / "posterior_decoders.json").read_text())["results"]["B7"]["families"]["b3"]["params"]; out["B7 + B3"] = {}
    for h in HALVES:
        probs = {q["id"]: np.load(pred_b7 / f"{q['id']}.npz")["prob"] for q in seqs[h]}
        m, t = evaluate(cfg, decode(params, probs), inputs.halves[h]); check_rows("B7 + B3", h, t, b7 / f"rows_B7_b3_{h}_multi_seg.csv")
        out["B7 + B3"][h] = {"metrics": summary(m), "tables": t}
    for k, run in asrf.items():
        out[f"E1 + ASRF s{k}"] = {}
        for h in HALVES:
            with np.load(run / f"predictions_{h}.npz") as z:
                preds = {q["id"]: z[q["id"]].astype(int) for q in seqs[h]}
            m, t = evaluate(cfg, preds, inputs.halves[h]); check_rows(f"E1 + ASRF s{k}", h, t, run / f"rows_{h}_multi_seg.csv")
            out[f"E1 + ASRF s{k}"][h] = {"metrics": summary(m), "tables": t}
    return out


def check_rows(name: str, h: str, tables: dict, rows_csv: Path) -> None:
    cols = ["seg_matched", "seg_pred", "seg_true"]
    if not np.array_equal(tables["multi_seg"][cols].sum().to_numpy(), pd.read_csv(rows_csv)[cols].sum().to_numpy()):
        raise SystemExit(f"{name} {h}: does not reproduce {rows_csv}")


# ----------------------------------------------------------------------------------------------- main
def freeze(paths, runs: dict, cs: dict, config: str, manifest_root: Path) -> dict:
    out = {}
    for key, man in runs.items():
        seed = man["seed"]; files = [man["final_checkpoint"], str(paths.repo_root / ARM_RUN.format(key=key) / "run_manifest.json"),
                                     man["init"], str(manifest_root / "index.json"), str(paths.repo_root / config)]
        cmd = [sys.executable, "scripts/freeze_model.py", "center_supervision_v1", key, *sum((["--file", f] for f in files), []),
               "--meta", f"identity={man['identity']}", "--meta", f"arm={man['arm']}", "--meta", f"seed={seed}",
               "--notes", "Target-time supervision round (2026-10-05); final epoch; trainable E1 blocks 8-11, ln_post, proj and the temporal head."]
        dest = paths.repo_root / "models_frozen" / "center_supervision_v1" / key
        if not dest.exists():
            subprocess.run(cmd, cwd=paths.repo_root, check=True, capture_output=True)
        out[key] = str(dest)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--status", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest-root", default="runs/center_supervision_v1/manifests"); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-freeze", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="code check only: initial weights, a few sequences, no metrics printed or written")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); cs = cfg["center_supervision"]; paths = load_paths(); ev_cfg = cs["evaluation"]
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    inputs, seqs = validation(cfg, cs, paths)
    if a.dry_run:
        dry_run(cfg, cs, paths, inputs, seqs, Path(a.manifest_root), device, a.workers); return
    runs = verify(cfg, cs, paths, Path(a.status), Path(a.manifest_root)); print("nine runs verified and paired", flush=True)
    triples = sorted({r for h in HALVES for q in seqs[h] for r in q["refs"]}); at = {t: i for i, t in enumerate(triples)}
    man = load_manifest(paths.manifest_path).set_index("video_id"); frame_paths = FramePaths({v: str(man.loc[v, "frames_dir"]) for v in {t[0] for t in triples}})
    stride = int(cs["window"]["stride"]); bs = {"draws": 2000, "seed": 20261002}
    models, timing = {}, {}
    for key, run in runs.items():
        t0 = time.time(); model, _ = build_model(cs, paths, Path(run["init"]), device)
        st = torch.load(run["final_checkpoint"], map_location="cpu", weights_only=False)
        with torch.no_grad():
            for n, p in model.encoder.trainable():
                p.copy_(st["encoder_trainable"][n].to(device))
        model.head.load_state_dict(st["head"]); model.eval()
        t1 = time.time(); feats = extract(model, triples, frame_paths, device, a.workers); t2 = time.time()
        outs = {h: {q["id"]: windows_of(model, feats, at, q, stride, device) for q in seqs[h]} for h in HALVES}; t3 = time.time()
        rows, tables = {}, {}
        for row in ("argmax", "b3"):
            preds = {h: {k: o["prob"].argmax(1) for k, o in outs[h].items()} for h in HALVES} if row == "argmax" else None
            if row == "b3":
                ch = b3_choose(cfg, cs, {k: o["prob"] for k, o in outs["calibration"].items()}, inputs.halves["calibration"])
                preds = {h: decode(ch["params"], {k: o["prob"] for k, o in outs[h].items()}) for h in HALVES}
                rows["b3_choice"] = {k: v for k, v in ch.items() if k != "grid"}; rows["b3_grid"] = ch["grid"]
            for h in HALVES:
                m, t = evaluate(cfg, preds[h], inputs.halves[h]); rows[f"{row}|{h}"] = summary(m); tables[f"{row}|{h}"] = t
        t4 = time.time()
        rows["census"] = {h: census(seqs[h], outs[h]) for h in HALVES}
        rows["probe"] = probe(model, feats, at, paths.repo_root / "runs/b7_window_target/probe", device, bs)
        n_samples = sum(len(q["labels"]) for h in HALVES for q in seqs[h])
        timing[key] = {"frames": len(triples), "samples": n_samples, "extract_s": t2 - t1, "windows_s": t3 - t2, "decode_and_metrics_s": t4 - t3,
                       "ms_per_sample_end_to_end": 1000 * (t4 - t0) / n_samples}
        models[key] = {"rows": rows, "tables": tables}
        print(f"{key}: B3 {rows['b3_choice']['status']} | sel {rows['b3|selection']['segment_f1_50_multi']:.4f} cal "
              f"{rows['b3|calibration']['segment_f1_50_multi']:.4f} | native {rows['b3|selection']['native_macro_f1']:.4f} / "
              f"{rows['b3|calibration']['native_macro_f1']:.4f} ({t4 - t0:.0f}s)", flush=True)
        del model, feats; torch.cuda.empty_cache()
    refs = references(cfg, paths, inputs, seqs); print("references reproduced", flush=True)
    mech = {}
    for seed in cs["seeds"]:
        for row in ("b3", "argmax"):
            for h in HALVES:
                c, s = models[f"C_s{seed}"]["tables"][f"{row}|{h}"], models[f"S_s{seed}"]["tables"][f"{row}|{h}"]
                mech[f"s{seed}|{row}|{h}"] = {"bootstrap": paired_bootstrap(c, s), "native_changes": native_changes(c["native"], s["native"]),
                                              "vs_E1_B3": native_changes(c["native"], refs["E1 + B3"][h]["tables"]["native"])}
    goal = {}
    ref = ev_cfg["references"]; floors = {"selection": 0.973, "calibration": 0.980}
    for arm in cs["arms"]:
        rows = [models[f"{arm}_s{seed}"]["rows"] for seed in cs["seeds"]]
        mean = {h: float(np.mean([r[f"b3|{h}"]["segment_f1_50_multi"] for r in rows])) for h in HALVES}
        per_seed = [{h: {"feasible": r["b3_choice"]["status"] == "feasible", "recall": r[f"b3|{h}"]["boundary_recall_multi"],
                         "native": r[f"b3|{h}"]["native_macro_f1"]} for h in HALVES} for r in rows]
        ok_seeds = all(p[h]["feasible"] and (p[h]["recall"] or 0) >= 0.95 and p[h]["native"] >= floors[h] for p in per_seed for h in HALVES)
        goal[f"{arm} + B3"] = {"mean_segment_f1": mean, "margin": {h: mean[h] - float(ref[h]) for h in HALVES}, "per_seed": per_seed,
                               "met": bool(ok_seeds and all(mean[h] >= float(ref[h]) + 0.01 for h in HALVES)),
                               "sensitivity_calibration": mean["calibration"] - float(ref["calibration_sensitivity"])}
    frozen = {} if a.no_freeze else freeze(paths, runs, cs, a.config, Path(a.manifest_root))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = {"runs": {k: {x: v[x] for x in ("identity", "arm", "seed", "final_sha256", "manifest_sha256", "init_sha256", "updates")} | {"epochs": v["epochs"]}
                    for k, v in runs.items()},
           "models": {k: v["rows"] for k, v in models.items()}, "mechanism": mech, "goal": goal, "timing": timing, "frozen": frozen,
           "references": {k: {h: v[h]["metrics"] for h in HALVES} for k, v in refs.items()}}
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    (out / "report_tables.md").write_text(tables_md(rep, cs)); print(json.dumps(goal, indent=1, default=str)); print(f"wrote {out}")


def dry_run(cfg: dict, cs: dict, paths, inputs: Inputs, seqs: dict, manifest_root: Path, device, workers: int) -> None:
    """Runs every evaluation step on the initial weights of seed 0 and two native clips, one pair and one multi stream per
    half. Prints shapes and timing only; no metric is printed or written."""
    from echo_routing.temporal.dev_data import EvalSet
    index = json.loads((manifest_root / "index.json").read_text())
    model, _ = build_model(cs, paths, manifest_root / index["seeds"]["0"]["files"]["init"]["file"], device); model.eval()
    sub, evs = {}, {}
    for h in HALVES:
        ev = inputs.halves[h]; keep = {s.seq_id for s in ev.natives[:2]} | {ev.pairs[0].recipe.recipe_id, ev.multi[0].recipe.recipe_id}
        sub[h] = [q for q in seqs[h] if q["id"] in keep]; evs[h] = EvalSet(h, ev.natives[:2], ev.pairs[:1], ev.multi[:1])
    triples = sorted({r for h in HALVES for q in sub[h] for r in q["refs"]}); at = {t: i for i, t in enumerate(triples)}
    man = load_manifest(paths.manifest_path).set_index("video_id"); fp = FramePaths({v: str(man.loc[v, "frames_dir"]) for v in {t[0] for t in triples}})
    t0 = time.time(); feats = extract(model, triples, fp, device, workers)
    outs = {h: {q["id"]: windows_of(model, feats, at, q, int(cs["window"]["stride"]), device) for q in sub[h]} for h in HALVES}
    ch = b3_choose(cfg, cs, {k: o["prob"] for k, o in outs["calibration"].items()}, evs["calibration"])
    for h in HALVES:
        evaluate(cfg, decode(ch["params"], {k: o["prob"] for k, o in outs[h].items()}), evs[h])
    _ = census(sub["selection"], outs["selection"])
    print(json.dumps({"dry_run": "ok", "frames": len(triples), "feature_shape": list(feats.shape), "windows": {h: int(sum(len(o["centres"]) for o in outs[h].values())) for h in HALVES},
                      "b3_grid_settings": len(ch["grid"]), "seconds": time.time() - t0}))


def tables_md(rep: dict, cs: dict) -> str:
    rows = []
    for key, r in rep["models"].items():
        for row in ("b3", "argmax"):
            for h in HALVES:
                m = r[f"{row}|{h}"]
                rows.append({"model": key, "row": row, "half": h, "B3 status": r["b3_choice"]["status"] if row == "b3" else "",
                             "segment F1@50": m["segment_f1_50_multi"], "boundary recall": m["boundary_recall_multi"],
                             "boundary precision": m["boundary_precision_multi"], "native macro-F1": m["native_macro_f1"],
                             "cuts per clip": m["native_internal_cuts_per_clip"], **{p: m[p] for p in PAIRS}})
    ref_rows = [{"system": k, "half": h, "segment F1@50": v[h]["segment_f1_50_multi"], "boundary recall": v[h]["boundary_recall_multi"],
                 "native macro-F1": v[h]["native_macro_f1"], **{p: v[h][p] for p in PAIRS}} for k, v in rep["references"].items() for h in HALVES]
    mech = [{"comparison": k, **{f"d {m}": f"{b[m]['delta']:+.4f} [{b[m]['ci95'][0]:+.4f}, {b[m]['ci95'][1]:+.4f}]"
                                 for m in ("multi segment F1@50", "native macro-F1", "multi boundary recall") for b in [v["bootstrap"]]}}
            for k, v in rep["mechanism"].items()]
    return "\n".join(["# Target-time supervision: tables\n", "## Models\n", md_table(rows), "## References\n", md_table(ref_rows),
                      "## C - S (paired bootstrap; streams reuse recordings)\n", md_table(mech),
                      "## Goal\n", "```\n" + json.dumps(rep["goal"], indent=1, default=str) + "\n```\n"])


if __name__ == "__main__":
    main()
