#!/usr/bin/env python
"""Locate the native-recognition regression of the N/S/C round (docs/superpowers/plans/2026-10-06-recognition-
regression-localisation.md; configs/experiments/center_recognition_diagnostic_v1.yaml). Inference only: nothing is
trained, tuned or relabelled; the test split and EchoNet-Pediatric are not read.

Five per-sample probability paths on both validation halves:
  F-dense   the frozen E1 probe's probabilities (E1 cache), every 10 Hz sample, as in the original evaluation
  F-grid    F-dense read only at the window target samples c and filled back by nearest_centre
  J-dense   a trained checkpoint's auxiliary frame head on that checkpoint's encoder features, every sample
  J-grid    J-dense read only at c and filled back the same way
  T-window  the checkpoint's temporal Transformer and window head (the previous round's argmax row)
Main rows: argmax per sample, majority vote per native clip. Reported per path, model and half: native macro-F1,
per-class recall with denominators, wrong clips, internal cuts, and the target-sample error at every window centre
(each centre recording weighted once; single-recording and mixed-view windows apart). Paired native comparisons
(grid effect, frame-path change, temporal readout) with a session bootstrap and fixed/new clips; the eight-cell table
of the C models; a supplementary table with the one archived E1 + B3 setting. Before any of this, the previous
argmax rows and E1 + B3 must reproduce. Per-recording ids and predictions stay in the server details directory.

  python scripts/diagnose_center_recognition.py --config configs/experiments/center_recognition_diagnostic_v1.yaml \
      --out docs/results/2026-10-06_center_recognition_diagnostic
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode, md_table  # noqa: E402
from scripts.report_center_supervision import evaluate, extract, validation, verify, windows_of  # noqa: E402
from scripts.train_center_supervision import build_model  # noqa: E402
from scripts.train_joint_visual_temporal import FramePaths  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import changes, sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.window_target import category, nearest_centre, window_frames  # noqa: E402
from echo_routing.manifest import load_manifest  # noqa: E402
from echo_routing.temporal.dev_run import HALVES  # noqa: E402
from echo_routing.temporal.windowed import window_centres  # noqa: E402

CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
PATHS = ("F-dense", "F-grid", "J-dense", "J-grid", "T-window")
STRIDE = 5
CHECK_KEYS = ("native_macro_f1", "segment_f1_50_multi", "boundary_recall_multi", "boundary_precision_multi", "native_internal_cuts_per_clip")


# ----------------------------------------------------------------------------------------------- pure helpers
def to_grid(prob: np.ndarray, stride: int = STRIDE) -> np.ndarray:
    """The probabilities read only at the window target samples, every sample taking its nearest centre's row."""
    n = len(prob); centres = window_centres(n, stride)
    return np.asarray(prob)[centres][nearest_centre(n, centres)]


def native_right(tab: pd.DataFrame) -> dict[str, bool]:
    """Native clip -> majority label correct, from dev_metrics.native_table (argmax per sample, bincount majority)."""
    return {str(v): bool(m == y) for v, m, y in zip(tab["video_id"], tab["majority"], tab["label"])}


def eight_cells(f: dict[str, bool], j: dict[str, bool], t: dict[str, bool]) -> dict[str, int]:
    if set(f) != set(j) or set(f) != set(t):
        raise ValueError("the three paths cover different clips")
    c = Counter((f[k], j[k], t[k]) for k in f)
    out = {f"F{'+' if a else '-'} J{'+' if b else '-'} T{'+' if d else '-'}": int(c.get((a, b, d), 0))
           for a in (True, False) for b in (True, False) for d in (True, False)}
    if sum(out.values()) != len(f):
        raise ValueError("the eight cells do not add up to the clip count")
    return out


def macro_f1(labels: np.ndarray, pred: np.ndarray, k: int = 5) -> float:
    conf = np.zeros((k, k)); np.add.at(conf, (labels, pred), 1)
    tp = np.diag(conf); den = 2 * tp + (conf.sum(0) - tp) + (conf.sum(1) - tp)
    return float(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0).mean())


def native_bootstrap(a: pd.DataFrame, b: pd.DataFrame, draws: int, seed: int) -> dict:
    """a - b on native clips (macro-F1 and accuracy), resampling recording sessions with the same draws for both."""
    b = b.set_index("video_id").loc[a["video_id"]].reset_index()
    y, pa, pb = a["label"].to_numpy(int), a["majority"].to_numpy(int), b["majority"].to_numpy(int)
    sess = a["video_id"].map(session_of).to_numpy(); keys = np.unique(sess); idx = {s: np.flatnonzero(sess == s) for s in keys}
    rng = np.random.default_rng(seed); d_f1, d_acc = [], []
    for _ in range(draws):
        ii = np.concatenate([idx[s] for s in rng.choice(keys, size=keys.size, replace=True)])
        d_f1.append(macro_f1(y[ii], pa[ii]) - macro_f1(y[ii], pb[ii])); d_acc.append(float((pa[ii] == y[ii]).mean() - (pb[ii] == y[ii]).mean()))
    return {"d_macro_f1": macro_f1(y, pa) - macro_f1(y, pb), "ci_macro_f1": [float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))],
            "d_accuracy": float((pa == y).mean() - (pb == y).mean()), "ci_accuracy": [float(np.percentile(d_acc, 2.5)), float(np.percentile(d_acc, 97.5))],
            "sessions": int(keys.size), "clips": int(len(y))}


def per_class(tab: pd.DataFrame) -> dict:
    out = {}
    for k, name in enumerate(CLASSES):
        sub = tab[tab["label"] == k]
        out[name] = {"right": int((sub["majority"] == k).sum()), "of": int(len(sub))}
    return out


# ----------------------------------------------------------------------------------------------- checks
def check_inputs(rd: dict, root: Path) -> dict:
    out = {}
    for key, spec in rd["inputs"].items():
        if isinstance(spec, dict) and "path" in spec:
            if (got := sha256_file(root / spec["path"])) != spec["sha256"]:
                raise SystemExit(f"{key} {spec['path']}: sha256 {got[:12]} != {spec['sha256'][:12]}")
            out[key] = got
    params = json.loads((root / rd["inputs"]["e1_b3_selected"]["path"]).read_text())["selected"]["b3"]["main"]
    want = rd["inputs"]["e1_b3_params"]
    if any(abs(float(params[k]) - float(want[k])) > 0 for k in ("half_window", "threshold", "nms_radius")):
        raise SystemExit(f"archived E1 + B3 parameters {params} differ from the config {want}")
    return out


def reproduce_e1_b3(cfg: dict, rd: dict, root: Path, inputs, f_dense: dict) -> dict:
    run = root / Path(rd["inputs"]["e1_b3_selected"]["path"]).parent
    ref = json.loads((run / "validation_metrics.json").read_text())["methods"]["B3 tuned (main)"]; out = {}
    for h in HALVES:
        m, t = evaluate(cfg, decode(rd["inputs"]["e1_b3_params"], f_dense[h]), inputs.halves[h])
        diff = max(abs(m[k] - ref[h][k]) for k in ("native_macro_f1", "segment_f1_50_multi", "boundary_recall_multi", "native_internal_cuts_per_clip"))
        cols = ["seg_matched", "seg_pred", "seg_true"]
        same = np.array_equal(t["multi_seg"][cols].sum().to_numpy(), pd.read_csv(run / f"rows_b3_tuned_main_{h}_multi_seg.csv")[cols].sum().to_numpy())
        if diff > 1e-12 or not same:
            raise SystemExit(f"E1 + B3 {h} not reproduced from F-dense (max difference {diff:.2e}, segments equal {same})")
        out[h] = {"max_abs_diff": diff, "segment_f1_50_multi": m["segment_f1_50_multi"]}
    return out


# ----------------------------------------------------------------------------------------------- per model
def frame_head_probs(model, feats: np.ndarray, at: dict, seqs_h: list, device) -> dict:
    out = {}
    with torch.no_grad():
        for q in seqs_h:
            x = torch.from_numpy(feats[[at[r] for r in q["refs"]]]).to(device)
            out[q["id"]] = torch.softmax(model.head.frame_head(x), -1).cpu().numpy()
    return out


def path_rows(cfg: dict, probs: dict, ev) -> tuple[dict, dict]:
    """argmax per sample -> (summary, native table)."""
    m, t = evaluate(cfg, {k: v.argmax(1) for k, v in probs.items()}, ev)
    nat = t["native"]
    return {"native_macro_f1": m["native_macro_f1"], "per_class": per_class(nat), "wrong_clips": int((nat["majority"] != nat["label"]).sum()),
            "clips": int(len(nat)), "cuts_per_clip": m["native_internal_cuts_per_clip"], "segment_f1_50_multi": m["segment_f1_50_multi"],
            "boundary_recall_multi": m["boundary_recall_multi"], **{k: m[k] for k in CHECK_KEYS}}, t


def centre_rows(seqs_h: list, probs: dict[str, dict]) -> pd.DataFrame:
    """One row per window target sample: category, label, recording and whether each path (F, J, T) is wrong there."""
    rows = []
    for q in seqs_h:
        n = len(q["labels"]); vids = [r[0] for r in q["refs"]]
        for c in window_centres(n, STRIDE):
            fr = window_frames(int(c), n, 16, 16); y = int(q["labels"][c])
            rows.append({"category": category([vids[i] for i in fr], q["labels"][fr]), "recording": vids[c],
                         **{p: int(probs[p][q["id"]][c].argmax()) != y for p in probs}})
    return pd.DataFrame(rows)


def centre_errors(df: pd.DataFrame, paths: list[str]) -> dict:
    out = {}
    for name, g in (("all", df), *((c, df[df["category"] == c]) for c in ("single recording", "mixed views"))):
        out[name] = {"windows": len(g), "recordings": int(g["recording"].nunique()),
                     **{p: {"error": float(g[p].mean()), "recording_weighted": float(g.groupby("recording")[p].mean().mean()),
                            "recordings_with_error": int(g[g[p]]["recording"].nunique())} for p in paths}}
    return out


def compare(a_tab: pd.DataFrame, b_tab: pd.DataFrame, bs: dict) -> dict:
    fixed, new = changes(native_right(b_tab), native_right(a_tab))
    cls = dict(zip(a_tab["video_id"].astype(str), a_tab["label"].astype(int)))
    return {**native_bootstrap(a_tab, b_tab, int(bs["draws"]), int(bs["seed"])), "fixed": len(fixed), "new": len(new),
            "fixed_sessions": len({session_of(v) for v in fixed}), "new_sessions": len({session_of(v) for v in new}),
            "fixed_by_class": dict(Counter(CLASSES[cls[v]] for v in fixed)), "new_by_class": dict(Counter(CLASSES[cls[v]] for v in new)),
            "_fixed_ids": fixed, "_new_ids": new}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--details", default="runs/center_recognition_diagnostic_v1"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); cs, rd = cfg["center_supervision"], cfg["recognition_diagnostic"]
    paths = load_paths(); root = paths.repo_root; bs = rd["bootstrap"]
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    hashes = check_inputs(rd, root)
    runs = verify(cfg, cs, paths, root / rd["inputs"]["status"]["path"], root / Path(rd["inputs"]["manifest_index"]["path"]).parent)
    if {k: v["final_sha256"] for k, v in runs.items()} != rd["inputs"]["final_checkpoints"]:
        raise SystemExit("final checkpoints differ from the config")
    prev = json.loads((root / rd["inputs"]["previous_report"]["path"]).read_text())["models"]
    inputs, seqs = validation(cfg, cs, paths)
    f_dense = {}
    for h in HALVES:
        ev = inputs.halves[h]; f = {s.seq_id: np.asarray(inputs.store.get(s.seq_id, "orig").prob) for s in ev.natives}
        f.update({s.recipe.recipe_id: np.asarray(s.prob) for s in (*ev.pairs, *ev.multi)}); f_dense[h] = f
        bad = [q["id"] for q in seqs[h] if f[q["id"]].shape != (len(q["labels"]), 5)]
        if bad:
            raise SystemExit(f"F-dense lengths differ from the sequences, e.g. {bad[:3]}")
    checks = {"e1_b3": reproduce_e1_b3(cfg, rd, root, inputs, f_dense), "f_dtype": str(next(iter(f_dense["selection"].values())).dtype)}
    print("E1 + B3 reproduced from F-dense", flush=True)
    triples = sorted({r for h in HALVES for q in seqs[h] for r in q["refs"]}); at = {t: i for i, t in enumerate(triples)}
    man = load_manifest(paths.manifest_path).set_index("video_id"); frame_paths = FramePaths({v: str(man.loc[v, "frames_dir"]) for v in {t[0] for t in triples}})
    details = root / a.details; details.mkdir(parents=True, exist_ok=True)
    f_grid = {h: {k: to_grid(v) for k, v in f_dense[h].items()} for h in HALVES}
    res = {"inputs": hashes, "checks": checks, "paths": dict(rd["paths"]), "models": {}, "F": {}, "supplementary": {}}
    f_tabs = {}
    for h in HALVES:
        for p, probs in (("F-dense", f_dense[h]), ("F-grid", f_grid[h])):
            s, t = path_rows(cfg, probs, inputs.halves[h]); f_tabs[(p, h)] = t["native"]; res["F"][f"{p}|{h}"] = s
            m, _ = evaluate(cfg, decode(rd["inputs"]["e1_b3_params"], probs), inputs.halves[h])
            res["supplementary"][f"F|{p}|{h}"] = {k: m[k] for k in CHECK_KEYS}
        res["F"][f"grid_effect|{h}"] = compare(f_tabs[("F-grid", h)], f_tabs[("F-dense", h)], bs)
    timing = {}
    for key, run in runs.items():
        t0 = time.time(); model, _ = build_model(cs, paths, Path(run["init"]), device)
        st = torch.load(run["final_checkpoint"], map_location="cpu", weights_only=False)
        with torch.no_grad():
            for n, p in model.encoder.trainable():
                p.copy_(st["encoder_trainable"][n].to(device))
        model.head.load_state_dict(st["head"]); model.eval()
        feats = extract(model, triples, frame_paths, device, a.workers); t1 = time.time()
        out = {"checks": {}, "rows": {}, "comparisons": {}, "centre_errors": {}, "eight_cells": {}}
        preds_store = {}
        for h in HALVES:
            j_dense = frame_head_probs(model, feats, at, seqs[h], device); j_grid = {k: to_grid(v) for k, v in j_dense.items()}
            t_win = {q["id"]: windows_of(model, feats, at, q, STRIDE, device)["prob"] for q in seqs[h]}
            tabs = {"F-dense": f_tabs[("F-dense", h)], "F-grid": f_tabs[("F-grid", h)]}
            for p, probs in (("J-dense", j_dense), ("J-grid", j_grid), ("T-window", t_win)):
                s, t = path_rows(cfg, probs, inputs.halves[h]); out["rows"][f"{p}|{h}"] = s; tabs[p] = t["native"]
                m, _ = evaluate(cfg, decode(rd["inputs"]["e1_b3_params"], probs), inputs.halves[h])
                res["supplementary"][f"{key}|{p}|{h}"] = {k: m[k] for k in CHECK_KEYS}
                preds_store.update({f"{h}|{p}|{k}": v.argmax(1).astype(np.int8) for k, v in probs.items()})
            old = prev[key][f"argmax|{h}"]; diff = max(abs(out["rows"][f"T-window|{h}"][k] - old[k]) for k in CHECK_KEYS)
            if diff > 1e-12:
                raise SystemExit(f"{key} {h}: T-window does not reproduce the previous argmax row (max difference {diff:.2e})")
            out["checks"][f"t_window_reproduction|{h}"] = diff
            for name, (x, y) in {"F-grid - F-dense": ("F-grid", "F-dense"), "J-grid - J-dense": ("J-grid", "J-dense"),
                                 "J-grid - F-grid": ("J-grid", "F-grid"), "J-dense - F-dense": ("J-dense", "F-dense"),
                                 "T-window - J-grid": ("T-window", "J-grid")}.items():
                out["comparisons"][f"{name}|{h}"] = compare(tabs[x], tabs[y], bs)
            df = centre_rows(seqs[h], {"F": f_dense[h], "J": j_dense, "T": t_win})
            out["centre_errors"][h] = centre_errors(df, ["F", "J", "T"])
            out["eight_cells"][h] = eight_cells(native_right(tabs["F-grid"]), native_right(tabs["J-grid"]), native_right(tabs["T-window"]))
        np.savez_compressed(details / f"predictions_{key}.npz", **preds_store)
        timing[key] = {"extract_s": t1 - t0, "total_s": time.time() - t0, "frames": len(triples)}
        res["models"][key] = out
        c = out["rows"]
        print(f"{key}: native F1 sel F-grid {res['F']['F-grid|selection']['native_macro_f1']:.4f} J-grid {c['J-grid|selection']['native_macro_f1']:.4f} "
              f"T {c['T-window|selection']['native_macro_f1']:.4f} | cal J-grid {c['J-grid|calibration']['native_macro_f1']:.4f} "
              f"T {c['T-window|calibration']['native_macro_f1']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        del model, feats; torch.cuda.empty_cache()
    res["timing"] = timing
    res["history_t_window_b3"] = {k: {h: {x: prev[k][f"b3|{h}"][x] for x in ("segment_f1_50_multi", "boundary_recall_multi", "native_macro_f1",
                                                                            "native_internal_cuts_per_clip")} | {"b3": prev[k]["b3_choice"]["status"]}
                                      for h in HALVES} for k in runs}
    (details / "details.json").write_text(json.dumps(res, indent=1, default=str))
    out_dir = root / a.out; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(strip_ids(res), indent=1, default=str)); (out_dir / "report_tables.md").write_text(tables_md(res))
    print(f"details={details}"); print(f"wrote {out_dir}")


def strip_ids(obj):
    if isinstance(obj, dict):
        return {k: strip_ids(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip_ids(v) for v in obj]
    return obj


def tables_md(res: dict) -> str:
    rows = []
    for h in HALVES:
        for p in ("F-dense", "F-grid"):
            r = res["F"][f"{p}|{h}"]
            rows.append({"model": "E1 (frozen)", "path": p, "half": h, "native macro-F1": r["native_macro_f1"], "wrong clips": f"{r['wrong_clips']}/{r['clips']}",
                         "A5C right": f"{r['per_class']['A5C']['right']}/{r['per_class']['A5C']['of']}", "A4C right": f"{r['per_class']['A4C']['right']}/{r['per_class']['A4C']['of']}",
                         "SC4C right": f"{r['per_class']['SC4C']['right']}/{r['per_class']['SC4C']['of']}", "cuts per clip": r["cuts_per_clip"]})
    for key, m in res["models"].items():
        for h in HALVES:
            for p in ("J-dense", "J-grid", "T-window"):
                r = m["rows"][f"{p}|{h}"]
                rows.append({"model": key, "path": p, "half": h, "native macro-F1": r["native_macro_f1"], "wrong clips": f"{r['wrong_clips']}/{r['clips']}",
                             "A5C right": f"{r['per_class']['A5C']['right']}/{r['per_class']['A5C']['of']}", "A4C right": f"{r['per_class']['A4C']['right']}/{r['per_class']['A4C']['of']}",
                             "SC4C right": f"{r['per_class']['SC4C']['right']}/{r['per_class']['SC4C']['of']}", "cuts per clip": r["cuts_per_clip"]})
    comp = []
    for key, m in res["models"].items():
        for name, c in m["comparisons"].items():
            comp.append({"model": key, "comparison": name, "d native macro-F1": f"{c['d_macro_f1']:+.4f} [{c['ci_macro_f1'][0]:+.4f}, {c['ci_macro_f1'][1]:+.4f}]",
                         "fixed / new clips": f"{c['fixed']} / {c['new']}", "sessions fixed / new": f"{c['fixed_sessions']} / {c['new_sessions']}",
                         "new by class": json.dumps(c["new_by_class"]), "fixed by class": json.dumps(c["fixed_by_class"])})
    for h in HALVES:
        c = res["F"][f"grid_effect|{h}"]
        comp.append({"model": "E1 (frozen)", "comparison": f"F-grid - F-dense|{h}", "d native macro-F1": f"{c['d_macro_f1']:+.4f} [{c['ci_macro_f1'][0]:+.4f}, {c['ci_macro_f1'][1]:+.4f}]",
                     "fixed / new clips": f"{c['fixed']} / {c['new']}", "sessions fixed / new": f"{c['fixed_sessions']} / {c['new_sessions']}",
                     "new by class": json.dumps(c["new_by_class"]), "fixed by class": json.dumps(c["fixed_by_class"])})
    cells = [{"model": k, "half": h, **m["eight_cells"][h]} for k, m in res["models"].items() if k.startswith("C_") for h in HALVES]
    cent = [{"model": k, "half": h, "windows": f"{m['centre_errors'][h][g]['windows']}", "group": g,
             **{f"{p} error (rec-weighted)": f"{m['centre_errors'][h][g][p]['error']:.4f} ({m['centre_errors'][h][g][p]['recording_weighted']:.4f})" for p in ("F", "J", "T")}}
            for k, m in res["models"].items() for h in HALVES for g in ("all", "single recording", "mixed views")]
    sup = [{"system": k, **v} for k, v in res["supplementary"].items()]
    return "\n".join(["# Recognition regression localisation: tables\n", "## Native recognition per path (argmax, majority vote)\n", md_table(rows),
                      "## Paired native comparisons (session bootstrap)\n", md_table(comp), "## Eight cells of the C models (F-grid, J-grid, T-window right?)\n", md_table(cells),
                      "## Target-sample error at window centres\n", md_table(cent), "## Supplementary: one archived E1 + B3 setting on every path\n", md_table(sup)])


if __name__ == "__main__":
    main()
