#!/usr/bin/env python
"""Evaluate, freeze and report the fold-output correction round (plan Tasks 3-5; configs/experiments/fold_output_correction_v1.yaml).

  --extract-aux  each frozen fold encoder on both validation halves (orig and gamma090, 10 Hz) into new caches
                 <cache_key>-validation; inputs only, no score. (GPU)
  --evaluate     after the nine runs: identities and pairing; the main evaluation on the frozen B0 v3 and the fixed
                 auxiliary evaluation on each fold encoder; per system, seed and half the metrics, the raw per-sample
                 argmax beside the full system, fixed and new errors with recordings and sessions; T-out - T-in (primary),
                 T-out - (P-out + fixed B3) and T-out - (B0 v3 + fixed B3) with the paired bootstrap; constraints, goal and
                 the fixed decision; the archived reference rows; freezing of the nine models.

Nothing here chooses an epoch, a decoding parameter or an encoder.

  python scripts/report_fold_output_correction.py --config configs/experiments/fold_output_correction_v1.yaml \
      --out docs/results/2026-10-08_fold_output_correction --evaluate
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
from scripts.ev9v_complementarity import md_table  # noqa: E402
import scripts.prepare_fold_output_correction as pf  # noqa: E402
import scripts.train_fold_output_correction as tf  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.dev_metrics import evaluate_predictions  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.metrics_boundary import match_boundaries  # noqa: E402
from echo_routing.temporal.baselines.b3_js_divergence import detect_boundaries, labels_from_boundaries  # noqa: E402
from echo_routing.temporal.dev_data import FeatureStore, Seq, eval_set  # noqa: E402
from echo_routing.temporal.dev_run import HALVES  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

NS = pf.NS
PAIRS = {"A5C read as A4C": (3, 2), "A4C read as A5C": (2, 3)}
FLAT = ("segment_f1_50_multi", "boundary_precision_multi", "boundary_recall_multi", "boundary_f1_multi", "native_macro_f1",
        "native_internal_cuts_per_clip", "same_class_cut_rate_multi", "segment_f1_50_pairs", "boundary_recall_pairs")
COMPARE = {"T-out - T-in": ("T-out", "T-in"), "T-out - (P-out + B3)": ("T-out", "P-out"), "T-out - (B0 + B3)": ("T-out", "baseline")}


# ----------------------------------------------------------------------------------------------- pure helpers
def fixed_b3(prob: np.ndarray, p: dict) -> np.ndarray:
    """The fixed B3: JS peaks at the given half-window, threshold and NMS radius with a persistent label change, then each
    interval's mean-probability argmax."""
    b = detect_boundaries(prob, int(p["half_window"]), float(p["threshold"]), int(p["nms_radius"]), bool(p["require_label_change"]))
    return labels_from_boundaries(prob, b)


def summary(m: dict) -> dict:
    conf = np.asarray(m["native_confusion"]); row = conf.sum(1)
    return {**{k: m[k] for k in FLAT}, "native_recall": {pf.CLASSES[k]: (float(conf[k, k] / row[k]) if row[k] else None) for k in range(5)},
            "native_confusion": conf.tolist(), **{p: int(conf[a, b]) for p, (a, b) in PAIRS.items()}}


def consistency(deltas: list[dict]) -> str:
    """Over seeds: 'consistent positive' when every 95% lower bound is above zero, 'consistent negative' when every upper
    bound is below zero, otherwise 'no consistent difference detected'."""
    if deltas and all(d["ci95"][0] > 0 for d in deltas):
        return "consistent positive"
    if deltas and all(d["ci95"][1] < 0 for d in deltas):
        return "consistent negative"
    return "no consistent difference detected"


def constraints_met(row: dict, half: str, c: dict) -> bool:
    return (row["boundary_recall_multi"] or 0) >= float(c["boundary_recall_min"]) and row["native_macro_f1"] >= float(c["native_macro_f1_min"][half])


def decide(primary: dict, secondary: dict, t_out_rows: list[dict], fo: dict) -> dict:
    """The fixed decision table on the main encoder. primary / secondary[name]: half -> consistency verdict; t_out_rows:
    per seed {half: summary row} of T-out."""
    c, g = fo["constraints"], fo["goal"]
    both = all(primary[h] == "consistent positive" for h in HALVES)
    beats = both and all(secondary[n][h] == "consistent positive" for n in secondary for h in HALVES)
    cons = all(constraints_met(r[h], h, c) for r in t_out_rows for h in HALVES)
    mean = {h: float(np.mean([r[h]["segment_f1_50_multi"] for r in t_out_rows])) for h in HALVES}
    goal = cons and mean["selection"] >= float(g["selection_min"]) and mean["calibration"] >= float(g["calibration_min"])
    rows = fo["decision"]
    case = rows[0] if not both else rows[1] if not beats else rows[2] if not cons else rows[3] if not goal else rows[4]
    return {"primary": primary, "secondary": secondary, "both_halves_positive": both, "beats_simple_controls": beats, "constraints_met": cons,
            "mean_segment_f1_T-out": mean, "goal_met": bool(goal), "goal_margin": {h: mean[h] - float(g[f"{h}_min"]) for h in HALVES},
            "calibration_sensitivity_margin": mean["calibration"] - (float(g["calibration_sensitivity"]) + 0.01), **case}


def change_sets(true_b: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, tol: float) -> tuple[set, set]:
    """True view changes found by a but not b, and by b but not a (one-to-one matching within tol)."""
    ha = {int(true_b[j]) for _, j in match_boundaries(pred_a, true_b, tol)}; hb = {int(true_b[j]) for _, j in match_boundaries(pred_b, true_b, tol)}
    return ha - hb, hb - ha


def boundary_changes(ev, preds_a: dict, preds_b: dict, tol: float) -> dict:
    """Reference view changes of multi streams found by a only (fixed) and by b only (lost), with the recordings on both
    sides of each change and their sessions."""
    out = {k: {"changes": 0, "recordings": set(), "sessions": set()} for k in ("found by a only", "found by b only")}
    for s in ev.multi:
        sid = s.recipe.recipe_id; starts = np.r_[0, s.joins]; vids = [f.video_id for f in s.recipe.fragments]
        a_b = np.flatnonzero(np.diff(np.asarray(preds_a[sid])) != 0) + 1; b_b = np.flatnonzero(np.diff(np.asarray(preds_b[sid])) != 0) + 1
        fa, fb = change_sets(np.asarray(s.semantic_boundaries, int), a_b, b_b, tol)
        for name, ch in (("found by a only", fa), ("found by b only", fb)):
            for r in ch:
                k = int(np.searchsorted(starts, r)); src = {vids[k - 1], vids[k]}
                out[name]["changes"] += 1; out[name]["recordings"] |= src; out[name]["sessions"] |= {session_of(v) for v in src}
    return {k: {"changes": v["changes"], "recordings": len(v["recordings"]), "sessions": len(v["sessions"])} for k, v in out.items()}


def native_changes(ta, tb) -> dict:
    """Native clips fixed by a (b wrong, a right) and newly wrong in a, per error pair and over all clips, with sessions."""
    j = ta.merge(tb, on=["video_id", "label"], suffixes=("_a", "_b")); out = {}
    for name, (y, x) in PAIRS.items():
        fixed = j[(j["label"] == y) & (j["majority_b"] == x) & (j["majority_a"] == y)]["video_id"].tolist()
        new = j[(j["label"] == y) & (j["majority_a"] == x) & (j["majority_b"] == y)]["video_id"].tolist()
        out[name] = {"fixed": len(fixed), "fixed_sessions": len({session_of(v) for v in fixed}), "new": len(new), "new_sessions": len({session_of(v) for v in new})}
    fx = j[(j["majority_b"] != j["label"]) & (j["majority_a"] == j["label"])]["video_id"]; nw = j[(j["majority_a"] != j["label"]) & (j["majority_b"] == j["label"])]["video_id"]
    out["all clips"] = {"fixed": int(fx.size), "fixed_sessions": len({session_of(v) for v in fx}), "new": int(nw.size), "new_sessions": len({session_of(v) for v in nw})}
    return out


def segment_changes(ev, preds_a: dict, preds_b: dict) -> dict:
    """Reference segments of multi streams matched (IoU 0.5, one-to-one) by a only and by b only, with recordings and sessions."""
    from scripts.report_fan224_visual import matched_true
    out = {k: {"segments": 0, "recordings": set(), "sessions": set()} for k in ("matched by a only", "matched by b only")}
    for s in ev.multi:
        sid = s.recipe.recipe_id; true = labels_to_segments(s.labels); vids = np.repeat([f.video_id for f in s.recipe.fragments], [f.end - f.start for f in s.recipe.fragments])
        ma = matched_true(labels_to_segments(np.asarray(preds_a[sid])), true); mb = matched_true(labels_to_segments(np.asarray(preds_b[sid])), true)
        for name, js in (("matched by a only", ma - mb), ("matched by b only", mb - ma)):
            for jx in js:
                src = set(vids[true[jx].start:true[jx].end].tolist())
                out[name]["segments"] += 1; out[name]["recordings"] |= src; out[name]["sessions"] |= {session_of(v) for v in src}
    return {k: {"segments": v["segments"], "recordings": len(v["recordings"]), "sessions": len(v["sessions"])} for k, v in out.items()}


def visual_vs_system(ev, head_preds: dict, sys_preds: dict) -> dict:
    """Native clips and multi-stream reference segments that the frozen visual head (per-sample argmax) gets right while
    the system gets them wrong, and the reverse; with recordings and sessions."""
    def nat_right(p, y):
        return int(np.bincount(np.asarray(p, int), minlength=5).argmax()) == int(y)
    out = {}
    hr_sw = [s.seq_id for s in ev.natives if nat_right(head_preds[s.seq_id], s.labels[0]) and not nat_right(sys_preds[s.seq_id], s.labels[0])]
    hw_sr = [s.seq_id for s in ev.natives if not nat_right(head_preds[s.seq_id], s.labels[0]) and nat_right(sys_preds[s.seq_id], s.labels[0])]
    out["native: head right, system wrong"] = {"clips": len(hr_sw), "sessions": len({session_of(v) for v in hr_sw})}
    out["native: head wrong, system right"] = {"clips": len(hw_sr), "sessions": len({session_of(v) for v in hw_sr})}
    seg = segment_changes(ev, head_preds, sys_preds)
    out["segments: head matched, system not"] = seg["matched by a only"]; out["segments: system matched, head not"] = seg["matched by b only"]
    return out


# ----------------------------------------------------------------------------------------------- inputs and systems
def prob_sequences(ev) -> list[Seq]:
    """Every validation sequence with its encoder probabilities as the temporal model's input."""
    out = [Seq(s.seq_id, True, np.asarray(s.prob, np.float32), s.labels, s.joins) for s in ev.natives]
    return out + [Seq(st.recipe.recipe_id, False, np.asarray(st.prob, np.float32), st.labels, st.joins) for st in (*ev.pairs, *ev.multi)]


def halves_of(proto: dict, labels: dict, cache_root: Path, key: str) -> dict:
    store = FeatureStore(cache_root, key)
    return {h: eval_set(proto, h, labels, store) for h in HALVES}


def run_keys(fo: dict) -> list[tuple[str, int]]:
    return [(arm, int(s)) for s in fo["training"]["seeds"] for arm in pf.ARMS]


def expected_identity(cfg: dict, fo: dict, root: Path, keys: dict, arm: str, seed: int) -> str:
    plan = root / "runs" / NS / "plans" / f"plan_s{seed}.json"; init = root / "runs" / NS / "inits" / ("p_out_identity.pt" if arm == "P-out" else f"asrf_init_s{seed}.pt")
    input_keys = {str(k): pf.arm_cache_key(arm, k, keys) for k in range(pf.N_FOLDS)}
    ctx = pf.load_context(cfg, fo, root); paths = load_paths()
    content = {k: pf.cache_digest(paths.cache_root, key, [v for v, _ in ctx.clips if ctx.fold_of_clip[v] == int(k)]) for k, key in input_keys.items()}
    return tf.identity(cfg, arm, seed, input_keys, content, sha256_file(plan), sha256_file(init), root)


def verify(cfg: dict, fo: dict, root: Path) -> dict:
    keys = {int(k): m["cache_key"] for k, m in fo["inputs"]["fold_models"].items()}; runs = {}
    for arm, seed in run_keys(fo):
        key = f"{arm}_s{seed}"; man_path = root / "runs" / NS / "train" / key / "run_manifest.json"
        if tf.finished_run(man_path, expected_identity(cfg, fo, root, keys, arm, seed)) is not True:
            raise SystemExit(f"{key}: not a finished run of the current identity")
        runs[key] = json.loads(man_path.read_text())
        if runs[key]["epochs"] != int(fo["training"]["epochs"]) or runs[key]["updates"] != int(fo["training"]["total_updates"]):
            raise SystemExit(f"{key}: {runs[key]['epochs']} epochs / {runs[key]['updates']} updates")
    for seed in [int(s) for s in fo["training"]["seeds"]]:
        a, b, p = runs[f"T-in_s{seed}"], runs[f"T-out_s{seed}"], runs[f"P-out_s{seed}"]
        if not (a["plan_sha256"] == b["plan_sha256"] == p["plan_sha256"] and a["init_sha256"] == b["init_sha256"] and a["class_weight"] == b["class_weight"] == p["class_weight"]
                and a["pos_weight"] == b["pos_weight"] and b["input_keys"] == p["input_keys"] and a["input_keys"] != b["input_keys"]):
            raise SystemExit(f"seed {seed}: the three arms are not paired")
    return runs


def load_models(cfg: dict, fo: dict, runs: dict, paths, device) -> dict:
    out = {}
    for key, man in runs.items():
        plan = json.loads(Path(man["plan"]).read_text()); head = tf.make_head(fo, paths.repo_root, man["arm"], plan, device)
        model = head.build().to(device); model.load_state_dict(torch.load(man["final_checkpoint"], map_location=device, weights_only=True)); model.eval()
        out[key] = (man["arm"], head, model)
    return out


def system_outputs(arm: str, head, model, seqs: list[Seq], b3: dict, device) -> tuple[dict, dict]:
    """(full-system labels, raw per-sample argmax) of one model on one half's sequences."""
    if arm == "P-out":
        q = {s.seq_id: head.probs(model, s.feat, device) for s in seqs}
        return {k: fixed_b3(v, b3) for k, v in q.items()}, {k: v.argmax(1) for k, v in q.items()}
    dec = head.decode(model, seqs, device)
    return {k: np.asarray(v["labels"], int) for k, v in dec.items()}, {k: np.asarray(v["frame_labels"], int) for k, v in dec.items()}


def evaluate_encoder(cfg: dict, fo: dict, models: dict, halves: dict, device) -> dict:
    """Every system on one encoder's two halves: full and raw rows, tables and predictions."""
    hz, tol_s = float(cfg["protocol"]["hz"]), float(cfg["selection"]["boundary_tolerance_s"]); b3 = fo["evaluation"]["fixed_b3"]; out = {}
    for h in HALVES:
        ev = halves[h]; seqs = prob_sequences(ev); probs = {s.seq_id: s.feat for s in seqs}
        systems = {"baseline": ({k: fixed_b3(p, b3) for k, p in probs.items()}, {k: p.argmax(1) for k, p in probs.items()})}
        for key, (arm, head, model) in models.items():
            systems[key] = system_outputs(arm, head, model, seqs, b3, device)
        for name, (full, raw) in systems.items():
            mf, tf_ = evaluate_predictions(full, ev, hz, tol_s); mr, _ = evaluate_predictions(raw, ev, hz, tol_s)
            out.setdefault(name, {})[h] = {"row": summary(mf), "raw_argmax": summary(mr), "tables": tf_, "preds": full}
        out.setdefault("_head", {})[h] = {"preds": {k: p.argmax(1) for k, p in probs.items()}}
    return out


def comparisons(fo: dict, res: dict, halves: dict, tol: float) -> dict:
    from scripts.report_model_development import paired_bootstrap
    out = {}
    for seed in [int(s) for s in fo["training"]["seeds"]]:
        for name, (a, b) in COMPARE.items():
            ka = f"{a}_s{seed}"; kb = "baseline" if b == "baseline" else f"{b}_s{seed}"
            for h in HALVES:
                A, B = res[ka][h], res[kb][h]
                out[f"s{seed}|{name}|{h}"] = {"bootstrap": paired_bootstrap(A["tables"], B["tables"]), "native_changes": native_changes(A["tables"]["native"], B["tables"]["native"]),
                                              "segment_changes": segment_changes(halves[h], A["preds"], B["preds"]),
                                              "boundary_changes": boundary_changes(halves[h], A["preds"], B["preds"], tol)}
    return out


def verdicts(fo: dict, comp: dict) -> dict:
    seeds = [int(s) for s in fo["training"]["seeds"]]
    return {name: {h: consistency([comp[f"s{s}|{name}|{h}"]["bootstrap"]["multi segment F1@50"] for s in seeds]) for h in HALVES} for name in COMPARE}


def strip(res: dict) -> dict:
    """Drop tables and predictions (they carry ids); keep rows and counts."""
    return {name: {h: {k: v for k, v in r.items() if k in ("row", "raw_argmax")} for h, r in per.items()} for name, per in res.items() if not name.startswith("_")}


# ----------------------------------------------------------------------------------------------- steps
def run_extract_aux(cfg: dict, fo: dict, paths) -> None:
    import pandas as pd

    from echo_routing.features.encoder import load_checkpoint
    from echo_routing.features.extract import extract_split
    root = paths.repo_root; pf.check_inputs(fo, root); proto = json.loads((root / fo["inputs"]["protocol"]["path"]).read_text())
    clips = sorted({c for h in HALVES for c in proto["validation"][h]["clips"]}); man = pd.read_csv(root / fo["inputs"]["manifest"]["path"])
    rows = man[man.video_id.isin(clips)]
    if len(rows) != len(clips) or set(rows.split) != {"validation"}:
        raise SystemExit("the validation clips are not all in the manifest's validation split")
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    for k, m in fo["inputs"]["fold_models"].items():
        model, _ = load_checkpoint(root / m["checkpoint"]); key = f"{m['cache_key']}-validation"
        for variant in pf.VARIANTS:
            extract_split(model, device, rows, "family5_index", paths.cache_root, key, variant, cfg["sampling"]["target_hz"], cfg["preprocess"]["image_size"],
                          cfg["features"]["batch_size"], cfg["features"]["num_workers"], meta={"ckpt": m["checkpoint"], "task": cfg["task"], "classes": list(pf.CLASSES), "use": "auxiliary evaluation"})
        print(json.dumps({"fold model": int(k), "cache": key, "clips": len(clips)}), flush=True)


def freeze(paths, runs: dict, config: str) -> dict:
    out = {}
    for key, man in runs.items():
        dest = paths.repo_root / "models_frozen" / NS / key
        if not dest.exists():
            files = [man["final_checkpoint"], str(paths.repo_root / "runs" / NS / "train" / key / "run_manifest.json"), man["plan"], man["init"], str(paths.repo_root / config)]
            subprocess.run([sys.executable, "scripts/freeze_model.py", NS, key, *[x for f in files for x in ("--file", f)], "--meta", f"identity={man['identity']}",
                            "--meta", f"arm={man['arm']}", "--meta", f"seed={man['seed']}", "--notes", "Fold-output correction round (2026-10-08); final epoch."],
                           cwd=paths.repo_root, check=True, capture_output=True)
        out[key] = str(dest)
    return out


def run_evaluate(a, cfg: dict, fo: dict, paths, out_dir: Path) -> None:
    from scripts.report_center_supervision import references, validation
    root = paths.repo_root; t0 = time.time(); runs = verify(cfg, fo, root); print("nine runs verified and paired", flush=True)
    device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    proto = json.loads((root / fo["inputs"]["protocol"]["path"]).read_text()); import pandas as pd
    man = pd.read_csv(root / fo["inputs"]["manifest"]["path"]).set_index("video_id")
    labels = {c: int(man.loc[c, "family5_index"]) for h in HALVES for c in proto["validation"][h]["clips"]}
    models = load_models(cfg, fo, runs, paths, device); tol = float(cfg["selection"]["boundary_tolerance_s"]) * float(cfg["protocol"]["hz"])
    res = {"main": {}, "auxiliary": {}, "runs": {k: {x: v[x] for x in ("identity", "arm", "seed", "epochs", "updates", "final_sha256", "plan_sha256", "init_sha256",
                                                                       "input_keys", "class_weight", "pos_weight")} for k, v in runs.items()}}
    encoders = {"B0 v3": fo["inputs"]["b0_v3"]["cache_key"], **{f"fold {k}": f"{m['cache_key']}-validation" for k, m in fo["inputs"]["fold_models"].items()}}
    for enc, key in encoders.items():
        halves = halves_of(proto, labels, paths.cache_root, key); out = evaluate_encoder(cfg, fo, models, halves, device); comp = comparisons(fo, out, halves, tol)
        block = {"systems": strip(out), "comparisons": comp, "verdicts": verdicts(fo, comp),
                 "visual_vs_system": {f"{k}|{h}": visual_vs_system(halves[h], out["_head"][h]["preds"], out[k][h]["preds"]) for k in out if not k.startswith("_") for h in HALVES}}
        if enc == "B0 v3":
            res["main"] = block
            t_out = [{h: out[f"T-out_s{s}"][h]["row"] for h in HALVES} for s in fo["training"]["seeds"]]
            res["decision"] = decide(block["verdicts"]["T-out - T-in"], {n: block["verdicts"][n] for n in ("T-out - (P-out + B3)", "T-out - (B0 + B3)")}, t_out, fo)
        else:
            res["auxiliary"][enc] = block
        print(f"{enc}: done", flush=True)
    try:
        inp_e1, seqs = validation(cfg, {"inputs": {"protocol": fo["inputs"]["protocol"], "feature_manifest": {"path": "models_frozen/e1_echoviewclip_probe/v1/feature_manifest.json"}}}, paths)
        refs = references({**cfg, "center_supervision": {"inputs": {"protocol": fo["inputs"]["protocol"]}}}, paths, inp_e1, seqs)
        res["references"] = {k: {h: v[h]["metrics"] for h in HALVES} for k, v in refs.items()}
    except SystemExit as e:                                 # reported, never replaced by another row
        res["references"] = {"not reproduced": str(e)}
    res["frozen"] = {} if a.no_freeze else freeze(paths, runs, a.config); res["seconds"] = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True); (out_dir / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    (out_dir / "report_tables.md").write_text(tables_md(res)); print(json.dumps(res["decision"], indent=1, default=str))


def tables_md(res: dict) -> str:
    rows = []
    for part, block in (("main (B0 v3)", res["main"]), *((f"auxiliary ({k})", v) for k, v in res["auxiliary"].items())):
        for name, per in block["systems"].items():
            for h in HALVES:
                r, w = per[h]["row"], per[h]["raw_argmax"]
                rows.append({"encoder": part, "system": name, "half": h, "segment F1@50": r["segment_f1_50_multi"], "boundary P": r["boundary_precision_multi"],
                             "boundary R": r["boundary_recall_multi"], "native macro-F1": r["native_macro_f1"], "cuts per clip": r["native_internal_cuts_per_clip"],
                             "same-class cut rate": r["same_class_cut_rate_multi"], **{p: r[p] for p in PAIRS}, "raw argmax segment F1@50": w["segment_f1_50_multi"]})
    comp = []
    for part, block in (("main", res["main"]), *((k, v) for k, v in res["auxiliary"].items())):
        for k, c in block["comparisons"].items():
            b = c["bootstrap"]["multi segment F1@50"]
            comp.append({"encoder": part, "comparison": k, "d segment F1@50": f"{b['delta']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]",
                         "native fixed/new": f"{c['native_changes']['all clips']['fixed']}/{c['native_changes']['all clips']['new']}"})
    return "\n".join(["# Fold-output correction: tables\n", "## Systems\n", md_table(rows, list(dict.fromkeys(x for r in rows for x in r))), "## Comparisons\n",
                      md_table(comp), "## Decision (main encoder)\n", "```\n" + json.dumps(res["decision"], indent=1, default=str) + "\n```\n"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True); ap.add_argument("--no-freeze", action="store_true")
    step = ap.add_mutually_exclusive_group(required=True); step.add_argument("--extract-aux", action="store_true"); step.add_argument("--evaluate", action="store_true")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); fo = cfg["fold_output_correction"]; pf.check_out(a.out, fo); paths = load_paths()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if a.extract_aux:
        run_extract_aux(cfg, fo, paths)
    else:
        run_evaluate(a, cfg, fo, paths, paths.repo_root / a.out)


if __name__ == "__main__":
    main()
