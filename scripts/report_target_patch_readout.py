#!/usr/bin/env python
"""Evaluate, freeze and report the target-anchored readout round (docs/superpowers/plans/2026-10-11-target-patch-readout.md,
Tasks 3-4; configs/experiments/target_patch_readout_v1.yaml). `--evaluate` runs once, after the nine trainings.

1. Runs: identities recomputed, 3,750 updates and final weights checked, interruptions read, the nine final weights frozen.
   Only then are development frames loaded.
2. Windows: the source round's (centres every 5 samples, 16 frames, the marker at the middle target slot, Ordered order).
   Per-window logits and per-sample probabilities of every model are saved on the server.
3. References: the frozen Ordered models from their saved per-window outputs with their stored B3 settings (their stored
   rows must be reproduced exactly); E0/E1/B7 + B3, the historical E1 + ASRF, Original FACT and the paired ASRF; G1 + ASRF
   from stored rows.
4. Mixed windows (primary: exactly 8 of 16 frames of the requested sample's reference view) with session-resampled and
   stream-resampled paired bootstraps; native recognition; direct argmax, B3 chosen on calibration, and the fixed Ordered
   B3 setting as a sensitivity; system comparisons, verdicts, goal, the fixed decision, units and costs.
Results go to the results directory through id guards after every stage.

  python scripts/report_target_patch_readout.py --config configs/experiments/target_patch_readout_v1.yaml --evaluate
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.raw_video_context_tables as rt  # noqa: E402
import scripts.report_center_supervision as rcs  # noqa: E402
import scripts.report_fact_asrf as ra  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.target_patch_readout_tables as pt  # noqa: E402
import scripts.train_raw_video_context as rvc  # noqa: E402
import scripts.train_target_patch_readout as tr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import decode  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.report_fold_output_correction import FLAT, boundary_changes, native_changes, segment_changes, summary, write_atomic  # noqa: E402
from scripts.report_model_development import paired_bootstrap  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.vit.target_patch_readout as tp  # noqa: E402
from echo_routing.config import gpu_memory_fraction  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402

NS = "target_patch_readout_v1"
HALVES = pt.HALVES
NAME = {"cls": "Continued CLS", "global_patches": "Global patches", "target_patches": "Target patches", "ordered": "Ordered"}
PAIRS = {"Target - Global": ("target_patches", "global_patches"), "Target - CLS": ("target_patches", "cls"), "Global - CLS": ("global_patches", "cls"),
         "Target - Ordered": ("target_patches", "ordered"), "CLS - Ordered": ("cls", "ordered"), "Global - Ordered": ("global_patches", "ordered")}
REFS = {"Target - (E1 + B3)": "E1 + B3", "Target - (E0 + B3)": "E0 + B3", "Target - (B7 + B3)": "B7 + B3", "Target - Original FACT": "Original FACT s{k}",
        "Target - paired ASRF": "paired ASRF s{k}", "Target - historical E1 + ASRF": "E1 + ASRF s{k}"}


# ----------------------------------------------------------------------------------------------- runs
def verify_runs(ctx) -> dict:
    runs = {}
    for key in ctx.tp["queue"]:
        r, s = key.rsplit("_s", 1)[0], int(key.rsplit("_s", 1)[1]); man_path = ctx.out / "train" / key / "run_manifest.json"
        if not man_path.is_file():
            raise SystemExit(f"{key}: no finished run")
        man = json.loads(man_path.read_text()); ident = tr.identity(ctx, r, s, ctx.manifest(s)[1], ctx.tp["source"]["weights"][s]["sha256"])
        if man["identity"] != ident or int(man["updates"]) != ctx.total or sha256_file(Path(man["final_checkpoint"])) != man["final_sha256"]:
            raise SystemExit(f"{key}: identity, update count or final weights do not verify")
        runs[key] = {"readout": r, "seed": s, "manifest": man}
    return runs


def freeze(ctx, runs: dict) -> dict:
    out = {}
    for key, r in runs.items():
        man, dest = r["manifest"], ctx.root / "models_frozen" / NS / key
        if not dest.exists():
            files = [man["final_checkpoint"], str(ctx.out / "train" / key / "run_manifest.json"), str(ctx.root / ctx.config)]
            subprocess.run([sys.executable, "scripts/freeze_model.py", NS, key, *[x for f in files for x in ("--file", f)], "--meta", f"identity={man['identity']}",
                            "--meta", f"readout={r['readout']}", "--meta", f"seed={r['seed']}", "--meta", f"source_weights_sha256={man['source_weights_sha256']}",
                            "--notes", "Target-anchored readout round (2026-10-11); final weights after 3,750 updates from the seed's Ordered model."],
                           cwd=ctx.root, check=True, capture_output=True)
        if sha256_file(dest / "final.pt") != man["final_sha256"]:
            raise SystemExit(f"{key}: the frozen copy differs from the final weights")
        out[key] = str(dest)
    return out


def load_model(ctx, run: dict, device) -> tp.ReadoutMViT:
    state = torch.load(run["manifest"]["final_checkpoint"], map_location="cpu", weights_only=True)
    return tr.make_model(state, run["readout"], device).eval()


# ----------------------------------------------------------------------------------------------- systems
def fixed_rows(cfg: dict, probs: dict, inputs, threshold: float, fx: dict) -> dict:
    params = {"half_window": int(fx["half_window"]), "threshold": float(threshold), "nms_radius": int(fx["nms_radius"])}; out = {}
    for h in HALVES:
        preds = decode(params, probs[h]); m, t = rcs.evaluate(cfg, preds, inputs.halves[h]); out[h] = {"row": summary(m), "tables": t, "preds": preds}
    return out


def ordered_reference(ctx, cfg: dict, W: dict, seqs: dict, inputs) -> tuple[dict, dict]:
    """The frozen Ordered models from the source round's saved per-window outputs, decoded with their stored B3 settings."""
    src = ctx.tp["source"]; stored = json.loads((ctx.root / src["report"]["summary"]["path"]).read_text())["systems"]; systems, logits = {}, {}
    for s in ctx.seeds:
        probs, key = {}, f"ordered_s{s}"; logits[key] = {}
        for h in HALVES:
            pin = src["ordered_outputs"][h][s]
            if sha256_file(ctx.root / pin["path"]) != pin["sha256"]:
                raise SystemExit(f"{key} {h}: saved outputs differ from the pin")
            with np.load(ctx.root / pin["path"]) as z:
                if z["sequence_ids"].tolist() != [q["id"] for q in seqs[h]] or not (np.array_equal(z["centre"], W[h]["centre"]) and np.array_equal(z["marker"], W[h]["marker"])):
                    raise SystemExit(f"{key} {h}: the saved windows differ from this round's windows")
                lg = z["logits"].astype(np.float32)
            logits[key][h] = lg; probs[h] = rr.per_sample(W[h], lg, seqs[h])
        params = stored[key]["b3_choice"]["params"]; systems[key] = {"argmax": {}, "b3": {}, "b3_choice": stored[key]["b3_choice"]}
        for h in HALVES:
            for row, preds in (("argmax", {k: v.argmax(1) for k, v in probs[h].items()}), ("b3", decode(params, probs[h]))):
                m, t = rcs.evaluate(cfg, preds, inputs.halves[h]); r = summary(m); want = stored[key][row][h]
                diff = {f: (r[f], want[f]) for f in FLAT if r[f] != want[f]}
                if diff:
                    raise SystemExit(f"{key} {row} {h}: the stored rows are not reproduced ({diff})")
                systems[key][row][h] = {"row": r, "tables": t, "preds": preds}
    return systems, logits


# ----------------------------------------------------------------------------------------------- windows
def window_meta(W: dict, seqs_h: list[dict]) -> dict:
    """Per window: kind, mixed flag, the requested sample's recording and session, its reference run id and its stream."""
    kind = np.array([seqs_h[q]["kind"] for q in W["seq"]]); mixed = (kind != "native") & np.array([len(set(x.tolist())) >= 2 for x in W["labels16"]])
    rec = np.array([seqs_h[q]["refs"][c][0] for q, c in zip(W["seq"], W["centre"])]); ses = np.array([session_of(v) for v in rec])
    run = []
    for q, c in zip(W["seq"], W["centre"]):
        lab = np.asarray(seqs_h[q]["labels"]); start = c
        while start > 0 and lab[start - 1] == lab[c]:
            start -= 1
        run.append(f"{q}:{start}")
    return {"kind": kind, "mixed": mixed, "recording": rec, "session": ses, "run": np.array(run), "stream": W["seq"].astype(str)}


def mixed_analysis(ctx, logits: dict, W: dict, meta: dict) -> dict:
    """Accuracy of every system on the mixed subsets, support counts and the paired comparisons per seed."""
    subs = pt.mixed_subsets(W["labels16"], W["label"], meta["mixed"]); y = W["label"]
    correct = {k: v.argmax(1) == y for k, v in logits.items()}; ends = W["marker"] != 8
    acc = {k: {n: (float(c[rows].mean()) if rows.size else None) for n, rows in subs.items()} | {"marker away from slot 8": float(c[ends].mean())}
           for k, c in correct.items()}
    support = {n: pt.subset_support(rows, meta["recording"], meta["session"], meta["run"], y) for n, rows in subs.items()}
    comp = {}
    for s in ctx.seeds:
        for name, (a, b) in PAIRS.items():
            ka, kb = f"{a}_s{s}", f"{b}_s{s}"
            for n, rows in subs.items():
                ca, cb = correct[ka][rows], correct[kb][rows]
                comp[f"s{s}|{name}|{n}"] = {"session": pt.group_bootstrap(ca, cb, meta["session"][rows]), "stream": pt.group_bootstrap(ca, cb, meta["stream"][rows]),
                                            "changes": rt.changes(ca, cb, meta["recording"][rows], meta["session"][rows])}
    return {"accuracy": acc, "support": support, "comparisons": comp}


def native_analysis(ctx, logits: dict, W: dict, meta: dict) -> tuple[dict, dict]:
    rows = np.flatnonzero(meta["kind"] == "native"); y = W["label"][rows]; ses = meta["session"][rows]
    sysd = {k: {"pred": v[rows].argmax(1), "margin": v[rows][:, pt.A5C] - v[rows][:, pt.A4C]} for k, v in logits.items()}
    metrics = {k: rt.recognition(v["pred"], v["margin"], y) | {"A4C read as A5C (windows)": int(((y == pt.A4C) & (v["pred"] == pt.A5C)).sum()),
                                                               "A5C read as A4C (windows)": int(((y == pt.A5C) & (v["pred"] == pt.A4C)).sum())} for k, v in sysd.items()}
    comp = {f"s{s}|{name}": rt.window_bootstrap(sysd[f"{a}_s{s}"], sysd[f"{b}_s{s}"], y, ses) for s in ctx.seeds for name, (a, b) in PAIRS.items()}
    return metrics, comp


# ----------------------------------------------------------------------------------------------- comparisons
def system_comparisons(ctx, systems: dict, inputs, tol: float) -> dict:
    out = {}
    for s in ctx.seeds:
        pairs = {n: (f"{a}_s{s}", f"{b}_s{s}") for n, (a, b) in PAIRS.items()} | {n: (f"target_patches_s{s}", r.format(k=s)) for n, r in REFS.items()}
        for name, (ka, kb) in pairs.items():
            for row in (("argmax", "b3", "fixed") if name in PAIRS else ("b3",)):
                if row == "fixed" and "fixed" not in systems[kb]:
                    continue
                for h in HALVES:
                    A, B = systems[ka][row][h], rr.pick(systems, kb, row)[h]
                    c = {"bootstrap": paired_bootstrap(A["tables"], B["tables"]), "native_changes": native_changes(A["tables"]["native"], B["tables"]["native"])}
                    if B.get("preds") is not None:
                        c["segment_changes"] = segment_changes(inputs.halves[h], A["preds"], B["preds"]); c["boundary_changes"] = boundary_changes(inputs.halves[h], A["preds"], B["preds"], tol)
                    out[f"s{s}|{row}|{name}|{h}"] = c
    return out


def verdicts(ctx, mixed_comp: dict, native_comp: dict, comp: dict) -> dict:
    """Consistency over seeds per half: mixed subsets (session bootstrap with the stream sensitivity), native recognition and systems."""
    seeds = ctx.seeds; subsets = ("8 of 16", "all mixed", "<= 7 of 16", "9-11 of 16", "12-15 of 16", "A4C/A5C pair windows")
    out = {"mixed": {name: {n: {h: pt.verdict([mixed_comp[f"s{s}|{name}|{n}|{h}"]["session"] for s in seeds],
                                                [mixed_comp[f"s{s}|{name}|{n}|{h}"]["stream"] for s in seeds]) for h in HALVES} for n in subsets} for name in PAIRS},
           "native": {name: {m: {h: rt.consistency([native_comp[f"s{s}|{name}|{h}"][m] for s in seeds]) for h in HALVES} for m in rt.REC_METRICS} for name in PAIRS},
           "system": {}}
    for row in ("argmax", "b3", "fixed"):
        names = list(PAIRS) + (list(REFS) if row == "b3" else [])
        out["system"][row] = {n: {h: rt.consistency([comp[f"s{s}|{row}|{n}|{h}"]["bootstrap"]["multi segment F1@50"] for s in seeds]) for h in HALVES}
                              for n in names if all(f"s{s}|{row}|{n}|{h}" in comp for s in seeds for h in HALVES)}
    return out


def strip(systems: dict) -> dict:
    out = {}
    for k, s in systems.items():
        out[k] = {r: {h: x["row"] for h, x in s[r].items()} for r in ("argmax", "b3", "fixed", "row") if r in s}
        if "b3_choice" in s:
            out[k]["b3_choice"] = s["b3_choice"]
    return out


def write(res: dict, out_dir: Path, ids: set) -> None:
    flat = json.loads(json.dumps(res, default=str)); assert_no_ids(flat, ids, "summary"); tables = pt.tables_md(flat)
    if any(s in tables for s in ids):
        raise SystemExit("an id appears in the report tables")
    write_atomic(out_dir / "summary.json", json.dumps(flat, indent=1)); write_atomic(out_dir / "report_tables.md", tables)


def plot_curves(runs: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2)); col = {"cls": "#8172b2", "global_patches": "#dd8452", "target_patches": "#4c72b0"}
    for key, r in runs.items():
        ep = r["manifest"]["epochs"]; x = [e["updates"] for e in ep]; kw = {"color": col[r["readout"]], "ls": ("-", "--", ":")[r["seed"]], "lw": 1.2, "label": key}
        ax[0].plot(x, [e["loss"] for e in ep], **kw); ax[1].plot(x, [e["train_accuracy"] for e in ep], **kw)
    for a, t in zip(ax, ("training loss (mean per epoch)", "training-batch accuracy (fit, not generalisation)")):
        a.set_title(t, fontsize=10); a.set_xlabel("updates"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7, ncol=3); fig.suptitle("Target-anchored readout round: final weights kept, nothing chosen from these curves", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def run_evaluate(ctx, workers: int) -> None:
    t0 = time.time(); device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction()); rvc.determinism()
    tpc, cfg, paths = ctx.tp, ctx.cfg, ctx.paths; out_dir = ctx.root / tpc["outputs"]["results_dir"]; out_dir.mkdir(parents=True, exist_ok=True)
    runs = verify_runs(ctx); frozen = freeze(ctx, runs); print("nine runs verified and frozen", flush=True)
    queue = json.loads((ctx.out / "queue_status.json").read_text()); train_ids = set(tr.allowlist(ctx.rv, ctx.root)); ids = train_ids | {session_of(v) for v in train_ids}
    res = {"runs": {k: {x: r["manifest"][x] for x in ("identity", "readout", "seed", "updates", "final_sha256", "source_weights_sha256", "interrupted", "parameters")}
                    | {"epochs": r["manifest"]["epochs"]} for k, r in runs.items()}, "frozen": frozen, "decision": "pending"}
    write(res, out_dir, ids); plot_curves(runs, out_dir / "training_curves.png")
    pin = ctx.rv["evaluation"]["reference_inputs"]["e1_feature_manifest"]
    if sha256_file(ctx.root / pin["path"]) != pin["sha256"]:
        raise SystemExit("the E1 feature manifest differs from its pin")
    inputs, seqs = rcs.validation(cfg, {"inputs": {"protocol": ctx.rv["source"]["protocol"], "feature_manifest": pin}}, paths)   # development data from here on
    dev_rec = {r[0] for h in HALVES for q in seqs[h] for r in q["refs"]}; ids |= dev_rec | {session_of(v) for v in dev_rec} | {q["id"] for h in HALVES for q in seqs[h]}
    refs = sorted({r for h in HALVES for q in seqs[h] for r in q["refs"]}); at = {r: i for i, r in enumerate(refs)}
    fp = tr.FramePaths(tr.frame_dirs(paths, ctx.rv, {r[0] for r in refs}))
    t1 = time.time(); frames = rr.decode_frames(refs, fp, workers); t_dec = time.time() - t1; print(f"{len(refs)} frames decoded ({t_dec:.0f} s)", flush=True)
    W = {h: rr.half_windows(seqs[h], at, 5) for h in HALVES}; meta = {h: window_meta(W[h], seqs[h]) for h in HALVES}
    systems, ordered_logits = ordered_reference(ctx, cfg, W, seqs, inputs); logits = {h: {k: v[h] for k, v in ordered_logits.items()} for h in HALVES}
    print("the frozen Ordered rows reproduce", flush=True); cost, fx = {}, tpc["evaluation"]["fixed_ordered_setting"]
    for s in ctx.seeds:
        systems[f"ordered_s{s}"]["fixed"] = {h: systems[f"ordered_s{s}"]["b3"][h] for h in HALVES}
    for key, r in runs.items():
        model = None; probs = {}; secs = 0.0
        for h in HALVES:
            orders = np.tile(np.arange(16), (len(W[h]["seq"]), 1)); path = ctx.out / "evaluation" / h / f"{key}.npz"
            lg, sec = rr.cached_logits(path, r["manifest"]["identity"], W[h], orders, seqs[h])
            if lg is None:
                model = model if model is not None else load_model(ctx, r, device)
                lg, sec = rr.model_logits(model, W[h], orders, frames, device)
                rr.save_outputs(path, r["manifest"]["identity"], W[h], orders, lg, rr.per_sample(W[h], lg, seqs[h]), seqs[h], sec)
            secs += sec; probs[h] = rr.per_sample(W[h], lg, seqs[h]); logits[h][key] = lg
        systems[key] = rr.system_rows(cfg, probs, inputs) | {"fixed": fixed_rows(cfg, probs, inputs, fx["thresholds"][r["seed"]], fx)}
        n_win = sum(len(W[h]["seq"]) for h in HALVES); n_smp = sum(len(q["labels"]) for h in HALVES for q in seqs[h])
        cost[key] = {"windows": n_win, "ms_per_window": 1000 * secs / n_win, "ms_per_sample": 1000 * secs / n_smp}
        print(f"{key}: B3 {systems[key]['b3_choice']['status']} | sel {systems[key]['b3']['selection']['row']['segment_f1_50_multi']:.4f} "
              f"cal {systems[key]['b3']['calibration']['row']['segment_f1_50_multi']:.4f}", flush=True)
        del model; torch.cuda.empty_cache()
    mixed = {h: mixed_analysis(ctx, logits[h], W[h], meta[h]) for h in HALVES}; native = {h: native_analysis(ctx, logits[h], W[h], meta[h]) for h in HALVES}
    res.update({"systems": strip(systems), "mixed": {"accuracy": {k: {h: mixed[h]["accuracy"][k] for h in HALVES} for k in mixed["selection"]["accuracy"]},
                                                      "support": {h: mixed[h]["support"] for h in HALVES},
                                                      "comparisons": {f"{k}|{h}": v for h in HALVES for k, v in mixed[h]["comparisons"].items()}},
                "native": {"metrics": {k: {h: native[h][0][k] for h in HALVES} for k in native["selection"][0]},
                           "comparisons": {f"{k}|{h}": v for h in HALVES for k, v in native[h][1].items()}},
                "cost": {"inference": cost, "frame_decode_seconds": t_dec}, "seconds": time.time() - t0})
    write(res, out_dir, ids); print("round systems, mixed and native windows evaluated", flush=True)
    try:
        refs_cs, e1_preds, g1 = ra.reference_systems(cfg, {"inputs": {"protocol": ctx.rv["source"]["protocol"]}}, paths, inputs, seqs)
        for name, by in refs_cs.items():
            systems[name] = {"row": {h: {"row": by[h]["metrics"], "tables": by[h]["tables"], "preds": e1_preds[h] if name == "E1 + B3" else None} for h in HALVES}}
        old = SimpleNamespace(rv=ctx.rv, root=ctx.root, cfg=cfg)
        systems |= {k: {"row": v} for k, v in rr.reference_fact_asrf(old, paths, inputs, device).items()}
    except (SystemExit, Exception) as e:  # noqa: BLE001 - recorded with its cause; comparisons then stay open
        res.update({"references": {"not reproduced": f"{type(e).__name__}: {e}"}, "decision": "not computed: the references did not reproduce", "seconds": time.time() - t0})
        write(res, out_dir, ids); print(json.dumps(res["references"], indent=1)); return
    rvc.determinism(); del frames
    tol = float(cfg["selection"]["boundary_tolerance_s"]) * float(cfg["protocol"]["hz"])
    comp = system_comparisons(ctx, systems, inputs, tol); ver = verdicts(ctx, res["mixed"]["comparisons"], res["native"]["comparisons"], comp)
    goals = {r: rt.goal([{h: systems[f"{r}_s{s}"]["b3"][h]["row"] for h in HALVES} for s in ctx.seeds],
                        [systems[f"{r}_s{s}"]["b3_choice"]["status"] == "feasible" for s in ctx.seeds], tpc["goal"]) for r in (*tp.READOUTS, "ordered")}
    primary = {c: ver["mixed"][c]["8 of 16"] for c in ("Target - Global", "Target - CLS")}
    native_v = {c: ver["native"][c]["A4C/A5C balanced accuracy"] for c in ("Target - Global", "Target - CLS")}
    dec = pt.decide(tpc["decision_rules"], primary, native_v, ver["system"]["b3"], goals)
    res.update({"systems": strip(systems), "references_g1": g1, "system_comparisons": comp, "verdicts": ver, "goal": goals, "decision": dec,
                "units": rr.units(inputs, systems, tol),
                "cost": {**res["cost"], "training_hours": {k: v.get("hours") for k, v in queue["jobs"].items()}, "check_hours": queue.get("check_hours"),
                         "failed_attempt_hours": queue.get("failed_attempt_hours", 0.0), "evaluation_hours": (time.time() - t0) / 3600},
                "seconds": time.time() - t0})
    write(res, out_dir, ids); print(json.dumps(dec, indent=1, default=str)); print(f"wrote {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--evaluate", action="store_true"); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); ctx = tr.Ctx(cfg, a.config)
    if not a.evaluate:
        raise SystemExit("nothing to do: pass --evaluate after the nine runs")
    run_evaluate(ctx, a.workers)


if __name__ == "__main__":
    main()
