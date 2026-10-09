#!/usr/bin/env python
"""Evaluate, freeze and report the paired FACT-ASRF round (plan docs/superpowers/plans/2026-10-08-fact-asrf-training.md,
Tasks 3-4; configs/experiments/fact_asrf_paired_v1.yaml).

  --evaluate   only after the six runs: each run's identity recomputed and its 18,750 updates checked; the six final
               weights frozen; both development halves decoded once by every system; the archived references reproduced
               with their stored settings; FACT - ASRF (primary, paired by seed) and FACT against E1 + B3, E0 + B3 and
               B7 + B3 with the paired bootstrap; verdicts, constraints, goal and the fixed decision; the mechanism tables
               (short segments, X -> Y -> X middles, false cuts, A4C/A5C changes, FACT's frame-branch readout, tokens);
               training curves and costs.
Nothing here chooses an epoch, a decoding parameter or a model. Recording and session ids stay on the server.

  python scripts/report_fact_asrf.py --config configs/experiments/fact_asrf_paired_v1.yaml --evaluate
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
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.report_fan224_visual import matched_true  # noqa: E402
from scripts.report_fold_output_correction import (boundary_changes, consistency, native_changes, segment_changes, summary,  # noqa: E402
                                                    write_atomic)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.metrics_segmentation import temporal_iou  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, sha256_file  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

NS = "fact_asrf_paired_v1"
CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
A4C, A5C = CLASSES.index("A4C"), CLASSES.index("A5C")
REFERENCES = ("E1 + B3", "E0 + B3", "B7 + B3")
COMPARE = {"FACT - ASRF": "ASRF", "FACT - (E1 + B3)": "E1 + B3", "FACT - (E0 + B3)": "E0 + B3", "FACT - (B7 + B3)": "B7 + B3"}


# ----------------------------------------------------------------------------------------------- decision
def constraints_met(row: dict, half: str, c: dict) -> bool:
    return (row["boundary_recall_multi"] or 0) >= float(c["boundary_recall_min"]) and row["native_macro_f1"] >= float(c["native_macro_f1_min"][half])


def decide(primary: dict, versus_b3: dict, fact_rows: list[dict], fa: dict) -> dict:
    """The fixed decision table. primary, versus_b3: half -> verdict of FACT - ASRF and FACT - (E1 + B3); fact_rows: per
    seed {half: summary row} of FACT."""
    c, g = fa["constraints"], fa["goal"]
    over_asrf = all(primary[h] == "consistent positive" for h in HALVES)
    over_b3 = over_asrf and all(versus_b3[h] == "consistent positive" for h in HALVES)
    failed = [{"seed": s, "half": h, "boundary_recall": r[h]["boundary_recall_multi"], "native_macro_f1": r[h]["native_macro_f1"]}
              for s, r in enumerate(fact_rows) for h in HALVES if not constraints_met(r[h], h, c)]
    mean = {h: float(np.mean([r[h]["segment_f1_50_multi"] for r in fact_rows])) for h in HALVES}
    goal = not failed and mean["selection"] >= float(g["selection_min"]) and mean["calibration"] >= float(g["calibration_min"])
    rows = fa["decision"]
    case = rows[0] if not over_asrf else rows[1] if not over_b3 else rows[2] if failed else rows[3] if not goal else rows[4]
    return {"primary": primary, "versus_E1_B3": versus_b3, "consistent_over_ASRF_both_halves": over_asrf, "consistent_over_E1_B3_both_halves": over_b3,
            "constraints_met": not failed, "constraint_failures": failed, "mean_segment_f1_FACT": mean, "goal_met": bool(goal),
            "goal_margin": {h: mean[h] - float(g[f"{h}_min"]) for h in HALVES},
            "calibration_sensitivity_margin": mean["calibration"] - float(g["calibration_sensitivity"]), **case}


# ----------------------------------------------------------------------------------------------- mechanism measures
def source_of(stream) -> np.ndarray:
    """The source recording of every sample of a constructed stream."""
    fr = stream.recipe.fragments
    return np.repeat([f.video_id for f in fr], [f.end - f.start for f in fr])


def miss_reason(ref, pred: list) -> str:
    """Why a reference segment was not matched: a predicted segment covers it at IoU >= 0.5 with another view, no
    predicted segment reaches IoU 0.5, or a same-view one does but was matched to another reference (one to one)."""
    best = max(pred, key=lambda p: temporal_iou(p, ref), default=None)
    if best is None or temporal_iou(best, ref) < 0.5:
        return "no segment at IoU 0.5"
    return "wrong view" if best.label != ref.label else "taken by another reference"


def unit_recall(ev, preds: dict, pick) -> dict:
    """Reference segments of the multi streams chosen by pick(runs, k): matched one to one at IoU 0.5, the reasons of the
    misses, and the distinct source recordings and sessions of the misses."""
    out = {"segments": 0, "matched": 0, "wrong view": 0, "no segment at IoU 0.5": 0, "taken by another reference": 0}
    rec, ses = set(), set()
    for s in ev.multi:
        true = labels_to_segments(s.labels); pred = labels_to_segments(np.asarray(preds[s.recipe.recipe_id]))
        chosen = [k for k in range(len(true)) if pick(true, k)]
        if not chosen:
            continue
        hit = matched_true(pred, true); src = source_of(s)
        for k in chosen:
            out["segments"] += 1
            if k in hit:
                out["matched"] += 1
            else:
                out[miss_reason(true[k], pred)] += 1; v = set(src[true[k].start:true[k].end].tolist()); rec |= v; ses |= {session_of(x) for x in v}
    return {**out, "recall": out["matched"] / out["segments"] if out["segments"] else None, "missed_recordings": len(rec), "missed_sessions": len(ses)}


def short(max_len: int, views: tuple | None = None):
    return lambda t, k: t[k].length <= max_len and (views is None or t[k].label in views)


def xyx_middle(pair: tuple | None = None):
    """The middle run of three consecutive reference runs X, Y, X with Y different from X; pair limits {X, Y}."""
    def pick(t, k):
        if k == 0 or k == len(t) - 1 or t[k - 1].label != t[k + 1].label or t[k].label == t[k - 1].label:
            return False
        return pair is None or {t[k - 1].label, t[k].label} == set(pair)
    return pick


def mechanism(ev, preds: dict, max_len: int) -> dict:
    return {"short segments": unit_recall(ev, preds, short(max_len)), "short A4C or A5C segments": unit_recall(ev, preds, short(max_len, (A4C, A5C))),
            "X-Y-X middles": unit_recall(ev, preds, xyx_middle()), "A4C/A5C X-Y-X middles": unit_recall(ev, preds, xyx_middle((A4C, A5C)))}


def token_runs(assign: np.ndarray | None) -> dict:
    """Runs of the frame-to-token assignment; a token that owns two or more separate runs covers separate intervals."""
    if assign is None or assign.size == 0:
        return {"runs": 0, "tokens": 0, "separate_intervals": False}
    runs = [g.label for g in labels_to_segments(assign)]
    return {"runs": len(runs), "tokens": len(set(runs)), "separate_intervals": len(set(runs)) < len(runs)}


def token_table(ev, fact_tokens: dict) -> dict:
    """Non-null tokens against the reference segment count (multi streams), and outputs with a token on separate intervals."""
    diff = [fact_tokens[s.recipe.recipe_id]["tokens_used"] - len(labels_to_segments(s.labels)) for s in ev.multi]
    sep = [k for k, v in fact_tokens.items() if v["separate_intervals"]]
    return {"multi_streams": len(diff), "tokens_equal_segments": int(sum(d == 0 for d in diff)), "fewer_tokens": int(sum(d < 0 for d in diff)),
            "more_tokens": int(sum(d > 0 for d in diff)), "mean_tokens_minus_segments": float(np.mean(diff)) if diff else None,
            "sequences": len(fact_tokens), "outputs_with_a_token_on_separate_intervals": len(sep)}


# ----------------------------------------------------------------------------------------------- systems
@torch.no_grad()
def fact_outputs(head, model, seqs: list, device) -> dict:
    """Official labels, the frame-branch argmax, the non-null token count and the frame-to-token assignment (as in the
    official Block._eval) of every sequence."""
    model.eval(); out = {}
    for s in seqs:
        x = torch.from_numpy(s.feat.astype(np.float32)).to(device); pr = head.predict(model, x); last = model.block_list[-1]
        cls = last.action_clogit[:, 0].argmax(-1); active = torch.where(cls != head.num_classes)[0]
        assign = active[last.a2f_attn[0][:, active].argmax(-1)].cpu().numpy() if active.numel() else None
        out[s.seq_id] = {"labels": pr["labels"], "frame_labels": pr["frame_labels"], "tokens_used": pr["tokens_used"], **token_runs(assign)}
    return out


def verify_runs(ctx, fa: dict, device) -> dict:
    """Every run of the queue: finished, 18,750 updates, intact final weights and the identity recomputed from the inputs."""
    import scripts.train_fact_asrf as tr
    runs = {}
    for key in fa["queue"]:
        arm, seed = key.split("_s"); seed = int(seed); man_path = ctx.server / "train" / key / "run_manifest.json"
        if not man_path.is_file():
            raise SystemExit(f"{key}: no finished run")
        man = json.loads(man_path.read_text()); plan, plan_sha = tr.load_plan(ctx, seed); head = tr.make_head(ctx, arm, plan, device)
        ident = tr.identity(ctx, arm, seed, head, plan_sha, sha256_file(ctx.init_path(arm, seed)))
        if man["identity"] != ident or int(man["completed_updates"]) != int(fa["data"]["total_updates"]) or sha256_file(Path(man["final_checkpoint"])) != man["final_sha256"]:
            raise SystemExit(f"{key}: identity, update count or final weights do not verify")
        model = head.build().to(device); model.load_state_dict(torch.load(man["final_checkpoint"], map_location=device, weights_only=True)); model.eval()
        runs[key] = {"arm": arm, "seed": seed, "head": head, "model": model, "manifest": man}
    return runs


def freeze(paths, runs: dict, config: str) -> dict:
    out = {}
    for key, r in runs.items():
        man, dest = r["manifest"], paths.repo_root / "models_frozen" / NS / key
        try:
            if not dest.exists():
                files = [man["final_checkpoint"], str(paths.repo_root / "runs" / NS / "train" / key / "run_manifest.json"), man["plan"], man["init"], str(paths.repo_root / config)]
                subprocess.run([sys.executable, "scripts/freeze_model.py", NS, key, *[x for f in files for x in ("--file", f)], "--meta", f"identity={man['identity']}",
                                "--meta", f"arm={r['arm']}", "--meta", f"seed={r['seed']}", "--notes", "Paired FACT-ASRF round (2026-10-08); final weights after 18,750 updates."],
                               cwd=paths.repo_root, check=True, capture_output=True)
            out[key] = str(dest) if sha256_file(dest / "final.pt") == man["final_sha256"] else f"frozen copy differs from {man['final_sha256'][:12]}"
        except (subprocess.CalledProcessError, OSError) as e:
            out[key] = f"not frozen: {type(e).__name__}"
    return out


def reference_systems(cfg: dict, fa: dict, paths, inp_e1, seqs: dict) -> tuple:
    """The archived reference rows reproduced with their stored settings on the halves built once, the E1 + B3 predictions
    (recomputed exactly as the reference does and checked against its rows) and the G1 + ASRF rows."""
    from scripts.ev9v_complementarity import decode
    from scripts.report_center_supervision import check_rows, evaluate, references
    from scripts.report_model_development_round2 import metrics, read
    root = paths.repo_root; inp = fa["inputs"]
    refs = references({**cfg, "center_supervision": {"inputs": {"protocol": inp["protocol"]}}}, paths, inp_e1, seqs)
    e1 = root / "runs/20261002-144641-md_simple_decoders-38bbcf7c-s0"; params = json.loads((e1 / "selected.json").read_text())["selected"]["b3"]["main"]; e1_preds = {}
    for h in HALVES:
        ev = inp_e1.halves[h]; probs = {s.seq_id: inp_e1.store.get(s.seq_id, "orig").prob for s in ev.natives}
        probs.update({s.recipe.recipe_id: s.prob for s in (*ev.pairs, *ev.multi)})
        e1_preds[h] = decode(params, probs); _, t = evaluate(cfg, e1_preds[h], ev); check_rows("E1 + B3", h, t, e1 / f"rows_b3_tuned_main_{h}_multi_seg.csv")
    g1 = {}
    for k in (0, 1, 2):
        st = json.loads((root / f"runs/geometry/downstream/G1_s{k}_status.json").read_text())
        g1[f"G1 + ASRF s{k}"] = {h: metrics(read(Path(st["M1"]["run_dir"]), f"rows_{h}")) for h in HALVES}
    g1_cal = float(np.mean([g1[f"G1 + ASRF s{k}"]["calibration"]["segment_f1_50_multi"] for k in (0, 1, 2)]))
    if abs(g1_cal - 0.9738935784747763) > 1e-12:
        raise SystemExit(f"G1 + ASRF calibration mean {g1_cal} does not reproduce 0.9738935784747763")
    return refs, e1_preds, g1


def evaluate_round(cfg: dict, runs: dict, inp_e1, device) -> dict:
    """FACT and ASRF on both halves: rows, tables, predictions and the FACT readouts; decode time per sequence. Each arm
    decodes under the determinism setting it trained and was smoke-tested with."""
    from echo_routing.evaluate.dev_metrics import evaluate_predictions
    hz, tol_s = float(cfg["protocol"]["hz"]), float(cfg["selection"]["boundary_tolerance_s"]); out, cost = {}, {}
    for h in HALVES:
        ev = inp_e1.halves[h]; seqs = ev.sequences()
        for key, r in runs.items():
            torch.use_deterministic_algorithms(r["arm"] == "FACT"); torch.cuda.synchronize(); t0 = time.time()
            if r["arm"] == "FACT":
                o = fact_outputs(r["head"], r["model"], seqs, device); full = {k: v["labels"] for k, v in o.items()}; frame = {k: v["frame_labels"] for k, v in o.items()}
            else:
                dec = r["head"].decode(r["model"], seqs, device); o = None
                full = {k: np.asarray(v["labels"], int) for k, v in dec.items()}; frame = {k: np.asarray(v["frame_labels"], int) for k, v in dec.items()}
            torch.cuda.synchronize(); cost.setdefault(key, []).append((time.time() - t0) / len(seqs))
            mf, tf_ = evaluate_predictions(full, ev, hz, tol_s); mr, _ = evaluate_predictions(frame, ev, hz, tol_s)
            out.setdefault(key, {})[h] = {"row": summary(mf), "frame_argmax": summary(mr), "tables": tf_, "preds": full, "tokens": o}
    return {"systems": out, "decode_seconds_per_sequence": {k: float(np.mean(v)) for k, v in cost.items()}}


def add_references(cfg: dict, systems: dict, refs: dict, e1_preds: dict, inp_e1) -> None:
    """E1 + B3 (with predictions), E0 + B3 and B7 + B3 (rows and tables) beside the round's systems."""
    from echo_routing.evaluate.dev_metrics import evaluate_predictions
    hz, tol_s = float(cfg["protocol"]["hz"]), float(cfg["selection"]["boundary_tolerance_s"])
    for h in HALVES:
        mb, tb = evaluate_predictions(e1_preds[h], inp_e1.halves[h], hz, tol_s)
        systems.setdefault("E1 + B3", {})[h] = {"row": summary(mb), "tables": tb, "preds": e1_preds[h]}
        for name in ("E0 + B3", "B7 + B3"):
            systems.setdefault(name, {})[h] = {"row": refs[name][h]["metrics"], "tables": refs[name][h]["tables"], "preds": None}


def comparisons(fa: dict, systems: dict, inp_e1, tol: float, names) -> dict:
    from scripts.report_model_development import paired_bootstrap
    out = {}
    for seed in [int(s) for s in fa["seeds"]]:
        for name in names:
            other = COMPARE[name]; kb = f"ASRF_s{seed}" if other == "ASRF" else other
            for h in HALVES:
                A, B = systems[f"FACT_s{seed}"][h], systems[kb][h]
                c = {"bootstrap": paired_bootstrap(A["tables"], B["tables"]), "native_changes": native_changes(A["tables"]["native"], B["tables"]["native"])}
                if B["preds"] is not None:
                    c["segment_changes"] = segment_changes(inp_e1.halves[h], A["preds"], B["preds"])
                    c["boundary_changes"] = boundary_changes(inp_e1.halves[h], A["preds"], B["preds"], tol)
                out[f"s{seed}|{name}|{h}"] = c
    return out


def verdicts(fa: dict, comp: dict, names) -> dict:
    seeds = [int(s) for s in fa["seeds"]]
    return {name: {h: consistency([comp[f"s{s}|{name}|{h}"]["bootstrap"]["multi segment F1@50"] for s in seeds]) for h in HALVES} for name in names}


def plot_curves(runs: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2)); colors = {"FACT": "#c44e52", "ASRF": "#4c72b0"}; styles = {0: "-", 1: "--", 2: ":"}
    for key, r in runs.items():
        rec = r["manifest"]["records"]; x = [q["updates"] for q in rec]; kw = {"color": colors[r["arm"]], "ls": styles[r["seed"]], "lw": 1.2, "label": key}
        ax[0].plot(x, [q["loss"]["total"] for q in rec], **kw); ax[1].plot(x, [q["lr_last"] for q in rec], **kw); ax[2].plot(x, [q["monitor"]["segment_f1_50"] for q in rec], **kw)
    ax[0].set_yscale("log"); ax[1].set_yscale("log")
    for a, t in zip(ax, ("training loss (mean per epoch; arms differ in loss scale)", "learning rate (last update of each epoch)", "monitor: 16 fixed training sequences, segment F1@50")):
        a.set_title(t, fontsize=10); a.set_xlabel("updates"); a.grid(alpha=0.3, which="both")
    ax[2].legend(fontsize=8, ncol=2); fig.suptitle("Paired FACT-ASRF training (final weights kept; nothing chosen from these curves)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=130)


def strip(systems: dict) -> dict:
    return {k: {h: {x: v for x, v in r.items() if x in ("row", "frame_argmax")} for h, r in per.items()} for k, per in systems.items()}


def tables_md(res: dict) -> str:
    rows = []
    for name, per in res.get("systems", {}).items():
        for h in HALVES:
            r = per[h]["row"]
            rows.append({"system": name, "half": h, "segment F1@50": r["segment_f1_50_multi"], "boundary P": r.get("boundary_precision_multi"), "boundary R": r["boundary_recall_multi"],
                         "native macro-F1": r["native_macro_f1"], "cuts per clip": r.get("native_internal_cuts_per_clip"), "same-class cut rate": r.get("same_class_cut_rate_multi"),
                         "A5C read as A4C": r.get("A5C read as A4C"), "A4C read as A5C": r.get("A4C read as A5C"),
                         "frame-branch segment F1@50": per[h].get("frame_argmax", {}).get("segment_f1_50_multi")})
    comp = []
    for k, c in res.get("comparisons", {}).items():
        b = c["bootstrap"]["multi segment F1@50"]
        comp.append({"comparison": k, "d segment F1@50": f"{b['delta']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]",
                     "native fixed/new": f"{c['native_changes']['all clips']['fixed']}/{c['native_changes']['all clips']['new']}"})
    parts = ["# Paired FACT-ASRF round: tables\n", "Recompute: `python scripts/report_fact_asrf.py --config configs/experiments/fact_asrf_paired_v1.yaml --evaluate`\n"]
    if rows:
        parts += ["## Systems\n", md_table(rows, list(dict.fromkeys(x for r in rows for x in r)))]
    if comp:
        parts += ["## Comparisons\n", md_table(comp)]
    return "\n".join(parts + ["## Verdicts\n", "```\n" + json.dumps(res.get("verdicts", "pending"), indent=1) + "\n```\n",
                               "## Decision\n", "```\n" + json.dumps(res.get("decision", "pending"), indent=1, default=str) + "\n```\n"])


def write(res: dict, out_dir: Path, ids: set) -> None:
    """Every write goes through both id guards; called after each stage so that nothing finished is lost."""
    assert_no_ids(json.loads(json.dumps(res, default=str)), ids, "summary")
    tables = tables_md(res)
    if any(s in tables for s in {session_of(v) for v in ids}):
        raise SystemExit("a session id appears in the report tables")
    write_atomic(out_dir / "summary.json", json.dumps(res, indent=1, default=str)); write_atomic(out_dir / "report_tables.md", tables)


def run_evaluate(cfg: dict, fa: dict, paths, config: str) -> None:
    import scripts.train_fact_asrf as tr
    from scripts.report_center_supervision import validation
    t0 = time.time(); device = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction()); tr.determinism("FACT")
    ctx = tr.Ctx(cfg, paths); runs = verify_runs(ctx, fa, device); print("six runs verified", flush=True)
    out_dir = paths.repo_root / fa["outputs"]["results_dir"]; out_dir.mkdir(parents=True, exist_ok=True)
    ids = ctx.data.ids(); tol = float(cfg["selection"]["boundary_tolerance_s"]) * float(cfg["protocol"]["hz"]); max_len = 16
    res = {"runs": {k: {x: r["manifest"][x] for x in ("identity", "arm", "seed", "completed_updates", "final_sha256", "plan_sha256", "init_sha256", "parameters", "minutes")}
                    for k, r in runs.items()}, "frozen": freeze(paths, runs, config), "references": "pending", "decision": "pending"}
    write(res, out_dir, ids); plot_curves(runs, out_dir / "training_curves.png")                     # stage 1: runs and frozen weights
    inp = fa["inputs"]; inp_e1, seqs = validation(cfg, {"inputs": {"protocol": inp["protocol"], "feature_manifest": inp["e1_feature_manifest"]}}, paths)
    ev = evaluate_round(cfg, runs, inp_e1, device); systems = ev["systems"]; first = ("FACT - ASRF",)
    comp = comparisons(fa, systems, inp_e1, tol, first)
    res.update({"systems": strip(systems), "comparisons": comp, "verdicts": verdicts(fa, comp, first),
                "mechanism": {k: {h: mechanism(inp_e1.halves[h], systems[k][h]["preds"], max_len) for h in HALVES} for k in systems},
                "tokens": {k: {h: token_table(inp_e1.halves[h], systems[k][h]["tokens"]) for h in HALVES} for k in systems if k.startswith("FACT")},
                "cost": {"training_hours": {k: r["manifest"]["minutes"] / 60 for k, r in runs.items()}, "decode_seconds_per_sequence": ev["decode_seconds_per_sequence"]},
                "seconds": time.time() - t0})
    write(res, out_dir, ids); print("FACT and ASRF evaluated", flush=True)                          # stage 2: the round's own systems
    try:
        refs, e1_preds, g1 = reference_systems(cfg, fa, paths, inp_e1, seqs); add_references(cfg, systems, refs, e1_preds, inp_e1)
    except (SystemExit, Exception) as e:  # noqa: BLE001 - a reproduction failure is reported with its cause; the decision then stays open
        res.update({"references": {"not reproduced": f"{type(e).__name__}: {e}"}, "decision": "not computed: the references did not reproduce",
                    "seconds": time.time() - t0})
        write(res, out_dir, ids); print(json.dumps(res["references"], indent=1)); return
    rest = tuple(n for n in COMPARE if n not in first); comp.update(comparisons(fa, systems, inp_e1, tol, rest)); ver = verdicts(fa, comp, tuple(COMPARE))
    fact_rows = [{h: systems[f"FACT_s{s}"][h]["row"] for h in HALVES} for s in fa["seeds"]]
    res["mechanism"]["E1 + B3"] = {h: mechanism(inp_e1.halves[h], systems["E1 + B3"][h]["preds"], max_len) for h in HALVES}
    res.update({"systems": strip(systems), "comparisons": comp, "verdicts": ver, "decision": decide(ver["FACT - ASRF"], ver["FACT - (E1 + B3)"], fact_rows, fa),
                "references": {k: {h: v[h]["metrics"] for h in HALVES} for k, v in refs.items()},
                "references_g1": {"schema": "scripts.report_model_development_round2.metrics on the stored rows", **g1}, "seconds": time.time() - t0})
    write(res, out_dir, ids); print(json.dumps(res["decision"], indent=1, default=str))            # stage 3: references, comparisons, decision


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--evaluate", action="store_true", required=True)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); paths = load_paths()
    run_evaluate(cfg, cfg["fact_asrf_paired"], paths, a.config)


if __name__ == "__main__":
    main()
