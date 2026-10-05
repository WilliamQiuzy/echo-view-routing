#!/usr/bin/env python
"""Post-hoc decision-calibration diagnostic on round 7 (configs/experiments/threshold_check_v1.yaml). No training.

For each seed, ASRF seed k decodes both validation halves into round 7's spans; the spans must equal round 7's saved
ones, and at threshold 0.5 the P0 and P1 systems must equal round 7's stored predictions. Then, for each arm (P0, P1)
and seed, every threshold of the grid is scored on the calibration half only; the feasible threshold (native macro-F1
>= 0.980, boundary recall >= 0.95) with the largest segment F1@50 is chosen (ties: closest to 0.5, then above it). The
selection half is scored only at the chosen threshold and at 0.5. The report adds paired per-seed bootstraps on the
selection half, spans and units turned right and wrong with their clips and sessions, the four common clips, the
terms and the reading. Writes report.json, report.md, the grids and the rows into --out.

  PYTHONPATH=. .venv/bin/python scripts/round7_threshold_check.py --spec runs/md_round4/spec.json --protocol P --out D
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.model_development_protocol import session_of  # noqa: E402
from scripts.report_model_development import paired_bootstrap  # noqa: E402
from scripts.report_model_development_round2 import category, metrics, read  # noqa: E402
from scripts.round4_failure_inventory import load_asrf  # noqa: E402
from scripts.round5_evidence_decider import unit_states  # noqa: E402
from scripts.round6_span_classifier import FLAT, validation_records  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.dev_metrics import evaluate_predictions  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402
from echo_routing.temporal.span_classifier import A4C, A5C, apply_decisions  # noqa: E402

ARMS, SEEDS = ("P0", "P1"), (0, 1, 2)
SYS = ("multi segment F1@50", "multi boundary recall", "native macro-F1", "native cuts per clip")
FLOOR_CAL, RECALL_MIN, FLOORS, MARGIN = 0.980, 0.95, {"selection": 0.973, "calibration": 0.980}, 0.01


def round7_dir(seed: int) -> Path:
    runs = sorted(p for p in glob.glob(f"runs/*-md7_local_readout_s{seed}-*") if "sanity" not in p)
    if len(runs) != 1:
        raise SystemExit(f"expected one round-7 run for seed {seed}, found {runs}")
    return Path(runs[0])


def decisions(margins: np.ndarray, p: float) -> list[int]:
    thr = 0.0 if p == 0.5 else math.log(p / (1 - p))
    return [A5C if m >= thr else A4C for m in margins]


def system(V: dict, dec: list[int]) -> dict:
    by_seq: dict[str, list[int]] = {}
    for i, r in enumerate(V["recs"]):
        by_seq.setdefault(r.seq_id, []).append(i)
    return {sid: apply_decisions(f, [V["recs"][i] for i in by_seq.get(sid, [])], [dec[i] for i in by_seq.get(sid, [])])
            for sid, (f, _) in V["decoded"].items()}


def choose(grid_rows: list[dict]) -> dict | None:
    ok = [g for g in grid_rows if g["feasible"]]
    if not ok:
        return None
    best = max(g["segment_f1_50_multi"] for g in ok)
    tied = [g for g in ok if abs(g["segment_f1_50_multi"] - best) < 1e-12]
    return min(tied, key=lambda g: (round(abs(g["p"] - 0.5), 6), -g["p"]))   # rounded: 0.45 and 0.55 are equally close


def span_changes(recs: list, new: list[int], ref: list[int], common: set[str]) -> dict:
    """On spans with an A4C or A5C majority: turned right and turned wrong, by class, with clips and sessions."""
    out = {}
    for cls, name in ((A4C, "A4C"), (A5C, "A5C")):
        for what, cond in (("turned right", lambda a, b: a and not b), ("turned wrong", lambda a, b: b and not a)):
            ks = [i for i, r in enumerate(recs) if r.label == cls and cond(new[i] == cls, ref[i] == cls)]
            clips = {recs[i].source for i in ks}
            out[f"{name} {what}"] = {"spans": len(ks), "clips": len(clips), "sessions": len({session_of(c) for c in clips}),
                                     "spans_from_common_clips": sum(recs[i].source in common for i in ks)}
    return out


def unit_changes(new: dict, ref: dict, common: set[str]) -> dict:
    """Stream units and native clips turned right and wrong, by class, with clips and sessions."""
    out = {}
    for kind, sel in (("units", lambda k: k[0] != "native"), ("natives", lambda k: k[0] == "native")):
        for cls, name in ((A4C, "A4C"), (A5C, "A5C")):
            for what, cond in (("turned right", lambda a, b: a and not b), ("turned wrong", lambda a, b: b and not a)):
                ks = [k for k in new if sel(k) and new[k][0] == cls and cond(new[k][1], ref[k][1])]
                clips = {c for k in ks for c in new[k][2]}
                out[f"{kind}: {name} {what}"] = {"n": len(ks), "clips": len(clips), "sessions": len({session_of(c) for c in clips}),
                                                 "n_touching_common_clips": sum(bool(set(new[k][2]) & common) for k in ks)}
    return out


def common_clips() -> set[str]:
    """Sources of A5C spans that every round-7 readout reads as A4C in every seed on the selection half."""
    sets = []
    for k in SEEDS:
        d = pd.read_csv(round7_dir(k) / "spans_selection.csv", dtype={"source": str})
        for arm in ("P0", "P1", "P2"):
            sets.append(set(d[(d["label"] == A5C) & (d[f"margin_{arm}"] < 0)]["source"]))
    return set.intersection(*sets)


def summarise(per_seed: dict) -> dict:
    return {"per_seed": per_seed, "category": {m: category([per_seed[k][m]["ci95"] for k in per_seed], len(SEEDS)) for m in SYS}}


def per_seed(k: int, inputs, cfg, e: dict, paths, device, hz: float, tol_s: float, common: set[str], out: Path) -> dict:
    rd7 = round7_dir(k); rd = Path(e["asrf"][str(k)]); asrf = load_asrf(rd, paths.repo_root, inputs.feature_dim, device)
    V = {h: validation_records(inputs, h, asrf, rd, set(cfg["sensitivity"]["exclude_clips"]), device) for h in HALVES}
    spans = {h: pd.read_csv(rd7 / f"spans_{h}.csv", dtype={"seq_id": str, "source": str}) for h in HALVES}
    for h in HALVES:
        if [(r.seq_id, r.start, r.end) for r in V[h]["recs"]] != list(zip(spans[h]["seq_id"], spans[h]["start"], spans[h]["end"])):
            raise SystemExit(f"seed {k} {h}: decoded spans differ from round 7's saved spans")
    stored = np.load(rd7 / "predictions.npz"); res = {"seed": k, "round7_run": str(rd7), "arms": {}}
    tables = {"default": {h: evaluate_predictions({sid: f for sid, (f, _) in V[h]["decoded"].items()}, V[h]["ev"], hz, tol_s)[1] for h in HALVES}}
    states = {"default": {h: unit_states({sid: f for sid, (f, _) in V[h]["decoded"].items()}, V[h]["ev"])[0] for h in HALVES}}
    vote = {h: [r.vote for r in V[h]["recs"]] for h in HALVES}
    for arm in ARMS:
        m = {h: spans[h][f"margin_{arm}"].to_numpy() for h in HALVES}
        half = {h: system(V[h], decisions(m[h], 0.5)) for h in HALVES}
        if any(not np.array_equal(half[h][sid], stored[f"{arm}|{h}|{sid}"].astype(int)) for h in HALVES for sid in half[h]):
            raise SystemExit(f"seed {k} {arm}: the 0.5 system does not reproduce round 7's stored predictions")
        grid = []
        for p in cfg["threshold"]["grid"]:
            mm, _ = evaluate_predictions(system(V["calibration"], decisions(m["calibration"], float(p))), V["calibration"]["ev"], hz, tol_s)
            grid.append({"p": float(p), **{x: mm[x] for x in FLAT},
                         "feasible": bool(mm["native_macro_f1"] >= FLOOR_CAL and mm["boundary_recall_multi"] >= RECALL_MIN)})
        pd.DataFrame(grid).to_csv(out / f"grid_{arm}_s{k}_calibration.csv", index=False)
        pick = choose(grid); a = {"grid": grid, "chosen": pick, "feasible_count": sum(g["feasible"] for g in grid)}
        runs = {"p050": (0.5, {h: (half[h], decisions(m[h], 0.5)) for h in HALVES})}
        if pick is not None:                           # the selection half is scored only at the chosen threshold
            runs["chosen"] = (pick["p"], {h: (system(V[h], decisions(m[h], pick["p"])), decisions(m[h], pick["p"])) for h in HALVES})
        a["metrics"], a["span_changes"], a["unit_changes"] = {}, {}, {}
        for tag, (p, by_h) in runs.items():
            for h in HALVES:
                mm, tabs = evaluate_predictions(by_h[h][0], V[h]["ev"], hz, tol_s)
                a["metrics"][f"{tag}|{h}"] = {"p": p, **{x: mm[x] for x in FLAT}}
                tables.setdefault(f"{arm} {tag}", {})[h] = tabs
                states.setdefault(f"{arm} {tag}", {})[h] = unit_states(by_h[h][0], V[h]["ev"])[0]
                for tn in ("native", "multi_seg", "multi_frag"):
                    tabs[tn].to_csv(out / f"rows_{arm}_{tag}_s{k}_{h}_{tn}.csv", index=False)
        if pick is not None:
            for h in HALVES:
                new = runs["chosen"][1][h][1]; recs = V[h]["recs"]
                a["span_changes"][h] = {"vs 0.5": span_changes(recs, new, runs["p050"][1][h][1], common), "vs ASRF vote": span_changes(recs, new, vote[h], common)}
                a["unit_changes"][h] = {"vs 0.5": unit_changes(states[f"{arm} chosen"][h], states[f"{arm} p050"][h], common),
                                        "vs ASRF default": unit_changes(states[f"{arm} chosen"][h], states["default"][h], common)}
        res["arms"][arm] = a
    res["_tables"] = tables
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True); ap.add_argument("--protocol", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("-c", "--config", default="configs/experiments/threshold_check_v1.yaml")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); paths = load_paths(); spec = json.loads(Path(a.spec).read_text()); e = spec["encoders"]["E1"]
    hz, tol_s = float(cfg["protocol"]["hz"]), float(cfg["selection"]["boundary_tolerance_s"])
    if [float(x) for x in cfg["threshold"]["grid"]] != [round(0.05 * i, 2) for i in range(1, 20)]:
        raise SystemExit("threshold grid differs from the config")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    inputs = Inputs(cfg, Path(a.protocol), Path(e["feature_manifest"]), paths.cache_root); common = common_clips()
    seeds = [per_seed(k, inputs, cfg, e, paths, device, hz, tol_s, common, out) for k in SEEDS]
    tables = {k: s.pop("_tables") for k, s in enumerate(seeds)}
    b3 = {h: read(Path(e["simple_run"]), f"rows_b3_tuned_main_{h}") for h in HALVES}
    strongest_t = {"selection": read(Path(spec["encoders"]["E0"]["simple_run"]), "rows_b3_tuned_main_selection"),
                   "calibration": read(Path(spec["b7_run"]), "rows_B7_b3_calibration")}
    strongest = {h: metrics(t)["segment_f1_50_multi"] for h, t in strongest_t.items()}
    feasible = {arm: all(s["arms"][arm]["chosen"] is not None for s in seeds) for arm in ARMS}
    contrasts, h = {}, "selection"
    for arm in ARMS:
        if not feasible[arm]:
            continue
        for ref, get in (("0.5", lambda k: tables[k][f"{arm} p050"][h]), ("ASRF default", lambda k: tables[k]["default"][h]),
                         ("B3 tuned (E1)", lambda k: b3[h]), ("strongest previous", lambda k: strongest_t[h])):
            contrasts[f"{arm} chosen - {ref}"] = summarise({k: paired_bootstrap(tables[k][f"{arm} chosen"][h], get(k)) for k in SEEDS})
    if all(feasible.values()):
        contrasts["P1 chosen - P0 chosen"] = summarise({k: paired_bootstrap(tables[k]["P1 chosen"][h], tables[k]["P0 chosen"][h]) for k in SEEDS})
    cat = lambda key: contrasts[key]["category"]["multi segment F1@50"] if key in contrasts else None  # noqa: E731
    terms = {"calibration_gain": {arm: cat(f"{arm} chosen - 0.5") == "consistent (higher)" for arm in ARMS},
             "ranking_into_decisions": cat("P1 chosen - P0 chosen") == "consistent (higher)",
             "beats_asrf": {arm: cat(f"{arm} chosen - ASRF default") == "consistent (higher)" for arm in ARMS},
             "feasible_in_all_seeds": feasible}
    goals = {}
    for arm in ARMS:
        if feasible[arm]:
            g = {}
            for hh in HALVES:
                ms = [metrics(tables[k][f"{arm} chosen"][hh]) for k in SEEDS]; mean = float(np.mean([x["segment_f1_50_multi"] for x in ms]))
                g[hh] = {"mean_segment_f1_50": mean, "target": strongest[hh] + MARGIN, "min_recall": min(x["boundary_recall_multi"] for x in ms),
                         "min_native_f1": min(x["native_macro_f1"] for x in ms)}
                g[hh]["passes"] = bool(mean >= g[hh]["target"] and g[hh]["min_recall"] >= RECALL_MIN and g[hh]["min_native_f1"] >= FLOORS[hh])
            goals[arm] = {**g, "passes_both": g["selection"]["passes"] and g["calibration"]["passes"]}
    gain = any(terms["calibration_gain"].values()) or terms["ranking_into_decisions"]
    reading = ("a calibration gain of a post-hoc decision rule (not a new model); the final goal is judged separately" if gain else
               "no gain: neither a calibration gain for either arm nor P1's ranking turning into better decisions; the search on these fixed spans and readout scores ends")
    report = {"common_clips": len(common), "strongest": strongest, "seeds": seeds, "contrasts_selection": contrasts, "terms": terms,
              "goals": goals, "reading": reading}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    (out / "report.md").write_text("\n".join(markdown(report)) + "\n"); print("\n".join(markdown(report)))


def markdown(r: dict) -> list[str]:
    md = ["# Round 7 threshold check: post-hoc decision calibration (generated by scripts/round7_threshold_check.py)", "",
          f"Common clips (A5C spans every round-7 readout misreads in every seed, selection half): {r['common_clips']}", "",
          "## Chosen thresholds (calibration half only)", "", "| arm | seed | feasible thresholds | chosen p | calibration F1@50 / recall / native F1 at chosen | at 0.5 |", "|---|---|---|---|---|---|"]
    for s in r["seeds"]:
        for arm in ARMS:
            a = s["arms"][arm]; c = a["chosen"]; g05 = next(g for g in a["grid"] if g["p"] == 0.5)
            ch = "none (failure)" if c is None else f"{c['p']:.2f}"
            cm = "-" if c is None else f"{c['segment_f1_50_multi']:.4f} / {c['boundary_recall_multi']:.3f} / {c['native_macro_f1']:.3f}"
            md.append(f"| {arm} | {s['seed']} | {a['feasible_count']}/19 | {ch} | {cm} | {g05['segment_f1_50_multi']:.4f} / {g05['boundary_recall_multi']:.3f} / {g05['native_macro_f1']:.3f} |")
    md += ["", "## Selection half (scored only at the chosen threshold and at 0.5)", "", "| arm | seed | p | F1@50 chosen (at 0.5) | recall chosen (0.5) | native F1 chosen (0.5) |", "|---|---|---|---|---|---|"]
    for s in r["seeds"]:
        for arm in ARMS:
            a = s["arms"][arm]
            if a["chosen"] is None:
                md.append(f"| {arm} | {s['seed']} | none | - | - | - |"); continue
            c, z = a["metrics"]["chosen|selection"], a["metrics"]["p050|selection"]
            md.append(f"| {arm} | {s['seed']} | {c['p']:.2f} | {c['segment_f1_50_multi']:.4f} ({z['segment_f1_50_multi']:.4f}) | {c['boundary_recall_multi']:.3f} ({z['boundary_recall_multi']:.3f}) | "
                      f"{c['native_macro_f1']:.3f} ({z['native_macro_f1']:.3f}) |")
    md += ["", "## Paired contrasts on the selection half (points per seed; 95% intervals)", ""]
    for key, c in r["contrasts_selection"].items():
        for m in SYS:
            v = c["per_seed"]
            md.append(f"- {key}, {m}: " + "; ".join(f"s{k} {100 * v[k][m]['delta']:+.2f} [{100 * v[k][m]['ci95'][0]:+.2f}, {100 * v[k][m]['ci95'][1]:+.2f}]" for k in v)
                      + f" -> {c['category'][m]}")
    md += ["", "## Turned right and turned wrong on the selection half (spans with an A4C or A5C majority; clips, sessions; from the common clips)", ""]
    for s in r["seeds"]:
        for arm in ARMS:
            a = s["arms"][arm]
            if a["chosen"] is None:
                continue
            for ref, d in a["span_changes"]["selection"].items():
                md.append(f"- seed {s['seed']} {arm} {ref}: " + "; ".join(f"{k} {v['spans']} ({v['clips']}, {v['sessions']}; {v['spans_from_common_clips']})" for k, v in d.items()))
    md += ["", "Units and native clips (selection half; clips, sessions; touching the common clips):", ""]
    for s in r["seeds"]:
        for arm in ARMS:
            a = s["arms"][arm]
            if a["chosen"] is None:
                continue
            for ref, d in a["unit_changes"]["selection"].items():
                md.append(f"- seed {s['seed']} {arm} {ref}: " + "; ".join(f"{k} {v['n']} ({v['clips']}, {v['sessions']}; {v['n_touching_common_clips']})" for k, v in d.items() if v["n"]))
    md += ["", "## Final goal (unchanged; calibration is optimistic here)", "", "```", json.dumps(r["goals"], indent=1), "```",
           "", "## Terms and reading", "", "```", json.dumps(r["terms"], indent=1), "```", "", f"Reading: **{r['reading']}**"]
    return md


if __name__ == "__main__":
    main()
