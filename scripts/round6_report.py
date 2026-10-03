#!/usr/bin/env python
"""Round-6 report over the three seed runs of scripts/round6_span_classifier.py (configs/experiments/
span_classifier_v1.yaml): span statistics, span classification, the four systems on both halves (all clips and
without the ten sensitivity clips), the success test, paired per-seed bootstrap contrasts, flips and false pieces, and
the pre-declared terms and interpretation rules. Writes report.json and report.md into --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.report_model_development import paired_bootstrap  # noqa: E402
from scripts.report_model_development_round2 import category, read  # noqa: E402

import numpy as np  # noqa: E402

HALVES = ("selection", "calibration")
SYSTEMS = ("default", "mean", "V0", "V1")
METRIC = "multi segment F1@50"
PIECES = ("A5C pieces inside A4C units", "A4C pieces inside A5C units")


def contrast(runs: list[Path], new: str, ref: str, scope: str, h: str, strongest=None) -> dict:
    per = {}
    for k, rd in enumerate(runs):
        b = strongest if strongest is not None else read(rd, f"rows_{ref}_{scope}_{h}")
        per[k] = paired_bootstrap(read(rd, f"rows_{new}_{scope}_{h}"), b)[METRIC]
    return {"per_seed": per, "category": category([per[k]["ci95"] for k in per], len(runs))}


def terms(res: dict, c: dict) -> dict:
    """The pre-declared terms of the config (section terms) for V0 and V1."""
    out = {}
    for m in ("V0", "V1"):
        cls = all(res[k]["classification"][h][m]["accuracy"] > res[k]["classification"][h]["vote"]["accuracy"] for k in range(3) for h in HALVES)
        cats = [c[f"{m} - default|all|{h}"]["category"] for h in HALVES]
        system = any(x == "consistent (higher)" for x in cats) and not any(x == "consistent (lower)" for x in cats)
        frag = any(sum(sum(res[k]["pieces"][f"{m}|{h}"][p]["n"] for p in PIECES) > sum(res[k]["pieces"][f"default|{h}"][p]["n"] for p in PIECES)
                       for k in range(3)) >= 2 for h in HALVES)
        out[m] = {"classification_improves": cls, "system_improves": system, "fragments_increase": frag}
    cats = [c[f"V1 - V0|all|{h}"]["category"] for h in HALVES]
    out["v1_beats_v0"] = any(x == "consistent (higher)" for x in cats) and not any(x == "consistent (lower)" for x in cats)
    return out


def reading(t: dict, success: dict) -> str:
    v0, v1 = t["V0"], t["V1"]
    improves = lambda x: x["classification_improves"] or x["system_improves"]  # noqa: E731
    if not improves(v0) and not improves(v1):
        return "neither improves: the short-span classification route ends"
    # rule 3 has two independent failure modes: classification improves while false pieces increase, or the full system
    # stays behind (no model passes the success test on both halves)
    cls = v0["classification_improves"] or v1["classification_improves"]
    frag = v0["fragments_increase"] or v1["fragments_increase"]
    behind = not any(success[m]["passes_both"] for m in ("V0", "V1"))
    notes = ["the model goal is not met (classification improves but false pieces increase, or the full system stays behind)"] \
        if (cls and frag) or behind else []
    if t["v1_beats_v0"] and v1["system_improves"]:
        notes.append("V1 beats V0 consistently and improves the full system: supports short-span supervision improving the representation")
    elif improves(v0) and not t["v1_beats_v0"]:
        notes.append("V0 improves and V1 does not improve further: the gain is not due to visual fine-tuning")
    return "; ".join(notes) or "the full system improves"


def fmt_ci(cn: dict) -> str:
    return " / ".join(f"{100 * v['delta']:+.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]" for v in cn["per_seed"].values()) + f", {cn['category']}"


def markdown(r: dict) -> list[str]:
    res, md = r["seeds"], ["# Round 6: span-level A4C/A5C training, V0 (frozen encoder) and V1 (last block trained)", "",
                           f"Targets: selection {r['target']['selection']:.4f}, calibration {r['target']['calibration']:.4f}; floors 0.973 and 0.980.", "",
                           "## Spans (per seed: training split / selection / calibration)", "",
                           "| seed | part | spans (A4C/A5C majority) | A5C | median length | p90 length | mixed share | vote differs from majority |",
                           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for k, x in enumerate(res):
        for part, st in x["span_stats"].items():
            md.append(f"| {k} | {part} | {st['with_A4C_A5C_majority']} | {st['A5C']} | {st['length_median']:.0f} | {st['length_p90']:.0f} | "
                      f"{100 * st['mixed_share']:.1f}% | {100 * st['vote_differs_share']:.1f}% |")
    md += ["", "## Span classification (validation spans with an A4C or A5C majority)", "",
           "| half | rule | accuracy s0 / s1 / s2 | A4C recall | A5C recall | source clips mostly right |", "|---|---|---|---|---|---|"]
    for h in HALVES:
        for rule in ("vote", "mean", "V0", "V1"):
            cs = [x["classification"][h][rule] for x in res]
            md.append(f"| {h} | {rule} | " + " / ".join(f"{c['accuracy']:.3f}" for c in cs) + " | " + " / ".join(f"{c['recall_A4C']:.3f}" for c in cs)
                      + " | " + " / ".join(f"{c['recall_A5C']:.3f}" for c in cs) + " | " + " / ".join(f"{100 * c['clips_mostly_right']:.0f}%" for c in cs) + " |")
    for scope, title in (("all", "All clips (main result)"), ("without", "Without the ten sensitivity clips (dependence only)")):
        md += ["", f"## Systems: {title}", "", "| half | system | segment F1@50 s0 / s1 / s2 (mean) | recall | native macro-F1 | predicted segments |",
               "|---|---|---|---|---|---|"]
        for h in HALVES:
            for m in SYSTEMS:
                ms = [x["metrics"][f"{m}|{scope}|{h}"] for x in res]; seg = [v["segment_f1_50_multi"] for v in ms]
                md.append(f"| {h} | {m} | " + " / ".join(f"{v:.3f}" for v in seg) + f" ({np.mean(seg):.3f}) | " + " / ".join(f"{v['boundary_recall_multi']:.3f}" for v in ms)
                          + " | " + " / ".join(f"{v['native_macro_f1']:.3f}" for v in ms) + " | " + " / ".join(str(v["predicted_segments"]) for v in ms) + " |")
    md += ["", "## Success test (all clips)", "", "| system | half | mean (target) | min recall | min native F1 (floor) | passes |", "|---|---|---|---|---|---|"]
    for m, ok in r["success"].items():
        for h in HALVES:
            o = ok[h]
            md.append(f"| {m} | {h} | {o['mean_segment_f1_50']:.3f} ({o['target']:.3f}) | {o['min_recall']:.3f} | {o['min_native_f1']:.3f} ({o['floor']:.3f}) | {'yes' if o['passes'] else 'no'} |")
    md += ["", "## Paired contrasts of segment F1@50 (points per seed, 95% intervals)", ""] + [f"- {k.replace('|', ', ')}: {fmt_ci(v)}" for k, v in r["contrasts"].items()]
    md += ["", "## A5C turning right and A4C turning wrong (stream units; native clips in brackets)", "",
           "| comparison | half | A5C fixed s0 / s1 / s2 (clips) | A4C broken s0 / s1 / s2 (clips) |", "|---|---|---|---|"]
    for comp in ("V0 vs default", "V1 vs default", "V0 vs mean", "V1 vs mean", "V1 vs V0"):
        for h in HALVES:
            f = [x["flips"][f"{comp}|{h}"] for x in res]
            cell = lambda key, nat: " / ".join(f"{y[key]['n']} ({y[key]['clips']}) [{y[nat]['n']}]" for y in f)  # noqa: E731
            md.append(f"| {comp} | {h} | {cell('units: A5C fixed', 'natives: A5C fixed')} | {cell('units: A4C broken', 'natives: A4C broken')} |")
    md += ["", "## False pieces (unmatched predicted segments inside a reference segment of the other class; clips in brackets)", "",
           "| system | half | A5C inside A4C s0 / s1 / s2 | A4C inside A5C s0 / s1 / s2 |", "|---|---|---|---|"]
    for m in SYSTEMS:
        for h in HALVES:
            pc = [x["pieces"][f"{m}|{h}"] for x in res]
            md.append(f"| {m} | {h} | " + " / ".join(f"{y[PIECES[0]]['n']} ({y[PIECES[0]]['clips']})" for y in pc) + " | "
                      + " / ".join(f"{y[PIECES[1]]['n']} ({y[PIECES[1]]['clips']})" for y in pc) + " |")
    md += ["", "## Training", ""] + [f"- seed {k}: V0 final loss {x['train_log']['V0'][-1]['loss']:.4f}; V1 final loss {x['train_log']['V1'][-1]['loss']:.4f}, "
                                     f"{sum(e['seconds'] for e in x['train_log']['V1']) / 60:.0f} min; parity {x['parity']['V1']['parity_reference']:.1e} / "
                                     f"{x['parity']['V1']['parity_cache']:.1e}" for k, x in enumerate(res)]
    md += ["", "## Pre-declared terms and reading", "", "```", json.dumps(r["terms"], indent=1), "```", "", f"Reading: **{r['reading']}**"]
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True, help="seed0_dir,seed1_dir,seed2_dir"); ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("-c", "--config", default="configs/experiments/span_classifier_v1.yaml")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = [Path(x) for x in a.runs.split(",")]; res = [json.loads((r / "seed.json").read_text()) for r in runs]
    if [r["seed"] for r in res] != [0, 1, 2] or any(r["sanity"] for r in res):
        raise SystemExit("expected the three full seed runs 0, 1, 2")
    spec = json.loads(Path(a.spec).read_text())
    strongest = {"selection": read(Path(spec["encoders"]["E0"]["simple_run"]), "rows_b3_tuned_main_selection"),
                 "calibration": read(Path(spec["b7_run"]), "rows_B7_b3_calibration")}
    target = {h: float(2 * t["multi_seg"].seg_matched.sum() / (t["multi_seg"].seg_pred.sum() + t["multi_seg"].seg_true.sum())) + float(cfg["success"]["segment_f1_margin"])
              for h, t in strongest.items()}
    floors = {"selection": 0.9730, "calibration": 0.9800}   # the E1 floors of rounds 2-5, fixed in the config
    c = {}
    for scope in ("all", "without"):
        for new, ref in (("V1", "V0"), ("V0", "mean"), ("V1", "mean"), ("V0", "default"), ("V1", "default"), ("mean", "default")):
            for h in HALVES:
                c[f"{new} - {ref}|{scope}|{h}"] = contrast(runs, new, ref, scope, h)
    for m in SYSTEMS:
        for h in HALVES:
            c[f"{m} - strongest previous|all|{h}"] = contrast(runs, m, None, "all", h, strongest[h])
    success = {}
    for m in SYSTEMS:
        ok = {}
        for h in HALVES:
            ms = [r["metrics"][f"{m}|all|{h}"] for r in res]; mean = float(np.mean([x["segment_f1_50_multi"] for x in ms]))
            ok[h] = {"mean_segment_f1_50": mean, "target": target[h], "min_recall": min(x["boundary_recall_multi"] for x in ms),
                     "min_native_f1": min(x["native_macro_f1"] for x in ms), "floor": floors[h],
                     "passes": bool(mean >= target[h] and min(x["boundary_recall_multi"] for x in ms) >= 0.95 and min(x["native_macro_f1"] for x in ms) >= floors[h])}
        success[m] = {**ok, "passes_both": ok["selection"]["passes"] and ok["calibration"]["passes"]}
    t = terms(res, c); report = {"target": target, "floors": floors, "success": success, "contrasts": c, "terms": t, "reading": reading(t, success),
                                 "seeds": res}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    (out / "report.md").write_text("\n".join(markdown(report)) + "\n"); print("\n".join(markdown(report)))


if __name__ == "__main__":
    main()
