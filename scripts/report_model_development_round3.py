#!/usr/bin/env python
"""Round-3 report: can separately learned boundaries beat B3 and ASRF? (configs/experiments/model_development_v3.yaml)

Systems per visual input (E0, E1) and seed, every metric computed from the per-row tables of each half:
- M5 and ASRF (M1) with their default decodings (rounds 1-2);
- step 1: their existing boundary heads, intervals labelled by the frozen classifier ("bdec", threshold chosen on the
  calibration half and then fixed for the selection half);
- step 2: the boundary-only model (BO), at the pre-declared checkpoint threshold ("default") and calibrated ("bdec");
- B3 tuned on the same input's frame-level probabilities, B7 with B3, and the strongest previous configuration per
  half, fixed in the config before round 3 (selection: E0 B3 tuned; calibration: B7 with B3).
Paired per-seed contrasts with 95% bootstrap intervals and seed categories (scripts/report_model_development_round2.py),
then the success test of the round-1 plan on each half: three-seed mean segment F1@50 at least 1 point above the
strongest previous configuration, multi-fragment boundary recall >= 0.95 in every seed, and native macro-F1 not below
the encoder's frame-majority native macro-F1 minus the tolerance in every seed. For the boundary-only runs the training
logs also give the best selection-half value over all epochs; it uses the selection half twice and is only an
optimistic bound.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.report_model_development import paired_bootstrap  # noqa: E402
from scripts.report_model_development_round2 import HALVES, SEEDS, category, contrast, fmt_row, metrics, read  # noqa: E402

import numpy as np  # noqa: E402

STRONGEST = {"selection": ("E0", "B3", "tuned"), "calibration": ("system", "B7", "b3")}
SEG = "multi segment F1@50"
CONTRAST_METRICS = ("multi segment F1@50", "native macro-F1", "multi boundary recall", "native cuts per clip")


def load(spec: dict) -> tuple[dict, dict, dict]:
    """(encoder, method, decoding, seed) -> {half: tables}; decoder statuses; native floors per encoder and half."""
    systems, status, floors = {}, {}, {}
    tol = float(spec["native_f1_tolerance"])
    for enc, e in spec["encoders"].items():
        for method, runs in e["default_runs"].items():          # the run's own decoding: rows_<half>_<table>.csv
            for seed, rd in runs.items():
                systems[(enc, method, "default", int(seed))] = {h: read(Path(rd), f"rows_{h}") for h in HALVES}
        for pdir in e["bdec_runs"]:                              # posterior_decoders.py --families bdec
            res = json.loads((Path(pdir) / "posterior_decoders.json").read_text())["results"]
            for row, r in res.items():
                method, _, seed = row.rpartition("_s")
                if "bdec" in r["families"]:
                    systems[(enc, method, "bdec", int(seed))] = {h: read(Path(pdir), f"rows_{row}_bdec_{h}") for h in HALVES}
                    status[(enc, method, "bdec", int(seed))] = {"status": r["families"]["bdec"]["status"],
                                                                "params": r["families"]["bdec"]["params"]}
        simple = Path(e["simple_run"])
        systems[(enc, "B3", "tuned", None)] = {h: read(simple, f"rows_b3_tuned_main_{h}") for h in HALVES}
        floors[enc] = {h: metrics(read(simple, f"rows_b0_argmax_{h}"))["native_macro_f1"] - tol for h in HALVES}
    systems[("system", "B7", "b3", None)] = {h: read(Path(spec["b7_run"]), f"rows_B7_b3_{h}") for h in HALVES}
    return systems, status, floors


def contrast_on_half(systems: dict, new: tuple, ref: tuple, h: str) -> dict:
    """contrast() on one half only: a strongest previous configuration is the strongest on its own half, so the other
    half is never paired with it. Seeded `new` against a seeded or seedless `ref`, as in contrast()."""
    out = {}
    for seed in SEEDS:
        a = systems.get((*new, seed)); b = systems.get((*ref, seed), systems.get((*ref, None)))
        if a is not None and b is not None:
            out[seed] = {h: paired_bootstrap(a[h], b[h])}
    cats = {f"{h}|{m}": category([out[k][h][m]["ci95"] for k in out], len(SEEDS)) for m in CONTRAST_METRICS}
    return {"new": new, "ref": ref, "half": h, "per_seed": out, "category": cats}


def success(systems: dict, cand: tuple, floors: dict) -> dict:
    res = {}
    for h in HALVES:
        vals = [metrics(systems[(*cand, s)][h]) for s in SEEDS if (*cand, s) in systems]
        best = metrics(systems[(*STRONGEST[h], None)][h])["segment_f1_50_multi"]
        mean = float(np.mean([v["segment_f1_50_multi"] for v in vals]))
        rec = min(v["boundary_recall_multi"] for v in vals); nat = min(v["native_macro_f1"] for v in vals)
        res[h] = {"n_seeds": len(vals), "mean_segment_f1_50": mean, "strongest": best, "margin": mean - best,
                  "min_recall": rec, "min_native_f1": nat, "native_floor": floors[h],
                  "passes": bool(len(vals) == len(SEEDS) and mean - best >= 0.01 and rec >= 0.95 and nat >= floors[h])}
    return res


def epoch_ceiling(run_dir: Path) -> dict:
    """Best selection-half segment F1@50 over all epochs at the checkpoint threshold (optimistic: the same half picks
    and scores), next to the chosen epoch and the number of epochs that met the native floor."""
    rows = [json.loads(x) for x in (run_dir / "train_log.jsonl").read_text().splitlines()]
    man = json.loads((run_dir / "run_manifest.json").read_text())
    best = max(rows, key=lambda r: r["segment_f1_50_multi"]); chosen = next(r for r in rows if r["epoch"] == man["chosen_epoch"])
    return {"epochs_run": len(rows), "chosen_epoch": man["chosen_epoch"], "chosen_segment_f1_50": chosen["segment_f1_50_multi"],
            "best_epoch": best["epoch"], "best_segment_f1_50": best["segment_f1_50_multi"],
            "epochs_meeting_floor": sum(r["native_macro_f1"] >= man["native_floor_selection"] for r in rows)}


def contrast_lines(label: str, c: dict, halves: tuple[str, ...]) -> list[str]:
    out = []
    for h in halves:
        vals = "; ".join(f"s{s} {c['per_seed'][s][h][SEG]['delta']:+.3f} [{c['per_seed'][s][h][SEG]['ci95'][0]:+.3f}, "
                         f"{c['per_seed'][s][h][SEG]['ci95'][1]:+.3f}]" for s in c["per_seed"])
        out.append(f"- {label} [{h}]: {vals} -> {c['category'][f'{h}|{SEG}']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="JSON: native_f1_tolerance, encoders{E0,E1:{default_runs{method:{seed:run_dir}}, "
                                                   "bdec_runs[], simple_run}}, b7_run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); spec = json.loads(Path(a.spec).read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    systems, status, floors = load(spec)
    table = {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": {h: metrics(v[h]) for h in HALVES} for k, v in systems.items()}
    pairs = [("BO", "bdec", "BO", "default", "calibrated threshold vs checkpoint threshold"),
             ("BO", "bdec", "M5", "bdec", "step 2 vs step 1 with M5's boundaries"),
             ("BO", "bdec", "M1", "bdec", "step 2 vs step 1 with ASRF's boundaries"),
             ("BO", "bdec", "M5", "default", "boundary-only vs the original M5"),
             ("BO", "bdec", "M1", "default", "boundary-only vs ASRF"),
             ("BO", "bdec", "B3", "tuned", "boundary-only vs B3 on the same input"),
             ("M5", "bdec", "M5", "default", "step 1 vs M5's default output"),
             ("M1", "bdec", "M1", "default", "step 1 vs ASRF's default output")]
    contrasts, lines = {}, []
    for enc in spec["encoders"]:
        for m1, d1, m2, d2, note in pairs:
            label = f"{enc}: {m1} ({d1}) - {m2} ({d2})"
            contrasts[label] = {**contrast(systems, (enc, m1, d1), (enc, m2, d2)), "note": note}
            lines += contrast_lines(label, contrasts[label], HALVES)
        for h in HALVES:   # each strongest configuration is compared only on the half where it is the strongest
            for m, d in (("BO", "bdec"), ("M5", "bdec"), ("M1", "bdec")):
                label = f"{enc}: {m} ({d}) - strongest previous on the {h} half"
                contrasts[label] = contrast_on_half(systems, (enc, m, d), STRONGEST[h], h)
                lines += contrast_lines(label, contrasts[label], (h,))
    candidates = [(enc, m, d) for enc in spec["encoders"] for m, d in (("BO", "bdec"), ("BO", "default"), ("M5", "bdec"), ("M1", "bdec"),
                                                                      ("M5", "default"), ("M1", "default"))]
    tests = {f"{c[0]} {c[1]} ({c[2]})": success(systems, c, floors[c[0]]) for c in candidates}
    ceilings = {f"{enc} BO s{s}": epoch_ceiling(Path(rd)) for enc, e in spec["encoders"].items() for s, rd in e["default_runs"]["BO"].items()}
    (out / "round3.json").write_text(json.dumps({"systems": table, "status": {"|".join(map(str, k)): v for k, v in status.items()},
                                                 "floors": floors, "contrasts": contrasts, "success": tests, "epoch_ceilings": ceilings,
                                                 "strongest": STRONGEST, "spec": spec}, indent=1, default=str))
    md = ["# Round 3 tables (generated by scripts/report_model_development_round3.py)", "",
          "Native floors (frame-majority native macro-F1 minus the tolerance): "
          + "; ".join(f"{enc} {h} {floors[enc][h]:.4f}" for enc in floors for h in HALVES), ""]
    for h in HALVES:
        md += [f"## {h} half", "", "| System (input, method, decoding, seed) | native macro-F1 | right/all PLAX / PSAX / A4C / A5C / SC4C | "
               "segment F1@50 | boundary recall | boundary F1 | cuts/clip | cuts/min | same-class cut rate | stream sample error |",
               "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for k in sorted(systems, key=lambda k: (k[0], k[1], k[2], -1 if k[3] is None else k[3])):
            st = status.get(k); note = "" if st is None else f" [threshold {st['params']['threshold']}, {st['status']}]"
            md.append(fmt_row(f"{k[0]}, {k[1]}, {k[2]}, {'-' if k[3] is None else 's' + str(k[3])}{note}", table[f"{k[0]}|{k[1]}|{k[2]}|{k[3]}"][h]))
        md.append("")
    md += ["## Success test (round-1 plan, per half)", "",
           "| Candidate | half | mean segment F1@50 | strongest | margin | min recall | min native F1 (floor) | passes |", "|---|---|---:|---:|---:|---:|---|---|"]
    for name, t in tests.items():
        for h in HALVES:
            x = t[h]
            md.append(f"| {name} | {h} | {x['mean_segment_f1_50']:.3f} | {x['strongest']:.3f} | {x['margin']:+.3f} | {x['min_recall']:.3f} | "
                      f"{x['min_native_f1']:.3f} ({x['native_floor']:.3f}) | {'yes' if x['passes'] else 'no'} |")
    md += ["", "## Boundary-only training logs (selection half, checkpoint threshold)", "",
           "| Run | epochs run | chosen epoch | its segment F1@50 | best epoch | best segment F1@50 (optimistic) | epochs meeting the floor |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for name, c in ceilings.items():
        md.append(f"| {name} | {c['epochs_run']} | {c['chosen_epoch']} | {c['chosen_segment_f1_50']:.3f} | {c['best_epoch']} | "
                  f"{c['best_segment_f1_50']:.3f} | {c['epochs_meeting_floor']} |")
    md += ["", "## Paired contrasts per seed (segment F1@50, 95% bootstrap intervals)", "", *lines]
    (out / "round3.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
