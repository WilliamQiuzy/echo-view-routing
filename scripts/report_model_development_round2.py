#!/usr/bin/env python
"""Round-2 report (docs/superpowers/plans/2026-10-02-model-development-round2.md, Tasks 4, 5b and 6).

Every system is (visual input E0/E1, method, decoding, seed). All metrics are computed from the same per-row tables
(native clips, multi-fragment segments and fragments) of each half, so every row uses one definition:
native macro-F1 with per-class right/all, multi segment F1@50, boundary recall and F1 (0.5 s), cuts per clip and per
minute, same-class cut rate and the per-sample error rate on multi-fragment streams.

Contrasts are paired per seed with 95% bootstrap intervals (native clips by session, streams by stream; the clip reuse
across streams is not modelled, so intervals are optimistic and seeds are not independent test sets): E1 - E0 for each
head, M5 - M3p (path term), and the candidate against ASRF and B3. The Task 5b start condition is evaluated as written:
(A or B) and C, then the candidate rule.

Inputs come from a JSON spec (--spec) listing run directories, posterior-decoder runs, simple-decoder runs, the
B7 posterior run and the oracle diagnostics; see main() for the keys.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.report_model_development import paired_bootstrap  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HALVES = ("selection", "calibration")
TABLES = ("native", "multi_seg", "multi_frag")
CLASSES = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]
SEEDS = (0, 1, 2)
EPS = 1e-6


def read(d: Path, prefix: str) -> dict[str, pd.DataFrame]:
    return {t: pd.read_csv(d / f"{prefix}_{t}.csv", dtype={"video_id": str, "stream": str}) for t in TABLES}


def metrics(tb: dict[str, pd.DataFrame], hz: float = 10.0) -> dict:
    nat, seg, frag = tb["native"], tb["multi_seg"], tb["multi_frag"]
    conf = np.zeros((5, 5), int); np.add.at(conf, (nat["label"].to_numpy(), nat["majority"].to_numpy()), 1)
    tp = np.diag(conf); den = 2 * tp + (conf.sum(0) - tp) + (conf.sum(1) - tp)
    s = frag.sum(numeric_only=True); g = seg.sum(numeric_only=True)
    return {"native_macro_f1": float(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0).mean()),
            "native_recall": {c: [int(tp[i]), int(conf[i].sum())] for i, c in enumerate(CLASSES)},
            "segment_f1_50_multi": float(2 * g["seg_matched"] / (g["seg_pred"] + g["seg_true"])),
            "boundary_recall_multi": float(s["t_matched"] / s["t_true"]),
            "boundary_f1_multi": float(2 * s["t_matched"] / (s["p_pred"] + s["t_true"])),
            "cuts_per_clip": float(nat["cuts"].mean()), "cuts_per_min": float(nat["cuts"].sum() / (nat["n"].sum() / (60 * hz))),
            "same_class_cut_rate": float(s["false_split"] / s["same_join"]),
            "multi_sample_error": float(1 - g["correct"] / g["n"]) if "correct" in seg else None}


def load_systems(spec: dict) -> dict[tuple, dict]:
    """(encoder, method, decoding, seed) -> {half: tables}. Seedless systems (B0, B3, B7) use seed None."""
    out: dict[tuple, dict] = {}
    for enc, e in spec["encoders"].items():
        for method, runs in e.get("learned", {}).items():
            for seed, rd in runs.items():
                out[(enc, method, "default", int(seed))] = {h: read(Path(rd), f"rows_{h}") for h in HALVES}
        for pdir in e.get("posterior_runs", []):
            res = json.loads((Path(pdir) / "posterior_decoders.json").read_text())["results"]
            for row, r in res.items():
                method, _, seed = row.rpartition("_s")
                if method.startswith("J_") and method[2:] in e.get("learned", {}):   # joint runs name their rows J_<method>
                    method = method[2:]
                if method not in e.get("learned", {}) or not seed.isdigit():
                    continue
                for fam in ("b3",):
                    if fam in r["families"]:
                        out[(enc, method, fam, int(seed))] = {h: read(Path(pdir), f"rows_{row}_{fam}_{h}") for h in HALVES}
        if e.get("simple_run"):
            for name, method, dec in (("B0 argmax", "B0", "default"), ("B3 tuned (main)", "B0", "b3"),
                                      ("B1 tuned (main)", "B1", "tuned"), ("B2 tuned (main)", "B2", "tuned")):
                tag = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                out[(enc, method, dec, None)] = {h: read(Path(e["simple_run"]), f"rows_{tag}_{h}") for h in HALVES}
    for name, pdir in spec.get("systems", {}).items():   # e.g. B7, the official windowed EchoViewCLIP
        res = json.loads((Path(pdir) / "posterior_decoders.json").read_text())["results"]
        for fam, dec in (("argmax", "default"), ("b3", "b3")):
            if fam in res[name]["families"]:
                out[("system", name, dec, None)] = {h: read(Path(pdir), f"rows_{name}_{fam}_{h}") for h in HALVES}
    return out


def category(intervals: list[list[float]], n_seeds: int) -> str:
    if len(intervals) == n_seeds and all(lo > 0 for lo, _ in intervals):
        return "consistent (higher)"
    if len(intervals) == n_seeds and all(hi < 0 for _, hi in intervals):
        return "consistent (lower)"
    return "no consistent difference detected"


def contrast(systems: dict, new: tuple, ref: tuple) -> dict:
    """new/ref = (encoder, method, decoding). Seeded systems pair seed by seed (a seedless side joins every seed);
    two seedless systems are compared once."""
    seedless = (*new, None) in systems and (*ref, None) in systems
    keys = [None] if seedless else list(SEEDS)
    out = {}
    for seed in keys:
        a = systems.get((*new, seed), systems.get((*new, None))); b = systems.get((*ref, seed), systems.get((*ref, None)))
        if a is not None and b is not None:
            out["-" if seed is None else seed] = {h: paired_bootstrap(a[h], b[h]) for h in HALVES}
    cats = {f"{h}|{m}": category([out[k][h][m]["ci95"] for k in out], len(keys))
            for h in HALVES for m in ("multi segment F1@50", "native macro-F1", "multi boundary recall", "native cuts per clip")}
    return {"new": new, "ref": ref, "per_seed": out, "category": cats}


def gate_5b(systems: dict, oracle_e0: dict, oracle_e1: dict) -> dict:
    """Round-2 plan, Task 5b: start iff (A or B) and C; then the candidate rule."""
    nat = {(enc, h): metrics(systems[(enc, "B0", "default", None)][h])["native_macro_f1"] for enc in ("E0", "E1") for h in HALVES}
    a = all(nat[("E1", h)] - nat[("E0", h)] > EPS for h in HALVES)
    orc = {(enc, h): o[h] for enc, o in (("E0", oracle_e0), ("E1", oracle_e1)) for h in HALVES}
    b = all(orc[("E0", h)] - orc[("E1", h)] > EPS for h in HALVES)
    rec = {m: [metrics(systems[("E1", m, "default", s)]["calibration"])["boundary_recall_multi"] for s in SEEDS
               if ("E1", m, "default", s) in systems] for m in ("M1", "M3p", "M5")}
    c_by = {m: (len(v) == len(SEEDS) and min(v) >= 0.95) for m, v in rec.items()}
    sel = {m: [metrics(systems[("E1", m, "default", s)]["selection"])["segment_f1_50_multi"] for s in SEEDS
               if ("E1", m, "default", s) in systems] for m in ("M3p", "M5")}
    means = {m: float(np.mean(v)) if len(v) == len(SEEDS) else None for m, v in sel.items()}
    cand = None
    if None not in means.values():
        cand = "M5" if means["M5"] - means["M3p"] >= 0.005 else "M3p"
    start = (a or b) and any(c_by.values())
    return {"A_native_f1": {"E0": {h: nat[("E0", h)] for h in HALVES}, "E1": {h: nat[("E1", h)] for h in HALVES}, "holds": a},
            "B_oracle_segment_error": {"E0": {h: orc[("E0", h)] for h in HALVES}, "E1": {h: orc[("E1", h)] for h in HALVES}, "holds": b},
            "C_calibration_recall_by_seed": {"values": rec, "holds_by_method": c_by, "holds": any(c_by.values())},
            "start": bool(start), "candidate_selection_mean_segment_f1": means, "candidate": cand if start else None}


def fmt_row(label: str, m: dict) -> str:
    rec = " / ".join(f"{a}/{b}" for a, b in m["native_recall"].values())
    err = "" if m["multi_sample_error"] is None else f"{m['multi_sample_error']:.4f}"
    return (f"| {label} | {m['native_macro_f1']:.3f} | {rec} | {m['segment_f1_50_multi']:.3f} | {m['boundary_recall_multi']:.3f} | "
            f"{m['boundary_f1_multi']:.3f} | {m['cuts_per_clip']:.3f} | {m['cuts_per_min']:.2f} | {m['same_class_cut_rate']:.3f} | {err} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="JSON: encoders{E0,E1:{learned{method:{seed:run_dir}}, posterior_runs[], "
                                                   "simple_run, oracle_run}}, systems{B7: posterior_run}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); spec = json.loads(Path(a.spec).read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    systems = load_systems(spec)
    table = {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": {h: metrics(v[h]) for h in HALVES} for k, v in systems.items()}
    oracle = {}
    for enc in ("E0", "E1"):
        r = spec["encoders"][enc].get("oracle_run")
        if r:
            rows = json.loads((Path(r) / "oracle.json").read_text())["rows"]
            oracle[enc] = {h: next(x for x in rows[h] if x["method"].startswith("B0"))["oracle"]["segment_error_rate"] for h in HALVES}
    contrasts = {}
    for m in ("M1", "M3p", "M5"):
        for dec in ("default", "b3"):
            contrasts[f"E1-E0 {m} ({dec})"] = contrast(systems, ("E1", m, dec), ("E0", m, dec))
    contrasts["E1-E0 B0 (default)"] = contrast(systems, ("E1", "B0", "default"), ("E0", "B0", "default"))
    for enc in ("E0", "E1"):
        contrasts[f"{enc} M5-M3p (default)"] = contrast(systems, (enc, "M5", "default"), (enc, "M3p", "default"))
        for cand in ("M3p", "M5"):
            contrasts[f"{enc} {cand}-ASRF (default)"] = contrast(systems, (enc, cand, "default"), (enc, "M1", "default"))
            contrasts[f"{enc} {cand}-B3 tuned"] = contrast(systems, (enc, cand, "default"), (enc, "B0", "b3"))
            contrasts[f"{enc} {cand}-ASRF (both b3)"] = contrast(systems, (enc, cand, "b3"), (enc, "M1", "b3"))
    if "E1J" in spec["encoders"]:   # joint fine-tuning (Task 5b): J - E1 for each system, and the joint candidate against ASRF and B3
        for m in spec["encoders"]["E1J"]["learned"]:
            for dec in ("default", "b3"):
                contrasts[f"E1J-E1 {m} ({dec})"] = contrast(systems, ("E1J", m, dec), ("E1", m, dec))
        cand = [m for m in spec["encoders"]["E1J"]["learned"] if m != "M1"]
        for c in cand:
            contrasts[f"E1J {c}-ASRF (default)"] = contrast(systems, ("E1J", c, "default"), ("E1J", "M1", "default"))
            contrasts[f"E1J {c}-ASRF (both b3)"] = contrast(systems, ("E1J", c, "b3"), ("E1J", "M1", "b3"))
            contrasts[f"E1J {c}-B3 tuned (E1)"] = contrast(systems, ("E1J", c, "default"), ("E1", "B0", "b3"))
            contrasts[f"E1J {c}-B7 (default)"] = contrast(systems, ("E1J", c, "default"), ("system", "B7", "default"))
    # strongest previous configuration per half (fixed before the joint results were read): the highest three-seed mean
    # segment F1@50 among all non-joint systems (every head and decoding, tuned B1/B2/B3 on either input, B7 with and
    # without B3); the raw per-sample argmax is excluded
    strongest = {}
    for h in HALVES:
        means = {}
        for (enc, m, dec, seed), v in systems.items():
            if enc != "E1J" and not (m == "B0" and dec == "default"):   # raw per-sample argmax is not a candidate
                means.setdefault((enc, m, dec), []).append(metrics(v[h])["segment_f1_50_multi"])
        best = max(means, key=lambda k: float(np.mean(means[k])))
        strongest[h] = {"system": best, "mean_segment_f1_50": float(np.mean(means[best])), "n": len(means[best])}
        if "E1J" in spec["encoders"]:
            for c in [m for m in spec["encoders"]["E1J"]["learned"] if m != "M1"]:
                contrasts[f"E1J {c}-strongest previous ({h})"] = contrast(systems, ("E1J", c, "default"), best)
    gate = gate_5b(systems, oracle.get("E0", {}), oracle.get("E1", {})) if set(oracle) >= {"E0", "E1"} else None
    (out / "round2.json").write_text(json.dumps({"systems": table, "contrasts": contrasts, "gate_5b": gate, "oracle_b0": oracle,
                                                 "strongest_previous": strongest, "spec": spec}, indent=1, default=str))
    md = ["# Round 2 tables (generated by scripts/report_model_development_round2.py)", ""]
    for h in HALVES:
        md += [f"## {h} half", "", "| System (input, method, decoding, seed) | native macro-F1 | right/all PLAX / PSAX / A4C / A5C / SC4C | "
               "segment F1@50 | boundary recall | boundary F1 | cuts/clip | cuts/min | same-class cut rate | stream sample error |",
               "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for k in sorted(systems, key=lambda k: (k[0], k[1], k[2], -1 if k[3] is None else k[3])):
            md.append(fmt_row(f"{k[0]}, {k[1]}, {k[2]}, {'-' if k[3] is None else 's' + str(k[3])}", table[f'{k[0]}|{k[1]}|{k[2]}|{k[3]}'][h]))
        md.append("")
    md += ["## Paired contrasts per seed (95% bootstrap intervals)", ""]
    for label, c in contrasts.items():
        if not c["per_seed"]:
            continue
        md.append(f"### {label}")
        for h in HALVES:
            for m in ("multi segment F1@50", "native macro-F1", "multi boundary recall", "native cuts per clip"):
                vals = "; ".join(f"s{s} {c['per_seed'][s][h][m]['delta']:+.3f} [{c['per_seed'][s][h][m]['ci95'][0]:+.3f}, "
                                 f"{c['per_seed'][s][h][m]['ci95'][1]:+.3f}]" for s in c["per_seed"])
                md.append(f"- {h}, {m}: {vals} -> {c['category'][f'{h}|{m}']}")
        md.append("")
    if gate:
        md += ["## Task 5b start condition", "", "```", json.dumps(gate, indent=1), "```", ""]
    (out / "round2.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:60])); print(json.dumps(gate, indent=1) if gate else "gate: oracle runs missing")


if __name__ == "__main__":
    main()
