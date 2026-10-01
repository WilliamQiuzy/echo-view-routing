#!/usr/bin/env python
"""Paired permutation effects with resampling intervals for the 2026-10-01 round (tasks 1 and 2).

Every endpoint is a ratio of sums over resampling units, so one draw of units serves every method, condition and seed
(paired). The effect of a permuted condition is the mean over its seeds of (value - true-order value); the interval is
the 2.5/97.5 percentile of that effect over the draws, and the per-seed effects on the full set give the seed range.
Units: EV9V native clips by presumed patient group (file-name prefix), EV9V pair streams by stream, EV9V multi streams
by cine (they reuse cines), EchoNet-Pediatric everything by first-token group (the frozen packages).
Writes <out>/effects.json and <out>/effects.md (aggregates only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT, BOOT_SEED = 2000, 20261001
NATIVE = {  # name -> (numerator column, denominator column)
    "cuts per clip": ("cuts", "one"), "cuts per min": ("cuts", "minutes"), "deviation rate": ("dev_dec", "n")}
RAW = {"raw deviations in runs >= 5": ("dev_raw_in_runs_ge5", "dev_raw"), "raw error runs per clip": ("runs_raw", "one"),
       "lag-1 autocorrelation": ("lag1_sum", "lag1_n")}
FRAG_COLS = ["t_true", "t_matched", "p_pred", "same_join", "false_split"]
FRAG = {"boundary F1": lambda c: 2 * c[1] / (c[2] + c[0]), "boundary recall": lambda c: c[1] / c[0],
        "same-class cut rate": lambda c: c[4] / c[3]}
STREAM_COLS = ["seg_matched50", "seg_pred", "seg_true", "changes_away_from_joins", "minutes"]
STREAM = {"segment F1@50": lambda c: 2 * c[0] / (c[1] + c[2]), "cuts away from joins per min": lambda c: c[3] / c[4]}


def read(path: Path) -> pd.DataFrame:
    """CSV reader that keeps condition names as text (pandas would turn a chunk of "true" into booleans)."""
    return pd.read_csv(path, dtype={"condition": str, "test_condition": str}, low_memory=False)


def weights(n_units: int, rng: np.random.Generator) -> np.ndarray:
    draws = rng.integers(0, n_units, size=(N_BOOT, n_units))
    return np.stack([np.bincount(d, minlength=n_units) for d in draws]).astype(float)


def sums(df: pd.DataFrame, unit: str, cols: list[str], units: pd.Index) -> np.ndarray:
    return df.groupby(unit)[cols].sum().reindex(units, fill_value=0).to_numpy(dtype=float)


def effects(df: pd.DataFrame, unit: str, cols: list[str], funcs: dict, conditions: list[str], W: np.ndarray, units: pd.Index) -> dict:
    """Per endpoint: true-order value [CI] and, per condition, mean-over-seeds effect [CI] with the per-seed range."""
    by = {(c, s): sums(g, unit, cols, units) for (c, s), g in df.groupby(["condition", "seed"])}
    f = lambda m: {k: fn(np.moveaxis(m, -1, 0)) for k, fn in funcs.items()}  # noqa: E731
    true_key = next(k for k in by if k[0] == "true")
    t_pt, t_rep = f(by[true_key].sum(0)), f(W @ by[true_key])
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in funcs:
            e = {"true": {"value": float(t_pt[k]), "ci95": np.nanpercentile(t_rep[k], [2.5, 97.5]).tolist()}}
            for c in conditions:
                keys = sorted(x for x in by if x[0] == c)
                if not keys:
                    continue
                per_seed = [float(f(by[x].sum(0))[k] - t_pt[k]) for x in keys]
                rep = np.mean([f(W @ by[x])[k] - t_rep[k] for x in keys], axis=0)
                e[c] = {"effect": float(np.mean(per_seed)), "ci95": np.nanpercentile(rep, [2.5, 97.5]).tolist(),
                        "seed_min": min(per_seed), "seed_max": max(per_seed), "seeds": len(keys),
                        "value": float(t_pt[k] + np.mean(per_seed))}
            out[k] = e
    return out


def ratio_funcs(spec: dict) -> tuple[list[str], dict]:
    cols = sorted({c for pair in spec.values() for c in pair})
    return cols, {name: (lambda c, a=cols.index(num), b=cols.index(den): c[a] / c[b]) for name, (num, den) in spec.items()}


def prep_natives(n: pd.DataFrame) -> pd.DataFrame:
    n = n.copy(); n["one"] = 1.0; n["minutes"] = n["n"] / 600.0
    n["lag1_n"] = n["lag1"].notna().astype(float); n["lag1_sum"] = n["lag1"].fillna(0.0)
    return n


def dataset_block(natives: pd.DataFrame, frags: pd.DataFrame, frag_units: dict, methods: list[str], conditions: list[str],
                  streams: pd.DataFrame | None = None) -> dict:
    rng = np.random.default_rng(BOOT_SEED); out = {"natives": {}, "raw": {}, "banks": {}, "units": {}}
    nat = prep_natives(natives); units = pd.Index(sorted(nat["group"].unique())); W = weights(len(units), rng)
    out["units"]["natives"] = {"unit": "group", "n_units": len(units), "n_clips": int(nat["video_id"].nunique())}
    cols, funcs = ratio_funcs(NATIVE)
    for m in methods:
        out["natives"][m] = effects(nat[nat["method"] == m], "group", cols, funcs, conditions, W, units)
    cols, funcs = ratio_funcs(RAW)
    out["raw"] = effects(nat[nat["method"] == methods[0]], "group", cols, funcs, conditions, W, units)
    for bank, unit in frag_units.items():
        fb = frags[frags["bank"] == bank]
        u = pd.Index(sorted(fb[unit].unique())); Wb = weights(len(u), rng)
        out["units"][bank] = {"unit": unit, "n_units": len(u), "n_streams": int(fb["stream"].nunique())}
        out["banks"][bank] = {m: effects(fb[fb["method"] == m], unit, FRAG_COLS, FRAG, conditions, Wb, u) for m in methods}
    if streams is not None:
        s = streams.copy(); s["minutes"] = s["n"] / 600.0
        for bank in sorted(s["bank"].unique()):
            sb = s[s["bank"] == bank]; u = pd.Index(sorted(sb["group"].unique())); Wb = weights(len(u), rng)
            out["banks"][f"{bank} (streams)"] = {m: effects(sb[sb["method"] == m], "group", STREAM_COLS, STREAM, conditions, Wb, u)
                                                 for m in methods}
    return out


def fmt(e: dict, key: str) -> str:
    if key == "true":
        lo, hi = e["true"]["ci95"]
        return f"{e['true']['value']:.3f} [{lo:.3f}, {hi:.3f}]"
    if key not in e:
        return "—"
    x = e[key]; lo, hi = x["ci95"]
    rng = f"; seeds {x['seed_min']:+.3f}…{x['seed_max']:+.3f}" if x["seeds"] > 1 else ""
    return f"{x['effect']:+.3f} [{lo:+.3f}, {hi:+.3f}]{rng}"


def md_table(title: str, block: dict, conditions: list[str]) -> list[str]:
    lines = [f"#### {title}", "", "| method | endpoint | true order [95% CI] | " + " | ".join(f"{c} − true" for c in conditions) + " |",
             "|---|---|---|" + "---|" * len(conditions)]
    for m, eps in block.items():
        for k, e in eps.items():
            lines.append(f"| {m} | {k} | {fmt(e, 'true')} | " + " | ".join(fmt(e, c) for c in conditions) + " |")
    return lines + [""]


def run_hist(runs: pd.DataFrame, encoder: str) -> pd.DataFrame:
    """Share of deviating samples by error-run length (raw B0 or decoded), per condition, pooled over seeds."""
    r = runs[runs["encoder"] == encoder].copy()
    r["bin"] = pd.cut(r["length"], [0, 1, 2, 4, 9, 19, 10**6], labels=["1", "2", "3-4", "5-9", "10-19", ">=20"])
    r["samples"] = r["length"] * r["count"]
    t = r.groupby(["source", "condition", "bin"], observed=False)["samples"].sum().unstack("bin")
    return t.div(t.sum(axis=1), axis=0).round(3)


def frame_md(t: pd.DataFrame) -> str:
    """Markdown for a small frame with a (source, condition) index (no optional dependencies)."""
    head = "| source | condition | " + " | ".join(str(c) for c in t.columns) + " |"
    rows = [f"| {a} | {b} | " + " | ".join(f"{v:.3f}" for v in r) + " |" for (a, b), r in zip(t.index, t.to_numpy())]
    return "\n".join([head, "|---|---|" + "---|" * len(t.columns), *rows])


def main() -> None:
    ap = argparse.ArgumentParser(description="Permutation effects with group-aware intervals")
    ap.add_argument("--ev9v", required=True, help="runs/_demo/permutation_mechanism_r2")
    ap.add_argument("--pediatric", default=None, help="runs/_demo/pediatric/eval_v1"); ap.add_argument("--out", required=True)
    args = ap.parse_args(); ev = Path(args.ev9v); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res: dict = {"bootstrap": {"replicates": N_BOOT, "seed": BOOT_SEED, "interval": "percentile 95%"}}
    md = ["# Permutation effects (2026-10-01 round)", "",
          "Effect = permuted minus true order, averaged over the condition's seeds; 95% percentile bootstrap interval over "
          "resampling units (paired across methods, conditions and seeds); seed range = per-seed effects on the full set.", ""]
    nat = read(ev / "natives.csv"); frag = read(ev / "fragments.csv.gz"); met = read(ev / "metrics.csv")
    ev_conds = ["name", "reverse", "random", "block5", "block10", "block20"]
    for enc, methods, conds in (("b0", ["b0", "b1", "b2", "b3", "b3a"], ev_conds), ("echoprime", ["b2", "b3", "b3a"], ["random"])):
        blk = dataset_block(nat[nat["encoder"] == enc], frag[frag["encoder"] == enc], {"pairs_test": "stream", "multi_test": "video_id"},
                            methods, conds)
        m = met[met["encoder"] == enc]
        blk["thresholds_unique_per_method"] = {k: int(g["thresholds"].nunique()) for k, g in m.groupby("method")}
        blk["b0_clip_scores_unique"] = (m[m["method"] == "b0"][["clip macro-F1", "frame acc"]].round(12).drop_duplicates().to_dict("records")
                                        if "b0" in methods else None)
        res[f"ev9v/{enc}"] = blk
        md += [f"## EV9V, encoder {enc}", "", f"Units: {json.dumps(blk['units'])}", "",
               f"Validation-selected thresholds per method across all conditions (1 = frozen): {blk['thresholds_unique_per_method']}", ""]
        if blk["b0_clip_scores_unique"] is not None:
            md += [f"B0 clip macro-F1 / frame accuracy across all conditions: {blk['b0_clip_scores_unique']}", ""]
        md += md_table("Native clips (unit: presumed patient group)", blk["natives"], conds)
        md += md_table(f"Raw {enc} deviation structure (method-independent)", {enc: blk["raw"]}, conds)
        for bank, unit in (("pairs_test", "stream"), ("multi_test", "cine")):
            md += md_table(f"{bank} at 0.5 s (unit: {unit})", blk["banks"][bank], conds)
    runs = read(ev / "runs.csv")
    for enc in ("b0", "echoprime"):
        h = run_hist(runs, enc)
        res[f"ev9v/{enc}"]["run_length_share"] = {f"{a}|{b}": row.to_dict() for (a, b), row in h.iterrows()}
        md += [f"#### Share of deviating samples by error-run length ({enc}; pooled over seeds)", "", frame_md(h), ""]
    cov = json.loads((ev / "coverage.json").read_text())
    blocks = {k: v for k, v in cov.items() if k.startswith("b0/block")}
    agg = {}
    for k, v in blocks.items():
        cond = k.split("/")[1]; a = agg.setdefault(cond, [0, 0, 0, 0])
        a[0] += v["fragments_perturbed"]; a[1] += v["fragments"]; a[2] += v["natives_perturbed"]; a[3] += v["natives"]
    res["ev9v/block_coverage"] = {c: {"fragments_perturbed_share": a[0] / a[1], "natives_perturbed_share": a[2] / a[3]} for c, a in agg.items()}
    md += ["#### Block-permutation coverage (share of test fragments / native clips actually perturbed, over 10 seeds)", "",
           "| condition | fragments | native clips |", "|---|---|---|"]
    md += [f"| {c} | {v['fragments_perturbed_share']:.3f} | {v['natives_perturbed_share']:.3f} |" for c, v in sorted(res["ev9v/block_coverage"].items())]
    md.append("")
    if args.pediatric:
        pe = Path(args.pediatric)
        pn = read(pe / "natives.csv.gz"); pf = read(pe / "fragments.csv.gz"); ps = read(pe / "streams.csv.gz")
        blk = dataset_block(pn, pf, {"exam": "group", "same": "group"}, ["b0", "b1", "b2", "b3", "b3a", "b4"], ["random"], streams=ps)
        cls = json.loads((pe / "classification.json").read_text())
        pt = pn[(pn["condition"] == "true")].copy(); pt["correct"] = (pt["majority"] == pt["label"]).astype(float)
        blk["clip_majority_accuracy_true_order"] = pt.groupby("method")["correct"].mean().round(4).to_dict()
        blk["classification"] = cls
        res["pediatric"] = blk
        md += ["## EchoNet-Pediatric (frozen protocol pediatric-external-v1, test folds 2-9)", "",
               f"Classification transfer adequate under the frozen rule: **{cls['transfer_adequate']}** ({cls['rule']}).", "",
               f"Units: {json.dumps(blk['units'])}", ""]
        md += md_table("Native clips (unit: first-token group)", blk["natives"], ["random"])
        md += md_table("Raw B0 deviation structure", {"b0": blk["raw"]}, ["random"])
        for bank in ("exam", "same", "exam (streams)", "same (streams)"):
            md += md_table(f"{bank} bank (unit: first-token group)", blk["banks"][bank], ["random"])
    (out / "effects.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "effects.md").write_text("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
