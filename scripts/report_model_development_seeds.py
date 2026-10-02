#!/usr/bin/env python
"""Multi-seed summary of round 1 (2026-10-02 plan, Tasks 4-6).

Reads the queue status files of the seeds (learned rows with their default decoding), one posterior_decoders run that
covers the same runs (calibrated decoders on each head's posteriors, rows named <row>_s<seed>) and the simple-decoder
run. Writes seeds.md and seeds.json to --out:
- per row and decoder, the mean and range over seeds on both validation halves;
- per seed, paired differences with 95% bootstrap intervals (natives by session, streams by stream) for pre-declared
  contrasts: the structural ones under the same decoder (M4-M3, M4-M2, M3-M2) and M4 against M0, M1 and the tuned
  simple decoders;
- the seed category of each contrast: "consistent" when all intervals exclude zero on the same side, otherwise
  "no consistent difference detected" (not evidence of equivalence).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.report_model_development import paired_bootstrap, read_rows  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HALVES = ("calibration", "selection")
ROWS = ("M0", "M1", "M2", "M3", "M4", "M3p", "M5")
KEYS = ("native_macro_f1", "segment_f1_50_multi", "boundary_recall_multi", "boundary_f1_multi", "native_internal_cuts_per_clip",
        "same_class_cut_rate_multi")
SHORT = {"native_macro_f1": "native F1", "segment_f1_50_multi": "segF1@50", "boundary_recall_multi": "recall",
         "boundary_f1_multi": "bound. F1", "native_internal_cuts_per_clip": "cuts/clip", "same_class_cut_rate_multi": "same-class cut"}
# (label, new system, reference system); a system is (row, decoder) with decoder "default" or a posterior family
CONTRASTS = [
    ("path term, default decoding", ("M4", "default"), ("M3", "default")),
    ("path term + boundary loss, default decoding", ("M4", "default"), ("M2", "default")),
    ("boundary loss, default decoding", ("M3", "default"), ("M2", "default")),
    ("path term, B3 on posteriors", ("M4", "b3"), ("M3", "b3")),
    ("path term + boundary loss, B3 on posteriors", ("M4", "b3"), ("M2", "b3")),
    ("path term, boundary segments", ("M4", "bseg"), ("M3", "bseg")),
    ("M4 vs MS-TCN, default decoding", ("M4", "default"), ("M0", "default")),
    ("M4 vs ASRF, default decoding (ASRF official refinement)", ("M4", "default"), ("M1", "default")),
    ("M4 boundary segments vs ASRF official refinement", ("M4", "bseg"), ("M1", "default")),
    ("M4 boundary segments vs ASRF boundary segments", ("M4", "bseg"), ("M1", "bseg")),
    ("M4 boundary segments vs MS-TCN + B3", ("M4", "bseg"), ("M0", "b3")),
    ("M4 boundary segments vs B3 tuned (encoder)", ("M4", "bseg"), ("B3 tuned (main)", "simple")),
    ("M4 boundary segments vs B3 audited (encoder)", ("M4", "bseg"), ("B3 audited", "simple")),
    ("M4 boundary segments vs HMM tuned (encoder)", ("M4", "bseg"), ("B2 tuned (main)", "simple")),
    # follow-up rows (segment pooling). Added after the pooled runs had printed their own calibration summaries; they
    # pair M5 with every reference row of the matrix rather than a chosen subset.
    ("segment pooling, default decoding", ("M5", "default"), ("M4", "default")),
    ("path term with pooling, default decoding", ("M5", "default"), ("M3p", "default")),
    ("M5 vs ASRF, default decodings", ("M5", "default"), ("M1", "default")),
    ("M5 vs ASRF, both with boundary segments", ("M5", "bseg"), ("M1", "bseg")),
    ("M5 vs MS-TCN, default decoding", ("M5", "default"), ("M0", "default")),
    ("M5 vs B3 tuned (encoder)", ("M5", "default"), ("B3 tuned (main)", "simple")),
    ("M5 vs B3 audited (encoder)", ("M5", "default"), ("B3 audited", "simple")),
    ("M5 + B3 vs ASRF, default decoding", ("M5", "b3"), ("M1", "default")),
]
BOOT_METRICS = ("multi segment F1@50", "native macro-F1", "multi boundary recall", "native cuts per clip")


def tag(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", required=True, help="comma list of status files, one per seed")
    ap.add_argument("--posterior-run", required=True, help="comma list of posterior_decoders run directories")
    ap.add_argument("--simple-run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    post, post_dirs = {}, {}
    for d in args.posterior_run.split(","):
        for row, res in json.loads((Path(d) / "posterior_decoders.json").read_text())["results"].items():
            post[row] = res; post_dirs[row] = Path(d)
    simple_dir = Path(args.simple_run); simple = json.loads((simple_dir / "validation_metrics.json").read_text())["methods"]
    runs: dict[tuple[str, int], Path] = {}; inputs_used: set[tuple[str, str]] = set()
    for path in args.status.split(","):
        st = json.loads(Path(path).read_text())
        for row, v in st.items():
            if v.get("state") == "done" and row in ROWS:
                man = json.loads((Path(v["run_dir"]) / "run_manifest.json").read_text())
                inputs_used.add((man["protocol_sha256"], man["feature_manifest_sha256"]))
                runs[(row, int(man["seed"]))] = Path(v["run_dir"])
    if len(inputs_used) != 1:
        raise SystemExit(f"runs were built from different protocols or feature manifests: {sorted(inputs_used)}")
    seeds = sorted({s for _, s in runs})
    metrics, tables = {}, {}
    for (row, seed), d in runs.items():
        vm = json.loads((d / "validation_metrics.json").read_text())["method"]
        metrics[(row, "default", seed)] = vm; tables[(row, "default", seed)] = {h: read_rows(d, h) for h in HALVES}
        for fam, e in post.get(f"{row}_s{seed}", {}).get("families", {}).items():
            if fam == "argmax":
                continue
            metrics[(row, fam, seed)] = {h: e[h] for h in HALVES}
            tables[(row, fam, seed)] = {h: {t: pd.read_csv(post_dirs[f"{row}_s{seed}"] / f"rows_{row}_s{seed}_{fam}_{h}_{t}.csv",
                                                           dtype={"video_id": str, "stream": str})
                                            for t in ("native", "multi_seg", "multi_frag")} for h in HALVES}
    for name in ("B0 argmax", "B2 tuned (main)", "B3 tuned (main)", "B3 audited"):
        for seed in seeds:
            metrics[(name, "simple", seed)] = {h: simple[name][h] for h in HALVES}
            tables[(name, "simple", seed)] = {h: read_rows(simple_dir, h, tag(name)) for h in HALVES}
    systems = sorted({(r, dec) for r, dec, _ in metrics}, key=lambda x: (x[1] == "simple", x[0], x[1]))
    summary = {}
    md = ["# Round 1 across seeds (generated by scripts/report_model_development_seeds.py)", "",
          f"Seeds: {seeds}. Learned rows: same v3 features, data, budget and checkpoint rule; decoders on posteriors are "
          "calibrated per run on the calibration half. Mean [min, max] over seeds; simple decoders do not depend on the seed.", ""]
    for h in HALVES:
        md += [f"## {h} half", "", "| System | " + " | ".join(SHORT[k] for k in KEYS) + " |", "|---|" + "---:|" * len(KEYS)]
        for r, dec in systems:
            vals = {k: [metrics[(r, dec, s)][h][k] for s in seeds if (r, dec, s) in metrics] for k in KEYS}
            if not vals[KEYS[0]]:
                continue
            summary.setdefault(f"{r} | {dec}", {})[h] = {k: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}
                                                         for k, v in vals.items()}
            cell = lambda v: f"{np.mean(v):.3f} [{np.min(v):.3f}, {np.max(v):.3f}]" if len(v) > 1 else f"{v[0]:.3f}"
            md.append(f"| {r} ({dec}) | " + " | ".join(cell(vals[k]) for k in KEYS) + " |")
        md.append("")
    contrasts = {}
    md += ["## Paired contrasts per seed (95% bootstrap intervals)", ""]
    for label, a, b in CONTRASTS:
        per = {}
        for seed in seeds:
            ka, kb = (*a, seed), (*b, seed)
            if ka in tables and kb in tables:
                per[seed] = {h: paired_bootstrap(tables[ka][h], tables[kb][h]) for h in HALVES}
        if not per:
            continue
        contrasts[label] = {"new": a, "ref": b, "per_seed": per, "category": {}}
        md.append(f"### {label}: {a[0]} ({a[1]}) minus {b[0]} ({b[1]})")
        for h in HALVES:
            for m in BOOT_METRICS:
                ivs = [per[s][h][m] for s in per]
                lo_pos = all(iv["ci95"][0] > 0 for iv in ivs); hi_neg = all(iv["ci95"][1] < 0 for iv in ivs)
                cat = "consistent (higher)" if lo_pos and len(ivs) == len(seeds) else (
                    "consistent (lower)" if hi_neg and len(ivs) == len(seeds) else "no consistent difference detected")
                contrasts[label]["category"][f"{h}|{m}"] = cat
                md.append(f"- {h}, {m}: " + "; ".join(f"s{s} {per[s][h][m]['delta']:+.3f} [{per[s][h][m]['ci95'][0]:+.3f}, "
                                                       f"{per[s][h][m]['ci95'][1]:+.3f}]" for s in per) + f" -> {cat}")
        md.append("")
    (out / "seeds.json").write_text(json.dumps({"seeds": seeds, "summary": summary, "contrasts": contrasts,
                                                "runs": {f"{r}_s{s}": str(d) for (r, s), d in runs.items()}}, indent=1, default=str))
    (out / "seeds.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
