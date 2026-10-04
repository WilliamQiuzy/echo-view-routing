#!/usr/bin/env python
"""Geometry-pretraining round report (configs/experiments/geometry_pretraining_v1.yaml, evaluation, interpretation and
terms). Arms G0 (E1 as it is: the round-2 probe, B3 run and ASRF seeds), G1 and G2 (three pretraining seeds each).

- Stage A, classification, from each arm's probe: A4C/A5C separation (AUC of the margin logit(A5C) - logit(A4C) over
  the cached samples of native A4C and A5C clips; the AUC of clip-mean margins beside it), A4C and A5C recall (clip
  majority), native macro-F1 and cuts per clip, all from the B0 argmax rows and the formal cache. Paired by session
  bootstrap per seed.
- Stage B, segmentation: ASRF (M1) and B3 tuned. Paired bootstrap per seed (report_model_development.paired_bootstrap),
  G2 - G1, G2 - G0, G1 - G0; every G1/G2 system against B3 or ASRF of G0 and the strongest previous configuration;
  floors, boundary recall and the final goal.
- The screening rules and terms of the config, mechanism evidence (tilt error against the zero predictor and the
  contrast baseline, embedding change by pair kind) and training records.
--self-test uses G0's runs for G1 and G2: every difference must be zero. Writes report.json and report.md into --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.model_development_protocol import session_of  # noqa: E402
from scripts.report_model_development import BOOT_SEED, N_BOOT, metric_fns, paired_bootstrap  # noqa: E402
from scripts.report_model_development_round2 import category, metrics, read  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from echo_routing.features.cache import load_video_features, variant_dir  # noqa: E402

HALVES, SEEDS, SYSTEMS = ("selection", "calibration"), (0, 1, 2), ("ASRF", "B3")
A4C, A5C = 2, 3
SEP, CLIP_SEP, SEG = "A4C/A5C AUC (samples)", "A4C/A5C AUC (clip mean)", "multi segment F1@50"
STAGE_A = (SEP, CLIP_SEP, "A4C recall", "A5C recall", "native macro-F1", "native cuts per clip")
STAGE_B = (SEG, "multi boundary recall", "native macro-F1", "native cuts per clip")
PAIRS = (("G2", "G1"), ("G2", "G0"), ("G1", "G0"))
FLOORS = {"selection": 0.973, "calibration": 0.980}
RECALL_MIN, MARGIN, SHORTCUT_RATIO = 0.95, 0.01, 1.1


def arm_inputs(spec: dict, geo: Path, self_test: bool) -> dict:
    """arm -> seed -> manifest, asrf and simple run dirs. G0 is E1 of rounds 2-5: one probe and B3 run, ASRF seed k."""
    e1 = spec["encoders"]["E1"]
    g0 = {k: {"manifest": e1["feature_manifest"], "asrf": e1["asrf"][str(k)], "simple": e1["simple_run"]} for k in SEEDS}
    if self_test:
        return {"G0": g0, "G1": g0, "G2": g0}
    mans = dict(line.split() for line in (geo / "manifests.txt").read_text().splitlines() if line.strip())
    out = {"G0": g0}
    for arm in ("G1", "G2"):
        out[arm] = {}
        for k in SEEDS:
            st = json.loads((geo / "downstream" / f"{arm}_s{k}_status.json").read_text())
            if any(st[j]["state"] != "done" for j in ("M1", "simple")):
                raise SystemExit(f"{arm}_s{k}: downstream jobs not done")
            out[arm][k] = {"manifest": mans[f"{arm}_s{k}"], "asrf": st["M1"]["run_dir"], "simple": st["simple"]["run_dir"]}
    return out


def auc(score: np.ndarray, is_a5c: np.ndarray) -> float:
    pos, neg = score[is_a5c], score[~is_a5c]
    if not len(pos) or not len(neg):
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def native_with_margin(run: dict, half: str) -> pd.DataFrame:
    """B0 argmax native rows of the half plus each A4C/A5C clip's margins over its cached samples and their mean."""
    nat = read(Path(run["simple"]), f"rows_b0_argmax_{half}")["native"]
    man = json.loads(Path(run["manifest"]).read_text())
    d = variant_dir(Path(man["cache_root"]), man["encoder"]["cache_key"], "orig"); margins = []
    for vid, label, n in zip(nat["video_id"], nat["label"], nat["n"]):
        if label not in (A4C, A5C):
            margins.append(np.zeros(0)); continue
        lg = load_video_features(d / f"{vid}.npz").logit
        if len(lg) != n:
            raise SystemExit(f"{vid}: {len(lg)} cached samples against {n} in the B0 rows")
        margins.append((lg[:, A5C] - lg[:, A4C]).astype(np.float64))
    return nat.assign(margins=margins, session=nat["video_id"].map(session_of)).set_index("video_id")


def stage_a(nat: pd.DataFrame) -> dict:
    lab, maj = nat["label"].to_numpy(), nat["majority"].to_numpy(); ac = np.isin(lab, (A4C, A5C))
    ms = nat["margins"].to_numpy()[ac]; a5c = lab[ac] == A5C
    return {SEP: auc(np.concatenate(ms), np.repeat(a5c, [len(m) for m in ms])), CLIP_SEP: auc(np.array([m.mean() for m in ms]), a5c),
            "A4C recall": float((maj[lab == A4C] == A4C).mean()), "A5C recall": float((maj[lab == A5C] == A5C).mean()),
            "native macro-F1": metric_fns()["native macro-F1"][1](nat), "native cuts per clip": float(nat["cuts"].mean())}


def stage_a_bootstrap(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """a - b on the same native clips, resampled by session (the draws of paired_bootstrap)."""
    if set(a.index) != set(b.index):
        raise ValueError("stage A: the two arms cover different native clips")
    b = b.loc[a.index]; rng = np.random.default_rng(BOOT_SEED)
    sessions = a["session"].unique(); rows = {s: np.flatnonzero(a["session"].to_numpy() == s) for s in sessions}
    pa, pb = stage_a(a), stage_a(b); draws = {k: [] for k in STAGE_A}
    for _ in range(N_BOOT):
        ni = np.concatenate([rows[s] for s in rng.choice(sessions, size=sessions.size, replace=True)])
        da, db = stage_a(a.iloc[ni]), stage_a(b.iloc[ni])
        for k in STAGE_A:
            draws[k].append(da[k] - db[k])
    return {k: {"delta": pa[k] - pb[k], "ci95": [float(np.nanpercentile(draws[k], 2.5)), float(np.nanpercentile(draws[k], 97.5))]} for k in STAGE_A}


def summarise(per_seed: dict, names: tuple) -> dict:
    return {"per_seed": per_seed, "category": {m: category([per_seed[k][m]["ci95"] for k in per_seed], len(SEEDS)) for m in names}}


def screening(ca: dict, cb: dict) -> dict:
    """Interpretation rules 1-4 and the terms of the config."""
    cat_a = lambda new, ref, h, m: ca[f"{new} - {ref}|{h}"]["category"][m]  # noqa: E731
    cat_b = lambda s, new, ref, h, m=SEG: cb[f"{s} {new} - {ref}|{h}"]["category"][m]  # noqa: E731
    r1 = not any(cat_a("G2", ref, h, m) == "consistent (lower)" for ref in ("G0", "G1") for h in HALVES for m in (SEP, "A5C recall"))
    sep = [cat_a("G2", "G1", h, SEP) for h in HALVES]
    r2 = "consistent (higher)" in sep and "consistent (lower)" not in sep
    r3 = not any(cat_a("G2", "G1", h, "native cuts per clip") == "consistent (higher)" for h in HALVES)
    seg = [cat_b(s, "G2", "G1", h) for s in SYSTEMS for h in HALVES]
    r4 = "consistent (lower)" not in seg and "consistent (higher)" in seg
    better = {}
    for arm in ("G1", "G2"):
        gains = [cat_a(arm, "G0", h, SEP) for h in HALVES] + [cat_b(s, arm, "G0", h) for s in SYSTEMS for h in HALVES]
        losses = [cat_a(arm, "G0", h, m) for h in HALVES for m in (SEP, "A5C recall")] + [cat_b(s, arm, "G0", h) for s in SYSTEMS for h in HALVES]
        better[arm] = "consistent (higher)" in gains and "consistent (lower)" not in losses
    adds_nothing = "consistent (higher)" not in sep + seg
    loses_sep = {arm: any(cat_a(arm, "G0", h, SEP) == "consistent (lower)" for h in HALVES) for arm in ("G1", "G2")}
    return {"rule_1_no_loss_in_separation_or_A5C_recall": r1, "rule_2_separation_gain_over_G1": r2,
            "rule_3_no_increase_in_false_changes": r3, "rule_4_stage_b_gain_over_G1": r4, "geometry_helps": r1 and r2 and r3 and r4,
            "better_than_G0": better, "g2_adds_nothing": adds_nothing, "loses_separation_against_G0": loses_sep}


def goal(tables: dict, strongest_value: dict) -> dict:
    """Final goal per half: three-seed mean segment F1@50 >= strongest + 1 point, recall and native floor in every seed
    and half, as in round 6. The config's comment on boundary_recall_min names the calibration half, where the jobs
    apply it to operating points; the verdict with recall checked on the calibration half only is kept beside it."""
    out = {}
    for h in HALVES:
        ms = [metrics(tables[k][h]) for k in SEEDS]; mean = float(np.mean([m["segment_f1_50_multi"] for m in ms]))
        rec, nat = min(m["boundary_recall_multi"] for m in ms), min(m["native_macro_f1"] for m in ms)
        gain, rec_ok = mean >= strongest_value[h] + MARGIN, rec >= RECALL_MIN
        out[h] = {"mean_segment_f1_50": mean, "strongest": strongest_value[h], "min_recall": rec, "min_native_f1": nat,
                  "floor": FLOORS[h], "thresholds_met": bool(rec_ok and nat >= FLOORS[h]),
                  "passes": bool(gain and rec_ok and nat >= FLOORS[h]),
                  "passes_recall_on_calibration_only": bool(gain and (rec_ok or h == "selection") and nat >= FLOORS[h])}
    return {**out, "passes_both": out["selection"]["passes"] and out["calibration"]["passes"],
            "passes_both_recall_on_calibration_only": all(out[h]["passes_recall_on_calibration_only"] for h in HALVES)}


def mechanism(geo: Path, baseline: dict | None, self_test: bool) -> dict:
    if self_test:
        return {}
    out = {f"{arm}_s{k}": json.loads((geo / f"{arm}_s{k}" / "mechanism.json").read_text()) for arm in ("G1", "G2") for k in SEEDS}
    warn = None
    if baseline is not None:
        b = baseline["final"]["signed"]["all"]["mae_deg"]
        warn = any(b <= SHORTCUT_RATIO * out[f"G2_s{k}"]["tilt_mae_deg"] for k in SEEDS)
    return {"runs": out, "contrast_baseline": baseline, "shortcut_warning": warn}


def training(geo: Path, self_test: bool) -> dict:
    if self_test:
        return {}
    out = {}
    for arm in ("G1", "G2"):
        for k in SEEDS:
            d = geo / f"{arm}_s{k}"; log = [json.loads(x) for x in (d / "train_log.jsonl").read_text().splitlines()]
            out[f"{arm}_s{k}"] = {"first": log[0], "last": log[-1], "minutes": round(log[-1]["seconds"] / 60, 1)}
    for k in SEEDS:
        out[f"batches_identical_s{k}"] = (geo / f"G1_s{k}" / "batch_digests.txt").read_text() == (geo / f"G2_s{k}" / "batch_digests.txt").read_text()
    return out


def reading(s: dict, goals: dict, mech: dict) -> list[str]:
    lines = []
    if s["geometry_helps"]:
        lines.append("G2 passes the four screening rules: the tilt supervision shows a research signal worth further testing; it is not yet the final goal")
    else:
        lines.append("G2 does not pass the four screening rules")
    if s["better_than_G0"]["G1"] and s["g2_adds_nothing"]:
        lines.append("G1 is better than G0 while G2 adds nothing over G1: the slice data help, the tilt supervision does not")
    if not s["better_than_G0"]["G1"] and not s["better_than_G0"]["G2"]:
        lines.append("neither G1 nor G2 is better than G0: this direction stops at this data scale")
    for arm, lost in s["loses_separation_against_G0"].items():
        if lost:
            lines.append(f"{arm} loses A4C/A5C separation against G0 consistently: not a success, whatever else improves")
    met = [f"{arm} {sys}" for arm in ("G1", "G2") for sys in SYSTEMS if goals[f"{arm} {sys}"]["passes_both"]]
    lines.append("final goal met by: " + ", ".join(met) if met else "no G1 or G2 system meets the final goal on both halves")
    differ = [f"{arm} {sys}" for arm in ("G1", "G2") for sys in SYSTEMS
              if goals[f"{arm} {sys}"]["passes_both"] != goals[f"{arm} {sys}"]["passes_both_recall_on_calibration_only"]]
    lines.append("with boundary recall checked on the calibration half only, the final-goal verdict changes for: " + ", ".join(differ)
                 if differ else "checking boundary recall on the calibration half only changes no final-goal verdict")
    if mech.get("shortcut_warning"):
        lines.append("shortcut warning: the contrast baseline on the final input predicts d_alpha about as well as G2's head")
    return lines


def ci(v: dict) -> str:
    return f"{v['delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"


def markdown(r: dict) -> list[str]:
    md = ["# Geometry pretraining round (generated by scripts/geometry_round_report.py)", ""]
    if r["self_test"]:
        md += ["**Self-test: G1 and G2 are G0's runs; every difference must be zero.**", ""]
    md += ["## Stage A: classification (per seed s0 / s1 / s2)", "", "| arm | half | " + " | ".join(STAGE_A) + " |", "|---|---|" + "---|" * len(STAGE_A)]
    for arm in ("G0", "G1", "G2"):
        for h in HALVES:
            md.append(f"| {arm} | {h} | " + " | ".join(" / ".join(f"{r['stage_a'][arm][k][h][m]:.4f}" for k in SEEDS) for m in STAGE_A) + " |")
    md += ["", "Paired differences (per seed, 95% session-bootstrap intervals):", ""]
    for key, c in r["contrasts_a"].items():
        for m in STAGE_A:
            md.append(f"- {key.replace('|', ', ')}, {m}: " + "; ".join(f"s{k} {ci(c['per_seed'][k][m])}" for k in SEEDS) + f" -> {c['category'][m]}")
    md += ["", "## Stage B: segmentation (per seed s0 / s1 / s2)", "", "| arm | system | half | segment F1@50 | boundary recall | native macro-F1 | cuts/clip |", "|---|---|---|---|---|---|---|"]
    for arm in ("G0", "G1", "G2"):
        for sys in SYSTEMS:
            for h in HALVES:
                ms = r["stage_b"][f"{arm} {sys}"][h]
                md.append(f"| {arm} | {sys} | {h} | " + " | ".join(" / ".join(f"{m[x]:.4f}" for m in ms) for x in
                                                                ("segment_f1_50_multi", "boundary_recall_multi", "native_macro_f1", "cuts_per_clip")) + " |")
    md += ["", "Paired differences (per seed, 95% bootstrap intervals; native clips by session, streams by stream):", ""]
    for key, c in r["contrasts_b"].items():
        for m in STAGE_B:
            md.append(f"- {key.replace('|', ', ')}, {m}: " + "; ".join(f"s{k} {ci(c['per_seed'][k][m])}" for k in SEEDS) + f" -> {c['category'][m]}")
    md += ["", "## Floors, boundary recall and the final goal", "", f"Floors {FLOORS['selection']}/{FLOORS['calibration']}; recall >= {RECALL_MIN}; "
           f"strongest previous {r['strongest']['selection']:.4f}/{r['strongest']['calibration']:.4f}, target + {MARGIN}.", "",
           "| arm and system | half | mean segment F1@50 (strongest) | min recall | min native F1 (floor) | thresholds met | final goal |", "|---|---|---|---|---|---|---|"]
    for name, g in r["goals"].items():
        for h in HALVES:
            o = g[h]
            md.append(f"| {name} | {h} | {o['mean_segment_f1_50']:.4f} ({o['strongest']:.4f}) | {o['min_recall']:.4f} | {o['min_native_f1']:.4f} ({o['floor']:.3f}) | "
                      f"{'yes' if o['thresholds_met'] else 'no'} | {'yes' if o['passes'] else 'no'} |")
    md += ["", "## Screening rules and terms (configs/experiments/geometry_pretraining_v1.yaml)", "", "```", json.dumps(r["screening"], indent=1), "```", ""]
    md += ["## Reading", ""] + [f"- {x}" for x in r["reading"]]
    if r["mechanism"]:
        md += ["", "## Mechanism evidence (held-out people; not a success criterion)", "",
               "| run | tilt MAE (deg) | zero predictor | moving planes | cosine distance same_plane / same_time / both |", "|---|---|---|---|---|"]
        for run, m in r["mechanism"]["runs"].items():
            e = m["embedding_change_by_kind"]
            md.append(f"| {run} | {m['tilt_mae_deg']:.3f} | {m['zero_predictor_mae_deg']:.3f} | {m['tilt_mae_deg_moving_plane']:.3f} | "
                      f"{e['same_plane']['mean_cosine_distance']:.3f} / {e['same_time']['mean_cosine_distance']:.3f} / {e['both']['mean_cosine_distance']:.3f} |")
        b = r["mechanism"]["contrast_baseline"]
        if b:
            md += ["", f"Contrast baseline (signed, all pairs): raw {b['raw']['signed']['all']['mae_deg']:.3f} deg, final input "
                   f"{b['final']['signed']['all']['mae_deg']:.3f} deg; zero predictor {b['zero_predictor']['all']['mae_deg']:.3f} deg. "
                   f"Shortcut warning: {r['mechanism']['shortcut_warning']}."]
    if r["training"]:
        md += ["", "## Training", ""] + [f"- {k}: {v}" for k, v in r["training"].items() if k.startswith("batches")] + \
              [f"- {k}: InfoNCE {v['first']['infonce']:.3f} -> {v['last']['infonce']:.3f}, tilt {v['first']['tilt']:.3f} -> {v['last']['tilt']:.3f}, "
               f"{v['minutes']} min" for k, v in r["training"].items() if not k.startswith("batches")]
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default="runs/md_round4/spec.json", help="E0/E1 runs and the B7 run of rounds 2-5")
    ap.add_argument("--geo", default="runs/geometry"); ap.add_argument("--baseline", default="runs/geometry/contrast_baseline.json")
    ap.add_argument("--config", default="configs/experiments/geometry_pretraining_v1.yaml")
    ap.add_argument("--out", required=True); ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); geo = Path(a.geo)
    cfg = yaml.safe_load(Path(a.config).read_text())["evaluation"]["stage_b_segmentation"]
    if cfg["native_f1_floors"] != FLOORS or cfg["boundary_recall_min"] != RECALL_MIN:
        raise SystemExit("floors or recall threshold differ from the config")
    spec = json.loads(Path(a.spec).read_text()); runs = arm_inputs(spec, geo, a.self_test); memo = {}
    for arm in runs:   # G0's probe rows are shared by its seeds
        for k in SEEDS:
            for h in HALVES:
                key = (runs[arm][k]["manifest"], runs[arm][k]["simple"], h)
                if key not in memo:
                    memo[key] = native_with_margin(runs[arm][k], h)
    nat = {arm: {k: {h: memo[(runs[arm][k]["manifest"], runs[arm][k]["simple"], h)] for h in HALVES} for k in SEEDS} for arm in runs}
    stage_a_pts = {arm: {k: {h: stage_a(nat[arm][k][h]) for h in HALVES} for k in SEEDS} for arm in runs}
    ca = {f"{new} - {ref}|{h}": summarise({k: stage_a_bootstrap(nat[new][k][h], nat[ref][k][h]) for k in SEEDS}, STAGE_A)
          for new, ref in PAIRS for h in HALVES}
    tables = {f"{arm} {sys}": {k: {h: read(Path(runs[arm][k]["asrf" if sys == "ASRF" else "simple"]),
                                           f"rows_{h}" if sys == "ASRF" else f"rows_b3_tuned_main_{h}") for h in HALVES} for k in SEEDS}
              for arm in runs for sys in SYSTEMS}
    strongest_t = {"selection": read(Path(spec["encoders"]["E0"]["simple_run"]), "rows_b3_tuned_main_selection"),
                   "calibration": read(Path(spec["b7_run"]), "rows_B7_b3_calibration")}
    strongest = {h: metrics(t)["segment_f1_50_multi"] for h, t in strongest_t.items()}
    cb = {}
    for sys in SYSTEMS:
        for new, ref in PAIRS:
            for h in HALVES:
                cb[f"{sys} {new} - {ref}|{h}"] = summarise({k: paired_bootstrap(tables[f"{new} {sys}"][k][h], tables[f"{ref} {sys}"][k][h]) for k in SEEDS}, STAGE_B)
    for arm in ("G1", "G2"):
        for sys, other in (("ASRF", "B3"), ("B3", "ASRF")):
            for h in HALVES:
                cb[f"{sys} {arm} - {other} G0|{h}"] = summarise({k: paired_bootstrap(tables[f"{arm} {sys}"][k][h], tables[f"G0 {other}"][k][h]) for k in SEEDS}, STAGE_B)
                cb[f"{sys} {arm} - strongest previous|{h}"] = summarise({k: paired_bootstrap(tables[f"{arm} {sys}"][k][h], strongest_t[h]) for k in SEEDS}, STAGE_B)
    goals = {name: goal(t, strongest) for name, t in tables.items()}
    s = screening(ca, cb)
    baseline = json.loads(Path(a.baseline).read_text()) if Path(a.baseline).exists() else None
    mech = mechanism(geo, baseline, a.self_test)
    report = {"self_test": a.self_test, "runs": runs, "stage_a": stage_a_pts, "contrasts_a": ca,
              "stage_b": {name: {h: [metrics(t[k][h]) for k in SEEDS] for h in HALVES} for name, t in tables.items()},
              "contrasts_b": cb, "strongest": strongest, "goals": goals, "screening": s, "mechanism": mech,
              "training": training(geo, a.self_test), "reading": reading(s, goals, mech)}
    if a.self_test:   # arm contrasts compare a run with itself
        arm_keys = list(ca) + [f"{sys} {new} - {ref}|{h}" for sys in SYSTEMS for new, ref in PAIRS for h in HALVES]
        bad = sorted({key for key in arm_keys for x in {**ca, **cb}[key]["per_seed"].values() for m in x.values()
                      if not (m["delta"] == 0 and m["ci95"] == [0.0, 0.0])})
        report["self_test_nonzero"] = bad
        print("self-test: contrasts with a nonzero difference:", bad)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    md = markdown(report); (out / "report.md").write_text("\n".join(md) + "\n"); print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
