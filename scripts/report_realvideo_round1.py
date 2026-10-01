#!/usr/bin/env python
"""Round-1 report: MIMIC ECG and image-cycle validation, EV9V cycle feasibility, MIMIC screening, preliminary phase
correspondence. Reads the outputs of mimic_round1.py, ev9v_cycles.py, realvideo_probs.py and realvideo_screen.py in
<dir>; writes <dir>/round1_report.json and <dir>/ecg_check.png (stays local: MIMIC data use agreement).

Image-cycle reliability thresholds are chosen on MIMIC, where the drawn ECG is the reference, as the least strict
(peak, dip) pair whose period agrees with the ECG R-R interval within 10% in >= 95% of reliable loops; they are then
applied unchanged to EV9V. Phase correspondence is reported only where the phase is reliable and is descriptive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from echo_routing.realvideo.cycles import phase_from_boundaries, phase_locking  # noqa: E402
from echo_routing.realvideo.ecg import _cursor_mask  # noqa: E402
from echo_routing.realvideo.screening import clip_scores, cohen_kappa, runs  # noqa: E402

CLASSES = {"B0": ["PLAX", "PSAX", "A4C", "A5C", "SC4C"], "EchoPrime": ["PLAX", "PSAX", "A4C", "A5C", "SC4C", "other"]}
PEAKS, DIPS = [0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 0.2, 0.4, 0.6]
AGREE, TARGET = 0.10, 0.95


def ratio_class(r: float) -> str:
    if not np.isfinite(r):
        return "none"
    return "agree" if abs(r - 1) <= AGREE else "double" if abs(r - 2) <= 2 * AGREE else "half" if abs(r - 0.5) <= AGREE / 2 else "other"


def mimic_tables(d: Path) -> pd.DataFrame:
    cines = pd.read_csv(d / "mimic_cines.csv")
    hdr = pd.read_csv(d / "mimic_headers.csv")
    hdr["key"] = hdr["study_id"].astype(str) + "_" + hdr["path"].map(lambda p: Path(p).stem)
    m = cines.merge(hdr[["key", "subject_id", "study_id", "hr_tag", "path"]], on="key", how="left")
    m["hr_ecg"] = 60 / m["ecg_rr_s"]
    m["ecg_tag_agree"] = (m["hr_ecg"] - m["hr_tag"]).abs() <= 0.10 * m["hr_tag"]
    m["img_vs_ecg"] = m["img_period_s"] / m["ecg_rr_s"]
    m["img_class"] = m["img_vs_ecg"].map(ratio_class)
    return m


def threshold_grid(m: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ref = m[(m["ecg_ok"] == True) & (m["ecg_n_r"] >= 2) & m["ecg_tag_agree"]]  # noqa: E712
    rows = []
    for pk in PEAKS:
        for dp in DIPS:
            rel = ref[(ref["img_peak"] >= pk) & (ref["img_dip"] >= dp)]
            rows.append({"peak": pk, "dip": dp, "coverage": len(rel) / len(ref), "n": len(rel),
                         "agree": float((rel["img_class"] == "agree").mean()) if len(rel) else np.nan,
                         "double": float((rel["img_class"] == "double").mean()) if len(rel) else np.nan,
                         "half": float((rel["img_class"] == "half").mean()) if len(rel) else np.nan})
    g = pd.DataFrame(rows)
    ok = g[g["agree"] >= TARGET].sort_values(["coverage", "peak"], ascending=[False, True])
    best = ok.iloc[0].to_dict() if len(ok) else g.sort_values("agree", ascending=False).iloc[0].to_dict()
    return g, {"n_reference_loops": len(ref), "chosen": best, "target_met": bool(len(ok))}


def screen_mimic(d: Path, m: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for r in m[m["error"].isna()].itertuples(index=False):
        z = np.load(d / "mimic" / f"{r.key}.npz")
        step = int(z["idx"][1] - z["idx"][0]) if len(z["idx"]) > 1 else 1
        hz = 1.0 / (step * float(z["frame_time_s"]))
        for rec, key in (("B0", "prob_b0"), ("EchoPrime", "prob_ep")):
            s = clip_scores(z[key], CLASSES[rec], hz)
            rows.append({"key": r.key, "subject_id": r.subject_id, "recognizer": rec, "n_samples": len(z[key]), **s})
    df = pd.DataFrame(rows)
    w = df.pivot(index="key", columns="recognizer", values="stratum")
    rep, sus = ("repeated", "mixed"), ("sustained", "mixed")
    out = {"loops": int(w.shape[0]), "patients": int(df["subject_id"].nunique()),
           "strata": {rec: df[df.recognizer == rec]["stratum"].value_counts().to_dict() for rec in CLASSES},
           "repeated_either": int((w["B0"].isin(rep) | w["EchoPrime"].isin(rep)).sum()),
           "repeated_both": int((w["B0"].isin(rep) & w["EchoPrime"].isin(rep)).sum()),
           "sustained_either": int((w["B0"].isin(sus) | w["EchoPrime"].isin(sus)).sum()),
           "sustained_both": int((w["B0"].isin(sus) & w["EchoPrime"].isin(sus)).sum()),
           "echoprime_dominant": df[df.recognizer == "EchoPrime"]["dominant"].value_counts().to_dict(),
           "duration_s_median": float(m["n_frames"].mul(m["frame_time_ms"]).div(1000).median())}
    ok = (w["B0"] != "short") & (w["EchoPrime"] != "short")
    out["kappa_strata_nonshort"] = cohen_kappa(list(w.loc[ok, "B0"]), list(w.loc[ok, "EchoPrime"])) if ok.sum() else None
    return df, out


def interior_window(off: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Drop off-runs that touch either end (ECG-gated loops start and end on R, which would fake locking)."""
    rr = runs(off); n = len(off)
    a = rr[0][1] if rr and rr[0][0] == 0 else 0
    b = rr[-1][0] if rr and rr[-1][1] == n else n
    m = np.zeros(n, dtype=bool)
    for s, e in rr:
        if s > 0 and e < n:
            m[s:e] = True
    return a, b, m


def locking_summary(items: list[tuple[str, np.ndarray, np.ndarray]], n_perm: int) -> dict:
    """items: (stratum, phase per sample, probability matrix); off = top-1 differs from the clip's dominant class.

    "all": every off sample. "interior": only off-runs that start after the first and end before the last sample,
    tested within the span between the boundary runs.
    """
    rows = []
    for k, (stratum, ph, p) in enumerate(items):
        top = p.argmax(1); dom = np.bincount(top).argmax(); off = top != dom
        r, pv = phase_locking(ph, off, n_perm=n_perm, seed=k)
        a, b, m = interior_window(off)
        ri, pi = phase_locking(ph[a:b], m[a:b], n_perm=n_perm, seed=k) if b - a > 2 else (np.nan, np.nan)
        rows.append({"stratum": stratum, "R": r, "p": pv, "R_int": ri, "p_int": pi})
    t = pd.DataFrame(rows)
    if t.empty:
        return {}
    out = {}
    for s, g in t.groupby("stratum"):
        ga, gi = g[g["p"].notna()], g[g["p_int"].notna()]
        out[s] = {"all": {"n": int(len(ga)), "median_R": round(float(ga["R"].median()), 3) if len(ga) else None,
                          "frac_p_lt_0.05": round(float((ga["p"] < 0.05).mean()), 3) if len(ga) else None},
                  "interior": {"n": int(len(gi)), "median_R": round(float(gi["R_int"].median()), 3) if len(gi) else None,
                               "frac_p_lt_0.05": round(float((gi["p_int"] < 0.05).mean()), 3) if len(gi) else None}}
    return out


def ev9v_locking(d: Path, cyc: pd.DataFrame, th: dict, screen: pd.DataFrame, n_perm: int) -> tuple[dict, dict]:
    bnd = json.loads((d / "ev9v_cycles.json").read_text())
    rel = cyc[(cyc["peak"] >= th["peak"]) & (cyc["dip"] >= th["dip"]) & (cyc["n_cycles"] >= 2)]
    strata = screen.pivot(index="video_id", columns="recognizer", values="stratum")
    out = {}
    for rec, key in (("B0", "prob_b0"), ("EchoPrime", "prob_ep")):
        items = []
        for r in rel.itertuples(index=False):
            if r.video_id not in strata.index:
                continue
            z = np.load(d / "probs_sorted" / f"{r.video_id}.npz")
            ph = phase_from_boundaries(int(r.n_frames), np.array(bnd[r.video_id]), r.period_s * 30.0)[z["frame_idx"]]
            items.append((strata.loc[r.video_id, rec], ph, z[key]))
        out[rec] = locking_summary(items, n_perm)
    return out, {"evaluable_clips": int(len(rel))}


def mimic_locking(d: Path, m: pd.DataFrame, mscreen: pd.DataFrame, n_perm: int) -> dict:
    good = m[(m["ecg_ok"] == True) & (m["ecg_n_r"] >= 3)]  # noqa: E712  (at least two complete beats)
    strata = mscreen.pivot(index="key", columns="recognizer", values="stratum")
    out = {}
    for rec, key in (("B0", "prob_b0"), ("EchoPrime", "prob_ep")):
        items = []
        for r in good.itertuples(index=False):
            z = np.load(d / "mimic" / f"{r.key}.npz")
            items.append((strata.loc[r.key, rec], z["ecg_phase"][z["idx"]], z[key]))
        out[rec] = locking_summary(items, n_perm)
    return out


def ecg_check(m: pd.DataFrame, out: Path, n: int, seed: int) -> None:
    from scripts.mimic_round1 import decode  # local import: needs pydicom
    pick = m[m["ecg_ok"] == True].sample(n=min(n, int((m["ecg_ok"] == True).sum())), random_state=seed)  # noqa: E712
    tiles = []
    for r in pick.itertuples(index=False):
        ims = decode(r.path); w, h = ims[0].size; top = int(h * 0.6)
        band = ims[0].crop((0, top, w, h)).convert("RGB"); dr = ImageDraw.Draw(band)
        z = np.load(Path(out).parent / "mimic" / f"{r.key}.npz")
        rf = z["r_frames"]
        cur = [np.flatnonzero(_cursor_mask(np.asarray(im.crop((0, top, w, h))).astype(np.int16)).any(0)) for im in (ims[0], ims[-1])]
        x0 = cur[0].mean() if cur[0].size else 0; x1 = cur[1].mean() if cur[1].size else w
        for f in rf:
            x = x0 + (x1 - x0) * f / max(1, len(ims) - 1)
            dr.line([(x, 0), (x, band.height)], fill=(255, 0, 255), width=1)
        dr.text((4, 4), f"HR ecg {60 / r.ecg_rr_s:.0f} tag {r.hr_tag:.0f} img {60 / r.img_period_s if r.img_period_s == r.img_period_s else float('nan'):.0f}", fill=(255, 255, 0))
        tiles.append(band.resize((band.width // 2, band.height // 2)))
    if tiles:
        g = Image.new("RGB", (tiles[0].width, sum(t.height + 4 for t in tiles)), "white"); y = 0
        for t in tiles:
            g.paste(t, (0, y)); y += t.height + 4
        g.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Round-1 report")
    ap.add_argument("--dir", required=True); ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--ecg-check-n", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); d = Path(args.dir); rep: dict = {}

    m = mimic_tables(d)
    ok = m[m["error"].isna()]
    rep["mimic_ecg"] = {
        "loops": int(len(m)), "decode_errors": int(m["error"].notna().sum()), "patients": int(ok["subject_id"].nunique()),
        "ecg_parsed": int((ok["ecg_ok"] == True).sum()),  # noqa: E712
        "ecg_fail_reasons": ok.loc[ok["ecg_ok"] != True, "ecg_reason"].value_counts().to_dict(),  # noqa: E712
        "ecg_with_rr": int((ok["ecg_n_r"] >= 2).sum()),
        "ecg_hr_within10pct_of_tag": int(ok["ecg_tag_agree"].sum()),
        "complete_beats_per_loop": (ok["ecg_n_r"] - 1).clip(lower=0).value_counts().sort_index().to_dict(),
        "irregular_rr_cv_gt_0.15": int((ok["ecg_rr_cv"] > 0.15).sum()), "rr_cv_evaluable": int(ok["ecg_rr_cv"].notna().sum()),
        "image_vs_ecg_all": ok.loc[ok["ecg_n_r"] >= 2, "img_class"].value_counts().to_dict(),
    }
    grid, choice = threshold_grid(ok)
    rep["image_threshold_grid"] = grid.round(3).to_dict("records"); rep["image_threshold"] = choice
    th = {"peak": choice["chosen"]["peak"], "dip": choice["chosen"]["dip"]}

    cyc = pd.read_csv(d / "ev9v_cycles.csv")
    rel = (cyc["peak"] >= th["peak"]) & (cyc["dip"] >= th["dip"])
    rep["ev9v_cycles"] = {
        "clips": int(len(cyc)), "no_estimate": int(cyc["period_s"].isna().sum()), "short_lt_2s": int((cyc["duration_s"] < 2).sum()),
        "reliable": int(rel.sum()), "reliable_frac": round(float(rel.mean()), 3),
        "reliable_ge_2_cycles": int((rel & (cyc["n_cycles"] >= 2)).sum()),
        "reliable_irregular_cv_gt_0.15": int((rel & (cyc["cv"] > 0.15)).sum()),
        "reliable_hr_median": round(float((60 / cyc.loc[rel, "period_s"]).median()), 1) if rel.any() else None,
        "peak_quartiles": cyc["peak"].quantile([0.25, 0.5, 0.75]).round(2).tolist(),
        "mimic_peak_quartiles": ok["img_peak"].quantile([0.25, 0.5, 0.75]).round(2).tolist(),
        "reliable_by_label": cyc.assign(rel=rel).merge(pd.read_csv(d / "screen_ev9v.csv").drop_duplicates("video_id")[["video_id", "label"]], on="video_id")
                               .groupby("label")["rel"].mean().round(3).to_dict(),
    }

    mscreen, rep["mimic_screen"] = screen_mimic(d, ok)
    screen = pd.read_csv(d / "screen_ev9v.csv")
    rep["phase_locking_ev9v"], extra = ev9v_locking(d, cyc, th, screen, args.n_perm); rep["phase_locking_ev9v"].update(extra)
    rep["phase_locking_mimic_ecg"] = mimic_locking(d, ok, mscreen, args.n_perm)
    (d / "round1_report.json").write_text(json.dumps(rep, indent=1, default=str))
    ecg_check(ok, d / "ecg_check.png", args.ecg_check_n, args.seed)
    print(json.dumps({k: v for k, v in rep.items() if k != "image_threshold_grid"}, indent=1, default=str))
    print(grid.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
