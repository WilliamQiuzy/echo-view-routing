#!/usr/bin/env python
"""Round-2 objective checks on the round-1 outputs (no expert labels needed).

1. Intersection: EV9V clips with a reliable image-motion cycle (>= 2 cycles) crossed with sustained / repeated change
   candidates, by view label, patient group and pilot coverage.
2. Cycle method, validated independently:
   a. thresholds chosen on half of the MIMIC patients and evaluated on the other half (5 random splits);
   b. same-phase correspondence against the drawn ECG: the ECG phase at each image "same-phase" frame, after removing
      one fixed offset per loop; reports the residual (ms), the drift per beat, missed beats and doubled frequency;
3. Phase-locking null: the interior-run test is re-run through the whole pipeline (screening, stratum, edge handling)
   on null data: (i) each loop's own off-pattern with the ECG phase of a different patient's loop of equal length,
   (ii) synthetic off-patterns from a two-state Markov chain matched to each loop and independent of phase.
Reads <dir> (runs/_demo/realvideo); writes <dir>/round2_checks.json and <dir>/round2_intersection.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.realvideo.cycles import phase_from_boundaries, phase_locking  # noqa: E402
from echo_routing.realvideo.screening import clip_scores  # noqa: E402
from scripts.report_realvideo_round1 import CLASSES, DIPS, PEAKS, TARGET, interior_window, mimic_tables  # noqa: E402

REP, SUS = ("repeated", "mixed"), ("sustained", "mixed")


# ---------- 1. intersection ----------
def intersection(d: Path, th: dict) -> tuple[pd.DataFrame, dict]:
    sc = pd.read_csv(d / "screen_ev9v.csv")
    w = sc.pivot(index="video_id", columns="recognizer", values="stratum")
    meta = sc.drop_duplicates("video_id").set_index("video_id")[["session", "label", "discovery", "duration_s"]]
    cyc = pd.read_csv(d / "ev9v_cycles.csv").set_index("video_id")
    key = pd.read_csv(d / "pilot" / "key" / "pilot_key.csv")
    in_pilot = set(key.loc[key["pilot_stratum"] != "practice", "video_id"])
    t = meta.join(w).join(cyc[["peak", "dip", "n_cycles"]])
    t["cycle_ok"] = (t["peak"] >= th["peak"]) & (t["dip"] >= th["dip"]) & (t["n_cycles"] >= 2)
    t["sus_either"] = t["B0"].isin(SUS) | t["EchoPrime"].isin(SUS)
    t["sus_both"] = t["B0"].isin(SUS) & t["EchoPrime"].isin(SUS)
    t["rep_either"] = t["B0"].isin(REP) | t["EchoPrime"].isin(REP)
    t["rep_both"] = t["B0"].isin(REP) & t["EchoPrime"].isin(REP)
    t["in_pilot"] = t.index.isin(in_pilot)
    t = t[~t["discovery"]]
    rows = []
    for name, m in (("all clips", t.index == t.index), ("sustained, either", t["sus_either"]), ("sustained, both", t["sus_both"]),
                    ("repeated, either", t["rep_either"]), ("repeated, both", t["rep_both"]),
                    ("neither", ~t["sus_either"] & ~t["rep_either"])):
        s = t[m]
        rows.append({"set": name, "clips": len(s), "groups": s["session"].nunique(),
                     "with_cycle": int(s["cycle_ok"].sum()), "with_cycle_groups": s.loc[s["cycle_ok"], "session"].nunique(),
                     "with_cycle_in_pilot": int((s["cycle_ok"] & s["in_pilot"]).sum()), "in_pilot": int(s["in_pilot"].sum()),
                     **{f"{lab}": f"{int(((s['label'] == lab) & s['cycle_ok']).sum())}/{int((s['label'] == lab).sum())}"
                        for lab in ("A4C", "A5C", "PLAX", "PSAX", "SC4C")}})
    table = pd.DataFrame(rows)
    both = t[t["sus_either"] & t["cycle_ok"]]
    return table, {"threshold": th, "sustained_with_cycle": both[["label", "B0", "EchoPrime", "peak", "n_cycles", "in_pilot"]]
                   .reset_index().to_dict("records")}


# ---------- 2a. patient-split thresholds ----------
def pick(ref: pd.DataFrame) -> dict:
    best = None
    for pk in PEAKS:
        for dp in DIPS:
            rel = ref[(ref["img_peak"] >= pk) & (ref["img_dip"] >= dp)]
            if len(rel) and (rel["img_class"] == "agree").mean() >= TARGET:
                cand = (len(rel) / len(ref), -pk, -dp)
                if best is None or cand > best[0]:
                    best = (cand, {"peak": pk, "dip": dp})
    return best[1] if best else {"peak": max(PEAKS), "dip": max(DIPS)}


def score(ref: pd.DataFrame, th: dict) -> dict:
    rel = ref[(ref["img_peak"] >= th["peak"]) & (ref["img_dip"] >= th["dip"])]
    return {"coverage": round(len(rel) / len(ref), 3), "agree": round(float((rel["img_class"] == "agree").mean()), 3),
            "double": round(float((rel["img_class"] == "double").mean()), 3), "n": int(len(rel))}


def patient_split(ref: pd.DataFrame, n_splits: int = 5) -> list[dict]:
    out = []
    subjects = np.array(sorted(ref["subject_id"].unique()))
    for seed in range(n_splits):
        rng = np.random.default_rng(seed); perm = rng.permutation(subjects); half = set(perm[: len(perm) // 2])
        a, b = ref[ref["subject_id"].isin(half)], ref[~ref["subject_id"].isin(half)]
        tha, thb = pick(a), pick(b)
        out.append({"seed": seed, "chosen_on_A": tha, "held_out_B": score(b, tha), "chosen_on_B": thb, "held_out_A": score(a, thb)})
    return out


# ---------- 2b. same-phase correspondence against the ECG ----------
def correspondence(d: Path, ref: pd.DataFrame, th: dict) -> dict:
    rel = ref[(ref["img_peak"] >= th["peak"]) & (ref["img_dip"] >= th["dip"])]
    err_ms, drift_ms, missed, extra, normal, n_loops, loop_max = [], [], 0, 0, 0, 0, []
    for r in rel.itertuples(index=False):
        z = np.load(d / "mimic" / f"{r.key}.npz")
        rf, b = np.asarray(z["r_frames"], float), np.asarray(z["img_boundaries"], int)
        if rf.size < 2 or b.size < 2:
            continue
        rr = float(np.median(np.diff(rf))); ms_per_frame = float(z["frame_time_s"]) * 1000.0
        ph = phase_from_boundaries(int(z["n_frames"]), rf, rr)
        inside = b[(b >= rf[0]) & (b <= rf[-1])]
        if inside.size < 2:
            continue
        phi = ph[inside]
        off = np.angle(np.exp(2j * np.pi * phi).mean()) / (2 * np.pi)
        res = (phi - off + 0.5) % 1.0 - 0.5  # residual phase after one fixed offset per loop, in cycles
        e = np.abs(res) * rr * ms_per_frame
        err_ms.extend(e.tolist()); loop_max.append(float(e.max())); n_loops += 1
        k = (inside - inside[0]) / rr
        if inside.size >= 3:
            drift_ms.append(abs(float(np.polyfit(k, res, 1)[0])) * rr * ms_per_frame)
        s = np.diff(b) / rr
        missed += int(((s >= 1.7) & (s <= 2.3)).sum()); extra += int((s <= 0.65).sum()); normal += int(((s >= 0.8) & (s <= 1.2)).sum())
    e = np.array(err_ms); lm = np.array(loop_max)
    gaps = missed + extra + normal
    return {"loops": n_loops, "boundaries": int(e.size), "median_abs_ms": round(float(np.median(e)), 1),
            "p90_abs_ms": round(float(np.percentile(e, 90)), 1), "share_within_50ms": round(float((e <= 50).mean()), 3),
            "loops_all_within_50ms": round(float((lm <= 50).mean()), 3), "median_drift_ms_per_beat": round(float(np.median(drift_ms)), 1) if drift_ms else None,
            "boundary_gaps": gaps, "missed_beat_gaps": missed, "doubled_frequency_gaps": extra, "one_beat_gaps": normal}


# ---------- 3. phase-locking null ----------
def markov_like(off: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Two-state chain with the loop's own off fraction and mean off-run length; independent of phase."""
    n = off.size; f = off.mean()
    if f in (0.0, 1.0):
        return off.copy()
    starts = max(1, int(((~off[:-1]) & off[1:]).sum() + off[0]))
    mean_run = max(1.0, off.sum() / starts)
    p_exit = 1.0 / mean_run; p_enter = min(1.0, p_exit * f / (1 - f))
    s = np.empty(n, dtype=bool); s[0] = rng.random() < f
    for i in range(1, n):
        s[i] = (rng.random() >= p_exit) if s[i - 1] else (rng.random() < p_enter)
    return s


def null_rates(d: Path, m: pd.DataFrame, n_perm: int, seed: int) -> dict:
    good = m[(m["ecg_ok"] == True) & (m["ecg_n_r"] >= 3)]  # noqa: E712
    loops = []
    for r in good.itertuples(index=False):
        z = np.load(d / "mimic" / f"{r.key}.npz")
        step = int(z["idx"][1] - z["idx"][0]) if len(z["idx"]) > 1 else 1
        hz = 1.0 / (step * float(z["frame_time_s"]))
        loops.append({"key": r.key, "subject": r.subject_id, "phase": z["ecg_phase"][z["idx"]], "hz": hz,
                      "probs": {"B0": z["prob_b0"].astype(float), "EchoPrime": z["prob_ep"].astype(float)}})
    by_len: dict[int, list[int]] = {}
    for i, L in enumerate(loops):
        by_len.setdefault(len(L["phase"]), []).append(i)
    rng = np.random.default_rng(seed)
    out = {}
    for rec in CLASSES:
        rows = []
        for i, L in enumerate(loops):
            p = L["probs"][rec]; top = p.argmax(1); dom = np.bincount(top).argmax(); off = top != dom
            stratum = clip_scores(p, CLASSES[rec], L["hz"])["stratum"]
            donors = [j for j in by_len[len(L["phase"])] if loops[j]["subject"] != L["subject"]]
            phases = {"real": L["phase"]}
            if donors:
                phases["swap"] = loops[int(rng.choice(donors))]["phase"]
            for kind, ph in phases.items():
                a, b, mask = interior_window(off)
                if b - a > 2:
                    rows.append({"kind": kind, "stratum": stratum, "p": phase_locking(ph[a:b], mask[a:b], n_perm=n_perm, seed=i)[1]})
            syn = markov_like(off, rng)
            onehot = np.eye(2)[syn.astype(int)]
            s_stratum = clip_scores(onehot, ["on", "off"], L["hz"])["stratum"]
            a, b, mask = interior_window(syn)
            if b - a > 2:
                rows.append({"kind": "synthetic", "stratum": s_stratum, "p": phase_locking(L["phase"][a:b], mask[a:b], n_perm=n_perm, seed=i)[1]})
        t = pd.DataFrame(rows).dropna()
        out[rec] = {kind: {s: {"n": int(len(g)), "rejection": round(float((g["p"] < 0.05).mean()), 3)}
                           for s, g in tk.groupby("stratum")} for kind, tk in t.groupby("kind")}
    return out


def excess_with_donors(d: Path, m: pd.DataFrame, n_donors: int, n_perm: int, n_boot: int, seed: int) -> dict:
    """Real-phase rejection minus the mean rejection over `n_donors` other-patient phases, per loop, with a
    patient-clustered bootstrap 95% interval (interior runs only, as in the round-1 report)."""
    good = m[(m["ecg_ok"] == True) & (m["ecg_n_r"] >= 3)]  # noqa: E712
    loops = []
    for r in good.itertuples(index=False):
        z = np.load(d / "mimic" / f"{r.key}.npz")
        step = int(z["idx"][1] - z["idx"][0]) if len(z["idx"]) > 1 else 1
        loops.append({"subject": r.subject_id, "phase": z["ecg_phase"][z["idx"]], "hz": 1.0 / (step * float(z["frame_time_s"])),
                      "probs": {"B0": z["prob_b0"].astype(float), "EchoPrime": z["prob_ep"].astype(float)}})
    by_len: dict[int, list[int]] = {}
    for i, L in enumerate(loops):
        by_len.setdefault(len(L["phase"]), []).append(i)
    rng = np.random.default_rng(seed); out = {}
    for rec in CLASSES:
        rows = []
        for i, L in enumerate(loops):
            p = L["probs"][rec]; top = p.argmax(1); dom = np.bincount(top).argmax(); off = top != dom
            a, b, mask = interior_window(off)
            if b - a <= 2:
                continue
            real = phase_locking(L["phase"][a:b], mask[a:b], n_perm=n_perm, seed=i)[1]
            donors = [j for j in by_len[len(L["phase"])] if loops[j]["subject"] != L["subject"]]
            if not np.isfinite(real) or not donors:
                continue
            pick = rng.choice(donors, size=min(n_donors, len(donors)), replace=False)
            sw = [phase_locking(loops[j]["phase"][a:b], mask[a:b], n_perm=n_perm, seed=i)[1] for j in pick]
            rows.append({"subject": L["subject"], "stratum": clip_scores(p, CLASSES[rec], L["hz"])["stratum"],
                         "real": float(real < 0.05), "swap": float(np.mean(np.array(sw) < 0.05))})
        t = pd.DataFrame(rows); res = {}
        for st, g in t.groupby("stratum"):
            subj = g["subject"].unique(); diffs = []
            for _ in range(n_boot):
                bs = rng.choice(subj, size=len(subj), replace=True)
                gb = pd.concat([g[g["subject"] == s_] for s_ in bs])
                diffs.append(gb["real"].mean() - gb["swap"].mean())
            res[st] = {"n": int(len(g)), "subjects": int(len(subj)), "real": round(float(g["real"].mean()), 3),
                       "swap": round(float(g["swap"].mean()), 3), "excess": round(float(g["real"].mean() - g["swap"].mean()), 3),
                       "excess_ci95": [round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)]}
        out[rec] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Round-2 objective checks")
    ap.add_argument("--dir", required=True); ap.add_argument("--n-perm", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--donors-only", type=int, default=0, help="only run the multi-donor excess with this many donors per loop")
    ap.add_argument("--heldout-correspondence", action="store_true", help="only run same-phase correspondence on held-out patients")
    args = ap.parse_args(); d = Path(args.dir)
    if args.heldout_correspondence:
        m = mimic_tables(d); m = m[m["error"].isna()]
        ref = m[(m["ecg_ok"] == True) & (m["ecg_n_r"] >= 2) & m["ecg_tag_agree"]]  # noqa: E712
        subjects = np.array(sorted(ref["subject_id"].unique())); res = []
        for seed in range(5):
            perm = np.random.default_rng(seed).permutation(subjects); half = set(perm[: len(perm) // 2])
            a, b = ref[ref["subject_id"].isin(half)], ref[~ref["subject_id"].isin(half)]
            res.append({"seed": seed, "threshold_from_A": pick(a), "B": correspondence(d, b, pick(a)),
                        "threshold_from_B": pick(b), "A": correspondence(d, a, pick(b))})
        (d / "round2_heldout_correspondence.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)); return
    if args.donors_only:
        m = mimic_tables(d); m = m[m["error"].isna()]
        res = excess_with_donors(d, m, args.donors_only, args.n_perm, 1000, args.seed)
        (d / "round2_null_donors.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)); return
    rep = json.loads((d / "round1_report.json").read_text())
    th = {"peak": rep["image_threshold"]["chosen"]["peak"], "dip": rep["image_threshold"]["chosen"]["dip"]}
    table, extra = intersection(d, th)
    table.to_csv(d / "round2_intersection.csv", index=False)
    m = mimic_tables(d); m = m[m["error"].isna()]
    ref = m[(m["ecg_ok"] == True) & (m["ecg_n_r"] >= 2) & m["ecg_tag_agree"]]  # noqa: E712
    out = {"intersection": extra, "patient_split": patient_split(ref), "correspondence_all_ref_threshold": correspondence(d, ref, th),
           "correspondence_no_threshold": correspondence(d, ref, {"peak": -1.0, "dip": -1.0})}
    out["phase_locking_null"] = null_rates(d, m, args.n_perm, args.seed)
    (d / "round2_checks.json").write_text(json.dumps(out, indent=1, default=str))
    print(table.to_string(index=False))
    print(json.dumps({k: v for k, v in out.items() if k != "intersection"}, indent=1, default=str))


if __name__ == "__main__":
    main()
