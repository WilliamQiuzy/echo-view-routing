#!/usr/bin/env python
"""Round-1 candidate screening on real EV9V clips: view change vs. locally insufficient evidence.

Uses only 10 Hz frame probabilities of two recognizers, in true temporal order (<out>/probs_sorted, written by
scripts/realvideo_probs.py), and never the clip label:
- B0: ResNet-18 frame classifier trained on the EV9V training split;
- EchoPrime: the released view classifier, never trained on EV9V. Its mass on views outside the five families (A2C,
  A3C, Doppler, SSN) is kept as a sixth class, "other".
Scores and strata are defined in echo_routing/realvideo/screening.py. Counts are candidates for mechanism discovery,
not prevalence. The random sample is two-stage (random patients, then one random clip each) so that its units are
independent; "session" below means patient (id without the clip index). Discovery clips (A5C analysis) are excluded from counts and samples and only checked for face validity.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.realvideo.screening import DEFAULT, clip_scores, cohen_kappa, patient_of  # noqa: E402

HZ = 10.0
GRID = {"min_run": [1, 2, 3], "max_exc": [0.5, 1.0, 2.0], "theta": [0.3, 0.5, 0.7], "rep_k": [2, 3]}
DISCOVERY = ["2022-05-24_14-30-39_320x240_1", "2022-07-26_16-43-25_320x240_6", "2022-07-11_09-51-34_320x240_7",
             "2022-05-24_14-30-39_320x240_15", "2022-05-23_15-30-01_320x240_5", "2022-07-10_10-58-17_320x240_7",
             "2022-05-05_10-38-37_320x240_1"]  # six hard A5C clips + the suspected A4C label error
CLASSES = {"B0": ["PLAX", "PSAX", "A4C", "A5C", "SC4C"], "EchoPrime": ["PLAX", "PSAX", "A4C", "A5C", "SC4C", "other"]}
REP, SUS = ("repeated", "mixed"), ("sustained", "mixed")


def load_probs(probs_dir: Path, vid: str) -> dict[str, np.ndarray]:
    """Temporally ordered sequences from scripts/realvideo_probs.py (the shared caches shuffle frames within clips)."""
    z = np.load(probs_dir / f"{vid}.npz")
    return {"B0": z["prob_b0"].astype(float), "EchoPrime": z["prob_ep"].astype(float)}


def sensitivity(probs: dict, vids: list[str]) -> pd.DataFrame:
    rows = []
    for vals in itertools.product(*GRID.values()):
        prm = dict(zip(GRID, vals))
        st = {rec: pd.Series({v: clip_scores(probs[v][rec], CLASSES[rec], HZ, prm)["stratum"] for v in vids}) for rec in CLASSES}
        for rec, s in st.items():
            rows.append({**prm, "recognizer": rec, "repeated_any": int(s.isin(REP).sum()), "sustained_any": int(s.isin(SUS).sum()),
                         "mixed": int((s == "mixed").sum()), "none": int((s == "none").sum()), "short": int((s == "short").sum())})
        a, b = st["B0"], st["EchoPrime"]
        rows.append({**prm, "recognizer": "either", "repeated_any": int((a.isin(REP) | b.isin(REP)).sum()),
                     "sustained_any": int((a.isin(SUS) | b.isin(SUS)).sum())})
        rows.append({**prm, "recognizer": "both", "repeated_any": int((a.isin(REP) & b.isin(REP)).sum()),
                     "sustained_any": int((a.isin(SUS) & b.isin(SUS)).sum())})
    return pd.DataFrame(rows)


def random_sample(man: pd.DataFrame, n: int, seed: int) -> list[str]:
    """n random sessions, then one random clip in each."""
    rng = np.random.default_rng(seed)
    sessions = sorted(man["session"].unique())
    pick = rng.choice(sessions, size=min(n, len(sessions)), replace=False)
    return sorted(rng.choice(sorted(man.loc[man.session == s, "video_id"])) for s in pick)


def main() -> None:
    ap = argparse.ArgumentParser(description="EV9V real-clip candidate screening")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-random", type=int, default=60); ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args(); paths = load_paths(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(paths.manifest_path)
    man = man[(man["dataset"] == "EV9V") & man["split"].isin(["validation", "test"]) & man["family5_label"].notna()].copy()
    man["session"] = man["video_id"].map(patient_of)  # patient (703 in EV9V)
    probs = {v: load_probs(out / "probs_sorted", v) for v in man["video_id"]}

    rows = [{"video_id": r.video_id, "session": r.session, "split": r.split, "label": r.family5_label,
             "duration_s": r.duration_s, "n_samples": len(p), "recognizer": rec, "discovery": r.video_id in DISCOVERY,
             **clip_scores(p, CLASSES[rec], HZ)}
            for r in man.itertuples(index=False) for rec, p in probs[r.video_id].items()]
    df = pd.DataFrame(rows); df.to_csv(out / "screen_ev9v.csv", index=False)
    main_df = df[~df["discovery"]]
    vids = sorted(main_df["video_id"].unique())
    sensitivity(probs, vids).to_csv(out / "screen_ev9v_sensitivity.csv", index=False)

    w = main_df.pivot(index="video_id", columns="recognizer", values="stratum")
    num = main_df.pivot(index="video_id", columns="recognizer", values=["t_ret", "contrast"]).astype(float)
    ok = w["B0"] != "short"
    sess = main_df.drop_duplicates("video_id").set_index("video_id")["session"]
    either_rep, either_sus = w["B0"].isin(REP) | w["EchoPrime"].isin(REP), w["B0"].isin(SUS) | w["EchoPrime"].isin(SUS)
    summary = {
        "clips": len(vids), "sessions": int(main_df.session.nunique()), "params": DEFAULT,
        "short_clips": int((~ok).sum()),
        "strata": {rec: main_df[main_df.recognizer == rec]["stratum"].value_counts().to_dict() for rec in CLASSES},
        "repeated_any": {"B0": int(w["B0"].isin(REP).sum()), "EchoPrime": int(w["EchoPrime"].isin(REP).sum()),
                         "either": int(either_rep.sum()), "both": int((w["B0"].isin(REP) & w["EchoPrime"].isin(REP)).sum()),
                         "either_sessions": int(sess[either_rep[either_rep].index].nunique())},
        "sustained_any": {"B0": int(w["B0"].isin(SUS).sum()), "EchoPrime": int(w["EchoPrime"].isin(SUS).sum()),
                          "either": int(either_sus.sum()), "both": int((w["B0"].isin(SUS) & w["EchoPrime"].isin(SUS)).sum()),
                          "either_sessions": int(sess[either_sus[either_sus].index].nunique())},
        "cross_table": pd.crosstab(w.loc[ok, "B0"], w.loc[ok, "EchoPrime"]).to_dict(),
        "kappa_strata": cohen_kappa(list(w.loc[ok, "B0"]), list(w.loc[ok, "EchoPrime"])),
        "spearman_t_ret": float(spearmanr(num.loc[ok, ("t_ret", "B0")], num.loc[ok, ("t_ret", "EchoPrime")]).statistic),
        "spearman_contrast": float(spearmanr(num.loc[ok, ("contrast", "B0")], num.loc[ok, ("contrast", "EchoPrime")]).statistic),
        "discovery_check": df[df["discovery"]][["video_id", "label", "recognizer", "stratum", "n_ret", "contrast", "sides"]].to_dict("records"),
        "random_sample": random_sample(man[~man.video_id.isin(DISCOVERY)], args.n_random, args.seed),
    }
    (out / "screen_ev9v_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("discovery_check", "random_sample", "cross_table")}, indent=1))


if __name__ == "__main__":
    main()
