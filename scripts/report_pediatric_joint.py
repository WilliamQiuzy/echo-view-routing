#!/usr/bin/env python
"""Pediatric joint table (2026-10-01 closing decisions, item 3): classification, class-aware segment F1@50 and
boundary F1 side by side for every frozen method, true order, test folds. Nothing is trained or tuned.

Classification and segment / boundary counts come from scripts/pediatric_eval.py outputs. One extra column re-decodes
the true-order exam streams from the cached predictions: of the true view changes, the share that is detected
within 0.5 s AND has the right view on both sides (majority decoded label of the fragment before and after). It shows
that locating a change is not the same as recognising the views around it. Intervals resample first-token groups
(2,000 draws, shared across methods). An EV9V column set from scripts/true_order_joint.py is added for contrast.
Writes <out>/pediatric_joint.json and .md (aggregates only).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.compose.recipe_builder import recipe_from_dict  # noqa: E402
from echo_routing.config import load_config, load_paths  # noqa: E402
from echo_routing.evaluate.metrics_boundary import match_boundaries  # noqa: E402
from echo_routing.realvideo.pediatric import PREPROCESS_VERSION, PROTOCOL_VERSION, recipe_group  # noqa: E402
from echo_routing.temporal.baselines.registry import AUDITED_B3, decode_with_prob  # noqa: E402
from echo_routing.temporal.segments import boundaries_from_labels  # noqa: E402
from scripts.pediatric_eval import METHODS, assemble, load_clips  # noqa: E402
from scripts.pediatric_freeze import B0_HASH, EV9V_CONFIG, HZ  # noqa: E402
from scripts.report_permutation_round import read  # noqa: E402

N_BOOT, BOOT_SEED = 2000, 20261001
NAMES = {"b0": "B0 frame classifier", "b1": "B1 smoothing", "b2": "B2 HMM", "b3": "B3, default", "b3a": "B3, audited",
         "b4": "B4 MS-TCN v2"}
EV9V_NAMES = {"b0": "B0 frame classifier", "b1": "B1 smoothing", "b2": "B2 HMM", "b3": "B3 change points, default",
              "b3a": "B3 change points, audited", "b4": "B4 MS-TCN v2"}


def labelled_transitions(streams, cfg: dict, method: str, tol: float) -> pd.DataFrame:
    rows = []
    for s in streams:
        lab = decode_with_prob(method, s.prob, cfg, s.feat, s.recipe.recipe_id)[0]
        pred_b = boundaries_from_labels(lab); sem = s.semantic_boundaries
        matched = {int(sem[j]) for _, j in match_boundaries(pred_b, sem, tol)}
        starts = np.r_[0, s.joins]; ends = np.r_[s.joins, len(lab)]
        maj = [int(np.bincount(lab[a:b], minlength=5).argmax()) for a, b in zip(starts, ends)]
        truth = [int(s.labels[a]) for a in starts]
        right = 0
        for k, j in enumerate(s.joins, start=1):
            if int(j) in matched and maj[k - 1] == truth[k - 1] and maj[k] == truth[k]:
                right += 1
        rows.append({"group": recipe_group(s.recipe), "sem": int(sem.size), "found": len(matched), "found_right": right})
    return pd.DataFrame(rows)


def unit_matrix(df: pd.DataFrame, cols: list[str], units: pd.Index) -> np.ndarray:
    return df.groupby("group")[cols].sum().reindex(units, fill_value=0).to_numpy(dtype=float)


def summarise(mats: dict, fn, W: np.ndarray) -> dict:
    with np.errstate(divide="ignore", invalid="ignore"):
        return {m: {"value": float(fn(x.sum(0))), "ci95": np.nanpercentile(fn(np.moveaxis(W @ x, -1, 0)), [2.5, 97.5]).tolist()}
                for m, x in mats.items()}


def macro_f1_two(c: np.ndarray) -> np.ndarray:
    """Macro-F1 over PSAX (1) and A4C (2) from per-true-class prediction counts c = [PSAX->0..4, A4C->0..4]."""
    m = c.reshape((2, 5) + c.shape[1:]); f = []
    for i, cls in enumerate((1, 2)):
        tp = m[i, cls]; fn = m[i].sum(0) - tp; fp = m[1 - i, cls]; den = 2 * tp + fp + fn
        f.append(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0))
    return (f[0] + f[1]) / 2


def main() -> None:
    ap = argparse.ArgumentParser(description="Pediatric joint classification / segmentation / boundary table")
    ap.add_argument("--eval", default="runs/_demo/pediatric/eval_v1"); ap.add_argument("--joint-ev9v", default="runs/_demo/true_order_joint/joint.json")
    ap.add_argument("--protocol-dir", default=f"data/manifests/pediatric/{PROTOCOL_VERSION}"); ap.add_argument("--out", required=True)
    args = ap.parse_args(); paths = load_paths(); ev = Path(args.eval); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    nat = read(ev / "natives.csv.gz"); nat = nat[nat["condition"] == "true"].copy()
    frag = read(ev / "fragments.csv.gz"); frag = frag[(frag["condition"] == "true") & (frag["bank"] == "exam")]
    strm = read(ev / "streams.csv.gz"); strm = strm[(strm["condition"] == "true") & (strm["bank"] == "exam")]
    for i, cls in enumerate((1, 2)):
        for k in range(5):
            nat[f"c{i}{k}"] = ((nat["label"] == cls) & (nat["majority"] == k)).astype(float)
    nat["correct"] = (nat["majority"] == nat["label"]).astype(float); nat["one"] = 1.0
    nat["right"] = nat["n"] - nat["dev_dec"]
    strm = strm.assign(right=strm["frame_acc"] * strm["n"])
    # true-order exam streams re-decoded for the labelled-transition column
    cfg = load_config(Path(EV9V_CONFIG)); cfg["decode"].setdefault("b3a", dict(AUDITED_B3))
    cfg["mstcn_official"] = {"pred_dir": str(paths.cache_root / "pediatric" / PREPROCESS_VERSION / "b4" / B0_HASH / "mstcn_v2_frameorder" / "true")}
    proto_dir = Path(args.protocol_dir)
    recipes = [recipe_from_dict(d) for d in json.loads((proto_dir / "recipes_test_exam.json").read_text())]
    clips = load_clips(paths.cache_root / "pediatric" / PREPROCESS_VERSION / "features" / B0_HASH,
                       sorted({f.video_id for r in recipes for f in r.fragments}))
    streams = [assemble(clips, r, "true", 0) for r in recipes]
    tol = cfg["policy"]["boundary_tolerances_s"][1] * HZ
    trans = {m: labelled_transitions(streams, copy.deepcopy(cfg), m, tol) for m in METHODS}
    rng = np.random.default_rng(BOOT_SEED)
    u_nat = pd.Index(sorted(nat["group"].unique())); W_nat = np.stack([np.bincount(d, minlength=len(u_nat)) for d in rng.integers(0, len(u_nat), (N_BOOT, len(u_nat)))]).astype(float)
    u_ex = pd.Index(sorted(strm["group"].unique())); W_ex = np.stack([np.bincount(d, minlength=len(u_ex)) for d in rng.integers(0, len(u_ex), (N_BOOT, len(u_ex)))]).astype(float)
    conf = [f"c{i}{k}" for i in range(2) for k in range(5)]
    res = {
        "sample accuracy, clips": summarise({m: unit_matrix(nat[nat["method"] == m], ["right", "n"], u_nat) for m in METHODS}, lambda c: c[0] / c[1], W_nat),
        "clip accuracy": summarise({m: unit_matrix(nat[nat["method"] == m], ["correct", "one"], u_nat) for m in METHODS}, lambda c: c[0] / c[1], W_nat),
        "clip macro-F1 (A4C, PSAX)": summarise({m: unit_matrix(nat[nat["method"] == m], conf, u_nat) for m in METHODS}, macro_f1_two, W_nat),
        "sample accuracy, exam streams": summarise({m: unit_matrix(strm[strm["method"] == m], ["right", "n"], u_ex) for m in METHODS}, lambda c: c[0] / c[1], W_ex),
        "segment F1@50, exam streams": summarise({m: unit_matrix(strm[strm["method"] == m], ["seg_matched50", "seg_pred", "seg_true"], u_ex) for m in METHODS},
                                                 lambda c: 2 * c[0] / (c[1] + c[2]), W_ex),
        "boundary F1, exam streams": summarise({m: unit_matrix(frag[frag["method"] == m], ["t_matched", "p_pred", "t_true"], u_ex) for m in METHODS},
                                               lambda c: 2 * c[0] / (c[1] + c[2]), W_ex),
        "boundary recall, exam streams": summarise({m: unit_matrix(frag[frag["method"] == m], ["t_matched", "t_true"], u_ex) for m in METHODS},
                                                   lambda c: c[0] / c[1], W_ex),
        "changes found with both views right": summarise({m: unit_matrix(trans[m], ["found_right", "sem"], u_ex) for m in METHODS},
                                                         lambda c: c[0] / c[1], W_ex),
        "found changes with both views right": summarise({m: unit_matrix(trans[m], ["found_right", "found"], u_ex) for m in METHODS},
                                                         lambda c: c[0] / c[1], W_ex),
    }
    ev9v = {}
    if Path(args.joint_ev9v).exists():
        j = json.loads(Path(args.joint_ev9v).read_text())["endpoints"]
        for m, name in EV9V_NAMES.items():
            if name in j:
                ev9v[m] = {k: j[name][k]["value"] for k in ("clip macro-F1", "segment F1@50, multi", "boundary F1, multi")}
    res_all = {"pediatric_test_true_order": res, "ev9v_test_true_order": ev9v,
               "units": {"clips": len(u_nat), "exam_stream_groups": len(u_ex), "exam_streams": len(streams),
                         "true_view_changes": int(trans["b0"]["sem"].sum())}}
    (out / "pediatric_joint.json").write_text(json.dumps(res_all, indent=1))
    cols = list(res)
    md = ["# EchoNet-Pediatric joint table (test folds, true order)", "",
          f"Native clips {len(u_nat)} groups; exam streams {len(streams)} in {len(u_ex)} groups with "
          f"{res_all['units']['true_view_changes']} true view changes. 95% intervals resample first-token groups.", "",
          "| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m in METHODS:
        md.append(f"| {NAMES[m]} | " + " | ".join(f"{res[c][m]['value']:.3f} [{res[c][m]['ci95'][0]:.3f}, {res[c][m]['ci95'][1]:.3f}]" for c in cols) + " |")
    if ev9v:
        md += ["", "EV9V test, true order, for contrast (from the joint comparison)", "",
               "| Method | clip macro-F1 | segment F1@50, 4–8 fragments | boundary F1, 4–8 fragments |", "|---|---|---|---|"]
        md += [f"| {NAMES[m]} | {v['clip macro-F1']:.3f} | {v['segment F1@50, multi']:.3f} | {v['boundary F1, multi']:.3f} |" for m, v in ev9v.items()]
    (out / "pediatric_joint.md").write_text("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    main()
