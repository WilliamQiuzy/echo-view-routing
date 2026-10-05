#!/usr/bin/env python
"""Cross-population step 1: the current strong systems on EchoNet-Pediatric native clips
(configs/experiments/pediatric_transfer_v2.yaml). No training or tuning.

Systems: E0 and E1 per-sample argmax (references), E0 + B3 and E1 + B3 with their EV9V-selected parameters, the E1 ASRF
seeds 0-2 (official decode) and B7 (the official EchoViewCLIP stage-1 video model, nine outputs kept; PMPALA is its own
wrong output). A clip's label is the majority of its per-sample labels. Per part (test folds, calibration folds):
A4C and PSAX recall, macro-F1 over the two present classes (as the frozen evaluation), where misread clips go, and the
groups behind the misread clips. Then the pre-declared decision. Every protocol video is scored. Writes clips.csv
(server only), summary.json and summary.md into --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pediatric_extract import verify_protocol  # noqa: E402
from scripts.round4_failure_inventory import load_asrf  # noqa: E402
from scripts.round5_reachability import asrf_decode, b3_decode  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.features.cache import load_video_features  # noqa: E402

NAMES = ["PLAX", "PSAX", "A4C", "A5C", "SC4C", "PMPALA"]
PSAX, A4C, SEEDS = 1, 2, (0, 1, 2)
RAW9_TO_OUT = {"PLHLA": 0, "PMASA": 1, "PMVLSA": 1, "PASA": 1, "A4C": 2, "A5C": 3, "PMPALA": 5, "PPMLSA": 1, "SC4C": 4}
E0_DIR = "pediatric/ped-ts10hz-letterbox224-v1/features/beb36d80df31"
E1_DIR = "pediatric/ped-ts10hz-e1crop224-v1/features/16efc85be645"
B7_DIR = "pediatric/ped-ts10hz-b7window-v1/a74c7b233e79"
B3 = {"E0": {"half_window": 9, "threshold": 0.39023423383871153, "nms_radius": 5},       # E0 simple run, selected.json
      "E1": {"half_window": 9, "threshold": 0.22457831805925244, "nms_radius": 5}}       # E1 simple run, selected.json


def majority(labels: np.ndarray) -> int:
    return int(np.bincount(np.asarray(labels, int), minlength=len(NAMES)).argmax())


def scores(df: pd.DataFrame, col: str) -> dict:
    y, yhat = df["label"].to_numpy(), df[col].to_numpy()
    out = {"clips": int(len(df)), "macro_f1_present": float(f1_score(y, yhat, labels=[PSAX, A4C], average="macro", zero_division=0))}
    for name, c in (("PSAX", PSAX), ("A4C", A4C)):
        sub = df[df["label"] == c]; wrong = sub[sub[col] != c]
        by_group = Counter(wrong["group"])
        out[name] = {"clips": int(len(sub)), "recall": float((sub[col] == c).mean()),
                     "misread_as": {NAMES[k] if k >= 0 else "no seed majority": int(v) for k, v in sorted(Counter(wrong[col]).items())},
                     "groups_with_misread": int(len(by_group)), "groups_with_class": int(sub["group"].nunique()),
                     "share_of_groups": float(len(by_group) / max(sub["group"].nunique(), 1)),
                     "top10_groups_share_of_misread": float(sum(v for _, v in by_group.most_common(10)) / len(wrong)) if len(wrong) else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default="runs/md_round4/spec.json"); ap.add_argument("--out", required=True)
    ap.add_argument("--protocol-dir", default="data/manifests/pediatric/pediatric-external-v1")
    a = ap.parse_args(); paths = load_paths(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    verify_protocol(Path(a.protocol_dir)); videos = pd.read_csv(Path(a.protocol_dir) / "videos.csv", dtype={"clip_no": str})
    spec = json.loads(Path(a.spec).read_text()); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    asrf = {k: load_asrf(Path(spec["encoders"]["E1"]["asrf"][str(k)]), paths.repo_root, 512, device) for k in SEEDS}
    root = paths.cache_root; rows = []
    for r in videos.itertuples(index=False):
        e0 = load_video_features(root / E0_DIR / f"{r.video_id}.npz"); e1 = load_video_features(root / E1_DIR / f"{r.video_id}.npz")
        b7 = np.load(root / B7_DIR / f"{r.video_id}.npz")
        if not (e0.n == e1.n == len(b7["raw"]) == int(r.n_samples)):
            raise SystemExit(f"{r.video_id}: sample counts differ (E0 {e0.n}, E1 {e1.n}, B7 {len(b7['raw'])}, protocol {r.n_samples})")
        b7_out = np.array([RAW9_TO_OUT[str(c)] for c in b7["classes"]])[b7["raw"].argmax(1)]
        seeds = [majority(asrf_decode(*asrf[k], e1.feat.astype(np.float16), device)[0]) for k in SEEDS]
        vote = Counter(seeds).most_common(1)[0]
        rows.append({"video_id": r.video_id, "group": r.group, "exam": r.exam, "role": r.role, "label": int(r.label), "n": int(r.n_samples),
                     "E0 argmax": majority(e0.prob.argmax(1)), "E0 + B3": majority(b3_decode(e0.prob, B3["E0"])[0]),
                     "E1 argmax": majority(e1.prob.argmax(1)), "E1 + B3": majority(b3_decode(e1.prob, B3["E1"])[0]),
                     **{f"E1 + ASRF s{k}": s for k, s in zip(SEEDS, seeds)}, "E1 + ASRF": vote[0] if vote[1] >= 2 else -1,
                     "B7": majority(b7_out)})
    df = pd.DataFrame(rows); df.to_csv(out / "clips.csv", index=False)
    systems = ["E0 argmax", "E0 + B3", "E1 argmax", "E1 + B3", *[f"E1 + ASRF s{k}" for k in SEEDS], "E1 + ASRF", "B7"]
    res = {"videos": int(len(df)), "parts": {}}
    for part in ("test", "calibration"):
        sub = df[df["role"] == part]
        res["parts"][part] = {"clips": int(len(sub)), "groups": int(sub["group"].nunique()), "systems": {s: scores(sub, s) for s in systems}}
    t = res["parts"]["test"]["systems"]
    adequate = {s: bool(t[s]["A4C"]["recall"] >= 0.80 and t[s]["PSAX"]["recall"] >= 0.80) for s in systems}
    broad = {s: any(t[s][c]["recall"] < 0.80 and t[s][c]["share_of_groups"] >= 0.10 for c in ("A4C", "PSAX")) for s in ("E1 + B3", "E1 + ASRF")}
    res["decision"] = {"adequate_transfer": adequate, "broad_failure_E1": broad, "proceed_to_adaptation": any(broad.values())}
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    md = ["# Pediatric transfer of the current systems (generated by scripts/pediatric_transfer_v2.py)", ""]
    for part, pr in res["parts"].items():
        md += [f"## {part} folds ({pr['clips']} clips, {pr['groups']} groups)", "",
               "| System | A4C recall | PSAX recall | Macro-F1 (A4C, PSAX) | PSAX misread as | A4C misread as | Groups with misread PSAX / A4C |", "|---|---|---|---|---|---|---|"]
        for s, v in pr["systems"].items():
            md.append(f"| {s} | {v['A4C']['recall']:.3f} | {v['PSAX']['recall']:.3f} | {v['macro_f1_present']:.3f} | {v['PSAX']['misread_as']} | {v['A4C']['misread_as']} | "
                      f"{v['PSAX']['groups_with_misread']}/{v['PSAX']['groups_with_class']} / {v['A4C']['groups_with_misread']}/{v['A4C']['groups_with_class']} |")
        md.append("")
    md += ["## Decision (pre-declared)", "", "```", json.dumps(res["decision"], indent=1), "```"]
    (out / "summary.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
