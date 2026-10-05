#!/usr/bin/env python
"""Round 8, step 1: A4C/A5C confusion in the training library (configs/experiments/training_library_check_v1.yaml).

Every training clip of the protocol is read by three existing systems, all trained on these clips (so in-sample):
E0's head (v3 cache), the E1 probe (E1 cache) and the E1 ASRF of rounds 2-5, seeds 0-2, which decodes each clip as a
native recording. A clip's prediction is the majority of its 10 Hz samples; ASRF's is the class most seeds give.
Reported, by clips and sessions: confusions in each direction, their concentration, co-failure across systems, margins
and lengths of correct and confused clips, and E1 clip-embedding similarities within and across sessions. Then the
pre-declared decision (S1-S3). Writes clips.csv (server only), summary.json and summary.md into --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.model_development_protocol import session_of  # noqa: E402
from scripts.round4_failure_inventory import load_asrf  # noqa: E402
from scripts.round5_reachability import asrf_decode  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.temporal.dev_data import FeatureStore  # noqa: E402
from echo_routing.temporal.dev_run import Inputs  # noqa: E402

A4C, A5C, SEEDS = 2, 3, (0, 1, 2)
NAMES = {0: "PLAX", 1: "PSAX", 2: "A4C", 3: "A5C", 4: "SC4C"}


def majority(labels: np.ndarray) -> int:
    return int(np.bincount(np.asarray(labels, int), minlength=5).argmax())


def concentration(clips: pd.DataFrame) -> dict:
    by_session = Counter(clips["session"])
    top3 = sum(c for _, c in by_session.most_common(3))
    return {"clips": int(len(clips)), "sessions": int(len(by_session)),
            "top3_sessions_share": float(top3 / len(clips)) if len(clips) else None,
            "clips_per_session_max": int(max(by_session.values())) if by_session else 0}


def describe(x: pd.Series) -> dict:
    x = x.dropna()
    return {"n": int(len(x)), "median": float(x.median()) if len(x) else None,
            "q25": float(x.quantile(0.25)) if len(x) else None, "q75": float(x.quantile(0.75)) if len(x) else None}


def pair_similarity(emb: np.ndarray, labels: np.ndarray, sessions: np.ndarray) -> dict:
    """Mean cosine similarity of clip pairs by (same class?, same session?) over A4C/A5C clips, and the same over the
    sessions that hold both classes."""
    sim = emb @ emb.T; n = len(labels); iu = np.triu_indices(n, 1)
    same_c = (labels[:, None] == labels[None, :])[iu]; same_s = (sessions[:, None] == sessions[None, :])[iu]; s = sim[iu]
    both = {x for x in set(sessions) if {A4C, A5C} <= set(labels[sessions == x])}
    in_both = np.isin(sessions, list(both)); mask_b = (in_both[:, None] & in_both[None, :])[iu]
    out = {"sessions_with_both_classes": len(both)}
    for name, m in (("same class, other session", same_c & ~same_s), ("same class, same session", same_c & same_s),
                    ("A4C-A5C, same session", ~same_c & same_s), ("A4C-A5C, other session", ~same_c & ~same_s)):
        out[name] = {"pairs": int(m.sum()), "mean": float(s[m].mean()) if m.any() else None}
        mb = m & mask_b
        out[f"{name} (sessions with both)"] = {"pairs": int(mb.sum()), "mean": float(s[mb].mean()) if mb.any() else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default="runs/md_round4/spec.json"); ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("-c", "--config", default="configs/experiments/training_library_check_v1.yaml")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); paths = load_paths(); spec = json.loads(Path(a.spec).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    e1 = spec["encoders"]["E1"]; inputs = Inputs(cfg, Path(a.protocol), Path(e1["feature_manifest"]), paths.cache_root)
    e0_key = json.loads(Path(spec["encoders"]["E0"]["feature_manifest"]).read_text())["encoder"]["cache_key"]
    e0 = FeatureStore(paths.cache_root, e0_key); asrf = {k: load_asrf(Path(e1["asrf"][str(k)]), paths.repo_root, inputs.feature_dim, device) for k in SEEDS}
    rows, embs = [], []
    for vid, label in inputs.train.clips:
        f1, f0 = inputs.store.get(vid, "orig"), e0.get(vid, "orig")
        if f0.n != f1.n:
            raise SystemExit(f"{vid}: E0 and E1 caches hold {f0.n} and {f1.n} samples")
        asrf_maj = [majority(asrf_decode(*asrf[k], f1.feat.astype(np.float16), device)[0]) for k in SEEDS]
        votes = Counter(asrf_maj).most_common(); asrf_vote = votes[0][0] if votes[0][1] >= 2 else -1
        rows.append({"video_id": vid, "session": session_of(vid), "label": int(label), "n": int(f1.n), "seconds": f1.n / float(cfg["protocol"]["hz"]),
                     "E0": majority(f0.prob.argmax(1)), "E1": majority(f1.logit.argmax(1)),
                     **{f"ASRF_s{k}": m for k, m in zip(SEEDS, asrf_maj)}, "ASRF": asrf_vote,
                     "margin_E0": float((f0.logit[:, A5C] - f0.logit[:, A4C]).mean()), "margin_E1": float((f1.logit[:, A5C] - f1.logit[:, A4C]).mean())})
        e = f1.feat.astype(np.float64).mean(0); embs.append(e / np.linalg.norm(e))
    df = pd.DataFrame(rows); emb = np.stack(embs); df.to_csv(out / "clips.csv", index=False)
    ac = df[df["label"].isin([A4C, A5C])].reset_index(drop=True); ac_emb = emb[df["label"].isin([A4C, A5C]).to_numpy()]
    res = {"training_clips": int(len(df)), "sessions": int(df["session"].nunique()),
           "per_class": {NAMES[c]: {"clips": int((df.label == c).sum()), "sessions": int(df[df.label == c]["session"].nunique())} for c in NAMES},
           "systems": {}}
    wrong = {}
    for sysn in ("E0", "E1", "ASRF"):
        r = {}
        for name, lab, other in (("A5C read as A4C", A5C, A4C), ("A4C read as A5C", A4C, A5C)):
            m = (ac["label"] == lab) & (ac[sysn] == other); wrong[(sysn, name)] = set(ac[m]["video_id"])
            r[name] = concentration(ac[m])
        for lab in (A4C, A5C):
            sub = ac[ac["label"] == lab]
            r[f"{NAMES[lab]} clip accuracy"] = float((sub[sysn] == lab).mean())
            r[f"{NAMES[lab]} read as another class (not A4C/A5C)"] = int((~sub[sysn].isin([A4C, A5C]) & (sub[sysn] != -1)).sum())
        res["systems"][sysn] = r
    res["ASRF_seed_disagreement_clips"] = int((ac[[f"ASRF_s{k}" for k in SEEDS]].nunique(axis=1) > 1).sum())
    confused = {v for sysn in ("E1", "ASRF") for d in ("A5C read as A4C", "A4C read as A5C") for v in wrong[(sysn, d)]}
    count = {v: sum(v in wrong[(s, d)] for s in ("E0", "E1", "ASRF") for d in ("A5C read as A4C", "A4C read as A5C")) for v in confused}
    cf = ac[ac["video_id"].isin(confused)]
    res["confused_by_E1_or_ASRF"] = {**concentration(cf), "by_direction": {d: int(sum(v in confused and v in (wrong[("E1", d)] | wrong[("ASRF", d)]) for v in confused))
                                                                           for d in ("A5C read as A4C", "A4C read as A5C")},
                                     "confused_by_2_or_3_systems": int(sum(c >= 2 for c in count.values())),
                                     "confused_by_all_3": int(sum(c >= 3 for c in count.values()))}
    res["correct_vs_confused"] = {}
    for lab in (A4C, A5C):
        sub = ac[ac["label"] == lab]; bad = sub["video_id"].isin(confused)
        res["correct_vs_confused"][NAMES[lab]] = {k: {"correct": describe(sub[~bad][k]), "confused": describe(sub[bad][k])}
                                                  for k in ("margin_E1", "margin_E0", "seconds")}
    res["cross_clip_similarity_E1"] = pair_similarity(ac_emb, ac["label"].to_numpy(), ac["session"].to_numpy())
    c = res["confused_by_E1_or_ASRF"]; ps = res["cross_clip_similarity_E1"]
    s3a, s3b = ps["same class, other session (sessions with both)"]["mean"], ps["A4C-A5C, same session (sessions with both)"]["mean"]
    res["decision"] = {"S1_breadth": bool(c["sessions"] >= 10 and (c["top3_sessions_share"] or 1.0) < 0.5),
                       "S2_shared": bool(c["clips"] > 0 and c["confused_by_2_or_3_systems"] >= 0.5 * c["clips"]),
                       "S3_cross_clip": bool(s3a is not None and s3b is not None and s3a < s3b)}
    res["decision"]["supported"] = all(res["decision"].values())
    (out / "summary.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
