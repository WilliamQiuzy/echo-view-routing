#!/usr/bin/env python
"""Where are Pediatric view changes lost? Bounded fault localisation on the frozen test exam streams (2026-10-05). No
training, no threshold change: the existing models and their own decoding rules only.

Systems, per seed k: D0 + ASRF (E1's ASRF seed k on E1 features), D2 + ASRF (D2's ASRF seed k on D2 features) and
D2 + B3 (D2's head with its EV9V-selected B3 parameters), as in scripts/pediatric_adapt_report.py.

Decoding path. ASRF: the boundary head's probability; candidate cuts are its accepted peaks (argrelmax at the head's
own threshold); each span between candidates takes the class vote; equal neighbours merge. B3: candidate boundaries
from its divergence detector, then each span's mean-probability class; equal neighbours merge. Final label changes can
only fall on candidates. For every true view change r (a join between different views), the first stage that loses it:
  found          a final boundary within 0.5 s (one-to-one matching, as boundary recall); split by both views right
  merged         a candidate within 0.5 s, but the spans on both sides got the same class and merged
  displaced      no candidate within 0.5 s, but an unmatched final boundary 0.5-1.5 s away
  no candidate   no candidate within 0.5 s and no final boundary within 1.5 s
Also: extra cuts (final boundaries more than 0.5 s from every true change), inside a clip or at a same-view join; for
ASRF the boundary probability near missed changes against the head's threshold; the class a merged span took; and,
per seed, which losses D0 + ASRF already had and which appear only after D2. Aggregates only (counts and groups).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pediatric_adapt_report import FEAT, b3_params, majority, sources  # noqa: E402
from scripts.pediatric_eval import assemble  # noqa: E402
from scripts.pediatric_extract import verify_protocol  # noqa: E402
from scripts.round4_failure_inventory import load_asrf  # noqa: E402
from scripts.round5_reachability import asrf_decode, b3_decode  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from echo_routing.compose.recipe_builder import recipe_from_dict  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.metrics_boundary import match_boundaries  # noqa: E402
from echo_routing.realvideo.pediatric import recipe_group  # noqa: E402
from echo_routing.temporal.segments import boundaries_from_labels  # noqa: E402

SEEDS, TOL, WIDE = (0, 1, 2), 5, 15
CATS = ("found, both views right", "found, a view wrong", "merged", "displaced", "no candidate")
SYSTEMS = ("D0 + ASRF", "D2 + ASRF", "D2 + B3")


@torch.no_grad()
def asrf_boundary_prob(model, feat: np.ndarray, device) -> np.ndarray:
    x = torch.from_numpy(feat.astype(np.float16).T.astype(np.float32))[None].to(device)
    return torch.sigmoid(model(x)[1]).cpu().numpy()[0, 0]


def classify(final: np.ndarray, cuts: list[int], s) -> tuple[list[dict], dict]:
    sem = np.asarray(s.semantic_boundaries, int); pred_b = boundaries_from_labels(final); joins = np.asarray(s.joins, int)
    pairs = match_boundaries(pred_b, sem, TOL); hit = {int(sem[j]) for _, j in pairs}; used = {int(pred_b[i]) for i, _ in pairs}
    starts, ends = np.r_[0, joins], np.r_[joins, len(final)]
    maj = [majority(final[a:b]) for a, b in zip(starts, ends)]; truth = [int(s.labels[a]) for a in starts]
    pos = {int(j): k for k, j in enumerate(joins, start=1)}; cuts = np.asarray(cuts, int); free = [int(b) for b in pred_b if int(b) not in used]
    rows = []
    for r in sem.tolist():
        k = pos[r]
        if r in hit:
            cat = CATS[0] if maj[k - 1] == truth[k - 1] and maj[k] == truth[k] else CATS[1]
        elif cuts.size and np.abs(cuts - r).min() <= TOL:
            cat = "merged"
        elif any(TOL < abs(b - r) <= WIDE for b in free):
            cat = "displaced"
        else:
            cat = "no candidate"
        took = int(final[r])
        rows.append({"r": r, "cat": cat, "merged_into": ("left view" if took == truth[k - 1] else "right view" if took == truth[k] else "neither view") if cat == "merged" else None})
    extra = [b for b in free if sem.size == 0 or np.abs(sem - b).min() > TOL]
    inside = sum(1 for b in extra if joins.size == 0 or np.abs(joins - b).min() > TOL)
    return rows, {"extra_inside_clip": inside, "extra_at_same_view_join": len(extra) - inside}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="runs/md_round4/spec.json"); ap.add_argument("--out", required=True)
    ap.add_argument("--protocol-dir", default="data/manifests/pediatric/pediatric-external-v1")
    a = ap.parse_args(); paths = load_paths(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    proto_dir = Path(a.protocol_dir); verify_protocol(proto_dir); spec = json.loads(Path(a.spec).read_text())
    recipes = [recipe_from_dict(d) for d in json.loads((proto_dir / "recipes_test_exam.json").read_text())]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    res = {"streams": len(recipes), "tolerance_s": TOL / 10, "wide_s": WIDE / 10, "seeds": {}}
    for k in SEEDS:
        src = sources(spec, k); per_ref = {}; seed_res = {}
        setups = {"D0 + ASRF": ("D0", "asrf"), "D2 + ASRF": ("D2", "asrf"), "D2 + B3": ("D2", "b3")}
        for name, (arm, kind) in setups.items():
            d = paths.cache_root / FEAT / src[arm]["feat"]; clips = {}
            for rec in recipes:
                for fr in rec.fragments:
                    if fr.video_id not in clips:
                        with np.load(d / f"{fr.video_id}.npz") as z:
                            clips[fr.video_id] = (z["prob"].astype(float), z["feat"].astype(np.float32))
            asrf = load_asrf(Path(src[arm]["asrf"]), paths.repo_root, 512, device) if kind == "asrf" else None
            b3 = b3_params(src[arm]["b3"]) if kind == "b3" else None
            cats, groups, extra, merged_into, near = Counter(), defaultdict(set), Counter(), Counter(), []
            for i, rec in enumerate(recipes):
                s = assemble(clips, rec, "true", 0); g = recipe_group(rec)
                if kind == "asrf":
                    final, cuts = asrf_decode(*asrf, s.feat.astype(np.float16), device); bprob = asrf_boundary_prob(asrf[1], s.feat, device)
                else:
                    final, cuts = b3_decode(s.prob, b3); bprob = None
                rows, ex = classify(np.asarray(final, int), cuts, s); extra.update(ex)
                for row in rows:
                    cats[row["cat"]] += 1; groups[row["cat"]].add(g); per_ref[(name, i, row["r"])] = row["cat"]
                    if row["merged_into"]:
                        merged_into[row["merged_into"]] += 1
                    if kind == "asrf" and row["cat"] in ("no candidate", "displaced"):
                        near.append(float(bprob[max(0, row["r"] - TOL): row["r"] + TOL + 1].max()))
            n = sum(cats.values()); th = float(asrf[0].boundary_th) if kind == "asrf" else None
            seed_res[name] = {"changes": n, "stages": {c: {"n": cats[c], "share": cats[c] / n, "groups": len(groups[c])} for c in CATS},
                              "extra_cuts": dict(extra), "merged_into": dict(merged_into), "boundary_threshold": th,
                              "missed_boundary_prob": None if not near else {"n": len(near), "median": float(np.median(near)),
                                                                            "share_at_or_above_threshold": float(np.mean(np.array(near) >= th)),
                                                                            "share_below_half_threshold": float(np.mean(np.array(near) < th / 2))}}
            print(f"seed {k} {name}: " + ", ".join(f"{c} {cats[c]}" for c in CATS), flush=True)
        lost = lambda name, key: per_ref[(name,) + key] not in CATS[:2]  # noqa: E731
        keys = [key[1:] for key in per_ref if key[0] == "D0 + ASRF"]
        both = [x for x in keys if lost("D0 + ASRF", x) and lost("D2 + ASRF", x)]
        new = [x for x in keys if lost("D2 + ASRF", x) and not lost("D0 + ASRF", x)]
        fixed = [x for x in keys if lost("D0 + ASRF", x) and not lost("D2 + ASRF", x)]
        seed_res["D0 vs D2 ASRF"] = {"lost by both": len(both), "lost only after D2": len(new), "lost only before (fixed by D2)": len(fixed),
                                     "stage of losses new after D2": dict(Counter(per_ref[("D2 + ASRF",) + x] for x in new)),
                                     "stage of losses shared, D2": dict(Counter(per_ref[("D2 + ASRF",) + x] for x in both)),
                                     "of D2 + ASRF losses, found by D2 + B3": int(sum(1 for x in keys if lost("D2 + ASRF", x) and not lost("D2 + B3", x)))}
        res["seeds"][k] = seed_res
    (out / "boundary_faults.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res["seeds"][0]["D0 vs D2 ASRF"], indent=1))


if __name__ == "__main__":
    main()
