#!/usr/bin/env python
"""Short-segment probe (configs/experiments/short_segment_probe_v1.yaml; docs/superpowers/plans/2026-10-07-short-segment-probe.md).

Question: in a decoded pattern X -> Y -> X, is the middle run a true short view segment (keep) or a wrong prediction
inside one view (remove), and do frozen visual features add information beyond the existing statistics? Training
recordings only; nothing enters the full system.

  --build   Session partition of the training clips (train / calibration / held-out), new constructed streams inside
            each part, the fixed generator (J0, kappa 8) on the L1_s{k} probabilities, the X -> Y -> X candidates with
            their reference category ((a) kept, (b) removed, (c) other) and splice flag, and the Q0 / Q1 features.
            Writes the libraries (server) and the composition report (counts only). No discriminator is trained.
  --train   Q0, Q1 and the diagnostic Q0+sim: L2 logistic regression with C chosen by grouped CV on the train part,
            operating points on the calibration part, evaluation on the held-out part with the session/stream
            bootstrap, the subgroups and the fixed decision.

  python scripts/short_segment_probe.py --config configs/experiments/short_segment_probe_v1.yaml --build
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402
from scripts.ev9v_complementarity import md_table  # noqa: E402

import numpy as np  # noqa: E402

from echo_routing.compose.recipe_builder import Pool, make_multi_recipes, recipe_from_dict, recipe_to_dict  # noqa: E402
from echo_routing.compose.stream_assembler import render  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.metrics_segmentation import temporal_iou  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.temporal.baselines.b3_js_divergence import boundary_scores  # noqa: E402
from echo_routing.temporal.dev_data import FeatureStore  # noqa: E402
from echo_routing.temporal.dev_run import load_json  # noqa: E402
from echo_routing.temporal.joint_decoding import joint_viterbi  # noqa: E402
from echo_routing.temporal.segments import Segment, labels_to_segments  # noqa: E402

NS = "short_segment_probe_v1"
PARTS = ("train", "calibration", "held_out")
CATS = ("kept", "removed", "other")
CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
MODELS = ("Q0", "Q1", "Q0+sim")
A4C, A5C = 2, 3


# ----------------------------------------------------------------------------------------------- pure helpers
def partition_sessions(sessions, fractions: dict, seed: int) -> dict[str, str]:
    """Every session to one part, by a seeded permutation of the sorted session ids."""
    order = sorted(set(sessions)); perm = np.random.default_rng(seed).permutation(len(order))
    n_tr = round(fractions["train"] * len(order)); n_cal = round(fractions["calibration"] * len(order))
    return {order[i]: ("train" if rank < n_tr else "calibration" if rank < n_tr + n_cal else "held_out") for rank, i in enumerate(perm)}


def sandwiches(pred: np.ndarray) -> list[tuple[Segment, Segment, Segment]]:
    """Consecutive runs (left, middle, right) whose outer runs share a class the middle's class differs from."""
    runs = labels_to_segments(np.asarray(pred, int))
    return [(runs[i - 1], runs[i], runs[i + 1]) for i in range(1, len(runs) - 1) if runs[i - 1].label == runs[i + 1].label != runs[i].label]


def majority(labels: np.ndarray) -> int:
    return int(np.bincount(np.asarray(labels, int), minlength=len(CLASSES)).argmax())


def categorise(left: Segment, mid: Segment, right: Segment, ref: np.ndarray, joins: np.ndarray, tol: int) -> tuple[str, bool]:
    """(a) kept, (b) removed or (c) other by the configuration's rules, and whether a physical join lies within tol of
    the middle's span."""
    x, y = left.label, mid.label; ref = np.asarray(ref, int)
    sides = majority(ref[left.start:left.end]) == x and majority(ref[right.start:right.end]) == x
    units = labels_to_segments(ref)
    if sides and any(u.label == y and temporal_iou(mid, u) >= 0.5 for u in units):
        cat = "kept"
    elif sides and bool(np.all(ref[mid.start:mid.end] == x)):
        cat = "removed"
    else:
        cat = "other"
    j = np.asarray(joins, int)
    return cat, bool(j.size and ((j >= mid.start - tol) & (j <= mid.end + tol)).any())


def side_windows(left: Segment, mid: Segment, right: Segment, w: int) -> tuple[slice, slice]:
    return slice(max(left.start, mid.start - w), mid.start), slice(mid.end, min(right.end, mid.end + w))


def q0_features(prob: np.ndarray, jsd: np.ndarray, left: Segment, mid: Segment, right: Segment, w: int, eps: float) -> np.ndarray:
    lw, rw = side_windows(left, mid, right, w); x, y = left.label, mid.label
    p = np.clip(np.asarray(prob, np.float64), eps, 1.0); pm = p[mid.start:mid.end]
    onehot = np.eye(len(CLASSES))
    return np.concatenate([[math.log(mid.length), math.log(left.length), math.log(right.length)], onehot[x], onehot[y],
                           pm.mean(0), p[lw].mean(0), p[rw].mean(0),
                           [pm[:, y].mean(), (pm[:, y] - pm[:, x]).mean(), (np.log(pm[:, y]) - np.log(pm[:, x])).mean(), jsd[mid.start], jsd[mid.end]]])


def visual_features(feat: np.ndarray, left: Segment, mid: Segment, right: Segment, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean frozen features of the middle and both side windows, and the cosine similarities middle-left, middle-right."""
    lw, rw = side_windows(left, mid, right, w); f = np.asarray(feat, np.float32)
    fm, fl, fr = f[mid.start:mid.end].mean(0), f[lw].mean(0), f[rw].mean(0)
    cos = lambda a, b: float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))  # noqa: E731
    return np.concatenate([fm, fl, fr]), np.array([cos(fm, fl), cos(fm, fr)])


def threshold_for(scores_kept: np.ndarray, keep_rate: float) -> float:
    """The highest threshold whose keep rate of (a) is at least keep_rate (kept when score >= threshold)."""
    s = np.sort(np.asarray(scores_kept, float))[::-1]
    if s.size == 0:
        raise ValueError("no (a) candidates to set the operating point")
    return float(s[math.ceil(keep_rate * s.size) - 1])


def rates(scores: np.ndarray, y: np.ndarray, t: float) -> dict:
    s, y = np.asarray(scores, float), np.asarray(y, int)
    return {"keep_rate_kept": float((s[y == 1] >= t).mean()) if (y == 1).any() else None,
            "removal_rate_removed": float((s[y == 0] < t).mean()) if (y == 0).any() else None,
            "n_kept": int((y == 1).sum()), "n_removed": int((y == 0).sum())}


def bootstrap_diff(sa: np.ndarray, sb: np.ndarray, y: np.ndarray, ta: float, tb: float, groups: np.ndarray, draws: int, seed: int) -> dict:
    """Removal rate of (b) and keep rate of (a), a - b at their thresholds, resampling groups with the same draws."""
    groups = np.asarray(groups); keys = np.unique(groups); idx = {k: np.flatnonzero(groups == k) for k in keys}
    rng = np.random.default_rng(seed); point = {}; out = {"removal": [], "keep": []}
    ra, rb = rates(sa, y, ta), rates(sb, y, tb)
    point = {"removal": ra["removal_rate_removed"] - rb["removal_rate_removed"], "keep": ra["keep_rate_kept"] - rb["keep_rate_kept"]}
    for _ in range(draws):
        ii = np.concatenate([idx[k] for k in rng.choice(keys, size=keys.size, replace=True)]); yy = y[ii]
        if (yy == 0).any():
            out["removal"].append(float((sa[ii][yy == 0] < ta).mean() - (sb[ii][yy == 0] < tb).mean()))
        if (yy == 1).any():
            out["keep"].append(float((sa[ii][yy == 1] >= ta).mean() - (sb[ii][yy == 1] >= tb).mean()))
    return {k: {"delta": point[k], "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in out.items()}


def sufficient(meta_held_out: list[dict], need: dict) -> dict:
    out = {}
    for cat in ("kept", "removed"):
        rows = [m for m in meta_held_out if m["cat"] == cat]
        out[cat] = {"candidates": len(rows), "sessions": len({s for m in rows for s in m["mid_sessions"]})}
    out["ok"] = all(out[c]["candidates"] >= need["candidates"] and out[c]["sessions"] >= need["sessions"] for c in ("kept", "removed"))
    return out


def decide(per_seed: list[dict]) -> dict:
    """The configuration's decision on the three seeds' held-out results at the 0.95 point."""
    checks = []
    for r in per_seed:
        d = r["Q1 - Q0"]["0.95"]
        checks.append({"removal_above_zero": d["bootstrap"]["removal"]["ci95"][0] > 0, "keep_within_2_points": d["bootstrap"]["keep"]["delta"] >= -0.02,
                       "splice_control_not_negative": (d["splice_removed_delta"] is None) or d["splice_removed_delta"] >= 0})
    return {"extra_value": bool(per_seed) and all(all(c.values()) for c in checks), "per_seed": checks}


# ----------------------------------------------------------------------------------------------- build
def check_inputs(sp: dict, root: Path) -> dict:
    out = {}
    specs = {"protocol": sp["inputs"]["protocol"], **{f"b_feature_manifest_s{k}": v for k, v in sp["inputs"]["b_feature_manifests"].items()}}
    for name, spec in specs.items():
        if (got := sha256_file(root / spec["path"])) != spec["sha256"]:
            raise SystemExit(f"input {name}: sha256 {got[:12]} != {spec['sha256'][:12]}")
        out[name] = got
    return out


def make_banks(cfg: dict, sp: dict, paths, proto: dict, labels: dict, n_samples: dict) -> tuple[dict, dict]:
    """The session partition and, per part, new constructed streams and the native clips."""
    train = list(proto["train"]["clips"]); part_of = partition_sessions([session_of(v) for v in train], sp["partition"]["fractions"], int(sp["partition"]["seed"]))
    mc, rc, st = cfg["multi"], cfg["recipes"], sp["streams"]; banks = {}
    for part in PARTS:
        clips = sorted(v for v in train if part_of[session_of(v)] == part)
        pool = Pool.from_index(clips, [labels[v] for v in clips], [n_samples[v] for v in clips])
        rec = make_multi_recipes(pool, int(st["n_streams"][part]), np.random.default_rng(int(st["seeds"][part])), n_fragments=tuple(mc["n_fragments"]),
                                 edit_variant=rc["edit_variant"], edit_prob=float(mc["edit_prob"]), min_len=int(rc["min_len"]), max_len=int(rc["max_len"]),
                                 reuse=True, p_same_next=float(mc["p_same_next"]))
        for r in rec:
            if any(part_of[session_of(f.video_id)] != part for f in r.fragments):
                raise SystemExit("a constructed stream crosses parts")
        banks[part] = {"recipes": [recipe_to_dict(r) for r in rec], "natives": clips}
    return part_of, banks


def sequences(part: str, bank: dict, store, labels: dict):
    """(group, kind, prob, feat, reference labels, joins, fragment bounds, fragment videos) of every sequence of a part."""
    for d in bank["recipes"]:
        r = recipe_from_dict(d); s = render(r, store.get)
        bounds = np.cumsum([0, *[f.end - f.start for f in r.fragments]])
        yield f"stream:{r.recipe_id}", "stream", s.prob, s.feat, s.labels, s.joins, bounds, [f.video_id for f in r.fragments]
    for v in bank["natives"]:
        vf = store.get(v, "orig")
        yield f"session:{session_of(v)}", "native", vf.prob, vf.feat, np.full(vf.n, labels[v]), np.zeros(0, int), np.array([0, vf.n]), [v]


def videos_over(bounds: np.ndarray, videos: list[str], start: int, end: int) -> list[str]:
    return sorted({videos[k] for k in range(len(videos)) if bounds[k] < end and bounds[k + 1] > start})


def build_library(sp: dict, gen: dict, store, banks: dict, labels: dict) -> tuple[dict, list[dict]]:
    w, eps, tol = int(sp["features"]["side_window"]), float(gen["eps"]), int(sp["candidates"]["tolerance_samples"])
    q0, vis, sim, meta = [], [], [], []
    for part in PARTS:
        for group, kind, prob, feat, ref, joins, bounds, videos in sequences(part, banks[part], store, labels):
            pred = joint_viterbi(prob, np.zeros(len(prob)), 0.0, float(gen["kappa"]), eps); jsd = boundary_scores(prob, 7)
            for left, mid, right in sandwiches(pred):
                cat, splice = categorise(left, mid, right, ref, joins, tol)
                q0.append(q0_features(prob, jsd, left, mid, right, w, eps)); v, c = visual_features(feat, left, mid, right, w); vis.append(v); sim.append(c)
                mv = videos_over(bounds, videos, mid.start, mid.end); av = videos_over(bounds, videos, left.start, right.end)
                meta.append({"part": part, "group": group, "kind": kind, "cat": cat, "splice": splice, "x": left.label, "y": mid.label,
                             "len_mid": mid.length, "len_left": left.length, "len_right": right.length,
                             "conf_mid": float(np.asarray(prob, float)[mid.start:mid.end, mid.label].mean()), "mid_videos": mv,
                             "mid_sessions": sorted({session_of(x) for x in mv}), "span_videos": av, "crosses_recordings": len(av) > 1})
    arrays = {"q0": np.asarray(q0, np.float32), "vis": np.asarray(vis, np.float32), "sim": np.asarray(sim, np.float32)}
    return arrays, meta


def describe(meta: list[dict]) -> dict:
    """Counts by part and category, with recordings and sessions, lengths, class pairs, confidence and splice flags."""
    out = {}
    for part in PARTS:
        out[part] = {}
        for cat in CATS:
            rows = [m for m in meta if m["part"] == part and m["cat"] == cat]
            lens = np.array([m["len_mid"] for m in rows]) if rows else np.zeros(0)
            out[part][cat] = {"candidates": len(rows), "recordings": len({v for m in rows for v in m["mid_videos"]}),
                              "sessions": len({s for m in rows for s in m["mid_sessions"]}), "from_streams": sum(m["kind"] == "stream" for m in rows),
                              "from_natives": sum(m["kind"] == "native" for m in rows), "at_splice": sum(m["splice"] for m in rows),
                              "crossing_recordings": sum(m["crosses_recordings"] for m in rows),
                              "len_mid_quartiles": [float(q) for q in np.percentile(lens, [25, 50, 75])] if rows else None,
                              "len_mid_10_40": int(((lens >= 10) & (lens <= 40)).sum()),
                              "conf_mid_median": float(np.median([m["conf_mid"] for m in rows])) if rows else None,
                              "a4c_a5c_pairs": sum({m["x"], m["y"]} == {A4C, A5C} for m in rows),
                              "class_pairs": {f"{CLASSES[x]}>{CLASSES[y]}": n for (x, y), n in sorted(Counter((m["x"], m["y"]) for m in rows).items())}}
    return out


def run_build(cfg: dict, sp: dict, paths) -> None:
    root = paths.repo_root; t0 = time.time(); inputs = check_inputs(sp, root); proto, _ = load_json(root / sp["inputs"]["protocol"]["path"])
    if not proto["train"]["clips"]:
        raise SystemExit("no training clips")
    out = root / "runs" / NS; out.mkdir(parents=True, exist_ok=True); comp = {"inputs": inputs, "seeds": {}}
    for seed in [int(s) for s in sp["seeds"]]:
        fman, _ = load_json(root / sp["inputs"]["b_feature_manifests"][seed]["path"]); key = fman["encoder"]["cache_key"]
        idx = read_index(variant_dir(paths.cache_root, key, "orig")).set_index("video_id")
        labels = {v: int(idx.loc[v, "label_index"]) for v in proto["train"]["clips"]}; n_samples = {v: int(idx.loc[v, "n_samples"]) for v in proto["train"]["clips"]}
        if seed == int(sp["seeds"][0]):
            part_of, banks = make_banks(cfg, sp, paths, proto, labels, n_samples)
            (out / "partition.json").write_text(json.dumps(part_of, indent=0)); (out / "banks.json").write_text(json.dumps(banks))
            comp["partition"] = {p: {"sessions": sum(v == p for v in part_of.values()),
                                     "clips_by_class": dict(Counter(CLASSES[labels[c]] for c in banks[p]["natives"])),
                                     "streams": len(banks[p]["recipes"])} for p in PARTS}
        arrays, meta = build_library(sp, sp["generator"], FeatureStore(paths.cache_root, key), banks, labels)
        np.savez_compressed(out / f"library_s{seed}.npz", **arrays); (out / f"library_meta_s{seed}.json").write_text(json.dumps(meta))
        comp["seeds"][str(seed)] = {"cache_key": key, "composition": describe(meta),
                                    "sufficiency": sufficient([m for m in meta if m["part"] == "held_out"], {"candidates": 30, "sessions": 15})}
        print(json.dumps({"seed": seed, "candidates": len(meta), "held_out": comp["seeds"][str(seed)]["sufficiency"]}), flush=True)
    comp["seconds"] = time.time() - t0; (out / "composition.json").write_text(json.dumps(comp, indent=1)); print(f"wrote {out / 'composition.json'}")


# ----------------------------------------------------------------------------------------------- train
def design(arrays: dict, model: str) -> np.ndarray:
    if model == "Q0":
        return arrays["q0"]
    if model == "Q1":
        return np.concatenate([arrays["q0"], arrays["vis"], arrays["sim"]], axis=1)
    if model == "Q0+sim":
        return np.concatenate([arrays["q0"], arrays["sim"]], axis=1)
    raise ValueError(model)


def fit(x: np.ndarray, y: np.ndarray, groups: np.ndarray, grid: list[float]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    make = lambda c: make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))  # noqa: E731
    cv = {c: float(np.mean(cross_val_score(make(c), x, y, groups=groups, cv=GroupKFold(n_splits=5), scoring="neg_log_loss"))) for c in grid}
    best = max(cv.values()); c = min(k for k, v in cv.items() if v == best)
    return make(c).fit(x, y), c, cv


def evaluate_seed(arrays: dict, meta: list[dict], sp: dict) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    part = np.array([m["part"] for m in meta]); cat = np.array([m["cat"] for m in meta]); groups = np.array([m["group"] for m in meta])
    use = cat != "other"; y = (cat == "kept").astype(int); res = {"models": {}}
    masks = {"all": np.ones(len(meta), bool), "A4C/A5C pairs": np.array([{m["x"], m["y"]} == {A4C, A5C} for m in meta]),
             "other pairs": np.array([{m["x"], m["y"]} != {A4C, A5C} for m in meta]), "middle 10-40 samples": np.array([10 <= m["len_mid"] <= 40 for m in meta])}
    splice = np.array([m["splice"] for m in meta]); scores = {}
    for name in MODELS:
        x = design(arrays, name); tr = use & (part == "train")
        model, c, cv = fit(x[tr], y[tr], groups[tr], [float(v) for v in sp["model"]["C_grid"]]); s = model.predict_proba(x)[:, 1]; scores[name] = s
        cal, ho = use & (part == "calibration"), use & (part == "held_out")
        th = {str(k): threshold_for(s[cal & (y == 1)], float(k)) for k in sp["operating_points"]["keep_rates"]}
        r = {"C": c, "cv_neg_log_loss": cv, "thresholds": th, "held_out_auc": float(roc_auc_score(y[ho], s[ho])),
             "held_out_average_precision_removed": float(average_precision_score(1 - y[ho], -s[ho])), "points": {}}
        for k, t in th.items():
            r["points"][k] = {g: rates(s[ho & m], y[ho & m], t) for g, m in masks.items()}
            r["points"][k]["removed at a splice"] = rates(s[ho & splice & (y == 0)], y[ho & splice & (y == 0)], t)
            r["points"][k]["removed inside a clip"] = rates(s[ho & ~splice & (y == 0)], y[ho & ~splice & (y == 0)], t)
            r["points"][k]["calibration"] = rates(s[cal], y[cal], t)
        res["models"][name] = r
    ho = use & (part == "held_out")
    for a, b in (("Q1", "Q0"), ("Q0+sim", "Q0"), ("Q1", "Q0+sim")):
        res[f"{a} - {b}"] = {}
        for k in res["models"][a]["thresholds"]:
            ta, tb = res["models"][a]["thresholds"][k], res["models"][b]["thresholds"][k]
            boot = bootstrap_diff(scores[a][ho], scores[b][ho], y[ho], ta, tb, groups[ho], int(sp["evaluation"]["bootstrap"]["draws"]), int(sp["evaluation"]["bootstrap"]["seed"]))
            sm = ho & splice & (y == 0)
            sub = {g: (rates(scores[a][ho & m], y[ho & m], ta)["removal_rate_removed"], rates(scores[b][ho & m], y[ho & m], tb)["removal_rate_removed"]) for g, m in masks.items()}
            res[f"{a} - {b}"][k] = {"bootstrap": boot, "subgroup_removal": {g: (None if None in v else v[0] - v[1]) for g, v in sub.items()},
                                    "splice_removed_delta": (float((scores[a][sm] < ta).mean() - (scores[b][sm] < tb).mean()) if sm.any() else None),
                                    "inside_removed_delta": (float((scores[a][ho & ~splice & (y == 0)] < ta).mean() - (scores[b][ho & ~splice & (y == 0)] < tb).mean())
                                                             if (ho & ~splice & (y == 0)).any() else None)}
    return res


def run_train(sp: dict, paths, out_dir: Path) -> None:
    root = paths.repo_root; lib = root / "runs" / NS; comp = json.loads((lib / "composition.json").read_text()); res = {"seeds": {}, "composition": comp}
    seeds = [int(s) for s in sp["seeds"]]; enough = all(comp["seeds"][str(s)]["sufficiency"]["ok"] for s in seeds)
    if not enough:
        res["decision"] = {"extra_value": False, "reason": "insufficient examples in the held-out part"}
    else:
        for s in seeds:
            with np.load(lib / f"library_s{s}.npz") as z:
                arrays = {k: z[k] for k in z.files}
            res["seeds"][str(s)] = evaluate_seed(arrays, json.loads((lib / f"library_meta_s{s}.json").read_text()), sp); print(f"seed {s} trained", flush=True)
        res["decision"] = decide([{"Q1 - Q0": {"0.95": res["seeds"][str(s)]["Q1 - Q0"]["0.95"]}} for s in seeds])
    out_dir.mkdir(parents=True, exist_ok=True); (out_dir / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    (out_dir / "report_tables.md").write_text(tables_md(res)); print(json.dumps(res["decision"], indent=1)); print(f"wrote {out_dir}")


def tables_md(res: dict) -> str:
    rows, comps = [], []
    for s, r in res["seeds"].items():
        for name, m in r["models"].items():
            for k, pts in m["points"].items():
                a = pts["all"]; rows.append({"seed": s, "model": name, "keep target": k, "C": m["C"], "AUC": m["held_out_auc"],
                                             "keep (a)": a["keep_rate_kept"], "remove (b)": a["removal_rate_removed"],
                                             "remove (b) at splice": pts["removed at a splice"]["removal_rate_removed"],
                                             "remove (b) inside": pts["removed inside a clip"]["removal_rate_removed"],
                                             "remove (b), 10-40": pts["middle 10-40 samples"]["removal_rate_removed"],
                                             "remove (b), A4C/A5C": pts["A4C/A5C pairs"]["removal_rate_removed"]})
        for name in ("Q1 - Q0", "Q0+sim - Q0", "Q1 - Q0+sim"):
            for k, d in r[name].items():
                b = d["bootstrap"]
                comps.append({"seed": s, "contrast": name, "keep target": k,
                              "removal (b)": f"{100 * b['removal']['delta']:+.1f} [{100 * b['removal']['ci95'][0]:+.1f}, {100 * b['removal']['ci95'][1]:+.1f}]",
                              "keep (a)": f"{100 * b['keep']['delta']:+.1f} [{100 * b['keep']['ci95'][0]:+.1f}, {100 * b['keep']['ci95'][1]:+.1f}]",
                              "splice (b)": d["splice_removed_delta"], "inside (b)": d["inside_removed_delta"]})
    return "\n".join(["# Short-segment probe: tables\n", "## Held-out rates\n", md_table(rows), "## Contrasts (points)\n", md_table(comps),
                      "## Decision\n", "```\n" + json.dumps(res["decision"], indent=1) + "\n```\n"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True); ap.add_argument("--out", default=None)
    step = ap.add_mutually_exclusive_group(required=True); step.add_argument("--build", action="store_true"); step.add_argument("--train", action="store_true")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); sp = cfg["short_segment_probe"]; paths = load_paths()
    if a.build:
        run_build(cfg, sp, paths)
    else:
        if not a.out:
            raise SystemExit("--out is required for --train")
        run_train(sp, paths, paths.repo_root / a.out)


if __name__ == "__main__":
    main()
