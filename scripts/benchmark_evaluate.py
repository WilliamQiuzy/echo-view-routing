#!/usr/bin/env python
"""Evaluator of the frozen EV9V test benchmark (metrics version ev9v-benchmark-metrics-v1).

It takes per-item label arrays on the benchmark's 10 Hz axis and nothing else from a model. Scoring runs through H09's
own code (scripts/true_order_joint.py: evaluate_spec on the same NativeCine / Stream objects, the same bootstrap draws),
with a registered decoder that only returns the supplied labels. Added endpoints (accuracy, per-class recall, A4C/A5C
confusions, multi boundary precision, pair boundary recall, multi FP / FN with short segments) reuse the same units, so
H09's intervals are unchanged. Reference labels are read here only.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.benchmark_manifest as bm  # noqa: E402
import scripts.true_order_joint as tj  # noqa: E402
from scripts.permutation_mechanism import Frozen, assemble, natives  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.compose.banks import multi_bank, pair_bank  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.metrics_segmentation import segment_f1, temporal_iou  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.temporal.baselines.registry import AUDITED_B3, register  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402

HZ, SHORT, CLASSES = 10.0, 16, ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
A4C, A5C = 2, 3


def match(pred: list, true: list) -> tuple[set, set]:
    """segment_f1's matching (same class, IoU >= 0.5, one-to-one, greedy by IoU), returning the matched indices."""
    cands = sorted(((temporal_iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(true) if p.label == t.label and temporal_iou(p, t) >= 0.5), reverse=True)
    mp, mt = set(), set()
    for _, i, j in cands:
        if i not in mp and j not in mt:
            mp.add(i); mt.add(j)
    return mp, mt


def seg_counts(lab: np.ndarray, ref: np.ndarray) -> dict:
    ps, ts = labels_to_segments(lab), labels_to_segments(ref); mp, mt = match(ps, ts)
    if len(mp) != segment_f1(ps, ts, 0.5)["matched"]:
        raise SystemExit("the matcher differs from segment_f1")
    return {"fp": len(ps) - len(mp), "fn": len(ts) - len(mt), "fp_short": sum(1 for i, p in enumerate(ps) if i not in mp and p.length <= SHORT),
            "fn_short": sum(1 for j, t in enumerate(ts) if j not in mt and t.length <= SHORT), "true_short": sum(1 for t in ts if t.length <= SHORT)}


def recall_of(k: int):
    return lambda c: c[6 * k] / np.maximum(1.0, sum(c[5 * k + j] for j in range(5)))


def endpoints() -> dict:
    """H09's endpoints first (their draws are unchanged), then the added ones on the same units."""
    ep = dict(tj.endpoint_values(None)); conf = tj.CONF
    ep.update({"clip accuracy": ("nat", "group", conf, lambda c: sum(c[6 * k] for k in range(5)) / sum(c)),
               **{f"recall {CLASSES[k]}": ("nat", "group", conf, recall_of(k)) for k in range(5)},
               "A4C read as A5C (clips)": ("nat", "group", [f"c{A4C}{A5C}"], lambda c: c[0]), "A5C read as A4C (clips)": ("nat", "group", [f"c{A5C}{A4C}"], lambda c: c[0]),
               "boundary precision, multi": ("multi_frag", "video_id", ["t_matched", "p_pred"], lambda c: c[0] / c[1]),
               "boundary recall, pairs": ("pairs_frag", "stream", ["t_matched", "t_true"], lambda c: c[0] / c[1]),
               "multi FP segments": ("multi_seg", "stream", ["fp"], lambda c: c[0]), "multi FP segments <= 16": ("multi_seg", "stream", ["fp_short"], lambda c: c[0]),
               "multi FN segments": ("multi_seg", "stream", ["fn"], lambda c: c[0]), "multi missed true segments <= 16": ("multi_seg", "stream", ["fn_short"], lambda c: c[0]),
               "pairs FP segments": ("pairs_seg", "stream", ["fp"], lambda c: c[0]), "pairs FN segments": ("pairs_seg", "stream", ["fn"], lambda c: c[0])})
    return ep


class Benchmark:
    """The benchmark's H09 scoring inputs, rebuilt by H09's functions and checked against the manifest."""

    def __init__(self, cfg: dict) -> None:
        man = bm.load(cfg); self.items = man["items"]; self.by_id = {x["id"]: x for x in self.items}; bench = cfg["frozen_external_benchmark"]["benchmark"]
        self.lc = bm.ladder(bench); self.lc["decode"].setdefault("b3a", dict(AUDITED_B3)); paths = load_paths()
        idx = read_index(variant_dir(paths.cache_root, bm.ENC, "orig")); fz = Frozen(paths, bm.ENC)
        nat = natives(fz, idx, "test", "true", 0, None, "b0")[0]
        pairs = [assemble(fz, r, "true", 0, None, "b0")[0] for r in pair_bank(idx, "test", self.lc).recipes]
        multi = [assemble(fz, r, "true", 0, None, "b0")[0] for r in multi_bank(idx, "test", self.lc).recipes]
        got = [(x.video_id, [x.label] * len(x.prob)) for x in nat] + [(s.recipe.recipe_id, s.labels.tolist()) for s in pairs + multi]
        if [(x["id"], x["labels"]) for x in self.items] != got:
            raise SystemExit("H09's rebuilt scoring inputs differ from the benchmark manifest")
        self.inputs = {"b0": (nat, pairs, multi)}; self.digest = bm.digest(self.items)

    def check_preds(self, name: str, preds: dict) -> None:
        if set(preds) != set(self.by_id):
            raise SystemExit(f"{name}: predictions cover {len(set(preds) & set(self.by_id))} of {len(self.by_id)} items")
        for k, v in preds.items():
            v = np.asarray(v)
            if v.shape != (self.by_id[k]["n"],) or not np.issubdtype(v.dtype, np.integer) or v.min() < 0 or v.max() > 4:
                raise SystemExit(f"{name}: item predictions have the wrong length or values")

    def evaluate(self, name: str, preds: dict) -> dict:
        self.check_preds(name, preds); method = f"benchmark::{name}"; fixed = {k: np.asarray(v, int) for k, v in preds.items()}
        register(method, lambda prob, feat, cfg, sid=None: (fixed[sid].copy(), prob))
        ev = tj.evaluate_spec({"name": name, "method": method, "encoder": "b0"}, self.lc, self.inputs, HZ)
        for bank, k in (("pairs", 1), ("multi", 2)):
            seg = ev["banks"][bank]["segments"]; extra = [seg_counts(fixed[s.recipe.recipe_id], s.labels) for s in self.inputs["b0"][k]]
            ev["banks"][bank]["segments"] = pd.concat([seg.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
        return ev


def bootstrap(evals: dict, reference: str) -> dict:
    """H09's bootstrap loop (same seed, unit order and draw order) over the extended endpoint list."""
    rng = np.random.default_rng(tj.BOOT_SEED); tabs = {k: tj.tables(v) for k, v in evals.items()}; ref = tabs[reference]; W: dict = {}; out = {k: {} for k in evals}
    for ep, (tab, unit, cols, fn) in endpoints().items():
        units = pd.Index(sorted(ref[tab][unit].unique()))
        if (tab, unit) not in W:
            d = rng.integers(0, len(units), size=(tj.N_BOOT, len(units))); W[(tab, unit)] = np.stack([np.bincount(x, minlength=len(units)) for x in d]).astype(float)
        w = W[(tab, unit)]; reps, pts = {}, {}
        for k, t in tabs.items():
            s = t[tab].groupby(unit)[cols].sum().reindex(units, fill_value=0).to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                pts[k] = float(fn(s.sum(0))); reps[k] = fn(np.moveaxis(w @ s, -1, 0))
        for k in tabs:
            e = {"value": pts[k], "ci95": np.nanpercentile(reps[k], [2.5, 97.5]).tolist()}
            if k != reference:
                e["diff"] = pts[k] - pts[reference]; e["diff_ci95"] = np.nanpercentile(reps[k] - reps[reference], [2.5, 97.5]).tolist()
            out[k][ep] = e
    return out


def hand_case() -> dict:
    """A hand-computed matching case: truth A(0-10) B(10-14) A(14-30); prediction A(0-12) B(12-13) A(13-30)."""
    ref = np.array([0] * 10 + [1] * 4 + [0] * 16); lab = np.array([0] * 12 + [1] * 1 + [0] * 17)
    c = seg_counts(lab, ref); f = segment_f1(labels_to_segments(lab), labels_to_segments(ref), 0.5)
    # by hand: A(0-12) vs A(0-10) IoU 10/12 matched; B(12-13) vs B(10-14) IoU 1/4 unmatched; A(13-30) vs A(14-30) IoU 16/17 matched
    want = {"fp": 1, "fn": 1, "fp_short": 1, "fn_short": 1, "true_short": 3, "matched": 2, "f1": 2 * 2 / (3 + 3)}
    got = {**c, "matched": f["matched"], "f1": f["f1"]}
    return {"got": got, "want": want, "ok": all(abs(got[k] - want[k]) < 1e-12 for k in want)}
