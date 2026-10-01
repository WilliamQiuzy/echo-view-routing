#!/usr/bin/env python
"""Order-versus-image-set attribution with frozen fragment frame sets (docs/research_plan.md, audit round; B0-B3).

Protocol
- Frame identity. Each sample is identified by the true frame number of the image it was computed on. In the legacy
  cache, sample k came from position 3k of the name-sorted listing; in cache_v2, from frame 3k.
- Frozen sets. For every fragment of every recipe (two-fragment and 4-8-fragment streams) and for every native clip:
  - S_L = the legacy frame IDs at the fragment's sample positions;
  - S_N = the cache_v2 frame IDs at those positions.
  Sets are fixed before any reordering, so all order conditions of one set contain exactly the same images.
- Orders.
  - name: IDs sorted as text, i.e. the legacy order;
  - true: IDs sorted numerically;
  - random: one seeded permutation per fragment, the same positions for both sets.
- Time axis. A uniform position grid: each sample is one 0.1 s step whatever its true frame number. Joins stay at the
  same positions, tolerances are 5 samples, and rates use n x 0.1 s. This is not physical time.
- Conditions. {S_L, S_N} x {name, true, random}, with B0-B3 scored by the ladder's evaluate_method.
  (S_L, name) must reproduce the legacy ladder and (S_N, true) the corrected one.
Also reports the per-fragment overlap of S_L and S_N, and the overlap of S_L with the sets the earlier decomposition
used (whole clip sorted before cutting). Legacy files are read directly for tracing.
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402
from scripts.compare_ladder_runs import METRICS, get  # noqa: E402
from scripts.legacy_set_true_order import name_order_numbers  # noqa: E402

import numpy as np  # noqa: E402

from echo_routing.compose.banks import multi_bank, pair_bank  # noqa: E402
from echo_routing.compose.stream_assembler import Stream  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.ladder import NativeCine, evaluate_method  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402

METHODS = ("b0", "b1", "b2", "b3")
SETS, ORDERS = ("legacy", "new"), ("name", "true", "random")


class Arrays:
    """Per (set, clip, variant): frame IDs, probabilities and features in cached order."""

    def __init__(self, paths, ckpt: str) -> None:
        self.paths, self.ckpt, self.memo, self.names = paths, ckpt, {}, {}

    def get(self, which: str, vid: str, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (which, vid, variant)
        if key not in self.memo:
            root = self.paths.repo_root / ("cache" if which == "legacy" else "cache_v2")
            with np.load(root / "features" / self.ckpt / variant / f"{vid}.npz") as z:
                pos, prob, feat = z["frame_idx"].astype(int), z["prob"].astype(float), z["feat"].astype(np.float32)
            if which == "legacy":
                if vid not in self.names:
                    self.names[vid] = name_order_numbers(self.paths.ev9v_images / vid)
                ids = self.names[vid][np.minimum(pos, len(self.names[vid]) - 1)]
            else:
                ids = pos
            self.memo[key] = (ids, prob, feat)
        return self.memo[key]


def order_index(ids: np.ndarray, order: str, seed: int) -> np.ndarray:
    if order == "name":
        return np.array(sorted(range(len(ids)), key=lambda i: str(int(ids[i]))), dtype=int)
    if order == "true":
        return np.argsort(ids, kind="stable")
    return np.random.default_rng(seed).permutation(len(ids))


def fragment(arr: Arrays, which: str, order: str, rid: str, i: int, fr) -> tuple[np.ndarray, ...]:
    ids, prob, feat = arr.get(which, fr.video_id, fr.variant)
    sl = slice(fr.start, fr.end)
    idx = order_index(ids[sl], order, zlib.crc32(f"{rid}/{i}".encode()))
    return ids[sl][idx], prob[sl][idx], feat[sl][idx]


def build_stream(arr: Arrays, which: str, order: str, recipe) -> Stream:
    probs, feats, labels, joins, pos = [], [], [], [], 0
    for i, fr in enumerate(recipe.fragments):
        _, p, f = fragment(arr, which, order, recipe.recipe_id, i, fr)
        probs.append(p); feats.append(f); labels.append(np.full(len(p), fr.label, dtype=int))
        if i > 0:
            joins.append(pos)
        pos += len(p)
    lab = np.concatenate(labels); j = np.array(joins, dtype=int)
    sem = np.array([x for x in j if lab[x] != lab[x - 1]], dtype=int)
    return Stream(recipe, np.concatenate(probs), np.concatenate(feats), lab, j, sem)


def natives(arr: Arrays, which: str, order: str, idx, split: str) -> list[NativeCine]:
    out = []
    for r in idx[idx["split"] == split].itertuples(index=False):
        ids, p, f = arr.get(which, r.video_id, "orig")
        o = order_index(ids, order, zlib.crc32(f"{r.video_id}/native".encode()))
        out.append(NativeCine(r.video_id, int(r.label_index), p[o], f[o]))
    return out


def set_overlap(arr: Arrays, banks: list) -> dict:
    jac, same, flawed = [], 0, []
    n = 0
    for bank in banks:
        for rec in bank.recipes:
            for fr in rec.fragments:
                a = set(arr.get("legacy", fr.video_id, fr.variant)[0][fr.start:fr.end].tolist())
                b = set(arr.get("new", fr.video_id, fr.variant)[0][fr.start:fr.end].tolist())
                jac.append(len(a & b) / len(a | b)); same += int(a == b); n += 1
                whole = np.sort(arr.get("legacy", fr.video_id, fr.variant)[0])  # earlier decomposition: sort, then cut
                c = set(whole[fr.start:fr.end].tolist()); flawed.append(len(a & c) / len(a | c))
    q = lambda x: [round(float(np.percentile(x, p)), 3) for p in (10, 50, 90)]  # noqa: E731
    return {"fragments": n, "jaccard_legacy_vs_new_p10_p50_p90": q(jac), "identical_sets": same,
            "jaccard_legacy_vs_earlier_decomposition_p10_p50_p90": q(flawed),
            "earlier_decomposition_identical": int(sum(1 for x in flawed if x == 1.0))}


def main() -> None:
    ap = base_parser("Order versus image set"); ap.add_argument("--out", required=True); ap.add_argument("--ckpt-hash", default="79f4a41af6e3")
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    idx = read_index(variant_dir(paths.cache_root, args.ckpt_hash, "orig"))
    banks = {s: (pair_bank(idx, s, cfg), multi_bank(idx, s, cfg)) for s in ("validation", "test")}
    arr = Arrays(paths, args.ckpt_hash)
    overlap = set_overlap(arr, [b for pair in banks.values() for b in pair])
    nat_overlap = [len(set(arr.get("legacy", v, "orig")[0]) & set(arr.get("new", v, "orig")[0])) / len(set(arr.get("legacy", v, "orig")[0]) | set(arr.get("new", v, "orig")[0]))
                   for v in idx["video_id"][idx["split"].isin(["validation", "test"])]]
    overlap["native_clips_jaccard_p10_p50_p90"] = [round(float(np.percentile(nat_overlap, p)), 3) for p in (10, 50, 90)]
    print(json.dumps(overlap), flush=True)
    num_classes = arr.get("new", idx["video_id"].iloc[0], "orig")[1].shape[1]
    results = {}
    for which in SETS:
        for order in ORDERS:
            st = {s: [build_stream(arr, which, order, r) for r in banks[s][0].recipes] for s in banks}
            mu = {s: [build_stream(arr, which, order, r) for r in banks[s][1].recipes] for s in banks}
            nv = {s: natives(arr, which, order, idx, s) for s in banks}
            for m in METHODS:
                res = evaluate_method(m, nv["validation"], nv["test"], st["validation"], st["test"], num_classes, cfg,
                                      test_multi=mu["test"], val_multi=mu["validation"])
                results[f"{which}/{order}/{m}"] = {name: get(res, key) for name, key in METRICS}
            print(f"done {which}/{order}", flush=True)
    (out / "conditions.json").write_text(json.dumps({"set_overlap": overlap, "metrics": results}, indent=1, default=float))
    lines = ["| Images | Order | Method | " + " | ".join(n for n, _ in METRICS) + " |", "|---|---|---|" + "---|" * len(METRICS)]
    for key, vals in results.items():
        which, order, m = key.split("/")
        lines.append(f"| {which} | {order} | {m} | " + " | ".join("" if v is None else f"{v:.3f}" for v in vals.values()) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
