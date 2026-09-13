#!/usr/bin/env python
"""Run the CPU baseline ladder on cached features: native cines + constructed 2x2 streams; write metrics + ladder."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg, new_run_dir  # noqa: E402

import numpy as np  # noqa: E402

from echo_routing.compose.recipe_builder import Pool, make_pair_recipes, recipe_to_dict  # noqa: E402
from echo_routing.compose.stream_assembler import render  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.ladder import NativeCine, evaluate_b_file, evaluate_method  # noqa: E402
from echo_routing.evaluate.report import write_ladder  # noqa: E402
from echo_routing.features.cache import load_video_features, read_index, variant_dir  # noqa: E402


def _strip_private(d):
    if isinstance(d, dict):
        return {k: _strip_private(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, list):
        return [_strip_private(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    return d


def main() -> None:
    ap = base_parser("Run baseline ladder")
    ap.add_argument("--ckpt-hash", required=True)
    ap.add_argument("--methods", default=None, help="comma list; default from config `methods`")
    ap.add_argument("--limit", type=int, default=None, help="max native cines per split (smoke)")
    args = ap.parse_args()
    cfg = load_cfg(args); paths = load_paths()
    methods = (args.methods or ",".join(cfg["methods"])).split(",")
    run_dir = new_run_dir(cfg, "baselines")
    hz = cfg["sampling"]["target_hz"]; rc = cfg["recipes"]

    orig = variant_dir(paths.cache_root, args.ckpt_hash, "orig")
    idx = read_index(orig)
    if args.limit:
        idx = idx.groupby("split", group_keys=False).head(args.limit)
    cache: dict[tuple[str, str], object] = {}

    def loader(video_id: str, variant: str):
        key = (video_id, variant)
        if key not in cache:
            cache[key] = load_video_features(variant_dir(paths.cache_root, args.ckpt_hash, variant) / f"{video_id}.npz")
        return cache[key]

    native = {s: [NativeCine(r.video_id, int(r.label_index), loader(r.video_id, "orig").prob, loader(r.video_id, "orig").feat)
                  for r in idx[idx["split"] == s].itertuples(index=False)] for s in ("validation", "test")}
    num_classes = native["validation"][0].prob.shape[1]
    streams = {}
    for s, seed_off in (("validation", 0), ("test", 1)):
        sub = idx[idx["split"] == s]
        pool = Pool.from_index(sub["video_id"], sub["label_index"], sub["n_samples"])
        recipes = make_pair_recipes(pool, int(rc["n_per_cell"]), np.random.default_rng(int(rc["seed"]) + seed_off),
                                    edit_variant=rc["edit_variant"], min_len=int(rc["min_len"]), max_len=int(rc["max_len"]))
        bank = paths.data_root / "manifests" / "recipes"; bank.mkdir(parents=True, exist_ok=True)
        (bank / f"{s}_pairs_{args.ckpt_hash}.json").write_text(json.dumps([recipe_to_dict(r) for r in recipes], indent=1))
        streams[s] = [render(r, loader) for r in recipes]
        print(f"{s}: {len(native[s])} native cines, {len(streams[s])} constructed streams", flush=True)

    mdir = run_dir / "metrics"; mdir.mkdir(exist_ok=True)
    results = []
    for m in methods:
        if m == "b_file":
            res = evaluate_b_file(native["validation"], native["test"], num_classes, cfg)
        else:
            res = evaluate_method(m, native["validation"], native["test"], streams["validation"], streams["test"], num_classes, cfg)
        res["ckpt_hash"] = args.ckpt_hash
        (mdir / f"{m}.json").write_text(json.dumps(_strip_private(res), indent=1))
        results.append(res); print(f"done {m}", flush=True)
    header = f"# Baseline ladder\n\nckpt_hash={args.ckpt_hash} · task classes={num_classes} · hz={hz} · " \
             f"native test cines={len(native['test'])} · constructed test streams={len(streams['test'])} · " \
             f"policy target risk={cfg['policy']['target_risk']} (validation-selected, frozen)"
    out = write_ladder(results, run_dir / "reports", header)
    print(out.read_text()); print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
