#!/usr/bin/env python
"""Write per-stream frame plans for the windowed clip-classifier baselines (banks + demo streams).

Output: data/window_plans/<ckpt_hash>/plans.json = {stream_id: {"n": N, "refs": [[video_id, frame_idx, gamma], ...], "bank": ...}}
plus frames_root so standalone runners (other Python environments) need nothing from echo_routing.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402

from echo_routing.compose.banks import Bank, multi_bank, native_bank, pair_bank  # noqa: E402
from echo_routing.compose.recipe_builder import recipe_from_dict  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.features.cache import load_video_features, read_index, variant_dir  # noqa: E402
from echo_routing.temporal.windowed import stream_plan  # noqa: E402


def main() -> None:
    ap = base_parser("Build window plans"); ap.add_argument("--ckpt-hash", required=True)
    ap.add_argument("--extra-recipes", default=None, help="JSON list of extra recipes (e.g. demo streams)")
    ap.add_argument("--protocol", default=None, help="protocol.json: plan only its validation halves' banks (never test)")
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths()
    idx = read_index(variant_dir(paths.cache_root, args.ckpt_hash, "orig")); cache = {}

    def loader(v, var):
        if (v, var) not in cache:
            cache[(v, var)] = load_video_features(variant_dir(paths.cache_root, args.ckpt_hash, var) / f"{v}.npz")
        return cache[(v, var)]

    banks = {}
    if args.protocol:
        proto_raw = Path(args.protocol).read_bytes(); proto = json.loads(proto_raw)
        for half, rec in proto["validation"].items():
            for kind, b in rec["banks"].items():
                banks[f"{kind}_{half}"] = Bank(f"{kind}_{half}", "validation", tuple(recipe_from_dict(d) for d in b["recipes"]))
    else:
        for split in ("validation", "test"):
            banks[f"pairs_{split}"] = pair_bank(idx, split, cfg); banks[f"multi_{split}"] = multi_bank(idx, split, cfg); banks[f"native_{split}"] = native_bank(idx, split)
    plans = {}
    for bank_id, b in banks.items():
        for r in b.recipes:
            refs = stream_plan(r, loader)
            plans[r.recipe_id] = {"bank": bank_id, "n": len(refs), "refs": [[x.video_id, x.frame_idx, x.gamma] for x in refs]}
    if args.extra_recipes:
        for d in json.loads(Path(args.extra_recipes).read_text()):
            r = recipe_from_dict(d); refs = stream_plan(r, loader)
            plans[r.recipe_id] = {"bank": "demo", "n": len(refs), "refs": [[x.video_id, x.frame_idx, x.gamma] for x in refs]}
    tag = args.ckpt_hash if not args.protocol else f"{args.ckpt_hash}_protocol_{hashlib.sha256(proto_raw).hexdigest()[:12]}"
    out = paths.data_root / "window_plans" / tag; out.mkdir(parents=True, exist_ok=True)
    (out / "plans.json").write_text(json.dumps({"frames_root": str(paths.ev9v_images), "hz": cfg["sampling"]["target_hz"], "streams": plans}))
    print(f"wrote {out / 'plans.json'}: {len(plans)} streams, {sum(p['n'] for p in plans.values())} samples")


if __name__ == "__main__":
    main()
