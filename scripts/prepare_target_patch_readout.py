#!/usr/bin/env python
"""Record the protocol inputs of the target-anchored readout round (plan Tasks 1-2; configs/experiments/target_patch_readout_v1.yaml).

Checks every pinned source: the task sheet, the source round's configuration, summary, README and tables, the three
Ordered final weights, the frozen manifests and their index, the saved Ordered per-window outputs and the environment,
including the server's torchvision MViT source. Every recording of the source manifests must be a training clip of the
protocol. For every seed, the three readouts are built from the same Ordered state: their parameter names, counts and
starting digests must agree, no parameter may be added, and the learned marker must be the source's. It records the token
grid, every block's query stride and the stem's temporal geometry. Nothing here trains or reads development data.

  python scripts/prepare_target_patch_readout.py --config configs/experiments/target_patch_readout_v1.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.train_target_patch_readout as tr  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.prepare_fold_output_correction import assert_no_ids  # noqa: E402
from scripts.prepare_raw_video_context import allowlist, environment  # noqa: E402
from scripts.train_joint_visual_temporal import params_digest  # noqa: E402


import echo_routing.vit.target_mvit as tm  # noqa: E402
import echo_routing.vit.target_patch_readout as tp  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402


def pins(src: dict) -> dict[str, dict]:
    out = {"task_sheet": None, "config": src["config"], **{f"report_{k}": v for k, v in src["report"].items()},
           **{f"weights_s{k}": v for k, v in src["weights"].items()}, "manifest_index": src["manifests"]["index"],
           **{f"manifest_s{k}": v for k, v in src["manifests"]["files"].items()}}
    for h, per in src["ordered_outputs"].items():
        out |= {f"ordered_{h}_s{k}": v for k, v in per.items()}
    return {k: v for k, v in out.items() if v is not None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); ctx = tr.Ctx(cfg, a.config); t0 = time.time(); root = ctx.root; src = ctx.tp["source"]
    checked = {"task_sheet": sha256_file(root / ctx.tp["task_sheet"]["path"])}
    if checked["task_sheet"] != ctx.tp["task_sheet"]["sha256"]:
        raise SystemExit("the task sheet differs from its pin")
    for name, pin in pins(src).items():
        if (got := sha256_file(root / pin["path"])) != pin["sha256"]:
            raise SystemExit(f"{name} {pin['path']}: sha256 {got[:12]} != {pin['sha256'][:12]}")
        checked[name] = got
    env = environment(); want = src["environment"]
    if any(str(env[k]) != str(want[k]) for k in ("torch", "torchvision", "cuda", "cudnn", "torchvision_mvit_sha256")):
        raise SystemExit(f"environment {env} differs from {want}")
    train = set(allowlist(ctx.rv, root)); outside = 0
    for s in ctx.seeds:
        outside += len({str(v) for v in ctx.manifest(s)[0]["videos"].tolist()} - train)
    if outside:
        raise SystemExit(f"{outside} manifest recordings are not training clips")
    rec = {"config": a.config, "checked": checked, "environment": env, "allowlist_clips": len(train), "seeds": {}}
    for s in ctx.seeds:
        state, _ = ctx.source_state(s); base = tm.build_model(None, tm.initial_new_state(s)); base.load_state_dict(state, strict=True)
        per = {}
        for r in tp.READOUTS:
            m = tp.build_readout_model(r, state)
            per[r] = {"keys": list(m.state_dict()), "parameters": int(sum(p.numel() for p in m.parameters())), "digest": params_digest(m.named_parameters()),
                      "grid": list(m.grid), "query_strides": [list(x) for x in m.query_strides], "stem_time": list(m.stem_time)}
        same = all(per[r]["keys"] == list(base.state_dict()) and per[r]["parameters"] == sum(p.numel() for p in base.parameters())
                   and per[r]["digest"] == params_digest(base.named_parameters()) for r in tp.READOUTS)
        marker = state["marker_stem.weight"]
        rec["seeds"][str(s)] = {"same_start_and_no_new_parameters": bool(same), "parameters": per["cls"]["parameters"], "start_digest": per["cls"]["digest"],
                                "grid": per["cls"]["grid"], "query_time_strides": sorted({x[0] for x in per["cls"]["query_strides"]}),
                                "query_strides": per["cls"]["query_strides"], "stem_time_kernel_stride_padding_dilation": per["cls"]["stem_time"],
                                "learned_marker_abs_max": float(marker.abs().max()), "marker_loaded": bool(float(marker.abs().max()) > 0)}
        if not same or not rec["seeds"][str(s)]["marker_loaded"]:
            raise SystemExit(f"seed {s}: the readouts do not share the source state, or the marker was not learned")
        print(f"seed {s}: {per['cls']['parameters']} parameters, grid {per['cls']['grid']}, marker max {float(marker.abs().max()):.4f}", flush=True)
    rec["seconds"] = time.time() - t0
    ids = train | {session_of(v) for v in train}; assert_no_ids(rec, ids, "preparation")
    out = ctx.out; out.mkdir(parents=True, exist_ok=True); res = root / ctx.tp["outputs"]["results_dir"]; res.mkdir(parents=True, exist_ok=True)
    for p in (out / "preparation.json", res / "preparation.json"):
        p.write_text(json.dumps(rec, indent=1))
    print(f"wrote {out / 'preparation.json'}")


if __name__ == "__main__":
    main()
