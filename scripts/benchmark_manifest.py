#!/usr/bin/env python
"""Benchmark manifest of the frozen EV9V test benchmark (configs/experiments/frozen_external_benchmark_v1.yaml).

Built once, on the server, from H09's own construction (scripts/true_order_joint.py): the cache index's test natives
(five-family task) and the ladder's pair and 4-8 fragment test banks (recipe seed 1234). Every item keeps its 10 Hz
reference labels, fragments, joins and (recording, frame, gamma) sample references; the references come from the
historical window plans and must equal cache_v2's frame indices and variant gammas. The manifest is private (server);
summary.json holds counts and digests only.

  python scripts/benchmark_manifest.py --config configs/experiments/frozen_external_benchmark_v1.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._common import load_cfg  # noqa: E402
from scripts.permutation_mechanism import Frozen  # noqa: E402

import numpy as np  # noqa: E402

from echo_routing.compose.banks import multi_bank, pair_bank  # noqa: E402
from echo_routing.compose.recipe_builder import recipe_to_dict  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import sha256_file  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.realvideo.screening import patient_of  # noqa: E402
from echo_routing.temporal.dev_run import digest  # noqa: E402

ENC = "79f4a41af6e3"
GAMMA = {"orig": 1.0, "gamma090": 0.9, "gamma110": 1.1}
KINDS = ("native", "pairs", "multi")


def pinned(spec: dict) -> Path:
    p = ROOT / spec["path"]
    if sha256_file(p) != spec["sha256"]:
        raise SystemExit(f"{spec['path']}: differs from its pin")
    return p


def ladder(bench: dict) -> dict:
    return load_cfg(SimpleNamespace(config=str(pinned(bench["ladder_config"])), set=[]))


def item_refs(fz: Frozen, plan: dict, frags: list[dict], iid: str) -> list[list]:
    """The plan's sample references, checked against cache_v2's frame indices and variant gammas."""
    refs = [[str(r[0]), int(r[1]), float(r[2])] for r in plan["refs"]]; want = []
    for f in frags:
        ids = fz.clip(f["video_id"], f["variant"])[0][f["start"]:f["end"]]; want += [[f["video_id"], int(i), GAMMA[f["variant"]]] for i in ids]
    if refs != want or int(plan["n"]) != len(want):
        raise SystemExit(f"{iid}: the window plan's references differ from cache_v2's frame indices / gammas")
    return refs


def build(cfg: dict) -> dict:
    bench = cfg["frozen_external_benchmark"]["benchmark"]; lc = ladder(bench); paths = load_paths()
    if (paths.cache_root / "FRAME_ORDER").read_text().strip() != bench["frame_order"].split()[-1]:
        raise SystemExit("cache_v2 frame order stamp differs")
    plans = json.loads(pinned(bench["frame_refs"]).read_text())["streams"]; idx = read_index(variant_dir(paths.cache_root, ENC, "orig")); fz = Frozen(paths, ENC)
    banks = {"pairs": pair_bank(idx, "test", lc), "multi": multi_bank(idx, "test", lc)}; items = []
    for r in idx[idx["split"] == "test"].itertuples(index=False):
        n = len(fz.clip(r.video_id, "orig")[0]); fr = [{"video_id": r.video_id, "variant": "orig", "start": 0, "end": n, "label": int(r.label_index)}]
        items.append({"id": r.video_id, "kind": "native", "n": n, "labels": [int(r.label_index)] * n, "fragments": fr, "joins": [], "semantic": [],
                      "refs": item_refs(fz, plans[r.video_id], fr, r.video_id)})
    for kind, bank in banks.items():
        for rec in bank.recipes:
            fr, labels, joins, pos = [], [], [], 0
            for i, f in enumerate(rec.fragments):
                m = f.end - f.start; fr.append({"video_id": f.video_id, "variant": f.variant, "start": int(f.start), "end": int(f.end), "label": int(f.label)})
                joins += [pos] if i else []; labels += [int(f.label)] * m; pos += m
            sem = [j for j in joins if labels[j] != labels[j - 1]]
            items.append({"id": rec.recipe_id, "kind": kind, "n": pos, "labels": labels, "fragments": fr, "joins": joins, "semantic": sem,
                          "refs": item_refs(fz, plans[rec.recipe_id], fr, rec.recipe_id)})
    bank_digests = {f"{k}_test": digest([recipe_to_dict(r) for r in b.recipes]) for k, b in banks.items()}
    return {"benchmark": bench["id"], "ladder_config_sha256": bench["ladder_config"]["sha256"], "frame_refs_sha256": bench["frame_refs"]["sha256"],
            "bank_digests": bank_digests, "items": items}


def summary(man: dict, bench: dict) -> dict:
    it = man["items"]; by = {k: [x for x in it if x["kind"] == k] for k in KINDS}
    clips = {k: {f["video_id"] for x in v for f in x["fragments"]} for k, v in by.items()}; allc = set().union(*clips.values())
    seg = {k: sum(1 + len(x["semantic"]) for x in v) for k, v in by.items()}
    short = {k: sum(int(np.sum(np.diff([0, *x["semantic"], x["n"]]) <= 16)) for x in v) for k, v in by.items()}
    exp = bench["expected"]; pre = bench["bank_digest_prefixes"]
    checks = {"natives": len(by["native"]) == exp["natives"], "native_sessions": len({patient_of(v) for v in clips["native"]}) == exp["native_sessions"],
              "pairs": len(by["pairs"]) == exp["pairs"], "multi": len(by["multi"]) == exp["multi"], "multi_clips": len(clips["multi"]) == exp["multi_clips"],
              "pairs_digest": man["bank_digests"]["pairs_test"].startswith(pre["pairs_test"]), "multi_digest": man["bank_digests"]["multi_test"].startswith(pre["multi_test"])}
    return {"benchmark": man["benchmark"], "manifest_digest": digest(man["items"]), "bank_digests": man["bank_digests"], "checks": checks,
            "counts": {k: {"items": len(v), "samples": int(sum(x["n"] for x in v)), "clips": len(clips[k]), "sessions": len({patient_of(c) for c in clips[k]}),
                           "true_segments": seg[k], "true_segments_le16": short[k], "true_changes": int(sum(len(x["semantic"]) for x in v)),
                           "same_class_joins": int(sum(len(x["joins"]) - len(x["semantic"]) for x in v)),
                           "gamma_samples": {str(g): int(sum(1 for x in v for r in x["refs"] if r[2] == g)) for g in (0.9, 1.0, 1.1)},
                           "per_class_items": [int(sum(1 for x in v if x["labels"][0] == c)) for c in range(5)] if k == "native" else None} for k, v in by.items()},
            "all_clips": len(allc), "all_sessions": len({patient_of(c) for c in allc}),
            "filters": "test natives = the five-family cache index rows of the official test split (PMPALA and clips without cached frames excluded upstream); banks drawn from them"}


def load(cfg: dict) -> dict:
    """The built manifest, refused unless its digest equals the recorded one."""
    out = ROOT / cfg["frozen_external_benchmark"]["outputs"]["root"] / "benchmark"; man = json.loads((out / "manifest.json").read_text())
    rec = json.loads((out / "summary.json").read_text())
    if digest(man["items"]) != rec["manifest_digest"]:
        raise SystemExit("the benchmark manifest differs from its recorded digest")
    return man


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--config", required=True)
    a = ap.parse_args(); cfg = load_cfg(SimpleNamespace(config=a.config, set=[])); fb = cfg["frozen_external_benchmark"]; out = ROOT / fb["outputs"]["root"] / "benchmark"
    if (out / "manifest.json").exists():
        raise SystemExit("the benchmark manifest exists; it is built once")
    t0 = time.time(); man = build(cfg); rec = summary(man, fb["benchmark"]); rec["config_sha256"] = sha256_file(ROOT / a.config); rec["seconds"] = time.time() - t0
    if not all(rec["checks"].values()):
        raise SystemExit(f"benchmark checks failed: {rec['checks']}")
    out.mkdir(parents=True, exist_ok=True); (out / "manifest.json").write_text(json.dumps(man)); (out / "summary.json").write_text(json.dumps(rec, indent=1))
    res = ROOT / fb["outputs"]["results_dir"]; res.mkdir(parents=True, exist_ok=True); (res / "benchmark_summary.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
