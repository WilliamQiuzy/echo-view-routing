#!/usr/bin/env python
"""Task 1 of the 2026-10-01 round (docs/research_plan.md): does shuffling help temporal decoders by breaking up runs of
label deviations? Uses existing caches only; no training.

Frozen inputs
- The corrected image set: every fragment of the validation/test two-fragment and 4-8-fragment recipes and every
  native clip, with frame IDs, features and probabilities from cache_v2. The recipes are the ladder's.
- Uniform 10 Hz position grid: "seconds" and "per minute" are grid-equivalent, not physiological time.
Conditions (test split only; validation stays in true order, so routing thresholds and decoder settings are frozen
from true-order validation and never re-selected per condition)
- true; name (frame IDs as text); random (10 seeds); reverse; block-5 / 10 / 20 samples (same 10 seeds).
Methods: B0 (order-invariance check of the clip majority vote), B1, B2 (HMM), B3, B3a (audited). The EchoPrime recheck
uses B2, B3 and an EchoPrime-specific audited B3 (chosen on EchoPrime true-order validation) under true and random.
Outputs (<out>): metrics.csv (ladder metrics per encoder/condition/seed/method, plus the validation-selected routing
thresholds as JSON so the freeze can be checked), natives.csv (per test native clip: deviations before/after decoding,
error runs, cuts, lag-1 autocorrelation), fragments.csv.gz (per test fragment counts of the constructed banks, for
resampling), runs.csv (error-run length histogram per source), coverage.json, echoprime_b3a.json.
"""
from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import base_parser, load_cfg  # noqa: E402
from scripts.compare_ladder_runs import METRICS, get  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.compose.banks import multi_bank, pair_bank  # noqa: E402
from echo_routing.compose.stream_assembler import Stream  # noqa: E402
from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.ladder import NativeCine, evaluate_method  # noqa: E402
from echo_routing.features.cache import read_index, variant_dir  # noqa: E402
from echo_routing.realvideo.permutations import error_runs, lag1_autocorrelation, order_index  # noqa: E402
from echo_routing.realvideo.screening import patient_of  # noqa: E402
from echo_routing.temporal.baselines.registry import AUDITED_B3, decode_with_prob  # noqa: E402
from echo_routing.temporal.segments import boundaries_from_labels  # noqa: E402

SEEDS = tuple(range(100, 110))
CONDITIONS = ([("true", "true", 0, None), ("name", "name", 0, None), ("reverse", "reverse", 0, None)]
              + [("random", "random", s, None) for s in SEEDS]
              + [(f"block{b}", "block", s, b) for b in (5, 10, 20) for s in SEEDS])
METHODS = ("b0", "b1", "b2", "b3", "b3a")
EP_METHODS = ("b2", "b3", "b3a")
HZ = 10.0


class Frozen:
    """Per (clip, variant): frame IDs, B0 probabilities and features from cache_v2; optional EchoPrime per stream."""

    def __init__(self, paths, ckpt: str) -> None:
        self.root = paths.cache_root / "features" / ckpt; self.ep_root = paths.cache_root / "echoprime_windowed" / ckpt
        self.memo: dict = {}

    def clip(self, vid: str, variant: str):
        if (vid, variant) not in self.memo:
            with np.load(self.root / variant / f"{vid}.npz") as z:
                self.memo[(vid, variant)] = (z["frame_idx"].astype(int), z["prob"].astype(float), z["feat"].astype(np.float32))
        return self.memo[(vid, variant)]

    def ep_stream(self, sid: str) -> np.ndarray:
        with np.load(self.ep_root / f"{sid}.npz") as z:
            return z["prob"].astype(float)


def assemble(fz: Frozen, recipe, kind: str, seed: int, block, encoder: str) -> tuple[Stream, int, int]:
    probs, feats, labels, joins, pos, changed = [], [], [], [], 0, 0
    ep = fz.ep_stream(recipe.recipe_id) if encoder == "echoprime" else None
    for i, fr in enumerate(recipe.fragments):
        ids, p, f = fz.clip(fr.video_id, fr.variant)
        sl = slice(fr.start, fr.end); n = fr.end - fr.start
        idx, ch = order_index(n, kind, seed, recipe.recipe_id, i, ids=ids[sl], block=block); changed += int(ch)
        pp = (ep[pos:pos + n] if ep is not None else p[sl])[idx]
        probs.append(pp); feats.append(f[sl][idx]); labels.append(np.full(n, fr.label, dtype=int))
        if i > 0:
            joins.append(pos)
        pos += n
    lab = np.concatenate(labels); j = np.array(joins, dtype=int)
    sem = np.array([x for x in j if lab[x] != lab[x - 1]], dtype=int)
    return Stream(recipe, np.concatenate(probs), np.concatenate(feats), lab, j, sem), changed, len(recipe.fragments)


def natives(fz: Frozen, idx: pd.DataFrame, split: str, kind: str, seed: int, block, encoder: str):
    out, changed = [], 0
    for r in idx[idx["split"] == split].itertuples(index=False):
        ids, p, f = fz.clip(r.video_id, "orig")
        if encoder == "echoprime":
            p = fz.ep_stream(r.video_id)
        o, ch = order_index(len(ids), kind, seed, r.video_id, 0, ids=ids, block=block); changed += int(ch)
        out.append(NativeCine(r.video_id, int(r.label_index), p[o], f[o]))
    return out, changed


def same_view_split_rate(res: dict, key: str) -> float | None:
    block = res.get(key) or {}
    cells = block.get("cells") or {}
    joins = sum(int(c.get("same_view_joins", 0)) for c in cells.values())
    splits = sum(int(c.get("false_splits", 0)) for c in cells.values())
    return splits / joins if joins else None


def native_rows(cines, cfg: dict, methods, encoder: str, cond: str, seed: int, hist: Counter | None = None) -> list[dict]:
    rows = []
    for c in cines:
        raw = c.prob.argmax(1); runs_raw = error_runs(raw, np.full(len(raw), c.label))
        if hist is not None:
            hist.update((encoder, cond, seed, "raw", int(x)) for x in runs_raw)
        base = {"encoder": encoder, "condition": cond, "seed": seed, "video_id": c.video_id, "group": patient_of(c.video_id),
                "n": len(raw), "lag1": lag1_autocorrelation(c.prob), "dev_raw": int(runs_raw.sum()),
                "runs_raw": int(runs_raw.size), "dev_raw_in_runs_ge5": int(runs_raw[runs_raw >= 5].sum())}
        for m in methods:
            lab = decode_with_prob(m, c.prob, cfg, c.feat, c.video_id)[0]
            rr = error_runs(lab, np.full(len(lab), c.label))
            if hist is not None:
                hist.update((encoder, cond, seed, m, int(x)) for x in rr)
            rows.append({**base, "method": m, "dev_dec": int(rr.sum()), "runs_dec": int(rr.size),
                         "cuts": int(boundaries_from_labels(lab).size)})
    return rows


def echoprime_b3a(paths, cfg: dict, ckpt: str) -> dict:
    """Audited-B3 setting for EchoPrime probabilities, chosen on its true-order validation by the problem A rule."""
    from scripts.pooling_audit import CONFIGS, curves_of, pick, run_config
    from scripts.problem_a import build_streams
    from echo_routing.realvideo.two_window import quantile_thresholds, two_window_score, window_lengths
    streams, _ = build_streams(paths, cfg, ckpt, "echoprime")
    rows, _ = run_config("ep A1b", CONFIGS["A1b JS fixed windows"], streams, {})
    op = pick(curves_of(pd.DataFrame(rows)))["ep A1b"]
    w = int(op["setting"].split()[0].split("=")[1]); q = float(op["setting"].split()[1].split("=")[1])
    sc = np.concatenate([two_window_score(s.prob, window_lengths(len(s.prob), fixed=w)[0], "js") for s in streams["validation/native"]])
    return {"half_window": w, "threshold": float(quantile_thresholds(sc, (q,))[0]), "nms_radius": 5,
            "chosen": op["setting"], "validation": op["validation"]}


def frozen_thresholds(res: dict) -> str:
    """Validation-selected routing thresholds of one evaluate_method call (identical across conditions if frozen)."""
    out = {"pair": {t: v["tau"] for t, v in res["policy"]["by_target"].items()}}
    if res.get("policy_multi"):
        out["multi"] = {t: v["tau"] for t, v in res["policy_multi"]["by_target"].items()}
        out["p2"] = {t: v["tau_by_class"] for t, v in res["policy2"]["by_target"].items()}
    return json.dumps(out, sort_keys=True, default=float)


def run_encoder(fz, idx, banks, cfg, encoder: str, methods, conditions, out_rows, nat_rows, coverage,
                frag_rows: list, hist: Counter) -> None:
    val_pairs = [assemble(fz, r, "true", 0, None, encoder)[0] for r in banks["validation"][0].recipes]
    val_multi = [assemble(fz, r, "true", 0, None, encoder)[0] for r in banks["validation"][1].recipes]
    val_nat, _ = natives(fz, idx, "validation", "true", 0, None, encoder)
    ncls = val_nat[0].prob.shape[1]
    for name, kind, seed, block in conditions:
        tp = [assemble(fz, r, kind, seed, block, encoder) for r in banks["test"][0].recipes]
        tm = [assemble(fz, r, kind, seed, block, encoder) for r in banks["test"][1].recipes]
        tn, nch = natives(fz, idx, "test", kind, seed, block, encoder)
        coverage[f"{encoder}/{name}/{seed}"] = {"fragments_perturbed": sum(x[1] for x in tp + tm), "fragments": sum(x[2] for x in tp + tm),
                                                "natives_perturbed": nch, "natives": len(tn)}
        pairs, multi = [x[0] for x in tp], [x[0] for x in tm]
        for m in methods:
            res = evaluate_method(m, val_nat, tn, val_pairs, pairs, ncls, cfg, test_multi=multi, val_multi=val_multi)
            row = {"encoder": encoder, "condition": name, "seed": seed, "method": m, **{k: get(res, p) for k, p in METRICS}}
            row["same_view_split_pairs"] = same_view_split_rate(res, "constructed")
            row["same_view_split_multi"] = same_view_split_rate(res, "constructed_multi")
            row["thresholds"] = frozen_thresholds(res)
            out_rows.append(row)
            for bank, fr in res["_fragments"].items():
                frag_rows.extend({"encoder": encoder, "condition": name, "seed": seed, "method": m, "bank": bank,
                                  **{k: v for k, v in f.items() if not k.startswith(("acc|", "wrong|"))}} for f in fr)
        nat_rows.extend(native_rows(tn, cfg, methods, encoder, name, seed, hist))
        print(f"done {encoder} {name} seed {seed}", flush=True)


def main() -> None:
    ap = base_parser("Permutation mechanism"); ap.add_argument("--out", required=True); ap.add_argument("--ckpt-hash", default="79f4a41af6e3")
    ap.add_argument("--skip-echoprime", action="store_true")
    args = ap.parse_args(); cfg = load_cfg(args); paths = load_paths(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg["decode"].setdefault("b3a", dict(AUDITED_B3))
    idx = read_index(variant_dir(paths.cache_root, args.ckpt_hash, "orig"))
    banks = {s: (pair_bank(idx, s, cfg), multi_bank(idx, s, cfg)) for s in ("validation", "test")}
    fz = Frozen(paths, args.ckpt_hash)
    rows, nat, cov, frags, hist = [], [], {}, [], Counter()
    run_encoder(fz, idx, banks, cfg, "b0", METHODS, CONDITIONS, rows, nat, cov, frags, hist)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False); pd.DataFrame(nat).to_csv(out / "natives.csv", index=False)
    if not args.skip_echoprime:
        ep_cfg = copy.deepcopy(cfg); setting = echoprime_b3a(paths, ep_cfg, args.ckpt_hash)
        ep_cfg["decode"]["b3a"] = {k: setting[k] for k in ("half_window", "threshold", "nms_radius")}
        (out / "echoprime_b3a.json").write_text(json.dumps(setting, indent=1, default=float))
        ep_conditions = [c for c in CONDITIONS if c[0] in ("true", "random")]
        run_encoder(fz, idx, banks, ep_cfg, "echoprime", EP_METHODS, ep_conditions, rows, nat, cov, frags, hist)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False); pd.DataFrame(nat).to_csv(out / "natives.csv", index=False)
    pd.DataFrame(frags).to_csv(out / "fragments.csv.gz", index=False)
    pd.DataFrame([dict(zip(("encoder", "condition", "seed", "source", "length"), k), count=v) for k, v in sorted(hist.items())]
                 ).to_csv(out / "runs.csv", index=False)
    (out / "coverage.json").write_text(json.dumps(cov, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
