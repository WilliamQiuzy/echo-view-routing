#!/usr/bin/env python
"""Round 4: paired neighbour-swap check for B3 and ASRF (configs/experiments/failure_analysis_v1.yaml, neighbour_check).

Run because the pre-declared condition held for A4C units next to A5C units (E0 main systems). The target unit's
samples stay fixed; only the neighbouring units change, with their lengths kept, so every position is unchanged
except in the isolated variant:
- original: the stream as it is;
- isolated: the unit alone;
- same_class_neighbours: every neighbour replaced by a block of the target's class from another clip;
- other_clip_neighbours: every neighbour replaced by a block of its own class from another clip;
- other_a5c_only: only the A5C neighbour(s) replaced by A5C from another clip; the other neighbour kept.
Controls: A4C units without an A5C neighbour that B3 and every ASRF seed get right. One neighbour (the next one, else
the previous one) is replaced by an A5C block from another clip ("inject_a5c"), or by a block of its own class from
another clip ("inject_same", the placebo for replacing a neighbour at all).
Donor blocks come from the same half's clips with the contrast variant of the replaced unit's first fragment, never
from the target's or the original neighbours' clips; each (unit, variant) uses three draws seeded by SHA-256. B3 decodes the E0 classifier's
probabilities with its selected setting and ASRF decodes the E0 features with each seed's checkpoint. A unit counts as
right when the majority of its samples has its class; for variants with different-class neighbours the segment F1@50
matching of the unit is recorded too. Reference limits are used only to build these inputs. Diagnostic only.
Writes swap.csv and swap.json into a new run directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg, new_run_dir  # noqa: E402
from scripts.round4_failure_inventory import b3, load_asrf, majority  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.failure_units import Unit, stream_units, unit_outcomes  # noqa: E402
from echo_routing.temporal.dev_data import Seq  # noqa: E402
from echo_routing.temporal.dev_run import HALVES, Inputs  # noqa: E402

A4C, A5C = 2, 3
DRAWS = 3


def donors(inputs: Inputs, half: str) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for s in inputs.halves[half].natives:
        out.setdefault(int(s.labels[0]), []).append(s.seq_id)
    return out


def block(store, pool: list[str], length: int, variant: str, exclude: set[str], rng: np.random.Generator) -> tuple:
    """`length` samples of one class from donor clips: one clip when one is long enough, otherwise consecutive pieces
    of several clips of that class (a same-class join, as in the constructed streams)."""
    cand = [v for v in pool if v not in exclude]
    if not cand:
        raise SystemExit(f"no donor clip outside {sorted(exclude)}")
    long_enough = [v for v in cand if store.get(v, variant).n >= length]
    if long_enough:
        vid = long_enough[int(rng.integers(len(long_enough)))]; vf = store.get(vid, variant); st = int(rng.integers(0, vf.n - length + 1))
        return vf.feat[st:st + length], vf.prob[st:st + length], vid
    feats, probs, used, need = [], [], [], length
    while need > 0:
        vid = cand[int(rng.integers(len(cand)))]; vf = store.get(vid, variant); take = min(need, vf.n); st = int(rng.integers(0, vf.n - take + 1))
        feats.append(vf.feat[st:st + take]); probs.append(vf.prob[st:st + take]); used.append(vid); need -= take
    return np.concatenate(feats), np.concatenate(probs), ";".join(used)


def variant_stream(s, units: list[Unit], j: int, plan: dict[int, int], store, pools, rng) -> tuple:
    """Copies of the stream's features, probabilities and reference labels with units `plan` (index -> new class)
    replaced by donor blocks of the same length, and per replaced block its donor clips and the class of the frozen
    classifier's mean probability over the block's own samples."""
    feat, prob, ref = s.feat.copy(), s.prob.copy(), s.labels.astype(int).copy()
    exclude = {v for u in units[max(0, j - 1):j + 2] for v, *_ in u.sources}; blocks = []
    for i, cls in plan.items():
        u = units[i]; variant = u.sources[0][1]
        f, p, vid = block(store, pools[cls], u.length, variant, exclude, rng)
        feat[u.start:u.end], prob[u.start:u.end], ref[u.start:u.end] = f, p, cls
        blocks.append({"class": cls, "donors": vid, "own_class": int(np.asarray(p, float).mean(0).argmax())})
    return feat, prob, ref, blocks


def plans(units: list[Unit], j: int, control: bool) -> dict[str, dict[int, int] | None]:
    nb = [i for i in (j - 1, j + 1) if 0 <= i < len(units)]
    if control:
        i = j + 1 if j + 1 < len(units) else j - 1
        return {"original": {}, "inject_a5c": {i: A5C}, "inject_same": {i: units[i].label}}
    return {"original": {}, "isolated": None, "same_class_neighbours": {i: units[j].label for i in nb},
            "other_clip_neighbours": {i: units[i].label for i in nb}, "other_a5c_only": {i: A5C for i in nb if units[i].label == A5C}}


def evaluate(labels: np.ndarray, ref: np.ndarray, u: Unit, isolated: bool, recipe_units_fn) -> dict:
    """Majority-label check of the target unit, and its segment F1@50 match when the variant has unit structure."""
    out = {"right": majority(labels if isolated else labels[u.start:u.end]) == u.label}
    if recipe_units_fn is not None:
        vu = recipe_units_fn(ref); k = next(i for i, x in enumerate(vu) if x.start <= u.start < x.end)
        out["matched"] = bool(unit_outcomes(labels, vu)[0][k]["matched"])
    return out


def units_from_ref(ref: np.ndarray, sid: str) -> list[Unit]:
    change = np.flatnonzero(np.diff(ref)) + 1; starts = np.r_[0, change]; ends = np.r_[change, ref.size]
    labs = [int(ref[a]) for a in starts]
    return [Unit(sid, i, int(a), int(b), labs[i], labs[i - 1] if i else None, labs[i + 1] if i + 1 < len(labs) else None, ())
            for i, (a, b) in enumerate(zip(starts, ends))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True); ap.add_argument("--spec", required=True); ap.add_argument("--protocol", required=True)
    ap.add_argument("-c", "--config", default="configs/experiments/failure_analysis_v1.yaml")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); paths = load_paths(); spec = json.loads(Path(a.spec).read_text())["encoders"]["E0"]
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = Inputs(cfg, Path(a.protocol), Path(spec["feature_manifest"]), paths.cache_root)
    b3_params = json.loads((Path(spec["simple_run"]) / "selected.json").read_text())["selected"]["b3"]["main"]
    asrf = {f"ASRF_s{k}": load_asrf(Path(rd), paths.repo_root, inputs.feature_dim, device) for k, rd in sorted(spec["asrf"].items())}
    inv = pd.read_csv(Path(a.inventory) / "units.csv", dtype={"stream": str}); inv = inv[inv.enc == "E0"]
    right_all = inv["B3_matched"] & inv["ASRF_s0_matched"] & inv["ASRF_s1_matched"] & inv["ASRF_s2_matched"]
    targets = inv[(inv.label == A4C) & ((inv.left == A5C) | (inv.right == A5C))]
    controls = inv[(inv.label == A4C) & (inv.left != A5C) & (inv.right != A5C) & right_all]
    rows = []
    for h in HALVES:
        streams = {s.recipe.recipe_id: s for s in inputs.halves[h].multi}; pools = donors(inputs, h)
        for kind, table in (("target", targets), ("control", controls)):
            for r in table[table.half == h].itertuples():
                s = streams[r.stream]; units = stream_units(s.recipe); u = units[int(r.unit)]
                for vname, plan in plans(units, u.index, kind == "control").items():
                    for draw in range(DRAWS if plan else 1):
                        rng = np.random.default_rng(int(hashlib.sha256(f"{h}|{r.stream}|{r.unit}|{vname}|{draw}".encode()).hexdigest()[:8], 16))
                        if plan is None:
                            feat, prob, ref, blocks = s.feat[u.start:u.end], s.prob[u.start:u.end], s.labels[u.start:u.end].astype(int), []
                        else:
                            feat, prob, ref, blocks = variant_stream(s, units, u.index, plan, inputs.store, pools, rng)
                        fn = None if plan is None or vname == "same_class_neighbours" else (lambda x: units_from_ref(x, r.stream))
                        row = {"half": h, "kind": kind, "stream": r.stream, "unit": int(r.unit), "length": u.length, "variant": vname, "draw": draw,
                               "a5c_neighbours": int((u.left == A5C) + (u.right == A5C)), "sources": r.sources,
                               "a5c_blocks": sum(b["class"] == A5C for b in blocks),
                               "a5c_blocks_read_as_a5c": sum(b["class"] == A5C and b["own_class"] == A5C for b in blocks),
                               "donors": json.dumps(blocks)}
                        res = {"B3": evaluate(b3(prob, b3_params), ref, u, plan is None, fn)}
                        seq = Seq("v", False, feat.astype(np.float16), ref.astype(np.int64), np.zeros(0, int), prob.astype(np.float32))
                        for name, (head, model) in asrf.items():
                            res[name] = evaluate(head.decode(model, [seq], device)["v"]["labels"], ref, u, plan is None, fn)
                        for name, v in res.items():
                            row[f"{name}_right"] = v["right"]; row[f"{name}_matched"] = v.get("matched")
                            row[f"{name}_orig_matched"] = bool(getattr(r, f"{name}_matched"))
                        rows.append(row)
        print(f"{h}: {len(rows)} rows", flush=True)
    df = pd.DataFrame(rows); run_dir = new_run_dir(cfg, "md4_neighbour_swap"); df.to_csv(run_dir / "swap.csv", index=False)
    summary = {}
    for kind in ("target", "control"):
        d = df[df.kind == kind]
        for name in ("B3", *asrf):
            for status in ((False, True) if kind == "target" else (True,)):
                g = d[d[f"{name}_orig_matched"] == status]
                summary[f"{kind}|{name}|originally {'right' if status else 'wrong'}"] = {
                    v: {"units": int(g[g.variant == v][["half", "stream", "unit"]].drop_duplicates().shape[0]),
                        "right": float(g[g.variant == v][f"{name}_right"].mean()) if len(g[g.variant == v]) else None,
                        "matched": float(g[g.variant == v][f"{name}_matched"].dropna().astype(float).mean()) if g[g.variant == v][f"{name}_matched"].notna().any() else None}
                    for v in d.variant.unique()}
    (run_dir / "swap.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1)); print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
