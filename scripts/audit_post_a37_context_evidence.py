#!/usr/bin/env python
"""Post-A37 cached diagnostic (configs/experiments/post_a37_context_evidence_audit_v1.yaml): sources of A37's S1 fixes,
one eW -> e224 substitution through the frozen W heads, and the matching chain of the extra missed short true segments.

CPU only. Reads A31's two session-fold training pools, the q / e224 / eW caches and the heads committed in f55c1ed;
reproduces A37's readout cells first. Writes id-free aggregates (diagnostic.json); per-unit traces stay on the server
under the run root's private/ directory. Nothing here is a candidate system.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""     # GPU 0 h: hide the GPU before A37's module sets its default

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.b3_coverage as bc  # noqa: E402
import scripts.diagnose_a5c_evidence as d32  # noqa: E402
import scripts.report_raw_video_context as rr  # noqa: E402
import scripts.report_video_switch_head as rsh  # noqa: E402
import scripts.train_session_correction as t31  # noqa: E402
import scripts.train_wider_temporal_evidence as t37  # noqa: E402
from scripts._common import load_cfg  # noqa: E402
from scripts.token_stability_attribution_tables import match_pairs, temporal_iou  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import echo_routing.temporal.interval_recognition as ir  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.temporal.segments import labels_to_segments  # noqa: E402
from echo_routing.temporal.windowed import FAMILY5  # noqa: E402

FOLDS, SEED, A4C, A5C, SHORT, FLOAT_TOL = (0, 1), 0, 2, 3, 16, 1e-9
MODELS = ("Base", "S", "W", "Wsub")
REPRO_KEYS = ("runs", "changes", "correct_A5C_kept", "S1_fixed", "S2_kept", "sequence_row", "counts")


class Ctx:
    def __init__(self, cfg: dict, config: str) -> None:
        self.config, self.v = config, cfg["context_evidence_audit"]; o = self.v["outputs"]
        self.out, self.results, self.ledger = ROOT / o["root"], ROOT / o["results_dir"], ROOT / self.v["budget"]["ledger"]
        a = self.v["inputs"]["a37_config"]; self.a37 = t37.Ctx(load_cfg(SimpleNamespace(config=a, set=[])), a); self.d = self.a37.d


# ----------------------------------------------------------------------------------------------- reproduction
def jsonable(x):
    return json.loads(json.dumps(x, default=float))


def mismatches(a, b, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        return [f"{path}: keys"] if set(a) != set(b) else [m for k in a for m in mismatches(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        return [f"{path}: length"] if len(a) != len(b) else [m for i, (x, y) in enumerate(zip(a, b)) for m in mismatches(x, y, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        return [] if a is not None and b is not None and abs(a - b) <= FLOAT_TOL else ([] if a == b else [path])
    return [] if a == b else [path]


def seq_preds(z: torch.Tensor, D: dict, off: int) -> dict:
    W = t31.pool_W(D["seqs"]); ps = rr.per_sample(W, z[off:off + len(W["seq"])].numpy(), D["seqs"])
    return {q["id"]: ir.relabel(D["base_preds"][q["id"]], ps[q["id"]]) for q in D["seqs"]}


def runs_are_base_rle(D: dict, preds: dict, p: np.ndarray) -> dict:
    """P runs must be the RLE runs of Base, and each run's pooled label must equal the sequence relabel."""
    pk, seqs = D["pk"], D["seqs"]; by = defaultdict(list)
    for k, (i, a, b) in enumerate(D["key"]):
        by[i].append((a, b, int(pk["base"][k])))
    rle_bad = sum(by[i] != ir.runs(D["base_preds"][q["id"]]) for i, q in enumerate(seqs))
    lab_bad = int(sum(int(preds[seqs[i]["id"]][a]) != int(p[k]) for k, (i, a, b) in enumerate(D["key"])))
    return {"sequences_with_rle_mismatch": int(rle_bad), "runs_with_label_mismatch": lab_bad}


# ----------------------------------------------------------------------------------------------- sources
def frame_index(seqs: list[dict]) -> dict:
    idx = defaultdict(set)
    for i, q in enumerate(seqs):
        for r in q["refs"]:
            idx[(r[0], int(r[1]))].add(i)
    return idx


def sources(units: list[tuple], pools: dict, index: dict) -> dict:
    """units: (fold, seq, a, b). Every source of a unit counts; shared source frames link units (union-find)."""
    if not units:
        return {"units": 0}
    frames = [{(r[0], int(r[1])) for r in pools[f][i]["refs"][a:b]} for f, i, a, b in units]; recs = [{v for v, _ in fr} for fr in frames]
    parent = list(range(len(units))); owner = {}

    def find(u: int) -> int:
        while parent[u] != u:
            parent[u] = parent[parent[u]]; u = parent[u]
        return u
    for u, ((f, i, _, _), fr) in enumerate(zip(units, frames)):
        for x in fr:
            if (f, x) in owner:
                parent[find(u)] = find(owner[(f, x)])
            else:
                owner[(f, x)] = u
    groups = Counter(find(u) for u in range(len(units))); per_rec = Counter(v for r in recs for v in r); allr = set(per_rec)
    streams = [len(set().union(*(index[f][x] for x in fr))) for (f, _, _, _), fr in zip(units, frames)]
    gstreams = defaultdict(set)
    for u, (f, i, _, _) in enumerate(units):
        gstreams[find(u)].add((f, i))
    cross = sum(len(gstreams[find(u)]) > 1 for u in range(len(units)))
    return {"units": len(units), "recordings": len(allr), "sessions": len({session_of(v) for v in allr}), "multi_source_units": int(sum(len(r) > 1 for r in recs)),
            "units_per_recording_max": int(max(per_rec.values())), "recordings_with_2plus_units": int(sum(c >= 2 for c in per_rec.values())),
            "source_frame_groups": len(groups), "largest_group": int(max(groups.values())), "units_sharing_frames_with_another_unit": int(sum(groups[find(u)] > 1 for u in range(len(units)))),
            "units_sharing_frames_across_streams": int(cross), "streams_holding_same_frames": {"mean": float(np.mean(streams)), "max": int(max(streams)), "units_in_2plus_streams": int(sum(s > 1 for s in streams))}}


def true_label_weights(Ds: dict, sel: list[tuple]) -> dict:
    """R only: A5C-coded source recordings (true_src) with their summed sample shares; aggregates only."""
    w, n = Counter(), Counter()
    for h, k in sel:
        for v, x in Ds[h]["true_src"][k].items():
            w[v] += x; n[v] += 1
    tot = sum(w.values())
    return {"recordings": len(w), "sessions": len({session_of(v) for v in w}), "weights_desc": sorted((round(float(x), 4) for x in w.values()), reverse=True),
            "runs_per_recording_desc": sorted(n.values(), reverse=True), "max_single_recording_share": float(max(w.values()) / tot) if tot else None, "total_weight": float(tot)}


# ----------------------------------------------------------------------------------------------- segment chain
def match_of(pred, labels) -> tuple[list, list, dict]:
    pa, ref = labels_to_segments(np.asarray(pred)), labels_to_segments(np.asarray(labels)); return pa, ref, {j: i for i, j in match_pairs(pa, ref)}


def holding(segs: list, x: int) -> int:
    return next(k for k, s in enumerate(segs) if s.start <= x < s.end)


def matched_after(new: np.ndarray, base: np.ndarray, a: int, b: int, labels, j: int) -> bool:
    u = new.copy(); u[a:b] = base[a:b]; return j in match_of(u, labels)[2]


def lost_chain(labels, base: np.ndarray, new: np.ndarray, j: int, B: tuple, N: tuple) -> dict:
    """Fixed priority: relabel > merge > rematch > other; single Base P run undos only (825cb5a)."""
    (pa_b, ref, mb), (pa_n, _, mn) = B, N; t = ref[j]; pb = pa_b[mb[j]]; lab = int(new[pb.start]); sn = pa_n[holding(pa_n, pb.start)]
    if len(set(new[pb.start:pb.end].tolist())) != 1:
        raise SystemExit("a Base run carries two labels after relabelling")
    out = {"label": int(t.label), "ref_length": t.length, "pB": (pb.start, pb.end), "W_label_on_pB": lab, "W_segment": (sn.start, sn.end, int(sn.label))}
    if lab != t.label:
        out.update(cls="relabel", absorbed_into_neighbour=bool(sn.length > pb.length), undo_tried=1, undo_restores=int(matched_after(new, base, pb.start, pb.end, labels, j)))
        return out
    cands = [(r[0], r[1]) for r in ir.runs(base) if sn.start <= r[0] and r[1] <= sn.end and (r[0], r[1]) != (pb.start, pb.end) and int(new[r[0]]) != r[2]]
    if sn.length > pb.length and temporal_iou(sn, t) < 0.5:
        ok = [matched_after(new, base, a, b, labels, j) for a, b in cands]
        out.update(cls="merge", merged_runs=len(cands), cands=cands, undo_tried=len(cands), undo_restores=int(any(ok)), undo_restoring_runs=int(sum(ok)))
        return out
    alt = [k for k, s in enumerate(pa_n) if s.label == t.label and temporal_iou(s, t) >= 0.5 and k in mn.values()]
    out.update(cls="rematch" if alt else "other", undo_tried=0, undo_restores=None)
    return out


def gained_chain(base: np.ndarray, new: np.ndarray, j: int, N: tuple) -> dict:
    pa_n, ref, mn = N; sn = pa_n[mn[j]]; changed = bool((new[sn.start:sn.end] != base[sn.start:sn.end]).any())
    same_in_base = any((s.start, s.end, s.label) == (sn.start, sn.end, sn.label) for s in labels_to_segments(base))
    return {"label": int(ref[j].label), "ref_length": ref[j].length, "cls": "relabel_gain" if changed else ("rematch_gain" if same_in_base else "other")}


def short_changes(D: dict, preds: dict, sub: dict | None) -> tuple[list, list]:
    """Short reference segments of multi sequences matched under Base and not under the arm (lost), and the reverse."""
    lost, gained = [], []
    for i, q in enumerate(D["seqs"]):
        if q["kind"] != "multi":
            continue
        base, new = np.asarray(D["base_preds"][q["id"]]), np.asarray(preds[q["id"]]); B, N = match_of(base, q["labels"]), match_of(new, q["labels"])
        U = match_of(np.asarray(sub[q["id"]]), q["labels"]) if sub is not None else None
        for j, t in enumerate(B[1]):
            if t.length > SHORT or (j in B[2]) == (j in N[2]):
                continue
            rec = {"seq": i, "j": j, "start": t.start, "end": t.end, "substitute_matched": (j in U[2]) if U else None}
            (lost if j in B[2] else gained).append({**rec, **(lost_chain(q["labels"], base, new, j, B, N) if j in B[2] else gained_chain(base, new, j, N))})
    return lost, gained


# ----------------------------------------------------------------------------------------------- input description
def window_fields(q: dict, base: np.ndarray, s: np.ndarray, w: np.ndarray, c: int) -> dict:
    lab = np.asarray(q["labels"]); refs = q["refs"]; tgt = lab[c]; sset = set(s.tolist())
    seam = lambda P, f: float(any(f(P[x]) != f(P[x + 1]) for x in range(len(P) - 1)))  # noqa: E731
    cut = lambda P: float(np.any(np.diff(base[P.min():P.max() + 1]) != 0))  # noqa: E731
    return {"other_S": float((lab[s] != tgt).mean()), "other_W": float((lab[w] != tgt).mean()), "new_other_W": float(sum(lab[x] != tgt and x not in sset for x in w.tolist()) / 16),
            "rec_seam_S": seam(s, lambda x: refs[x][0]), "rec_seam_W": seam(w, lambda x: refs[x][0]), "label_seam_S": seam(s, lambda x: lab[x]), "label_seam_W": seam(w, lambda x: lab[x]),
            "base_cut_S": cut(s), "base_cut_W": cut(w), "base_cut_W_only": float(cut(w) and not cut(s))}


def describe(units: list[tuple], Ds: dict, Wins: dict) -> dict:
    """units: (fold, P-run index). Each unit is the multiplicity-weighted mean over its nearest-centre windows."""
    if not units:
        return {"units": 0}
    vals = []
    for h, k in units:
        D, (Wh, pos), pk = Ds[h], Wins[h], Ds[h]["pk"]; sl = slice(pk["offsets"][k], pk["offsets"][k + 1]); rows, wts = pk["rows"][sl] - D["off"], pk["weights"][sl]
        i = int(pk["seq"][k]); q = D["seqs"][i]; base = np.asarray(D["base_preds"][q["id"]])
        f = [window_fields(q, base, Wh["s"][r], pos[r], int(Wh["centre"][r])) for r in rows]
        vals.append({**{key: float(np.average([x[key] for x in f], weights=wts)) for key in f[0]}, "length": int(pk["b"][k] - pk["a"][k]), "pure": float(pk["purity"][k] >= 1.0)})
    keys = vals[0].keys()
    return {"units": len(vals), "mean": {key: float(np.mean([v[key] for v in vals])) for key in keys}, "units_with_new_other_W": int(sum(v["new_other_W"] > 0 for v in vals)),
            "units_with_W_only_base_cut": int(sum(v["base_cut_W_only"] > 0 for v in vals)), "short_units": int(sum(v["length"] <= SHORT for v in vals)), "length_median": float(np.median([v["length"] for v in vals]))}


# ----------------------------------------------------------------------------------------------- per fold
def fold_outputs(ctx: Ctx, X: dict, G: dict, h: int, a8, data, old: dict) -> tuple[dict, dict, dict, dict]:
    """A37's Base / S / W cells reproduced first; only then the one substitution of eW by e224 through the frozen W head."""
    D = d32.fold_data(ctx.d, SEED, h); D["M"] = d32.set_masks(D); D["off"] = off = X["offset"][h]
    P = {"seqs": D["seqs"], "n_of": dict(data.n), "label_of": dict(data.label)}; ev = bc.evalset(P, list(range(len(D["seqs"]))), f"heldout_f{h}"); T, preds, repro = {}, {}, {}
    with torch.no_grad():
        z = {"Base": X["q"], "S": G[h]["S"](X["q"], X["e224"], X["S"]), "W": G[h]["W"](X["q"], X["e224"], X["W"])}
    for m, zz in z.items():
        T[m] = t37.tables(zz, D, off, a8.cfg, P, ev); preds[m] = seq_preds(zz, D, off); want = old[f"heldout_f{h}"][m]
        bad = mismatches(jsonable({k: T[m][k] for k in REPRO_KEYS}), {k: want[k] for k in REPRO_KEYS})
        if m != "Base":
            bad += mismatches(jsonable(t37.coverage(D, D["pk"]["labels"], D["pk"]["base"], T[m]["pred"])[0]), want["coverage"], ".coverage")
        repro[m] = {"mismatches": len(bad), "first": bad[:10], **runs_are_base_rle(D, preds[m], T[m]["pred"])}
    if any(r["mismatches"] or r["sequences_with_rle_mismatch"] or r["runs_with_label_mismatch"] for r in repro.values()):
        raise SystemExit(f"A37 fold {h} not reproduced: {repro}")
    with torch.no_grad():
        zs = G[h]["W"](X["q"], X["e224"], X["e224"])
    T["Wsub"] = t37.tables(zs, D, off, a8.cfg, P, ev); preds["Wsub"] = seq_preds(zs, D, off); repro["Wsub"] = runs_are_base_rle(D, preds["Wsub"], T["Wsub"]["pred"])
    return D, T, preds, repro


def masks(D: dict, T: dict) -> dict:
    y, b = D["pk"]["labels"], D["pk"]["base"]; pS, pW = T["S"]["pred"], T["W"]["pred"]; s1 = D["M"]["S1"]; ok5, ok4 = (y == A5C) & (b == A5C), (y == A4C) & (b == A4C)
    return {"all_P": np.ones(len(y), bool), "R": s1 & (pW == A5C), "S1_rest": s1 & (pW != A5C), "D5": ok5 & (pW != A5C), "D5_S_and_W": ok5 & (pW != A5C) & (pS != A5C),
            "D5_W_only": ok5 & (pW != A5C) & (pS == A5C), "D5_S_only": ok5 & (pS != A5C) & (pW == A5C), "D4": ok4 & (pW == A5C), "A5C_kept_by_W": ok5 & (pW == A5C),
            "W_fixed": (b != y) & (pW == y), "W_broken": (b == y) & (pW != y)}


def retention(y: np.ndarray, b: np.ndarray, pW: np.ndarray, pU: np.ndarray, m: np.ndarray) -> dict:
    fx, br, ch = m & (b != y) & (pW == y), m & (b == y) & (pW != y), m & (pW != b); nm = FAMILY5
    tr = Counter(f"{nm[x]}>{nm[w]}>{nm[u]}" for x, w, u in zip(b[m], pW[m], pU[m]) if w != x or u != x)
    return {"runs": int(m.sum()), "W_changed": int(ch.sum()), "W_changed_kept_by_substitute": int((ch & (pU == pW)).sum()), "W_changed_back_to_Base": int((ch & (pU == b)).sum()),
            "W_fixed": int(fx.sum()), "W_fixed_kept": int((fx & (pU == y)).sum()), "W_broken": int(br.sum()), "W_broken_same_class": int((br & (pU == pW)).sum()), "W_broken_still_wrong": int((br & (pU != y)).sum()),
            "substitute_changes_where_W_kept_Base": int((m & (pW == b) & (pU != b)).sum()), "transitions_Base_W_substitute": dict(tr.most_common())}


def row_of(T: dict) -> dict:
    r, c = T["sequence_row"], T["counts"]
    return {**{k: r[k] for k in ("segment_f1_50_multi", "native_macro_f1", "boundary_recall_multi", "boundary_precision_multi")}, **{k: c[k] for k in ("fp", "fp_short", "fn", "fn_short")},
            "net_fixed": T["runs"]["all"]["net_fixed"], "correct_A5C_kept": T["correct_A5C_kept"]["kept"], "S1_fixed": T["S1_fixed"]["fixed"]}


def chain_summary(lost: list, gained: list, run_set) -> dict:
    nm = FAMILY5; rel = [x for x in lost if x["cls"] == "relabel"]; mer = [x for x in lost if x["cls"] == "merge"]
    undo = lambda xs: {"units": len(xs), "restored_by_one_single_run_undo": int(sum(x["undo_restores"] for x in xs)), "not_confirmed_by_single_run_undo": int(sum(not x["undo_restores"] for x in xs))}  # noqa: E731
    out = {"lost": len(lost), "gained": len(gained), "net_lost": len(lost) - len(gained), "lost_by_class": dict(Counter(x["cls"] for x in lost)), "gained_by_class": dict(Counter(x["cls"] for x in gained)),
           "lost_by_label": dict(Counter(nm[x["label"]] for x in lost)), "gained_by_label": dict(Counter(nm[x["label"]] for x in gained)),
           "relabel_destination": dict(Counter(f"{nm[x['label']]}->{nm[x['W_label_on_pB']]}" for x in rel)), "relabel_absorbed_into_neighbour": int(sum(x["absorbed_into_neighbour"] for x in rel)),
           "relabel_undo": undo(rel), "merge_undo": undo(mer), "merge_neighbour_runs": dict(Counter(int(x["merged_runs"]) for x in mer))}
    if run_set is not None:
        out["pB_run_set"] = dict(Counter(run_set(x) for x in lost)); out["merge_neighbour_run_sets"] = dict(Counter(s for x in mer for s in x["neighbour_sets"]))
    if lost and lost[0].get("substitute_matched") is not None:
        out["substitute_matched_again"] = {"lost": int(sum(x["substitute_matched"] for x in lost)), "gained_still_matched": int(sum(x["substitute_matched"] for x in gained))}
    return out


# ----------------------------------------------------------------------------------------------- run
def run_set_of(D: dict, Mk: dict, T: dict, kmap: dict):
    def which(a: int, b: int, i: int) -> str:
        k = kmap[(i, a, b)]
        for name in ("D5", "D4", "W_broken", "W_fixed"):
            if Mk[name][k]:
                return name
        return "W_changed_other" if T["W"]["pred"][k] != D["pk"]["base"][k] else "unchanged"
    return which


def diagnose(ctx: Ctx) -> str:
    X = t37.feats(ctx.a37, SEED, committed=True); G = {h: {arm: t37.load_head(ctx.a37, arm, SEED, 1 - h) for arm in t37.ARMS} for h in FOLDS}
    a8 = rsh.a08_ctx(); data = t31.vpr.training_data(ctx.a37.a31.a15.a10.a07p); rp = ctx.a37.results / f"readout_s{SEED}.json"; old = json.loads(rp.read_text())["cells"]
    Ds, Ts, Ms, Wins, pools, index, chains, private = {}, {}, {}, {}, {}, {}, {}, {"folds": {}}
    rec = {"readout_sha256": t37.sha256_file(rp), "reproduction": {}, "per_fold": {}}
    for h in FOLDS:
        D, T, preds, repro = fold_outputs(ctx, X, G, h, a8, data, old); Ds[h], Ts[h], Ms[h] = D, T, masks(D, T); rec["reproduction"][f"heldout_f{h}"] = repro
        _, Wh, pos = t37.fold_windows(ctx.a37, h); Wins[h] = (Wh, pos); pools[h] = D["seqs"]; index[h] = frame_index(D["seqs"])
        kmap = {key: k for k, key in enumerate(D["key"])}; which = run_set_of(D, Ms[h], T, kmap); chains[h] = {}
        for arm in ("W", "S", "Wsub"):
            lost, gained = short_changes(D, preds[arm], preds["Wsub"] if arm == "W" else None)
            for x in lost:
                x["fold"] = h; x["pB_set"] = which(*x["pB"], x["seq"]) if arm == "W" else None
                x["neighbour_sets"] = [which(a, b, x["seq"]) for a, b in x.get("cands", [])] if arm == "W" else []
            for x in gained:
                x["fold"] = h
            dn = T[arm]["counts"]["fn_short"] - T["Base"]["counts"]["fn_short"]
            if len(lost) - len(gained) != dn:
                raise SystemExit(f"fold {h} {arm}: lost - gained != fn_short difference")
            chains[h][arm] = (lost, gained)
        rec["per_fold"][f"heldout_f{h}"] = {m: row_of(T[m]) for m in MODELS}
        private["folds"][str(h)] = {arm: {"lost": [{**x, "seq_id": D["seqs"][x["seq"]]["id"]} for x in chains[h][arm][0]], "gained": [{**x, "seq_id": D["seqs"][x["seq"]]["id"]} for x in chains[h][arm][1]]} for arm in chains[h]}
    cat = lambda f: np.concatenate([f(h) for h in FOLDS])  # noqa: E731
    y, b = cat(lambda h: Ds[h]["pk"]["labels"]), cat(lambda h: Ds[h]["pk"]["base"]); p = {m: cat(lambda h, m=m: Ts[h][m]["pred"]) for m in ("S", "W", "Wsub")}
    M = {k: cat(lambda h, k=k: Ms[h][k]) for k in Ms[0]}; units = {k: [(h, k_) for h in FOLDS for k_ in np.flatnonzero(Ms[h][k])] for k in Ms[0]}
    run_units = lambda k: [(h, int(Ds[h]["pk"]["seq"][i]), int(Ds[h]["pk"]["a"][i]), int(Ds[h]["pk"]["b"][i])) for h, i in units[k]]  # noqa: E731
    rec["sets"] = {k: {"sources": sources(run_units(k), pools, index), "substitute": retention(y, b, p["W"], p["Wsub"], M[k]),
                       "description": describe(units[k], Ds, Wins) if k not in ("W_fixed", "W_broken") else None} for k in M}
    rec["sets"]["R"]["A5C_coded_sources"] = true_label_weights(Ds, units["R"]); rec["sets"]["S1_rest"]["A5C_coded_sources"] = true_label_weights(Ds, units["S1_rest"])
    ok5 = (y == A5C) & (b == A5C)
    rec["correct_A5C_losses"] = {arm: {"runs": int(ok5.sum()), "kept": int((ok5 & (p[arm] == A5C)).sum()), "destinations": dict(Counter(FAMILY5[x] for x in p[arm][ok5 & (p[arm] != A5C)]))} for arm in ("S", "W", "Wsub")}
    rec["broken_by_destination"] = {arm: dict(Counter(f"{FAMILY5[t]}->{FAMILY5[x]}" for t, c, x in zip(y, b, p[arm]) if c == t and x != t)) for arm in ("S", "W", "Wsub")}
    seg_units = lambda xs: [(x["fold"], x["seq"], x["start"], x["end"]) for x in xs]  # noqa: E731
    rec["short_segments"] = {}
    for arm in ("W", "S", "Wsub"):
        lost = [x for h in FOLDS for x in chains[h][arm][0]]; gained = [x for h in FOLDS for x in chains[h][arm][1]]
        rec["short_segments"][arm] = {**chain_summary(lost, gained, (lambda x: x["pB_set"]) if arm == "W" else None), "per_fold": {f"heldout_f{h}": [len(chains[h][arm][0]), len(chains[h][arm][1])] for h in FOLDS},
                                      "lost_sources": sources(seg_units(lost), pools, index), "gained_sources": sources(seg_units(gained), pools, index)}
        if arm == "W":
            kmaps = {h: {key: k for k, key in enumerate(Ds[h]["key"])} for h in FOLDS}
            rec["short_segments"]["W"]["lost_description_pB_runs"] = describe([(x["fold"], kmaps[x["fold"]][(x["seq"], *x["pB"])]) for x in lost], Ds, Wins)
    rec["scope"] = ctx.v["substitute"]["limits"]; rec["finished"] = t31.now()
    (ctx.out / "private").mkdir(parents=True, exist_ok=True); t31.write_json(ctx.out / "private" / "units.json", jsonable(private)); t37.publish(ctx, "diagnostic.json", rec)
    s = rec["short_segments"]["W"]; print(json.dumps({"R": rec["sets"]["R"]["sources"], "R_sub": rec["sets"]["R"]["substitute"], "L": {k: s[k] for k in ("lost", "gained", "lost_by_class")}}, indent=1, default=str), flush=True)
    return f"R {rec['sets']['R']['sources']['units']} runs / {rec['sets']['R']['sources']['recordings']} recordings; L {s['lost']} lost / {s['gained']} gained"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--config", required=True)
    a = ap.parse_args(); a.set = []; ctx = Ctx(load_cfg(a), a.config); ctx.out.mkdir(parents=True, exist_ok=True)
    t37.staged(ctx, "diagnostic s0 (CPU)", "cpu", lambda: diagnose(ctx))


if __name__ == "__main__":
    main()
