"""Pure parts of the segment objective round (configs/experiments/segment_objective_v1.yaml).

- Transcripts (run-length-collapsed class sequences) and their normalised Levenshtein distance to the reference.
- A joint bootstrap that draws exactly as scripts/report_model_development.paired_bootstrap (native clips by session,
  streams by stream, 2,000 draws, seed 20261002) once per system, so every paired difference uses the same draws; the
  metrics add the multi-stream FP and FN segment counts to the existing six.
- Each reference segment's status between two systems with the evaluator's class-aware greedy IoU 0.5 matching
  (recovered = matched by a only, lost = matched by b only), the unmatched predicted segments and the counts per
  population, with distinct source recordings and sessions.
- The fixed nonexclusive readings and the markdown tables.
Nothing here reads a file or returns a recording id; callers pass arrays and get counts.
"""
from __future__ import annotations

import json

import numpy as np

from echo_routing.evaluate.failure_units import session_of
from echo_routing.temporal.losses.segment_objective import edit_distance  # noqa: F401  (one definition for the monitor and the report)
from echo_routing.temporal.segments import labels_to_segments
from scripts.raw_video_context_tables import NEG, NONE, POS, ci, consistency, md_table  # noqa: F401  (re-exported for the report)
from scripts.report_model_development import BOOT_SEED, N_BOOT
from scripts.report_model_development import metric_fns as base_metric_fns
from scripts.token_stability_attribution_tables import f1_of, match_pairs, xyx_middle

HALVES = ("selection", "calibration")
KINDS = ("native", "pairs", "multi")
CLASS_ORDER = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
FEW, SHORT = 10, 16
BOOT_METRICS = ("multi segment F1@50", "native macro-F1", "multi boundary recall", "multi FP segments", "multi FN segments", "native cuts per clip")
READING_ORDER = ("complete_system_progress", "segment_balancing_effect", "order_objective_effect", "transcript_only_or_local_effect", "trade_off",
                 "uncertain_or_no_qualifying_gain", "end")


# ----------------------------------------------------------------------------------------------- transcripts
def transcript(labels) -> tuple[int, ...]:
    return tuple(int(s.label) for s in labels_to_segments(np.asarray(labels)))


def transcript_distance(pred, ref) -> float:
    """Edit distance between the predicted and the reference transcript over the reference transcript's length."""
    r = transcript(ref)
    return edit_distance(transcript(pred), r) / len(r)


# ----------------------------------------------------------------------------------------------- bootstrap
def metric_fns() -> dict:
    fns = base_metric_fns()
    fns["multi FP segments"] = ("multi_seg", lambda d: float(d["seg_pred"].sum() - d["seg_matched"].sum()))
    fns["multi FN segments"] = ("multi_seg", lambda d: float(d["seg_true"].sum() - d["seg_matched"].sum()))
    return fns


def system_draws(tables: dict, draws: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Point values and the per-draw values of every metric for one system, drawn as paired_bootstrap draws: sessions in
    the order of first appearance among the native clips, streams in table order, sessions then streams per draw."""
    fns = metric_fns(); nat = tables["native"].assign(session=tables["native"]["video_id"].map(session_of))
    sessions = nat["session"].unique(); by_session = {s: np.flatnonzero(nat["session"].to_numpy() == s) for s in sessions}
    streams = tables["multi_seg"]["stream"].unique()
    if set(tables["multi_frag"]["stream"]) != set(streams):
        raise ValueError("the stream tables cover different streams")
    idx = {t: {s: np.flatnonzero(tables[t]["stream"].to_numpy() == s) for s in streams} for t in ("multi_seg", "multi_frag")}
    rng = np.random.default_rng(seed); reps = {k: np.empty(int(draws)) for k in fns}
    for d in range(int(draws)):
        ss = rng.choice(sessions, size=sessions.size, replace=True); st = rng.choice(streams, size=streams.size, replace=True)
        rows = {"native": nat.iloc[np.concatenate([by_session[s] for s in ss])],
                **{t: tables[t].iloc[np.concatenate([idx[t][s] for s in st])] for t in ("multi_seg", "multi_frag")}}
        for k, (t, f) in fns.items():
            reps[k][d] = f(rows[t])
    order = {"native": nat["video_id"].tolist(), "streams": streams.tolist(), "draws": int(draws), "seed": int(seed)}
    return {"point": {k: f(nat if t == "native" else tables[t]) for k, (t, f) in fns.items()}, "draws": reps, "order": order}


def paired(a: dict, b: dict) -> dict:
    """a - b per metric with the 95% percentile interval of the per-draw differences (the same draws for both)."""
    if a["order"] != b["order"]:
        raise ValueError("the two systems were drawn over different clips, streams or draws")
    return {k: {"delta": a["point"][k] - b["point"][k], "ci95": [float(np.percentile(a["draws"][k] - b["draws"][k], 2.5)),
                                                                 float(np.percentile(a["draws"][k] - b["draws"][k], 97.5))]}
            for k in a["point"]}


# ----------------------------------------------------------------------------------------------- segments
def _counts(pred: list, ref: list, m: list) -> dict:
    return {"tp": len(m), "fp": len(pred) - len(m), "fn": len(ref) - len(m), "pred": len(pred), "ref": len(ref)}


def segment_status(labels, pred_a, pred_b, thr: float = 0.5) -> dict:
    """One sequence: every reference segment's status (both, a only = recovered by a, b only = lost by a, neither), the
    unmatched predicted segments new in a and removed from b (as (start, end, label)) and both systems' counts."""
    ref = labels_to_segments(np.asarray(labels)); pa = labels_to_segments(np.asarray(pred_a)); pb = labels_to_segments(np.asarray(pred_b))
    ma, mb = match_pairs(pa, ref, thr), match_pairs(pb, ref, thr); ra, rb = {j for _, j in ma}, {j for _, j in mb}
    status = ["both" if j in ra and j in rb else "a only" if j in ra else "b only" if j in rb else "neither" for j in range(len(ref))]
    ia, ib = {i for i, _ in ma}, {i for i, _ in mb}
    ua = {(p.start, p.end, p.label) for i, p in enumerate(pa) if i not in ia}; ub = {(p.start, p.end, p.label) for i, p in enumerate(pb) if i not in ib}
    return {"ref": ref, "status": status, "new_unmatched": ua - ub, "removed_unmatched": ub - ua, "counts_a": _counts(pa, ref, ma), "counts_b": _counts(pb, ref, mb)}


def units(recordings) -> dict:
    rec = set(recordings); ses = {session_of(v) for v in rec}
    return {"recordings": len(rec), "sessions": len(ses), "few_units": len(ses) < FEW}


def _bucket() -> dict:
    return {"segments": 0, "by_class": dict.fromkeys(CLASS_ORDER, 0), "short": 0, "long": 0, "xyx_a4c_a5c": 0, "_rec": set()}


def _close(b: dict) -> dict:
    rec = b.pop("_rec")
    return b | units(rec)


def segment_accounting(seqs: list[dict], preds_a: dict, preds_b: dict) -> tuple[dict, list]:
    """Per population: both systems' TP / FP / FN and F1, the status counts, the recovered and lost reference segments
    and the new and removed unmatched predicted segments (class, at most 16 samples, A4C/A5C X-Y-X middles, distinct
    source recordings and sessions). The private list (sequence ids, positions) stays on the server."""
    out = {k: {"a": dict.fromkeys(("tp", "fp", "fn", "pred", "ref"), 0), "b": dict.fromkeys(("tp", "fp", "fn", "pred", "ref"), 0),
               "status": dict.fromkeys(("both", "a only", "b only", "neither"), 0),
               **{n: _bucket() for n in ("recovered", "lost", "new_unmatched_pred", "removed_unmatched_pred")}} for k in KINDS}
    private = []
    for q in seqs:
        c = segment_status(q["labels"], preds_a[q["id"]], preds_b[q["id"]]); k = q["kind"]; src = np.array([r[0] for r in q["refs"]])
        for who, cc in (("a", c["counts_a"]), ("b", c["counts_b"])):
            for m, v in cc.items():
                out[k][who][m] += v
        for j, (r, s) in enumerate(zip(c["ref"], c["status"])):
            out[k]["status"][s] += 1
            if s in ("a only", "b only"):
                b = out[k]["recovered" if s == "a only" else "lost"]
                b["segments"] += 1; b["by_class"][CLASS_ORDER[r.label]] += 1; b["short" if r.length <= SHORT else "long"] += 1
                b["xyx_a4c_a5c"] += int(xyx_middle(c["ref"], j)); b["_rec"] |= set(src[r.start:r.end].tolist())
                private.append({"sequence": q["id"], "segment": j, "start": r.start, "end": r.end, "label": r.label, "status": s})
        for name, segs in (("new_unmatched_pred", c["new_unmatched"]), ("removed_unmatched_pred", c["removed_unmatched"])):
            for s0, e0, lab in sorted(segs):
                b = out[k][name]; b["segments"] += 1; b["by_class"][CLASS_ORDER[lab]] += 1; b["short" if e0 - s0 <= SHORT else "long"] += 1
                b["_rec"] |= set(src[s0:e0].tolist()); private.append({"sequence": q["id"], "start": s0, "end": e0, "label": lab, "status": name})
    for k in KINDS:
        for who in ("a", "b"):
            out[k][who]["f1"] = f1_of(out[k][who])
        for n in ("recovered", "lost", "new_unmatched_pred", "removed_unmatched_pred"):
            out[k][n] = _close(out[k][n])
        for n in ("new_unmatched_pred", "removed_unmatched_pred"):
            out[k][n]["xyx_a4c_a5c"] = None              # not applicable: X-Y-X middles are reference segments
    return out, private


def kind_counts(seqs: list[dict], preds: dict, tables: dict) -> dict:
    """TP / FP / FN per population with the exposed matching, checked against the evaluator's stream tables."""
    out = {k: dict.fromkeys(("tp", "fp", "fn", "pred", "ref"), 0) for k in KINDS}
    for q in seqs:
        pa, ref = labels_to_segments(np.asarray(preds[q["id"]])), labels_to_segments(np.asarray(q["labels"]))
        for m, v in _counts(pa, ref, match_pairs(pa, ref)).items():
            out[q["kind"]][m] += v
    for k, t in (("multi", "multi_seg"), ("pairs", "pairs_seg")):
        want = (int(tables[t]["seg_matched"].sum()), int(tables[t]["seg_pred"].sum()), int(tables[t]["seg_true"].sum()))
        if (out[k]["tp"], out[k]["pred"], out[k]["ref"]) != want:
            raise SystemExit(f"{k}: the exposed matching disagrees with the evaluator's tables")
    return {k: v | {"f1": f1_of(v)} for k, v in out.items()}


def transcript_summary(seqs: list[dict], preds: dict) -> dict:
    """Mean normalised transcript edit distance per population, and how many sequences have an exact transcript."""
    out = {}
    for k in KINDS:
        d = [transcript_distance(preds[q["id"]], q["labels"]) for q in seqs if q["kind"] == k]
        out[k] = {"sequences": len(d), "mean": float(np.mean(d)) if d else None, "exact": int(sum(x == 0 for x in d))}
    return out


def switch_summary(seqs: list[dict], preds: dict) -> dict:
    """Label switches inside native clips: their number, the clips with a switch and those clips' recordings/sessions."""
    cuts = {q["id"]: int(np.count_nonzero(np.diff(np.asarray(preds[q["id"]])))) for q in seqs if q["kind"] == "native"}
    clips = [c for c, n in cuts.items() if n]
    return {"switches": int(sum(cuts.values())), "clips": len(cuts), "clips_with_switches": len(clips), **units(clips)}


# ----------------------------------------------------------------------------------------------- readings
def both_halves(v: dict, want: str) -> bool:
    return all(v[h] == want for h in HALVES)


def readings(ver: dict, goals: dict, extra: dict) -> dict:
    """The fixed nonexclusive readings. ver[comparison][half]: the B3 multi segment F1@50 verdict over the three seeds
    for 'O - B', 'B - F', 'O - F', 'B - frozen Ordered' and 'O - frozen Ordered'; goals['B'|'O']: rt.goal records;
    extra: {'transcript_or_recovery_gain': bool, 'tradeoff': bool} computed by the report."""
    ok = {c: not goals[c]["constraint_failures"] for c in ("B", "O")}
    progress = [c for c in ("B", "O") if both_halves(ver[f"{c} - F"], POS) and both_halves(ver[f"{c} - frozen Ordered"], POS) and ok[c] and goals[c]["met"]]
    applied, notes = [], {}
    if progress:
        applied.append("complete_system_progress")
        notes["complete_system_progress"] = {"candidates": progress, "order_contribution_claimed": both_halves(ver["O - B"], POS)}
    if both_halves(ver["B - F"], POS) and ok["B"]:
        applied.append("segment_balancing_effect")
    if both_halves(ver["O - B"], POS) and ok["O"]:
        applied.append("order_objective_effect")
    if extra["transcript_or_recovery_gain"] and not {"segment_balancing_effect", "order_objective_effect"} & set(applied):
        applied.append("transcript_only_or_local_effect")
    if extra["tradeoff"]:
        applied.append("trade_off")
    if "order_objective_effect" not in applied:
        applied.append("uncertain_or_no_qualifying_gain")
    applied.append("end")
    return {"applied": [r for r in READING_ORDER if r in applied], "notes": notes,
            "inputs": {"verdicts": ver, "constraints_met": ok, "goal_met": {c: goals[c]["met"] for c in ("B", "O")}, **extra}}


# ----------------------------------------------------------------------------------------------- markdown
def tables_md(res: dict) -> str:
    parts = ["# Segment objective round: tables\n",
             "Recompute: `python scripts/report_segment_objective.py --config configs/experiments/segment_objective_v1.yaml --selection "
             "--choices-sha256 <sha> --choices-commit <commit>`\n"]
    rows = [{"system": k, "half": h, "segment F1@50": r.get("segment_f1_50_multi"), "boundary P": r.get("boundary_precision_multi"),
             "boundary R": r.get("boundary_recall_multi"), "boundary F1": r.get("boundary_f1_multi"), "native macro-F1": r.get("native_macro_f1"),
             "switches per clip": r.get("native_internal_cuts_per_clip"), "same-view join cuts": r.get("same_class_cut_rate_multi"),
             "pairs F1@50": r.get("segment_f1_50_pairs"), "A4C read as A5C": r.get("A4C read as A5C"), "A5C read as A4C": r.get("A5C read as A4C")}
            for k, by in res.get("systems", {}).items() for h, r in by.items()]
    if rows:
        parts += ["## Systems\n", md_table(rows)]
    comp = [{"comparison": k, **{m: ci(v[m]) for m in BOOT_METRICS if m in v}} for k, v in res.get("comparisons", {}).items()]
    if comp:
        parts += ["## Paired comparisons (seed | decoding | comparison | half; 95% bootstrap intervals)\n", md_table(comp)]
    ver = [{"decoding | comparison | metric": k, **v} for k, v in res.get("verdicts", {}).items()]
    if ver:
        parts += ["## Three-seed verdicts\n", md_table(ver)]
    acc = []
    for key, by in res.get("accounting", {}).items():
        for kind in KINDS:
            x = by["segments"][kind]
            acc.append({"seed | pair | half": key, "population": kind, "TP a": x["a"]["tp"], "FP a": x["a"]["fp"], "FN a": x["a"]["fn"],
                        "TP b": x["b"]["tp"], "FP b": x["b"]["fp"], "FN b": x["b"]["fn"], "recovered": x["recovered"]["segments"],
                        "recovered sessions": x["recovered"]["sessions"], "lost": x["lost"]["segments"], "lost sessions": x["lost"]["sessions"],
                        "recovered <=16": x["recovered"]["short"], "lost <=16": x["lost"]["short"], "recovered X-Y-X": x["recovered"]["xyx_a4c_a5c"],
                        "lost X-Y-X": x["lost"]["xyx_a4c_a5c"], "new unmatched pred": x["new_unmatched_pred"]["segments"],
                        "removed unmatched pred": x["removed_unmatched_pred"]["segments"]})
    if acc:
        parts += ["## Reference segments recovered and lost (a against b)\n", md_table(acc)]
    goal = [{"group": g, "selection mean": v["mean_segment_f1"]["selection"], "calibration mean": v["mean_segment_f1"]["calibration"],
             "constraint failures": len(v["constraint_failures"]), "met": v["met"]} for g, v in res.get("goal", {}).items()]
    if goal:
        parts += ["## Goal\n", md_table(goal)]
    if "decision" in res:
        parts += ["## Readings\n", "```\n" + json.dumps(res["decision"]["applied"], indent=1) + "\n```\n"]
    return "\n".join(parts)
