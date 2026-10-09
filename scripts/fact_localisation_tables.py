"""Per-reference-run measures and the summary tables of the FACT short-segment localisation (CPU only; plan
docs/superpowers/plans/2026-10-09-fact-localisation.md). scripts/diagnose_fact_localisation.py saves the per-sequence
records on the server and calls these to write summary.json and report_tables.md; they read records, never a model.

A record holds the reference labels, the source fragments, the readouts of every stage (E1, FACT i, u, U, the official
FACT output, ASRF's last stage and its refined output), the tokens, the partition entering U and the outputs of the
intervention conditions B and C. Units are reference runs. Matching is segment_f1's class-aware one-to-one rule at IoU
0.5; boundaries are label changes matched one to one within the 0.5 s tolerance; every count carries its distinct source
recordings and sessions, because constructed streams reuse recordings.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from scripts.ev9v_complementarity import md_table
from scripts.report_fact_asrf import xyx_middle
from scripts.report_fan224_visual import matched_true

from echo_routing.evaluate.failure_units import session_of
from echo_routing.evaluate.metrics_boundary import match_boundaries
from echo_routing.evaluate.metrics_segmentation import segment_f1, temporal_iou
from echo_routing.temporal.segments import boundaries_from_labels, labels_to_segments

N_CLASSES = 5
CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
A4C, A5C = CLASSES.index("A4C"), CLASSES.index("A5C")
BLOCKS = ("i", "u", "U")                                       # gtea.yaml: InputBlock, UpdateBlock, UpdateBlockTDU
STAGES = ("E1", "FACT_i", "FACT_u", "FACT_U", "FACT_final", "ASRF_frame", "ASRF_final")
INTERVENTION = {"A": ("FACT_final", "FACT_U"), "B": ("B_final", "B_U"), "C": ("C_final", "C_U")}   # official output, U frame branch
READOUTS = (*STAGES, "B_final", "C_final", "B_U", "C_U")
FACT_CHAIN, ASRF_CHAIN = ("E1", "FACT_i", "FACT_u", "FACT_U", "FACT_final"), ("E1", "ASRF_frame", "ASRF_final")
PAIRS = (*zip(FACT_CHAIN, FACT_CHAIN[1:]), ("E1", "FACT_final"), *zip(ASRF_CHAIN, ASRF_CHAIN[1:]), ("E1", "ASRF_final"))
DIFFS = (("B", "A"), ("C", "A"), ("B", "C"))
GROUPS = ("all", "short", "xyx", "a4c_a5c_xyx")
FOCUS = ("stream:all", "stream:short", "stream:xyx", "stream:a4c_a5c_xyx", "native:all")
MISS_FLAGS = ("no_run_at_iou_50", "other_view_at_iou_50", "same_view_run_taken")
WHAT = ("Post hoc training-set diagnostic of the frozen FACT, ASRF and E1 of the paired FACT-ASRF round. No training and no "
        "development-set result. Conditions B and C of the intervention use the reference partition; they are diagnostics, "
        "never model results.")


# ----------------------------------------------------------------------------------------------- per reference run
def match_pairs(pred: list, true: list, thr: float = 0.5) -> dict[int, int]:
    """segment_f1's one-to-one rule (candidates by IoU, then index, descending): reference index -> predicted index,
    checked against the shared matched_true."""
    cands = sorted(((temporal_iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(true)
                    if p.label == t.label and temporal_iou(p, t) >= thr), reverse=True)
    used, out = set(), {}
    for _, i, j in cands:
        if i not in used and j not in out:
            used.add(i); out[j] = i
    if set(out) != matched_true(pred, true, thr):
        raise AssertionError("the pairs disagree with the shared one-to-one rule")
    return out


def shape(p, t) -> str:
    """The one same-view predicted run overlapping a missed reference run: longer (covers it), shorter (inside it) or
    shifted (crosses one of its ends)."""
    if p.start <= t.start and p.end >= t.end:
        return "longer"
    return "shorter" if p.start >= t.start and p.end <= t.end else "shifted"


def segment_measures(ref: np.ndarray, pred: np.ndarray, tol: float) -> list[dict]:
    """Per reference run: matched one to one (class-aware, IoU 0.5), the largest IoU with any and with a same-view
    predicted run, the predicted classes inside it, and whether each inner end has a predicted change within tol samples
    (any change; not one to one). A miss also gets three non-exclusive flags (no predicted run at IoU 0.5; a run of
    another view at IoU 0.5; a same-view run at IoU 0.5 matched to another reference) and the shape of the same-view
    prediction (absent, fragmented, longer, shorter, shifted). None of these is a cause."""
    true, ps = labels_to_segments(ref), labels_to_segments(pred); pairs = match_pairs(ps, true); owner = {i: j for j, i in pairs.items()}
    cuts = boundaries_from_labels(pred); out = []
    for j, t in enumerate(true):
        over = [i for i, p in enumerate(ps) if p.start < t.end and t.start < p.end]; iou = {i: temporal_iou(ps[i], t) for i in over}
        same = [i for i in over if ps[i].label == t.label]
        row = {"matched": j in pairs, "iou_any": max(iou.values(), default=0.0), "iou_same": max((iou[i] for i in same), default=0.0),
               "inside": np.bincount(pred[t.start:t.end], minlength=N_CLASSES).tolist(),
               "start_found": None if t.start == 0 else bool(np.any(np.abs(cuts - t.start) <= tol)),
               "end_found": None if t.end == ref.size else bool(np.any(np.abs(cuts - t.end) <= tol))}
        if not row["matched"]:
            row["miss"] = {"no_run_at_iou_50": row["iou_any"] < 0.5, "other_view_at_iou_50": any(iou[i] >= 0.5 and ps[i].label != t.label for i in over),
                           "same_view_run_taken": any(iou[i] >= 0.5 and owner.get(i, j) != j for i in same),
                           "shape": "absent" if not same else "fragmented" if len(same) > 1 else shape(ps[same[0]], t)}
        out.append(row)
    return out


def unit_groups(true: list, k: int, short_max: int) -> list[str]:
    g = ["all"] + (["short"] if true[k].length <= short_max else [])
    return g + (["xyx"] if xyx_middle()(true, k) else []) + (["a4c_a5c_xyx"] if xyx_middle((A4C, A5C))(true, k) else [])


def readout(rec: dict, name: str) -> np.ndarray:
    """A stage readout, or an intervention readout (B_final, C_final, B_U, C_U; a native window's are A's)."""
    if name in rec["stages"]:
        return np.asarray(rec["stages"][name])
    cond, part = name.split("_")
    return np.asarray(rec[cond]["final" if part == "final" else "U"])


def own_share(t, part: list) -> float:
    """The largest share of one interval of the partition entering U that a reference run fills. Below one half, the run
    is the minority in every interval it touches, so the U block averages it with more samples of other runs."""
    return max((min(e, t.end) - max(s, t.start)) / (e - s) for s, e in part if s < t.end and t.start < e)


def units_of(rec: dict, tol: float, short_max: int) -> list[dict]:
    """One unit per reference run: its groups, distinct source recordings and sessions, the non-null U tokens of its
    view and the reference runs of its view in the sequence, its share of the partition entering U, and its measures per
    readout."""
    ref = np.asarray(rec["ref"]); true = labels_to_segments(ref); per = {n: segment_measures(ref, readout(rec, n), tol) for n in READOUTS}
    active = [c for c in rec["tokens"]["U"] if c != N_CLASSES]; part = [tuple(i) for i in rec["partition_A"]]; out = []
    runs_of_view = Counter(g.label for g in true)
    for k, t in enumerate(true):
        src = sorted({v for v, a, b in rec["frags"] if a < t.end and t.start < b})
        out.append({"kind": "native" if rec["native"] else "stream", "seq": rec["seq_id"], "k": k, "label": t.label, "length": t.length,
                    "groups": unit_groups(true, k, short_max), "recordings": src, "sessions": sorted({session_of(v) for v in src}),
                    "view_among_U_tokens": t.label in active, "U_tokens_of_view": active.count(t.label), "runs_of_view": runs_of_view[t.label],
                    "own_share": own_share(t, part), "m": {n: per[n][k] for n in READOUTS}})
    return out


def sources(units: list[dict]) -> dict:
    return {"segments": len(units), "recordings": len({v for u in units for v in u["recordings"]}), "sessions": len({v for u in units for v in u["sessions"]})}


# ----------------------------------------------------------------------------------------------- tables of one seed
def fit_table(recs: list[dict], name: str, tol: float) -> dict:
    """Pooled over sequences: frame error; segment F1@50 = 2 x matched / (predicted + reference runs); boundary recall
    and precision (label changes matched one to one within tol samples); exact transcripts. Beside them the per-sequence
    F1 distribution (the training monitor reported the mean of per-sequence F1 over its 16 sequences)."""
    t, f1 = Counter(), []
    for r in recs:
        ref, p = np.asarray(r["ref"]), readout(r, name); ts, ps = labels_to_segments(ref), labels_to_segments(p); m = segment_f1(ps, ts, 0.5)
        tb, pb = boundaries_from_labels(ref), boundaries_from_labels(p); f1.append(m["f1"])
        t.update({"sequences": 1, "samples": int(ref.size), "errors": int((p != ref).sum()), "matched": m["matched"], "pred_runs": len(ps), "ref_runs": len(ts),
                  "exact": int([g.label for g in ps] == [g.label for g in ts]), "b_true": int(tb.size), "b_pred": int(pb.size),
                  "b_matched": len(match_boundaries(pb, tb, tol)), "with_change": int(pb.size > 0)})
    q = dict(zip(("q10", "q25", "median", "q75", "q90"), (float(v) for v in np.percentile(f1, [10, 25, 50, 75, 90]))))
    return {"sequences": t["sequences"], "frame_error": t["errors"] / t["samples"], "segment_f1_50": 2 * t["matched"] / (t["pred_runs"] + t["ref_runs"]),
            "matched_reference_runs": t["matched"], "reference_runs": t["ref_runs"], "predicted_runs": t["pred_runs"], "exact_transcripts": t["exact"],
            "boundary_recall": t["b_matched"] / t["b_true"] if t["b_true"] else None, "boundary_precision": t["b_matched"] / t["b_pred"] if t["b_pred"] else None,
            "predicted_changes": t["b_pred"], "sequences_with_a_change": t["with_change"],
            "per_sequence_f1": {"mean": float(np.mean(f1)), **q, "share_at_1": float(np.mean(np.asarray(f1) == 1.0))}}


def confusions(recs: list[dict], name: str, top: int = 6) -> list[dict]:
    """The most frequent (reference view, predicted view) pairs among wrong samples."""
    c = Counter()
    for r in recs:
        ref, p = np.asarray(r["ref"]), readout(r, name); w = ref != p; c.update(zip(ref[w].tolist(), p[w].tolist()))
    total = sum(c.values())
    return [{"reference": CLASSES[a], "read_as": CLASSES[b], "samples": n, "share_of_errors": n / total} for (a, b), n in c.most_common(top)]


def miss_lengths(units: list[dict], name: str) -> dict:
    miss = [u for u in units if not u["m"][name]["matched"]]
    return {k: sources([u for u in miss if a <= u["length"] <= b]) for k, (a, b) in (("1-10", (1, 10)), ("11-16", (11, 16)), ("17-40", (17, 40)),
                                                                                    ("41-80", (41, 80)), (">80", (81, 10**9)))}


def group_table(units: list[dict], name: str) -> dict:
    """Matched reference runs of one group at one readout with distinct sources, and the descriptors of the misses."""
    miss = [u for u in units if not u["m"][name]["matched"]]; ms = [u["m"][name] for u in miss]
    ends = [e for m in ms for e in (m["start_found"], m["end_found"]) if e is not None]
    mostly = Counter("own view" if int(np.argmax(m["inside"])) == u["label"] else CLASSES[int(np.argmax(m["inside"]))] for u, m in zip(miss, ms))
    med = lambda v: float(np.median(v)) if v else None  # noqa: E731
    return {**sources(units), "matched": len(units) - len(miss), "recall": (len(units) - len(miss)) / len(units) if units else None,
            "unmatched": sources(miss), "unmatched_flags": dict(Counter(" + ".join(k for k in MISS_FLAGS if m["miss"][k]) or "none" for m in ms)),
            "unmatched_shape": dict(Counter(m["miss"]["shape"] for m in ms)), "unmatched_inner_ends_found": [int(sum(ends)), len(ends)],
            "unmatched_read_mostly_as": dict(mostly), "unmatched_iou_any_median": med([m["iou_any"] for m in ms]),
            "unmatched_iou_same_median": med([m["iou_same"] for m in ms]),
            "unmatched_own_view_share_inside_median": med([m["inside"][u["label"]] / u["length"] for u, m in zip(miss, ms)])}


def transitions(units: list[dict]) -> dict:
    """Between readouts a -> b: reference runs matched at a but not at b (lost) and the reverse (recovered)."""
    out = {}
    for a, b in PAIRS:
        ma, mb = [u["m"][a]["matched"] for u in units], [u["m"][b]["matched"] for u in units]
        out[f"{a} -> {b}"] = {"lost": sources([u for u, x, y in zip(units, ma, mb) if x and not y]), "recovered": sources([u for u, x, y in zip(units, ma, mb) if y and not x]),
                              "matched_both": sum(x and y for x, y in zip(ma, mb)), "unmatched_both": sum(not x and not y for x, y in zip(ma, mb))}
    return out


def first_loss(units: list[dict], chain: tuple = FACT_CHAIN) -> dict:
    """The earliest readout of the chain where each reference run is unmatched, how many of those the chain's last
    readout matches again, and the matched pattern along the chain (1 = matched)."""
    first, pattern = {k: [] for k in (*chain, "never unmatched")}, Counter()
    for u in units:
        hits = [u["m"][s]["matched"] for s in chain]; pattern["".join("1" if h else "0" for h in hits)] += 1
        first[next((chain[i] for i, h in enumerate(hits) if not h), "never unmatched")].append(u)
    return {"chain": list(chain), "patterns": dict(pattern.most_common()),
            "first_unmatched_at": {k: {**sources(v), f"matched_at_{chain[-1]}": sum(x["m"][chain[-1]]["matched"] for x in v)} for k, v in first.items() if v}}


def token_summary(kinds: dict) -> dict:
    """Non-null tokens of each block against the sequence's reference runs, and decodes that fell back to the frame
    branch (every final token null)."""
    out = {}
    for kind, rs in kinds.items():
        n_ref = [len(labels_to_segments(np.asarray(r["ref"]))) for r in rs]; d = {}
        for b in BLOCKS:
            act = [sum(c != N_CLASSES for c in r["tokens"][b]) for r in rs]; diff = [a - n for a, n in zip(act, n_ref)]
            d[b] = {"sequences": len(diff), "active_equals_reference_runs": sum(x == 0 for x in diff), "fewer": sum(x < 0 for x in diff),
                    "more": sum(x > 0 for x in diff), "mean_active_minus_runs": float(np.mean(diff)), "all_null": sum(a == 0 for a in act)}
        out[kind] = {**d, "final_decode_fell_back_to_frame_branch": sum(r["assign"] is None for r in rs)}
    return out


def partition_mixing(streams: list[dict], short_max: int) -> dict:
    """The partition entering U (runs of the u block's argmax) against the reference runs of the constructed streams:
    how many reference runs each interval covers and how mixed its reference views are; and per reference run whether
    it has its own interval(s), lies inside a larger interval, or has an end inside an interval while spanning several.
    This describes the grouping; it does not show a cause."""
    cover, minority, cats, on = Counter(), [], {g: {} for g in GROUPS}, Counter()
    for r in streams:
        ref = np.asarray(r["ref"]); true = labels_to_segments(ref); part = [tuple(i) for i in r["partition_A"]]; bounds = {s for s, _ in part} | {ref.size}
        for c in boundaries_from_labels(ref).tolist():   # added after the run (summary only)
            pair = {int(ref[c - 1]), int(ref[c])} == {A4C, A5C}; on["all"] += int(c in bounds); on["all_n"] += 1
            on["a4c_a5c"] += int(pair and c in bounds); on["a4c_a5c_n"] += int(pair)
        for k, t in enumerate(true):
            if xyx_middle((A4C, A5C))(true, k):
                on["middle_ends"] += int(t.start in bounds) + int(t.end in bounds); on["middle_ends_n"] += 2
        for s, e in part:
            cover[min(sum(1 for t in true if t.start < e and s < t.end), 4)] += 1; cnt = np.bincount(ref[s:e], minlength=N_CLASSES)
            if (cnt > 0).sum() > 1:
                minority.append(1 - cnt.max() / (e - s))
        for k, t in enumerate(true):
            aligned, pieces = t.start in bounds and t.end in bounds, sum(1 for s, e in part if s < t.end and t.start < e)
            c = ("own interval" if pieces == 1 else "own intervals, cut inside") if aligned else \
                "inside a larger interval" if any(s <= t.start and t.end <= e for s, e in part) else "spans intervals, an end inside one"
            src = {v for v, a, b in r["frags"] if a < t.end and t.start < b}
            for g in unit_groups(true, k, short_max):
                d = cats[g].setdefault(c, {"segments": 0, "recordings": set(), "sessions": set()})
                d["segments"] += 1; d["recordings"] |= src; d["sessions"] |= {session_of(v) for v in src}
    return {"intervals": sum(cover.values()), "intervals_by_reference_runs_covered": {("4 or more" if k == 4 else str(k)): v for k, v in sorted(cover.items())},
            "mixed_intervals": len(minority), "minority_share_of_mixed_median": float(np.median(minority)) if minority else None,
            "reference_changes_on_a_boundary": {"all": [on["all"], on["all_n"]], "A4C_A5C": [on["a4c_a5c"], on["a4c_a5c_n"]],
                                                "A4C_A5C_middle_ends": [on["middle_ends"], on["middle_ends_n"]]},
            "reference_runs": {g: {c: {"segments": d["segments"], "recordings": len(d["recordings"]), "sessions": len(d["sessions"])} for c, d in v.items()}
                               for g, v in cats.items()}}


def by_own_share(units: list[dict]) -> dict:
    """Added after the run (summary only): constructed-stream reference runs that are the minority in every interval of
    the partition entering U (own share below one half) or the majority of at least one, with distinct sources and the
    runs the official output matches under A, B and C. The exact-alignment categories above put a run inside an own-view
    interval a few samples too long together with a run absorbed by a neighbour; this split separates the two."""
    out = {}
    for g in GROUPS:
        us = [u for u in units if u["kind"] == "stream" and g in u["groups"]]
        for name, pick in (("minority in every interval", lambda u: u["own_share"] < 0.5), ("majority of an interval", lambda u: u["own_share"] >= 0.5)):
            sel = [u for u in us if pick(u)]
            out.setdefault(g, {})[name] = {**sources(sel), "matched": {c: sum(u["m"][INTERVENTION[c][0]]["matched"] for u in sel) for c in "ABC"},
                                           "matched_by_U_frame_branch": {c: sum(u["m"][INTERVENTION[c][1]]["matched"] for u in sel) for c in "ABC"}}
    return out


def changes_inside_intervals(recs: list[dict]) -> dict:
    """Added after the run (summary only): partition intervals entering U inside which the official output (A) or the
    U block's frame branch changes its label, and those whose official label differs from the class of the token the
    decode assigned (on every frame or on some). The official decode gives every frame of one interval the same token and
    mixes in the frame branch at weight 0.1, so the label can leave the token's class only where the frame branch
    outweighs it."""
    out = {}
    for kind, rs in (("stream", [r for r in recs if not r["native"]]), ("native", [r for r in recs if r["native"]])):
        c = Counter()
        for r in rs:
            fin, tok = r["stages"]["FACT_final"], r["tokens"]["U"]
            for s, e in r["partition_A"]:
                c["intervals"] += 1; c["official_output_changes_inside"] += int(len(set(fin[s:e])) > 1)
                c["U_frame_branch_changes_inside"] += int(len(set(r["stages"]["FACT_U"][s:e])) > 1)
                if r["assign"] is None:
                    c["intervals_decoded_by_the_frame_branch_fallback"] += 1; continue
                differ = [fin[i] != tok[r["assign"][i]] for i in range(s, e)]
                c["label_differs_from_the_assigned_token_class_on_every_frame"] += int(all(differ))
                c["label_differs_from_the_assigned_token_class_on_some_frames"] += int(any(differ))
        out[kind] = {k: c[k] for k in ("intervals", "official_output_changes_inside", "U_frame_branch_changes_inside",
                                       "label_differs_from_the_assigned_token_class_on_every_frame",
                                       "label_differs_from_the_assigned_token_class_on_some_frames", "intervals_decoded_by_the_frame_branch_fallback")}
    return out


def changes(units: list[dict], x: str, y: str) -> dict:
    """Reference runs matched under x but not y (fixed) and under y but not x (new errors), with distinct sources."""
    return {"fixed": sources([u for u in units if u["m"][x]["matched"] and not u["m"][y]["matched"]]),
            "new": sources([u for u in units if u["m"][y]["matched"] and not u["m"][x]["matched"]])}


def intervention(streams: list[dict], natives: list[dict], units: list[dict], tol: float) -> dict:
    """A, B and C on the constructed streams: the official output and the U block's frame branch (pooled segment F1@50,
    boundary recall and precision, and their differences); per group the matched runs (no subset F1 or precision) and the
    fixed and new reference runs with distinct sources; the cuts added. A native window's B and C equal A by construction."""
    keys, out = ("segment_f1_50", "boundary_recall", "boundary_precision"), {}
    for label, idx in (("official_output", 0), ("U_frame_branch", 1)):
        rows = {c: fit_table(streams, INTERVENTION[c][idx], tol) for c in "ABC"}
        out[label] = {"rows": {c: {k: rows[c][k] for k in (*keys, "matched_reference_runs", "reference_runs", "predicted_runs", "per_sequence_f1")} for c in "ABC"},
                      "differences": {f"{x} - {y}": {k: None if rows[x][k] is None or rows[y][k] is None else rows[x][k] - rows[y][k] for k in keys}
                                      for x, y in DIFFS}}
    su = [u for u in units if u["kind"] == "stream"]; out["groups"] = {}
    for g in GROUPS:
        us = [u for u in su if g in u["groups"]]; hit = {c: sum(u["m"][INTERVENTION[c][0]]["matched"] for u in us) for c in "ABC"}
        out["groups"][g] = {"segments": len(us), **sources(us), "matched": hit, "recall": {c: hit[c] / len(us) if us else None for c in "ABC"},
                            "changes": {f"{x} - {y}": changes(us, INTERVENTION[x][0], INTERVENTION[y][0]) for x, y in DIFFS}}
    n = [len(r["B"]["cuts"]) for r in streams]
    out["cuts"] = {"streams": len(n), "streams_with_added_cuts": sum(k > 0 for k in n), "added_cuts": sum(n), "per_stream": {str(k): v for k, v in sorted(Counter(n).items())},
                   "C_counts_equal_B": all(len(r["C"]["cuts"]) == len(r["B"]["cuts"]) for r in streams),
                   "streams_with_B_and_C_executed": sum(r["B"]["executed"] and r["C"]["executed"] for r in streams)}
    out["native_windows"] = {"windows": len(natives), "no_cut_added": sum(not r["B"]["cuts"] and not r["C"]["cuts"] for r in natives),
                             "B_C_outputs_reused_from_A": sum(not r["B"]["executed"] for r in natives)}
    return out


def seed_summary(recs: list[dict], units: list[dict], tol: float, short_max: int) -> dict:
    kinds = {"stream": [r for r in recs if not r["native"]], "native": [r for r in recs if r["native"]]}
    groups = {f"{k}:{g}": us for k in kinds for g in GROUPS for us in [[u for u in units if u["kind"] == k and g in u["groups"]]] if us}
    return {"fit": {k: {n: fit_table(rs, n, tol) for n in STAGES} for k, rs in kinds.items()},
            "errors": {k: {n: confusions(rs, n) for n in STAGES} for k, rs in kinds.items()},
            "groups": {gk: {n: group_table(us, n) for n in STAGES} for gk, us in groups.items()},
            "missed_lengths": {gk: {n: miss_lengths(us, n) for n in ("E1", "FACT_u", "FACT_final", "ASRF_final")} for gk, us in groups.items() if gk.endswith(":all")},
            "transitions": {gk: transitions(us) for gk, us in groups.items()}, "first_unmatched": {gk: first_loss(us) for gk, us in groups.items()},
            "tokens": token_summary(kinds), "logit_argmax_mismatches": sum(r["logit_argmax_mismatches"] for r in recs),
            "unmatched_at_FACT_final_with_its_view_among_U_tokens": {gk: [sum(u["view_among_U_tokens"] for u in us if not u["m"]["FACT_final"]["matched"]),
                                                                          sum(not u["m"]["FACT_final"]["matched"] for u in us)] for gk, us in groups.items()},
            "unmatched_at_FACT_final_with_as_many_U_tokens_of_its_view_as_runs_of_it": {   # added after the run (summary only)
                gk: [sum(u["U_tokens_of_view"] >= u["runs_of_view"] for u in us if not u["m"]["FACT_final"]["matched"]),
                     sum(not u["m"]["FACT_final"]["matched"] for u in us)] for gk, us in groups.items()},
            "partition_entering_U": {**partition_mixing(kinds["stream"], short_max), "by_own_share": by_own_share(units),
                                     "changes_inside_intervals": changes_inside_intervals(recs)},
            "intervention": intervention(kinds["stream"], kinds["native"], units, tol)}


def across_seeds(units_by_seed: dict) -> dict:
    """The constructed streams are the same 1,500 bank streams in every seed (three models, not three samples): reference
    runs unmatched in all three seeds and in at least one, per group, with distinct sources. Native windows differ by seed."""
    seeds = sorted(units_by_seed); key = lambda u: (u["seq"], u["k"])  # noqa: E731
    base = {key(u): u for u in units_by_seed[seeds[0]] if u["kind"] == "stream"}
    if any({key(u) for u in units_by_seed[s] if u["kind"] == "stream"} != set(base) for s in seeds):
        raise SystemExit("the seeds do not share the same bank streams")
    out = {}
    for name in ("E1", "FACT_u", "FACT_U", "FACT_final", "ASRF_final"):
        miss = [{key(u) for u in units_by_seed[s] if u["kind"] == "stream" and not u["m"][name]["matched"]} for s in seeds]
        every, some = set.intersection(*miss), set.union(*miss)
        out[name] = {g: {"all_seeds": sources([base[k] for k in every if g in base[k]["groups"]]), "any_seed": sources([base[k] for k in some if g in base[k]["groups"]])}
                     for g in GROUPS}
    nat = [u for s in seeds for u in units_by_seed[s] if u["kind"] == "native"]
    out["native_windows_all_seeds"] = sources(nat)
    return out


# ----------------------------------------------------------------------------------------------- report tables
def ratio(d: dict) -> str:
    return f"{d['matched']}/{d['segments']}"


def tables_md(res: dict) -> str:
    """The generated tables of report_tables.md (a subset of summary.json)."""
    parts = ["# FACT short-segment localisation: tables\n", WHAT + "\n",
             "Recompute: `python scripts/diagnose_fact_localisation.py --config configs/experiments/fact_localisation_v1.yaml --run`\n"]
    fit, grp, first, trans, part, share, on, iv, chg, tok = [], [], [], [], [], [], [], [], [], []
    for seed, s in res["seeds"].items():
        for kind, per in s["fit"].items():
            fit += [{"seed": seed, "kind": kind, "readout": n, "frame error": r["frame_error"], "segment F1@50 (pooled)": r["segment_f1_50"],
                     "per-sequence F1 mean": r["per_sequence_f1"]["mean"], "per-sequence F1 median": r["per_sequence_f1"]["median"],
                     "matched/reference runs": f"{r['matched_reference_runs']}/{r['reference_runs']}", "exact transcripts": f"{r['exact_transcripts']}/{r['sequences']}",
                     "boundary R": r["boundary_recall"], "boundary P": r["boundary_precision"], "predicted changes": r["predicted_changes"]} for n, r in per.items()]
        for gk in FOCUS:
            if gk not in s["groups"]:
                continue
            g = s["groups"][gk]; grp.append({"seed": seed, "group": gk, **{n: ratio(g[n]) for n in STAGES},
                                             "unmatched at FACT_final (recordings/sessions)": f"{g['FACT_final']['unmatched']['recordings']}/{g['FACT_final']['unmatched']['sessions']}"})
            first += [{"seed": seed, "group": gk, "first unmatched at": k, **v} for k, v in s["first_unmatched"][gk]["first_unmatched_at"].items()]
            trans += [{"seed": seed, "group": gk, "readouts": p, "lost": v["lost"]["segments"], "lost rec/ses": f"{v['lost']['recordings']}/{v['lost']['sessions']}",
                       "recovered": v["recovered"]["segments"], "recovered rec/ses": f"{v['recovered']['recordings']}/{v['recovered']['sessions']}"}
                      for p, v in s["transitions"][gk].items()]
        for g, cats in s["partition_entering_U"]["reference_runs"].items():
            part += [{"seed": seed, "group": g, "reference run in the partition entering U": c, **v} for c, v in cats.items()]
        for g, cats in s["partition_entering_U"]["by_own_share"].items():
            share += [{"seed": seed, "group": g, "place": c, "segments": v["segments"], "rec/ses": f"{v['recordings']}/{v['sessions']}",
                       "matched A/B/C": f"{v['matched']['A']}/{v['matched']['B']}/{v['matched']['C']}",
                       "U frame branch A/B/C": "/".join(str(v["matched_by_U_frame_branch"][c]) for c in "ABC")} for c, v in cats.items()]
        share += [{"seed": seed, "group": f"{k} intervals", "place": "changes inside an interval", "segments": v["intervals"],
                   "matched A/B/C": f"official output {v['official_output_changes_inside']} (label off its token's class on every frame: "
                                    f"{v['label_differs_from_the_assigned_token_class_on_every_frame']})",
                   "U frame branch A/B/C": f"{v['U_frame_branch_changes_inside']}"}
                  for k, v in s["partition_entering_U"]["changes_inside_intervals"].items()]
        on += [{"seed": seed, **{k: f"{a}/{n}" for k, (a, n) in s["partition_entering_U"]["reference_changes_on_a_boundary"].items()}}]
        o = s["intervention"]
        for label in ("official_output", "U_frame_branch"):
            iv += [{"seed": seed, "readout": label, "condition": c, **{k: r[k] for k in ("segment_f1_50", "boundary_recall", "boundary_precision")},
                    "matched/reference runs": f"{r['matched_reference_runs']}/{r['reference_runs']}"} for c, r in o[label]["rows"].items()]
            iv += [{"seed": seed, "readout": label, "condition": d, **v} for d, v in o[label]["differences"].items()]
        for g, v in o["groups"].items():
            chg += [{"seed": seed, "group": g, "comparison": d, "matched A/B/C": f"{v['matched']['A']}/{v['matched']['B']}/{v['matched']['C']} of {v['segments']}",
                     "fixed": c["fixed"]["segments"], "fixed rec/ses": f"{c['fixed']['recordings']}/{c['fixed']['sessions']}",
                     "new": c["new"]["segments"], "new rec/ses": f"{c['new']['recordings']}/{c['new']['sessions']}"} for d, c in v["changes"].items()]
        tok += [{"seed": seed, "kind": k, "block": b, **v[b]} for k, v in s["tokens"].items() for b in BLOCKS]
    cuts = [{"seed": seed, **{k: v for k, v in s["intervention"]["cuts"].items() if k != "per_stream"}} for seed, s in res["seeds"].items()]
    parts += ["## Fit of every readout (pooled over sequences; per-sequence F1 beside it)\n", md_table(fit),
              "## Reference runs matched (one to one, class-aware, IoU 0.5), by group\n", md_table(grp),
              "## Earliest unmatched readout along E1 -> FACT i -> u -> U -> final\n", md_table(first),
              "## Lost and recovered between readouts\n", md_table(trans), "## Reference runs in the partition entering U (constructed streams)\n", md_table(part),
              "### By the largest share of one interval a run fills (added after the run; summary only)\n", md_table(share),
              "### Reference changes on a partition boundary (added after the run; summary only)\n", md_table(on),
              "## Intervention at the last U block (constructed streams; B and C use the reference partition)\n", md_table(iv), "### Cuts\n", md_table(cuts),
              "### Matched, fixed and new reference runs by group\n", md_table(chg), "## Tokens\n", md_table(tok),
              "## Across seeds\n", "```\n" + json.dumps(res.get("across_seeds", {}), indent=1) + "\n```\n", "## Cost\n", "```\n" + json.dumps(res.get("cost", {}), indent=1) + "\n```\n"]
    return "\n".join(parts)
