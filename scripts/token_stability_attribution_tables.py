"""Pure parts of the token stability and segmentation attribution round (configs/experiments/token_stability_attribution_v1.yaml).

- The four reference-conditioned readout cells: a source (CLS or token) per window from a reference-label mask, chosen
  before the nearest-request mapping. They are diagnostics, never deployable systems or attainable scores.
- The evaluator's class-aware one-to-one segment matching with its exact greedy order (descending temporal IoU, then
  prediction index, then reference index), TP / FP / FN, each reference segment's status between two systems and their
  unmatched predicted segments.
- The distance from a request to the nearest reference view change, and its bins.
- Overlapping-anchor observations and pairs: one anchor per sample and window, the complete three-slot neighbourhood,
  the sample and marker triples, A03's selected output anchor, group weights and the base-2 Jensen-Shannon divergence.
- A contrast bootstrap that draws exactly as scripts/report_model_development.paired_bootstrap, for the four-cell
  interaction.
- The fixed nonexclusive readings and the markdown tables.
Nothing here reads a file or returns a recording id; callers pass arrays and get counts.
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations

import numpy as np

from echo_routing.evaluate.failure_units import session_of
from echo_routing.evaluate.metrics_segmentation import segment_f1, temporal_iou
from echo_routing.temporal.segments import labels_to_segments
from scripts.raw_video_context_tables import NEG, NONE, POS, ci, md_table  # noqa: F401  (re-exported for the report)
from scripts.report_model_development import BOOT_SEED, N_BOOT, metric_fns

HALVES = ("selection", "calibration")
KINDS = ("native", "pairs", "multi")
CELLS = ("CC", "TC", "CT", "TT")
SOURCES = {"CC": (0, 0), "TC": (1, 0), "CT": (0, 1), "TT": (1, 1)}          # (mixed-window source, pure-window source); 1 = token
CONTRASTS = {"TC - CC": {"TC": 1, "CC": -1}, "CT - CC": {"CT": 1, "CC": -1}, "TT - CC": {"TT": 1, "CC": -1},
             "TT - TC - CT + CC": {"TT": 1, "TC": -1, "CT": -1, "CC": 1}}
BINS = ("0-5", "6-10", ">10", "none")
STATUS = ("both", "token only", "cls only", "neither")
FEW, SHORT, A4C, A5C = 10, 16, 2, 3
RULE_ORDER = ("local_correction_without_segment_recovery", "mixed_vs_pure_tradeoff", "anchor_sensitivity_observed", "unresolved", "end")


# ----------------------------------------------------------------------------------------------- the four cells
def diagnostic_probs(p_cls, p_token, mixed, mixed_source: int, pure_source: int) -> np.ndarray:
    """Window probabilities from a literal source per window (0 = CLS, 1 = token); the mask uses reference labels."""
    p_cls, p_token, mixed = np.asarray(p_cls), np.asarray(p_token), np.asarray(mixed, bool)
    if p_cls.shape != p_token.shape or mixed.shape != p_cls.shape[:1]:
        raise ValueError("the two readouts and the mask must cover the same windows")
    sources = np.where(mixed, mixed_source, pure_source)
    return np.where(sources[:, None] == 1, p_token, p_cls)


def cell_probs(p_cls, p_token, mixed) -> dict:
    return {c: diagnostic_probs(p_cls, p_token, mixed, *SOURCES[c]) for c in CELLS}


# ----------------------------------------------------------------------------------------------- segments
def match_pairs(pred: list, ref: list, thr: float = 0.5) -> list[tuple[int, int]]:
    """The evaluator's one-to-one pairs (prediction index, reference index), in acceptance order."""
    cands = sorted(((temporal_iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(ref)
                    if p.label == t.label and temporal_iou(p, t) >= thr), reverse=True)
    used_p, used_t, out = set(), set(), []
    for _, i, j in cands:
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j); out.append((i, j))
    if len(out) != segment_f1(pred, ref, thr)["matched"]:
        raise AssertionError("the exposed matching disagrees with segment_f1")
    return out


def seg_counts(pred: list, ref: list, thr: float = 0.5) -> dict:
    m = len(match_pairs(pred, ref, thr))
    return {"tp": m, "fp": len(pred) - m, "fn": len(ref) - m, "pred": len(pred), "ref": len(ref)}


def f1_of(c: dict) -> float | None:
    den = 2 * c["tp"] + c["fp"] + c["fn"]
    return 2 * c["tp"] / den if den else None


def compare_segments(labels, pred_a, pred_b, thr: float = 0.5) -> dict:
    """One sequence, a = the token system and b = the CLS system: each reference segment's status, the unmatched
    predicted segments of both (as (start, end, label)) and the counts."""
    ref = labels_to_segments(np.asarray(labels)); pa = labels_to_segments(np.asarray(pred_a)); pb = labels_to_segments(np.asarray(pred_b))
    ma, mb = match_pairs(pa, ref, thr), match_pairs(pb, ref, thr)
    ra, rb = {j for _, j in ma}, {j for _, j in mb}
    status = ["both" if j in ra and j in rb else "token only" if j in ra else "cls only" if j in rb else "neither" for j in range(len(ref))]
    ia, ib = {i for i, _ in ma}, {i for i, _ in mb}
    ua = {(p.start, p.end, p.label) for i, p in enumerate(pa) if i not in ia}; ub = {(p.start, p.end, p.label) for i, p in enumerate(pb) if i not in ib}
    count = lambda p, m: {"tp": len(m), "fp": len(p) - len(m), "fn": len(ref) - len(m), "pred": len(p), "ref": len(ref)}  # noqa: E731
    return {"ref": ref, "status": status, "new_unmatched": ua - ub, "removed_unmatched": ub - ua, "counts_a": count(pa, ma), "counts_b": count(pb, mb)}


def segment_index(ref: list, c: int) -> int:
    """The reference segment that holds sample c."""
    for j, r in enumerate(ref):
        if r.start <= c < r.end:
            return j
    raise ValueError("the sample lies outside the sequence")


def xyx_middle(ref: list, k: int, pair: tuple = (A4C, A5C)) -> bool:
    """report_fact_asrf.xyx_middle: the middle of three consecutive reference segments X, Y, X, with {X, Y} = pair."""
    if k == 0 or k == len(ref) - 1 or ref[k - 1].label != ref[k + 1].label or ref[k].label == ref[k - 1].label:
        return False
    return {ref[k - 1].label, ref[k].label} == set(pair)


# ----------------------------------------------------------------------------------------------- distances
def view_changes(labels) -> np.ndarray:
    labels = np.asarray(labels)
    return np.flatnonzero(np.diff(labels)) + 1 if labels.size > 1 else np.zeros(0, int)


def request_distance(c: int, changes) -> int | None:
    """Samples between the request and the nearest view change b (first sample of the new view); None without one."""
    ch = np.asarray(changes, int)
    if ch.size == 0:
        return None
    return int(np.where(c >= ch, c - ch, ch - 1 - c).min())


def distance_bin(d) -> str:
    return "none" if d is None else "0-5" if d <= 5 else "6-10" if d <= 10 else ">10"


# ----------------------------------------------------------------------------------------------- overlapping anchors
def window_observations(S, M) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(window row, anchor j, sample) once per sample and window: the anchor nearest the marker slot, then the lower j."""
    I, J, T = [], [], []
    for i, (s, m) in enumerate(zip(np.asarray(S), np.asarray(M))):
        best = {}
        for j in range(8):
            t, key = int(s[2 * j]), (abs(2 * j - int(m)), j)
            if t not in best or key < best[t][0]:
                best[t] = (key, j)
        for t, (_, j) in sorted(best.items()):
            I.append(i); J.append(j); T.append(t)
    return np.asarray(I, int), np.asarray(J, int), np.asarray(T, int)


def _triples(S, M, I, J) -> tuple[np.ndarray, np.ndarray]:
    slots = 2 * J[:, None] + np.array([-1, 0, 1])[None, :]
    samp = np.where(slots >= 0, np.asarray(S)[I[:, None], np.clip(slots, 0, 15)], -1)       # slot -1 is padding
    mark = (np.asarray(M)[I][:, None] == slots).astype(int)
    return samp, mark


def anchor_pairs(S, C, M, seq, sel) -> dict:
    """Unordered pairs of observations of the same sequence and sample from windows with distinct centres."""
    S, C, M, seq, sel = (np.asarray(x) for x in (S, C, M, seq, sel))
    I, J, T = window_observations(S, M)
    groups = defaultdict(list)
    for k, (i, t) in enumerate(zip(I, T)):
        groups[(int(seq[i]), int(t))].append(k)
    rows = []
    for g, (key, ks) in enumerate(sorted(groups.items())):
        for a, b in combinations(ks, 2):
            if I[a] == I[b] or C[I[a]] == C[I[b]]:
                continue
            a, b = (a, b) if I[a] < I[b] else (b, a)
            rows.append((g, key[0], key[1], a, b))
    if not rows:
        return {k: np.zeros(0, int) for k in ("group", "seq", "t", "i1", "j1", "i2", "j2", "centre_distance", "pos1", "pos2")} | {
            k: np.zeros(0, bool) for k in ("complete", "rgb_equal", "marker_equal", "main", "broad", "sel1", "sel2")} | {
            "mtriple1": np.zeros((0, 3), int), "mtriple2": np.zeros((0, 3), int)}
    g, q, t, a, b = (np.asarray(x) for x in zip(*rows))
    i1, j1, i2, j2 = I[a], J[a], I[b], J[b]
    s1, m1 = _triples(S, M, i1, j1); s2, m2 = _triples(S, M, i2, j2)
    complete = (j1 >= 1) & (j2 >= 1); rgb = complete & (s1 == s2).all(1)
    return {"group": g, "seq": q, "t": t, "i1": i1, "j1": j1, "i2": i2, "j2": j2, "complete": complete, "rgb_equal": rgb,
            "marker_equal": (m1 == m2).all(1), "main": rgb, "broad": np.ones(len(g), bool), "sel1": j1 == sel[i1], "sel2": j2 == sel[i2],
            "centre_distance": np.abs(C[i1] - C[i2]), "pos1": 2 * j1 - M[i1], "pos2": 2 * j2 - M[i2], "mtriple1": m1, "mtriple2": m2}


def group_weights(group, mask) -> np.ndarray:
    """Each occurrence group has total weight 1 within the mask, split equally over its pairs."""
    group, mask = np.asarray(group), np.asarray(mask, bool); w = np.zeros(group.size)
    if mask.any():
        _, inv, cnt = np.unique(group[mask], return_inverse=True, return_counts=True)
        w[mask] = 1.0 / cnt[inv]
    return w


def js_divergence(p, q) -> np.ndarray:
    """Row-wise Jensen-Shannon divergence in bits (0 for equal distributions, 1 for disjoint ones)."""
    p, q = np.asarray(p, float), np.asarray(q, float); m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = lambda a: np.where(a > 0, a * np.log2(a / m), 0.0).sum(-1)  # noqa: E731
        return 0.5 * kl(p) + 0.5 * kl(q)


def weighted_rate(values, weights) -> float | None:
    w = np.asarray(weights, float)
    return float((np.asarray(values, float) * w).sum() / w.sum()) if w.sum() > 0 else None


# ----------------------------------------------------------------------------------------------- bootstrap
def contrast_bootstrap(tabs: dict, coef: dict, draws: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """sum(coef[c] * metric(c)) with native clips resampled by session and streams by stream, drawn exactly as the
    paired bootstrap draws (the first cell sets the order; every cell sees the same draws)."""
    cells = list(coef); first = tabs[cells[0]]; fns = metric_fns()
    order = first["native"]["video_id"].tolist(); streams = first["multi_seg"]["stream"].unique(); nat = {}
    for c in cells:
        t = tabs[c]
        if set(t["native"]["video_id"]) != set(order) or set(t["multi_seg"]["stream"]) != set(streams) or set(t["multi_frag"]["stream"]) != set(first["multi_frag"]["stream"]):
            raise ValueError("the cells cover different native clips or streams")
        nat[c] = t["native"].assign(session=t["native"]["video_id"].map(session_of)) if c == cells[0] else t["native"].set_index("video_id").loc[order].reset_index()
    ses = nat[cells[0]]["session"]; sessions = ses.unique(); by_session = {s: np.flatnonzero(ses.to_numpy() == s) for s in sessions}
    idx = {c: {t: {s: np.flatnonzero(tabs[c][t]["stream"].to_numpy() == s) for s in streams} for t in ("multi_seg", "multi_frag")} for c in cells}
    point = {k: sum(w * f(tabs[c][t]) for c, w in coef.items()) for k, (t, f) in fns.items()}
    rng = np.random.default_rng(seed); reps = {k: [] for k in fns}
    for _ in range(int(draws)):
        ss = rng.choice(sessions, size=sessions.size, replace=True); st = rng.choice(streams, size=streams.size, replace=True)
        ni = np.concatenate([by_session[s] for s in ss])
        rows = {c: {"native": nat[c].iloc[ni], **{t: tabs[c][t].iloc[np.concatenate([idx[c][t][s] for s in st])] for t in ("multi_seg", "multi_frag")}}
                for c in cells}
        for k, (t, f) in fns.items():
            reps[k].append(sum(w * f(rows[c][t]) for c, w in coef.items()))
    return {k: {"delta": point[k], "ci95": [float(np.percentile(reps[k], 2.5)), float(np.percentile(reps[k], 97.5))]} for k in fns}


# ----------------------------------------------------------------------------------------------- readings
def readings(local_shares: list, trade: dict, anchor_disagreements: list, flags: dict, rules: dict | None = None) -> dict:
    """local_shares: per seed and half, the share of fixed mixed-window requests of multi streams lying in reference
    segments that the CLS system matched (None without such a request); trade[h]: per seed the TC - CC and CT - CC
    segment F1@50 point estimates with their verdicts; anchor_disagreements: per seed and half, the aligned arm's
    main-subset argmax disagreements; flags: the unresolved conditions."""
    local = bool(local_shares) and all(x is not None and x > 0.5 for x in local_shares)
    halves = {}
    for h in HALVES:
        x = trade[h]; opposite = all(a * b < 0 for a, b in zip(x["tc"], x["ct"]))
        consistent = opposite and {x["tc_verdict"], x["ct_verdict"]} == {POS, NEG}
        halves[h] = "consistent" if consistent else "observed" if opposite else "not observed"
    rows = {"local_correction_without_segment_recovery": {"applies": local, "shares": list(local_shares)},
            "mixed_vs_pure_tradeoff": {"applies": any(v != "not observed" for v in halves.values()), "halves": halves},
            "anchor_sensitivity_observed": {"applies": any(int(x) > 0 for x in anchor_disagreements), "disagreements": [int(x) for x in anchor_disagreements]},
            "unresolved": {"applies": any(bool(v) for v in flags.values()), "flags": {k: bool(v) for k, v in flags.items()}},
            "end": {"applies": True}}
    for k in rows:
        if rules and k in rules:
            rows[k]["reading"] = rules[k]
    return {"rows": rows, "applied": [k for k in RULE_ORDER if rows[k]["applies"]]}


# ----------------------------------------------------------------------------------------------- markdown
def _key(k: str, names: tuple) -> dict:
    return dict(zip(names, k.split("|")))


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out |= _flat(v, f"{prefix}{k} ")
        else:
            out[f"{prefix}{k}"] = v
    return out


def tables_md(res: dict) -> str:
    """Every table of the summary; recording, session and stream ids never reach it (callers check). Keys are split into
    columns because markdown cells must not hold '|'."""
    parts = ["# Token stability and segmentation attribution: tables\n",
             "Recompute: `python scripts/token_stability_attribution.py --config configs/experiments/token_stability_attribution_v1.yaml --run`\n",
             "Reference-conditioned cells (TC, CT) use reference labels to choose a readout per window. They are diagnostics, "
             "not deployable systems, attainable scores or upper bounds. Seeds are not extra cases; development halves are reused.\n"]
    for part, title in (("actual", "Actual systems (token at A03's decoder against CLS at A02's decoder)"), ("fixed", "Fixed decoder (TT against CC at A02's CLS setting)")):
        acc = res.get(part, {})
        if not acc:
            continue
        rows = [{**_key(k, ("seed", "half")), "population": kind, **{f"{who} {m}": a["segments"][kind][who][m] for who in ("token", "cls") for m in ("tp", "fp", "fn", "f1")},
                 **{s: a["segments"][kind]["status"][s] for s in STATUS}} for k, a in acc.items() for kind in KINDS]
        parts += [f"## {title}: segments\n", md_table(rows)]
        rows = [{**_key(k, ("seed", "half")), "population": kind, "change": ch, **_flat(a["segments"][kind][ch])}
                for k, a in acc.items() for kind in KINDS for ch in ("newly_matched", "lost", "new_unmatched_pred", "removed_unmatched_pred")]
        parts += [f"## {title}: changed segments\n", md_table(rows)]
        rows = [{**_key(k, ("seed", "half")), "population": kind, "windows": ctx, "distance": b, **_flat(v)}
                for k, a in acc.items() for kind, by in a["windows"].items() for ctx, bins in by.items() for b, v in bins.items()]
        parts += [f"## {title}: changed requested-window correctness by distance\n", md_table(rows)]
        rows = [{**_key(k, ("seed", "half")), "population": kind, "windows": ctx, **_flat(v)}
                for k, a in acc.items() for kind, by in a["containing"].items() for ctx, v in by.items()]
        parts += [f"## {title}: reference segments holding changed requests\n", md_table(rows)]
        rows = [{**_key(k, ("seed", "half")), "population": kind, **_flat(a["probability_only"][kind])} for k, a in acc.items() for kind in KINDS]
        parts += [f"## {title}: probability-only changes\n", md_table(rows)]
        rows = [{**_key(k, ("seed", "half")), "system": who, "decoding": dec, **v} for k, a in acc.items() for who, by in a["switches"].items() for dec, v in by.items()]
        parts += [f"## {title}: native label switches\n", md_table(rows)]
    cells = res.get("cells", {})
    if cells:
        rows = [{**_key(k, ("seed", "half")), "decoding": dec, "cell": c, **{m: x["row"].get(m) for m in ("segment_f1_50_multi", "native_macro_f1", "boundary_precision_multi",
                                                                                                         "boundary_recall_multi", "boundary_f1_multi", "native_internal_cuts_per_clip",
                                                                                                         "same_class_cut_rate_multi")},
                 **{f"{kind} {m}": x["segments"][kind][m] for kind in KINDS for m in ("tp", "fp", "fn")}}
                for k, by in cells.items() for dec, cc in by.items() for c, x in cc.items()]
        parts += ["## Four reference-conditioned cells at the fixed decoder (diagnostics only)\n", md_table(rows)]
    if res.get("contrasts"):
        rows = [{**_key(k, ("decoding", "seed", "half", "contrast")), **{m: ci(v[m]) for m in ("multi segment F1@50", "native macro-F1", "multi boundary recall",
                                                                                             "multi boundary F1", "native cuts per clip", "same-class cut rate")}}
                for k, v in res["contrasts"].items()]
        parts += ["## Cell contrasts (paired bootstrap; native clips by session, streams by stream)\n", md_table(rows)]
    if res.get("contrast_verdicts"):
        rows = [{"decoding": d, "contrast": c, "metric": m, **v} for d, by in res["contrast_verdicts"].items() for c, mm in by.items() for m, v in mm.items()]
        parts += ["## Cell contrast verdicts over seeds\n", md_table(rows)]
    st = res.get("stability", {})
    if st.get("structure"):
        rows = [{"half": h, "subset": sub, "population": kind, "context": ctx, **_flat(v)}
                for h, by in st["structure"].items() for sub, kk in by["support"].items() for kind, cc in kk.items() for ctx, v in cc.items()]
        parts += ["## Overlapping-anchor support\n", md_table(rows)]
        rows = [{"half": h, "population": kind, "distribution": name, "values": json.dumps(v, sort_keys=True)}
                for h, by in st["structure"].items() for kind, dd in by["distributions"].items() for name, v in dd.items()]
        parts += ["## Main-subset distributions\n", md_table(rows)]
    if st.get("rates"):
        rows = [{**_key(k, ("readout", "half")), "subset": sub, "population": kind, "context": ctx, **v}
                for k, by in st["rates"].items() for sub, kk in by.items() for kind, cc in kk.items() for ctx, v in cc.items()]
        parts += ["## Overlapping-anchor rates (group-weighted)\n", md_table(rows)]
    if st.get("union"):
        rows = [{**_key(k, ("readout", "half")), "subset": sub, **v} for k, by in st["union"].items() for sub, v in by.items()]
        parts += ["## Disagreeing pairs over all populations\n", md_table(rows)]
    if st.get("roles"):
        rows = [{**_key(k, ("readout", "half")), "subset": sub, "population": kind, **{n: json.dumps(v, sort_keys=True) for n, v in dd.items()}}
                for k, by in st["roles"].items() for sub, kk in by.items() for kind, dd in kk.items() if dd.get("selected output anchor", {}).get("at least one", 0)
                or any(dd.get(n) for n in ("anchor pairs",))]
        parts += ["## Roles of the disagreeing pairs\n", md_table(rows)]
    for key, title in (("across_seeds", "Distinct cases over the three seeds"), ("readings", "Fixed readings"), ("post_hoc", "Post hoc additions"),
                       ("identity", "Run identity"), ("run", "Stage record"), ("cost", "Cost")):
        if key in res:
            parts += [f"## {title}\n", "```\n" + json.dumps(res[key], indent=1, default=str) + "\n```\n"]
    return "\n".join(parts)
