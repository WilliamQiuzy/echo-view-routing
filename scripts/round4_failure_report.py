#!/usr/bin/env python
"""Round 4 report: where the strong baselines fail and what the failures cost (diagnostic only; nothing here is a
model result). Applies configs/experiments/failure_analysis_v1.yaml to the inventory of
scripts/round4_failure_inventory.py:
1. categories per visual input and half: B3 right or wrong x number of ASRF seeds wrong, with units, source clips and
   sessions, and how many of them B7 with B3 also gets wrong;
2. causes of the wrong units per system and per category;
3. visual checks of the wrong units next to units that every main system gets right, of the same class and length
   (controls, standardised over class x length cells);
4. failure rates by length, class, neighbour class and class pair at each change, and the neighbour-check condition;
5. cost in segment F1@50: an ideal repair of each category, and the ideal, segmentation-only and class-only repairs of
   the largest cause, against the +1 point target on each half;
6. the pre-declared decision rule;
7. representative cases for scripts/round4_case_figures.py: per category, the units of its most common cause from the
   source clip with the most units in that category (ties by half, stream and unit).
Writes report.json, report.md and cases.csv into --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.evaluate.failure_units import CAUSES, Unit, repair, segment_counts  # noqa: E402

HALVES = ("selection", "calibration")
ASRF = ("ASRF_s0", "ASRF_s1", "ASRF_s2")
MAIN = ("B3", *ASRF)
SYSTEMS = (*MAIN, "B7_B3")
CLASSES = ["PLAX", "PSAX", "A4C", "A5C", "SC4C"]
CATEGORIES = {"stable_common": lambda d: d.b3 & (d.k == 3), "common_two_plus": lambda d: d.b3 & (d.k >= 2),
              "b3_only": lambda d: d.b3 & (d.k == 0), "asrf_only": lambda d: ~d.b3 & (d.k >= 2)}


def bin_of(n: int, bins: list) -> str:
    return next(str(lo) if lo == hi else (f"{lo}+" if hi >= 1000 else f"{lo}-{hi}") for lo, hi in bins if lo <= n <= hi)


def load(inv: Path, cfg: dict) -> pd.DataFrame:
    u = pd.read_csv(inv / "units.csv", dtype={"stream": str})
    for s in SYSTEMS:
        u[f"{s}_wrong"] = ~u[f"{s}_matched"].astype(bool)
    u["b3"] = u["B3_wrong"]; u["k"] = sum(u[f"{s}_wrong"].astype(int) for s in ASRF)
    u["bin"] = [bin_of(int(n), cfg["rates"]["length_bins"]) for n in u["length"]]
    u["clips"] = u["sources"].str.split(";"); u["sess"] = u["sessions"].str.split(";")
    u["crop_right"] = u["crop_class"] == u["label"]
    for x in range(len(CLASSES)):
        u[f"nb_{x}"] = (u["left"] == x) | (u["right"] == x)
    return u


def count(d: pd.DataFrame) -> dict:
    return {"units": int(len(d)), "clips": len({c for x in d["clips"] for c in x}), "sessions": len({s for x in d["sess"] for s in x})}


def scope(u: pd.DataFrame, enc: str, half: str) -> pd.DataFrame:
    return u[(u.enc == enc) & (u.half == half)] if half != "both" else u[u.enc == enc]


def categories(u: pd.DataFrame) -> dict:
    out = {}
    for enc in ("E0", "E1"):
        for h in (*HALVES, "both"):
            d = scope(u, enc, h)
            out[f"{enc}|{h}"] = {"total": count(d), "wrong_per_system": {s: count(d[d[f"{s}_wrong"]]) for s in SYSTEMS},
                                 "grid": {f"B3 {'wrong' if b else 'right'}, ASRF wrong in {k}": count(d[(d.b3 == b) & (d.k == k)])
                                          for b in (False, True) for k in range(4)},
                                 "categories": {n: {**count(d[f(d)]), "b7_also_wrong": int(d[f(d)]["B7_B3_wrong"].sum())}
                                                for n, f in CATEGORIES.items()}}
    return out


def causes(u: pd.DataFrame, spurious: pd.DataFrame) -> dict:
    out = {}
    for enc in ("E0", "E1"):
        for h in HALVES:
            d = scope(u, enc, h); sp = spurious[(spurious.enc == enc) & (spurious.half == h)]
            out[f"{enc}|{h}"] = {s: {**{c: int((d[f"{s}_cause"] == c).sum()) for c in CAUSES}, "spurious": int((sp.system == s).sum()),
                                     "spurious_inside_a_unit": int(((sp.system == s) & sp.inside).sum())} for s in SYSTEMS}
        d = scope(u, enc, "both")
        for n, f in CATEGORIES.items():
            c = d[f(d)]
            out[f"{enc}|both|{n}"] = {"B3": {x: int((c["B3_cause"] == x).sum()) for x in CAUSES},
                                      "ASRF (wrong seeds pooled)": {x: int(sum((c[f"{s}_cause"] == x).sum() for s in ASRF)) for x in CAUSES}}
    return out


CHECKS = {"full source clip right": lambda d: d.full_clip_right_all.astype(float),
          "own samples right (reference interval)": lambda d: d.crop_right.astype(float),
          "full clip right, own samples wrong": lambda d: (d.full_clip_right_all & ~d.crop_right).astype(float),
          "every fragment's own samples right": lambda d: d.fragments_crop_right_all.astype(float),
          "B3 on the unit alone right": lambda d: (d.iso_B3 == d.label).astype(float),
          "B3 in the stream right (majority)": lambda d: (d.B3_majority == d.label).astype(float),
          "ASRF on the unit alone right (seed mean)": lambda d: sum((d[f"iso_{s}"] == d.label).astype(float) for s in ASRF) / 3,
          "ASRF in the stream right (seed mean, majority)": lambda d: sum((d[f"{s}_majority"] == d.label).astype(float) for s in ASRF) / 3}


def standardised(wrong: pd.DataFrame, controls: pd.DataFrame, f) -> tuple[float | None, float]:
    """Control rate re-weighted to the wrong units' class x length cells; the share of wrong units with controls."""
    cells = wrong.groupby(["label", "bin"]).size(); num = den = 0.0
    for (lab, b), n in cells.items():
        c = controls[(controls.label == lab) & (controls.bin == b)]
        if len(c):
            num += n * float(f(c).mean()); den += n
    return (num / den if den else None), den / max(len(wrong), 1)


def checks(u: pd.DataFrame) -> dict:
    out = {}
    for enc in ("E0", "E1"):
        d = scope(u, enc, "both"); controls = d[~d.b3 & (d.k == 0)]
        groups = {n: d[f(d)] for n, f in CATEGORIES.items()}
        groups.update({"B3 wrong (all)": d[d.b3], "ASRF wrong in at least one seed": d[d.k >= 1]})
        for g, w in groups.items():
            out[f"{enc}|{g}"] = {"n": count(w), **{name: {"wrong": float(f(w).mean()) if len(w) else None,
                                                          "controls": standardised(w, controls, f)[0],
                                                          "coverage": standardised(w, controls, f)[1]} for name, f in CHECKS.items()},
                                 "crop_margin_median": {"wrong": float(w.crop_margin.median()) if len(w) else None,
                                                        "controls": float(controls.crop_margin.median())}}
    return out


def long_form(d: pd.DataFrame, systems: tuple[str, ...]) -> pd.DataFrame:
    return pd.concat([d.assign(system=s, wrong=d[f"{s}_wrong"]) for s in systems], ignore_index=True)


def rates(u: pd.DataFrame, by: list[str]) -> dict:
    out = {}
    for enc in ("E0", "E1"):
        d = scope(u, enc, "both")
        for name, systems in (("B3", ("B3",)), ("ASRF (3 seeds pooled)", ASRF), ("B7 with B3", ("B7_B3",))):
            lf = long_form(d, systems)
            for key, g in lf.groupby(by):
                key = key if isinstance(key, tuple) else (key,)
                out.setdefault(f"{enc}|{name}", {})["|".join(map(str, key))] = {
                    "rate": float(g.wrong.mean()), "wrong": int(g.wrong.sum()), "total": int(len(g)),
                    "wrong_clips": len({c for x in g[g.wrong]["clips"] for c in x})}
    return out


def neighbour_condition(u: pd.DataFrame) -> list[dict]:
    """Pre-declared: units of class c with a neighbour of class x fail at least twice as often as units of class c
    without one (length-standardised), with at least 10 wrong units from at least 5 source clips (main systems, E0)."""
    lf = long_form(scope(u, "E0", "both"), MAIN); out = []
    for c in range(len(CLASSES)):
        for x in range(len(CLASSES)):
            if x == c:
                continue
            w = lf[(lf.label == c) & lf[f"nb_{x}"]]; o = lf[(lf.label == c) & ~lf[f"nb_{x}"]]
            if not len(w):
                continue
            rate_o = standardised(w, o, lambda z: z.wrong.astype(float))[0]
            bad = w[w.wrong]; units = bad.drop_duplicates(["half", "stream", "unit"])
            n_clips = len({k for x_ in units["clips"] for k in x_})
            holds = bool(rate_o is not None and w.wrong.mean() >= 2 * rate_o and len(units) >= 10 and n_clips >= 5)
            out.append({"class": CLASSES[c], "neighbour": CLASSES[x], "rate_with": float(w.wrong.mean()), "rate_without_standardised": rate_o,
                        "wrong_units": int(len(units)), "wrong_clips": n_clips, "units_with": int(len(w) // len(MAIN)), "holds": holds})
    return out


def edge_rates(u: pd.DataFrame, hit: int, near: int) -> dict:
    out = {}
    for enc in ("E0", "E1"):
        d = scope(u, enc, "both"); d = d[d.right.notna()]
        for name, systems in (("B3", ("B3",)), ("ASRF (3 seeds pooled)", ASRF), ("B7 with B3", ("B7_B3",))):
            for (a, b), g in d.groupby([d.label.astype(int), d.right.astype(int)]):
                pair = " / ".join(sorted((CLASSES[a], CLASSES[b])))
                dist = [json.loads(x)[-1] for s in systems for x in g[f"{s}_edge_dist"]]
                rec = out.setdefault(f"{enc}|{name}", {}).setdefault(pair, {"n": 0, "missed": 0, "shifted": 0})
                rec["n"] += len(dist); rec["missed"] += sum(x is None for x in dist); rec["shifted"] += sum(x is not None and x > hit for x in dist)
    return out


class Streams:
    """Reference units, labels, classifier probabilities and system predictions of one input and half."""

    def __init__(self, u: pd.DataFrame, arrays, enc: str, half: str) -> None:
        d = scope(u, enc, half); self.enc, self.half = enc, half
        self.units = {sid: [Unit(sid, int(r.unit), int(r.start), int(r.end), int(r.label), None if pd.isna(r.left) else int(r.left),
                                 None if pd.isna(r.right) else int(r.right), ()) for r in g.sort_values("unit").itertuples()]
                      for sid, g in d.groupby("stream")}
        self.ref = {sid: arrays[f"{half}|ref|{sid}"].astype(int) for sid in self.units}
        self.clf = {sid: arrays[f"{enc}|{half}|clf|{sid}"] for sid in self.units}
        self.pred = {(s, sid): arrays[f"{enc}|{half}|{s}|labels|{sid}"].astype(int) for s in SYSTEMS for sid in self.units}
        self.own = {(s, sid): arrays[f"{enc}|{half}|{s}|prob|{sid}"] for s in ASRF for sid in self.units}

    def f1(self, system: str, which: dict[str, list[int]] | None = None, mode: str = "ideal", own: bool = False) -> float:
        tot = np.zeros(3)
        for sid, units in self.units.items():
            p = self.pred[(system, sid)]
            if which and which.get(sid):
                scores = (self.own[(system, sid)] if own else self.clf[sid]) if mode == "segmentation" else None
                p = repair(p, self.ref[sid], units, which[sid], mode, scores)
            tot += segment_counts(p, units)
        return float(2 * tot[0] / (tot[1] + tot[2]))


def which_of(d: pd.DataFrame) -> dict[str, list[int]]:
    return {sid: g["unit"].astype(int).tolist() for sid, g in d.groupby("stream")}


def costs(u: pd.DataFrame, arrays, target: dict[str, float], target_cause: str) -> dict:
    out = {}
    for enc in ("E0", "E1"):
        for h in HALVES:
            st = Streams(u, arrays, enc, h); d = scope(u, enc, h)
            for s in SYSTEMS:
                base = st.f1(s); wrong = d[d[f"{s}_wrong"]]; tc = wrong[wrong[f"{s}_cause"] == target_cause]
                rec = {"base": base, "target": target[h], "all wrong units, ideal": st.f1(s, which_of(wrong)),
                       **{f"{n}, ideal": st.f1(s, which_of(wrong[f(wrong)])) for n, f in CATEGORIES.items()},
                       f"{target_cause}: ideal": st.f1(s, which_of(tc)), f"{target_cause}: segmentation only": st.f1(s, which_of(tc), "segmentation"),
                       f"{target_cause}: class only": st.f1(s, which_of(tc), "class"), f"{target_cause}: units": int(len(tc))}
                if s in ASRF:
                    rec[f"{target_cause}: segmentation only, own posteriors"] = st.f1(s, which_of(tc), "segmentation", own=True)
                out[f"{enc}|{h}|{s}"] = rec
    return out


def decision(u: pd.DataFrame) -> dict:
    d = scope(u, "E0", "both"); w = d[CATEGORIES["common_two_plus"](d)]
    per_clip = pd.Series([c for x in w["clips"] for c in set(x)]).value_counts(); top5 = set(per_clip.index[:5])
    share_top5 = float(w["clips"].apply(lambda x: bool(top5 & set(x))).mean()); share_clip_wrong = float((~w.full_clip_right_all).mean())
    share_crop_wrong = float((~w.crop_right).mean())
    rule = "data" if share_top5 >= 0.5 or share_clip_wrong >= 0.5 else ("visual" if share_crop_wrong >= 0.5 else "segmentation")
    return {"units": int(len(w)), "top5_clips": per_clip.head(5).to_dict(), "share_in_top5_clips": share_top5,
            "share_full_clip_wrong": share_clip_wrong, "share_own_samples_wrong": share_crop_wrong, "rule": rule}


def decision_metrics(w: pd.DataFrame) -> dict:
    per_clip = pd.Series([c for x in w["clips"] for c in set(x)]).value_counts(); top5 = set(per_clip.index[:5])
    return {"units": int(len(w)), "share_in_top5_clips": float(w["clips"].apply(lambda x: bool(top5 & set(x))).mean()) if len(w) else None,
            "share_full_clip_wrong": float((~w.full_clip_right_all).mean()) if len(w) else None,
            "share_own_samples_wrong": float((~w.crop_right).mean()) if len(w) else None,
            "own_x_full": {f"own {'right' if a else 'wrong'}, clip {'right' if b else 'wrong'}": int(((w.crop_right == a) & (w.full_clip_right_all == b)).sum())
                           for a in (True, False) for b in (True, False)}}


def extra(u: pd.DataFrame, arrays, top5: set[str]) -> dict:
    """Not pre-declared: robustness of the decision inputs, the cost of the five most affected clips and the repairs of
    the second cause (missed_cut)."""
    out = {"decision_inputs": {f"{enc}|{n}": decision_metrics(scope(u, enc, "both")[CATEGORIES[n](scope(u, enc, "both"))])
                               for enc in ("E0", "E1") for n in ("stable_common", "common_two_plus")}}
    for enc in ("E0", "E1"):
        for h in HALVES:
            st = Streams(u, arrays, enc, h); d = scope(u, enc, h)
            for s in SYSTEMS:
                wrong = d[d[f"{s}_wrong"]]; mc = wrong[wrong[f"{s}_cause"] == "missed_cut"]
                in5 = wrong[wrong["clips"].apply(lambda x: bool(top5 & set(x)))]
                out[f"{enc}|{h}|{s}"] = {"base": st.f1(s), "all wrong: segmentation only": st.f1(s, which_of(wrong), "segmentation"),
                                         "all wrong: class only": st.f1(s, which_of(wrong), "class"),
                                         "top5 clips: units": int(len(in5)), "top5 clips: ideal": st.f1(s, which_of(in5)),
                                         "missed_cut: units": int(len(mc)), "missed_cut: ideal": st.f1(s, which_of(mc)),
                                         "missed_cut: segmentation only": st.f1(s, which_of(mc), "segmentation"),
                                         "missed_cut: class only": st.f1(s, which_of(mc), "class")}
    return out


def cases(u: pd.DataFrame) -> pd.DataFrame:
    d = scope(u, "E0", "both"); picks = []
    for n in ("stable_common", "b3_only", "asrf_only"):
        c = d[CATEGORIES[n](d)].copy()
        if not len(c):
            continue
        c["cause"] = c["B3_cause"] if n != "asrf_only" else c[[f"{s}_cause" for s in ASRF]].mode(axis=1)[0]
        top = c["cause"].value_counts().index[0]; c = c[c.cause == top]
        freq = pd.Series([k for x in c["clips"] for k in set(x)]).value_counts()
        c["clip_units"] = c["clips"].apply(lambda x: max(freq[k] for k in x))
        for h in HALVES:
            ch = c[c.half == h].sort_values(["clip_units", "stream", "unit"], ascending=[False, True, True]).head(1)
            picks += [{**r, "category": n, "cause": top} for r in ch[["enc", "half", "stream", "unit", "label", "length", "sources"]].to_dict("records")]
    und = d[CATEGORIES["common_two_plus"](d) & ~d.crop_right].sort_values(["half", "stream", "unit"]).head(1)
    picks += [{**r, "category": "common_two_plus, own samples wrong", "cause": r_cause} for r, r_cause in
              zip(und[["enc", "half", "stream", "unit", "label", "length", "sources"]].to_dict("records"), und["B3_cause"])]
    return pd.DataFrame(picks)


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def cnt(c: dict) -> str:
    return f"{c['units']} ({c['clips']} clips, {c['sessions']} sessions)"


def md_categories(cat: dict) -> list[str]:
    md = ["## 1. Categories (units = reference segments; clips and sessions deduplicated)", ""]
    for enc in ("E0", "E1"):
        md += [f"### {enc}", "", "| | " + " | ".join(("selection", "calibration", "both halves")) + " |", "|---|---|---|---|"]
        md.append("| all units | " + " | ".join(cnt(cat[f"{enc}|{h}"]["total"]) for h in (*HALVES, "both")) + " |")
        for s in SYSTEMS:
            md.append(f"| wrong for {s} | " + " | ".join(cnt(cat[f"{enc}|{h}"]["wrong_per_system"][s]) for h in (*HALVES, "both")) + " |")
        for g in cat[f"{enc}|both"]["grid"]:
            md.append(f"| {g} | " + " | ".join(cnt(cat[f"{enc}|{h}"]["grid"][g]) for h in (*HALVES, "both")) + " |")
        for n in CATEGORIES:
            md.append(f"| **{n}** (B7 with B3 also wrong) | " + " | ".join(
                f"{cnt(cat[f'{enc}|{h}']['categories'][n])}; B7 {cat[f'{enc}|{h}']['categories'][n]['b7_also_wrong']}" for h in (*HALVES, "both")) + " |")
        md.append("")
    return md


def md_causes(cz: dict) -> list[str]:
    md = ["## 2. Causes of wrong units, and spurious predicted segments", "", "| input, half, system | " + " | ".join(CAUSES) +
          " | spurious segments | of which inside one unit |", "|---|" + "---:|" * (len(CAUSES) + 2)]
    for enc in ("E0", "E1"):
        for h in HALVES:
            for s in SYSTEMS:
                r = cz[f"{enc}|{h}"][s]
                md.append(f"| {enc}, {h}, {s} | " + " | ".join(str(r[c]) for c in CAUSES) + f" | {r['spurious']} | {r['spurious_inside_a_unit']} |")
    md += ["", "| input, category (both halves) | causes under B3 | causes under ASRF (wrong seeds pooled) |", "|---|---|---|"]
    for enc in ("E0", "E1"):
        for n in CATEGORIES:
            r = cz[f"{enc}|both|{n}"]
            md.append(f"| {enc}, {n} | " + "; ".join(f"{c} {r['B3'][c]}" for c in CAUSES) + " | " +
                      "; ".join(f"{c} {r['ASRF (wrong seeds pooled)'][c]}" for c in CAUSES) + " |")
    return md + [""]


def md_checks(ck: dict) -> list[str]:
    md = ["## 3. Checks of the wrong units against matched right units", "",
          "Controls are units that B3 and all three ASRF seeds get right, re-weighted to the wrong units' class x length cells.", ""]
    for key, r in ck.items():
        md += [f"### {key.replace('|', ', ')}: {cnt(r['n'])}", "", "| check | wrong units | matched controls |", "|---|---:|---:|"]
        md += [f"| {name} | {pct(r[name]['wrong'])} | {pct(r[name]['controls'])} |" for name in CHECKS]
        md += [f"| median margin of the label in the own-sample mean probability | {r['crop_margin_median']['wrong']:.2f} | "
               f"{r['crop_margin_median']['controls']:.2f} |" if r['crop_margin_median']['wrong'] is not None else "", ""]
    return md


def md_rates(rt: dict, title: str, keys_fmt=lambda k: k) -> list[str]:
    md = [title, ""]
    for sysname, table in rt.items():
        md += [f"**{sysname.replace('|', ', ')}**: " + "; ".join(f"{keys_fmt(k)} {pct(v['rate'])} ({v['wrong']}/{v['total']}, {v['wrong_clips']} clips)"
                                                         for k, v in table.items()), ""]
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("-c", "--config", default="configs/experiments/failure_analysis_v1.yaml")
    a = ap.parse_args(); a.set = []; cfg = load_cfg(a); inv = Path(a.inventory); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    u = load(inv, cfg); spurious = pd.read_csv(inv / "spurious.csv", dtype={"stream": str}); arrays = dict(np.load(inv / "predictions.npz"))
    ij = json.loads((inv / "inventory.json").read_text()); ver = ij["verified_against_reported_rows"]
    strongest = {"selection": ver["E0|selection|B3"]["segment_f1_50"], "calibration": ver["E0|calibration|B7_B3"]["segment_f1_50"]}
    target = {h: v + float(cfg["success"]["segment_f1_margin"]) for h, v in strongest.items()}
    cz = causes(u, spurious); main_counts = {c: sum(cz[f"E0|{h}"][s][c] for h in HALVES for s in MAIN) for c in CAUSES}
    target_cause = max(main_counts, key=main_counts.get)
    res = {"verified": ver, "strongest": strongest, "target": target, "categories": categories(u), "causes": cz,
           "main_cause_counts": main_counts, "target_cause": target_cause, "checks": checks(u),
           "rates_length": rates(u, ["bin"]), "rates_class": rates(u, ["label"]), "rates_class_length": rates(u, ["label", "bin"]),
           "neighbour_condition": neighbour_condition(u), "edge_rates": edge_rates(u, int(cfg["causes"]["hit"]), int(cfg["causes"]["near"])),
           "costs": costs(u, arrays, target, target_cause), "decision": decision(u),
           "natives": {k: {"wrong": len(v["wrong"]), "cut": len(v["cut"]), "n": v["n"], "wrong_ids": v["wrong"]} for k, v in ij["natives"].items()}}
    res["extra_not_predeclared"] = extra(u, arrays, set(res["decision"]["top5_clips"]))
    cs = cases(u); cs.to_csv(out / "cases.csv", index=False)
    (out / "report.json").write_text(json.dumps(res, indent=1, default=str))
    md = ["# Round 4 failure tables (generated by scripts/round4_failure_report.py; diagnostic only)", "",
          f"Strongest previous configuration: selection {strongest['selection']:.4f}, calibration {strongest['calibration']:.4f}; "
          f"target (+1 point): {target['selection']:.4f} and {target['calibration']:.4f}.", ""]
    md += md_categories(res["categories"]) + md_causes(cz) + md_checks(res["checks"])
    md += md_rates(res["rates_length"], "## 4. Failure rates by unit length (samples at 10 Hz)")
    md += md_rates(res["rates_class"], "## 4b. Failure rates by class", lambda k: CLASSES[int(k)])
    md += md_rates(res["rates_class_length"], "## 4c. Failure rates by class and length", lambda k: f"{CLASSES[int(k.split('|')[0])]} {k.split('|')[1]}")
    md += ["## 4d. Neighbour condition (E0 main systems, both halves; pre-declared)", "",
           "| class | neighbour | failure rate with | without (length-standardised) | wrong units | wrong clips | holds |", "|---|---|---:|---:|---:|---:|---|"]
    md += [f"| {r['class']} | {r['neighbour']} | {pct(r['rate_with'])} | {pct(r['rate_without_standardised'])} | {r['wrong_units']} | "
           f"{r['wrong_clips']} | {'yes' if r['holds'] else 'no'} |" for r in res["neighbour_condition"]]
    md += ["", "## 4e. Reference changes by class pair: missed (no predicted change within 10 samples) and shifted (6-10)", ""]
    for k, table in res["edge_rates"].items():
        md.append(f"**{k.replace('|', ', ')}**: " + "; ".join(f"{p} missed {pct(v['missed'] / v['n'])}, shifted {pct(v['shifted'] / v['n'])} (n={v['n']})"
                                                         for p, v in sorted(table.items())))
        md.append("")
    md += [f"## 5. Cost in segment F1@50 (diagnostic repairs; target cause: {target_cause})", "",
           "| input, half, system | now | all wrong units ideal | stable_common | common_two_plus | b3_only | asrf_only | "
           f"{target_cause}: units | ideal | segmentation only | class only | target |", "|---|" + "---:|" * 11]
    for k, r in res["costs"].items():
        md.append(f"| {k.replace('|', ', ')} | {r['base']:.3f} | {r['all wrong units, ideal']:.3f} | " +
                  " | ".join(f"{r[f'{n}, ideal']:.3f}" for n in CATEGORIES) + f" | {r[f'{target_cause}: units']} | {r[f'{target_cause}: ideal']:.3f} | "
                  f"{r[f'{target_cause}: segmentation only']:.3f}" + (f" (own {r[f'{target_cause}: segmentation only, own posteriors']:.3f})"
                                                                    if f"{target_cause}: segmentation only, own posteriors" in r else "") +
                  f" | {r[f'{target_cause}: class only']:.3f} | {r['target']:.3f} |")
    dc = res["decision"]
    md += ["", "## 6. Decision rule (pre-declared)", "", f"E0 common_two_plus units: {dc['units']}. Share touching the five most affected "
           f"clips: {pct(dc['share_in_top5_clips'])} ({dc['top5_clips']}). Whole source clip assigned to another class: "
           f"{pct(dc['share_full_clip_wrong'])}. Own samples assigned to another class: {pct(dc['share_own_samples_wrong'])}. "
           f"Rule: **{dc['rule']}**.", "", "## 7. Native clips (wrong majority label; clips with internal cuts)", ""]
    md += [f"- {k.replace('|', ', ')}: wrong {v['wrong']}/{v['n']}, cut {v['cut']}" for k, v in res["natives"].items()]
    ex = res["extra_not_predeclared"]
    md += ["", "## 9. Additional analyses (not pre-declared)", "", "| input, group | units | in the five most affected clips | whole clip "
           "another class | own samples another class | own right & clip right / own right & clip wrong / own wrong & clip right / own wrong & clip wrong |",
           "|---|---:|---:|---:|---:|---|"]
    for k, v in ex["decision_inputs"].items():
        md.append(f"| {k.replace('|', ', ')} | {v['units']} | {pct(v['share_in_top5_clips'])} | {pct(v['share_full_clip_wrong'])} | "
                  f"{pct(v['share_own_samples_wrong'])} | " + " / ".join(str(x) for x in v["own_x_full"].values()) + " |")
    md += ["", "| input, half, system | now | all wrong units: segmentation only | all wrong units: class only | units in the five clips | "
           "their ideal repair | missed_cut units | ideal | segmentation only | class only |", "|---|" + "---:|" * 9]
    for k, v in ex.items():
        if k != "decision_inputs":
            md.append(f"| {k.replace('|', ', ')} | {v['base']:.3f} | {v['all wrong: segmentation only']:.3f} | {v['all wrong: class only']:.3f} | "
                      f"{v['top5 clips: units']} | {v['top5 clips: ideal']:.3f} | {v['missed_cut: units']} | "
                      f"{v['missed_cut: ideal']:.3f} | {v['missed_cut: segmentation only']:.3f} | {v['missed_cut: class only']:.3f} |")
    md += ["", "## 8. Representative cases (cases.csv)", "", "| " + " | ".join(cs.columns) + " |", "|" + "---|" * len(cs.columns)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in cs.itertuples(index=False)]
    (out / "report.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
