#!/usr/bin/env python
"""Does B7's window output answer "which view at this time"? (user plan of 2026-10-05, steps 2-4; no training.)

configs/experiments/b7_window_target_v1.yaml fixes every definition before any result. Only the EV9V validation halves
are read; the test split stays sealed; no EchoNet-Pediatric input. The inputs are those pinned by
ev9v_complementarity_v1, whose load() re-checks every hash and the per-sample alignment on every run.

--phase census      step 2: every B7 window of every validation sequence, as single recording, cross-recording with one
                    view, or mixed views; errors at the centre sample, counted by recording and session
--phase probe-plan  step 3: writes the probe windows (each target's centre frames with its own, a same-view and each
                    other-view context; the reproduction and order checks) and prints the B7 command
--phase probe       step 3: scores the probe outputs; paired other-view minus same-view contrasts, session bootstrap
--phase room        step 4: recordings affected; ideal repair of wrong mixed-views windows (and of every wrong window,
                    as a reference) through B3 with two existing settings; the training gate
census, probe and room write <phase>.json, <phase>.md and tables into a new run directory; probe-plan writes the
windows into the configured probe directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common import load_cfg, new_run_dir  # noqa: E402
from scripts.ev9v_complementarity import decode, evaluate, frozen_systems, load, md_table, spot_ids, unit_frame  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from echo_routing.config import load_paths  # noqa: E402
from echo_routing.evaluate.complementarity import changes, concentration, map_b7, mapping_matrix, normalise_rows  # noqa: E402
from echo_routing.evaluate.failure_units import session_of  # noqa: E402
from echo_routing.evaluate.window_target import (CATEGORIES, category, middle, pick_context, probe_window,  # noqa: E402
                                                 repair, share_bin, window_frames)
from echo_routing.temporal.dev_run import HALVES  # noqa: E402
from echo_routing.temporal.windowed import window_centres  # noqa: E402

CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")


def plans_of(cfg: dict, paths) -> dict:
    return json.loads((paths.repo_root / cfg["inputs"]["B7"]["window_plans"]).read_text())


def window_rows(cfg: dict, data: dict, plans: dict) -> pd.DataFrame:
    """One row per B7 window: its category, centre label and recording, and B7's output (the cached row at the centre)."""
    w, rows = cfg["inputs"]["B7"]["window"], []
    for h in HALVES:
        for sid, sd in data[h].items():
            recs, n = [r[0] for r in plans["streams"][sid]["refs"]], sd.labels.size
            out = sd.b7_stored.argmax(1)
            for j, c in enumerate(window_centres(n, int(w["stride_samples"]))):
                fr = window_frames(int(c), n, int(w["samples"]), int(w["frames"])); labs = sd.labels[fr]
                y, pred, present = int(sd.labels[c]), int(out[c]), {int(x) for x in labs}
                rows.append({"half": h, "kind": sd.kind, "seq": sid, "j": j, "centre": int(c), "category": category([recs[i] for i in fr], labs),
                             "label": y, "pred": pred, "wrong": pred != y, "recording": recs[c], "session": session_of(recs[c]),
                             "centre_frames": int((labs == y).sum()), "classes": len(present), "pred_in_window": pred in present})
    return pd.DataFrame(rows)


def dedup(g: pd.DataFrame) -> dict:
    wr = g[g["wrong"]]
    return {"windows": len(g), "wrong": int(g["wrong"].sum()), "error_rate": float(g["wrong"].mean()) if len(g) else None,
            "recordings": int(g["recording"].nunique()), "recordings_wrong": int(wr["recording"].nunique()),
            "sessions": int(g["session"].nunique()), "sessions_wrong": int(wr["session"].nunique()),
            "recording_averaged_error": float(g.groupby("recording")["wrong"].mean().mean()) if len(g) else None}


def table(df: pd.DataFrame, keys: list[str]) -> list[dict]:
    return [{**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **dedup(g)} for k, g in df.groupby(keys, sort=True)]


def census(cfg: dict, paths, args) -> None:
    _, data, meta = load(cfg, paths)
    win = window_rows(cfg, data, plans_of(cfg, paths)); win["class"] = win["label"].map(dict(enumerate(CLASSES)))
    mixed = win[win["category"] == CATEGORIES[2]].assign(share=lambda d: d["centre_frames"].map(share_bin))
    wm = mixed[mixed["wrong"]].assign(output=lambda d: d["pred"].map(dict(enumerate(CLASSES))))
    pairs = [{"half": h, "centre class": a, "output": b, "windows": len(g), "recordings": int(g["recording"].nunique()),
              "sessions": int(g["session"].nunique()), "output present in window": int(g["pred_in_window"].sum())}
             for (h, a, b), g in wm.groupby(["half", "class", "output"], sort=True)]
    rep = {"inputs": meta, "by_half_kind_category": table(win, ["half", "kind", "category"]), "by_half_category": table(win, ["half", "category"]),
           "by_half_category_class": table(win, ["half", "category", "class"]), "mixed_by_share": table(mixed, ["half", "share"]),
           "mixed_wrong_pairs": pairs,
           "mixed_wrong_output_present": {h: {"present": int(wm[wm["half"] == h]["pred_in_window"].sum()),
                                              "absent": int((~wm[wm["half"] == h]["pred_in_window"]).sum())} for h in HALVES}}
    run_dir = new_run_dir(cfg, "b7_window_target_census")
    win.to_csv(run_dir / "windows.csv", index=False); (run_dir / "census.json").write_text(json.dumps(rep, indent=1, default=str))
    (run_dir / "census.md").write_text("\n".join(["# Step 2: B7 windows on the validation halves\n", md_table(rep["by_half_kind_category"]),
                                                  md_table(rep["by_half_category"]), md_table(rep["by_half_category_class"]),
                                                  "## Mixed-views windows by frames of the centre's class\n", md_table(rep["mixed_by_share"]),
                                                  "## Wrong mixed-views windows: centre class and output\n", md_table(pairs),
                                                  json.dumps(rep["mixed_wrong_output_present"])]))
    print(json.dumps(rep["by_half_category"], indent=1)); print(f"run_dir={run_dir}")


# ----------------------------------------------------------------------------------------------- step 3
def probe_plan(cfg: dict, paths, args) -> None:
    pc, w = cfg["probe"], cfg["inputs"]["B7"]["window"]; size = int(w["samples"])
    _, data, _ = load(cfg, paths); plans = plans_of(cfg, paths); windows, meta = {}, []
    for hi, h in enumerate(HALVES):
        nat = {s: (int(sd.labels[0]), plans["streams"][s]["refs"]) for s, sd in data[h].items() if sd.kind == "native" and sd.labels.size >= size}
        by_class = {k: [s for s, (y, _) in nat.items() if y == k] for k in range(5)}
        for ti, t in enumerate(sorted(nat)):
            y, refs = nat[t]; tw = middle(refs, size); base = {"half": h, "target": t, "label": y, "session": session_of(t)}
            windows[f"{t}|own"] = tw; meta.append({"id": f"{t}|own", **base, "m": None, "condition": "own", "context": t, "context_label": y})
            for x in range(5):
                ctx = pick_context([s for s in by_class[x] if s != t], {session_of(t)}, np.random.default_rng([int(pc["seed"]), hi, ti, x]), session_of)
                cond = "same view" if x == y else "other view"
                if ctx is None:
                    meta.append({"id": None, **base, "m": None, "condition": cond, "context": None, "context_label": x}); continue
                cw = middle(nat[ctx][1], size)
                for m in pc["centre_frames"]:
                    wid = f"{t}|{'same' if x == y else 'other'}|{x}|{int(m)}"
                    windows[wid] = probe_window(tw, cw, int(m), size)
                    meta.append({"id": wid, **base, "m": int(m), "condition": cond, "context": ctx, "context_label": x})
    for sid in spot_ids(plans["streams"], int(cfg["audit"]["spot_check"]["per_bank"])):
        refs = plans["streams"][sid]["refs"]; n = len(refs)
        for c in window_centres(n, int(w["stride_samples"]))[:4]:
            windows[f"repro|{sid}|{int(c)}"] = [list(refs[i]) for i in window_frames(int(c), n, size, int(w["frames"]))]
    for k in sorted(k for k in windows if "|other|" in k and k.endswith("|8"))[:40]:
        windows[f"order|{k}"] = windows[k][::-1]
    d = paths.repo_root / pc["dir"]; d.mkdir(parents=True, exist_ok=True)
    (d / "windows.json").write_text(json.dumps({"frames_root": plans["frames_root"], "windows": windows}))
    pd.DataFrame(meta).to_csv(d / "meta.csv", index=False)
    b = cfg["inputs"]["B7"]
    print(f"wrote {len(windows)} windows to {d / 'windows.json'}. Run:\n  envs/echoviewclip/bin/python scripts/b7_window_probe.py "
          f"--windows {pc['dir']}/windows.json --ckpt {b['checkpoint']} --config {b['runner_config']} --out {pc['dir']}/pred")


def session_ci(df: pd.DataFrame, col: str, draws: int, seed: int) -> list[float]:
    """95% interval of the mean of `col`, resampling the target recordings' sessions."""
    groups = {s: g[col].to_numpy(float) for s, g in df.groupby("session")}; keys = np.array(sorted(groups))
    rng = np.random.default_rng(seed); stats = []
    for _ in range(draws):
        stats.append(np.concatenate([groups[s] for s in rng.choice(keys, size=keys.size, replace=True)]).mean())
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def probe(cfg: dict, paths, args) -> None:
    pc, c = cfg["probe"], cfg["classes"]; d = paths.repo_root / pc["dir"]; bs = pc["bootstrap"]
    with np.load(d / "pred" / "probe.npz") as z:
        ids, raw, stored = [str(x) for x in z["ids"]], z["raw"].astype(float), z["prob"].astype(float)
    mapped, _ = map_b7(raw, mapping_matrix(c["b7_raw9_order"], c["b7_raw9_to_family5"], c["family5"]), stored, c["b7_map_tolerance"])
    p = normalise_rows(mapped, c["b7_min_mapped_mass"]); at = {k: i for i, k in enumerate(ids)}
    diffs = []
    for k in (k for k in ids if k.startswith("repro|")):
        _, sid, cc = k.split("|")
        with np.load(paths.repo_root / cfg["inputs"]["B7"]["pred_dir"] / f"{sid}.npz") as z:
            diffs.append(float(max(np.abs(z["raw"][int(cc)] - raw[at[k]]).max(), np.abs(z["prob"][int(cc)] - stored[at[k]]).max())))
    if not diffs or max(diffs) > 1e-4:
        raise SystemExit(f"the probe runner does not reproduce the B7 cache (max difference {max(diffs, default=float('nan')):.2e})")
    order = [float(np.abs(raw[at[k]] - raw[at[k[len('order|'):]]]).max()) for k in ids if k.startswith("order|")]
    meta = pd.read_csv(d / "meta.csv")
    missing = meta[meta["id"].isna()]; meta = meta.dropna(subset=["id"]).copy()
    meta["pred"] = [int(p[at[k]].argmax()) for k in meta["id"]]; meta["correct"] = meta["pred"] == meta["label"]
    meta["p_target"] = [float(p[at[k]][int(y)]) for k, y in zip(meta["id"], meta["label"])]
    meta["names_context"] = meta["pred"] == meta["context_label"]
    own = meta[meta["condition"] == "own"].set_index("target")
    res = {"reproduction_max_diff": max(diffs), "reproduction_windows": len(diffs), "order_max_diff": max(order), "order_windows": len(order),
           "missing_contexts": {f"{h}|{c}": int(n) for (h, c), n in missing.groupby(["half", "condition"]).size().items()},
           "own_accuracy": {h: float(own[own["half"] == h]["correct"].mean()) for h in HALVES}, "sizes": {}}
    for m in pc["centre_frames"]:
        same = meta[(meta["condition"] == "same view") & (meta["m"] == m)].set_index("target")
        oth = meta[(meta["condition"] == "other view") & (meta["m"] == m)].copy()
        oth = oth[oth["target"].isin(same.index)]
        oth["d_correct"] = oth["correct"].astype(float).to_numpy() - same.loc[oth["target"], "correct"].astype(float).to_numpy()
        oth["d_p"] = oth["p_target"].to_numpy() - same.loc[oth["target"], "p_target"].to_numpy()
        new_err = oth[(oth["d_correct"] < 0)]
        sv = same.assign(d_own=same["correct"].astype(float) - own.loc[same.index, "correct"].astype(float), session=same["session"])
        cells = [{"target": CLASSES[a], "context": CLASSES[b], "pairs": len(g), "targets": int(g["target"].nunique()),
                  "same-view accuracy": float(same.loc[g["target"], "correct"].mean()), "other-view accuracy": float(g["correct"].mean()),
                  "other view names context": int(g["names_context"].sum())}
                 for (a, b), g in oth.groupby(["label", "context_label"], sort=True)]
        res["sizes"][int(m)] = {
            "same_view_accuracy": float(same["correct"].mean()), "other_view_accuracy": float(oth["correct"].mean()),
            "own_accuracy_same_targets": float(own.loc[same.index, "correct"].mean()),
            "d_correct": float(oth["d_correct"].mean()), "d_correct_ci": session_ci(oth, "d_correct", int(bs["draws"]), int(bs["seed"])),
            "d_p_target": float(oth["d_p"].mean()), "d_p_target_ci": session_ci(oth, "d_p", int(bs["draws"]), int(bs["seed"])),
            "same_minus_own": float(sv["d_own"].mean()), "same_minus_own_ci": session_ci(sv.reset_index(), "d_own", int(bs["draws"]), int(bs["seed"])),
            "additional_errors": len(new_err), "additional_errors_naming_context": int(new_err["names_context"].sum()),
            "additional_error_targets": int(new_err["target"].nunique()), "additional_error_sessions": int(new_err["session"].nunique()),
            "pairs": len(oth), "targets": int(oth["target"].nunique()), "cells": cells}
    res["support"] = {int(m): bool(res["sizes"][int(m)]["d_correct_ci"][1] < 0 and res["sizes"][int(m)]["additional_errors"] > 0
                                   and res["sizes"][int(m)]["additional_errors_naming_context"] >= 0.5 * res["sizes"][int(m)]["additional_errors"])
                      for m in pc["primary_centre_frames"]}
    run_dir = new_run_dir(cfg, "b7_window_target_probe")
    meta.to_csv(run_dir / "probe_windows.csv", index=False); (run_dir / "probe.json").write_text(json.dumps(res, indent=1, default=str))
    rows = [{"centre frames": m, **{k: v for k, v in r.items() if k != "cells"}} for m, r in res["sizes"].items()]
    (run_dir / "probe.md").write_text("\n".join(["# Step 3: paired context probe\n", md_table(rows)]
                                                + [f"## {m} centre frames\n\n" + md_table(r["cells"]) for m, r in res["sizes"].items()]
                                                + [json.dumps({k: res[k] for k in ("reproduction_max_diff", "order_max_diff", "own_accuracy", "support")})]))
    print(json.dumps({k: v for k, v in res.items() if k != "sizes"}, indent=1, default=str)); print(md_table(rows)); print(f"run_dir={run_dir}")


# ----------------------------------------------------------------------------------------------- step 4
def room(cfg: dict, paths, args) -> None:
    inputs, data, meta = load(cfg, paths); plans = plans_of(cfg, paths); win = window_rows(cfg, data, plans)
    rc, ref = cfg["room"], cfg["fusion"]["references"]; stride = int(cfg["inputs"]["B7"]["window"]["stride_samples"])
    fz = frozen_systems(cfg, paths)
    settings = {"frozen": (fz["B7 frozen"]["params"], "b7_stored"), "common rule": (rc["decodings"]["common_rule"]["params"], "b7")}
    wrong = win[win["wrong"]]; chosen = {"R1": wrong[wrong["category"] == CATEGORIES[2]], "R2": wrong}
    picks = {v: {(h, s): g["j"].tolist() for (h, s), g in df.groupby(["half", "seq"])} for v, df in chosen.items()}
    wm = chosen["R1"].assign(pair=lambda d: d["label"].map(dict(enumerate(CLASSES))) + " -> " + d["pred"].map(dict(enumerate(CLASSES))))
    affected = {h: {"windows": int((wm["half"] == h).sum()), "recordings": int(wm[wm["half"] == h]["recording"].nunique()),
                    "sessions": int(wm[wm["half"] == h]["session"].nunique()),
                    "by_pair": [{"pair": k, "windows": len(g), "recordings": int(g["recording"].nunique()), "sessions": int(g["session"].nunique())}
                                for k, g in wm[wm["half"] == h].groupby("pair", sort=True)]} for h in HALVES}
    runs, metrics, unit_changes = {}, [], {}
    for name, (params, attr) in settings.items():
        for variant in ("original", "R1", "R2"):
            for h in HALVES:
                probs = {}
                for sid, sd in data[h].items():
                    js = picks[variant].get((h, sid), []) if variant != "original" else []
                    rows = getattr(sd, attr)
                    probs[sid] = repair(rows, window_centres(sd.labels.size, stride), js, sd.labels) if js else rows
                preds = decode(params, probs); m, tables = evaluate(cfg, preds, inputs.halves[h])
                runs[(name, variant, h)] = preds
                metrics.append({"decoding": name, "variant": variant, "half": h, "segment_f1_50_multi": m["segment_f1_50_multi"],
                                "boundary_recall_multi": m["boundary_recall_multi"], "native_macro_f1": m["native_macro_f1"],
                                "native_internal_cuts_per_clip": m["native_internal_cuts_per_clip"], "repaired_windows": sum(len(v) for (hh, _), v in picks.get(variant, {}).items() if hh == h) if variant != "original" else 0})
        for h in HALVES:
            u = unit_frame(cfg, data[h], h, {"o": runs[(name, "original", h)], "r": runs[(name, "R1", h)]})
            fixed, new = changes(dict(zip(u["key"], u["o_right"])), dict(zip(u["key"], u["r_right"])))
            src = dict(zip(u["key"], u["sources"].str.split(";")))
            unit_changes[f"{name}|{h}"] = {"fixed": fixed, "new_errors": new, "concentration": concentration([src[k] for k in fixed], [src[k] for k in new])}
    mt = pd.DataFrame(metrics); goal = {}
    for name in settings:
        r1 = mt[(mt["decoding"] == name) & (mt["variant"] == "R1")].set_index("half")
        seg_ok = {h: float(r1.loc[h, "segment_f1_50_multi"]) >= float(ref[h]["value"]) + 0.01 for h in HALVES}
        floors = {h: float(r1.loc[h, "boundary_recall_multi"] or 0.0) >= float(cfg["fusion"]["boundary_recall_floor"])
                  and float(r1.loc[h, "native_macro_f1"]) >= float(cfg["fusion"]["native_floor"][h]) for h in HALVES}
        goal[name] = {"segment_part": seg_ok, "floors_part": floors, "met": all(seg_ok.values()) and all(floors.values())}
    rep = {"inputs": meta, "affected": affected, "metrics": metrics, "unit_changes": unit_changes, "goal": goal,
           "gate_met": any(g["met"] for g in goal.values())}
    run_dir = new_run_dir(cfg, "b7_window_target_room")
    (run_dir / "room.json").write_text(json.dumps(rep, indent=1, default=str))
    (run_dir / "room.md").write_text("\n".join(["# Step 4: room and gate\n", md_table(metrics),
                                                md_table([{"half": h, **{k: v for k, v in a.items() if k != "by_pair"}} for h, a in affected.items()]),
                                                *[md_table(a["by_pair"]) for a in affected.values()],
                                                "```\n" + json.dumps({"goal": goal, "gate_met": rep["gate_met"]}, indent=1) + "\n```\n"]))
    print(md_table(metrics)); print(json.dumps({"goal": goal, "gate_met": rep["gate_met"]}, indent=1)); print(f"run_dir={run_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("census", "probe-plan", "probe", "room"), required=True)
    ap.add_argument("-c", "--config", default="configs/experiments/b7_window_target_v1.yaml")
    args = ap.parse_args(); args.set = []; cfg = load_cfg(args); paths = load_paths()
    {"census": census, "probe-plan": probe_plan, "probe": probe, "room": room}[args.phase](cfg, paths, args)


if __name__ == "__main__":
    main()
