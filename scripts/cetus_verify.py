#!/usr/bin/env python
"""Intake check of the CETUS ED/ES release (echo_routing/geometry/cetus.py), one patient at a time, then a summary.
Runs in envs/cetus (nibabel). Writes data/cetus/verify/<patient>.json and data/cetus/summary.json (numbers only).

A patient counts as usable when all four criteria hold. They were fixed after the first case and before the others
were read:
  U1 decode     the four files read, share one grid, and hold integer grey values 0-255
  U2 annotation the masks hold exactly the values 0 and 255 at both phases; the largest connected component holds at
                least 99% of the cavity; the ES cavity is smaller than the ED cavity
  U3 alignment  at both phases the cavity is darker inside than in a 3-voxel shell just outside it
  U4 sector     at both phases one array axis gives a straight sector taper (R^2 >= 0.98, apex within 15 mm of the
                visible region), the same axis at ED and ES, with apices within 2 mm and half-angles within 1 degree
Everything else (LV axis against the beam, depth, share of the cavity outside the fitted sector, spacing, grey fill)
is reported, not used to exclude.

  envs/cetus/bin/python scripts/cetus_verify.py [--patients patient01,patient02] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echo_routing.geometry import cetus  # noqa: E402

RAW = Path("data/cetus/raw/dataset")
OUT = Path("data/cetus")
APEX_TOL_MM, ANGLE_TOL_DEG, COMPONENT_SHARE = 2.0, 1.0, 0.99


def check(folder: Path) -> dict:
    pid = folder.name; r = {"patient": pid, "criteria": {}}
    try:
        case = cetus.read_case(folder)
    except Exception as e:  # noqa: BLE001 - any failure to read is the U1 result
        r["criteria"]["U1_decode"] = False; r["error"] = f"{type(e).__name__}: {e}"; r["usable"] = False
        return r
    r["criteria"]["U1_decode"] = True
    r.update({"shape": list(case.volumes["ED"].shape), "spacing_mm": case.spacing_mm.round(5).tolist(),
              "header_spacing": case.header_spacing.tolist(), "mask_values": case.mask_values})
    sec, phases = {}, {}
    for ph in cetus.PHASES:
        v, m = case.volumes[ph], case.masks[ph]; bg, share = cetus.background(v)
        p = {"background": bg, "background_share": share, "mean_grey_in_sector": float(v[v > bg].mean())}
        try:
            axis, s, outcome = cetus.fit_sector(v, case.spacing_mm, bg); sec[ph] = (axis, s)
            p["sector"] = {"beam_axis": axis, "r2": s.width_fit_r2, "apex_mm": s.apex_mm.round(3).tolist(),
                           "half_angles_deg": [s.half_angle_x_deg, s.half_angle_y_deg], "rho_max_mm": s.rho_max_mm,
                           "centre_tilt_deg": s.centre_tilt_deg, "per_axis": outcome}
            p["lv"] = cetus.lv_geometry(m, case.spacing_mm, axis, s)
        except ValueError as e:
            p["sector_error"] = str(e)
            p["lv"] = {"volume_ml": float(m.sum() * np.prod(case.spacing_mm) / 1000.0)}
        p["contrast"] = cetus.annotation_contrast(v, m, bg)
        phases[ph] = p
    r["phases"] = phases
    ed, es = phases["ED"], phases["ES"]
    r["criteria"]["U2_annotation"] = bool(
        all(case.mask_values[ph] == [0.0, 255.0] for ph in cetus.PHASES)
        and all(phases[ph]["lv"].get("largest_component_share", 0.0) >= COMPONENT_SHARE for ph in cetus.PHASES)
        and es["lv"]["volume_ml"] < ed["lv"]["volume_ml"])
    r["criteria"]["U3_alignment"] = bool(all(phases[ph]["contrast"]["mean_inside"] < phases[ph]["contrast"]["mean_shell_outside"]
                                             for ph in cetus.PHASES))
    same = False
    if set(sec) == set(cetus.PHASES):
        (a1, s1), (a2, s2) = sec["ED"], sec["ES"]
        gap = float(np.linalg.norm(s1.apex_mm - s2.apex_mm))
        dang = max(abs(s1.half_angle_x_deg - s2.half_angle_x_deg), abs(s1.half_angle_y_deg - s2.half_angle_y_deg))
        r["ed_es_sector"] = {"same_axis": a1 == a2, "apex_gap_mm": gap, "half_angle_gap_deg": float(dang)}
        same = a1 == a2 and gap <= APEX_TOL_MM and dang <= ANGLE_TOL_DEG
    r["criteria"]["U4_sector"] = bool(same)
    r["ejection_fraction"] = float(1 - es["lv"]["volume_ml"] / ed["lv"]["volume_ml"]) if ed["lv"]["volume_ml"] > 0 else None
    r["usable"] = all(r["criteria"].values())
    return r


def summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["usable"]]
    def rng(vals):
        vals = [v for v in vals if v is not None]
        return [float(np.min(vals)), float(np.median(vals)), float(np.max(vals))] if vals else None
    lv = lambda ph, k: [r["phases"][ph]["lv"].get(k) for r in ok]  # noqa: E731
    return {"patients": len(rows), "usable": len(ok),
            "failing": {r["patient"]: [k for k, v in r["criteria"].items() if not v] for r in rows if not r["usable"]},
            "criteria_counts": {k: sum(r["criteria"].get(k, False) for r in rows) for k in ("U1_decode", "U2_annotation", "U3_alignment", "U4_sector")},
            "shapes": sorted({tuple(r["shape"]) for r in rows if "shape" in r}),
            "spacing_mm": sorted({tuple(r["spacing_mm"]) for r in rows if "spacing_mm" in r}),
            "beam_axis": {str(a): sum(r["phases"]["ED"]["sector"]["beam_axis"] == a for r in ok) for a in range(3)},
            "min_median_max": {
                "ed_volume_ml": rng(lv("ED", "volume_ml")), "es_volume_ml": rng(lv("ES", "volume_ml")),
                "ejection_fraction": rng([r["ejection_fraction"] for r in ok]),
                "lv_long_axis_vs_beam_deg_ed": rng(lv("ED", "long_axis_vs_beam_deg")),
                "lv_length_mm_ed": rng(lv("ED", "length_along_long_axis_mm")),
                "lv_shallowest_depth_mm_ed": rng([r["phases"]["ED"]["lv"]["depth_mm"][0] for r in ok]),
                "cavity_share_outside_sector_ed": rng(lv("ED", "share_outside_sector")),
                "cavity_share_outside_sector_es": rng(lv("ES", "share_outside_sector")),
                "half_angle_x_deg": rng([r["phases"]["ED"]["sector"]["half_angles_deg"][0] for r in ok]),
                "half_angle_y_deg": rng([r["phases"]["ED"]["sector"]["half_angles_deg"][1] for r in ok]),
                "rho_max_mm": rng([r["phases"]["ED"]["sector"]["rho_max_mm"] for r in ok]),
                "apex_gap_ed_es_mm": rng([r["ed_es_sector"]["apex_gap_mm"] for r in ok]),
                "background_ed": rng([r["phases"]["ED"]["background"] for r in ok]),
                "background_es": rng([r["phases"]["ES"]["background"] for r in ok]),
                "mean_grey_in_sector_ed": rng([r["phases"]["ED"]["mean_grey_in_sector"] for r in ok]),
                "mean_grey_in_sector_es": rng([r["phases"]["ES"]["mean_grey_in_sector"] for r in ok]),
                "contrast_inside_ed": rng([r["phases"]["ED"]["contrast"]["mean_inside"] for r in ok]),
                "contrast_shell_ed": rng([r["phases"]["ED"]["contrast"]["mean_shell_outside"] for r in ok])},
            "background_differs_ed_es": sum(r["phases"]["ED"]["background"] != r["phases"]["ES"]["background"] for r in ok)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", default=None); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    folders = sorted(p for p in RAW.glob("patient*") if p.is_dir())
    if a.patients:
        folders = [RAW / x for x in a.patients.split(",")]
    (OUT / "verify").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(check, folders))
    for r in rows:
        (OUT / "verify" / f"{r['patient']}.json").write_text(json.dumps(r, indent=1))
        print(r["patient"], "usable" if r["usable"] else f"NOT usable {r['criteria']}", r.get("error", ""), flush=True)
    if not a.patients:
        s = summary(rows); (OUT / "summary.json").write_text(json.dumps(s, indent=1)); print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
