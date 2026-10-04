#!/usr/bin/env python
"""Tables and the shared probe-frame convention for the STACOM 2011 verification (scripts/stacom_verify.py output).

The convention is decided once for all volunteers: the frame (azimuth axis direction and handedness) chosen most often
as best_frame (highest mean view probability) is used for every volunteer. Per-case agreement is reported. Writes
<out>/probe_frames.json (what the slice pipeline loads) and prints Markdown tables. Numbers only.
python scripts/summarize_stacom_verify.py data/stacom2011/verify [--check data/stacom2011/pipeline_check/pipeline_check.json] --out DIR
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

CASES = ["v1", "v2", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"]
VIEWS = ("A4C", "A5C", "A2C", "A3C")


AXES = {"+x": np.array([1.0, 0, 0]), "-x": np.array([-1.0, 0, 0]), "+y": np.array([0, 1.0, 0]), "-y": np.array([0, -1.0, 0])}


def load(root: Path, extra: list[Path]) -> dict:
    """Each volunteer's summary, with the frames of extra runs (the other azimuth axis family, forced) merged in, so
    best_frame is chosen over every frame that was run. Each run remembers the folder holding its views CSV."""
    out = {}
    for c in CASES:
        f = root / c / "summary.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text()); d["families_run"] = [d["azimuth_axis_family"]]
        for r in d["echoslicer"].values():
            r["dir"] = str(root / c)
        for e in extra:
            g = e / c / "summary.json"
            if g.exists():
                x = json.loads(g.read_text()); d["families_run"].append(x["azimuth_axis_family"])
                for k, r in x["echoslicer"].items():
                    if k not in d["echoslicer"]:
                        d["echoslicer"][k] = {**r, "dir": str(e / c)}
        scored = {k: r for k, r in d["echoslicer"].items() if "mean_prob" in r}
        d["best_frame"] = max(scored, key=lambda k: scored[k]["mean_prob"]) if scored else None
        out[c] = d
    return out


def frame_rows(tag: str, beam: list) -> list:
    """Rows (beam, elevation, azimuth) of a frame tag such as '-y_mirrored', as scripts/stacom_verify.py builds them."""
    az = AXES[tag.split("_")[0]]; el = (1 if tag.endswith("right") else -1) * np.cross(az, np.array(beam))
    return [list(map(float, beam)), el.tolist(), az.tolist()]


def table(rows: list[list], head: list[str]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path); ap.add_argument("--extra", type=Path, nargs="*", default=[])
    ap.add_argument("--check", type=Path, default=None); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); s = load(a.root, a.extra)
    convention = Counter(d["best_frame"] for d in s.values() if d["best_frame"]).most_common(1)[0][0]
    flipped = ("+" if convention.startswith("-") else "-") + convention[1:]       # the same frame turned 180 degrees about the beam

    decode = [[c, d["decode"]["frames"], d["decode"]["readme_frames"], f'{d["decode"]["dicom_vtk_equal_fraction"]:.3f}',
               f'{d["time"]["volumes_per_s"]:.1f}', f'{d["time"]["cycle_s"]:.2f}', " x ".join(f"{v:.0f}" for v in d["decode"]["fov_mm_xyz"]),
               f'{d["sector"]["half_angle_deg_x_y"][0]:.0f} / {d["sector"]["half_angle_deg_x_y"][1]:.0f}', f'{d["sector"]["rho_max_mm"]:.0f}',
               f'{d["sector"]["centre_tilt_deg"]:.1f}', f'{d["sector"]["width_fit_rmse_mm"]:.2f}',
               f'{d["sector"]["width_fit_r2"]:.4f}'] for c, d in s.items()]
    print("### Decode, time and sector\n"); print(table(decode, ["Case", "Frames", "Readme frames", "DICOM = VTK", "Volumes/s", "Cycle (s)",
          "Field of view (mm, x y z)", "Half-angle x / y (°)", "Depth (mm)", "Centre tilt (°)", "Width-fit RMSE (mm)", "Width-fit R²"])); print()

    lm = []
    for c, d in s.items():
        o = d["landmarks"]["obs1"]; r = d["echoslicer"].get(convention, {})
        if "mean_prob" not in r:
            lm.append([c, "n/a" if o["contrast_grey"] is None else f'{o["contrast_grey"]:.0f}', f'{o["shift_rank"]}/{o["shift_copies"]}', f'{o["axis_mapping_rank"]}/{o["axis_mappings"]}',
                       " / ".join(f"{x:.0f}" for x in o["ring_depth_mm_apical_mid_basal"]), f'{o["room_beyond_basal_ring_mm"]:.0f}', "failed", "failed"])
            continue
        ring = [max(r["obs1_ring_centre_distance_mm"][v]) for v in ("A4C", "A2C")]
        lm.append([c, "n/a" if o["contrast_grey"] is None else f'{o["contrast_grey"]:.0f}', f'{o["shift_rank"]}/{o["shift_copies"]}', f'{o["axis_mapping_rank"]}/{o["axis_mappings"]}',
                   " / ".join(f"{x:.0f}" for x in o["ring_depth_mm_apical_mid_basal"]), f'{o["room_beyond_basal_ring_mm"]:.0f}',
                   f"{ring[0]:.1f}", f"{ring[1]:.1f}"])
    print("### Coordinates against the expert landmarks (observer 1, first frame)\n")
    print(table(lm, ["Case", "Wall minus cavity (grey)", "Rank among shifts", "Rank among axis mappings", "Ring depth apical / mid / basal (mm)",
                     "Room beyond base (mm)", "A4C plane to farthest ring centre (mm)", "A2C plane to farthest ring centre (mm)"])); print()

    views = []
    for c, d in s.items():
        r = d["echoslicer"].get(convention, {}); q = d["echoslicer"].get(flipped, {})
        if "mean_prob" not in r or "mean_prob" not in q:
            views.append([c, d["best_frame"], *(["failed"] * (len(VIEWS) + 3))]); continue
        cov = r["a4c_tilt_coverage"]["max_tilt_with_80pct_coverage_deg"]
        views.append([c, d["best_frame"], *(f'{r["probs"][v]:.2f}' for v in VIEWS), f'{r["a5c_tilt_from_a4c_deg"]:+.0f}',
                      f'{q["probs"]["A5C"]:.2f}', f'+{cov["positive"]:.1f} / -{cov["negative"]:.1f}'])
    print(f"### EchoSlicer's search in the shared frame ({convention}); classifier scores are checks, not labels\n")
    print(table(views, ["Case", "Best frame alone", *VIEWS, "A5C tilt (°)", f"A5C in {flipped}", "Tilt keeping 80% of A4C (+ / -, °)"])); print()

    agree = sum(d["best_frame"] == convention for d in s.values())
    print(f"- Shared frame: {convention}, best on its own in {agree} of {len(s)} volunteers.")
    both = [d for d in s.values() if "mean_prob" in d["echoslicer"].get(convention, {}) and "mean_prob" in d["echoslicer"].get(flipped, {})]
    a5c_here = [d["echoslicer"][convention]["probs"]["A5C"] for d in both]; a5c_flip = [d["echoslicer"][flipped]["probs"]["A5C"] for d in both]
    print(f"- A5C median {np.median(a5c_here):.2f} in {convention} against {np.median(a5c_flip):.2f} in {flipped}; higher in "
          f"{sum(x > y for x, y in zip(a5c_here, a5c_flip))} of {len(both)} volunteers with both frames.")

    frames = {}
    for c, d in s.items():
        run = d["echoslicer"].get(convention, {})
        frames[c] = {"frame_tag": convention, "frame_rows_beam_elevation_azimuth": frame_rows(convention, d["sector"]["axis"]),
                     "apex_mm_xyz": d["sector"]["apex_mm_xyz"], "frame_time_ms": d["time"]["frame_time_ms"],
                     "frames": d["decode"]["frames"], "readme_es_frame": d["time"]["readme_es_frame"],
                     "best_frame_alone": d["best_frame"], "families_run": d["families_run"],
                     "views_csv": str(Path(run["dir"]) / f"views_{convention}" / "views.csv") if "mean_prob" in run else None}
    if a.check and a.check.exists():
        chk = json.loads(a.check.read_text())
        rows = [[c, f'{v["a4c_prob_pipeline"]:.3f}' if "pass" in v else "n/a", f'{v["a4c_prob_echoslicer"]:.3f}' if "pass" in v else "n/a",
                 ("yes" if v["pass"] else "NO") if "pass" in v else "masks only", f'{v["masks"]["median_coverage"]:.2f}',
                 sum(m["pixels_outside_at_random_tilts"] for k, m in v["masks"].items() if k != "median_coverage")]
                for c, v in chk["echoslicer"].items()]
        rows += [[c, f'{v["a4c_prob_pipeline"]:.3f}', f'{v["a4c_prob_verification"]:.3f}', "yes" if v["pass"] else "NO",
                  f'{v["masks"]["median_coverage"]:.2f}', sum(m["pixels_outside_at_random_tilts"] for k, m in v["masks"].items() if k != "median_coverage")]
                 for c, v in chk["stacom"].items()]
        print("\n### Pipeline checks (start condition 3)\n")
        print(table(rows, ["Volume", "A4C score, pipeline", "A4C score, EchoSlicer", "Pass", "Median fixed-mask coverage", "Mask pixels outside"]))
        paths = [[c, r["plane"], f'{r["pearson_r"]:.3f}', r["best_shift_px"], f'{r["mean_abs_grey"]:.1f}', f'{r["shared_fraction"]:.2f}']
                 for c, v in chk["stacom"].items() for r in v["paths"]]
        print(); print(table(paths, ["Volume", "Plane", "Pearson r", "Best shift (px)", "Mean abs difference (grey)", "Shared fraction"]))
        for c, v in chk["stacom"].items():
            frames[c]["median_fixed_mask_coverage"] = v["masks"]["median_coverage"]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "probe_frames.json").write_text(json.dumps({"convention": convention, "cases": frames}, indent=1))


if __name__ == "__main__":
    main()
