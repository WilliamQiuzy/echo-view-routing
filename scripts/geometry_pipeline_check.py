#!/usr/bin/env python
"""Real-data checks of the slice pipeline: start condition 3 of configs/experiments/geometry_pretraining_v1.yaml.
Diagnostic only; classifier scores are checks, not labels.

(a) EchoSlicer volumes. EchoSlicer's own A4C plane (its CSV, for the volumes its search was run on), converted with
    planes.from_echoslicer and rendered by the pipeline, must score within 0.1 of EchoSlicer's recorded A4C probability
    on the same classifier. Every volume gets the mask check (c).
(b) STACOM volumes. In the shared probe frame (probe_frames.json from scripts/summarize_stacom_verify.py), the Cartesian path and a fine
    spherical resampling must render the same planes with Pearson r >= 0.95 on shared pixels and a best integer shift
    of zero; the pipeline's A4C rendering must also score within 0.1 of the verification run.
(c) Fixed masks. For psi in -135, -90, -45, 0, 30 and 40 random tilts in the range, no mask pixel may fall outside
    the volume.
    Mask coverage per volume feeds the inclusion rule (median over psi at least 20% of the slice).
Writes <out>/pipeline_check.json. Runs in EchoSlicer's environment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from echo_routing.geometry import pairs, planes, stacom, volumes  # noqa: E402

GRID = pairs.SliceGrid(pixel_mm=0.5, width_mm=160.0, depth_mm=150.0)
ALPHA = (-15.0, 15.0)
PSIS = (-135.0, -90.0, -45.0, 0.0, 30.0)                  # inside pairs.PSI_RANGE


def best_shift(a: np.ndarray, b: np.ndarray, m: np.ndarray, reach: int = 3) -> tuple[int, int]:
    """Integer (row, column) shift of b that best matches a on the shared mask (highest Pearson r)."""
    best = None
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            bs = np.roll(np.roll(b, dy, 0), dx, 1); ms = m & np.roll(np.roll(m, dy, 0), dx, 1)
            r = np.corrcoef(a[ms], bs[ms])[0, 1]
            if best is None or r > best[0]:
                best = (r, (dy, dx))
    return best[1]


def mask_checks(vol, rng: np.random.Generator) -> dict:
    out = {}
    for psi in PSIS:
        m = pairs.fixed_mask(vol, psi, GRID, ALPHA); bad = 0; s, t = GRID.axes()
        for a in rng.uniform(*ALPHA, size=40):
            inside = vol.inside(planes.ProbePlane(psi, float(a)).points(s, t).reshape(-1, 3)).reshape(len(t), len(s))
            bad += int((m & ~inside).sum())
        out[str(int(psi))] = {"coverage": float(m.mean()), "pixels_outside_at_random_tilts": bad}
    out["median_coverage"] = float(np.median([v["coverage"] for v in out.values()]))
    return out


def echoslicer_volume(dicom: Path, subject: str):
    import pydicom
    from extractor.utils_3d import read_3d
    dcm = pydicom.dcmread(dicom); data, bounds = read_3d(dcm)
    grid = tuple(np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, data.shape[:3]))
    ft = float(dcm[(0x0018, 0x1063)].value) / 1000.0
    return volumes.SphericalVolume(data, grid, np.arange(data.shape[3]) * ft, subject)


def stacom_volumes(root: Path, probe_frames: dict, case: str):
    """The volunteer's CartesianVolume in the shared probe frame (scripts/summarize_stacom_verify.py), a fine spherical
    resampling of its first frame in the same frame, and EchoSlicer's views found in that frame."""
    import stacom_verify as sv
    c = stacom.read_case(root / case / "3DUS"); sector = stacom.sector_geometry(c.volume, c.spacing)
    pf = probe_frames["cases"][case]; tag = pf["frame_tag"]
    beam, el, az = (np.array(v) for v in pf["frame_rows_beam_elevation_azimuth"])
    if not (np.allclose(beam, sector.axis) and np.allclose(pf["apex_mm_xyz"], sector.apex_mm)):
        raise ValueError(f"{case}: probe frame file does not match the sector fitted now")
    times = np.arange(c.volume.shape[0]) * c.frame_time_ms / 1000.0
    cart = volumes.CartesianVolume(c.volume, c.spacing, sector.apex_mm, np.stack([sector.axis, el, az]), sector.mask, times, case)
    first = stacom.Case(c.volume[:1], c.spacing, c.frame_time_ms, c.dicom_equal_fraction)   # the check renders frame 0 only
    data, bounds = sv.to_spherical(first, sector, el, az, step_mm=0.25, step_deg=0.5)
    sph = volumes.SphericalVolume(data, tuple(np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, data.shape[:3])), times[:1], case)
    views = pd.read_csv(pf["views_csv"]).set_index("name")
    return cart, sph, views, tag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--echoslicer", required=True, help="third_party/3d-echo_run")
    ap.add_argument("--echoslicer-data", required=True, help="data/echoslicer (unpacked/dataset, extracted)")
    ap.add_argument("--echoslicer-volumes", nargs="*", default=[f"{p}_{i}" for p, n in (("A", 15), ("B", 5), ("C", 5), ("D", 4)) for i in range(n)])
    ap.add_argument("--stacom-root", required=True)
    ap.add_argument("--probe-frames", required=True, help="probe_frames.json from scripts/summarize_stacom_verify.py")
    ap.add_argument("--stacom-cases", nargs="*", default=[]); ap.add_argument("--out", required=True)
    a = ap.parse_args(); sys.path.insert(0, a.echoslicer)
    from extractor import extractor as ex
    ve = ex.ViewExtractor(); rng = np.random.default_rng(0); report = {"echoslicer": {}, "stacom": {}}
    probe_frames = json.loads(Path(a.probe_frames).read_text())
    es = Path(a.echoslicer_data)
    for v in a.echoslicer_volumes:
        vol = echoslicer_volume(es / "unpacked" / "dataset" / f"{v}.dcm", v)
        if not (es / "extracted" / v / "A4C.csv").exists():          # EchoSlicer's search was run on A_0, B_0, C_0 only
            report["echoslicer"][v] = {"masks": mask_checks(vol, rng)}; print(v, "masks only", flush=True); continue
        row = pd.read_csv(es / "extracted" / v / "A4C.csv").iloc[0]
        img, _ = pairs.render(vol, planes.from_echoslicer(row.d, row.phi, row.theta), 0, GRID)
        p = float(ve._view_probs(img, "A4C"))
        base = planes.from_echoslicer(row.d, row.phi, row.theta); sweep = {}
        for dpsi in range(-150, 151, 30):                     # where standard views appear (display orientation)
            x, _ = pairs.render(vol, planes.ProbePlane(base.psi_deg + dpsi, base.alpha_deg, base.offset_mm), 0, GRID)
            sweep[str(dpsi)] = {k: float(ve._view_probs(x, k)) for k in ("A4C", "A2C", "A3C")}
        report["echoslicer"][v] = {"a4c_prob_pipeline": p, "a4c_prob_echoslicer": float(row.p),
                                   "pass": abs(p - float(row.p)) <= 0.1, "masks": mask_checks(vol, rng),
                                   "psi_sweep_from_a4c": sweep}
        print(v, report["echoslicer"][v]["a4c_prob_pipeline"], row.p, flush=True)
    for c in a.stacom_cases:
        cart, sph, views, tag = stacom_volumes(Path(a.stacom_root), probe_frames, c)
        a4c = views.loc["A4C"]; base = planes.from_echoslicer(a4c.d, a4c.phi, a4c.theta); rows = []
        for name, pl in (("A4C", base), ("A4C+10", planes.ProbePlane(base.psi_deg, base.alpha_deg + 10, base.offset_mm)),
                         ("A4C-10", planes.ProbePlane(base.psi_deg, base.alpha_deg - 10, base.offset_mm)),
                         ("A4C psi+60", planes.ProbePlane(base.psi_deg + 60, base.alpha_deg, base.offset_mm))):
            x, mx = pairs.render(cart, pl, 0, GRID); y, my = pairs.render(sph, pl, 0, GRID); m = mx & my
            rows.append({"plane": name, "shared_fraction": float(m.mean()), "pearson_r": float(np.corrcoef(x[m], y[m])[0, 1]),
                         "mean_abs_grey": float(np.abs(x[m] - y[m]).mean()), "best_shift_px": list(best_shift(x, y, m))})
        img, _ = pairs.render(cart, base, 0, GRID); p = float(ve._view_probs(img, "A4C")); sweep = {}
        for dpsi in range(-150, 151, 30):                     # display orientation in the shared STACOM frame
            x, _ = pairs.render(cart, planes.ProbePlane(base.psi_deg + dpsi, base.alpha_deg, base.offset_mm), 0, GRID)
            sweep[str(dpsi)] = {k: float(ve._view_probs(x, k)) for k in ("A4C", "A2C", "A3C")}
        ok = all(r["pearson_r"] >= 0.95 and r["best_shift_px"] == [0, 0] for r in rows) and abs(p - float(a4c.p)) <= 0.1
        report["stacom"][c] = {"frame": tag, "paths": rows, "a4c_prob_pipeline": p, "a4c_prob_verification": float(a4c.p),
                               "pass": ok, "masks": mask_checks(cart, rng), "psi_sweep_from_a4c": sweep,
                               "a4c_plane_psi_deg": base.psi_deg}
        print(c, tag, [(r["plane"], round(r["pearson_r"], 3), r["best_shift_px"]) for r in rows], round(p, 3), flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "pipeline_check.json").write_text(json.dumps(report, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
