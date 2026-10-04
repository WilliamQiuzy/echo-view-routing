#!/usr/bin/env python
"""Fixed masks and pair banks for geometry pretraining (Task 3; configs/experiments/geometry_pretraining_v1.yaml).

masks: for every volume and every whole-degree psi in bank.PSI_GRID, the fixed mask over alpha in [-15, 15] (1-degree
  steps, 2-pixel margin) and its coverage, saved packed per volume. Volumes whose median coverage over psi is below 20%
  are excluded and listed (inclusion rule).
render: one bank. Pairs are sampled by person (bank.sample_specs) from the pretraining people or from the held-out
  people, rendered with the cached masks, standardised to the EV9V sector statistics and put through E1's geometry
  (bank.final_input). Workers load one volume at a time and write their pairs at fixed positions, so the bank's order
  is the sampling order. Writes <out>/{u8.npy (N, 2, 224, 224) uint8, features.npz, specs.csv, meta.json}.
Runs in EchoSlicer's environment (its DICOM reader): third_party/3d-echo_run/.venv/bin/python scripts/geometry_bank.py ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echo_routing.geometry import bank, pairs, planes, stacom, volumes  # noqa: E402

GRID = pairs.SliceGrid(pixel_mm=0.5, width_mm=160.0, depth_mm=150.0)
ALPHA = (-15.0, 15.0)
ECHOSLICER = [f"{p}_{i}" for p, n in (("A", 15), ("B", 5), ("C", 5), ("D", 4)) for i in range(n)]
STACOM = ["v1", "v2", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"]
HELD_OUT = {"echoslicer:C", "stacom:v14", "stacom:v15", "stacom:v16"}
PATHS = {}


def load_volume(volume_id: str):
    source, name = volume_id.split(":")
    if source == "echoslicer":
        import pydicom
        sys.path.insert(0, PATHS["echoslicer"])
        from extractor.utils_3d import read_3d
        dcm = pydicom.dcmread(Path(PATHS["echoslicer_data"]) / "unpacked" / "dataset" / f"{name}.dcm"); data, bounds = read_3d(dcm)
        grid = tuple(np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, data.shape[:3]))
        return volumes.SphericalVolume(data, grid, np.arange(data.shape[3]) * float(dcm[(0x0018, 0x1063)].value) / 1000.0, f"echoslicer:{name[0]}")
    case = stacom.read_case(Path(PATHS["stacom_root"]) / name / "3DUS"); sector = stacom.sector_geometry(case.volume, case.spacing)
    pf = json.loads(Path(PATHS["probe_frames"]).read_text())["cases"][name]
    if not (np.allclose(pf["apex_mm_xyz"], sector.apex_mm) and np.allclose(pf["frame_rows_beam_elevation_azimuth"][0], sector.axis)):
        raise SystemExit(f"{name}: probe frame file does not match the sector fitted now")
    return volumes.CartesianVolume(case.volume, case.spacing, sector.apex_mm, np.array(pf["frame_rows_beam_elevation_azimuth"]),
                                   sector.mask, np.arange(case.volume.shape[0]) * case.frame_time_ms / 1000.0, f"stacom:{name}")


def person_of(volume_id: str) -> str:
    source, name = volume_id.split(":")
    return f"echoslicer:{name[0]}" if source == "echoslicer" else volume_id


def n_times(vol) -> int:
    return int(vol.data.shape[3]) if isinstance(vol, volumes.SphericalVolume) else int(vol.volume.shape[0])


def masks_for(volume_id: str) -> dict:
    vol = load_volume(volume_id); out = Path(PATHS["masks"]) / f"{volume_id.replace(':', '_')}.npz"
    ms = np.stack([pairs.fixed_mask(vol, float(p), GRID, ALPHA) for p in bank.PSI_GRID])
    np.savez_compressed(out, psi=bank.PSI_GRID, coverage=ms.mean((1, 2)), packed=np.packbits(ms, axis=-1), shape=np.array(ms.shape[1:]),
                        n_times=n_times(vol))
    return {"volume_id": volume_id, "median_coverage": float(np.median(ms.mean((1, 2)))), "n_times": n_times(vol)}


def load_masks(volume_id: str) -> tuple[np.ndarray, np.ndarray, int]:
    z = np.load(Path(PATHS["masks"]) / f"{volume_id.replace(':', '_')}.npz")
    return np.unpackbits(z["packed"], axis=-1)[..., : int(z["shape"][1])].astype(bool), z["coverage"], int(z["n_times"])


def render_volume(job: tuple) -> int:
    """Render every pair of one volume into the bank's arrays (opened as memmaps by index)."""
    volume_id, rows, out, mean, sd = job
    vol = load_volume(volume_id); masks, _, _ = load_masks(volume_id)
    u8 = np.load(Path(out) / "u8.npy", mmap_mode="r+"); feats = np.load(Path(out) / "features_raw.npy", mmap_mode="r+")
    fin = np.load(Path(out) / "features_final.npy", mmap_mode="r+")
    for r in rows:
        spec = pairs.PairSpec(r["psi_deg"], r["alpha1_deg"], r["alpha2_deg"], int(r["t1"]), int(r["t2"]), r["kind"])
        mask = masks[int(spec.psi_deg) - int(bank.PSI_GRID[0])]
        for k, (a, t) in enumerate(((spec.alpha1_deg, spec.t1), (spec.alpha2_deg, spec.t2))):
            img, inside = pairs.render(vol, planes.ProbePlane(spec.psi_deg, a), t, GRID)
            if not inside[mask].all():
                raise SystemExit(f"{volume_id} pair {r['index']}: mask reaches outside the volume at alpha {a}")
            feats[r["index"], k] = pairs.contrast_features(img, mask)
            x, m = bank.final_input(img, mask, mean, sd); u8[r["index"], k] = x; fin[r["index"], k] = bank.final_features(x, m)
    u8.flush(); feats.flush(); fin.flush()
    return len(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["masks", "render"]); ap.add_argument("--echoslicer", required=True)
    ap.add_argument("--echoslicer-data", required=True); ap.add_argument("--stacom-root", required=True)
    ap.add_argument("--probe-frames", required=True); ap.add_argument("--masks", required=True)
    ap.add_argument("--out", default=None); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--pairs", type=int, default=96000)
    ap.add_argument("--held-out", action="store_true", help="sample from the held-out people instead of the pretraining people")
    ap.add_argument("--stats", default="data/geometry/ev9v_sector_stats.json"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    PATHS.update(echoslicer=a.echoslicer, echoslicer_data=a.echoslicer_data, stacom_root=a.stacom_root, probe_frames=a.probe_frames, masks=a.masks)
    Path(a.masks).mkdir(parents=True, exist_ok=True)
    every = [f"echoslicer:{v}" for v in ECHOSLICER] + [f"stacom:{v}" for v in STACOM]
    if a.mode == "masks":
        with ProcessPoolExecutor(a.workers, initializer=PATHS.update, initargs=(dict(PATHS),)) as ex:
            res = list(ex.map(masks_for, every))
        incl = {"included": [r["volume_id"] for r in res if r["median_coverage"] >= bank.MIN_COVERAGE],
                "excluded": {r["volume_id"]: r["median_coverage"] for r in res if r["median_coverage"] < bank.MIN_COVERAGE},
                "volumes": res, "rule": "median fixed-mask coverage over psi at least 20%"}
        (Path(a.masks) / "inclusion.json").write_text(json.dumps(incl, indent=1)); print(json.dumps({k: incl[k] for k in ("excluded",)}, indent=1))
        return
    incl = json.loads((Path(a.masks) / "inclusion.json").read_text()); stats = json.loads(Path(a.stats).read_text())
    info, cov = [], {}
    for v in incl["included"]:
        _, c, nt = load_masks(v); cov[v] = c; info.append(bank.VolumeInfo(v, person_of(v), nt))
    people = bank.people_of(info, HELD_OUT, pretraining=not a.held_out)
    specs = bank.sample_specs(np.random.default_rng(a.seed), people, cov, a.pairs, ALPHA)
    rows = [{"index": i, "volume_id": v.volume_id, "person": v.person, "psi_deg": s.psi_deg, "alpha1_deg": s.alpha1_deg, "alpha2_deg": s.alpha2_deg,
             "t1": s.t1, "t2": s.t2, "kind": s.kind, "d_alpha": s.d_alpha} for i, (v, s) in enumerate(specs)]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); n = len(rows)
    np.lib.format.open_memmap(out / "u8.npy", mode="w+", dtype=np.uint8, shape=(n, 2, 224, 224))
    for name in ("features_raw", "features_final"):
        np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=np.float32, shape=(n, 2, 4))
    df = pd.DataFrame(rows); df.to_csv(out / "specs.csv", index=False)
    jobs = [(v, g.to_dict("records"), str(out), float(stats["mean"]), float(stats["sd"])) for v, g in df.groupby("volume_id")]
    jobs.sort(key=lambda j: -len(j[1]))
    with ProcessPoolExecutor(a.workers, initializer=PATHS.update, initargs=(dict(PATHS),)) as ex:
        done = sum(ex.map(render_volume, jobs))
    meta = {"seed": a.seed, "pairs": n, "rendered": done, "held_out_people": a.held_out, "people": sorted(people),
            "pairs_per_person": df.groupby("person").size().to_dict(), "pairs_per_kind": df.groupby("kind").size().to_dict(),
            "grid": vars(GRID), "alpha_range": ALPHA, "psi_grid": [int(bank.PSI_GRID[0]), int(bank.PSI_GRID[-1])],
            "standardise_to": {"mean": stats["mean"], "sd": stats["sd"], "source": a.stats}, "augmentation": "none",
            "sha256": {f: sha256_file(out / f) for f in ("u8.npy", "features_raw.npy", "features_final.npy", "specs.csv")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1)); print(json.dumps({k: meta[k] for k in ("pairs", "rendered", "pairs_per_kind")}, indent=1))


if __name__ == "__main__":
    main()
