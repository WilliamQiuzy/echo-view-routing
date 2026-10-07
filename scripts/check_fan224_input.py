#!/usr/bin/env python
"""Training-image checks and export of the fan224_v1 input (plan Task 1;
docs/superpowers/plans/2026-10-06-fan224-visual-adaptation.md). Training recordings only.

--phase qc      1. The QC list: the pilot's 30 training recordings (seed 20261006, 6 per class) and 14 more per class
                   from the remaining training recordings (seed 20261007), 100 in total.
                2. The size scan: every frame file of every training recording is opened (header) and must be
                   320 x 240 and non-empty; the first, middle and last frames are decoded and must not be blank.
                   Anomalies are listed, nothing is removed.
                3. Per QC recording, over all its frames: constant bright pixels (temporal minimum >= 40 and range
                   <= 25) inside the template beyond the near field (more than 25 px from the apex), i.e. interface
                   marks left in the input; and the share of the recording's changing pixels (temporal std > 4) inside
                   the pilot sector that the template drops, i.e. a truncation indicator. Both also for the pilot.
                4. Review sheets: first, middle and last frame of every QC recording with the template outline, next to
                   the fan224 outputs; 5 recordings per sheet, labelled Q001-Q100 only.
--phase export  PNG of every 10 Hz sample of the training split in both variants (orig, gamma090) under
                data/derived/<version>/<variant>/<video_id>/<frame_idx>.png, plus a manifest with the source frame,
                time stamp, frame-order version, source file sha256, output sha256 and transform digest.

Identifiers and images stay on the server (runs/fan224_visual_adaptation_v1/qc, data/derived); --summary writes an
id-free summary for the repository.

  python scripts/check_fan224_input.py --phase qc --summary docs/results/2026-10-06_fan224_visual_adaptation/input_qc.json
  python scripts/check_fan224_input.py --phase export --workers 8
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from echo_routing.ingest import fan224  # noqa: E402
from echo_routing.ingest.frames import list_frame_paths  # noqa: E402

CLASSES = ("PLAX", "PSAX", "A4C", "A5C", "SC4C")
PROTOCOL = "runs/20261002-051140-model_development_protocol-fa45e7f4-s0/protocol.json"
PROTOCOL_SHA = "a897483b776e8cff1ba8b92774c6a82537758bcba6abd362f2d8603f37997593"
MANIFEST = "data/manifests/ev9v_native.csv"
CACHE_KEY = "16efc85be645"           # cache_v2 (numeric frame order): the 10 Hz sample indices and time stamps
FRAME_ORDER = "numeric (cache_v2)"
PILOT_SEED, PILOT_PER_CLASS = 20261006, 6
QC_SEED, QC_PER_CLASS = 20261007, 14
QC_DIR = "runs/fan224_visual_adaptation_v1/qc"
CONST_MIN, CONST_RANGE, NEAR_FIELD_PX, ACTIVE_STD = 40, 25, 25.0, 4.0
PILOT = {"apex_x": 157, "apex_y": 26, "radius": 207, "half_angle_degrees": 46.5}   # scripts/pilot_fan224.py


# ----------------------------------------------------------------------------------------------- lists
def train_rows(root: Path) -> list[dict]:
    proto_bytes = (root / PROTOCOL).read_bytes()
    if hashlib.sha256(proto_bytes).hexdigest() != PROTOCOL_SHA:
        raise SystemExit("the protocol differs from its recorded sha256")
    allowed = set(json.loads(proto_bytes)["train"]["clips"])
    with (root / MANIFEST).open() as f:
        rows = [r for r in csv.DictReader(f) if r["video_id"] in allowed]
    if len(rows) != len(allowed) or any(r["split"] != "train" for r in rows):
        raise SystemExit("training recordings and manifest disagree")
    return rows


def pick(rows: list[dict], seed: int, per_class: int, exclude: set) -> list[dict]:
    """Per class: the training recordings sorted by id, shuffled with random.Random(seed), the first `per_class`
    (the pilot's rule; it reads ids and classes only, no image, score or validation data)."""
    rng = random.Random(seed); out = []
    for label in CLASSES:
        pool = sorted((r for r in rows if r["family5_label"] == label and r["video_id"] not in exclude), key=lambda r: r["video_id"])
        rng.shuffle(pool)
        if len(pool) < per_class:
            raise SystemExit(f"{label}: only {len(pool)} recordings")
        out.extend(pool[:per_class])
    return out


def qc_list(rows: list[dict]) -> list[dict]:
    pilot = pick(rows, PILOT_SEED, PILOT_PER_CLASS, set())
    extra = pick(rows, QC_SEED, QC_PER_CLASS, {r["video_id"] for r in pilot})
    out = [{**r, "qc": f"Q{k + 1:03d}", "source": "pilot" if k < len(pilot) else "new"} for k, r in enumerate(pilot + extra)]
    if len({r["video_id"] for r in out}) != 100:
        raise SystemExit("the QC list does not hold 100 distinct recordings")
    return out


def frames_dir(root: Path, row: dict) -> Path:
    d = Path(row["frames_dir"])
    return d if d.is_absolute() else root / d


# ----------------------------------------------------------------------------------------------- size scan
def scan_one(args) -> dict:
    vid, d = args
    try:
        paths = list_frame_paths(d)
    except Exception as e:   # noqa: BLE001 - every failure is reported as an anomaly, nothing is skipped silently
        return {"video_id": vid, "frames": 0, "problems": [f"no frames: {e}"]}
    problems = []
    for p in paths:
        if p.stat().st_size == 0:
            problems.append(f"empty file {p.name}")
            continue
        with Image.open(p) as im:
            if im.size != fan224.SOURCE_SIZE:
                problems.append(f"{p.name}: {im.size[0]}x{im.size[1]}")
    for p in (paths[0], paths[len(paths) // 2], paths[-1]):
        if p.stat().st_size:
            with Image.open(p) as im:
                if int(np.asarray(im.convert("L")).max()) < 10:
                    problems.append(f"{p.name}: blank")
    return {"video_id": vid, "frames": len(paths), "problems": problems}


# ----------------------------------------------------------------------------------------------- QC metrics and sheets
def pilot_mask() -> np.ndarray:
    yy, xx = np.mgrid[:240, :320]; dx, dy = xx - PILOT["apex_x"], yy - PILOT["apex_y"]
    return (dy >= 0) & (dx * dx + dy * dy <= PILOT["radius"] ** 2) & (np.abs(np.degrees(np.arctan2(dx, dy))) <= PILOT["half_angle_degrees"])


def far_field(mask: np.ndarray, apex: tuple[float, float]) -> np.ndarray:
    yy, xx = np.mgrid[:240, :320]
    return mask & (np.hypot(xx - apex[0], yy - apex[1]) > NEAR_FIELD_PX)


def recording_metrics(gray: np.ndarray, new: np.ndarray, old: np.ndarray) -> dict:
    """gray: (T, 240, 320) uint8 frames of one recording."""
    lo, hi = gray.min(0).astype(int), gray.max(0).astype(int); std = gray.astype(np.float32).std(0)
    const = (lo >= CONST_MIN) & ((hi - lo) <= CONST_RANGE); active = std > ACTIVE_STD
    new_far = far_field(new, (fan224.TEMPLATE["apex_x"], fan224.TEMPLATE["apex_y"])); old_far = far_field(old, (PILOT["apex_x"], PILOT["apex_y"]))
    act_old = int((active & old).sum())
    return {"constant_bright_inside": int((const & new_far).sum()), "constant_bright_inside_pilot": int((const & old_far).sum()),
            "active_dropped_share": float((active & old & ~new).sum() / max(act_old, 1)),
            "active_outside_both": int((active & ~old & ~new).sum())}


def outline(mask: np.ndarray) -> np.ndarray:
    m = mask
    return m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))


def qc_one(args) -> tuple[dict, list]:
    row, root = args; paths = list_frame_paths(frames_dir(root, row))
    gray = np.stack([np.asarray(Image.open(p).convert("L"), dtype=np.uint8) for p in paths])
    met = recording_metrics(gray, fan224.MASK, pilot_mask())
    picks = [0, len(paths) // 2, len(paths) - 1]; edge = outline(fan224.MASK); tiles = []
    for i in picks:
        with Image.open(paths[i]) as im:
            raw = np.asarray(im.convert("RGB")).copy(); out = fan224.fan224(im)
        raw[edge] = [255, 0, 0]; tiles.append((raw, out))
    return {"qc": row["qc"], "label": row["family5_label"], "source": row["source"], "frames": len(paths), **met}, tiles


def sheet(rows: list[tuple[str, list]]) -> Image.Image:
    w = 3 * 320 + 3 * 224; im = Image.new("RGB", (w, 240 * len(rows))); draw = ImageDraw.Draw(im)
    for j, (name, tiles) in enumerate(rows):
        for k, (raw, out) in enumerate(tiles):
            im.paste(Image.fromarray(raw), (320 * k, 240 * j)); im.paste(Image.fromarray(out), (960 + 224 * k, 240 * j + 8))
        draw.text((4, 240 * j + 2), name, fill=(255, 255, 0))
    return im


def phase_qc(root: Path, workers: int, summary_path: Path | None) -> None:
    out = root / QC_DIR; (out / "sheets").mkdir(parents=True, exist_ok=True)
    rows = train_rows(root); qc = qc_list(rows)
    (out / "qc_list.json").write_text(json.dumps([{k: r[k] for k in ("qc", "source", "video_id", "family5_label")} for r in qc], indent=1))
    with ProcessPoolExecutor(workers) as ex:
        scan = list(ex.map(scan_one, [(r["video_id"], frames_dir(root, r)) for r in rows], chunksize=16))
        res = list(ex.map(qc_one, [(r, root) for r in qc]))
    anomalies = [s for s in scan if s["problems"] or not s["frames"]]
    (out / "size_scan.json").write_text(json.dumps({"recordings": len(scan), "frames": sum(s["frames"] for s in scan), "anomalies": anomalies}, indent=1))
    metrics = [m for m, _ in res]
    with (out / "metrics.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(metrics[0])); wr.writeheader(); wr.writerows(metrics)
    for page in range(0, len(res), 5):
        sheet([(m["qc"] + " " + m["label"], tiles) for m, tiles in res[page:page + 5]]).save(out / "sheets" / f"sheet_{page // 5 + 1:02d}.png")
    cb, cbp = np.array([m["constant_bright_inside"] for m in metrics]), np.array([m["constant_bright_inside_pilot"] for m in metrics])
    drop = np.array([m["active_dropped_share"] for m in metrics])
    summ = {"transform": fan224.spec(), "transform_digest": fan224.spec_digest(), "scale_factor": fan224.scale_factor(),
            "qc_recordings": len(metrics), "per_class": {c: sum(m["label"] == c for m in metrics) for c in CLASSES},
            "size_scan": {"recordings": len(scan), "frames": int(sum(s["frames"] for s in scan)), "anomalous_recordings": len(anomalies)},
            "constant_bright_inside": {"template": {"recordings_with_any": int((cb > 0).sum()), "total": int(cb.sum()), "max": int(cb.max())},
                                       "pilot": {"recordings_with_any": int((cbp > 0).sum()), "total": int(cbp.sum()), "max": int(cbp.max())}},
            "active_dropped_share_vs_pilot": {"median": float(np.median(drop)), "p90": float(np.percentile(drop, 90)), "max": float(drop.max())},
            "per_recording": [{k: m[k] for k in ("qc", "label", "source", "frames", "constant_bright_inside", "constant_bright_inside_pilot", "active_dropped_share")} for m in metrics]}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True); summary_path.write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("per_recording", "transform")}, indent=1))


# ----------------------------------------------------------------------------------------------- export
def export_one(args) -> list[dict]:
    vid, d, frame_idx, t_sec, root, version, digest = args; paths = list_frame_paths(d); recs = []
    for variant, gamma in (("orig", 1.0), ("gamma090", 0.9)):
        dest = root / "data" / "derived" / version / variant / vid; dest.mkdir(parents=True, exist_ok=True)
        for k, (fi, t) in enumerate(zip(frame_idx, t_sec)):
            src = paths[min(int(fi), len(paths) - 1)]; raw = src.read_bytes()
            with Image.open(src) as im:
                arr = fan224.fan224(im, gamma)
            png = dest / f"{int(fi):06d}.png"; Image.fromarray(arr).save(png, optimize=False)
            recs.append({"video_id": vid, "variant": variant, "gamma": gamma, "sample": k, "frame_idx": int(fi), "t_sec": float(t),
                         "source_file": src.name, "source_sha256": hashlib.sha256(raw).hexdigest(), "png": str(png.relative_to(root)),
                         "png_sha256": hashlib.sha256(png.read_bytes()).hexdigest(), "frame_order": FRAME_ORDER, "transform_digest": digest})
    return recs


def phase_export(root: Path, workers: int) -> None:
    from echo_routing.config import load_paths
    from echo_routing.temporal.dev_data import FeatureStore
    rows = train_rows(root); store = FeatureStore(load_paths().cache_root, CACHE_KEY); version, digest = fan224.VERSION, fan224.spec_digest()
    base = root / "data" / "derived" / version; man = base / "manifest_train.csv"
    if man.exists():
        raise SystemExit(f"{man} exists; a new transform needs a new version")
    jobs = []
    for r in rows:
        a, b = store.get(r["video_id"], "orig"), store.get(r["video_id"], "gamma090")
        if not np.array_equal(a.frame_idx, b.frame_idx):
            raise SystemExit(f"{r['video_id']}: the variants disagree on frame indices")
        jobs.append((r["video_id"], frames_dir(root, r), a.frame_idx.tolist(), a.t_sec.tolist(), root, version, digest))
    with ProcessPoolExecutor(workers) as ex:
        recs = [x for chunk in ex.map(export_one, jobs, chunksize=8) for x in chunk]
    with man.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(recs[0])); wr.writeheader(); wr.writerows(recs)
    (base / "spec.json").write_text(json.dumps({**fan224.spec(), "transform_digest": digest, "split": "train", "frames": len(recs),
                                               "manifest_sha256": hashlib.sha256(man.read_bytes()).hexdigest()}, indent=1))
    print(json.dumps({"version": version, "frames": len(recs), "recordings": len(jobs), "manifest": str(man.relative_to(root))}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["qc", "export"], required=True); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summary", type=Path, default=None)
    a = ap.parse_args(); root = Path(__file__).resolve().parents[1]
    if a.phase == "qc":
        phase_qc(root, a.workers, root / a.summary if a.summary else None)
    else:
        phase_export(root, a.workers)


if __name__ == "__main__":
    main()
