#!/usr/bin/env python
"""Round-1 pass over the downloaded MIMIC-IV-Echo B-mode cines (tasks 1-3).

Scope: every multi-frame DICOM whose ultrasound regions are all 2D tissue (B-mode), in studies whose DICOMs are all
downloaded. Per cine:
- ECG phase from the trace drawn on the frames (echo_routing/realvideo/ecg.py), checked against the HeartRate tag;
- image-motion cycle on the tissue region above the trace (echo_routing/realvideo/cycles.py), for validating the
  image method against the ECG before it is used on EV9V, which has no ECG;
- 10 Hz frame probabilities from B0 (ResNet-18 trained on EV9V) and EchoPrime's released view classifier.
Each DICOM is one stored loop; loops are never concatenated. Outputs stay on the server (MIMIC data use agreement):
<out>/mimic_cines.csv and <out>/mimic/<study>_<file>.npz.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydicom  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402
from pydicom.encaps import generate_frames  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from echo_routing.config import gpu_memory_fraction, load_paths  # noqa: E402
from echo_routing.features.encoder import load_checkpoint  # noqa: E402
from echo_routing.ingest.frames import letterbox, to_tensor_array  # noqa: E402
from echo_routing.realvideo.cycles import estimate_cycle  # noqa: E402
from echo_routing.realvideo.ecg import parse_burned_ecg, trace_top_row  # noqa: E402
from echo_routing.temporal.windowed import ECHOPRIME_TO_FAMILY5, ECHOPRIME_VIEWS, map_probs  # noqa: E402

ROOTS = [Path("/mnt/mimic/mimic-iv-echo/1.0.1"), Path.home() / "mimic-iv-echo/1.0.1"]
RECORDS = ROOTS[0] / "echo-record-list.csv"
B0_CKPT = "checkpoints/20260913-033411-b0_frame_encoder-b072725b-s0/best.pt"
EP_MEAN = np.array([29.110628, 28.076836, 29.096405], dtype=np.float32).reshape(3, 1, 1)
EP_STD = np.array([47.989223, 46.456997, 47.20083], dtype=np.float32).reshape(3, 1, 1)
DOWN = 4  # block size for the image-motion signal
CROP_AND_SCALE = None  # EchoPrime's official preprocessing, imported in main()


def locate(rel: str) -> Path | None:
    return next((r / rel for r in ROOTS if (r / rel).is_file()), None)


def select_cines() -> pd.DataFrame:
    rec = pd.read_csv(RECORDS)
    rec["path"] = rec["dicom_filepath"].map(locate)
    complete = rec.groupby("study_id")["path"].apply(lambda s: s.notna().all())
    rec = rec[rec["study_id"].isin(complete[complete].index)].copy()
    rows = []
    for r in rec.itertuples(index=False):
        ds = pydicom.dcmread(r.path, stop_before_pixels=True)
        regs = list(getattr(ds, "SequenceOfUltrasoundRegions", []))
        n = int(getattr(ds, "NumberOfFrames", 1) or 1)
        bmode = bool(regs) and all(int(g.RegionSpatialFormat) == 1 and int(g.RegionDataType) == 1 for g in regs)
        row = {"subject_id": r.subject_id, "study_id": r.study_id, "path": str(r.path), "n_frames": n,
               "frame_time_ms": float(getattr(ds, "FrameTime", np.nan) or np.nan),
               "hr_tag": float(ds.HeartRate) if str(getattr(ds, "HeartRate", "")).strip() else np.nan,
               "n_regions": len(regs), "bmode_cine": bmode and n > 1}
        if regs:
            row.update(x0=min(int(g.RegionLocationMinX0) for g in regs), y0=min(int(g.RegionLocationMinY0) for g in regs),
                       x1=max(int(g.RegionLocationMaxX1) for g in regs), y1=max(int(g.RegionLocationMaxY1) for g in regs))
        rows.append(row)
    return pd.DataFrame(rows)


def decode(path: str) -> list[Image.Image]:
    """All frames as RGB PIL images; Pillow decodes each JPEG frame (identical pixels to pydicom, ~7x faster)."""
    ds = pydicom.dcmread(path)
    return [Image.open(BytesIO(fb)).convert("RGB") for fb in generate_frames(ds.PixelData, number_of_frames=int(ds.NumberOfFrames))]


class Cines(Dataset):
    def __init__(self, rows: list[dict], b0_size: int) -> None:
        self.rows, self.size = rows, b0_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        key = f"{r['study_id']}_{Path(r['path']).stem}"
        try:
            ims = decode(r["path"])
        except Exception as e:  # corrupt or undecodable file: record and move on
            return {"key": key, "error": repr(e)[:200]}
        ft = r["frame_time_ms"] / 1000.0
        w, h = ims[0].size
        band_row = int(h * 0.6)
        band = np.stack([np.asarray(im.crop((0, band_row, w, h))) for im in ims])
        ecg = parse_burned_ecg(band, ft, search_from=0.0)
        top = trace_top_row(band[0], search_from=0.0)
        y1 = min(int(r["y1"]), band_row + top - 3) if top is not None else int(r["y1"])
        box = (int(r["x0"]), int(r["y0"]), int(r["x1"]), y1)
        grey = np.stack([np.asarray(im.crop(box).convert("L").reduce(DOWN), dtype=np.float32) for im in ims])
        cyc = estimate_cycle(grey, 1.0 / ft)
        idx = np.arange(0, len(ims), max(1, int(round(0.1 / ft))))
        ep = np.stack([(CROP_AND_SCALE(np.asarray(ims[k])).astype(np.float32).transpose(2, 0, 1) - EP_MEAN) / EP_STD for k in idx])
        b0 = np.stack([to_tensor_array(letterbox(ims[k], self.size)) for k in idx])
        n = len(ims)
        return {"key": key, "error": "", "idx": idx, "n": n, "ft": ft, "ecg": ecg, "cyc": cyc,
                "ep": torch.from_numpy(ep), "b0": torch.from_numpy(b0)}


def _one_thread(_: int) -> None:
    torch.set_num_threads(1)


def load_echoprime(repo_root: Path, device) -> torch.nn.Module:
    vc = torchvision.models.convnext_base(); vc.classifier[-1] = torch.nn.Linear(vc.classifier[-1].in_features, 11)
    sd = torch.load(repo_root / "models_frozen" / "echoprime_view_classifier" / "release_v1.0.0" / "view_classifier.pt", map_location="cpu")
    vc.load_state_dict(sd)
    return vc.to(device).eval()


def main() -> None:
    global CROP_AND_SCALE
    ap = argparse.ArgumentParser(description="MIMIC-IV-Echo round-1 pass")
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(); paths = load_paths(); out = Path(args.out); (out / "mimic").mkdir(parents=True, exist_ok=True)
    repo = paths.repo_root / "third_party" / "EchoPrime"; cwd = os.getcwd(); os.chdir(repo); sys.path.insert(0, str(repo))
    import utils  # noqa: E402  (official EchoPrime crop_and_scale)
    os.chdir(cwd); CROP_AND_SCALE = utils.crop_and_scale

    hdr_csv = out / "mimic_headers.csv"
    hdr = pd.read_csv(hdr_csv) if hdr_csv.is_file() else select_cines()
    hdr.to_csv(hdr_csv, index=False)
    sel = hdr[hdr["bmode_cine"] & hdr["frame_time_ms"].notna()]
    print(f"complete-study DICOMs {len(hdr)}, B-mode cines {len(sel)}, patients {sel.subject_id.nunique()}, studies {sel.study_id.nunique()}", flush=True)
    rows = sel.head(args.limit).to_dict("records") if args.limit else sel.to_dict("records")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction())
    b0, meta = load_checkpoint(paths.repo_root / B0_CKPT); b0 = b0.to(device).eval()
    ep_model = load_echoprime(paths.repo_root, device)
    dl = DataLoader(Cines(rows, int(meta.get("image_size", 224))), batch_size=None, num_workers=args.workers,
                    worker_init_fn=_one_thread)
    fields = ["key", "error", "n_frames", "frame_time_ms", "ecg_ok", "ecg_reason", "ecg_rr_s", "ecg_rr_cv", "ecg_n_r",
              "first_r_frame", "img_period_s", "img_peak", "img_dip", "img_cv", "img_n_cycles"]
    with open(out / "mimic_cines.csv", "w", newline="") as fh, torch.no_grad():
        wr = csv.DictWriter(fh, fieldnames=fields); wr.writeheader()
        for k, it in enumerate(dl, 1):
            if it["error"]:
                wr.writerow({"key": it["key"], "error": it["error"]}); continue
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pb = torch.softmax(b0(it["b0"].to(device)).float(), 1).cpu().numpy()
                raw = torch.softmax(ep_model(it["ep"].to(device)).float(), 1).cpu().numpy()
            fam = map_probs(raw, ECHOPRIME_VIEWS, ECHOPRIME_TO_FAMILY5)
            pe = np.c_[fam, np.clip(1 - fam.sum(1), 0, None)]
            ecg, cyc = it["ecg"], it["cyc"]
            np.savez_compressed(out / "mimic" / f"{it['key']}.npz", idx=it["idx"], n_frames=it["n"], frame_time_s=it["ft"],
                                prob_b0=pb.astype(np.float32), prob_ep=pe.astype(np.float32), raw_ep=raw.astype(np.float32),
                                ecg_phase=ecg.phase(it["n"]).astype(np.float32), r_frames=ecg.r_frames,
                                img_boundaries=cyc.boundaries, img_period_s=cyc.period_s)
            wr.writerow({"key": it["key"], "error": "", "n_frames": it["n"], "frame_time_ms": it["ft"] * 1000,
                         "ecg_ok": ecg.ok, "ecg_reason": ecg.reason, "ecg_rr_s": ecg.rr_s, "ecg_rr_cv": ecg.rr_cv,
                         "ecg_n_r": ecg.r_frames.size, "first_r_frame": ecg.r_frames[0] if ecg.r_frames.size else np.nan,
                         "img_period_s": cyc.period_s, "img_peak": cyc.peak, "img_dip": cyc.dip, "img_cv": cyc.cv,
                         "img_n_cycles": cyc.n_cycles})
            if k % 200 == 0:
                fh.flush(); print(f"  {k}/{len(rows)}", flush=True)
    print(f"wrote {out / 'mimic_cines.csv'}")


if __name__ == "__main__":
    main()
