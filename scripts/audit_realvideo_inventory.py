#!/usr/bin/env python
"""Round-1 inventory for the view-change vs insufficient-evidence study (task 1).

Reports, without running any model:
- MIMIC-IV-Echo: downloaded DICOMs, complete studies, and header tags that could give an independent cardiac cycle
  (heart rate, frame time, ECG waveform, ultrasound regions).
- EV9V: clips, sessions and durations per split, and which recognizers have per-frame outputs for which split.
- EV9V frames: whether the ECG strip at the bottom of the image actually moves.
- EchoNet-Dynamic: ED/ES frame labels (a reference for validating an image-based phase estimate).
- EchoXFlow sample: whether an ECG array is stored.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path.home() / "echo-view-routing"
OUT: dict = {}


def mimic() -> None:
    rec = list(csv.DictReader(open("/mnt/mimic/mimic-iv-echo/1.0.1/echo-record-list.csv")))
    roots = [Path("/mnt/mimic/mimic-iv-echo/1.0.1"), Path.home() / "mimic-iv-echo/1.0.1"]
    have = set()
    part = 0
    for r in roots:
        for dp, _, fs in os.walk(r / "files"):
            for f in fs:
                if f.endswith(".dcm"):
                    have.add(os.path.relpath(os.path.join(dp, f), r))
                elif f.endswith(".part"):
                    part += 1
    by_study = collections.defaultdict(list)
    for r in rec:
        by_study[r["study_id"]].append(r["dicom_filepath"] in have)
    complete = [s for s, v in by_study.items() if all(v)]
    started = [s for s, v in by_study.items() if any(v)]
    subj = {r["subject_id"] for r in rec if r["study_id"] in set(complete)}
    OUT["mimic"] = {"records": len(rec), "downloaded_dcm": len(have), "part_files": part,
                    "studies_total": len(by_study), "studies_started": len(started), "studies_complete": len(complete),
                    "patients_with_complete_study": len(subj)}
    try:
        import pydicom
    except ImportError:
        OUT["mimic"]["pydicom"] = "not installed"
        return
    tags = collections.Counter(); frames = []; hr = []
    for rel in sorted(have)[:: max(1, len(have) // 60)][:60]:
        path = next(r / rel for r in roots if (r / rel).is_file())
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        for name in ("HeartRate", "FrameTime", "CineRate", "NumberOfFrames", "SequenceOfUltrasoundRegions", "WaveformSequence",
                     "RWaveTimeVector", "TriggerTime", "Modality", "ImageType"):
            if name in ds:
                tags[name] += 1
        if "NumberOfFrames" in ds:
            frames.append(int(ds.NumberOfFrames))
        if "HeartRate" in ds and str(ds.HeartRate).strip():
            hr.append(float(ds.HeartRate))
        regs = getattr(ds, "SequenceOfUltrasoundRegions", [])
        for g in regs:
            tags[f"region_datatype_{int(getattr(g, 'RegionDataType', -1))}"] += 1
    OUT["mimic"]["header_sample"] = {"n": 60, "tag_presence": dict(tags), "frames_median": float(np.median(frames)) if frames else None,
                                     "heart_rate_present": len(hr), "heart_rate_median": float(np.median(hr)) if hr else None}


def ev9v() -> None:
    man = [r for r in csv.DictReader(open(ROOT / "data/manifests/ev9v_native.csv")) if r["dataset"] == "EV9V"]
    per = {}
    for sp in ("train", "validation", "test"):
        rows = [r for r in man if r["split"] == sp]
        dur = np.array([float(r["duration_s"]) for r in rows])
        sess = {r["video_id"].split("_320x240")[0] for r in rows}
        per[sp] = {"clips": len(rows), "sessions": len(sess), "duration_s_median": round(float(np.median(dur)), 2),
                   "duration_s_p10_p90": [round(float(np.percentile(dur, 10)), 2), round(float(np.percentile(dur, 90)), 2)],
                   "clips_under_2s": int((dur < 2).sum()), "fps": sorted({r["fps_playback"] for r in rows})}
    OUT["ev9v"] = per
    # per-frame outputs available per split
    ids = {sp: [r["video_id"] for r in man if r["split"] == sp] for sp in ("train", "validation", "test")}
    cfg_dirs = {"B0 ResNet-18 (feature cache)": ROOT / "cache/features/79f4a41af6e3/orig",
                "D1 ViT-S (vit_frame)": ROOT / "cache/vit_streams/vit_frame_ad3cdda5ec67",
                "EchoPrime (per frame)": ROOT / "cache/echoprime_windowed/79f4a41af6e3",
                "EchoViewCLIP (1.6 s windows)": ROOT / "cache/echoviewclip_windowed/79f4a41af6e3",
                "STFM (1.6 s windows)": ROOT / "cache/stfm_windowed/79f4a41af6e3"}
    cov = {}
    for name, d in cfg_dirs.items():
        files = {p.stem for p in d.glob("*.npz")} if d.is_dir() else set()
        cov[name] = {sp: sum(1 for v in ids[sp] if v in files) for sp in ids}
    OUT["ev9v_per_frame_outputs"] = cov


def ecg_strip() -> None:
    """Does the ECG strip (bottom rows, green trace) change across frames of a clip?"""
    res = []
    for vid in ("2022-07-10_10-58-17_320x240_7", "2022-07-18_14-02-18_320x240_8", "2022-05-24_14-30-39_320x240_1"):
        d = ROOT / "data/ev9v/Images" / vid
        fr = sorted(d.glob("*.jpg"), key=lambda p: int(p.stem))[:90]
        strips = []
        for p in fr:
            a = np.asarray(Image.open(p).convert("RGB")).astype(int)[200:240, :, :]
            green = (a[..., 1] > 120) & (a[..., 0] < 100) & (a[..., 2] < 120)
            strips.append(green)
        s = np.stack(strips)
        per_frame = s.sum((1, 2))
        change = np.abs(np.diff(s.astype(int), axis=0)).sum((1, 2))
        rows_used = np.flatnonzero(s.any((0, 2)))
        res.append({"video": vid, "green_px_per_frame_median": float(np.median(per_frame)),
                    "green_px_changed_between_frames_median": float(np.median(change)),
                    "green_rows": [int(rows_used.min()) + 200, int(rows_used.max()) + 200] if rows_used.size else None})
    OUT["ev9v_ecg_strip"] = res


def echonet_dynamic() -> None:
    base = ROOT / "data/echonet/dynamic/EchoNet-Dynamic"
    fl = list(csv.DictReader(open(base / "FileList.csv")))
    vt = list(csv.DictReader(open(base / "VolumeTracings.csv")))
    frames = collections.defaultdict(set)
    for r in vt:
        frames[r["FileName"]].add(int(r["Frame"]))
    OUT["echonet_dynamic"] = {"videos": len(fl), "fields": list(fl[0].keys()), "videos_with_2_traced_frames": sum(1 for v in frames.values() if len(v) == 2),
                              "fps_median": float(np.median([float(r["FPS"]) for r in fl])),
                              "frames_median": float(np.median([float(r["NumberOfFrames"]) for r in fl]))}


def echoxflow() -> None:
    root = ROOT / "data/echoxflow"
    names = collections.Counter()
    for dp, dirs, fs in os.walk(root):
        for x in dirs + fs:
            low = x.lower()
            if any(k in low for k in ("ecg", "ekg", "tissue", "bmode", "b-mode", "doppler")):
                names[x] += 1
        if sum(names.values()) > 200:
            break
    OUT["echoxflow"] = {"exams_dirs": len(list(root.glob("*"))), "matching_names": names.most_common(12)}


for fn in (mimic, ev9v, ecg_strip, echonet_dynamic, echoxflow):
    try:
        fn()
    except Exception as e:  # report and continue: this is an inventory
        OUT[fn.__name__ + "_error"] = repr(e)[:300]
print(json.dumps(OUT, indent=1))
