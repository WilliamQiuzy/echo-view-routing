#!/usr/bin/env python
"""EchoNet-Pediatric inventory (task 2 of the 2026-10-01 round): aggregate statistics only, no data leaves the server.

Reports readable videos per view, frame rate and length distributions, the official 10-fold `Split`, and whether the
apparent patient (first file-name token) and exam (first two tokens) groupings stay within one fold. File names look
like <patient>-<exam>-<clip>.avi; this reading is checked against the metadata (one exam should carry one age, sex
and EF across views) but not against an official key.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("data/echonet/pediatric/echonetpediatric/pediatric_echo_avi/pediatric_echo_avi")


def main() -> None:
    ap = argparse.ArgumentParser(description="EchoNet-Pediatric inventory"); ap.add_argument("--out", required=True)
    args = ap.parse_args(); out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for view in ("A4C", "PSAX"):
        fl = pd.read_csv(ROOT / view / "FileList.csv")
        for r in fl.itertuples(index=False):
            cap = cv2.VideoCapture(str(ROOT / view / "Videos" / r.FileName))
            ok = cap.isOpened()
            fps = cap.get(cv2.CAP_PROP_FPS) if ok else np.nan; n = cap.get(cv2.CAP_PROP_FRAME_COUNT) if ok else np.nan
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) if ok else np.nan; h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if ok else np.nan
            cap.release()
            tok = r.FileName[:-4].split("-")
            rows.append({"view": view, "file": r.FileName, "patient": tok[0], "exam": "-".join(tok[:2]), "split": int(r.Split),
                         "ef": r.EF, "age": r.Age, "sex": r.Sex, "readable": ok, "fps": fps, "frames": n, "width": w, "height": h})
    df = pd.DataFrame(rows)
    df.to_csv(out.with_suffix(".csv"), index=False)  # server-side table (file names and metadata only)
    q = lambda s: [round(float(x), 2) for x in np.nanpercentile(s, [0, 10, 50, 90, 100])]  # noqa: E731
    per_patient_folds = df.groupby("patient")["split"].nunique(); per_exam_folds = df.groupby("exam")["split"].nunique()
    exam_meta = df.groupby("exam")[["age", "sex", "ef"]].nunique()
    summary = {
        "videos": df.groupby("view").size().to_dict(), "readable": df.groupby("view")["readable"].sum().astype(int).to_dict(),
        "fps_min_p10_p50_p90_max": {v: q(g["fps"]) for v, g in df.groupby("view")},
        "duration_s_min_p10_p50_p90_max": {v: q(g["frames"] / g["fps"]) for v, g in df.groupby("view")},
        "frame_size": df.groupby("view").apply(lambda g: g[["width", "height"]].drop_duplicates().astype(int).values.tolist()).to_dict(),
        "patients": int(df["patient"].nunique()), "exams": int(df["exam"].nunique()),
        "patients_in_more_than_one_fold": int((per_patient_folds > 1).sum()), "exams_in_more_than_one_fold": int((per_exam_folds > 1).sum()),
        "exams_with_both_views": int(df.groupby("exam")["view"].nunique().eq(2).sum()),
        "exams_with_inconsistent_metadata": int((exam_meta > 1).any(axis=1).sum()),
        "patients_with_several_exams": int(df.groupby("patient")["exam"].nunique().gt(1).sum()),
        "videos_per_fold": df.groupby("split").size().to_dict(), "patients_per_fold": df.groupby("split")["patient"].nunique().to_dict(),
    }
    out.write_text(json.dumps(summary, indent=1, default=str)); print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
