"""EchoNet-Pediatric as an external set (docs/research_plan.md, round of 2026-10-01, task 2).

Frozen before any prediction:
- labels come from the view folder (A4C -> family5 "A4C", PSAX -> family5 "PSAX"); the model keeps its five outputs;
- a file name reads <group>-<exam>-<clip number>. The official 10 folds never split a first-token group, so the group
  is the resampling unit. It is not verified to be a patient identity;
- the official folds have no designated test fold: folds 0-1 are the external calibration part, folds 2-9 the test part;
- samples sit on a 10 Hz grid of the clip's own time axis (nearest frame at the AVI's own frame rate);
- stitch banks are built inside one group only, so no source is shared between groups:
  `exam` puts all clips of one exam in acquisition (clip-number) order, `same` pairs same-view clips of one group.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from echo_routing.compose.recipe_builder import Fragment, Recipe, recipe_to_dict

PROTOCOL_VERSION = "pediatric-external-v1"
PREPROCESS_VERSION = "ped-ts10hz-letterbox224-v1"
VIEW_TO_FAMILY5 = {"A4C": 2, "PSAX": 1}
VIEWS = tuple(VIEW_TO_FAMILY5)
CALIBRATION_FOLDS = (0, 1)
VIDEO_ROOT = Path("data/echonet/pediatric/echonetpediatric/pediatric_echo_avi/pediatric_echo_avi")


def parse_name(file_name: str) -> tuple[str, str, str]:
    """(group, exam, clip number) from '<group>-<exam>-<clip>.avi'; exam ids are made unique by keeping the group."""
    stem = file_name[:-4] if file_name.lower().endswith(".avi") else file_name
    tok = stem.split("-")
    if len(tok) != 3 or not tok[2].isdigit():
        raise ValueError(f"unexpected EchoNet-Pediatric file name {file_name!r}")
    return tok[0], f"{tok[0]}-{tok[1]}", tok[2]


def by_clip(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Sort by `keys` with the clip number compared as an integer (the release pads it to six digits, but the
    acquisition order must not depend on that)."""
    return df.assign(_clip=df["clip_no"].astype(int)).sort_values([*keys, "_clip"]).drop(columns="_clip")


def read_filelists(root: Path) -> pd.DataFrame:
    """One row per listed video: video_id, view, file, path, group, exam, clip_no, fold, role, label."""
    rows = []
    for view in VIEWS:
        fl = pd.read_csv(Path(root) / view / "FileList.csv")
        for r in fl.itertuples(index=False):
            group, exam, clip = parse_name(r.FileName)
            fold = int(r.Split)
            rows.append({"video_id": r.FileName[:-4], "view": view, "file": r.FileName,
                         "path": str(Path(root) / view / "Videos" / r.FileName), "group": group, "exam": exam,
                         "clip_no": clip, "fold": fold, "role": "calibration" if fold in CALIBRATION_FOLDS else "test",
                         "label": VIEW_TO_FAMILY5[view]})
    df = pd.DataFrame(rows)
    if df["video_id"].duplicated().any():
        raise ValueError("duplicate video ids across views")
    return by_clip(df, ["role", "group", "exam"]).reset_index(drop=True)


def timestamp_indices(n_frames: int, fps: float, hz: float) -> np.ndarray:
    """Nearest frame to t = k / hz for k = 0, 1, ... while the frame exists (true frame rate, no integer stride)."""
    if n_frames <= 0 or fps <= 0 or hz <= 0:
        raise ValueError("n_frames, fps and hz must be positive")
    idx = np.floor(np.arange(0.0, n_frames / fps, 1.0 / hz) * fps + 0.5).astype(int)
    return np.unique(idx[idx < n_frames])


def _recipe(rid: str, cell: str, rows: pd.DataFrame) -> Recipe:
    return Recipe(rid, cell, tuple(Fragment(r.video_id, int(r.label), "orig", 0, int(r.n_samples))
                                   for r in rows.itertuples(index=False)))


def exam_recipes(df: pd.DataFrame) -> list[Recipe]:
    """Every exam with at least two clips: its clips in clip-number order. Each video is used once."""
    out = []
    for exam, g in by_clip(df, ["group", "exam"]).groupby("exam", sort=False):
        if len(g) >= 2:
            out.append(_recipe(f"pedexam_{exam}", "exam", g))
    return out


def same_view_recipes(df: pd.DataFrame) -> list[Recipe]:
    """Inside each group and view, consecutive non-overlapping pairs in (exam, clip-number) order; each video at most once."""
    out = []
    for (group, view), g in by_clip(df, ["group", "view", "exam"]).groupby(["group", "view"], sort=False):
        for k in range(len(g) // 2):
            pair = g.iloc[2 * k:2 * k + 2]
            out.append(_recipe(f"pedsame_{group}_{view}_{k}", "same", pair))
    return out


def recipe_group(recipe: Recipe) -> str:
    return parse_name(recipe.fragments[0].video_id + ".avi")[0]


def digest_rows(rows: list[dict] | pd.DataFrame) -> str:
    """SHA-256 of a canonical JSON rendering (order-sensitive, so the frozen order is part of the digest)."""
    data = rows.to_dict(orient="records") if isinstance(rows, pd.DataFrame) else rows
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def recipes_digest(recipes: list[Recipe]) -> str:
    return digest_rows([recipe_to_dict(r) for r in recipes])
