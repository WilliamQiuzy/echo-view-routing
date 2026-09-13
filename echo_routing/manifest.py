"""Native-recording manifest for EV9V (proposal section 9.1 data contract).

One row per native cine. Columns:
  dataset, dataset_revision, split, video_id, raw_label, mapping_version,
  family5_label (nullable), family5_index (nullable), raw9_index,
  n_frames (nullable until frames are on disk), fps_playback, duration_s (nullable),
  frames_dir (nullable), video_path (nullable)
No patient identifiers are inferred from filenames.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from echo_routing.errors import ManifestError

from echo_routing.audit.label_map import LabelOntology

SPLIT_FILES = {"train": "train_labeled.txt", "validation": "validation_labeled.txt", "test": "test_labeled.txt"}
FPS_PLAYBACK = 30.0
MANIFEST_COLUMNS = [
    "dataset", "dataset_revision", "split", "video_id", "raw_label", "mapping_version",
    "family5_label", "family5_index", "raw9_index", "n_frames", "fps_playback", "duration_s",
    "frames_dir", "video_path",
]


@dataclass(frozen=True)
class SplitEntry:
    split: str
    video_id: str
    raw_label: str


def read_split_file(path: Path, split: str) -> list[SplitEntry]:
    entries: list[SplitEntry] = []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ManifestError(f"{path}:{lineno}: expected '<video_id> <label>', got {raw!r}")
        entries.append(SplitEntry(split=split, video_id=parts[0], raw_label=parts[1]))
    return entries


def read_all_splits(raw_dir: Path) -> list[SplitEntry]:
    entries: list[SplitEntry] = []
    for split, fname in SPLIT_FILES.items():
        entries.extend(read_split_file(Path(raw_dir) / fname, split))
    _check_integrity(entries)
    return entries


def _check_integrity(entries: Iterable[SplitEntry]) -> None:
    seen: dict[str, SplitEntry] = {}
    for e in entries:
        prev = seen.get(e.video_id)
        if prev is not None:
            raise ManifestError(f"duplicate video_id across splits: {e.video_id} ({prev.split}, {e.split})")
        seen[e.video_id] = e


def count_frames(frames_dir: Path) -> int | None:
    if not frames_dir.is_dir():
        return None
    return sum(1 for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def build_manifest(
    raw_dir: Path,
    ontology: LabelOntology,
    images_root: Path | None = None,
    videos_root: Path | None = None,
    dataset: str = "EV9V",
    dataset_revision: str = "",
) -> pd.DataFrame:
    """Assemble the native manifest. Frame counts are filled only if `images_root` exists."""
    rows = [
        _row(e, ontology, images_root, videos_root, dataset, dataset_revision)
        for e in read_all_splits(raw_dir)
    ]
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    return df


def _row(e: SplitEntry, ont: LabelOntology, images_root, videos_root, dataset, revision) -> dict:
    frames_dir = (Path(images_root) / e.video_id) if images_root else None
    video_path = (Path(videos_root) / f"{e.video_id}.mp4") if videos_root else None
    n_frames = count_frames(frames_dir) if frames_dir else None
    fam = ont.family(e.raw_label)
    return {
        "dataset": dataset,
        "dataset_revision": revision,
        "split": e.split,
        "video_id": e.video_id,
        "raw_label": e.raw_label,
        "mapping_version": ont.mapping_version,
        "family5_label": fam,
        "family5_index": ont.family_index(e.raw_label),
        "raw9_index": ont.raw9_index(e.raw_label),
        "n_frames": n_frames,
        "fps_playback": FPS_PLAYBACK,
        "duration_s": (n_frames / FPS_PLAYBACK) if n_frames else None,
        "frames_dir": str(frames_dir) if frames_dir and frames_dir.is_dir() else None,
        "video_path": str(video_path) if video_path and video_path.is_file() else None,
    }


def save_manifest(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return path


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"video_id": str, "raw_label": str})
    missing = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing:
        raise ManifestError(f"manifest missing columns: {sorted(missing)}")
    return df


def task_subset(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Rows usable for a task: family5 drops excluded codes (PMPALA); raw9 keeps all."""
    if task == "family5":
        return df[df["family5_index"].notna()].copy()
    if task == "raw9":
        return df.copy()
    raise ManifestError(f"unknown task {task!r}")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="raw_label", columns="split", values="video_id", aggfunc="count", fill_value=0)
