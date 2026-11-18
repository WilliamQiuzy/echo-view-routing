"""EV9V frames at 10 Hz as one grey uint8 array, so that training never waits for a JPEG decode.

Layout: `<folder>/frames.u8` is a raw (N, 240, 320) uint8 array (18 GB); `<folder>/index.json` lists every recording
with its split, label, first row and row count. Training recordings come first, then the two development halves. The
EV9V test split is never read. Grey is the integer mean of the three channels.

Which recordings are read, and in which split and order, is fixed by the list that ships with the package
(`data/ev9v_recordings.json.gz`). It names a recording by its key (echoview.evaluation.ev9v.recording_key); the cache
is built from a local copy of EV9V by hashing the names of its folders.

    python -m echoview.training.cache --images /data/EV9V/Images --out /dev/shm/ev9v_cache
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import time
from importlib import resources
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..evaluation.ev9v import DataError, folders_by_key, recording_key

HEIGHT, WIDTH = 240, 320
STEP = 3  # EV9V plays at 30 frames per second; every third frame is a 10 Hz sample
SPLITS = ("train", "selection", "calibration")
RECORDINGS = "ev9v_recordings.json.gz"
LOG = logging.getLogger("echoview.training.cache")


def recordings() -> list[dict]:
    """The recordings of the cache in their fixed order: key, split, label and number of source frames."""
    raw = resources.files("echoview").joinpath("data", RECORDINGS).read_bytes()
    return json.loads(gzip.decompress(raw))["recordings"]


def layout(items: list[dict]) -> list[dict]:
    """Every recording with its first row and its number of rows in the frame array."""
    rows, start = [], 0
    for item in items:
        count = len(range(0, int(item["n_frames"]), STEP))
        rows.append({**item, "count": count, "start": start})
        start += count
    return rows


def decode_recording(job: tuple[str, str, int, int, int]) -> int:
    """Decodes one recording's 10 Hz frames into the shared array; returns the number of rows written."""
    from PIL import Image

    folder, target, start, count, total = job
    frames = np.memmap(target, np.uint8, "r+", shape=(total, HEIGHT, WIDTH))
    for k in range(count):
        with Image.open(Path(folder) / f"{k * STEP}.jpg") as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint16)
        if rgb.shape != (HEIGHT, WIDTH, 3):
            raise ValueError(f"unexpected frame size {rgb.shape} in {folder}")
        frames[start + k] = (rgb.sum(-1) // 3).astype(np.uint8)
    frames.flush()
    return count


def build(images: str | Path, out: str | Path, workers: int = 8, items: list[dict] | None = None) -> dict:
    """Writes frames.u8 and index.json into `out` from the Images folder of a local EV9V copy. `items` replaces the
    package's list of recordings (tests)."""
    images, out = Path(images), Path(out)
    rows = layout(recordings() if items is None else items)
    names = folders_by_key(images, {row["key"] for row in rows})
    out.mkdir(parents=True, exist_ok=True)
    total = rows[-1]["start"] + rows[-1]["count"]
    target = out / "frames.u8"
    np.memmap(target, np.uint8, "w+", shape=(total, HEIGHT, WIDTH)).flush()
    jobs = [(str(images / names[row["key"]]), str(target), row["start"], row["count"], total) for row in rows]
    with Pool(max(1, workers)) as pool:
        written = sum(pool.imap_unordered(decode_recording, jobs, chunksize=8))
    if written != total:
        raise DataError(f"wrote {written} of {total} rows")
    index = {"total": total, "height": HEIGHT, "width": WIDTH, "clips": rows}
    (out / "index.json").write_text(json.dumps(index) + "\n")
    return index


class Cache:
    """The frame array of a cache folder, read-only, and where every recording's rows are."""

    def __init__(self, folder: str | Path) -> None:
        folder = Path(folder)
        meta = json.loads((folder / "index.json").read_text())
        self.clips = meta["clips"]
        for clip in self.clips:  # a cache written by the research code names recordings; the key follows from the name
            clip.setdefault("key", recording_key(clip["name"]) if "name" in clip else "")
        self.total = int(meta["total"])
        self.frames = np.memmap(folder / "frames.u8", np.uint8, "r", shape=(self.total, int(meta["height"]), int(meta["width"])))
        self.by_key = {clip["key"]: clip for clip in self.clips}

    def rows(self, split: str) -> list[dict]:
        return [clip for clip in self.clips if clip["split"] == split]

    @property
    def train_end(self) -> int:
        """One past the last row of the training recordings (they come first)."""
        last = self.rows("train")[-1]
        return int(last["start"]) + int(last["count"])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="the Images folder of EV9V")
    parser.add_argument("--out", required=True, help="folder for frames.u8 (18 GB) and index.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    began = time.time()
    try:
        index = build(args.images, args.out, args.workers)
    except DataError as error:
        raise SystemExit(str(error)) from None
    counts = {split: sum(clip["split"] == split for clip in index["clips"]) for split in SPLITS}
    LOG.info("recordings %s, rows %d, %.1f GB, %d s", counts, index["total"], index["total"] * HEIGHT * WIDTH / 1e9, time.time() - began)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
