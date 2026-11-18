"""EV9V frames as the evaluation reads them.

EV9V (bgx666/EV9V on the Hugging Face Hub, CC BY 4.0) ships every recording as a folder of JPEG frames:
`Images/<recording>/<frame number>.jpg`. A benchmark sample names a recording by its key (see `recording_key`), a frame
number and a gamma. The frame
is read, the gamma is applied to its grey levels (a table look-up; 1.0 changes nothing), the short side is resized to
256, the central 224 x 224 is cut out and the three channels are averaged: the same picture the model gets from a
video file.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from ..frames import SIDE, crop_uint8
from .benchmark import Ref, Sequence


class DataError(RuntimeError):
    """The dataset is not where it should be; the message says what was looked for."""


def recording_key(name: str) -> str:
    """The benchmark's name for a recording: the first 16 hex digits of the SHA-256 of its folder name."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def folders_by_key(images: Path, keys: set[str]) -> dict[str, str]:
    """The folder name of every wanted key in a local EV9V Images folder."""
    if not images.is_dir():
        raise DataError(f"{images} is not a folder. It should be the Images folder of EV9V (bgx666/EV9V on the Hugging Face Hub).")
    found = {}
    for entry in images.iterdir():
        if entry.is_dir() and (key := recording_key(entry.name)) in keys:
            found[key] = entry.name
    missing = len(keys - set(found))
    if missing:
        raise DataError(f"{missing} of the {len(keys)} recordings of the benchmark are not in {images}: it is not a complete EV9V Images folder")
    return found


def gamma_table(gamma: float) -> list[int]:
    return [int(255 * ((i / 255.0) ** gamma) + 0.5) for i in range(256)] * 3


def load_frame(path: Path, gamma: float) -> np.ndarray:
    """One frame as uint8 (224, 224, 3): the JPEG, the gamma, short side 256, central crop."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        if abs(gamma - 1.0) > 1e-9:
            image = image.point(gamma_table(gamma))
        return crop_uint8(image)


def frame_path(images: Path, video: str, frame: int) -> Path:
    return images / video / f"{frame}.jpg"


def grey_plane(frame: np.ndarray) -> np.ndarray:
    """(224, 224, 3) uint8 -> (224, 224) uint8: the integer mean of the three channels."""
    return (frame.astype(np.uint16).sum(-1) // 3).astype(np.uint8)


class FrameTable:
    """Every frame a set of sequences refers to, in grey, held in memory once (about 50 kB per frame)."""

    def __init__(self, refs: tuple[Ref, ...], grey: np.ndarray) -> None:
        if grey.shape != (len(refs), SIDE, SIDE) or grey.dtype != np.uint8:
            raise ValueError(f"expected one uint8 plane per reference, got {grey.shape} {grey.dtype}")
        self.refs, self.grey = refs, grey
        self._row = {ref: row for row, ref in enumerate(refs)}

    @staticmethod
    def references(sequences: Iterable[Sequence]) -> tuple[Ref, ...]:
        return tuple(sorted({ref for sequence in sequences for ref in sequence.refs}))

    @staticmethod
    def digest(refs: tuple[Ref, ...]) -> str:
        text = "\n".join(f"{video}\t{frame}\t{round(gamma, 6)}" for video, frame, gamma in refs)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    @classmethod
    def build(cls, sequences: Iterable[Sequence], images: str | Path, workers: int = 8, cache: str | Path | None = None) -> FrameTable:
        """Reads the frames from `images` (the dataset's Images folder). With `cache`, the table is kept there as one
        file and read back on the next run for the same sequences."""
        images = Path(images)
        refs = cls.references(sequences)
        stored = Path(cache) / f"ev9v_frames_{cls.digest(refs)}.npy" if cache else None
        if stored is not None and stored.is_file():
            return cls(refs, np.load(stored))
        names = folders_by_key(images, {key for key, _, _ in refs})
        missing = next((frame_path(images, names[k], f) for k, f, _ in refs if not frame_path(images, names[k], f).is_file()), None)
        if missing is not None:
            raise DataError(f"{missing} does not exist: {images} is not a complete EV9V Images folder")

        def one(ref: Ref) -> np.ndarray:
            key, frame, gamma = ref
            return grey_plane(load_frame(frame_path(images, names[key], frame), round(float(gamma), 6)))

        with ThreadPoolExecutor(max(1, workers)) as pool:
            grey = np.stack(list(pool.map(one, refs)))
        if stored is not None:
            stored.parent.mkdir(parents=True, exist_ok=True)
            partial = stored.with_suffix(".partial.npy")
            np.save(partial, grey)
            partial.replace(stored)
        return cls(refs, grey)

    def frames(self, sequence: Sequence) -> np.ndarray:
        """The sequence as the model reads it: (samples, 224, 224, 3) uint8, grey in all three channels."""
        rows = [self._row[ref] for ref in sequence.refs]
        return np.repeat(self.grey[rows][..., None], 3, axis=-1)
