"""Reading a recording as the model reads it.

Every recording is brought to a 10 Hz grid (the nearest source frame at each 0.1 s), each frame is resized so that its
short side is 256 pixels, cropped to the central 224 x 224 and reduced to grey levels (the mean of the three channels,
repeated in three channels). Nothing else is done to the image: no masking, no intensity change.

Three sources are read:
  DICOM cine    compressed 8-bit frames are taken from the file without decoding the others (dicom_frames.py) and
                decoded by Pillow. This is the path that produced the project's MIMIC-IV-Echo numbers. Other DICOM
                files (uncompressed, more than 8 bits, a compression Pillow does not read) go through pydicom's
                own decoder; `Recording.reader` says which path a file took.
  video file    decoded by OpenCV (mp4, avi, mov, ...).
  image folder  one 8-bit image per frame, in natural file-name order; needs the frame rate.

A file's own frame rate is always used. The rate passed by the caller serves only inputs that carry none.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import dicom_frames

HZ = 10.0
SIDE = 224
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv")
DICOM_SUFFIXES = (".dcm", ".dicom")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
EIGHT_BIT_MODES = ("1", "L", "LA", "P", "RGB", "RGBA", "YCbCr", "CMYK")
PILLOW_READER = "dicom, compressed frames decoded by Pillow"
PYDICOM_READER = "dicom, decoded by pydicom"


class ReadError(ValueError):
    """The recording cannot be read; the message says why."""


@dataclass(frozen=True)
class Recording:
    name: str
    frames: np.ndarray  # (n, 224, 224, 3) uint8: the 10 Hz samples as the model reads them
    fps: float
    n_source_frames: int
    reader: str


def crop_uint8(image: Image.Image) -> np.ndarray:
    """Short side to 256 (bilinear), central 224 x 224, uint8 (224, 224, 3)."""
    image = image.convert("RGB")
    width, height = image.size
    scale = 256 / min(width, height)
    image = image.resize((max(SIDE, round(width * scale)), max(SIDE, round(height * scale))), Image.BILINEAR)
    width, height = image.size
    left, top = (width - SIDE) // 2, (height - SIDE) // 2
    return np.asarray(image.crop((left, top, left + SIDE, top + SIDE)), dtype=np.uint8)


def to_grey(frames: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 -> (..., 3) uint8 with the mean of the three channels in each."""
    grey = (frames.astype(np.uint16).sum(-1) // 3).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=-1)


def is_dicom(path: Path) -> bool:
    if path.suffix.lower() in DICOM_SUFFIXES:
        return True
    if path.suffix.lower() in VIDEO_SUFFIXES + IMAGE_SUFFIXES:
        return False
    try:
        with path.open("rb") as handle:
            handle.seek(128)
            return handle.read(4) == b"DICM"
    except OSError:
        return False


def _usable(rate: float | None) -> bool:
    return rate is not None and bool(np.isfinite(rate)) and rate > 0


def _rate(own: float | None, given: float | None, where: str) -> float:
    """The input's own frame rate; the caller's only when the input has none."""
    if _usable(own):
        return float(own)
    if _usable(given):
        return float(given)
    raise ReadError(f"{where}: no usable frame rate; pass one with --fps")


def _from_crops(name: str, crops: dict[int, np.ndarray], n_frames: int, fps: float, reader: str) -> Recording:
    chosen = dicom_frames.sample_indices(n_frames, fps, HZ)
    frames = np.stack([crops[int(i)] for i in chosen])
    return Recording(name=name, frames=to_grey(frames), fps=float(fps), n_source_frames=int(n_frames), reader=reader)


def _read_dicom(path: Path, fps: float | None) -> Recording:
    data = path.read_bytes()
    try:
        head, blobs = dicom_frames.select(data, HZ, fps)
    except dicom_frames.NoFrameRate as error:
        raise ReadError(f"{path.name}: {error}") from error
    except dicom_frames.FrameError:
        return _read_dicom_decoded(path, fps)  # not a compressed cine: let pydicom decode it
    try:
        crops = {}
        for index, blob in zip(head["sent"], blobs):
            image = Image.open(io.BytesIO(blob))
            image.load()
            if image.size != (head["cols"], head["rows"]) or image.mode not in EIGHT_BIT_MODES:
                raise ValueError("not an 8-bit frame of the header's size")
            crops[index] = crop_uint8(image)
    except (OSError, SyntaxError, ValueError):
        return _read_dicom_decoded(path, fps)  # a compression or a bit depth that Pillow does not read as it should
    frames = np.stack([crops[i] for i in head["chosen"]])
    return Recording(name=path.name, frames=to_grey(frames), fps=float(head["fps"]), n_source_frames=int(head["n_frames"]),
                     reader=PILLOW_READER)


def _read_dicom_decoded(path: Path, fps: float | None) -> Recording:
    """Any other DICOM image through pydicom. 8 bits are kept as they are; deeper data keeps its 8 highest stored bits."""
    import pydicom

    ds = pydicom.dcmread(str(path))
    photometric = str(getattr(ds, "PhotometricInterpretation", ""))
    if photometric == "PALETTE COLOR":
        raise ReadError(f"{path.name}: palette colour images are not supported")
    if int(getattr(ds, "PixelRepresentation", 0) or 0) == 1:
        raise ReadError(f"{path.name}: signed pixel data is not supported")
    if hasattr(ds, "pixel_array_options"):
        ds.pixel_array_options(as_rgb=False)  # pydicom 3 would turn YBR into RGB itself; it is converted once, below
    array = np.asarray(ds.pixel_array)
    n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    colour = int(getattr(ds, "SamplesPerPixel", 1) or 1) == 3
    if n_frames == 1:
        array = array[None]
    if array.ndim != (4 if colour else 3) or len(array) != n_frames:
        raise ReadError(f"{path.name}: unexpected pixel array shape {array.shape}")
    if colour and photometric.startswith("YBR"):
        try:
            from pydicom.pixels import convert_color_space
        except ImportError:  # pydicom 2
            from pydicom.pixel_data_handlers.util import convert_color_space
        array = convert_color_space(array, photometric, "RGB")
    if array.dtype != np.uint8:
        stored = int(getattr(ds, "BitsStored", 0) or array.dtype.itemsize * 8)
        array = np.clip(array.astype(np.int64) >> max(0, stored - 8), 0, 255).astype(np.uint8)
    if photometric == "MONOCHROME1":
        array = 255 - array
    if not colour:
        array = np.repeat(array[..., None], 3, axis=-1)
    rate = HZ if n_frames == 1 else _rate(dicom_frames.frame_rate(ds)[0], fps, path.name)
    wanted = set(dicom_frames.sample_indices(n_frames, rate, HZ).tolist())
    crops = {i: crop_uint8(Image.fromarray(array[i])) for i in wanted}
    return _from_crops(path.name, crops, n_frames, rate, PYDICOM_READER)


def _read_video(path: Path, fps: float | None) -> Recording:
    try:
        import cv2
    except ImportError as error:
        raise ReadError("reading video files needs OpenCV: pip install opencv-python-headless") from error
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ReadError(f"{path.name}: the video could not be opened")
        rate = _rate(capture.get(cv2.CAP_PROP_FPS), fps, path.name)
        target = lambda k: int(np.rint(k * rate / HZ))  # the source frame of the k-th 10 Hz sample
        crops: dict[int, np.ndarray] = {}
        index, k = 0, 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index == target(k):
                crops[index] = crop_uint8(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                while target(k) <= index:
                    k += 1
            index += 1
    finally:
        capture.release()
    if index == 0:
        raise ReadError(f"{path.name}: the video holds no frame")
    return _from_crops(path.name, crops, index, rate, "video, decoded by OpenCV")


def _natural(path: Path) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _visible_files(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")]


def _read_image_folder(path: Path, fps: float | None) -> Recording:
    files = sorted((p for p in _visible_files(path) if p.suffix.lower() in IMAGE_SUFFIXES), key=_natural)
    if not files:
        raise ReadError(f"{path.name}: the folder holds no image")
    rate = _rate(None, fps, path.name)
    wanted = set(dicom_frames.sample_indices(len(files), rate, HZ).tolist())
    crops = {}
    for i in wanted:
        with Image.open(files[i]) as image:
            if image.mode not in EIGHT_BIT_MODES:
                raise ReadError(f"{path.name}: {files[i].name} has more than 8 bits per channel, which is not supported")
            crops[i] = crop_uint8(image)
    return _from_crops(path.name, crops, len(files), rate, "image folder")


def read_recording(path: str | Path, fps: float | None = None) -> Recording:
    """One recording on the 10 Hz grid. `fps` is used only for an input that carries no frame rate of its own
    (an image folder always needs it). Every failure is a ReadError that names the file."""
    path = Path(path)
    try:
        if not path.exists():
            raise ReadError(f"{path}: no such file or folder")
        if path.is_dir():
            return _read_image_folder(path, fps)
        if is_dicom(path):
            return _read_dicom(path, fps)
        if path.suffix.lower() in VIDEO_SUFFIXES:
            return _read_video(path, fps)
        raise ReadError(f"{path.name}: not a DICOM file, a video file or an image folder")
    except ReadError:
        raise
    except Exception as error:
        raise ReadError(f"{path.name}: {type(error).__name__}: {str(error)[:200]}") from error


def _is_frame_folder(folder: Path) -> bool:
    """Images and nothing else that could be a recording: a stray picture beside videos is not a recording."""
    files = _visible_files(folder)
    has_image = any(p.suffix.lower() in IMAGE_SUFFIXES for p in files)
    return has_image and not any(p.suffix.lower() in VIDEO_SUFFIXES or is_dicom(p) for p in files)


def _skipped(relative: Path) -> bool:
    """Hidden files, macOS archive leftovers and the DICOM directory file are never recordings."""
    return any(part.startswith(".") or part == "__MACOSX" for part in relative.parts) or relative.name.upper() == "DICOMDIR"


def find_recordings(inputs: list[str | Path]) -> list[Path]:
    """Files as given; a folder gives its DICOM and video files (searched in sub-folders too, in sorted order) and
    every folder inside it that holds frame images and no other recording."""
    found: list[Path] = []
    for item in inputs:
        path = Path(item)
        if not path.is_dir():
            found.append(path)
            continue
        if _is_frame_folder(path):
            found.append(path)
            continue
        for child in sorted(path.rglob("*")):
            if _skipped(child.relative_to(path)):
                continue
            if child.is_dir():
                if _is_frame_folder(child):
                    found.append(child)
            elif child.suffix.lower() in VIDEO_SUFFIXES or (child.suffix.lower() not in IMAGE_SUFFIXES and is_dicom(child)):
                found.append(child)
    return found
