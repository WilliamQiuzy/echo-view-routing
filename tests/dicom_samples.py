"""Synthetic DICOM files for the tests. No real recording is used anywhere in the tests of the readers."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import JPEGBaseline8Bit, generate_uid

H, W = 64, 96


def textured(level: int, height: int = H, width: int = W) -> np.ndarray:
    """A grey picture with texture, so that resizing and JPEG have something to do."""
    y, x = np.mgrid[0:height, 0:width]
    return np.clip(level + 30 * np.sin(x / 7.0) + 20 * np.cos(y / 5.0), 0, 255).astype(np.uint8)


def dataset(transfer_syntax, **tags) -> FileDataset:
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = transfer_syntax
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.3.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset("x", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = H, W
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    for key, value in tags.items():
        setattr(ds, key, value)
    return ds


def save(ds: FileDataset, path: Path) -> Path:
    pydicom.dcmwrite(path, ds)
    return path


def jpeg(grey: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.repeat(grey[..., None], 3, -1)).save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


def write_compressed_cine(path: Path, frames: int = 30, frame_time_ms: float = 20.0) -> Path:
    """JPEG frames at 50 frames per second by default, as an ultrasound machine stores a cine."""
    ds = dataset(JPEGBaseline8Bit, NumberOfFrames=frames, SamplesPerPixel=3, PhotometricInterpretation="YBR_FULL_422",
                 PlanarConfiguration=0, FrameTime=frame_time_ms)
    ds.PixelData = encapsulate([jpeg(textured(60 + 4 * k)) for k in range(frames)])
    ds["PixelData"].is_undefined_length = True
    return save(ds, path)
