"""Frames of an encapsulated multi-frame DICOM cine on the project's 10 Hz grid, taken without decoding.

The pixel data of a compressed cine is a sequence of items: an offset table, then the compressed frames. This module
takes a whole file's bytes, finds the frame items and returns the bytes of the frames the 10 Hz grid asks for.
It is the module that read the MIMIC-IV-Echo files for the project's numbers, with one addition: `select` accepts a
frame rate for files whose header has none.

Only technical header fields are read. No name, date, identifier or free text is kept.
"""
from __future__ import annotations

import io
import struct

import numpy as np

ITEM = (0xFFFE, 0xE000)
SEQUENCE_END = (0xFFFE, 0xE0DD)
PIXEL_DATA = (0x7FE0, 0x0010)
UNDEFINED = 0xFFFFFFFF
JPEG_START = b"\xff\xd8"
REGION_FIELDS = ("RegionSpatialFormat", "RegionDataType", "RegionFlags", "RegionLocationMinX0", "RegionLocationMinY0",
                 "RegionLocationMaxX1", "RegionLocationMaxY1", "PhysicalUnitsXDirection", "PhysicalUnitsYDirection",
                 "PhysicalDeltaX", "PhysicalDeltaY")


class FrameError(ValueError):
    """The file cannot give its 10 Hz frames; the message says why."""


class NoFrameRate(FrameError):
    """The header names no frame rate and none was given."""


def sample_indices(n_frames: int, fps: float, hz: float = 10.0) -> np.ndarray:
    """Nearest decoded frame at each 10-Hz time not beyond the last frame."""
    if n_frames < 1 or not np.isfinite(fps) or fps <= 0 or hz <= 0:
        raise ValueError('positive frame count and finite positive source rate required')
    count = int(np.floor((n_frames - 1) * hz / fps + 1e-9)) + 1
    return np.minimum(np.rint(np.arange(count) * fps / hz).astype(np.int64), n_frames - 1)


def _positive(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def frame_rate(ds) -> tuple[float | None, str]:
    """Acquisition frames per second and the header field it came from."""
    frame_time = _positive(getattr(ds, "FrameTime", None))
    if frame_time is not None:
        return 1000.0 / frame_time, "FrameTime"
    vector = getattr(ds, "FrameTimeVector", None)
    if vector is not None:
        steps = [s for s in (_positive(v) for v in list(vector)[1:]) if s is not None]
        if steps:
            return 1000.0 * len(steps) / sum(steps), "FrameTimeVector"
    for name in ("CineRate", "RecommendedDisplayFrameRate"):
        rate = _positive(getattr(ds, name, None))
        if rate is not None:
            return rate, name
    return None, "none"


def read_header(data: bytes):
    """The header dataset and the byte offset of the pixel data element."""
    import pydicom

    stream = io.BytesIO(data)
    ds = pydicom.dcmread(stream, stop_before_pixels=True)
    return ds, stream.tell()


def _plain(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)[:40]
    return int(number) if number == int(number) else number


def technical_header(ds) -> dict:
    """Image geometry, encoding and timing. Nothing that identifies a person, a study or a date."""
    regions = [{name: _plain(getattr(r, name, None)) for name in REGION_FIELDS}
               for r in (getattr(ds, "SequenceOfUltrasoundRegions", None) or [])]
    return {
        "n_frames": int(getattr(ds, "NumberOfFrames", 1) or 1),
        "rows": int(ds.Rows), "cols": int(ds.Columns),
        "photometric": str(getattr(ds, "PhotometricInterpretation", "")),
        "samples_per_pixel": _plain(getattr(ds, "SamplesPerPixel", None)),
        "planar": _plain(getattr(ds, "PlanarConfiguration", None)),
        "bits": _plain(getattr(ds, "BitsAllocated", None)),
        "transfer_syntax": str(ds.file_meta.TransferSyntaxUID),
        "frame_time_ms": _positive(getattr(ds, "FrameTime", None)),
        "cine_rate": _positive(getattr(ds, "CineRate", None)),
        "heart_rate": _positive(getattr(ds, "HeartRate", None)),
        "color_data": _plain(getattr(ds, "UltrasoundColorDataPresent", None)),
        "regions": regions,
    }


def fragment_spans(data: bytes, offset: int) -> tuple[bytes, list[tuple[int, int]]]:
    """The offset table and the (start, length) of every fragment of the encapsulated pixel data at `offset`."""
    if offset + 12 > len(data) or struct.unpack_from("<HH", data, offset) != PIXEL_DATA:
        raise FrameError("no pixel data element")
    if data[offset + 4:offset + 6] in (b"OB", b"OW"):
        length, pos = struct.unpack_from("<I", data, offset + 8)[0], offset + 12
    else:
        length, pos = struct.unpack_from("<I", data, offset + 4)[0], offset + 8
    if length != UNDEFINED:
        raise FrameError("pixel data is not encapsulated")
    table, spans = None, []
    while True:
        if pos + 8 > len(data):
            raise FrameError("pixel data sequence has no end")
        group, element, size = struct.unpack_from("<HHI", data, pos)
        pos += 8
        if (group, element) == SEQUENCE_END:
            break
        if (group, element) != ITEM or size == UNDEFINED or pos + size > len(data):
            raise FrameError("malformed pixel data item")
        if table is None:
            table = data[pos:pos + size]
        else:
            spans.append((pos, size))
        pos += size
    if table is None or not spans:
        raise FrameError("pixel data sequence holds no frame")
    return table, spans


def frame_groups(data: bytes, spans: list[tuple[int, int]], n_frames: int) -> list[list[tuple[int, int]]]:
    """The fragments of each frame: one per frame, or fragments joined from one JPEG start marker to the next."""
    if len(spans) == n_frames:
        return [[span] for span in spans]
    groups: list[list[tuple[int, int]]] = []
    for start, size in spans:
        if not groups or data[start:start + 2] == JPEG_START:
            groups.append([])
        groups[-1].append((start, size))
    if len(groups) != n_frames:
        raise FrameError(f"{len(spans)} fragments do not form {n_frames} frames")
    return groups


def select(data: bytes, hz: float = 10.0, fps: float | None = None) -> tuple[dict, list[bytes]]:
    """Header fields and the compressed bytes of the distinct frames on the 10 Hz grid.

    head['chosen'] lists the source frame of every 10 Hz sample; head['sent'] lists the distinct frames in the order
    of the returned byte strings. The header's frame rate is used; `fps` serves only a file whose header names
    none, and a single-frame file needs none."""
    ds, offset = read_header(data)
    head = technical_header(ds)
    given = fps
    fps, source = frame_rate(ds)
    if fps is None and head["n_frames"] == 1:
        fps, source = hz, "single frame"
    if fps is None and _positive(given) is not None:
        fps, source = _positive(given), "given"
    head["fps"], head["fps_source"] = fps, source
    if fps is None:
        raise NoFrameRate("no frame rate in the header; pass one with --fps")
    table, spans = fragment_spans(data, offset)
    groups = frame_groups(data, spans, head["n_frames"])
    chosen = sample_indices(len(groups), fps, hz)
    sent = sorted(set(chosen.tolist()))
    frames = [b"".join(data[start:start + size] for start, size in groups[i]) for i in sent]
    head.update(chosen=chosen.tolist(), sent=sent, sizes=[len(f) for f in frames], n_fragments=len(spans),
                offset_table_bytes=len(table), status="ok")
    return head, frames
