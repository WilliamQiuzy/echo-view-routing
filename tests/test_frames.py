"""Reading recordings: the picture the model gets, the 10 Hz grid, and every kind of input that must be refused."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydicom.dataset import FileDataset
from pydicom.encaps import encapsulate
from pydicom.uid import ExplicitVRLittleEndian, JPEGBaseline8Bit

from echoview import frames as fr

from .conftest import write_video
from .dicom_samples import H, W, dataset, jpeg, save, textured, write_compressed_cine


@pytest.fixture()
def compressed_cine(tmp_path) -> Path:
    return write_compressed_cine(tmp_path / "cine_jpeg.dcm")


def centre_level(recording: fr.Recording, index: int = 0) -> int:
    return int(recording.frames[index, 112, 112, 0])


# ---------------------------------------------------------------------------------------------- the picture
def test_crop_resizes_the_short_side_to_256_and_keeps_the_centre():
    picture = np.zeros((240, 320, 3), np.uint8)
    picture[:, 160:] = 200  # right half bright
    crop = fr.crop_uint8(Image.fromarray(picture))
    assert crop.shape == (224, 224, 3) and crop.dtype == np.uint8
    assert crop[:, :100].max() == 0 and crop[:, 124:].min() == 200


def test_grey_is_the_integer_mean_of_the_channels_in_all_three():
    colour = np.array([[[10, 20, 31]]], np.uint8)
    assert fr.to_grey(colour).tolist() == [[[20, 20, 20]]]


# ------------------------------------------------------------------------------------------------- DICOM
def test_a_compressed_cine_is_read_at_its_own_frame_rate(compressed_cine):
    recording = fr.read_recording(compressed_cine)
    assert recording.reader == fr.PILLOW_READER
    assert recording.fps == 50.0 and recording.n_source_frames == 30 and len(recording.frames) == 6
    assert recording.frames.shape[1:] == (224, 224, 3)


def test_a_given_frame_rate_never_replaces_the_files_own(compressed_cine):
    own, given = fr.read_recording(compressed_cine), fr.read_recording(compressed_cine, fps=30.0)
    assert given.fps == 50.0 and np.array_equal(own.frames, given.frames)


def test_the_pydicom_reader_gives_the_same_picture_as_the_pillow_reader(compressed_cine):
    pillow, fallback = fr.read_recording(compressed_cine), fr._read_dicom_decoded(compressed_cine, None)
    gap = np.abs(fallback.frames.astype(int) - pillow.frames.astype(int))
    assert fallback.reader == fr.PYDICOM_READER
    assert gap.mean() < 1.0 and gap.max() <= 3


def test_uncompressed_ybr_keeps_its_grey_levels(tmp_path):
    levels = [20, 36, 52, 69, 85, 101]
    pixels = np.stack([np.dstack([np.full((H, W), v, np.uint8), np.full((H, W), 128, np.uint8), np.full((H, W), 128, np.uint8)])
                       for v in levels])
    ds = dataset(ExplicitVRLittleEndian, NumberOfFrames=6, SamplesPerPixel=3, PhotometricInterpretation="YBR_FULL",
                 PlanarConfiguration=0, FrameTime=100.0)
    ds.PixelData = pixels.tobytes()
    recording = fr.read_recording(save(ds, tmp_path / "ybr_raw.dcm"))
    assert recording.reader == fr.PYDICOM_READER
    assert all(abs(centre_level(recording, k) - level) <= 1 for k, level in enumerate(levels))


def test_single_frames_need_no_frame_rate(tmp_path):
    ds = dataset(JPEGBaseline8Bit, NumberOfFrames=1, SamplesPerPixel=3, PhotometricInterpretation="YBR_FULL_422", PlanarConfiguration=0)
    ds.PixelData = encapsulate([jpeg(textured(90))])
    ds["PixelData"].is_undefined_length = True
    compressed = fr.read_recording(save(ds, tmp_path / "still_jpeg.dcm"))
    ds = dataset(ExplicitVRLittleEndian, SamplesPerPixel=1, PhotometricInterpretation="MONOCHROME2")
    ds.PixelData = textured(90).tobytes()
    raw = fr.read_recording(save(ds, tmp_path / "still_raw.dcm"))
    assert len(compressed.frames) == 1 and len(raw.frames) == 1
    assert abs(centre_level(compressed) - centre_level(raw)) <= 3


def deep_grey(stored_bits: int, value: int, **tags) -> FileDataset:
    ds = dataset(ExplicitVRLittleEndian, SamplesPerPixel=1, PhotometricInterpretation="MONOCHROME2", BitsAllocated=16,
                 BitsStored=stored_bits, HighBit=stored_bits - 1, **tags)
    ds.PixelData = np.full((H, W), value, np.uint16).tobytes()
    return ds


def test_deep_grey_keeps_its_highest_eight_stored_bits(tmp_path):
    assert centre_level(fr.read_recording(save(deep_grey(16, 0x8000), tmp_path / "deep16.dcm"))) == 128
    assert centre_level(fr.read_recording(save(deep_grey(12, 0x800), tmp_path / "deep12.dcm"))) == 128


def test_inverted_grey_is_turned_round(tmp_path):
    ds = dataset(ExplicitVRLittleEndian, SamplesPerPixel=1, PhotometricInterpretation="MONOCHROME1")
    ds.PixelData = np.full((H, W), 10, np.uint8).tobytes()
    assert centre_level(fr.read_recording(save(ds, tmp_path / "inverse.dcm"))) == 245


def test_signed_and_palette_data_are_refused_with_a_reason(tmp_path):
    signed = deep_grey(16, 0, PixelRepresentation=1)
    signed.PixelData = np.full((H, W), -100, np.int16).tobytes()
    with pytest.raises(fr.ReadError, match="signed"):
        fr.read_recording(save(signed, tmp_path / "signed.dcm"))
    palette = dataset(ExplicitVRLittleEndian, SamplesPerPixel=1, PhotometricInterpretation="PALETTE COLOR")
    palette.PixelData = np.zeros((H, W), np.uint8).tobytes()
    with pytest.raises(fr.ReadError, match="palette"):
        fr.read_recording(save(palette, tmp_path / "palette.dcm"))


def test_files_that_are_not_images_are_read_errors(tmp_path):
    report = dataset(ExplicitVRLittleEndian)
    del report.Rows
    del report.Columns
    with pytest.raises(fr.ReadError):
        fr.read_recording(save(report, tmp_path / "report.dcm"))
    (tmp_path / "notes.dcm").write_text("not a dicom file")
    with pytest.raises(fr.ReadError):
        fr.read_recording(tmp_path / "notes.dcm")
    with pytest.raises(fr.ReadError, match="no such file"):
        fr.read_recording(tmp_path / "missing.mp4")
    (tmp_path / "table.csv").write_text("a,b\n")
    with pytest.raises(fr.ReadError, match="not a DICOM file"):
        fr.read_recording(tmp_path / "table.csv")


# ------------------------------------------------------------------------------------------ video and folders
def test_a_video_is_read_on_the_10_hz_grid(tmp_path):
    # Neighbouring source frames are 60 grey levels apart, so a sample taken one frame off shows, whatever the codec does.
    source = [40 + 60 * (i % 3) for i in range(50)]
    clip = write_video(tmp_path / "clip.mp4", source, fps=25.0)
    recording = fr.read_recording(clip)
    assert recording.reader == "video, decoded by OpenCV"
    assert recording.fps == 25.0 and recording.n_source_frames == 50 and len(recording.frames) == 20
    assert recording.frames.shape[1:] == (224, 224, 3) and recording.frames.dtype == np.uint8
    wanted = [source[int(np.rint(k * 25.0 / fr.HZ))] for k in range(20)]  # the nearest source frame of every 0.1 s
    assert all(abs(centre_level(recording, k) - level) <= 20 for k, level in enumerate(wanted))
    assert np.array_equal(recording.frames[..., 0], recording.frames[..., 1])  # grey in all channels


@pytest.fixture()
def frame_folder(tmp_path) -> Path:
    folder = tmp_path / "frames"
    folder.mkdir()
    for k in range(12):
        Image.fromarray(textured(60 + k)).save(folder / f"{k}.png")
    return folder


def test_an_image_folder_needs_a_frame_rate(frame_folder):
    with pytest.raises(fr.ReadError, match="--fps"):
        fr.read_recording(frame_folder)
    assert len(fr.read_recording(frame_folder, fps=20.0).frames) == 6


def test_image_files_are_read_in_natural_order(frame_folder):
    recording = fr.read_recording(frame_folder, fps=10.0)
    levels = [centre_level(recording, k) for k in range(12)]
    assert levels == sorted(levels) and len(set(levels)) > 6  # 10.png comes after 9.png, not after 1.png


def test_a_16_bit_image_folder_is_refused(tmp_path):
    folder = tmp_path / "deep"
    folder.mkdir()
    Image.fromarray(np.full((H, W), 40000, np.uint16)).save(folder / "0.png")
    with pytest.raises(fr.ReadError, match="8 bits"):
        fr.read_recording(folder, fps=10.0)


def test_a_mixed_folder_gives_its_video_and_its_frame_folder_only(tmp_path, frame_folder):
    mixed = tmp_path / "mixed"
    (mixed / "__MACOSX").mkdir(parents=True)
    (mixed / "sub").mkdir()
    clip = write_video(tmp_path / "clip.mp4", [100] * 10)
    shutil.copy(clip, mixed / "a.mp4")
    (mixed / "._a.mp4").write_bytes(b"\0" * 200)
    shutil.copy(clip, mixed / "__MACOSX" / "x.mp4")
    Image.fromarray(textured(80)).save(mixed / "thumb.jpg")
    (mixed / "DICOMDIR").write_bytes(b"\0" * 128 + b"DICM" + b"\0" * 64)
    shutil.copytree(frame_folder, mixed / "sub" / "frames")
    found = [str(path.relative_to(mixed)) for path in fr.find_recordings([mixed])]
    assert found == ["a.mp4", "sub/frames"]


def test_a_file_named_directly_is_kept_as_given(tmp_path):
    assert fr.find_recordings([tmp_path / "anything.bin"]) == [tmp_path / "anything.bin"]
