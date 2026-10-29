import hashlib
import io
import tarfile

import numpy as np
import pytest

from scripts.lvot_caliper_contract import normalized_calipers, verify_publisher_object, stream_verified_tar, bind_main_lvot_caliper


def test_publisher_hashes_reject_corruption_and_size_mismatch():
    body = b"a complete public object"
    git_oid = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    assert verify_publisher_object(body, {"size": len(body), "oid": git_oid}) == "git_blob_sha1"
    lfs = {"size": len(body), "lfs": {"oid": hashlib.sha256(body).hexdigest()}}
    assert verify_publisher_object(body, lfs) == "lfs_sha256"
    with pytest.raises(ValueError, match="hash"):
        verify_publisher_object(body[:-1] + b"x", lfs)
    with pytest.raises(ValueError, match="size"):
        verify_publisher_object(body[:-1], lfs)


def test_caliper_polar_roundtrip_and_midpoint():
    geometry = dict(depth_start_m=.02, depth_end_m=.12, tilt_rad=.1, width_rad=1.)
    rc = np.array([[.25, .3], [.65, .7]])
    radius = .02 + rc[:, 0] * .1
    angle = -.4 + rc[:, 1]
    points = np.stack((radius * np.sin(angle), radius * np.cos(angle)), axis=1)
    result = normalized_calipers(points, geometry)
    np.testing.assert_allclose(result["row_col"], rc, atol=1e-12)
    np.testing.assert_allclose(result["midpoint_row_col"], [.45, .5], atol=1e-12)
    assert result["roundtrip_error_m"] < 1e-12


@pytest.mark.parametrize("points", [np.zeros((2, 2)), np.ones((3, 2)),
    np.array([[np.nan, .1], [.01, .1]]), np.array([[0., .01], [.01, .1]])])
def test_invalid_calipers_are_not_clipped_or_broadcast(points):
    with pytest.raises(ValueError):
        normalized_calipers(points, dict(depth_start_m=.02, depth_end_m=.12,
                                        tilt_rad=0., width_rad=1.))


def test_streamed_archive_hash_covers_unselected_payload_and_trailer(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as archive:
        for name, data in [("chosen/.zattrs", b"{}"), ("other/payload", b"discard")]:
            item = tarfile.TarInfo(name); item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    body = buf.getvalue() + b"trailer"
    meta = {"size": len(body), "lfs": {"oid": hashlib.sha256(body).hexdigest()}}
    result = stream_verified_tar(io.BytesIO(body), meta, {"chosen"}, tmp_path, 100)
    assert result["publisher_verified"] and result["bytes"] == len(body)
    assert (tmp_path / "chosen/.zattrs").read_bytes() == b"{}"
    assert not (tmp_path / "other").exists()
    corrupt = body[:-1] + b"x"
    with pytest.raises(ValueError, match="hash"):
        stream_verified_tar(io.BytesIO(corrupt), meta, {"chosen"}, tmp_path / "corrupt", 100)


def test_streamed_archive_blocks_path_escape(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as archive:
        item = tarfile.TarInfo("chosen/../../escape"); item.size = 1
        archive.addfile(item, io.BytesIO(b"x"))
    body = buf.getvalue()
    with pytest.raises(ValueError, match="path"):
        stream_verified_tar(io.BytesIO(body), {"size": len(body), "lfs": {
            "oid": hashlib.sha256(body).hexdigest()}}, {"chosen"}, tmp_path, 100)


def _metadata():
    points = [[.001, .06], [.009, .08]]
    a = {"content_types": ["2d_brightness_mode"],
         "frame_counts_by_content_type": {"2d_brightness_mode": 1},
         "arrays": [{"name": "annotations/measurement_000_points", "shape": [2, 2],
                     "units": "physical_xy_m", "coordinate_system": "physical_xy_m"}],
         "recording_manifest": {"annotations": [{"kind": "measurement",
             "target": {"type": "recording", "semantic_id": "global"},
             "value": {"zarr_path": "annotations/measurement_000_points"},
             "links": {"coordinate_type": "spatial", "geometry_kind": "caliper",
                       "measure_id": "Cardiac/2D/Dimension/LVOT Diam",
                       "measurement_kind": "dimension_lvot_diam", "point_count": 2,
                       "init_from_cad": False, "source_entry_ref": {"element_index": 2}}}],
             "sectors": [{"semantic_id": "bmode",
                "frames": {"zarr_path": "data/2d_brightness_mode"},
                "geometry": dict(sample_type="Tissue(8)", depth_start_m=.02,
                                 depth_end_m=.12, tilt_rad=0., width_rad=1.),
                "overlays": {"physical_points": [{"points": points}]}}]}}
    return a, np.array(points)


def test_main_schema_keeps_global_link_and_derived_geometry_join_separate():
    a, points = _metadata()
    result = bind_main_lvot_caliper(a, {"annotations/measurement_000_points": points})
    assert result["original_target"]["semantic_id"] == "global"
    assert result["binding_method"] == "unique_sector_overlay_geometry_join"
    assert result["target_path"] == "annotations/measurement_000_points"


@pytest.mark.parametrize("change", ["no_overlay", "two_matches", "cad", "second_sector", "other_stream"])
def test_main_schema_rejects_global_label_without_unique_binding(change):
    a, points = _metadata(); m = a["recording_manifest"]
    if change == "no_overlay": m["sectors"][0]["overlays"] = {}
    if change == "two_matches": m["sectors"][0]["overlays"]["physical_points"] *= 2
    if change == "cad": m["annotations"][0]["links"]["init_from_cad"] = True
    if change == "second_sector": m["sectors"] *= 2
    if change == "other_stream": a["content_types"].append("1d_pulsed_wave_doppler")
    with pytest.raises(ValueError):
        bind_main_lvot_caliper(a, {"annotations/measurement_000_points": points})
