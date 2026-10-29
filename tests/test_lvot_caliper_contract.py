import hashlib
import io
import tarfile

import numpy as np
import pytest

from scripts.lvot_caliper_contract import normalized_calipers, verify_publisher_object, stream_verified_tar


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
