import hashlib

import numpy as np
import pytest

from scripts.lvot_caliper_contract import normalized_calipers, verify_publisher_object


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
