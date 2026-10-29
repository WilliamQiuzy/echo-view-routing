"""A49 public-object integrity and static LVOT caliper geometry contract.

No downloader, model, inferred annotation, clipping or target fallback.
"""
import hashlib

import numpy as np


def verify_publisher_object(body, metadata):
    if len(body) != metadata["size"]:
        raise ValueError("publisher object size mismatch")
    if metadata.get("lfs"):
        expected = metadata["lfs"].get("oid", metadata["lfs"].get("sha256"))
        actual = hashlib.sha256(body).hexdigest()
        method = "lfs_sha256"
    else:
        expected = metadata["oid"]
        actual = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
        method = "git_blob_sha1"
    if actual != expected:
        raise ValueError("publisher object hash mismatch")
    return method


def normalized_calipers(points, geometry):
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("two finite physical endpoints required")
    d0, d1 = float(geometry["depth_start_m"]), float(geometry["depth_end_m"])
    if "angle_start_rad" in geometry:
        a0, a1 = float(geometry["angle_start_rad"]), float(geometry["angle_end_rad"])
    else:
        tilt, width = float(geometry.get("tilt_rad", 0)), float(geometry["width_rad"])
        a0, a1 = tilt - width / 2, tilt + width / 2
    if not np.isfinite([d0, d1, a0, a1]).all() or d1 <= d0 or a1 <= a0:
        raise ValueError("invalid sector geometry")
    length = float(np.linalg.norm(points[1] - points[0]))
    if length <= 0:
        raise ValueError("distinct endpoints required")
    radius = np.hypot(points[:, 0], points[:, 1])
    angle = np.arctan2(points[:, 0], points[:, 1])
    rc = np.stack(((radius - d0) / (d1 - d0), (angle - a0) / (a1 - a0)), axis=1)
    if not ((rc > 0) & (rc < 1)).all():
        raise ValueError("endpoint outside native sector")
    r = d0 + rc[:, 0] * (d1 - d0)
    a = a0 + rc[:, 1] * (a1 - a0)
    reconstructed = np.stack((r * np.sin(a), r * np.cos(a)), axis=1)
    return {"row_col": rc.tolist(), "midpoint_row_col": rc.mean(0).tolist(),
            "length_m": length,
            "roundtrip_error_m": float(np.max(np.abs(reconstructed - points)))}
