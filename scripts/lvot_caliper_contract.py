"""A49 public-object integrity and static LVOT caliper geometry contract.

No downloader, model, inferred annotation, clipping or target fallback.
"""
import hashlib
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np


def bind_main_lvot_caliper(attrs, decoded_targets):
    """Bind the main exporter global named caliper by a unique sector geometry join.

    Caller decodes targets from fully publisher-verified archive objects only.
    This establishes a weak clinical measurement location, never view/absence truth.
    """
    if attrs.get("content_types") != ["2d_brightness_mode"]:
        raise ValueError("exclusive B-mode content required")
    if attrs.get("frame_counts_by_content_type") != {"2d_brightness_mode": 1}:
        raise ValueError("exactly one static B-mode frame required")
    manifest = attrs["recording_manifest"]
    sectors = manifest.get("sectors", [])
    if len(sectors) != 1 or sectors[0].get("semantic_id") != "bmode":
        raise ValueError("exactly one B-mode sector required")
    sector = sectors[0]
    if sector.get("frames", {}).get("zarr_path") != "data/2d_brightness_mode":
        raise ValueError("explicit B-mode sector frame binding required")
    if sector.get("geometry", {}).get("sample_type") != "Tissue(8)":
        raise ValueError("native tissue sample required")
    named = [a for a in manifest.get("annotations", [])
             if a.get("kind") == "measurement" and
             a.get("links", {}).get("measurement_kind") == "dimension_lvot_diam"]
    if len(named) != 1:
        raise ValueError("one named LVOT diameter measurement required")
    annotation = named[0]
    links = annotation["links"]
    if (links.get("measure_id") != "Cardiac/2D/Dimension/LVOT Diam" or
        links.get("coordinate_type") != "spatial" or links.get("geometry_kind") != "caliper" or
        links.get("point_count") != 2 or links.get("init_from_cad") is not False or
        not links.get("source_entry_ref") or links.get("burned_in_only", False)):
        raise ValueError("LVOT clinical caliper provenance mismatch")
    path = annotation.get("value", {}).get("zarr_path")
    metadata = [v for v in attrs.get("arrays", []) if v.get("name") == path]
    if (len(metadata) != 1 or metadata[0].get("shape") != [2, 2] or
        metadata[0].get("units") != "physical_xy_m" or
        metadata[0].get("coordinate_system") != "physical_xy_m"):
        raise ValueError("physical two-endpoint array metadata required")
    points = np.asarray(decoded_targets[path], dtype=np.float64)
    geometry = normalized_calipers(points, sector["geometry"])
    def same(other):
        other = np.asarray(other, dtype=np.float64)
        return other.shape == (2, 2) and (np.allclose(points, other, atol=5e-7, rtol=0) or
                                        np.allclose(points[::-1], other, atol=5e-7, rtol=0))
    matches = [v for v in sector.get("overlays", {}).get("physical_points", [])
               if same(v.get("points", []))]
    if len(matches) != 1:
        raise ValueError("unique sector-owned overlay match required")
    for other in manifest.get("annotations", []):
        other_path = other.get("value", {}).get("zarr_path")
        if other is annotation or other.get("kind") != "measurement":
            continue
        if other_path in decoded_targets and same(decoded_targets[other_path]):
            raise ValueError("alternative measurement geometry binding")
    return {"target_path": path, "original_target": annotation.get("target"),
            "binding_method": "unique_sector_overlay_geometry_join",
            "measure_id": links["measure_id"], "geometry": geometry}


def stream_verified_tar(source, metadata, selected_prefixes, destination, max_retained_bytes):
    """Retain selected members in quarantine; admission requires returned full SHA proof.

    Includes unselected payload and trailing bytes in the publisher archive digest.
    Caller enforces network/time/global storage limits and deletes failed quarantine.
    """
    if not metadata.get("lfs"):
        raise ValueError("streamed archive requires publisher LFS SHA256")
    expected = metadata["lfs"].get("oid", metadata["lfs"].get("sha256"))
    destination = Path(destination)
    class Reader:
        def __init__(self):
            self.count = 0
            self.digest = hashlib.sha256()
        def read(self, n=-1):
            data = source.read(n)
            self.count += len(data)
            if self.count > metadata["size"]:
                raise ValueError("publisher archive size exceeded")
            self.digest.update(data)
            return data
    reader = Reader()
    retained = 0
    members = []
    with tarfile.open(fileobj=reader, mode="r|", bufsize=1024 * 1024) as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe archive path")
            if not any(member.name == prefix or member.name.startswith(prefix + "/")
                       for prefix in selected_prefixes):
                continue
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("selected archive member must be a regular file")
            if retained + member.size > max_retained_bytes:
                raise ValueError("retained archive byte cap exceeded")
            target = destination.joinpath(*path.parts)
            if target.exists():
                raise ValueError("duplicate archive member")
            target.parent.mkdir(parents=True, exist_ok=True)
            body = archive.extractfile(member).read()
            if len(body) != member.size:
                raise ValueError("truncated archive member")
            target.write_bytes(body)
            retained += len(body)
            members.append({"name": member.name, "bytes": len(body),
                            "sha256": hashlib.sha256(body).hexdigest()})
    while reader.read(8 * 1024 * 1024):
        pass
    if reader.count != metadata["size"]:
        raise ValueError("publisher archive size mismatch")
    if reader.digest.hexdigest() != expected:
        raise ValueError("publisher archive hash mismatch")
    return {"publisher_verified": True, "bytes": reader.count,
            "sha256": reader.digest.hexdigest(), "retained_bytes": retained, "members": members}


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
