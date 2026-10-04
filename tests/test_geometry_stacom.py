"""Coordinate checks for the STACOM 2011 reader on synthetic volumes with known geometry."""
import numpy as np
import pytest

from echo_routing.geometry import stacom


def write_vtk(path, vol_zyx, spacing, origin=(0, 0, 0)):
    z, y, x = vol_zyx.shape
    head = (f"# vtk DataFile Version 3.0\nvtk output\nBINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS {x} {y} {z}\n"
            f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\nORIGIN {origin[0]} {origin[1]} {origin[2]}\n"
            f"POINT_DATA {x * y * z}\n").encode()
    path.write_bytes(head + stacom.VTK_KEY + vol_zyx.astype(np.uint8).tobytes() + b"\n")


def pyramid(shape=(60, 70, 80), spacing=(0.7, 0.6, 0.5), apex=(28.0, 21.0, 31.0), half=(30.0, 35.0), depth_max=28.0):
    """Visible region of a probe at `apex` (mm, x y z) looking towards -z, with half-angles along x and y."""
    z, y, x = np.meshgrid(*(np.arange(n) * s for n, s in zip(shape, spacing[::-1])), indexing="ij")
    d = apex[2] - z
    inside = (d > 0) & (np.abs(x - apex[0]) <= d * np.tan(np.radians(half[0]))) & \
             (np.abs(y - apex[1]) <= d * np.tan(np.radians(half[1]))) & (d <= depth_max)
    return inside


def test_read_vtk_volume_roundtrip(tmp_path):
    vol = np.random.default_rng(0).integers(0, 256, size=(5, 6, 7)).astype(np.uint8)
    write_vtk(tmp_path / "a.vtk", vol, (0.5, 0.6, 0.7))
    got, sp = stacom.read_vtk_volume(tmp_path / "a.vtk")
    assert np.array_equal(got, vol) and np.allclose(sp, [0.5, 0.6, 0.7])


def test_read_vtk_volume_rejects_nonzero_origin(tmp_path):
    write_vtk(tmp_path / "a.vtk", np.zeros((2, 2, 2)), (1, 1, 1), origin=(1, 0, 0))
    with pytest.raises(ValueError, match="origin"):
        stacom.read_vtk_volume(tmp_path / "a.vtk")


def test_sector_geometry_recovers_apex_and_angles():
    spacing = np.array([0.7, 0.6, 0.5]); apex = (28.0, 21.0, 31.0)
    vol = pyramid(spacing=spacing, apex=apex).astype(np.uint8)[None] * 100
    s = stacom.sector_geometry(vol, spacing)
    assert np.allclose(s.apex_mm, apex, atol=1.0)
    assert abs(s.half_angle_x_deg - 30) < 1.5 and abs(s.half_angle_y_deg - 35) < 1.5
    assert np.allclose(s.axis, [0, 0, -1]) and s.centre_tilt_deg < 1.0


def test_sector_geometry_apex_at_low_z():
    spacing = np.array([0.7, 0.6, 0.5])
    vol = pyramid(spacing=spacing)[::-1].astype(np.uint8)[None] * 100     # probe now at the low-z end
    s = stacom.sector_geometry(vol, spacing)
    assert np.allclose(s.axis, [0, 0, 1]) and abs(s.apex_mm[2] - (59 * 0.5 - 31.0)) < 1.0


def capped_pyramid(shape, apex, half_deg, radius):
    """Probe at `apex` looking towards -z (1 mm voxels), with a spherical far cap of `radius`."""
    z, y, x = np.meshgrid(*(np.arange(n, dtype=float) for n in shape), indexing="ij")
    d = apex[2] - z; t = np.tan(np.radians(half_deg))
    return (d > 0) & (np.abs(x - apex[0]) <= d * t) & (np.abs(y - apex[1]) <= d * t) & \
           (np.sqrt((x - apex[0]) ** 2 + (y - apex[1]) ** 2 + d ** 2) <= radius)


@pytest.mark.parametrize("probe_at_low_z", [False, True])
def test_sector_geometry_ignores_a_long_clipped_plateau_with_a_curved_end(probe_at_low_z):
    # 45-degree sector: the box clips it after 20 mm, leaving a long plateau whose last slices narrow along the cap.
    # With the probe at low z, the plateau-and-cap side is the longer run from the first widest slice, which a rule
    # choosing the longer side would fit, giving a meaningless apex.
    m = capped_pyramid((90, 40, 40), (19.5, 19.5, 89.5), 45.0, 80.0)
    vol = (m[::-1] if probe_at_low_z else m)[None].astype(np.uint8)
    s = stacom.sector_geometry(vol, np.ones(3))
    want_axis, want_z = ([0, 0, 1], 89 - 89.5) if probe_at_low_z else ([0, 0, -1], 89.5)
    assert np.allclose(s.axis, want_axis) and abs(s.apex_mm[2] - want_z) < 1.5 and s.width_fit_r2 >= 0.98


def test_sector_geometry_with_the_whole_far_cap_inside_the_box():
    vol = capped_pyramid((100, 90, 90), (44.5, 44.5, 99.5), 30.0, 85.0)[None].astype(np.uint8)
    s = stacom.sector_geometry(vol, np.ones(3))
    assert np.allclose(s.axis, [0, 0, -1]) and abs(s.apex_mm[2] - 99.5) < 1.5
    assert abs(s.half_angle_x_deg - 30) < 1.5 and abs(s.half_angle_y_deg - 30) < 1.5


def test_sector_geometry_raises_without_a_taper():
    with pytest.raises(ValueError):
        stacom.sector_geometry(np.ones((1, 30, 20, 20), np.uint8), np.ones(3))


def test_ranking_drops_nan_candidates():
    assert stacom._rank({"a": 5.0, "b": float("nan"), "c": 3.0, "d": 7.0, "e": float("nan")}, "d") == (1, 3)
    assert stacom._rank({"a": 5.0, "b": float("nan")}, "b") == (None, 1)


def test_landmark_levels_apical_first():
    rng = np.random.default_rng(1); pts = []
    for h, r in ((0.0, 12.0), (20.0, 20.0), (40.0, 25.0)):      # apical ring is the smallest
        for a in rng.uniform(0, 2 * np.pi, 4):
            pts.append([r * np.cos(a), r * np.sin(a), h])
    pts = np.array(pts)[rng.permutation(12)]
    levels = stacom.landmark_levels(pts)
    assert [sorted(np.round(pts[g][:, 2]).tolist()) for g in levels] == [[0] * 4, [20] * 4, [40] * 4]


def test_landmark_contrast_peaks_at_true_position():
    spacing = np.array([1.0, 1.0, 1.0]); vol = np.full((40, 40, 40), 50, np.uint8)
    pts = []
    for h, r in ((10, 6), (20, 9), (30, 11)):
        for a in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
            pts.append([20 + r * np.cos(a), 20 + r * np.sin(a), h])
    pts = np.array(pts)
    for p in pts:                                                  # bright wall at the landmarks only
        x, y, z = np.round(p).astype(int); vol[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2] = 200
    assert stacom.landmark_contrast(vol, spacing, pts) > 100
    rank, total = stacom.shift_rank(vol, spacing, pts, step=2.0, reach=2)
    assert rank == 1 and total == 125


def test_mapping_rank_prefers_the_volume_as_read():
    spacing = np.array([1.0, 1.0, 1.0]); vol = np.full((40, 44, 48), 50, np.uint8)
    pts = []
    for h, r in ((10, 6), (20, 9), (30, 11)):
        for a in (0.3, 1.9, 3.4, 5.0):
            pts.append([24 + r * np.cos(a), 22 + r * np.sin(a), h])
    pts = np.array(pts)
    for p in pts:
        x, y, z = np.round(p).astype(int); vol[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2] = 200
    rank, total = stacom.mapping_rank(vol, spacing, pts)
    assert rank == 1 and total >= 8


def test_depth_and_margin():
    spacing = np.array([0.7, 0.6, 0.5]); apex = (28.0, 21.0, 31.0)
    vol = pyramid(spacing=spacing, apex=apex).astype(np.uint8)[None] * 100
    s = stacom.sector_geometry(vol, spacing)
    centre = np.array([28.0, 21.0, 16.0])                         # 15 mm below the apex on the beam
    assert abs(stacom.depth(centre, s)[0] - 15.0) < 1.0
    assert stacom.margin_to_edge(centre, s, spacing)[0] > 5.0
    assert stacom.margin_to_edge(np.array([1.0, 1.0, 1.0]), s, spacing)[0] == 0.0
