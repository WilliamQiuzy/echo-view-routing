"""Coordinate checks for the slice pipeline: plane geometry, both volume types, and pair sampling."""
import numpy as np
import pytest

from echo_routing.geometry import pairs, planes, volumes


def echoslicer_normal(phi_deg, theta_deg):
    p, t = np.radians(phi_deg), np.radians(theta_deg)
    return np.array([np.cos(t) * np.cos(p), np.sin(t), np.cos(t) * np.sin(p)])


def random_frame(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q.T                                                   # orthonormal rows


def test_basis_orthonormal_and_zero_plane():
    rng = np.random.default_rng(0)
    for psi, alpha in rng.uniform(-180, 180, size=(20, 2)):
        b, l, n = planes.ProbePlane(psi, alpha).basis()
        assert np.allclose(np.stack([b, l, n]) @ np.stack([b, l, n]).T, np.eye(3), atol=1e-12)
    b, l, n = planes.ProbePlane(0.0, 0.0).basis()
    assert np.allclose(b, [1, 0, 0]) and np.allclose(l, [0, 0, 1]) and np.allclose(n, [0, 1, 0])


@pytest.mark.parametrize("alpha", [-20.0, -5.0, 0.0, 10.0, 35.0])
def test_alpha_is_echoslicer_theta_minus_90_at_psi_0(alpha):
    n = planes.ProbePlane(0.0, alpha).basis()[2]
    assert np.allclose(n, echoslicer_normal(0.0, 90.0 + alpha), atol=1e-12)


def test_image_columns_follow_echoslicer_display():
    # EchoSlicer's A4C image runs along -azimuth (u = n x (1,0,0) = -z at theta = 90, flipu False): the left column
    # is the +z side. Our columns run along -l, so they must agree.
    pts = planes.ProbePlane(0.0, 0.0).points(np.array([-10.0, 10.0]), np.array([50.0]))[0]
    assert pts[0][2] > 0 > pts[1][2] and np.allclose(pts[:, 0], 50.0)


def test_psi_90_is_echoslicer_a2c_start_plane():
    n = planes.ProbePlane(90.0, 0.0).basis()[2]
    assert np.isclose(abs(n @ echoslicer_normal(90.0, 0.0)), 1.0)


def test_tilt_between_equals_d_alpha_for_same_psi():
    rng = np.random.default_rng(1)
    for psi, a1, a2 in zip(rng.uniform(0, 180, 20), rng.uniform(-15, 15, 20), rng.uniform(-15, 15, 20)):
        assert np.isclose(planes.tilt_between(planes.ProbePlane(psi, a1), planes.ProbePlane(psi, a2)), abs(a2 - a1), atol=1e-6)


def test_from_echoslicer_gives_the_same_plane():
    rng = np.random.default_rng(2)
    for d, phi, theta in zip(rng.uniform(-30, 30, 20), rng.uniform(-60, 60, 20), rng.uniform(40, 130, 20)):
        p = planes.from_echoslicer(d, phi, theta); n_es = echoslicer_normal(phi, theta); n = p.basis()[2]
        assert np.isclose(abs(n @ n_es), 1.0, atol=1e-9)
        pts = p.points(np.array([-20.0, 0.0, 30.0]), np.array([0.0, 40.0])).reshape(-1, 3)
        assert np.allclose((pts - d * n_es) @ n_es, 0.0, atol=1e-9)      # every rendered point is on EchoSlicer's plane


def cartesian_from_field(field, shape=(50, 60, 70), spacing=(0.8, 0.7, 0.9), apex=(25.0, 20.0, 40.0), frame=None):
    spacing = np.array(spacing); z, y, x = np.meshgrid(*(np.arange(n) * s for n, s in zip(shape, spacing[::-1])), indexing="ij")
    world = np.stack([x, y, z], -1)
    frame = np.eye(3) if frame is None else frame
    probe = (world - np.asarray(apex)) @ frame.T                 # world -> probe coordinates
    vol = field(probe.reshape(-1, 3)).reshape(shape)[None].astype(np.float32)
    return volumes.CartesianVolume(vol, spacing, np.asarray(apex, float), frame, np.ones(shape, bool), np.array([0.0]), "synthetic")


def test_cartesian_sampling_is_exact_for_a_linear_field():
    rng = np.random.default_rng(3); frame = random_frame(rng); w = np.array([0.3, -0.7, 1.1])
    vol = cartesian_from_field(lambda p: 100 + p @ w, apex=(28.0, 21.0, 22.0), frame=frame)   # apex at the box centre
    pts = rng.uniform(-8, 8, size=(200, 3))                      # within 14 mm of the apex, inside the box
    v, inside = vol.sample(pts, 0)
    assert inside.all() and np.allclose(v, 100 + pts @ w, atol=1e-3)


def test_cartesian_inside_needs_the_whole_interpolation_support():
    vol = cartesian_from_field(lambda p: np.ones(len(p)), shape=(10, 10, 10), spacing=(1.0, 1.0, 1.0), apex=(0.0, 0.0, 0.0))
    v, inside = vol.sample(np.array([[5.0, 5.0, 5.0], [8.6, 5.0, 5.0], [9.0, 5.0, 5.0], [9.2, 5.0, 5.0]]), 0)
    # voxel 9 is the last one; the eroded support ends at 8, so 8.6 rounds to 9 and is out, 9.2 is past the box
    assert inside.tolist() == [True, False, False, False] and v[0] == 1.0


def test_cartesian_rejects_non_orthonormal_frame():
    with pytest.raises(ValueError):
        cartesian_from_field(lambda p: p[:, 0], frame=np.array([[1.0, 0, 0], [1.0, 1.0, 0], [0, 0, 1.0]]))


def spherical_from_field(field, rho_max=60.0, half_deg=35.0):
    grid = (np.linspace(0, rho_max, 241), np.radians(np.linspace(-half_deg, half_deg, 141)), np.radians(np.linspace(-half_deg, half_deg, 141)))
    R, P, T = np.meshgrid(*grid, indexing="ij")
    pts = np.stack([R * np.cos(T) * np.cos(P), R * np.sin(T), R * np.cos(T) * np.sin(P)], -1)
    return volumes.SphericalVolume(field(pts.reshape(-1, 3)).reshape(R.shape)[..., None].astype(np.float32), grid, np.array([0.0]), "synthetic")


def test_spherical_sampling_and_inside():
    vol = spherical_from_field(lambda p: np.linalg.norm(p, axis=1))
    pts = np.array([[30.0, 2.0, -3.0], [10.0, 0.0, 0.0], [80.0, 0.0, 0.0], [10.0, 0.0, 20.0]])   # last two outside
    v, inside = vol.sample(pts, 0)
    assert inside.tolist() == [True, True, False, False]
    assert np.allclose(v[:2], np.linalg.norm(pts[:2], axis=1), atol=1e-3)


def test_cartesian_and_spherical_render_the_same_scene():
    blob = lambda p: 200 * np.exp(-np.sum((p - np.array([35.0, 4.0, -6.0])) ** 2, axis=1) / (2 * 8.0 ** 2))
    rng = np.random.default_rng(4)
    cart = cartesian_from_field(blob, shape=(110, 120, 130), spacing=(0.6, 0.6, 0.6), apex=(39.0, 36.0, 60.0), frame=random_frame(rng))
    sph = spherical_from_field(blob)
    grid = pairs.SliceGrid(pixel_mm=0.5, width_mm=40.0, depth_mm=55.0)
    for psi, alpha in ((0.0, 0.0), (30.0, 8.0), (120.0, -10.0)):
        a, ma = pairs.render(cart, planes.ProbePlane(psi, alpha), 0, grid); b, mb = pairs.render(sph, planes.ProbePlane(psi, alpha), 0, grid)
        both = ma & mb
        assert both.mean() > 0.3
        assert np.abs(a[both] - b[both]).max() < 2.0              # grey levels out of 200


def test_sample_spec_kinds():
    rng = np.random.default_rng(5)
    specs = [pairs.sample_spec(rng, 12) for _ in range(300)]
    assert {s.kind for s in specs} == set(pairs.KINDS)
    for s in specs:
        assert -150 <= s.psi_deg <= 30 and -15 <= s.alpha1_deg <= 15 and -15 <= s.alpha2_deg <= 15
        if s.kind == "same_plane":
            assert s.d_alpha == 0 and s.t1 != s.t2
        elif s.kind == "same_time":
            assert s.d_alpha != 0 and s.t1 == s.t2
        else:
            assert s.d_alpha != 0 and s.t1 != s.t2
    with pytest.raises(ValueError):
        pairs.sample_spec(rng, 1)


def test_fixed_mask_and_render_pair():
    vol = spherical_from_field(lambda p: 50 + np.linalg.norm(p, axis=1))
    vol = volumes.SphericalVolume(np.repeat(vol.data, 3, axis=-1), vol.grid, np.arange(3.0), "synthetic")
    grid = pairs.SliceGrid(pixel_mm=0.5, width_mm=60.0, depth_mm=55.0)
    mask = pairs.fixed_mask(vol, 40.0, grid, (-15.0, 15.0), erode_px=2)
    assert 0.1 < mask.mean() < 0.9
    for a in np.arange(-15.0, 15.0 + 1e-9, 2.5):
        _, inside = pairs.render(vol, planes.ProbePlane(40.0, float(a)), 0, grid)
        assert inside[mask].all()
    out = pairs.render_pair(vol, pairs.PairSpec(40.0, -10.0, 12.0, 0, 2, "both"), grid, mask, mean=128.0, sd=20.0)
    assert out["d_alpha"] == 22.0 and out["img1"].shape == mask.shape and (out["img1"][~mask] == 0).all()
    # no clipping at this mean and SD, so both are met exactly (clipping at 0 or 255 would move them)
    assert abs(out["img1"][mask].mean() - 128.0) < 1e-3 and abs(out["img1"][mask].std() - 20.0) < 1e-3
    assert out["raw_features1"].shape == (4,)
