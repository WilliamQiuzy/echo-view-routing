"""CETUS reader and checks (echo_routing/geometry/cetus.py) on synthetic volumes."""
import numpy as np
import pytest

from echo_routing.geometry import cetus


def sector_volume(shape=(80, 80, 40), spacing=0.6, apex=(24.0, 24.0, -1.5), half=(35.0, 30.0), fill=13, seed=0):
    """(I, J, K) uint8: a probe at `apex` (mm, i j k) looking along +K, sector values 14-200, constant `fill` outside.
    As in CETUS, the apex is centred laterally and the box holds the sector's full width down to its far end."""
    i, j, k = np.meshgrid(*(np.arange(n) * spacing for n in shape), indexing="ij")
    d = k - apex[2]
    inside = (d > 0) & (np.abs(i - apex[0]) <= d * np.tan(np.radians(half[0]))) & \
             (np.abs(j - apex[1]) <= d * np.tan(np.radians(half[1])))
    rng = np.random.default_rng(seed)
    return np.where(inside, rng.integers(14, 201, size=shape), fill).astype(np.uint8)


def test_background_is_the_constant_fill():
    v = sector_volume(fill=8)
    bg, share = cetus.background(v)
    assert bg == 8 and 0.2 < share < 1.0


def test_spacing_order_matches_moved_axes():
    sp = np.array([1.0, 2.0, 3.0])
    assert cetus.spacing_xyz(sp, 0).tolist() == [3.0, 2.0, 1.0]      # moved: (I, J, K) -> Z=I, Y=J, X=K
    assert cetus.spacing_xyz(sp, 2).tolist() == [2.0, 1.0, 3.0]      # Z=K, Y=I, X=J
    assert cetus.moved(np.zeros((2, 3, 4)), 1).shape == (3, 2, 4)


def test_fit_sector_finds_the_beam_axis_and_apex():
    v = sector_volume()
    axis, sec, outcome = cetus.fit_sector(v, np.full(3, 0.6), 13)
    assert axis == 2 and sec.width_fit_r2 > 0.98
    # moved frame for axis 2: x = J, y = I, z = K
    assert np.allclose(sec.apex_mm, [24.0, 24.0, -1.5], atol=1.0)
    assert abs(sec.half_angle_x_deg - 30.0) < 2.0 and abs(sec.half_angle_y_deg - 35.0) < 2.0


def test_fit_sector_raises_on_a_box():
    with pytest.raises(ValueError):
        cetus.fit_sector(np.full((30, 30, 30), 40, np.uint8), np.ones(3), 13)


def test_lv_geometry_of_an_ellipsoid_along_the_beam():
    v = sector_volume()
    axis, sec, _ = cetus.fit_sector(v, np.full(3, 0.6), 13)
    i, j, k = np.meshgrid(*(np.arange(n) * 0.6 for n in v.shape), indexing="ij")
    mask = ((i - 24.0) / 5.0) ** 2 + ((j - 24.0) / 5.0) ** 2 + ((k - 14.0) / 8.0) ** 2 <= 1.0
    g = cetus.lv_geometry(mask, np.full(3, 0.6), axis, sec)
    assert g["components"] == 1 and g["largest_component_share"] == 1.0
    assert g["long_axis_vs_beam_deg"] < 3.0
    assert abs(g["volume_ml"] - 4 / 3 * np.pi * 5 * 5 * 8 / 1000) < 0.05
    assert g["share_outside_sector"] == 0.0 and abs(g["length_along_long_axis_mm"] - 16.0) < 1.5


def test_annotation_contrast_is_darker_inside_a_dark_cavity():
    v = sector_volume().astype(np.int32)
    i, j, k = np.meshgrid(*(np.arange(n) * 0.6 for n in v.shape), indexing="ij")
    mask = ((i - 24.0) / 5.0) ** 2 + ((j - 24.0) / 5.0) ** 2 + ((k - 14.0) / 8.0) ** 2 <= 1.0
    v[mask] = 20
    c = cetus.annotation_contrast(v.astype(np.uint8), mask, 13)
    assert c["mean_inside"] < c["mean_shell_outside"]


def test_read_case_converts_metre_spacing(tmp_path):
    nib = pytest.importorskip("nibabel")
    folder = tmp_path / "patient99"; folder.mkdir()
    v = sector_volume(shape=(20, 22, 24)).astype(np.float32); m = np.zeros_like(v); m[8:12, 8:12, 8:12] = 255
    aff = np.diag([-0.0006, -0.0006, 0.0006, 1.0])
    for ph in ("ED", "ES"):
        for s, a in (("", v), ("_gt", m)):
            im = nib.Nifti1Image(a, aff); im.header.set_zooms((0.0006, 0.0006, 0.0006)); im.header.set_xyzt_units("mm")
            nib.save(im, folder / f"patient99_{ph}{s}.nii.gz")
    case = cetus.read_case(folder)
    assert np.allclose(case.spacing_mm, 0.6) and case.volumes["ED"].dtype == np.uint8
    assert case.mask_values["ED"] == [0.0, 255.0] and case.masks["ES"].sum() == 64
