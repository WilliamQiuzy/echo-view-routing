import importlib.util
from pathlib import Path

import numpy as np


def api():
    path = Path(__file__).parents[1] / "echo_routing/temporal/image_dependence.py"
    assert path.exists(), "image-dependence implementation missing"
    spec = importlib.util.spec_from_file_location("image_dependence", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_outcomes_are_disjoint_and_keep_fixed_denominators():
    m = api()
    y = np.array([3, 3, 2, 2, 0, 3])
    b = np.array([2, 2, 2, 2, 2, 3])
    p = np.array([3, 2, 3, 2, 3, 3])
    g = m.outcome_groups(y, b, p)
    assert {k: v.tolist() for k, v in g.items()} == {
        "A5C_recovered": [0], "A5C_missed": [1],
        "A4C_harmed": [2], "A4C_retained": [3]}


def test_selection_prefers_sessions_then_recordings_and_is_repeatable():
    m = api()
    src = np.array(["a", "b", "c", "a", "d", "d"])
    sess = np.array(["s1", "s1", "s2", "s1", "s3", "s3"])
    chosen, levels = m.select_indices(np.arange(6), src, sess, [9, 2], 6)
    assert len(set(sess[chosen[:3]])) == 3
    assert len(set(src[chosen[:4]])) == 4
    assert levels == ["session"] * 3 + ["recording"] + ["remaining"] * 2
    assert m.select_indices(np.arange(6), src, sess, [9, 2], 6) == (chosen, levels)


def test_grid_is_exact_partition_and_equal_area_controls():
    m = api()
    cells = m.grid_masks()
    assert cells.shape == (49, 224, 224)
    assert np.all(cells.sum(0) == 1)
    assert np.all(cells.sum((1, 2)) == 1024)
    mask = m.border_mask()
    assert mask.sum() == 224 * 224 - 192 * 192
    ctl = m.area_control(mask, 91)
    assert ctl.sum() == mask.sum()
    assert not np.array_equal(ctl, mask)
    assert m.area_control(np.zeros_like(mask), 91).sum() == 0


def test_mask_changes_only_selected_pixels_and_preserves_time_shape():
    m = api()
    x = np.arange(16 * 224 * 224 * 3, dtype=np.uint8).reshape(16, 224, 224, 3)
    mask = m.grid_masks()[9]
    for fill in ["mean", "blur"]:
        z = m.mask_window(x, mask, fill)
        assert z.dtype == x.dtype and z.shape == x.shape
        assert np.array_equal(z[:, ~mask], x[:, ~mask])
    assert np.array_equal(m.mask_window(x, np.zeros_like(mask), "mean"), x)


def test_label_free_proxies_are_empty_on_dark_gray_and_detect_chroma():
    m = api()
    x = np.full((16, 224, 224, 3), 40, np.uint8)
    assert m.proxy_masks(x)["chroma"].sum() == 0
    assert m.proxy_masks(x)["static_bright"].sum() == 0
    x[:, 20:24, 30:34] = [255, 0, 0]
    x[:, 60:64, 70:74] = [240, 240, 240]
    masks = m.proxy_masks(x)
    assert masks["chroma"].sum() == 16
    assert masks["static_bright"].sum() == 16


def test_duration_weighted_mean_has_explicit_occurrence_reference():
    m = api()
    scores = np.array([-1., 3., 9.])
    assert m.weighted_mean(scores, np.array([0, 1]), np.array([.25, .75])) == 2.
    assert m.weighted_mean(scores, np.array([0, 1]), np.array([.25, .75])) == np.mean([-1., 3., 3., 3.])
    try:
        m.weighted_mean(scores, np.array([0, 1]), np.array([.5, .6]))
        assert False, "bad weights must be rejected"
    except ValueError:
        pass


def test_full_frame_footprint_removes_only_new_context_at_identical_scale():
    m = api()
    raw = np.full((300, 600, 3), 90, np.uint8)
    full, retained, rec = m.letterbox_pair(raw, 1.)
    assert full.shape == retained.shape == (224, 224, 3)
    keep = np.asarray(rec["retained_mask"], bool)
    assert np.array_equal(full[keep], retained[keep])
    assert np.all(retained[~keep] == 0)
    assert np.any(full[~keep] > 0)
    assert 0 < rec["raw_crop_retained_fraction"] < 1


def test_global_variants_keep_contract_order_and_marker_frame_identity():
    m = api()
    x = np.random.default_rng(7).integers(0, 256, (16, 224, 224, 3), dtype=np.uint8)
    variants = dict(m.global_variants(x, 6, x, x, 17))
    assert tuple(variants) == m.GLOBAL_NAMES
    assert np.array_equal(variants["original"], x)
    assert all(np.array_equal(f, x[6]) for f in variants["target_repeat"])
    for k in ("periphery", "chroma", "static_bright"):
        control = m.area_control(m.proxy_masks(x)[k], [17, ("periphery", "chroma", "static_bright").index(k)])
        assert control.sum() == m.proxy_masks(x)[k].sum()
