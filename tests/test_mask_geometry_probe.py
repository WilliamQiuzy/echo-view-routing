from pathlib import Path
import importlib.util

import numpy as np


def test_translations_preserve_shape_area_and_fixed_order():
    p = Path(__file__).parents[1] / "scripts/probe_a5c_mask_geometry.py"
    assert p.exists(), "shape-preserving probe missing"
    spec = importlib.util.spec_from_file_location("shape_probe", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    x = np.zeros((224, 224), bool); x[4:7, 20:45] = True; x[221:, 216:] = True
    masks = m.translated_masks(x)
    assert len(masks) == 4
    assert np.array_equal(masks[0], x)
    assert all(v.sum() == x.sum() for v in masks)
    for a, shift in zip(masks, [(0, 0), (0, 112), (112, 0), (112, 112)]):
        assert np.array_equal(a, np.roll(x, shift, axis=(0, 1)))
    assert all(np.array_equal(v, np.zeros_like(x)) for v in m.translated_masks(np.zeros_like(x)))


def test_control_reports_planar_seam_confound_instead_of_claiming_equivalence():
    p = Path(__file__).parents[1] / "scripts/probe_a5c_mask_geometry.py"
    spec = importlib.util.spec_from_file_location("shape_probe", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    mask = np.ones((224, 224), bool); mask[16:-16, 16:-16] = False
    shifted = m.translated_masks(mask)[-1]
    a, b = m.mask_properties(mask, mask), m.mask_properties(shifted, mask)
    assert a["area_pixels"] == b["area_pixels"]
    assert a["periodic_boundary_edges"] == b["periodic_boundary_edges"]
    assert a["planar_boundary_edges"] != b["planar_boundary_edges"]
    assert a["jaccard_with_actual"] == 1
    assert b["jaccard_with_actual"] < 1
