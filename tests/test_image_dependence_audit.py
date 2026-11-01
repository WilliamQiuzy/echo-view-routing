import importlib.util
from pathlib import Path

import numpy as np
import pytest


def audit():
    p = Path(__file__).parents[1] / "scripts/audit_a5c_image_dependence.py"
    assert p.exists(), "independent image-dependence auditor missing"
    spec = importlib.util.spec_from_file_location("independent_audit", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_reference_scores_expand_memberships_and_reject_bad_weight_mass():
    m = audit()
    scores = np.array([[[1., 10.], [3., 30.]], [[5., 50.], [7., 70.]]])
    result = m.reference_run_scores(np.array([10, 20]), scores, [20, 10], [.75, .25])
    assert np.array_equal(result, np.array([[4., 40.], [6., 60.]]))
    with pytest.raises(ValueError):
        m.reference_run_scores(np.array([10, 20]), scores, [20, 10], [.75, .35])
    with pytest.raises(ValueError):
        m.reference_run_scores(np.array([10, 10]), scores, [10], [1.])


def test_zero_is_not_a_positive_promotion_and_flips_are_paired():
    m = audit()
    counts = m.effect_counts([0., 1., -1., 2.], [1., 0., 0., 3.])
    assert counts["original_positive"] == 2
    assert counts["positive_after"] == 2
    assert counts["positive_to_nonpositive"] == 1
    assert counts["nonpositive_to_positive"] == 1


def test_constant_maps_are_undefined_not_claimed_to_agree():
    m = audit()
    assert m.correlation([1., 1., 1.], [2., 3., 4.]) is None
    assert m.correlation([1., 2., 3.], [4., 5., 6.]) == pytest.approx(1.)


def test_original_score_alignment_is_not_inferred_from_counts():
    m = audit()
    with pytest.raises(ValueError):
        m.assert_original_parity(np.array([1., -1.]), np.array([-1., 1.]))
    assert m.assert_original_parity(np.array([1., -1.]), np.array([1., -1.])) == 0


def test_independent_ocr_reference_rejects_changed_token_selection():
    m = audit()
    tsv = 'left\ttop\twidth\theight\tconf\ttext\n300\t220\t40\t40\t91\tA5C\n'
    q = {'raw_shape': [480, 640], 'raw_tsv': tsv,
         'processed_tsv': 'left\ttop\twidth\theight\tconf\ttext\n',
         'raw_boxes': [{'left':300,'top':220,'width':40,'height':40,'text':'A5C','confidence':91.}],
         'processed_boxes': []}
    rm, pm = m.reference_ocr_source(q)
    assert rm[112,112] and not pm.any()
    q['raw_boxes'] = []
    with pytest.raises(AssertionError): m.reference_ocr_source(q)
