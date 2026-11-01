import numpy as np
import pytest
from echo_routing.temporal.point_detail import point_canvas, source_gates


def test_point_canvas_preserves_geometry_and_degrades_only_detail():
    x=np.zeros((240,320,3),np.uint8);x[::2,:,0]=255;x[:,::2,1]=255
    h=point_canvas(x,'H');r=point_canvas(x,'R')
    assert h.shape==r.shape==(448,448,3)
    assert not h[:56].any() and not h[-56:].any()
    assert np.mean(abs(h.astype(float)-r.astype(float)))>1
    assert np.array_equal(point_canvas(x,'H'),h)


def test_invalid_pixel_contract_is_rejected():
    with pytest.raises(ValueError):point_canvas(np.zeros((5,5,3),float),'H')
    with pytest.raises(ValueError):point_canvas(np.zeros((5,5,3),np.uint8),'unknown')


def test_high_recall_cannot_override_a4c_protection():
    good={'balanced_accuracy':.9,'A4C_recall':.9,'A5C_recall':.9,'AUROC':.95}
    assert not source_gates({'H':good,'R':good},.6,.5)['absolute_pass']
    safe={**good,'A4C_recall':.96}
    assert source_gates({'H':safe,'R':good},.6,.5)['absolute_pass']
    assert not source_gates({'H':safe,'R':good},.6,.5)['detail_increment_pass']
