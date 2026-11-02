import copy,importlib
import pytest


def rows(coefficient):
    return [{'u':i,'schedule_index':i,'coefficient':coefficient,'physical_forwards':16,
             'physical_backwards':16,'source_plan_sha256':str(i),'replay_plan_sha256':str(i+3),
             'input_window_sha256':'pixels'+str(i)} for i in range(3)]


def test_matched_actual_pixels_are_required_in_every_update():
    fn=importlib.import_module('echo_routing.temporal.source_retention_audit').matched_journals
    left,right=rows(0),rows(1)
    assert fn(left,right,3)['matched_updates']==3
    bad=copy.deepcopy(right);bad[1]['input_window_sha256']='differentpixels'
    with pytest.raises(ValueError,match='pixels'):fn(left,bad,3)


@pytest.mark.parametrize('field,value',[('u',0),('physical_backwards',8),('coefficient',0)])
def test_incomplete_or_unmatched_update_cannot_pass(field,value):
    fn=importlib.import_module('echo_routing.temporal.source_retention_audit').matched_journals
    left,right=rows(0),rows(1);right[1][field]=value
    with pytest.raises(ValueError):fn(left,right,3)
