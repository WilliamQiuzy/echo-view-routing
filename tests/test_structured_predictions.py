import importlib
import numpy as np
import pytest

def test_label_prefix_resume_never_replays_committed_inference(tmp_path):
    m=importlib.import_module('echo_routing.temporal.structured_predictions');j=m.LabelReadout(tmp_path,'fixed',[3,2]);calls=[];j.run_one(0,lambda:(calls.append(0) or np.array([0,1,4])));j=m.LabelReadout(tmp_path,'fixed',[3,2]);assert j.cursor==1
    with pytest.raises(ValueError):j.run_one(0,lambda:calls.append('replay'))
    j.run_one(1,lambda:(calls.append(1) or np.array([2,2])));assert calls==[0,1] and j.load(1).tolist()==[2,2]
    with pytest.raises(ValueError):m.LabelReadout(tmp_path,'changed',[3,2])
    with pytest.raises(ValueError):m.LabelReadout(tmp_path,'fixed',[2,2])

def test_ambiguous_or_corrupted_label_prefix_stops_before_more_decode(tmp_path):
    m=importlib.import_module('echo_routing.temporal.structured_predictions');a=tmp_path/'a';j=m.LabelReadout(a,'x',[2])
    with pytest.raises(ValueError):j.run_one(0,lambda:np.array([1.,2.]))
    with pytest.raises(RuntimeError):m.LabelReadout(a,'x',[2])
    b=tmp_path/'b';j=m.LabelReadout(b,'y',[2]);j.run_one(0,lambda:np.array([0,4],np.int8));j.file(0).write_bytes(b'changed')
    with pytest.raises(ValueError):m.LabelReadout(b,'y',[2])
