import importlib.util
import json
import numpy as np
import pytest


def live():
    spec=importlib.util.find_spec('echo_routing.temporal.pw_transfer_live')
    assert spec is not None, 'durable score and exact-resume protections not implemented'
    from echo_routing.temporal import pw_transfer_live
    return pw_transfer_live


def test_resume_refuses_uncheckpointed_or_duplicate_updates():
    m=live()
    assert m.validate_journal([{'step':1},{'step':2}],2)==2
    for log,step in [([{'step':1},{'step':2}],1),([{'step':1},{'step':1}],2),([],1)]:
        with pytest.raises(ValueError):m.validate_journal(log,step)
    with pytest.raises(ValueError):m.validate_journal([{'step':1}],1,inflight=True)


def test_raw_score_cell_cannot_be_inferred_twice_or_rebound(tmp_path):
    m=live();identity={'model':'abc','input':'def','fit_fold':1,'heldout_fold':0}
    assert m.claim_cell(tmp_path,'G_f0',identity)
    with pytest.raises(RuntimeError):m.claim_cell(tmp_path,'G_f0',identity)
    m.finish_cell(tmp_path,'G_f0',identity,np.array([3,6]),np.array([-.1,.5],np.float32))
    assert m.claim_cell(tmp_path,'G_f0',identity) is False
    rows,scores=m.load_cell(tmp_path,'G_f0',identity)
    assert rows.tolist()==[3,6] and scores.tolist()==pytest.approx([-.1,.5])
    with pytest.raises(ValueError):m.load_cell(tmp_path,'G_f0',{**identity,'model':'changed'})
    (tmp_path/'G_f0.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):m.claim_cell(tmp_path,'G_f0',identity)


def test_independent_membership_mean_and_one_way_zero_threshold():
    m=live()
    pk={'base':np.array([2,2,1]),'offsets':np.array([0,2,3,4]),'rows':np.array([10,12,11,12]),'weights':np.array([.25,.75,1.,1.])}
    means,p=m.reference_labels(pk,np.array([0,1,2]),np.array([-3.,0.,1.]),10)
    assert means[:2].tolist()==[0.,0.] and p.tolist()==[2,2,1]
    means,p=m.reference_labels(pk,np.array([0,1,2]),np.array([-3.,.1,2.]),10)
    assert means[:2].tolist()==pytest.approx([.75,.1]) and p.tolist()==[3,3,1]
    with pytest.raises(ValueError):m.reference_labels(pk,np.array([0,1]),np.array([-3.,0.]),10)
