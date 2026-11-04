import numpy as np
import pytest
import torch
from echo_routing.temporal.proposal_data import prepare_training, validate, construct_confirmation
from echo_routing.temporal.proposal_runtime import infer
from echo_routing.temporal.proposal_training import draw_plan
from echo_routing.temporal.proposal_features import fit_normalizer, fit_target_scales


def panel():
    bs = [np.array([0]*9+[1]*4+[0]*7), np.array([0]*7+[1]*6+[0]*7)]
    aa = [np.eye(5)[b]*.8+.04 for b in bs]
    jj = [np.eye(5)[np.array([0]*8+[2]*4+[0]*8)]*.8+.04 for b in bs]
    qs = [{'id':str(i),'kind':k,'labels':np.array([0]*8+[2]*4+[0]*8),'refs':[(str(i),t,1.) for t in range(20)]} for i,k in enumerate(['native','multi'])]
    return qs,aa,jj,bs


def test_full_candidate_rms_and_canonical_seed_identity():
    qs,aa,jj,bs=panel();d=prepare_training(qs,aa,jj,bs)
    n={a:fit_normalizer(d['features'][a]) for a in ['A','J']}
    s=fit_target_scales(d['targets'])
    plans={str(seed):draw_plan(*(len(d['kind_indices'][k]) for k in ['native','constructed']),seed) for seed in [2026110360,2026110361]}
    assert validate(d,qs,aa,jj,bs,n,s,plans)['passed']
    assert d['targets'].shape[1]==3 and np.array_equal(d['features']['A'][:,:69],d['features']['J'][:,:69])
    for bad in [{},dict(list(plans.items())[:1]),{**plans,'02026110360':plans['2026110360']}]:
        with pytest.raises(ValueError):validate(d,qs,aa,jj,bs,n,s,bad)
    changed=d.copy();changed['targets']=d['targets'].copy();changed['targets'][0,2]+=1
    with pytest.raises(ValueError):validate(changed,qs,aa,jj,bs,n,s,plans)


def test_empty_full_inference_avoids_model_scaler_and_forbidden_fields():
    class Never(torch.nn.Module):
        def forward(self,*args):raise AssertionError('empty domain must not forward')
    p=np.eye(5)[np.zeros(17,int)]*.8+.04
    result=infer(Never(),None,None,p,p,np.zeros(17,int),'A')
    assert result['outputs'].shape==(0,3)
    for policy in ['O','E','R']:
        assert np.array_equal(result[policy]['prediction'],np.zeros(17,int))
        assert result[policy]['accepted'] is None
    assert result['operational_required_producers']==['A08','JEPA']


def test_new_confirmation_domain_and_no_train_footprint():
    natives=[{'id':f'c{i}','kind':'native','refs':[(f'c{i}',t,1.) for t in range(90)],'labels':np.full(90,i%5,int)} for i in range(15)]
    a,ar,summary=construct_confirmation(natives,4,'selection')
    b,br,_=construct_confirmation(natives,4,'selection')
    assert ar==br and summary['RNG_domain']==[2026110360,60,2]
    assert all(q['id'].startswith('a60_') for q in a)
    assert 4<=summary['realized_fragment_minmax'][0]<=summary['realized_fragment_minmax'][1]<=80
    with pytest.raises(ValueError):construct_confirmation(natives,4,'fit')
