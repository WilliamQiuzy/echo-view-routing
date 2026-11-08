"""Hand-built A63 source/signature/probability/draw contracts; no data/NN."""
import importlib
import hashlib
import numpy as np
import pytest

try: m=importlib.import_module('echo_routing.temporal.selected_risk_supply')
except ModuleNotFoundError: m=None

def api(name):
    assert m is not None and callable(getattr(m,name,None)), f'new A63 {name} missing'
    return getattr(m,name)

def row(i,*,D=True,X=False,session=None,H=8,dFP=0,overlap=1.,safe=False,J=0,target=3):
    return {'canonical_id':str(i),'kind':'multi','D':D,'X':X,
      'event_signature':f'e{i}','draw_signature':f'd{i}',
      'responsible_sessions':[session or f's{i}'],'H':0 if safe else H,'R':2 if safe else 0,
      'dFP':0 if safe else dFP,'u':1. if safe else -1.,'short16_lost':0,
      'Jexclusive_short16_recovered':J,'min_overlap':overlap,
      'harm_transitions':[] if safe else [[2,target]],'replacement_class':target}

def test_partition_exact_whole_session_nul_hash_no_redraw():
    f=api('partition_sources'); qs=[{'id':f's{i}_{c}', 'labels':np.array([c]),'refs':[(f's{i}_{c}',0,1.)]} for i in range(402) for c in range(5)]
    out=f(qs); expected=sorted([f's{i}' for i in range(402)],key=lambda s:(hashlib.sha256(b'A63-supply-v1\0'+s.encode()).digest(),s))
    assert out['sessions']==[expected[:201],expected[201:]]
    assert [len(x) for x in out['native_indices']]==[1005,1005]
    assert not set(out['sessions'][0])&set(out['sessions'][1])
    with pytest.raises(ValueError):f([q for q in qs if q['labels'][0]!=4])

def test_signature_ignores_gamma_offset_axis_but_real_harm_source_only():
    f=api('event_signature');refs=[('a_0',7,.9),('b_0',9,1.),('noise_0',0,1.)]
    a=f(refs,[2,3,1],[3,3,1],[2,3,1],new_fp_segments=[])
    b=f([('a_0',7,1.),('b_0',9,.9),('noise_0',0,1.)],[2,3,1],[3,3,1],[2,3,1],new_fp_segments=[])
    assert a['signature']==b['signature'] and a['responsible_sessions']==['a']
    assert a['H']==1 and a['harm_transitions']==[[2,3]]
    c=f(refs,[2,3,1],[2,3,4],[2,3,1],new_fp_segments=[(2,3,4)])
    assert c['signature']!=a['signature'] and c['responsible_sessions']==['noise']

def test_safe_draw_signature_ordered_changed_refs_gamma_invariant():
    f=api('draw_signature')
    assert f([('a_0',2,.9),('b_0',3,1.)],[0,1],[1,1])==f([('a_0',2,1.),('b_0',3,.9)],[0,1],[1,1])
    assert f([('a_0',3,1.),('b_0',3,1.)],[0,1],[1,1])!=f([('a_0',2,1.),('b_0',3,1.)],[0,1],[1,1])

def test_duplicate_invariance_and_true_half_half_responsibility():
    f=api('coverage');rows=[row(i,session='dominant' if i<2 else f's{i}') for i in range(10)]
    rows += [row(100+i,D=False,X=True,session='dominant' if i<4 else f'x{i}') for i in range(10)]
    a=f(rows); b=f(rows+[dict(rows[0],canonical_id=f'copy{i}') for i in range(100)])
    assert a['D']['dangerous_signatures']==b['D']['dangerous_signatures']==10
    assert a['max_p_D']==b['max_p_D']==.2
    assert a['max_p_X']==b['max_p_X']==.4
    assert a['max_p_mix']==b['max_p_mix']==pytest.approx(.3)
    assert not a['passed'] and not b['passed']

def test_cross_membership_union_once_xonly_subtracts_canonical_D():
    f=api('coverage');rows=[row(1,D=True,X=True),row(2,D=False,X=True),dict(row(1),canonical_id='other',D=False,X=True)]
    a=f(rows);assert a['D']['dangerous_signatures']==1 and a['X_only']['dangerous_signatures']==2 and a['union']['dangerous_signatures']==2

def test_all_member_geometry_and_safeJ_conflicts_never_fill_quota():
    f=api('coverage'); a=row(1);b=dict(a,canonical_id='copy',min_overlap=.2)
    c=row(2,safe=True,J=1);d=dict(c,canonical_id='safecopy',Jexclusive_short16_recovered=0)
    out=f([a,b,c,d]);assert out['union']['full_run_signatures']==0 and out['union']['partial_run_signatures']==0
    assert out['D']['safe_J_signatures']==0 and out['D']['safe_J_conflicts']==1
    assert out['union']['geometry_conflicts']==1

def test_signature_responsibility_disagreement_is_audit_error():
    f=api('coverage');a=row(1)
    with pytest.raises(ValueError):f([a,dict(a,canonical_id='copy',responsible_sessions=['other'])])

def test_shortloss_only_kept_negative_target_but_not_safe_useful():
    f=api('coverage');a=row(1,safe=True);a['short16_lost']=1
    out=f([a]);assert out['D']['dangerous_signatures']==0 and out['D']['safe_useful_signatures']==0

def test_matching_draw_replay_and_uniform_signature_then_canonical_member():
    f=api('draws'); pools=[{f'{k}{i}':[100+10*k+i*2,101+10*k+i*2] for i in range(3)} for k in range(4)]
    out=f(np.arange(10),np.arange(100,180),pools,2026110461,steps=6000)
    assert out['C0'].shape==out['C1'].shape==(6000,64)
    assert np.array_equal(out['C0'][:,:48],out['C1'][:,:48])
    assert np.isin(out['C0'][:,:32],np.arange(10)).all()
    for slot,p in enumerate(pools):assert np.isin(out['C1'][:,48+slot*4:52+slot*4],sum(p.values(),[])).all()
    assert np.array_equal(out['C1'],f(np.arange(10),np.arange(100,180),pools,2026110461,steps=6000)['C1'])
    with pytest.raises(ValueError):f(np.arange(10),np.arange(100,180),[{},*pools[1:]],2026110461)

def test_strict_same_top_no_fallback_lexical_tie():
    f=api('top'); keys=[(1,2,(0,)),(0,1,(0,))]
    assert f(keys,[1,1],[0,2],[1,0],[.1,.1])=={'top_index':1,'N_admitted':False}
    assert f(keys,[0,-1],[0,0],[1,1],[.1,.1])=={'top_index':None,'N_admitted':False}
    assert f(keys,[2,1],[1,0],[1,1],[.5,.1])=={'top_index':0,'N_admitted':False}

@pytest.mark.parametrize('bad',[0,999])
def test_selected_members_cannot_escape_mixed_bank(bad):
    f=api('draws');pools=[{'a':[100,101]} for _ in range(4)];pools[0]={'bad':[bad]}
    with pytest.raises(ValueError):f(np.arange(10),np.arange(100,180),pools,2026110461)

@pytest.mark.parametrize('native,mixed', [([0,0],[100]),([0],[0,100]),([.5],[100]),([[0]],[100]),([True],[100]),([-1],[100])])
def test_exact_unique_disjoint_one_dimensional_index_pools(native,mixed):
    f=api('draws')
    with pytest.raises(ValueError):f(native,mixed,[{'a':[100]} for _ in range(4)],2026110461)

def test_unsigned_int64_overflow_is_rejected_before_cast():
    f=api('draws')
    with pytest.raises(ValueError):f(np.asarray([2**63],np.uint64),[100],[{'a':[100]} for _ in range(4)],2026110461)
