import numpy as np
import pytest
from echo_routing.temporal.proposal_actions import Action


def test_all_eight_policies_keep_original_top_and_strict_veto():
    from echo_routing.temporal.selected_risk_resume_readout import decoded
    b=np.zeros(8,np.int64);actions=[Action(2,3,(1,),False,True),Action(4,5,(2,),False,True)]
    old={'actions':actions,'old_scores':np.array([[2.,.8,.2,0.,.1,.8],[1.,0.,1.,0.,.1,.1]]),'old_q':np.array([[.1,.1],[0.,0.]])}
    v=decoded(b,old,np.zeros((2,2)),dict(L0=1.,M0=1.,R1=1.,R2=1.),dict(C0=1.,C1=1.))
    assert v['E']['accepted_index']==0
    assert all(v[p]['accepted_index'] is None for p in ['N','L0','M0','R1','R2','C0','C1'])


def test_calibration_only_original_admitted_top_and_no_top_none():
    from echo_routing.temporal.selected_risk_resume_readout import calibration_inputs
    b=np.zeros(8,np.int64);a=Action(2,3,(1,),False,True);q={'labels':b.copy()}
    scores=[{'actions':[a],'old_scores':np.array([[1.,.2,.8,0.,.1,.2]]),'q':np.array([[.03,.04]])}]
    rows=calibration_inputs([q],[b],scores,'CAL')
    assert rows['C0'][0]['H']==1 and rows['C0'][0]['q']==.03
    empty=[{'actions':[],'old_scores':np.empty((0,6)),'q':np.empty((0,2))}]
    assert calibration_inputs([q],[b],empty,'CAL')['C0'][0]['q'] is None
    with pytest.raises(ValueError):calibration_inputs([{}],[b],scores,'AUDIT')


def test_all_table_role_mapping_is_prespecified():
    from echo_routing.temporal.selected_risk_resume_readout import role_of
    assert role_of('main','selection')=='selection'
    assert role_of('a58_stress','calibration')=='old_stress_calibration'
    assert role_of('a63_confirmation','selection')=='a63_confirmation_selection'
    with pytest.raises(ValueError):role_of('new_table','selection')


def test_full_context_uses_main_nonmulti_then_new_whole_multi_axes():
    from echo_routing.temporal.selected_risk_resume_readout import full_context
    main=([dict(id='n',kind='native'),dict(id='om',kind='multi'),dict(id='p',kind='pairs')],[1,2,3],[4,5,6],[7,8,9])
    stress=([dict(id='s'+str(i),kind='multi') for i in range(120)],list(range(120)),list(range(120)),list(range(120)))
    old=['nscore','omscore','pscore'];new=['s'+str(i) for i in range(120)]
    bank,values=full_context(main,stress,old,new)
    assert [q['id'] for q in bank[0]][:2]==['n','p'] and len(bank[0])==122
    assert values[:3]==['nscore','pscore','s0'] and bank[1][:2]==[1,3]
    assert main[0][2]['kind']=='pairs'
    with pytest.raises(ValueError):full_context(main,stress[:1]+([0],)+stress[2:],old,new)
    bad=([dict(id='n',kind='multi')]+stress[0][1:],*stress[1:])
    with pytest.raises(ValueError):full_context(main,bad,old,new)


def test_composite_identity_binds_both_physical_input_roles():
    from echo_routing.temporal.selected_risk_resume_readout import input_identity
    bank=([dict(id='private',kind='native')],[np.zeros((1,5))],[np.ones((1,5))],[np.zeros(1,np.int64)])
    a=input_identity({'main':'a'*64,'stress':'b'*64},bank)
    assert a!=input_identity({'main':'c'*64,'stress':'b'*64},bank)
    assert a!=input_identity({'main':'a'*64,'stress':'d'*64},bank)



def _post_risk_module():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'echo_routing/temporal/selected_risk_resume_readout.py'
    spec=importlib.util.spec_from_file_location('private_post_risk_readout',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def _post_bank(role,kinds):
    qs=[];aa=[];jj=[];bb=[];values=[]
    for i,kind in enumerate(kinds):
        b=np.zeros(12,np.int64);j=b.copy();j[2]=1;truth=j.copy() if kind=='multi' else b.copy()
        q=dict(id=role+str(i),kind=kind,labels=truth);a=Action(2,3,(1,),False,True)
        c=np.array([[1.,1. if kind in ['pair','pairs'] else 0.,0.,0.,0.,0.]],np.float64)
        score=1. if kind=='multi' else 0.;other=.75 if kind=='multi' else .25
        v=dict(actions=[a],all_actions=[a],G_rejected=[],old_scores=c,old_q=np.array([[score,other]],np.float64),q=np.array([[score,score]],np.float64))
        qs.append(q);aa.append(np.eye(5)[b]);jj.append(np.eye(5)[j]);bb.append(b);values.append(v)
    return (qs,aa,jj,bb),values


def test_post_risk_zero_one_whole_ties_types_strict_rejection_and_regret():
    m=_post_risk_module();d=object.__new__(m.Readout);bank,values=_post_bank('toy',['native','multi','pairs'])
    before=[(v['old_scores'].copy(),v['old_q'].copy(),v['q'].copy()) for v in values]
    taus=dict(R1=1.,R2=.5,C0=0.,C1=1.)
    r=d._risk_diagnostics(bank,values,role='selection',seed=2026110461,taus=taus)
    assert r['axis_denominators']['axes_pair']==1 and r['axis_denominators']['original_Ntop_axes']==2
    allr=r['R1']['all_G_eligible_candidates'];ntop=r['R1']['original_N_top']
    assert (allr['N'],ntop['N'])==(3,2) and allr['q_zero_count']==2 and allr['q_one_count']==1
    assert allr['BCE'] is None and allr['infinite_BCE_count']==3
    assert allr['tail_rows'][0]['N']==2 and allr['tail_rows'][0]['actual_fraction']==pytest.approx(2/3)
    assert allr['H_and_FP']==2 and allr['H_only']==0 and allr['safe_R']==1 and allr['safe_J16']==1
    assert ntop['safe_useful_rejected']['safe_R']==1 and ntop['strict_tau_counts']['unsafe']==1
    assert allr['by_kind']['pair']['N']==1 and allr['by_kind']['native']['unsafe']==1
    assert r['R1']['gate_coverage']['all']['accepted_N_top']==1
    assert r['C0']['gate_coverage']['all']['accepted_N_top']==0
    assert r['R1']['regret']['all']['after_veto_regret']['mean']>0
    assert r['C0']['regret']['native']['after_veto_regret']['mean']==0
    assert r['training_exposure']['counts'] is None
    for v,old in zip(values,before):
        assert all(np.array_equal(v[k],b) for k,b in zip(['old_scores','old_q','q'],old))


def test_post_risk_14_physical_source_roles_do_not_repeat_main_or_call_inference():
    m=_post_risk_module();banks={};scores={};calls=[]
    roles=[m.role_of(t,h) for t in m.TABLES for h in ['selection','calibration']]
    for role in roles:
        kinds=['native','pairs']+['multi']*120 if role in ['selection','calibration'] else ['multi']*120
        banks[role],scores[role]=_post_bank(role,kinds)
    class Refs:
        def bank(self,role):return banks[role]
    class Inputs:
        def bank(self,role,**kw):return banks[role]
    class Cached:
        def score(self,seed,role,bank,**kw):calls.append((seed,role,kw));return scores[role]
    d=m.Readout(None,Inputs(),None,{})
    d._data_identity=lambda *a,**kw:'synthetic-only'
    captured={};infer=Cached();refs=Refs()
    for t in m.TABLES:
        for h in ['selection','calibration']:d._outer_bank(t,h,refs,infer,readonly=False,risk_sources=captured)
    assert len(calls)==28
    oldtau={s:dict(R1=.5,R2=.5) for s in m.SEEDS}
    thresholds=[dict(seed=s,policy=a,tau=.5) for s in m.SEEDS for a in ['C0','C1']]
    result=d._source_risk_diagnostics(captured,oldtau,thresholds)
    assert len(calls)==28 and result['physical_source_roles']==14 and len(result['records'])==28
    stress=next(r for r in result['records'] if r['role']=='old_stress_selection')
    assert stress['axis_denominators']['axes_multi']==120
    assert stress['axis_denominators']['axes_native']==stress['axis_denominators']['axes_pair']==0
    assert stress['R1']['all_G_eligible_candidates']['by_kind']['native']['N']==0
    assert result['new_NN_forwards']==0 and result['FIT_scope']['score_cache_available'] is None
    assert result['required_risk_scope_complete'] is False
    captured.pop(next(iter(captured)))
    with pytest.raises(ValueError):d._source_risk_diagnostics(captured,oldtau,thresholds)


def test_post_risk_exposure_uses_only_observed_global_prefix_with_duplicates():
    m=_post_risk_module();identity={'fixture':'independently verified by future caller'}
    rows=[dict(u=0,candidate_indices=[0]*32+[1]*32,identity=identity),dict(u=1,candidate_indices=[2]*64,identity=identity)]
    r=m._actual_global_exposure(3,rows,identity)
    assert r['counts'].tolist()==[32,32,64] and r['actual_candidate_exposures']==128
    assert r['actual_updates_observed']==2 and r['full6000'] is False and r['scheduled_future_exposures_included'] is False
    with pytest.raises(ValueError):m._actual_global_exposure(3,[rows[1]],identity)



def _fit_context(tmp_path):
    from echo_routing.temporal.selected_risk_resume_context import Context
    class C(Context):
        def revalidate(self):return True
    ctx=object.__new__(C);ctx.root=tmp_path.resolve();return ctx


def test_fit_risk_maps_global_hashes_DX_offsets_and_exact_native_post_scores():
    from echo_routing.temporal.selected_risk_supply import digest as candidate_digest
    m=_post_risk_module();a=Action(2,3,(1,),False,True);b=np.zeros(12,np.int64)
    q1=dict(id='native',kind='native',refs=[('native',0)],labels=b.copy());q2=dict(id='mixed',kind='multi',refs=[('mixed',0)],labels=b.copy());q3=dict(id='empty',kind='native',refs=[('empty',0)],labels=b.copy())
    c=np.array([[1.,0.,0.,1.,.1,.1]],np.float64)
    def value(actions,D,X):
        dc=c.copy() if actions else np.empty((0,6));xc=c.copy() if actions else np.empty((0,6))
        if actions and not D:dc[:,4]=.9;dc[:,5]=.9
        if actions and not X:xc[:,4]=.9;xc[:,5]=.9
        return dict(actions=actions,all_actions=actions,G_rejected=[],D=dict(top_index=0 if actions else None,N_admitted=D),X=dict(top_index=0 if actions else None,N_admitted=X),D_C6=dc,X_C6=xc,x109=np.zeros((len(actions),109)),context96=np.zeros((len(actions),96)))
    rows=[(q1,b,value([a],True,False)),(q2,b,value([a],False,True)),(q3,b,value([],False,False))]
    ids=np.asarray([bytes.fromhex(candidate_digest([q['id'],a.key])) for q in [q1,q2]],dtype='V32');data={'candidate_digests':ids,'unsafe':np.ones(2,np.float32)}
    maps=m._fit_candidate_map(rows,data)
    assert maps['axis_offsets'].tolist()==[0,1,2,2] and maps['D'].tolist()==[1,0] and maps['X'].tolist()==[0,1]
    empty_id=m._fit_axis_identity('a'*64,2,q3,b,rows[2][2],ids[2:])
    assert empty_id!=m._fit_axis_identity('a'*64,2,{**q3,'id':'another-empty'},b,rows[2][2],ids[2:])
    old={'groups':[np.array([1],np.int64),np.empty(0,np.int64)]};post={'C6':np.vstack([c,c]),'q':np.array([[.9,.8],[.125,.25]],np.float64)}
    native=m._fit_native_q([q1,q3],rows,old,post)
    assert np.array_equal(native[0],post['q'][[1]]) and native[1] is None and native[2].shape==(0,2)
    q4=[np.array([[.125,.25,.5,.75]],np.float64),np.full((1,4),.5,np.float64),np.empty((0,4),np.float64)]
    cached=[{**v,'old_scores':v['D_C6'],'old_q':score[:,:2],'q':score[:,2:]} for (q,b,v),score in zip(rows,q4)]
    _,flat=object.__new__(m.Readout)._risk_diagnostics(([q for q,b,v in rows],[b for q,b,v in rows]),cached,role='FIT_all',seed=2026110461,taus=dict(R1=1.,R2=1.,C0=1.,C1=1.),return_candidates=True)
    assert [v['unsafe'] for v in flat]==[True,True] and [v['Ntop'] for v in flat]==maps['D'].astype(bool).tolist()
    from echo_routing.temporal.selected_risk_resume_storage import storage_bound
    table={'identity':'a'*64,**maps,'C0_exposures':np.zeros(2,np.int64),'C1_exposures':np.zeros(2,np.int64)}
    assert storage_bound(table)['artifact_bytes']<=18*2+8*4+4096
    with pytest.raises(ValueError):m._fit_candidate_map(rows,{**data,'candidate_digests':ids[::-1]})
    post['C6'][1,0]=2.
    with pytest.raises(ValueError):m._fit_native_q([q1,q3],rows,old,post)


def test_fit_risk_axis_empty_compact_limits_and_completed_restart(tmp_path):
    m=_post_risk_module();ctx=_fit_context(tmp_path);calls=[]
    def predict():calls.append(1);return np.array([[.125,.25,.5,.75]],np.float64),{'calls':2,'candidate_outputs':2,'CPU_seconds':0.}
    q,r=m._fit_score_axis(ctx,'FIT_risk/seed1',0,1,'a'*64,2,predict)
    assert calls==[1] and r['new_calls']==2 and r['score_bytes']-32<=640 and r['receipt_bytes']<=512
    stored=ctx.load_value(r['score_pin']);assert set(stored)=={'identity','q'} and stored['identity']=='a'*64 and np.array_equal(stored['q'],q)
    def forbidden():raise AssertionError('completed or empty axis repeated a forward')
    again,reused=m._fit_score_axis(ctx,'FIT_risk/seed1',0,1,'a'*64,2,forbidden)
    assert reused['new_calls']==0 and np.array_equal(again,q)
    empty,er=m._fit_score_axis(ctx,'FIT_risk/seed1',1,0,'b'*64,0,forbidden)
    assert empty.shape==(0,4) and er['new_calls']==0 and er['score_pin'] is None
    assert not ctx.output('FIT_risk/seed1/axes/00001.npz').exists()


def test_fit_risk_pending_never_replays_and_public_limit_is_hard(tmp_path):
    m=_post_risk_module();ctx=_fit_context(tmp_path);calls=[]
    def crash():calls.append(1);raise RuntimeError('synthetic forward failure, no real NN')
    with pytest.raises(RuntimeError):m._fit_score_axis(ctx,'FIT_risk/seed2',0,1,'c'*64,4,crash)
    assert len(ctx.output('FIT_risk/seed2/pending.json').read_bytes())<=512
    with pytest.raises(ValueError):m._fit_score_axis(ctx,'FIT_risk/seed2',0,1,'c'*64,4,crash)
    assert calls==[1]
    class NoWrite:
        def write_json(self,*a,**k):raise AssertionError('oversized public aggregate reached writer')
    with pytest.raises(ValueError):m._write_fit_aggregate(NoWrite(),{'oversize':'x'*2_000_001})
