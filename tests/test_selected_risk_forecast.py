import importlib
import pytest

def module():
    try:m=importlib.import_module('echo_routing.temporal.selected_risk_forecast')
    except ModuleNotFoundError:m=None
    assert m is not None,'new prospective cost forecast module missing'
    return m

def test_density_projection_preserves_distinct_nonnegative_terms_and_scope():
    m=module();old={'N':10,'T':100,'T2':1000,'C':20,'C2':40,'TC':200,'K':50,'KT':500,'CR':80,'all_C':25,'all_C2':65,'all_TC':250,'all_K':60,'all_KT':600,'all_CR':100}
    out=m.project(old,{'sequences':30,'positions':450,'max_length':30},density=2)
    assert out['N']==30 and out['T']==450 and out['T2']==13500
    for k in ['C','C2','TC','K','KT','CR','all_C','all_C2','all_TC','all_K','all_KT','all_CR']:
        assert out[k]>=old[k]*9
    assert m.SCOPE=={'aux_updates':48000,'formal_updates':24000,'outer_rows':238,'inner_rows':34,'comparisons':140,'replicates':5000,'support':56,'native':34,'regression':32,'fullstates':48}

def test_teacher_bound_counts_payload_temporary_journal_and_both_phases():
    m=module();r=[{'positions':100,'sequences':10,'a08_stride5_windows':20,'jepa_chunks':10}]
    b=m.teacher_bytes(r);assert b>=2*(100*5*8+10*4096) and b>100*5*8
    assert m.teacher_forecast(20,10,100,50,60,100)==pytest.approx(115.)
    with pytest.raises(ValueError):m.teacher_forecast(1,1,0,0,60,0)

def test_complete_remaining_never_drops_training_closure_or_executed_storage():
    m=module();g={'N':1,'T':10,'T2':100,'C':2,'C2':4,'TC':20,'K':3,'KT':30,'CR':8,'all_C':3,'all_C2':9,'all_TC':30,'all_K':4,'all_KT':40,'all_CR':12}
    p={'training':{'utility_update_seconds':.01,'risk_update_seconds':.01,'event_update_seconds':.01,'fullstate_write_read_seconds':.01},'geometry':{'native':g,'mixed':g},'backend_coefficients':{k:.000001 for k in m.TERMS},'CI5000_seconds':.01,'max_training_C':100,'max_training_K':1000}
    roles={k:{'sequences':1,'positions':10,'max_length':10,'a08_stride5_windows':2,'jepa_chunks':1} for k in ['F0_mixed','F1_mixed','a63_confirmation_selection','a63_confirmation_calibration']}
    out=m.remaining(p,roles,old_source_wall=60,old_source_units=100,current_upper=200000000)
    assert out['scope']==m.SCOPE and out['phases']['aux48000_updates']>=480
    assert out['phases']['formal24000_updates']>=240 and out['phases']['resource_OS_independent_audit_publication']>=3600
    assert out['storage_upper_bytes']>=200000000+128000000 and out['remaining_wall_seconds']>=sum(out['phases'].values())+7200

def test_full_auxiliary_and_formal_draw_arrays_are_explicit_storage():
    m=module()
    assert getattr(m,'DRAW_PLAN_BYTES',0)>=12*6000*64*8+6000*64*8,'eight aux plus four formal arrays and one atomic coexistence must fit'

def test_historical_cost_proof_verifies_on_runtime_without_Git(tmp_path,monkeypatch):
    import json,subprocess,hashlib
    from echo_routing.temporal import safety_git
    m=module();repo=tmp_path/'source';repo.mkdir();subprocess.run(['git','init','-q',str(repo)],check=True)
    (repo/'payload.json').write_text('{"frozen":true}\n');oid=subprocess.check_output(['git','-C',str(repo),'hash-object','-w','payload.json']).decode().strip()
    tree=subprocess.check_output(['git','-C',str(repo),'mktree'],input=f'100644 blob {oid}\tpayload.json\n'.encode()).decode().strip()
    commit=subprocess.check_output(['git','-C',str(repo),'-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit-tree',tree],input=b'fixture\n').decode().strip()
    proof=safety_git.build(repo,commit,['payload.json']);runtime=tmp_path/'deployed_no_git';runtime.mkdir();(runtime/'payload.json').write_bytes((repo/'payload.json').read_bytes());raw=(json.dumps(proof,sort_keys=True)+'\n').encode();(runtime/'proof.json').write_bytes(raw)
    monkeypatch.setattr(m,'COST_INPUTS',{'payload.json':hashlib.sha256((runtime/'payload.json').read_bytes()).hexdigest()});monkeypatch.setattr(m,'COST_INPUT_GIT',commit);monkeypatch.setattr(m,'COST_INPUT_PROOF','proof.json',raising=False);monkeypatch.setattr(m,'COST_INPUT_PROOF_SHA',hashlib.sha256(raw).hexdigest(),raising=False)
    monkeypatch.setattr(safety_git,'build',lambda *a,**k:(_ for _ in ()).throw(AssertionError('runtime cannot require historical Git')))
    assert m.verify_cost_inputs(runtime)==proof
