import numpy as np
import pytest


def test_geometry_never_reads_truth_and_keeps_empty_axes():
    from echo_routing.temporal.selected_risk_resume_S import observable_geometry
    class Bomb:
        def __array__(self,*a):raise AssertionError('geometry read truth')
    q={'kind':'multi','refs':[('session_a_1',i,1.) for i in range(8)],'labels':Bomb()}
    p=np.eye(5)[np.zeros(8,dtype=int)]
    out=observable_geometry([q],[p],[p],[np.zeros(8,dtype=int)])
    assert out['axes']==1 and out['private_candidate_K_per_axis']==[[]]
    assert out['geometry']['C']==0 and out['truth_or_head_outcomes_used'] is False


def test_observable_predraw_uses_full_queries_and_both_kinds():
    from echo_routing.temporal.selected_risk_resume_S import layout_data
    d=layout_data({'F0_native':[[2,3],[]],'F0_mixed':[[7],[1,2,4]]},0)
    assert np.array_equal(d['position_offsets'],[0,2,5,12,13,15,19])
    assert [g.tolist() for g in d['groups']]==[[0,1],[],[2],[3,4,5]]
    assert d['kind_groups']['native'].tolist()==[0]
    assert d['kind_indices']['constructed'].tolist()==[2,3,4,5]


def test_completed_head_is_verified_and_never_reenters_loop(tmp_path,monkeypatch):
    from echo_routing.temporal.selected_risk_resume_S import completed_head
    import hashlib,json,gzip
    f=tmp_path/'head';f.mkdir();trace=f/'physical_updates.jsonl.gz';identity={'code':'a'*64}
    raw=b''.join(json.dumps({'u':i,'identity':identity}).encode()+b'\n' for i in range(6000));trace.write_bytes(gzip.compress(raw))
    ar={'identity':identity,'archive_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'uncompressed_bytes':len(raw)}
    (f/'physical_updates.jsonl.archive.json').write_text(json.dumps(ar))
    row={'identity':identity,'physical_updates':6000,'fullstate_count':4,'raw_trace_sha256':ar['uncompressed_sha256'],'state_pins':[{'path':str(f/str(k)),'sha256':'c'*64} for k in range(4)],'passed':True}
    (f/'completion.json').write_text(json.dumps(row))
    with pytest.raises(ValueError):completed_head(f,{'code':'b'*64},lambda p:None)
    assert completed_head(f,{'code':'a'*64},lambda p:None)==row
    (f/'pending.json').write_text('{}')
    with pytest.raises(ValueError):completed_head(f,row['identity'],lambda p:None)


def test_aux_training_driver_calls_real_loop_audit_then_archive_in_order(monkeypatch,tmp_path):
    # No Torch model is instantiated. This checks the driver protocol against
    # stubs at its three qualified boundaries, not any optimizer/runtime result.
    from echo_routing.temporal.selected_risk_resume_S import finish_one_head
    calls=[]
    monkeypatch.setattr('echo_routing.temporal.selected_risk_resume_S._train',lambda *a,**k:calls.append('real-loop-boundary') or {'committed_physical_updates':6000})
    monkeypatch.setattr('echo_routing.temporal.selected_risk_resume_S._audit_head',lambda *a,**k:calls.append('independent-trace-state-audit') or {'passed':True,'physical_updates':6000,'fullstate_count':4,'state_pins':[]})
    monkeypatch.setattr('echo_routing.temporal.selected_risk_resume_S._archive',lambda *a,**k:calls.append('archive') or {'archive_sha256':'a'*64})
    class C:
        def write_json(self,path,v):calls.append('immutable-completion');return {'path':path,'sha256':'b'*64}
    result=finish_one_head(C(),None,None,{},None,{'code':'a'*64},tmp_path,{},'head')
    assert calls==['real-loop-boundary','independent-trace-state-audit','immutable-completion','archive']
    assert result['audit']['passed']
