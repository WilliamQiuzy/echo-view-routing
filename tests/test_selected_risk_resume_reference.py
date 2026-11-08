"""Pure/read-only synthetic reference fixtures; no private data or model execution."""
import importlib
import io
import json
import hashlib
import numpy as np
import pytest


def module():return importlib.import_module('echo_routing.temporal.selected_risk_resume_reference')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def record(role='CAL',seed=2026110461):
    return dict(role=role,seed=seed,path=f'runs/a62_anchor_context_risk_v1/q_scores/seed{seed}/{role}',identity='a'*64,journal_sha256='b'*64,sequences=469 if role=='CAL' else 459 if role=='AUDIT' else 120,candidate_vectors=0)


def test_old_q_record_duplicates_are_receipt_reuse_not_extra_cells():
    m=module();r=record();s=dict(r,new_event_MLP_forwards_this_process=0)
    assert m.cache_records([r,s])[2026110461,'CAL']==r
    s['journal_sha256']='c'*64
    with pytest.raises(ValueError):m.cache_records([r,s])
    with pytest.raises(ValueError):m.cache_records([record('a63_confirmation_selection')])


def test_original_CAL_mapping_is_319_native_not_new_FIT_201():
    m=module();qs=[dict(kind='native',refs=[['x',i]],labels=[0]) for i in range(319)]+[dict(kind='multi',refs=[['m',i]],labels=[0]) for i in range(150)]
    original=[dict(q) for q in qs[:319]]+[dict(kind='native',refs=[['other',i]],labels=[0]) for i in range(309)]+qs[319:]+qs[319:]
    mapping={'native_original_inner_indices':{str(i):i for i in range(319)},'new_mixed_role':'CAL_mixed'}
    assert m.native_mapping('CAL',qs,original,mapping)==dict(enumerate(range(319)))
    bad=dict(mapping,native_original_inner_indices={str(i):i for i in range(201)})
    with pytest.raises(ValueError):m.native_mapping('CAL',qs,original,bad)
    mapping['native_original_inner_indices']['0']=319
    with pytest.raises(ValueError):m.native_mapping('CAL',qs,original,mapping)


def qfixture():
    m=module();base=np.array([0,0]);old=np.empty((0,6),np.float64);q=np.empty((0,2),np.float64)
    ref={'kind':'readonly_exact_fixed_C6_payload','path':'runs/a61_action_safety_ranking_v1/x/sequence000000.private.npz','payload_sha256':'c'*64,'journal_sha256':'d'*64,'scores_sha256':sha(old.tobytes())}
    stream=io.BytesIO();np.savez(stream,old_scores=old,q=q,actions_json=json.dumps([]),old_reference_json=json.dumps(ref),taus_json='null',seconds=0.)
    return base,ref,stream.getvalue()


def test_q_payload_requires_public_journal_hash_canonical_actions_and_C6_source():
    m=module();base,ref,raw=qfixture()
    value=m.parse_q_payload(raw,base,base,base,ref)
    assert value['old_scores'].shape==(0,6) and value['old_q'].shape==(0,2)
    with pytest.raises(ValueError):m.parse_q_payload(raw,base,base,base,dict(ref,scores_sha256='e'*64))
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:payload={k:z[k] for k in z.files}
    payload['actions_json']=np.array('[[0,1,[1],true,false]]');stream=io.BytesIO();np.savez(stream,**payload)
    with pytest.raises(ValueError):m.parse_q_payload(stream.getvalue(),base,base,base,ref)


def test_q_journal_requires_exact_committed_axis_and_public_SHA():
    m=module();r=record('selection');r['sequences']=1
    row={'index':0,'identity':r['identity'],'positions':2,'sha256':'c'*64};raw=(json.dumps(row)+'\n').encode();r['journal_sha256']=sha(raw)
    assert m.committed_rows(raw,r,[2])==[row]
    with pytest.raises(ValueError):m.committed_rows(raw,r,[3])
    with pytest.raises(ValueError):m.committed_rows(raw+b' ',r,[2])
    with pytest.raises(ValueError):m.committed_rows(raw+raw,dict(r,journal_sha256=sha(raw+raw)),[2])


def test_new_confirmation_never_falls_back_to_readonly_old_reference():
    m=module();bank=object.__new__(m.ReferenceBank)
    with pytest.raises(ValueError):bank.bank('a63_confirmation_selection')
    with pytest.raises(ValueError):bank.scores(2026110461,'a63_confirmation_selection',())


def test_module_has_no_model_or_writing_cli_dependency():
    from pathlib import Path
    text=Path(module().__file__).read_text()
    assert 'scripts.event_risk_execution' not in text and 'torch' not in text
    assert '.run_one(' not in text and 'allow_pickle=True' in text


def test_CAL_native_cannot_substitute_mixed_physical_C6_even_empty_actions():
    m=module();obj=object.__new__(m.ReferenceBank);obj.mapping={'CAL':{'native_original_inner_indices':{'0':12}}}
    obj._banks={'CAL':([{'kind':'native'}],[],[],[])}
    class NoRead:
        def raw(self,*args):raise AssertionError('wrong native reference reached private IO')
    obj.ctx=NoRead();obj._raw={}
    ref=dict(kind='readonly_exact_fixed_C6_payload',path='runs/a62_anchor_context_risk_v1/fixed_scores/seed2026110461/CAL_mixed/sequence000000.private.npz',payload_sha256='a'*64,journal_sha256='b'*64,scores_sha256=sha(b''))
    with pytest.raises(ValueError):obj._C6(ref,2026110461,'CAL',0,[])
