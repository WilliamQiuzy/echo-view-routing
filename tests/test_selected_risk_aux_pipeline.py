import importlib
from pathlib import Path
import numpy as np
import pytest

def api():
    try: return importlib.import_module('echo_routing.temporal.selected_risk_aux_pipeline')
    except ModuleNotFoundError: pytest.fail('independent auxiliary pipeline contracts absent')

def data():
    return {'groups':[np.array([0,1]),np.array([],dtype=np.int64),np.array([2,3,4])],
            'kind_groups':{'native':np.array([0]),'constructed':np.array([2])},
            'kind_indices':{'native':np.array([0,1]),'constructed':np.array([2,3,4])},
            'position_offsets':np.array([0,1,3,6,10,15])}

def plan(role):
    return {'indices':np.zeros((6000,64),dtype=np.int64),'source_kinds':[0,1],
            'role':role,'seed':2026110461,'fallback':False}

def test_full_predraw_retains_all_candidates_positions_and_noop_pairs():
    m=api();v=m.predraw_summary(data(),plan('utility'),'utility')
    assert v['updates']==6000 and v['query_exposures']==384000
    assert v['candidate_exposures']==6000*(32*2+32*3)
    assert v['changed_position_exposures']==6000*(32*3+32*12)
    assert v['max_candidates']==160 and v['trace_raw_byte_upper']>v['variable_JSON_bytes']
    r=m.predraw_summary(data(),plan('risk'),'risk')
    assert r['candidate_exposures']==384000 and r['query_exposures']==0
    assert r['changed_position_exposures']==6000*(32*1+32*3)
    wrong=plan('utility');wrong['indices'][0,0]=1
    with pytest.raises(ValueError):m.predraw_summary(data(),wrong,'utility')

def test_expected_identity_binds_independent_fold_and_all_seven_inputs():
    m=api();parts={k:str(i+1)*64 for i,k in enumerate(['code','config','sources','input','draw_plan','normalizer','target_scales'])}
    x=m.expected_identity(parts,2026110461,0,'utility')
    assert set(x)==set(parts) and x!=m.expected_identity(parts,2026110461,1,'utility')
    assert x!=m.expected_identity(parts,2026110462,0,'utility')
    assert x!=m.expected_identity(parts,2026110461,0,'risk')
    with pytest.raises(ValueError):m.expected_identity({**parts,'input':'not_sha'},2026110461,0,'risk')

def test_archived_or_pending_head_never_reenters_old_updatejournal(tmp_path):
    m=api();p=tmp_path/'head';assert m.guard_head_start(p)
    p.mkdir();(p/'physical_updates.jsonl.gz').write_bytes(b'x')
    with pytest.raises(ValueError):m.guard_head_start(p)
    (p/'physical_updates.jsonl.gz').unlink();(p/'completion.json').write_text('{}')
    with pytest.raises(ValueError):m.guard_head_start(p)
    (p/'completion.json').unlink();(p/'pending.json').write_text('{}')
    with pytest.raises(ValueError):m.guard_head_start(p)

def test_actual_geometry_retains_preG_volume_and_kind_bounds():
    m=api();valid={'native':{'max_C':9,'max_candidate_K':86,'max_T':1081},
                   'mixed':{'max_C':16,'max_candidate_K':383,'max_T':383}}
    assert m.validate_measured_shapes(valid)
    for kind,key in [('native','max_C'),('mixed','max_candidate_K')]:
        bad={k:dict(v) for k,v in valid.items()};bad[kind][key]+=1
        with pytest.raises(ValueError):m.validate_measured_shapes(bad)

def test_only_actual_archive_receipt_or_dangling_marker_forbids_reentry(tmp_path):
    m=api();p=tmp_path/'head';p.mkdir();r=p/'physical_updates.jsonl.archive.json';r.write_text('{}')
    with pytest.raises(ValueError):m.guard_head_start(p)
    r.unlink();r.symlink_to(p/'missing')
    with pytest.raises(ValueError):m.guard_head_start(p)

@pytest.mark.parametrize('offsets',[
    np.array([0,5,4,6,10,15],dtype=np.uint64),
    np.array([0,2**63,2**63+1,2**63+2,2**63+3,2**63+4],dtype=np.uint64),
    np.array([0,np.iinfo(np.int64).max,-2,-1,0,1],dtype=np.int64)])
def test_offset_validation_precedes_signed_diff_or_unsigned_conversion(offsets):
    d=data();d['position_offsets']=offsets
    with pytest.raises(ValueError):api().predraw_summary(d,plan('risk'),'risk')
