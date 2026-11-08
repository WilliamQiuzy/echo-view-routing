"""Hand-counted existing predictions only; no data access, fitting or forwards."""
import copy
import importlib

import numpy as np
import pytest

from echo_routing.temporal.proposal_actions import canonical, apply
from echo_routing.temporal.selected_risk_metrics import _validate_row, system

MODEL_IDENTITY = dict(producer_manifest_sha256='c'*64, head_seal_sha256='d'*64,
                      policy_manifest_sha256='e'*64,
                      model_roles={'A08':'1'*64,'JEPA':'2'*64,'S-L':'3'*64,'N':'4'*64,'C1':'5'*64})


def module():
    return importlib.import_module('echo_routing.temporal.selected_risk_analysis')


def fixture():
    qs = [dict(id='native-fixture', kind='native', labels=np.zeros(30, np.int64)),
          dict(id='multi-fixture', kind='multi', labels=np.array([0]*9+[1]*2+[2]*20)),
          dict(id='pair-fixture', kind='pairs', labels=np.array([3]*4+[4]*20))]
    for q in qs:
        q['refs'] = [(q['id']+'-source', i, 1.) for i in range(len(q['labels']))]
        q['joins'] = [] if q['kind']=='native' else [9,11] if q['kind']=='multi' else [4]
    bases = [np.zeros(30, np.int64), np.array([0]*11+[2]*20), np.array([4]*24)]
    actions = [canonical(bases[0], 0, 10, [1]*10, old=True),
               canonical(bases[1], 0, 11, [1]*11, jepa=True),
               canonical(bases[2], 0, 4, [3]*4, jepa=True)]
    values = [dict(actions=[a], C1=dict(accepted_index=0, prediction=apply(b, a)))
              for b, a in zip(bases, actions)]
    return qs, bases, values


def qualified(policy='C1', table='a63_confirmation', half='selection', seed=2026110461,
              identity='a'*64):
    qs,bases,values=fixture()
    return module().qualified_evaluate_endpoint(qs,bases,None if policy=='Base' else values,
        table,half,seed,policy,identity,MODEL_IDENTITY,.89)


def expected_provenance(table='a63_confirmation', half='selection', seed=2026110461, policy='C1', identity='a'*64):
    qs,bases,values=fixture()
    return module().endpoint_provenance(qs,bases,None if policy=='Base' else values,
        table,half,seed,policy,identity,MODEL_IDENTITY,.89)


def normalize(table,half,seed,policy,identity,rich_metrics,summary,**kwargs):
    # Expected evidence is rebuilt from fixture input/model pins, never from rich/summary.
    expected=expected_provenance(table,half,None if policy=='Base' else 2026110461,
                                 policy,identity if identity in ['a'*64,'b'*64] else 'a'*64)
    return module().normalize_endpoint(table,half,seed,policy,identity,rich_metrics,summary,
        expected_provenance=kwargs.pop('expected_provenance',expected),**kwargs)


def test_hand_counted_gross_harm_repairs_short_loss_and_pair_alias():
    qs, bases, values = fixture(); result = module().safety_summary(qs, bases, values, 'C1', .89)
    assert result['H'] == dict(native=10, multi=9, pair=0)
    assert result['R'] == dict(native=0, multi=2, pair=4)
    assert result['dE'] == dict(native=10, multi=7, pair=-4)
    assert result['dFP16'] == dict(native=1, multi=1, pair=0)
    assert result['short16_lost'] == dict(native=0, multi=1, pair=0)
    assert result['short16_recovered'] == dict(native=0, multi=0, pair=1)
    assert result['H_by_true_class']['multi'] == [9,0,0,0,0]
    assert result['class_H'] == [19,0,0,0,0]
    assert result['JEPA_short16_recovered'] == 0
    assert result['JEPA_short16_recovered_by_kind'] == dict(native=0,multi=0,pair=1)
    assert result['native_confusion'] == [[1,0,0,0,0]]+[[0]*5 for _ in range(4)]
    assert result['native_vote_changes'] == 0
    assert result['axis_counts'] == dict(native=1,multi=1,pair=1)
    assert result['reference_positions'] == dict(native=30,multi=31,pair=24)
    assert qs[2]['kind'] == 'pairs'


def test_Base_noop_and_rejected_top_do_not_become_fewer_cut_success():
    qs, bases, values = fixture(); m = module()
    base = m.safety_summary(qs, bases, None, 'Base', .89)
    for row, b in zip(values, bases): row['C1'] = dict(accepted_index=None, prediction=b.copy())
    noop = m.safety_summary(qs, bases, values, 'C1', .89)
    for s in (base, noop):
        assert all(v == 0 for name in ('H','R','dE','dFP16','short16_lost') for v in s[name].values())
        assert s['JEPA_short16_recovered'] == 0 and s['accepted_actions'] == 0
        assert 'goal_met' not in s
    assert base['native_confusion'] == noop['native_confusion']


def test_native_vote_change_and_lower_class_majority_tie_are_measured():
    qs = [dict(id='n', kind='native', labels=np.array([4]*5),
               refs=[('native-source',i,1.) for i in range(5)], joins=[])]
    bases = [np.array([3,3,3,4,4])]; a = canonical(bases[0],0,3,[4,4,4],jepa=True)
    values = [dict(actions=[a], C1=dict(prediction=apply(bases[0],a),accepted_index=0))]
    s = module().safety_summary(qs,bases,values,'C1',.89)
    assert s['native_confusion'][4][4] == 1 and s['native_vote_changes'] == 1 and s['R']['native'] == 3
    qs[0]['labels'] = np.array([4]*4); bases[0] = np.array([3,3,4,4])
    qs[0]['refs'] = qs[0]['refs'][:4]
    s = module().safety_summary(qs,bases,None,'Base',.89)
    assert s['native_confusion'][4][3] == 1


@pytest.mark.parametrize('change', ['unknown_kind','duplicate_id','shape','float_labels','nonconstant_native',
                                   'missing_values','bool_index','negative_index','large_index','prediction_mismatch','noop_mismatch','unknown_policy'])
def test_invalid_or_unaligned_readout_cannot_silently_change_the_axis(change):
    qs,bases,values=fixture(); policy='C1'
    if change=='unknown_kind': qs[2]['kind']='mixed'
    if change=='duplicate_id': qs[2]['id']=qs[0]['id']
    if change=='shape': bases[1]=bases[1][:-1]
    if change=='float_labels': qs[0]['labels']=qs[0]['labels'].astype(float)
    if change=='nonconstant_native': qs[0]['labels'][0]=1
    if change=='missing_values': values=values[:-1]
    if change=='bool_index': values[0]['C1']['accepted_index']=False
    if change=='negative_index': values[0]['C1']['accepted_index']=-1
    if change=='large_index': values[0]['C1']['accepted_index']=1
    if change=='prediction_mismatch': values[0]['C1']['prediction']=bases[0]
    if change=='noop_mismatch': values[0]['C1']['accepted_index']=None
    if change=='unknown_policy': policy='C2'
    with pytest.raises(ValueError): module().safety_summary(qs,bases,values,policy,.89)


def test_normalized_endpoint_preserves_qualified_F1_and_declared_identity():
    r,s,_=qualified()
    row=normalize('a63_confirmation','selection',2026110461,'C1','a'*64,r,s)
    _validate_row(row)
    assert row['multi_f1'] == r['segment_f1_50_multi']
    assert row['system'] == system(2026110461,'C1') and row['step']==6000
    assert row['class_H']==[19,0,0,0,0] and row['H']['multi']==9 and row['short16_lost']['multi']==1
    assert row['role']=='policy' and row['input_identity']=='a'*64


@pytest.mark.parametrize('change',['confusion','gross','Hsum','position','bad_F1','wrong_seed','wrong_identity'])
def test_normalization_rejects_inconsistent_rich_metrics_and_aggregate_schema(change):
    r,s,_=qualified(); seed=2026110461; identity='a'*64
    if change=='confusion': r['native_confusion'][0][0]=2
    if change=='gross': r['gross_reference_position_effects']['multi']['H']=0
    if change=='Hsum': s['class_H'][0]=0
    if change=='position': r['decoded_position_confusion']['multi'][0][1]=0
    if change=='bad_F1': r['segment_f1_50_multi']=float('nan')
    if change=='wrong_seed': seed=True
    if change=='wrong_identity': identity='oops'
    with pytest.raises(ValueError): normalize('a63_confirmation','selection',seed,'C1',identity,r,s)


def test_Base_endpoint_is_single_seedless_reference_and_input_arrays_are_unchanged():
    qs,bases,values=fixture(); before=[b.copy() for b in bases]
    r,s,_=qualified('Base','main',seed=None,identity='b'*64)
    row=normalize('main','selection',None,'Base','b'*64,r,s)
    _validate_row(row)
    assert row['role']=='reference' and row['seed'] is None and 'step' not in row
    assert all(np.array_equal(a,b) for a,b in zip(before,bases))
    with pytest.raises(ValueError): normalize('main','selection',2026110461,'Base','b'*64,r,s)


@pytest.mark.parametrize('change',['gross_partition','accepted','J_exceeds_recovery','bool_rich_count'])
def test_effect_conservation_and_J_recovery_bounds_cannot_be_forged(change):
    r,s,_=qualified()
    if change=='gross_partition': s['changed_positions']['multi']=1
    if change=='accepted': s['accepted_actions']=0
    if change=='J_exceeds_recovery': s['JEPA_short16_recovered']=2
    if change=='bool_rich_count': r['gross_reference_position_effects']['pair']['H']=False
    with pytest.raises(ValueError): normalize('a63_confirmation','selection',2026110461,'C1','a'*64,r,s)


def test_unbound_opaque_F1_cannot_be_promoted_by_normalization():
    s=module().safety_summary(*fixture(),'C1',.89)
    r=dict(segment_f1_50_multi=.5,native_confusion=s['native_confusion'],
           decoded_position_confusion=s['position_confusion'],
           gross_reference_position_effects={k:{'H':s['H'][k],'R':s['R'][k],
             'dE':s['dE'][k],'H_by_true_class':s['H_by_true_class'][k],
             'reference_positions':s['reference_positions'][k]} for k in ('native','multi','pair')})
    with pytest.raises(ValueError,match='binding|provenance'):
        normalize('a63_confirmation','selection',2026110461,'C1','a'*64,r,s)


@pytest.mark.parametrize('change',['refs','joins','base','action_flags','accepted_index'])
def test_intrinsic_binding_distinguishes_same_labels_and_aggregate_counts(change):
    qs,bases,values=fixture();m=module()
    old=m.safety_summary(qs,bases,values,'C1',.89)['intrinsic_identity']
    if change=='refs': qs[1]['refs'][0]=('another-source',0,1.)
    if change=='joins': qs[1]['joins']=[8,11]
    if change=='base': bases[2][5]=3;values[2]['C1']['prediction']=apply(bases[2],values[2]['actions'][0])
    if change=='action_flags':
        from dataclasses import replace
        values[0]['actions'][0]=replace(values[0]['actions'][0],jepa=True)
    if change=='accepted_index':
        a=canonical(bases[2],1,4,[3]*3,jepa=True)
        values[2]['actions'].insert(0,a);values[2]['C1']['accepted_index']=1
    new=m.safety_summary(qs,bases,values,'C1',.89)['intrinsic_identity']
    assert old!=new
    if change in ['refs','joins']: assert old['data_sha256']!=new['data_sha256']
    if change=='base': assert old['base_sha256']!=new['base_sha256']
    if change in ['action_flags','accepted_index']: assert old['prediction_sha256']!=new['prediction_sha256']


@pytest.mark.parametrize('change',['rich_scope','summary_scope','external_input','data','prediction',
                                  'model','evaluator','F1','summary_effect','missing_rich_binding'])
def test_normalization_rejects_endpoint_provenance_or_payload_substitution(change):
    r,s,_=qualified();m=module()
    if change=='rich_scope': r['endpoint_binding']['scope']['half']='calibration'
    if change=='summary_scope': s['endpoint_binding']['scope']['policy']='C0'
    if change=='external_input': r['endpoint_binding']['input_identity']='f'*64
    if change=='data': s['intrinsic_identity']['data_sha256']='f'*64
    if change=='prediction': s['intrinsic_identity']['prediction_sha256']='f'*64
    if change=='model': r['endpoint_binding']['model_identity']['head_seal_sha256']='f'*64
    if change=='evaluator': r['endpoint_binding']['evaluator_identity']={}
    if change=='F1': r['segment_f1_50_multi']=.123
    if change=='summary_effect': s['fit_f1']=.88
    if change=='missing_rich_binding': del r['endpoint_binding']
    with pytest.raises(ValueError): normalize('a63_confirmation','selection',2026110461,'C1','a'*64,r,s)


def test_external_manifest_SHA_is_kept_separate_from_intrinsic_data_digest():
    r,s,_=qualified(identity='a'*64)
    row=normalize('a63_confirmation','selection',2026110461,'C1','a'*64,r,s)
    assert row['input_identity']=='a'*64
    assert row['intrinsic_identity']['data_sha256']!='a'*64
    assert row['endpoint_binding']==r['endpoint_binding']==s['endpoint_binding']


@pytest.mark.parametrize('change',['model','data','prediction','evaluator','scope'])
def test_internally_consistent_payload_pair_still_must_match_independent_expected_pins(change):
    r,s,_=qualified(); expected=expected_provenance()
    # Both incoming copies agree: requested evidence must still reject substitution.
    for obj in [r,s]:
        b=obj['endpoint_binding']
        if change=='model': b['model_identity']['head_seal_sha256']='f'*64
        if change=='data': b['intrinsic_identity']['data_sha256']='f'*64
        if change=='prediction': b['intrinsic_identity']['prediction_sha256']='f'*64
        if change=='evaluator': b['evaluator_identity']={}
        if change=='scope': b['scope']['half']='calibration'
    with pytest.raises(ValueError,match='provenance|binding'):
        normalize('a63_confirmation','selection',2026110461,'C1','a'*64,r,s,expected_provenance=expected)


def test_absent_expected_evidence_cannot_default_to_incoming_payload():
    r,s,_=qualified()
    with pytest.raises(ValueError,match='expected|provenance|binding'):
        module().normalize_endpoint('a63_confirmation','selection',2026110461,'C1','a'*64,r,s)


def test_qualified_provenance_retains_exact_model_role_to_SHA_mapping():
    model=copy.deepcopy(MODEL_IDENTITY)
    model['model_roles']={'A08':'1'*64,'JEPA':'2'*64,'S-L':'3'*64,'N':'4'*64,'C1':'5'*64}
    p=module().endpoint_provenance(*fixture(),'a63_confirmation','selection',2026110461,
                                    'C1','a'*64,model,.89)
    assert p['model_identity']['model_roles']==model['model_roles']
    model['model_roles']['C1']='not-a-sha'
    with pytest.raises(ValueError):
        module().endpoint_provenance(*fixture(),'a63_confirmation','selection',2026110461,
                                     'C1','a'*64,model,.89)
