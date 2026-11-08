"""Synthetic decoded axes, using the real historical metric/count backend."""
import copy
import importlib
from pathlib import Path
import runpy

import numpy as np
import pytest

from echo_routing.temporal.selected_risk_analysis import endpoint_provenance

_fixture_module=runpy.run_path(str(Path(__file__).with_name('test_selected_risk_analysis.py')))
fixture=_fixture_module['fixture']; MODEL_IDENTITY=_fixture_module['MODEL_IDENTITY']


def module():
    return importlib.import_module('echo_routing.temporal.selected_risk_resume_pairing')


def produce(policy='C1', table='a63_confirmation', half='selection', seed=2026110461):
    qs,bases,values=fixture()
    for v in values:
        v[policy]=copy.deepcopy(v['C1'])
    values=None if policy=='Base' else values
    expected=endpoint_provenance(qs,bases,values,table,half,seed,policy,'a'*64,MODEL_IDENTITY,.89)
    output=module().evaluate_and_prepare(qs,bases,values,table,half,seed,policy,'a'*64,
        MODEL_IDENTITY,.89,expected_provenance=expected)
    return output,expected


def test_actual_metric_pass_produces_row_and_matching_private_pair_bundle():
    out,expected=produce();m=module()
    m.validate_pair_bundle(out['pair_bundle'],out['row'],expected)
    assert out['row']['JEPA_short16_recovered']==0
    assert out['pair_bundle']['binding']['endpoint_provenance']==expected
    assert 'session_confusion' in out['pair_bundle']['system']


@pytest.mark.parametrize('change',['counts','F1','identity','scope','model','prediction','row','missing_expected'])
def test_pair_bundle_or_expected_evidence_substitution_fails_before_bootstrap(change):
    out,expected=produce();m=module();bundle=out['pair_bundle'];row=out['row']
    if change=='counts': bundle['system']['streams']['multi'][0][1][0]=0
    if change=='F1': row['multi_f1']=.1
    if change=='identity': expected['input_identity']='f'*64
    if change=='scope': expected['scope']['half']='calibration'
    if change=='model': expected['model_identity']['model_roles']['C1']='f'*64
    if change=='prediction': expected['intrinsic_identity']['prediction_sha256']='f'*64
    if change=='row': row['JEPA_short16_recovered']=1
    if change=='missing_expected': expected=None
    with pytest.raises(ValueError): m.validate_pair_bundle(bundle,row,expected)


def test_wrong_expected_data_cannot_trigger_the_metric_backend(monkeypatch):
    qs,bases,values=fixture()
    expected=endpoint_provenance(qs,bases,values,'a63_confirmation','selection',2026110461,
                                  'C1','a'*64,MODEL_IDENTITY,.89)
    expected['intrinsic_identity']['data_sha256']='f'*64
    import echo_routing.temporal.event_risk_metrics as backend
    def forbidden(*args,**kwargs): raise AssertionError('unqualified backend called')
    monkeypatch.setattr(backend,'evaluate',forbidden)
    with pytest.raises(ValueError):
        module().evaluate_and_prepare(qs,bases,values,'a63_confirmation','selection',2026110461,
            'C1','a'*64,MODEL_IDENTITY,.89,expected_provenance=expected)


def test_missing_model_expected_field_cannot_become_a_valid_subset():
    out,expected=produce();m=module()
    del expected['model_identity']
    del out['pair_bundle']['binding']['endpoint_provenance']['model_identity']
    with pytest.raises(ValueError): m.validate_pair_bundle(out['pair_bundle'],out['row'],expected)


@pytest.mark.parametrize('change',['flat_input','intrinsic_data','intrinsic_base','J_recovery','dFP','short_loss'])
def test_rehashed_flat_row_must_still_equal_the_producer_qualified_canonical_row(change):
    out,expected=produce();m=module();row=out['row'];bundle=out['pair_bundle']
    if change=='flat_input': row['input_identity']='f'*64
    if change=='intrinsic_data': row['intrinsic_identity']['data_sha256']='f'*64
    if change=='intrinsic_base': row['intrinsic_identity']['base_sha256']='f'*64
    if change=='J_recovery': row['JEPA_short16_recovered']=1
    if change=='dFP': row['dFP16']['native']=2
    if change=='short_loss': row['short16_lost']['multi']=0
    bundle['binding']['normalized_row_sha256']=m._digest(row)
    with pytest.raises(ValueError): m.validate_pair_bundle(bundle,row,expected)
