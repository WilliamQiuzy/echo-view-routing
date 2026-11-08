"""Complete synthetic 238/140 matrix with the actual paired-count backend."""
import copy
import importlib
from pathlib import Path
import runpy

import pytest

from echo_routing.temporal.selected_risk_metrics import readout_plan

_fixtures=runpy.run_path(str(Path(__file__).with_name('test_selected_risk_resume_pairing.py')))
produce=_fixtures['produce']


def module():
    return importlib.import_module('echo_routing.temporal.selected_risk_resume_aggregate')


@pytest.fixture(scope='module')
def matrix():
    rows=[];bundles={};expected={}
    for spec in readout_plan()['rows']:
        out,pin=produce(spec['policy'],spec['table'],spec['half'],spec['seed'])
        row=out['row'];key=(row['table'],row['half'],row['system'])
        rows.append(row);bundles[key]=out['pair_bundle'];expected[key]=pin
    return rows,bundles,expected


def test_complete_matrix_uses_all140_exact5000_pairs_and_preserves_negative_result(matrix):
    rows,bundles,expected=matrix
    out=module().aggregate(rows,bundles,expected)
    assert len(out['comparisons'])==140
    assert sum(c['primary_family'] for c in out['comparisons'])==8
    assert all(c['bootstrap']['replicates']==5000 for c in out['comparisons'])
    assert all(c['bootstrap']['weight_matrices_generated_once_per_unit_kind'] for c in out['comparisons'])
    assert out['acceptance']['H63_claim_supported'] is False
    assert out['acceptance']['hard_protection_pass'] is False
    assert out['counts']==dict(outer_rows=238,paired_comparisons=140,primary_comparisons=8)


@pytest.mark.parametrize('change',['missing_row','missing_bundle','missing_expected','swapped_bundle',
                                  'changed_count','wrong_input','wrong_expected_model','wrong_F1'])
def test_invalid_matrix_is_rejected_before_any_bootstrap(matrix,change,monkeypatch):
    rows,bundles,expected=copy.deepcopy(matrix)
    first=rows[0];key=(first['table'],first['half'],first['system'])
    if change=='missing_row': rows.pop()
    if change=='missing_bundle': bundles.pop(key)
    if change=='missing_expected': expected.pop(key)
    if change=='swapped_bundle': bundles[key]=next(v for k,v in bundles.items() if k!=key)
    if change=='changed_count': bundles[key]['system']['streams']['multi'][0][1][0]=0
    if change=='wrong_input': first['input_identity']='f'*64
    if change=='wrong_expected_model': expected[key]['model_identity']['model_roles']['A08']='f'*64
    if change=='wrong_F1': first['multi_f1']=.001
    import echo_routing.temporal.event_risk_metrics as backend
    def forbidden(*args,**kwargs): raise AssertionError('bootstrap before full qualification')
    monkeypatch.setattr(backend,'paired',forbidden)
    with pytest.raises(ValueError): module().aggregate(rows,bundles,expected)
