"""CAL/AUDIT synthetic decoded predictions, no model or calibration fitting."""
import copy
import builtins
import ast
import inspect
import textwrap
import importlib
from pathlib import Path
import runpy

import pytest

_fixture=runpy.run_path(str(Path(__file__).with_name('test_selected_risk_analysis.py')))
MODEL_IDENTITY=_fixture['MODEL_IDENTITY']


def module(): return importlib.import_module('echo_routing.temporal.selected_risk_resume_inner')


def args(policy='C1',seed=2026110461,population='CAL'):
    q,b,v=_fixture['fixture']();q=q[:2];b=b[:2];v=v[:2]
    for row in v: row[policy]=copy.deepcopy(row['C1'])
    return q,b,None if policy=='Base' else v,population,seed,policy,'a'*64,MODEL_IDENTITY,.89


@pytest.mark.parametrize('population',['CAL','AUDIT'])
def test_qualified_inner_preserves_absent_pair_and_original_metrics(population):
    a=args(population=population);m=module();expected=m.inner_provenance(*a)
    out=m.evaluate_inner(*a,expected_provenance=expected)
    r=out['record']
    assert r['population']==population and r['system']=='seed2026110461_C1_fixed6000'
    assert r['segment_f1_50_pairs'] is None and r['pair_reference_positions']==0
    assert r['inner_pair_bank_absent'] is True
    assert out['safety_summary']['axis_counts']['pair']==0
    assert r['endpoint_binding']['scope']==expected['scope']
    assert 'fit_threshold' not in out and 'paired_comparisons' not in out


def test_Base_is_seedless_and_diagnostics_keep_noop_visible():
    a=args(policy='Base',seed=None,population='AUDIT');m=module()
    out=m.evaluate_inner(*a,expected_provenance=m.inner_provenance(*a))
    assert out['record']['role']=='reference'
    assert not any(out['safety_summary']['H'].values())
    assert not any(out['safety_summary']['R'].values())


@pytest.mark.parametrize('change',['population','seed','policy','pairs','missing_expected',
                                  'data','model','scope','prediction'])
def test_inner_schema_or_expected_mismatch_cannot_produce_metrics(change):
    a=list(args());m=module();expected=m.inner_provenance(*a)
    if change=='population': a[3]='a63_confirmation'
    if change=='seed': a[4]=True
    if change=='policy': a[5]='C2'
    if change=='pairs': a[0][1]['kind']='pair'
    if change=='missing_expected': expected=None
    if change=='data': expected['intrinsic_identity']['data_sha256']='f'*64
    if change=='model': expected['model_identity']['model_roles']['C1']='f'*64
    if change=='scope': expected['scope']['population']='AUDIT'
    if change=='prediction': expected['intrinsic_identity']['prediction_sha256']='f'*64
    with pytest.raises(ValueError): m.evaluate_inner(*a,expected_provenance=expected)


def test_inner_runtime_does_not_import_old_writing_CLI_globals(monkeypatch):
    a=args();m=module();expected=m.inner_provenance(*a)
    original=builtins.__import__
    def restricted(name,*args,**kwargs):
        if name in ['scripts.calibrate_event_risk','scripts.event_risk_execution','scripts.event_risk_common']:
            raise AssertionError('old writing CLI imported by new inner runtime')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',restricted)
    m.evaluate_inner(*a,expected_provenance=expected)


def test_original_inner_metric_statement_projection_is_exact():
    root=Path(__file__).resolve().parents[1]
    old=ast.parse((root/'scripts/calibrate_event_risk.py').read_text())
    original=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='inner_metric')
    new=ast.parse(textwrap.dedent(inspect.getsource(module()._inner_metric))).body[0]
    projected=[n for n in new.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    assert ast.dump(ast.Module(body=projected,type_ignores=[]),include_attributes=False)==ast.dump(
        ast.Module(body=original.body,type_ignores=[]),include_attributes=False)
