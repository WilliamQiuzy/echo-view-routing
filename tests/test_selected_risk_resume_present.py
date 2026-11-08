"""Synthetic aggregate presentation only; no new experiment/model output."""
import copy
import importlib
from pathlib import Path
import runpy

import numpy as np
import pytest

from echo_routing.temporal.selected_risk_metrics import readout_plan


def module(): return importlib.import_module('echo_routing.temporal.selected_risk_resume_present')


def inputs():
    f=runpy.run_path(str(Path(__file__).with_name('test_selected_risk_scope.py')))
    rows,ci=f['complete']();p=readout_plan()
    inner=[dict(s,segment_f1_50_multi=.9,inner_pair_bank_absent=True,
                segment_f1_50_pairs=None,pair_reference_positions=0) for s in p['inner_rows']]
    zero={k:0 for k in ('H','R','dE','dFP','u','true_recovered','true_lost','short10_recovered',
        'short10_lost','short16_recovered','short16_lost','JEPA_exclusive_short16_recovered')}
    def kind(axes,active=False):
        effects=dict(zero)
        if active:effects.update(R=5,dE=-5,true_recovered=5,short16_recovered=5,JEPA_exclusive_short16_recovered=5)
        return dict(axes=axes,N_admitted_tops=5 if active else 0,available_tops=5 if active else 0,
            positive_u_available_tops=0,all_N_top_effects=dict(effects),all_available_top_effects=dict(effects),
            union_recovered_true_units=5 if active else 0,union_recovered_short10_true_units=0,
            union_recovered_short16_true_units=5 if active else 0,pooled_F1_objective_member=False)
    def result(safety):
        multi=kind(120,True);multi.pop('pooled_F1_objective_member')
        multi.update(oracle={'F1':.95,'M':19,'P':20,'T':20,'numerator':38,'denominator':40,'iterations':1},
            joint_chosen_tops=5,joint_chosen_effects=dict(multi['all_available_top_effects']))
        return dict(safety=safety,multi=multi,native=kind(50),pair=kind(2))
    support=[dict(table=s['table'],half=s['half'],seed=s['seed'],safety=s['safety'],
        readonly_A62_reference=s['readonly_A62'],result={'safety':s['safety'],
          'multi':{'oracle':{'F1':.95},'joint_chosen_effects':{'JEPA_exclusive_short16_recovered':5}},
          'native':{'axes':50},'pair':{'axes':2}}) for s in p['support_cells']]
    for row in support:row['result']=result(row['safety'])
    cube=np.zeros((5,5,5),dtype=int)
    for c in range(5):cube[c,c,c]=10
    native=[dict(population=s['half'],seed=s['seed'],policy=s['policy'],result={
        'true_Base_prediction_joint5cube':cube.tolist(),'Base_correct_to_wrong':0,
        'Base_wrong_to_correct':0}) for s in p['base_native_references']+p['native_cells']]
    regression=[dict(axis=s['axis'],seed=s['seed'],policy=s['policy'],top_exists=False,N_admitted=False,
        accepted=False,gate_score=None,tau=None,H=None,R=None,dFP16=None,short16_lost=None,
        actual_H=0,actual_R=0,actual_dFP16=0,actual_short16_lost=0,
        context_summary=None,candidate_key_sha256=None) for s in p['regression_records']]
    curves=[dict(seed=s,arm=a,update=u,loss=1/(u+1)) for s in [2026110461,2026110462]
            for a in ['C0','C1'] for u in range(1,6001)]
    resources=dict(known_CPU_seconds=10.,CPU_accounting_seconds=20.,incremental_GPU_seconds=0.,
        incremental_tracked_storage_upper_bytes=None,actual_caps_fit=None,
        CPU_cap=28800,GPU_cap=5400,storage_cap=1000000000)
    return rows,ci,inner,support,native,regression,curves,resources


def test_full_renderer_requires_all_fixed_inputs_before_writing(tmp_path):
    a=list(inputs());a[0]=a[0][:-1]
    with pytest.raises(ValueError): module().render_full(tmp_path/'invalid',*a,synthetic=True)
    assert not (tmp_path/'invalid').exists()


@pytest.mark.parametrize('index',[2,3,4,5,6])
def test_missing_inner_support_native_regression_or_curve_cannot_render(tmp_path,index):
    a=list(inputs());a[index]=a[index][:-1]
    with pytest.raises(ValueError): module().render_full(tmp_path/'invalid',*a,synthetic=True)
    assert not (tmp_path/'invalid').exists()


def test_synthetic_full_png_pdf_csv_outputs_are_immutable_and_never_visual_QA(tmp_path):
    m=module();a=inputs();r=m.render_full(tmp_path,*a,synthetic=True)
    assert r['synthetic'] is True and r['actual_visual_QA_passed'] is False
    assert len([x for x in r['files'] if x['path'].endswith('.png')])==6
    assert len([x for x in r['files'] if x['path'].endswith('.pdf')])==6
    assert r['counts']==dict(outer=238,comparisons=140,inner=34,support=56,native=34,regression=32,curves=24000)
    text=(tmp_path/'synthetic_full_figures'/'README.md').read_text()
    assert 'SYNTHETIC' in text and 'unknown' in text
    with pytest.raises(FileExistsError): m.render_full(tmp_path,*a,synthetic=True)


def test_inconsistent_native_or_case_effects_are_not_just_plotted(tmp_path):
    a=list(inputs());a[4][0]['result']['Base_correct_to_wrong']=1
    with pytest.raises(ValueError): module().render_full(tmp_path/'bad-native',*a,synthetic=True)
    a=list(inputs());a[5][0]['actual_H']=1
    with pytest.raises(ValueError): module().render_full(tmp_path/'bad-case',*a,synthetic=True)


def test_native_plot_keeps_both_matched_arms_visible(tmp_path,monkeypatch):
    m=module();seen={}
    def capture(fig,name,*args,**kwargs):
        seen[name]=[a.get_title() for a in fig.axes]
        import matplotlib.pyplot as plt
        plt.close(fig)
    monkeypatch.setattr(m,'_finish',capture)
    m.render_full(tmp_path,*inputs(),synthetic=True)
    titles=seen['05_native_regression']
    assert any('C0' in x for x in titles) and any('C1' in x for x in titles)
    gross_titles=seen['02_gross_short_effects']
    assert all(any(k in t for t in gross_titles) for k in ('native','multi','pair'))


def test_no_top_regression_keeps_absence_null_not_fake_zero_potential(tmp_path):
    a=list(inputs());a[5][0]['H']=0
    with pytest.raises(ValueError): module().render_full(tmp_path,*a,synthetic=True)
    assert not (tmp_path/'synthetic_full_figures').exists()


def test_stageS_uses_new_namespace_and_preserves_coverage_failure(tmp_path):
    f=runpy.run_path(str(Path(__file__).with_name('test_selected_risk_figures.py')))
    coverage,curves=f['fixture']();m=module()
    r=m.render_stageS(tmp_path,coverage,curves,synthetic=True)
    assert r['coverage_passed'] is False and r['formal_optimizer_updates']==0
    assert r['actual_visual_QA_passed'] is False
    assert len([x for x in r['files'] if x['path'].endswith('.png')])==4
    assert len([x for x in r['files'] if x['path'].endswith('.pdf')])==4
    with pytest.raises(FileExistsError): m.render_stageS(tmp_path,coverage,curves,synthetic=True)


def test_actual_renderer_checks_full_namespace_before_any_directory_work(monkeypatch):
    m=module()
    def forbidden(*args,**kwargs):raise AssertionError('directory helper reached before full namespace guard')
    monkeypatch.setattr(m,'_parent',forbidden)
    bad=Path(m.__file__).resolve().parents[2]/'docs/results/2026-11-05_selected_action_risk/2026-11-05_a63_resume_execution'
    with pytest.raises(ValueError):m._setup(bad,'full_figures',False,[])


def test_existing_top_requires_potential_effects_and_candidate_identity(tmp_path):
    a=list(inputs());a[5][0]['top_exists']=True
    with pytest.raises(ValueError):module().render_full(tmp_path,*a,synthetic=True)
    assert not (tmp_path/'synthetic_full_figures').exists()


def test_support_export_preserves_all_three_kind_availability_and_effects(tmp_path):
    import csv,json
    m=module();m.render_full(tmp_path,*inputs(),synthetic=True)
    with (tmp_path/'synthetic_full_figures/support.csv').open() as f:first=next(csv.DictReader(f))
    for kind in ('native','multi','pair'):
        out=json.loads(first[kind+'_result'])
        assert 'N_admitted_tops' in out and 'available_tops' in out and 'all_available_top_effects' in out
        assert out['all_N_top_effects']['H']==0
    assert 'joint_chosen_effects' in json.loads(first['multi_result'])


def test_synthetic_renderer_rejects_external_non_temporary_study_path(monkeypatch):
    m=module()
    def forbidden(*args,**kwargs):raise AssertionError('directory helper reached for non-temp study')
    monkeypatch.setattr(m,'_parent',forbidden)
    with pytest.raises(ValueError):m._setup(Path('/Users/another-study/results'),'synthetic_full_figures',True,[])


@pytest.mark.parametrize('field,value',[('positions','private-id'),('all_native_majorities_mechanically_equal_Base','private-id'),('true_Base_frame_confusion5',[['private-id']]*5)])
def test_native_anonymous_known_fields_reject_wrong_value_schema(field,value):
    m=module();row=inputs()[4][0]['result'];row[field]=value
    with pytest.raises(ValueError):m._native_public(row)
