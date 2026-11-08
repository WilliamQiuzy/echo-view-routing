"""Gate/CSV-only synthetic fixtures; never render fake plots or claim visual QA."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'echo_routing/temporal/selected_risk_incomplete_present.py'
    spec=importlib.util.spec_from_file_location('private_incomplete_present',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def data(tmp_path):
    def pin(name,value):
        path=tmp_path/name;raw=json.dumps(value,sort_keys=True).encode();path.write_bytes(raw)
        return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
    owner='a'*64
    spec_sha='c'*64;stage='seal_aux'
    guardian=pin('synthetic_guardian.json',{'stage':stage,'runtime_spec_sha256':spec_sha,'supervisor_pid':1234,'supervisor_start_ticks':5678,'stage_exit_status':1,'model_pid':None,'runtime_authorization_sha256':owner,'actual_fresh_completed_aux_admission':{'all_caps_fit':False,'failed_checks':['deadline']}})
    parent=pin('synthetic_parent.json',{'guardian_snapshot':guardian,'stage':stage,'runtime_spec_sha256':spec_sha,'guardian_pid':1234,'guardian_start_ticks':5678,'stage_exit_status':1,'model_pid':None,'whole_child_accounting_complete':True,'runtime_authorization_sha256':owner})
    stop=pin('synthetic_stop.json',{'schema':'A63_incomplete_resource_stop_decision_v1','synthetic':True,'actual_caps_refused':True,'reason':'complete_remaining_work_forecast_refused','refusal_parent_pin':parent,'refusal_guardian_pin':guardian,'failed_checks':['deadline'],'coverage':None,'coverage_evaluated':False,'aux_optimizer_updates':48000,'formal_optimizer_updates':0,'C0_C1_instances':0,'not_run':['coverage','CONFIRM','C0','C1','CAL_AUDIT','outer','paired_CI','support','native_regression'],'overall_complete':False,'runtime_authorization_sha256':owner,'T0_epoch':0,'deadline_epoch':43200})
    models=[];curves=[]
    for seed in [2026110461,2026110462]:
        for held in [0,1]:
            for role in ['utility','risk']:
                models.append({'seed':seed,'held_fold':held,'role':role,'step':6000,'path':f'never_open_weights/{seed}/{held}/{role}.pt','sha256':'b'*64})
                curves.extend({'seed':seed,'held_fold':held,'role':role,'update':u,'loss':1./(u+1)} for u in range(1,6001))
    resources=[{'metric':'GPU job wall','actual':None,'forecast':None,'cap':1.5,'units':'h','scope':'synthetic unknown, not zero; source pin only','source_pin':guardian},
               {'metric':'Window from original T0','actual':11.,'forecast':12.5,'cap':12.,'units':'h','scope':'synthetic elapsed versus full remaining forecast','source_pin':guardian}]
    return curves,models,resources,stop


def test_complete_loss_grid_and_models_reject_missing_duplicate_or_mismatch_before_output(tmp_path):
    m=module();curves,models,resources,stop=data(tmp_path);output=tmp_path/'must_not_exist'
    for bad in [curves[:-1],curves[:-1]+[curves[0]]]:
        with pytest.raises(ValueError):m.render_incomplete(output,bad,models,resources,stop)
        assert not output.exists()
    badmodels=[dict(row) for row in models];badmodels[-1]['step']=5999
    with pytest.raises(ValueError):m.render_incomplete(output,curves,badmodels,resources,stop)
    assert not output.exists()
    badcurves=[dict(row) for row in curves];badcurves[0]['loss']=float('nan')
    with pytest.raises(ValueError):m.render_incomplete(output,badcurves,models,resources,stop)
    assert not output.exists()


def test_unknown_resource_values_survive_gate_and_raw_CSV_without_plotting(tmp_path):
    m=module();curves,models,resources,stop=data(tmp_path)
    rows,checked_models,checked_resources,decision=m._prepare(curves,models,resources,stop)
    assert len(rows)==48000 and len(checked_models)==8
    assert checked_resources[0]['actual'] is None and checked_resources[0]['forecast'] is None
    path=tmp_path/'synthetic_resources.csv'
    m._csv(path,['metric','actual','forecast','cap','units','scope','source_pin_path','source_pin_sha256'],checked_resources)
    with path.open(newline='') as handle:written=list(csv.DictReader(handle))
    assert written[0]['actual']=='unknown' and written[0]['forecast']=='unknown'
    assert written[0]['cap']=='1.5' and written[0]['units']=='h'
    assert written[0]['source_pin_sha256']==resources[0]['source_pin']['sha256']
    assert decision['coverage'] is None and decision['synthetic'] is True
    # These gate-only fixtures produce no PNG/PDF and prove no visual QA.
    assert not list(tmp_path.rglob('*.png')) and not list(tmp_path.rglob('*.pdf'))

def test_parent_must_bind_exact_guardian_raw_pin_and_process_spec_stage(tmp_path):
    m=module();curves,models,resources,stop=data(tmp_path)
    m._prepare(curves,models,resources,stop) # Correct current caller/guardian fields qualify.
    decision=json.loads(Path(stop['path']).read_bytes())
    parent=json.loads(Path(decision['refusal_parent_pin']['path']).read_bytes())
    changes=[('guardian_snapshot',{'path':str(tmp_path/'different_failure.json'),'sha256':'d'*64}),
             ('guardian_pid',1235),('guardian_start_ticks',5679),
             ('runtime_spec_sha256','d'*64),('stage','admit_formal'),('guardian_start_ticks',None)]
    for i,(key,value) in enumerate(changes):
        badparent=dict(parent);badparent[key]=value
        parent_path=tmp_path/f'bad_parent_{i}.json';raw=json.dumps(badparent,sort_keys=True).encode();parent_path.write_bytes(raw)
        badstop=dict(decision);badstop['refusal_parent_pin']={'path':str(parent_path),'sha256':hashlib.sha256(raw).hexdigest()}
        stop_path=tmp_path/f'bad_stop_{i}.json';raw=json.dumps(badstop,sort_keys=True).encode();stop_path.write_bytes(raw)
        badpin={'path':str(stop_path),'sha256':hashlib.sha256(raw).hexdigest()}
        with pytest.raises(ValueError,match='parent.*guardian'):
            m._prepare(curves,models,resources,badpin)
    assert not list(tmp_path.rglob('*.png')) and not list(tmp_path.rglob('*.pdf'))
