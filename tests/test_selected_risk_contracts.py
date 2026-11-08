import importlib,json,time
from pathlib import Path
import pytest

def module(name):
 try:m=importlib.import_module('echo_routing.temporal.selected_risk_'+name)
 except ModuleNotFoundError:m=None
 assert m is not None, f'new A63 {name} contract missing'
 return m

def test_budgets_baseline_not_recharged_and_actual_vs_forecast():
 m=module('budget');spent=m.totals([])
 assert spent['cumulative_CPU_accounting_hours']==20.231205854833334
 assert spent['cumulative_GPU_hours']==18.657838855
 a=m.forecast(spent,4500,20000,900000000,1793855000,30000)
 assert a['all_caps_fit'] and a['forecast_cumulative_GPU_hours_including_reserve']>21
 assert a['unused_forecast_reserves_are_actual_spend'] is False
 assert not m.forecast(spent,5401,20000,900000000,1793855000,30000)['all_caps_fit']
 assert not m.forecast(spent,4500,28801,900000000,1793855000,30000)['all_caps_fit']
 assert not m.forecast(spent,4500,20000,1000000001,1793855000,30000)['all_caps_fit']
 assert m.remaining_limits(spent,1793890000,potential_GPU_context=False,closing=True)['stage_wall_seconds']>0

def test_import_closure_package_alias_relative_and_oldcli_rejection(tmp_path):
 m=module('contracts');(tmp_path/'scripts').mkdir();(tmp_path/'echo_routing/pkg').mkdir(parents=True)
 (tmp_path/'echo_routing/pkg/__init__.py').write_text('')
 (tmp_path/'echo_routing/pkg/sub.py').write_text('from . import other')
 (tmp_path/'echo_routing/pkg/other.py').write_text('')
 e=tmp_path/'scripts/selected_risk_test.py';e.write_text('from echo_routing.pkg import sub')
 out=m.source_closure(tmp_path,['scripts/selected_risk_test.py'])
 assert 'echo_routing/pkg/other.py' in out and 'echo_routing/pkg/__init__.py' in out
 (tmp_path/'scripts/old_cli.py').write_text('')
 e.write_text('from scripts import old_cli')
 with pytest.raises(ValueError):m.source_closure(tmp_path,['scripts/selected_risk_test.py'])

def test_stage_mandatory_entries_inputs_and_failbranch_has_no_formal():
 m=module('contracts');entries,inputs=m.stage_spec('coverage_terminal')
 assert any('close_selected_risk' in p for p in entries)
 assert any('coverage.json' in p for p in inputs)
 assert not any(Path(p).name in ['models_manifest.json','summary.json'] for p in inputs)
 assert m.stage_spec('aux_train')[0]==['scripts/train_selected_risk_aux.py']
 with pytest.raises(ValueError):m.stage_spec('whatever')

def test_source_roles_actual_domains_and_fold_refs():
 m=module('data');import numpy as np
 qs=[{'id':f's{i}_{c}','kind':'native','labels':np.full(90,c,dtype=np.int64),'refs':[(f's{i}_{c}',j,1.) for j in range(90)]} for i in range(10) for c in range(5)]
 a,recipes,info=m.construct(qs,0,count=4)
 b,recipes2,info2=m.construct(qs,0,count=4)
 assert recipes==recipes2 and info['RNG_domain']==[2026110563,63,0]
 assert len(a)==4 and all({r[0] for r in q['refs']}<={q['id'] for q in qs} for q in a)
 for q in a:
  runs=[]
  for r in q['refs']:
   if not runs or runs[-1]!=r[0]:runs.append(r[0])
  assert 4<=len(runs)<=8 and len(runs)==len(set(runs))
 with pytest.raises(ValueError):m.construct(qs,4,count=4)

def test_unknown_admin_CPU_is_null_measured_plus_separate_allowance():
 m=module('budget');r={'receipt_id':'unknown-admin','started_epoch':1.,'finished_epoch':2.,'full_process_wall_seconds':1.,'measured_CPU_core_seconds':None,'CPU_accounting_allowance_seconds':1.,'measurement_kind':'administrative_partial_receipt','GPU_job':False,'CUDA_context_occupied':False,'observed_worker_exit_code':None}
 a=m.totals([r]);assert a['new_measured_CPU_seconds']==0 and a['new_CPU_accounting_seconds']==1 and a['unknown_CPU_receipts']==1
 assert a['receipt_records'][0]['measured_CPU_core_seconds'] is None
 assert a['cumulative_CPU_accounting_hours']==20.231205854833334+1/3600
