"""One reproducible aggregate-only plot smoke; no actual A62 results or NN."""
import importlib.util,json,hashlib,csv
from pathlib import Path
import pytest
from echo_routing.temporal.event_risk_figures import render
from echo_routing.temporal.event_risk_metrics import plan,risk_summary

def test_six_aggregate_families_immutable_artifacts_and_fixture_watermark(tmp_path):
 spec=importlib.util.spec_from_file_location('acceptancetoy',str(Path(__file__).with_name('test_event_risk_acceptance.py')));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);rows,comparisons,info=m.EventRiskAcceptanceTests().fixture()
 inner=[]
 for r in plan()['inner_rows']:
  inner.append({**r,'action_statistics_full_task':{'accepted_actions':3,'N_admitted_tops':30},'risk_diagnostics':{arm:{'N_top':risk_summary([.1,.3,.9],[False,True,True])} for arm in ['R1','R2']}})
 cal={'thresholds':[{'seed':s,'policy':p,'tau':.3} for s in [2026110461,2026110462] for p in ['M0','R1','R2']],'legacy_thresholds':[{'seed':s,'policy':'L0','tau':.01} for s in [2026110461,2026110462]]}
 reg=[{**r,'accepted':r['policy']=='L0','gate_score':.2 if r['policy']=='L0' else .4,'tau':.3,'H':40 if r['axis']=='Case_A' else 9,'R':0 if r['axis']=='Case_A' else 2,'short16_lost':0 if r['axis']=='Case_A' else 2,'context_summary':{'minimum_overlap_over_full_run':.2,'minimum_full_Base_run_A08_margin':.8}} for r in plan()['regression_records']]
 curves=[{'seed':s,'arm':a,'update':u,'loss':.7-u/20000} for s in [2026110461,2026110462] for a in ['R1','R2'] for u in [1,2000,4000,6000]]
 for r in reg:
  if r['axis']=='Case_A' and r['seed']==2026110462:
   r.update(accepted=False,gate_score=None,H=None,R=None,short16_lost=None,context_summary=None)
 pub=tmp_path/'synthetic';pub.mkdir()
 resources={'incremental_GPU_seconds':100,'incremental_CPU_seconds':200,'incremental_tracked_storage_upper_bytes':1000000}
 result=render(pub,rows,comparisons,inner,cal,reg,curves,resources,synthetic_fixture=True)
 assert result['synthetic_fixture'] and not result['actual_visual_QA_complete']
 assert result['families']==result['PNG']==result['PDF']==result['CSV']==6
 assert len(result['artifacts'])==18
 with (pub/'figures/05_context_regression.csv').open() as f: exported=list(csv.DictReader(f))
 missing=[r for r in exported if r['axis']=='Case_A' and r['seed']=='2026110462' and not r.get('context_observable')]
 assert len(missing)==4 and all(r['H']==r['R']==r['gate_score']=='' for r in missing)
 for row in result['artifacts']:
  raw=(pub/row['path']).read_bytes()
  assert hashlib.sha256(raw).hexdigest()==row['sha256'] and len(raw)>100
 with pytest.raises(ValueError):render(pub,rows,comparisons,inner,cal,reg,curves,resources,synthetic_fixture=True)
