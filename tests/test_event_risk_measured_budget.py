import copy
from echo_routing.temporal.event_risk_measured_budget import remaining


def profile():
 return {'profiles':[{'T':20,'C':0,'T_times_C':0,'K_total':0,'full_q_inference_seconds':.001,'six_gate_summary_seconds':.001,'raw_cache_init_write_read_seconds':.001,'payload_bytes':400},{'T':20,'C':30,'T_times_C':600,'K_total':40,'full_q_inference_seconds':.01,'six_gate_summary_seconds':.02,'raw_cache_init_write_read_seconds':.003,'payload_bytes':1000}], 'max_64candidate_forward_seconds':.001,'synthetic_fulltask_metric_seconds':.1,'synthetic_CI5000_seconds':.2,'synthetic_CI_unit_upper':{'native_sessions':100,'multi_streams':150,'pair_streams':300}, 'geometry':{'selection':{'N':10,'T':100,'C':10,'TC':100,'K':20},'calibration':{'N':10,'T':100,'C':10,'TC':100,'K':20},'inner_cal':{'N':10,'T':100,'C':10,'TC':100,'K':20}}}


def manifest():
 return {'roles':{r:{'sequences':n,'positions':n*20,'max_length':20,'a08_stride5_windows':n*4,'jepa_chunks':n} for r,n in [('CAL',155),('AUDIT',155),('CAL_mixed',150),('AUDIT_mixed',150),('a62_confirmation_selection',120),('a62_confirmation_calibration',120)]}}


def test_complete_remaining_and_monotone():
 p=profile();m=manifest();v=remaining(p,m,None,False,False)
 assert v['scope']=={'formal_updates':24000,'engineering_updates':72,'outer_rows':156,'inner_rows':26,'comparisons':240,'replicates':5000,'new_full_oracles':6,'old_readonly_oracles':30,'support':48,'native':24,'base_native':2,'regression':16,'fullstates':16}
 assert v['GPU_seconds']>0 and v['CPU_seconds']>=3600
 doubled=copy.deepcopy(m)
 for r in doubled['roles'].values():r['positions']*=2;r['max_length']*=2
 assert remaining(p,doubled,None,False,False)['CPU_seconds']>=v['CPU_seconds']
 after=remaining(p,m,{'max_complete_update_seconds':.002,'max_disk_save_load_seconds':.002},True,False)
 assert after['GPU_seconds']==0 and after['scope']['engineering_updates']==0
 assert after['storage_upper_bytes']<=1250000000


def test_cost_fixture_schema_without_main_benchmark():
 from scripts.probe_event_risk_cost import bundle,benchmark_metrics
 from echo_routing.temporal.event_risk_metrics import paired
 b=bundle(2,2,2);c=paired(b,b,primary=True)
 assert c['bootstrap_replicates']==5000
 assert benchmark_metrics(8)>=0
