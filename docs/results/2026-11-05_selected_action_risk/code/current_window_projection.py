from pathlib import Path
import json,time
from echo_routing.temporal.selected_risk_cost_units import remaining
R=Path('.');P=R/'docs/results/2026-11-05_selected_action_risk'
read=lambda p:json.loads(p.read_text())
profile=read(P/'cost_profile.json');shape=read(P/'cost_shape_profile.json');m=read(P/'input_manifest.json')
old=P.parent/'2026-11-04_anchor_context_risk';pr=read(old/'producer_manifest.json');wall=read(old/'process_producer_1.json')['full_process_wall_seconds']
u={n:sum(r.get('committed_logical_neural_units',r.get('actual_neural_forward_units',0)) for r in pr['records'] if r['source']==n) for n in ['a08','jepa']}
w=remaining(profile,shape,m['roles'],old_source_wall=wall,old_a08_units=u['a08'],old_jepa_units=u['jepa'],current_upper=436049885,sessions=['synthetic_session_'+str(i) for i in range(402)])
now=time.time();v={'diagnostic_only_not_stage_admission':True,'wall_CPU_projection_independent_of_session_string_dictionary':True,'synthetic_dictionary_used_only_to_satisfy_unused_storage_calculation':'not_actual_storage_or_source_mapping','CPU_remaining_seconds':w['CPU_seconds'],'remaining_wall_seconds':w['remaining_wall_seconds'],'source_FIT_GPU_forecast':w['source_FIT_GPU_forecast'],'GPU_remaining':w['GPU_seconds'],'last_start_epoch_for_full_original_window':1793897040-w['remaining_wall_seconds'],'now_epoch':now,'forecast_end_if_started_now':now+w['remaining_wall_seconds'],'deadline_epoch':1793897040,'source_and_aux_updates_not_executed':True,'phases':w['phases']}
q=P/'current_window_projection_readonly.json';assert not q.exists();q.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
