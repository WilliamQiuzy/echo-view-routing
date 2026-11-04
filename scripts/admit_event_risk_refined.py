#!/usr/bin/env python3
"""Versioned complete admission from independently frozen legal geometry timing."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.event_risk_common import *
from echo_routing.temporal.event_risk_engineering import verify
from echo_routing.temporal.event_risk_budget import totals,forecast
from echo_routing.temporal.event_risk_refined_budget import remaining
from echo_routing.temporal.structured_guard import storage_bytes


def main():
    p=execution_parser(__doc__);p.add_argument('--stage',choices=['sources','formal','outer'],required=True);p.add_argument('--version',type=int,default=1);a=p.parse_args()
    if not authorize(a):return 0
    freeze=verify(f'{a.stage}_refined_admission_v{a.version}_freeze.json');probe=verify('refinement_probe_freeze.json');old=read(PUB/'cost_profile.json');profile=read(PUB/'cost_refinement_profile.json');m=read(PUB/'input_manifest.json')
    if not profile['passed'] or not old['passed'] or profile['engineering_freeze_git']!=probe['git_commit'] or profile['original_cost_profile_sha256']!=sha(PUB/'cost_profile.json') or profile['input_manifest_sha256']!=sha(PUB/'input_manifest.json') or old['input_manifest_sha256']!=sha(PUB/'input_manifest.json'):raise ValueError('actual new/original measurement/input bindings differ')
    spent=totals(actual_receipts());checks=read(PUB/'FIT_checks.json') if a.stage!='sources' else None
    if a.stage!='sources' and not read(PUB/'producer_manifest.json')['complete']:raise ValueError('complete1080 production required before formal/outer')
    if a.stage=='outer' and read(PUB/'model_audit.json')['formal_optimizer_updates']!=24000:raise ValueError('allfour final model audit before outer')
    cfg=read(CFG);r=remaining(old,profile,m,checks,a.stage!='sources',a.stage=='outer',read(verify_pin(cfg['prior_A61']['producer_manifest'])))
    current=storage_bytes([RUN,PUB]);peak=max(current,r['storage_upper_bytes']+105000000);now=time.time();f=forecast(spent,r['GPU_seconds'],r['CPU_seconds'],peak,now,r['remaining_wall_seconds'])
    value={**f,'remaining_complete_workload':r,'already_spent':spent,'input_manifest_sha256':sha(PUB/'input_manifest.json'),'training_inputs_sha256':sha(PUB/'training_inputs.json') if (PUB/'training_inputs.json').exists() else None,'cost_profile_sha256':sha(PUB/'cost_profile.json'),'refinement_profile_sha256':sha(PUB/'cost_refinement_profile.json'),'measured_current_tracked_upper_bytes':current,'observed_current_du_bytes':current-105000000,'admission_freeze_git':freeze['git_commit'],'created_epoch':now,'stage':a.stage,'version':a.version,'independent_engineering_contract':pin(RUN/'engineering_contracts'/f'{a.stage}_refined_admission_v{a.version}_freeze.json'),'forecast_is_not_actual_spend':True,'original_preview_not_overwritten':sha(PUB/'cost_preview_v1.json')}
    output=PUB/f'{a.stage}_refined_admission_v{a.version}.json'
    if output.exists():raise ValueError('immutable prior admission attempt; use fresh version')
    write(output,value);write(PUB/f'{a.stage}_admission.json',value);print({k:f[k] for k in ['all_caps_fit','forecast_end_epoch','forecast_incremental_GPU_seconds','forecast_incremental_CPU_seconds_including_closure','forecast_cumulative_tracked_storage_upper_bytes_including_reserve']},flush=True);return 0 if f['all_caps_fit'] else 1


if __name__=='__main__':raise SystemExit(main())
