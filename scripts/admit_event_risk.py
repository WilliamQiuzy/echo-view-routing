#!/usr/bin/env python3
"""Versioned actual-spend+complete remaining A62 admission, never a pilot."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.event_risk_common import *
from echo_routing.temporal.event_risk_budget import totals,forecast
from echo_routing.temporal.event_risk_measured_budget import remaining
from echo_routing.temporal.structured_guard import storage_bytes


def main():
    p=execution_parser(__doc__);p.add_argument('--stage',choices=['sources','formal','outer'],required=True);p.add_argument('--version',type=int,default=1);a=p.parse_args()
    if not authorize(a):return 0
    freeze=verify_freeze(a.stage+'_admission_v'+str(a.version)+'_freeze.json');spent=totals(actual_receipts());m=read(PUB/'input_manifest.json');profile=read(PUB/'cost_profile.json');checks=read(PUB/'FIT_checks.json') if a.stage!='sources' else None
    if profile['input_manifest_sha256']!=sha(PUB/'input_manifest.json') or not profile['passed']:raise ValueError('measured profile must bind actualcomplete inputs')
    if a.stage!='sources' and not read(PUB/'producer_manifest.json')['complete']:raise ValueError('all1080source payloads must already exist')
    if a.stage=='outer' and not read(PUB/'model_audit.json')['formal_optimizer_updates']==24000:raise ValueError('allfour final models before outer admission')
    cfg=read(CFG);prior=read(verify_pin(cfg['prior_A61']['producer_manifest']));r=remaining(profile,m,checks,a.stage!='sources',a.stage=='outer',prior)
    measured=storage_bytes([RUN,PUB]);peak=max(measured,r['storage_upper_bytes']+105000000);now=time.time();f=forecast(spent,r['GPU_seconds'],r['CPU_seconds'],peak,now,r['remaining_wall_seconds']);value={**f,'remaining_complete_workload':r,'already_spent':spent,'input_manifest_sha256':sha(PUB/'input_manifest.json'),'training_inputs_sha256':sha(PUB/'training_inputs.json') if (PUB/'training_inputs.json').exists() else None,'cost_profile_sha256':sha(PUB/'cost_profile.json'),'measured_current_tracked_upper_bytes':measured,'observed_current_du_bytes':measured-105000000,'admission_freeze_git':freeze['git_commit'],'created_epoch':now,'stage':a.stage,'version':a.version,'forecast_is_not_actual_spend':True}
    versioned=PUB/(a.stage+'_admission_v'+str(a.version)+'.json')
    if versioned.exists():raise ValueError('retain every admission attempt immutable; choose fresh version')
    write(versioned,value);write(PUB/(a.stage+'_admission.json'),value);print({k:value[k] for k in ['all_caps_fit','forecast_end_epoch','forecast_incremental_GPU_seconds','forecast_incremental_CPU_seconds_including_closure','forecast_cumulative_tracked_storage_upper_bytes_including_reserve']},flush=True);return 0 if f['all_caps_fit'] else 1
if __name__=='__main__':raise SystemExit(main())
