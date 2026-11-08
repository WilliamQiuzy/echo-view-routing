#!/usr/bin/env python3
"""A63 versioned actual-spent plus FULL remaining empirical forecast admission."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.selected_risk_common import *
from echo_routing.temporal.selected_risk_forecast import remaining,verify_cost_inputs,teacher_bytes
from echo_routing.temporal.selected_risk_budget import totals,forecast

def measured_namespace_bytes():
    paths=[p for root in [RUN,PUB] for p in root.rglob('*') if p.is_file()]
    if any(p.is_symlink() for p in paths):raise ValueError('tracked namespace symlink forbidden')
    return sum(p.stat().st_size for p in paths)

def admit(stage,version):
    if stage not in ['sources','confirm','formal','outer'] or isinstance(version,bool) or version<1:raise ValueError('canonical prospective stage/version required')
    freeze=verify_freeze(stage+'_admission');verify_cost_inputs(ROOT);m=read(PUB/'input_manifest.json');profile=read(PUB/'cost_profile.json')
    if profile['input_manifest_sha256']!=sha(PUB/'input_manifest.json') or profile.get('passed') is not True:raise ValueError('qualified completed measurement and exact input required')
    prior=read(ROOT/'docs/results/2026-11-04_anchor_context_risk/producer_manifest.json');oldwall=read(ROOT/'docs/results/2026-11-04_anchor_context_risk/process_producer_1.json')['full_process_wall_seconds'];units={n:sum(r['committed_logical_neural_units'] if 'committed_logical_neural_units' in r else r['actual_neural_forward_units'] for r in prior['records'] if r['source']==n) for n in ['a08','jepa']}
    measured=measured_namespace_bytes();upper=measured+300000000;work=remaining(profile,m['roles'],old_source_wall=oldwall,old_a08_units=units['a08'],old_jepa_units=units['jepa'],current_upper=upper)
    # Complete initial pass branch is estimated even if later coverage stops it.
    if (PUB/'producer_FIT_manifest.json').exists():
        r=read(PUB/'producer_FIT_manifest.json')
        if r['complete'] is not True or r['input_manifest_sha256']!=sha(PUB/'input_manifest.json'):raise ValueError('completed FIT source evidence must be exact')
        work['GPU_seconds']-=work['source_FIT_GPU_forecast'];work['phases']['teacher_CPU_including_entry_load_IO']-=work['source_FIT_GPU_forecast'];work['CPU_seconds']-=work['source_FIT_GPU_forecast'];work['remaining_wall_seconds']-=work['source_FIT_GPU_forecast']
    spent=totals(actual_receipts());now=time.time();out=forecast(spent,work['GPU_seconds'],work['CPU_seconds'],work['storage_upper_bytes'],now,work['remaining_wall_seconds']);path=PUB/f'{stage}_admission_v{version}.json'
    if path.exists():raise ValueError('versioned admitted/failed estimates immutable')
    value={**out,'stage':stage,'version':version,'source_phase':'FIT' if stage=='sources' else 'CONFIRM','remaining_complete_workload':work,'already_spent':spent,'measured_current_namespace_du_bytes':measured,'current_namespace_plus_admin_allowance_upper_bytes':upper,'source_stage_payload_byte_upper':teacher_bytes([m['roles'][r] for r in (['F0_mixed','F1_mixed'] if stage=='sources' else ['a63_confirmation_selection','a63_confirmation_calibration'])]),'input_manifest_sha256':sha(PUB/'input_manifest.json'),'cost_profile_sha256':sha(PUB/'cost_profile.json'),'admission_freeze_git':freeze['git_commit'],'forecast_is_not_actual_spend_or_guarantee':True,'old_A08_J_units_separately_scaled':units,'created_epoch':now}
    write(path,value);write(PUB/(stage+'_admission.json'),value);print({'stage':stage,'all_caps_fit':out['all_caps_fit'],'CPU_remaining':work['CPU_seconds'],'GPU_remaining':work['GPU_seconds'],'storage_upper':work['storage_upper_bytes'],'forecast_end':out['forecast_end_epoch'],'deadline':1793897040},flush=True);return 0 if out['all_caps_fit'] else 1

def main():
    p=execution_parser(__doc__);p.add_argument('--stage',choices=['sources','confirm','formal','outer'],required=True);p.add_argument('--version',type=int,required=True);a=p.parse_args()
    if not authorize(a):return 0
    cpu_only();return admit(a.stage,a.version)
if __name__=='__main__':raise SystemExit(main())
