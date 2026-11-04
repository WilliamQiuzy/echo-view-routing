#!/usr/bin/env python3
"""A60 actual closed receipts + complete matrix forecasts; original cumulative caps."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.proposal_common import *
from echo_routing.temporal.proposal_budget import complete
from echo_routing.temporal.proposal_resources import totals,forecast
from echo_routing.temporal.structured_guard import storage_bytes


def main(mode):
    if mode not in ['formal','outer']:raise ValueError('fixed formal or outer admission required')
    freeze=verify_freeze(mode+'_admission_freeze.json');inputs=read(PUB/'training_inputs.json');checks=read(PUB/'wholeFIT_checks.json');costs=read(PUB/'FIT_inference_cost_bounds.json')
    if not checks['passed'] or not costs['passed'] or checks['training_inputs_sha256']!=sha(PUB/'training_inputs.json'):raise ValueError('actual complete FIT graph and worst geometry timings required')
    if mode=='formal' and (RUN/'models').exists():raise ValueError('prospective allmatrix admission must precede formal models')
    if mode=='outer':
        models=read(PUB/'models_manifest.json')
        if not models['passed'] or models['formal_optimizer_updates']!=24000 or not models['all_models_and12_inner_records_frozen']:raise ValueError('all models/12inner must freeze')
    cfg=read(CFG);roles=read(PUB/'input_manifest.json')['roles'];old=read(verify_pin(cfg['prior_A59']['cache_outer_manifest.json']))['records']
    remaining=complete(roles,costs['role_geometry'],checks,costs,inputs,old,storage_bytes([RUN,PUB]),formal_pending=mode=='formal')
    spent=totals(actual_receipts());caps=forecast(spent,remaining['remaining_GPU_seconds'],remaining['remaining_CPU_seconds'],remaining['incremental_tracked_path_peak_upper_bytes'],time.time(),remaining['remaining_serial_wall_seconds'])
    write(PUB/(mode+'_admission.json'),{**caps,**remaining,'closed_actual_totals_at_snapshot':{k:v for k,v in spent.items() if k!='receipt_records'},'training_inputs_sha256':sha(PUB/'training_inputs.json'),'wholeFIT_checks_sha256':sha(PUB/'wholeFIT_checks.json'),'cost_bounds_sha256':sha(PUB/'FIT_inference_cost_bounds.json'),'admission_freeze_git':freeze['git_commit'],'formal_updates_at_admission':0 if mode=='formal' else 24000,'engineering_updates_prior_formal':76,'original_budget_not_reset':True,'epoch':time.time()})
    if not caps['all_caps_fit']:raise ValueError('complete full matrix workflow exceeds inherited caps or earlier deadline')
    print(caps,flush=True)
if __name__=='__main__':main(sys.argv[1])
