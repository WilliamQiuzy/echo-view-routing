#!/usr/bin/env python3
"""A61 complete measured forecast under unchanged original cumulative caps."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.safety_common import *
from echo_routing.temporal.safety_budget import complete
from echo_routing.temporal.safety_resources import totals,forecast
from echo_routing.temporal.structured_guard import storage_bytes


def main(mode):
    if mode not in ['formal','outer']:raise ValueError('fixed formal/outer admission required')
    freeze=verify_freeze(mode+'_admission_freeze.json');checks=read(PUB/'FIT_checks.json');costs=read(PUB/'inference_cost_bounds.json')
    if not checks['passed'] or checks['actual_engineering_optimizer_updates']!=130 or checks['mutable_check_models']!=22 or not costs['passed']:raise ValueError('complete new FIT graph/structural cost checks required')
    if mode=='formal' and (RUN/'models').exists():raise ValueError('prospective formal admission must precede formalmodels')
    if mode=='outer':
        models=read(PUB/'models_manifest.json')
        if not models['passed'] or models['formal_optimizer_updates']!=60000 or not models['all10finals24inner8taus56sets_frozen']:raise ValueError('complete model/calibration barrier required')
    roles=read(PUB/'input_manifest.json')['roles'];remaining=complete(roles,checks,costs,storage_bytes([RUN,PUB]),formal_pending=mode=='formal')
    spent=totals(actual_receipts());caps=forecast(spent,remaining['remaining_GPU_seconds'],remaining['remaining_CPU_seconds'],remaining['incremental_tracked_path_peak_upper_bytes'],time.time(),remaining['remaining_serial_wall_seconds'])
    write(PUB/(mode+'_admission.json'),{**caps,**remaining,'actual_closed_costs_at_snapshot':{k:v for k,v in spent.items() if k!='receipt_records'},'training_inputs_sha256':sha(PUB/'training_inputs.json'),'FIT_checks_sha256':sha(PUB/'FIT_checks.json'),'structural_cost_sha256':sha(PUB/'inference_cost_bounds.json'),'A61_outer_utility_or_risk_outcomes_used_for_budget_choice':False,'only_FIT_or_synthetic_timing_and_compute_metadata_used':True,'engineering_optimizer_updates_prior_formal':130,'cost_only_optimizer_updates':0,'original_caps_deadline_retained':True,'freeze_git':freeze['git_commit'],'epoch':time.time()})
    if not caps['all_caps_fit']:raise ValueError('whole remaining matrix exceeds inherited caps/deadline;no hidden reduction')
    print(caps,flush=True)
if __name__=='__main__':main(sys.argv[1])
