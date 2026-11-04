#!/usr/bin/env python3
"""Versioned component-measured admission; retain original failed forecast bytes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.safety_common import *
from echo_routing.temporal.safety_measured_budget import complete_measured
from echo_routing.temporal.safety_resources import totals,forecast
from echo_routing.temporal.structured_guard import storage_bytes


def main(mode):
    if mode not in ['formal','outer']:raise ValueError('fixed formal/outer measured admission required')
    freeze=verify_freeze(mode+'_admission_freeze.json');failed=PUB/'formal_admission_v1_failed.json';old=read(failed)
    if old['all_caps_fit'] or sha(failed)!='5006e43eddeac9e2aba8437d0b7d0c5600f1be7e255d6eb1ac21c45710f02063':raise ValueError('exact initial failed forecast must remain unchanged')
    output=PUB/(mode+'_admission_v2.json');canonical=PUB/(mode+'_admission.json')
    if output.exists():raise ValueError('measured admission already attempted; preserve result, no silent overwrite')
    if mode=='formal':
        if (RUN/'models').exists() or sha(canonical)!=sha(failed):raise ValueError('first formal training must follow unchanged retained prior failure')
    elif canonical.exists() or not read(PUB/'models_manifest.json')['all10finals24inner8taus56sets_frozen']:raise ValueError('fresh outer admission after full60000/24/8/56 audit required')
    checks=read(PUB/'FIT_checks.json');costs=read(PUB/'inference_cost_bounds.json');measured=read(PUB/'component_cost_measurements.json')
    if not checks['passed'] or checks['actual_engineering_optimizer_updates']!=130 or not costs['passed']:raise ValueError('qualified original FIT graph and structural timing required')
    remaining=complete_measured(read(PUB/'input_manifest.json')['roles'],checks,costs,measured,storage_bytes([RUN,PUB]),formal_pending=mode=='formal');spent=totals(actual_receipts());caps=forecast(spent,remaining['remaining_GPU_seconds'],remaining['remaining_CPU_seconds'],remaining['incremental_tracked_path_peak_upper_bytes'],time.time(),remaining['remaining_serial_wall_seconds'])
    value={**caps,**remaining,'actual_closed_costs_at_snapshot':{k:v for k,v in spent.items() if k!='receipt_records'},'supersedes_original_failed_forecast_sha256':sha(failed),'original_failed_forecast_path':str(failed.relative_to(ROOT)),'component_cost_measurements_sha256':sha(PUB/'component_cost_measurements.json'),'structural_cost_sha256':sha(PUB/'inference_cost_bounds.json'),'FIT_checks_sha256':sha(PUB/'FIT_checks.json'),'training_inputs_sha256':sha(PUB/'training_inputs.json'),'engineering_only_no_training_model_threshold_data_or_endpoint_change':True,'A61_outer_head_outcomes_used_for_forecast':False,'all_original_actual_caps_deadline_and_GPU_reserve_preserved':True,'canonical_report_replaced_only_if_complete_measured_caps_pass':bool(caps['all_caps_fit']),'freeze_git':freeze['git_commit'],'epoch':time.time()};write(output,value)
    if not caps['all_caps_fit']:raise ValueError('complete measured workload still exceeds unchanged caps; no formal launch')
    write(canonical,value);print(caps,flush=True)
if __name__=='__main__':main(sys.argv[1])
