#!/usr/bin/env python3
"""Independent stage-only source admission; never admits A61 formal training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.safety_common import *
from echo_routing.temporal.safety_resources import totals,forecast
from echo_routing.temporal.structured_guard import storage_bytes


def main():
    freeze=verify_freeze('producer_precache_admission_freeze.json')
    if (PUB/'producer_precache_admission.json').exists() or (RUN/'models').exists():raise ValueError('fresh stage-only admission before formal models required')
    inputs=read(PUB/'input_manifest.json');roles=['a61_confirmation_selection','a61_confirmation_calibration']
    counts={s:sum(inputs['roles'][r][k] for r in roles) for s,k in [('a08','a08_stride5_windows'),('jepa','jepa_chunks')]}
    if counts!={'a08':6574,'jepa':1127} or sum(inputs['roles'][r]['sequences'] for r in roles)!=240:raise ValueError('exact fixed new240 genuine joined producer geometry required')
    spent=totals(actual_receipts());peak=storage_bytes([RUN,PUB])+128_000_000;caps=forecast(spent,1200.,2400.,peak,time.time(),1200.)
    write(PUB/'producer_precache_admission.json',{**caps,'scope':'ONLY opt-in frozen teacher source cache;formal training NOT admitted','GPU_occupied_process_wall_upper_seconds':1200,'CPU_upper_core_seconds':2400,'incremental_tracked_path_peak_upper_bytes':peak,'additional_payload_and_journal_room_bytes':128_000_000,'source_neural_units':counts,'genuine_streams_per_producer':240,'formal_optimizer_updates_at_admission':0,'A61_action_head_forward_or_GTmetric_admitted':False,'source_code_Git_commit':freeze['git_commit'],'actual_closed_costs_at_snapshot':{k:v for k,v in spent.items() if k!='receipt_records'},'epoch':time.time()})
    if not caps['all_caps_fit']:raise ValueError('precache stage cannot fit inherited actual caps/deadline;do not launch')
    print(caps,flush=True)
if __name__=='__main__':main()
