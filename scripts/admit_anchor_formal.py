#!/usr/bin/env python3
"""Post-cache/actualwholeFIT admission: actual spent plus complete remaining matrix."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.anchor_common import RUN,PUB,read,write,sha,verify_freeze
from echo_routing.temporal.anchor_budget import source_seconds,storage_forecast
from echo_routing.temporal.anchor_resources import totals,forecast
from echo_routing.temporal.structured_guard import storage_bytes


def main():
    freeze=verify_freeze('formal_admission_freeze.json');inputs=read(PUB/'training_inputs.json');check=read(PUB/'wholeFIT_checks.json')
    cached=read(PUB/'cache_fit_manifest.json');roles=read(PUB/'input_manifest.json')['roles'];gpu=read(PUB/'FIT_GPU_checks.json')
    if not all(v['passed'] for v in [inputs,check,cached]) or check['training_inputs_sha256']!=sha(PUB/'training_inputs.json'):
        raise ValueError('complete cache and actualwholeFIT frozen graph checks required')
    if (RUN/'models').exists():raise ValueError('new formal admission required before formal outcomes')
    rows=[json.loads(line) for line in (RUN/'processes.jsonl').read_text().splitlines()]
    local=read(PUB/'local_cpu_receipts.json')
    for path,expected in local['file_pins'].items():
        if sha(ROOT/path)!=expected:raise ValueError('local measured CPU receipt changed')
    spent=totals(rows+local['receipts']);io=gpu['maximum_durable_cache_IO_seconds_per_sequence']
    fresh=['confirmation_selection','confirmation_calibration'];outer=['selection','calibration','old_stress_selection','old_stress_calibration']+fresh
    units_a=sum(roles[k]['a08_stride5_windows'] for k in fresh);units_j=sum(roles[k]['jepa_chunks'] for k in fresh)
    seq=sum(roles[k]['sequences'] for k in fresh)
    source=1.5*(source_seconds(gpu['source_timings']['a08'],units_a,seq,32)+source_seconds(gpu['source_timings']['jepa'],units_j,seq,4)
                +2*sum(roles[k]['sequences'] for k in outer)*io+2*sum(gpu['source_load_seconds'].values()))+120
    old=read(PUB/'cache_admission.json')['remaining']['remaining_CPU_components_seconds']
    cpu={k:v for k,v in old.items() if k not in ['FIT_candidate_targets_features_normalizers_plans','source_cache_worker_and_probes',
         'four_fixed6000_heads_checkpoint_and_trace','inflight_admission_and_transition_margin']}
    cpu['source_outer_cache_worker_and_probes']=1.2*source
    cpu['four_fixed6000_heads_checkpoint_and_trace']=2*24000*max(v['conservative_full_update_timing_seconds'] for v in check['heads'].values())+240*io+120
    cpu['inflight_admission_transition_fullFIT_audit_margin']=120.
    storage=storage_forecast(roles,storage_bytes([RUN,PUB]));wall=source+sum(v for k,v in cpu.items() if k!='source_outer_cache_worker_and_probes')
    caps=forecast(spent,source,sum(cpu.values()),storage['incremental_tracked_path_peak_upper_bytes'],time.time(),wall)
    write(PUB/'formal_admission.json',{**caps,'remaining_GPU_seconds':source,'remaining_CPU_components_seconds':cpu,
        'remaining_CPU_seconds':sum(cpu.values()),'remaining_serial_wall_seconds':wall,'storage':storage,
        'closed_actual_totals_at_snapshot':{k:v for k,v in spent.items() if k!='receipt_records'},
        'wholeFIT_checks_sha256':sha(PUB/'wholeFIT_checks.json'),'training_inputs_sha256':sha(PUB/'training_inputs.json'),
        'processes_sha256':sha(RUN/'processes.jsonl'),'local_cpu_receipts_sha256':sha(PUB/'local_cpu_receipts.json'),
        'remaining_neural_units':{'original5A08_windows':units_a,'JEPA64_chunks':units_j},
        'all4x6000_12inner_70rows_96bootstrap_oracle_audits_figures_closure_included':True,
        'formal_optimizer_updates_at_admission':0,'engineering_updates_before_formal':84+check['engineering_optimizer_updates'],
        'engineering_mutable_models_before_formal':16+check['engineering_mutable_models'],
        'admission_freeze_git':freeze['git_commit'],'epoch':time.time()})
    if not caps['all_caps_fit']:raise ValueError('complete actual remaining workflow does not fit inherited caps')
    print(caps,flush=True)

if __name__=='__main__':main()
