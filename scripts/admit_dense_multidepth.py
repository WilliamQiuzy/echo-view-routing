#!/usr/bin/env python3
"""Freeze a cost-only MD seed tier after actual newgraph timing and core confirmation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_dense_multidepth as run
from scripts import train_dense_sequence as base
from echo_routing.temporal.multidepth_budget import storage_components,select_tier

def main():
    cfg,freeze=run.context(False);checks=run.read(run.PUB/'checks.json');assert checks['passed'] and checks['stage_freeze_git']==freeze['git_commit'];assert not any((run.RUN/f'seed{s}'/a/'physical_updates.jsonl').exists() for s in cfg['training_seeds'] for a in cfg['arms']);basis=run.read(base.PUB/'multidepth_preparation/forecast_basis.json');assert run.read(base.PUB/'raw_outer_results.json')['status']=='complete fixed frozen exploratory rows'
    past=[json.loads(x) for x in (base.RUN/'processes.jsonl').read_text().splitlines()];spent_gpu=sum(r['full_process_wall_seconds'] for r in past if r['GPU_job'])/3600;spent_cpu=sum(r['process_CPU_seconds'] for r in past)/3600
    # Exact current namespace includes all original models/latest optimizers/source
    # and caches. Add only new MD components plus one future atomic coexistence.
    paths=[base.RUN,ROOT/'checkpoints/a57_jepa21',ROOT/'third_party/VJEPA2_1',base.PUB];existing=sum(int(x.split()[0]) for x in subprocess.check_output(['du','-sb',*map(str,paths)],text=True).splitlines());update=max(checks['raw_J_seconds_per2x64_update'],max(p['mean_seconds'] for p in basis['raw_training_profiles'].values()));chunk=max(checks['raw_decode_infer_stitch_seconds_per_chunk'],basis['full_inner_readout_max_seconds']/2769);outer=basis['canonical64_stride32_chunk_bound_bothhalves_per_row'];elapsed=(time.time()-1793648044.1596537)/3600;tiers=[]
    for seed_count in [2,1]:
        models=2*seed_count;updates=models*6000;inner=models*12;rows=models*3;component=storage_components(seed_count,checks);hours={'new_training':updates*update/3600,'all_inner_full_readouts':inner*2769*chunk/3600,'all_outer_fixed_and_packages':rows*outer*chunk/3600,'all_latest_optimizer_writes':updates/100*checks['atomic_full_optimizer_checkpoint_seconds']/3600,'coldstart_allowance':.0833333};future=1.15*sum(hours.values());cpu_extra=8 if seed_count==2 else 5 # all CPU decoder grids, aggregation, hashing, plots and verification
        row={'seed_count':seed_count,'training_models':models,'effective_updates':updates,'inner_readouts':inner,'outer_logical_rows':rows,'measured_raw_update_seconds_used_for_both_F_and_J':update,'measured_raw_full_pipeline_seconds_per_chunk':chunk,'hours_components_before_runtime_contingency':hours,'runtime_contingency':1.15,'shared_closure_GPUh_reserve':2,'already_GPUjob_h':spent_gpu,'GPU_total_including_reserve':spent_gpu+future+2,'CPU_total':spent_cpu+future+cpu_extra,'CPU_nonGPU_remaining_allowance_h':cpu_extra,'storage_existing_actual_bytes':existing,'storage_new_components':component,'peak_bytes':existing+sum(component.values()),'wall_hours':elapsed+future+cpu_extra+2};tiers.append(row)
    chosen=select_tier(tiers);result={'stage_freeze_git':freeze['git_commit'],'checks_sha256':run.sha(run.PUB/'checks.json'),'source_config_sha256':run.sha(run.CFG),'formal_MD_updates_at_admission':0,'tier_selection_uses_cost_only':True,'selected_seed_count':chosen,'training_seeds':[] if not chosen else [2026110257,2026110258][:chosen],'all_shared_caps_fit':bool(chosen),'candidates':tiers,'original_core32k_and_same_axis_A08_reproduction_complete':True,'global_caps':{'GPUjob_h':24,'actual_CPUcore_h':72,'peak_bytes':40000000000,'wall_h':48},'all_core_GPU_inference_obligations_completed_before_MD_admission':True,'protected_media_forwards':0,'epoch':time.time()};run.write(run.PUB/'admission.json',result);print(json.dumps({'selected_MD_seed_count':chosen,'tier_forecasts':[{k:r[k] for k in ['seed_count','GPU_total_including_reserve','CPU_total','peak_bytes','wall_hours']} for r in tiers]}),flush=True)
if __name__=='__main__':main()
