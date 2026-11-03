#!/usr/bin/env python3
"""Measured complete admission for adaptivepair plus preserved two-seed MD obligations."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_dense_refit as run,train_dense_sequence as base
from echo_routing.temporal.multidepth_budget import storage_components

def main():
    cfg,freeze=run.context(False);checks=run.read(run.PUB/'checks.json');md=run.read(base.PUB/'multidepth/checks.json');assert checks['passed'] and md['passed'] and checks['stage_freeze_git']==freeze['git_commit'];assert not any((run.RUN/a/'physical_updates.jsonl').exists() for a in cfg['arms']);past=[json.loads(x) for x in (base.RUN/'processes.jsonl').read_text().splitlines()];aux=[json.loads(x) for x in (base.RUN/'auxiliary_CPU_receipts.jsonl').read_text().splitlines()];assert not {r['model_pid'] for r in past}&{r['model_pid'] for r in aux};gpu=sum(r['full_process_wall_seconds'] for r in past if r['GPU_job'])/3600;cpu=sum(r['process_CPU_seconds'] for r in past+aux);prep={}
    for name in ['source_acquisition.json','metadata_attempt1_failed_process.json','metadata_process.json','smoke_pixels_process.json','input_exposure_process.json','pilot_summary_process.json']:
        p=base.RUN/name
        if p.exists():prep[name]=run.read(p)['CPU_core_seconds']
    prep['source_qualification']=run.read(base.PUB/'source.json')['CPU_core_seconds'];prep['POST_A56_ROC']=run.read(ROOT/'docs/results/2026-11-02_post_a56_score_diagnostic/verification.json')['worker_CPU_core_seconds'];cpu=(cpu+sum(prep.values()))/3600;basis=run.read(base.PUB/'multidepth_preparation/forecast_basis.json');pipe=run.read(base.PUB/'pipeline_checks.json');old=max(x['mean_seconds'] for x in basis['raw_training_profiles'].values());chunk=max(pipe['raw_full_infer_seconds_per_chunk'],basis['full_inner_readout_max_seconds']/2769,md['raw_decode_infer_stitch_seconds_per_chunk']);outer=run.read(base.PUB/'outer_inputs.json')['total_chunks_both_halves'];inner=pipe['inner_total_chunks'];components={
        'adaptive_12000_updates':12000*max(checks['raw20_RST_seconds_per_update'],old)/3600,
        'adaptive_2fixed6000_outer_rows':2*outer*chunk/3600,
        'adaptive_120_latest_optimizer_writes':120*checks['atomic_full_optimizer_seconds']/3600,
        'preserved_MD_24000_updates':24000*max(md['raw_J_seconds_per2x64_update'],old)/3600,
        'preserved_MD_48complete_inner_readouts':48*inner*chunk/3600,
        'preserved_MD_12logical_outer_rows':12*outer*chunk/3600,
        'preserved_MD_240_latest_optimizer_writes':240*md['atomic_full_optimizer_checkpoint_seconds']/3600,
        'coldstart_audit_freeze_allowance':.15}
    future=1.2*sum(components.values());paths=[base.RUN,ROOT/'checkpoints/a57_jepa21',ROOT/'third_party/VJEPA2_1',base.PUB];existing=sum(int(x.split()[0]) for x in subprocess.check_output(['du','-sb',*map(str,paths)],text=True).splitlines());mdstorage=storage_components(2,md);fullweights=(86833152+889994)*4+2_000_000;newstorage={'adaptive_4fixed2000_6000_weights':4*fullweights,'adaptive_2latest_fulloptimizers':2*(checks['atomic_full_optimizer_bytes']+1_000_000),'adaptive_bank_cache_logs_and_serialization':200_000_000,'shared_atomic_extra_if_refit_larger':max(0,checks['atomic_full_optimizer_bytes']-mdstorage['one_atomic_optimizer_coexistence'])};peak=existing+sum(mdstorage.values())+sum(newstorage.values());total=gpu+future+2;cpu_total=cpu+future+10+.25;wall=(time.time()-1793648044.1596537)/3600+future+10+2;passed=total<=24 and cpu_total<=72 and peak<=40_000_000_000 and wall<=48
    result={'all_caps_fit':passed,'stage_freeze_git':freeze['git_commit'],'checks_sha256':run.sha(run.PUB/'checks.json'),'MD_checks_sha256':run.sha(base.PUB/'multidepth/checks.json'),'formal_refit_updates_at_admission':0,'preserved_MD_two_seed_obligations_included':True,'already_GPUjob_h':gpu,'already_measured_CPUcore_h':cpu,'measured_auxiliary_CPU_receipts_count':len(aux),'measured_prep_CPU_seconds':prep,'future_GPU_components_before_1point2contingency':components,'runtime_contingency':1.2,'global_closure_GPUh_reserve':2,'total_GPUjob_h_including_reserve':total,'CPU_total_conservative_h':cpu_total,'CPU_nonGPU_remaining_allowance_h':10,'unmetered_short_setup_forecast_CPU_allowance_h':.25,'CPU_scope':'supervisedworkers+separateauxiliaryCPUPIDs+listedprep; noLLMwall or idle time; allowances areforecast,notspent','existing_actual_bytes':existing,'preserved_MD_new_storage_components':mdstorage,'adaptive_new_storage_components':newstorage,'peak_bytes_conservative':peak,'phase_wall_total_conservative_h':wall,'no_remaining_core_model_readouts':True,'priority':'adaptivepairformalfirst; preservedMDafter measured cost-only admission','epoch':time.time()};run.write(run.PUB/'admission.json',result);print(json.dumps({k:result[k] for k in ['all_caps_fit','total_GPUjob_h_including_reserve','CPU_total_conservative_h','peak_bytes_conservative','phase_wall_total_conservative_h']}),flush=True)
if __name__=='__main__':main()
