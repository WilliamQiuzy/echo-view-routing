#!/usr/bin/env python3
"""Terminal actual A57 worker receipts; refuses an active or incomplete phase."""
import os,time,resource
START=time.time()
import json,sys,subprocess,hashlib,socket
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from echo_routing.temporal.dense_resource_accounting import closed_totals,terminal_queue_lock
RUN=ROOT/'runs/a57_dense_sequence_v1';PUB=ROOT/'docs/results/2026-11-02_dense_target_sequence';ORIGIN=1793648044.1596537
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

def collect():
    assert not (RUN/'active_process.json').exists(),'owned queue still active'
    groups=[('',32000),('full_refit',12000),('multidepth',24000)];model_count=0
    for folder,updates in groups:
        path=PUB/folder;manifest=read(path/'models_manifest.json');summary=read(path/'summary.json');assert manifest['passed'] and manifest['formal_physical_updates']==updates and summary['model_manifest_sha256']==sha(path/'models_manifest.json');model_count+=len(manifest['models'])
    assert read(PUB/'summary.json')['raw_reader_metrics_independently_reproduced'] and read(PUB/'full_refit/summary.json')['raw_metrics_independently_reproduced'] and read(PUB/'multidepth/summary.json')['raw_reader_metrics_independently_reproduced'];assert model_count==12
    pending=sum(1 for _ in RUN.rglob('pending.json'));assert pending==0,'ambiguous model/readout pending remains'
    past=[json.loads(x) for x in (RUN/'processes.jsonl').read_text().splitlines()];aux=[json.loads(x) for x in (RUN/'auxiliary_CPU_receipts.jsonl').read_text().splitlines()];prep=[]
    for row in past:row['receipt_identity']=str((RUN/f"process_{row['stage']}_{row['attempt']}.json").relative_to(ROOT));row['historical_measurement_host_not_recorded']='measurement_host' not in row
    for i,row in enumerate(aux):row['receipt_identity']=str((RUN/'auxiliary_CPU_receipts.jsonl').relative_to(ROOT))+f':line{i+1}';row['historical_measurement_host_not_recorded']='measurement_host' not in row
    for name in ['source_acquisition.json','metadata_attempt1_failed_process.json','metadata_process.json','smoke_pixels_process.json','input_exposure_process.json','pilot_summary_process.json']:
        p=RUN/name
        if p.exists():
            x=read(p);prep.append({'model_pid':x.get('model_pid',x.get('pid')),'stage':name,'process_CPU_seconds':x['CPU_core_seconds'],'GPU_job':False,'ok':x.get('ok',True),'receipt_identity':str(p.relative_to(ROOT)),'measurement_source':str(p.relative_to(ROOT)),**{k:x[k] for k in ['measurement_host','start_epoch','end_epoch','wall_seconds'] if k in x}})
    for path,stage,field in [(PUB/'source.json','source_qualification','CPU_core_seconds'),(ROOT/'docs/results/2026-11-02_post_a56_score_diagnostic/verification.json','POST_A56_ROC','worker_CPU_core_seconds')]:
        x=read(path);prep.append({'model_pid':x.get('model_pid',x.get('pid')),'stage':stage,'process_CPU_seconds':x[field],'GPU_job':False,'ok':True,'receipt_identity':str(path.relative_to(ROOT)),**{k:x[k] for k in ['measurement_host','start_epoch','end_epoch'] if k in x}})
    # Current stable producers/readers must still have the exact pinned bytes.
    pins={}
    for name in ['pipeline_freeze.json','outer_readout_freeze.json','full_refit_v1/stage_freeze.json','full_refit_v1/formal_admission_freeze.json','full_refit_v1/outer_readout_freeze.json','multidepth_v1/stage_freeze.json','multidepth_v1/formal_admission_freeze.json','multidepth_v1/outer_readout_freeze.json']:
        f=read(RUN/name);bad=[p for p,h in f['objects'].items() if sha(ROOT/p)!=h];assert not bad,(name,bad);pins[name]={'git_commit':f['git_commit'],'verified_objects':len(f['objects']),'current_pinned_bytes_unchanged':True}
    source=read(PUB/'source.json');raw=source['source_checkpoint'];source_objects={raw['path']:raw['sha256'],'checkpoints/a57_jepa21/target_encoder.pt':source['target_encoder_sha256'],**{'third_party/VJEPA2_1/'+p:h for p,h in source['loaded_model_subset_SHA'].items()}}
    assert all(sha(ROOT/p)==h for p,h in source_objects.items());pins['official_source_and_imported_subset']={'verified_objects':len(source_objects),'current_pinned_bytes_unchanged':True}
    paths=[RUN,ROOT/'checkpoints/a57_jepa21',ROOT/'third_party/VJEPA2_1',PUB];actual=sum(int(x.split()[0]) for x in subprocess.check_output(['du','-sb',*map(str,paths)],text=True).splitlines());atomic=max(read(PUB/'pipeline_checks.json')['atomic_checkpoint_bytes'],read(PUB/'full_refit/checks.json')['atomic_full_optimizer_bytes'],read(PUB/'multidepth/checks.json')['atomic_full_checkpoint_bytes']);upper=actual+atomic+100_000_000
    gpu_query=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True).strip();cuda_pids={int(line.split(',')[0]) for line in gpu_query.splitlines() if line.strip()};owned_gpu={r['model_pid'] for r in past if r['GPU_job']};assert not cuda_pids&owned_gpu
    current=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN);snapshot=time.time();self_record={'stage':'terminal_resource_collection','model_pid':os.getpid(),'measurement_host':socket.gethostname(),'start_epoch':START,'end_epoch':snapshot,'receipt_identity':f'terminal_resource_collection:{socket.gethostname()}:{os.getpid()}:{START}','ok':True,'SELF_CPU_seconds':current.ru_utime+current.ru_stime,'waited_CHILDREN_CPU_seconds':children.ru_utime+children.ru_stime,'process_CPU_seconds':current.ru_utime+current.ru_stime+children.ru_utime+children.ru_stime,'full_process_wall_seconds':snapshot-START,'GPU_job':False};records=past+aux+prep+[self_record];total=closed_totals(records);now=time.time();gpu=total['GPU_job_wall_seconds']/3600;cpu=total['measured_CPU_core_seconds']/3600;elapsed=(now-ORIGIN)/3600;conservative_cpu=cpu+.25;passed=gpu<=24 and conservative_cpu<=72 and upper<=40_000_000_000 and elapsed<=48
    result={'phase':'A57newresearchphase, allcore+adaptive+MDadmittedworkclosed','asof_epoch':now,'phase_origin_epoch':ORIGIN,'phase_origin_basis':'prospectively fixedworktreebirth;notmeasuredCPU','phase_deadline_epoch':1793820844.1596537,'phase_wall_elapsed_hours':elapsed,**total,'actual_GPU_job_wall_hours':gpu,'measured_actual_worker_CPU_core_hours':cpu,'short_unmetered_setup_CPU_budget_allowance_hours':.25,'CPU_measured_plus_setup_budget_allowance_hours':conservative_cpu,'actual_additional_namespace_bytes':actual,'largest_known_atomic_checkpoint_bytes':atomic,'completed_storage_peak_conservative_upper_bytes':upper,'storage_upper_method':'summed retained named taskpaths+one largestknown atomiccheckpoint+100MBcode allowance;tracked-pathupper,notwhole-filesystemdeltaoranobservedpeak; preflighttemporariesboundedbyfinalretainedtaskpaths','storage_paths':[str(p.relative_to(ROOT)) for p in paths],'all_shared_caps_fit':passed,'caps':{'GPUjob_h':24,'CPUcore_h':72,'added_peak_bytes':40000000000,'wall_h':48},'formal_optimizer_updates':68000,'formal_models':12,'new_protected_media_forwards':0,'protected_labels_used_for_training_or_selection':False,'GPU_query_empty':not bool(gpu_query),'owned_GPU_pids_absent':True,'owned_queue_inactive':True,'ambiguous_pending_count':pending,'stable_source_identity_checks':pins,'historical_A55_A56_reference':'docs/results/2026-11-02_source_retention/resources.json; historicalCPUupper/lifetimecharges retainedseparately,neversummedwithactualCPU','CPU_scope':'sumofactualcompletedworker/prep/auxiliary/localtestCPUmeasurements+collectorSELFandwaitedCHILDRENthroughsnapshot;PIDdeduplicatedwhereavailable,butlegacyhost/start/PIDgapsmeanCPUcrossreceiptoverlapisnotfullyproven;notwholehostCPU;LLM/coordinatorwallandidlewaitingnotcharged;shortunwrappedcommandsunmeteredandcoveredonlybyseparatebudgetallowance','closure_queue_lock':'exclusive_supervisor.lock heldnonblockingthroughactive/GPUchecksandoutputwrites','historical_host_provenance':'recordedhostretainedwhereavailable;fixedSSHphasecontextaloneisnotaretrospectivelymeasuredhostname;unknownsremainexplicit','reserved_closure_GPU_hours_during_admission':2,'unused_reserve_or_future_cost_charged':0,'scientific_success_is_not_inferred_from_resource_completion':True,'clinical_independent_validation':False,'publication_goal_achieved':False,'result_commit_before_resource_close':sys.argv[1] if len(sys.argv)>1 else None,'subsequent_root_integration_monitor_tail':'root_coordination_tail.json recordedseparatelyafterworkerclosure;nofutureprecharge'}
    write(PUB/'process_receipts.json',{'records':records,'scope':'newphasecosts only; bothfailedandsuccessfuljobs retained, nohistoricalCPUupperconversion'});write(PUB/'resources.json',result)
    with (RUN/'auxiliary_CPU_receipts.jsonl').open('a') as f:f.write(json.dumps(self_record)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({k:result[k] for k in ['all_shared_caps_fit','actual_GPU_job_wall_hours','measured_actual_worker_CPU_core_hours','completed_storage_peak_conservative_upper_bytes','formal_optimizer_updates','formal_models','GPU_query_empty']}),flush=True);assert passed
def main():
    with terminal_queue_lock(RUN/'exclusive_supervisor.lock'):collect()
if __name__=='__main__':main()
