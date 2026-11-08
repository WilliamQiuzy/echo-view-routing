#!/usr/bin/env python3
"""Local-only A63 resource-stop collection; never a remote/model collector.

Remote current state remains unknown if SSH is unavailable. This process reads
only local frozen evidence, records its own CPU, and leaves root publication
pending. It neither launches work nor edits any predecessor namespace.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,time,resource,hashlib,socket
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def merge_receipts(known,recovered):
    seen={};out=[]
    for bank in [known,recovered]:
        own=set()
        for row in bank:
            key=row['receipt_id']
            if key in own:raise ValueError('duplicate within one actual receipt bank')
            own.add(key)
            if key in seen:
                if row!=seen[key]:raise ValueError('recovered alias changes actual process receipt')
                continue
            seen[key]=row;out.append(row)
    return out

def recovered_parent_rows(value):
    if value.get('actual_supervisor_OS_exit_code')!=0 or value.get('actual_supervisor_directly_waited') is not True or value.get('error') is not None or value.get('scope_is_admission_no_new_model_execution') is not True or any(type(value.get(k)) is not int or value[k]!=0 for k in ['new_MLP_module_forward_calls','new_optimizer_updates','teacher_forwards','production_C0_C1_instances']):raise ValueError('only actual waited zero-neural source admission may be recovered')
    rows=value['new_process_receipts']+value['new_entry_receipts']
    if not value['new_process_receipts'] or any(r.get('observed_worker_exit_code')!=0 or r.get('child_exit_code_directly_waited') is not True for r in value['new_process_receipts']):raise ValueError('all recovered admission workers must have actual direct OS0')
    return rows

def build_stop_state(stop,projection,rows,workers):
    from echo_routing.temporal.selected_risk_budget import totals,DEADLINE_EPOCH
    from echo_routing.temporal.selected_risk_terminal import terminal_state
    counters=['teacher_payloads','A08_windows','JEPA_chunks','teacher_forwards','aux_optimizer_updates','formal_optimizer_updates','C0_C1_instances']
    if stop.get('mode')!='resource-stopped' or stop.get('coverage_assessed') is not False or stop.get('scientific_queues_stopped') is not True or stop.get('baseline_not_reset') is not True or set(stop['source_counters'])!=set(counters) or any(type(v) is not int or v!=0 for v in stop['source_counters'].values()):raise ValueError('this stop requires explicit zero executed A63 source/aux/formal, unassessed coverage')
    p=projection
    if p.get('diagnostic_only_not_stage_admission') is not True or p['deadline_epoch']!=DEADLINE_EPOCH or p['forecast_end_if_started_now']!=p['now_epoch']+p['remaining_wall_seconds'] or p['forecast_end_if_started_now']<=DEADLINE_EPOCH:raise ValueError('unchanged full-work projection must actually exceed original deadline; not a fabricated failed admission')
    return terminal_state('resource-stopped',None,None,stop['source_counters'],totals(rows),workers)

def _mandatory():
    from scripts.selected_risk_common import PUB,relative
    from scripts.selected_risk_shared_contracts import required as old_required
    from echo_routing.temporal.selected_risk_contracts import source_closure
    from scripts.selected_risk_common import ROOT
    from echo_routing.temporal.selected_risk_forecast import COST_INPUTS,COST_INPUT_PROOF
    names=['execution_stop_decision.json','current_window_projection_readonly.json','current_window_projection_binding.json','admin_current_window_projection_v2_CPU.json','cost_profile.json','cost_shape_profile.json','input_manifest.json','late_sourcegate_v3_parent_observation.json','local_known_worker_receipts.json','local_cpu_receipts.json','root_local_closure_storage_snapshot.json']
    return old_required('metadata')|set(source_closure(ROOT,['scripts/a63_collect_selected_risk_stop.py','scripts/a63_wait_selected_risk_stop.py',relative(PUB/'code/current_window_projection.py'),COST_INPUT_PROOF,*COST_INPUTS,*[relative(PUB/n) for n in names]]))

def verify_projection_inputs():
    from scripts.selected_risk_common import ROOT,PUB,read,sha,verify_pin
    from echo_routing.temporal.selected_risk_forecast import verify_cost_inputs
    from echo_routing.temporal.selected_risk_cost_units import remaining
    import math
    binding=read(PUB/'current_window_projection_binding.json');p=read(PUB/'current_window_projection_readonly.json');meter=read(PUB/'admin_current_window_projection_v2_CPU.json')
    expected={'projection_sha256':'current_window_projection_readonly.json','actual_local_measured_wrapper_sha256':'admin_current_window_projection_v2_CPU.json','input_manifest_sha256':'input_manifest.json','cost_profile_sha256':'cost_profile.json','cost_shape_profile_sha256':'cost_shape_profile.json'}
    if any(binding[k]!=sha(PUB/n) for k,n in expected.items()) or binding['script_path']!='docs/results/2026-11-05_selected_action_risk/code/current_window_projection.py':raise ValueError('projection source/input/result/meter binding differs')
    verify_pin({'path':binding['script_path'],'sha256':binding['script_sha256']})
    if meter['actual_child_OS_exit_code']!=0 or not meter['started_epoch']<=p['now_epoch']<=meter['sampled_epoch']:raise ValueError('projection clock must be actually bracketed by measured OS0 wrapper')
    verify_cost_inputs(ROOT);old=PUB.parent/'2026-11-04_anchor_context_risk';pr=read(old/'producer_manifest.json');wall=read(old/'process_producer_1.json')['full_process_wall_seconds'];u={n:sum(r.get('committed_logical_neural_units',r.get('actual_neural_forward_units',0)) for r in pr['records'] if r['source']==n) for n in ['a08','jepa']}
    work=remaining(read(PUB/'cost_profile.json'),read(PUB/'cost_shape_profile.json'),read(PUB/'input_manifest.json')['roles'],old_source_wall=wall,old_a08_units=u['a08'],old_jepa_units=u['jepa'],current_upper=436049885,sessions=['synthetic_session_'+str(i) for i in range(402)])
    for k,key in [('CPU_remaining_seconds','CPU_seconds'),('remaining_wall_seconds','remaining_wall_seconds'),('source_FIT_GPU_forecast','source_FIT_GPU_forecast'),('GPU_remaining','GPU_seconds')]:
        if not math.isclose(p[k],work[key],rel_tol=0,abs_tol=1e-8):raise ValueError('original reviewed full-work CPU/wall recount differs')
    return p

def freeze():
    from scripts.selected_risk_common import ROOT,RUN,PUB,CFG,read,verify_pin,verify_freeze,validate_frozen_prior,sha,pin,write,relative,verify_Git
    from echo_routing.temporal.safety_git import build
    from echo_routing.temporal.selected_risk_storage_engineering import pack_proof,expand_proof
    import subprocess
    target=PUB/'local_resource_stop_identity.json'
    if target.exists():raise ValueError('never overwrite prepared/executed local closure')
    old=verify_freeze('metadata');objects={p:sha(ROOT/p) for p in _mandatory()};validate_frozen_prior(objects);refs=read(verify_pin(read(CFG)['source_refs']));current=dict(objects);historical={}
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    for p,v in refs['historical_frozen_imports'].items():
        if p not in current:continue
        if current.pop(p)!=v['sha256']:raise ValueError('historical overlay drift')
        proof=build(ROOT,v['git_commit'],[p]);verify_Git(proof,v['git_commit'],{p:v['sha256']});fp=RUN/'local_closure_Git_proofs'/(Path(p).name+'.json');write(fp,proof);historical[p]={**v,'proof':pin(fp)}
    full=build(ROOT,commit,current);pool=read(verify_pin(old['actual_Git_object_proof']));packed=pack_proof(full,pool)
    if expand_proof(packed,pool)!=full:raise ValueError('lossless original Git proof required')
    verify_Git(full,commit,current);pp=RUN/'local_closure_Git_proofs/current.shared.private.json';write(pp,packed)
    value={'stage':'local_resource_stop_v1','git_commit':commit,'objects':objects,'historical_frozen_imports':historical,'actual_Git_object_proof':pin(pp),'shared_object_pool':old['actual_Git_object_proof'],'scope':'local evidence closure only; no remote launch or fresh remote validation'};write(target,value);verify_contract();return value

def verify_contract():
    from scripts.selected_risk_common import ROOT,PUB,CFG,read,verify_pin,verify_freeze,validate_frozen_prior,sha,verify_Git
    from echo_routing.temporal.selected_risk_storage_engineering import expand_proof
    c=read(PUB/'local_resource_stop_identity.json');old=verify_freeze('metadata');refs=read(verify_pin(read(CFG)['source_refs']));objects=c['objects']
    if c['stage']!='local_resource_stop_v1' or c['shared_object_pool']!=old['actual_Git_object_proof'] or not _mandatory()<=set(objects):raise ValueError('complete independent local stop closure required')
    validate_frozen_prior(objects);expected=dict(objects)
    for p,h in objects.items():
        verify_pin({'path':p,'sha256':h})
    for p,v in c['historical_frozen_imports'].items():
        if expected.pop(p)!=v['sha256'] or refs['historical_frozen_imports'].get(p)!={k:v[k] for k in ['git_commit','blob_oid','sha256']}:raise ValueError('historical proof scope differs')
        verify_Git(read(verify_pin(v['proof'])),v['git_commit'],{p:v['sha256']})
    verify_Git(expand_proof(read(verify_pin(c['actual_Git_object_proof'])),read(verify_pin(c['shared_object_pool']))),c['git_commit'],expected)
    return c

def collect():
    from scripts.selected_risk_common import ROOT,RUN,PUB,read,sha,write,relative,actual_receipts,verify_freeze
    from scripts.selected_risk_shared_contracts import verify_shared_freeze
    from scripts.a63_selected_risk_costunit_contract import verify_shared_freeze as verify_units,STAGE
    from echo_routing.temporal.selected_risk_budget import totals
    identity=verify_contract()
    if any((PUB/n).exists() for n in ['resources_resource_stopped.json','terminal_resource_stopped.json','local_resource_stop_CPU_snapshot.json']):raise ValueError('one immutable local stop collection only')
    contracts={}
    for name in ['metadata','cost_probe','sources_admission']:contracts[name]=verify_freeze(name)['git_commit']
    contracts['cost_shape_probe']=verify_shared_freeze('cost_shape_probe')['git_commit'];contracts[STAGE]=verify_units(STAGE)['git_commit']
    parent=read(PUB/'late_sourcegate_v3_parent_observation.json');raw=(PUB/'costunit_sourcegate_launch_v1.stdout').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=parent['actual_parent_stdout_sha256']:raise ValueError('actual parent observation raw changed')
    observed=[json.loads(s)['sources_admission_enclosing'] for s in raw.decode().splitlines() if '"sources_admission_enclosing"' in s]
    if len(observed)!=1 or observed[0]!=parent['actual_enclosing_receipt']:raise ValueError('recovered semantic receipt differs from actual directly-waited parent stdout')
    recovered=recovered_parent_rows(observed[0]);known=read(PUB/'local_known_worker_receipts.json');server=[json.loads(s) for s in (RUN/'processes.jsonl').read_text().splitlines()]
    workers=merge_receipts(server,observed[0]['new_process_receipts'])
    if workers!=known['receipts']:raise ValueError('known actual remote worker bank differs from preserved raw')
    base=actual_receipts();rows=merge_receipts(base,recovered)
    stop=read(PUB/'execution_stop_decision.json');projection=verify_projection_inputs();state=build_stop_state(stop,projection,rows,workers)
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime;sample=time.time()
    row={'receipt_id':'local_resource_stop_snapshot:'+str(os.getpid()),'measurement_kind':'administrative_partial_receipt','measurement_host':socket.gethostname(),'model_pid':os.getpid(),'started_epoch':PROCESS_START,'finished_epoch':sample,'full_process_wall_seconds':sample-PROCESS_START,'measured_CPU_core_seconds':cpu,'CPU_accounting_allowance_seconds':0.,'GPU_job':False,'CUDA_context_occupied':False,'observed_worker_exit_code':None,'SELF_OS_unobserved':True}
    total=totals(rows+[row]);state=build_stop_state(stop,projection,rows+[row],workers);snapshot={'whole_fresh_process_CPU_through_snapshot':cpu,'SELF_OS_exit_code':None,'snapshot_epoch':sample,'process_PID':os.getpid(),'collector_receipt':row,'imports_included':True,'serialization_exit_CPU_not_in_snapshot':True,'snapshot_wall_interval_excludes_pre_main_stdlib_import_wall':True}
    resources={'mode':'resource-stopped','known_actual_costs_through_local_snapshot':total,'local_snapshot_CPU_seconds':cpu,'actual_remote_worker_count':len(workers),'remote_workers':workers,'executed_contracts_local_current_SHA_and_Git_verified':contracts,'identity_sha256':sha(PUB/'local_resource_stop_identity.json'),'remote_current_boot_PID_GPUowners_active_pending_du':'unknown; bounded SSH failed; last direct wait retained','latest_remote_stage_receipt_source':'actual parent stdout; original v3 result/receipt JSON not mirrored','fresh_remote_collector_executed':False,'source_aux_formal_updates':stop['source_counters'],'coverage':None,'storage_assessment':'root separate scoped measurement/addendum; not inferred from forecast','actual_caps_fit':None,'overall_complete':False,'root_protected_publication_pending':True,'no_A64':True}
    write(PUB/'local_resource_stop_CPU_snapshot.json',snapshot);write(PUB/'terminal_resource_stopped.json',state)
    resources['terminal_state_sha256']=sha(PUB/'terminal_resource_stopped.json');resources['CPU_snapshot_sha256']=sha(PUB/'local_resource_stop_CPU_snapshot.json');write(PUB/'resources_resource_stopped.json',resources)
    print(json.dumps({'local_collection_complete':True,'overall_complete':False,'snapshot_CPU_seconds':cpu,'remote_actual_workers':len(workers),'source_aux_formal':stop['source_counters']}),flush=True);return 0

def main():
    from scripts.selected_risk_common import execution_parser,authorize
    p=execution_parser(__doc__);p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    if not authorize(a):return 0
    if a.freeze_only:print(json.dumps({'local_prepared_identity':freeze()['git_commit']}),flush=True);return 0
    return collect()

PROCESS_START=time.time()
if __name__=='__main__':raise SystemExit(main())
