"""Synthetic metadata witnesses only: no probe/NN/optimizer/private model access."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from echo_routing.temporal.selected_risk_completed_supply_controller import CompletedSupplyRuntimeState
from echo_routing.temporal.selected_risk_resume_contract import RUN, PUB, digest
from echo_routing.temporal.selected_risk_resume_S import SEEDS

SOURCE='echo_routing/temporal/selected_risk_measured_cache_controller.py'


def api():
    p=Path(__file__).resolve().parents[1]/SOURCE
    spec=importlib.util.spec_from_file_location('private_measured_cache',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Context:
    def __init__(self,root):
        self.root=root;self.owner={'mock':'same window'};self.owner_pin=self.put('owner.json',self.owner);self.owner_sha=self.owner_pin['sha256'];self.freeze={'objects':{}}
    def put(self,path,value):
        raw=value if isinstance(value,bytes) else json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        return {'path':path,'sha256':hashlib.sha256(raw).hexdigest()}
    def raw(self,pin):
        assert set(pin)=={'path','sha256'}
        p=self.root/pin['path']
        if p.is_symlink() or not p.is_file():raise ValueError('mock pinned file missing')
        raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=pin['sha256']:raise ValueError('mock current SHA mismatch')
        return raw
    def read(self,pin):return json.loads(self.raw(pin))
    def evidence(self,desc):return {'payload':self.read(desc['payload_pin'])}
    def enter(self,stage,freeze,descriptors,*,required):
        for path in required:
            if path not in self.freeze['objects']:raise ValueError('mock execution freeze source absent')
            self.raw({'path':path,'sha256':self.freeze['objects'][path]})
        return self.freeze,{key:self.evidence(desc) for key,desc in descriptors.items()}
    def closed_publication_path(self,path):return path
    def contract_proof(self,contract,pin):return self.read(pin)


def case(tmp_path,monkeypatch,cpus=(2.,3.,2.5)):
    m=api();ctx=Context(tmp_path)
    import echo_routing.temporal.selected_risk_resume_runtime as runtime
    qualified=Path(runtime.__file__).resolve().parents[2]
    old={}
    for path,expected in {**m._OLD8_PINS,**m._FORMULA_PINS}.items():
        raw=(qualified/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
        pin=ctx.put(path,raw);old[path]=pin['sha256']
    ctx.freeze['objects'].update(old)
    for path in m._EXECUTION_SOURCES:
        if path in old:continue
        if path==SOURCE:raw=Path(m.__file__).read_bytes()
        elif path==m._NEW_ROUTE:
            raw=(qualified/m._BASE_ROUTE).read_text().replace('completed_supply','measured_cache').replace('CompletedSupplyRuntimeState','MeasuredCacheRuntimeState').encode()
            # The expanded mandatory source set is checked independently.
            import ast
            t=ast.parse(raw);n=next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='EXTRA_SOURCES' for x in n.targets));lines=raw.decode().splitlines(keepends=True)
            raw=(''.join(lines[:n.lineno-1])+'EXTRA_SOURCES='+repr(tuple(m._EXECUTION_SOURCES))+'\n'+''.join(lines[n.end_lineno:])).encode()
        else:raw=b'# synthetic metadata-only source fixture\n'
        pin=ctx.put(path,raw);ctx.freeze['objects'][path]=pin['sha256']
    profile_raw=(qualified/m._PROFILE_PIN['path']).read_bytes();assert hashlib.sha256(profile_raw).hexdigest()==m._PROFILE_PIN['sha256'];ctx.put(m._PROFILE_PIN['path'],profile_raw);profile=json.loads(profile_raw)
    geometry_path=PUB+'/actual_FIT_geometry.json';geometry_raw=(qualified/geometry_path).read_bytes();geometry=json.loads(geometry_raw);geometry_pin=ctx.put(geometry_path,geometry_raw)
    metadata=ctx.put('input_manifest.json',{'roles':'synthetic metadata placeholder'})
    producer=ctx.put('FIT_producer.json',{'synthetic':True})
    geometry['input_manifest_sha256']=metadata['sha256'];geometry['producer_FIT_payload_sha256']=producer['sha256'];geometry_pin=ctx.put(geometry_path,geometry)
    spec={'stage':'seal_aux','metadata':metadata,'FIT_producer':producer,'CONFIRM_producer':None,'freeze':{'path':'newfreeze.json','sha256':'0'*64},'predecessors':{},'arguments':{'stage_CPU_upper_seconds':900.,'stage_wall_upper_seconds':900.},'expected_boot_id':'fixture-boot'}
    workers=[]
    def closed(stage,value,exitcode=0):
        pin=ctx.put(PUB+'/mock_'+stage+'.json',value);spec['predecessors'][stage]={'payload_pin':pin,'runtime_authorization_sha256':ctx.owner_sha}
        stdout=ctx.put('mock_logs/'+stage+'.stdout',{'stage':stage,'stage_returned_success':True,'runtime_authorization_sha256':ctx.owner_sha,'output':pin})
        row={'stage':stage,'model_pid':100+len(workers),'stage_exit_status':exitcode,'observed_worker_exit_code':exitcode,'inner_model_observed_OS_exit_code':0,'whole_child_accounting_complete':True,'actual_child_directly_waited':True,'unresolved_worker_uncertainty':False,'runtime_authorization_sha256':ctx.owner_sha,'log_pins':[stdout]}
        workers.append(row);return pin,row
    closed('adopt_metadata',{'passed':True});closed('admit_FIT',{'all_caps_fit':True})
    closed('source_FIT',{'complete':True,'phase':'FIT','committed_payloads':6000,'aux_updates_at_production':0,'formal_updates_at_production':0},1)
    closed('source_FIT_adoption',{'passed':True,'complete':True,'committed_payloads':6000,'original_source_stage_exit_status':1,'original_teacher_inner_OS_exit_code':0,'original_source_stage_cap_failure_preserved':True,'new_teacher_forwards':0,'new_optimizer_updates':0,'cache_writes':0})
    spec['predecessors']['observe_geometry']={'payload_pin':geometry_pin,'runtime_authorization_sha256':ctx.owner_sha}
    log=ctx.put('mock_logs/observe_geometry.stdout',{'stage':'observe_geometry','stage_returned_success':True,'runtime_authorization_sha256':ctx.owner_sha,'output':geometry_pin});workers.append({'stage':'observe_geometry','model_pid':120,'stage_exit_status':0,'observed_worker_exit_code':0,'inner_model_observed_OS_exit_code':0,'whole_child_accounting_complete':True,'actual_child_directly_waited':True,'unresolved_worker_uncertainty':False,'runtime_authorization_sha256':ctx.owner_sha,'log_pins':[log]})
    closed('admit_aux',{'all_caps_fit':True})
    training,_=closed('prepare_aux',{'passed':True,'folds':[{'held_fold':0},{'held_fold':1}]})
    models=[];records=[]
    for seed in SEEDS:
        for held in [0,1]:
            for role in ['utility','risk']:
                prefix=RUN+f'/aux/seed{seed}/held{held}/{role}'
                states=[ctx.put(prefix+'/'+name,b'opaque synthetic state') for name in ['step2000.pt','step4000.pt','step6000.pt','latest.pt']]
                identity={key:digest([seed,held,role,key]) for key in ['code','config','sources','input','draw_plan','normalizer','target_scales']}
                complete=ctx.put(prefix+'/completion.json',{'passed':True,'physical_updates':6000,'fullstate_count':4,'identity':identity,'state_pins':states})
                models.append({'seed':seed,'held_fold':held,'role':role,'completion':complete})
                records.append({'seed':seed,'held_fold':held,'role':role,'completion':complete,'identity':identity,'fullstate_pins_and_bytes':[{'pin':p,'bytes':len(ctx.raw(p))} for p in states]})
    train,row=closed('train_aux',{'passed':True,'models':models,'aux_optimizer_updates':48000,'formal_optimizer_updates':0,'physical_models':8,'fullstate_count':32})
    train_parent=ctx.put('train_parent.json',row)
    audit,_=closed('audit_aux',{'passed':True,'aux_optimizer_updates':48000,'physical_models':8,'fullstate_count':32})
    proofpin=ctx.put('closed_aux_proof.json',{'schema':'A63_closed_aux_storage_materialization_readonly_v1','passed':True,'complete_train_manifest':train,'actual_train_parent_OS_receipt':train_parent,'physical_updates':48000,'fullstate_count':32,'records':records})
    spec['arguments']['completed_aux_storage_proof']=proofpin
    core={'models_manifest':ctx.put(PUB+'/aux_models_manifest.json',{'models':'opaque metadata'}),'audit':audit,'training_inputs':training}
    science={'stage':'seal_aux','metadata':metadata,'FIT_producer':producer,'expected_boot_id':'fixture-boot','arguments':{'core':core},'predecessors':copy.deepcopy(spec['predecessors'])}
    oldfreeze=ctx.put('old_source_freeze.json',{'objects':dict(old)})
    science['freeze']=oldfreeze;sciencepin=ctx.put('old_science_spec.json',science)
    oldctx=SimpleNamespace(owner_sha=ctx.owner_sha,raw=ctx.raw,read=ctx.read)
    def verify_old(stage,pin,*,required):
        assert pin==oldfreeze and set(m._OLD8_PINS)<=set(required)
        for path,sha in old.items():ctx.raw({'path':path,'sha256':sha})
        return {'objects':dict(old)}
    oldctx.verify_stage=verify_old
    def bootstrap(root,pin,owner):
        if pin!=sciencepin or owner!=ctx.owner_sha:raise ValueError('mock wrong original context')
        return oldctx,science
    monkeypatch.setattr(runtime,'bootstrap',bootstrap)
    import echo_routing.temporal.safety_git as git
    def verify(proof,commit,objects):
        if proof.get('objects')!=objects or commit!='c'*40:raise ValueError('mock tiny Git proof invalid')
    monkeypatch.setattr(git,'verify',verify)
    output=RUN+'/cost_only_synthetic_cache_probe/fixture'
    probe_spec={'schema':'A63_full_cache_verifier_cost_probe_spec_v1','purpose':'cost_only_synthetic_nonadoptable','nonce':'fixture','runtime_authorization_sha256':ctx.owner_sha,'source_runtime_spec_pin':sciencepin,'input_manifest_pin':metadata,'geometry_pin':geometry_pin,'core_pins':core,'output_prefix':output,'limits':{'CPU_seconds':600,'wall_seconds':600,'temp_bytes':80000000}}
    new={path:ctx.freeze['objects'][path] for path in m._PROBE_SOURCES}
    gitpin=ctx.put(output+'/tiny_git_proof.json',{'objects':new})
    contract=ctx.put(output+'/source_contract.json',{'schema':'A63_full_cache_verifier_cost_probe_sources_v1','runtime_authorization_sha256':ctx.owner_sha,'readonly_source_runtime_spec_pin':sciencepin,'readonly_source_freeze_pin':oldfreeze,'objects':new,'git_commit':'c'*40,'actual_Git_object_proof':gitpin,'historical_frozen_imports':{},'combined_source_objects_sha256':digest({**old,**new}),'probe_spec_semantics_sha256':digest(probe_spec)})
    probe_spec['source_Git_contract_pin']=contract;probe_spec_pin=ctx.put(output+'/spec.json',probe_spec)
    source_pins=[{'path':p,'sha256':sha} for p,sha in sorted({**old,**new}.items()) if p.endswith('.py')]
    bindings={'probe_spec_pin':probe_spec_pin,'source_runtime_spec_pin':sciencepin,'source_Git_contract_pin':contract,'input_manifest_pin':metadata,'geometry_pin':geometry_pin,'core_pins':core,'source_pins':source_pins}
    pool={'purpose':'cost_only_synthetic_nonadoptable','axes':5736,'admitted_G':8111,'ALL_G':8522,'seeds':list(SEEDS),'cache_records':8}
    cache_records=[]
    for seed in SEEDS:
        for role,g in geometry['roles'].items():
            cache_records.append({'seed':seed,'role':role,'axes':g['geometry']['N'],'admitted_G':g['geometry']['C'],'relative_path':f'seed{seed}/{role}','feature_reference_seed':None if seed==SEEDS[0] else SEEDS[0]})
    cachemanifest=ctx.put(output+'/cache_manifest.private.json',{'schema':'A63_cost_only_cache_manifest_v1','purpose':pool['purpose'],'records':cache_records})
    files=[{'path':f'{i:08d}.npz','sha256':'f'*64,'bytes':1} for i in range(11472)]+[{'path':f'meta{i}.json','sha256':'e'*64,'bytes':1} for i in range(16)]
    deleted=ctx.put(output+'/deleted_files.json',files);flags={'NN_calls':0,'SGD_updates':0,'model_instances':0,'GT_or_head_effect_used_for_cost':False,'metadata_label_parse_exception':'synthetic fixture: no actual labels read'}
    def process(index,cpu,prepare=False):
        summary={'schema':'A63_cost_only_cache_prepare_summary_v1' if prepare else 'A63_cost_only_cache_verifier_summary_v1','passed':True,'complete':True,'bindings':bindings,'synthetic_pool':pool,'cache_manifest_pin':cachemanifest,**flags}
        if prepare:summary.update(deleted_file_manifest_pin=deleted,payload_regular_bytes=len(files),structural_peak_upper_bytes=79000000,NPZ_STORED=True)
        else:summary.update(verified_payloads=11472,read_only_cache_payloads_verified=True,canonical_actions_verified=True,reference_raw_features_bitwise_verified=True,no_pending=True)
        pin=ctx.put(output+f'/summary{index}.json',summary)
        stdout=ctx.put(output+f'/child{index}.stdout',{'passed':True,'output_pin':pin});stderr=ctx.put(output+f'/child{index}.stderr',b'')
        row={'pid':600+index,'start_ticks':100+index,'boot_id':'fixture-boot','actual_OS_exit_code':0,'CPU_core_seconds':cpu,'wall_seconds':cpu+1.,'full_accounting_complete':True,'actual_directly_waited':True,'raw_stdout_pin':stdout,'raw_stderr_pin':stderr,'GPU_context_unknown':False,'CUDA_context_observed':False,'unresolved_worker_uncertainty':False}
        raw=ctx.put(output+f'/child{index}.json',row);return {**row,'raw_receipt_pin':raw}
    prepared=process(0,5.,True);three=[process(i+1,cpu) for i,cpu in enumerate(cpus)]
    cleanup={'owned_payload_root':output+'/payloads','namespace':pool['purpose'],'before_regular_bytes':len(files),'after_regular_bytes':0,'fully_removed':True,'owned_only':True,'deleted_file_manifest_pin':deleted}
    payload={'schema':'A63_full_cache_verifier_cost_probe_v1','passed':True,'runtime_authorization_sha256':ctx.owner_sha,**bindings,'synthetic_pool':pool,'cache_manifest_pin':cachemanifest,'prepare_process':prepared,'three_fresh_verifier_processes':three,'cleanup':cleanup,**flags}
    parent={'schema':'A63_cost_probe_outer_wait4_receipt_v1','stage':'cost_only_cache_probe','runtime_authorization_sha256':ctx.owner_sha,'probe_spec_pin':probe_spec_pin,'source_runtime_spec_pin':sciencepin,'source_Git_contract_pin':contract,'actual_child_OS_exit_code':0,'actual_child_directly_waited':True,'whole_child_accounting_complete':True,'stage_exit_status':0,'observed_worker_exit_code':0,'model_pid':500,'model_start_ticks':50,'boot_id':'fixture-boot','started_epoch':1000.,'finished_epoch':1500.,'full_process_wall_seconds':500.,'measured_CPU_core_seconds':5.+sum(cpus)+10.,'sampled_temp_peak_bytes':len(files),'SELF_OS_exit_code':None,'GPU_job':False,'CUDA_context_occupied':False,'unresolved_worker_uncertainty':False,'cuda_context_observation':{'known':True,'initialized':False},'caller_exit_CPU_allowance_seconds':1.,'prepare_process_raw_receipt_pin':prepared['raw_receipt_pin'],'three_fresh_verifier_raw_receipt_pins':[r['raw_receipt_pin'] for r in three]}
    def rewrite():
        payloadpin=ctx.put(output+'/payload.json',payload);parent['payload_pin']=payloadpin;parent['successful_stdout_pin']=ctx.put(output+'/parent.stdout',{'passed':True,'payload_pin':payloadpin})
        parentpin=ctx.put(output+'/parent.json',parent)
        spec['arguments']['measured_cache_probe']=payloadpin;spec['arguments']['measured_cache_probe_parent']=parentpin
        if parent not in workers:workers.append(parent)
    rewrite()
    from echo_routing.temporal.selected_risk_forecast import project,add,backend
    from echo_routing.temporal.selected_risk_cost_units import project_coefficients
    coef,coefproof=project_coefficients(profile)
    projected={r:project(profile['geometry']['mixed'],{'positions':220448,'sequences':1500,'max_length':1000}) for r in ['F0_mixed','F1_mixed']}
    oldB=backend(add([profile['geometry']['native'],*[projected[r] for r in ['F0_mixed','F1_mixed']]]),coef)
    phases={'complement_prepare_and_independent_recount':4*oldB,'D_X_held_score_all_features_2seeds_and_cache_IO':8*oldB,'selected_effects_signatures_coverage_recount':4*oldB,'completed_supply_cache_revalidation_remaining_CPU':3000.,'completed_aux_revalidation_remaining_CPU':300.,'continuation_provenance_codec_and_protocol_CPU':360.,'resource_OS_independent_audit_publication':3600.,'unknown_CONFIRM_and_outer_unchanged':2800.}
    storage={'current_namespace_plus_admin_upper':900000000,'retained_fullstates_atomic_upper':12500000,'closing_reserve':128000000,'completed_aux_controller_compact_future_metadata_upper':33554432}
    work={'phases':phases,'CPU_seconds':sum(phases.values()),'remaining_wall_seconds':sum(phases.values())+3600.,'GPU_seconds':400.,'storage_components':storage,'storage_upper_bytes':sum(storage.values()),'projected_geometry':projected,'actual_four_profile_projection_proof':coefproof,'completed_aux_future_trace_refinement':{'proof_pin':proofpin},'scope':'same full matrix'}
    base={'remaining_complete_workload':work,'already_spent':{'legacy':True},'actual_storage_cap':1300000000,'all_caps_fit':False,'evaluated_epoch':1.}
    monkeypatch.setattr(CompletedSupplyRuntimeState,'admission',lambda self:copy.deepcopy(base))
    state=m.MeasuredCacheRuntimeState(ctx,spec,{'path':'new_spec.json','sha256':'9'*64})
    monkeypatch.setattr(state,'_worker_rows',lambda:workers);monkeypatch.setattr(state,'spent',lambda:{'fresh_actual_spent':True})
    import echo_routing.temporal.selected_risk_resume_budget as budget
    calls=[]
    def forecast(*a,**kw):calls.append((a,kw));return {'all_caps_fit':True}
    monkeypatch.setattr(budget,'forecast',forecast);monkeypatch.setattr(budget,'remaining_limits',lambda *a,**kw:{'remaining_CPU_seconds':1000.,'stage_wall_seconds':1000.})
    return SimpleNamespace(m=m,ctx=ctx,spec=spec,state=state,work=work,base=base,payload=payload,parent=parent,workers=workers,rewrite=rewrite,calls=calls,closed=closed)


def test_actual_geometry_measured_quantity_and_state_unit_retirement_preserve_other_work(tmp_path,monkeypatch):
    c=case(tmp_path,monkeypatch);r=c.state.admission();w=r['remaining_complete_workload'];detail=w['measured_cache_future_cost_refinement']
    assert detail['remaining_guardian_calls']==22 and detail['remaining_admission_child_calls']==3 and detail['remaining_full_verifier_calls']==25
    assert w['phases']['completed_supply_cache_revalidation_remaining_CPU']==2*3.*25+60
    assert w['phases']['complement_prepare_and_independent_recount']==pytest.approx(824.5957261863382)
    assert w['phases']['D_X_held_score_all_features_2seeds_and_cache_IO']==pytest.approx(1649.1914523726764)
    assert w['phases']['selected_effects_signatures_coverage_recount']==pytest.approx(824.5957261863382)
    assert w['storage_components']['retained_fullstates_atomic_upper']==4500000
    assert c.work['storage_upper_bytes']-w['storage_upper_bytes']==8000000
    assert {k:v for k,v in w['storage_components'].items() if k!='retained_fullstates_atomic_upper'}=={k:v for k,v in c.work['storage_components'].items() if k!='retained_fullstates_atomic_upper'}
    changed={'complement_prepare_and_independent_recount','D_X_held_score_all_features_2seeds_and_cache_IO','selected_effects_signatures_coverage_recount','completed_supply_cache_revalidation_remaining_CPU'}
    assert {k:v for k,v in w['phases'].items() if k not in changed}=={k:v for k,v in c.work['phases'].items() if k not in changed}
    delta=sum(w['phases'][k]-c.work['phases'][k] for k in changed)
    assert w['CPU_seconds']==pytest.approx(c.work['CPU_seconds']+delta)
    assert w['remaining_wall_seconds']==pytest.approx(c.work['remaining_wall_seconds']+delta+50.)
    assert detail['measured_future_CPU_room_seconds']==210.
    assert detail['measured_future_wall_room_seconds']==260.
    assert detail['raw_three_verifier_CPU_seconds']==[2.,3.,2.5]
    assert detail['raw_three_verifier_wall_seconds']==[3.,4.,3.5]
    assert w['GPU_seconds']==400. and r['already_spent']=={'fresh_actual_spent':True}
    assert c.calls[0][0][6]==c.ctx.owner and r['actual_storage_cap']==1300000000
    c.state.stage='close_full';c.spec['stage']='close_full';del c.spec['arguments']['measured_cache_probe'];assert c.state.admission()==c.base


def test_slow_qualified_probe_is_not_clipped_to_old_room(tmp_path,monkeypatch):
    c=case(tmp_path,monkeypatch,(80.,79.,78.));w=c.state.admission()['remaining_complete_workload']
    assert w['phases']['completed_supply_cache_revalidation_remaining_CPU']==4060.>3000.
    assert w['measured_cache_future_cost_refinement']['measured_future_wall_room_seconds']==4110.


def test_missing_failed_foreign_or_incomplete_measurement_and_bad_layout_reject(tmp_path,monkeypatch):
    c=case(tmp_path,monkeypatch)
    for key in ['GPU_job','CUDA_context_occupied','unresolved_worker_uncertainty']:
        original=c.parent.pop(key);c.rewrite()
        with pytest.raises(ValueError):c.state.admission()
        c.parent[key]=True;c.rewrite()
        with pytest.raises(ValueError):c.state.admission()
        c.parent[key]=original;c.rewrite()
    c.parent['cuda_context_observation']['known']=False;c.rewrite()
    with pytest.raises(ValueError):c.state.admission()
    c.parent['cuda_context_observation']['known']=True;c.parent['cuda_context_observation']['initialized']=True;c.rewrite()
    with pytest.raises(ValueError):c.state.admission()
    c.parent['cuda_context_observation']['initialized']=False;c.rewrite()
    # A process's own current raw wait record must explicitly establish no GPU.
    for entry in [c.payload['prepare_process'],*c.payload['three_fresh_verifier_processes']]:
        for key in ['GPU_context_unknown','CUDA_context_observed','unresolved_worker_uncertainty']:
            saved=entry.pop(key);rawpin=entry['raw_receipt_pin'];entry['raw_receipt_pin']=c.ctx.put(rawpin['path'],{k:v for k,v in entry.items() if k!='raw_receipt_pin'})
            if entry is c.payload['prepare_process']:c.parent['prepare_process_raw_receipt_pin']=entry['raw_receipt_pin']
            else:c.parent['three_fresh_verifier_raw_receipt_pins']=[r['raw_receipt_pin'] for r in c.payload['three_fresh_verifier_processes']]
            c.rewrite()
            with pytest.raises(ValueError):c.state.admission()
            entry[key]=True;entry['raw_receipt_pin']=c.ctx.put(rawpin['path'],{k:v for k,v in entry.items() if k!='raw_receipt_pin'})
            if entry is c.payload['prepare_process']:c.parent['prepare_process_raw_receipt_pin']=entry['raw_receipt_pin']
            else:c.parent['three_fresh_verifier_raw_receipt_pins']=[r['raw_receipt_pin'] for r in c.payload['three_fresh_verifier_processes']]
            c.rewrite()
            with pytest.raises(ValueError):c.state.admission()
            entry[key]=saved;entry['raw_receipt_pin']=c.ctx.put(rawpin['path'],{k:v for k,v in entry.items() if k!='raw_receipt_pin'})
            if entry is c.payload['prepare_process']:c.parent['prepare_process_raw_receipt_pin']=entry['raw_receipt_pin']
            else:c.parent['three_fresh_verifier_raw_receipt_pins']=[r['raw_receipt_pin'] for r in c.payload['three_fresh_verifier_processes']]
            c.rewrite()
    c.parent['actual_child_OS_exit_code']=7;c.rewrite()
    with pytest.raises(ValueError):c.state.admission()
    c.parent['actual_child_OS_exit_code']=0;c.rewrite()
    pin=c.spec['arguments'].pop('measured_cache_probe_parent')
    with pytest.raises(ValueError):c.state.admission()
    c.spec['arguments']['measured_cache_probe_parent']=pin
    c.payload['cleanup']['after_regular_bytes']=1;c.rewrite()
    with pytest.raises(ValueError):c.state.admission()
    c.payload['cleanup']['after_regular_bytes']=0;c.rewrite()
    row=c.payload['three_fresh_verifier_processes'].pop();c.rewrite()
    with pytest.raises(ValueError):c.state.admission()
    c.payload['three_fresh_verifier_processes'].append(row);c.rewrite()
    c.work['phases']['complement_prepare_and_independent_recount']+=1
    with pytest.raises(ValueError):c.state.admission()
    c.work['phases']['complement_prepare_and_independent_recount']-=1
    c.work['storage_components']['retained_fullstates_atomic_upper']-=1
    with pytest.raises(ValueError):c.state.admission()
