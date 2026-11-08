"""New supply controller only; mock authority/Git/budget, real tiny readonly caches."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from echo_routing.temporal.selected_risk_completed_aux_controller import CompletedAuxRuntimeState
from echo_routing.temporal.selected_risk_resume_contract import RUN, PUB, digest
from echo_routing.temporal.selected_risk_resume_S import SEEDS
from echo_routing.temporal.selected_risk_resume_supply import POST_NATIVE_PINS

SOURCE = 'echo_routing/temporal/selected_risk_completed_supply_controller.py'
COMPONENT = 'minimal_aux_plus_compact_shared_cache_phase_peak'
ROLES = ['F0_native','F0_mixed','F1_native','F1_mixed']


def api():
    path = Path(__file__).resolve().parents[1]/SOURCE
    spec = importlib.util.spec_from_file_location('private_completed_supply', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class Context:
    def __init__(self, root):
        self.root=root; self.owner={'scope':'synthetic'}; self.entered=[]
        self.owner_pin=self.put('owner.json',self.owner); self.owner_sha=self.owner_pin['sha256']
        self.proposal_pin=self.put('proposal.json',{'scope':'synthetic'})
        self.baseline_pin=self.put('baseline.json',{'scope':'synthetic'})
        self.freeze={'objects':{p['path']:p['sha256'] for p in [self.owner_pin,self.proposal_pin,self.baseline_pin]}}
    def put(self,path,value):
        raw=value if isinstance(value,bytes) else (json.dumps(value,sort_keys=True)+'\n').encode()
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        return {'path':path,'sha256':hashlib.sha256(raw).hexdigest()}
    def raw(self,pin):
        assert set(pin)=={'path','sha256'}
        p=self.root/pin['path']
        if p.is_symlink() or not p.is_file():raise ValueError('mock ordinary current pinned file missing')
        raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=pin['sha256']:raise ValueError('mock current SHA mismatch')
        return raw
    def read(self,pin):return json.loads(self.raw(pin))
    def evidence(self,desc):return {'payload':self.read(desc['payload_pin'])}
    def enter(self,stage,freeze,descriptors,*,required):
        self.entered.append(set(required))
        for path in required:
            if path not in self.freeze['objects']:raise ValueError('mock actual-Git source omitted')
            self.raw({'path':path,'sha256':self.freeze['objects'][path]})
        return self.freeze,{k:self.evidence(v) for k,v in descriptors.items()}
    def contract_proof(self,contract,pin):return self.read(pin)
    def _historical_expected(self):return {}


def case(tmp_path,monkeypatch,held=False):
    module=api();ctx=Context(tmp_path)
    from echo_routing.temporal import selected_risk_resume_runtime as runtime
    qualified=Path(runtime.__file__).resolve().parents[2]
    old={}
    for path,sha in {**module._LAYOUT_PINS,**module._OLD_EXTRA_PINS}.items():
        raw=(qualified/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
        pin=ctx.put(path,raw);ctx.freeze['objects'][path]=pin['sha256']
        if path in module._LAYOUT_PINS:old[path]=pin['sha256']
    pin=ctx.put(SOURCE,Path(module.__file__).read_bytes());ctx.freeze['objects'][SOURCE]=pin['sha256']
    metadata=ctx.put('metadata.json',{'input':'mock'});producer=ctx.put('producer.json',{'input':'mock'})
    spec={'stage':'seal_aux','metadata':metadata,'FIT_producer':producer,'CONFIRM_producer':None,'freeze':{'path':'newfreeze.json','sha256':'a'*64},'predecessors':{},'arguments':{'stage_CPU_upper_seconds':500.,'stage_wall_upper_seconds':700.}}
    def evidence(key,path,payload):
        pin=ctx.put(path,payload);spec['predecessors'][key]={'payload_pin':pin,'runtime_authorization_sha256':ctx.owner_sha};return pin
    folds=[{'held_fold':i,'code_pins':dict(old)} for i in [0,1]]
    prepared={'passed':True,'folds':folds,'input_manifest_sha256':metadata['sha256']}
    training=evidence('prepare_aux',PUB+'/aux_training_inputs.json',prepared)
    counts={'F0_native':(1368,156),'F1_native':(1368,109),'F0_mixed':(1500,3901),'F1_mixed':(1500,3945)}
    # Original geometry totals N5736/C8111. B uses tiny independent synthetic axes.
    if held:counts={role:(1,1) for role in ROLES}
    geometry={'passed':True,'complete':True,'input_manifest_sha256':metadata['sha256'],'producer_FIT_payload_sha256':producer['sha256'],'roles':{role:{'geometry':{'N':n,'C':c}} for role,(n,c) in counts.items()}}
    geometry_pin=evidence('observe_geometry',PUB+'/actual_FIT_geometry.json',geometry)
    prepared['actual_geometry_sha256']=digest(geometry);prepared['producer_FIT_sha256']=producer['sha256']
    training=evidence('prepare_aux',PUB+'/aux_training_inputs.json',prepared)
    components={'current_namespace_plus_admin_upper':880972679,COMPONENT:76628035,'retained_fullstates_atomic_upper':12500000,'closing_reserve':128000000,'completed_aux_controller_compact_future_metadata_upper':33554432,'retained_archive_plus_one_active_raw_and_atomic_peak':53542859,'12_draw_arrays_plus_atomic_temp':39936000,'selected_records_unchanged':7944314}
    work={'storage_components':components,'storage_upper_bytes':sum(components.values()),'feature_layout_components':{'complete_two_seed_compact_cache':75977048,'formal_targets_only_no_persistent205':650987,'held_score_and_aux_prepare_temporaries_cannot_coexist':True},'verified_future_storage_obligations_completed':['prepare_aux'],'GPU_seconds':0.,'CPU_seconds':14231.638,'remaining_wall_seconds':14231.638+3600.,'phases':{'original360protocol':360.,'originalCPUclosure':3600.,'completed_aux_revalidation_remaining_CPU':300.}}
    base={'all_caps_fit':False,'remaining_complete_workload':work,'already_spent':{'old':'same'},'stage_CPU_limit':490.,'stage_wall_limit':690.,'actual_storage_cap':1300000000,'evaluated_epoch':1.}
    monkeypatch.setattr(CompletedAuxRuntimeState,'admission',lambda self:copy.deepcopy(base))
    instance=module.CompletedSupplyRuntimeState(ctx,spec,{'path':'mock_spec.json','sha256':'c'*64})
    workers=[]
    def add_stage(stage,payload):
        pin=evidence(stage,PUB+'/'+stage+'_mock.json',payload)
        log=ctx.put(stage+'.stdout',{'stage':stage,'stage_returned_success':True,'runtime_authorization_sha256':ctx.owner_sha,'output':pin})
        row={'stage':stage,'model_pid':123,'stage_exit_status':0,'observed_worker_exit_code':0,'inner_model_observed_OS_exit_code':0,'whole_child_accounting_complete':True,'actual_child_directly_waited':True,'unresolved_worker_uncertainty':False,'runtime_authorization_sha256':ctx.owner_sha,'log_pins':[log]}
        workers.append(row);return row
    log=ctx.put('prepare_aux.stdout',{'stage':'prepare_aux','stage_returned_success':True,'runtime_authorization_sha256':ctx.owner_sha,'output':training})
    workers.append({'stage':'prepare_aux','model_pid':122,'stage_exit_status':0,'observed_worker_exit_code':0,'inner_model_observed_OS_exit_code':0,'whole_child_accounting_complete':True,'actual_child_directly_waited':True,'unresolved_worker_uncertainty':False,'runtime_authorization_sha256':ctx.owner_sha,'log_pins':[log]})
    geometry_log=ctx.put('observe_geometry.stdout',{'stage':'observe_geometry','stage_returned_success':True,'runtime_authorization_sha256':ctx.owner_sha,'output':geometry_pin})
    geometry_worker={'stage':'observe_geometry','model_pid':121,'stage_exit_status':0,'observed_worker_exit_code':0,'inner_model_observed_OS_exit_code':0,'whole_child_accounting_complete':True,'actual_child_directly_waited':True,'unresolved_worker_uncertainty':False,'runtime_authorization_sha256':ctx.owner_sha,'log_pins':[geometry_log]}
    workers.append(geometry_worker)
    monkeypatch.setattr(instance,'_worker_rows',lambda:workers);monkeypatch.setattr(instance,'spent',lambda:{'fresh_actual_spent':True})
    import echo_routing.temporal.selected_risk_resume_budget as budget
    calls=[]
    def forecast(*args,**kwargs):calls.append((args,kwargs));return {'all_caps_fit':True,'forecast_incremental_tracked_storage_upper_bytes':args[3]}
    monkeypatch.setattr(budget,'forecast',forecast);monkeypatch.setattr(budget,'remaining_limits',lambda *a,**kw:{'remaining_CPU_seconds':500.,'stage_wall_seconds':700.})
    banks={};manifest={'roles':{}}
    for role in ROLES:
        b=np.zeros(8,np.int64);j=b.copy();j[3]=1
        aa=np.eye(5)[b];jj=np.eye(5)[j];banks[role]=([{'id':role}],[aa],[jj],[b])
        manifest['roles'][role]={'sha256':digest(role)}
    inputs=SimpleNamespace(metadata_pin=metadata,manifest=manifest,bank=lambda role,*,purpose:banks[role])
    import echo_routing.temporal.selected_risk_resume_inputs as inputs_module
    monkeypatch.setattr(inputs_module,'ReadOnlyInputs',lambda *a,**kw:inputs)
    c=SimpleNamespace(module=module,ctx=ctx,state=instance,spec=spec,base=base,work=work,calls=calls,workers=workers,inputs=inputs,geometry_pin=geometry_pin,geometry_worker=geometry_worker,add_stage=add_stage)
    if not held:return c
    # Mock Git object verifier checks the declared object map; no real repository.
    import echo_routing.temporal.safety_git as git_module
    def verify(proof,commit,objects):
        if proof.get('objects')!=objects or commit!='d'*40:raise ValueError('mock Git object proof rejected')
    monkeypatch.setattr(git_module,'verify',verify)
    models=[]
    for seed in SEEDS:
        for held_fold in [0,1]:
            for role in ['utility','risk']:
                pin=ctx.put(RUN+f'/mock_aux/{seed}/{held_fold}/{role}.pt',b'mock-not-a-model')
                models.append({**pin,'seed':seed,'held_fold':held_fold,'role':role,'step':6000})
    models_pin=ctx.put(PUB+'/aux_models_manifest.json',{'models':models,'aux_training_inputs_sha256':training['sha256']})
    audit_pin=ctx.put(PUB+'/aux_model_audit.json',{'passed':True,'models':models,'aux_training_inputs_sha256':training['sha256'],'aux_optimizer_updates':48000,'physical_models':8,'fullstate_count':32})
    core={'models_manifest':models_pin,'audit':audit_pin,'training_inputs':training}
    core_objects={p['path']:p['sha256'] for p in core.values()}
    coreproof=ctx.put('core_proof.json',{'objects':core_objects})
    seal=evidence('seal_aux',PUB+'/aux_identity.json',{'passed':True,**core,'git_commit':'d'*40,'actual_Git_object_proof':coreproof})
    origin_objects=dict(ctx.freeze['objects']);origin_py={p:h for p,h in origin_objects.items() if p.endswith('.py')}
    original_spec={'stage':'held_score','predecessors':{'seal_aux':spec['predecessors']['seal_aux']},'metadata':metadata,'FIT_producer':producer,'arguments':{}}
    origin_proof=ctx.put('origin_proof.json',{'objects':origin_objects})
    origin_freeze=ctx.put('origin_freeze.json',{'schema':'A63_resume_stage_freeze_v1','stage':'held_score','git_commit':'d'*40,'objects':origin_objects,'historical_frozen_imports':{},'actual_Git_object_proof':origin_proof,'runtime_authorization_sha256':ctx.owner_sha,'runtime_spec_semantics_sha256':digest(original_spec)})
    original_spec['freeze']=origin_freeze;original_spec_pin=ctx.put('original_spec.json',original_spec)
    intent=ctx.put('original_intent.json',{'stage':'held_score','runtime_spec':original_spec_pin,'runtime_authorization_sha256':ctx.owner_sha,'caller_pid':100,'caller_start_ticks':200,'boot_id':'boot','caller_nonce':'nonce'})
    from echo_routing.temporal.selected_risk_compact_cache import CompactSupplyCache
    from echo_routing.temporal.anchor_cache import sha
    refs={};records=[]
    for seed in SEEDS:
        for role in ROLES:
            held_fold=int(role[1]);identity=digest({'stage_code':origin_py,'seed':seed,'held_fold':held_fold,'role':role,'metadata':metadata['sha256'],'aux_core_pins':core,'native_frozen_C6_pin':POST_NATIVE_PINS[seed] if role.endswith('_native') else None})
            path=RUN+f'/held/seed{seed}/{role}';qs,aa,jj,bb=banks[role]
            cache=CompactSupplyCache(ctx.root/path,identity,bb,[x.argmax(1) for x in aa],[x.argmax(1) for x in jj],feature_reference=refs.get(role))
            value={'D_C6':np.array([[1.,.2,.7,.1,.1,.2]]),'X_C6':np.array([[1.,.1,.8,.1,.1,.1]]),'x109':np.zeros((1,109)),'context96':np.zeros((1,96))}
            cache.run_one(0,lambda:value)
            records.append({'seed':seed,'role':role,'path':path,'identity':identity,'journal_sha256':sha(cache.journal),'axes':1,'input_sha256':manifest['roles'][role]['sha256'],'feature_reference_seed':None if seed==SEEDS[0] else SEEDS[0]})
            if seed==SEEDS[0]:refs[role]=CompactSupplyCache(ctx.root/path,identity,bb,[x.argmax(1) for x in aa],[x.argmax(1) for x in jj],readonly=True)
    held_payload={'passed':True,'complete':True,'records':records,'effect_lookup_before_seal':False,'aux_core_pins':core,'native_D_C6_frozen_pins':{str(s):p for s,p in POST_NATIVE_PINS.items()}}
    heldpin=evidence('held_score',PUB+'/held_selector_cache_manifest.json',held_payload)
    stdout=ctx.put('held.stdout',{'stage':'held_score','stage_returned_success':True,'runtime_authorization_sha256':ctx.owner_sha,'output':heldpin})
    worker={'stage':'held_score','model_pid':300,'stage_exit_status':0,'observed_worker_exit_code':0,'inner_model_observed_OS_exit_code':0,'whole_child_accounting_complete':True,'actual_child_directly_waited':True,'unresolved_worker_uncertainty':False,'runtime_authorization_sha256':ctx.owner_sha,'log_pins':[stdout],'caller_intent':intent,'runtime_spec_sha256':original_spec_pin['sha256'],'caller_pid':100,'caller_start_ticks':200,'caller_boot_id':'boot','caller_nonce':'nonce'}
    workers.append(worker)
    extra=ctx.put('echo_routing/temporal/added_after_held_score.py',b'extra-current-source')
    ctx.freeze['objects'][extra['path']]=extra['sha256']
    c.records=records;c.worker=worker;c.origin_py=origin_py;c.original_spec_pin=original_spec_pin
    return c


def test_actual_geometry_future_bound_keeps_all_original_other_components(tmp_path,monkeypatch):
    c=case(tmp_path,monkeypatch);r=c.state.admission();w=r['remaining_complete_workload']
    assert w['feature_layout_components']['complete_two_seed_compact_cache']==61848664
    assert w['storage_components'][COMPONENT]==61848664+650987
    assert c.work['storage_upper_bytes']-w['storage_upper_bytes']==14128384
    assert {k:v for k,v in w['storage_components'].items() if k!=COMPONENT}=={k:v for k,v in c.work['storage_components'].items() if k!=COMPONENT}
    assert w['CPU_seconds']==c.work['CPU_seconds']+3000. and w['remaining_wall_seconds']==c.work['remaining_wall_seconds']+3000.
    assert {k:v for k,v in w['phases'].items() if k!='completed_supply_cache_revalidation_remaining_CPU'}==c.work['phases']
    assert r['already_spent']=={'fresh_actual_spent':True} and r['actual_storage_cap']==1300000000
    assert r['stage_CPU_limit']==490. and r['stage_wall_limit']==690.
    assert c.calls[0][0][6]==c.ctx.owner and c.calls[0][1]=={'closing':False}
    c.work['verified_future_storage_obligations_completed']=[]
    assert c.state.admission()==c.base
    for stage in ['close_negative','close_full']:
        c.state.stage=stage;c.spec['stage']=stage;assert c.state.admission()==c.base


def test_completed_readonly_eight_cache_origin_identity_and_no_double_credit(tmp_path,monkeypatch):
    c=case(tmp_path,monkeypatch,held=True);before={str(p.relative_to(c.ctx.root)) for p in c.ctx.root.rglob('*') if p.is_file()}
    r=c.state.admission();w=r['remaining_complete_workload']
    assert w['storage_components'][COMPONENT]==650987
    assert c.work['storage_upper_bytes']-w['storage_upper_bytes']==75977048
    assert w['storage_components']['current_namespace_plus_admin_upper']==880972679
    assert {str(p.relative_to(c.ctx.root)) for p in c.ctx.root.rglob('*') if p.is_file()}==before
    assert not (c.ctx.root/PUB/'selector_identity.json').exists()
    c.add_stage('seal_selectors',{'passed':True,'complete':True})
    c.work['verified_future_storage_obligations_completed'].append('seal_selectors')
    c.work['storage_components'][COMPONENT]=650987;c.work['storage_upper_bytes']-=75977048
    r=c.state.admission();assert r['remaining_complete_workload']['storage_upper_bytes']==c.work['storage_upper_bytes']
    c.add_stage('prepare_formal',{'passed':True,'seeds':list(SEEDS)})
    c.work['verified_future_storage_obligations_completed'].append('prepare_formal')
    c.work['storage_components'][COMPONENT]=0;c.work['storage_upper_bytes']-=650987
    assert c.state.admission()['remaining_complete_workload']['storage_components'][COMPONENT]==0


def test_bad_closed_parent_missing_origin_corruption_pending_and_source_change_reject(tmp_path,monkeypatch):
    c=case(tmp_path,monkeypatch,held=True)
    c.geometry_worker['inner_model_observed_OS_exit_code']=7
    with pytest.raises(ValueError):c.state.admission()
    c.geometry_worker['inner_model_observed_OS_exit_code']=0
    c.workers.remove(c.geometry_worker)
    with pytest.raises(ValueError):c.state.admission()
    c.workers.append(c.geometry_worker)
    c.worker['inner_model_observed_OS_exit_code']=7
    with pytest.raises(ValueError):c.state.admission()
    c.worker['inner_model_observed_OS_exit_code']=0
    intent=c.worker.pop('caller_intent')
    with pytest.raises(ValueError):c.state.admission()
    c.worker['caller_intent']=intent
    folder=c.ctx.root/c.records[0]['path'];pending=folder/'pending.json';pending.write_text('{}')
    with pytest.raises(ValueError):c.state.admission()
    pending.unlink()
    payload=next(folder.glob('*.npz'));raw=payload.read_bytes();payload.write_bytes(raw+b'corrupt')
    with pytest.raises(ValueError):c.state.admission()
    payload.write_bytes(raw)
    path=next(iter(c.module._LAYOUT_PINS));old=c.ctx.freeze['objects'][path];c.ctx.freeze['objects'][path]='0'*64
    with pytest.raises(ValueError):c.state.admission()
    c.ctx.freeze['objects'][path]=old
    c.work['storage_components'][COMPONENT]-=1
    with pytest.raises(ValueError):c.state.admission()
