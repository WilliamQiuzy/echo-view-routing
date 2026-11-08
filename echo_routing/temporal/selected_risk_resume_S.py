"""Real continuation Stage-S preparation, eight-head loop and independent audit.

Every public method enters the new raw-owner/current-Git predecessor gate before
data/model work. No production C0/C1 is imported or constructed here. Completion
is audited per head before lossless archive and the next head; archived heads
are verified and skipped rather than reentering an update journal.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from echo_routing.temporal.selected_risk_resume_contract import digest

SEEDS=(2026110461,2026110462)
SOURCE='echo_routing/temporal/selected_risk_resume_S.py'


def array_sha(value):
    a=np.asarray(value)
    if a.dtype.hasobject:raise ValueError('typed non-object mathematical array required')
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def observable_geometry(qs,aa,jj,bb):
    from echo_routing.temporal.event_risk_cost_bounds import observed_geometry
    from echo_routing.temporal.proposal_actions import propose,partition,apply
    if not len(qs)==len(aa)==len(jj)==len(bb):raise ValueError('all complete matched teacher axes required')
    rows=[];layout=[];maximum={'max_C':0,'max_candidate_K':0,'max_T':0}
    for q,a,j,b in zip(qs,aa,jj,bb):
        if q['kind'] not in ['native','multi'] or len(q['refs'])!=len(b) or np.asarray(a).shape!=(len(b),5) or np.asarray(j).shape!=(len(b),5):raise ValueError('matched original FIT geometry required')
        ar=np.asarray(a).argmax(1);jr=np.asarray(j).argmax(1);g=observed_geometry(b,ar,jr)
        eligible,_=partition(b,propose(b,ar,jr));ks=[int(np.count_nonzero(apply(b,x)!=b)) for x in eligible]
        layout.append(ks);maximum['max_C']=max(maximum['max_C'],g['C']);maximum['max_candidate_K']=max(maximum['max_candidate_K'],max(ks,default=0));maximum['max_T']=max(maximum['max_T'],len(b));g['nonempty_query_axes']=int(bool(eligible));rows.append(g)
    return {'geometry':{k:sum(r[k] for r in rows) for k in rows[0]} if rows else {},'maxima':maximum,'axes':len(rows),'truth_or_head_outcomes_used':False,'private_candidate_K_per_axis':layout}


def layout_data(layout,train_fold):
    if type(train_fold) is not int or train_fold not in [0,1]:raise ValueError('original training complement fold required')
    groups=[];lengths=[];kg={'native':[],'constructed':[]};ki={k:[] for k in kg}
    for kind,role in [('native',f'F{train_fold}_native'),('constructed',f'F{train_fold}_mixed')]:
        for ks in layout[role]:
            if any(type(k) is not int or k<1 for k in ks):raise ValueError('all actual positive candidate K values required')
            g=np.arange(len(lengths),len(lengths)+len(ks),dtype=np.int64);q=len(groups);groups.append(g);lengths.extend(ks);ki[kind].extend(g.tolist())
            if len(g):kg[kind].append(q)
    total=sum(lengths)
    if total>np.iinfo(np.int64).max:raise ValueError('candidate changed-position offsets overflow')
    return {'groups':groups,'position_offsets':np.r_[0,np.cumsum(lengths,dtype=np.int64)],'kind_groups':{k:np.asarray(v,np.int64) for k,v in kg.items()},'kind_indices':{k:np.asarray(v,np.int64) for k,v in ki.items()}}


def _cpu():
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('entry must hide CUDA before Torch import')
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():raise ValueError('anomalous CUDA context before CPU Stage-S work')


def _model(role,seed):
    from echo_routing.temporal.safety_training import Utility,Risk,optimizer
    m=Utility('S-L',seed) if role=='utility' else Risk(seed)
    return m,optimizer(m)


def _train(*args,**kwargs):
    from echo_routing.temporal.selected_risk_training import train_aux_one
    return train_aux_one(*args,**kwargs)


def _archive(path,identity):
    from echo_routing.temporal.selected_risk_storage_engineering import archive_trace
    return archive_trace(path,identity)


def _audit_head(model,opt,data,plan,identity,folder,initial_state):
    import torch
    from echo_routing.temporal.selected_risk_audit import audit_aux_trace,validate_aux_states
    from echo_routing.temporal.anchor_cache import sha
    raw=folder/'physical_updates.jsonl';rows=[json.loads(v) for v in raw.read_text().splitlines()]
    states={k:torch.load(folder/('latest.pt' if k=='latest' else f'step{k}.pt'),map_location='cpu',weights_only=True) for k in [2000,4000,6000,'latest']}
    trace=audit_aux_trace(rows,data,plan,model.seed,model.role,identity)
    state=validate_aux_states(model,opt,states,identity,data,initial_state)
    return {'passed':True,'physical_updates':trace['physical_updates'],'fullstate_count':state['fullstates'],'trace_audit':trace,'state_audit':state,'raw_trace_sha256':sha(raw),'state_pins':[{'path':str(folder/('latest.pt' if k=='latest' else f'step{k}.pt')),'sha256':sha(folder/('latest.pt' if k=='latest' else f'step{k}.pt'))} for k in [2000,4000,6000,'latest']]}


def finish_one_head(context,model,opt,data,plan,identity,folder,initial_state,relative_folder):
    physical=_train(model,opt,data,plan,identity,folder)
    audit=_audit_head(model,opt,data,plan,identity,folder,initial_state)
    if audit.get('passed') is not True or (audit.get('physical_updates'),audit.get('fullstate_count'))!=(6000,4):raise ValueError('independent exact6000/four-fullstate audit precedes immutable completion')
    complete={**audit,'identity':identity,'physical_result':physical}
    completion=context.write_json(relative_folder+'/completion.json',complete)
    archived=_archive(folder/'physical_updates.jsonl',identity)
    return {'audit':audit,'completion':completion,'archive':archived}


def completed_head(folder,identity,verify_pin):
    """Completion is an audit seal, not permission to trust an unchecked archive."""
    from echo_routing.temporal.selected_risk_storage_engineering import load_trace
    folder=Path(folder);p=folder/'completion.json'
    if not p.exists():return None
    if p.is_symlink() or (folder/'pending.json').exists() or (folder/'pending.json').is_symlink():raise ValueError('ambiguous/pending completed head must not replay')
    row=json.loads(p.read_text())
    if row.get('passed') is not True or row.get('identity')!=identity or (row.get('physical_updates'),row.get('fullstate_count'))!=(6000,4):raise ValueError('completed head audit identity/count differs')
    if len(row['state_pins'])!=4 or len({p['path'] for p in row['state_pins']})!=4:raise ValueError('four distinct independently audited fullstate pins required')
    for pin in row['state_pins']:verify_pin(pin)
    # Archive after audited completion is a distinct durable operation. If the
    # process died between these writes, require explicit recovery; never train.
    receipt=folder/'physical_updates.jsonl.archive.json'
    if not receipt.is_file() or receipt.is_symlink():raise ValueError('audited completion missing archive: explicit no-training recovery required')
    ar=json.loads(receipt.read_text())
    if ar['identity']!=identity or ar['uncompressed_sha256']!=row['raw_trace_sha256']:raise ValueError('archive does not reproduce independently audited original trace')
    rows=load_trace(folder/'physical_updates.jsonl.gz',ar)
    if len(rows)!=6000 or any(r.get('u')!=i or r.get('identity')!=identity for i,r in enumerate(rows)):raise ValueError('archived physical6000 trace/identity differs')
    return row


class StageS:
    def __init__(self,context,inputs,freeze_pin,descriptors):
        self.ctx=context;self.inputs=inputs;self.freeze_pin=freeze_pin;self.descriptors=descriptors

    def _enter(self,stage):
        freeze,e=self.ctx.enter(stage,self.freeze_pin,self.descriptors,required={SOURCE,'echo_routing/temporal/selected_risk_resume_context.py','echo_routing/temporal/selected_risk_resume_inputs.py','configs/experiments/selected_action_risk_v1.json'})
        _cpu();return freeze,e

    def observe_geometry(self):
        freeze,e=self._enter('observe_geometry');out={};layout={}
        from echo_routing.temporal.selected_risk_training import aux_plans
        from echo_routing.temporal.selected_risk_aux_pipeline import predraw_summary,validate_measured_shapes
        for role in ['F0_native','F1_native','F0_mixed','F1_mixed']:
            out[role]=observable_geometry(*self.inputs.bank(role,purpose='geometry'));layout[role]=out[role].pop('private_candidate_K_per_axis')
        maxima={kind:{k:max(out[f'F{f}_{kind}']['maxima'][k] for f in [0,1]) for k in ['max_C','max_candidate_K','max_T']} for kind in ['native','mixed']}
        shape_error=None
        try:validate_measured_shapes(maxima)
        except ValueError as ex:shape_error=str(ex)
        native=sum(out[f'F{f}_native']['geometry']['C'] for f in [0,1])
        if native!=265:raise ValueError('full original native candidate265 parity is mandatory')
        draws=[]
        for held in [0,1]:
            data=layout_data(layout,1-held)
            for seed in SEEDS:
                for role,plan in aux_plans(data,seed).items():draws.append({'seed':seed,'held_fold':held,'role':role,**predraw_summary(data,plan,role)})
        pin=self.ctx.save_value('geometry/axis_layout.private.npz',layout)
        value={'passed':shape_error is None,'complete':True,'roles':out,'axis_layout':{k:pin[k] for k in ['path','sha256']},'aux_full_predraw':draws,'shape_maxima':maxima,'fits_measured_training_shapes':shape_error is None,'shape_domain_error':shape_error,'input_manifest_sha256':self.inputs.metadata_pin['sha256'],'producer_FIT_payload_sha256':e['source_FIT']['payload_sha256'],'geometry_freeze_git':freeze['git_commit'],'teacher_forwards':0,'optimizer_updates':0,'C0_C1_instances':0}
        return self.ctx.write_json('actual_FIT_geometry.json',value,public=True)

    def prepare_aux(self):
        freeze,e=self._enter('prepare_aux');admit=e['admit_aux']['payload'];geo=e['observe_geometry']['payload']
        if admit.get('geometry_sha256')!=e['observe_geometry']['payload_sha256'] or geo.get('passed') is not True:raise ValueError('aux admission must bind complete actual teacher geometry')
        from echo_routing.temporal.selected_risk_fit import prepare_aux_training
        from echo_routing.temporal.selected_risk_inputs import minimal_aux_data
        from echo_routing.temporal.selected_risk_training import aux_plans
        from echo_routing.temporal.selected_risk_resume_storage import storage_bound
        part=self.inputs.partition();folds=[];allocated=0
        for held in [0,1]:
            bank=self.inputs.fold_bank(1-held,purpose='complement_targets')
            data=minimal_aux_data(prepare_aux_training(*bank,part['sessions'][1-held],part['sessions'][held]));plans={str(s):aux_plans(data,s) for s in SEEDS}
            allocated+=storage_bound({'data':data,'plans':plans})['own_atomic_coexistence_bytes']
            if type(admit.get('prepared_data_atomic_bytes_upper')) is not int or allocated>admit['prepared_data_atomic_bytes_upper']:raise ValueError('actual typed full-feature/draw atomic bytes exceed admitted allocation: stop before persistent large cache')
            artifact=self.ctx.save_value(f'aux/held{held}/training_inputs.private.npz',{'data':data,'plans':plans})
            folds.append({'held_fold':held,'train_fold':1-held,'data':{k:artifact[k] for k in ['path','sha256']},'F_fit':data['F_fit'],'scale':data['scale'],'candidate_count':len(data['targets']),'changed_positions':len(data['position_labels']),'train_sessions':201,'held_sessions':201,'training_source_footprint_verified':True,'code_pins':{p:h for p,h in freeze['objects'].items() if p.endswith('.py')}})
        out={'passed':True,'folds':folds,'input_manifest_sha256':self.inputs.metadata_pin['sha256'],'actual_geometry_sha256':e['observe_geometry']['payload_sha256'],'producer_FIT_sha256':geo['producer_FIT_payload_sha256'],'prepared_data_atomic_bytes':allocated,'aux_optimizer_updates':0,'formal_optimizer_updates':0,'C0_C1_instances':0,'normalizers_targets_RMS_F_fit_complement_only':True}
        return self.ctx.write_json('aux_training_inputs.json',out,public=True)

    def _identity(self,fold,data,plan,freeze,seed,role):
        from echo_routing.temporal.selected_risk_aux_pipeline import expected_identity
        for p,h in fold['code_pins'].items():
            if freeze['objects'].get(p)!=h:raise ValueError('training code changed from prepared independently pinned input graph')
        parts={'code':digest(fold['code_pins']),'config':freeze['objects']['configs/experiments/selected_action_risk_v1.json'],'sources':digest({'metadata':self.inputs.metadata_pin,'FIT':self.descriptors['admit_aux']['payload_pin']}),'input':fold['data']['sha256'],'draw_plan':array_sha(plan['indices']),'normalizer':digest({k:{kk:array_sha(vv) for kk,vv in v.items()} for k,v in data['normalizers'].items()}),'target_scales':digest({'F_fit':data['F_fit'],'RMS':data['scale'],'targets':array_sha(data['targets'])})}
        return expected_identity(parts,seed,fold['held_fold'],role)

    def train_aux(self):
        freeze,e=self._enter('train_aux');prepared=e['prepare_aux']['payload'];records=[]
        import torch
        from echo_routing.temporal.selected_risk_aux_pipeline import guard_head_start
        from echo_routing.temporal.proposal_training import atomic_save
        from echo_routing.temporal.anchor_cache import durable_directory,sha
        for fold in prepared['folds']:
            value=self.ctx.load_value(fold['data']);data=value['data'];plans=value['plans']
            for seed in SEEDS:
                for role in ['utility','risk']:
                    plan=plans[str(seed)][role];identity=self._identity(fold,data,plan,freeze,seed,role);rel=f'aux/seed{seed}/held{fold["held_fold"]}/{role}';folder=self.ctx.output(rel)
                    def check(pin):
                        p=Path(pin['path']);p=p.relative_to(self.ctx.root) if p.is_absolute() else p
                        self.ctx.raw({'path':str(p),'sha256':pin['sha256']})
                    done=completed_head(folder,identity,check)
                    if done is not None:records.append({'seed':seed,'held_fold':fold['held_fold'],'role':role,'reused_complete':True});continue
                    guard_head_start(folder);model,opt=_model(role,seed);initial={n:t.detach().clone() for n,t in model.state_dict().items()}
                    durable_directory(folder);ip=folder/'initial.pt'
                    if ip.exists():
                        old=torch.load(ip,weights_only=True,map_location='cpu')
                        if set(old)!=set(initial) or any(not torch.equal(old[k],initial[k]) for k in old):raise ValueError('auxiliary initializer differs; never warm-start')
                    else:atomic_save(initial,ip)
                    result=finish_one_head(self.ctx,model,opt,data,plan,identity,folder,initial,rel)
                    records.append({'seed':seed,'held_fold':fold['held_fold'],'role':role,'reused_complete':False,'final':{'path':str((folder/'step6000.pt').relative_to(self.ctx.root)),'sha256':sha(folder/'step6000.pt')},'completion':result['completion'],'archive_sha256':result['archive']['archive_sha256']})
                    del model,opt
        return self.ctx.write_json('aux_physical_completion.json',{'passed':True,'models':records,'aux_optimizer_updates':48000,'formal_optimizer_updates':0,'physical_models':8,'fullstate_count':32},public=True)

    def audit_aux(self):
        freeze,e=self._enter('audit_aux');prepared=e['prepare_aux']['payload'];part=self.inputs.partition();models=[];audits=[]
        import torch
        from echo_routing.temporal.selected_risk_fit import prepare_aux_training
        from echo_routing.temporal.selected_risk_inputs import minimal_aux_data
        from echo_routing.temporal.selected_risk_training import aux_plans
        from echo_routing.temporal.safety_data import exact
        from echo_routing.temporal.selected_risk_audit import audit_aux_trace,validate_aux_states
        from echo_routing.temporal.selected_risk_storage_engineering import load_trace
        from echo_routing.temporal.anchor_cache import sha
        for fold in prepared['folds']:
            data=minimal_aux_data(prepare_aux_training(*self.inputs.fold_bank(1-fold['held_fold'],purpose='complement_targets'),part['sessions'][1-fold['held_fold']],part['sessions'][fold['held_fold']]))
            stored=self.ctx.load_value(fold['data'])
            if not exact(stored['data'],data) or not exact(stored['plans'],{str(s):aux_plans(data,s) for s in SEEDS}):raise ValueError('independent entire complement feature/target/norm/RMS/draw recount failed')
            for seed in SEEDS:
                for role in ['utility','risk']:
                    plan=aux_plans(data,seed)[role];identity=self._identity(fold,data,plan,freeze,seed,role);folder=self.ctx.output(f'aux/seed{seed}/held{fold["held_fold"]}/{role}')
                    completed_head(folder,identity,lambda p:self.ctx.raw({'path':str(Path(p['path']).relative_to(self.ctx.root)),'sha256':p['sha256']}))
                    model,opt=_model(role,seed);initial={n:t.detach().clone() for n,t in model.state_dict().items()}
                    rows=load_trace(folder/'physical_updates.jsonl.gz',json.loads((folder/'physical_updates.jsonl.archive.json').read_text()));trace=audit_aux_trace(rows,data,plan,seed,role,identity)
                    states={k:torch.load(folder/('latest.pt' if k=='latest' else f'step{k}.pt'),map_location='cpu',weights_only=True) for k in [2000,4000,6000,'latest']};state=validate_aux_states(model,opt,states,identity,data,initial)
                    row={'seed':seed,'held_fold':fold['held_fold'],'role':role,'step':6000,'path':str((folder/'step6000.pt').relative_to(self.ctx.root)),'sha256':sha(folder/'step6000.pt'),'parameters':state['parameters']};models.append(row);audits.append({'seed':seed,'held_fold':fold['held_fold'],'role':role,'identity':identity,'trace':trace,'states':state})
                    del model,opt
        if len(models)!=8 or sum(v['trace']['physical_updates'] for v in audits)!=48000 or sum(v['states']['fullstates'] for v in audits)!=32:raise ValueError('complete independent eight-head48000/32 audit required')
        common={'passed':True,'models':models,'aux_optimizer_updates':48000,'formal_optimizer_updates':0,'physical_models':8,'fullstate_count':32,'aux_training_inputs_sha256':e['prepare_aux']['payload_sha256']}
        manifest=self.ctx.write_json('aux_models_manifest.json',common,public=True)
        audit=self.ctx.write_json('aux_model_audit.json',{**common,'independent_recounts':audits,'no_forward_during_audit':True},public=True)
        return {'models_manifest':manifest,'audit':audit}
