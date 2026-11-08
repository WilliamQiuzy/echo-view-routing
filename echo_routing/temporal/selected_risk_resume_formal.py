"""C0/C1 actual matched-draw training after independently recounted coverage.

No model is instantiated on import. All normalized205 matrices remain in RAM;
private persistence contains only targets, canonical digests, explicit draws and
read-only cache identities. New journals use the original qualified BCE/Adam
step and complete checkpoint/restore, not the old A62 uniform-draw loop.
"""
import io
import json
import time
from pathlib import Path
import numpy as np
from echo_routing.temporal.selected_risk_resume_contract import digest,validate_coverage_pass
from echo_routing.temporal.selected_risk_resume_S import _cpu,array_sha,completed_head,SEEDS

SOURCE='echo_routing/temporal/selected_risk_resume_formal.py'


def new_head(seed,arm,coverage):
    if not isinstance(coverage,dict):raise ValueError('actual full coverage required before production head construction')
    validate_coverage_pass(coverage)
    if seed not in SEEDS or arm not in ['C0','C1']:raise ValueError('exact matched formal arm/seed required')
    from echo_routing.temporal.event_risk_model import EventRisk,optimizer
    m=EventRisk('R2',seed);m.arm=arm
    if sum(p.numel() for p in m.parameters())!=6625:raise ValueError('unchanged 205x32x1 architecture required')
    return m,optimizer(m)


def unsafe_target(effect):return int(effect['H']>0 or effect['dFP']>0)


def validate_matched_draws(draws,pools,n):
    if set(draws)!={'C0','C1'} or set(pools)!={'native','constructed'}:raise ValueError('two fixed arms and complete two-kind pools required')
    for arm,a in draws.items():
        a=np.asarray(a)
        if a.shape!=(6000,64) or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>np.iinfo(np.int64).max) or np.any(a>=n) or not np.isin(a[:,:32],pools['native']).all() or not np.isin(a[:,32:],pools['constructed']).all():raise ValueError('actual global6000x64 native32/mixed32 candidate indices required')
    if not np.array_equal(draws['C0'][:,:48],draws['C1'][:,:48]):raise ValueError('same native32 plus mixed16 replay must match every update')
    return True


def train_explicit(model,opt,x,y,draws,identity,norms,folder):
    """Durable actual6000, retaining pending/uncheckpointed prefixes on failure."""
    import torch
    from echo_routing.temporal.event_risk_model import checkpoint,restore,step
    from echo_routing.temporal.proposal_training import UpdateJournal,atomic_save
    if torch.cuda.is_initialized():raise ValueError('CPU formal entry must reject CUDA before journal/model access')
    if x.dtype!=np.float32 or x.shape!=(len(y),205) or y.dtype!=np.float32 or not np.isfinite(x).all() or not np.isin(y,[0,1]).all():raise ValueError('complete unchanged normalized205 and unscaled binary targets required')
    folder=Path(folder);journal=UpdateJournal(folder,identity);latest=folder/'latest.pt'
    if not latest.exists() and journal.cursor==0:atomic_save(checkpoint(model,opt,0,identity,norms),latest)
    start=journal.resume_cursor();state=torch.load(latest,map_location='cpu',weights_only=True)
    if restore(model,opt,state,identity,norms)!=start:raise ValueError('formal latest/physical cursor mismatch: never replay')
    tx=torch.from_numpy(x);ty=torch.from_numpy(y)
    for u in range(start,6000):
        if torch.cuda.is_initialized():raise ValueError('unexpected CUDA before actual formal update')
        idx=draws[u];journal.begin(u);t=time.perf_counter();row=step(model,opt,{'x':tx[idx],'y':ty[idx]},u)
        row.update(candidate_indices=idx.tolist(),source_kind_blocks=[0,1],optimizer_update_wall_seconds=time.perf_counter()-t)
        cursor=u+1;payload=checkpoint(model,opt,cursor,identity,norms) if cursor%100==0 else None
        if cursor in [2000,4000,6000]:atomic_save(payload,folder/f'step{cursor}.pt')
        journal.finish(row,payload)
    if any(float(opt.state[p]['step'])!=6000 for p in model.parameters()):raise ValueError('actual formal Adam cursors must all equal6000')
    return {'physical_updates':6000,'new_updates_this_process':6000-start,'candidate_exposures':384000}


class Formal:
    def __init__(self,context,inputs,freeze_pin,descriptors):
        self.ctx=context;self.inputs=inputs;self.freeze_pin=freeze_pin;self.descriptors=descriptors

    def _enter(self,stage):
        f,e=self.ctx.enter(stage,self.freeze_pin,self.descriptors,required={SOURCE,'echo_routing/temporal/selected_risk_resume_context.py','configs/experiments/selected_action_risk_v1.json'})
        _cpu();return f,e

    def _cache_rows(self,manifest,seed):
        from echo_routing.temporal.selected_risk_compact_cache import CompactSupplyCache
        from echo_routing.temporal.anchor_cache import sha
        refs={};out=[]
        # Reopen first seed even for second seed to verify every shared feature
        # payload SHA and every original feature bit through the qualified cache.
        for s in SEEDS:
            for role in ['F0_native','F0_mixed','F1_native','F1_mixed']:
                qs,a,j,b=self.inputs.bank(role,purpose='complement_targets');r=next(r for r in manifest['records'] if (r['seed'],r['role'])==(s,role))
                c=CompactSupplyCache(self.ctx.root/r['path'],r['identity'],b,[p.argmax(1) for p in a],[p.argmax(1) for p in j],readonly=True,feature_reference=refs.get(role))
                if sha(c.journal)!=r['journal_sha256'] or c.cursor!=len(qs) or r['input_sha256']!=self.inputs.manifest['roles'][role]['sha256']:raise ValueError('formal inputs require every immutable source/cache payload identity')
                if s==seed:out.extend((q,bb,c.load(i)) for i,(q,bb) in enumerate(zip(qs,b)))
                if s==SEEDS[0]:refs[role]=c
            if s==seed:break
        return out

    def _recount(self,coverage,selector,seed):
        from echo_routing.temporal.selected_risk_resume_supply import selected_indices,require_selector_seal
        from echo_routing.temporal.selected_risk_effects import record,F_FIT
        from echo_routing.temporal.selected_risk_supply import coverage as recount,digest as candidate_digest,draws as selected_draws
        from echo_routing.temporal.selected_risk_record_storage import read_records
        from echo_routing.temporal.safety_targets import effects
        from echo_routing.temporal.safety_git import verify
        require_selector_seal(selector,selector['cache_manifest']['sha256']);manifest=self.ctx.read(selector['cache_manifest'])
        objects={selector['cache_manifest']['path']:selector['cache_manifest']['sha256'],**{p['path']:p['sha256'] for p in manifest['aux_core_pins'].values()}}
        if selector['objects']!=objects:raise ValueError('exact selector plus auxiliary core object set required')
        verify(self.ctx.read(selector['actual_Git_object_proof']),selector['git_commit'],objects)
        rows=self._cache_rows(manifest,seed);selected=[];ids=[];y=[];raw=[];contexts=[];pools={'native':[],'constructed':[]}
        for q,b,v in rows:
            for index,(D,X) in selected_indices(v).items():selected.append(record(q,b,v['actions'][index],v['context96'][index],D=D,X=X))
            for i,action in enumerate(v['actions']):
                idx=len(ids);ids.append(candidate_digest([q['id'],action.key]));y.append(unsafe_target(effects(b,action,q['labels'],F_FIT)));raw.append(v['x109'][i]);contexts.append(v['context96'][i]);pools['native' if q['kind']=='native' else 'constructed'].append(idx)
        if len(ids)!=len(set(ids)):raise ValueError('complete canonical candidate identity may not duplicate')
        sessions=sorted(sum(self.inputs.partition()['sessions'],[]));table=next(t for t in coverage['selected_record_tables'] if t['seed']==seed)
        self.ctx.raw({k:table[k] for k in ['path','sha256']});stored=read_records(self.ctx.root/table['path'],expected_sha256=table['sha256'],expected_sessions=sessions)
        if selected!=stored:raise ValueError('independent original selected actions/effects/signatures differ from sealed coverage inputs')
        detail=recount(selected,private_details=True)
        if {k:v for k,v in detail.items() if k!='private_pools'}!=coverage['seeds'][str(seed)] or detail['passed'] is not True:raise ValueError('independent all23 signature-based coverage criteria must pass for each seed')
        lookup={h:i for i,h in enumerate(ids)};selected_pools=[{signature:[lookup[h] for h in members] for signature,members in p.items()} for p in detail['private_pools']]
        p={k:np.asarray(v,np.int64) for k,v in pools.items()};generated=selected_draws(p['native'],p['constructed'],selected_pools,seed);draws={a:generated[a] for a in ['C0','C1']};validate_matched_draws(draws,p,len(ids))
        return {'candidate_digests':np.asarray([bytes.fromhex(h) for h in ids],dtype='V32'),'unsafe':np.asarray(y,np.float32),'kind_indices':p,'draws':draws},np.asarray(raw,np.float64),np.asarray(contexts,np.float64)

    def _norms(self):
        from scripts.selected_risk_common import prior_path
        p=prior_path('docs/results/2026-11-04_anchor_context_risk/training_inputs.json');t=json.loads(p.read_text());norms=self.ctx.read(t['normalizers'])
        if set(norms)!={'candidate','context'}:raise ValueError('original frozen A62 R2 normalizers only; no refit')
        return norms,t['normalizers']

    def prepare_formal(self):
        freeze,e=self._enter('prepare_formal');cov=e['coverage']['payload'];validate_coverage_pass(cov);selector=e['seal_selectors']['payload'];norms,normpin=self._norms();records=[]
        from echo_routing.temporal.selected_risk_resume_storage import storage_bound
        allocated=0;admit=e['admit_formal']['payload']
        for seed in SEEDS:
            data,raw,context=self._recount(cov,selector,seed)
            allocated+=storage_bound(data)['own_atomic_coexistence_bytes']
            if type(admit.get('formal_data_atomic_bytes_upper')) is not int or allocated>admit['formal_data_atomic_bytes_upper']:raise ValueError('actual explicit formal draw/target atomic bytes exceed current admitted bound')
            pin=self.ctx.save_value(f'formal/seed{seed}/training_inputs.private.npz',data)
            records.append({'seed':seed,'data':{k:pin[k] for k in ['path','sha256']},'candidate_count':len(data['unsafe']),'kind_candidate_counts':{k:len(v) for k,v in data['kind_indices'].items()},'unsafe_count':int(data['unsafe'].sum()),'raw109_sha256':array_sha(raw),'context96_sha256':array_sha(context),'draw_SHA':{a:array_sha(v) for a,v in data['draws'].items()},'original_fixed_normalizers':normpin,'code_pins':{p:h for p,h in freeze['objects'].items() if p.endswith('.py')}})
        return self.ctx.write_json('formal_training_inputs.json',{'passed':True,'seeds':list(SEEDS),'records':records,'coverage_sha256':e['coverage']['payload_sha256'],'selector_seal_sha256':e['seal_selectors']['payload_sha256'],'aux_optimizer_updates':48000,'formal_optimizer_updates':0,'C0_C1_instances':0,'unsafe_definition':'int(H>0 or dFP16>0); no target scaler','normalized205_persisted':False,'formal_data_atomic_bytes':allocated},public=True)

    def _data(self,row,cov,selector):
        from echo_routing.temporal.safety_data import exact
        from echo_routing.temporal.event_risk_features import event_matrix
        stored=self.ctx.load_value(row['data']);actual,raw,context=self._recount(cov,selector,row['seed'])
        if not exact(stored,actual) or array_sha(raw)!=row['raw109_sha256'] or array_sha(context)!=row['context96_sha256']:raise ValueError('independent formal targets/draws/features/supply source binding failed')
        norms,pin=self._norms()
        if pin!=row['original_fixed_normalizers']:raise ValueError('normalizer source cannot change between matched arms')
        return actual,event_matrix(raw,context,'R2',norms['candidate'],norms['context']),norms

    def _identity(self,row,data,freeze,arm):
        for p,h in row['code_pins'].items():
            if freeze['objects'].get(p)!=h:raise ValueError('formal code changed from prepared inputs')
        base={'code':digest(row['code_pins']),'config':freeze['objects']['configs/experiments/selected_action_risk_v1.json'],'sources':self.inputs.metadata_pin['sha256'],'input':row['data']['sha256'],'draw_plan':array_sha(data['draws'][arm]),'normalizer':row['original_fixed_normalizers']['sha256'],'target_scales':digest({'unscaled_binary_event':array_sha(data['unsafe']),'seed':row['seed'],'arm':arm})}
        return {k:digest({'SHA':v,'seed':row['seed'],'arm':arm,'updates':6000}) for k,v in base.items()}

    def train_formal(self):
        freeze,e=self._enter('train_formal');cov=e['coverage']['payload'];selector=self.ctx.evidence(self.descriptors['seal_selectors'])['payload'];prepared=e['prepare_formal']['payload'];out=[]
        import torch
        from echo_routing.temporal.proposal_training import atomic_save
        from echo_routing.temporal.anchor_cache import durable_directory,sha
        from echo_routing.temporal.selected_risk_audit import audit_event_trace,validate_event_states
        from echo_routing.temporal.selected_risk_storage_engineering import archive_trace
        from echo_routing.temporal.selected_risk_aux_pipeline import guard_head_start
        for row in prepared['records']:
            data,x,norms=self._data(row,cov,selector);initial_reference=None
            for arm in ['C0','C1']:
                identity=self._identity(row,data,freeze,arm);rel=f'models/seed{row["seed"]}/{arm}';folder=self.ctx.output(rel)
                done=completed_head(folder,identity,lambda p:self.ctx.raw({'path':str(Path(p['path']).relative_to(self.ctx.root)),'sha256':p['sha256']}))
                if done is not None:
                    completion_pin={'path':str((folder/'completion.json').relative_to(self.ctx.root)),'sha256':sha(folder/'completion.json')};self.ctx.raw(completion_pin)
                    archived=json.loads((folder/'physical_updates.jsonl.archive.json').read_text())
                    out.append({'seed':row['seed'],'arm':arm,'completion':completion_pin,'archive_sha256':archived['archive_sha256'],'reused_complete':True});continue
                guard_head_start(folder);model,opt=new_head(row['seed'],arm,cov);initial={n:t.detach().clone() for n,t in model.state_dict().items()}
                if initial_reference is not None and any(not torch.equal(initial_reference[n],t) for n,t in initial.items()):raise ValueError('C0/C1 matched initialization differs')
                initial_reference=initial;durable_directory(folder);ip=folder/'initial.pt'
                if ip.exists():
                    old=torch.load(ip,weights_only=True,map_location='cpu')
                    if set(old)!=set(initial) or any(not torch.equal(old[k],v) for k,v in initial.items()):raise ValueError('formal fresh initializer differs; no warm start')
                else:atomic_save(initial,ip)
                result=train_explicit(model,opt,x,data['unsafe'],data['draws'][arm],identity,norms,folder)
                rows=[json.loads(v) for v in (folder/'physical_updates.jsonl').read_text().splitlines()];trace=audit_event_trace(rows,data['draws'][arm],data['kind_indices'],data['unsafe'],identity)
                states={k:torch.load(folder/('latest.pt' if k=='latest' else f'step{k}.pt'),map_location='cpu',weights_only=True) for k in [2000,4000,6000,'latest']};state=validate_event_states(model,opt,states,identity,norms,initial)
                completion={'passed':True,'identity':identity,'physical_updates':6000,'fullstate_count':4,'raw_trace_sha256':sha(folder/'physical_updates.jsonl'),'state_pins':[{'path':str(folder/('latest.pt' if k=='latest' else f'step{k}.pt')),'sha256':sha(folder/('latest.pt' if k=='latest' else f'step{k}.pt'))} for k in [2000,4000,6000,'latest']],'trace_audit':trace,'state_audit':state,'physical_result':result}
                pin=self.ctx.write_json(rel+'/completion.json',completion);ar=archive_trace(folder/'physical_updates.jsonl',identity)
                out.append({'seed':row['seed'],'arm':arm,'completion':pin,'archive_sha256':ar['archive_sha256'],'reused_complete':False});del model,opt
        return self.ctx.write_json('formal_physical_completion.json',{'passed':True,'records':out,'formal_optimizer_updates':24000,'physical_models':4,'fullstate_count':16,'aux_optimizer_updates':48000},public=True)

    def audit_formal(self):
        freeze,e=self._enter('audit_formal');prepared=e['prepare_formal']['payload'];cov=self.ctx.evidence(self.descriptors['coverage'])['payload'];selector=self.ctx.evidence(self.descriptors['seal_selectors'])['payload'];models=[];fullstates=[];audits=[]
        import torch
        from echo_routing.temporal.selected_risk_audit import audit_event_trace,validate_event_states
        from echo_routing.temporal.selected_risk_storage_engineering import load_trace
        from echo_routing.temporal.anchor_cache import sha
        for row in prepared['records']:
            data,x,norms=self._data(row,cov,selector);initial_reference=None
            for arm in ['C0','C1']:
                identity=self._identity(row,data,freeze,arm);folder=self.ctx.output(f'models/seed{row["seed"]}/{arm}');completed_head(folder,identity,lambda p:self.ctx.raw({'path':str(Path(p['path']).relative_to(self.ctx.root)),'sha256':p['sha256']}))
                model,opt=new_head(row['seed'],arm,cov);initial={n:t.detach().clone() for n,t in model.state_dict().items()}
                if initial_reference is not None and any(not torch.equal(initial_reference[n],v) for n,v in initial.items()):raise ValueError('independent matched C0/C1 initializer recount failed')
                initial_reference=initial;rows=load_trace(folder/'physical_updates.jsonl.gz',json.loads((folder/'physical_updates.jsonl.archive.json').read_text()));trace=audit_event_trace(rows,data['draws'][arm],data['kind_indices'],data['unsafe'],identity)
                states={k:torch.load(folder/('latest.pt' if k=='latest' else f'step{k}.pt'),map_location='cpu',weights_only=True) for k in [2000,4000,6000,'latest']};state=validate_event_states(model,opt,states,identity,norms,initial)
                models.append({'seed':row['seed'],'arm':arm,'step':6000,'parameters':6625,'identity':identity,'path':str((folder/'step6000.pt').relative_to(self.ctx.root)),'sha256':sha(folder/'step6000.pt')})
                fullstates.extend({'seed':row['seed'],'arm':arm,'step':6000 if k=='latest' else k,'kind':'latest' if k=='latest' else 'milestone','path':str((folder/('latest.pt' if k=='latest' else f'step{k}.pt')).relative_to(self.ctx.root)),'sha256':sha(folder/('latest.pt' if k=='latest' else f'step{k}.pt'))} for k in [2000,4000,6000,'latest']);audits.append({'seed':row['seed'],'arm':arm,'trace':trace,'states':state});del model,opt
        if len(models)!=4 or len(fullstates)!=16 or sum(a['trace']['physical_updates'] for a in audits)!=24000:raise ValueError('independent complete four/24000/16 audit required')
        return self.ctx.write_json('formal_model_audit.json',{'passed':True,'formal_optimizer_updates':24000,'physical_models':4,'fullstate_count':16,'models':models,'fullstates':fullstates,'audits':audits,'formal_training_inputs_sha256':e['prepare_formal']['payload_sha256'],'no_forward_during_audit':True},public=True)
