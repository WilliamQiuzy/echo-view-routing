"""Frozen D / opposite-fold X label-free scoring, then sealed effect lookup.

This is a real CPU inference driver when separately authorized. Preparation
tests do not instantiate or forward any head. Native D reuses the exact frozen
post-A62 FIT C6; new mixed D and held X use qualified per-axis complete-C/all-K
infer, with no fallback and without any reference-effect lookup until Git seal.
"""
import io
import json
from pathlib import Path
import numpy as np
from echo_routing.temporal.selected_risk_resume_contract import digest
from echo_routing.temporal.selected_risk_resume_S import _cpu,SEEDS

SOURCE='echo_routing/temporal/selected_risk_resume_supply.py'
POST_NATIVE_PINS={2026110461:{'path':'runs/post_a62_frozen_score_diagnostic_v1/FIT_final_scores_seed2026110461.private.npz','sha256':'53e43ef337703cbfdda872b90de864d168c0417899a6662cfe2db16d19f58813'},2026110462:{'path':'runs/post_a62_frozen_score_diagnostic_v1/FIT_final_scores_seed2026110462.private.npz','sha256':'24216ecf00421c043fce3b66404c8e586d0d160923fce485f26d3727bfd6d9e0'}}


def selected_indices(value):
    out={}
    for role in ['D','X']:
        s=value[role];i=s['top_index']
        if s['N_admitted']:
            if isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or not 0<=i<len(value['actions']):raise ValueError('one original positive N-admitted top required')
            flags=out.setdefault(int(i),[False,False]);flags[['D','X'].index(role)]=True
    return {i:tuple(v) for i,v in out.items()}


def native_C6(matrix,oldfit,original_axis_indices,action_counts):
    c=np.asarray(matrix)
    if c.ndim!=2 or c.shape[1]!=6 or c.dtype!=np.float64 or not np.isfinite(c).all() or len(original_axis_indices)!=len(action_counts):raise ValueError('complete original float64 C6 candidate rows required')
    out=[]
    for qi,n in zip(original_axis_indices,action_counts):
        idx=np.asarray(oldfit['groups'][qi])
        if idx.ndim!=1 or idx.dtype.kind not in 'iu' or np.any(idx<0) or np.any(idx>=len(c)) or len(idx)!=n:raise ValueError('unchanged canonical native candidate group/order required')
        out.append(c[idx].copy())
    return out


def require_selector_seal(seal,cache_manifest_sha):
    if seal.get('passed') is not True or seal.get('complete') is not True or seal.get('cache_manifest_sha256')!=cache_manifest_sha or seal.get('effect_lookup_before_seal') is not False or (seal.get('aux_optimizer_updates'),seal.get('formal_optimizer_updates'))!=(48000,0) or not isinstance(seal.get('actual_Git_object_proof'),dict):raise ValueError('complete label-free selectors actual-Git sealed before any effect lookup required')
    return True


class Supply:
    def __init__(self,context,inputs,freeze_pin,descriptors):
        self.ctx=context;self.inputs=inputs;self.freeze_pin=freeze_pin;self.descriptors=descriptors

    def _enter(self,stage):
        f,e=self.ctx.enter(stage,self.freeze_pin,self.descriptors,required={SOURCE,'echo_routing/temporal/selected_risk_resume_context.py','echo_routing/temporal/selected_risk_resume_inputs.py','configs/experiments/selected_action_risk_v1.json'})
        _cpu();return f,e

    def _aux(self,seal):
        from echo_routing.temporal.safety_git import verify
        pins={n:seal[n] for n in ['models_manifest','audit','training_inputs']};values={k:self.ctx.read(v) for k,v in pins.items()}
        verify(self.ctx.read(seal['actual_Git_object_proof']),seal['git_commit'],{p['path']:p['sha256'] for p in pins.values()})
        m,a,t=(values[k] for k in ['models_manifest','audit','training_inputs'])
        if a.get('passed') is not True or (a.get('aux_optimizer_updates'),a.get('physical_models'),a.get('fullstate_count'))!=(48000,8,32) or a['models']!=m['models'] or a['aux_training_inputs_sha256']!=pins['training_inputs']['sha256'] or m['aux_training_inputs_sha256']!=pins['training_inputs']['sha256']:raise ValueError('actual auxiliary three-core model/independent-audit/input identities differ')
        if len(m['models'])!=8 or {(r['seed'],r['held_fold'],r['role']) for r in m['models']}!={(s,h,r) for s in SEEDS for h in [0,1] for r in ['utility','risk']} or any(r['step']!=6000 for r in m['models']):raise ValueError('exact eight final auxiliary heads required')
        for r in m['models']:self.ctx.raw({k:r[k] for k in ['path','sha256']})
        return values,pins

    def _load_heads(self,rows,seed,norms,scale):
        import torch
        from echo_routing.temporal.safety_training import Utility,Risk,norm_tensors
        heads={'utility':Utility('S-L',seed),'risk':Risk(seed)};expected=norm_tensors(norms)
        for role,model in heads.items():
            row=next(r for r in rows if r['seed']==seed and r['role']==role);pin={k:row[k] for k in ['path','sha256']};self.ctx.raw(pin)
            state=torch.load(self.ctx.root/pin['path'],map_location='cpu',weights_only=True)
            if (state['update'],state['seed'],state['role'],state['arm'],state['utility_scale'])!=(6000,seed,role,'S-L' if role=='utility' else 'risk',scale) or set(state['normalizers'])!=set(expected) or any(set(state['normalizers'][k])!=set(v) or any(not torch.equal(state['normalizers'][k][n],x) for n,x in v.items()) for k,v in expected.items()):raise ValueError('frozen final auxiliary/fixed head scale/normalizer identity differs')
            model.load_state_dict(state['model'],strict=True);model.eval();model.requires_grad_(False)
        return heads

    def _fixed(self,seed):
        from scripts.selected_risk_common import prior_path
        # prior_path independently enforces the literal approved86 table and
        # aliases. The old A62 config's prior_A61 map is itself pinned there.
        path=prior_path('configs/experiments/anchor_context_risk_v1.json');oldcfg=json.loads(path.read_text())
        manifest=self.ctx.read(oldcfg['prior_A61']['models_manifest.json']);training=self.ctx.read(oldcfg['prior_A61']['training_inputs.json']);norms=self.ctx.read(training['normalizers'])
        rows=[]
        for r in manifest['models']:
            if r['seed']==seed and r['arm'] in ['S-L','risk']:rows.append({**r,'role':'utility' if r['arm']=='S-L' else 'risk'})
        if len(rows)!=2:raise ValueError('both original final S-L/risk weights required')
        return self._load_heads(rows,seed,norms,training['utility_scale']),norms,training['utility_scale'],training

    def held_score(self):
        freeze,e=self._enter('held_score');values,pins=self._aux(e['seal_aux']['payload']);prepared=values['training_inputs'];records=[];calls=0;candidate_outputs=positions=0
        from echo_routing.temporal.selected_risk_compact_cache import CompactSupplyCache
        from echo_routing.temporal.safety_runtime import infer
        from echo_routing.temporal.event_risk_features import context_features
        from echo_routing.temporal.safety_features import utility_features
        from echo_routing.temporal.proposal_actions import propose,partition,apply
        from echo_routing.temporal.selected_risk_label_free_inputs import native_indices
        from echo_routing.temporal.anchor_cache import sha
        import torch
        first={};banks={r:self.inputs.bank(r,purpose='selector') for r in ['F0_native','F0_mixed','F1_native','F1_mixed']}
        for seed in SEEDS:
            fixed,oldnorm,oldscale,oldtraining=self._fixed(seed)
            oldfit=np.load(io.BytesIO(self.ctx.raw(oldtraining['dataset'])),allow_pickle=True).item()
            with np.load(io.BytesIO(self.ctx.raw(POST_NATIVE_PINS[seed])),allow_pickle=False) as z:
                if set(z.files)!={'C6','q'} or z['C6'].shape!=(4183,6):raise ValueError('entire frozen post-A62 FIT C6 required, no native recomputation')
                native_scores=z['C6'].copy()
            for held in [0,1]:
                fold=next(r for r in prepared['folds'] if r['held_fold']==held);data=self.ctx.load_value(fold['data'])['data'];rows=[r for r in values['models_manifest']['models'] if r['held_fold']==held];heads=self._load_heads(rows,seed,data['normalizers'],data['scale'])
                for kind in ['native','mixed']:
                    role=f'F{held}_{kind}';qs,aa,jj,bb=banks[role];rawa=[a.argmax(1) for a in aa];rawj=[j.argmax(1) for j in jj]
                    acts=[partition(b,propose(b,a,j))[0] for b,a,j in zip(bb,rawa,rawj)]
                    oldc=native_C6(native_scores,oldfit,native_indices(self.inputs.sequences('fit'),qs),[len(v) for v in acts]) if kind=='native' else None
                    identity=digest({'stage_code':{p:h for p,h in freeze['objects'].items() if p.endswith('.py')},'seed':seed,'held_fold':held,'role':role,'metadata':self.inputs.metadata_pin['sha256'],'aux_core_pins':pins,'native_frozen_C6_pin':POST_NATIVE_PINS[seed] if kind=='native' else None})
                    folder=self.ctx.output(f'held/seed{seed}/{role}');cache=CompactSupplyCache(folder,identity,bb,rawa,rawj,feature_reference=first.get(role));prefix=cache.cursor
                    for i in range(cache.cursor,len(qs)):
                        def predict(i=i):
                            nonlocal calls,candidate_outputs,positions
                            actions=acts[i];b=bb[i];a=aa[i];j=jj[i]
                            with torch.no_grad():
                                xvalue=infer(heads['utility'],heads['risk'],data['normalizers'],data['scale'],a,j,b)
                                dvalue=None if oldc is not None else infer(fixed['utility'],fixed['risk'],oldnorm,oldscale,a,j,b)
                            if xvalue['actions']!=actions or dvalue is not None and dvalue['actions']!=actions:raise ValueError('canonical G action/order mismatch')
                            calls+=3*bool(actions)*(1 if oldc is not None else 2);candidate_outputs+=2*len(actions)*(1 if oldc is not None else 2);positions+=sum(np.count_nonzero(apply(b,ac)!=b) for ac in actions)*(1 if oldc is not None else 2)
                            return {'D_C6':oldc[i] if oldc is not None else dvalue['score_matrix'],'X_C6':xvalue['score_matrix'],'x109':np.stack([utility_features(a,j,b,ac) for ac in actions]) if actions else np.empty((0,109),np.float64),'context96':np.stack([context_features(a,j,b,ac) for ac in actions]) if actions else np.empty((0,96),np.float64)}
                        cache.run_one(i,predict)
                    if cache.cursor!=len(qs):raise ValueError('all held axes including empty axes must be committed')
                    row={'seed':seed,'role':role,'path':str(folder.relative_to(self.ctx.root)),'identity':identity,'journal_sha256':sha(cache.journal),'axes':cache.cursor,'input_sha256':self.inputs.manifest['roles'][role]['sha256'],'resumed_committed_prefix_axes':prefix,'feature_reference_seed':None if role not in first else SEEDS[0]};records.append(row)
                    if seed==SEEDS[0]:first[role]=CompactSupplyCache(folder,identity,bb,rawa,rawj,readonly=True)
                del heads
            del fixed,oldfit
        # Weight pin reads are repeated after all forwards; no optimizer exists.
        for r in values['models_manifest']['models']:self.ctx.raw({k:r[k] for k in ['path','sha256']})
        out={'passed':True,'complete':True,'records':records,'seeds':list(SEEDS),'aux_optimizer_updates':48000,'formal_optimizer_updates':0,'effect_lookup_before_seal':False,'actual_new_MLP_module_calls_this_process':calls,'actual_new_candidate_head_outputs_this_process':candidate_outputs,'actual_new_changed_position_outputs_this_process':int(positions),'resumed_prefix_actual_total_calls_unknown':any(r['resumed_committed_prefix_axes'] for r in records),'no_teacher_or_optimizer':True,'native_D_C6_frozen_pins':POST_NATIVE_PINS,'aux_core_pins':pins}
        return self.ctx.write_json('held_selector_cache_manifest.json',out,public=True)

    def coverage(self):
        freeze,e=self._enter('coverage');seal=e['seal_selectors']['payload'];cache_pin=seal['cache_manifest'];manifest=self.ctx.read(cache_pin);require_selector_seal(seal,cache_pin['sha256'])
        from echo_routing.temporal.safety_git import verify
        objects={cache_pin['path']:cache_pin['sha256'],**{p['path']:p['sha256'] for p in manifest['aux_core_pins'].values()}}
        if seal['objects']!=objects:raise ValueError('entire label-free cache and auxiliary three-core identities must be sealed together')
        verify(self.ctx.read(seal['actual_Git_object_proof']),seal['git_commit'],objects)
        from echo_routing.temporal.selected_risk_compact_cache import CompactSupplyCache
        from echo_routing.temporal.selected_risk_effects import record
        from echo_routing.temporal.selected_risk_supply import coverage,session_of
        from echo_routing.temporal.selected_risk_record_storage import write_records,storage_bound
        from echo_routing.temporal.anchor_cache import sha
        part=self.inputs.partition();sessions=sorted(set(part['sessions'][0])|set(part['sessions'][1]));seeds={};tables=[];refs={};source_ok=True;atomic=0
        for seed in SEEDS:
            selected=[]
            for role in ['F0_native','F0_mixed','F1_native','F1_mixed']:
                qs,aa,jj,bb=self.inputs.bank(role,purpose='selector');held=int(role[1]);actual={session_of(v) for q in qs for v,*_ in q['refs']}
                if actual-set(part['sessions'][held]) or actual&set(part['sessions'][1-held]):source_ok=False
                row=next(r for r in manifest['records'] if (r['seed'],r['role'])==(seed,role));cache=CompactSupplyCache(self.ctx.root/row['path'],row['identity'],bb,[a.argmax(1) for a in aa],[j.argmax(1) for j in jj],readonly=True,feature_reference=refs.get(role))
                if cache.cursor!=len(qs) or sha(cache.journal)!=row['journal_sha256'] or row['input_sha256']!=self.inputs.manifest['roles'][role]['sha256']:raise ValueError('complete immutable label-free cache source/input/journal required')
                for qi,(q,b) in enumerate(zip(qs,bb)):
                    value=cache.load(qi)
                    for index,(D,X) in selected_indices(value).items():selected.append(record(q,b,value['actions'][index],value['context96'][index],D=D,X=X))
                if seed==SEEDS[0]:refs[role]=cache
            atomic+=storage_bound(len(selected),sessions)['atomic_coexistence_upper_bytes']
            if type(seal.get('selected_table_atomic_bytes_upper')) is not int or atomic>seal['selected_table_atomic_bytes_upper']:raise ValueError('actual402-dictionary selected-table atomic bytes exceed admitted bound')
            path=self.ctx.output(f'supply/seed{seed}.private.npz');pin=write_records(path,selected,sessions);tables.append({'seed':seed,'path':str(path.relative_to(self.ctx.root)),'sha256':pin['sha256'],'bytes':pin['bytes'],'rows':len(selected)})
            seeds[str(seed)]=coverage(selected)
        value={'passed':source_ok and all(s['passed'] for s in seeds.values()),'seeds':seeds,'source_isolation':{'passed':source_ok,'source_sessions':402,'fold_sizes':[201,201],'unit':'session, not patient'},'aux_optimizer_updates':48000,'formal_optimizer_updates':0,'C0_C1_instances':0,'selected_record_tables':tables,'selected_table_atomic_bytes':atomic,'selector_seal_sha256':e['seal_selectors']['payload_sha256'],'all_effect_lookup_after_actual_selector_Git_seal':True}
        return self.ctx.write_json('coverage.json',value,public=True)
