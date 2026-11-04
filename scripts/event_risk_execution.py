"""A62 stage helpers: pinned readonly sources, fixed C6 and new event score caches."""
import hashlib,time
import numpy as np
import torch
from scripts.event_risk_common import *
from echo_routing.temporal.event_risk_model import EventRisk
from echo_routing.temporal.event_risk_cache import EventCache
from echo_routing.temporal.event_risk_runtime import infer_events,C6_sha
from echo_routing.temporal.event_risk_policy import decode
from echo_routing.temporal.event_risk_audit import validate_barrier,FIXED_A61
from echo_routing.temporal.safety_policy_cache import SafetyCache
from echo_routing.temporal.safety_runtime import infer as old_infer
from echo_routing.temporal.safety_training import Utility,Risk
from echo_routing.temporal.baselines.registry import decode_with_prob
from echo_routing.temporal.structured_metrics import stress_context
BANKS={};OLD={};OLD_SCORES={};FIXED={};EVENT={};QINFO=[];FIXED_INFO=[]
ROLE_PREFIX={'a58_stress':'old_stress_','a59_confirmation':'confirmation_','a60_confirmation':'a60_confirmation_','a61_confirmation':'a61_confirmation_','a62_confirmation':'a62_confirmation_'}


def old_records():
    if not OLD:OLD.update({k:read(verify_pin(v)) for k,v in read(CFG)['prior_A61'].items()})
    return OLD

def role_for(table,half):return half if table=='main' else ROLE_PREFIX[table]+half

def preouter():
    models=read(PUB/'models_manifest.json');seal=read(PUB/'outer_identity.json');validate_barrier(models,seal,sha(PUB/'models_manifest.json'),sha(PUB/'calibration.json'),sha(PUB/'readout_plan.json'),read(verify_pin(seal['actual_Git_object_proof'])))
    for r in [*models['models'],*models['fixed_A61_models']]:verify_pin({'path':r['path'],'sha256':r['sha256']})
    return models

def bank(role):
    if role in BANKS:return BANKS[role]
    qs=sequences(role);pa=[];pj=[]
    if role in ['CAL','AUDIT']:
        mapping=read(verify_pin(read(PUB/'input_manifest.json')['partition_mapping']))[role];indices={int(i):int(j) for i,j in mapping['native_original_inner_indices'].items()};inner={s:probability_reader(s,'inner_cal') for s in ['a08','jepa']};mixed=bank(mapping['new_mixed_role']);offset=0
        for i,q in enumerate(qs):
            if q['kind']=='native':pa.append(inner['a08'].load(indices[i]));pj.append(inner['jepa'].load(indices[i]))
            else:pa.append(mixed[1][offset]);pj.append(mixed[2][offset]);offset+=1
        if offset!=150:raise ValueError('exactnew150 mixed plus unchanged native subset required')
    else:
        readers={s:probability_reader(s,role) for s in ['a08','jepa']}
        for i in range(len(qs)):pa.append(readers['a08'].load(i));pj.append(readers['jepa'].load(i))
    params=read(CFG)['source']['a08']['original_B3'];base=[decode_with_prob('b3',p,{'decode':{'b3':params}})[0] for p in pa];BANKS[role]=(qs,pa,pj,base);return BANKS[role]

def contexts():
    cfg=old_records()['config'];p=ROOT/cfg['data']['old_outer_metadata']
    if sha(p)!=cfg['data']['old_outer_metadata_sha256']:raise ValueError('original fulltask metadata changed')
    main=np.load(p,allow_pickle=True).item()
    from echo_routing.temporal.event_risk_metrics import TABLES,HALVES
    return {t:{h:main[h] if t=='main' else stress_context(main[h],sequences(role_for(t,h))) for h in HALVES} for t in TABLES}

def _fixed(seed):
    if seed in FIXED:return FIXED[seed]
    if not read(PUB/'model_audit.json')['passed']:raise ValueError('allfour actualfinal model audit precedes fixed-head new stream scores')
    old=old_records();rows=[r for r in old['models_manifest.json']['models'] if r['seed']==seed and r['arm'] in ['S-L','risk']];lookup={r['arm']:r for r in rows};heads={'S-L':Utility('S-L',seed),'risk':Risk(seed)}
    for arm,head in heads.items():
        row=lookup[arm];state=torch.load(verify_pin({'path':row['path'],'sha256':row['sha256']}),weights_only=True,map_location='cpu')['model'];head.load_state_dict(state,strict=True);head.eval()
        for p in head.parameters():p.requires_grad_(False)
    norms=read(verify_pin(old['training_inputs.json']['normalizers']));FIXED[seed]=(heads,norms,old['training_inputs.json']['utility_scale']);return FIXED[seed]

def old_scores(seed,role,readonly=False):
    key=(seed,role)
    if key in OLD_SCORES:return OLD_SCORES[key]
    qs,pa,pj,base=bank(role);old=old_records();pairs=[]
    if role in ['CAL','AUDIT']:
        mapping=read(verify_pin(read(PUB/'input_manifest.json')['partition_mapping']))[role];indices={int(i):int(j) for i,j in mapping['native_original_inner_indices'].items()};pinrow=next(r for r in old['models_manifest.json']['inner_records'] if r['seed']==seed and r['arm']=='S-L' and r['step']==6000);inner=read(verify_pin(pinrow));cache=np.load(verify_pin(inner['private_outputs']),allow_pickle=True).item();mix=old_scores(seed,mapping['new_mixed_role'],readonly=readonly);offset=0
        if len(cache['scores'])!=928:raise ValueError('original928 inner oldC6 source required')
        for i,q in enumerate(qs):
            if q['kind']=='native':
                index=indices[i];matrix=np.asarray(cache['scores'][index],np.float64);ref={'kind':'readonly_original_A61_inner','source_receipt_sha256':pinrow['sha256'],'payload_sha256':inner['private_outputs']['sha256'],'original_inner_index':index,'scores_sha256':C6_sha(matrix)};pairs.append((matrix,ref))
            else:pairs.append(mix[offset]);offset+=1
    else:
        matches=[r for r in old['outer_metrics.json']['rows'] if r.get('seed')==seed and r.get('arm')=='S-L' and r.get('policy')=='C' and r.get('source_role')==role]
        if matches:
            if len(matches)!=1:raise ValueError('one exact historical S-L/source cache required')
            info=matches[0]['policy_cache'];folder=ROOT/info['path']
            if sha(folder/'committed.jsonl')!=info['journal_sha256']:raise ValueError('original oldC6 journal changed')
            cache=SafetyCache(folder,info['identity'],base,[p.argmax(1) for p in pa],[p.argmax(1) for p in pj],matches[0]['tau'],readonly=True)
        else:
            if role.startswith('a62_confirmation_'):preouter()
            if role not in ['CAL_mixed','AUDIT_mixed','a62_confirmation_selection','a62_confirmation_calibration']:raise ValueError('new fixed heads only on four declared new roles')
            weights=[dict(r) for r in FIXED_A61 if r['seed']==seed];identity=digest({'old_fixed_weights':weights,'input':read(PUB/'input_manifest.json')['roles'][role]['sha256'],'producer_manifest':sha(PUB/'producer_manifest.json'),'old_pure_code':read(CFG)['readonly_code_pins']});folder=RUN/'fixed_scores'/f'seed{seed}'/role;cache=SafetyCache(folder,identity,base,[p.argmax(1) for p in pa],[p.argmax(1) for p in pj],read(CFG)['legacy_SL_tau'][str(seed)],readonly=readonly);before=cache.cursor
            if readonly and cache.cursor!=len(qs):raise ValueError('incomplete readonly fixed C6 cache cannot instantiate heads or forward')
            if cache.cursor<len(qs):heads,norms,scale=_fixed(seed)
            for i in range(cache.cursor,len(qs)):cache.run_one(i,lambda i=i:old_infer(heads['S-L'],heads['risk'],norms,scale,pa[i],pj[i],base[i],read(CFG)['legacy_SL_tau'][str(seed)]))
            FIXED_INFO.append({'seed':seed,'role':role,'new_fixed_MLP_module_calls':3*sum(bool(cache.load(i)['actions']) for i in range(before,len(qs))),'path':str(folder.relative_to(ROOT)),'journal_sha256':sha(cache.journal)})
        for i in range(len(qs)):
            matrix=cache.load(i)['score_matrix'];pairs.append((matrix,{'kind':'readonly_exact_fixed_C6_payload','path':str(cache.file(i).relative_to(ROOT)),'payload_sha256':sha(cache.file(i)),'journal_sha256':sha(cache.journal),'scores_sha256':C6_sha(matrix)}))
    OLD_SCORES[key]=pairs;return pairs

def event_heads(seed):
    if seed in EVENT:return EVENT[seed]
    audit=read(PUB/'model_audit.json');rows=[r for r in audit['models'] if r['seed']==seed]
    if not audit['passed'] or audit['formal_optimizer_updates']!=24000 or len(rows)!=2:raise ValueError('allfour audited final event models required')
    heads={}
    for r in rows:
        model=EventRisk(r['arm'],seed);state=torch.load(verify_pin({'path':r['path'],'sha256':r['sha256']}),weights_only=True,map_location='cpu');model.load_state_dict(state['model'],strict=True);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        heads[r['arm']]=model
    EVENT[seed]=heads;return heads

def score_values(seed,role,readonly=False):
    inner=role in ['CAL','AUDIT']
    if not inner:
        preouter();adopt=read(PUB/'source_adoption.json')
        if not adopt['passed'] or adopt['formal_updates_at_binding']!=24000 or adopt['producer_manifest_sha256']!=sha(PUB/'producer_manifest.json'):raise ValueError('finalhead barrier plus complete readonly teacher adoption required beforeouterhead outputs')
    stage=verify_freeze('calibration_freeze.json' if inner else 'readout_freeze.json');qs,pa,pj,base=bank(role);pairs=old_scores(seed,role,readonly=readonly);oldrefs=[r for _,r in pairs];inputs=read(PUB/'training_inputs.json');heads=None if readonly else event_heads(seed);norms=read(verify_pin(inputs['normalizers']));rows=[r for r in read(PUB/'model_audit.json')['models'] if r['seed']==seed];ident=digest({'model_pins':[{k:r[k] for k in ['arm','path','sha256']} for r in rows],'input':read(PUB/'input_manifest.json')['roles'][role]['sha256'],'old_reference_digest':digest(oldrefs),'normalizer':inputs['normalizers']['sha256'],'stage_source':stage['objects'],'raw_scores_only_before_strict_CAL_thresholds':inner});cache=EventCache(RUN/'q_scores'/f'seed{seed}'/role,ident,base,[p.argmax(1) for p in pa],[p.argmax(1) for p in pj],None,oldrefs,readonly=readonly);before=cache.cursor
    if not readonly:
        for i in range(cache.cursor,len(qs)):cache.run_one(i,lambda i=i:infer_events(heads,norms,pa[i],pj[i],base[i],pairs[i][0],pairs[i][1]))
    values=[cache.load(i) for i in range(len(qs))];info={'seed':seed,'role':role,'path':str(cache.path.relative_to(ROOT)),'identity':ident,'journal_sha256':sha(cache.journal),'sequences':len(qs),'new_event_MLP_forwards_this_process':2*sum(bool(values[i]['actions']) for i in range(before,len(qs))),'candidate_vectors':sum(len(v['actions']) for v in values),'raw_q_cache_does_not_fabricate_CAL_thresholds':True};QINFO.append(info);return values,info

def decoded(values,bases,taus):return [{**v,**decode(b,v['actions'],v['old_scores'],v['q'][:,0],v['q'][:,1],taus)} for b,v in zip(bases,values)]
