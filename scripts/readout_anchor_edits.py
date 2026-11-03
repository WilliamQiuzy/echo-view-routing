#!/usr/bin/env python3
"""Frozen A59 four physical heads, eight policies, all70 fixed task rows."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.anchor_common import RUN,PUB,CFG,read,write,sha,digest,verify_pin,verify_freeze,sequences,save_array,array_pin
from scripts.train_anchor_edits import role_values
from scripts.cache_anchor_edits import original_a08
from echo_routing.temporal.anchor_training import UtilityRegressor,optimizer,restore
from echo_routing.temporal.anchor_runtime import infer
from echo_routing.temporal.anchor_policy_cache import PolicyPairCache
from echo_routing.temporal.anchor_metrics import evaluate
from echo_routing.temporal.anchor_source_binding import validate as validate_sources
from echo_routing.temporal.anchor_edits import majority
from echo_routing.temporal.anchor_readout_plan import plan as fixed_plan,system,TABLES,HALVES,BASE
from echo_routing.temporal.structured_metrics import stress_context,compose_stress_predictions,verify_reference
from echo_routing.temporal.baselines.registry import decode_with_prob


def role_for(table,half):
    return half if table=='main' else ('old_stress_' if table=='old_stress' else 'confirmation_')+half


def contexts(cfg):
    path=ROOT/cfg['data']['old_outer_metadata']
    if sha(path)!=cfg['data']['old_outer_metadata_sha256']:raise ValueError('frozen task context changed')
    main=np.load(path,allow_pickle=True).item()
    return {table:{half:main[half] if table=='main' else stress_context(main[half],sequences(role_for(table,half)))
                   for half in HALVES} for table in TABLES}


def validate_source_banks(outer,banks):
    observed={}
    manifest=read(PUB/'input_manifest.json')
    for role,(qs,a,j,b,sources) in banks.items():
        for source in ['a08','jepa']:
            observed[(source,role)]={'identity':sources[source],
                'journal_sha256':sha(RUN/'cache'/source/role/'committed.jsonl'),
                'committed_sequences':len(qs),'input_sha256':manifest['roles'][role]['sha256']}
    return validate_sources(outer['records'],observed)


def pair_identity(row,role,sources):
    return digest({'checkpoint_sha256':row['final_sha256'],'head_identity':row['identity'],
                   'source_cache_identities':sources,'role':role,'input_sha256':read(PUB/'input_manifest.json')['roles'][role]['sha256'],
                   'readout_freeze':verify_freeze('readout_freeze.json')['objects']})


def pair_cache(row,role,qs,a,b,sources,readonly=False):
    identity=pair_identity(row,role,sources)
    return PolicyPairCache(RUN/'policy_readout'/row['final_sha256']/role,identity,b,[p.argmax(1) for p in a],readonly=readonly)


def candidate_lengths(values):
    lengths=[e-s for value in values for s,e in value['spans']]
    return {'count':len(lengths),'minimum':min(lengths) if lengths else None,
            'median':float(np.median(lengths)) if lengths else None,'maximum':max(lengths) if lengths else None,
            'total':sum(lengths),'atmost10':sum(n<=10 for n in lengths),'atmost16':sum(n<=16 for n in lengths)}


def main():
    freeze=verify_freeze('readout_freeze.json');verify_freeze('outer_freeze.json')
    manifest=read(PUB/'models_manifest.json');cfg=read(CFG);plan=read(PUB/'readout_plan.json')
    if (not manifest['passed'] or manifest['formal_optimizer_updates']!=24000 or not manifest['all_models_and12_inner_records_frozen']
        or plan!=fixed_plan()):raise ValueError('complete frozen matrix/70rowplan required')
    seal=read(PUB/'outer_identity.json')
    if seal['models_manifest_sha256']!=sha(PUB/'models_manifest.json') or seal['readout_plan_sha256']!=sha(PUB/'readout_plan.json'):
        raise ValueError('pre-outer actualGit model/plan seal changed')
    outer=read(PUB/'cache_outer_manifest.json')
    if not outer['passed'] or not all(r['complete'] for r in outer['records']):raise ValueError('complete matched source caches required')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    ctx=contexts(cfg);banks={role_for(t,h):role_values(role_for(t,h)) for t in TABLES for h in HALVES}
    validate_source_banks(outer,banks)
    predictions={t:{h:{} for h in HALVES} for t in TABLES};rows=[];journals=[]
    original=read(ROOT/'docs/results/2026-11-02_dense_target_sequence/A08_reference_metrics.json')['systems']
    def publish(table,half,name,pred,info):
        metric,_=evaluate(pred,ctx[table][half]);predictions[table][half][name]=pred
        rows.append({'table':table,'half':half,'system':name,**info,**metric})
        write(PUB/'completed_outer_rows.json',{'rows':rows,'metric_rows':len(rows),'provisional_until_independent_aggregate_audit':True,
              'all4heads12inner_frozen_before_outer':True,'readout_freeze_git':freeze['git_commit']})
        print(json.dumps({'completed_outer_row':name,'table':table,'half':half,'native_macro_f1':metric['native_macro_f1'],
              'segment_f1_50_multi':metric['segment_f1_50_multi'],'short16_FN':metric['short_true_unmatched_16samples'],
              'short16_FP':metric['short_spurious_segments_16samples']}),flush=True)
    for half in HALVES:
        qs,a,j,b,_=banks[half]
        if [q['id'] for q in qs]!=[q['id'] for q in ctx['main'][half]['sequences']]:raise ValueError('main task axes changed')
        for seed in [0,1,2]:
            p,params=original_a08(half,qs,seed,cfg)
            pred={q['id']:decode_with_prob('b3',value,{'decode':{'b3':params}})[0].astype(np.int8) for q,value in zip(qs,p)}
            name=f'A08_s{seed}_original_B3'
            publish('main',half,name,pred,{'role':'reference','historical_original_parameters':True})
            verify_reference(rows[-1],original[half][f'A08_original_B3_s{seed}'])
            if seed==0 and any(not np.array_equal(pred[q['id']],base) for q,base in zip(qs,b)):
                raise ValueError('original main Base differs from pinned source cache')
        for name,values in [('A08_s0_original_raw',a),('JEPA_core_fixed6000_raw',j)]:
            publish('main',half,name,{q['id']:p.argmax(1).astype(np.int8) for q,p in zip(qs,values)},{'role':'reference'})
        for table in TABLES[1:]:
            role=role_for(table,half);sq,sa,sj,sb,_=banks[role]
            for name,values in [(BASE,sb),('A08_s0_original_raw',[p.argmax(1) for p in sa]),('JEPA_core_fixed6000_raw',[p.argmax(1) for p in sj])]:
                pred=compose_stress_predictions(predictions['main'][half][name],{q['id']:np.asarray(p,np.int8) for q,p in zip(sq,values)},qs,sq)
                publish(table,half,name,pred,{'role':'reference','native_pair_predictions_reused_identically':True})
    for row in manifest['models']:
        seed,arm=row['seed'],row['arm'];normalizer=read(verify_pin(row['normalizer']))
        path=ROOT/row['final_path']
        if sha(path)!=row['final_sha256']:raise ValueError('frozen final weight changed')
        model=UtilityRegressor(arm,seed);opt=optimizer(model)
        if restore(model,opt,torch.load(path,map_location='cpu',weights_only=True),row['identity'],normalizer)!=6000:
            raise ValueError('fixed6000 readout checkpoint changed')
        model.eval()
        for table in TABLES:
            for half in HALVES:
                role=role_for(table,half);qs,a,j,b,sources=banks[role]
                cache=pair_cache(row,role,qs,a,b,sources)
                for i in range(cache.cursor,len(qs)):
                    cache.run_one(i,lambda i=i:infer(model,normalizer,a[i],j[i],b[i],arm))
                values=[cache.load(i) for i in range(len(qs))]
                if any(majority(v['G']['prediction'])!=majority(base) for v,base in zip(values,b)):
                    raise ValueError('whole-task source-only majority guard failed')
                info={'identity':cache.identity,'journal_sha256':sha(cache.journal),'sequences':len(qs),
                      'path':str(cache.path.relative_to(ROOT)),'source_cache_identities':sources,
                      'actual_head_score_forwards_shared_GU':sum(bool(v['spans']) for v in values),
                      'candidate_scores_shared_GU':sum(len(v['scores']) for v in values)}
                journals.append(info)
                for policy in ['U','G']:
                    name=system(seed,arm+'-'+policy)
                    pred={q['id']:value[policy]['prediction'] for q,value in zip(qs,values)}
                    if table!='main':pred=compose_stress_predictions(predictions['main'][half][name],pred,banks[half][0],qs)
                    publish(table,half,name,pred,{'role':'policy','seed':seed,'arm':arm,'policy':policy,'step':6000,
                        'physical_model':f'seed{seed}_{arm}_fixed6000','weight_sha256':row['final_sha256'],'policy_cache':info,
                        'candidate_count':sum(len(v['spans']) for v in values),
                        'positive_scores':sum(int((v['scores']>0).sum()) for v in values),
                        'zero_scores':sum(int((v['scores']==0).sum()) for v in values),
                        'negative_scores':sum(int((v['scores']<0).sum()) for v in values),
                        'candidate_lengths':candidate_lengths(values),
                        'candidate_statistics_scope':'source role only;stress native/pairs reuse main statistics',
                        'accepted_actions':sum(v[policy]['accepted'] is not None for v in values),
                        'guard_vetoes':sum(len(v[policy]['vetoed']) for v in values),
                        'cached_feature_head_policy_seconds':sum(v['cached_feature_head_policy_seconds'] for v in values),
                        'full_system_inference_seconds':None,'producer_cost_excluded_from_cached_head_latency':True,
                        'operational_required_producers':['A08'] if arm=='A' else ['A08','JEPA'],
                        'G_U_scores_compute_shared':True,'native_pair_predictions_reused_identically':table!='main'})
                write(RUN/'runtimecoordination.json',{'phase':'frozen_anchor_policy_readout','model_pid':os.getpid(),'table':table,
                      'half':half,'seed':seed,'arm':arm,'metric_rows':len(rows),'epoch':time.time(),
                      'next_action':'complete all70 fixed rows; independent96paired comparisons and declared action-space oracle'})
    actual={(r['table'],r['half'],r['system'],r['role']) for r in rows}
    expected={(r['table'],r['half'],r['system'],r['role']) for r in plan['rows']}
    if len(rows)!=70 or actual!=expected or len(journals)!=24:raise ValueError('exact70rows/24shared-policycells required')
    private=RUN/'readout_predictions.private.npy';save_array(private,predictions)
    write(PUB/'outer_metrics.json',{'rows':rows,'metric_rows':70,'policy_metric_rows':48,'reference_metric_rows':22,
        'readout_freeze_git':freeze['git_commit'],'models_manifest_sha256':sha(PUB/'models_manifest.json'),
        'readout_plan_sha256':sha(PUB/'readout_plan.json'),'source_outer_cache_manifest_sha256':sha(PUB/'cache_outer_manifest.json'),
        'private_predictions':array_pin(private),'policy_pair_cache_cells':journals,
        'G_U_scores_shared_physical_forward_once':True,
        'actual_physical_head_score_forwards':sum(r['actual_head_score_forwards_shared_GU'] for r in journals),
        'actual_physical_candidate_scores':sum(r['candidate_scores_shared_GU'] for r in journals),'historical_A08_all3_reproduced':True,
        'all4heads12inner_frozen_before_outer':True,'no_outer_or_inner_model_selection':True,'epoch':time.time()})
    write(RUN/'runtimecoordination.json',{'phase':'all70fixed_readouts_complete_pending96paired_recounts','model_pid':None,
          'metric_rows':70,'epoch':time.time(),'next_action':'freeze predictions then independent70recount/96bootstrap/oracle'})

if __name__=='__main__':main()
