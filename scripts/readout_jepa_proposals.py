#!/usr/bin/env python3
"""A60 frozen four heads, three same-weight policies, exact124 task rows."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.proposal_common import *
from scripts.train_jepa_proposals import role_values
from scripts.cache_jepa_proposals import original_a08
from echo_routing.temporal.proposal_training import ProposalRegressor,optimizer,restore
from echo_routing.temporal.proposal_runtime import infer
from echo_routing.temporal.proposal_policy_cache import PolicyCache
from echo_routing.temporal.proposal_metrics import evaluate,action_summary
from echo_routing.temporal.anchor_edits import majority
from echo_routing.temporal.proposal_readout_plan import plan as fixed_plan,system,TABLES,HALVES,BASE
from echo_routing.temporal.structured_metrics import stress_context,compose_stress_predictions,verify_reference
from echo_routing.temporal.baselines.registry import decode_with_prob


def role_for(table,half):
    return half if table=='main' else {'a58_stress':'old_stress_','a59_confirmation':'confirmation_','a60_confirmation':'a60_confirmation_'}[table]+half


def contexts(cfg):
    path=ROOT/cfg['data']['old_outer_metadata']
    if sha(path)!=cfg['data']['old_outer_metadata_sha256']:raise ValueError('frozen task context changed')
    main=np.load(path,allow_pickle=True).item()
    return {t:{h:main[h] if t=='main' else stress_context(main[h],sequences(role_for(t,h))) for h in HALVES} for t in TABLES}


def validate_source_banks(outer,banks):
    manifest=read(PUB/'input_manifest.json');fresh={(r['source'],r['role']):r for r in outer['records']}
    if len(outer['records'])!=4 or set(fresh)!={(s,'a60_confirmation_'+h) for s in ['a08','jepa'] for h in HALVES}:raise ValueError('exact4 new source-role cells required')
    count=0
    for role,(qs,a,j,b,identities) in banks.items():
        if not len(qs)==len(a)==len(j)==len(b):raise ValueError('whole source axes changed')
        for s in ['a08','jepa']:
            old=manifest['source_cache_roles'].get(role,{}).get(s);row=old or fresh[(s,role)];folder=ROOT/old['path'] if old else RUN/'cache'/s/role
            if not row['complete'] or row['identity']!=identities[s] or row['journal_sha256']!=sha(folder/'committed.jsonl') or row['input_sha256']!=manifest['roles'][role]['sha256'] or row['committed_sequences']!=len(qs):raise ValueError('exact producer/ref/gamma/order/completeaxis manifest changed')
            count+=1
    if count!=16:raise ValueError('exact16 outer source-role posterior cells required')
    return True


def pair_identity(row,role,sources):
    return digest({'checkpoint_sha256':row['final_sha256'],'head_identity':row['identity'],'source_cache_identities':sources,'role':role,'input_sha256':read(PUB/'input_manifest.json')['roles'][role]['sha256'],'readout_freeze':verify_freeze('readout_freeze.json')['objects']})


def pair_cache(row,role,qs,a,j,b,sources,readonly=False):
    return PolicyCache(RUN/'policy_readout'/row['final_sha256']/role,pair_identity(row,role,sources),b,[p.argmax(1) for p in a],[p.argmax(1) for p in j],readonly=readonly)


def main():
    freeze=verify_freeze('readout_freeze.json');verify_freeze('outer_freeze.json');manifest=read(PUB/'models_manifest.json');cfg=read(CFG);plan=read(PUB/'readout_plan.json');inputs=read(PUB/'training_inputs.json')
    if not manifest['passed'] or manifest['formal_optimizer_updates']!=24000 or not manifest['all_models_and12_inner_records_frozen'] or plan!=fixed_plan():raise ValueError('complete frozen4/12/36 and124/144/32 endpointplan required')
    seal=read(PUB/'outer_identity.json')
    if seal['models_manifest_sha256']!=sha(PUB/'models_manifest.json') or seal['readout_plan_sha256']!=sha(PUB/'readout_plan.json'):raise ValueError('pre-outer actualGit weight/endpointseal changed')
    outer=read(PUB/'cache_outer_manifest.json');torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    ctx=contexts(cfg);banks={role_for(t,h):role_values(role_for(t,h)) for t in TABLES for h in HALVES};validate_source_banks(outer,banks)
    predictions={t:{h:{} for h in HALVES} for t in TABLES};rows=[];journals=[];main_values={}
    original=read(ROOT/'docs/results/2026-11-02_dense_target_sequence/A08_reference_metrics.json')['systems']
    def publish(table,half,name,pred,info):
        metric,_=evaluate(pred,ctx[table][half]);predictions[table][half][name]=pred;rows.append({'table':table,'half':half,'system':name,**info,**metric})
        write(PUB/'completed_outer_rows.json',{'rows':rows,'metric_rows':len(rows),'provisional_until_independent_aggregate_audit':True,'all4heads12inner36policysets_frozen_before_outer':True,'readout_freeze_git':freeze['git_commit']})
        print(json.dumps({'completed_outer_row':name,'table':table,'half':half,'native_macro_f1':metric['native_macro_f1'],'segment_f1_50_multi':metric['segment_f1_50_multi'],'short16_FN':metric['short_true_unmatched_16samples'],'short16_FP':metric['short_spurious_segments_16samples']}),flush=True)
    for half in HALVES:
        qs,a,j,b,_=banks[half]
        for seed in [0,1,2]:
            prob,params=original_a08(half,qs,seed,cfg);name=f'A08_s{seed}_original_B3';pred={q['id']:decode_with_prob('b3',p,{'decode':{'b3':params}})[0].astype(np.int8) for q,p in zip(qs,prob)}
            publish('main',half,name,pred,{'role':'reference','historical_original_parameters':True});verify_reference(rows[-1],original[half][f'A08_original_B3_s{seed}'])
            if seed==0 and any(not np.array_equal(pred[q['id']],base) for q,base in zip(qs,b)):raise ValueError('Base original A08 cache mismatch')
        for name,p in [('A08_s0_original_raw',a),('JEPA_core_fixed6000_raw',j)]:publish('main',half,name,{q['id']:v.argmax(1).astype(np.int8) for q,v in zip(qs,p)},{'role':'reference'})
        for table in TABLES[1:]:
            sq,sa,sj,sb,_=banks[role_for(table,half)]
            for name,p in [(BASE,sb),('A08_s0_original_raw',[v.argmax(1) for v in sa]),('JEPA_core_fixed6000_raw',[v.argmax(1) for v in sj])]:publish(table,half,name,compose_stress_predictions(predictions['main'][half][name],{q['id']:np.asarray(v,np.int8) for q,v in zip(sq,p)},qs,sq),{'role':'reference','native_pair_predictions_reused_identically':True})
    scales=np.asarray(read(verify_pin(inputs['target_scales']))['scales'])
    for row in manifest['models']:
        seed,arm=row['seed'],row['arm'];norm=read(verify_pin(row['normalizer']));path=ROOT/row['final_path']
        if sha(path)!=row['final_sha256']:raise ValueError('fixed6000 fileSHAchanged')
        model=ProposalRegressor(arm,seed);opt=optimizer(model)
        if restore(model,opt,torch.load(path,map_location='cpu',weights_only=True),row['identity'],norm,scales)!=6000:raise ValueError('fixed6000 checkpointchanged')
        for table in TABLES:
            for half in HALVES:
                role=role_for(table,half);qs,a,j,b,sources=banks[role];cache=pair_cache(row,role,qs,a,j,b,sources)
                for i in range(cache.cursor,len(qs)):cache.run_one(i,lambda i=i:infer(model,norm,scales,a[i],j[i],b[i],arm))
                values=[cache.load(i) for i in range(len(qs))]
                if table=='main':main_values[(seed,arm,half)]=values
                info={'identity':cache.identity,'journal_sha256':sha(cache.journal),'sequences':len(qs),'path':str(cache.path.relative_to(ROOT)),'source_cache_identities':sources,'physical_head_score_forwards_shared_OER':sum(bool(v['actions']) for v in values),'physical_candidate_output_vectors':sum(len(v['outputs']) for v in values)};journals.append(info)
                for policy in ['O','E','R']:
                    name=system(seed,arm+'-'+policy);pred={q['id']:v[policy]['prediction'] for q,v in zip(qs,values)}
                    stats=action_summary(qs,a,j,b,values,policy,inputs['F_fit']);fullstats=stats
                    if table!='main':
                        mq,ma,mj,mb,_=banks[half];mv=main_values[(seed,arm,half)];lookup={q['id']:(q,x,y,z,v) for q,x,y,z,v in zip(mq,ma,mj,mb,mv) if q['kind']!='multi'};lookup.update({q['id']:(q,x,y,z,v) for q,x,y,z,v in zip(qs,a,j,b,values)})
                        group=[lookup[q['id']] for q in ctx[table][half]['sequences']];fullstats=action_summary(*[list(v) for v in zip(*group)],policy,inputs['F_fit'])
                        pred=compose_stress_predictions(predictions['main'][half][name],pred,mq,qs)
                    if any(majority(v[policy]['prediction'])!=majority(base) for v,base in zip(values,b)):raise ValueError('everysequenceguard failed')
                    publish(table,half,name,pred,{'role':'policy','seed':seed,'arm':arm,'policy':policy,'step':6000,'physical_model':f'seed{seed}_{arm}_fixed6000','weight_sha256':row['final_sha256'],'policy_cache':info,'candidate_count':fullstats['candidate_count'],'accepted_actions':fullstats['accepted_actions'],'action_statistics_full_task':fullstats,'action_statistics_source_role':stats,'source_role_candidate_count':stats['candidate_count'],'source_role_accepted_actions':stats['accepted_actions'],'source_role':role,'native_pair_predictions_reused_identically':table!='main','all_sequences_G_majority_unchanged':True,'cached_feature_head_policy_seconds_source_role':sum(v['cached_feature_head_policy_seconds'] for v in values),'full_system_inference_seconds':None,'producer_cost_excluded_from_cached_head_latency':True,'operational_required_producers':['A08','JEPA'],'OER_shared_weights_scores_and_physical_forward':True,'A_control_also_uses_JEPA_proposals':True})
                write(RUN/'runtimecoordination.json',{'phase':'frozen_A60_OER_readout','model_pid':os.getpid(),'seed':seed,'arm':arm,'table':table,'half':half,'metric_rows':len(rows),'epoch':time.time(),'next_action':'all124thenindependentrecount/144CIs/32oracleandclosure'})
    expected={(r['table'],r['half'],r['system'],r['role']) for r in plan['rows']};actual=[(r['table'],r['half'],r['system'],r['role']) for r in rows]
    if len(rows)!=124 or set(actual)!=expected or len(journals)!=32:raise ValueError('exact124rows/32sharedcells required')
    target=RUN/'readout_predictions.private.npy';save_array(target,predictions)
    write(PUB/'outer_metrics.json',{'rows':rows,'metric_rows':124,'policy_metric_rows':96,'reference_metric_rows':28,'readout_freeze_git':freeze['git_commit'],'models_manifest_sha256':sha(PUB/'models_manifest.json'),'readout_plan_sha256':sha(PUB/'readout_plan.json'),'source_outer_cache_manifest_sha256':sha(PUB/'cache_outer_manifest.json'),'private_predictions':array_pin(target),'joint_policy_cache_cells':journals,'OER_scores_shared_physical_forward_once':True,'actual_physical_head_score_forwards':sum(x['physical_head_score_forwards_shared_OER'] for x in journals),'actual_physical_candidate_vectors':sum(x['physical_candidate_output_vectors'] for x in journals),'historical_A08_all3_reproduced':True,'all4heads12inner36sets_frozen_before_outer':True,'no_outer_or_inner_model_selection':True,'all_length_changed_fraction_wholeaxis_and_harm_descriptors_in_first_reader':True,'epoch':time.time()})
    write(RUN/'runtimecoordination.json',{'phase':'all124readouts_complete_pending144CIs32oracles','model_pid':None,'metric_rows':124,'epoch':time.time(),'next_action':'freezeallreadouts_then144paired/32oracle/closure'})
if __name__=='__main__':main()
